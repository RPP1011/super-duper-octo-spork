#!/usr/bin/env python3
"""SAC-Discrete training module for V4/V5 dual-head actor-critic.

Implements Soft Actor-Critic for discrete action spaces with factored Q-values:
  - Move: 9 discrete actions (8 directions + stay)
  - Combat: up to 10 discrete actions (attack, hold, ability_0..7)
  - Pointer: variable-size discrete (target selection among entities)

Uses shared actor encoder with detached features for lightweight Q-heads,
avoiding full encoder duplication (DrQ/CURL-style).

Usage:
    from sac_learner import SACTrainer
    trainer = SACTrainer(actor, d_model=128)
    trainer.add_episodes("episodes.jsonl", registry, cls_dim=128)
    metrics = trainer.train_n_steps(1000)
"""

from __future__ import annotations

import copy
import json
import math
import os
import struct
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import (
    MAX_ABILITIES,
    NUM_COMBAT_TYPES,
    NUM_MOVE_DIRS,
)

ENTITY_DIM = 30
THREAT_DIM = 8
POSITION_DIM = 8
AGG_DIM = 16

LOG_PROB_MIN = -10.0
Q_CLAMP = 50.0  # loose clamp — rely on Huber loss + target network for stability
LOG_EPS = 1e-8

# SHM offsets — must match gpu_inference_server.py and gpu_client.rs
OFF_RELOAD_PATH = 0x80
RELOAD_PATH_LEN = 256
OFF_RELOAD_REQ = 0x180
OFF_RELOAD_ACK = 0x184
SHM_PATH_PREFIX = "/dev/shm/"


# ---------------------------------------------------------------------------
# Lightweight Q-heads (shared encoder, separate heads)
# ---------------------------------------------------------------------------


class SACCriticHeads(nn.Module):
    """Twin Q-heads operating on pooled features from the actor encoder.

    Each twin has independent Q-value heads for move, combat, and pointer
    actions. The encoder is shared with the actor but features are detached
    before being fed to critic heads (no encoder gradients from critic loss).
    """

    def __init__(
        self,
        d_model: int,
        n_move: int = NUM_MOVE_DIRS,
        n_combat: int = NUM_COMBAT_TYPES,
    ):
        super().__init__()
        self.d_model = d_model

        # Q1 heads
        self.q1_move = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, n_move),
        )
        self.q1_combat = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, n_combat),
        )
        self.q1_ptr_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )

        # Q2 heads
        self.q2_move = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, n_move),
        )
        self.q2_combat = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, n_combat),
        )
        self.q2_ptr_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )

        self.scale = d_model ** -0.5

    def forward(
        self,
        pooled: torch.Tensor,
        entity_tokens: torch.Tensor,
        entity_mask: torch.Tensor,
    ) -> dict:
        """Compute twin Q-values for all action heads.

        Args:
            pooled: (B, D) pooled representation from actor encoder (detached)
            entity_tokens: (B, N, D) entity tokens from actor encoder (detached)
            entity_mask: (B, N) True where padded

        Returns:
            dict with q1_move, q2_move (B, 9), q1_combat, q2_combat (B, 10),
            q1_ptr, q2_ptr (B, N) — pointer Q-values per entity
        """
        q1_move = self.q1_move(pooled)
        q2_move = self.q2_move(pooled)

        q1_combat = self.q1_combat(pooled)
        q2_combat = self.q2_combat(pooled)

        # Pointer Q-values: query-key dot product over entity tokens
        q1_query = self.q1_ptr_proj(pooled).unsqueeze(1)  # (B, 1, D)
        q1_ptr = (q1_query @ entity_tokens.transpose(-1, -2)).squeeze(1) * self.scale
        q1_ptr = q1_ptr.masked_fill(entity_mask, -1e9)

        q2_query = self.q2_ptr_proj(pooled).unsqueeze(1)
        q2_ptr = (q2_query @ entity_tokens.transpose(-1, -2)).squeeze(1) * self.scale
        q2_ptr = q2_ptr.masked_fill(entity_mask, -1e9)

        return {
            "q1_move": q1_move, "q2_move": q2_move,
            "q1_combat": q1_combat, "q2_combat": q2_combat,
            "q1_ptr": q1_ptr, "q2_ptr": q2_ptr,
        }


# ---------------------------------------------------------------------------
# Replay buffer — struct-of-arrays on CPU
# ---------------------------------------------------------------------------


class ReplayBuffer:
    """Fixed-size replay buffer storing transitions as contiguous numpy arrays.

    All arrays are pre-allocated at capacity. New transitions overwrite the
    oldest ones via a circular write pointer.
    """

    def __init__(self, capacity: int, max_ents: int, max_threats: int,
                 max_positions: int, cls_dim: int, has_aggregate: bool):
        self.capacity = capacity
        self.size = 0
        self.pos = 0
        self.max_ents = max_ents
        self.max_threats = max_threats
        self.max_positions = max_positions
        self.cls_dim = cls_dim
        self.has_aggregate = has_aggregate

        # --- Current state ---
        self.ent_feat = np.zeros((capacity, max_ents, ENTITY_DIM), dtype=np.float32)
        self.ent_types = np.zeros((capacity, max_ents), dtype=np.int64)
        self.ent_mask = np.ones((capacity, max_ents), dtype=np.bool_)
        self.thr_feat = np.zeros((capacity, max_threats, THREAT_DIM), dtype=np.float32)
        self.thr_mask = np.ones((capacity, max_threats), dtype=np.bool_)
        self.pos_feat = np.zeros((capacity, max_positions, POSITION_DIM), dtype=np.float32)
        self.pos_mask = np.ones((capacity, max_positions), dtype=np.bool_)
        self.agg_feat = np.zeros((capacity, AGG_DIM), dtype=np.float32) if has_aggregate else None
        self.ability_cls = np.zeros((MAX_ABILITIES, capacity, cls_dim), dtype=np.float32)
        self.ability_cls_valid = np.zeros((MAX_ABILITIES, capacity), dtype=np.bool_)

        # --- Next state ---
        self.next_ent_feat = np.zeros((capacity, max_ents, ENTITY_DIM), dtype=np.float32)
        self.next_ent_types = np.zeros((capacity, max_ents), dtype=np.int64)
        self.next_ent_mask = np.ones((capacity, max_ents), dtype=np.bool_)
        self.next_thr_feat = np.zeros((capacity, max_threats, THREAT_DIM), dtype=np.float32)
        self.next_thr_mask = np.ones((capacity, max_threats), dtype=np.bool_)
        self.next_pos_feat = np.zeros((capacity, max_positions, POSITION_DIM), dtype=np.float32)
        self.next_pos_mask = np.ones((capacity, max_positions), dtype=np.bool_)
        self.next_agg_feat = np.zeros((capacity, AGG_DIM), dtype=np.float32) if has_aggregate else None
        self.next_ability_cls = np.zeros((MAX_ABILITIES, capacity, cls_dim), dtype=np.float32)
        self.next_ability_cls_valid = np.zeros((MAX_ABILITIES, capacity), dtype=np.bool_)

        # --- Actions ---
        self.move_dirs = np.zeros(capacity, dtype=np.int64)
        self.combat_types = np.zeros(capacity, dtype=np.int64)
        self.target_indices = np.zeros(capacity, dtype=np.int64)
        self.combat_masks = np.zeros((capacity, NUM_COMBAT_TYPES), dtype=np.bool_)

        # --- Reward, done ---
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

    def _store_state(self, idx: int, step: dict, ability_cls_list: list,
                     prefix: str):
        """Write one state into the buffer arrays at `idx`."""
        ent_feat = getattr(self, f"{prefix}ent_feat")
        ent_types = getattr(self, f"{prefix}ent_types")
        ent_mask = getattr(self, f"{prefix}ent_mask")
        thr_feat = getattr(self, f"{prefix}thr_feat")
        thr_mask = getattr(self, f"{prefix}thr_mask")
        pos_feat = getattr(self, f"{prefix}pos_feat")
        pos_mask = getattr(self, f"{prefix}pos_mask")
        agg_feat = getattr(self, f"{prefix}agg_feat")
        ab_cls = getattr(self, f"{prefix}ability_cls")
        ab_valid = getattr(self, f"{prefix}ability_cls_valid")

        # Entities
        ents = step.get("entities", [])
        n_e = min(len(ents), self.max_ents)
        ent_feat[idx] = 0
        ent_types[idx] = 0
        ent_mask[idx] = True
        if n_e > 0:
            ent_feat[idx, :n_e] = ents[:n_e]
            ent_types[idx, :n_e] = step["entity_types"][:n_e]
            ent_mask[idx, :n_e] = False

        # Threats
        threats = step.get("threats")
        thr_feat[idx] = 0
        thr_mask[idx] = True
        if threats:
            n_t = min(len(threats), self.max_threats)
            thr_feat[idx, :n_t] = threats[:n_t]
            thr_mask[idx, :n_t] = False

        # Positions
        positions = step.get("positions")
        pos_feat[idx] = 0
        pos_mask[idx] = True
        if positions:
            n_p = min(len(positions), self.max_positions)
            pos_feat[idx, :n_p] = positions[:n_p]
            pos_mask[idx, :n_p] = False

        # Aggregate features
        if agg_feat is not None:
            agg = step.get("aggregate_features")
            if agg and len(agg) >= AGG_DIM:
                agg_feat[idx] = agg[:AGG_DIM]
            else:
                agg_feat[idx] = 0

        # Ability CLS
        ab_valid[:, idx] = False
        if ability_cls_list:
            for ab_idx in range(min(MAX_ABILITIES, len(ability_cls_list))):
                if ability_cls_list[ab_idx] is not None:
                    ab_cls[ab_idx, idx] = ability_cls_list[ab_idx]
                    ab_valid[ab_idx, idx] = True

    def add(
        self,
        step: dict,
        next_step: dict,
        ability_cls_list: list,
        next_ability_cls_list: list,
        move_dir: int,
        combat_type: int,
        target_idx: int,
        combat_mask: np.ndarray,
        reward: float,
        done: bool,
    ):
        """Add a single transition to the buffer."""
        i = self.pos
        self._store_state(i, step, ability_cls_list, "")
        self._store_state(i, next_step, next_ability_cls_list, "next_")

        self.move_dirs[i] = move_dir
        self.combat_types[i] = combat_type
        self.target_indices[i] = min(target_idx, self.max_ents - 1)
        self.combat_masks[i] = combat_mask
        self.rewards[i] = reward
        self.dones[i] = done

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: str = "cuda") -> dict:
        """Sample a random batch and transfer to device."""
        idx = np.random.randint(0, self.size, size=batch_size)
        return self._get_batch(idx, device)

    def _get_batch(self, idx: np.ndarray, device: str) -> dict:
        """Fetch batch by index array, transfer to device."""

        def to_dev(arr):
            return torch.from_numpy(arr).to(device, non_blocking=True)

        def build_cls(ab_arr, valid_arr):
            result = []
            for ab_idx in range(MAX_ABILITIES):
                if valid_arr[ab_idx, idx].any():
                    result.append(to_dev(ab_arr[ab_idx, idx]))
                else:
                    result.append(None)
            return result

        state = {
            "entity_features": to_dev(self.ent_feat[idx]),
            "entity_type_ids": to_dev(self.ent_types[idx]),
            "threat_features": to_dev(self.thr_feat[idx]),
            "entity_mask": to_dev(self.ent_mask[idx]),
            "threat_mask": to_dev(self.thr_mask[idx]),
            "position_features": to_dev(self.pos_feat[idx]),
            "position_mask": to_dev(self.pos_mask[idx]),
        }
        if self.agg_feat is not None:
            state["aggregate_features"] = to_dev(self.agg_feat[idx])

        next_state = {
            "entity_features": to_dev(self.next_ent_feat[idx]),
            "entity_type_ids": to_dev(self.next_ent_types[idx]),
            "threat_features": to_dev(self.next_thr_feat[idx]),
            "entity_mask": to_dev(self.next_ent_mask[idx]),
            "threat_mask": to_dev(self.next_thr_mask[idx]),
            "position_features": to_dev(self.next_pos_feat[idx]),
            "position_mask": to_dev(self.next_pos_mask[idx]),
        }
        if self.next_agg_feat is not None:
            next_state["aggregate_features"] = to_dev(self.next_agg_feat[idx])

        return {
            "state": state,
            "next_state": next_state,
            "ability_cls": build_cls(self.ability_cls, self.ability_cls_valid),
            "next_ability_cls": build_cls(self.next_ability_cls, self.next_ability_cls_valid),
            "move_dirs": to_dev(self.move_dirs[idx]),
            "combat_types": to_dev(self.combat_types[idx]),
            "target_indices": to_dev(self.target_indices[idx]),
            "combat_masks": to_dev(self.combat_masks[idx]),
            "rewards": to_dev(self.rewards[idx]),
            "dones": to_dev(self.dones[idx]),
        }


# ---------------------------------------------------------------------------
# Combat mask helper
# ---------------------------------------------------------------------------


def _build_combat_mask_single(step: dict) -> np.ndarray:
    """Build a single combat mask (10,) for one step."""
    mask_raw = step.get("mask", [])
    cm = np.zeros(NUM_COMBAT_TYPES, dtype=np.bool_)
    cm[0] = any(mask_raw[j] for j in range(min(3, len(mask_raw))))
    cm[1] = True  # hold is always valid
    for ab_idx in range(MAX_ABILITIES):
        if 3 + ab_idx < len(mask_raw):
            cm[2 + ab_idx] = mask_raw[3 + ab_idx]
    return cm


# ---------------------------------------------------------------------------
# Actor log probs and entropy helpers
# ---------------------------------------------------------------------------


def compute_all_log_probs(
    output: dict,
    combat_masks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute log probs for ALL actions (for expected Q computation).

    Returns:
        move_logp: (B, 9)
        combat_logp: (B, 10)
        ptr_logp: (B, N) — log probs over pointer targets (attack pointer)
    """
    move_logp = F.log_softmax(output["move_logits"], dim=-1)

    combat_logits = output["combat_logits"].masked_fill(~combat_masks, -1e9)
    combat_logp = F.log_softmax(combat_logits, dim=-1)

    ptr_logp = F.log_softmax(output["attack_ptr"], dim=-1)

    return move_logp, combat_logp, ptr_logp


def compute_entropy(
    output: dict,
    combat_masks: torch.Tensor,
) -> torch.Tensor:
    """Compute total entropy across all heads. Returns (B,)."""
    move_logp = F.log_softmax(output["move_logits"], dim=-1)
    move_ent = -(move_logp.exp() * move_logp).sum(-1)

    combat_logits = output["combat_logits"].masked_fill(~combat_masks, -1e9)
    combat_logp = F.log_softmax(combat_logits, dim=-1)
    combat_ent = -(combat_logp.exp() * combat_logp).sum(-1)

    ptr_logp = F.log_softmax(output["attack_ptr"], dim=-1)
    ptr_ent = -(ptr_logp.exp() * ptr_logp).sum(-1)

    return move_ent + combat_ent + ptr_ent


# ---------------------------------------------------------------------------
# SAC Trainer
# ---------------------------------------------------------------------------


class SACTrainer:
    """SAC-Discrete trainer with factored Q-heads and shared actor encoder.

    The actor encoder provides features (detached) to lightweight twin Q-heads.
    Target Q-heads are soft-updated for stability. Alpha is auto-tuned per the
    SAC formulation with target entropy = -scale * log(action_size) per head.
    """

    def __init__(
        self,
        actor: nn.Module,
        critic_cls=None,  # unused, kept for interface compat
        d_model: int = 128,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 100_000,
        batch_size: int = 256,
        target_entropy_scale: float = 0.5,
        device: str = "cuda",
        reward_scale: float = 1.0,
        max_grad_norm: float = 1.0,
        actor_delay: int = 4,  # update actor every N critic steps
    ):
        self.actor = actor.to(device)
        self.d_model = d_model
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.device = device
        self.reward_scale = reward_scale
        self.max_grad_norm = max_grad_norm
        self.actor_delay = actor_delay
        self.train_steps = 0

        # Twin Q-heads (lightweight — only ~50K params)
        self.critic = SACCriticHeads(d_model).to(device)
        self.critic_target = SACCriticHeads(d_model).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.requires_grad_(False)

        # Auto-tuned temperature alpha (shared across heads)
        # Target entropy (SAC-Discrete, Christodoulou 2019):
        #   H_target = scale * sum(log(action_sizes))
        # This is POSITIVE — we want the policy to maintain some exploration.
        # scale=0.5 means target entropy is ~50% of maximum (uniform).
        target_h_move = target_entropy_scale * math.log(NUM_MOVE_DIRS)
        target_h_combat = target_entropy_scale * math.log(NUM_COMBAT_TYPES)
        target_h_ptr = target_entropy_scale * math.log(7)
        self.target_entropy = target_h_move + target_h_combat + target_h_ptr
        self.log_alpha = torch.zeros(1, device=device, requires_grad=True)
        self.alpha_min = 0.01  # floor to prevent entropy signal from vanishing

        # Optimizers — separate for actor, critic, alpha
        self.actor_optimizer = torch.optim.AdamW(
            [p for p in actor.parameters() if p.requires_grad],
            lr=lr_actor, weight_decay=1.0, betas=(0.9, 0.98),
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(), lr=lr_critic,
            weight_decay=1.0, betas=(0.9, 0.98),
        )
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=lr_critic,
        )

        # Replay buffer (created lazily when episodes are first added)
        self.buffer: ReplayBuffer | None = None
        self.buffer_size = buffer_size

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp().detach()

    # -------------------------------------------------------------------
    # Encoder helpers — work with both V4 and V5 actor models
    # -------------------------------------------------------------------

    def _encode_state(
        self,
        state: dict,
        ability_cls: list[torch.Tensor | None],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run actor encoder, return (pooled, tokens, full_mask).

        Uses encode_state() for V5, or entity_encoder for V4.
        """
        if hasattr(self.actor, "encode_state"):
            # V5 path: EntityEncoderV5 + LatentInterface
            enc = self.actor.encode_state(
                state["entity_features"], state["entity_type_ids"],
                state["threat_features"], state["entity_mask"],
                state["threat_mask"], ability_cls,
                state.get("position_features"),
                state.get("position_mask"),
                state.get("aggregate_features"),
            )
            return enc["pooled"], enc["tokens"], enc["full_mask"]
        else:
            # V4 path: EntityEncoderV3
            tokens, full_mask = self.actor.entity_encoder(
                state["entity_features"], state["entity_type_ids"],
                state["threat_features"], state["entity_mask"],
                state["threat_mask"],
                state.get("position_features"),
                state.get("position_mask"),
            )
            exist = (~full_mask).float().unsqueeze(-1)
            pooled = (tokens * exist).sum(1) / exist.sum(1).clamp(min=1)
            return pooled, tokens, full_mask

    def _actor_decide(
        self,
        pooled: torch.Tensor,
        tokens: torch.Tensor,
        full_mask: torch.Tensor,
        ability_cls: list[torch.Tensor | None],
        state: dict,
    ) -> dict:
        """Run actor decision heads. Returns output dict with logits."""
        if hasattr(self.actor, "decide"):
            # V5: build ability_cross_embs + full_type_ids, call decide()
            ability_cross_embs = []
            for i in range(MAX_ABILITIES):
                if ability_cls[i] is not None:
                    cls_i = self.actor.project_cls(ability_cls[i])
                    cross_emb = self.actor.cross_attn(cls_i, tokens, full_mask)
                    ability_cross_embs.append(cross_emb)
                else:
                    ability_cross_embs.append(None)

            n_threats = state["threat_features"].shape[1]
            n_pos = state["position_features"].shape[1] if state.get("position_features") is not None else 0
            has_agg = state.get("aggregate_features") is not None
            full_type_ids = self.actor._build_full_type_ids(
                state["entity_type_ids"], n_threats, n_pos, has_agg,
                state["entity_features"].device,
            )
            output = self.actor.decide(
                pooled, tokens, full_mask, ability_cross_embs, full_type_ids,
                aggregate_features=state.get("aggregate_features"),
            )
            return output
        else:
            # V4: full forward, discard value + hidden state
            output, _value, _h = self.actor(
                state["entity_features"], state["entity_type_ids"],
                state["threat_features"], state["entity_mask"],
                state["threat_mask"], ability_cls,
                state.get("position_features"),
                state.get("position_mask"),
            )
            return output

    # -------------------------------------------------------------------
    # Core SAC update
    # -------------------------------------------------------------------

    def train_step(self) -> dict:
        """One SAC update step. Returns metrics dict.

        1. Sample batch from replay buffer
        2. Compute target Q via next-state policy + target critic
        3. Update critic heads (Huber loss)
        4. Update actor (maximize expected Q - alpha * log_pi)
        5. Update alpha (temperature auto-tuning)
        6. Soft-update target critic
        """
        if self.buffer is None or self.buffer.size < self.batch_size:
            return {}

        batch = self.buffer.sample(self.batch_size, self.device)
        alpha = self.log_alpha.exp()

        # ---------------------------------------------------------------
        # 1. Encode current and next states (no grad — for critic)
        # ---------------------------------------------------------------
        with torch.no_grad():
            pooled, tokens, full_mask = self._encode_state(
                batch["state"], batch["ability_cls"],
            )
            pooled_det = pooled.detach()
            tokens_det = tokens.detach()

            pooled_next, tokens_next, full_mask_next = self._encode_state(
                batch["next_state"], batch["next_ability_cls"],
            )
            pooled_next_det = pooled_next.detach()
            tokens_next_det = tokens_next.detach()

        # ---------------------------------------------------------------
        # 2. Compute target Q-values
        # ---------------------------------------------------------------
        with torch.no_grad():
            # Next-state policy from actor
            next_output = self._actor_decide(
                pooled_next_det, tokens_next_det, full_mask_next,
                batch["next_ability_cls"], batch["next_state"],
            )
            next_move_logp, next_combat_logp, next_ptr_logp = compute_all_log_probs(
                next_output, batch["combat_masks"],
            )

            # Target Q-values
            tgt_q = self.critic_target(pooled_next_det, tokens_next_det, full_mask_next)

            # Expected Q under next policy: E_a'[Q(s',a') - alpha * log pi(a'|s')]
            # Move
            min_q_move = torch.min(tgt_q["q1_move"], tgt_q["q2_move"]).clamp(-Q_CLAMP, Q_CLAMP)
            v_move = (next_move_logp.exp() * (min_q_move - alpha * next_move_logp)).sum(-1)

            # Combat
            min_q_combat = torch.min(tgt_q["q1_combat"], tgt_q["q2_combat"]).clamp(-Q_CLAMP, Q_CLAMP)
            v_combat = (next_combat_logp.exp() * (min_q_combat - alpha * next_combat_logp)).sum(-1)

            # Pointer
            min_q_ptr = torch.min(tgt_q["q1_ptr"], tgt_q["q2_ptr"]).clamp(-Q_CLAMP, Q_CLAMP)
            v_ptr = (next_ptr_logp.exp() * (min_q_ptr - alpha * next_ptr_logp)).sum(-1)

            v_next = v_move + v_combat + v_ptr

            # Bellman target
            rewards = batch["rewards"] * self.reward_scale
            dones = batch["dones"].float()
            y = (rewards + (1.0 - dones) * self.gamma * v_next).clamp(-Q_CLAMP, Q_CLAMP)

        # ---------------------------------------------------------------
        # 3. Update critics (Huber loss)
        # ---------------------------------------------------------------
        critic_out = self.critic(pooled_det, tokens_det, full_mask)

        move_dirs = batch["move_dirs"]
        combat_types = batch["combat_types"]
        target_indices = batch["target_indices"].clamp(max=tokens_det.shape[1] - 1)

        q1_move_sel = critic_out["q1_move"].gather(1, move_dirs.unsqueeze(1)).squeeze(1)
        q2_move_sel = critic_out["q2_move"].gather(1, move_dirs.unsqueeze(1)).squeeze(1)
        q1_combat_sel = critic_out["q1_combat"].gather(1, combat_types.unsqueeze(1)).squeeze(1)
        q2_combat_sel = critic_out["q2_combat"].gather(1, combat_types.unsqueeze(1)).squeeze(1)
        q1_ptr_sel = critic_out["q1_ptr"].gather(1, target_indices.unsqueeze(1)).squeeze(1)
        q2_ptr_sel = critic_out["q2_ptr"].gather(1, target_indices.unsqueeze(1)).squeeze(1)

        # Factored Q: Q_total = Q_move + Q_combat + Q_ptr
        q1_total = (q1_move_sel + q1_combat_sel + q1_ptr_sel).clamp(-Q_CLAMP, Q_CLAMP)
        q2_total = (q2_move_sel + q2_combat_sel + q2_ptr_sel).clamp(-Q_CLAMP, Q_CLAMP)

        critic_loss = F.huber_loss(q1_total, y) + F.huber_loss(q2_total, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # ---------------------------------------------------------------
        # 4. Update actor (delayed — every actor_delay critic steps)
        # ---------------------------------------------------------------
        actor_loss = torch.tensor(0.0)
        mean_entropy = torch.tensor(0.0)
        if self.train_steps % self.actor_delay == 0:
            # Re-encode with grad through actor encoder
            pooled_actor, tokens_actor, full_mask_actor = self._encode_state(
                batch["state"], batch["ability_cls"],
            )
            output = self._actor_decide(
                pooled_actor, tokens_actor, full_mask_actor,
                batch["ability_cls"], batch["state"],
            )

            curr_move_logp, curr_combat_logp, curr_ptr_logp = compute_all_log_probs(
                output, batch["combat_masks"],
            )

            # Critic Q-values for actor loss (detach from critic grad graph)
            with torch.no_grad():
                critic_for_actor = self.critic(
                    pooled_actor.detach(), tokens_actor.detach(), full_mask_actor,
                )

            min_q_move_a = torch.min(critic_for_actor["q1_move"], critic_for_actor["q2_move"])
            min_q_combat_a = torch.min(critic_for_actor["q1_combat"], critic_for_actor["q2_combat"])
            min_q_ptr_a = torch.min(critic_for_actor["q1_ptr"], critic_for_actor["q2_ptr"])

            # Actor loss: E_a[alpha * log pi(a|s) - Q(s,a)] — minimize
            safe_move_lp = curr_move_logp.clamp(min=-20)
            safe_combat_lp = curr_combat_logp.clamp(min=-20)
            safe_ptr_lp = curr_ptr_logp.clamp(min=-20)
            actor_loss_move = (safe_move_lp.exp() * (alpha * safe_move_lp - min_q_move_a.clamp(-Q_CLAMP, Q_CLAMP))).sum(-1)
            actor_loss_combat = (safe_combat_lp.exp() * (alpha * safe_combat_lp - min_q_combat_a.clamp(-Q_CLAMP, Q_CLAMP))).sum(-1)
            actor_loss_ptr = (safe_ptr_lp.exp() * (alpha * safe_ptr_lp - min_q_ptr_a.clamp(-Q_CLAMP, Q_CLAMP))).sum(-1)
            actor_loss = (actor_loss_move + actor_loss_combat + actor_loss_ptr).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

            # ---------------------------------------------------------------
            # 5. Update alpha (temperature) — also delayed with actor
            # ---------------------------------------------------------------
            with torch.no_grad():
                entropy = compute_entropy(output, batch["combat_masks"])
            mean_entropy = entropy.mean()

            alpha_loss = self.log_alpha.exp() * (self.target_entropy - mean_entropy.detach())

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            # Clamp alpha to minimum floor
            with torch.no_grad():
                self.log_alpha.clamp_(min=math.log(self.alpha_min))

        # ---------------------------------------------------------------
        # 6. Soft update target networks
        # ---------------------------------------------------------------
        with torch.no_grad():
            for param, tgt_param in zip(self.critic.parameters(),
                                        self.critic_target.parameters()):
                tgt_param.data.mul_(1.0 - self.tau).add_(param.data, alpha=self.tau)

        self.train_steps += 1

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "alpha": alpha.item(),
            "entropy": mean_entropy.item(),
            "q_mean": ((q1_total + q2_total) / 2).mean().item(),
            "reward_mean": rewards.mean().item(),
        }

    def train_n_steps(self, n: int = 1000) -> dict:
        """Run n training steps, return averaged metrics."""
        accum: dict[str, float] = {}
        count = 0

        for _ in range(n):
            m = self.train_step()
            if not m:
                continue
            for k, v in m.items():
                accum[k] = accum.get(k, 0.0) + v
            count += 1

        if count == 0:
            return {}

        return {k: v / count for k, v in accum.items()}

    # -------------------------------------------------------------------
    # Episode loading
    # -------------------------------------------------------------------

    def add_episodes(
        self,
        episodes_path: str | Path,
        embedding_registry: dict | None,
        cls_dim: int,
        reward_scale: float = 1.0,
    ):
        """Load episodes from JSONL, convert to transitions, add to replay buffer.

        Each episode contains sequential steps per unit. Consecutive steps from
        the same unit form (s, a, r, s') transitions. The terminal reward from
        the episode is added to the last transition's reward.
        """
        episodes_path = Path(episodes_path)

        # Pre-convert embedding registry to numpy for fast lookup
        embs_np: dict[str, np.ndarray] = {}
        if embedding_registry:
            for k, v in embedding_registry["embeddings"].items():
                if isinstance(v, torch.Tensor):
                    embs_np[k] = v.cpu().numpy()
                elif isinstance(v, np.ndarray):
                    embs_np[k] = v
                else:
                    embs_np[k] = np.array(v, dtype=np.float32)

        # First pass: scan dimensions + collect trajectories
        trajectories: list[dict] = []
        max_ents = 7
        max_threats = 1
        max_positions = 1
        has_agg = False

        with open(episodes_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ep = json.loads(line)
                is_terminal = ep.get("outcome", "") in ("Victory", "Defeat")
                ab_names_map = ep.get("unit_ability_names", {})

                unit_steps: dict[int, list[dict]] = defaultdict(list)
                for step in ep["steps"]:
                    if step.get("move_dir") is None:
                        continue
                    unit_steps[step["unit_id"]].append(step)

                    # Track max dims
                    ents = step.get("entities", [])
                    max_ents = max(max_ents, len(ents))
                    threats = step.get("threats", [])
                    max_threats = max(max_threats, len(threats))
                    positions = step.get("positions", [])
                    max_positions = max(max_positions, len(positions))
                    if step.get("aggregate_features"):
                        has_agg = True

                for uid, steps in unit_steps.items():
                    steps.sort(key=lambda s: s["tick"])
                    uid_str = str(uid)
                    names = ab_names_map.get(uid_str, [])
                    trajectories.append({
                        "steps": steps,
                        "terminal": is_terminal,
                        "episode_reward": ep.get("reward", 0.0),
                        "ability_names": names,
                    })

        if not trajectories:
            print(f"  No trajectories in {episodes_path}")
            return

        max_threats = max(max_threats, 1)
        max_positions = max(max_positions, 1)

        # Create buffer if needed
        if self.buffer is None:
            self.buffer = ReplayBuffer(
                capacity=self.buffer_size,
                max_ents=max_ents,
                max_threats=max_threats,
                max_positions=max_positions,
                cls_dim=cls_dim,
                has_aggregate=has_agg,
            )

        # Resolve ability CLS helper
        def resolve_cls(names: list[str]) -> list:
            result: list = [None] * MAX_ABILITIES
            if not embs_np:
                return result
            for ab_idx in range(min(MAX_ABILITIES, len(names))):
                if names[ab_idx]:
                    lookup = names[ab_idx].replace(" ", "_")
                    if lookup in embs_np:
                        result[ab_idx] = embs_np[lookup]
            return result

        # Second pass: add transitions
        n_transitions = 0
        for traj in trajectories:
            steps = traj["steps"]
            names = traj["ability_names"]
            ab_cls = resolve_cls(names)

            for i in range(len(steps) - 1):
                s = steps[i]
                s_next = steps[i + 1]

                combat_mask = _build_combat_mask_single(s)
                done = (i == len(steps) - 2) and traj["terminal"]

                # Reward: shaped step reward only (outcome bonus already included
                # in the final step by transform_drill_rewards)
                reward = s.get("step_reward", 0.0) * reward_scale

                self.buffer.add(
                    step=s,
                    next_step=s_next,
                    ability_cls_list=ab_cls,
                    next_ability_cls_list=ab_cls,  # same unit, same abilities
                    move_dir=s.get("move_dir", 0),
                    combat_type=s.get("combat_type", 0),
                    target_idx=s.get("target_idx", 0),
                    combat_mask=combat_mask,
                    reward=reward,
                    done=done,
                )
                n_transitions += 1

        print(f"  Added {n_transitions} transitions from {len(trajectories)} trajectories, "
              f"buffer: {self.buffer.size}/{self.buffer.capacity}")

    # -------------------------------------------------------------------
    # GPU server sync
    # -------------------------------------------------------------------

    def sync_actor_to_gpu(self, checkpoint_path: str, shm_name: str = "impala_inf"):
        """Save actor weights and signal GPU inference server to reload."""
        checkpoint_path = str(Path(checkpoint_path).resolve())
        torch.save({"model_state_dict": self.actor.state_dict()}, checkpoint_path)

        shm_path = f"{SHM_PATH_PREFIX}{shm_name}"
        if not os.path.exists(shm_path):
            print(f"  GPU server SHM not found at {shm_path}, skipping reload")
            return

        import mmap as mmap_mod
        abs_bytes = checkpoint_path.encode("utf-8")[:RELOAD_PATH_LEN - 1] + b"\x00"

        fd = os.open(shm_path, os.O_RDWR)
        mm = mmap_mod.mmap(fd, 0)
        os.close(fd)

        mm[OFF_RELOAD_PATH:OFF_RELOAD_PATH + len(abs_bytes)] = abs_bytes
        struct.pack_into("<I", mm, OFF_RELOAD_ACK, 0)
        struct.pack_into("<I", mm, OFF_RELOAD_REQ, 1)

        for _ in range(100_000):
            ack = struct.unpack_from("<I", mm, OFF_RELOAD_ACK)[0]
            if ack != 0:
                mm.close()
                print(f"  GPU server reloaded from {checkpoint_path}")
                return
            time.sleep(50e-6)

        mm.close()
        print("  WARNING: GPU server reload timed out")

    # -------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------

    def save_checkpoint(self, path: str):
        """Save full training state (actor, critics, optimizers, alpha)."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "train_steps": self.train_steps,
        }, path)

    def load_checkpoint(self, path: str):
        """Load full training state."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.critic_target.load_state_dict(ckpt["critic_target"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer"])
        self.alpha_optimizer.load_state_dict(ckpt["alpha_optimizer"])
        self.log_alpha.data.copy_(ckpt["log_alpha"].to(self.device))
        self.train_steps = ckpt.get("train_steps", 0)
        print(f"  Loaded SAC checkpoint from {path} (step {self.train_steps})")
