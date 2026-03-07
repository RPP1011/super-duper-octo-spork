"""Tokenizer for the ability DSL.

Converts raw .ability DSL strings into integer token sequences suitable for
transformer training.  The vocabulary is closed — every terminal symbol in the
grammar maps to a fixed token ID.  Numbers are bucketed into semantic ranges,
ability names and arbitrary string literals are replaced with placeholder tokens.

Usage:
    from tokenizer import AbilityTokenizer
    tok = AbilityTokenizer()
    ids = tok.encode("ability Fireball { target: enemy, range: 5.0 ... }")
    text = tok.decode(ids)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

# Special tokens
PAD = "[PAD]"
CLS = "[CLS]"
MASK = "[MASK]"
SEP = "[SEP]"
UNK = "[UNK]"
NAME = "[NAME]"    # ability/passive names (arbitrary identifiers)
STR = "[STR]"      # quoted string literals
TAG = "[TAG]"      # tag names inside [ ] brackets (e.g. FIRE, ICE)

SPECIAL_TOKENS = [PAD, CLS, MASK, SEP, UNK, NAME, STR, TAG]

# Punctuation
PUNCTUATION = ["{", "}", "(", ")", "[", "]", ":", ",", "+", "%"]

# All DSL keywords — flat list, context-free.  The transformer learns from
# positional context whether "damage" is a hint or an effect.
KEYWORDS = sorted(set([
    # top-level
    "ability", "passive",
    # properties
    "target", "range", "cooldown", "cast", "hint", "cost", "charges",
    "recharge", "recast", "recast_window", "unstoppable", "toggle",
    "form", "swap_form",
    # targeting
    "enemy", "ally", "self", "self_aoe", "self_cast", "target_enemy",
    "target_ally", "ground", "ground_target", "direction", "vector", "global",
    # delivery
    "deliver", "instant", "projectile", "channel", "zone", "tether", "trap",
    "chain",
    # delivery params
    "speed", "pierce", "width", "duration", "tick", "trigger_radius",
    "arm_time", "max_range", "bounces", "bounce_range", "falloff",
    # hooks
    "on_hit", "on_arrival", "on_complete", "on_hit_buff",
    # area
    "in", "circle", "cone", "line", "ring", "spread",
    # area params
    "radius", "angle_deg", "inner_radius", "outer_radius", "length",
    "max_targets",
    # effects
    "damage", "heal", "shield", "stun", "slow", "knockback", "dash", "buff",
    "debuff", "duel", "summon", "command_summons", "dispel", "root", "silence",
    "fear", "taunt", "pull", "swap", "reflect", "lifesteal", "damage_modify",
    "self_damage", "execute", "blind", "resurrect", "overheal_shield",
    "absorb_to_heal", "shield_steal", "status_clone", "immunity", "detonate",
    "status_transfer", "death_mark", "polymorph", "banish", "confuse", "charm",
    "stealth", "leash", "link", "redirect", "rewind", "cooldown_modify",
    "apply_stacks", "obstacle", "suppress", "grounded", "projectile_block",
    "attach", "evolve_ability", "blink",
    # effect modifiers
    "for", "to_target", "to_position", "is_blink", "clone",
    "clone_damage_percent", "directed", "break_on_damage", "break_on_ability",
    "share_percent", "lookback_ms", "template", "hp_percent", "max_stacks",
    "max", "stacking", "chance",
    # conditions
    "when", "else", "and", "or", "not",
    "target_hp_below", "target_hp_above",
    "target_is_stunned", "target_is_slowed", "target_is_rooted",
    "target_is_silenced", "target_is_feared", "target_is_taunted",
    "target_is_banished", "target_is_stealthed", "target_is_charmed",
    "target_is_polymorphed",
    "caster_hp_below", "caster_hp_above", "hit_count_above",
    "target_has_tag", "caster_has_status", "target_has_status",
    "target_debuff_count", "caster_buff_count",
    "ally_count_below", "enemy_count_below", "target_stack_count",
    "target_distance_below", "target_distance_above",
    "caster_resource_below", "caster_resource_above",
    # damage types
    "physical", "magic", "true",
    # scaling stats
    "target_max_hp", "target_current_hp", "target_missing_hp",
    "caster_max_hp", "caster_current_hp", "caster_missing_hp",
    "caster_attack_damage", "target_stacks", "caster_stacks", "consume", "cap",
    # stat names (buff/debuff)
    "damage_output", "attack_speed", "move_speed", "cooldown_reduction",
    "armor", "magic_resist",
    # passive triggers
    "on_damage_taken", "on_damage_dealt", "on_kill", "on_death",
    "on_ability_hit", "on_heal", "on_shield_break", "on_cc_applied",
    "on_buff_applied", "on_debuff_applied", "on_stack_reached",
    "on_hp_below", "on_hp_above", "on_ally_death", "on_ally_low_hp",
    "on_interval", "on_cooldown_end", "on_combat_start", "on_combat_end",
    "on_enter_zone",
    # hints (not already covered by effect names)
    "cc", "crowd_control", "defense", "utility",
    # misc
    "x",
]))

# Numeric bucket tokens
NUM_TOKENS = [
    "NUM_0", "NUM_1", "NUM_2", "NUM_3", "NUM_4", "NUM_5",
    "NUM_6", "NUM_7", "NUM_8", "NUM_9", "NUM_10",
    "NUM_SMALL",   # 11–50
    "NUM_MED",     # 51–200
    "NUM_LARGE",   # 201–1000
    "NUM_HUGE",    # 1001+
    "FRAC_TINY",   # 0.01–0.2
    "FRAC_LOW",    # 0.2–0.4
    "FRAC_MID",    # 0.4–0.6
    "FRAC_HIGH",   # 0.6–0.8
    "FRAC_MAX",    # 0.8–1.0
]

# Duration bucket tokens — durations are parsed from "5s" / "1000ms" forms
DUR_TOKENS = [
    "DUR_INSTANT",   # < 200ms
    "DUR_SHORT",     # 200ms – 2s
    "DUR_MED",       # 2s – 8s
    "DUR_LONG",      # 8s – 30s
    "DUR_VLONG",     # > 30s
]


def _build_vocab() -> tuple[dict[str, int], dict[int, str]]:
    """Build token-to-id and id-to-token mappings."""
    tokens = SPECIAL_TOKENS + PUNCTUATION + KEYWORDS + NUM_TOKENS + DUR_TOKENS
    tok2id = {t: i for i, t in enumerate(tokens)}
    id2tok = {i: t for t, i in tok2id.items()}
    return tok2id, id2tok


_TOK2ID, _ID2TOK = _build_vocab()
VOCAB_SIZE = len(_TOK2ID)


# ---------------------------------------------------------------------------
# Number bucketing
# ---------------------------------------------------------------------------

def _bucket_number(value: float) -> str:
    """Map a numeric value to a bucket token."""
    if value == int(value) and 0 <= value <= 10:
        return f"NUM_{int(value)}"
    if 0 < value < 1:
        if value <= 0.2:
            return "FRAC_TINY"
        if value <= 0.4:
            return "FRAC_LOW"
        if value <= 0.6:
            return "FRAC_MID"
        if value <= 0.8:
            return "FRAC_HIGH"
        return "FRAC_MAX"
    v = abs(value)
    if v <= 10:
        return f"NUM_{round(v)}"
    if v <= 50:
        return "NUM_SMALL"
    if v <= 200:
        return "NUM_MED"
    if v <= 1000:
        return "NUM_LARGE"
    return "NUM_HUGE"


def _bucket_duration_ms(ms: float) -> str:
    """Map a duration in milliseconds to a bucket token."""
    if ms < 200:
        return "DUR_INSTANT"
    if ms <= 2000:
        return "DUR_SHORT"
    if ms <= 8000:
        return "DUR_MED"
    if ms <= 30000:
        return "DUR_LONG"
    return "DUR_VLONG"


# ---------------------------------------------------------------------------
# Lexer
# ---------------------------------------------------------------------------

# Regex patterns for lexing — order matters
_COMMENT = re.compile(r"(?://|#).*$", re.MULTILINE)
_STRING = re.compile(r'"([^"]*)"')
_DURATION = re.compile(r"(-?\d+(?:\.\d+)?)(ms|s)\b")
_PERCENT = re.compile(r"(-?\d+(?:\.\d+)?)%")
_NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PUNCT = re.compile(r"[{}()\[\]:,+%]")

# Multiplier pattern: x followed by digits (e.g. x2, x3)
_MULTIPLIER = re.compile(r"\bx(\d+)\b")


@dataclass
class AbilityTokenizer:
    """Tokenizes ability DSL strings into integer sequences."""

    tok2id: dict[str, int] = field(default_factory=lambda: dict(_TOK2ID))
    id2tok: dict[int, str] = field(default_factory=lambda: dict(_ID2TOK))
    max_length: int = 256

    @property
    def vocab_size(self) -> int:
        return len(self.tok2id)

    @property
    def pad_id(self) -> int:
        return self.tok2id[PAD]

    @property
    def cls_id(self) -> int:
        return self.tok2id[CLS]

    @property
    def mask_id(self) -> int:
        return self.tok2id[MASK]

    @property
    def unk_id(self) -> int:
        return self.tok2id[UNK]

    def _lex(self, text: str) -> list[str]:
        """Convert raw DSL text into a list of token strings."""
        # Strip comments
        text = _COMMENT.sub("", text)

        tokens: list[str] = []
        pos = 0
        # Track whether we're right after "ability" or "passive" keyword
        # to replace the name with [NAME]
        expect_name = False
        # Track bracket depth for tag names
        bracket_depth = 0

        while pos < len(text):
            # Skip whitespace
            if text[pos].isspace():
                pos += 1
                continue

            # String literal
            m = _STRING.match(text, pos)
            if m:
                tokens.append(STR)
                pos = m.end()
                expect_name = False
                continue

            # Duration (must check before plain number)
            m = _DURATION.match(text, pos)
            if m:
                value = float(m.group(1))
                suffix = m.group(2)
                ms = value if suffix == "ms" else value * 1000
                tokens.append(_bucket_duration_ms(ms))
                pos = m.end()
                expect_name = False
                continue

            # Percentage (must check before plain number)
            m = _PERCENT.match(text, pos)
            if m:
                value = float(m.group(1))
                tokens.append(_bucket_number(value / 100.0))
                tokens.append("%")
                pos = m.end()
                expect_name = False
                continue

            # Multiplier (x2, x3)
            m = _MULTIPLIER.match(text, pos)
            if m:
                tokens.append("x")
                tokens.append(_bucket_number(int(m.group(1))))
                pos = m.end()
                expect_name = False
                continue

            # Punctuation
            if text[pos] in "{}()[]:,+%":
                ch = text[pos]
                if ch == "[":
                    bracket_depth += 1
                elif ch == "]":
                    bracket_depth = max(0, bracket_depth - 1)
                tokens.append(ch)
                pos += 1
                continue

            # Plain number (after duration/percent checks)
            m = _NUMBER.match(text, pos)
            if m:
                value = float(m.group())
                tokens.append(_bucket_number(value))
                pos = m.end()
                expect_name = False
                continue

            # Identifier or keyword
            m = _IDENT.match(text, pos)
            if m:
                word = m.group()
                pos = m.end()

                if expect_name:
                    # This is an ability/passive name — replace with [NAME]
                    tokens.append(NAME)
                    expect_name = False
                elif bracket_depth > 0 and word not in self.tok2id:
                    # Inside [ ] brackets and not a keyword — it's a tag name
                    tokens.append(TAG)
                elif word in self.tok2id:
                    tokens.append(word)
                    if word in ("ability", "passive"):
                        expect_name = True
                else:
                    # Unknown identifier — could be a stat name or status name
                    tokens.append(UNK)
                continue

            # Skip unknown characters
            pos += 1

        return tokens

    def encode(self, text: str, add_cls: bool = True) -> list[int]:
        """Tokenize DSL text and convert to integer IDs.

        Returns a list of token IDs, optionally prepended with [CLS].
        Truncated to max_length (including [CLS]).
        """
        token_strs = self._lex(text)
        if add_cls:
            token_strs = [CLS] + token_strs

        ids = [self.tok2id.get(t, self.tok2id[UNK]) for t in token_strs]
        return ids[: self.max_length]

    def encode_padded(self, text: str, add_cls: bool = True) -> list[int]:
        """Encode and pad to max_length."""
        ids = self.encode(text, add_cls=add_cls)
        pad_len = self.max_length - len(ids)
        if pad_len > 0:
            ids = ids + [self.pad_id] * pad_len
        return ids

    def decode(self, ids: list[int]) -> str:
        """Convert token IDs back to a readable string."""
        tokens = [self.id2tok.get(i, UNK) for i in ids if i != self.tok2id[PAD]]
        return " ".join(tokens)

    def batch_encode(
        self, texts: list[str], add_cls: bool = True
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Encode a batch of texts, returning (token_ids, attention_masks).

        All sequences are padded to the max length in the batch.
        """
        encoded = [self.encode(t, add_cls=add_cls) for t in texts]
        max_len = min(max(len(e) for e in encoded), self.max_length)

        ids_batch = []
        mask_batch = []
        for ids in encoded:
            ids = ids[:max_len]
            mask = [1] * len(ids)
            pad_len = max_len - len(ids)
            ids = ids + [self.pad_id] * pad_len
            mask = mask + [0] * pad_len
            ids_batch.append(ids)
            mask_batch.append(mask)

        return ids_batch, mask_batch


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    tok = AbilityTokenizer()
    print(f"Vocabulary size: {tok.vocab_size}")

    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if path.is_dir():
            files = sorted(path.glob("*.ability"))[:5]
        else:
            files = [path]
        for f in files:
            text = f.read_text()
            ids = tok.encode(text)
            decoded = tok.decode(ids)
            print(f"\n--- {f.name} ({len(ids)} tokens) ---")
            print(decoded)
    else:
        # Inline example
        example = '''ability Fireball {
            target: enemy, range: 5.0
            cooldown: 5s, cast: 300ms
            hint: damage
            deliver projectile { speed: 8.0, width: 0.3 } {
                on_hit { damage 55 [FIRE: 60] }
            }
        }'''
        ids = tok.encode(example)
        print(f"\nExample ({len(ids)} tokens):")
        print(tok.decode(ids))
