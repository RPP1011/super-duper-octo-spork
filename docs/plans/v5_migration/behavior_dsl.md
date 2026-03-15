# Behavior DSL Grammar

A declarative language for defining unit AI behaviors. Compiles to a tick function
that produces a single `IntentAction` each game tick. Integrates with the ability
eval system for smart ability usage while allowing full manual control when needed.

## Design Principles

1. **Priority-ordered rules** — evaluated top-down, first match wins
2. **Composable** — behaviors reference other behaviors, L1 primitives compose into L2 patterns
3. **Ability eval integration** — `use_best_ability` delegates to the existing 9-category eval system
4. **Hardcoded effects** — can directly specify damage, heal, CC amounts for training dummies
5. **Reusable** — same DSL for drill opponents, game NPCs, boss AI, and default squad behavior

---

## Grammar (EBNF-ish)

```ebnf
program        = behavior_def+

behavior_def   = "behavior" STRING "{" rule+ "}"

rule           = priority_rule | default_rule | fallback_rule

priority_rule  = "priority" action ("when" condition)? ("cooldown" INT)?
default_rule   = "default" action ("when" condition)?
fallback_rule  = "fallback" action

(* ── Actions ── *)

action         = move_action
               | attack_action
               | ability_action
               | eval_action
               | effect_action
               | composite_action
               | "hold"
               | "run" STRING          (* invoke another behavior by name *)

(* L1: Movement primitives *)
move_action    = "chase" target
               | "flee" target
               | "move_to" position
               | "maintain_distance" target NUMBER
               | "orbit" target NUMBER             (* circle at distance *)
               | "move_dir" direction
               | "patrol" position ("," position)+ (* cycle through waypoints *)
               | "body_block" target target        (* stand between target1 and target2 *)

(* L1: Attack primitives *)
attack_action  = "attack" target
               | "focus" target                    (* attack + chase if out of range *)

(* L1: Ability primitives — manual ability selection *)
ability_action = "cast" ABILITY_SLOT "on" target          (* cast specific ability slot 0-7 *)
               | "cast" ABILITY_SLOT "at" position        (* ground-targeted *)
               | "cast_if_ready" ABILITY_SLOT "on" target (* only if off cooldown *)

(* L1: Ability eval integration — delegates to existing system *)
eval_action    = "use_best_ability"                             (* full ability eval, pick highest urgency *)
               | "use_best_ability" "on" target                 (* eval but force target *)
               | "use_ability_type" ABILITY_TYPE                (* eval within category: damage|heal|cc|buff *)
               | "use_ability_type" ABILITY_TYPE "on" target

(* L1: Hardcoded effect primitives — for training dummies and scripted NPCs *)
effect_action  = "deal_damage" NUMBER "to" target               (* instant damage *)
               | "heal_amount" NUMBER "on" target               (* instant heal *)
               | "apply_cc" CC_TYPE NUMBER "to" target          (* stun/root/slow for N ms *)
               | "spawn_zone" ZONE_TYPE NUMBER "at" position "for" NUMBER  (* AoE zone: type, radius, pos, duration *)
               | "knockback" target NUMBER                      (* push target N units away *)
               | "telegraph" position NUMBER "for" NUMBER       (* warning indicator: pos, radius, ticks *)
                 "then" effect_action                           (* ... then apply this effect *)

(* L2: Composite patterns — named combinations of L1 primitives *)
composite_action = "combo" "{" rule+ "}"                        (* inline sub-behavior *)
               | "sequence" "{" seq_step+ "}"                   (* ordered steps with waits *)

seq_step       = action ("wait" NUMBER)?                        (* action then wait N ticks *)

(* ── Targets ── *)

target         = entity_target | position_target

entity_target  = "self"
               | "nearest_enemy"
               | "nearest_ally"
               | "lowest_hp_enemy"
               | "lowest_hp_ally"
               | "highest_dps_enemy"
               | "highest_threat_enemy"     (* uses threat = dps * (1 / distance) *)
               | "casting_enemy"            (* enemy with is_casting == 1 *)
               | "enemy_attacking" target   (* enemy whose target is X *)
               | "unit" INT                 (* specific unit by ID *)
               | "tagged" STRING            (* unit with a named tag, set in scenario TOML *)

position_target = "position" NUMBER NUMBER
               | "random_position"
               | "target_position"          (* drill objective marker *)

position       = position_target
               | entity_target              (* uses entity's position *)

direction      = "N" | "NE" | "E" | "SE" | "S" | "SW" | "W" | "NW" | "random"

(* ── Conditions ── *)

condition      = condition "and" condition
               | condition "or" condition
               | "not" condition
               | "(" condition ")"
               | comparison
               | state_check

comparison     = value COMP_OP value

value          = NUMBER
               | "self.hp"
               | "self.hp_pct"
               | "self.mana" | "self.mana_pct"
               | "self.move_speed"
               | target ".hp" | target ".hp_pct"
               | target ".dps"
               | target ".distance"             (* distance from self to target *)
               | target ".attack_range"
               | target ".cc_remaining"
               | target ".cast_progress"
               | "ally_count"
               | "enemy_count"
               | "enemy_count_in_range" NUMBER  (* enemies within N units *)
               | "ally_count_below_hp" NUMBER   (* allies below X% HP *)
               | "tick"                         (* current game tick *)
               | "elapsed"                      (* ticks since behavior started *)
               | "ability" ABILITY_SLOT ".cooldown_pct"
               | "ability" ABILITY_SLOT ".range"
               | "best_ability_urgency"         (* from ability eval system *)

state_check    = "heal_ready"                   (* any heal ability off cooldown *)
               | "stun_ready" | "cc_ready"      (* any CC ability off cooldown *)
               | "aoe_ready"                    (* any AoE ability off cooldown *)
               | "ability_ready" ABILITY_SLOT   (* specific slot off cooldown *)
               | "is_casting"
               | "is_cc_d"                      (* self is CC'd *)
               | "target_is_casting" target
               | "target_is_cc_d" target
               | "can_attack"                   (* attack off cooldown + enemy in range *)
               | "in_danger_zone"               (* standing in hostile zone *)
               | "ally_in_danger"               (* any ally in hostile zone or < 30% HP *)
               | "has_line_of_sight" target     (* clear line to target, no walls *)
               | "near_wall" NUMBER             (* within N units of a wall *)
               | "every" NUMBER                 (* true every N ticks, for periodic actions *)

COMP_OP        = "<" | ">" | "<=" | ">=" | "==" | "!="
NUMBER         = float literal
INT            = integer literal
STRING         = quoted string
ABILITY_SLOT   = "ability0" | "ability1" | ... | "ability7"
ABILITY_TYPE   = "damage" | "heal" | "cc" | "buff" | "debuff" | "movement" | "aoe"
CC_TYPE        = "stun" | "root" | "slow" | "silence" | "knockback"
ZONE_TYPE      = "damage" | "heal" | "slow"
```

---

## Examples

### Training Dummy (Phase 3.1)
```
behavior "stationary_dummy" {
  fallback hold
}
```

### Moving Target (Phase 3.2)
```
behavior "fleeing_target" {
  default flee(nearest_enemy)
  fallback move_dir random
}
```

### Melee Chaser (Phase 2.1)
```
behavior "melee_chaser" {
  priority attack(nearest_enemy) when nearest_enemy.distance < 1.5
  default chase(nearest_enemy)
  fallback hold
}
```

### Healer Bot (Phase 3.5, 4.1)
```
behavior "healer_bot" {
  priority use_ability_type heal on lowest_hp_ally when ally_count_below_hp 0.5 > 0 and heal_ready
  priority attack(enemy_attacking lowest_hp_ally) when ally_count_below_hp 0.3 > 0
  default attack(nearest_enemy) when can_attack
  default chase(nearest_enemy)
  fallback hold
}
```

### AoE Caster (Phase 2.3)
```
behavior "telegraphed_aoe_caster" {
  priority sequence {
    telegraph nearest_enemy 3.0 for 30
    then deal_damage 100 to nearest_enemy
  } when every 60
  default maintain_distance(nearest_enemy, 8)
  fallback hold
}
```

### Kite Dummy (Phase 2.4)
```
behavior "melee_with_lunge" {
  priority attack(nearest_enemy) when nearest_enemy.distance < 1.5
  default chase(nearest_enemy)
  fallback hold
}
```

### Tank + Healer Pair (Phase 3.8)
```
behavior "tank_guard" {
  priority body_block(nearest_enemy, tagged "healer")
  priority attack(enemy_attacking tagged "healer") when tagged "healer".hp_pct < 0.8
  default attack(nearest_enemy) when can_attack
  default chase(nearest_enemy)
  fallback hold
}

behavior "pocket_healer" {
  priority use_ability_type heal on tagged "tank" when tagged "tank".hp_pct < 0.6 and heal_ready
  priority flee(nearest_enemy) when nearest_enemy.distance < 3
  default maintain_distance(tagged "tank", 4)
  fallback hold
}
```

### CC Gatekeeper (Phase 4.8)
```
behavior "cc_gatekeeper" {
  priority cast_if_ready ability0 on nearest_enemy when nearest_enemy.distance < 4
  priority attack(nearest_enemy) when can_attack
  default chase(nearest_enemy)
  fallback hold
}
```

### Boss with Rotation (game NPC)
```
behavior "flame_lord" {
  priority sequence {
    telegraph self 5.0 for 45
    then spawn_zone damage 5.0 at self for 120
  } when every 200 and enemy_count_in_range 6.0 > 2

  priority cast_if_ready ability0 on lowest_hp_enemy when ability_ready ability0
  priority cast_if_ready ability1 on nearest_enemy when nearest_enemy.distance < 3

  default use_best_ability
  default attack(highest_threat_enemy) when can_attack
  default chase(nearest_enemy)
  fallback hold
}
```

### Smart Default AI (replacement for current squad AI)
```
behavior "default_fighter" {
  priority use_best_ability when best_ability_urgency > 0.4
  priority attack(lowest_hp_enemy) when can_attack and lowest_hp_enemy.hp_pct < 0.3
  priority attack(enemy_attacking lowest_hp_ally) when ally_in_danger
  default attack(nearest_enemy) when can_attack
  default chase(nearest_enemy)
  fallback hold
}
```

---

## Ability Eval Integration

The `use_best_ability` action delegates to the existing ability eval system
(`src/ai/core/ability_eval.rs`). This means:
- All 9 ability categories are evaluated (single damage, AoE, heal, CC, etc.)
- Per-ability urgency scores computed from game state
- Terrain awareness (knockback into wall detection) included
- The behavior DSL decides WHEN to consider abilities; the eval decides WHICH ability

`use_ability_type` narrows the eval to one category — e.g., `use_ability_type heal`
only considers healing abilities from the eval system.

`best_ability_urgency` exposes the eval score as a condition value, so behaviors
can gate ability usage on the eval's confidence: "only use abilities when the eval
is confident one is useful."

---

## Hardcoded Effects (L1 Primitives)

For training drills, we need dummy enemies that produce specific effects
without real abilities:

- `deal_damage 50 to target` — bypass the ability system, just apply damage
- `heal_amount 30 on target` — direct heal
- `apply_cc stun 1500 to target` — apply CC directly
- `spawn_zone damage 3.0 at position for 120` — create a damage zone
- `knockback target 4` — push target away
- `telegraph position 3.0 for 30 then effect` — show warning, then apply

These are implemented as sim-level commands, not routed through the ability system.
This lets us create precise drill scenarios without needing to define full abilities.

---

## Scenario TOML Integration

Behaviors are referenced in scenario files:

```toml
[scenario]
name = "drill_2_3_dodge_aoe"
hero_count = 1
enemy_count = 1
hero_templates = ["scout"]
room_type = "Pressure"
max_ticks = 500

[[enemies]]
behavior = "telegraphed_aoe_caster"
hp = 9999
dps = 0
position = [10.0, 10.0]
tag = "caster"

[[objectives]]
type = "survive"
duration = 500
damage_taken_max = 0
```

Or for tagged multi-unit drills:

```toml
[[enemies]]
behavior = "tank_guard"
template = "brute"
tag = "tank"

[[enemies]]
behavior = "pocket_healer"
template = "cleric"
tag = "healer"
```

---

## Implementation Plan

### Phase 1: Parser
- `src/ai/behavior/parser.rs` — tokenizer + recursive descent parser
- Input: behavior DSL text → Output: `BehaviorTree` (Vec of prioritized rules)
- Each rule: `(Option<Condition>, Action)`

### Phase 2: Interpreter
- `src/ai/behavior/interpreter.rs` — tick function
- `fn evaluate(tree: &BehaviorTree, sim: &SimState, unit_id: u32) -> IntentAction`
- Evaluates conditions against current sim state
- Returns first matching action as an IntentAction (same type the squad AI uses)

### Phase 3: Effect primitives
- `src/ai/behavior/effects.rs` — hardcoded effect execution
- Hooks into sim's damage/heal/CC application directly
- Used by `deal_damage`, `heal_amount`, `apply_cc`, `spawn_zone`, `telegraph`

### Phase 4: Scenario integration
- Load `.behavior` files from `assets/behaviors/`
- Reference in scenario TOML via `behavior = "name"`
- Override default squad AI per-unit when behavior is set

### Phase 5: Ability eval bridge
- `use_best_ability` calls existing `evaluate_abilities()` from `ability_eval.rs`
- `best_ability_urgency` reads the max urgency score
- `use_ability_type` filters eval results by category
