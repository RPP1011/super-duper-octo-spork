# Spatial Awareness: Geometry Corner Tokens

## What it does

Extracts structural features of the room geometry and encodes them as spatial tokens that can feed into the entity encoder's cross-attention. Each unit gets a per-unit view: only corners visible from its current position are included.

## How it works

**Corner extraction** (`extract_corners`): Scans the grid for walkable cells adjacent to walls where the boundary changes direction (L-shapes, dead ends, freestanding wall tips). Each corner has:
- Position (world-space)
- Convex/concave (wall sticking out vs receding — cover vs exposed)
- Blocked neighbor count (1-3, enclosure level)
- Opening direction (unit vector into open space)
- Passage width (how wide the opening is at this point)
- Elevation

Corners are sorted by passage width (narrowest first = most tactically relevant). Capped at 16 per room.

**Precomputed visibility** (`VisibilityMap`): Built once per room. For every walkable cell, stores a u32 bitset of which corners are visible via Bresenham LOS checks. Lookup at tick time is a single HashMap get.

**Dynamic obstacles** (`update_obstacle_placed` / `update_obstacle_removed`): When a barricade is placed, new corners are detected at the obstacle edges, existing corner passage widths are updated, and visibility is recomputed for cells within a radius. When the obstacle expires, the process reverses. Geometry changes bump a generation counter that invalidates the spatial cache.

**Per-unit cache** (`SpatialCache`): Stores computed spatial tokens per unit. Only recomputes when the unit moves to a different grid cell or geometry changes. On cache hit, returns immediately.

## Token format

Each spatial token is 11 floats (`SPATIAL_TOKEN_DIM`):

| Index | Feature | Range |
|-------|---------|-------|
| 0-1 | Corner position (normalized 0-1 in room) | [0, 1] |
| 2 | Convex (1) vs concave (0) | {0, 1} |
| 3 | Enclosure level (blocked_neighbors / 3) | [0, 1] |
| 4-5 | Opening direction (unit vector) | [-1, 1] |
| 6 | Passage width (normalized, capped) | [0, 1] |
| 7 | Elevation (normalized) | [0, 1] |
| 8-9 | Relative direction from unit to corner | [-1, 1] |
| 10 | Normalized distance from unit to corner | [0, 1] |

Features 0-7 are static per corner. Features 8-10 are per-unit (relative to the querying unit's position).

## Performance

| Operation | Cost | Frequency |
|-----------|------|-----------|
| Room build (`VisibilityMap::build`) | ~500us | Once per room |
| Cache hit (same cell, no geometry change) | ~20ns | Most decision ticks |
| Cache miss (unit moved cells) | ~150ns | When unit crosses cell boundary |
| Obstacle placed/removed | ~300us | Per barricade event (~every 8s) |

## Threat tokens (zones, cast indicators, projectiles)

Ephemeral spatial tokens extracted from live game state every decision tick.
Unlike geometry corners (static or slowly changing), these represent active
threats and opportunities that units need to react to.

**Zone tokens**: Active zones on the ground — damage fields, healing circles,
barricade obstacles. Extracted from `SimState.zones`. Includes position,
radius, hostile flag, remaining duration.

**Cast indicator tokens**: Abilities being cast with a ground target position
(`CastState.target_pos`). This is the "telegraph" — shows where an AoE is
about to land and how long until it fires. Extracted from units with active
`casting` state that has a `target_pos`.

**Projectile tokens**: Projectiles in flight with position, direction, width.
Extracted from `SimState.projectiles`.

Each threat token is 10 floats (`THREAT_TOKEN_DIM`):

| Index | Feature | Range |
|-------|---------|-------|
| 0-1 | Position (normalized 0-1 in room) | [0, 1] |
| 2-3 | Relative direction from unit | [-1, 1] |
| 4 | Normalized distance from unit | [0, 1] |
| 5 | Radius (normalized) | [0, 1] |
| 6 | Hostile flag | {0, 1} |
| 7 | Duration remaining (seconds, capped) | [0, 1] |
| 8 | Kind (zone=0.25, obstacle=0.5, cast=0.75, projectile=1.0) | [0, 1] |
| 9 | Line-of-sight visibility from unit | {0, 1} |

## Files

- `src/ai/goap/spatial.rs` — all implementation (corner extraction, visibility map, dynamic obstacles, threat tokens, cache)
- `docs/plans/spatial_encoder.md` — original design doc (sector map approach, partially superseded by corner approach)
