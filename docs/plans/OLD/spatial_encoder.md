# Spatial Encoder Design

## Problem

The entity encoder sees per-unit features but has no geometric understanding of
the battlefield. It can't distinguish an open field from a narrow corridor, can't
identify choke points or cover positions, and can't reason about where to place
obstacles or form up.

## Architecture: Sector Map + Topology Scalars + Spatial Tokens

Three complementary representations, each addressing a different scale of spatial
reasoning. All are fixed-size and efficient to compute.

### Layer 1: Sector Map (per-unit, directional awareness)

For each unit, cast rays in 8 compass directions from its position. Each ray
measures properties until it hits the map boundary:

```
Per sector (8 directions × 6 features = 48 features per unit):
  wall_distance      — distance to nearest wall/obstacle in this direction
  enemy_count        — enemies in this 45° sector within engagement range
  ally_count          — allies in this sector
  cover_density      — fraction of cells with cover in first 4 tiles
  zone_threat        — max hostile zone damage in this sector
  obstacle_density   — fraction of blocked cells in this sector
```

This gives each unit directional awareness: "wall close to my east, enemies
approaching from north, open flank to my south." The model can learn:
- Place barricade toward sectors with low obstacle_density + high enemy_count
- Move toward sectors with high cover_density when low HP
- Spread formation when all sectors have low enemy_count

**Computation:** O(units × 8 × ray_length). With ray_length ≈ 20 cells and 8
units, that's ~1280 grid lookups per tick. Negligible.

### Layer 2: Topology Scalars (global, room structure)

Pre-computed once per room (or cached), these describe the room's strategic shape:

```
Global features (12 scalars):
  room_area           — total walkable cells
  room_aspect_ratio   — width / height of bounding box
  choke_count         — number of narrow passages (width ≤ 2 cells)
  choke_narrowest     — width of tightest choke point
  connectivity        — number of distinct open regions
  open_fraction       — fraction of cells with no adjacent walls
  cover_count         — number of cover-providing obstacles
  elevation_variance  — terrain height variation
  perimeter_ratio     — perimeter / sqrt(area) — measures room compactness
  team_centroid_x     — hero team centroid (normalized 0-1)
  team_centroid_y
  centroid_separation — distance between team centroids (normalized)
```

These let the model learn room-level tactics:
- Low choke_count + high open_fraction → spread formation, no fortification value
- High choke_count + low choke_narrowest → funnel formation, barricades at chokes
- High elevation_variance → high-ground control matters

**Computation:** Room topology is static per scenario. Compute once at fight start.
Centroids update per tick but are O(units).

### Layer 3: Spatial Tokens (key positions as pseudo-entities)

The most expressive layer. Identify strategic positions and encode them as
additional "entity" tokens that the entity encoder cross-attends to alongside
real units:

```
Spatial token types (max 8 tokens):
  CHOKE_POINT    — position of each detected choke, width, orientation
  COVER_POS      — best cover positions near team, quality score
  ZONE_CENTER    — active zone positions, remaining duration, threat level
  OBSTACLE_EDGE  — endpoints of player-placed barricades
  HIGH_GROUND    — elevated positions with tactical advantage
  TEAM_ANCHOR    — team centroid position + movement direction
```

Each spatial token gets the same feature dimensionality as entity tokens
(~30 features), so they can participate in cross-attention naturally:

```
Per spatial token (30 features, matching entity dim):
  type_embedding[4]   — one-hot: choke/cover/zone/obstacle/high_ground/anchor
  position[2]         — x, y (normalized)
  relevance           — how important this point is (choke width, cover quality)
  distance_to_heroes  — avg distance from hero team
  distance_to_enemies — avg distance from enemy team
  control_score       — which team controls this position (-1 to 1)
  threat_level        — incoming damage potential at this position
  accessibility       — pathfinding cost to reach from team centroid
  orientation[2]      — direction the feature "faces" (choke opening direction, cover angle)
  padding[16]         — zero-padded to match entity feature dim
```

The entity encoder's self-attention layer sees both real entities (units) and
spatial tokens. It learns to attend to relevant positions — e.g., an engineer
attends strongly to CHOKE_POINT tokens when deciding where to barricade.

**Computation:** Choke detection is O(grid_size) with a flood-fill / skeleton
algorithm. Cover positions are O(units × cover_candidates). Total spatial token
extraction: ~1ms per room, cached between ticks (only zone/obstacle tokens update).

## Integration with Entity Encoder

```
Current:  [unit_0, unit_1, ..., unit_7] → self-attention → entity embeddings
Proposed: [unit_0, ..., unit_7, spatial_0, ..., spatial_7] → self-attention → embeddings
                                 ^^^^^^^^^^^^^^^^^^^^^^
                                 spatial tokens (same dim)
```

The spatial tokens participate in self-attention but don't produce actions —
they're context-only. The action head still only selects over unit entities.
This means the model learns which spatial features matter for each decision
without any architectural changes to the action head.

Entity feature dim stays at 30. Max entities goes from 7 to 15 (7 units + 8
spatial tokens). Cross-attention computation scales as O(n²) so 15² vs 7² is
~4.6x more attention computation, but at d_model=32 this is still <1μs.

## Sector Map Implementation

```rust
pub struct SectorMap {
    /// 8 sectors × 6 features = 48 floats per unit
    pub sectors: [[f32; 6]; 8],
}

const SECTOR_WALL_DIST: usize = 0;
const SECTOR_ENEMY_COUNT: usize = 1;
const SECTOR_ALLY_COUNT: usize = 2;
const SECTOR_COVER_DENSITY: usize = 3;
const SECTOR_ZONE_THREAT: usize = 4;
const SECTOR_OBSTACLE_DENSITY: usize = 5;

impl SectorMap {
    pub fn extract(state: &SimState, unit_idx: usize, nav: &GridNav) -> Self {
        let unit = &state.units[unit_idx];
        let pos = unit.position;
        let team = unit.team;
        let mut sectors = [[0.0f32; 6]; 8];

        // 8 directions: N, NE, E, SE, S, SW, W, NW
        let dirs: [(f32, f32); 8] = [
            (0.0, -1.0), (0.707, -0.707), (1.0, 0.0), (0.707, 0.707),
            (0.0, 1.0), (-0.707, 0.707), (-1.0, 0.0), (-0.707, -0.707),
        ];

        for (i, (dx, dy)) in dirs.iter().enumerate() {
            // Ray march for wall distance
            let mut dist = 0.0f32;
            let step = 0.5;
            while dist < 20.0 {
                dist += step;
                let probe = SimVec2 { x: pos.x + dx * dist, y: pos.y + dy * dist };
                if !nav.is_walkable_pos(probe) {
                    break;
                }
            }
            sectors[i][SECTOR_WALL_DIST] = dist / 20.0; // normalized

            // Count units in sector (45° cone)
            let sector_angle = i as f32 * std::f32::consts::FRAC_PI_4;
            let half_cone = std::f32::consts::FRAC_PI_4 * 0.5;

            for u in &state.units {
                if u.hp <= 0 || u.id == unit.id { continue; }
                let ux = u.position.x - pos.x;
                let uy = u.position.y - pos.y;
                let angle = uy.atan2(ux);
                let d = (ux * ux + uy * uy).sqrt();
                if d > 15.0 { continue; }

                let angle_diff = (angle - sector_angle).abs();
                let angle_diff = angle_diff.min(2.0 * std::f32::consts::PI - angle_diff);
                if angle_diff <= half_cone {
                    if u.team == team {
                        sectors[i][SECTOR_ALLY_COUNT] += 1.0;
                    } else {
                        sectors[i][SECTOR_ENEMY_COUNT] += 1.0;
                    }
                }
            }

            // Cover and obstacle density from grid
            // (scan cells in sector cone, count cover/blocked)
            // ... grid sampling omitted for brevity
        }

        SectorMap { sectors }
    }
}
```

## Topology Extraction (Choke Detection)

Choke points are detected by finding the "skeleton" of the walkable area
(medial axis transform) and measuring the width at each skeleton point.
Points where width drops below a threshold are choke points.

Simplified approach for grid nav:
1. Distance transform: for each walkable cell, compute distance to nearest wall
2. Local minima in distance field along paths between team positions → choke points
3. Width at choke = 2 × distance_field value at that point

```rust
pub struct RoomTopology {
    pub choke_points: Vec<ChokePoint>,  // max 8
    pub room_area: f32,
    pub aspect_ratio: f32,
    pub open_fraction: f32,
    pub cover_count: u32,
    pub elevation_variance: f32,
}

pub struct ChokePoint {
    pub position: SimVec2,
    pub width: f32,          // narrowest width at this point
    pub orientation: f32,    // angle of the passage
    pub connects: [usize; 2], // which regions it connects
}
```

## Training Data Considerations

- Sector maps are per-unit per-tick, so they inflate the feature vector from 30 to 78
  (30 base + 48 sector). This is fine — the encoder handles it.
- Topology scalars are per-game, append as global conditioning to the entity encoder
  (similar to how delta projection works in the next-state predictor).
- Spatial tokens are the most valuable for learning emergent tactics (formation,
  fortification) because they give the model concrete positions to reference.
- For training: spatial features should be computed lazily and cached per tick
  to avoid slowing down episode generation.

## Phased Rollout

1. **Phase 1: Sector map only** — append 48 features to entity dim, retrain
   entity encoder. Validates that directional awareness improves fight outcomes.
2. **Phase 2: Topology scalars** — add 12 global features as conditioning.
   Test if room-adaptive formation emerges.
3. **Phase 3: Spatial tokens** — add up to 8 spatial pseudo-entities. This is
   the big architectural change but enables the richest spatial reasoning.
