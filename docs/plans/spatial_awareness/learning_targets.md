# Spatial Awareness: What the Model Could Learn

## Tactical behaviors enabled by corner tokens

### Barricade placement (engineer)
Engineer sees a corner with low `passage_width` + concave + enemies approaching
from that direction → place barricade to seal the choke. Two engineers see the
same corners but after one places a barricade, new corners appear at the obstacle
edges and the sealed corner's passage_width drops to 0 — the second engineer
targets the next gap.

### Cover usage
Low-HP unit sees concave corner nearby with short distance → move toward it. The
`convex` flag distinguishes cover (concave, wall shields you) from exposed
positions (convex, wall sticks out and you're flanked).

### Formation selection
The distribution of visible corners encodes the room shape implicitly:
- Many corners with low passage_width → corridor room → tight formation
- Few corners, all high passage_width → open room → spread formation
- Asymmetric corner distribution → flanking opportunities on one side

No explicit room-type classification needed — the model infers it from the
corner set.

### Flanking detection
A unit sees a convex corner → the other side is exposed. If an enemy is near
that corner, they can be attacked from around the wall. The `open_direction`
vector points toward the open side.

### Choke control
Corners with `passage_width < 2.0` are choke points. A team that controls
the choke (has units near it) forces enemies through a narrow opening.
The model can learn that holding position near a low-width concave corner
is high-value in corridor rooms.

### Zone placement
AoE abilities are most effective at choke points — placing a damage zone
at a narrow corner forces enemies to walk through it or take a long detour.
The corner's `passage_width` directly indicates AoE value at that location.

## Training data considerations

- Spatial tokens change rarely (only when units cross cell boundaries or
  obstacles change), so they're cheap to include in training data
- Per-unit visibility means training data naturally captures "what this unit
  can see" — important for learning to use cover
- Corner features are normalized [0,1] or [-1,1], matching the entity feature
  scale — no additional normalization needed
- The variable number of visible corners per unit requires masking in attention
  layers (or zero-padding to MAX_CORNERS)

## Evaluation metrics

Beyond win rate, spatial awareness should improve:
- **Damage taken in corridor rooms** — units should take less damage by using cover
- **Barricade effectiveness** — barricades at choke points should block more enemies
- **Formation spread** — should correlate with room openness (corner distribution)
- **Choke control time** — time spent controlling narrow passages
- **Out-of-sight deaths** — should decrease (units aware of blind spots)
