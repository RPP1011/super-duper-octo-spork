use super::lcg::{Lcg, ObstacleRegion};
use super::nav::NavGrid;

// ---------------------------------------------------------------------------
// Obstacle Primitives
// ---------------------------------------------------------------------------

/// Place a linear wall segment with an optional gap; forces flanking.
pub(crate) fn place_wall_segment(
    nav: &mut NavGrid,
    rng: &mut Lcg,
    col: usize,
    row: usize,
    length: usize,
    horizontal: bool,
    gap: usize,
    height: f32,
) -> Vec<ObstacleRegion> {
    let mut regions = Vec::new();
    if length == 0 {
        return regions;
    }
    let gap_start = if gap > 0 && length > gap + 2 {
        rng.next_usize_range(1, length - gap - 1)
    } else if gap > 0 {
        length // gap too large -- skip it
    } else {
        length // no gap requested
    };

    for i in 0..length {
        if i >= gap_start && i < gap_start + gap {
            continue;
        }
        let (c, r) = if horizontal {
            (col + i, row)
        } else {
            (col, row + i)
        };
        if c == 0 || r == 0 || c >= nav.cols.saturating_sub(1) || r >= nav.rows.saturating_sub(1) {
            continue;
        }
        nav.set_walkable_rect(c, r, c, r, false);
        regions.push(ObstacleRegion {
            col0: c,
            col1: c,
            row0: r,
            row1: r,
            height,
        });
    }
    regions
}

/// Place evenly spaced pillars in a rectangular region; breaks sightlines.
pub(crate) fn place_pillar_grid(
    nav: &mut NavGrid,
    _rng: &mut Lcg,
    col0: usize,
    row0: usize,
    col1: usize,
    row1: usize,
    spacing: usize,
    pillar_size: usize,
    height: f32,
) -> Vec<ObstacleRegion> {
    let mut regions = Vec::new();
    let sp = spacing.max(pillar_size + 1);
    let ps = pillar_size.max(1);
    let mut c = col0;
    while c + ps - 1 <= col1 && c + ps - 1 < nav.cols.saturating_sub(1) {
        let mut r = row0;
        while r + ps - 1 <= row1 && r + ps - 1 < nav.rows.saturating_sub(1) {
            if c > 0 && r > 0 {
                let ec = (c + ps - 1).min(nav.cols - 2);
                let er = (r + ps - 1).min(nav.rows - 2);
                nav.set_walkable_rect(c, r, ec, er, false);
                regions.push(ObstacleRegion {
                    col0: c,
                    col1: ec,
                    row0: r,
                    row1: er,
                    height,
                });
            }
            r += sp;
        }
        c += sp;
    }
    regions
}

/// Place an L-shaped cover block; asymmetric defender advantage.
pub(crate) fn place_l_shape(
    nav: &mut NavGrid,
    _rng: &mut Lcg,
    anchor_col: usize,
    anchor_row: usize,
    arm_len: usize,
    thickness: usize,
    orientation: usize,
    height: f32,
) -> Vec<ObstacleRegion> {
    let mut regions = Vec::new();
    let t = thickness.max(1);
    let a = arm_len.max(2);

    // orientation: 0=right-down, 1=left-down, 2=left-up, 3=right-up
    let (hc0, hc1, hr0, hr1) = match orientation {
        0 => (
            anchor_col,
            anchor_col + a - 1,
            anchor_row,
            anchor_row + t - 1,
        ),
        1 => (
            anchor_col.saturating_sub(a - 1),
            anchor_col,
            anchor_row,
            anchor_row + t - 1,
        ),
        2 => (
            anchor_col.saturating_sub(a - 1),
            anchor_col,
            anchor_row.saturating_sub(t - 1),
            anchor_row,
        ),
        _ => (
            anchor_col,
            anchor_col + a - 1,
            anchor_row.saturating_sub(t - 1),
            anchor_row,
        ),
    };
    let (vc0, vc1, vr0, vr1) = match orientation {
        0 => (
            anchor_col,
            anchor_col + t - 1,
            anchor_row,
            anchor_row + a - 1,
        ),
        1 => (
            anchor_col.saturating_sub(t - 1),
            anchor_col,
            anchor_row,
            anchor_row + a - 1,
        ),
        2 => (
            anchor_col.saturating_sub(t - 1),
            anchor_col,
            anchor_row.saturating_sub(a - 1),
            anchor_row,
        ),
        _ => (
            anchor_col,
            anchor_col + t - 1,
            anchor_row.saturating_sub(a - 1),
            anchor_row,
        ),
    };

    for &(c0, c1, r0, r1) in &[(hc0, hc1, hr0, hr1), (vc0, vc1, vr0, vr1)] {
        let c0c = c0.max(1);
        let c1c = c1.min(nav.cols.saturating_sub(2));
        let r0c = r0.max(1);
        let r1c = r1.min(nav.rows.saturating_sub(2));
        if c0c <= c1c && r0c <= r1c {
            nav.set_walkable_rect(c0c, r0c, c1c, r1c, false);
            regions.push(ObstacleRegion {
                col0: c0c,
                col1: c1c,
                row0: r0c,
                row1: r1c,
                height,
            });
        }
    }
    regions
}

/// Place 2-4 small scattered blocks; peeking cover.
pub(crate) fn place_cover_cluster(
    nav: &mut NavGrid,
    rng: &mut Lcg,
    centre_col: usize,
    centre_row: usize,
    spread: usize,
    count: usize,
    height: f32,
) -> Vec<ObstacleRegion> {
    let mut regions = Vec::new();
    let n = count.clamp(2, 4);
    for _ in 0..n {
        let dc = rng.next_usize_range(0, spread);
        let dr = rng.next_usize_range(0, spread);
        let sign_c: isize = if rng.next_u64() % 2 == 0 { 1 } else { -1 };
        let sign_r: isize = if rng.next_u64() % 2 == 0 { 1 } else { -1 };
        let c =
            (centre_col as isize + sign_c * dc as isize).clamp(1, nav.cols as isize - 2) as usize;
        let r =
            (centre_row as isize + sign_r * dr as isize).clamp(1, nav.rows as isize - 2) as usize;
        let w = rng.next_usize_range(1, 2);
        let h = rng.next_usize_range(1, 2);
        let ec = (c + w - 1).min(nav.cols - 2);
        let er = (r + h - 1).min(nav.rows - 2);
        nav.set_walkable_rect(c, r, ec, er, false);
        regions.push(ObstacleRegion {
            col0: c,
            col1: ec,
            row0: r,
            row1: er,
            height,
        });
    }
    regions
}

/// Place two parallel walls forming a corridor lane; kill zone.
pub(crate) fn place_corridor_walls(
    nav: &mut NavGrid,
    rng: &mut Lcg,
    col_lo: usize,
    col_hi: usize,
    centre_row: usize,
    gap_width: usize,
    height: f32,
) -> Vec<ObstacleRegion> {
    let mut regions = Vec::new();
    let half_gap = gap_width / 2;
    let top_row = centre_row.saturating_sub(half_gap + 1);
    let bot_row = centre_row + half_gap + 1;
    let len = col_hi.saturating_sub(col_lo);

    if top_row > 0 && top_row < nav.rows - 1 {
        regions.extend(place_wall_segment(
            nav, rng, col_lo, top_row, len, true, 0, height,
        ));
    }
    if bot_row > 0 && bot_row < nav.rows - 1 {
        regions.extend(place_wall_segment(
            nav, rng, col_lo, bot_row, len, true, 0, height,
        ));
    }
    regions
}

/// Place elevated cover positions: walkable cells with elevation providing
/// a tactical high-ground advantage without blocking movement.
pub(crate) fn place_elevated_platform(
    nav: &mut NavGrid,
    _rng: &mut Lcg,
    col0: usize,
    row0: usize,
    width: usize,
    depth: usize,
    elevation: f32,
) -> Vec<ObstacleRegion> {
    let c0 = col0.max(1);
    let r0 = row0.max(1);
    let c1 = (col0 + width - 1).min(nav.cols.saturating_sub(2));
    let r1 = (row0 + depth - 1).min(nav.rows.saturating_sub(2));
    // Set elevation but keep cells walkable
    nav.set_elevation_rect(c0, r0, c1, r1, elevation);
    // Place small lip/railing obstacles on two edges for cover
    let mut regions = Vec::new();
    // Front edge (toward enemy side, right)
    if c1 + 1 < nav.cols - 1 {
        for r in r0..=r1 {
            nav.set_walkable_rect(c1 + 1, r, c1 + 1, r, false);
            regions.push(ObstacleRegion {
                col0: c1 + 1,
                col1: c1 + 1,
                row0: r,
                row1: r,
                height: 0.6,
            });
        }
    }
    regions
}

/// Place a sandbag arc: scattered low cover blocks in a semicircle pattern.
/// Provides cover without fully blocking sightlines.
pub(crate) fn place_sandbag_arc(
    nav: &mut NavGrid,
    rng: &mut Lcg,
    centre_col: usize,
    centre_row: usize,
    radius: usize,
    count: usize,
    height: f32,
) -> Vec<ObstacleRegion> {
    let mut regions = Vec::new();
    let n = count.clamp(3, 8);
    for i in 0..n {
        // Place blocks in arc from -90 to +90 degrees (facing right toward enemy side)
        let angle = -std::f32::consts::FRAC_PI_2
            + std::f32::consts::PI * (i as f32 / (n - 1).max(1) as f32);
        let dc = (angle.cos() * radius as f32).round() as isize;
        let dr = (angle.sin() * radius as f32).round() as isize;
        // Add small random jitter
        let jitter_c = if rng.next_u64() % 3 == 0 { 1isize } else { 0 };
        let jitter_r = if rng.next_u64() % 3 == 0 { 1isize } else { 0 };
        let c = (centre_col as isize + dc + jitter_c).clamp(1, nav.cols as isize - 2) as usize;
        let r = (centre_row as isize + dr + jitter_r).clamp(1, nav.rows as isize - 2) as usize;
        nav.set_walkable_rect(c, r, c, r, false);
        regions.push(ObstacleRegion {
            col0: c,
            col1: c,
            row0: r,
            row1: r,
            height,
        });
    }
    regions
}

/// Place a moat (ring of unwalkable cells) around a central area with bridge gaps.
/// Creates defensive terrain forcing attackers through chokepoints.
pub(crate) fn place_moat(
    nav: &mut NavGrid,
    rng: &mut Lcg,
    centre_col: usize,
    centre_row: usize,
    inner_radius: usize,
    bridge_count: usize,
) -> Vec<ObstacleRegion> {
    let mut regions = Vec::new();
    let r = inner_radius.max(2);
    // Pick bridge directions (expressed as octant indices 0..8)
    let mut bridge_octants: Vec<usize> = Vec::new();
    let n_bridges = bridge_count.clamp(1, 4);
    let step = 8 / n_bridges;
    let offset = rng.next_usize_range(0, step);
    for i in 0..n_bridges {
        bridge_octants.push((offset + i * step) % 8);
    }

    for angle_step in 0..32 {
        let angle = (angle_step as f32 / 32.0) * std::f32::consts::TAU;
        let dc = (angle.cos() * r as f32).round() as isize;
        let dr = (angle.sin() * r as f32).round() as isize;
        let c = (centre_col as isize + dc).clamp(1, nav.cols as isize - 2) as usize;
        let row = (centre_row as isize + dr).clamp(1, nav.rows as isize - 2) as usize;

        // Check if this cell is on a bridge
        let octant = ((angle / std::f32::consts::TAU * 8.0).round() as usize) % 8;
        if bridge_octants.contains(&octant) {
            continue;
        }

        nav.set_walkable_rect(c, row, c, row, false);
        regions.push(ObstacleRegion {
            col0: c,
            col1: c,
            row0: row,
            row1: row,
            height: 0.3,
        });
    }
    regions
}

/// Place a spiral pattern of obstacles radiating from centre.
/// Creates winding paths that force non-linear movement.
pub(crate) fn place_spiral(
    nav: &mut NavGrid,
    _rng: &mut Lcg,
    centre_col: usize,
    centre_row: usize,
    max_radius: usize,
    turns: f32,
    height: f32,
) -> Vec<ObstacleRegion> {
    let mut regions = Vec::new();
    let steps = (max_radius as f32 * turns * 6.0) as usize;
    for i in 0..steps {
        let t = i as f32 / steps.max(1) as f32;
        let cur_r = t * max_radius as f32;
        let angle = t * turns * std::f32::consts::TAU;
        let dc = (angle.cos() * cur_r).round() as isize;
        let dr = (angle.sin() * cur_r).round() as isize;
        let c = (centre_col as isize + dc).clamp(1, nav.cols as isize - 2) as usize;
        let row = (centre_row as isize + dr).clamp(1, nav.rows as isize - 2) as usize;
        nav.set_walkable_rect(c, row, c, row, false);
        regions.push(ObstacleRegion {
            col0: c,
            col1: c,
            row0: row,
            row1: row,
            height,
        });
    }
    regions
}

/// Place alternating block/gap segments along a row; barricade.
pub(crate) fn place_barricade_line(
    nav: &mut NavGrid,
    _rng: &mut Lcg,
    col_lo: usize,
    col_hi: usize,
    row: usize,
    segment_len: usize,
    gap_len: usize,
    height: f32,
) -> Vec<ObstacleRegion> {
    let mut regions = Vec::new();
    if row == 0 || row >= nav.rows - 1 {
        return regions;
    }
    let seg = segment_len.max(1);
    let gap = gap_len.max(1);
    let mut c = col_lo.max(1);
    let limit = col_hi.min(nav.cols - 2);
    let mut placing = true;
    let mut run = 0usize;
    while c <= limit {
        if placing {
            nav.set_walkable_rect(c, row, c, row, false);
            regions.push(ObstacleRegion {
                col0: c,
                col1: c,
                row0: row,
                row1: row,
                height,
            });
            run += 1;
            if run >= seg {
                placing = false;
                run = 0;
            }
        } else {
            run += 1;
            if run >= gap {
                placing = true;
                run = 0;
            }
        }
        c += 1;
    }
    regions
}
