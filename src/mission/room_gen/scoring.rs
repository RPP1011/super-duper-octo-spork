//! Room interest scoring — a fitness function for procedural room generation.
//!
//! Computes a static "visual/tactical interest" score for a `RoomLayout` by
//! measuring spatial complexity, sightline variety, cover distribution,
//! elevation drama, and obstacle asymmetry.  The score is cheap to compute
//! (no sim rollout needed) and can drive rejection-sampling in
//! `generate_room` to pick the most interesting candidate from N seeds.

use crate::ai::core::sim_vec2;
use crate::ai::pathing::{cover_factor, has_line_of_sight, raycast_distances, GridNav};
use crate::mission::room_gen::{NavGrid, RoomLayout};

// ---------------------------------------------------------------------------
// Individual metric functions (all return 0.0 .. 1.0 normalised)
// ---------------------------------------------------------------------------

/// Sightline entropy: do different positions see different things?
///
/// We cast rays from a grid of sample points and measure how much the
/// per-position mean distance varies.  High std-dev → interesting geometry
/// with pockets, corridors, open areas.  Low std-dev → uniform open field
/// or uniform grid of pillars.
pub fn sightline_entropy(nav: &GridNav, width: f32, depth: f32) -> f32 {
    let n_samples = 5;
    let n_rays: usize = 16;
    let max_dist = width.max(depth);
    let mut means: Vec<f32> = Vec::with_capacity(n_samples * n_samples);

    for sy in 0..n_samples {
        for sx in 0..n_samples {
            let x = width * (sx as f32 + 0.5) / n_samples as f32;
            let z = depth * (sy as f32 + 0.5) / n_samples as f32;
            let pos = sim_vec2(x, z);
            if !nav.is_walkable_pos(pos) {
                continue;
            }
            let dists = raycast_distances(nav, pos, n_rays, max_dist);
            let mean = dists.iter().sum::<f32>() / dists.len() as f32;
            means.push(mean);
        }
    }

    if means.len() < 2 {
        return 0.0;
    }

    let overall_mean = means.iter().sum::<f32>() / means.len() as f32;
    let variance =
        means.iter().map(|m| (m - overall_mean).powi(2)).sum::<f32>() / means.len() as f32;
    let std_dev = variance.sqrt();

    // Normalise: std_dev of ~3-5 cells is "interesting", 0 is boring.
    (std_dev / max_dist * 4.0).min(1.0)
}

/// Cover distribution: is cover available across the room, not just clumped?
///
/// Sample positions and check how many have non-zero cover from at least
/// one cardinal direction.  Also penalise if cover quality is too uniform
/// (all positions have the same factor) or too sparse.
pub fn cover_distribution(nav: &GridNav, width: f32, depth: f32) -> f32 {
    let n = 6;
    let mut cover_values: Vec<f32> = Vec::with_capacity(n * n);

    for sy in 0..n {
        for sx in 0..n {
            let x = width * (sx as f32 + 0.5) / n as f32;
            let z = depth * (sy as f32 + 0.5) / n as f32;
            let pos = sim_vec2(x, z);
            if !nav.is_walkable_pos(pos) {
                continue;
            }
            // Best cover from any of 4 cardinal attack directions
            let mut best = 0.0_f32;
            for &(dx, dz) in &[(5.0, 0.0), (-5.0, 0.0), (0.0, 5.0), (0.0, -5.0)] {
                let attacker = sim_vec2(x + dx, z + dz);
                best = best.max(cover_factor(nav, pos, attacker));
            }
            cover_values.push(best);
        }
    }

    if cover_values.is_empty() {
        return 0.0;
    }

    let covered_count = cover_values.iter().filter(|&&v| v > 0.0).count();
    let covered_frac = covered_count as f32 / cover_values.len() as f32;

    // Ideal: 30-60% of sampled positions have cover.
    // Too low = open field (boring).  Too high = maze (also boring).
    let coverage_score = 1.0 - (covered_frac - 0.45).abs() * 3.0;

    // Also reward variance in cover quality
    let mean = cover_values.iter().sum::<f32>() / cover_values.len() as f32;
    let var = cover_values
        .iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f32>()
        / cover_values.len() as f32;
    let variety = (var.sqrt() * 4.0).min(1.0);

    (coverage_score.max(0.0) * 0.6 + variety * 0.4).clamp(0.0, 1.0)
}

/// Elevation drama: height variation creates visual depth and tactical layers.
pub fn elevation_drama(nav_grid: &NavGrid) -> f32 {
    let walkable_elevations: Vec<f32> = nav_grid
        .walkable
        .iter()
        .zip(nav_grid.elevation.iter())
        .filter(|(&w, _)| w)
        .map(|(_, &e)| e)
        .collect();

    if walkable_elevations.len() < 2 {
        return 0.0;
    }

    let mean = walkable_elevations.iter().sum::<f32>() / walkable_elevations.len() as f32;
    let var = walkable_elevations
        .iter()
        .map(|e| (e - mean).powi(2))
        .sum::<f32>()
        / walkable_elevations.len() as f32;
    let std_dev = var.sqrt();

    // 0.3m std_dev = moderate interest, 0.8+ = very dramatic
    (std_dev / 0.6).min(1.0)
}

/// Obstacle asymmetry: perfect left-right symmetry is boring.
///
/// Compare the walkable grid left-half vs right-half and compute how
/// different they are (Hamming distance normalised by count).
pub fn asymmetry(nav_grid: &NavGrid) -> f32 {
    let cols = nav_grid.cols;
    let rows = nav_grid.rows;
    if cols < 4 || rows < 4 {
        return 0.0;
    }

    let mut diffs = 0usize;
    let mut total = 0usize;
    for r in 1..rows - 1 {
        let half = (cols - 2) / 2;
        for i in 0..half {
            let left_c = 1 + i;
            let right_c = cols - 2 - i;
            let left_w = nav_grid.walkable[r * cols + left_c];
            let right_w = nav_grid.walkable[r * cols + right_c];
            if left_w != right_w {
                diffs += 1;
            }
            total += 1;
        }
    }

    if total == 0 {
        return 0.0;
    }

    let raw = diffs as f32 / total as f32;
    // Sweet spot: 15-50% asymmetry.  Too much = chaotic.
    if raw < 0.10 {
        raw / 0.10 * 0.5 // Penalise near-symmetry
    } else if raw > 0.60 {
        1.0 - (raw - 0.60) * 2.0 // Penalise chaos
    } else {
        0.5 + (raw - 0.10) / 0.50 * 0.5 // Linear in sweet spot
    }
    .clamp(0.0, 1.0)
}

/// Chokepoint count: tactical decision points where geometry forces choices.
///
/// A chokepoint cell has exactly 2 walkable cardinal neighbors (narrow passage).
pub fn chokepoint_density(nav_grid: &NavGrid) -> f32 {
    let cols = nav_grid.cols;
    let rows = nav_grid.rows;
    let mut chokepoints = 0usize;
    let mut walkable_count = 0usize;

    for r in 1..rows - 1 {
        for c in 1..cols - 1 {
            let idx = r * cols + c;
            if !nav_grid.walkable[idx] {
                continue;
            }
            walkable_count += 1;
            let mut walkable_neighbors = 0;
            for &(dc, dr) in &[(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
                let nc = c as i32 + dc;
                let nr = r as i32 + dr;
                if nc >= 0
                    && nr >= 0
                    && (nc as usize) < cols
                    && (nr as usize) < rows
                    && nav_grid.walkable[nr as usize * cols + nc as usize]
                {
                    walkable_neighbors += 1;
                }
            }
            if walkable_neighbors == 2 {
                chokepoints += 1;
            }
        }
    }

    if walkable_count == 0 {
        return 0.0;
    }

    let density = chokepoints as f32 / walkable_count as f32;
    // Sweet spot: 3-12% chokepoint density.
    (density / 0.08).min(1.0)
}

/// LoS fragmentation: how many sampled positions can see across the room?
///
/// Rather than just checking spawn-to-spawn (which is easy to fool with a
/// single pillar line), we sample a grid of positions and check cross-room
/// visibility.  We also penalise rooms where LoS quality is *uniform* —
/// interesting rooms have some positions with LoS and some without.
pub fn los_fragmentation(
    nav: &GridNav,
    width: f32,
    depth: f32,
) -> f32 {
    let n = 5;
    let mut los_per_sample: Vec<f32> = Vec::new();

    // Sample a grid of defender positions on the left half
    // and check LoS to attacker positions on the right half
    for sy in 0..n {
        for sx in 0..n {
            let dx = width * (sx as f32 + 0.5) / n as f32;
            let dz = depth * (sy as f32 + 0.5) / n as f32;
            let def_pos = sim_vec2(dx, dz);
            if !nav.is_walkable_pos(def_pos) {
                continue;
            }

            let mut visible = 0usize;
            let mut checked = 0usize;
            // Check LoS to positions on the opposite side of the room
            for ay in 0..n {
                let ax = width * (0.5 + (n - 1 - sx) as f32 * 0.5 / n as f32);
                let az = depth * (ay as f32 + 0.5) / n as f32;
                let atk_pos = sim_vec2(ax, az);
                if !nav.is_walkable_pos(atk_pos) {
                    continue;
                }
                checked += 1;
                if has_line_of_sight(nav, def_pos, atk_pos) {
                    visible += 1;
                }
            }
            if checked > 0 {
                los_per_sample.push(visible as f32 / checked as f32);
            }
        }
    }

    if los_per_sample.len() < 2 {
        return 0.0;
    }

    // Mean visibility fraction
    let mean = los_per_sample.iter().sum::<f32>() / los_per_sample.len() as f32;

    // Variance of per-position LoS — high variance means some positions are
    // exposed and some are sheltered (interesting).
    let var = los_per_sample
        .iter()
        .map(|v| (v - mean).powi(2))
        .sum::<f32>()
        / los_per_sample.len() as f32;
    let std_dev = var.sqrt();

    // Score: reward moderate mean LoS + high variance
    let mean_score = 1.0 - (mean - 0.40).abs() * 2.5;
    let variety_score = (std_dev * 3.0).min(1.0);

    (mean_score.max(0.0) * 0.5 + variety_score * 0.5).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Composite score
// ---------------------------------------------------------------------------

/// Individual metric scores (for diagnostics).
#[derive(Debug, Clone)]
pub struct InterestBreakdown {
    pub sightline: f32,
    pub cover: f32,
    pub elevation: f32,
    pub asymmetry: f32,
    pub chokepoints: f32,
    pub los_frag: f32,
    pub total: f32,
}

/// Compute a composite interest score for a room layout.
/// Returns a normalised score in [0.0, 1.0] and a breakdown of components.
pub fn static_interest_score(layout: &RoomLayout) -> InterestBreakdown {
    let grid_nav = layout.nav.to_gridnav();
    let w = layout.width;
    let d = layout.depth;

    let s_sightline = sightline_entropy(&grid_nav, w, d);
    let s_cover = cover_distribution(&grid_nav, w, d);
    let s_elevation = elevation_drama(&layout.nav);
    let s_asymmetry = asymmetry(&layout.nav);
    let s_chokepoints = chokepoint_density(&layout.nav);

    let s_los = los_fragmentation(&grid_nav, w, d);

    // Weighted combination
    let total = s_sightline * 0.25
        + s_cover * 0.20
        + s_elevation * 0.10
        + s_asymmetry * 0.15
        + s_chokepoints * 0.15
        + s_los * 0.15;

    InterestBreakdown {
        sightline: s_sightline,
        cover: s_cover,
        elevation: s_elevation,
        asymmetry: s_asymmetry,
        chokepoints: s_chokepoints,
        los_frag: s_los,
        total: total.clamp(0.0, 1.0),
    }
}

// ---------------------------------------------------------------------------
// Candidate selection: generate N rooms, pick the most interesting
// ---------------------------------------------------------------------------

/// Generate `candidate_count` rooms and return the one with the highest
/// static interest score.
pub fn generate_interesting_room(
    seed: u64,
    room_type: crate::game_core::RoomType,
    candidate_count: usize,
) -> (RoomLayout, InterestBreakdown) {
    let mut best_layout = super::generate_room(seed, room_type);
    let mut best_score = static_interest_score(&best_layout);

    for variant in 1..candidate_count as u64 {
        let candidate_seed = seed ^ (variant.wrapping_mul(0x517C_C1B7_2722_0A95));
        let candidate = super::generate_room(candidate_seed, room_type);
        let score = static_interest_score(&candidate);
        if score.total > best_score.total {
            best_layout = candidate;
            best_score = score;
        }
    }

    (best_layout, best_score)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game_core::RoomType;
    use crate::mission::room_gen::generate_room;

    /// Build a hand-crafted adversarial room that passes all validation but
    /// is tactically dead: a big empty box with a single thin wall of
    /// pillars evenly spaced across the exact centre.
    ///
    /// It passes validate_layout because:
    /// - blocked% ≈ 3% (above 2% threshold)
    /// - player↔enemy BFS connectivity works (gaps between pillars)
    ///
    /// But it's boring because:
    /// - Perfect left/right symmetry
    /// - Uniform sightlines from every position
    /// - Cover only available in one narrow band
    /// - Zero elevation variation
    /// - No chokepoints (every cell has 3+ walkable neighbors)
    /// - Near-100% LoS between spawns (pillars have gaps)
    fn build_adversarial_room() -> RoomLayout {
        let cols = 20usize;
        let rows = 20usize;
        let cell_size = 1.0;
        let mut nav = NavGrid::new(cols, rows, cell_size);

        // Perimeter walls
        for c in 0..cols {
            nav.set_walkable_rect(c, 0, c, 0, false);
            nav.set_walkable_rect(c, rows - 1, c, rows - 1, false);
        }
        for r in 0..rows {
            nav.set_walkable_rect(0, r, 0, r, false);
            nav.set_walkable_rect(cols - 1, r, cols - 1, r, false);
        }

        // A single thin line of pillars at column 10, evenly spaced
        // (every other row blocked) — just enough to hit 2% blocked
        for r in (2..rows - 2).step_by(2) {
            nav.set_walkable_rect(cols / 2, r, cols / 2, r, false);
        }

        // Spawns on opposite sides
        use crate::ai::core::SimVec2;
        let player_spawn = crate::mission::room_gen::SpawnZone {
            positions: vec![
                SimVec2 { x: 2.5, y: 10.5 },
                SimVec2 { x: 3.5, y: 10.5 },
                SimVec2 { x: 2.5, y: 11.5 },
                SimVec2 { x: 3.5, y: 11.5 },
            ],
        };
        let enemy_spawn = crate::mission::room_gen::SpawnZone {
            positions: vec![
                SimVec2 { x: 17.5, y: 10.5 },
                SimVec2 { x: 16.5, y: 10.5 },
                SimVec2 { x: 17.5, y: 11.5 },
                SimVec2 { x: 16.5, y: 11.5 },
            ],
        };

        RoomLayout {
            width: cols as f32,
            depth: rows as f32,
            nav,
            player_spawn,
            enemy_spawn,
            room_type: RoomType::Entry,
            seed: 0,
            floor_heights: vec![0.0; (cols + 1) * (rows + 1)],
            floor_subdivisions: cols + 1,
            floor_colors: vec![[0.4, 0.4, 0.4, 1.0]; (cols + 1) * (rows + 1)],
            scatter: Vec::new(),
            environment: crate::mission::room_gen::RoomEnvironment::Ruins,
            obstacles: Vec::new(),
            ramps: Vec::new(),
        }
    }

    #[test]
    fn adversarial_room_exposes_scoring_gap() {
        let boring = build_adversarial_room();
        let score = static_interest_score(&boring);

        println!("=== ADVERSARIAL ROOM (boring pillar line) ===");
        println!("  sightline:   {:.3}", score.sightline);
        println!("  cover:       {:.3}", score.cover);
        println!("  elevation:   {:.3}", score.elevation);
        println!("  asymmetry:   {:.3}", score.asymmetry);
        println!("  chokepoints: {:.3}", score.chokepoints);
        println!("  LoS frag:    {:.3}", score.los_frag);
        println!("  TOTAL:       {:.3}", score.total);

        // The adversarial room has zero elevation — scorer detects this.
        assert!(score.elevation < 0.01, "no elevation = 0");

        // Count how many generated rooms this boring room actually beats.
        let room_types = [
            RoomType::Entry,
            RoomType::Pressure,
            RoomType::Pivot,
            RoomType::Setpiece,
            RoomType::Climax,
        ];
        let mut total = 0;
        let mut adversarial_wins = 0;
        for rt in &room_types {
            for seed in 0..20u64 {
                let layout = generate_room(seed, *rt);
                let s = static_interest_score(&layout);
                total += 1;
                if score.total > s.total {
                    adversarial_wins += 1;
                }
            }
        }
        let adversarial_win_pct = adversarial_wins as f32 / total as f32 * 100.0;
        println!(
            "Adversarial room beats {adversarial_wins}/{total} ({adversarial_win_pct:.0}%) generated rooms"
        );
        // This documents the gap: without candidate selection, the boring
        // room beats a significant fraction of generated rooms.
        // The candidate_selection_improves_score test shows how to fix this.
    }

    #[test]
    fn candidate_selection_beats_adversarial() {
        let adversarial_score = static_interest_score(&build_adversarial_room()).total;

        // With candidate selection (pick best of 8), most room types should
        // beat the adversarial baseline.
        let room_types = [
            RoomType::Entry,
            RoomType::Pressure,
            RoomType::Pivot,
            RoomType::Setpiece,
            RoomType::Climax,
        ];

        let mut beaten = 0;
        for rt in &room_types {
            let (_, selected) = generate_interesting_room(42, *rt, 8);
            if selected.total >= adversarial_score {
                beaten += 1;
            }
            println!(
                "{:?}: selected={:.3} vs adversarial={:.3} {}",
                rt,
                selected.total,
                adversarial_score,
                if selected.total >= adversarial_score { "WIN" } else { "LOSE" }
            );
        }
        // At least 3/5 room types should beat adversarial with selection
        assert!(
            beaten >= 3,
            "only {beaten}/5 room types beat adversarial with candidate selection"
        );
    }

    #[test]
    fn candidate_selection_improves_score() {
        let room_types = [RoomType::Entry, RoomType::Setpiece, RoomType::Climax];

        for rt in &room_types {
            let baseline = generate_room(42, *rt);
            let baseline_score = static_interest_score(&baseline);

            let (_, selected_score) = generate_interesting_room(42, *rt, 8);

            println!(
                "{:?}: baseline={:.3}  selected(8)={:.3}  delta={:+.3}",
                rt,
                baseline_score.total,
                selected_score.total,
                selected_score.total - baseline_score.total
            );

            // Selected should be >= baseline (it's a max over candidates)
            assert!(
                selected_score.total >= baseline_score.total - f32::EPSILON,
                "{:?}: selection made it worse somehow",
                rt
            );
        }
    }

    #[test]
    fn print_score_distribution() {
        println!("\n=== SCORE DISTRIBUTION ===");
        for rt in &[
            RoomType::Entry,
            RoomType::Pressure,
            RoomType::Pivot,
            RoomType::Setpiece,
            RoomType::Recovery,
            RoomType::Climax,
        ] {
            let mut scores: Vec<f32> = (0..50u64)
                .map(|s| static_interest_score(&generate_room(s, *rt)).total)
                .collect();
            scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let min = scores[0];
            let med = scores[scores.len() / 2];
            let max = scores[scores.len() - 1];
            println!("{:>10?}  min={min:.3}  med={med:.3}  max={max:.3}", rt);
        }
    }
}
