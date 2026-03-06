use super::lcg::{Lcg, ObstacleRegion};
use super::nav::NavGrid;
use super::primitives::*;

// ---------------------------------------------------------------------------
// Room-Type Template Generators
// ---------------------------------------------------------------------------

pub(crate) fn generate_entry_obstacles(nav: &mut NavGrid, rng: &mut Lcg) -> Vec<ObstacleRegion> {
    let cols = nav.cols;
    let rows = nav.rows;
    let template = rng.next_usize_range(0, 5);
    let mut obs = Vec::new();
    let mid_c = cols / 2;
    let mid_r = rows / 2;
    match template {
        0 => {
            // "Pillared Hall": symmetric pillar grid with wide spacing — clean lanes
            obs.extend(place_pillar_grid(
                nav,
                rng,
                cols / 4,
                rows / 4,
                3 * cols / 4,
                3 * rows / 4,
                5,
                1,
                1.5,
            ));
        }
        1 => {
            // "Divided Room": central horizontal wall with gap, mirrored cover blocks
            obs.extend(place_wall_segment(
                nav,
                rng,
                cols / 4,
                mid_r,
                cols / 2,
                true,
                3,
                1.5,
            ));
            // Symmetric cover blocks in each third
            obs.extend(place_barricade_line(
                nav,
                rng,
                cols / 5,
                cols / 5 + 3,
                rows / 3,
                2,
                1,
                1.0,
            ));
            obs.extend(place_barricade_line(
                nav,
                rng,
                cols / 5,
                cols / 5 + 3,
                2 * rows / 3,
                2,
                1,
                1.0,
            ));
            obs.extend(place_barricade_line(
                nav,
                rng,
                3 * cols / 5,
                3 * cols / 5 + 3,
                rows / 3,
                2,
                1,
                1.0,
            ));
            obs.extend(place_barricade_line(
                nav,
                rng,
                3 * cols / 5,
                3 * cols / 5 + 3,
                2 * rows / 3,
                2,
                1,
                1.0,
            ));
        }
        2 => {
            // "Fortified Positions": elevated platform at centre with sandbag arcs on flanks
            obs.extend(place_elevated_platform(
                nav,
                rng,
                mid_c - 1,
                mid_r - 1,
                3,
                3,
                0.8,
            ));
            obs.extend(place_sandbag_arc(nav, rng, cols / 4, mid_r, 2, 4, 0.7));
            obs.extend(place_sandbag_arc(nav, rng, 3 * cols / 4, mid_r, 2, 4, 0.7));
        }
        3 => {
            // "Spiral Approach": spiral obstacle path forcing winding movement
            obs.extend(place_spiral(nav, rng, mid_c, mid_r, rows / 3, 1.2, 1.2));
            obs.extend(place_cover_cluster(nav, rng, cols / 4, rows / 4, 2, 2, 1.0));
            obs.extend(place_cover_cluster(
                nav,
                rng,
                3 * cols / 4,
                3 * rows / 4,
                2,
                2,
                1.0,
            ));
        }
        _ => {
            // "Mirrored Ruins": symmetric L-shapes creating flanking corridors
            obs.extend(place_l_shape(nav, rng, cols / 4, rows / 3, 3, 1, 0, 1.5));
            obs.extend(place_l_shape(
                nav,
                rng,
                3 * cols / 4,
                2 * rows / 3,
                3,
                1,
                2,
                1.5,
            ));
            // Short horizontal barricades near each side for cover
            obs.extend(place_barricade_line(
                nav,
                rng,
                cols / 4 - 1,
                cols / 4 + 2,
                2 * rows / 3,
                2,
                1,
                1.0,
            ));
            obs.extend(place_barricade_line(
                nav,
                rng,
                3 * cols / 4 - 2,
                3 * cols / 4 + 1,
                rows / 3,
                2,
                1,
                1.0,
            ));
        }
    }
    obs
}

pub(crate) fn generate_pressure_obstacles(nav: &mut NavGrid, rng: &mut Lcg) -> Vec<ObstacleRegion> {
    let cols = nav.cols;
    let rows = nav.rows;
    let template = rng.next_usize_range(0, 4);
    let mut obs = Vec::new();
    match template {
        0 => {
            obs.extend(place_corridor_walls(
                nav,
                rng,
                cols / 4,
                3 * cols / 4,
                rows / 2,
                4,
                1.5,
            ));
            obs.extend(place_cover_cluster(nav, rng, cols / 2, rows / 2, 2, 3, 1.0));
        }
        1 => {
            obs.extend(place_barricade_line(
                nav,
                rng,
                cols / 4,
                3 * cols / 4,
                rows / 3,
                3,
                2,
                1.2,
            ));
            obs.extend(place_barricade_line(
                nav,
                rng,
                cols / 4 + 1,
                3 * cols / 4 + 1,
                2 * rows / 3,
                3,
                2,
                1.2,
            ));
            obs.extend(place_pillar_grid(
                nav,
                rng,
                cols / 3,
                rows / 4,
                2 * cols / 3,
                3 * rows / 4,
                5,
                1,
                1.5,
            ));
        }
        2 => {
            let gap_col = cols / 2;
            let wall_len = rows / 2 - 2;
            obs.extend(place_wall_segment(
                nav,
                rng,
                gap_col - 1,
                1,
                wall_len,
                false,
                0,
                1.8,
            ));
            obs.extend(place_wall_segment(
                nav,
                rng,
                gap_col + 1,
                1,
                wall_len,
                false,
                0,
                1.8,
            ));
            obs.extend(place_wall_segment(
                nav,
                rng,
                gap_col - 1,
                rows / 2 + 2,
                wall_len,
                false,
                0,
                1.8,
            ));
            obs.extend(place_wall_segment(
                nav,
                rng,
                gap_col + 1,
                rows / 2 + 2,
                wall_len,
                false,
                0,
                1.8,
            ));
            obs.extend(place_cover_cluster(nav, rng, cols / 3, rows / 2, 2, 2, 1.0));
        }
        3 => {
            // "Elevated Chokepoint": raised centre with sandbag defenses
            obs.extend(place_elevated_platform(
                nav,
                rng,
                cols / 3,
                rows / 4,
                cols / 3,
                rows / 2,
                1.0,
            ));
            obs.extend(place_sandbag_arc(nav, rng, cols / 3, rows / 2, 2, 4, 0.7));
            obs.extend(place_sandbag_arc(
                nav,
                rng,
                2 * cols / 3,
                rows / 2,
                2,
                4,
                0.7,
            ));
            obs.extend(place_barricade_line(
                nav,
                rng,
                cols / 4,
                3 * cols / 4,
                rows / 3,
                2,
                2,
                0.8,
            ));
        }
        _ => {
            obs.extend(place_cover_cluster(nav, rng, cols / 4, rows / 2, 3, 3, 1.0));
            obs.extend(place_cover_cluster(nav, rng, cols / 2, rows / 3, 3, 3, 1.0));
            obs.extend(place_cover_cluster(
                nav,
                rng,
                3 * cols / 4,
                rows / 2,
                3,
                3,
                1.0,
            ));
            obs.extend(place_wall_segment(
                nav,
                rng,
                cols / 2 - 2,
                rows / 2,
                5,
                true,
                0,
                1.2,
            ));
        }
    }
    obs
}

pub(crate) fn generate_pivot_obstacles(nav: &mut NavGrid, rng: &mut Lcg) -> Vec<ObstacleRegion> {
    let cols = nav.cols;
    let rows = nav.rows;
    let mid_c = cols / 2;
    let mid_r = rows / 2;
    let template = rng.next_usize_range(0, 4);
    let mut obs = Vec::new();
    match template {
        0 => {
            obs.extend(place_wall_segment(
                nav,
                rng,
                cols / 4,
                rows / 2,
                cols / 2,
                true,
                2,
                1.5,
            ));
            obs.extend(place_wall_segment(
                nav,
                rng,
                cols / 2,
                rows / 4,
                rows / 2,
                false,
                2,
                1.5,
            ));
        }
        1 => {
            obs.extend(place_l_shape(nav, rng, cols / 4, rows / 4, 3, 1, 0, 1.5));
            obs.extend(place_l_shape(
                nav,
                rng,
                3 * cols / 4,
                3 * rows / 4,
                3,
                1,
                2,
                1.5,
            ));
        }
        2 => {
            // "High Ground Pivot": two elevated platforms with cover lips
            obs.extend(place_elevated_platform(
                nav,
                rng,
                cols / 4,
                rows / 4,
                3,
                3,
                1.0,
            ));
            obs.extend(place_elevated_platform(
                nav,
                rng,
                3 * cols / 4 - 2,
                3 * rows / 4 - 2,
                3,
                3,
                1.0,
            ));
            obs.extend(place_cover_cluster(nav, rng, cols / 2, rows / 2, 2, 3, 1.0));
        }
        3 => {
            // "Moated Pivot": central moat ring with bridges, forcing pivots around chokepoints
            obs.extend(place_moat(nav, rng, mid_c, mid_r, 3, 2));
            obs.extend(place_cover_cluster(nav, rng, mid_c, mid_r, 1, 2, 0.8));
        }
        _ => {
            obs.extend(place_pillar_grid(
                nav,
                rng,
                cols / 4,
                rows / 4,
                3 * cols / 4,
                3 * rows / 4,
                3,
                1,
                1.5,
            ));
        }
    }
    obs
}

pub(crate) fn generate_setpiece_obstacles(nav: &mut NavGrid, rng: &mut Lcg) -> Vec<ObstacleRegion> {
    let cols = nav.cols;
    let rows = nav.rows;
    let mid_c = cols / 2;
    let mid_r = rows / 2;
    let template = rng.next_usize_range(0, 7);
    let mut obs = Vec::new();
    match template {
        0 => {
            obs.extend(place_corridor_walls(
                nav,
                rng,
                cols / 3,
                2 * cols / 3,
                rows / 2,
                6,
                1.8,
            ));
            obs.extend(place_l_shape(nav, rng, cols / 3, rows / 3, 4, 1, 0, 1.5));
            obs.extend(place_l_shape(
                nav,
                rng,
                2 * cols / 3,
                2 * rows / 3,
                4,
                1,
                2,
                1.5,
            ));
            obs.extend(place_cover_cluster(nav, rng, cols / 2, rows / 2, 3, 3, 1.0));
        }
        1 => {
            obs.extend(place_pillar_grid(
                nav,
                rng,
                cols / 4,
                rows / 4,
                3 * cols / 4,
                3 * rows / 4,
                5,
                2,
                1.5,
            ));
            obs.extend(place_wall_segment(
                nav,
                rng,
                cols / 3,
                rows / 3,
                cols / 4,
                true,
                3,
                1.2,
            ));
            obs.extend(place_wall_segment(
                nav,
                rng,
                cols / 2,
                2 * rows / 3,
                cols / 4,
                true,
                2,
                1.2,
            ));
            obs.extend(place_barricade_line(
                nav,
                rng,
                cols / 4,
                3 * cols / 4,
                rows / 2,
                4,
                3,
                1.0,
            ));
        }
        2 => {
            let wall_len_v = rows / 2;
            let wall_len_h = cols / 2;
            obs.extend(place_wall_segment(
                nav,
                rng,
                cols / 4,
                rows / 4,
                wall_len_v,
                false,
                3,
                1.5,
            ));
            obs.extend(place_wall_segment(
                nav,
                rng,
                3 * cols / 4,
                rows / 4,
                wall_len_v,
                false,
                3,
                1.5,
            ));
            obs.extend(place_wall_segment(
                nav,
                rng,
                cols / 4,
                rows / 4,
                wall_len_h,
                true,
                3,
                1.5,
            ));
            obs.extend(place_wall_segment(
                nav,
                rng,
                cols / 4,
                3 * rows / 4,
                wall_len_h,
                true,
                3,
                1.5,
            ));
            obs.extend(place_cover_cluster(nav, rng, cols / 3, rows / 3, 3, 3, 1.0));
            obs.extend(place_cover_cluster(
                nav,
                rng,
                2 * cols / 3,
                rows / 2,
                3,
                3,
                1.0,
            ));
            obs.extend(place_cover_cluster(
                nav,
                rng,
                cols / 2,
                2 * rows / 3,
                3,
                3,
                1.0,
            ));
        }
        3 => {
            // "Fortress Moat": concentric moat ring with elevated centre
            obs.extend(place_moat(nav, rng, mid_c, mid_r, rows / 4, 3));
            obs.extend(place_elevated_platform(
                nav,
                rng,
                mid_c - 2,
                mid_r - 2,
                4,
                4,
                1.2,
            ));
            obs.extend(place_cover_cluster(nav, rng, cols / 4, rows / 4, 3, 2, 1.0));
            obs.extend(place_cover_cluster(
                nav,
                rng,
                3 * cols / 4,
                3 * rows / 4,
                3,
                2,
                1.0,
            ));
        }
        4 => {
            // "Spiral Arena": dual spirals creating maze-like approach
            obs.extend(place_spiral(nav, rng, cols / 3, mid_r, rows / 4, 1.0, 1.5));
            obs.extend(place_spiral(
                nav,
                rng,
                2 * cols / 3,
                mid_r,
                rows / 4,
                1.0,
                1.5,
            ));
            obs.extend(place_barricade_line(
                nav,
                rng,
                cols / 3,
                2 * cols / 3,
                mid_r,
                3,
                2,
                1.0,
            ));
        }
        5 => {
            // "Broken Ruins": asymmetric walls and scattered debris.
            // One side has heavy cover, the other is exposed — forces
            // tactical decisions about which flank to push.
            obs.extend(place_wall_segment(
                nav, rng, cols / 5, rows / 4, rows / 2, false, 2, 1.8,
            ));
            obs.extend(place_l_shape(nav, rng, cols / 5, rows / 4, 4, 1, 0, 1.5));
            obs.extend(place_cover_cluster(nav, rng, cols / 4, 2 * rows / 3, 3, 3, 1.2));
            obs.extend(place_cover_cluster(nav, rng, cols / 3, rows / 3, 2, 2, 1.0));
            // Right side: sparse — only sandbag arcs
            obs.extend(place_sandbag_arc(nav, rng, 3 * cols / 4, rows / 3, 3, 5, 0.8));
            obs.extend(place_barricade_line(
                nav, rng, cols / 2 + 2, 3 * cols / 4, 2 * rows / 3, 3, 2, 1.0,
            ));
            // Central elevated platform off-centre
            obs.extend(place_elevated_platform(
                nav, rng, mid_c + 2, mid_r - 1, 4, 3, 1.2,
            ));
        }
        6 => {
            // "Diagonal Split": diagonal wall bisects the room, creating
            // two distinct zones with different cover characteristics.
            // Stagger wall segments diagonally from top-left to bottom-right.
            let steps = 5;
            for i in 0..steps {
                let c = cols / 6 + i * (2 * cols / 3) / steps;
                let r = rows / 6 + i * (2 * rows / 3) / steps;
                obs.extend(place_wall_segment(
                    nav, rng, c, r, 3, true, 1, 1.5,
                ));
            }
            // Heavy cover in top-right zone
            obs.extend(place_l_shape(nav, rng, 3 * cols / 4, rows / 4, 3, 1, 1, 1.5));
            obs.extend(place_cover_cluster(nav, rng, 2 * cols / 3, rows / 3, 3, 2, 1.0));
            obs.extend(place_pillar_grid(
                nav, rng, 3 * cols / 5, rows / 5, 4 * cols / 5, 2 * rows / 5, 4, 1, 1.2,
            ));
            // Sparse cover in bottom-left zone — exposed approach
            obs.extend(place_sandbag_arc(nav, rng, cols / 4, 2 * rows / 3, 2, 4, 0.7));
            obs.extend(place_barricade_line(
                nav, rng, cols / 5, cols / 3, 3 * rows / 4, 2, 2, 0.8,
            ));
        }
        _ => {
            let wall_len = rows / 3;
            obs.extend(place_wall_segment(
                nav, rng, cols / 4, rows / 4, wall_len, false, 2, 1.5,
            ));
            obs.extend(place_wall_segment(
                nav, rng, cols / 2, rows / 3, wall_len, false, 2, 1.5,
            ));
            obs.extend(place_wall_segment(
                nav, rng, 3 * cols / 4, rows / 4, wall_len, false, 2, 1.5,
            ));
            obs.extend(place_cover_cluster(nav, rng, cols / 3, rows / 2, 3, 2, 1.0));
            obs.extend(place_cover_cluster(
                nav, rng, 2 * cols / 3, rows / 2, 3, 2, 1.0,
            ));
        }
    }
    obs
}

pub(crate) fn generate_recovery_obstacles(nav: &mut NavGrid, rng: &mut Lcg) -> Vec<ObstacleRegion> {
    let cols = nav.cols;
    let rows = nav.rows;
    let mid_c = cols / 2;
    let mid_r = rows / 2;
    let template = rng.next_usize_range(0, 3);
    let mut obs = Vec::new();
    match template {
        0 => {
            obs.extend(place_pillar_grid(
                nav,
                rng,
                cols / 3,
                rows / 3,
                2 * cols / 3,
                2 * rows / 3,
                3,
                1,
                1.2,
            ));
        }
        1 => {
            obs.extend(place_cover_cluster(nav, rng, cols / 2, rows / 2, 2, 4, 1.0));
            obs.extend(place_cover_cluster(
                nav,
                rng,
                cols / 3,
                2 * rows / 3,
                2,
                2,
                1.0,
            ));
        }
        2 => {
            // "Sanctuary Ring": gentle moat with wide bridges — safe but structured
            obs.extend(place_moat(nav, rng, mid_c, mid_r, 3, 4));
            obs.extend(place_cover_cluster(nav, rng, mid_c, mid_r, 1, 2, 0.6));
        }
        _ => {
            obs.extend(place_wall_segment(
                nav,
                rng,
                cols / 3,
                rows / 3,
                4,
                false,
                0,
                1.2,
            ));
            obs.extend(place_wall_segment(
                nav,
                rng,
                2 * cols / 3,
                rows / 3,
                4,
                false,
                0,
                1.2,
            ));
            obs.extend(place_cover_cluster(nav, rng, cols / 2, rows / 2, 2, 2, 1.0));
        }
    }
    obs
}

pub(crate) fn generate_climax_obstacles(nav: &mut NavGrid, rng: &mut Lcg) -> Vec<ObstacleRegion> {
    let cols = nav.cols;
    let rows = nav.rows;
    let mid_c = cols / 2;
    let mid_r = rows / 2;
    let template = rng.next_usize_range(0, 6);
    let mut obs = Vec::new();
    match template {
        0 => {
            obs.extend(place_pillar_grid(
                nav,
                rng,
                cols / 4,
                rows / 4,
                3 * cols / 4,
                3 * rows / 4,
                4,
                1,
                1.8,
            ));
            obs.extend(place_barricade_line(
                nav,
                rng,
                cols / 3,
                2 * cols / 3,
                rows / 3,
                3,
                2,
                1.2,
            ));
            obs.extend(place_barricade_line(
                nav,
                rng,
                cols / 3,
                2 * cols / 3,
                2 * rows / 3,
                3,
                2,
                1.2,
            ));
            obs.extend(place_cover_cluster(nav, rng, cols / 3, rows / 2, 3, 2, 1.0));
            obs.extend(place_cover_cluster(
                nav,
                rng,
                2 * cols / 3,
                rows / 2,
                3,
                2,
                1.0,
            ));
        }
        1 => {
            obs.extend(place_corridor_walls(
                nav,
                rng,
                cols / 4,
                3 * cols / 4,
                rows / 2,
                8,
                1.8,
            ));
            obs.extend(place_l_shape(nav, rng, cols / 4, rows / 4, 4, 1, 0, 1.5));
            obs.extend(place_l_shape(
                nav,
                rng,
                3 * cols / 4,
                rows / 4,
                4,
                1,
                1,
                1.5,
            ));
            obs.extend(place_l_shape(
                nav,
                rng,
                cols / 4,
                3 * rows / 4,
                4,
                1,
                3,
                1.5,
            ));
            obs.extend(place_l_shape(
                nav,
                rng,
                3 * cols / 4,
                3 * rows / 4,
                4,
                1,
                2,
                1.5,
            ));
            obs.extend(place_cover_cluster(nav, rng, cols / 2, rows / 2, 4, 3, 1.0));
        }
        2 => {
            let wall_len = cols / 3;
            obs.extend(place_wall_segment(
                nav,
                rng,
                cols / 4,
                rows / 3,
                wall_len,
                true,
                3,
                1.5,
            ));
            obs.extend(place_wall_segment(
                nav,
                rng,
                cols / 3,
                rows / 2,
                wall_len,
                true,
                3,
                1.5,
            ));
            obs.extend(place_wall_segment(
                nav,
                rng,
                cols / 4,
                2 * rows / 3,
                wall_len,
                true,
                3,
                1.5,
            ));
            obs.extend(place_cover_cluster(nav, rng, cols / 4, rows / 4, 3, 2, 1.0));
            obs.extend(place_cover_cluster(
                nav,
                rng,
                3 * cols / 4,
                rows / 4,
                3,
                2,
                1.0,
            ));
            obs.extend(place_cover_cluster(
                nav,
                rng,
                cols / 4,
                3 * rows / 4,
                3,
                2,
                1.0,
            ));
            obs.extend(place_cover_cluster(
                nav,
                rng,
                3 * cols / 4,
                3 * rows / 4,
                3,
                2,
                1.0,
            ));
        }
        3 => {
            // "Throne Room Siege": moated centre with elevated throne and spiral approaches
            obs.extend(place_moat(nav, rng, mid_c, mid_r, rows / 5, 2));
            obs.extend(place_elevated_platform(
                nav,
                rng,
                mid_c - 2,
                mid_r - 2,
                4,
                4,
                1.5,
            ));
            obs.extend(place_spiral(nav, rng, cols / 4, mid_r, rows / 6, 0.8, 1.2));
            obs.extend(place_spiral(
                nav,
                rng,
                3 * cols / 4,
                mid_r,
                rows / 6,
                0.8,
                1.2,
            ));
            obs.extend(place_sandbag_arc(nav, rng, mid_c, rows / 4, 3, 4, 0.7));
            obs.extend(place_sandbag_arc(nav, rng, mid_c, 3 * rows / 4, 3, 4, 0.7));
        }
        4 => {
            // "Shattered Keep": asymmetric ruin with one intact wing and one
            // collapsed wing.  Left side has tall walls (intact), right side
            // has scattered rubble (collapsed).
            obs.extend(place_wall_segment(
                nav, rng, cols / 5, rows / 5, rows / 2, false, 2, 2.0,
            ));
            obs.extend(place_wall_segment(
                nav, rng, cols / 5, rows / 5, cols / 3, true, 1, 2.0,
            ));
            obs.extend(place_l_shape(nav, rng, cols / 5 + 1, 2 * rows / 5, 4, 1, 0, 1.8));
            obs.extend(place_elevated_platform(
                nav, rng, cols / 5 + 2, rows / 5 + 1, 3, 3, 1.0,
            ));
            // Right side: scattered rubble from the collapse
            obs.extend(place_cover_cluster(nav, rng, 3 * cols / 5, rows / 3, 3, 3, 0.8));
            obs.extend(place_cover_cluster(nav, rng, 2 * cols / 3, rows / 2, 2, 4, 0.6));
            obs.extend(place_cover_cluster(nav, rng, 3 * cols / 4, 2 * rows / 3, 3, 2, 0.9));
            obs.extend(place_barricade_line(
                nav, rng, cols / 2, 3 * cols / 4, 3 * rows / 4, 4, 2, 0.7,
            ));
            obs.extend(place_sandbag_arc(nav, rng, 2 * cols / 3, rows / 4, 2, 5, 0.6));
        }
        5 => {
            // "Flanking Gauntlet": heavy central obstacle forces teams around
            // the sides, with asymmetric cover on each flank.
            // Centre: impassable block
            obs.extend(place_wall_segment(
                nav, rng, mid_c - 2, mid_r - 3, 6, false, 0, 2.0,
            ));
            obs.extend(place_wall_segment(
                nav, rng, mid_c - 2, mid_r - 3, 5, true, 0, 2.0,
            ));
            obs.extend(place_wall_segment(
                nav, rng, mid_c + 2, mid_r - 3, 6, false, 0, 2.0,
            ));
            obs.extend(place_elevated_platform(
                nav, rng, mid_c - 1, mid_r - 2, 3, 4, 1.5,
            ));
            // Top flank: dense cover (defender advantage)
            obs.extend(place_l_shape(nav, rng, cols / 4, rows / 5, 3, 1, 0, 1.5));
            obs.extend(place_pillar_grid(
                nav, rng, cols / 3, rows / 6, 2 * cols / 3, rows / 3, 5, 1, 1.2,
            ));
            // Bottom flank: open with sparse sandbags (attacker's fast route)
            obs.extend(place_sandbag_arc(nav, rng, cols / 3, 3 * rows / 4, 2, 4, 0.7));
            obs.extend(place_barricade_line(
                nav, rng, cols / 2, 3 * cols / 4, 4 * rows / 5, 2, 2, 0.8,
            ));
        }
        _ => {
            let wall_len_h = cols / 2;
            let wall_len_v = rows / 2;
            obs.extend(place_wall_segment(
                nav,
                rng,
                cols / 4,
                rows / 4,
                wall_len_h,
                true,
                3,
                1.5,
            ));
            obs.extend(place_wall_segment(
                nav,
                rng,
                cols / 4,
                3 * rows / 4,
                wall_len_h,
                true,
                3,
                1.5,
            ));
            obs.extend(place_wall_segment(
                nav,
                rng,
                cols / 4,
                rows / 4,
                wall_len_v,
                false,
                3,
                1.5,
            ));
            obs.extend(place_wall_segment(
                nav,
                rng,
                3 * cols / 4,
                rows / 4,
                wall_len_v,
                false,
                3,
                1.5,
            ));
            obs.extend(place_pillar_grid(
                nav,
                rng,
                cols / 3,
                rows / 3,
                2 * cols / 3,
                2 * rows / 3,
                4,
                1,
                1.5,
            ));
        }
    }
    obs
}

/// Fallback: deterministic grid of 1x1 blocks in the centre (always valid).
pub(crate) fn generate_fallback_obstacles(
    nav: &mut NavGrid,
    _rng: &mut Lcg,
) -> Vec<ObstacleRegion> {
    let cols = nav.cols;
    let rows = nav.rows;
    let mut obs = Vec::new();
    let c_lo = cols / 4;
    let c_hi = 3 * cols / 4;
    let r_lo = rows / 4;
    let r_hi = 3 * rows / 4;
    for r in (r_lo..=r_hi).step_by(2) {
        for c in (c_lo..=c_hi).step_by(2) {
            if c > 0 && c < cols - 1 && r > 0 && r < rows - 1 {
                nav.set_walkable_rect(c, r, c, r, false);
                obs.push(ObstacleRegion {
                    col0: c,
                    col1: c,
                    row0: r,
                    row1: r,
                    height: 1.0,
                });
            }
        }
    }
    obs
}
