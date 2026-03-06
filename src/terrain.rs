use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology, VertexAttributeValues};
use bevy::render::render_asset::RenderAssetUsages;

use crate::game_core;
use crate::hub_types::OverworldTerrainVisual;

// ---------------------------------------------------------------------------
// Hash utilities
// ---------------------------------------------------------------------------

fn splitmix(v: u64) -> u64 {
    let mut z = v;
    z ^= z >> 30;
    z = z.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z ^= z >> 27;
    z = z.wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    z
}

fn hash_cell(seed: u64, ix: i32, iz: i32) -> u64 {
    splitmix(
        seed ^ ((ix as i64 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15))
            ^ ((iz as i64 as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)),
    )
}

fn hash01(seed: u64, ix: i32, iz: i32) -> f32 {
    (hash_cell(seed, ix, iz) >> 11) as f32 / ((1u64 << 53) as f32)
}

// ---------------------------------------------------------------------------
// Value noise (kept for backward compatibility)
// ---------------------------------------------------------------------------

pub fn sample_value_noise(seed: u64, x: f32, z: f32) -> f32 {
    let x0 = x.floor() as i32;
    let z0 = z.floor() as i32;
    let tx = x - x.floor();
    let tz = z - z.floor();
    let smooth = |t: f32| t * t * t * (t * (t * 6.0 - 15.0) + 10.0); // quintic
    let sx = smooth(tx);
    let sz = smooth(tz);
    let v00 = hash01(seed, x0, z0);
    let v10 = hash01(seed, x0 + 1, z0);
    let v01 = hash01(seed, x0, z0 + 1);
    let v11 = hash01(seed, x0 + 1, z0 + 1);
    let a = v00 + (v10 - v00) * sx;
    let b = v01 + (v11 - v01) * sx;
    a + (b - a) * sz
}

// ---------------------------------------------------------------------------
// Gradient noise (Perlin-style with analytic gradient vectors)
// ---------------------------------------------------------------------------

fn grad2(seed: u64, ix: i32, iz: i32) -> (f32, f32) {
    let h = hash_cell(seed, ix, iz);
    let angle = (h & 0xFFFF) as f32 * (std::f32::consts::TAU / 65536.0);
    (angle.cos(), angle.sin())
}

/// Sample 2D gradient noise returning a value roughly in [-1, 1].
pub fn sample_gradient_noise(seed: u64, x: f32, z: f32) -> f32 {
    let x0 = x.floor() as i32;
    let z0 = z.floor() as i32;
    let tx = x - x.floor();
    let tz = z - z.floor();
    let fade = |t: f32| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    let sx = fade(tx);
    let sz = fade(tz);

    let dot_grid = |ix: i32, iz: i32, dx: f32, dz: f32| -> f32 {
        let (gx, gz) = grad2(seed, ix, iz);
        gx * dx + gz * dz
    };

    let n00 = dot_grid(x0, z0, tx, tz);
    let n10 = dot_grid(x0 + 1, z0, tx - 1.0, tz);
    let n01 = dot_grid(x0, z0 + 1, tx, tz - 1.0);
    let n11 = dot_grid(x0 + 1, z0 + 1, tx - 1.0, tz - 1.0);
    let a = n00 + (n10 - n00) * sx;
    let b = n01 + (n11 - n01) * sx;
    a + (b - a) * sz
}

// ---------------------------------------------------------------------------
// Domain warping: distort input coordinates for organic feel
// ---------------------------------------------------------------------------

fn domain_warp(seed: u64, x: f32, z: f32, strength: f32) -> (f32, f32) {
    let wx = sample_gradient_noise(seed ^ 0x1234_5678, x * 0.4, z * 0.4) * strength;
    let wz = sample_gradient_noise(seed ^ 0x8765_4321, x * 0.4, z * 0.4) * strength;
    (x + wx, z + wz)
}

// ---------------------------------------------------------------------------
// fBm (fractal Brownian motion) using gradient noise
// ---------------------------------------------------------------------------

fn fbm_gradient(seed: u64, x: f32, z: f32, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    let mut amplitude = 1.0_f32;
    let mut frequency = 1.0_f32;
    let mut sum = 0.0_f32;
    let mut norm = 0.0_f32;
    for i in 0..octaves {
        let n = sample_gradient_noise(
            seed ^ ((i as u64).wrapping_mul(0xD6E8_FD9A_5A11_6D23)),
            x * frequency,
            z * frequency,
        );
        sum += n * amplitude;
        norm += amplitude;
        amplitude *= gain;
        frequency *= lacunarity;
    }
    if norm > 0.0 {
        sum / norm
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Ridge noise: absolute-value inversion for mountain ridges
// ---------------------------------------------------------------------------

fn ridge_noise(seed: u64, x: f32, z: f32, octaves: u32, lacunarity: f32, gain: f32) -> f32 {
    let mut amplitude = 1.0_f32;
    let mut frequency = 1.0_f32;
    let mut sum = 0.0_f32;
    let mut norm = 0.0_f32;
    let mut prev = 1.0_f32;
    for i in 0..octaves {
        let n = sample_gradient_noise(
            seed ^ ((i as u64).wrapping_mul(0xA3F1_9C5E_7B2D_4018)),
            x * frequency,
            z * frequency,
        );
        let ridge = 1.0 - n.abs();
        let ridge = ridge * ridge * prev;
        sum += ridge * amplitude;
        norm += amplitude;
        prev = ridge;
        amplitude *= gain;
        frequency *= lacunarity;
    }
    if norm > 0.0 {
        sum / norm
    } else {
        0.0
    }
}

// ---------------------------------------------------------------------------
// Biome classification
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Biome {
    DeepWater,
    ShallowWater,
    Beach,
    Grassland,
    Forest,
    Highland,
    Mountain,
    Snow,
}

impl Biome {
    /// Classify a biome from normalised height (0..1) and moisture (0..1).
    pub fn from_height_moisture(height: f32, moisture: f32) -> Self {
        if height < 0.18 {
            Biome::DeepWater
        } else if height < 0.25 {
            Biome::ShallowWater
        } else if height < 0.30 {
            Biome::Beach
        } else if height < 0.55 {
            if moisture > 0.5 {
                Biome::Forest
            } else {
                Biome::Grassland
            }
        } else if height < 0.72 {
            Biome::Highland
        } else if height < 0.88 {
            Biome::Mountain
        } else {
            Biome::Snow
        }
    }

    /// Return an (r, g, b) base colour for this biome.
    pub fn color(self) -> (f32, f32, f32) {
        match self {
            Biome::DeepWater => (0.04, 0.10, 0.22),
            Biome::ShallowWater => (0.08, 0.20, 0.32),
            Biome::Beach => (0.76, 0.70, 0.50),
            Biome::Grassland => (0.28, 0.42, 0.18),
            Biome::Forest => (0.12, 0.30, 0.10),
            Biome::Highland => (0.38, 0.36, 0.28),
            Biome::Mountain => (0.46, 0.44, 0.42),
            Biome::Snow => (0.88, 0.90, 0.92),
        }
    }
}

// ---------------------------------------------------------------------------
// Improved overworld height sampling
// ---------------------------------------------------------------------------

pub fn sample_overworld_height(seed: u64, x: f32, z: f32) -> f32 {
    // Apply domain warping for organic, non-grid-aligned features
    let (wx, wz) = domain_warp(seed, x, z, 1.8);

    // Continental shape: large-scale gradient noise
    let continent = fbm_gradient(seed, wx * 0.15, wz * 0.15, 4, 2.0, 0.5);

    // Detail layer: higher-frequency fBm
    let detail = fbm_gradient(seed ^ 0xCAFE_BABE, wx * 0.7, wz * 0.7, 6, 2.1, 0.48);

    // Ridge mountains: sharp peaks along geological fault lines
    let ridges = ridge_noise(seed ^ 0xDEAD_BEEF, wx * 0.28, wz * 0.28, 5, 2.2, 0.5);

    // Basin depressions: broad gentle valleys
    let basin = sample_gradient_noise(seed ^ 0x5F4C_3AA1, wx * 0.12, wz * 0.12);

    // Terrace effect: creates stepped plateaus at certain elevations
    let raw = continent * 5.5 + detail * 1.8 + ridges * 3.2 - basin.abs() * 1.5;
    let terrace_strength = 0.25;
    let terrace_count = 6.0_f32;
    let terraced = (raw * terrace_count / 10.0).round() * (10.0 / terrace_count);
    raw * (1.0 - terrace_strength) + terraced * terrace_strength
}

/// Sample a moisture value for biome classification.
pub fn sample_moisture(seed: u64, x: f32, z: f32) -> f32 {
    let n = fbm_gradient(seed ^ 0x7777_AAAA, x * 0.22, z * 0.22, 3, 2.0, 0.5);
    (n * 0.5 + 0.5).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Thermal erosion simulation on the heightmap
// ---------------------------------------------------------------------------

fn apply_thermal_erosion(heights: &mut [f32], side: usize, iterations: usize, talus: f32) {
    let rate = 0.4_f32;
    for _ in 0..iterations {
        for z in 1..side - 1 {
            for x in 1..side - 1 {
                let idx = z * side + x;
                let h = heights[idx];
                let neighbors = [
                    (z * side + x + 1),
                    (z * side + x - 1),
                    ((z + 1) * side + x),
                    ((z - 1) * side + x),
                ];
                let mut max_diff = 0.0_f32;
                let mut max_idx = idx;
                for &ni in &neighbors {
                    let diff = h - heights[ni];
                    if diff > max_diff {
                        max_diff = diff;
                        max_idx = ni;
                    }
                }
                if max_diff > talus && max_idx != idx {
                    let transfer = (max_diff - talus) * rate * 0.5;
                    heights[idx] -= transfer;
                    heights[max_idx] += transfer;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Mesh generation
// ---------------------------------------------------------------------------

pub fn build_overworld_terrain_mesh(seed: u64, subdivisions: usize, half_extent: f32) -> Mesh {
    let vert_side = subdivisions + 1;
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(vert_side * vert_side);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(vert_side * vert_side);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(vert_side * vert_side);
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(vert_side * vert_side);
    let mut heights = vec![0.0_f32; vert_side * vert_side];
    let idx = |x: usize, z: usize| z * vert_side + x;

    // --- Pass 1: compute raw heights ---
    for z in 0..=subdivisions {
        for x in 0..=subdivisions {
            let tx = x as f32 / subdivisions as f32;
            let tz = z as f32 / subdivisions as f32;
            let px = (tx * 2.0 - 1.0) * half_extent;
            let pz = (tz * 2.0 - 1.0) * half_extent;
            let h = sample_overworld_height(seed, px * 0.16, pz * 0.16);
            heights[idx(x, z)] = h;
        }
    }

    // --- Pass 2: thermal erosion for natural slopes ---
    apply_thermal_erosion(&mut heights, vert_side, 3, 0.8);

    // --- Find height range for normalisation ---
    let h_min = heights.iter().copied().fold(f32::INFINITY, f32::min);
    let h_max = heights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let h_range = (h_max - h_min).max(0.01);

    // --- Pass 3: build positions, UVs, and per-vertex biome colours ---
    for z in 0..=subdivisions {
        for x in 0..=subdivisions {
            let tx = x as f32 / subdivisions as f32;
            let tz = z as f32 / subdivisions as f32;
            let px = (tx * 2.0 - 1.0) * half_extent;
            let pz = (tz * 2.0 - 1.0) * half_extent;
            let h = heights[idx(x, z)];

            positions.push([px, h, pz]);
            uvs.push([tx, tz]);

            // Biome colouring
            let h_norm = (h - h_min) / h_range;
            let moisture = sample_moisture(seed, px * 0.16, pz * 0.16);
            let biome = Biome::from_height_moisture(h_norm, moisture);
            let (br, bg, bb) = biome.color();

            // Add subtle noise variation to prevent banding
            let color_jitter =
                sample_value_noise(seed ^ 0xC010_8888, px * 0.5, pz * 0.5) * 0.06 - 0.03;
            colors.push([
                (br + color_jitter).clamp(0.0, 1.0),
                (bg + color_jitter).clamp(0.0, 1.0),
                (bb + color_jitter).clamp(0.0, 1.0),
                1.0,
            ]);
        }
    }

    // --- Pass 4: normals from finite differences ---
    for z in 0..=subdivisions {
        for x in 0..=subdivisions {
            let x_l = x.saturating_sub(1);
            let x_r = (x + 1).min(subdivisions);
            let z_d = z.saturating_sub(1);
            let z_u = (z + 1).min(subdivisions);
            let h_l = heights[idx(x_l, z)];
            let h_r = heights[idx(x_r, z)];
            let h_d = heights[idx(x, z_d)];
            let h_u = heights[idx(x, z_u)];
            let n = Vec3::new(h_l - h_r, 2.4, h_d - h_u).normalize_or_zero();
            normals.push([n.x, n.y, n.z]);
        }
    }

    // --- Pass 5: triangle indices ---
    let mut indices: Vec<u32> = Vec::with_capacity(subdivisions * subdivisions * 6);
    for z in 0..subdivisions {
        for x in 0..subdivisions {
            let i0 = idx(x, z) as u32;
            let i1 = idx(x + 1, z) as u32;
            let i2 = idx(x, z + 1) as u32;
            let i3 = idx(x + 1, z + 1) as u32;
            indices.extend_from_slice(&[i0, i2, i1, i1, i2, i3]);
        }
    }

    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_attribute(
        Mesh::ATTRIBUTE_COLOR,
        VertexAttributeValues::Float32x4(colors),
    );
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

// ---------------------------------------------------------------------------
// Scene setup
// ---------------------------------------------------------------------------

pub fn setup_overworld_terrain_scene(
    mut commands: Commands,
    overworld: Res<game_core::OverworldMap>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
) {
    let terrain_mesh = meshes.add(build_overworld_terrain_mesh(overworld.map_seed, 128, 52.0));
    let terrain_material = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        perceptual_roughness: 0.96,
        metallic: 0.02,
        ..default()
    });
    let water_mesh = meshes.add(Mesh::from(bevy::math::primitives::Plane3d::default()));
    let water_material = materials.add(StandardMaterial {
        base_color: Color::rgba(0.06, 0.16, 0.2, 0.82),
        perceptual_roughness: 0.22,
        metallic: 0.0,
        alpha_mode: AlphaMode::Blend,
        ..default()
    });

    commands.spawn((
        PbrBundle {
            mesh: terrain_mesh,
            material: terrain_material,
            visibility: Visibility::Hidden,
            transform: Transform::from_xyz(0.0, -2.2, 0.0),
            ..default()
        },
        OverworldTerrainVisual,
    ));
    commands.spawn((
        PbrBundle {
            mesh: water_mesh,
            material: water_material,
            visibility: Visibility::Hidden,
            transform: Transform::from_xyz(0.0, -1.3, 0.0).with_scale(Vec3::new(140.0, 1.0, 140.0)),
            ..default()
        },
        OverworldTerrainVisual,
    ));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gradient_noise_range() {
        let mut min_v = f32::INFINITY;
        let mut max_v = f32::NEG_INFINITY;
        for i in 0..1000 {
            let x = (i as f32 - 500.0) * 0.13;
            let z = (i as f32 * 0.7 - 350.0) * 0.13;
            let v = sample_gradient_noise(42, x, z);
            min_v = min_v.min(v);
            max_v = max_v.max(v);
        }
        assert!(min_v >= -1.5, "gradient noise too low: {min_v}");
        assert!(max_v <= 1.5, "gradient noise too high: {max_v}");
    }

    #[test]
    fn domain_warp_produces_offset() {
        let (wx, wz) = domain_warp(42, 5.0, 5.0, 2.0);
        assert!((wx - 5.0).abs() > 0.001 || (wz - 5.0).abs() > 0.001);
    }

    #[test]
    fn fbm_deterministic() {
        let a = fbm_gradient(99, 1.0, 2.0, 5, 2.0, 0.5);
        let b = fbm_gradient(99, 1.0, 2.0, 5, 2.0, 0.5);
        assert!((a - b).abs() < 1e-6);
    }

    #[test]
    fn ridge_noise_non_negative() {
        for i in 0..200 {
            let x = (i as f32 - 100.0) * 0.1;
            let z = (i as f32 * 0.3 - 30.0) * 0.1;
            let v = ridge_noise(42, x, z, 4, 2.0, 0.5);
            assert!(v >= -0.01, "ridge noise negative: {v}");
        }
    }

    #[test]
    fn biome_classification_boundaries() {
        assert_eq!(Biome::from_height_moisture(0.10, 0.5), Biome::DeepWater);
        assert_eq!(Biome::from_height_moisture(0.20, 0.5), Biome::ShallowWater);
        assert_eq!(Biome::from_height_moisture(0.27, 0.5), Biome::Beach);
        assert_eq!(Biome::from_height_moisture(0.40, 0.3), Biome::Grassland);
        assert_eq!(Biome::from_height_moisture(0.40, 0.7), Biome::Forest);
        assert_eq!(Biome::from_height_moisture(0.60, 0.5), Biome::Highland);
        assert_eq!(Biome::from_height_moisture(0.80, 0.5), Biome::Mountain);
        assert_eq!(Biome::from_height_moisture(0.95, 0.5), Biome::Snow);
    }

    #[test]
    fn moisture_in_range() {
        for i in 0..200 {
            let x = (i as f32 - 100.0) * 0.2;
            let z = (i as f32 * 0.5) * 0.2;
            let m = sample_moisture(42, x, z);
            assert!(m >= 0.0 && m <= 1.0, "moisture out of range: {m}");
        }
    }

    #[test]
    fn thermal_erosion_smooths() {
        let side = 10;
        let mut heights = vec![0.0_f32; side * side];
        // Create a spike
        heights[5 * side + 5] = 10.0;
        let original_peak = heights[5 * side + 5];
        apply_thermal_erosion(&mut heights, side, 5, 0.5);
        assert!(
            heights[5 * side + 5] < original_peak,
            "erosion should reduce peak"
        );
        // Neighbours should have gained some height
        let neighbor_sum: f32 = [
            heights[5 * side + 6],
            heights[5 * side + 4],
            heights[6 * side + 5],
            heights[4 * side + 5],
        ]
        .iter()
        .sum();
        assert!(neighbor_sum > 0.0, "neighbors should have gained material");
    }

    #[test]
    fn overworld_height_deterministic() {
        let a = sample_overworld_height(42, 3.0, 7.0);
        let b = sample_overworld_height(42, 3.0, 7.0);
        assert!((a - b).abs() < 1e-6);
    }

    #[test]
    fn value_noise_backward_compat() {
        // Value noise should still produce values in [0,1]
        for i in 0..100 {
            let x = i as f32 * 0.3;
            let z = i as f32 * 0.7;
            let v = sample_value_noise(42, x, z);
            assert!(v >= 0.0 && v <= 1.0, "value noise out of range: {v}");
        }
    }
}
