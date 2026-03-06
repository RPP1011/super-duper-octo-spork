use bevy::prelude::*;
use bevy::render::mesh::{Indices, PrimitiveTopology, VertexAttributeValues};
use bevy::render::render_asset::RenderAssetUsages;

use super::lcg::Lcg;
use super::RoomLayout;

// ---------------------------------------------------------------------------
// Bevy components
// ---------------------------------------------------------------------------

#[derive(Component)]
pub struct RoomFloor;

#[derive(Component)]
pub struct RoomWall;

#[derive(Component)]
pub struct RoomObstacle;

#[derive(Component)]
pub struct RoomScatter;

// ---------------------------------------------------------------------------
// Floor mesh generation (per-vertex height + colour)
// ---------------------------------------------------------------------------

fn build_floor_mesh(layout: &RoomLayout) -> Mesh {
    let sub = layout.floor_subdivisions;
    let w = layout.width;
    let d = layout.depth;

    let vert_count = sub * sub;
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(vert_count);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(vert_count);
    let mut uvs: Vec<[f32; 2]> = Vec::with_capacity(vert_count);

    // Positions and UVs
    for vz in 0..sub {
        for vx in 0..sub {
            let tx = vx as f32 / (sub - 1).max(1) as f32;
            let tz = vz as f32 / (sub - 1).max(1) as f32;
            let px = tx * w;
            let pz = tz * d;
            let idx = vz * sub + vx;
            let h = layout.floor_heights[idx];
            positions.push([px, h, pz]);
            uvs.push([tx, tz]);
        }
    }

    // Normals via finite differences
    for vz in 0..sub {
        for vx in 0..sub {
            let x_l = vx.saturating_sub(1);
            let x_r = (vx + 1).min(sub - 1);
            let z_d = vz.saturating_sub(1);
            let z_u = (vz + 1).min(sub - 1);
            let h_l = layout.floor_heights[vz * sub + x_l];
            let h_r = layout.floor_heights[vz * sub + x_r];
            let h_d = layout.floor_heights[z_d * sub + vx];
            let h_u = layout.floor_heights[z_u * sub + vx];
            let n = Vec3::new(h_l - h_r, 2.0, h_d - h_u).normalize_or_zero();
            normals.push([n.x, n.y, n.z]);
        }
    }

    // Triangle indices
    let quad_side = sub - 1;
    let mut indices: Vec<u32> = Vec::with_capacity(quad_side * quad_side * 6);
    for vz in 0..quad_side {
        for vx in 0..quad_side {
            let i0 = (vz * sub + vx) as u32;
            let i1 = (vz * sub + vx + 1) as u32;
            let i2 = ((vz + 1) * sub + vx) as u32;
            let i3 = ((vz + 1) * sub + vx + 1) as u32;
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
        VertexAttributeValues::Float32x4(layout.floor_colors.clone()),
    );
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

// ---------------------------------------------------------------------------
// Room visual spawning
// ---------------------------------------------------------------------------

/// Spawn all Bevy PBR meshes for a generated room.
///
/// - Floor: subdivided mesh with per-vertex height and biome-aware colour.
/// - Perimeter walls: four `Cuboid` meshes in environment wall colour.
/// - Obstacle blocks: `Cuboid` meshes with noise-derived height variation.
/// - Ramp regions: inclined `Cuboid` approximations.
/// - Scatter: small detail cuboids (rubble, vines, frost patches).
pub fn spawn_room(
    layout: &RoomLayout,
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    materials: &mut ResMut<Assets<StandardMaterial>>,
) {
    let w = layout.width;
    let d = layout.depth;
    let cs = layout.nav.cell_size;
    let env = layout.environment;

    // --- Floor mesh (subdivided with height + colour) ---
    let floor_mesh = meshes.add(build_floor_mesh(layout));
    let floor_mat = materials.add(StandardMaterial {
        base_color: Color::WHITE, // vertex colours drive the tint
        perceptual_roughness: env.roughness(),
        metallic: 0.0,
        ..default()
    });
    commands.spawn((
        PbrBundle {
            mesh: floor_mesh,
            material: floor_mat,
            ..default()
        },
        RoomFloor,
    ));

    // --- Perimeter walls ---
    let (wr, wg, wb) = env.wall_color();
    let wall_mat = materials.add(StandardMaterial {
        base_color: Color::rgb(wr, wg, wb),
        perceptual_roughness: 0.95,
        metallic: 0.0,
        ..default()
    });
    let wall_thick = 1.0_f32;
    let wall_h = 2.0_f32;

    spawn_wall(commands, meshes, wall_mat.clone(), Vec3::new(w / 2.0, wall_h / 2.0, wall_thick / 2.0), Vec3::new(w, wall_h, wall_thick));
    spawn_wall(commands, meshes, wall_mat.clone(), Vec3::new(w / 2.0, wall_h / 2.0, d - wall_thick / 2.0), Vec3::new(w, wall_h, wall_thick));
    spawn_wall(commands, meshes, wall_mat.clone(), Vec3::new(wall_thick / 2.0, wall_h / 2.0, d / 2.0), Vec3::new(wall_thick, wall_h, d));
    spawn_wall(commands, meshes, wall_mat.clone(), Vec3::new(w - wall_thick / 2.0, wall_h / 2.0, d / 2.0), Vec3::new(wall_thick, wall_h, d));

    // --- Interior obstacle blocks ---
    let (obsr, obsg, obsb) = env.obstacle_color();
    let mut obs_rng = Lcg::new(layout.seed ^ 0x0B57_AC1E);
    for obs in &layout.obstacles {
        let x0 = obs.col0 as f32 * cs;
        let x1 = (obs.col1 + 1) as f32 * cs;
        let z0 = obs.row0 as f32 * cs;
        let z1 = (obs.row1 + 1) as f32 * cs;

        let obs_w = x1 - x0;
        let obs_d = z1 - z0;
        let obs_h = obs.height;

        // Per-obstacle colour variation
        let var = obs_rng.next_f32() * 0.08 - 0.04;
        let height_factor = (obs_h * 0.02).min(0.05);
        let obs_mat = materials.add(StandardMaterial {
            base_color: Color::rgb(
                (obsr - height_factor + var).clamp(0.0, 1.0),
                (obsg - height_factor + var).clamp(0.0, 1.0),
                (obsb - height_factor + var).clamp(0.0, 1.0),
            ),
            perceptual_roughness: env.roughness(),
            metallic: 0.0,
            ..default()
        });
        let obs_mesh = meshes.add(Cuboid::new(obs_w, obs_h, obs_d));
        commands.spawn((
            PbrBundle {
                mesh: obs_mesh,
                material: obs_mat,
                transform: Transform::from_translation(Vec3::new(
                    x0 + obs_w / 2.0,
                    obs_h / 2.0,
                    z0 + obs_d / 2.0,
                )),
                ..default()
            },
            RoomObstacle,
        ));
    }

    // --- Ramp regions ---
    let ramp_mat = materials.add(StandardMaterial {
        base_color: Color::rgb(wr, wg, wb),
        perceptual_roughness: env.roughness(),
        metallic: 0.0,
        ..default()
    });
    for ramp in &layout.ramps {
        let x0 = ramp.col0 as f32 * cs;
        let x1 = (ramp.col1 + 1) as f32 * cs;
        let z0 = ramp.row0 as f32 * cs;
        let z1 = (ramp.row1 + 1) as f32 * cs;

        let ramp_w = x1 - x0;
        let ramp_d = z1 - z0;
        let slab_h = 0.1_f32;

        let angle = (ramp.elevation / ramp_d).atan();
        let centre = Vec3::new(
            x0 + ramp_w / 2.0,
            ramp.elevation / 2.0,
            z0 + ramp_d / 2.0,
        );
        let mut transform = Transform::from_translation(centre);
        transform.rotation = Quat::from_rotation_x(angle);

        let ramp_mesh = meshes.add(Cuboid::new(ramp_w, slab_h, ramp_d));
        commands.spawn((
            PbrBundle {
                mesh: ramp_mesh,
                material: ramp_mat.clone(),
                transform,
                ..default()
            },
            RoomObstacle,
        ));
    }

    // --- Scatter details (rubble, vines, frost, embers) ---
    let (scr, scg, scb) = env.scatter_color();
    let scatter_mat = materials.add(StandardMaterial {
        base_color: Color::rgb(scr, scg, scb),
        perceptual_roughness: env.roughness().min(0.95),
        metallic: 0.0,
        ..default()
    });
    for detail in &layout.scatter {
        let mesh = meshes.add(Cuboid::new(
            detail.scale,
            detail.height,
            detail.scale * 0.8,
        ));
        commands.spawn((
            PbrBundle {
                mesh,
                material: scatter_mat.clone(),
                transform: Transform::from_translation(Vec3::new(
                    detail.x,
                    detail.height / 2.0,
                    detail.z,
                )),
                ..default()
            },
            RoomScatter,
        ));
    }
}

fn spawn_wall(
    commands: &mut Commands,
    meshes: &mut ResMut<Assets<Mesh>>,
    material: Handle<StandardMaterial>,
    centre: Vec3,
    size: Vec3,
) {
    let mesh = meshes.add(Cuboid::new(size.x, size.y, size.z));
    commands.spawn((
        PbrBundle {
            mesh,
            material,
            transform: Transform::from_translation(centre),
            ..default()
        },
        RoomWall,
    ));
}
