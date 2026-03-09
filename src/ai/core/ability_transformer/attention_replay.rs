//! Bevy 3D attention replay plugin.
//!
//! Loads `.attn` recordings and renders attention maps as a navigable 3D scene:
//!
//! - **Y-axis**: transformer layers stacked vertically
//! - **X-axis**: query tokens
//! - **Z-axis**: key tokens
//! - **Column height/color**: attention weight magnitude
//! - **Timeline**: scrub through training steps or game ticks
//!
//! Cross-attention shown as colored beams between ability CLS and entity tokens.
//! egui panel provides head selection, layer toggling, and playback controls.

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use super::attention_recording::{AttentionRecording, LayerAttention};

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

/// Add this plugin to a Bevy `App` to enable 3D attention replay.
pub struct AttentionReplayPlugin;

impl Plugin for AttentionReplayPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ReplayState>()
            .add_systems(Startup, setup_replay_scene)
            .add_systems(
                Update,
                (
                    replay_ui_panel,
                    update_attention_meshes,
                    animate_playback,
                    update_cross_attention_beams,
                ),
            );
    }
}

// ---------------------------------------------------------------------------
// Resources
// ---------------------------------------------------------------------------

/// State for the attention replay viewer.
#[derive(Resource)]
pub struct ReplayState {
    /// The loaded recording (None until a file is loaded).
    pub recording: Option<AttentionRecording>,
    /// Current frame index.
    pub frame_index: usize,
    /// Which attention head to display (0..n_heads).
    pub selected_head: usize,
    /// Which attention source to show.
    pub attention_source: AttentionSource,
    /// Layer visibility toggles.
    pub layer_visible: Vec<bool>,
    /// Auto-playback.
    pub playing: bool,
    /// Playback speed in frames per second.
    pub playback_fps: f32,
    /// Accumulated time for playback stepping.
    pub playback_timer: f32,
    /// Visual scale for attention columns.
    pub column_scale: f32,
    /// Spacing between layers on Y-axis.
    pub layer_spacing: f32,
    /// Show cross-attention beams.
    pub show_cross_attention: bool,
    /// Which ability slot's cross-attention to show (None = all).
    pub cross_attention_slot: Option<usize>,
    /// Whether meshes need rebuild.
    pub dirty: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AttentionSource {
    Transformer,
    EntityEncoder,
}

impl Default for ReplayState {
    fn default() -> Self {
        Self {
            recording: None,
            frame_index: 0,
            selected_head: 0,
            attention_source: AttentionSource::Transformer,
            layer_visible: Vec::new(),
            playing: false,
            playback_fps: 2.0,
            playback_timer: 0.0,
            column_scale: 5.0,
            layer_spacing: 4.0,
            show_cross_attention: true,
            cross_attention_slot: None,
            dirty: true,
        }
    }
}

impl ReplayState {
    /// Load an `.attn` file and reset state.
    pub fn load_recording(&mut self, recording: AttentionRecording) {
        let n_layers = match self.attention_source {
            AttentionSource::Transformer => recording.n_layers,
            AttentionSource::EntityEncoder => recording.n_entity_layers,
        };
        self.layer_visible = vec![true; n_layers.max(recording.n_layers).max(recording.n_entity_layers)];
        self.frame_index = 0;
        self.selected_head = 0;
        self.recording = Some(recording);
        self.dirty = true;
    }
}

// ---------------------------------------------------------------------------
// Components
// ---------------------------------------------------------------------------

/// Marker for attention column meshes so we can despawn/rebuild them.
#[derive(Component)]
pub struct AttentionColumn {
    pub layer: usize,
    pub query: usize,
    pub key: usize,
}

/// Marker for the base grid plane per layer.
#[derive(Component)]
pub struct LayerPlane {
    pub layer: usize,
}

/// Marker for cross-attention beam meshes.
#[derive(Component)]
pub struct CrossAttentionBeam {
    pub ability_slot: usize,
}

/// Marker for token label text.
#[derive(Component)]
pub struct TokenLabel;

// ---------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------

fn setup_replay_scene(mut commands: Commands) {
    // Camera
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(8.0, 12.0, 16.0).looking_at(Vec3::ZERO, Vec3::Y),
        ..default()
    });

    // Ambient light
    commands.insert_resource(AmbientLight {
        color: Color::WHITE,
        brightness: 400.0,
    });

    // Directional light
    commands.spawn(DirectionalLightBundle {
        directional_light: DirectionalLight {
            illuminance: 3000.0,
            shadows_enabled: false,
            ..default()
        },
        transform: Transform::from_rotation(Quat::from_euler(
            EulerRot::XYZ,
            -0.7,
            0.3,
            0.0,
        )),
        ..default()
    });
}

// ---------------------------------------------------------------------------
// egui control panel
// ---------------------------------------------------------------------------

fn replay_ui_panel(mut contexts: EguiContexts, mut state: ResMut<ReplayState>) {
    let Some(recording) = &state.recording else {
        egui::Window::new("Attention Replay").show(contexts.ctx_mut(), |ui| {
            ui.label("No recording loaded.");
            ui.label("Load a .attn file via ReplayState::load_recording()");
        });
        return;
    };

    let n_frames = recording.frames.len();
    let n_heads = recording.n_heads;
    let description = recording.description.clone();

    egui::Window::new("Attention Replay")
        .default_width(320.0)
        .show(contexts.ctx_mut(), |ui| {
            if !description.is_empty() {
                ui.label(&description);
                ui.separator();
            }

            // Source selection
            ui.horizontal(|ui| {
                ui.label("Source:");
                if ui
                    .selectable_label(
                        state.attention_source == AttentionSource::Transformer,
                        "Transformer",
                    )
                    .clicked()
                {
                    state.attention_source = AttentionSource::Transformer;
                    state.dirty = true;
                }
                if ui
                    .selectable_label(
                        state.attention_source == AttentionSource::EntityEncoder,
                        "Entity Encoder",
                    )
                    .clicked()
                {
                    state.attention_source = AttentionSource::EntityEncoder;
                    state.dirty = true;
                }
            });

            ui.separator();

            // Head selection
            ui.horizontal(|ui| {
                ui.label("Head:");
                for h in 0..n_heads {
                    if ui
                        .selectable_label(state.selected_head == h, format!("{h}"))
                        .clicked()
                    {
                        state.selected_head = h;
                        state.dirty = true;
                    }
                }
            });

            // Layer visibility
            let n_layers = state.layer_visible.len();
            ui.horizontal_wrapped(|ui| {
                ui.label("Layers:");
                for l in 0..n_layers {
                    let mut vis = state.layer_visible[l];
                    if ui.checkbox(&mut vis, format!("L{l}")).changed() {
                        state.layer_visible[l] = vis;
                        state.dirty = true;
                    }
                }
            });

            ui.separator();

            // Timeline
            ui.horizontal(|ui| {
                if ui.button(if state.playing { "⏸" } else { "▶" }).clicked() {
                    state.playing = !state.playing;
                }
                ui.label(format!(
                    "Frame {}/{}",
                    state.frame_index + 1,
                    n_frames
                ));
            });

            let mut fi = state.frame_index;
            if n_frames > 0 {
                let max = n_frames - 1;
                if ui
                    .add(egui::Slider::new(&mut fi, 0..=max).text("Step"))
                    .changed()
                {
                    state.frame_index = fi;
                    state.dirty = true;
                }
            }

            // Show frame info
            if let Some(rec) = &state.recording {
                if let Some(frame) = rec.frames.get(state.frame_index) {
                    ui.label(format!("Training step: {}", frame.step));
                    if let Some(ref dec) = frame.decision {
                        ui.label(format!("Value: {:.4}", dec.value));
                    }
                }
            }

            ui.separator();

            // Visual controls
            ui.add(
                egui::Slider::new(&mut state.column_scale, 1.0..=20.0)
                    .text("Column Scale"),
            );
            ui.add(
                egui::Slider::new(&mut state.layer_spacing, 1.0..=10.0)
                    .text("Layer Spacing"),
            );
            ui.add(
                egui::Slider::new(&mut state.playback_fps, 0.5..=30.0)
                    .text("Playback FPS"),
            );

            ui.separator();

            // Cross-attention controls
            ui.checkbox(&mut state.show_cross_attention, "Show Cross-Attention");
            if state.show_cross_attention {
                ui.horizontal(|ui| {
                    ui.label("Ability slot:");
                    if ui
                        .selectable_label(state.cross_attention_slot.is_none(), "All")
                        .clicked()
                    {
                        state.cross_attention_slot = None;
                        state.dirty = true;
                    }
                    for s in 0..8 {
                        if ui
                            .selectable_label(
                                state.cross_attention_slot == Some(s),
                                format!("{s}"),
                            )
                            .clicked()
                        {
                            state.cross_attention_slot = Some(s);
                            state.dirty = true;
                        }
                    }
                });
            }
        });
}

// ---------------------------------------------------------------------------
// Playback animation
// ---------------------------------------------------------------------------

fn animate_playback(time: Res<Time>, mut state: ResMut<ReplayState>) {
    if !state.playing {
        return;
    }
    let n_frames = match &state.recording {
        Some(r) => r.frames.len(),
        None => return,
    };
    if n_frames == 0 {
        return;
    }

    state.playback_timer += time.delta_seconds();
    let interval = 1.0 / state.playback_fps;
    if state.playback_timer >= interval {
        state.playback_timer -= interval;
        let next = state.frame_index + 1;
        if next < n_frames {
            state.frame_index = next;
        } else {
            state.frame_index = 0; // loop
        }
        state.dirty = true;
    }
}

// ---------------------------------------------------------------------------
// Mesh generation — attention columns
// ---------------------------------------------------------------------------

/// Color ramp: dark blue → cyan → yellow → white for attention weights 0→1.
fn attention_color(weight: f32) -> Color {
    let t = weight.clamp(0.0, 1.0);
    if t < 0.33 {
        let s = t / 0.33;
        Color::rgb(0.05, 0.05 + s * 0.4, 0.2 + s * 0.6)
    } else if t < 0.66 {
        let s = (t - 0.33) / 0.33;
        Color::rgb(s * 0.9, 0.45 + s * 0.45, 0.8 - s * 0.5)
    } else {
        let s = (t - 0.66) / 0.34;
        Color::rgb(0.9 + s * 0.1, 0.9 + s * 0.1, 0.3 + s * 0.7)
    }
}

/// Color for cross-attention beams by ability slot.
fn ability_color(slot: usize) -> Color {
    const COLORS: [Color; 8] = [
        Color::rgb(0.34, 0.85, 0.39), // green
        Color::rgb(0.97, 0.32, 0.29), // red
        Color::rgb(0.22, 0.55, 0.99), // blue
        Color::rgb(0.82, 0.60, 0.13), // gold
        Color::rgb(0.74, 0.55, 1.00), // purple
        Color::rgb(0.99, 0.55, 0.22), // orange
        Color::rgb(0.22, 0.88, 0.82), // teal
        Color::rgb(0.88, 0.22, 0.66), // pink
    ];
    COLORS[slot % 8]
}

fn update_attention_meshes(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut state: ResMut<ReplayState>,
    columns: Query<Entity, With<AttentionColumn>>,
    planes: Query<Entity, With<LayerPlane>>,
) {
    if !state.dirty {
        return;
    }
    state.dirty = false;

    // Despawn old meshes
    for entity in columns.iter() {
        commands.entity(entity).despawn_recursive();
    }
    for entity in planes.iter() {
        commands.entity(entity).despawn_recursive();
    }

    let Some(recording) = &state.recording else {
        return;
    };
    let Some(frame) = recording.frames.get(state.frame_index) else {
        return;
    };

    let layers: &[LayerAttention] = match state.attention_source {
        AttentionSource::Transformer => &frame.transformer_attention,
        AttentionSource::EntityEncoder => &frame.entity_attention,
    };

    let n_heads = recording.n_heads;
    let head = state.selected_head.min(n_heads.saturating_sub(1));
    let col_scale = state.column_scale;
    let layer_spacing = state.layer_spacing;
    let cell_size = 0.4_f32;
    let cell_gap = 0.05_f32;
    let cell_total = cell_size + cell_gap;

    for layer_attn in layers {
        let l = layer_attn.layer;
        if l < state.layer_visible.len() && !state.layer_visible[l] {
            continue;
        }

        let q_len = layer_attn.query_len;
        let k_len = layer_attn.key_len;
        let y_base = l as f32 * layer_spacing;

        // Spawn translucent base plane
        let plane_w = q_len as f32 * cell_total;
        let plane_d = k_len as f32 * cell_total;
        commands.spawn((
            PbrBundle {
                mesh: meshes.add(Mesh::from(Cuboid::new(plane_w, 0.02, plane_d))),
                material: materials.add(StandardMaterial {
                    base_color: Color::rgba(0.15, 0.18, 0.25, 0.3),
                    alpha_mode: AlphaMode::Blend,
                    unlit: true,
                    ..default()
                }),
                transform: Transform::from_translation(Vec3::new(
                    plane_w * 0.5 - cell_total * 0.5,
                    y_base - 0.01,
                    plane_d * 0.5 - cell_total * 0.5,
                )),
                ..default()
            },
            LayerPlane { layer: l },
        ));

        // Spawn columns for each (q, k) cell
        if head >= n_heads {
            continue;
        }
        let matrix = layer_attn.head_matrix(n_heads, head);
        for q in 0..q_len {
            for k in 0..k_len {
                let w = matrix[q * k_len + k];
                if w < 0.005 {
                    continue; // skip near-zero for performance
                }

                let height = w * col_scale;
                let x = q as f32 * cell_total;
                let z = k as f32 * cell_total;
                let y = y_base + height * 0.5;

                commands.spawn((
                    PbrBundle {
                        mesh: meshes.add(Mesh::from(Cuboid::new(
                            cell_size, height, cell_size,
                        ))),
                        material: materials.add(StandardMaterial {
                            base_color: attention_color(w),
                            emissive: attention_color(w) * 0.3,
                            ..default()
                        }),
                        transform: Transform::from_translation(Vec3::new(x, y, z)),
                        ..default()
                    },
                    AttentionColumn {
                        layer: l,
                        query: q,
                        key: k,
                    },
                ));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Cross-attention beam visualization
// ---------------------------------------------------------------------------

fn update_cross_attention_beams(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    state: Res<ReplayState>,
    beams: Query<Entity, With<CrossAttentionBeam>>,
) {
    // Only rebuild when dirty flag was just cleared (same frame as mesh rebuild)
    // We check by looking at the dirty flag — it's false after rebuild
    if state.dirty {
        return;
    }

    // Despawn old beams every frame when recording exists (rebuild below)
    let Some(recording) = &state.recording else {
        return;
    };

    // Only despawn/rebuild if we have cross attention data
    let Some(frame) = recording.frames.get(state.frame_index) else {
        return;
    };

    if !state.show_cross_attention || frame.cross_attention.is_empty() {
        // Just despawn any lingering beams
        for entity in beams.iter() {
            commands.entity(entity).despawn_recursive();
        }
        return;
    }

    // Check if beams already exist for current state (avoid per-frame rebuild)
    if !beams.is_empty() {
        return;
    }

    let n_heads = recording.n_heads;
    let head = state.selected_head.min(n_heads.saturating_sub(1));

    // Cross-attention beams render above all layers
    let beam_y_start = (recording.n_layers.max(recording.n_entity_layers)) as f32
        * state.layer_spacing
        + 1.0;
    let cell_total = 0.45_f32;

    for ca in &frame.cross_attention {
        if let Some(slot_filter) = state.cross_attention_slot {
            if ca.ability_slot != slot_filter {
                continue;
            }
        }

        let k_len = ca.key_len;
        if k_len == 0 || head >= n_heads {
            continue;
        }

        let color = ability_color(ca.ability_slot);
        let beam_x = ca.ability_slot as f32 * 1.5; // space ability slots apart

        // Get weights for selected head
        let head_start = head * k_len;
        let head_end = head_start + k_len;
        if head_end > ca.weights_flat.len() {
            continue;
        }
        let weights = &ca.weights_flat[head_start..head_end];

        for (k, &w) in weights.iter().enumerate() {
            if w < 0.01 {
                continue;
            }

            let target_z = k as f32 * cell_total;
            let beam_height = 2.0;
            let beam_thickness = w * 0.3; // thickness proportional to weight

            commands.spawn((
                PbrBundle {
                    mesh: meshes.add(Mesh::from(Cuboid::new(
                        beam_thickness,
                        beam_height,
                        beam_thickness,
                    ))),
                    material: materials.add(StandardMaterial {
                        base_color: color.with_a(0.5 + w * 0.5),
                        emissive: color * w,
                        alpha_mode: AlphaMode::Blend,
                        ..default()
                    }),
                    transform: Transform::from_translation(Vec3::new(
                        beam_x,
                        beam_y_start + beam_height * 0.5,
                        target_z,
                    )),
                    ..default()
                },
                CrossAttentionBeam {
                    ability_slot: ca.ability_slot,
                },
            ));
        }
    }
}
