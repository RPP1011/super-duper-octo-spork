use super::lcg::Lcg;
use crate::terrain;

// ---------------------------------------------------------------------------
// Room environment context — bridges overworld biome into mission rooms
// ---------------------------------------------------------------------------

/// Visual/atmospheric theme for a mission room, derived from overworld context.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RoomEnvironment {
    /// Stone ruins, grey tones, rubble scatter
    Ruins,
    /// Overgrown structure, mossy green, vine scatter
    Overgrown,
    /// Underground cavern, damp blues, stalactite scatter
    Cavern,
    /// Scorched battlefield, charred reds, ember scatter
    Scorched,
    /// Frozen outpost, icy whites, frost scatter
    Frozen,
    /// Sandy fortress, warm yellows, sand drift scatter
    Desert,
}

impl RoomEnvironment {
    /// Pick an environment from seed (can be extended to use overworld biome).
    pub fn from_seed(seed: u64) -> Self {
        let mut rng = Lcg::new(seed ^ 0xE4B1_0000_0000_0000);
        match rng.next_usize_range(0, 5) {
            0 => RoomEnvironment::Ruins,
            1 => RoomEnvironment::Overgrown,
            2 => RoomEnvironment::Cavern,
            3 => RoomEnvironment::Scorched,
            4 => RoomEnvironment::Frozen,
            _ => RoomEnvironment::Desert,
        }
    }

    /// Pick environment from an overworld biome.
    pub fn from_biome(biome: terrain::Biome, seed: u64) -> Self {
        let mut rng = Lcg::new(seed ^ 0xB10E_0000_0000_0000);
        match biome {
            terrain::Biome::DeepWater | terrain::Biome::ShallowWater => RoomEnvironment::Cavern,
            terrain::Biome::Beach => RoomEnvironment::Desert,
            terrain::Biome::Grassland => {
                if rng.next_u64() % 2 == 0 {
                    RoomEnvironment::Ruins
                } else {
                    RoomEnvironment::Overgrown
                }
            }
            terrain::Biome::Forest => RoomEnvironment::Overgrown,
            terrain::Biome::Highland => {
                if rng.next_u64() % 2 == 0 {
                    RoomEnvironment::Ruins
                } else {
                    RoomEnvironment::Scorched
                }
            }
            terrain::Biome::Mountain => RoomEnvironment::Ruins,
            terrain::Biome::Snow => RoomEnvironment::Frozen,
        }
    }

    /// Floor base colour.
    pub fn floor_color(self) -> (f32, f32, f32) {
        match self {
            RoomEnvironment::Ruins => (0.42, 0.40, 0.38),
            RoomEnvironment::Overgrown => (0.28, 0.40, 0.22),
            RoomEnvironment::Cavern => (0.30, 0.32, 0.38),
            RoomEnvironment::Scorched => (0.38, 0.28, 0.22),
            RoomEnvironment::Frozen => (0.72, 0.76, 0.80),
            RoomEnvironment::Desert => (0.68, 0.58, 0.42),
        }
    }

    /// Wall colour (darker variant of floor).
    pub fn wall_color(self) -> (f32, f32, f32) {
        let (r, g, b) = self.floor_color();
        ((r - 0.12).max(0.0), (g - 0.12).max(0.0), (b - 0.12).max(0.0))
    }

    /// Obstacle colour (darker variant with slight hue shift).
    pub fn obstacle_color(self) -> (f32, f32, f32) {
        let (r, g, b) = self.floor_color();
        match self {
            RoomEnvironment::Overgrown => ((r + 0.05).min(1.0), (g - 0.10).max(0.0), (b + 0.02).min(1.0)),
            RoomEnvironment::Scorched => ((r - 0.08).max(0.0), (g - 0.12).max(0.0), (b - 0.10).max(0.0)),
            _ => ((r - 0.15).max(0.0), (g - 0.15).max(0.0), (b - 0.15).max(0.0)),
        }
    }

    /// Detail scatter colour (rubble, vines, frost, embers).
    pub fn scatter_color(self) -> (f32, f32, f32) {
        match self {
            RoomEnvironment::Ruins => (0.52, 0.48, 0.44),
            RoomEnvironment::Overgrown => (0.18, 0.48, 0.14),
            RoomEnvironment::Cavern => (0.22, 0.28, 0.36),
            RoomEnvironment::Scorched => (0.55, 0.22, 0.10),
            RoomEnvironment::Frozen => (0.80, 0.85, 0.92),
            RoomEnvironment::Desert => (0.76, 0.68, 0.50),
        }
    }

    /// Floor height variation amplitude (how wavy the floor is).
    pub fn floor_noise_amplitude(self) -> f32 {
        match self {
            RoomEnvironment::Ruins => 0.15,
            RoomEnvironment::Overgrown => 0.20,
            RoomEnvironment::Cavern => 0.30,
            RoomEnvironment::Scorched => 0.12,
            RoomEnvironment::Frozen => 0.08,
            RoomEnvironment::Desert => 0.10,
        }
    }

    /// Material roughness.
    pub fn roughness(self) -> f32 {
        match self {
            RoomEnvironment::Ruins => 0.92,
            RoomEnvironment::Overgrown => 0.88,
            RoomEnvironment::Cavern => 0.80,
            RoomEnvironment::Scorched => 0.95,
            RoomEnvironment::Frozen => 0.35,
            RoomEnvironment::Desert => 0.90,
        }
    }

    /// How many scatter details to place per 100 sq metres of floor.
    pub fn scatter_density(self) -> f32 {
        match self {
            RoomEnvironment::Ruins => 3.5,
            RoomEnvironment::Overgrown => 5.0,
            RoomEnvironment::Cavern => 2.5,
            RoomEnvironment::Scorched => 4.0,
            RoomEnvironment::Frozen => 2.0,
            RoomEnvironment::Desert => 3.0,
        }
    }
}
