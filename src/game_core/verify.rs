use std::collections::HashSet;

use super::overworld_types::*;
use super::roster_types::*;
use super::types::*;

// ---------------------------------------------------------------------------
// Campaign-layer violation types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum CampaignViolation {
    // --- Roster ---
    /// Two heroes share the same ID.
    DuplicateHeroId { hero_id: u32 },
    /// `next_id` is not greater than every existing hero ID.
    RosterNextIdTooLow { next_id: u32, max_existing: u32 },
    /// Hero stat is outside the valid [0, 100] range.
    HeroStatOutOfRange { hero_id: u32, stat: &'static str, value_x100: i32 },
    /// Recruit candidate ID collides with an active hero.
    RecruitIdCollidesWithHero { recruit_id: u32 },
    /// Player hero ID doesn't reference any hero in the roster.
    InvalidPlayerHeroId { id: u32 },
    /// Hero marked as both active and deserter.
    ActiveDeserter { hero_id: u32 },

    // --- Parties ---
    /// Party leader references a hero that doesn't exist or is inactive/deserted.
    InvalidPartyLeader { party_id: u32, leader_hero_id: u32 },
    /// Party `order_target_region_id` references a region that doesn't exist.
    PartyOrderTargetInvalid { party_id: u32, region_id: usize },
    /// Party's current region is out of bounds.
    PartyRegionOutOfBounds { party_id: u32, region_id: usize, num_regions: usize },
    /// Duplicate party IDs.
    DuplicatePartyId { party_id: u32 },
    /// No player-controlled party exists.
    NoPlayerParty,
    /// Multiple player-controlled parties.
    MultiplePlayerParties { count: usize },
    /// Party supply is negative.
    NegativePartySupply { party_id: u32, supply_x100: i32 },

    // --- Overworld ---
    /// Region neighbor references a region ID that doesn't exist.
    RegionNeighborOutOfBounds { region_id: usize, neighbor_id: usize, num_regions: usize },
    /// Region owner faction doesn't exist.
    RegionOwnerInvalid { region_id: usize, faction_id: usize, num_factions: usize },
    /// Region unrest is outside [0, 100].
    RegionUnrestOutOfRange { region_id: usize, value_x100: i32 },
    /// Region control is outside [0, 100].
    RegionControlOutOfRange { region_id: usize, value_x100: i32 },
    /// `current_region` index is out of bounds.
    CurrentRegionOutOfBounds { index: usize, num_regions: usize },

    // --- Diplomacy ---
    /// Diplomacy relation matrix is not square or doesn't match faction count.
    DiplomacyMatrixMismatch { rows: usize, expected: usize },
    /// Diplomacy diagonal entry is non-zero (faction relation to itself).
    DiplomacyDiagonalNonZero { faction_id: usize, value: i32 },
    /// Player faction ID is out of bounds.
    PlayerFactionOutOfBounds { player_faction_id: usize, num_factions: usize },

    // --- Missions ---
    /// Completed mission is still marked active.
    CompletedMissionStillActive { mission_name: String },
    /// Mission reactor integrity is negative.
    NegativeReactorIntegrity { mission_name: String, value_x100: i32 },
    /// Mission room index exceeds the room count.
    MissionRoomIndexOutOfBounds { mission_name: String, room_index: usize, num_rooms: usize },
    /// Mission bound to a region that doesn't exist.
    MissionBoundRegionInvalid { mission_name: String, region_id: usize },

    // --- Flashpoints ---
    /// Flashpoint stage is outside [1, 3].
    FlashpointStageInvalid { chain_id: u32, stage: u8 },
    /// Flashpoint references a region that doesn't exist.
    FlashpointRegionInvalid { chain_id: u32, region_id: usize },
    /// Flashpoint attacker or defender faction doesn't exist.
    FlashpointFactionInvalid { chain_id: u32, faction_id: usize },

    // --- Companion quests ---
    /// Quest references a hero that isn't in the roster.
    QuestHeroMissing { quest_id: u32, hero_id: u32 },
    /// Quest progress exceeds target.
    QuestProgressExceedsTarget { quest_id: u32, progress: u32, target: u32 },

    // --- Attention ---
    /// Attention energy exceeds max.
    AttentionEnergyExceedsMax { energy_x100: i32, max_x100: i32 },
    /// Attention energy is negative.
    NegativeAttentionEnergy { energy_x100: i32 },
}

#[derive(Debug, Clone)]
pub struct CampaignVerificationReport {
    pub violations: Vec<CampaignViolation>,
}

impl CampaignVerificationReport {
    pub fn is_ok(&self) -> bool {
        self.violations.is_empty()
    }

    pub fn violation_count(&self) -> usize {
        self.violations.len()
    }
}

/// Verify the integrity of a campaign roster.
pub fn verify_roster(roster: &CampaignRoster) -> Vec<CampaignViolation> {
    let mut violations = Vec::new();
    let mut seen_ids = HashSet::new();

    // Duplicate hero IDs
    for hero in &roster.heroes {
        if !seen_ids.insert(hero.id) {
            violations.push(CampaignViolation::DuplicateHeroId { hero_id: hero.id });
        }
    }

    // next_id must be higher than all existing hero IDs
    if let Some(&max_id) = seen_ids.iter().max() {
        if roster.next_id <= max_id {
            violations.push(CampaignViolation::RosterNextIdTooLow {
                next_id: roster.next_id,
                max_existing: max_id,
            });
        }
    }

    // Player hero ID validity
    if let Some(player_id) = roster.player_hero_id {
        if !roster.heroes.iter().any(|h| h.id == player_id) {
            violations.push(CampaignViolation::InvalidPlayerHeroId { id: player_id });
        }
    }

    // Per-hero checks
    for hero in &roster.heroes {
        // Stat bounds [0.0, 100.0]
        let stats: [(&'static str, f32); 5] = [
            ("loyalty", hero.loyalty),
            ("stress", hero.stress),
            ("fatigue", hero.fatigue),
            ("injury", hero.injury),
            ("resolve", hero.resolve),
        ];
        for (name, val) in stats {
            if val < 0.0 || val > 100.0 {
                violations.push(CampaignViolation::HeroStatOutOfRange {
                    hero_id: hero.id,
                    stat: name,
                    value_x100: (val * 100.0) as i32,
                });
            }
        }

        // Active + deserter is contradictory
        if hero.active && hero.deserter {
            violations.push(CampaignViolation::ActiveDeserter { hero_id: hero.id });
        }
    }

    // Recruit IDs should not collide with hero IDs
    for recruit in &roster.recruit_pool {
        if seen_ids.contains(&recruit.id) {
            violations.push(CampaignViolation::RecruitIdCollidesWithHero {
                recruit_id: recruit.id,
            });
        }
    }

    violations
}

/// Verify the integrity of campaign parties against roster and overworld.
pub fn verify_parties(
    parties: &CampaignParties,
    roster: &CampaignRoster,
    overworld: &OverworldMap,
) -> Vec<CampaignViolation> {
    let mut violations = Vec::new();
    let hero_ids: HashSet<u32> = roster.heroes.iter()
        .filter(|h| h.active && !h.deserter)
        .map(|h| h.id)
        .collect();
    let num_regions = overworld.regions.len();
    let mut seen_party_ids = HashSet::new();
    let mut player_party_count = 0usize;

    for party in &parties.parties {
        // Duplicate party IDs
        if !seen_party_ids.insert(party.id) {
            violations.push(CampaignViolation::DuplicatePartyId { party_id: party.id });
        }

        // Leader must be an active, non-deserter hero
        if !hero_ids.contains(&party.leader_hero_id) {
            violations.push(CampaignViolation::InvalidPartyLeader {
                party_id: party.id,
                leader_hero_id: party.leader_hero_id,
            });
        }

        // Region bounds
        if party.region_id >= num_regions {
            violations.push(CampaignViolation::PartyRegionOutOfBounds {
                party_id: party.id,
                region_id: party.region_id,
                num_regions,
            });
        }

        // Order target region bounds
        if let Some(target_region) = party.order_target_region_id {
            if target_region >= num_regions {
                violations.push(CampaignViolation::PartyOrderTargetInvalid {
                    party_id: party.id,
                    region_id: target_region,
                });
            }
        }

        // Negative supply
        if party.supply < 0.0 {
            violations.push(CampaignViolation::NegativePartySupply {
                party_id: party.id,
                supply_x100: (party.supply * 100.0) as i32,
            });
        }

        if party.is_player_controlled {
            player_party_count += 1;
        }
    }

    if !parties.parties.is_empty() && player_party_count == 0 {
        violations.push(CampaignViolation::NoPlayerParty);
    }
    if player_party_count > 1 {
        violations.push(CampaignViolation::MultiplePlayerParties {
            count: player_party_count,
        });
    }

    violations
}

/// Verify overworld map structural integrity.
pub fn verify_overworld(overworld: &OverworldMap) -> Vec<CampaignViolation> {
    let mut violations = Vec::new();
    let num_regions = overworld.regions.len();
    let num_factions = overworld.factions.len();

    if overworld.current_region >= num_regions && num_regions > 0 {
        violations.push(CampaignViolation::CurrentRegionOutOfBounds {
            index: overworld.current_region,
            num_regions,
        });
    }

    for region in &overworld.regions {
        // Neighbor validity
        for &neighbor in &region.neighbors {
            if neighbor >= num_regions {
                violations.push(CampaignViolation::RegionNeighborOutOfBounds {
                    region_id: region.id,
                    neighbor_id: neighbor,
                    num_regions,
                });
            }
        }

        // Owner faction validity
        if region.owner_faction_id >= num_factions {
            violations.push(CampaignViolation::RegionOwnerInvalid {
                region_id: region.id,
                faction_id: region.owner_faction_id,
                num_factions,
            });
        }

        // Unrest/control bounds [0, 100]
        if region.unrest < 0.0 || region.unrest > 100.0 {
            violations.push(CampaignViolation::RegionUnrestOutOfRange {
                region_id: region.id,
                value_x100: (region.unrest * 100.0) as i32,
            });
        }
        if region.control < 0.0 || region.control > 100.0 {
            violations.push(CampaignViolation::RegionControlOutOfRange {
                region_id: region.id,
                value_x100: (region.control * 100.0) as i32,
            });
        }
    }

    violations
}

/// Verify diplomacy state consistency.
pub fn verify_diplomacy(
    diplomacy: &DiplomacyState,
    num_factions: usize,
) -> Vec<CampaignViolation> {
    let mut violations = Vec::new();

    if diplomacy.player_faction_id >= num_factions && num_factions > 0 {
        violations.push(CampaignViolation::PlayerFactionOutOfBounds {
            player_faction_id: diplomacy.player_faction_id,
            num_factions,
        });
    }

    if diplomacy.relations.len() != num_factions {
        violations.push(CampaignViolation::DiplomacyMatrixMismatch {
            rows: diplomacy.relations.len(),
            expected: num_factions,
        });
    } else {
        for (i, row) in diplomacy.relations.iter().enumerate() {
            if row.len() != num_factions {
                violations.push(CampaignViolation::DiplomacyMatrixMismatch {
                    rows: row.len(),
                    expected: num_factions,
                });
            } else if row[i] != 0 {
                violations.push(CampaignViolation::DiplomacyDiagonalNonZero {
                    faction_id: i,
                    value: row[i],
                });
            }
        }
    }

    violations
}

/// Verify mission snapshot consistency.
pub fn verify_missions(
    missions: &[MissionSnapshot],
    overworld: &OverworldMap,
    mission_map: &MissionMap,
) -> Vec<CampaignViolation> {
    let mut violations = Vec::new();
    let num_rooms = mission_map.rooms.len();

    for mission in missions {
        // Completed but still active
        if mission.result != MissionResult::InProgress && mission.mission_active {
            violations.push(CampaignViolation::CompletedMissionStillActive {
                mission_name: mission.mission_name.clone(),
            });
        }

        // Negative reactor integrity
        if mission.reactor_integrity < 0.0 {
            violations.push(CampaignViolation::NegativeReactorIntegrity {
                mission_name: mission.mission_name.clone(),
                value_x100: (mission.reactor_integrity * 100.0) as i32,
            });
        }

        // Room index out of bounds
        if num_rooms > 0 && mission.room_index >= num_rooms {
            violations.push(CampaignViolation::MissionRoomIndexOutOfBounds {
                mission_name: mission.mission_name.clone(),
                room_index: mission.room_index,
                num_rooms,
            });
        }

        // Bound region validity
        if let Some(region_id) = mission.bound_region_id {
            if !overworld.regions.iter().any(|r| r.id == region_id) {
                violations.push(CampaignViolation::MissionBoundRegionInvalid {
                    mission_name: mission.mission_name.clone(),
                    region_id,
                });
            }
        }
    }

    violations
}

/// Verify flashpoint chain consistency.
pub fn verify_flashpoints(
    flashpoints: &FlashpointState,
    overworld: &OverworldMap,
) -> Vec<CampaignViolation> {
    let mut violations = Vec::new();
    let num_regions = overworld.regions.len();
    let num_factions = overworld.factions.len();

    for chain in &flashpoints.chains {
        if chain.completed {
            continue;
        }

        if chain.stage < 1 || chain.stage > 3 {
            violations.push(CampaignViolation::FlashpointStageInvalid {
                chain_id: chain.id,
                stage: chain.stage,
            });
        }

        if chain.region_id >= num_regions {
            violations.push(CampaignViolation::FlashpointRegionInvalid {
                chain_id: chain.id,
                region_id: chain.region_id,
            });
        }

        if chain.attacker_faction_id >= num_factions {
            violations.push(CampaignViolation::FlashpointFactionInvalid {
                chain_id: chain.id,
                faction_id: chain.attacker_faction_id,
            });
        }
        if chain.defender_faction_id >= num_factions {
            violations.push(CampaignViolation::FlashpointFactionInvalid {
                chain_id: chain.id,
                faction_id: chain.defender_faction_id,
            });
        }
    }

    violations
}

/// Verify companion quest consistency.
pub fn verify_companion_quests(
    story: &super::companion::CompanionStoryState,
    roster: &CampaignRoster,
) -> Vec<CampaignViolation> {
    let mut violations = Vec::new();
    let hero_ids: HashSet<u32> = roster.heroes.iter().map(|h| h.id).collect();

    for quest in &story.quests {
        if !hero_ids.contains(&quest.hero_id) {
            violations.push(CampaignViolation::QuestHeroMissing {
                quest_id: quest.id,
                hero_id: quest.hero_id,
            });
        }
        if quest.progress > quest.target {
            violations.push(CampaignViolation::QuestProgressExceedsTarget {
                quest_id: quest.id,
                progress: quest.progress,
                target: quest.target,
            });
        }
    }

    violations
}

/// Verify attention state bounds.
pub fn verify_attention(attention: &AttentionState) -> Vec<CampaignViolation> {
    let mut violations = Vec::new();

    if attention.global_energy > attention.max_energy {
        violations.push(CampaignViolation::AttentionEnergyExceedsMax {
            energy_x100: (attention.global_energy * 100.0) as i32,
            max_x100: (attention.max_energy * 100.0) as i32,
        });
    }
    if attention.global_energy < 0.0 {
        violations.push(CampaignViolation::NegativeAttentionEnergy {
            energy_x100: (attention.global_energy * 100.0) as i32,
        });
    }

    violations
}

/// Run all campaign-layer verification checks against a save data snapshot.
pub fn verify_campaign(save: &super::save::CampaignSaveData) -> CampaignVerificationReport {
    let mut violations = Vec::new();

    violations.extend(verify_roster(&save.campaign_roster));
    violations.extend(verify_parties(
        &save.campaign_parties,
        &save.campaign_roster,
        &save.overworld_map,
    ));
    violations.extend(verify_overworld(&save.overworld_map));
    violations.extend(verify_diplomacy(
        &save.diplomacy_state,
        save.overworld_map.factions.len(),
    ));
    violations.extend(verify_missions(
        &save.mission_snapshots,
        &save.overworld_map,
        &save.mission_map,
    ));
    violations.extend(verify_flashpoints(
        &save.flashpoint_state,
        &save.overworld_map,
    ));
    violations.extend(verify_companion_quests(
        &save.companion_story_state,
        &save.campaign_roster,
    ));
    violations.extend(verify_attention(&save.attention_state));

    CampaignVerificationReport { violations }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn minimal_roster() -> CampaignRoster {
        CampaignRoster {
            heroes: vec![
                HeroCompanion {
                    id: 1,
                    name: "Test Hero".into(),
                    origin_faction_id: 0,
                    origin_region_id: 0,
                    backstory: String::new(),
                    archetype: PersonalityArchetype::Vanguard,
                    loyalty: 50.0,
                    stress: 10.0,
                    fatigue: 5.0,
                    injury: 0.0,
                    resolve: 60.0,
                    active: true,
                    deserter: false,
                    xp: 0,
                    level: 1,
                    equipment: HeroEquipment::default(),
                    traits: Vec::new(),
                },
            ],
            recruit_pool: Vec::new(),
            player_hero_id: Some(1),
            next_id: 2,
            generation_counter: 0,
        }
    }

    #[test]
    fn clean_roster_passes() {
        let roster = minimal_roster();
        let violations = verify_roster(&roster);
        assert!(violations.is_empty(), "{:?}", violations);
    }

    #[test]
    fn detects_duplicate_hero_ids() {
        let mut roster = minimal_roster();
        let mut clone = roster.heroes[0].clone();
        clone.name = "Duplicate".into();
        roster.heroes.push(clone);
        let violations = verify_roster(&roster);
        assert!(violations.iter().any(|v| matches!(v,
            CampaignViolation::DuplicateHeroId { hero_id: 1 }
        )));
    }

    #[test]
    fn detects_next_id_too_low() {
        let mut roster = minimal_roster();
        roster.next_id = 1; // should be > 1
        let violations = verify_roster(&roster);
        assert!(violations.iter().any(|v| matches!(v,
            CampaignViolation::RosterNextIdTooLow { .. }
        )));
    }

    #[test]
    fn detects_hero_stat_out_of_range() {
        let mut roster = minimal_roster();
        roster.heroes[0].loyalty = 150.0;
        let violations = verify_roster(&roster);
        assert!(violations.iter().any(|v| matches!(v,
            CampaignViolation::HeroStatOutOfRange { hero_id: 1, stat: "loyalty", .. }
        )));
    }

    #[test]
    fn detects_active_deserter() {
        let mut roster = minimal_roster();
        roster.heroes[0].active = true;
        roster.heroes[0].deserter = true;
        let violations = verify_roster(&roster);
        assert!(violations.iter().any(|v| matches!(v,
            CampaignViolation::ActiveDeserter { hero_id: 1 }
        )));
    }

    #[test]
    fn detects_invalid_player_hero_id() {
        let mut roster = minimal_roster();
        roster.player_hero_id = Some(999);
        let violations = verify_roster(&roster);
        assert!(violations.iter().any(|v| matches!(v,
            CampaignViolation::InvalidPlayerHeroId { id: 999 }
        )));
    }

    #[test]
    fn detects_recruit_id_collision() {
        let mut roster = minimal_roster();
        roster.recruit_pool.push(RecruitCandidate {
            id: 1, // same as hero
            codename: "Spy".into(),
            origin_faction_id: 0,
            origin_region_id: 0,
            backstory: String::new(),
            archetype: PersonalityArchetype::Tactician,
            resolve: 50.0,
            loyalty_bias: 0.5,
            risk_tolerance: 0.5,
        });
        let violations = verify_roster(&roster);
        assert!(violations.iter().any(|v| matches!(v,
            CampaignViolation::RecruitIdCollidesWithHero { recruit_id: 1 }
        )));
    }

    #[test]
    fn detects_diplomacy_diagonal_nonzero() {
        let diplomacy = DiplomacyState {
            player_faction_id: 0,
            relations: vec![vec![5, 10], vec![10, 0]],
        };
        let violations = verify_diplomacy(&diplomacy, 2);
        assert!(violations.iter().any(|v| matches!(v,
            CampaignViolation::DiplomacyDiagonalNonZero { faction_id: 0, value: 5 }
        )));
    }

    #[test]
    fn detects_attention_energy_exceeds_max() {
        let attention = AttentionState {
            global_energy: 150.0,
            max_energy: 100.0,
            ..AttentionState::default()
        };
        let violations = verify_attention(&attention);
        assert!(violations.iter().any(|v| matches!(v,
            CampaignViolation::AttentionEnergyExceedsMax { .. }
        )));
    }
}
