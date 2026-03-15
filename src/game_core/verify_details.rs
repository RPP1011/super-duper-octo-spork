//! Detailed campaign verification checks: roster, parties, overworld, diplomacy,
//! missions, flashpoints, companion quests, attention.
//!
//! Split from `verify.rs` to keep files under 500 lines.

use std::collections::HashSet;

use super::overworld_types::*;
use super::roster_types::*;
use super::types::*;
use super::verify::CampaignViolation;

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
