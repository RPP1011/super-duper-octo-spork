//! `goap_probe` pin — the load-bearing, end-to-end proof for the `goap`
//! declaration (real backward-chained precondition search, resolved at
//! compile time, executing per-agent on GPU every tick).
//!
//! `dsl_ast::goap`'s own unit tests prove the compile-time search
//! algorithm is correct in isolation; `goap_parses_and_desugars.rs` proves
//! the source syntax desugars into an ordinary physics rule that resolves
//! with zero goap-specific compiler support. This file is the one that
//! matters most: it seeds FOUR agents at four different points along the
//! SAME compiled dependency chain and asserts each one independently picks
//! the objectively correct next action from its OWN current field values —
//! genuine per-agent, GPU-native, plan-directed divergence, not four
//! separate hand-tuned cases.

use sims::goap_probe::GeneratedRuntime;

const N: u32 = 4;

const CHOP_WOOD: u32 = 1;
const BUILD_HEARTH: u32 = 2;
const FORAGE: u32 = 3;
const COOK: u32 = 4;

#[test]
fn four_agents_at_four_stages_of_the_chain_each_pick_the_right_next_action() {
    let Some(mut state) = GeneratedRuntime::try_new(0x60A9_0001, N) else {
        eprintln!("[goap_probe] skipping: no wgpu adapter on host.");
        return;
    };

    // slot 0: nothing               -> expect ChopWood    (deepest unmet prereq)
    // slot 1: has_timber only       -> expect BuildHearth  (next link in the SAME chain)
    // slot 2: has_hearth only       -> expect Forage        (the OTHER, independent prereq)
    // slot 3: has_hearth + raw_food -> expect Cook           (goal is one step away)
    let inv_timber: [f32; N as usize] = [0.0, 5.0, 0.0, 0.0];
    let inv_hearth: [u32; N as usize] = [0, 0, 1, 1];
    let inv_raw_food: [f32; N as usize] = [0.0, 0.0, 0.0, 3.0];
    let inv_meal: [f32; N as usize] = [0.0, 0.0, 0.0, 0.0];

    state.gpu.queue.write_buffer(&state.agent_inv_timber_buf, 0, bytemuck::cast_slice(&inv_timber));
    state.gpu.queue.write_buffer(&state.agent_inv_hearth_buf, 0, bytemuck::cast_slice(&inv_hearth));
    state.gpu.queue.write_buffer(&state.agent_inv_raw_food_buf, 0, bytemuck::cast_slice(&inv_raw_food));
    state.gpu.queue.write_buffer(&state.agent_inv_meal_buf, 0, bytemuck::cast_slice(&inv_meal));

    state.step();

    let buf = state.agent_chosen_action_buf.clone();
    let chosen = read_u32(&mut state, &buf);

    assert_eq!(chosen[0], CHOP_WOOD, "slot 0 (nothing) should plan to chop wood first");
    assert_eq!(chosen[1], BUILD_HEARTH, "slot 1 (has timber) should move on to building the hearth");
    assert_eq!(chosen[2], FORAGE, "slot 2 (has hearth only) should work the OTHER prerequisite: forage");
    assert_eq!(chosen[3], COOK, "slot 3 (has hearth + raw food) should finally cook");

    // The load-bearing per-agent claim, stated as an assertion rather than
    // just four independent equalities: the SAME tick, the SAME compiled
    // kernel, four DIFFERENT outcomes purely from each agent's own state.
    let unique: std::collections::BTreeSet<u32> = chosen.iter().copied().collect();
    assert_eq!(unique.len(), 4, "expected four distinct chosen actions, got {chosen:?}");
}

/// A goal already fully met plans nothing (`0`) rather than looping back
/// through the chain redundantly.
#[test]
fn an_agent_who_already_has_a_meal_plans_nothing() {
    let Some(mut state) = GeneratedRuntime::try_new(0x60A9_0002, 1) else {
        eprintln!("[goap_probe] skipping: no wgpu adapter on host.");
        return;
    };
    state.gpu.queue.write_buffer(&state.agent_inv_timber_buf, 0, bytemuck::cast_slice(&[5.0f32]));
    state.gpu.queue.write_buffer(&state.agent_inv_hearth_buf, 0, bytemuck::cast_slice(&[1u32]));
    state.gpu.queue.write_buffer(&state.agent_inv_raw_food_buf, 0, bytemuck::cast_slice(&[3.0f32]));
    state.gpu.queue.write_buffer(&state.agent_inv_meal_buf, 0, bytemuck::cast_slice(&[2.0f32]));
    state.step();
    let buf = state.agent_chosen_action_buf.clone();
    let chosen = read_u32(&mut state, &buf);
    assert_eq!(chosen[0], 0, "goal already satisfied — nothing to plan");
}

fn read_u32(state: &mut GeneratedRuntime, buf: &wgpu::Buffer) -> Vec<u32> {
    let bytes = (state.agent_count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("goap_probe::readback_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("goap_probe::readback") });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |res| res.expect("readback_staging map_async failed"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("device poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&view);
        words[..state.agent_count as usize].to_vec()
    };
    staging.unmap();
    out
}
