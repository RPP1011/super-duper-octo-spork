//! `verb_belief_score_probe` pin — falsifies (or confirms) the other half
//! of webband_colony.sim's "epistemic split": whether a `verb`'s `score`
//! clause can read a pair-keyed `belief` at all, and whether doing so
//! actually changes which candidate wins the argmax (not just that the
//! kernel compiles). `dsl_ast::eval::mod.rs`'s own `ReadContext` doc
//! comment already claimed scoring is a belief/view consumer
//! ("pure reads; used by masks, scoring, and lazy views") — this is the
//! GPU proof.
//!
//! 3 colonists. Agent 0 is the actor under test; agents 1 and 2 are its
//! only candidates (`target != self`). `grudge(0, 1)` is seeded high,
//! `grudge(0, 2)` is seeded low. `Approach`'s score is `-grudge(self,
//! target)`, so agent 0's argmax should pick agent 2 — a genuinely
//! belief-driven target selection, not a fixed/positional one.

use sims::verb_belief_score_probe::GeneratedRuntime;

#[test]
fn a_verbs_argmax_target_selection_is_driven_by_a_pair_keyed_belief() {
    let Some(mut state) = GeneratedRuntime::try_new(0xB311EF_0001, 3) else {
        eprintln!("[verb_belief_score_probe] skipping: no wgpu adapter on host.");
        return;
    };
    let n = state.agent_count as usize;
    assert_eq!(n, 3);

    // Storage buffers start zero-initialized; `Approach`'s mask gates on
    // `self.alive && target.alive`, so every slot needs an explicit
    // alive stamp (no init/spawn block in this fixture seeds it).
    state.gpu.queue.write_buffer(&state.agent_alive_buf, 0, bytemuck::cast_slice(&vec![1u32; n]));

    // Distinct tags so the winning target is identifiable via
    // `agents.tag(t)` without needing an AgentId-typed field.
    let tag: [u32; 3] = [1, 100, 5];
    state.gpu.queue.write_buffer(&state.agent_tag_buf, 0, bytemuck::cast_slice(&tag));

    // Pair storage is row-major `[observer * n + subject]`. Agent 0
    // holds a strong grudge against agent 1, almost none against agent 2.
    let mut grudge = vec![0.0f32; n * n];
    grudge[0 * n + 1] = 80.0;
    grudge[0 * n + 2] = 2.0;
    state.gpu.queue.write_buffer(&state.view_storage_grudge_primary_buf, 0, bytemuck::cast_slice(&grudge));

    // The verb cascade (mask -> scoring -> ActionSelected -> chronicle
    // consumer emits Approached -> RecordApproach post-consumer) spans
    // more than one tick's event-ring latency; a few ticks give it room
    // to land without pinning the exact stage count.
    for _ in 0..5 {
        state.step();
    }

    let buf = state.agent_last_target_tag_buf.clone();
    let last_target_tag = read_u32(&mut state, &buf, n);

    assert_eq!(
        last_target_tag[0], 5,
        "agent 0 should have approached the LOW-grudge candidate (tag 5), not the high-grudge one (tag 100): {last_target_tag:?}"
    );
}

fn read_u32(state: &mut GeneratedRuntime, buf: &wgpu::Buffer, count: usize) -> Vec<u32> {
    let bytes = (count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("verb_belief_score_probe::readback_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("verb_belief_score_probe::readback") });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |res| res.expect("readback_staging map_async failed"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("device poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&view);
        words[..count].to_vec()
    };
    staging.unmap();
    out
}
