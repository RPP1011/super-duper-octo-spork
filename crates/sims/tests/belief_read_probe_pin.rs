//! `belief_read_probe` pin — falsifies (or confirms) `webband_colony.sim`'s
//! header claim that "physics cannot read views" (and, since beliefs share
//! the same `ViewIR` storage slot per `dsl_ast::resolve`'s `Decl::Belief`
//! handling, that beliefs are equally unreadable from physics). The
//! `belief_read_probe.sim` fixture compiled to a real kernel
//! (`physics_ReadGrudge`) binding the pair-keyed `view_storage_grudge_primary`
//! buffer directly — this test proves that binding actually reads the right
//! cell at runtime, not just that the kernel exists.
//!
//! Seeds the diagonal cell `[i*n+i]` of the belief's pair storage directly
//! (the same direct-buffer-write pattern `webband_colony.rs`'s own pins
//! use), runs one tick, and asserts the physics rule's `grudge(self, self)`
//! read landed in `seen_grudge` unchanged (no fold ran this tick — no
//! `Brawl` event was emitted — so the value is exactly the seed, not a
//! folded delta).

use sims::belief_read_probe::GeneratedRuntime;

#[test]
fn a_per_agent_physics_rule_reads_a_pair_keyed_belief_directly() {
    let Some(mut state) = GeneratedRuntime::try_new(0x8E11E5_0001, 4) else {
        eprintln!("[belief_read_probe] skipping: no wgpu adapter on host.");
        return;
    };
    let n = state.agent_count as usize;

    // Seed the diagonal cells (self, self) at three different values, one
    // agent left at zero, so a genuine per-agent divergent read is the
    // load-bearing claim (not "the read always comes back 0").
    let mut seed = vec![0.0f32; n * n];
    seed[0 * n + 0] = 7.0;
    seed[1 * n + 1] = 42.0;
    seed[2 * n + 2] = 0.0;
    seed[3 * n + 3] = 99.5;
    state.gpu.queue.write_buffer(&state.view_storage_grudge_primary_buf, 0, bytemuck::cast_slice(&seed));

    state.step();

    let buf = state.agent_seen_grudge_buf.clone();
    let seen = read_f32(&mut state, &buf, n);

    assert_eq!(seen, vec![7.0, 42.0, 0.0, 99.5], "per-agent belief read did not round-trip: {seen:?}");
}

fn read_f32(state: &mut GeneratedRuntime, buf: &wgpu::Buffer, count: usize) -> Vec<f32> {
    let bytes = (count as u64 * 4).max(16);
    let staging = state.gpu.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("belief_read_probe::readback_staging"),
        size: bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = state
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("belief_read_probe::readback") });
    encoder.copy_buffer_to_buffer(buf, 0, &staging, 0, bytes);
    state.gpu.queue.submit(Some(encoder.finish()));
    let slice = staging.slice(..bytes);
    slice.map_async(wgpu::MapMode::Read, |res| res.expect("readback_staging map_async failed"));
    state.gpu.device.poll(wgpu::PollType::Wait).expect("device poll");
    let out = {
        let view = slice.get_mapped_range();
        let words: &[f32] = bytemuck::cast_slice(&view);
        words[..count].to_vec()
    };
    staging.unmap();
    out
}
