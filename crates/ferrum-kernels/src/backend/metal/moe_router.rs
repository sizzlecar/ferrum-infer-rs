//! GPU-side MoE router — softmax + top-K + optional renormalize over
//! `[batch, num_experts]` router logits, output `[batch, top_k]` ids
//! and weights. See `moe_router.metal` for the algorithmic notes.
//!
//! Eliminates the per-layer `B::sync + B::to_vec + host route()` round
//! trip used by the previous decode-style stacked dispatch path.

#![cfg(all(target_os = "macos", feature = "metal"))]

use std::ffi::c_void;
use std::sync::OnceLock;

use metal::{
    Buffer, CompileOptions, ComputeCommandEncoderRef, ComputePipelineState, Device, MTLSize,
};

const SHADER_SRC: &str = include_str!("moe_router.metal");
const KERNEL_NAME: &str = "moe_router_topk_softmax_f32";
const IDS_TPE_KERNEL_NAME: &str = "moe_compute_ids_tpe_f32";
const IDS_TPE_THREADS: u64 = 256;

static PIPELINE: OnceLock<ComputePipelineState> = OnceLock::new();
static IDS_TPE_PIPELINE: OnceLock<ComputePipelineState> = OnceLock::new();

fn pipeline(device: &Device) -> &'static ComputePipelineState {
    PIPELINE.get_or_init(|| {
        let lib = device
            .new_library_with_source(SHADER_SRC, &CompileOptions::new())
            .expect("compile moe_router.metal");
        let function = lib
            .get_function(KERNEL_NAME, None)
            .expect("find moe_router_topk_softmax_f32");
        device
            .new_compute_pipeline_state_with_function(&function)
            .expect("build moe_router_topk_softmax_f32 pipeline")
    })
}

fn ids_tpe_pipeline(device: &Device) -> &'static ComputePipelineState {
    IDS_TPE_PIPELINE.get_or_init(|| {
        let lib = device
            .new_library_with_source(SHADER_SRC, &CompileOptions::new())
            .expect("compile moe_router.metal");
        let function = lib
            .get_function(IDS_TPE_KERNEL_NAME, None)
            .expect("find moe_compute_ids_tpe_f32");
        device
            .new_compute_pipeline_state_with_function(&function)
            .expect("build moe_compute_ids_tpe_f32 pipeline")
    })
}

/// Dispatch the GPU-side router kernel on an existing compute encoder.
///
/// Inputs:
/// - `logits`: `[batch, num_experts]` f32
///
/// Outputs:
/// - `out_ids`: `[batch, top_k]` i32 (selected expert indices, smaller-
///   index-wins tie-breaking to match the host `route()` reference)
/// - `out_weights`: `[batch, top_k]` f32 (post-softmax probabilities,
///   optionally renormalised so each row sums to 1)
pub fn dispatch_route_topk_softmax(
    device: &Device,
    enc: &ComputeCommandEncoderRef,
    logits: &Buffer,
    out_ids: &Buffer,
    out_weights: &Buffer,
    batch: usize,
    num_experts: usize,
    top_k: usize,
    norm_topk_prob: bool,
) {
    debug_assert!(top_k <= num_experts);
    debug_assert!(top_k > 0);

    #[repr(C)]
    struct P {
        num_experts: i32,
        top_k: i32,
        norm_topk_prob: i32,
    }
    let params = P {
        num_experts: num_experts as i32,
        top_k: top_k as i32,
        norm_topk_prob: if norm_topk_prob { 1 } else { 0 },
    };

    let pipe = pipeline(device);
    enc.set_compute_pipeline_state(pipe);
    enc.set_buffer(0, Some(logits), 0);
    enc.set_buffer(1, Some(out_ids), 0);
    enc.set_buffer(2, Some(out_weights), 0);
    enc.set_bytes(
        3,
        std::mem::size_of::<P>() as u64,
        &params as *const _ as *const c_void,
    );

    // Threadgroup memory: probs[num_experts] f32 + sel_weights[top_k] f32
    //                   + sel_idxs[top_k] i32 + renorm_slot[1] f32.
    let shmem_bytes = (num_experts * 4 + top_k * 4 + top_k * 4 + 4).max(64);
    enc.set_threadgroup_memory_length(0, shmem_bytes as u64);

    let grid = MTLSize::new(batch as u64, 1, 1);
    let tg = MTLSize::new(32, 1, 1);
    enc.dispatch_thread_groups(grid, tg);
}

/// Dispatch the GPU-side bucket-sort that turns the `[batch, top_k]`
/// selected expert IDs from `dispatch_route_topk_softmax` into the
/// `(tpe, ids)` arrays consumed by `gemm_quant_moe_id`, plus the
/// indirect-dispatch args buffers `gate_up_args` and `down_args`.
///
/// Buffer expectations:
/// - `selected_ids`: `[batch * top_k]` i32 (output of router)
/// - `tpe`: `[num_experts]` i32 — overwritten (zeroed + filled)
/// - `ids`: `[num_experts * row_stride]` i32 — only the first `tpe[e]`
///   entries of each expert's row are written; consumer reads only
///   that prefix.
/// - `gate_up_args` / `down_args`: 12-byte buffers receiving
///   `(grid_x, grid_y, grid_z)` u32 triples. `grid_x` is shared (same
///   `max(tpe[e])`); `grid_y` differs because `M` differs between
///   gate/up (`m_gate_up`) and down (`m_down`).
///
/// `row_stride` is the worst-case `batch * top_k`. Tightening it would
/// require a separate compaction pass; the GEMM kernel's
/// `r1 >= tpe[e]` early-exit handles the over-strided indices for free.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_compute_ids_tpe(
    device: &Device,
    enc: &ComputeCommandEncoderRef,
    selected_ids: &Buffer,
    tpe: &Buffer,
    ids: &Buffer,
    gate_up_args: &Buffer,
    down_args: &Buffer,
    batch: usize,
    num_experts: usize,
    top_k: usize,
    m_gate_up: usize,
    m_down: usize,
) {
    debug_assert!(top_k > 0);

    #[repr(C)]
    struct P {
        num_experts: i32,
        row_stride: i32,
        total_pairs: i32,
        m_gate_up: i32,
        m_down: i32,
    }
    let total_pairs = batch * top_k;
    let params = P {
        num_experts: num_experts as i32,
        row_stride: total_pairs as i32,
        total_pairs: total_pairs as i32,
        m_gate_up: m_gate_up as i32,
        m_down: m_down as i32,
    };

    let pipe = ids_tpe_pipeline(device);
    enc.set_compute_pipeline_state(pipe);
    enc.set_buffer(0, Some(selected_ids), 0);
    enc.set_buffer(1, Some(tpe), 0);
    enc.set_buffer(2, Some(ids), 0);
    enc.set_buffer(3, Some(gate_up_args), 0);
    enc.set_buffer(4, Some(down_args), 0);
    enc.set_bytes(
        5,
        std::mem::size_of::<P>() as u64,
        &params as *const _ as *const c_void,
    );

    // Single threadgroup with `IDS_TPE_THREADS` threads; the kernel
    // strides over experts (zeroing / max) and pairs (bucketing).
    let grid = MTLSize::new(1, 1, 1);
    let tg = MTLSize::new(IDS_TPE_THREADS, 1, 1);
    enc.dispatch_thread_groups(grid, tg);
}

#[cfg(test)]
mod tests {
    use super::*;
    use metal::MTLResourceOptions;

    /// Compare GPU router output against a host softmax+top-K reference
    /// for a small synthetic logits matrix. Catches mass-misordering
    /// bugs (e.g. using wrong tie-break, off-by-one in masking, weights
    /// not normalised).
    #[test]
    fn router_matches_host_reference() {
        let Some(device) = Device::system_default() else {
            eprintln!("no Metal device — skipping");
            return;
        };
        let queue = device.new_command_queue();

        let batch = 3;
        let num_experts = 16;
        let top_k = 4;

        // Build deterministic logits with one clear winner per row.
        let mut logits: Vec<f32> = Vec::with_capacity(batch * num_experts);
        for b in 0..batch {
            for e in 0..num_experts {
                let v = ((b as f32) * 0.7 + (e as f32) * 0.13).sin() + (e as f32) * 0.05;
                logits.push(v);
            }
        }

        let logits_buf = device.new_buffer_with_data(
            logits.as_ptr() as *const _,
            (logits.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let ids_buf = device.new_buffer(
            (batch * top_k * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let w_buf = device.new_buffer(
            (batch * top_k * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        dispatch_route_topk_softmax(
            &device,
            enc,
            &logits_buf,
            &ids_buf,
            &w_buf,
            batch,
            num_experts,
            top_k,
            true,
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let gpu_ids: &[i32] =
            unsafe { std::slice::from_raw_parts(ids_buf.contents() as *const i32, batch * top_k) };
        let gpu_w: &[f32] =
            unsafe { std::slice::from_raw_parts(w_buf.contents() as *const f32, batch * top_k) };

        for b in 0..batch {
            // Host softmax + top-K + renorm reference.
            let row = &logits[b * num_experts..(b + 1) * num_experts];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|x| (x - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|e| e / sum).collect();

            // Top-K via partial sort, smaller-index tie-break.
            let mut indexed: Vec<(usize, f32)> =
                probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
            indexed.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            let topk: Vec<usize> = indexed.iter().take(top_k).map(|(i, _)| *i).collect();
            let renorm_sum: f32 = topk.iter().map(|&i| probs[i]).sum();
            let weights_ref: Vec<f32> = topk.iter().map(|&i| probs[i] / renorm_sum).collect();

            for k in 0..top_k {
                let gpu_id = gpu_ids[b * top_k + k] as usize;
                let gpu_weight = gpu_w[b * top_k + k];
                assert_eq!(
                    gpu_id, topk[k],
                    "row {} slot {}: gpu id {} != host id {}",
                    b, k, gpu_id, topk[k]
                );
                let diff = (gpu_weight - weights_ref[k]).abs();
                assert!(
                    diff < 1e-5,
                    "row {} slot {}: gpu weight {} vs host {} (diff {})",
                    b,
                    k,
                    gpu_weight,
                    weights_ref[k],
                    diff
                );
            }
        }
    }

    /// Verify the GPU bucket-sort that builds (tpe, ids) from a flat
    /// `[batch, top_k]` selected-expert array. Compares against the
    /// host-side reference in `ferrum_kernels::moe_host::compute_ids_tpe`.
    ///
    /// The GPU kernel uses the worst-case row stride `batch * top_k`,
    /// while the host reference returns a tight `max_per_expert` stride
    /// — so we re-pack the host output into a worst-case layout for the
    /// comparison and only check the first `tpe[e]` entries of each row
    /// (the only ones the consumer GEMM reads).
    #[test]
    fn compute_ids_tpe_matches_host() {
        let Some(device) = Device::system_default() else {
            eprintln!("no Metal device — skipping");
            return;
        };
        let queue = device.new_command_queue();

        let batch = 5usize;
        let num_experts = 8usize;
        let top_k = 3usize;
        let total_pairs = batch * top_k;
        let row_stride = total_pairs;

        // Deterministic synthetic routing — exercises hot/cold experts
        // and one expert that gets >1 slot from the same token.
        let selected: Vec<i32> = (0..total_pairs)
            .map(|i| {
                let b = i / top_k;
                let k = i % top_k;
                let e = (b * 3 + k * 2 + b) % num_experts;
                e as i32
            })
            .collect();

        let sel_buf = device.new_buffer_with_data(
            selected.as_ptr() as *const _,
            (selected.len() * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let tpe_buf = device.new_buffer(
            (num_experts * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let ids_buf = device.new_buffer(
            (num_experts * row_stride * 4) as u64,
            MTLResourceOptions::StorageModeShared,
        );
        let gate_up_args = device.new_buffer(12, MTLResourceOptions::StorageModeShared);
        let down_args = device.new_buffer(12, MTLResourceOptions::StorageModeShared);

        // Synthetic M values — only used to derive grid_y in the args.
        let m_gate_up = 768usize;
        let m_down = 2048usize;

        let cmd = queue.new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        dispatch_compute_ids_tpe(
            &device,
            enc,
            &sel_buf,
            &tpe_buf,
            &ids_buf,
            &gate_up_args,
            &down_args,
            batch,
            num_experts,
            top_k,
            m_gate_up,
            m_down,
        );
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();

        let gpu_tpe: &[i32] =
            unsafe { std::slice::from_raw_parts(tpe_buf.contents() as *const i32, num_experts) };
        let gpu_ids: &[i32] = unsafe {
            std::slice::from_raw_parts(ids_buf.contents() as *const i32, num_experts * row_stride)
        };
        let gpu_gate_up_args: &[u32] =
            unsafe { std::slice::from_raw_parts(gate_up_args.contents() as *const u32, 3) };
        let gpu_down_args: &[u32] =
            unsafe { std::slice::from_raw_parts(down_args.contents() as *const u32, 3) };

        // Host reference. Note: GPU output uses worst-case row_stride;
        // host output uses tight max_per_expert. Compare row-by-row
        // contents up to `tpe[e]`, ignoring stride differences.
        let selected_u32: Vec<u32> = selected.iter().map(|&v| v as u32).collect();
        let (host_tpe, host_ids, host_mpe) =
            crate::moe_host::compute_ids_tpe(&selected_u32, num_experts, batch, top_k);

        assert_eq!(gpu_tpe, host_tpe.as_slice(), "tpe mismatch");

        for e in 0..num_experts {
            let count = host_tpe[e] as usize;
            let mut gpu_row: Vec<i32> = gpu_ids[e * row_stride..e * row_stride + count].to_vec();
            let mut host_row: Vec<i32> = host_ids[e * host_mpe..e * host_mpe + count].to_vec();
            // Slot order is non-deterministic across atomic claims —
            // sort both before comparing the (multi-)set.
            gpu_row.sort_unstable();
            host_row.sort_unstable();
            assert_eq!(gpu_row, host_row, "expert {e}: id set mismatch");
        }

        // Verify the indirect-dispatch args track max(tpe[e]).
        let expected_max_pe = host_tpe.iter().copied().max().unwrap_or(0).max(1);
        let expected_grid_x = (expected_max_pe as u32 + 31) / 32;
        let expected_grid_y_gu = (m_gate_up as u32 + 63) / 64;
        let expected_grid_y_dn = (m_down as u32 + 63) / 64;
        let expected_grid_z = num_experts as u32;
        assert_eq!(
            gpu_gate_up_args,
            [expected_grid_x, expected_grid_y_gu, expected_grid_z],
            "gate_up indirect args mismatch"
        );
        assert_eq!(
            gpu_down_args,
            [expected_grid_x, expected_grid_y_dn, expected_grid_z],
            "down indirect args mismatch"
        );
    }
}
