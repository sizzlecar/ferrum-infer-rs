//! Test-only gate/up F32-scale Q8 integer MMQ/Stream-K; strict down remains.
use super::*;
use cudarc::driver::{sys, CudaGraph, CudaSlice, DevicePtr, DevicePtrMut};
use ferrum_interfaces::vnext::{ElementType, WeightId};
mod fixture;
mod qualification;

use fixture::{Case, Matrix, PAD, SENTINEL};
const SHARED_BYTES: u32 = 45696;
fn tiles(rows: usize, outputs: usize) -> usize {
    rows.div_ceil(8) * outputs.div_ceil(128)
}
fn ctas(rows: usize, inputs: usize, outputs: usize, cta_budget: usize) -> usize {
    assert!(cta_budget > 0 && inputs > 0 && inputs % 256 == 0);
    (tiles(rows, outputs) * (inputs / 256)).min(cta_budget)
}
fn scratch_len(rows: usize, inputs: usize, outputs: usize, cta_budget: usize) -> usize {
    (tiles(rows, outputs) + ctas(rows, inputs, outputs, cta_budget)) * 1024
}
struct Candidate {
    project: CudaFunction,
    metadata: CudaFunction,
    pack: CudaFunction,
    fixup: CudaFunction,
    silu: CudaFunction,
    cta_budget: usize,
    product: super::super::stream_mmq::StreamMmq,
}
impl Candidate {
    fn load(ctx: &Arc<CudaContext>) -> Self {
        let m = ctx
            .load_module(Ptx::from_src(include_str!(concat!(
                env!("OUT_DIR"),
                "/vnext_q4_stream_mmq.ptx"
            ))))
            .unwrap();
        let activation = ctx
            .load_module(Ptx::from_src(crate::ptx::FUSED_SILU_MUL))
            .unwrap();
        let project = m.load_function("vnext_q4_stream_mmq").unwrap();
        project
            .set_attribute(
                sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                SHARED_BYTES as i32,
            )
            .unwrap();
        let sm_count = usize::try_from(
            ctx.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
                .unwrap(),
        )
        .unwrap();
        let active_ctas_per_sm = project
            .occupancy_max_active_blocks_per_multiprocessor(256, SHARED_BYTES as usize, None)
            .unwrap();
        assert!(sm_count > 0 && active_ctas_per_sm > 0);
        let cta_budget = sm_count.checked_mul(active_ctas_per_sm as usize).unwrap();
        println!(
            "{}",
            serde_json::json!({
                "event":"occupancy_grid_policy", "revision":"r2", "sm_count":sm_count,
                "max_active_ctas_per_sm":active_ctas_per_sm, "cta_budget":cta_budget,
                "block_threads":256, "dynamic_shared_bytes":SHARED_BYTES,
                "static_shared_bytes":project.shared_size_bytes().unwrap(),
                "registers_per_thread":project.num_regs().unwrap(),
                "local_bytes_per_thread":project.local_size_bytes().unwrap(),
                "achieved_occupancy_measured":false
            })
        );
        Self {
            project,
            metadata: m.load_function("vnext_q4_stream_metadata").unwrap(),
            pack: m.load_function("vnext_q4_stream_pack").unwrap(),
            fixup: m.load_function("vnext_q4_stream_fixup").unwrap(),
            silu: activation
                .load_function("fused_silu_mul_interleaved_f16")
                .unwrap(),
            cta_budget,
            product: super::super::stream_mmq::StreamMmq::load(ctx).unwrap(),
        }
    }
    fn report_shape(&self, rows: usize, inputs: usize, outputs: usize) {
        println!(
            "{}",
            serde_json::json!({
                "event":"stream_k_shape", "revision":"r2", "rows":rows, "inputs":inputs,
                "outputs":outputs, "output_tiles":tiles(rows, outputs),
                "k256_units":tiles(rows, outputs) * (inputs / 256),
                "selected_ctas":ctas(rows, inputs, outputs, self.cta_budget),
                "cta_budget":self.cta_budget,
                "scratch_bytes":scratch_len(rows, inputs, outputs, self.cta_budget) * 4
            })
        );
    }
    fn pack(
        &self,
        s: &Arc<CudaStream>,
        x: u64,
        q: u64,
        d: u64,
        sum: u64,
        rows: usize,
        inputs: usize,
    ) {
        let dims = [rows, inputs].map(|v| u32::try_from(v).unwrap());
        unsafe {
            s.launch_builder(&self.pack)
                .arg(&x)
                .arg(&q)
                .arg(&d)
                .arg(&sum)
                .arg(&dims[0])
                .arg(&dims[1])
                .launch(LaunchConfig {
                    grid_dim: ((rows * (inputs / 32)).div_ceil(8) as u32, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                })
        }
        .unwrap();
    }
    fn project(
        &self,
        s: &Arc<CudaStream>,
        q: u64,
        d: u64,
        sum: u64,
        w: u64,
        y: u64,
        partial: u64,
        rows: usize,
        inputs: usize,
        outputs: usize,
        stride: usize,
        offset: usize,
    ) {
        assert!(offset + outputs <= stride);
        let count = ctas(rows, inputs, outputs, self.cta_budget);
        let dims = [rows, inputs, outputs, count].map(|v| u32::try_from(v).unwrap());
        let mut b = s.launch_builder(&self.project);
        b.arg(&q).arg(&d).arg(&sum).arg(&w).arg(&partial);
        for v in &dims {
            b.arg(v);
        }
        unsafe {
            b.launch(LaunchConfig {
                grid_dim: (count as u32, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: SHARED_BYTES,
            })
        }
        .unwrap();
        let dims =
            [rows, inputs, outputs, stride, offset, count].map(|v| u32::try_from(v).unwrap());
        let mut b = s.launch_builder(&self.fixup);
        b.arg(&partial).arg(&y);
        for v in &dims {
            b.arg(v);
        }
        unsafe { b.launch(LaunchConfig::for_num_elems((rows * outputs) as u32)) }.unwrap();
    }
}
fn strict(
    s: &Arc<CudaStream>,
    k: &CudaNativeBlockKernels,
    x: u64,
    w: u64,
    y: u64,
    rows: usize,
    inputs: usize,
    outputs: usize,
    stride: usize,
    offset: usize,
    format: GgufBlockFormat,
) {
    let part = weights::MatrixPart {
        component_id: WeightId::new("test.q4-stream-mmq").unwrap(),
        format: weights::MatrixFormat::Block(format),
        rows: outputs as u32,
        columns: inputs as u32,
        output_offset: offset as u32,
        transform: None,
        signs_region: None,
    };
    k.linear_with_precision(
        s,
        x,
        w,
        y,
        &part,
        rows as u32,
        stride as u32,
        ElementType::F16,
        ElementType::F16,
    )
    .unwrap();
}
#[test]
fn q4_stream_mmq_layout_and_stream_k_coverage() {
    let mut seen = vec![0u8; 8 * 128];
    for warp in 0..8 {
        for lane in 0..32 {
            for n in 0..4 {
                seen[(2 * (lane % 4) + n % 2) * 128 + warp * 16 + lane / 4 + (n / 2) * 8] += 1;
            }
        }
    }
    assert!(seen.iter().all(|n| *n == 1));
    // Hardware-independent range tests, including more/fewer CTAs than tiles,
    // partial first/last tiles and multiple full tiles in a single CTA.
    for tiles in [1usize, 2, 7, 96, 171, 513] {
        for blocks in [1usize, 2, 3, 16, 48] {
            for cta_budget in [1usize, 3, 17, 170, 340, 509] {
                let total = tiles * blocks;
                let count = total.min(cta_budget);
                let mut covered = vec![0u8; total];
                let mut complete = vec![0u8; tiles];
                let mut tails = vec![None; count];
                for c in 0..count {
                    let mut pos = c * total / count;
                    let stop = (c + 1) * total / count;
                    for n in &mut covered[pos..stop] {
                        *n += 1;
                    }
                    while pos < stop {
                        let tile = pos / blocks;
                        let end = ((tile + 1) * blocks).min(stop);
                        if end % blocks == 0 {
                            complete[tile] += 1;
                        } else {
                            assert!(tails[c].is_none());
                            tails[c] = Some(tile);
                        }
                        pos = end;
                    }
                }
                assert!(covered.iter().all(|v| *v == 1));
                assert!(complete.iter().all(|v| *v == 1));
                for tile in 0..tiles {
                    let begin = tile * blocks;
                    let end = begin + blocks;
                    let first = ((begin + 1) * count - 1) / total;
                    let last = (end * count - 1) / total;
                    let mut units = 0;
                    for c in first..=last {
                        let start = c * total / count;
                        let stop = (c + 1) * total / count;
                        units += stop.min(end) - start.max(begin);
                        if stop < end {
                            assert_eq!(tails[c], Some(tile));
                        } else {
                            assert_eq!(c, last);
                        }
                    }
                    assert_eq!(units, blocks);
                }
            }
        }
    }
}
#[test]
fn q4_stream_mmq_fixture_has_finite_coefficients_and_dense_inputs() {
    for f in [GgufBlockFormat::Q4K, GgufBlockFormat::Q6K] {
        let (_, x) = fixture::templates(f, 512, 7);
        assert!(x.iter().all(|v| v.is_finite()));
        assert!(x.iter().any(|v| *v < 0.) && x.iter().any(|v| *v > 0.));
    }
    for g in 0..2 {
        let x = fixture::inputs(8, 4096, g);
        assert!(x[PAD..x.len() - PAD].iter().all(|x| x.is_finite()));
        assert!(
            x[PAD..x.len() - PAD]
                .iter()
                .filter(|x| x.to_f32() != 0.)
                .count()
                > 32000
        );
    }
}
