//! Test-only gate/up F32-scale Q8 integer MMQ/Stream-K; strict down remains.
use super::*;
use cudarc::driver::{sys, CudaGraph, CudaSlice, DevicePtr, DevicePtrMut};
use ferrum_interfaces::vnext::{ElementType, WeightId};
mod fixture;
mod screen;

use fixture::{Case, Matrix, PAD, SENTINEL};

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
    terms: usize,
    shared_bytes: u32,
}
impl Candidate {
    fn load(ctx: &Arc<CudaContext>, terms: usize) -> Self {
        assert!(matches!(terms, 1 | 2));
        let shared_bytes = if terms == 1 { 45696 } else { 48384 };
        let suffix = if terms == 1 { "" } else { "_residual2" };
        let m = ctx
            .load_module(Ptx::from_src(include_str!(concat!(
                env!("OUT_DIR"),
                "/vnext_q4_stream_mmq.ptx"
            ))))
            .unwrap();
        let activation = ctx
            .load_module(Ptx::from_src(crate::ptx::FUSED_SILU_MUL))
            .unwrap();
        let project = m
            .load_function(&format!("vnext_q4_stream_mmq{suffix}"))
            .unwrap();
        project
            .set_attribute(
                sys::CUfunction_attribute_enum::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                shared_bytes as i32,
            )
            .unwrap();
        let sm_count = usize::try_from(
            ctx.attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
                .unwrap(),
        )
        .unwrap();
        let active_ctas_per_sm = project
            .occupancy_max_active_blocks_per_multiprocessor(256, shared_bytes as usize, None)
            .unwrap();
        assert!(sm_count > 0 && active_ctas_per_sm > 0);
        let cta_budget = sm_count.checked_mul(active_ctas_per_sm as usize).unwrap();
        println!(
            "{}",
            serde_json::json!({
                "event":"occupancy_grid_policy", "revision":"residual2-r1", "sm_count":sm_count,
                "max_active_ctas_per_sm":active_ctas_per_sm, "cta_budget":cta_budget,
                "block_threads":256, "dynamic_shared_bytes":shared_bytes,
                "static_shared_bytes":project.shared_size_bytes().unwrap(),
                "registers_per_thread":project.num_regs().unwrap(),
                "local_bytes_per_thread":project.local_size_bytes().unwrap(),
                "achieved_occupancy_measured":false
            })
        );
        Self {
            project,
            metadata: m.load_function("vnext_q4_stream_metadata").unwrap(),
            pack: m
                .load_function(&format!("vnext_q4_stream_pack{suffix}"))
                .unwrap(),
            fixup: m.load_function("vnext_q4_stream_fixup").unwrap(),
            silu: activation
                .load_function("fused_silu_mul_interleaved_f16")
                .unwrap(),
            cta_budget,
            terms,
            shared_bytes,
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
                shared_mem_bytes: self.shared_bytes,
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
