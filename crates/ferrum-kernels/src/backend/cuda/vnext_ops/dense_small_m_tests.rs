//! Diagnostic only: a dense F16 small-row kernel against the current cuBLAS
//! launcher. There is no production provider/selector for the new module.
//!
//! Every output has an independent, serial F64 dot-product reference. The
//! mathematical F32 forward-error bound and relative 1/1000 comparisons to the
//! rounded F64 reference are reported separately. Different F32 reduction trees
//! are not assumed bit-identical, nor does this benchmark qualify model output.

use super::transformer::launch_gemm_f16;
use cudarc::cublas::CudaBlas;
use cudarc::driver::{
    sys::CUevent_flags, CudaContext, CudaFunction, CudaSlice, CudaStream, DevicePtr, LaunchConfig,
    PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use half::f16;
use sha2::{Digest, Sha256};
use std::{sync::Arc, time::Instant};

mod oracle;
mod reduction;
mod workspace;
use oracle::{reference as f64_reference, validate, Dot};

const GUARD: u16 = 0xd755;
const GUARD_ELEMENTS: usize = 8;

#[derive(Clone, Copy)]
struct Shape {
    name: &'static str,
    rows: usize,
    outputs: usize,
    inputs: usize,
}

impl Shape {
    fn new(name: &'static str, rows: usize, outputs: usize, inputs: usize) -> Self {
        for extent in [rows, outputs, inputs] {
            assert!(extent != 0 && i32::try_from(extent).is_ok());
        }
        for elements in [
            rows.checked_mul(inputs),
            outputs.checked_mul(inputs),
            rows.checked_mul(outputs),
        ] {
            elements
                .unwrap()
                .checked_add(2 * GUARD_ELEMENTS)
                .unwrap()
                .checked_mul(2)
                .unwrap();
        }
        Self {
            name,
            rows,
            outputs,
            inputs,
        }
    }

    fn uses_candidate(self) -> bool {
        matches!(self.rows, 1 | 2) && self.inputs.is_multiple_of(2)
    }
}

// One allocation per buffer, retained until the stream is synchronized. Unlike
// the tiny Guarded fixture, this does not clone the 2 GB LM-head host matrix.
struct Buffer {
    device: CudaSlice<u16>,
    host: Vec<u16>,
}

impl Buffer {
    fn new(stream: &Arc<CudaStream>, elements: usize, mut value: impl FnMut(usize) -> u16) -> Self {
        let mut host = Vec::with_capacity(elements.checked_add(2 * GUARD_ELEMENTS).unwrap());
        host.extend(std::iter::repeat_n(GUARD, GUARD_ELEMENTS));
        host.extend((0..elements).map(&mut value));
        host.extend(std::iter::repeat_n(GUARD, GUARD_ELEMENTS));
        let device = stream.clone_htod(&host).unwrap();
        Self { device, host }
    }

    fn values(&self) -> &[u16] {
        &self.host[GUARD_ELEMENTS..self.host.len() - GUARD_ELEMENTS]
    }

    fn pointer(&self, stream: &Arc<CudaStream>) -> u64 {
        let pointer = self.device.device_ptr(stream).0 + (GUARD_ELEMENTS * 2) as u64;
        assert_eq!(pointer % 4, 0);
        pointer
    }

    fn read(&self, stream: &Arc<CudaStream>) -> Vec<u16> {
        let result = stream.clone_dtoh(&self.device).unwrap();
        assert!(result[..GUARD_ELEMENTS].iter().all(|&v| v == GUARD));
        assert!(result[result.len() - GUARD_ELEMENTS..]
            .iter()
            .all(|&v| v == GUARD));
        result[GUARD_ELEMENTS..result.len() - GUARD_ELEMENTS].to_vec()
    }

    fn assert_read_only(&self, stream: &Arc<CudaStream>) {
        // Check the entire matrix, with at most 1 MiB of readback staging.
        for start in (0..self.host.len()).step_by(512 * 1024) {
            let end = (start + 512 * 1024).min(self.host.len());
            let actual = stream.clone_dtoh(&self.device.slice(start..end)).unwrap();
            assert_eq!(
                actual,
                self.host[start..end],
                "read-only input modified at {start}"
            );
        }
    }
}

struct Kernels {
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    one: CudaFunction,
    two: CudaFunction,
}

struct PreparedLaunch<'a> {
    shape: Shape,
    pointers: [u64; 3],
    // Raw CUDA entrypoints do not take ownership. Keep all original buffers
    // borrowed until the caller has observed the completion fence.
    _owners: [&'a Buffer; 3],
}

impl Kernels {
    fn new() -> Self {
        let context = CudaContext::new(0).expect("dense small-M diagnostic requires CUDA");
        let stream = context.default_stream();
        let module = context
            .load_module(Ptx::from_src(crate::ptx::DENSE_SMALL_M_F16.to_owned()))
            .unwrap();
        Self {
            blas: CudaBlas::new(stream.clone()).unwrap(),
            one: module.load_function("dense_small_m_f16_m1").unwrap(),
            two: module.load_function("dense_small_m_f16_m2").unwrap(),
            stream,
        }
    }

    fn prepare<'a>(
        &self,
        shape: Shape,
        x: &'a Buffer,
        w: &'a Buffer,
        y: &'a Buffer,
    ) -> PreparedLaunch<'a> {
        assert!(x.values().len() >= shape.rows * shape.inputs);
        assert_eq!(w.values().len(), shape.outputs * shape.inputs);
        assert!(y.values().len() >= shape.rows * shape.outputs);
        // The three separately owned allocations remain live through the fence.
        let xp = x.pointer(&self.stream);
        let wp = w.pointer(&self.stream);
        let yp = y.pointer(&self.stream);
        let intervals = [
            xp..xp
                .checked_add((shape.rows * shape.inputs * 2) as u64)
                .unwrap(),
            wp..wp
                .checked_add((shape.outputs * shape.inputs * 2) as u64)
                .unwrap(),
            yp..yp
                .checked_add((shape.rows * shape.outputs * 2) as u64)
                .unwrap(),
        ];
        for i in 0..3 {
            for j in i + 1..3 {
                assert!(
                    intervals[i].end <= intervals[j].start
                        || intervals[j].end <= intervals[i].start
                );
            }
        }
        PreparedLaunch {
            shape,
            pointers: [xp, wp, yp],
            _owners: [x, w, y],
        }
    }

    fn launch(&self, candidate: bool, prepared: &PreparedLaunch<'_>) {
        let shape = prepared.shape;
        let [xp, wp, yp] = prepared.pointers;
        let outputs = i32::try_from(shape.outputs).unwrap();
        let inputs = i32::try_from(shape.inputs).unwrap();
        if candidate && shape.uses_candidate() {
            let function = if shape.rows == 1 {
                &self.one
            } else {
                &self.two
            };
            let mut args = self.stream.launch_builder(function);
            args.arg(&xp).arg(&wp).arg(&yp).arg(&outputs).arg(&inputs);
            unsafe {
                args.launch(LaunchConfig {
                    grid_dim: (u32::try_from(shape.outputs.div_ceil(4)).unwrap(), 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                })
            }
            .unwrap();
        } else {
            // This fallback exists only in this diagnostic adapter. Production
            // selection remains entirely unchanged, including long prefills.
            launch_gemm_f16(
                &self.blas,
                xp,
                wp,
                yp,
                i32::try_from(shape.rows).unwrap(),
                outputs,
                inputs,
                "dense small-M diagnostic cuBLAS reference",
            )
            .unwrap();
        }
    }
}

fn value(index: usize, salt: u32, scale: f32) -> u16 {
    let mut mixed = (index as u32).wrapping_add(salt).wrapping_mul(0x9e37_79b1);
    mixed ^= mixed >> 16;
    mixed = mixed.wrapping_mul(0x85eb_ca6b);
    mixed ^= mixed >> 13;
    f16::from_f32((((mixed >> 12) & 4095) as i32 - 2048) as f32 * scale).to_bits()
}

fn validation_report(
    shape: Shape,
    old: &[u16],
    candidate: &[u16],
    reference: &[Dot],
) -> serde_json::Value {
    let old_stats = validate(old, reference);
    let candidate_stats = validate(candidate, reference);
    let mut max_abs = 0.0_f64;
    let mut squared_error = 0.0;
    let mut squared_reference = 0.0;
    let mut bit_mismatches = 0;
    for (&a, &b) in candidate.iter().zip(old) {
        bit_mismatches += usize::from(a != b);
        let (a, b) = (f16::from_bits(a).to_f64(), f16::from_bits(b).to_f64());
        if a.is_finite() && b.is_finite() {
            let difference = a - b;
            max_abs = max_abs.max(difference.abs());
            squared_error += difference * difference;
            squared_reference += b * b;
        }
    }
    serde_json::json!({
        "diagnostic":"dense_small_m_f16_validation", "shape":shape.name,
        "m":shape.rows,"n":shape.outputs,"k":shape.inputs,
        "route":if shape.uses_candidate() {"warp_shared_weight"} else {"unchanged_cublas_fallback"},
        "all_outputs_checked":reference.len(), "cublas":old_stats,"candidate":candidate_stats,
        "candidate_vs_cublas":{"bit_mismatches":bit_mismatches,"max_abs":max_abs,
            "rmse":(squared_error / old.len() as f64).sqrt(),
            "rel_l2":(squared_reference != 0.0).then(||(squared_error / squared_reference).sqrt())},
        "oracle":"serial F64 dot from actual F16 carriers, then F16 RNE; complete output, not sampling",
        "qualification":"forward-error checks are mandatory; the 1/1000 comparison uses rounded F64, not the production CPU operation; it is evidence, not product numerical qualification"
    })
}

#[test]
#[ignore = "requires CUDA; complete F64 oracle, guard and fallback diagnostic"]
fn cuda_dense_small_m_f16_full_oracle() {
    let kernels = Kernels::new();
    for (m, n, k) in [
        (1, 7, 2),
        (2, 13, 70),
        (2, 17, 4098),
        (1, 9, 12290),
        (2, 9, 31),
        (17, 7, 130),
    ] {
        let shape = Shape::new("tail_or_fallback", m, n, k);
        let x = Buffer::new(&kernels.stream, m * k, |i| value(i, 37, 1.0 / 1024.0));
        let w = Buffer::new(&kernels.stream, n * k, |i| value(i, 731, 1.0 / 65536.0));
        let outputs = [
            Buffer::new(&kernels.stream, m * n, |_| GUARD),
            Buffer::new(&kernels.stream, m * n, |_| GUARD),
        ];
        let reference = f64_reference(shape, x.values(), w.values());
        let prepared = outputs
            .each_ref()
            .map(|output| kernels.prepare(shape, &x, &w, output));
        for (arm, prepared) in prepared.iter().enumerate() {
            kernels.launch(arm == 1, prepared);
        }
        kernels.stream.synchronize().unwrap();
        let old = outputs[0].read(&kernels.stream);
        let candidate = outputs[1].read(&kernels.stream);
        println!("{}", validation_report(shape, &old, &candidate, &reference));
        if !shape.uses_candidate() {
            assert_eq!(old, candidate);
        }
        x.assert_read_only(&kernels.stream);
        w.assert_read_only(&kernels.stream);
    }
    let shape = Shape::new("subnormal_and_rounding_ties", 2, 5, 64);
    let x = Buffer::new(&kernels.stream, 128, |i| {
        f16::from_f32(if i < 64 { 1.0 } else { 2.0 }).to_bits()
    });
    let w = Buffer::new(&kernels.stream, 5 * 64, |i| match (i / 64, i % 64) {
        (0, 0) => 1,
        (1, 0) => 0x8001,
        (2, 0) => f16::from_f32(0.5).to_bits(),
        (3, 0) => f16::from_f32(0.50048828125).to_bits(),
        (2 | 3, 1) => f16::from_f32(0.000244140625).to_bits(),
        (4, _) => 0x8000,
        _ => 0,
    });
    let reference = f64_reference(shape, x.values(), w.values());
    for candidate in [false, true] {
        let output = Buffer::new(&kernels.stream, 10, |_| GUARD);
        let prepared = kernels.prepare(shape, &x, &w, &output);
        kernels.launch(candidate, &prepared);
        kernels.stream.synchronize().unwrap();
        let actual = output.read(&kernels.stream);
        validate(&actual, &reference);
        assert_eq!(
            actual,
            reference
                .iter()
                .map(|dot| f16::from_f64(dot.value).to_bits())
                .collect::<Vec<_>>(),
            "exact dyadic sums exercise RNE ties, subnormal and signed-zero stores"
        );
    }
    x.assert_read_only(&kernels.stream);
    w.assert_read_only(&kernels.stream);
    // Separate nonfinite carriers make propagation explicit without allowing
    // NaNs to disappear into a numeric error/relative-tolerance calculation.
    let shape = Shape::new("nonfinite", 2, 7, 64);
    let x = Buffer::new(&kernels.stream, 128, |i| {
        f16::from_f32(if i < 64 { 1.0 } else { 2.0 }).to_bits()
    });
    let w = Buffer::new(&kernels.stream, 7 * 64, |i| match (i / 64, i % 64) {
        (2, 0) => 0x7e01,
        (3, 0) | (5, 0) => 0x7c00,
        (4, 0) | (5, 1) => 0xfc00,
        (6, _) => 0x8000,
        (1, _) => f16::from_f32(-0.25).to_bits(),
        _ => f16::from_f32(0.25).to_bits(),
    });
    let outputs = [
        Buffer::new(&kernels.stream, 14, |_| GUARD),
        Buffer::new(&kernels.stream, 14, |_| GUARD),
    ];
    let reference = f64_reference(shape, x.values(), w.values());
    let prepared = outputs
        .each_ref()
        .map(|output| kernels.prepare(shape, &x, &w, output));
    for (arm, prepared) in prepared.iter().enumerate() {
        kernels.launch(arm == 1, prepared);
    }
    kernels.stream.synchronize().unwrap();
    for output in &outputs {
        validate(&output.read(&kernels.stream), &reference);
    }
    x.assert_read_only(&kernels.stream);
    w.assert_read_only(&kernels.stream);
}

#[test]
#[ignore = "requires CUDA and about 2.1 GiB each of host/device memory; full-size diagnostic, no latency PASS threshold"]
fn cuda_dense_small_m_f16_qwen35_microbench() {
    const ITERATIONS: usize = 8;
    const PAIRS: usize = 8;
    let kernels = Kernels::new();
    // One matrix at a time: these are the actual dense Qwen3.5 dimensions, not
    // a full model load. The single gate/up shape is a diagnostic subprojection;
    // production already uses the larger packed gate/up matrix.
    for (name, n, k) in [
        ("ffn_single_gate_or_up", 12288, 4096),
        ("ffn_packed_gate_up", 24576, 4096),
        ("ffn_down", 4096, 12288),
        ("gdn_packed_qkvzba", 12352, 4096),
        ("gdn_output", 4096, 4096),
        ("lm_head", 248320, 4096),
    ] {
        let rows: &[usize] = if name == "ffn_down" {
            &[1, 2, 8]
        } else {
            &[1, 2]
        };
        let maximum_rows = *rows.last().unwrap();
        let maximum = Shape::new(name, maximum_rows, n, k);
        let x = Buffer::new(&kernels.stream, maximum_rows * k, |i| {
            value(i, 37, 1.0 / 1024.0)
        });
        let w = Buffer::new(&kernels.stream, n * k, |i| value(i, 731, 1.0 / 65536.0));
        let outputs = [
            Buffer::new(&kernels.stream, maximum_rows * n, |_| GUARD),
            Buffer::new(&kernels.stream, maximum_rows * n, |_| GUARD),
        ];
        let oracle_start = Instant::now();
        let reference = f64_reference(maximum, x.values(), w.values());
        let oracle_wall_ms = oracle_start.elapsed().as_secs_f64() * 1000.0;
        for &m in rows {
            let shape = Shape::new(name, m, n, k);
            let prepared = outputs
                .each_ref()
                .map(|output| kernels.prepare(shape, &x, &w, output));
            for (arm, prepared) in prepared.iter().enumerate() {
                kernels.launch(arm == 1, prepared);
            }
            kernels.stream.synchronize().unwrap();
            let old_full = outputs[0].read(&kernels.stream);
            let candidate_full = outputs[1].read(&kernels.stream);
            assert!(old_full[m * n..].iter().all(|&v| v == GUARD));
            assert!(candidate_full[m * n..].iter().all(|&v| v == GUARD));
            let old = &old_full[..m * n];
            let candidate = &candidate_full[..m * n];
            println!(
                "{}",
                validation_report(shape, old, candidate, &reference[..m * n])
            );
            // All arithmetic and readback validation precedes timing. Relative
            // oracle violations are not hidden by, or used as a latency gate.
            for _ in 0..4 {
                kernels.launch(false, &prepared[0]);
                kernels.launch(true, &prepared[1]);
            }
            kernels.stream.synchronize().unwrap();
            for pair in 0..PAIRS {
                let order = if pair % 2 == 0 { [0, 1] } else { [1, 0] };
                let mut timing = [[0.0_f64; 3]; 2];
                for arm in order {
                    kernels.stream.synchronize().unwrap();
                    let wall = Instant::now();
                    let start = kernels
                        .stream
                        .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                        .unwrap();
                    let enqueue = Instant::now();
                    for _ in 0..ITERATIONS {
                        kernels.launch(arm == 1, &prepared[arm]);
                    }
                    let enqueue_ns = enqueue.elapsed().as_nanos() as f64 / ITERATIONS as f64;
                    let end = kernels
                        .stream
                        .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
                        .unwrap();
                    end.synchronize().unwrap();
                    timing[arm] = [
                        f64::from(start.elapsed_ms(&end).unwrap()) * 1e6 / ITERATIONS as f64,
                        enqueue_ns,
                        wall.elapsed().as_nanos() as f64 / ITERATIONS as f64,
                    ];
                }
                println!(
                    "{}",
                    serde_json::json!({
                        "diagnostic":"dense_small_m_f16_projection", "shape":name,"m":m,"n":n,"k":k,
                        "pair":pair,"order":order,"iterations":ITERATIONS,
                        "candidate_route":if shape.uses_candidate() {"warp_shared_weight"} else {"unchanged_cublas_fallback"},
                        "weight_bytes":n * k * 2,"oracle_wall_ms":oracle_wall_ms,
                        "cublas":{"cuda_event_ns":timing[0][0],"host_enqueue_ns":timing[0][1],"completion_wall_ns":timing[0][2]},
                        "candidate":{"cuda_event_ns":timing[1][0],"host_enqueue_ns":timing[1][1],"completion_wall_ns":timing[1][2]},
                        "kernel_sha256":format!("{:x}",Sha256::digest(include_bytes!("../../../../kernels/dense_small_m_f16.cu"))),
                        "timing_scope":"same owners/stream, warmed and alternating; events may include host-feed gaps; allocation, full F64 oracle and readback excluded",
                        "performance_threshold":null,"production_selected":false
                    })
                );
            }
            assert_eq!(&outputs[0].read(&kernels.stream)[..m * n], old);
            assert_eq!(&outputs[1].read(&kernels.stream)[..m * n], candidate);
        }
        x.assert_read_only(&kernels.stream);
        w.assert_read_only(&kernels.stream);
    }
}
