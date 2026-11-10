use super::{fixture::*, Result};
use cudarc::driver::{sys, CudaContext, DevicePtr, DevicePtrMut};
use ferrum_native_ops::cutlass_fused_f32_attention::{
    CutlassFusedF32AttentionBindingsV1 as Bindings, CutlassFusedF32AttentionInfoV1 as Info,
    CutlassFusedF32AttentionPrepareFn as Prepare, CutlassFusedF32AttentionRequestV1 as Request,
    CutlassFusedF32AttentionRunFn as Launch, CutlassFusedF32AttentionSpanV1 as Span,
};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{path::Path, time::Instant};

const GUARD: usize = 128;
const CANARY: u32 = 0x7e12_3456;

fn checked(status: i32, operation: &str) -> Result<()> {
    if status == 0 {
        Ok(())
    } else {
        Err(format!("{operation}: native status {status}").into())
    }
}
fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn guarded(values: &[f32]) -> Vec<f32> {
    let mut buffer = vec![f32::from_bits(CANARY); values.len() + 2 * GUARD];
    buffer[GUARD..GUARD + values.len()].copy_from_slice(values);
    buffer
}
fn canaries_ok(values: &[f32]) -> bool {
    values[..GUARD]
        .iter()
        .chain(&values[values.len() - GUARD..])
        .all(|x| x.to_bits() == CANARY)
}
fn unchanged(actual: &[f32], expected: &[f32]) -> bool {
    actual.len() == expected.len()
        && actual
            .iter()
            .zip(expected)
            .all(|(a, b)| a.to_bits() == b.to_bits())
}

fn qualify(
    context: &std::sync::Arc<CudaContext>,
    prepare: Prepare,
    launch: Launch,
    shape: Shape,
    iterations: u32,
) -> Result<serde_json::Value> {
    let fixture = Fixture::new(shape);
    let oracle_start = Instant::now();
    let reference = fixture.reference();
    let oracle_seconds = oracle_start.elapsed().as_secs_f64();
    let stream = context.new_stream()?;
    let host_q = guarded(&fixture.query);
    let host_k = guarded(&fixture.key);
    let host_v = guarded(&fixture.value);
    let host_output = guarded(&vec![f32::NAN; shape.elements()]);
    let query = stream.clone_htod(&host_q)?;
    let key = stream.clone_htod(&host_k)?;
    let value = stream.clone_htod(&host_v)?;
    let mut output = stream.clone_htod(&host_output)?;
    let request = Request {
        abi: 1,
        size: std::mem::size_of::<Request>() as u32,
        batches: shape.batches as u32,
        sequence: shape.sequence as u32,
        heads: shape.heads as u32,
        dimension: shape.dimension as u32,
        scale: fixture.scale,
        reserved: 0,
    };
    request.validate()?;
    let mut info = Info::default();
    // Warmup and all preparation, events, allocation and H2D are outside timing.
    let execution = (|| -> Result<_> {
        unsafe {
            checked(prepare(&request, &mut info), "prepare")?;
        }
        info.validate_for(&request)?;
        let start = context.new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let stop = context.new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
        let (q_ptr, _q_guard) = query.device_ptr(&stream);
        let (k_ptr, _k_guard) = key.device_ptr(&stream);
        let (v_ptr, _v_guard) = value.device_ptr(&stream);
        let (o_ptr, _o_guard) = output.device_ptr_mut(&stream);
        let span = |address: u64| Span {
            address: address + (GUARD * std::mem::size_of::<f32>()) as u64,
            elements: shape.elements() as u64,
        };
        let bindings = Bindings {
            query: span(q_ptr),
            key: span(k_ptr),
            value: span(v_ptr),
            output: span(o_ptr),
        };
        bindings.validate(&request)?;
        stream.synchronize()?;
        for _ in 0..2 {
            unsafe {
                checked(
                    launch(&request, &bindings, stream.cu_stream().cast()),
                    "warmup",
                )?;
            }
        }
        stream.synchronize()?;
        start.record(&stream)?;
        let host_started = Instant::now();
        for _ in 0..iterations {
            unsafe {
                checked(
                    launch(&request, &bindings, stream.cu_stream().cast()),
                    "timed launch",
                )?;
            }
        }
        let host_launch_seconds = host_started.elapsed().as_secs_f64();
        stop.record(&stream)?;
        stop.synchronize()?;
        let elapsed_ms = start.elapsed_ms(&stop)?;
        Ok((elapsed_ms, host_launch_seconds))
    })();
    // Native errors can follow enqueue. Always drain BEFORE releasing payloads.
    // If completion is unknown, retain owners (and the outer library on error).
    if let Err(error) = stream.synchronize() {
        std::mem::forget((query, key, value, output, stream));
        return Err(format!("completion unknown; retained all device owners: {error}").into());
    }
    let (elapsed_ms, host_launch_seconds) = execution?;
    let actual = stream.clone_dtoh(&output)?;
    let input_unchanged = unchanged(&stream.clone_dtoh(&query)?, &host_q)
        && unchanged(&stream.clone_dtoh(&key)?, &host_k)
        && unchanged(&stream.clone_dtoh(&value)?, &host_v);
    let canary_passed = canaries_ok(&actual);
    let accuracy = reference.compare(&actual[GUARD..GUARD + shape.elements()]);
    let valid_timing = elapsed_ms.is_finite() && elapsed_ms > 0.0;
    let passed = accuracy.passed && canary_passed && input_unchanged && valid_timing;
    Ok(json!({
        "shape":shape, "scale_f32_bits":fixture.scale.to_bits(), "native_info":{ "abi":info.abi,"size":info.size,"shared_bytes":info.shared_bytes,"threads":info.threads,"query_tile":info.query_tile,"key_tile":info.key_tile,"global_workspace_bytes":info.global_workspace_bytes },
        "query_classes_per_head":CLASSES, "oracle":"F64 dot/stable-softmax/weighted sum of actual F32 input and scale; exact repeated-Q reuse per batch/head/class",
        "oracle_cpu_seconds":oracle_seconds, "accuracy":accuracy,
        "input_payloads_and_canaries_unchanged":input_unchanged, "output_canaries_intact":canary_passed,
        "warmup_launches":2,"timed_launches":iterations,
        "device_stream_elapsed_ms":elapsed_ms,"device_stream_ms_per_launch":f64::from(elapsed_ms)/f64::from(iterations),
        "host_timed_launch_seconds":host_launch_seconds, "passed":passed
    }))
}

pub fn run(library_path: &Path, receipt_path: &Path, device: usize, iterations: u32) -> Result<()> {
    let library_sha = digest(&std::fs::read(library_path)?);
    let context = CudaContext::new(device)?;
    let device_name = context.name()?;
    let compute_capability = context.compute_capability()?;
    let library = unsafe { libloading::Library::new(library_path)? };
    let mut cases = Vec::new();
    let outcome = (|| -> Result<()> {
        let prepare =
            unsafe { *library.get::<Prepare>(b"ferrum_cutlass_fused_f32_attention_prepare_v1\0")? };
        let launch =
            unsafe { *library.get::<Launch>(b"ferrum_cutlass_fused_f32_attention_run_v1\0")? };
        for shape in [
            Shape {
                batches: 1,
                sequence: 4096,
                heads: 16,
                dimension: 72,
            },
            Shape {
                batches: 4,
                sequence: 261,
                heads: 32,
                dimension: 64,
            },
        ] {
            let result = qualify(&context, prepare, launch, shape, iterations)?;
            println!("{result}");
            let passed = result["passed"].as_bool() == Some(true);
            cases.push(result);
            if !passed {
                return Err("qualification numerical/canary/timing gate failed".into());
            }
        }
        if digest(&std::fs::read(library_path)?) != library_sha {
            return Err("library bytes changed during qualification".into());
        }
        Ok(())
    })();
    let receipt = json!({
        "schema_version":1,"purpose":"independent qualification only, not product/model acceptance",
        "profile":"F32 input/output; CUTLASS Sm80 OpMultiplyAddFastF32; FP32 accumulator/softmax; no strict-F32 bitwise claim",
        "mask":"full_bidirectional_all_valid","dropout":false,"bias":false,
        "upstream_revision":"cb4247394dd82148787aed73e5dc7cef33cbf862",
        "library":library_path,"library_sha256":library_sha,
        "adapter_sha256":digest(include_bytes!("../../native/cutlass-fused-f32-attention/boundary.cu")),
        "abi_sha256":digest(include_bytes!("../../native/cutlass-fused-f32-attention/abi.h")),
        "seed_sha256":digest(include_bytes!("../../native/cutlass-fused-f32-attention/source.seed.json")),
        "rust_fixture_sha256":digest(include_bytes!("fixture.rs")),
        "device":device,"device_name":device_name,"compute_capability":compute_capability,
        "timing_scope":"CUDA events on one stream, warmed launches only; includes possible enqueue gaps; excludes allocation/transfer/oracle; not E2E or kernel-sum",
        "strict_same_process_baseline":null,"cases":cases,
        "passed":outcome.is_ok(),"error":outcome.as_ref().err().map(ToString::to_string)
    });
    // On an error, including unknown completion, do not unload native code.
    if outcome.is_err() {
        std::mem::forget(library);
    }
    std::fs::write(receipt_path, serde_json::to_vec_pretty(&receipt)?)?;
    outcome
}
