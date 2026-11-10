use super::{fixture::*, Result};
use cudarc::cublas::{result::gemm_ex, sys as blas_sys, CudaBlas};
use cudarc::driver::{sys, CudaContext, CudaStream, DevicePtr, DevicePtrMut};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{ffi::c_void, path::Path, sync::Arc};

const GUARD: usize = 128;
const CANARY: u16 = 0x7fc1;
#[repr(C)]
#[derive(Clone, Copy)]
struct Bindings {
    activation: u64,
    weight: u64,
    row_scale: u64,
    channel_scale: u64,
    output: u64,
}
#[repr(C)]
#[derive(Default)]
struct Info {
    workspace_bytes: u64,
    tile_m: u32,
    tile_n: u32,
    tile_k: u32,
    threads: u32,
}
type Prepare = unsafe extern "C" fn(*const Shape, *mut Info) -> i32;
type Launch = unsafe extern "C" fn(*const Shape, *const Bindings, *mut c_void) -> i32;
type Epilogue = unsafe extern "C" fn(*const Shape, *const Bindings, u64, *mut c_void) -> i32;
type Fill = unsafe extern "C" fn(u64, u64, u16, *mut c_void) -> i32;
type Compare = unsafe extern "C" fn(u64, u64, u64, u64, *mut c_void) -> i32;
struct Native {
    prepare: Prepare,
    launch: Launch,
    epilogue: Epilogue,
    fill: Fill,
    compare: Compare,
}
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
fn baseline(blas: &CudaBlas, shape: Shape, b: &Bindings, acc: u64) -> Result<()> {
    let one = 1_i32;
    let zero = 0_i32;
    unsafe {
        gemm_ex(
            *blas.handle(),
            blas_sys::cublasOperation_t::CUBLAS_OP_T,
            blas_sys::cublasOperation_t::CUBLAS_OP_N,
            shape.outputs as i32,
            shape.rows as i32,
            shape.inputs as i32,
            (&one as *const i32).cast(),
            b.weight as *const c_void,
            blas_sys::cudaDataType_t::CUDA_R_8I,
            shape.inputs as i32,
            b.activation as *const c_void,
            blas_sys::cudaDataType_t::CUDA_R_8I,
            shape.inputs as i32,
            (&zero as *const i32).cast(),
            acc as *mut c_void,
            blas_sys::cudaDataType_t::CUDA_R_32I,
            shape.outputs as i32,
            blas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32I,
            blas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
        )?;
    }
    Ok(())
}
fn timed(
    context: &Arc<CudaContext>,
    stream: &Arc<CudaStream>,
    enqueue: impl FnOnce() -> Result<()>,
) -> Result<f32> {
    let start = context.new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    let stop = context.new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
    start.record(stream)?;
    enqueue()?;
    stop.record(stream)?;
    stop.synchronize()?;
    Ok(start.elapsed_ms(&stop)?)
}
fn qualify(
    context: &Arc<CudaContext>,
    native: &Native,
    shape: Shape,
    iterations: u32,
) -> Result<serde_json::Value> {
    let stream = context.new_stream()?;
    let blas = CudaBlas::new(Arc::clone(&stream))?;
    // Match the existing ConvRot GEMM's zero-workspace, deterministic settings.
    unsafe {
        let handle = *blas.handle();
        blas_sys::cublasSetWorkspace_v2(handle, std::ptr::null_mut(), 0).result()?;
        blas_sys::cublasSetPointerMode_v2(
            handle,
            blas_sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST,
        )
        .result()?;
        blas_sys::cublasSetMathMode(handle, blas_sys::cublasMath_t::CUBLAS_DEFAULT_MATH)
            .result()?;
        blas_sys::cublasSetAtomicsMode(
            handle,
            blas_sys::cublasAtomicsMode_t::CUBLAS_ATOMICS_NOT_ALLOWED,
        )
        .result()?;
    }
    let m = shape.rows as usize;
    let n = shape.outputs as usize;
    let k = shape.inputs as usize;
    let elements = shape.output_elements();
    let a = stream.clone_htod(&(0..m * k).map(activation).collect::<Vec<_>>())?;
    let w = stream.clone_htod(&(0..n * k).map(weight).collect::<Vec<_>>())?;
    let rs = stream.clone_htod(&(0..m).map(row_scale).collect::<Vec<_>>())?;
    let cs = stream.clone_htod(&(0..n).map(column_scale).collect::<Vec<_>>())?;
    let mut expected = stream.alloc_zeros::<u16>(elements + 2 * GUARD)?;
    let mut actual = stream.alloc_zeros::<u16>(elements + 2 * GUARD)?;
    let mut comparison = stream.alloc_zeros::<u64>(3)?;
    stream.synchronize()?;
    let before_scratch = context.mem_get_info()?;
    let mut accumulator = stream.alloc_zeros::<i32>(elements)?;
    stream.synchronize()?;
    let after_scratch = context.mem_get_info()?;
    let mut info = Info::default();
    unsafe { checked((native.prepare)(&shape, &mut info), "prepare")? };
    if info.workspace_bytes != 0 {
        return Err("prototype requires zero candidate workspace".into());
    }
    let execution = (|| -> Result<_> {
        let (ap, _ag) = a.device_ptr(&stream);
        let (wp, _wg) = w.device_ptr(&stream);
        let (rp, _rg) = rs.device_ptr(&stream);
        let (cp, _cg) = cs.device_ptr(&stream);
        let (ep, _eg) = expected.device_ptr_mut(&stream);
        let (op, _og) = actual.device_ptr_mut(&stream);
        let (ip, _ig) = accumulator.device_ptr_mut(&stream);
        let (compare_ptr, _compare_guard) = comparison.device_ptr_mut(&stream);
        let raw_stream = stream.cu_stream().cast();
        let expected_bindings = Bindings {
            activation: ap,
            weight: wp,
            row_scale: rp,
            channel_scale: cp,
            output: ep + (GUARD * 2) as u64,
        };
        let actual_bindings = Bindings {
            output: op + (GUARD * 2) as u64,
            ..expected_bindings
        };
        for ptr in [ep, op] {
            unsafe {
                checked(
                    (native.fill)(ptr, (elements + 2 * GUARD) as u64, CANARY, raw_stream),
                    "initialize output/canaries",
                )?;
            }
        }
        let reference_launch = || -> Result<()> {
            baseline(&blas, shape, &expected_bindings, ip)?;
            unsafe {
                checked(
                    (native.epilogue)(&shape, &expected_bindings, ip, raw_stream),
                    "reference epilogue",
                )
            }
        };
        let fused_launch = || -> Result<()> {
            unsafe {
                checked(
                    (native.launch)(&shape, &actual_bindings, raw_stream),
                    "fused output",
                )
            }
        };
        reference_launch()?;
        fused_launch()?;
        stream.synchronize()?;
        let mut baseline_ms = Vec::new();
        let mut candidate_ms = Vec::new();
        // Alternate order; allocation, transfer and correctness work stay outside events.
        for iteration in 0..iterations {
            if iteration % 2 == 0 {
                baseline_ms.push(timed(context, &stream, reference_launch)?);
                candidate_ms.push(timed(context, &stream, fused_launch)?);
            } else {
                candidate_ms.push(timed(context, &stream, fused_launch)?);
                baseline_ms.push(timed(context, &stream, reference_launch)?);
            }
        }
        unsafe {
            checked(
                (native.compare)(
                    expected_bindings.output,
                    actual_bindings.output,
                    elements as u64,
                    compare_ptr,
                    raw_stream,
                ),
                "full output comparison",
            )?;
        }
        Ok((baseline_ms, candidate_ms))
    })();
    if let Err(error) = stream.synchronize() {
        std::mem::forget((
            a,
            w,
            rs,
            cs,
            expected,
            actual,
            comparison,
            accumulator,
            blas,
            stream,
        ));
        return Err(format!("completion unknown; retained device owners: {error}").into());
    }
    let (baseline_ms, candidate_ms) = execution?;
    let counts = stream.clone_dtoh(&comparison)?;
    let max_abs = f32::from_bits(counts[2] as u32);
    let mut sample_results = Vec::new();
    let mut samples_passed = true;
    for (row, column) in samples(shape) {
        let index = row * n + column;
        let (dot, bf16) = reference(shape, row, column);
        let actual_dot = stream.clone_dtoh(&accumulator.slice(index..index + 1))?[0];
        let baseline_bits =
            stream.clone_dtoh(&expected.slice(GUARD + index..GUARD + index + 1))?[0];
        let fused_bits = stream.clone_dtoh(&actual.slice(GUARD + index..GUARD + index + 1))?[0];
        let passed = dot == actual_dot && bf16 == baseline_bits && bf16 == fused_bits;
        samples_passed &= passed;
        sample_results.push(json!({"row":row,"column":column,"cpu_i32":dot,
            "cublas_i32":actual_dot,"cpu_bf16_bits":bf16,"baseline_bits":baseline_bits,
            "fused_bits":fused_bits,"passed":passed}));
    }
    let mut canaries_intact = true;
    for output in [&expected, &actual] {
        for range in [0..GUARD, elements + GUARD..elements + 2 * GUARD] {
            canaries_intact &= stream
                .clone_dtoh(&output.slice(range))?
                .iter()
                .all(|&x| x == CANARY);
        }
    }
    let timing_valid = baseline_ms
        .iter()
        .chain(&candidate_ms)
        .all(|x| x.is_finite() && *x > 0.0);
    let mean =
        |values: &[f32]| values.iter().map(|x| f64::from(*x)).sum::<f64>() / values.len() as f64;
    let passed = counts[0] == 0
        && counts[1] == 0
        && max_abs == 0.0
        && samples_passed
        && canaries_intact
        && timing_valid;
    Ok(json!({"shape":shape,"compared_output_elements":elements,
        "accuracy":{"bitwise_mismatches":counts[0],"nonfinite":counts[1],
            "maximum_absolute_error":max_abs,"cpu_samples":sample_results,
            "samples_passed":samples_passed,"output_canaries_intact":canaries_intact},
        "baseline_ms":baseline_ms,"candidate_ms":candidate_ms,
        "baseline_mean_ms":mean(&baseline_ms),"candidate_mean_ms":mean(&candidate_ms),
        "baseline_over_candidate":mean(&baseline_ms)/mean(&candidate_ms),
        "scratch":{"baseline_i32_bytes":elements as u64*4,"candidate_queried_workspace_bytes":info.workspace_bytes,
            "cuda_memory_before_baseline_scratch":{"free":before_scratch.0,"total":before_scratch.1},
            "cuda_memory_after_baseline_scratch":{"free":after_scratch.0,"total":after_scratch.1},
            "observed_free_decrease":before_scratch.0.saturating_sub(after_scratch.0),
            "scope":"Only GEMM output scratch; A/W/scales and both guarded outputs coexist for comparison. No allocation occurs in the candidate launch; no process peak claim."},
        "native":{"tile":[info.tile_m,info.tile_n,info.tile_k],"threads":info.threads},
        "passed":passed}))
}

pub fn run(library_path: &Path, receipt_path: &Path, device: usize, iterations: u32) -> Result<()> {
    use std::io::Write;
    let mut receipt_file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(receipt_path)?;
    let library_sha = digest(&std::fs::read(library_path)?);
    let context = CudaContext::new(device)?;
    let library = unsafe { libloading::Library::new(library_path)? };
    let mut cases = Vec::new();
    let outcome = (|| -> Result<()> {
        let native = unsafe {
            Native {
                prepare: *library.get::<Prepare>(b"ferrum_int8_fused_output_prepare_v1\0")?,
                launch: *library.get::<Launch>(b"ferrum_int8_fused_output_run_v1\0")?,
                epilogue: *library
                    .get::<Epilogue>(b"ferrum_int8_output_reference_epilogue_v1\0")?,
                fill: *library.get::<Fill>(b"ferrum_int8_output_fill_v1\0")?,
                compare: *library.get::<Compare>(b"ferrum_int8_output_compare_v1\0")?,
            }
        };
        for rows in [6296, 19389] {
            let result = qualify(&context, &native, Shape::fc1(rows), iterations)?;
            let passed = result["passed"].as_bool() == Some(true);
            println!("{result}");
            cases.push(result);
            if !passed {
                return Err("INT8 output qualification failed".into());
            }
        }
        if digest(&std::fs::read(library_path)?) != library_sha {
            return Err("library changed during qualification".into());
        }
        Ok(())
    })();
    let receipt = json!({"schema_version":1,"purpose":"independent single-layer primitive qualification, no product routing or E2E claim",
        "threshold":{"bitwise_mismatches":0,"maximum_absolute_error":0,"nonfinite":0},
        "arithmetic":"Exact bounded I8 dot to I32; RNE F32 conversion; row scale then channel scale; final BF16 RNE. Qualification targets bitwise equality; failures are retained, not relabelled a different profile.",
        "data":"Deterministic coordinate-hashed full-range I8 activations and decoded weights; varying positive F32 row/column scales. Synthetic primitive fixture, not captured model activations.",
        "timing_scope":"One warmup per path; alternating CUDA-event samples; excludes allocation/H2D/D2H/oracle. Baseline original cuBLAS GEMM plus epilogue; candidate CUTLASS fused output. Existing ConvRot rotation, quantization and W4 decoding are unchanged and excluded from both.",
        "library":library_path,"library_sha256":library_sha,"iterations":iterations,
        "adapter_sha256":digest(include_bytes!("../../native/cutlass-int8-fused-output/boundary.cu")),
        "abi_sha256":digest(include_bytes!("../../native/cutlass-int8-fused-output/abi.h")),
        "seed_sha256":digest(include_bytes!("../../native/cutlass-int8-fused-output/source.seed.json")),
        "fixture_sha256":digest(include_bytes!("fixture.rs")),"device":device,"device_name":context.name()?,
        "compute_capability":context.compute_capability()?,"cases":cases,"passed":outcome.is_ok(),
        "error":outcome.as_ref().err().map(ToString::to_string)});
    if outcome.is_err() {
        std::mem::forget(library);
    }
    receipt_file.write_all(&serde_json::to_vec_pretty(&receipt)?)?;
    receipt_file.sync_all()?;
    outcome
}
