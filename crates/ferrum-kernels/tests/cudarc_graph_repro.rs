//! Minimal Rust-cudarc graph reproducer, mirrors `scripts/graph_repro_v4.cu`
//! but goes through cudarc's CudaContext / CudaStream / alloc APIs instead
//! of raw driver FFI. Isolates cudarc as a variable.
//!
//! Two variants:
//! - `cudarc_graph_default_pool` — use cudarc defaults (pool alloc enabled)
//! - `cudarc_graph_no_event_tracking` — same + `disable_event_tracking()`
//!   immediately after ctx creation (matches our ferrum-kernels fix)
//!
//! If either SEGFAULTs at `cuGraphLaunch` on Blackwell + CUDA 13, cudarc's
//! default path is what's triggering the issue we see in ferrum. If both
//! pass, ferrum's extra layers (PTX module loading, cuBLAS handle,
//! set_decode_state memcpy, etc.) are the real trigger and need more bisection.

#![cfg(feature = "cuda")]

use cudarc::driver::sys::{self, CUgraphInstantiate_flags, CUstreamCaptureMode};
use cudarc::driver::{CudaContext, DevicePtr, DevicePtrMut, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

// A trivial kernel — matches our touch_kernel from the C++ repros.
const TOUCH_PTX: &str = r#"
extern "C" __global__ void touch(float* buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = buf[i] + 1.0f;
}
"#;

fn run_graph_repro(disable_event_tracking: bool) {
    eprintln!("[cudarc-repro] disable_event_tracking={disable_event_tracking}");
    let ctx = CudaContext::new(0).expect("CudaContext::new");
    if disable_event_tracking {
        unsafe {
            ctx.disable_event_tracking();
        }
    }
    let stream = ctx.new_stream().expect("new_stream");

    // Compile + load the touch kernel via NVRTC (mirrors cudarc's typical
    // PTX load path — our ferrum code does the same for its kernels).
    let ptx = compile_ptx(TOUCH_PTX).expect("compile_ptx");
    let module = ctx.load_module(ptx).expect("load_module");
    let touch = module.load_function("touch").expect("load_function");

    // Allocate on stream (pool-backed when has_async_alloc is true).
    let n: usize = 4096;
    let mut buf = stream.alloc_zeros::<f32>(n).expect("alloc_zeros");
    stream.synchronize().expect("sync before warmup");

    // Warm-up launch outside capture.
    let mut builder = stream.launch_builder(&touch);
    let n_i32 = n as i32;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (((n + 127) / 128) as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };
    builder.arg(&mut buf).arg(&n_i32);
    unsafe { builder.launch(cfg).expect("warmup launch") };
    stream.synchronize().expect("sync after warmup");

    eprintln!("[cudarc-repro] begin capture");
    stream
        .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        .expect("begin_capture");

    // Four kernels inside capture — matches our decode-step pattern.
    for _ in 0..4 {
        let mut builder = stream.launch_builder(&touch);
        builder.arg(&mut buf).arg(&n_i32);
        unsafe { builder.launch(cfg).expect("captured launch") };
    }

    let graph = stream
        .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
        .expect("end_capture")
        .expect("end_capture returned None");

    graph.upload().expect("graph upload");
    stream.synchronize().expect("sync after upload");

    for i in 0..6 {
        eprintln!("[cudarc-repro] cuGraphLaunch #{}", i + 1);
        // Raw cuGraphLaunch to match our ferrum replay_last_graph path.
        let st = unsafe { sys::cuGraphLaunch(graph.cu_graph_exec(), stream.cu_stream()) };
        assert_eq!(
            st,
            sys::CUresult::CUDA_SUCCESS,
            "cuGraphLaunch returned {st:?}"
        );
        stream.synchronize().expect("sync after graph launch");
    }
    eprintln!("[cudarc-repro] SUCCESS (disable_event_tracking={disable_event_tracking})");
}

#[test]
#[ignore = "requires CUDA hardware"]
fn cudarc_graph_default_pool() {
    run_graph_repro(false);
}

#[test]
#[ignore = "requires CUDA hardware"]
fn cudarc_graph_no_event_tracking() {
    run_graph_repro(true);
}

/// Mirrors ferrum's load pattern: allocate LOTS of buffers (like weights)
/// then try graph capture + replay. If this fails where the simpler test
/// passes, allocation count or memory pool state is the trigger.
#[test]
#[ignore = "requires CUDA hardware"]
fn cudarc_graph_with_many_allocs() {
    eprintln!("[cudarc-repro-heavy] init");
    let ctx = CudaContext::new(0).expect("CudaContext::new");
    unsafe {
        ctx.disable_event_tracking();
    }
    let stream = ctx.new_stream().expect("new_stream");

    let ptx = compile_ptx(TOUCH_PTX).expect("compile_ptx");
    let module = ctx.load_module(ptx).expect("load_module");
    let touch = module.load_function("touch").expect("load_function");

    // Allocate ~200 buffers (each f32 weights-sized) to mimic model load.
    let mut weight_bufs = Vec::with_capacity(200);
    for _ in 0..200 {
        let buf = stream
            .alloc_zeros::<f32>(1024 * 1024) // 4 MB each
            .expect("weight alloc");
        weight_bufs.push(buf);
    }
    eprintln!(
        "[cudarc-repro-heavy] {} weight bufs allocated",
        weight_bufs.len()
    );

    // Scratch buffer for graph work.
    let n: usize = 4096;
    let mut scratch = stream.alloc_zeros::<f32>(n).expect("scratch");
    stream.synchronize().expect("sync");

    let n_i32 = n as i32;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (((n + 127) / 128) as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };

    // Warm up.
    let mut builder = stream.launch_builder(&touch);
    builder.arg(&mut scratch).arg(&n_i32);
    unsafe { builder.launch(cfg).expect("warmup launch") };
    stream.synchronize().expect("sync");

    eprintln!("[cudarc-repro-heavy] begin capture");
    stream
        .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        .expect("begin_capture");

    for _ in 0..8 {
        let mut builder = stream.launch_builder(&touch);
        builder.arg(&mut scratch).arg(&n_i32);
        unsafe { builder.launch(cfg).expect("captured launch") };
    }
    let graph = stream
        .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
        .expect("end_capture")
        .expect("non-null graph");
    graph.upload().expect("upload");
    stream.synchronize().expect("sync");

    for i in 0..6 {
        eprintln!("[cudarc-repro-heavy] cuGraphLaunch #{}", i + 1);
        let st = unsafe { sys::cuGraphLaunch(graph.cu_graph_exec(), stream.cu_stream()) };
        assert_eq!(st, sys::CUresult::CUDA_SUCCESS, "cuGraphLaunch: {st:?}");
        stream.synchronize().expect("sync");
    }
    eprintln!("[cudarc-repro-heavy] SUCCESS (200 bufs + graph)");
}

/// Tests captured graph with INTERNAL `stream.memcpy_htod` from a stable
/// host Box array. Mirrors ferrum's batched_decode capture path:
/// `flash_attention_batched_per_cache` and `kv_cache_append_batched_per_cache`
/// each call `stream.memcpy_htod(host_slice, dev_buf)` for cache pointers
/// + lengths, INSIDE the captured region.
///
/// If post-capture replay works (we already verified) but a SECOND
/// cuGraphLaunch hangs, the captured async memcpy interaction with
/// repeat-launch is suspect. This isolates that.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA hardware"]
fn cudarc_graph_internal_memcpy() {
    eprintln!("[cudarc-internal-memcpy] init");
    let ctx = CudaContext::new(0).expect("CudaContext::new");
    unsafe {
        ctx.disable_event_tracking();
    }
    let stream = ctx.new_stream().expect("new_stream");

    let ptx = compile_ptx(TOUCH_PTX).expect("compile_ptx");
    let module = ctx.load_module(ptx).expect("load_module");
    let touch = module.load_function("touch").expect("load_function");

    let n: usize = 4096;
    let mut buf = stream.alloc_zeros::<f32>(n).expect("alloc");

    // Stable host pointer scratch on heap — mirrors ferrum's
    // `batched_host_cache_ptrs: Box<[u64; 64]>`.
    let host_ptrs: Box<[u64; 4]> = Box::new([1u64, 2, 3, 4]);
    let mut device_ptrs = stream.alloc_zeros::<u64>(4).expect("alloc ptrs");

    stream.synchronize().expect("warmup sync");

    let n_i32 = n as i32;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (((n + 127) / 128) as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };

    // Warm-up outside capture.
    let mut b = stream.launch_builder(&touch);
    b.arg(&mut buf).arg(&n_i32);
    unsafe { b.launch(cfg).expect("warmup") };
    stream.synchronize().expect("warmup sync");

    eprintln!("[cudarc-internal-memcpy] begin capture");
    stream
        .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        .expect("begin_capture");

    // INSIDE capture: async memcpy from stable host array to device.
    let host_slice: &[u64] = &host_ptrs[..4];
    stream
        .memcpy_htod(host_slice, &mut device_ptrs)
        .expect("captured memcpy");

    // INSIDE capture: kernel launch.
    let mut b = stream.launch_builder(&touch);
    b.arg(&mut buf).arg(&n_i32);
    unsafe { b.launch(cfg).expect("captured launch") };

    let graph = stream
        .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
        .expect("end_capture")
        .expect("non-null");
    graph.upload().expect("upload");
    stream.synchronize().expect("post-upload sync");

    for i in 0..6 {
        eprintln!("[cudarc-internal-memcpy] cuGraphLaunch #{}", i + 1);
        let st = unsafe { sys::cuGraphLaunch(graph.cu_graph_exec(), stream.cu_stream()) };
        assert_eq!(st, sys::CUresult::CUDA_SUCCESS, "launch: {st:?}");
        stream.synchronize().expect("sync");
    }
    eprintln!("[cudarc-internal-memcpy] SUCCESS");
}

/// Captures a graph with TWO `stream.memcpy_htod` calls — both reading
/// from the SAME stable host array but writing to different device
/// buffers. Mirrors ferrum's batched_decode pattern where 32 layers
/// each call kv_cache_append_batched + flash_attn_batched, which all
/// memcpy from the SAME `batched_host_*_ptrs` arrays in CudaState
/// (because each captured call OVERWRITES that single shared host
/// array with its own layer's data, then memcpy reads it).
///
/// On replay, all captured memcpys re-execute reading the same host
/// pointer; if the host array has only the LAST-written data, every
/// memcpy reads layer N's data — earlier kernels see wrong inputs.
///
/// If this test hangs or asserts, the multi-captured-memcpy-shared-host
/// pattern is the root of ferrum's Phase 4d hang.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA hardware"]
fn cudarc_graph_shared_host_array_multi_memcpy() {
    eprintln!("[shared-host-multi] init");
    let ctx = CudaContext::new(0).expect("CudaContext::new");
    unsafe {
        ctx.disable_event_tracking();
    }
    let stream = ctx.new_stream().expect("new_stream");

    let ptx = compile_ptx(TOUCH_PTX).expect("compile_ptx");
    let module = ctx.load_module(ptx).expect("load_module");
    let touch = module.load_function("touch").expect("load_function");

    let n: usize = 4096;
    let mut buf = stream.alloc_zeros::<f32>(n).expect("alloc");

    // ONE shared host array (mirrors ferrum's batched_host_cache_ptrs)
    let mut host_array: Box<[u64; 4]> = Box::new([100u64, 200, 300, 400]);
    // TWO device buffers
    let mut dev1 = stream.alloc_zeros::<u64>(4).expect("alloc dev1");
    let mut dev2 = stream.alloc_zeros::<u64>(4).expect("alloc dev2");

    stream.synchronize().expect("warmup sync");

    let n_i32 = n as i32;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (((n + 127) / 128) as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut b = stream.launch_builder(&touch);
    b.arg(&mut buf).arg(&n_i32);
    unsafe { b.launch(cfg).expect("warmup") };
    stream.synchronize().expect("warmup sync");

    eprintln!("[shared-host-multi] begin capture");
    stream
        .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        .expect("begin_capture");

    // Captured op #1: write [1,2,3,4] to host_array, memcpy to dev1
    *host_array = [1u64, 2, 3, 4];
    let host_slice: &[u64] = &host_array[..4];
    stream
        .memcpy_htod(host_slice, &mut dev1)
        .expect("captured memcpy 1");
    let mut b = stream.launch_builder(&touch);
    b.arg(&mut buf).arg(&n_i32);
    unsafe { b.launch(cfg).expect("captured launch 1") };

    // Captured op #2: OVERWRITE host_array with [5,6,7,8], memcpy to dev2
    // This mirrors ferrum's per-layer overwrite of batched_host_cache_ptrs
    *host_array = [5u64, 6, 7, 8];
    let host_slice: &[u64] = &host_array[..4];
    stream
        .memcpy_htod(host_slice, &mut dev2)
        .expect("captured memcpy 2");
    let mut b = stream.launch_builder(&touch);
    b.arg(&mut buf).arg(&n_i32);
    unsafe { b.launch(cfg).expect("captured launch 2") };

    let graph = stream
        .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
        .expect("end_capture")
        .expect("non-null");
    graph.upload().expect("upload");
    stream.synchronize().expect("post-upload sync");

    for i in 0..6 {
        eprintln!("[shared-host-multi] cuGraphLaunch #{}", i + 1);
        let st = unsafe { sys::cuGraphLaunch(graph.cu_graph_exec(), stream.cu_stream()) };
        assert_eq!(st, sys::CUresult::CUDA_SUCCESS, "launch: {st:?}");
        stream.synchronize().expect("sync");
    }

    // Verify dev1 vs dev2 actually got DIFFERENT data on replay
    let dev1_host: Vec<u64> = stream.memcpy_dtov(&dev1).expect("dtov dev1");
    let dev2_host: Vec<u64> = stream.memcpy_dtov(&dev2).expect("dtov dev2");
    eprintln!("[shared-host-multi] dev1 = {:?}", dev1_host);
    eprintln!("[shared-host-multi] dev2 = {:?}", dev2_host);
    eprintln!("[shared-host-multi] SUCCESS — ran without hang. Data correctness:");
    if dev1_host == [5u64, 6, 7, 8] && dev2_host == [5u64, 6, 7, 8] {
        eprintln!("  BUG CONFIRMED: both dev buffers read latest host value (race)");
    } else if dev1_host == [1u64, 2, 3, 4] && dev2_host == [5u64, 6, 7, 8] {
        eprintln!("  WORKING: each memcpy got its own captured snapshot");
    } else {
        eprintln!("  UNEXPECTED");
    }
}

/// Mirrors ferrum's path even more closely — adds:
/// - cuBLAS handle + 32 MB workspace + DEVICE pointer mode + sgemm inside capture
/// - multiple different PTX modules loaded dynamically
/// - set_decode_state-style memcpy_htod BEFORE graph launch
///
/// This is the final "if this passes, the trigger is in our kernel
/// dispatch / stream semantics / something we haven't imagined"
/// sanity check for cudarc.
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA hardware"]
fn cudarc_graph_like_ferrum() {
    use cudarc::cublas::{sys as blas_sys, CudaBlas};

    const ANOTHER_PTX: &str = r#"
extern "C" __global__ void square(float* buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = buf[i] * buf[i];
}
"#;

    eprintln!("[cudarc-ferrum-like] init");
    let ctx = CudaContext::new(0).expect("CudaContext::new");
    unsafe {
        ctx.disable_event_tracking();
    }
    let stream = ctx.new_stream().expect("new_stream");

    // Load TWO PTX modules — mirrors ferrum's multi-kernel setup.
    let touch_ptx = compile_ptx(TOUCH_PTX).expect("compile touch");
    let touch_mod = ctx.load_module(touch_ptx).expect("load touch");
    let touch_fn = touch_mod.load_function("touch").expect("touch fn");
    let square_ptx = compile_ptx(ANOTHER_PTX).expect("compile square");
    let square_mod = ctx.load_module(square_ptx).expect("load square");
    let square_fn = square_mod.load_function("square").expect("square fn");

    // cuBLAS handle + workspace + DEVICE pointer mode (matches our config).
    let blas = CudaBlas::new(stream.clone()).expect("CudaBlas::new");
    let alpha_f32 = stream.memcpy_stod(&[1.0f32]).expect("alpha");
    let beta_f32 = stream.memcpy_stod(&[0.0f32]).expect("beta");
    unsafe {
        blas_sys::cublasSetPointerMode_v2(
            *blas.handle(),
            blas_sys::cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE,
        );
    }

    // GEMM buffers.
    let m = 1;
    let n = 1024;
    let k = 1024;
    let d_a = stream.alloc_zeros::<f32>(m * k).expect("a");
    let d_b = stream.alloc_zeros::<f32>(k * n).expect("b");
    let mut d_c = stream.alloc_zeros::<f32>(m * n).expect("c");

    // Decode-state mimic: three tiny buffers we'll memcpy_htod into pre-replay.
    let mut state_token = stream.alloc_zeros::<u32>(1).expect("token");
    let mut state_pos = stream.alloc_zeros::<i32>(1).expect("pos");
    let mut state_kv = stream.alloc_zeros::<i32>(1).expect("kv");
    stream.synchronize().expect("sync");

    // Scratch for kernel launches.
    let scratch_n = 4096;
    let mut scratch = stream.alloc_zeros::<f32>(scratch_n).expect("scratch");
    let n_i32 = scratch_n as i32;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (((scratch_n + 127) / 128) as u32, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };

    // Warm-up outside capture.
    let mut b = stream.launch_builder(&touch_fn);
    b.arg(&mut scratch).arg(&n_i32);
    unsafe { b.launch(cfg).expect("warmup touch") };
    stream.synchronize().expect("sync");

    eprintln!("[cudarc-ferrum-like] begin capture");
    stream
        .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
        .expect("begin_capture");

    // Inside capture: mix kernels, modules, cuBLAS — like a real decode step.
    for _ in 0..3 {
        let mut b = stream.launch_builder(&touch_fn);
        b.arg(&mut scratch).arg(&n_i32);
        unsafe { b.launch(cfg).expect("cap touch") };
        let mut b = stream.launch_builder(&square_fn);
        b.arg(&mut scratch).arg(&n_i32);
        unsafe { b.launch(cfg).expect("cap square") };
        // cuBLAS sgemm (DEVICE alpha/beta)
        unsafe {
            let (a_ptr, _ga) = alpha_f32.device_ptr(&stream);
            let (b_ptr, _gb) = beta_f32.device_ptr(&stream);
            let (da_ptr, _gda) = d_a.device_ptr(&stream);
            let (db_ptr, _gdb) = d_b.device_ptr(&stream);
            let (dc_ptr, _gdc) = d_c.device_ptr_mut(&stream);
            blas_sys::cublasSgemm_v2(
                *blas.handle(),
                blas_sys::cublasOperation_t::CUBLAS_OP_N,
                blas_sys::cublasOperation_t::CUBLAS_OP_N,
                n as i32,
                m as i32,
                k as i32,
                a_ptr as *const f32,
                db_ptr as *const f32,
                n as i32,
                da_ptr as *const f32,
                k as i32,
                b_ptr as *const f32,
                dc_ptr as *mut f32,
                n as i32,
            );
        }
    }

    let graph = stream
        .end_capture(CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH)
        .expect("end_capture")
        .expect("non-null");
    graph.upload().expect("upload");
    stream.synchronize().expect("sync");

    for i in 0..6 {
        // set_decode_state-style pre-launch htod
        stream
            .memcpy_htod(&[i as u32], &mut state_token)
            .expect("htod token");
        stream
            .memcpy_htod(&[i as i32], &mut state_pos)
            .expect("htod pos");
        stream
            .memcpy_htod(&[(i + 1) as i32], &mut state_kv)
            .expect("htod kv");

        eprintln!("[cudarc-ferrum-like] cuGraphLaunch #{}", i + 1);
        let st = unsafe { sys::cuGraphLaunch(graph.cu_graph_exec(), stream.cu_stream()) };
        assert_eq!(st, sys::CUresult::CUDA_SUCCESS, "cuGraphLaunch: {st:?}");
        stream.synchronize().expect("sync");
    }
    eprintln!("[cudarc-ferrum-like] SUCCESS — cuBLAS + multi-module + htod pre-launch");
}
