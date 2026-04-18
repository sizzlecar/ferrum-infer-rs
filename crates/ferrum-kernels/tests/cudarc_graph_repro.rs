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
use cudarc::driver::{CudaContext, PushKernelArg};
use cudarc::nvrtc::compile_ptx;

// A trivial kernel — matches our touch_kernel from the C++ repros.
const TOUCH_PTX: &str = r#"
extern "C" __global__ void touch(float* buf, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) buf[i] = buf[i] + 1.0f;
}
"#;

fn run_graph_repro(disable_event_tracking: bool) {
    eprintln!(
        "[cudarc-repro] disable_event_tracking={disable_event_tracking}"
    );
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
    eprintln!(
        "[cudarc-repro] SUCCESS (disable_event_tracking={disable_event_tracking})"
    );
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
