//! Cross-backend GPU-side timer trait — PLAYBOOK § Phase 1.1.
//!
//! Replaces the `Instant::now()` calls inside `FERRUM_*_PROF` probes
//! (`crates/ferrum-models/src/moe/forward.rs`, `qwen3_moe.rs`, etc.).
//! Those measure CPU-side dispatch + queue depth — they DON'T see GPU
//! execution time, so the "per-op µs" they report has been misleading
//! all the perf debugging this session has built on top of.
//!
//! ## Backend behaviour
//!
//! - **CUDA** (`cudarc::driver::sys` events) — `cuEventRecord` is
//!   asynchronous on the stream; `elapsed_ms()` calls `cuEventSynchronize`
//!   + `cuEventElapsedTime`. Overhead per scope: ~5µs (event create +
//!   record × 2 + sync at read). Accuracy: ±0.5µs.
//!
//! - **Metal** — Metal's `MTLCommandBuffer` exposes `gpuStartTime` /
//!   `gpuEndTime` per command buffer. For sub-command-buffer scope we
//!   wrap the section in an explicit `sync()` boundary. This adds
//!   command-buffer commit overhead (~50-100µs) but gives accurate
//!   on-GPU timing. **Caveat**: on Metal the sync-wrap inflates each
//!   timed scope's CPU side; use sparingly.
//!
//! - **CPU** — `Instant`. (CPU is the "GPU" here — wall-clock is correct.)
//!
//! ## Usage
//!
//! ```ignore
//! use ferrum_kernels::backend::timer::BackendTimer;
//!
//! let mut timer = <B as Backend>::Timer::new();
//! timer.record_start(&mut ctx);
//! Backend::rms_norm(&mut ctx, &x, &w, eps, &mut out, tokens, dim);
//! timer.record_end(&mut ctx);
//! let us = timer.elapsed_ms() * 1000.0;
//! tracing::info!("rms_norm: {us:.1} us");
//! ```
//!
//! Hot loops should reuse a single `Timer` instance across scopes via
//! `record_start` / `record_end` — `new()` allocates events on CUDA.

use crate::backend::Backend;

/// Start a timer iff `enabled` is true — `None` is the disabled state.
/// Pair with [`finish_probe_timer`] at the end of the scope. The env/config
/// gate is intentionally resolved by the caller so hot probes do not read
/// process env while a token/layer loop is running.
pub fn start_probe_timer_if<B: Backend>(enabled: bool, ctx: &mut B::Context) -> Option<B::Timer> {
    if enabled {
        let mut t = B::make_timer();
        t.record_start(ctx);
        Some(t)
    } else {
        None
    }
}

/// Close a timer started by [`start_probe_timer`] and return the
/// elapsed microseconds. `None` propagates the "disabled" state so the
/// caller can keep the `if let Some(us) = ... { record(us) }` pattern.
pub fn finish_probe_timer<B: Backend>(
    timer: Option<B::Timer>,
    ctx: &mut B::Context,
) -> Option<u64> {
    let mut t = timer?;
    t.record_end(ctx);
    Some((t.elapsed_ms() * 1000.0) as u64)
}

/// Convenience wrapper: close a timer AND push a chrome-trace event in
/// one call. When `FERRUM_TRACE_OUT` is unset, the trace push is a
/// no-op (cheap atomic check inside [`global_trace`]).
///
/// PLAYBOOK § 1.5 — Phase 4 `visualize_layerwise.py` reads chrome-trace
/// JSON populated by these probe sites.
pub fn finish_probe_timer_traced<B: Backend>(
    timer: Option<B::Timer>,
    ctx: &mut B::Context,
    name: &str,
    cat: &str,
    tid: u32,
) -> Option<u64> {
    let us = finish_probe_timer::<B>(timer, ctx)?;
    ferrum_bench_core::trace::global_trace().push(name, cat, (us as f64) / 1000.0, tid);
    Some(us)
}

/// GPU-side timer scoped to a single Backend context.
pub trait BackendTimer<B: Backend>: Send {
    /// Allocate timer state. On CUDA this creates two `cuEvent_t`
    /// handles; on Metal it's a no-op; on CPU it's two `Option<Instant>`.
    fn new() -> Self
    where
        Self: Sized;

    /// Record the "start" timestamp on the current ctx's stream/command
    /// buffer. Returns immediately on CUDA (async); on Metal forces a
    /// sync to flush any pending work first.
    fn record_start(&mut self, ctx: &mut B::Context);

    /// Record the "end" timestamp.
    fn record_end(&mut self, ctx: &mut B::Context);

    /// Synchronize on the recorded events and return the elapsed time
    /// in milliseconds. Blocks the calling thread on CUDA; instant on CPU/Metal.
    ///
    /// Calling `elapsed_ms` before both record_start + record_end have
    /// fired returns `0.0`.
    fn elapsed_ms(&self) -> f64;
}

// ─────────────────────────────────────────────────────────────────────
// CPU implementation
// ─────────────────────────────────────────────────────────────────────

/// CPU timer — wall-clock via `Instant`. There's no GPU to wait on,
/// so the "GPU time" is just the CPU work duration.
pub struct CpuTimer {
    start: Option<std::time::Instant>,
    end: Option<std::time::Instant>,
}

impl Default for CpuTimer {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuTimer {
    pub fn new() -> Self {
        Self {
            start: None,
            end: None,
        }
    }
}

impl BackendTimer<crate::backend::cpu::CpuBackend> for CpuTimer {
    fn new() -> Self {
        CpuTimer::new()
    }

    fn record_start(&mut self, _ctx: &mut <crate::backend::cpu::CpuBackend as Backend>::Context) {
        self.start = Some(std::time::Instant::now());
    }

    fn record_end(&mut self, _ctx: &mut <crate::backend::cpu::CpuBackend as Backend>::Context) {
        self.end = Some(std::time::Instant::now());
    }

    fn elapsed_ms(&self) -> f64 {
        match (self.start, self.end) {
            (Some(s), Some(e)) => e.duration_since(s).as_secs_f64() * 1000.0,
            _ => 0.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Metal implementation
// ─────────────────────────────────────────────────────────────────────
//
// Metal exposes per-command-buffer `gpuStartTime`/`gpuEndTime`. To time
// a sub-CB scope we explicitly `sync()` to commit the current CB,
// record `Instant::now()` between syncs. The sync forces a flush+wait
// so the wall-clock delta IS the GPU time — same property as CUDA events,
// just paid with extra commits.
//
// Future improvement: use `MTLCounterSampleBuffer` with the timestamp
// counter set when the backend's pipeline state is configured to sample
// counters — would avoid the sync overhead. Not yet wired.

#[cfg(all(target_os = "macos", feature = "metal"))]
pub struct MetalTimer {
    start: Option<std::time::Instant>,
    end: Option<std::time::Instant>,
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl Default for MetalTimer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl MetalTimer {
    pub fn new() -> Self {
        Self {
            start: None,
            end: None,
        }
    }
}

#[cfg(all(target_os = "macos", feature = "metal"))]
impl BackendTimer<crate::backend::metal::MetalBackend> for MetalTimer {
    fn new() -> Self {
        MetalTimer::new()
    }

    fn record_start(
        &mut self,
        ctx: &mut <crate::backend::metal::MetalBackend as Backend>::Context,
    ) {
        // Force any pending work to drain before we anchor the clock.
        crate::backend::metal::MetalBackend::sync(ctx);
        self.start = Some(std::time::Instant::now());
    }

    fn record_end(&mut self, ctx: &mut <crate::backend::metal::MetalBackend as Backend>::Context) {
        // Sync so the wall-clock delta is bounded by actual GPU completion.
        crate::backend::metal::MetalBackend::sync(ctx);
        self.end = Some(std::time::Instant::now());
    }

    fn elapsed_ms(&self) -> f64 {
        match (self.start, self.end) {
            (Some(s), Some(e)) => e.duration_since(s).as_secs_f64() * 1000.0,
            _ => 0.0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// CUDA implementation
// ─────────────────────────────────────────────────────────────────────
//
// Uses cudarc's raw event API — same one quant.rs:889 already uses for
// Marlin split-K cross-stream coordination.

#[cfg(feature = "cuda")]
pub struct CudaTimer {
    start: Option<cudarc::driver::sys::CUevent>,
    end: Option<cudarc::driver::sys::CUevent>,
    recorded_start: bool,
    recorded_end: bool,
}

// `CUevent` is `*mut c_void` (a CUDA driver handle). The CUDA driver API
// is documented as thread-safe for these handles; the trait bound
// `BackendTimer<B>: Send` requires it.
#[cfg(feature = "cuda")]
unsafe impl Send for CudaTimer {}

#[cfg(feature = "cuda")]
impl Default for CudaTimer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "cuda")]
impl Drop for CudaTimer {
    fn drop(&mut self) {
        // Best-effort destroy — ignore errors during drop.
        use cudarc::driver::sys as cu;
        unsafe {
            if let Some(e) = self.start.take() {
                let _ = cu::cuEventDestroy_v2(e);
            }
            if let Some(e) = self.end.take() {
                let _ = cu::cuEventDestroy_v2(e);
            }
        }
    }
}

#[cfg(feature = "cuda")]
impl CudaTimer {
    pub fn new() -> Self {
        use cudarc::driver::sys as cu;
        let mut start: cu::CUevent = std::ptr::null_mut();
        let mut end: cu::CUevent = std::ptr::null_mut();
        unsafe {
            // Flag 0 = default (with timing). We don't disable timing
            // (cuEventDisableTiming) since elapsed_ms needs it.
            let _ = cu::cuEventCreate(&mut start, 0);
            let _ = cu::cuEventCreate(&mut end, 0);
        }
        Self {
            start: Some(start),
            end: Some(end),
            recorded_start: false,
            recorded_end: false,
        }
    }
}

#[cfg(feature = "cuda")]
impl BackendTimer<crate::backend::cuda::CudaBackend> for CudaTimer {
    fn new() -> Self {
        CudaTimer::new()
    }

    fn record_start(&mut self, ctx: &mut <crate::backend::cuda::CudaBackend as Backend>::Context) {
        use cudarc::driver::sys as cu;
        if let Some(evt) = self.start {
            unsafe {
                let _ = cu::cuEventRecord(evt, ctx.stream.cu_stream());
            }
            self.recorded_start = true;
        }
    }

    fn record_end(&mut self, ctx: &mut <crate::backend::cuda::CudaBackend as Backend>::Context) {
        use cudarc::driver::sys as cu;
        if let Some(evt) = self.end {
            unsafe {
                let _ = cu::cuEventRecord(evt, ctx.stream.cu_stream());
            }
            self.recorded_end = true;
        }
    }

    fn elapsed_ms(&self) -> f64 {
        if !self.recorded_start || !self.recorded_end {
            return 0.0;
        }
        use cudarc::driver::sys as cu;
        let (Some(s), Some(e)) = (self.start, self.end) else {
            return 0.0;
        };
        unsafe {
            // cuEventSynchronize blocks until the event is observed
            // on the stream. Required before reading elapsed.
            let _ = cu::cuEventSynchronize(e);
            // Safe wrapper dispatches cuEventElapsedTime vs _v2 based on
            // cudarc's auto-detected CUDA version (cuda-version-from-
            // build-system feature). The two are only visible under
            // CUDA 12.x and 13.x respectively — calling either directly
            // is non-portable.
            cudarc::driver::result::event::elapsed(s, e)
                .ok()
                .map(|ms| ms as f64)
                .unwrap_or(0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_timer_basic() {
        let mut t = CpuTimer::new();
        assert_eq!(t.elapsed_ms(), 0.0);
        // Construct a CpuBackend Context — which is unit.
        let mut ctx: <crate::backend::cpu::CpuBackend as Backend>::Context = ();
        BackendTimer::<crate::backend::cpu::CpuBackend>::record_start(&mut t, &mut ctx);
        std::thread::sleep(std::time::Duration::from_millis(2));
        BackendTimer::<crate::backend::cpu::CpuBackend>::record_end(&mut t, &mut ctx);
        let ms = BackendTimer::<crate::backend::cpu::CpuBackend>::elapsed_ms(&t);
        assert!(ms >= 2.0 && ms < 50.0, "elapsed_ms = {ms}");
    }

    #[test]
    fn cpu_timer_returns_zero_if_unrecorded() {
        let t = CpuTimer::new();
        let ms = BackendTimer::<crate::backend::cpu::CpuBackend>::elapsed_ms(&t);
        assert_eq!(ms, 0.0);
    }
}
