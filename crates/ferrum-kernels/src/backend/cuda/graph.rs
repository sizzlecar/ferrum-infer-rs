//! `BackendGraph for CudaBackend` — CUDA graph capture / replay.
//!
//! Extracted from `cuda/mod.rs` (#8 Phase 2). Owns:
//!
//! - `impl BackendGraph for CudaBackend` — `set_decode_state` /
//!   `begin_graph_capture` / `end_graph_capture` / `replay_graph` /
//!   `reset_graph` / `reset_all_graphs`. The `set_decode_state` method
//!   writes to the process-global `DECODE_STATE` (defined in `mod.rs`
//!   because the core Backend ops also read it).
//! - `GraphSlotRaw` — raw `cuGraph` + `cuGraphExec` pointers, bypassing
//!   cudarc's `CudaGraph` wrapper (whose `end_capture` corrupts the
//!   context on Blackwell + CUDA 13).
//! - `DECODE_GRAPHS` multi-slot graph cache + `graph_slots` helper +
//!   `install_decode_graph_raw` + `with_decode_graph` + the `pub`
//!   eviction helpers `invalidate_decode_graph` / `invalidate_all_decode_graphs`.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use cudarc::driver::CudaStream;
use ferrum_bench_core::{global_profile, profile_fields_from_json};
use ferrum_types::{FerrumError, Result};

use super::{decode_state_slot_for_ordinal, CudaBackend};
use crate::backend::{Backend, BackendGraph};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CudaGraphRuntimeConfig {
    graph_prof: bool,
    skip_upload: bool,
    skip_sync: bool,
}

impl CudaGraphRuntimeConfig {
    fn from_env() -> Self {
        Self::from_env_vars(std::env::vars())
    }

    fn from_env_vars<I, K, V>(vars: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: AsRef<str>,
    {
        let mut config = Self {
            graph_prof: false,
            skip_upload: false,
            skip_sync: false,
        };
        for (name, value) in vars {
            let value = value.as_ref();
            match name.as_ref() {
                "FERRUM_GRAPH_PROF" => config.graph_prof = true,
                "FERRUM_GRAPH_SKIP_UPLOAD" => config.skip_upload = value == "1",
                "FERRUM_GRAPH_SKIP_SYNC" => config.skip_sync = value == "1",
                _ => {}
            }
        }
        config
    }
}

fn cuda_graph_runtime_config() -> &'static CudaGraphRuntimeConfig {
    static CONFIG: OnceLock<CudaGraphRuntimeConfig> = OnceLock::new();
    CONFIG.get_or_init(CudaGraphRuntimeConfig::from_env)
}

fn captured_graph_summary(cu_graph: cudarc::driver::sys::CUgraph) -> String {
    use cudarc::driver::sys;

    let mut parts = Vec::new();
    let mut node_count = 0usize;
    let nodes_status =
        unsafe { sys::cuGraphGetNodes(cu_graph, std::ptr::null_mut(), &mut node_count) };
    if nodes_status == sys::CUresult::CUDA_SUCCESS {
        parts.push(format!("nodes={node_count}"));
        if node_count > 0 {
            let mut nodes = vec![std::ptr::null_mut(); node_count];
            let mut filled = node_count;
            let fill_status =
                unsafe { sys::cuGraphGetNodes(cu_graph, nodes.as_mut_ptr(), &mut filled) };
            if fill_status == sys::CUresult::CUDA_SUCCESS {
                let mut by_type = std::collections::BTreeMap::<String, usize>::new();
                let mut type_errors = 0usize;
                for &node in nodes.iter().take(filled) {
                    let mut ty = std::mem::MaybeUninit::<sys::CUgraphNodeType>::uninit();
                    let ty_status = unsafe { sys::cuGraphNodeGetType(node, ty.as_mut_ptr()) };
                    if ty_status == sys::CUresult::CUDA_SUCCESS {
                        let ty = unsafe { ty.assume_init() };
                        *by_type.entry(format!("{ty:?}")).or_default() += 1;
                    } else {
                        type_errors += 1;
                    }
                }
                if !by_type.is_empty() {
                    let counts = by_type
                        .into_iter()
                        .map(|(ty, count)| format!("{ty}:{count}"))
                        .collect::<Vec<_>>()
                        .join(",");
                    parts.push(format!("node_types={counts}"));
                }
                if type_errors > 0 {
                    parts.push(format!("node_type_errors={type_errors}"));
                }
                if filled != node_count {
                    parts.push(format!("nodes_filled={filled}"));
                }
            } else {
                parts.push(format!("nodes_fill_status={fill_status:?}"));
            }
        }
    } else {
        parts.push(format!("nodes_status={nodes_status:?}"));
    }

    let mut root_count = 0usize;
    let roots_status =
        unsafe { sys::cuGraphGetRootNodes(cu_graph, std::ptr::null_mut(), &mut root_count) };
    if roots_status == sys::CUresult::CUDA_SUCCESS {
        parts.push(format!("roots={root_count}"));
    } else {
        parts.push(format!("roots_status={roots_status:?}"));
    }

    match cudarc::driver::result::mem_get_info() {
        Ok((free, total)) => parts.push(format!(
            "mem_free_mib={} mem_total_mib={}",
            free / (1024 * 1024),
            total / (1024 * 1024)
        )),
        Err(err) => parts.push(format!("mem_get_info={err:?}")),
    }
    parts.join(" ")
}

impl BackendGraph for CudaBackend {
    fn set_decode_state(ctx: &mut Self::Context, token: u32, step: u32) {
        let valid_kv = (step as i32) + 1;
        let step_i = step as i32;
        let stream = ctx.stream.clone();
        let mut w = decode_state_slot_for_ordinal(ctx.ordinal)
            .write()
            .expect("DECODE_STATE poisoned");
        let bufs = w.as_mut().expect("DecodeStateBufs not initialised");
        stream
            .memcpy_htod(&[token], &mut bufs.token)
            .expect("token_buf memcpy");
        stream
            .memcpy_htod(&[step_i], &mut bufs.pos)
            .expect("pos_buf memcpy");
        stream
            .memcpy_htod(&[valid_kv], &mut bufs.kv)
            .expect("kv_buf memcpy");
    }

    fn set_dev_state_mode(ctx: &mut Self::Context, enable: bool) {
        ctx.use_dev_state = enable;
    }

    fn begin_graph_capture(ctx: &mut Self::Context) -> Result<()> {
        use cudarc::driver::sys::CUstreamCaptureMode;
        // Event tracking already disabled globally in default_stream; begin
        // capture directly in relaxed mode. Bare-Rust cudarc reproducer
        // confirms this configuration works on Blackwell + CUDA 13
        // (`cudarc_graph_no_event_tracking` test). The full ferrum bench
        // path still SIGSEGVs though — remaining delta is likely one of
        // PTX module load timing, cuBLAS workspace interaction, or a
        // specific kernel's use of constant memory that doesn't survive
        // capture. See `docs/phase-e-cuda-status.md` graph section.
        ctx.stream
            .begin_capture(CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED)
            .map_err(|e| FerrumError::unsupported(format!("begin_capture: {e}")))?;
        ctx.capture_in_flight = true;
        Ok(())
    }

    fn end_graph_capture(ctx: &mut Self::Context, key: u64) -> Result<()> {
        use cudarc::driver::sys;
        if !ctx.capture_in_flight {
            return Err(FerrumError::unsupported("end_capture without begin"));
        }
        ctx.capture_in_flight = false;

        // Bypass cudarc's end_capture — it does cuStreamEndCapture +
        // cuGraphInstantiateWithFlags in one call, and one of those corrupts
        // the context on Blackwell. Call them separately so we can see which.
        ctx.ctx
            .bind_to_thread()
            .map_err(|e| FerrumError::unsupported(format!("bind pre-end: {e}")))?;

        let cu_stream = ctx.stream.cu_stream();
        let mut cu_graph: sys::CUgraph = std::ptr::null_mut();
        let st1 = unsafe { sys::cuStreamEndCapture(cu_stream, &mut cu_graph) };
        if st1 != sys::CUresult::CUDA_SUCCESS {
            return Err(FerrumError::unsupported(format!(
                "cuStreamEndCapture failed: {st1:?}"
            )));
        }
        if cu_graph.is_null() {
            return Err(FerrumError::unsupported(
                "cuStreamEndCapture returned null graph",
            ));
        }

        // flags=0: no AUTO_FREE_ON_LAUNCH. The captured graph contains
        // only kernel launches (memcpys are sync via cuMemcpyHtoD_v2
        // outside capture, see populate_batched_pointers), so
        // AUTO_FREE has nothing to free. With AUTO_FREE on, replays
        // worked for ~14 iters then SIGSEGV — likely device-side
        // launch resources getting freed mid-launch sequence.
        let flags = 0u64;
        let mut cu_graph_exec: sys::CUgraphExec = std::ptr::null_mut();
        let st2 = unsafe { sys::cuGraphInstantiateWithFlags(&mut cu_graph_exec, cu_graph, flags) };
        if st2 != sys::CUresult::CUDA_SUCCESS {
            let graph_summary = captured_graph_summary(cu_graph);
            unsafe {
                sys::cuGraphDestroy(cu_graph);
            }
            return Err(FerrumError::unsupported(format!(
                "cuGraphInstantiate failed: {st2:?} key={key} {graph_summary}"
            )));
        }

        // Upload graph to GPU before first launch. Without this, the first
        // cuGraphLaunch does lazy JIT + resource upload while the stream
        // still has pending ops — libcuda dereferences not-yet-uploaded
        // graph state and SIGSEGVs on Blackwell + CUDA 13.
        let st3 = unsafe { sys::cuGraphUpload(cu_graph_exec, cu_stream) };
        if st3 != sys::CUresult::CUDA_SUCCESS {
            let graph_summary = captured_graph_summary(cu_graph);
            unsafe {
                sys::cuGraphExecDestroy(cu_graph_exec);
                sys::cuGraphDestroy(cu_graph);
            }
            return Err(FerrumError::unsupported(format!(
                "cuGraphUpload failed: {st3:?} key={key} {graph_summary}"
            )));
        }

        // Install into the multi-slot cache keyed by `key`. Replaces any
        // existing graph for the same key; the old GraphSlotRaw drops
        // (cuCtxSync + cuGraphExecDestroy + cuGraphDestroy in its Drop
        // impl) before the new one takes its place.
        install_decode_graph_raw(key, cu_graph, cu_graph_exec, ctx.stream.clone());
        Ok(())
    }

    fn reset_graph(_ctx: &mut Self::Context, key: u64) {
        invalidate_decode_graph(key);
    }

    fn reset_all_graphs(_ctx: &mut Self::Context) {
        invalidate_all_decode_graphs();
    }

    fn replay_graph(ctx: &mut Self::Context, key: u64) -> Result<bool> {
        use cudarc::driver::sys;
        let cu_stream = ctx.stream.cu_stream();
        ctx.ctx
            .bind_to_thread()
            .map_err(|e| FerrumError::unsupported(format!("bind pre-replay: {e}")))?;
        with_decode_graph(key, |g_opt| {
            if let Some(g) = g_opt {
                let runtime_config = cuda_graph_runtime_config();
                let prof = runtime_config.graph_prof;
                let t_pre = if prof {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                // Re-upload before each launch. Without it, c=4 throughput
                // drops 257→178 tok/s (post-Phase-8 measurement). The
                // graph instantiate-then-upload-once design didn't pan out
                // empirically; keep the per-replay upload until we
                // understand why removing it slows things down.
                if !runtime_config.skip_upload {
                    let st_up = unsafe { sys::cuGraphUpload(g.cu_graph_exec, cu_stream) };
                    if st_up != sys::CUresult::CUDA_SUCCESS {
                        return Err(FerrumError::unsupported(format!(
                            "cuGraphUpload: {st_up:?}"
                        )));
                    }
                }
                let t_after_upload = if prof {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                let st = unsafe { sys::cuGraphLaunch(g.cu_graph_exec, cu_stream) };
                if st != sys::CUresult::CUDA_SUCCESS {
                    return Err(FerrumError::unsupported(format!("cuGraphLaunch: {st:?}")));
                }
                let t_after_launch = if prof {
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                if !runtime_config.skip_sync {
                    let st_sync = unsafe { sys::cuStreamSynchronize(cu_stream) };
                    if st_sync != sys::CUresult::CUDA_SUCCESS {
                        return Err(FerrumError::unsupported(format!(
                            "post-launch sync: {st_sync:?}"
                        )));
                    }
                }
                if let (Some(t0), Some(t1), Some(t2)) = (t_pre, t_after_upload, t_after_launch) {
                    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if n.is_multiple_of(64) {
                        let upload = t1.duration_since(t0).as_micros();
                        let launch = t2.duration_since(t1).as_micros();
                        let sync = t2.elapsed().as_micros();
                        eprintln!(
                            "[graph-prof] call#{n} upload={upload}us launch={launch}us sync={sync}us total={}us",
                            t0.elapsed().as_micros()
                        );
                        let profile = global_profile();
                        if profile.is_enabled() {
                            let _ = profile.push_event(
                                "graph_prof",
                                profile_fields_from_json(serde_json::json!({
                                    "call": n,
                                })),
                                profile_fields_from_json(serde_json::json!({
                                    "upload": upload,
                                    "launch": launch,
                                    "sync": sync,
                                    "total": t0.elapsed().as_micros(),
                                })),
                                true,
                            );
                        }
                    }
                }
                Ok(true)
            } else {
                Ok(false)
            }
        })
    }
}

// ────────────────────────────────────────────────────────────────────────
// Process-global decode graph slot
// ────────────────────────────────────────────────────────────────────────
//
// Stored here (not on `CudaState`) because:
//   - Backend::Context isn't Send+Sync for all backends (Metal holds a
//     raw CommandBufferRef) — the model struct gets Send issues if ctx
//     is stored on it.
//   - Only CUDA uses graph capture, so global-per-process is fine.
//   - Kernel arg pointers captured in the graph reference model-owned
//     scratch buffers; the model outlives any graph, so no dangling refs.
//
// `CudaGraph` isn't automatically `Send+Sync` in cudarc's public API —
// we wrap in our own marker struct with `unsafe impl`. The stream itself
// is single-threaded per model (engine serialises via Mutex), so graph
// launch from the same thread is safe.

/// Raw graph slot holding cuGraph + cuGraphExec pointers directly, bypassing
/// cudarc's CudaGraph wrapper. The wrapper's `end_capture` does
/// cuStreamEndCapture + cuGraphInstantiateWithFlags in one non-overridable
/// call, and one of those corrupts the context on Blackwell; bypassing lets
/// us split the FFI calls and choose which instantiate variant to use.
struct GraphSlotRaw {
    cu_graph: cudarc::driver::sys::CUgraph,
    cu_graph_exec: cudarc::driver::sys::CUgraphExec,
    // Keep the stream Arc alive so its underlying cu_stream stays valid.
    _stream: std::sync::Arc<cudarc::driver::CudaStream>,
}

impl Drop for GraphSlotRaw {
    fn drop(&mut self) {
        use cudarc::driver::sys;
        unsafe {
            // Sync device before destroying graph resources to ensure no
            // kernel launches from this graph are still in flight.
            sys::cuCtxSynchronize();
            if !self.cu_graph_exec.is_null() {
                sys::cuGraphExecDestroy(self.cu_graph_exec);
            }
            if !self.cu_graph.is_null() {
                sys::cuGraphDestroy(self.cu_graph);
            }
            // Sync again after destroy so any cleanup completes.
            sys::cuCtxSynchronize();
        }
    }
}

// SAFETY: graph launch is serialised through the model's stream, which
// is accessed from one thread at a time (engine Mutex-wraps the model).
unsafe impl Send for GraphSlotRaw {}
unsafe impl Sync for GraphSlotRaw {}

// Multi-slot graph cache, keyed by an opaque `u64`. Caller chooses the
// key — the model uses `m_padded` (or 0 for single-item) so that
// different batch shapes get their own captured graph instead of
// thrashing a single slot at every shape change.
//
// Native CUDA microbench (graph_upload_bench.cu, 320 launches × 500 iters,
// alternating two graph sizes) confirmed multi-slot replay is stable
// at ~0.26ms/iter with no degradation vs single slot.
static DECODE_GRAPHS: std::sync::OnceLock<std::sync::RwLock<HashMap<u64, GraphSlotRaw>>> =
    std::sync::OnceLock::new();

fn graph_slots() -> &'static std::sync::RwLock<HashMap<u64, GraphSlotRaw>> {
    DECODE_GRAPHS.get_or_init(|| std::sync::RwLock::new(HashMap::new()))
}

fn install_decode_graph_raw(
    key: u64,
    cu_graph: cudarc::driver::sys::CUgraph,
    cu_graph_exec: cudarc::driver::sys::CUgraphExec,
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
) {
    let mut g = graph_slots().write().expect("DECODE_GRAPHS poisoned");
    g.insert(
        key,
        GraphSlotRaw {
            cu_graph,
            cu_graph_exec,
            _stream: stream,
        },
    );
}

fn with_decode_graph<R>(key: u64, f: impl FnOnce(Option<&GraphSlotRaw>) -> Result<R>) -> Result<R> {
    let guard = graph_slots().read().expect("DECODE_GRAPHS poisoned");
    f(guard.get(&key))
}

/// Evict ONE cached graph — call when its kernel-arg pointers (KV cache,
/// scratch buffers) might be invalidated.
pub fn invalidate_decode_graph(key: u64) {
    graph_slots()
        .write()
        .expect("DECODE_GRAPHS poisoned")
        .remove(&key);
}

/// Evict ALL cached graphs — used by hard reset (model reload, scratch
/// realloc) when every captured pointer might be stale.
pub fn invalidate_all_decode_graphs() {
    graph_slots()
        .write()
        .expect("DECODE_GRAPHS poisoned")
        .clear();
}

#[cfg(test)]
mod tests {
    use super::CudaGraphRuntimeConfig;

    #[test]
    fn cuda_graph_runtime_config_parses_graph_knobs() {
        let config = CudaGraphRuntimeConfig::from_env_vars([
            ("FERRUM_GRAPH_PROF", "0"),
            ("FERRUM_GRAPH_SKIP_UPLOAD", "1"),
            ("FERRUM_GRAPH_SKIP_SYNC", "1"),
        ]);

        assert!(config.graph_prof);
        assert!(config.skip_upload);
        assert!(config.skip_sync);
    }

    #[test]
    fn cuda_graph_runtime_config_keeps_default_replay_path() {
        let config = CudaGraphRuntimeConfig::from_env_vars([
            ("FERRUM_GRAPH_SKIP_UPLOAD", "true"),
            ("FERRUM_GRAPH_SKIP_SYNC", "0"),
        ]);

        assert!(!config.graph_prof);
        assert!(!config.skip_upload);
        assert!(!config.skip_sync);
    }
}
