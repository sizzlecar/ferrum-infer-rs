//! `BackendGraph for CudaBackend` — CUDA graph capture / replay.
//!
//! Extracted from `cuda/mod.rs` (#8 Phase 2). Owns:
//!
//! - `impl BackendGraph for CudaBackend` — `set_decode_state` /
//!   `begin_graph_capture` / `end_graph_capture` / `replay_graph` /
//!   `reset_graph` / `reset_all_graphs`. The `set_decode_state` method
//!   writes to the original context's decode state, also read by core ops.
//! - `GraphSlotRaw` — raw `cuGraph` + `cuGraphExec` pointers, bypassing
//!   cudarc's `CudaGraph` wrapper (whose `end_capture` corrupts the
//!   context on Blackwell + CUDA 13).
//! - The original context's multi-slot graph cache and exact in-flight leases,
//!   with instance-scoped eviction through the `BackendGraph` methods.

use std::sync::{
    atomic::{AtomicPtr, Ordering},
    Arc,
};

use ferrum_types::{FerrumError, Result};

use super::CudaBackend;
use crate::backend::{Backend, BackendGraph};

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
        let bufs = &mut ctx.decode_state;
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
        if ctx.completion_observed {
            ctx.graph_leases.retire_completed(&ctx.graph_cache)?;
        }
        // Event tracking is disabled on this original CUDA context; begin
        // capture directly in relaxed mode. Bare-Rust cudarc reproducer
        // confirms this configuration works on Blackwell + CUDA 13
        // (`cudarc_graph_no_event_tracking` test). The full ferrum bench
        // path may still fail — remaining delta is likely one of
        // PTX module load timing, cuBLAS workspace interaction, or a
        // specific kernel's use of constant memory that doesn't survive
        // capture.
        ctx.completion_observed = false;
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
            if !cu_graph.is_null() {
                // Even an unsuccessful driver call can return an owned handle.
                // Capture remains unresolved, so retain it without destroying.
                ctx.graph_leases
                    .retain(Arc::new(GraphSlotRaw::new(cu_graph, ctx.stream.clone())));
            }
            return Err(FerrumError::unsupported(format!(
                "cuStreamEndCapture failed: {st1:?}"
            )));
        }
        // A failed EndCapture is not proof that capture ended. Keep the
        // context unavailable for completion/reuse until the driver confirms
        // the transition; the frame owner retains it on failure.
        ctx.capture_in_flight = false;
        if cu_graph.is_null() {
            return Err(FerrumError::unsupported(
                "cuStreamEndCapture returned null graph",
            ));
        }

        // The original lease takes the actual captured handle before any
        // fallible instantiate/upload or cleanup. Even failed instantiation
        // leaves its original graph available for checked retirement.
        let graph = Arc::new(GraphSlotRaw::new(cu_graph, ctx.stream.clone()));
        ctx.graph_leases.retain(graph.clone());
        let flags = 0u64;
        let mut cu_graph_exec: sys::CUgraphExec = std::ptr::null_mut();
        let st2 = unsafe { sys::cuGraphInstantiateWithFlags(&mut cu_graph_exec, cu_graph, flags) };
        graph.cu_graph_exec.store(cu_graph_exec, Ordering::Relaxed);
        if st2 != sys::CUresult::CUDA_SUCCESS {
            let graph_summary = captured_graph_summary(cu_graph);
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
            return Err(FerrumError::unsupported(format!(
                "cuGraphUpload failed: {st3:?} key={key} {graph_summary}"
            )));
        }

        // Replace only this context's cache entry. A previously submitted
        // executable remains retained by its exact in-flight lease.
        if let Some(previous) = ctx.graph_cache.insert(key, graph) {
            ctx.graph_leases.retain(previous);
        }
        Ok(())
    }

    fn reset_graph(ctx: &mut Self::Context, key: u64) {
        if let Some(previous) = ctx.graph_cache.remove(&key) {
            ctx.graph_leases.retain(previous);
        }
    }

    fn abort_graph_capture(ctx: &mut Self::Context) -> Result<()> {
        use cudarc::driver::sys;
        if !ctx.capture_in_flight {
            return Ok(());
        }
        ctx.ctx.bind_to_thread().map_err(|error| {
            FerrumError::backend(format!("bind before capture cancellation: {error}"))
        })?;
        let mut graph = std::ptr::null_mut();
        let ended = unsafe { sys::cuStreamEndCapture(ctx.stream.cu_stream(), &mut graph) };
        if !graph.is_null() {
            // A successful capture prefix is deliberately never instantiated
            // or replayed. Even an error may return an owned handle: preserve
            // it until capture status and real completion permit cleanup.
            ctx.graph_leases
                .retain(Arc::new(GraphSlotRaw::new(graph, ctx.stream.clone())));
        }
        let mut status = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE;
        let queried = unsafe { sys::cuStreamIsCapturing(ctx.stream.cu_stream(), &mut status) };
        if queried != sys::CUresult::CUDA_SUCCESS
            || status != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
        {
            return Err(FerrumError::backend(format!(
                "CUDA capture cancellation remains unresolved: end={ended:?}, query={queried:?}, status={status:?}",
            )));
        }
        // EndCapture can report an invalidated capture while still ending it.
        // Only this real query proves that subsequent completion observation
        // is legal; cancellation alone does not prove stream quiescence.
        ctx.capture_in_flight = false;
        ctx.completion_observed = false;
        ctx.use_dev_state = false;
        if ended != sys::CUresult::CUDA_SUCCESS {
            return Err(FerrumError::backend(format!(
                "CUDA capture ended during cancellation with error: {ended:?}",
            )));
        }
        Ok(())
    }

    fn reset_all_graphs(ctx: &mut Self::Context) {
        for (_, previous) in ctx.graph_cache.drain() {
            ctx.graph_leases.retain(previous);
        }
    }

    fn retire_graphs_for_storage_mutation(ctx: &mut Self::Context) -> Result<()> {
        if !ctx.capture_in_flight && ctx.graph_cache.is_empty() && ctx.graph_leases.is_empty() {
            // No captured executable can reference this storage. Repeated
            // per-layer growth must not introduce repeated device waits.
            return Ok(());
        }
        Self::reset_all_graphs(ctx);
        // Failed cleanup remains in the original graph leases. In particular,
        // no captured buffer may be replaced on either terminal-cleanup Err
        // or unresolved execution Err.
        Self::observe_completion(ctx).into_result()?;
        debug_assert!(ctx.graph_cache.is_empty() && ctx.graph_leases.is_empty());
        Ok(())
    }

    fn replay_graph(ctx: &mut Self::Context, key: u64) -> Result<bool> {
        use cudarc::driver::sys;
        if ctx.completion_observed {
            ctx.graph_leases.retire_completed(&ctx.graph_cache)?;
        }
        let cu_stream = ctx.stream.cu_stream();
        ctx.ctx
            .bind_to_thread()
            .map_err(|e| FerrumError::unsupported(format!("bind pre-replay: {e}")))?;
        {
            if let Some(g) = ctx.graph_cache.get(&key).cloned() {
                // Cache eviction is not execution retirement. Retain the
                // selected object before either upload or launch can fail.
                ctx.graph_leases.retain(g.clone());
                let executable = g.cu_graph_exec.load(Ordering::Relaxed);
                debug_assert!(
                    !executable.is_null(),
                    "cached graph must have an executable"
                );
                ctx.completion_observed = false;
                let runtime_config = &ctx.configuration.cuda;
                // Re-upload before each launch. Without it, c=4 throughput
                // drops 257→178 tok/s (post-Phase-8 measurement). The
                // graph instantiate-then-upload-once design didn't pan out
                // empirically; keep the per-replay upload until we
                // understand why removing it slows things down.
                if !runtime_config.graph_skip_upload {
                    let st_up = unsafe { sys::cuGraphUpload(executable, cu_stream) };
                    if st_up != sys::CUresult::CUDA_SUCCESS {
                        return Err(FerrumError::unsupported(format!(
                            "cuGraphUpload: {st_up:?}"
                        )));
                    }
                }
                let st = unsafe { sys::cuGraphLaunch(executable, cu_stream) };
                if st != sys::CUresult::CUDA_SUCCESS {
                    return Err(FerrumError::unsupported(format!("cuGraphLaunch: {st:?}")));
                }
                if !runtime_config.graph_skip_sync {
                    let st_sync = unsafe { sys::cuStreamSynchronize(cu_stream) };
                    if st_sync != sys::CUresult::CUDA_SUCCESS {
                        return Err(FerrumError::unsupported(format!(
                            "post-launch sync: {st_sync:?}"
                        )));
                    }
                }
                Ok(true)
            } else {
                Ok(false)
            }
        }
    }
}

// ────────────────────────────────────────────────────────────────────────
// Instance-owned decode graph slot
// ────────────────────────────────────────────────────────────────────────
//
// The cache owns reusable executables. Each CudaState additionally retains the
// exact executable before upload/launch; removing a cache entry does not retire
// an in-flight use. Model-owned capture buffers still need their model/frame
// owner until that use has been observed terminal.
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
pub(super) struct GraphSlotRaw {
    cu_graph: AtomicPtr<cudarc::driver::sys::CUgraph_st>,
    cu_graph_exec: AtomicPtr<cudarc::driver::sys::CUgraphExec_st>,
    #[cfg(test)]
    fail_next_exec_destroy: std::sync::atomic::AtomicBool,
    #[cfg(test)]
    fail_next_graph_destroy: std::sync::atomic::AtomicBool,
    // Keep the stream Arc alive so its underlying cu_stream stays valid.
    _stream: std::sync::Arc<cudarc::driver::CudaStream>,
}

/// The usual one-graph decode needs no new heap allocation. Multiple distinct
/// graphs (for example separate layer/head captures) retain all exact owners.
#[derive(Default)]
pub(super) struct GraphLeases {
    first: Option<Arc<GraphSlotRaw>>,
    additional: Vec<Arc<GraphSlotRaw>>,
}

impl GraphLeases {
    fn retain(&mut self, graph: Arc<GraphSlotRaw>) {
        match &self.first {
            None => self.first = Some(graph),
            Some(first) if Arc::ptr_eq(first, &graph) => {}
            Some(_) => {
                if !self.additional.iter().any(|held| Arc::ptr_eq(held, &graph)) {
                    self.additional.push(graph);
                }
            }
        }
    }

    pub(super) fn is_empty(&self) -> bool {
        self.first.is_none() && self.additional.is_empty()
    }

    pub(super) fn retire_completed(
        &mut self,
        cache: &std::collections::HashMap<u64, Arc<GraphSlotRaw>>,
    ) -> Result<()> {
        let mut first_error = None;
        let mut release = |graph: &Arc<GraphSlotRaw>| {
            if cache.values().any(|cached| Arc::ptr_eq(cached, graph)) {
                return true;
            }
            match graph.try_destroy() {
                Ok(()) => true,
                Err(error) => {
                    first_error.get_or_insert(error);
                    false
                }
            }
        };
        if self.first.as_ref().is_some_and(&mut release) {
            self.first = None;
        }
        self.additional.retain(|graph| !release(graph));
        first_error.map_or(Ok(()), Err)
    }
}

impl GraphSlotRaw {
    fn new(graph: cudarc::driver::sys::CUgraph, stream: Arc<cudarc::driver::CudaStream>) -> Self {
        Self {
            cu_graph: AtomicPtr::new(graph),
            cu_graph_exec: AtomicPtr::new(std::ptr::null_mut()),
            _stream: stream,
            #[cfg(test)]
            fail_next_exec_destroy: std::sync::atomic::AtomicBool::new(false),
            #[cfg(test)]
            fail_next_graph_destroy: std::sync::atomic::AtomicBool::new(false),
        }
    }

    // Called only by the original CudaState, after actual stream completion.
    // Atomic storage permits mutation through an exact shared lease; it is not
    // authorization for concurrent replay/retirement. The context remains the
    // sole serialized execution owner.
    fn try_destroy(&self) -> Result<()> {
        use cudarc::driver::sys;
        self._stream.context().bind_to_thread().map_err(|error| {
            FerrumError::backend(format!(
                "CUDA terminal graph cleanup could not bind context: {error}"
            ))
        })?;
        let executable = self.cu_graph_exec.load(Ordering::Relaxed);
        if !executable.is_null() {
            #[cfg(test)]
            if self.fail_next_exec_destroy.swap(false, Ordering::Relaxed) {
                return Err(FerrumError::backend(
                    "CUDA terminal graph executable cleanup fault",
                ));
            }
            let status = unsafe { sys::cuGraphExecDestroy(executable) };
            if status != sys::CUresult::CUDA_SUCCESS {
                return Err(FerrumError::backend(format!(
                    "CUDA terminal graph executable cleanup failed: {status:?}"
                )));
            }
            self.cu_graph_exec
                .store(std::ptr::null_mut(), Ordering::Relaxed);
        }
        let graph = self.cu_graph.load(Ordering::Relaxed);
        if !graph.is_null() {
            #[cfg(test)]
            if self.fail_next_graph_destroy.swap(false, Ordering::Relaxed) {
                return Err(FerrumError::backend("CUDA terminal graph cleanup fault"));
            }
            let status = unsafe { sys::cuGraphDestroy(graph) };
            if status != sys::CUresult::CUDA_SUCCESS {
                return Err(FerrumError::backend(format!(
                    "CUDA terminal graph cleanup failed: {status:?}"
                )));
            }
            self.cu_graph.store(std::ptr::null_mut(), Ordering::Relaxed);
        }
        Ok(())
    }
}

impl Drop for GraphSlotRaw {
    fn drop(&mut self) {
        // Last-reference destruction must follow checked retirement. In
        // particular, an eviction never guesses that a submitted graph ended.
        if !self.cu_graph_exec.load(Ordering::Relaxed).is_null()
            || !self.cu_graph.load(Ordering::Relaxed).is_null()
        {
            eprintln!("fatal: CUDA graph handles dropped before checked retirement");
            std::process::abort();
        }
    }
}

// SAFETY: graph launch is serialised through the model's stream, which
// is accessed from one thread at a time (engine Mutex-wraps the model).
unsafe impl Send for GraphSlotRaw {}
unsafe impl Sync for GraphSlotRaw {}

#[cfg(test)]
mod tests {
    #[test]
    #[ignore = "requires an actual CUDA device"]
    fn graph_eviction_keeps_the_actual_submitted_executable_alive() {
        use super::*;
        static NEXT_KEY: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1 << 63);
        let key = NEXT_KEY.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let mut ctx =
            CudaBackend::new_context(&ferrum_types::BackendContextConfig::default()).unwrap();
        let mut buffer = CudaBackend::from_slice_in_context(&mut ctx, &[1.0, 2.0, 3.0, 4.0]);
        CudaBackend::begin_graph_capture(&mut ctx).unwrap();
        CudaBackend::zero_buffer(&mut ctx, &mut buffer, 4).unwrap();
        CudaBackend::end_graph_capture(&mut ctx, key).unwrap();
        let peer_key = key.wrapping_add(1);
        CudaBackend::begin_graph_capture(&mut ctx).unwrap();
        CudaBackend::zero_buffer(&mut ctx, &mut buffer, 4).unwrap();
        CudaBackend::end_graph_capture(&mut ctx, peer_key).unwrap();
        assert!(matches!(
            CudaBackend::observe_completion(&mut ctx),
            crate::backend::BackendCompletion::Succeeded
        ));
        assert!(CudaBackend::replay_graph(&mut ctx, key).unwrap());
        let exact_graph = Arc::downgrade(ctx.graph_leases.first.as_ref().unwrap());
        CudaBackend::reset_graph(&mut ctx, key);
        assert!(exact_graph.upgrade().is_some());
        // The public graph API must invalidate the prior observation even
        // without an explicit begin_execution. Replaying another key cannot
        // retire A using the observation from before A's latest submission.
        assert!(CudaBackend::replay_graph(&mut ctx, peer_key).unwrap());
        assert!(exact_graph.upgrade().is_some());
        assert!(matches!(
            CudaBackend::observe_completion(&mut ctx),
            crate::backend::BackendCompletion::Succeeded
        ));
        assert!(exact_graph.upgrade().is_none());
        assert_eq!(
            CudaBackend::to_vec_in_context(&mut ctx, &buffer, 4),
            vec![0.0; 4]
        );
    }

    #[test]
    #[ignore = "requires an actual CUDA device and graph replay"]
    fn scratch_growth_keeps_captured_storage_until_graph_and_execution_retire() {
        use super::*;
        let mut ctx =
            CudaBackend::new_context(&ferrum_types::BackendContextConfig::default()).unwrap();
        let stream = ctx.stream.clone();
        let expected = [11i32, 13, 17, 19];
        crate::backend::cuda::with_argmax_out(
            &stream,
            &mut ctx.argmax_out,
            false,
            &mut ctx.captured_storage,
            4,
            |scratch| {
                stream
                    .memcpy_htod(&expected, &mut scratch.slice_mut(..4))
                    .unwrap()
            },
        );
        let mut output = stream.alloc_zeros::<i32>(4).unwrap();
        CudaBackend::observe_completion(&mut ctx)
            .into_result()
            .unwrap();
        CudaBackend::begin_execution(&mut ctx).unwrap();
        CudaBackend::begin_graph_capture(&mut ctx).unwrap();
        stream
            .memcpy_dtod(&ctx.argmax_out.as_ref().unwrap().slice(..4), &mut output)
            .unwrap();
        CudaBackend::end_graph_capture(&mut ctx, 37).unwrap();
        CudaBackend::observe_completion(&mut ctx)
            .into_result()
            .unwrap();

        crate::backend::cuda::with_argmax_out(
            &stream,
            &mut ctx.argmax_out,
            false,
            &mut ctx.captured_storage,
            128,
            |scratch| stream.memset_zeros(scratch).unwrap(),
        );
        assert_eq!(ctx.captured_storage.len(), 1);
        assert!(CudaBackend::replay_graph(&mut ctx, 37).unwrap());
        CudaBackend::observe_completion(&mut ctx)
            .into_result()
            .unwrap();
        assert_eq!(stream.clone_dtoh(&output).unwrap(), expected);
        // Completion alone cannot release storage referenced by a cached graph.
        assert_eq!(ctx.captured_storage.len(), 1);

        assert!(CudaBackend::replay_graph(&mut ctx, 37).unwrap());
        CudaBackend::reset_graph(&mut ctx, 37);
        // Eviction alone cannot release the original in-flight resources.
        assert_eq!(ctx.captured_storage.len(), 1);
        CudaBackend::observe_completion(&mut ctx)
            .into_result()
            .unwrap();
        assert!(ctx.captured_storage.is_empty());
        assert_eq!(stream.clone_dtoh(&output).unwrap(), expected);
        CudaBackend::retire_context(&mut ctx).unwrap();
    }

    #[test]
    #[ignore = "requires an actual CUDA device and graph retirement"]
    fn terminal_graph_cleanup_failure_keeps_exact_handles_until_retry() {
        use super::*;
        let mut ctx =
            CudaBackend::new_context(&ferrum_types::BackendContextConfig::default()).unwrap();
        let mut buffer = CudaBackend::from_slice_in_context(&mut ctx, &[3.0, 5.0, 7.0, 9.0]);
        CudaBackend::begin_graph_capture(&mut ctx).unwrap();
        CudaBackend::zero_buffer(&mut ctx, &mut buffer, 4).unwrap();
        CudaBackend::end_graph_capture(&mut ctx, 41).unwrap();
        assert!(CudaBackend::replay_graph(&mut ctx, 41).unwrap());
        CudaBackend::observe_completion(&mut ctx)
            .into_result()
            .unwrap();
        assert_eq!(
            CudaBackend::to_vec_in_context(&mut ctx, &buffer, 4),
            vec![0.0; 4]
        );
        ctx.captured_storage.push(Box::new(buffer));
        let graph = ctx.graph_cache.get(&41).unwrap().clone();
        let original_graph = graph.cu_graph.load(Ordering::Relaxed);
        let original_exec = graph.cu_graph_exec.load(Ordering::Relaxed);
        graph.fail_next_exec_destroy.store(true, Ordering::Relaxed);
        graph.fail_next_graph_destroy.store(true, Ordering::Relaxed);
        CudaBackend::reset_graph(&mut ctx, 41);

        let outcome = CudaBackend::observe_completion(&mut ctx);
        assert!(
            matches!(outcome, crate::backend::BackendCompletion::Failed(_)),
            "a completed device with failed cleanup must not become Unresolved"
        );
        assert_eq!(graph.cu_graph.load(Ordering::Relaxed), original_graph);
        assert_eq!(graph.cu_graph_exec.load(Ordering::Relaxed), original_exec);
        assert_eq!(ctx.captured_storage.len(), 1);
        assert!(Arc::ptr_eq(
            ctx.graph_leases.first.as_ref().unwrap(),
            &graph
        ));

        ctx.use_dev_state = true;
        assert!(CudaBackend::begin_execution(&mut ctx).is_err());
        // The real executable was destroyed once; only the graph remains.
        assert!(graph.cu_graph_exec.load(Ordering::Relaxed).is_null());
        assert_eq!(graph.cu_graph.load(Ordering::Relaxed), original_graph);
        assert!(
            ctx.use_dev_state,
            "failed begin must not reset invocation state"
        );
        assert_eq!(ctx.captured_storage.len(), 1);

        // Storage mutation has the same checked obligation: terminal device
        // work alone does not permit dropping buffers referenced by an
        // undestroyed captured graph.
        graph.fail_next_graph_destroy.store(true, Ordering::Relaxed);
        assert!(CudaBackend::retire_graphs_for_storage_mutation(&mut ctx).is_err());
        assert_eq!(graph.cu_graph.load(Ordering::Relaxed), original_graph);
        assert!(graph.cu_graph_exec.load(Ordering::Relaxed).is_null());
        assert_eq!(ctx.captured_storage.len(), 1);
        assert!(Arc::ptr_eq(
            ctx.graph_leases.first.as_ref().unwrap(),
            &graph
        ));

        CudaBackend::retire_graphs_for_storage_mutation(&mut ctx).unwrap();
        assert!(graph.cu_graph_exec.load(Ordering::Relaxed).is_null());
        assert!(graph.cu_graph.load(Ordering::Relaxed).is_null());
        assert!(ctx.graph_cache.is_empty());
        assert!(ctx.graph_leases.is_empty());
        assert!(ctx.captured_storage.is_empty());
        // Explicit close is idempotent and cannot double-destroy either handle.
        CudaBackend::retire_context(&mut ctx).unwrap();
    }

    #[test]
    #[ignore = "requires an actual CUDA device; child intentionally fails stop"]
    fn bare_context_drop_with_terminal_cleanup_failure_fails_stop() {
        const CHILD: &str = "FERRUM_TEST_CUDA_TERMINAL_CLEANUP_DROP_CHILD";
        if std::env::var_os(CHILD).is_some() {
            use super::*;
            let mut ctx =
                CudaBackend::new_context(&ferrum_types::BackendContextConfig::default()).unwrap();
            let mut buffer = CudaBackend::from_slice_in_context(&mut ctx, &[1.0; 4]);
            CudaBackend::begin_graph_capture(&mut ctx).unwrap();
            CudaBackend::zero_buffer(&mut ctx, &mut buffer, 4).unwrap();
            CudaBackend::end_graph_capture(&mut ctx, 43).unwrap();
            assert!(CudaBackend::replay_graph(&mut ctx, 43).unwrap());
            CudaBackend::observe_completion(&mut ctx)
                .into_result()
                .unwrap();
            ctx.graph_cache
                .get(&43)
                .unwrap()
                .fail_next_exec_destroy
                .store(true, Ordering::Relaxed);
            // Keep external captured storage alive while the original context
            // tries its checked cleanup and reports the distinct failure.
            drop(ctx);
            drop(buffer);
            panic!("terminal cleanup failure unexpectedly returned from Drop");
        }
        let output = std::process::Command::new(std::env::current_exe().unwrap())
            .arg("backend::cuda::graph::tests::bare_context_drop_with_terminal_cleanup_failure_fails_stop")
            .args(["--exact", "--ignored", "--nocapture"])
            .env(CHILD, "1")
            .output().unwrap();
        assert!(!output.status.success());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("fatal: dropping CUDA instance after unsuccessful checked retirement"),
            "{stderr}"
        );
        assert!(
            stderr.contains("terminal graph executable cleanup fault"),
            "{stderr}"
        );
        assert!(
            !stderr.contains("CUDA completion is unresolved"),
            "{stderr}"
        );
    }

    #[test]
    #[ignore = "requires an actual CUDA device and retained copy destinations"]
    fn contiguous_kv_growth_panic_keeps_original_layout_and_submitted_destinations() {
        use super::*;
        use crate::backend::KvLayer;
        use cudarc::driver::DevicePtr;
        use ferrum_interfaces::kv_dtype::KvFp16;

        let mut ctx =
            CudaBackend::new_context(&ferrum_types::BackendContextConfig::default()).unwrap();
        let mut cache = <KvFp16 as KvLayer<CudaBackend>>::alloc_contig(&mut ctx, 2, 2, 2);
        let original_k = [1.0, 2.0, 3.0, 4.0, 11.0, 12.0, 13.0, 14.0];
        let short_v = [21.0, 22.0, 23.0, 24.0];
        cache.k = CudaBackend::from_slice_in_context(&mut ctx, &original_k);
        // The intentionally short second source fails on head 1 after real
        // head-0 K/V and head-1 K copies were encoded on the original stream.
        cache.v = CudaBackend::from_slice_in_context(&mut ctx, &short_v);
        cache.len = 2;
        CudaBackend::observe_completion(&mut ctx)
            .into_result()
            .unwrap();
        let k_pointer = cache.k.as_f16().device_ptr(&ctx.stream).0;
        let v_pointer = cache.v.as_f16().device_ptr(&ctx.stream).0;
        let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            <KvFp16 as KvLayer<CudaBackend>>::ensure_contig_capacity(&mut ctx, &mut cache, 5, 2, 2)
                .unwrap();
        }));
        assert!(failed.is_err());
        assert_eq!(
            (
                cache.capacity,
                cache.len,
                cache.num_kv_heads,
                cache.head_dim
            ),
            (2, 2, 2, 2)
        );
        assert_eq!(cache.k.as_f16().device_ptr(&ctx.stream).0, k_pointer);
        assert_eq!(cache.v.as_f16().device_ptr(&ctx.stream).0, v_pointer);
        assert_eq!(ctx.captured_storage.len(), 2);
        assert_eq!(
            CudaBackend::to_vec_in_context(&mut ctx, &cache.k, original_k.len()),
            original_k
        );
        assert_eq!(
            CudaBackend::to_vec_in_context(&mut ctx, &cache.v, short_v.len()),
            short_v
        );
        // Readback is not a substitute for releasing the original execution
        // custody through its completion observer.
        assert_eq!(ctx.captured_storage.len(), 2);
        CudaBackend::observe_completion(&mut ctx)
            .into_result()
            .unwrap();
        assert!(ctx.captured_storage.is_empty());
        CudaBackend::retire_context(&mut ctx).unwrap();
    }
}
