//! Shared test-only CUDA graph lifetime and timing helpers.

use cudarc::driver::{sys, sys::CUevent_flags, CudaStream};
use std::{
    panic::{catch_unwind, resume_unwind, AssertUnwindSafe},
    sync::Arc,
};

// A test-local adapter of vnext_replay.rs's raw capture/instantiate/upload
// lifecycle. Its private production helper also owns DeviceCommands/BLAS;
// these test-only functions are deliberately not registered as production ops.
// The caller keeps all Fixture allocations alive and fixed until this drops.
pub(super) struct Captured {
    graph: sys::CUgraph,
    executable: sys::CUgraphExec,
    stream: Arc<CudaStream>,
    pub(super) node_count: usize,
}

impl Captured {
    pub(super) fn new(stream: &Arc<CudaStream>, enqueue: impl FnOnce()) -> Self {
        stream.context().bind_to_thread().unwrap();
        stream.synchronize().unwrap();
        let begin = unsafe {
            sys::cuStreamBeginCapture_v2(
                stream.cu_stream(),
                sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
            )
        };
        assert_eq!(begin, sys::CUresult::CUDA_SUCCESS, "begin graph capture");
        // Always end capture, including a launch-builder panic. No allocation,
        // copy, event timing or synchronization occurs inside this closure.
        let encoded = catch_unwind(AssertUnwindSafe(enqueue));
        let mut captured = Self {
            graph: std::ptr::null_mut(),
            executable: std::ptr::null_mut(),
            stream: Arc::clone(stream),
            node_count: 0,
        };
        let end = unsafe { sys::cuStreamEndCapture(stream.cu_stream(), &mut captured.graph) };
        let mut state = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_INVALIDATED;
        let query = unsafe { sys::cuStreamIsCapturing(stream.cu_stream(), &mut state) };
        if query != sys::CUresult::CUDA_SUCCESS
            || state != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE
        {
            // Terminate an invalidated capture before any possible unwind or
            // buffer destruction. Do not resume testing an uncleared stream.
            let mut abandoned = std::ptr::null_mut();
            unsafe {
                sys::cuStreamEndCapture(stream.cu_stream(), &mut abandoned);
                if !abandoned.is_null() {
                    sys::cuGraphDestroy(abandoned);
                }
            }
            let retry = unsafe { sys::cuStreamIsCapturing(stream.cu_stream(), &mut state) };
            assert_eq!(retry, sys::CUresult::CUDA_SUCCESS, "query capture cleanup");
            assert_eq!(
                state,
                sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE,
                "capture cleanup did not terminate the stream"
            );
        }
        if let Err(panic) = encoded {
            resume_unwind(panic);
        }
        assert_eq!(end, sys::CUresult::CUDA_SUCCESS, "end graph capture");
        assert!(!captured.graph.is_null(), "empty captured graph");
        let nodes = unsafe {
            sys::cuGraphGetNodes(
                captured.graph,
                std::ptr::null_mut(),
                &mut captured.node_count,
            )
        };
        assert_eq!(nodes, sys::CUresult::CUDA_SUCCESS, "read graph node count");
        // Match current vNext flags=0 and prepare/upload outside replay. Avoid
        // cudarc end_capture/AUTO_FREE_ON_LAUNCH on Blackwell + CUDA 13.
        let instantiate = unsafe {
            sys::cuGraphInstantiateWithFlags(&mut captured.executable, captured.graph, 0)
        };
        assert_eq!(
            instantiate,
            sys::CUresult::CUDA_SUCCESS,
            "instantiate graph"
        );
        assert!(!captured.executable.is_null());
        let upload = unsafe { sys::cuGraphUpload(captured.executable, stream.cu_stream()) };
        assert_eq!(upload, sys::CUresult::CUDA_SUCCESS, "upload graph");
        stream.synchronize().unwrap();
        captured
    }

    pub(super) fn launch(&self) {
        let status = unsafe { sys::cuGraphLaunch(self.executable, self.stream.cu_stream()) };
        assert_eq!(status, sys::CUresult::CUDA_SUCCESS, "replay graph");
    }
}

impl Drop for Captured {
    fn drop(&mut self) {
        if self.stream.context().bind_to_thread().is_err() {
            return;
        }
        // Graph resources die before Fixture buffers; pending work must finish.
        let _ = self.stream.synchronize();
        unsafe {
            if !self.executable.is_null() {
                sys::cuGraphExecDestroy(self.executable);
            }
            if !self.graph.is_null() {
                sys::cuGraphDestroy(self.graph);
            }
        }
    }
}

pub(super) fn measure(stream: &Arc<CudaStream>, launch: impl FnOnce()) -> (f64, f64) {
    let wall = std::time::Instant::now();
    let start = stream
        .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .unwrap();
    launch();
    let end = stream
        .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .unwrap();
    end.synchronize().unwrap();
    (
        wall.elapsed().as_secs_f64() * 1e9,
        f64::from(start.elapsed_ms(&end).unwrap()) * 1e6,
    )
}
