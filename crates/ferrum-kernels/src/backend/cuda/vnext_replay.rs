//! Stream-local CUDA executable capture for vNext command segments.
//!
//! The cache never infers replay safety. Core command phases provide eager
//! barriers and each CUDA provider must attach an exact replay key to every
//! command in a captured segment. A key binds provider semantics, launch
//! scalars, physical regions, and retained host bytes.

use std::collections::HashMap;
use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use cudarc::cublas::CudaBlas;
use cudarc::driver::sys;
use cudarc::driver::{CudaContext, CudaStream};
use sha2::{Digest, Sha256};

use super::vnext_runtime::{CudaCommandPayload, CudaDeviceCommand, CudaDeviceRuntimeError};

const COMMAND_KEY_DOMAIN: &[u8] = b"ferrum.cuda-vnext.command-replay.v1\0";
const SEGMENT_KEY_DOMAIN: &[u8] = b"ferrum.cuda-vnext.executable-segment.v1\0";

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct CudaCommandReplayKey([u8; 32]);

impl CudaCommandReplayKey {
    pub(crate) fn bind_runtime_payload(
        self,
        operation: &'static str,
        regions: impl IntoIterator<Item = (u64, u64, u64)>,
        host_storage: &[Box<[u8]>],
    ) -> Self {
        let mut digest = Sha256::new();
        digest.update(COMMAND_KEY_DOMAIN);
        digest.update(self.0);
        digest.update((operation.len() as u64).to_le_bytes());
        digest.update(operation.as_bytes());
        for (pointer, length_bytes, element_bytes) in regions {
            digest.update(pointer.to_le_bytes());
            digest.update(length_bytes.to_le_bytes());
            digest.update(element_bytes.to_le_bytes());
        }
        digest.update((host_storage.len() as u64).to_le_bytes());
        for storage in host_storage {
            digest.update((storage.len() as u64).to_le_bytes());
            digest.update(storage.as_ref());
        }
        Self(digest.finalize().into())
    }

    fn bytes(self) -> [u8; 32] {
        self.0
    }
}

/// Explicit provider-side builder for scalar and topology identity. Buffer
/// addresses and retained host bytes are added by `CudaDeviceCommand`.
pub(crate) struct CudaCommandReplayKeyBuilder(Sha256);

impl CudaCommandReplayKeyBuilder {
    pub(crate) fn new(provider_fingerprint: &str, operation: &'static str) -> Self {
        let mut digest = Sha256::new();
        digest.update(COMMAND_KEY_DOMAIN);
        digest.update((provider_fingerprint.len() as u64).to_le_bytes());
        digest.update(provider_fingerprint.as_bytes());
        digest.update((operation.len() as u64).to_le_bytes());
        digest.update(operation.as_bytes());
        Self(digest)
    }

    pub(crate) fn u64(mut self, value: u64) -> Self {
        self.0.update(value.to_le_bytes());
        self
    }

    pub(crate) fn i32(mut self, value: i32) -> Self {
        self.0.update(value.to_le_bytes());
        self
    }

    pub(crate) fn u32(mut self, value: u32) -> Self {
        self.0.update(value.to_le_bytes());
        self
    }

    pub(crate) fn f32(mut self, value: f32) -> Self {
        self.0.update(value.to_bits().to_le_bytes());
        self
    }

    pub(crate) fn boolean(mut self, value: bool) -> Self {
        self.0.update([u8::from(value)]);
        self
    }

    pub(crate) fn bytes(mut self, value: &[u8]) -> Self {
        self.0.update((value.len() as u64).to_le_bytes());
        self.0.update(value);
        self
    }

    pub(crate) fn finish(self) -> CudaCommandReplayKey {
        CudaCommandReplayKey(self.0.finalize().into())
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct CudaExecutableSegmentKey([u8; 32]);

impl CudaExecutableSegmentKey {
    fn from_commands(commands: &[CudaDeviceCommand]) -> Option<Self> {
        let mut digest = Sha256::new();
        digest.update(SEGMENT_KEY_DOMAIN);
        digest.update((commands.len() as u64).to_le_bytes());
        for command in commands {
            digest.update(command.replay_key()?.bytes());
        }
        Some(Self(digest.finalize().into()))
    }
}

#[derive(Debug)]
pub(crate) struct CudaReplayError {
    stage: &'static str,
    detail: String,
    eager_fallback_safe: bool,
}

impl CudaReplayError {
    fn cuda(stage: &'static str, status: sys::CUresult, eager_fallback_safe: bool) -> Self {
        Self {
            stage,
            detail: format!("{status:?}"),
            eager_fallback_safe,
        }
    }

    fn runtime(
        stage: &'static str,
        error: CudaDeviceRuntimeError,
        eager_fallback_safe: bool,
    ) -> Self {
        Self {
            stage,
            detail: error.to_string(),
            eager_fallback_safe,
        }
    }

    fn provider_panic(capture_terminated: bool) -> Self {
        Self {
            stage: "encode reusable executable capture",
            detail: if capture_terminated {
                "provider command panicked; capture was terminated".to_owned()
            } else {
                "provider command panicked and capture termination could not be proven".to_owned()
            },
            eager_fallback_safe: false,
        }
    }

    pub(crate) const fn eager_fallback_safe(&self) -> bool {
        self.eager_fallback_safe
    }
}

impl fmt::Display for CudaReplayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.stage, self.detail)
    }
}

struct CudaExecutableSegment {
    graph: sys::CUgraph,
    executable: sys::CUgraphExec,
    context: Arc<CudaContext>,
    _stream: Arc<CudaStream>,
    _blas: Arc<CudaBlas>,
    _payloads: Vec<Arc<CudaCommandPayload>>,
    last_used: u64,
}

impl CudaExecutableSegment {
    fn capture(
        context: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        blas: &Arc<CudaBlas>,
        commands: &[CudaDeviceCommand],
        last_used: u64,
    ) -> Result<Self, CudaReplayError> {
        let capture_status = unsafe {
            sys::cuStreamBeginCapture_v2(
                stream.cu_stream(),
                sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_RELAXED,
            )
        };
        if capture_status != sys::CUresult::CUDA_SUCCESS {
            return Err(CudaReplayError::cuda(
                "begin reusable executable capture",
                capture_status,
                true,
            ));
        }

        let encoded = catch_unwind(AssertUnwindSafe(|| {
            commands
                .iter()
                .try_for_each(|command| command.enqueue(stream, blas))
        }));
        let mut graph: sys::CUgraph = std::ptr::null_mut();
        let end_status = unsafe { sys::cuStreamEndCapture(stream.cu_stream(), &mut graph) };
        let mut capture_state = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_INVALIDATED;
        let capture_query_status =
            unsafe { sys::cuStreamIsCapturing(stream.cu_stream(), &mut capture_state) };
        let capture_terminated = capture_query_status == sys::CUresult::CUDA_SUCCESS
            && capture_state == sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_NONE;
        let encoded = match encoded {
            Ok(encoded) => encoded,
            Err(_panic) => {
                if !graph.is_null() {
                    unsafe {
                        sys::cuGraphDestroy(graph);
                    }
                }
                return Err(CudaReplayError::provider_panic(capture_terminated));
            }
        };
        if let Err(error) = encoded {
            if !graph.is_null() {
                unsafe {
                    sys::cuGraphDestroy(graph);
                }
            }
            return Err(CudaReplayError::runtime(
                "encode reusable executable capture",
                error,
                capture_terminated,
            ));
        }
        if end_status != sys::CUresult::CUDA_SUCCESS || graph.is_null() || !capture_terminated {
            if !graph.is_null() {
                unsafe {
                    sys::cuGraphDestroy(graph);
                }
            }
            return Err(CudaReplayError::cuda(
                "end reusable executable capture",
                end_status,
                capture_terminated,
            ));
        }

        let mut executable: sys::CUgraphExec = std::ptr::null_mut();
        let instantiate_status =
            unsafe { sys::cuGraphInstantiateWithFlags(&mut executable, graph, 0) };
        if instantiate_status != sys::CUresult::CUDA_SUCCESS || executable.is_null() {
            unsafe {
                sys::cuGraphDestroy(graph);
            }
            return Err(CudaReplayError::cuda(
                "instantiate reusable executable",
                instantiate_status,
                true,
            ));
        }
        let upload_status = unsafe { sys::cuGraphUpload(executable, stream.cu_stream()) };
        if upload_status != sys::CUresult::CUDA_SUCCESS {
            unsafe {
                sys::cuGraphExecDestroy(executable);
                sys::cuGraphDestroy(graph);
            }
            return Err(CudaReplayError::cuda(
                "upload reusable executable",
                upload_status,
                false,
            ));
        }

        Ok(Self {
            graph,
            executable,
            context: Arc::clone(context),
            _stream: Arc::clone(stream),
            _blas: Arc::clone(blas),
            _payloads: commands.iter().map(CudaDeviceCommand::payload).collect(),
            last_used,
        })
    }

    fn launch(&mut self, stream: &CudaStream, last_used: u64) -> Result<(), CudaReplayError> {
        let status = unsafe { sys::cuGraphLaunch(self.executable, stream.cu_stream()) };
        if status != sys::CUresult::CUDA_SUCCESS {
            return Err(CudaReplayError::cuda(
                "launch reusable executable",
                status,
                false,
            ));
        }
        self.last_used = last_used;
        Ok(())
    }
}

impl Drop for CudaExecutableSegment {
    fn drop(&mut self) {
        if self.context.bind_to_thread().is_err() {
            // Destruction in an unknown context is unsafe. The process will
            // reclaim these handles; normal lane shutdown binds successfully.
            return;
        }
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

// CUDA graph handles are launched only through the owning execution lane.
unsafe impl Send for CudaExecutableSegment {}

pub(crate) struct CudaExecutableCache {
    entries: HashMap<CudaExecutableSegmentKey, CudaExecutableSegment>,
    rejected: HashMap<CudaExecutableSegmentKey, u64>,
    maximum_entries: usize,
    clock: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CudaExecutablePreparation {
    Unavailable,
    CacheHit,
    Captured,
}

impl CudaExecutableCache {
    pub(crate) fn new(maximum_entries: usize) -> Self {
        assert!(
            maximum_entries > 0,
            "CUDA executable cache must be non-empty"
        );
        Self {
            entries: HashMap::new(),
            rejected: HashMap::new(),
            maximum_entries,
            clock: 0,
        }
    }

    fn tick(&mut self) -> u64 {
        self.clock = self.clock.wrapping_add(1).max(1);
        self.clock
    }

    pub(crate) fn prepare(
        &mut self,
        context: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        blas: &Arc<CudaBlas>,
        commands: &[CudaDeviceCommand],
        capture_allowed: bool,
    ) -> Result<CudaExecutablePreparation, CudaReplayError> {
        let Some(key) = CudaExecutableSegmentKey::from_commands(commands) else {
            return Ok(CudaExecutablePreparation::Unavailable);
        };
        let now = self.tick();
        if let Some(entry) = self.entries.get_mut(&key) {
            entry.last_used = now;
            return Ok(CudaExecutablePreparation::CacheHit);
        }
        if self.rejected.contains_key(&key) || !capture_allowed {
            return Ok(CudaExecutablePreparation::Unavailable);
        }
        if self.entries.len() >= self.maximum_entries {
            let oldest = self
                .entries
                .iter()
                .min_by_key(|(_, entry)| entry.last_used)
                .map(|(key, _)| *key);
            if let Some(oldest) = oldest {
                self.entries.remove(&oldest);
            }
        }
        match CudaExecutableSegment::capture(context, stream, blas, commands, now) {
            Ok(entry) => {
                self.entries.insert(key, entry);
                Ok(CudaExecutablePreparation::Captured)
            }
            Err(error) => {
                if error.eager_fallback_safe() {
                    if self.rejected.len() >= self.maximum_entries {
                        let oldest = self
                            .rejected
                            .iter()
                            .min_by_key(|(_, last_used)| *last_used)
                            .map(|(key, _)| *key);
                        if let Some(oldest) = oldest {
                            self.rejected.remove(&oldest);
                        }
                    }
                    self.rejected.insert(key, now);
                }
                Err(error)
            }
        }
    }

    pub(crate) fn launch(
        &mut self,
        stream: &CudaStream,
        commands: &[CudaDeviceCommand],
    ) -> Result<bool, CudaReplayError> {
        let Some(key) = CudaExecutableSegmentKey::from_commands(commands) else {
            return Ok(false);
        };
        let now = self.tick();
        let Some(entry) = self.entries.get_mut(&key) else {
            return Ok(false);
        };
        entry.launch(stream, now)?;
        Ok(true)
    }

    pub(crate) fn leak_if_in_flight(&mut self) {
        for (_, entry) in self.entries.drain() {
            std::mem::forget(entry);
        }
    }
}
