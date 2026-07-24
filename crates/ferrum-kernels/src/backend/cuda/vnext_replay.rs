//! Stream-local CUDA executable capture for vNext command segments.
//!
//! The cache never infers replay safety. Core command phases provide eager
//! barriers and each CUDA provider must attach an exact replay key to every
//! command in a captured segment. A key binds provider semantics, launch
//! scalars, physical regions, and retained host bytes.

use std::collections::HashMap;
use std::ffi::CString;
use std::fmt;
use std::fmt::Write;
use std::ops::Range;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, OnceLock};

use cudarc::cublas::CudaBlas;
use cudarc::driver::sys;
use cudarc::driver::{CudaContext, CudaStream};
use ferrum_interfaces::vnext::{
    DeviceCommandPhase, DeviceReusableAddressScope, DeviceReusableExecutionCapture,
    DeviceReusableExecutionInvocation, DeviceReusableExecutionPlan,
    DeviceReusableExecutionPreparation, DeviceReusableExecutionPreparationState,
    DeviceReusableExecutionProgram, DeviceReusableExecutionProgramId,
    DeviceReusableExecutionSegment, ElementType,
};
use sha2::{Digest, Sha256};

use crate::backend::reusable_execution::{
    discover_reusable_segments, plan_bounded_reusable_execution,
    ReusableExecutionPreparationTracker,
};

use super::vnext_runtime::{CudaCommandExecutable, CudaDeviceCommand, CudaDeviceRuntimeError};
use super::vnext_tool_correlation::correlate_replay_launch;

const COMMAND_KEY_DOMAIN: &[u8] = b"ferrum.cuda-vnext.command-replay.v1\0";
const SEGMENT_KEY_DOMAIN: &[u8] = b"ferrum.cuda-vnext.executable-segment.v1\0";

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct CudaCommandReplayKey([u8; 32]);

impl CudaCommandReplayKey {
    pub(crate) fn bind_runtime_payload(
        self,
        operation: &'static str,
        regions: impl ExactSizeIterator<Item = (u64, u64, ElementType)>,
        host_storage: &[Box<[u8]>],
    ) -> Self {
        let mut digest = Sha256::new();
        digest.update(COMMAND_KEY_DOMAIN);
        digest.update(self.0);
        digest.update((operation.len() as u64).to_le_bytes());
        digest.update(operation.as_bytes());
        digest.update((regions.len() as u64).to_le_bytes());
        for (pointer, length_bytes, element_type) in regions {
            digest.update(pointer.to_le_bytes());
            digest.update(length_bytes.to_le_bytes());
            digest.update([element_type_tag(element_type)]);
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

const fn element_type_tag(element_type: ElementType) -> u8 {
    match element_type {
        ElementType::Bool => 0,
        ElementType::U8 => 1,
        ElementType::U32 => 2,
        ElementType::I8 => 3,
        ElementType::I32 => 4,
        ElementType::F16 => 5,
        ElementType::Bf16 => 6,
        ElementType::F32 => 7,
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
        let mut lane_scope = None;
        for command in commands {
            match command.reusable_address_scope()? {
                DeviceReusableAddressScope::Plan => {}
                DeviceReusableAddressScope::ExecutionLane(lane_id) => match lane_scope {
                    Some(current) if current != lane_id => return None,
                    Some(_) => {}
                    None => lane_scope = Some(lane_id),
                },
            }
            digest.update(command.replay_key()?.bytes());
        }
        Some(Self(digest.finalize().into()))
    }

    fn fingerprint(self) -> String {
        let mut fingerprint = String::with_capacity(64);
        for byte in self.0 {
            write!(&mut fingerprint, "{byte:02x}")
                .expect("writing a byte to an allocated String cannot fail");
        }
        fingerprint
    }
}

pub(crate) struct CudaExecutableCandidate {
    range: Range<usize>,
    key: CudaExecutableSegmentKey,
}

impl CudaExecutableCandidate {
    pub(crate) fn start(&self) -> usize {
        self.range.start
    }

    pub(crate) fn end(&self) -> usize {
        self.range.end
    }

    fn commands<'commands>(
        &self,
        commands: &'commands [CudaDeviceCommand],
    ) -> &'commands [CudaDeviceCommand] {
        &commands[self.range.clone()]
    }
}

pub(crate) fn cuda_executable_candidates(
    phases: &[DeviceCommandPhase],
    commands: &[CudaDeviceCommand],
) -> Result<Vec<CudaExecutableCandidate>, CudaDeviceRuntimeError> {
    if phases.len() != commands.len() {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA command phase count differs from its command count",
        ));
    }
    Ok(discover_reusable_segments(phases, |index| {
        commands[index].replay_key().is_some() && commands[index].reusable_address_scope().is_some()
    })
    .into_iter()
    .filter_map(|segment| {
        let range = segment.range();
        CudaExecutableSegmentKey::from_commands(&commands[range.clone()])
            .map(|key| CudaExecutableCandidate { range, key })
    })
    .collect())
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
    key: CudaExecutableSegmentKey,
    graph: sys::CUgraph,
    executable: sys::CUgraphExec,
    context: Arc<CudaContext>,
    _stream: Arc<CudaStream>,
    _blas: Arc<CudaBlas>,
    _executables: Vec<Arc<CudaCommandExecutable>>,
    command_graph_node_counts: Option<Arc<[u32]>>,
    uploaded: bool,
    last_used: u64,
    profile_identity: OnceLock<CudaExecutableProfileIdentity>,
}

struct CudaExecutableProfileIdentity {
    fingerprint: Arc<str>,
    nvtx_label: CString,
}

impl CudaExecutableProfileIdentity {
    fn new(key: CudaExecutableSegmentKey) -> Self {
        let fingerprint = Arc::<str>::from(key.fingerprint());
        let nvtx_label = CString::new(format!("ferrum.cuda.replay/{fingerprint}"))
            .expect("a lowercase hexadecimal fingerprint contains no NUL bytes");
        Self {
            fingerprint,
            nvtx_label,
        }
    }
}

fn graph_node_count(graph: sys::CUgraph) -> Option<usize> {
    if graph.is_null() {
        return None;
    }
    let mut node_count = 0;
    let status = unsafe { sys::cuGraphGetNodes(graph, std::ptr::null_mut(), &mut node_count) };
    (status == sys::CUresult::CUDA_SUCCESS).then_some(node_count)
}

fn capture_graph_node_count(stream: &CudaStream) -> Option<usize> {
    let mut capture_state = sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_INVALIDATED;
    let mut capture_id = 0;
    let mut graph: sys::CUgraph = std::ptr::null_mut();
    let mut dependencies: *const sys::CUgraphNode = std::ptr::null();
    let mut edge_data: *const sys::CUgraphEdgeData = std::ptr::null();
    let mut dependency_count = 0;
    let status = unsafe {
        sys::cuStreamGetCaptureInfo_v3(
            stream.cu_stream(),
            &mut capture_state,
            &mut capture_id,
            &mut graph,
            &mut dependencies,
            &mut edge_data,
            &mut dependency_count,
        )
    };
    if status != sys::CUresult::CUDA_SUCCESS
        || capture_state != sys::CUstreamCaptureStatus::CU_STREAM_CAPTURE_STATUS_ACTIVE
    {
        return None;
    }
    graph_node_count(graph)
}

impl CudaExecutableSegment {
    fn capture(
        key: CudaExecutableSegmentKey,
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

        let mut previous_graph_node_count = capture_graph_node_count(stream);
        let mut command_graph_node_counts = Vec::with_capacity(commands.len());
        let encoded = catch_unwind(AssertUnwindSafe(|| {
            for command in commands {
                command.enqueue(stream, blas)?;
                if let Some(previous) = previous_graph_node_count {
                    let observed = capture_graph_node_count(stream)
                        .and_then(|current| {
                            current.checked_sub(previous).map(|delta| (current, delta))
                        })
                        .and_then(|(current, delta)| {
                            u32::try_from(delta).ok().map(|delta| (current, delta))
                        });
                    match observed {
                        Some((current, delta)) => {
                            command_graph_node_counts.push(delta);
                            previous_graph_node_count = Some(current);
                        }
                        None => {
                            command_graph_node_counts.clear();
                            previous_graph_node_count = None;
                        }
                    }
                }
            }
            Ok::<(), CudaDeviceRuntimeError>(())
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
        let command_graph_node_counts = (command_graph_node_counts.len() == commands.len())
            .then(|| {
                command_graph_node_counts
                    .iter()
                    .map(|count| *count as usize)
                    .sum::<usize>()
            })
            .filter(|captured_count| graph_node_count(graph) == Some(*captured_count))
            .map(|_| Arc::<[u32]>::from(command_graph_node_counts));

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
        Ok(Self {
            key,
            graph,
            executable,
            context: Arc::clone(context),
            _stream: Arc::clone(stream),
            _blas: Arc::clone(blas),
            _executables: commands.iter().map(CudaDeviceCommand::executable).collect(),
            command_graph_node_counts,
            uploaded: false,
            last_used,
            profile_identity: OnceLock::new(),
        })
    }

    fn upload(&mut self, stream: &CudaStream) -> Result<(), CudaReplayError> {
        let status = unsafe { sys::cuGraphUpload(self.executable, stream.cu_stream()) };
        if status != sys::CUresult::CUDA_SUCCESS {
            return Err(CudaReplayError::cuda(
                "upload reusable executable",
                status,
                false,
            ));
        }
        self.uploaded = true;
        Ok(())
    }

    fn launch(
        &mut self,
        stream: &CudaStream,
        last_used: u64,
        retain_profile_identity: bool,
        tool_correlation: bool,
    ) -> Result<(Option<Arc<str>>, Option<Arc<[u32]>>), CudaReplayError> {
        if !self.uploaded {
            return Err(CudaReplayError {
                stage: "launch reusable executable",
                detail: "executable was not uploaded by its preparation phase".to_owned(),
                eager_fallback_safe: false,
            });
        }
        debug_assert!(!tool_correlation || retain_profile_identity);
        let profile_identity = retain_profile_identity.then(|| {
            self.profile_identity
                .get_or_init(|| CudaExecutableProfileIdentity::new(self.key))
        });
        let status = profile_identity.map_or_else(
            || unsafe { sys::cuGraphLaunch(self.executable, stream.cu_stream()) },
            |identity| {
                correlate_replay_launch(&identity.nvtx_label, || unsafe {
                    sys::cuGraphLaunch(self.executable, stream.cu_stream())
                })
            },
        );
        if status != sys::CUresult::CUDA_SUCCESS {
            return Err(CudaReplayError::cuda(
                "launch reusable executable",
                status,
                false,
            ));
        }
        self.last_used = last_used;
        Ok((
            profile_identity.map(|identity| Arc::clone(&identity.fingerprint)),
            tool_correlation
                .then(|| self.command_graph_node_counts.as_ref().map(Arc::clone))
                .flatten(),
        ))
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
    programs: HashMap<DeviceReusableExecutionProgramId, CudaExecutableProgram>,
    preparation: ReusableExecutionPreparationTracker,
    clock: u64,
}

#[derive(Clone, PartialEq, Eq)]
struct CudaExecutableProgramSegment {
    descriptor: DeviceReusableExecutionSegment,
    key: CudaExecutableSegmentKey,
}

#[derive(Clone, PartialEq, Eq)]
struct CudaExecutableProgram {
    descriptor: DeviceReusableExecutionProgram,
    segments: Vec<CudaExecutableProgramSegment>,
}

pub(crate) struct CudaExecutableLaunch {
    reusable_executable_fingerprint: Option<Arc<str>>,
    reusable_graph_node_counts: Option<Arc<[u32]>>,
}

impl CudaExecutableLaunch {
    pub(crate) fn reusable_executable_fingerprint(&self) -> Option<Arc<str>> {
        self.reusable_executable_fingerprint
            .as_ref()
            .map(Arc::clone)
    }

    pub(crate) fn reusable_graph_node_counts(&self) -> Option<Arc<[u32]>> {
        self.reusable_graph_node_counts.as_ref().map(Arc::clone)
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct CudaExecutablePreparation {
    cache_hit_segments: usize,
    captured_segments: usize,
    uploaded_segments: usize,
    capture_rejected_segments: usize,
    cached_rejected_segments: usize,
    quiescence_deferred_segments: usize,
    capacity_deferred_segments: usize,
    outside_preparation_segments: usize,
    evicted_segments: usize,
}

impl CudaExecutablePreparation {
    pub(crate) fn cache_hit_segments(self) -> usize {
        self.cache_hit_segments
    }

    pub(crate) fn captured_segments(self) -> usize {
        self.captured_segments
    }

    pub(crate) fn uploaded_segments(self) -> usize {
        self.uploaded_segments
    }

    pub(crate) fn capture_rejected_segments(self) -> usize {
        self.capture_rejected_segments
    }

    pub(crate) fn cached_rejected_segments(self) -> usize {
        self.cached_rejected_segments
    }

    pub(crate) fn quiescence_deferred_segments(self) -> usize {
        self.quiescence_deferred_segments
    }

    pub(crate) fn capacity_deferred_segments(self) -> usize {
        self.capacity_deferred_segments
    }

    pub(crate) fn outside_preparation_segments(self) -> usize {
        self.outside_preparation_segments
    }

    pub(crate) fn evicted_segments(self) -> usize {
        self.evicted_segments
    }
}

impl CudaExecutableCache {
    pub(crate) fn new() -> Self {
        Self {
            entries: HashMap::new(),
            rejected: HashMap::new(),
            programs: HashMap::new(),
            preparation: ReusableExecutionPreparationTracker::default(),
            clock: 0,
        }
    }

    pub(crate) fn configure(
        &mut self,
        plan: DeviceReusableExecutionPlan,
    ) -> Result<DeviceReusableExecutionPreparation, String> {
        self.preparation
            .configure(plan, self.entries.len(), self.rejected.len())
    }

    pub(crate) fn seal(&mut self) -> Result<DeviceReusableExecutionPreparation, String> {
        self.preparation
            .seal(self.entries.len(), self.rejected.len())
    }

    pub(crate) fn preparation(&self) -> Result<DeviceReusableExecutionPreparation, String> {
        self.preparation
            .snapshot(self.entries.len(), self.rejected.len())
    }

    fn tick(&mut self) -> u64 {
        self.clock = self.clock.wrapping_add(1).max(1);
        self.clock
    }

    fn remove_entry_and_dependent_programs(&mut self, key: CudaExecutableSegmentKey) -> bool {
        if self.entries.remove(&key).is_none() {
            return false;
        }
        self.programs
            .retain(|_, program| !program.segments.iter().any(|segment| segment.key == key));
        true
    }

    pub(crate) fn prepare_all(
        &mut self,
        context: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        blas: &Arc<CudaBlas>,
        commands: &[CudaDeviceCommand],
        candidates: &[CudaExecutableCandidate],
        capture_allowed: bool,
    ) -> Result<CudaExecutablePreparation, CudaReplayError> {
        let mut report = CudaExecutablePreparation::default();
        for candidate in candidates {
            let now = self.tick();
            if let Some(entry) = self.entries.get_mut(&candidate.key) {
                entry.last_used = now;
                report.cache_hit_segments += 1;
            }
        }
        report.cached_rejected_segments = candidates
            .iter()
            .filter(|candidate| self.rejected.contains_key(&candidate.key))
            .count();

        if candidates.is_empty() {
            return Ok(report);
        }
        if !self.preparation.capture_is_open() {
            report.outside_preparation_segments = candidates
                .len()
                .saturating_sub(report.cache_hit_segments)
                .saturating_sub(report.cached_rejected_segments);
            return Ok(report);
        }
        if !capture_allowed {
            report.quiescence_deferred_segments = candidates
                .len()
                .saturating_sub(report.cache_hit_segments)
                .saturating_sub(report.cached_rejected_segments);
            return Ok(report);
        }

        let candidate_keys = candidates
            .iter()
            .map(|candidate| candidate.key)
            .collect::<Vec<_>>();
        let resident_last_used = self
            .entries
            .iter()
            .map(|(key, entry)| (*key, entry.last_used))
            .collect::<Vec<_>>();
        let rejected_keys = self.rejected.keys().copied().collect::<Vec<_>>();
        let plan = plan_bounded_reusable_execution(
            &candidate_keys,
            &resident_last_used,
            &rejected_keys,
            self.preparation
                .maximum_executables()
                .expect("open reusable execution preparation has a capacity"),
        );
        report.capacity_deferred_segments = plan.capacity_deferred_misses().len();

        let mut captured = Vec::with_capacity(plan.admitted_misses().len());
        for key in plan.admitted_misses().iter().copied() {
            let candidate = candidates
                .iter()
                .find(|candidate| candidate.key == key)
                .expect("admitted CUDA executable key came from its candidate set");
            let now = self.tick();
            match CudaExecutableSegment::capture(
                key,
                context,
                stream,
                blas,
                candidate.commands(commands),
                now,
            ) {
                Ok(entry) => captured.push((key, entry)),
                Err(error) if error.eager_fallback_safe() => {
                    self.remember_rejected(key, now);
                    report.capture_rejected_segments += 1;
                    tracing::debug!(
                        error = %error,
                        command_count = candidate.end() - candidate.start(),
                        "CUDA reusable executable capture rejected; using eager fallback"
                    );
                }
                Err(error) => return Err(error),
            }
        }

        for key in plan.required_evictions(captured.len()) {
            if self.remove_entry_and_dependent_programs(*key) {
                report.evicted_segments += 1;
            }
        }

        let upload_keys = captured.iter().map(|(key, _)| *key).collect::<Vec<_>>();
        report.captured_segments = captured.len();
        for (key, entry) in captured {
            let previous = self.entries.insert(key, entry);
            debug_assert!(previous.is_none());
        }
        for key in upload_keys {
            self.entries
                .get_mut(&key)
                .expect("captured CUDA executable was committed before upload")
                .upload(stream)?;
            report.uploaded_segments += 1;
        }
        self.preparation
            .record_batch(
                report.captured_segments,
                report.uploaded_segments,
                report.capacity_deferred_segments,
            )
            .map_err(|detail| CudaReplayError {
                stage: "record reusable executable preparation",
                detail,
                eager_fallback_safe: false,
            })?;
        debug_assert!(
            self.entries.len()
                <= self
                    .preparation
                    .maximum_executables()
                    .expect("open reusable execution preparation has a capacity")
        );
        Ok(report)
    }

    fn remember_rejected(&mut self, key: CudaExecutableSegmentKey, now: u64) {
        let maximum_entries = self
            .preparation
            .maximum_executables()
            .expect("capture rejection occurs only during preparation");
        if self.rejected.len() >= maximum_entries {
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

    pub(crate) fn register_program(
        &mut self,
        capture: &DeviceReusableExecutionCapture,
        candidates: &[CudaExecutableCandidate],
        command_node_indices: &[Option<u32>],
    ) -> Result<(), CudaReplayError> {
        if !self.preparation.capture_is_open() {
            return Ok(());
        }
        let mut segments = Vec::new();
        for candidate in candidates
            .iter()
            .filter(|candidate| self.entries.contains_key(&candidate.key))
        {
            let node_indices = command_node_indices
                .get(candidate.range.clone())
                .ok_or_else(|| CudaReplayError {
                    stage: "register reusable execution program",
                    detail: "candidate command range exceeds node attribution".to_owned(),
                    eager_fallback_safe: false,
                })?;
            let Some(start_node_index) = node_indices.first().copied().flatten() else {
                return Err(CudaReplayError {
                    stage: "register reusable execution program",
                    detail: "resident segment has no first plan node".to_owned(),
                    eager_fallback_safe: false,
                });
            };
            if node_indices.iter().enumerate().any(|(offset, observed)| {
                let expected = start_node_index.checked_add(offset as u32);
                *observed != expected
            }) {
                return Err(CudaReplayError {
                    stage: "register reusable execution program",
                    detail: "resident segment does not cover one contiguous plan-node range"
                        .to_owned(),
                    eager_fallback_safe: false,
                });
            }
            let end_node_index = start_node_index
                .checked_add(
                    u32::try_from(node_indices.len()).map_err(|_| CudaReplayError {
                        stage: "register reusable execution program",
                        detail: "resident segment node count exceeds u32".to_owned(),
                        eager_fallback_safe: false,
                    })?,
                )
                .ok_or_else(|| CudaReplayError {
                    stage: "register reusable execution program",
                    detail: "resident segment node range overflows u32".to_owned(),
                    eager_fallback_safe: false,
                })?;
            let ordinal = u32::try_from(segments.len()).map_err(|_| CudaReplayError {
                stage: "register reusable execution program",
                detail: "resident segment ordinal exceeds u32".to_owned(),
                eager_fallback_safe: false,
            })?;
            let logical_command_count = u32::try_from(candidate.end() - candidate.start())
                .map_err(|_| CudaReplayError {
                    stage: "register reusable execution program",
                    detail: "resident segment command count exceeds u32".to_owned(),
                    eager_fallback_safe: false,
                })?;
            let descriptor = DeviceReusableExecutionSegment::new(
                ordinal,
                start_node_index,
                end_node_index,
                logical_command_count,
            )
            .map_err(|error| CudaReplayError {
                stage: "register reusable execution program",
                detail: error.to_string(),
                eager_fallback_safe: false,
            })?;
            segments.push(CudaExecutableProgramSegment {
                descriptor,
                key: candidate.key,
            });
        }
        if segments.is_empty() {
            return Ok(());
        }
        let binding_node_indices = capture
            .per_wave_binding_node_indices()
            .iter()
            .copied()
            .filter(|node_index| {
                segments
                    .iter()
                    .any(|segment| segment.descriptor.contains_node(*node_index))
            })
            .collect::<Vec<_>>();
        let descriptor = DeviceReusableExecutionProgram::new(
            capture.program_id().clone(),
            segments
                .iter()
                .map(|segment| segment.descriptor.clone())
                .collect(),
            binding_node_indices,
        )
        .map_err(|error| CudaReplayError {
            stage: "register reusable execution program",
            detail: error.to_string(),
            eager_fallback_safe: false,
        })?;
        let program = CudaExecutableProgram {
            descriptor,
            segments,
        };
        match self.programs.get(capture.program_id()) {
            Some(current) if current == &program => Ok(()),
            Some(_) => Err(CudaReplayError {
                stage: "register reusable execution program",
                detail: "one typed program id resolved to different resident segments".to_owned(),
                eager_fallback_safe: false,
            }),
            None => {
                self.programs.insert(capture.program_id().clone(), program);
                Ok(())
            }
        }
    }

    pub(crate) fn catalog(&self) -> Result<Vec<DeviceReusableExecutionProgram>, String> {
        let preparation = self
            .preparation
            .snapshot(self.entries.len(), self.rejected.len())?;
        if preparation.state() != DeviceReusableExecutionPreparationState::Ready {
            return Err("reusable execution catalog requires a sealed cache".to_owned());
        }
        if self.programs.values().any(|program| {
            program
                .segments
                .iter()
                .any(|segment| !self.entries.contains_key(&segment.key))
        }) {
            return Err(
                "reusable execution catalog references a non-resident executable".to_owned(),
            );
        }
        let mut catalog = self
            .programs
            .values()
            .map(|program| program.descriptor.clone())
            .collect::<Vec<_>>();
        catalog.sort_by(|left, right| left.program_id().cmp(right.program_id()));
        Ok(catalog)
    }

    pub(crate) fn launch_program_segment(
        &mut self,
        stream: &CudaStream,
        invocation: &DeviceReusableExecutionInvocation,
        retain_profile_identity: bool,
        tool_correlation: bool,
    ) -> Result<Option<CudaExecutableLaunch>, CudaReplayError> {
        let Some(program) = self.programs.get(invocation.program_id()) else {
            return Ok(None);
        };
        let Some(segment) = program
            .segments
            .get(invocation.segment().ordinal() as usize)
            .filter(|segment| &segment.descriptor == invocation.segment())
        else {
            return Err(CudaReplayError {
                stage: "launch reusable execution program",
                detail: "segment reference differs from the sealed program catalog".to_owned(),
                eager_fallback_safe: false,
            });
        };
        let key = segment.key;
        let now = self.tick();
        let Some(entry) = self.entries.get_mut(&key) else {
            return Err(CudaReplayError {
                stage: "launch reusable execution program",
                detail: "sealed program executable is no longer resident".to_owned(),
                eager_fallback_safe: false,
            });
        };
        let (reusable_executable_fingerprint, reusable_graph_node_counts) =
            entry.launch(stream, now, retain_profile_identity, tool_correlation)?;
        Ok(Some(CudaExecutableLaunch {
            reusable_executable_fingerprint,
            reusable_graph_node_counts,
        }))
    }

    pub(crate) fn contains_program_segment(
        &self,
        invocation: &DeviceReusableExecutionInvocation,
    ) -> Result<bool, CudaReplayError> {
        let Some(program) = self.programs.get(invocation.program_id()) else {
            return Ok(false);
        };
        let Some(segment) = program
            .segments
            .get(invocation.segment().ordinal() as usize)
        else {
            return Err(CudaReplayError {
                stage: "validate reusable execution program",
                detail: "segment ordinal is outside the sealed program catalog".to_owned(),
                eager_fallback_safe: true,
            });
        };
        if &segment.descriptor != invocation.segment() {
            return Err(CudaReplayError {
                stage: "validate reusable execution program",
                detail: "segment reference differs from the sealed program catalog".to_owned(),
                eager_fallback_safe: true,
            });
        }
        Ok(self.entries.contains_key(&segment.key))
    }

    pub(crate) fn launch(
        &mut self,
        stream: &CudaStream,
        candidate: &CudaExecutableCandidate,
        retain_profile_identity: bool,
        tool_correlation: bool,
    ) -> Result<Option<CudaExecutableLaunch>, CudaReplayError> {
        let now = self.tick();
        let Some(entry) = self.entries.get_mut(&candidate.key) else {
            return Ok(None);
        };
        let (reusable_executable_fingerprint, reusable_graph_node_counts) =
            entry.launch(stream, now, retain_profile_identity, tool_correlation)?;
        Ok(Some(CudaExecutableLaunch {
            reusable_executable_fingerprint,
            reusable_graph_node_counts,
        }))
    }

    pub(crate) fn contains(&self, candidate: &CudaExecutableCandidate) -> bool {
        self.entries.contains_key(&candidate.key)
    }

    pub(crate) fn trim_quiescent(&mut self) -> (usize, usize) {
        let released_executables = self.entries.len();
        let released_rejections = self.rejected.len();
        self.entries.clear();
        self.rejected.clear();
        self.programs.clear();
        (released_executables, released_rejections)
    }

    pub(crate) fn leak_if_in_flight(&mut self) {
        self.programs.clear();
        for (_, entry) in self.entries.drain() {
            std::mem::forget(entry);
        }
    }
}
