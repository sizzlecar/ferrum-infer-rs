use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::num::NonZeroU64;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Duration;

use super::{
    CapabilityId, DeviceAllocationPermit, DeviceId, DynamicStorageProfile, ElementType,
    ExecutionIdentityEnvelope, FailureDomain, FailureEnvelope, IdentifiedFailure, VNextError,
    WeightComponentPayload,
};

/// Backend-neutral device capability for an explicit cold-path reusable
/// executable preparation lifecycle.
pub const DEVICE_REUSABLE_EXECUTION_CAPABILITY_ID: &str = "capability.device.reusable_execution.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
#[serde(transparent)]
pub struct ExecutionLaneId(NonZeroU64);

impl ExecutionLaneId {
    pub(crate) fn mint() -> Result<Self, VNextError> {
        static NEXT_LANE_ID: AtomicU64 = AtomicU64::new(1);
        let raw = NEXT_LANE_ID
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                current.checked_add(1)
            })
            .map_err(|_| VNextError::InvalidExecutionPlan {
                reason: "execution lane identity space is exhausted".to_owned(),
            })?;
        NonZeroU64::new(raw)
            .map(Self)
            .ok_or_else(|| VNextError::InvalidExecutionPlan {
                reason: "execution lane identity must be non-zero".to_owned(),
            })
    }

    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// Cleanup pressure is independent from model size and normal request
/// concurrency. Once one plan accumulates this many non-quiescent owners, new
/// execution authority is rejected until an explicit recovery worker drains
/// the backlog.
pub const MAX_DEFERRED_DEVICE_CLEANUP_TASKS: usize = 64;
pub const MAX_DEFERRED_DEVICE_CLEANUP_MAINTENANCE_TASKS: usize = 64;

const _: () = assert!(
    MAX_DEFERRED_DEVICE_CLEANUP_TASKS > 0
        && MAX_DEFERRED_DEVICE_CLEANUP_TASKS <= 64
        && MAX_DEFERRED_DEVICE_CLEANUP_MAINTENANCE_TASKS > 0
        && MAX_DEFERRED_DEVICE_CLEANUP_MAINTENANCE_TASKS <= 64
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct DeferredDeviceCleanupDomainId(NonZeroU64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DeferredDeviceCleanupDisposition {
    Completed,
    Retryable,
    Quarantined,
}

pub(crate) trait DeferredDeviceCleanupTask: Send + 'static {
    fn try_cleanup(&mut self) -> DeferredDeviceCleanupDisposition;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeferredDeviceCleanupTaskState {
    Pending,
    Retryable,
    Quarantined,
    Panicked,
}

struct DeferredDeviceCleanupEntry {
    task_id: NonZeroU64,
    task: Box<dyn DeferredDeviceCleanupTask>,
    state: DeferredDeviceCleanupTaskState,
}

#[derive(Default)]
struct DeferredDeviceCleanupDomain {
    queued: VecDeque<DeferredDeviceCleanupEntry>,
    in_progress: usize,
    submitted_total: u64,
    attempted_total: u64,
    completed_total: u64,
    panicked_total: u64,
}

#[derive(Default)]
struct DeferredDeviceCleanupRegistry {
    domains: BTreeMap<DeferredDeviceCleanupDomainId, DeferredDeviceCleanupDomain>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeferredDeviceCleanupStatus {
    queued: usize,
    in_progress: usize,
    retryable: usize,
    quarantined: usize,
    panicked: usize,
    submitted_total: u64,
    attempted_total: u64,
    completed_total: u64,
    panicked_total: u64,
}

impl DeferredDeviceCleanupStatus {
    pub const fn queued(&self) -> usize {
        self.queued
    }

    pub const fn in_progress(&self) -> usize {
        self.in_progress
    }

    pub const fn pending(&self) -> usize {
        self.queued + self.in_progress
    }

    pub const fn retryable(&self) -> usize {
        self.retryable
    }

    pub const fn quarantined(&self) -> usize {
        self.quarantined
    }

    pub const fn panicked(&self) -> usize {
        self.panicked
    }

    pub const fn submitted_total(&self) -> u64 {
        self.submitted_total
    }

    pub const fn attempted_total(&self) -> u64 {
        self.attempted_total
    }

    pub const fn completed_total(&self) -> u64 {
        self.completed_total
    }

    pub const fn panicked_total(&self) -> u64 {
        self.panicked_total
    }

    pub const fn is_saturated(&self) -> bool {
        self.pending() >= MAX_DEFERRED_DEVICE_CLEANUP_TASKS
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeferredDeviceCleanupMaintenanceReceipt {
    attempted: usize,
    completed: usize,
    retryable: usize,
    quarantined: usize,
    panicked: usize,
    status_after: DeferredDeviceCleanupStatus,
}

impl DeferredDeviceCleanupMaintenanceReceipt {
    pub const fn attempted(&self) -> usize {
        self.attempted
    }

    pub const fn completed(&self) -> usize {
        self.completed
    }

    pub const fn retryable(&self) -> usize {
        self.retryable
    }

    pub const fn quarantined(&self) -> usize {
        self.quarantined
    }

    pub const fn panicked(&self) -> usize {
        self.panicked
    }

    pub const fn status_after(&self) -> &DeferredDeviceCleanupStatus {
        &self.status_after
    }
}

static NEXT_DEFERRED_DEVICE_CLEANUP_DOMAIN_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_DEFERRED_DEVICE_CLEANUP_TASK_ID: AtomicU64 = AtomicU64::new(1);
static DEFERRED_DEVICE_CLEANUP_REGISTRY: OnceLock<Mutex<DeferredDeviceCleanupRegistry>> =
    OnceLock::new();

fn deferred_device_cleanup_registry() -> &'static Mutex<DeferredDeviceCleanupRegistry> {
    DEFERRED_DEVICE_CLEANUP_REGISTRY
        .get_or_init(|| Mutex::new(DeferredDeviceCleanupRegistry::default()))
}

fn lock_deferred_device_cleanup_registry() -> MutexGuard<'static, DeferredDeviceCleanupRegistry> {
    deferred_device_cleanup_registry()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

pub(crate) fn new_deferred_device_cleanup_domain() -> DeferredDeviceCleanupDomainId {
    let raw = NEXT_DEFERRED_DEVICE_CLEANUP_DOMAIN_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .expect("deferred device cleanup domain identity space is exhausted");
    DeferredDeviceCleanupDomainId(
        NonZeroU64::new(raw).expect("deferred device cleanup domain ids start at one"),
    )
}

/// Transfers one aggregate owner into a process-reachable registry. This path
/// performs no backend call and never drops or forgets a non-quiescent task.
/// Recovery is driven explicitly through bounded maintenance on a scheduler
/// recovery thread.
pub(crate) fn defer_device_cleanup<T>(domain_id: DeferredDeviceCleanupDomainId, task: T)
where
    T: DeferredDeviceCleanupTask,
{
    let task_id = NEXT_DEFERRED_DEVICE_CLEANUP_TASK_ID
        .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
            current.checked_add(1)
        })
        .expect("deferred device cleanup task identity space is exhausted");
    let mut registry = lock_deferred_device_cleanup_registry();
    let domain = registry.domains.entry(domain_id).or_default();
    domain.queued.push_back(DeferredDeviceCleanupEntry {
        task_id: NonZeroU64::new(task_id).expect("deferred device cleanup task ids start at one"),
        task: Box::new(task),
        state: DeferredDeviceCleanupTaskState::Pending,
    });
    domain.submitted_total = domain.submitted_total.saturating_add(1);
}

pub(crate) fn deferred_device_cleanup_status(
    domain_id: DeferredDeviceCleanupDomainId,
) -> DeferredDeviceCleanupStatus {
    let registry = lock_deferred_device_cleanup_registry();
    registry
        .domains
        .get(&domain_id)
        .map(deferred_device_cleanup_domain_status)
        .unwrap_or_else(empty_deferred_device_cleanup_status)
}

pub(crate) fn maintain_deferred_device_cleanups(
    domain_id: DeferredDeviceCleanupDomainId,
    maximum_tasks: usize,
) -> DeferredDeviceCleanupMaintenanceReceipt {
    debug_assert!(
        maximum_tasks > 0 && maximum_tasks <= MAX_DEFERRED_DEVICE_CLEANUP_MAINTENANCE_TASKS
    );
    let selected = {
        let registry = lock_deferred_device_cleanup_registry();
        registry
            .domains
            .get(&domain_id)
            .map(|domain| {
                domain
                    .queued
                    .iter()
                    .take(maximum_tasks)
                    .map(|entry| entry.task_id)
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default()
    };

    let mut attempted = 0;
    let mut completed = 0;
    let mut retryable = 0;
    let mut quarantined = 0;
    let mut panicked = 0;
    for task_id in selected {
        let Some(mut entry) = ({
            let mut registry = lock_deferred_device_cleanup_registry();
            let domain = registry.domains.entry(domain_id).or_default();
            let entry = domain
                .queued
                .iter()
                .position(|entry| entry.task_id == task_id)
                .and_then(|position| domain.queued.remove(position));
            if entry.is_some() {
                domain.in_progress = domain.in_progress.saturating_add(1);
            }
            entry
        }) else {
            continue;
        };
        attempted += 1;
        let outcome = catch_unwind(AssertUnwindSafe(|| entry.task.try_cleanup()));
        let mut registry = lock_deferred_device_cleanup_registry();
        let domain = registry.domains.entry(domain_id).or_default();
        domain.in_progress = domain.in_progress.saturating_sub(1);
        domain.attempted_total = domain.attempted_total.saturating_add(1);
        match outcome {
            Ok(DeferredDeviceCleanupDisposition::Completed) => {
                domain.completed_total = domain.completed_total.saturating_add(1);
                completed += 1;
            }
            Ok(DeferredDeviceCleanupDisposition::Retryable) => {
                entry.state = DeferredDeviceCleanupTaskState::Retryable;
                domain.queued.push_back(entry);
                retryable += 1;
            }
            Ok(DeferredDeviceCleanupDisposition::Quarantined) => {
                entry.state = DeferredDeviceCleanupTaskState::Quarantined;
                domain.queued.push_back(entry);
                quarantined += 1;
            }
            Err(_) => {
                entry.state = DeferredDeviceCleanupTaskState::Panicked;
                domain.queued.push_back(entry);
                domain.panicked_total = domain.panicked_total.saturating_add(1);
                panicked += 1;
            }
        }
    }

    let status_after = deferred_device_cleanup_status(domain_id);
    DeferredDeviceCleanupMaintenanceReceipt {
        attempted,
        completed,
        retryable,
        quarantined,
        panicked,
        status_after,
    }
}

pub(crate) fn retire_deferred_device_cleanup_domain(
    domain_id: DeferredDeviceCleanupDomainId,
) -> bool {
    let mut registry = lock_deferred_device_cleanup_registry();
    if registry
        .domains
        .get(&domain_id)
        .is_some_and(|domain| !domain.queued.is_empty() || domain.in_progress != 0)
    {
        return false;
    }
    registry.domains.remove(&domain_id);
    true
}

fn deferred_device_cleanup_domain_status(
    domain: &DeferredDeviceCleanupDomain,
) -> DeferredDeviceCleanupStatus {
    DeferredDeviceCleanupStatus {
        queued: domain.queued.len(),
        in_progress: domain.in_progress,
        retryable: domain
            .queued
            .iter()
            .filter(|entry| entry.state == DeferredDeviceCleanupTaskState::Retryable)
            .count(),
        quarantined: domain
            .queued
            .iter()
            .filter(|entry| entry.state == DeferredDeviceCleanupTaskState::Quarantined)
            .count(),
        panicked: domain
            .queued
            .iter()
            .filter(|entry| entry.state == DeferredDeviceCleanupTaskState::Panicked)
            .count(),
        submitted_total: domain.submitted_total,
        attempted_total: domain.attempted_total,
        completed_total: domain.completed_total,
        panicked_total: domain.panicked_total,
    }
}

const fn empty_deferred_device_cleanup_status() -> DeferredDeviceCleanupStatus {
    DeferredDeviceCleanupStatus {
        queued: 0,
        in_progress: 0,
        retryable: 0,
        quarantined: 0,
        panicked: 0,
        submitted_total: 0,
        attempted_total: 0,
        completed_total: 0,
        panicked_total: 0,
    }
}

/// Backend-neutral device classes. Concrete backend names do not belong here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceClass {
    Host,
    Accelerator,
    Reference,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceDescriptor {
    pub id: DeviceId,
    pub class: DeviceClass,
    pub ordinal: u32,
    pub total_memory_bytes: u64,
    pub runtime_implementation_fingerprint: String,
    pub capabilities: BTreeSet<CapabilityId>,
    pub dynamic_storage_profiles: BTreeSet<DynamicStorageProfile>,
}

impl DeviceDescriptor {
    pub fn validate(&self) -> Result<(), VNextError> {
        if self.total_memory_bytes == 0
            || self.dynamic_storage_profiles.is_empty()
            || self.runtime_implementation_fingerprint.len() != 64
            || !self
                .runtime_implementation_fingerprint
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!(
                    "device `{}` has invalid capacity or runtime implementation fingerprint",
                    self.id
                ),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BufferRequest {
    resource_id: super::ResourceId,
    size_bytes: u64,
    alignment_bytes: u64,
    usage: BufferUsage,
    element_type: ElementType,
}

impl BufferRequest {
    pub fn new(
        resource_id: super::ResourceId,
        size_bytes: u64,
        alignment_bytes: u64,
        usage: BufferUsage,
        element_type: ElementType,
    ) -> Result<Self, super::VNextError> {
        if size_bytes == 0 || alignment_bytes == 0 || !alignment_bytes.is_power_of_two() {
            return Err(super::VNextError::InvalidExecutionPlan {
                reason: "buffer request has invalid size or alignment".to_owned(),
            });
        }
        Ok(Self {
            resource_id,
            size_bytes,
            alignment_bytes,
            usage,
            element_type,
        })
    }

    pub fn resource_id(&self) -> &super::ResourceId {
        &self.resource_id
    }

    pub fn size_bytes(&self) -> u64 {
        self.size_bytes
    }

    pub fn alignment_bytes(&self) -> u64 {
        self.alignment_bytes
    }

    pub fn usage(&self) -> BufferUsage {
        self.usage
    }

    pub fn element_type(&self) -> ElementType {
        self.element_type
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BufferDescriptor {
    pub resource_id: super::ResourceId,
    pub size_bytes: u64,
    pub alignment_bytes: u64,
    pub usage: BufferUsage,
    pub element_type: ElementType,
}

/// Opaque core ownership retained by backend commands that outlive the
/// borrowed buffer view used to encode them. Backends may clone and store this
/// value, but cannot inspect or manufacture resource ownership.
#[derive(Clone)]
pub struct DeviceBufferRetention {
    _primary_owner: Arc<dyn Send + Sync + 'static>,
    _secondary_owner: Option<Arc<dyn Send + Sync + 'static>>,
    reusable_address_scope: Option<DeviceReusableAddressScope>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceReusableAddressScope {
    Plan,
    ExecutionLane(ExecutionLaneId),
}

impl DeviceBufferRetention {
    pub(crate) fn plan<T>(owner: Arc<T>) -> Self
    where
        T: Send + Sync + 'static,
    {
        Self {
            _primary_owner: owner,
            _secondary_owner: None,
            reusable_address_scope: Some(DeviceReusableAddressScope::Plan),
        }
    }

    pub(crate) fn pair<T, U>(primary_owner: Arc<T>, secondary_owner: Arc<U>) -> Self
    where
        T: Send + Sync + 'static,
        U: Send + Sync + 'static,
    {
        Self {
            _primary_owner: primary_owner,
            _secondary_owner: Some(secondary_owner),
            reusable_address_scope: None,
        }
    }

    pub(crate) fn lane_pair<T, U>(
        lane_id: ExecutionLaneId,
        primary_owner: Arc<T>,
        secondary_owner: Arc<U>,
    ) -> Self
    where
        T: Send + Sync + 'static,
        U: Send + Sync + 'static,
    {
        Self {
            _primary_owner: primary_owner,
            _secondary_owner: Some(secondary_owner),
            reusable_address_scope: Some(DeviceReusableAddressScope::ExecutionLane(lane_id)),
        }
    }

    pub const fn reusable_address_scope(&self) -> Option<DeviceReusableAddressScope> {
        self.reusable_address_scope
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BufferUsage {
    Weights,
    Activations,
    State,
    /// Provider/runtime workspace whose lifetime spans operations but is not
    /// model semantic state (for example packed metadata or persistent scratch).
    Persistent,
    /// Request-shaped provider control data written before reusable compute.
    Binding,
    Scratch,
    Transfer,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CopyRegion {
    source_offset_bytes: u64,
    destination_offset_bytes: u64,
    length_bytes: u64,
}

impl CopyRegion {
    pub fn new(
        source_offset_bytes: u64,
        destination_offset_bytes: u64,
        length_bytes: u64,
    ) -> Result<Self, super::VNextError> {
        if length_bytes == 0
            || source_offset_bytes.checked_add(length_bytes).is_none()
            || destination_offset_bytes.checked_add(length_bytes).is_none()
        {
            return Err(super::VNextError::InvalidExecutionPlan {
                reason: "copy region is empty or overflows u64".to_owned(),
            });
        }
        Ok(Self {
            source_offset_bytes,
            destination_offset_bytes,
            length_bytes,
        })
    }

    pub fn validate_bounds(
        &self,
        source: &BufferDescriptor,
        destination: &BufferDescriptor,
    ) -> Result<(), super::VNextError> {
        let source_end = self
            .source_offset_bytes
            .checked_add(self.length_bytes)
            .ok_or_else(|| super::VNextError::InvalidExecutionPlan {
                reason: "source copy range overflows u64".to_owned(),
            })?;
        let destination_end = self
            .destination_offset_bytes
            .checked_add(self.length_bytes)
            .ok_or_else(|| super::VNextError::InvalidExecutionPlan {
                reason: "destination copy range overflows u64".to_owned(),
            })?;
        if source_end > source.size_bytes || destination_end > destination.size_bytes {
            return Err(super::VNextError::InvalidExecutionPlan {
                reason: "copy region exceeds a buffer boundary".to_owned(),
            });
        }
        Ok(())
    }

    pub fn source_offset_bytes(self) -> u64 {
        self.source_offset_bytes
    }

    pub fn destination_offset_bytes(self) -> u64 {
        self.destination_offset_bytes
    }

    pub fn length_bytes(self) -> u64 {
        self.length_bytes
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamState {
    Ready,
    Recording,
    Submitted,
    Failed,
}

/// Backend timing is enabled monotonically before product requests start.
/// `Off` must not allocate backend events or add host clock reads to the hot
/// path; `Completion` measures only the existing submission terminal and
/// readback boundaries; `Kernel` additionally attributes backend-observed
/// physical work to the core-issued immutable-plan node index.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum DeviceTimingMode {
    #[default]
    Off,
    Completion,
    Kernel,
}

impl DeviceTimingMode {
    pub const fn completion_enabled(self) -> bool {
        !matches!(self, Self::Off)
    }

    pub const fn kernel_attribution_enabled(self) -> bool {
        matches!(self, Self::Kernel)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceExecutionPath {
    Eager,
    Replayed,
}

impl DeviceExecutionPath {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Eager => "eager",
            Self::Replayed => "replayed",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceBatchingForm {
    Scalar,
    Packed,
    ParticipantLoop,
}

impl DeviceBatchingForm {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Packed => "packed",
            Self::ParticipantLoop => "participant_loop",
        }
    }
}

/// Typed host boundaries inside one backend submission. These intervals use
/// the host monotonic clock and must not be combined with device-event time.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceSubmissionStage {
    ValidateAndPrepare,
    BeginTiming,
    EnqueueCommands,
    RecordFenceAndAccount,
}

/// Aggregate reusable-execution work observed inside one backend submission.
///
/// The observation carries no execution authority and is recorded only by an
/// enabled diagnostic sink. Backends update one stack value while submitting;
/// the product sink aggregates it without allocating on the hot path.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct DeviceReusableExecutionObservation {
    candidate_segments: u64,
    captured_segments: u64,
    uploaded_segments: u64,
    cache_hit_segments: u64,
    cached_rejected_segments: u64,
    capture_rejected_segments: u64,
    quiescence_deferred_segments: u64,
    capacity_deferred_segments: u64,
    outside_preparation_segments: u64,
    evicted_segments: u64,
    replayed_segments: u64,
    replayed_commands: u64,
    eager_commands: u64,
}

impl DeviceReusableExecutionObservation {
    pub fn observe_candidate_segment(&mut self) {
        self.candidate_segments = self.candidate_segments.saturating_add(1);
    }

    pub fn observe_captured_segment(&mut self) {
        self.captured_segments = self.captured_segments.saturating_add(1);
    }

    pub fn observe_uploaded_segment(&mut self) {
        self.uploaded_segments = self.uploaded_segments.saturating_add(1);
    }

    pub fn observe_cache_hit_segment(&mut self) {
        self.cache_hit_segments = self.cache_hit_segments.saturating_add(1);
    }

    pub fn observe_cached_rejected_segment(&mut self) {
        self.cached_rejected_segments = self.cached_rejected_segments.saturating_add(1);
    }

    pub fn observe_capture_rejection(&mut self) {
        self.capture_rejected_segments = self.capture_rejected_segments.saturating_add(1);
    }

    pub fn observe_quiescence_deferred_segment(&mut self) {
        self.quiescence_deferred_segments = self.quiescence_deferred_segments.saturating_add(1);
    }

    pub fn observe_capacity_deferred_segment(&mut self) {
        self.capacity_deferred_segments = self.capacity_deferred_segments.saturating_add(1);
    }

    pub fn observe_outside_preparation_segment(&mut self) {
        self.outside_preparation_segments = self.outside_preparation_segments.saturating_add(1);
    }

    pub fn observe_evicted_segment(&mut self) {
        self.evicted_segments = self.evicted_segments.saturating_add(1);
    }

    pub fn observe_replayed_segment(&mut self, command_count: usize) {
        self.replayed_segments = self.replayed_segments.saturating_add(1);
        self.replayed_commands = self
            .replayed_commands
            .saturating_add(u64::try_from(command_count).unwrap_or(u64::MAX));
    }

    pub fn observe_eager_command(&mut self) {
        self.eager_commands = self.eager_commands.saturating_add(1);
    }

    pub const fn candidate_segments(self) -> u64 {
        self.candidate_segments
    }

    pub const fn captured_segments(self) -> u64 {
        self.captured_segments
    }

    pub const fn uploaded_segments(self) -> u64 {
        self.uploaded_segments
    }

    pub const fn cache_hit_segments(self) -> u64 {
        self.cache_hit_segments
    }

    pub const fn cached_rejected_segments(self) -> u64 {
        self.cached_rejected_segments
    }

    pub const fn capture_rejected_segments(self) -> u64 {
        self.capture_rejected_segments
    }

    pub const fn quiescence_deferred_segments(self) -> u64 {
        self.quiescence_deferred_segments
    }

    pub const fn capacity_deferred_segments(self) -> u64 {
        self.capacity_deferred_segments
    }

    pub const fn outside_preparation_segments(self) -> u64 {
        self.outside_preparation_segments
    }

    pub const fn evicted_segments(self) -> u64 {
        self.evicted_segments
    }

    pub const fn replayed_segments(self) -> u64 {
        self.replayed_segments
    }

    pub const fn replayed_commands(self) -> u64 {
        self.replayed_commands
    }

    pub const fn eager_commands(self) -> u64 {
        self.eager_commands
    }
}

/// Cold-path receipt for releasing backend reusable executables after an
/// execution lane has reached proven quiescence.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct DeviceReusableExecutionTrim {
    released_executables: u64,
    released_rejections: u64,
}

/// Cold-path capacity selected by the model execution plan before reusable
/// device executables are prepared.
///
/// The value is an upper bound on resident executable descriptors, not a
/// hardware- or model-name heuristic. Product composition derives it from the
/// immutable execution plan and the startup shape matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct DeviceReusableExecutionPlan {
    maximum_executables: usize,
}

impl DeviceReusableExecutionPlan {
    pub fn new(maximum_executables: usize) -> Result<Self, super::VNextError> {
        if maximum_executables == 0 {
            return Err(super::VNextError::InvalidExecutionPlan {
                reason: "reusable execution plan requires non-zero capacity".to_owned(),
            });
        }
        Ok(Self {
            maximum_executables,
        })
    }

    pub const fn maximum_executables(self) -> usize {
        self.maximum_executables
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceReusableExecutionPreparationState {
    Unsupported,
    Preparing,
    Ready,
}

/// Backend receipt for the explicit configure -> prepare -> seal lifecycle.
///
/// Captures happen only between `Preparing` and `Ready`. Once sealed, a
/// backend must replay a resident executable or use eager execution; it must
/// not compile new work on a product request.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct DeviceReusableExecutionPreparation {
    state: DeviceReusableExecutionPreparationState,
    maximum_executables: u64,
    resident_executables: u64,
    rejected_executables: u64,
    captured_executables: u64,
    uploaded_executables: u64,
    capacity_deferred_executables: u64,
}

impl DeviceReusableExecutionPreparation {
    pub const fn unsupported() -> Self {
        Self {
            state: DeviceReusableExecutionPreparationState::Unsupported,
            maximum_executables: 0,
            resident_executables: 0,
            rejected_executables: 0,
            captured_executables: 0,
            uploaded_executables: 0,
            capacity_deferred_executables: 0,
        }
    }

    pub fn preparing(plan: DeviceReusableExecutionPlan) -> Self {
        Self {
            state: DeviceReusableExecutionPreparationState::Preparing,
            maximum_executables: u64::try_from(plan.maximum_executables()).unwrap_or(u64::MAX),
            ..Self::unsupported()
        }
    }

    pub fn preparing_with_progress(
        plan: DeviceReusableExecutionPlan,
        resident_executables: usize,
        rejected_executables: usize,
        captured_executables: u64,
        uploaded_executables: u64,
        capacity_deferred_executables: u64,
    ) -> Result<Self, super::VNextError> {
        Self::with_progress(
            DeviceReusableExecutionPreparationState::Preparing,
            plan,
            resident_executables,
            rejected_executables,
            captured_executables,
            uploaded_executables,
            capacity_deferred_executables,
        )
    }

    pub fn ready(
        plan: DeviceReusableExecutionPlan,
        resident_executables: usize,
        rejected_executables: usize,
        captured_executables: u64,
        uploaded_executables: u64,
        capacity_deferred_executables: u64,
    ) -> Result<Self, super::VNextError> {
        Self::with_progress(
            DeviceReusableExecutionPreparationState::Ready,
            plan,
            resident_executables,
            rejected_executables,
            captured_executables,
            uploaded_executables,
            capacity_deferred_executables,
        )
    }

    fn with_progress(
        state: DeviceReusableExecutionPreparationState,
        plan: DeviceReusableExecutionPlan,
        resident_executables: usize,
        rejected_executables: usize,
        captured_executables: u64,
        uploaded_executables: u64,
        capacity_deferred_executables: u64,
    ) -> Result<Self, super::VNextError> {
        if resident_executables > plan.maximum_executables()
            || uploaded_executables < u64::try_from(resident_executables).unwrap_or(u64::MAX)
            || captured_executables < uploaded_executables
        {
            return Err(super::VNextError::InvalidExecutionPlan {
                reason: "reusable execution preparation receipt is internally inconsistent"
                    .to_owned(),
            });
        }
        Ok(Self {
            state,
            maximum_executables: u64::try_from(plan.maximum_executables()).unwrap_or(u64::MAX),
            resident_executables: u64::try_from(resident_executables).unwrap_or(u64::MAX),
            rejected_executables: u64::try_from(rejected_executables).unwrap_or(u64::MAX),
            captured_executables,
            uploaded_executables,
            capacity_deferred_executables,
        })
    }

    pub const fn state(self) -> DeviceReusableExecutionPreparationState {
        self.state
    }

    pub const fn maximum_executables(self) -> u64 {
        self.maximum_executables
    }

    pub const fn resident_executables(self) -> u64 {
        self.resident_executables
    }

    pub const fn rejected_executables(self) -> u64 {
        self.rejected_executables
    }

    pub const fn captured_executables(self) -> u64 {
        self.captured_executables
    }

    pub const fn uploaded_executables(self) -> u64 {
        self.uploaded_executables
    }

    pub const fn capacity_deferred_executables(self) -> u64 {
        self.capacity_deferred_executables
    }
}

impl DeviceReusableExecutionTrim {
    pub fn new(released_executables: usize, released_rejections: usize) -> Self {
        Self {
            released_executables: u64::try_from(released_executables).unwrap_or(u64::MAX),
            released_rejections: u64::try_from(released_rejections).unwrap_or(u64::MAX),
        }
    }

    pub const fn released_executables(self) -> u64 {
        self.released_executables
    }

    pub const fn released_rejections(self) -> u64 {
        self.released_rejections
    }
}

/// Diagnostic-only sink for backend submission attribution.
///
/// `ENABLED = false` is the compile-time off path: a backend must not read a
/// clock or call `record_device_submission` in that specialization. Enabled
/// implementations run on the submission thread and must not block, allocate,
/// or panic.
pub trait DeviceSubmissionTimingSink: Send + Sync {
    const ENABLED: bool;

    fn record_device_submission(&self, stage: DeviceSubmissionStage, elapsed: Duration);

    fn record_reusable_execution(&self, _observation: DeviceReusableExecutionObservation) {}
}

pub struct DisabledDeviceSubmissionTimingSink;

impl DeviceSubmissionTimingSink for DisabledDeviceSubmissionTimingSink {
    const ENABLED: bool = false;

    fn record_device_submission(&self, _stage: DeviceSubmissionStage, _elapsed: Duration) {
        unreachable!("disabled device submission timing cannot record")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceTimingUnavailableReason {
    BackendUnsupported,
    BackendMeasurementFailed,
    DurationOverflow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "status", content = "detail")]
pub enum DeviceTimingMeasurement<T> {
    NotRequested,
    Measured(T),
    Unavailable(DeviceTimingUnavailableReason),
}

impl<T> DeviceTimingMeasurement<T> {
    pub const fn measured(&self) -> Option<&T> {
        match self {
            Self::Measured(measured) => Some(measured),
            Self::NotRequested | Self::Unavailable(_) => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceTimingClock {
    DeviceEventElapsed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeviceExecutionTiming {
    elapsed_ns: u64,
    clock: DeviceTimingClock,
}

impl DeviceExecutionTiming {
    pub const fn device_event_elapsed(elapsed_ns: u64) -> Self {
        Self {
            elapsed_ns,
            clock: DeviceTimingClock::DeviceEventElapsed,
        }
    }

    pub const fn elapsed_ns(self) -> u64 {
        self.elapsed_ns
    }

    pub const fn clock(self) -> DeviceTimingClock {
        self.clock
    }
}

/// One backend-counter interval relative to the first sampled command in an
/// exact submission. Intervals remain in a device elapsed-time domain; they
/// must not be subtracted from host timestamps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceExecutionIntervalKind {
    Compute,
    Transfer,
}

impl DeviceExecutionIntervalKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Compute => "compute",
            Self::Transfer => "transfer",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct DeviceExecutionInterval {
    kind: DeviceExecutionIntervalKind,
    start_offset_ns: u64,
    end_offset_ns: u64,
    subwork_id: Option<&'static str>,
}

impl DeviceExecutionInterval {
    pub fn new(
        kind: DeviceExecutionIntervalKind,
        start_offset_ns: u64,
        end_offset_ns: u64,
    ) -> Option<Self> {
        (end_offset_ns > start_offset_ns).then_some(Self {
            kind,
            start_offset_ns,
            end_offset_ns,
            subwork_id: None,
        })
    }

    pub fn new_labeled(
        kind: DeviceExecutionIntervalKind,
        start_offset_ns: u64,
        end_offset_ns: u64,
        subwork_id: &'static str,
    ) -> Option<Self> {
        (!subwork_id.is_empty() && end_offset_ns > start_offset_ns).then_some(Self {
            kind,
            start_offset_ns,
            end_offset_ns,
            subwork_id: Some(subwork_id),
        })
    }

    pub const fn kind(self) -> DeviceExecutionIntervalKind {
        self.kind
    }

    pub const fn start_offset_ns(self) -> u64 {
        self.start_offset_ns
    }

    pub const fn end_offset_ns(self) -> u64 {
        self.end_offset_ns
    }

    pub const fn subwork_id(self) -> Option<&'static str> {
        self.subwork_id
    }

    pub const fn elapsed_ns(self) -> u64 {
        self.end_offset_ns - self.start_offset_ns
    }
}

/// Backend-counter timing for one command entry in a core-owned submission.
/// A command may own multiple physical encoder intervals, for example a
/// gather-compute-scatter implementation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeviceCommandExecutionTiming {
    command_index: u32,
    intervals: Box<[DeviceExecutionInterval]>,
    elapsed_ns: u64,
}

impl DeviceCommandExecutionTiming {
    pub fn new(command_index: u32, intervals: Vec<DeviceExecutionInterval>) -> Option<Self> {
        if intervals.is_empty()
            || intervals
                .windows(2)
                .any(|pair| pair[0].end_offset_ns() > pair[1].start_offset_ns())
        {
            return None;
        }
        let elapsed_ns = intervals.iter().try_fold(0_u64, |total, interval| {
            total.checked_add(interval.elapsed_ns())
        })?;
        Some(Self {
            command_index,
            intervals: intervals.into_boxed_slice(),
            elapsed_ns,
        })
    }

    pub const fn command_index(&self) -> u32 {
        self.command_index
    }

    pub fn intervals(&self) -> &[DeviceExecutionInterval] {
        &self.intervals
    }

    pub fn elapsed_ns(&self) -> u64 {
        self.elapsed_ns
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceExecutionSpanKind {
    EagerCommand,
    ReusableExecutable,
}

impl DeviceExecutionSpanKind {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EagerCommand => "eager_command",
            Self::ReusableExecutable => "reusable_executable",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case", tag = "status", content = "detail")]
pub enum DeviceExecutionSpanMeasurement {
    Measured {
        intervals: Box<[DeviceExecutionInterval]>,
        elapsed_ns: u64,
    },
    Unavailable(DeviceTimingUnavailableReason),
}

impl DeviceExecutionSpanMeasurement {
    pub fn measured(intervals: Vec<DeviceExecutionInterval>) -> Option<Self> {
        if intervals.is_empty()
            || intervals
                .windows(2)
                .any(|pair| pair[0].end_offset_ns() > pair[1].start_offset_ns())
        {
            return None;
        }
        let elapsed_ns = intervals.iter().try_fold(0_u64, |total, interval| {
            total.checked_add(interval.elapsed_ns())
        })?;
        Some(Self::Measured {
            intervals: intervals.into_boxed_slice(),
            elapsed_ns,
        })
    }

    pub const fn unavailable(reason: DeviceTimingUnavailableReason) -> Self {
        Self::Unavailable(reason)
    }

    pub fn intervals(&self) -> Option<&[DeviceExecutionInterval]> {
        match self {
            Self::Measured { intervals, .. } => Some(intervals),
            Self::Unavailable(_) => None,
        }
    }

    pub const fn elapsed_ns(&self) -> Option<u64> {
        match self {
            Self::Measured { elapsed_ns, .. } => Some(*elapsed_ns),
            Self::Unavailable(_) => None,
        }
    }

    pub const fn unavailable_reason(&self) -> Option<DeviceTimingUnavailableReason> {
        match self {
            Self::Measured { .. } => None,
            Self::Unavailable(reason) => Some(*reason),
        }
    }
}

/// One physical device interval owner inside an exact submission.
///
/// Eager spans own one core command. Reusable executable spans own one
/// contiguous command range, because a single CUDA graph launch cannot be
/// truthfully duplicated across each logical command it contains.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeviceSubmissionExecutionSpan {
    start_command_index: u32,
    end_command_index: u32,
    kind: DeviceExecutionSpanKind,
    measurement: DeviceExecutionSpanMeasurement,
}

impl DeviceSubmissionExecutionSpan {
    pub fn measured(
        start_command_index: u32,
        end_command_index: u32,
        kind: DeviceExecutionSpanKind,
        intervals: Vec<DeviceExecutionInterval>,
    ) -> Option<Self> {
        let measurement = DeviceExecutionSpanMeasurement::measured(intervals)?;
        Self::new(start_command_index, end_command_index, kind, measurement)
    }

    pub fn unavailable(
        start_command_index: u32,
        end_command_index: u32,
        kind: DeviceExecutionSpanKind,
        reason: DeviceTimingUnavailableReason,
    ) -> Option<Self> {
        Self::new(
            start_command_index,
            end_command_index,
            kind,
            DeviceExecutionSpanMeasurement::unavailable(reason),
        )
    }

    fn new(
        start_command_index: u32,
        end_command_index: u32,
        kind: DeviceExecutionSpanKind,
        measurement: DeviceExecutionSpanMeasurement,
    ) -> Option<Self> {
        if end_command_index <= start_command_index
            || (kind == DeviceExecutionSpanKind::EagerCommand
                && end_command_index != start_command_index.checked_add(1)?)
        {
            return None;
        }
        Some(Self {
            start_command_index,
            end_command_index,
            kind,
            measurement,
        })
    }

    fn from_command(command: DeviceCommandExecutionTiming) -> Option<Self> {
        let end_command_index = command.command_index.checked_add(1)?;
        Self::measured(
            command.command_index,
            end_command_index,
            DeviceExecutionSpanKind::EagerCommand,
            command.intervals.into_vec(),
        )
    }

    pub const fn start_command_index(&self) -> u32 {
        self.start_command_index
    }

    pub const fn end_command_index(&self) -> u32 {
        self.end_command_index
    }

    pub const fn command_count(&self) -> u32 {
        self.end_command_index - self.start_command_index
    }

    pub const fn kind(&self) -> DeviceExecutionSpanKind {
        self.kind
    }

    pub const fn measurement(&self) -> &DeviceExecutionSpanMeasurement {
        &self.measurement
    }

    pub const fn contains_command(&self, command_index: u32) -> bool {
        command_index >= self.start_command_index && command_index < self.end_command_index
    }
}

/// Terminal backend-counter evidence for one exact submission. Physical spans
/// cover every core command exactly once, remain ordered, and may explicitly
/// mark a range unavailable without discarding measured sibling spans.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeviceSubmissionExecutionTiming {
    command_count: u32,
    spans: Box<[DeviceSubmissionExecutionSpan]>,
}

impl DeviceSubmissionExecutionTiming {
    pub fn new(commands: Vec<DeviceCommandExecutionTiming>) -> Option<Self> {
        let command_count = commands.last()?.command_index().checked_add(1)?;
        let spans = commands
            .into_iter()
            .map(DeviceSubmissionExecutionSpan::from_command)
            .collect::<Option<Vec<_>>>()?;
        Self::from_spans(command_count, spans)
    }

    pub fn from_spans(
        command_count: u32,
        spans: Vec<DeviceSubmissionExecutionSpan>,
    ) -> Option<Self> {
        if command_count == 0 || spans.is_empty() {
            return None;
        }
        let mut expected_start = 0_u32;
        for span in &spans {
            if span.start_command_index() != expected_start
                || span.end_command_index() > command_count
            {
                return None;
            }
            expected_start = span.end_command_index();
        }
        if expected_start != command_count {
            return None;
        }
        Some(Self {
            command_count,
            spans: spans.into_boxed_slice(),
        })
    }

    pub const fn command_count(&self) -> u32 {
        self.command_count
    }

    pub fn spans(&self) -> &[DeviceSubmissionExecutionSpan] {
        &self.spans
    }

    pub fn span_for_command(&self, command_index: u32) -> Option<&DeviceSubmissionExecutionSpan> {
        let index = self
            .spans
            .partition_point(|span| span.end_command_index() <= command_index);
        self.spans
            .get(index)
            .filter(|span| span.contains_command(command_index))
    }
}

/// A terminal and its optional backend clock evidence are inseparable. This
/// prevents timing from being queried before the exact fence proves quiescence.
#[derive(Debug, Serialize)]
#[must_use = "a device terminal receipt owns exact fence timing evidence"]
pub struct DeviceTerminalReceipt<E> {
    terminal: DeviceTerminal<E>,
    execution_timing: DeviceTimingMeasurement<DeviceExecutionTiming>,
    submission_timing: DeviceTimingMeasurement<DeviceSubmissionExecutionTiming>,
}

impl<E> DeviceTerminalReceipt<E> {
    pub fn unprofiled(terminal: DeviceTerminal<E>) -> Self {
        Self {
            terminal,
            execution_timing: DeviceTimingMeasurement::NotRequested,
            submission_timing: DeviceTimingMeasurement::NotRequested,
        }
    }

    pub fn profiled(
        terminal: DeviceTerminal<E>,
        execution_timing: DeviceTimingMeasurement<DeviceExecutionTiming>,
    ) -> Self {
        Self {
            terminal,
            execution_timing,
            submission_timing: DeviceTimingMeasurement::NotRequested,
        }
    }

    pub fn profiled_with_submission_timing(
        terminal: DeviceTerminal<E>,
        execution_timing: DeviceTimingMeasurement<DeviceExecutionTiming>,
        submission_timing: DeviceTimingMeasurement<DeviceSubmissionExecutionTiming>,
    ) -> Self {
        Self {
            terminal,
            execution_timing,
            submission_timing,
        }
    }

    pub const fn terminal(&self) -> &DeviceTerminal<E> {
        &self.terminal
    }

    pub const fn execution_timing(&self) -> &DeviceTimingMeasurement<DeviceExecutionTiming> {
        &self.execution_timing
    }

    pub const fn submission_timing(
        &self,
    ) -> &DeviceTimingMeasurement<DeviceSubmissionExecutionTiming> {
        &self.submission_timing
    }

    pub fn into_parts(
        self,
    ) -> (
        DeviceTerminal<E>,
        DeviceTimingMeasurement<DeviceExecutionTiming>,
        DeviceTimingMeasurement<DeviceSubmissionExecutionTiming>,
    ) {
        (self.terminal, self.execution_timing, self.submission_timing)
    }
}

/// A submit failure that guarantees no device-visible work was enqueued.
///
/// This wrapper is deliberately distinct from an arbitrary runtime error:
/// only this outcome may release prepared resources or authorize an exact
/// retry without first reaching a fence terminal state.
#[derive(Debug, Serialize)]
#[must_use = "a definitely-not-submitted failure owns the only safe retry classification"]
pub struct DefinitelyNotSubmitted<E> {
    error: E,
}

impl<E> DefinitelyNotSubmitted<E> {
    pub fn new(error: E) -> Self {
        Self { error }
    }

    pub fn error(&self) -> &E {
        &self.error
    }

    pub fn into_error(self) -> E {
        self.error
    }
}

/// A quiescent device terminal. Both variants prove that command-owned
/// buffers are no longer accessed by the device.
#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case", tag = "status", content = "error")]
#[must_use = "a device terminal determines whether in-flight resources are release-safe"]
pub enum DeviceTerminal<E> {
    Succeeded,
    FailedButQuiescent(E),
}

impl<E> DeviceTerminal<E> {
    pub const fn is_succeeded(&self) -> bool {
        matches!(self, Self::Succeeded)
    }
}

/// Non-blocking fence observation. An indeterminate query retains the fence
/// and routes ownership to blocking recovery; it is not a terminal failure.
#[derive(Debug, Serialize)]
#[serde(rename_all = "snake_case", tag = "status", content = "detail")]
#[must_use = "a fence query must preserve pending or indeterminate ownership"]
pub enum FenceQuery<E> {
    Pending,
    Terminal(DeviceTerminalReceipt<E>),
    Indeterminate(E),
}

impl<E> FenceQuery<E> {
    pub const fn is_pending(&self) -> bool {
        matches!(self, Self::Pending)
    }
}

/// Blocking wait could not prove fence quiescence. The fence and all
/// in-flight ownership must remain retained for lane recovery or quarantine.
#[derive(Debug, Serialize)]
#[must_use = "an indeterminate fence retains recovery and quarantine ownership"]
pub struct FenceIndeterminate<E> {
    error: E,
}

impl<E> FenceIndeterminate<E> {
    pub fn new(error: E) -> Self {
        Self { error }
    }

    pub fn error(&self) -> &E {
        &self.error
    }

    pub fn into_error(self) -> E {
        self.error
    }
}

/// Backend-provided description of one device error. The backend cannot pick
/// a failure domain or execution identity; core attaches both after checking
/// the concrete runtime device.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeviceErrorReport {
    failure: FailureEnvelope,
}

impl DeviceErrorReport {
    pub fn new(
        code: impl Into<String>,
        message: impl Into<String>,
        retryable: bool,
    ) -> Result<Self, VNextError> {
        Ok(Self {
            failure: FailureEnvelope::new(FailureDomain::Device, code, message, retryable)?,
        })
    }

    pub fn code(&self) -> &str {
        self.failure.code()
    }

    pub fn message(&self) -> &str {
        self.failure.message()
    }

    pub const fn retryable(&self) -> bool {
        self.failure.retryable()
    }

    fn into_failure(self) -> FailureEnvelope {
        self.failure
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct HostTransferLayout {
    element_type: ElementType,
    element_count: u64,
}

impl HostTransferLayout {
    pub fn new(element_type: ElementType, element_count: u64) -> Result<Self, super::VNextError> {
        if element_count == 0
            || element_count
                .checked_mul(element_type.size_bytes())
                .is_none()
        {
            return Err(super::VNextError::InvalidExecutionPlan {
                reason: "host transfer layout is empty or overflows u64".to_owned(),
            });
        }
        Ok(Self {
            element_type,
            element_count,
        })
    }

    pub fn byte_len(self) -> Result<u64, super::VNextError> {
        self.element_count
            .checked_mul(self.element_type.size_bytes())
            .ok_or_else(|| super::VNextError::InvalidExecutionPlan {
                reason: "host transfer byte count overflows u64".to_owned(),
            })
    }

    pub fn validate_bytes(self, bytes: usize) -> Result<(), super::VNextError> {
        if self.byte_len()? != bytes as u64 {
            return Err(super::VNextError::InvalidExecutionPlan {
                reason: "host transfer byte count does not match its element layout".to_owned(),
            });
        }
        Ok(())
    }

    pub fn element_type(self) -> ElementType {
        self.element_type
    }

    pub fn element_count(self) -> u64 {
        self.element_count
    }
}

/// Semantic phase of one command inside a core-owned submission batch.
///
/// Backends may use this phase to compile reusable device executables, but
/// they must preserve the original ordering and may only reuse `Compute`
/// commands whose backend provider supplied an exact replay contract.
/// Initialization, dynamic binding, and result binding commands are explicit
/// eager barriers: replaying them can reset live state, reuse stale request
/// data, or write results into an earlier request's backing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceCommandPhase {
    Initialization,
    DynamicBinding,
    Compute,
    ResultBinding,
}

/// Backend-observed physical work for one core-owned command entry.
///
/// Rows are created only in `DeviceTimingMode::Kernel`. The node index is
/// issued by core and binds backend work back to the immutable plan; backend
/// labels and counters carry observation only and grant no execution authority.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeviceNativeWorkAttribution {
    command_index: u32,
    node_index: Option<u32>,
    command_phase: DeviceCommandPhase,
    native_op_id: &'static str,
    execution_path: DeviceExecutionPath,
    batching_form: DeviceBatchingForm,
    participant_count: u32,
    token_count: u64,
    compute_dispatch_count: u64,
    transfer_command_count: u64,
}

impl DeviceNativeWorkAttribution {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        command_index: u32,
        node_index: Option<u32>,
        command_phase: DeviceCommandPhase,
        native_op_id: &'static str,
        execution_path: DeviceExecutionPath,
        batching_form: DeviceBatchingForm,
        participant_count: u32,
        token_count: u64,
        compute_dispatch_count: u64,
        transfer_command_count: u64,
    ) -> Option<Self> {
        if native_op_id.is_empty()
            || (compute_dispatch_count == 0 && transfer_command_count == 0)
            || (node_index.is_some() && participant_count == 0)
        {
            return None;
        }
        Some(Self {
            command_index,
            node_index,
            command_phase,
            native_op_id,
            execution_path,
            batching_form,
            participant_count,
            token_count,
            compute_dispatch_count,
            transfer_command_count,
        })
    }

    pub const fn command_index(&self) -> u32 {
        self.command_index
    }

    pub const fn node_index(&self) -> Option<u32> {
        self.node_index
    }

    pub const fn command_phase(&self) -> DeviceCommandPhase {
        self.command_phase
    }

    pub const fn native_op_id(&self) -> &'static str {
        self.native_op_id
    }

    pub const fn execution_path(&self) -> DeviceExecutionPath {
        self.execution_path
    }

    pub const fn batching_form(&self) -> DeviceBatchingForm {
        self.batching_form
    }

    pub const fn participant_count(&self) -> u32 {
        self.participant_count
    }

    pub const fn token_count(&self) -> u64 {
        self.token_count
    }

    pub const fn compute_dispatch_count(&self) -> u64 {
        self.compute_dispatch_count
    }

    pub const fn transfer_command_count(&self) -> u64 {
        self.transfer_command_count
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeviceSubmissionAttribution {
    commands: Box<[DeviceNativeWorkAttribution]>,
}

impl DeviceSubmissionAttribution {
    pub fn new(commands: Vec<DeviceNativeWorkAttribution>) -> Option<Self> {
        if commands.is_empty()
            || commands
                .windows(2)
                .any(|pair| pair[0].command_index() >= pair[1].command_index())
        {
            return None;
        }
        Some(Self {
            commands: commands.into_boxed_slice(),
        })
    }

    pub fn commands(&self) -> &[DeviceNativeWorkAttribution] {
        &self.commands
    }
}

/// Provider-encoded work for one logical operation.
///
/// Core owns the phase boundaries: request-specific inputs are written before
/// compute, and request-specific outputs are materialized afterwards. A
/// backend may optimize the compute command, but cannot accidentally capture
/// either dynamic boundary into a reusable executable.
#[must_use = "encoded device operations must be appended to a submission batch"]
pub struct EncodedDeviceOperation<C> {
    program_bindings: Vec<C>,
    dynamic_bindings: Vec<C>,
    compute: C,
    result_bindings: Vec<C>,
}

impl<C> EncodedDeviceOperation<C> {
    pub fn compute(command: C) -> Self {
        Self {
            program_bindings: Vec::new(),
            dynamic_bindings: Vec::new(),
            compute: command,
            result_bindings: Vec::new(),
        }
    }

    /// Adds a binding that may execute in the wave-level program prelude.
    ///
    /// Providers may use this only for writes into their own non-aliasing
    /// binding workspace. The command must not read provider outputs or depend
    /// on earlier compute in the same wave.
    pub fn with_program_binding(mut self, command: C) -> Self {
        self.program_bindings.push(command);
        self
    }

    pub fn with_dynamic_binding(mut self, command: C) -> Self {
        self.dynamic_bindings.push(command);
        self
    }

    pub fn with_result_binding(mut self, command: C) -> Self {
        self.result_bindings.push(command);
        self
    }

    pub fn dynamic_binding_count(&self) -> usize {
        self.dynamic_bindings.len()
    }

    pub fn program_binding_count(&self) -> usize {
        self.program_bindings.len()
    }

    pub fn result_binding_count(&self) -> usize {
        self.result_bindings.len()
    }

    pub(crate) fn into_parts(self) -> (Vec<C>, Vec<C>, C, Vec<C>) {
        (
            self.program_bindings,
            self.dynamic_bindings,
            self.compute,
            self.result_bindings,
        )
    }
}

/// One command plus the core-issued semantic phase that constrains backend
/// execution optimizations.
pub struct DeviceCommandEntry<C> {
    phase: DeviceCommandPhase,
    node_index: Option<u32>,
    command: C,
}

impl<C> DeviceCommandEntry<C> {
    pub const fn phase(&self) -> DeviceCommandPhase {
        self.phase
    }

    pub const fn node_index(&self) -> Option<u32> {
        self.node_index
    }

    pub const fn command(&self) -> &C {
        &self.command
    }

    pub fn into_parts(self) -> (DeviceCommandPhase, Option<u32>, C) {
        (self.phase, self.node_index, self.command)
    }
}

/// Core-owned physical submission unit.
///
/// Operation providers produce individual commands, while the execution
/// runtime decides which commands share one ordered device submission and
/// completion fence. Construction stays private to core so a backend cannot
/// silently split one admitted lane segment into unrelated submissions.
#[must_use = "encoded device command batches must be submitted"]
pub struct DeviceCommandBatch<C> {
    commands: Vec<DeviceCommandEntry<C>>,
    timing_mode: DeviceTimingMode,
}

impl<C> DeviceCommandBatch<C> {
    pub(crate) fn singleton(command: C) -> Self {
        Self {
            commands: vec![DeviceCommandEntry {
                phase: DeviceCommandPhase::Compute,
                node_index: None,
                command,
            }],
            timing_mode: DeviceTimingMode::Off,
        }
    }

    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            commands: Vec::with_capacity(capacity),
            timing_mode: DeviceTimingMode::Off,
        }
    }

    pub(crate) fn with_capacity_and_timing(capacity: usize, timing_mode: DeviceTimingMode) -> Self {
        Self {
            commands: Vec::with_capacity(capacity),
            timing_mode,
        }
    }

    pub(crate) fn push_initialization(&mut self, command: C) {
        self.commands.push(DeviceCommandEntry {
            phase: DeviceCommandPhase::Initialization,
            node_index: None,
            command,
        });
    }

    pub(crate) fn push_dynamic_binding(&mut self, command: C) {
        self.commands.push(DeviceCommandEntry {
            phase: DeviceCommandPhase::DynamicBinding,
            node_index: None,
            command,
        });
    }

    pub(crate) fn push_compute(&mut self, command: C) {
        self.commands.push(DeviceCommandEntry {
            phase: DeviceCommandPhase::Compute,
            node_index: None,
            command,
        });
    }

    pub(crate) fn push_result_binding(&mut self, command: C) {
        self.commands.push(DeviceCommandEntry {
            phase: DeviceCommandPhase::ResultBinding,
            node_index: None,
            command,
        });
    }

    pub(crate) fn push_operation(&mut self, node_index: u32, operation: EncodedDeviceOperation<C>) {
        let (program_bindings, dynamic_bindings, compute, result_bindings) = operation.into_parts();
        for command in program_bindings {
            self.commands.push(DeviceCommandEntry {
                phase: DeviceCommandPhase::DynamicBinding,
                node_index: Some(node_index),
                command,
            });
        }
        self.push_operation_parts(node_index, dynamic_bindings, compute, result_bindings);
    }

    pub(crate) fn push_operation_parts(
        &mut self,
        node_index: u32,
        dynamic_bindings: Vec<C>,
        compute: C,
        result_bindings: Vec<C>,
    ) {
        for command in dynamic_bindings {
            self.commands.push(DeviceCommandEntry {
                phase: DeviceCommandPhase::DynamicBinding,
                node_index: Some(node_index),
                command,
            });
        }
        self.commands.push(DeviceCommandEntry {
            phase: DeviceCommandPhase::Compute,
            node_index: Some(node_index),
            command: compute,
        });
        for command in result_bindings {
            self.commands.push(DeviceCommandEntry {
                phase: DeviceCommandPhase::ResultBinding,
                node_index: Some(node_index),
                command,
            });
        }
    }

    pub(crate) fn push(&mut self, command: C) {
        self.push_compute(command);
    }

    pub fn len(&self) -> usize {
        self.commands.len()
    }

    pub fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    pub const fn timing_mode(&self) -> DeviceTimingMode {
        self.timing_mode
    }

    pub fn into_commands(self) -> Vec<C> {
        self.commands
            .into_iter()
            .map(|entry| entry.command)
            .collect()
    }

    pub fn into_entries(self) -> Vec<DeviceCommandEntry<C>> {
        self.commands
    }
}

/// Cold-path transaction for backends that can bind immutable weight
/// components directly instead of allocating and uploading one contiguous
/// physical arena.
///
/// The session must not publish a partially imported arena. [`Self::seal`]
/// consumes the complete transaction and is the only point at which imported
/// regions may become visible to device execution.
pub trait StaticWeightImportSession<B, E> {
    fn import_component(
        &mut self,
        payload: &WeightComponentPayload<'_>,
        destination: &B,
        destination_offset_bytes: u64,
    ) -> Result<(), E>;

    fn seal(self: Box<Self>) -> Result<(), E>;
}

/// Stable primitive boundary implemented by a concrete device runtime.
///
/// Associated buffer, stream, command, and error types preserve compile-time
/// type safety. Every operation is required; unsupported work cannot inherit a
/// success-returning default implementation.
pub trait DeviceRuntime: Send + Sync + 'static {
    type Buffer: Send + Sync + 'static;
    type Stream: Send + 'static;
    type Command: Send + 'static;
    type Fence: Send + 'static;
    type Error: Error + Send + Sync + 'static;

    fn descriptor(&self) -> &DeviceDescriptor;

    /// Allocates only after the resource transaction has authorized the exact
    /// request. `DeviceAllocationPermit` has no public constructor and borrows
    /// the live transaction context, so raw device allocation cannot bypass
    /// admission or outlive the transaction action that authorized it.
    fn allocate(&self, permit: DeviceAllocationPermit<'_>) -> Result<Self::Buffer, Self::Error>;

    fn buffer_descriptor(&self, buffer: &Self::Buffer) -> BufferDescriptor;

    /// Begins an optional all-or-nothing static-weight import transaction.
    /// Returning `None` selects the portable zero-and-upload path. The default
    /// preserves existing CUDA, CPU, and test runtime behavior.
    fn begin_static_weight_import(
        &self,
    ) -> Option<
        Result<Box<dyn StaticWeightImportSession<Self::Buffer, Self::Error> + '_>, Self::Error>,
    > {
        None
    }

    fn create_stream(&self) -> Result<Self::Stream, Self::Error>;

    fn stream_state(&self, stream: &Self::Stream) -> StreamState;

    /// Opens the bounded cold-path preparation window for one stream.
    /// Backends without reusable executable support retain the no-op receipt.
    fn configure_reusable_executables(
        &self,
        _stream: &mut Self::Stream,
        _plan: DeviceReusableExecutionPlan,
    ) -> Result<DeviceReusableExecutionPreparation, Self::Error> {
        Ok(DeviceReusableExecutionPreparation::unsupported())
    }

    /// Permanently closes the preparation window for this stream. A sealed
    /// stream may replay or fall back to eager execution but cannot capture on
    /// a later product request.
    fn seal_reusable_executables(
        &self,
        _stream: &mut Self::Stream,
    ) -> Result<DeviceReusableExecutionPreparation, Self::Error> {
        Ok(DeviceReusableExecutionPreparation::unsupported())
    }

    /// Returns the current preparation receipt without changing lifecycle
    /// state. Product startup uses two snapshots to prove that its validation
    /// pass replayed stable executables instead of compiling more work.
    fn reusable_executable_preparation(
        &self,
        _stream: &Self::Stream,
    ) -> Result<DeviceReusableExecutionPreparation, Self::Error> {
        Ok(DeviceReusableExecutionPreparation::unsupported())
    }

    /// Releases reusable executable cache entries on a proven-quiescent
    /// stream. Backends without such a cache retain the no-op default.
    fn trim_reusable_executables(
        &self,
        _stream: &mut Self::Stream,
    ) -> Result<DeviceReusableExecutionTrim, Self::Error> {
        Ok(DeviceReusableExecutionTrim::default())
    }

    fn encode_copy(
        &self,
        source: &Self::Buffer,
        destination: &Self::Buffer,
        region: CopyRegion,
    ) -> Result<Self::Command, Self::Error>;

    fn encode_upload(
        &self,
        source: &[u8],
        source_layout: HostTransferLayout,
        destination: &Self::Buffer,
        destination_offset_bytes: u64,
    ) -> Result<Self::Command, Self::Error>;

    fn encode_zero(
        &self,
        destination: &Self::Buffer,
        destination_offset_bytes: u64,
        length_bytes: u64,
    ) -> Result<Self::Command, Self::Error>;

    /// Coalesces independent provider binding writes into a wave prelude.
    ///
    /// The default preserves one command per provider. Backends may return a
    /// smaller ordered set, but must retain every command-owned resource and
    /// preserve the exact enqueue order and failure semantics.
    fn coalesce_program_bindings(
        &self,
        commands: Vec<Self::Command>,
    ) -> Result<Vec<Self::Command>, Self::Error> {
        Ok(commands)
    }

    /// Submits one non-empty ordered command batch and returns its exact
    /// completion fence. A backend must preserve command order and must not
    /// manufacture intermediate host-visible completion boundaries.
    ///
    /// The error type is intentionally closed over `DefinitelyNotSubmitted`:
    /// an ordinary backend error is not sufficient evidence that invocation
    /// resources may be released or retried. Backends that cannot prove that
    /// no work was enqueued must panic or retain/return a fence through their
    /// implementation boundary; core treats an unwind as possibly submitted.
    fn submit(
        &self,
        stream: &mut Self::Stream,
        commands: DeviceCommandBatch<Self::Command>,
    ) -> Result<Self::Fence, DefinitelyNotSubmitted<Self::Error>>;

    /// Profile-attached submission entrypoint. Backends override this only
    /// when they can expose typed internal boundaries without changing
    /// submission ownership or error semantics.
    fn submit_with_timing<S>(
        &self,
        stream: &mut Self::Stream,
        commands: DeviceCommandBatch<Self::Command>,
        timing_sink: &S,
    ) -> Result<Self::Fence, DefinitelyNotSubmitted<Self::Error>>
    where
        Self: Sized,
        S: DeviceSubmissionTimingSink,
    {
        let _ = timing_sink;
        self.submit(stream, commands)
    }

    /// Returns backend-observed native work for an already submitted fence.
    /// Only `DeviceTimingMode::Kernel` fences may carry attribution. The
    /// returned rows are diagnostic evidence and never grant completion or
    /// resource-release authority.
    fn submission_attribution(&self, _fence: &Self::Fence) -> Option<DeviceSubmissionAttribution> {
        None
    }

    /// Observes a fence without blocking. `Indeterminate` is not terminal and
    /// therefore cannot release command-owned resources.
    fn query_fence(&self, fence: &Self::Fence) -> FenceQuery<Self::Error>;

    /// Waits for a quiescent terminal. Failure to prove quiescence retains the
    /// fence and every resource reachable from the submitted invocation.
    fn wait_fence(
        &self,
        fence: &Self::Fence,
    ) -> Result<DeviceTerminalReceipt<Self::Error>, FenceIndeterminate<Self::Error>>;

    fn synchronize(&self, stream: &mut Self::Stream) -> Result<(), Self::Error>;

    fn readback(
        &self,
        stream: &mut Self::Stream,
        source: &Self::Buffer,
        region: CopyRegion,
        output_layout: HostTransferLayout,
    ) -> Result<Vec<u8>, Self::Error>;

    fn describe_error(&self, error: &Self::Error) -> Result<DeviceErrorReport, VNextError>;
}

/// Closes a backend error over the exact runtime instance and a core-owned
/// device failure domain.
pub fn classify_device_error<R: DeviceRuntime + ?Sized>(
    runtime: &R,
    identity: ExecutionIdentityEnvelope,
    error: &R::Error,
) -> Result<IdentifiedFailure, VNextError> {
    runtime.descriptor().validate()?;
    if identity.parts().device_id.as_ref() != Some(&runtime.descriptor().id)
        || identity
            .parts()
            .runtime_implementation_fingerprint
            .as_deref()
            != Some(
                runtime
                    .descriptor()
                    .runtime_implementation_fingerprint
                    .as_str(),
            )
    {
        return Err(VNextError::InvalidExecutionPlan {
            reason: "device error identity differs from the concrete runtime device implementation"
                .to_owned(),
        });
    }
    IdentifiedFailure::new(identity, runtime.describe_error(error)?.into_failure())
}

#[cfg(test)]
mod execution_timing_tests {
    use super::*;

    #[test]
    fn command_timing_requires_positive_ordered_nonoverlapping_intervals() {
        assert!(
            DeviceExecutionInterval::new(DeviceExecutionIntervalKind::Compute, 10, 10).is_none()
        );
        assert!(
            DeviceExecutionInterval::new(DeviceExecutionIntervalKind::Compute, 11, 10).is_none()
        );

        let first =
            DeviceExecutionInterval::new(DeviceExecutionIntervalKind::Compute, 10, 20).unwrap();
        let adjacent =
            DeviceExecutionInterval::new(DeviceExecutionIntervalKind::Transfer, 20, 30).unwrap();
        let overlapping =
            DeviceExecutionInterval::new(DeviceExecutionIntervalKind::Transfer, 19, 30).unwrap();
        assert!(DeviceCommandExecutionTiming::new(0, vec![first, adjacent]).is_some());
        assert!(DeviceCommandExecutionTiming::new(0, vec![first, overlapping]).is_none());
        let labeled = DeviceExecutionInterval::new_labeled(
            DeviceExecutionIntervalKind::Compute,
            30,
            40,
            "projection.qkv",
        )
        .unwrap();
        assert_eq!(labeled.subwork_id(), Some("projection.qkv"));
        assert!(DeviceExecutionInterval::new_labeled(
            DeviceExecutionIntervalKind::Compute,
            30,
            40,
            "",
        )
        .is_none());
    }

    #[test]
    fn submission_timing_requires_complete_nonoverlapping_command_coverage() {
        let command = |command_index| {
            DeviceCommandExecutionTiming::new(
                command_index,
                vec![
                    DeviceExecutionInterval::new(DeviceExecutionIntervalKind::Compute, 0, 1)
                        .unwrap(),
                ],
            )
            .unwrap()
        };
        let commands = DeviceSubmissionExecutionTiming::new(vec![command(0), command(1)]).unwrap();
        assert_eq!(commands.command_count(), 2);
        assert_eq!(commands.spans().len(), 2);
        assert!(commands
            .spans()
            .iter()
            .all(|span| span.kind() == DeviceExecutionSpanKind::EagerCommand));
        assert!(DeviceSubmissionExecutionTiming::new(vec![command(0), command(2)]).is_none());
        assert!(DeviceSubmissionExecutionTiming::new(vec![command(1), command(1)]).is_none());
        assert!(DeviceSubmissionExecutionTiming::new(vec![command(2), command(1)]).is_none());
    }

    #[test]
    fn submission_timing_preserves_measured_and_unavailable_physical_spans() {
        let eager = DeviceSubmissionExecutionSpan::measured(
            0,
            1,
            DeviceExecutionSpanKind::EagerCommand,
            vec![
                DeviceExecutionInterval::new(DeviceExecutionIntervalKind::Transfer, 0, 10).unwrap(),
            ],
        )
        .unwrap();
        let replay = DeviceSubmissionExecutionSpan::measured(
            1,
            4,
            DeviceExecutionSpanKind::ReusableExecutable,
            vec![DeviceExecutionInterval::new_labeled(
                DeviceExecutionIntervalKind::Compute,
                10,
                40,
                "cuda reusable executable",
            )
            .unwrap()],
        )
        .unwrap();
        let unavailable = DeviceSubmissionExecutionSpan::unavailable(
            4,
            5,
            DeviceExecutionSpanKind::EagerCommand,
            DeviceTimingUnavailableReason::BackendMeasurementFailed,
        )
        .unwrap();

        let timing =
            DeviceSubmissionExecutionTiming::from_spans(5, vec![eager, replay, unavailable])
                .unwrap();
        assert_eq!(timing.command_count(), 5);
        assert_eq!(
            timing.span_for_command(2).unwrap().kind(),
            DeviceExecutionSpanKind::ReusableExecutable
        );
        assert_eq!(
            timing
                .span_for_command(4)
                .unwrap()
                .measurement()
                .unavailable_reason(),
            Some(DeviceTimingUnavailableReason::BackendMeasurementFailed)
        );
        assert!(timing.span_for_command(5).is_none());
    }

    #[test]
    fn physical_spans_reject_gaps_overlaps_and_invalid_eager_ranges() {
        let interval = || {
            vec![DeviceExecutionInterval::new(DeviceExecutionIntervalKind::Compute, 0, 1).unwrap()]
        };
        assert!(DeviceSubmissionExecutionSpan::measured(
            0,
            2,
            DeviceExecutionSpanKind::EagerCommand,
            interval(),
        )
        .is_none());
        let first = DeviceSubmissionExecutionSpan::measured(
            0,
            1,
            DeviceExecutionSpanKind::EagerCommand,
            interval(),
        )
        .unwrap();
        let gap = DeviceSubmissionExecutionSpan::measured(
            2,
            3,
            DeviceExecutionSpanKind::EagerCommand,
            interval(),
        )
        .unwrap();
        assert!(DeviceSubmissionExecutionTiming::from_spans(3, vec![first.clone(), gap]).is_none());
        let overlap = DeviceSubmissionExecutionSpan::measured(
            0,
            2,
            DeviceExecutionSpanKind::ReusableExecutable,
            interval(),
        )
        .unwrap();
        assert!(DeviceSubmissionExecutionTiming::from_spans(2, vec![first, overlap]).is_none());
    }
}

#[cfg(test)]
mod deferred_cleanup_tests {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::sync::{Arc, Barrier};

    struct ScriptedCleanup {
        outcomes: VecDeque<DeferredDeviceCleanupDisposition>,
        attempts: Arc<AtomicUsize>,
        dropped: Arc<AtomicBool>,
    }

    impl DeferredDeviceCleanupTask for ScriptedCleanup {
        fn try_cleanup(&mut self) -> DeferredDeviceCleanupDisposition {
            self.attempts.fetch_add(1, Ordering::AcqRel);
            self.outcomes
                .pop_front()
                .unwrap_or(DeferredDeviceCleanupDisposition::Completed)
        }
    }

    impl Drop for ScriptedCleanup {
        fn drop(&mut self) {
            self.dropped.store(true, Ordering::Release);
        }
    }

    struct PanicOnceCleanup {
        first: bool,
        attempts: Arc<AtomicUsize>,
    }

    impl DeferredDeviceCleanupTask for PanicOnceCleanup {
        fn try_cleanup(&mut self) -> DeferredDeviceCleanupDisposition {
            self.attempts.fetch_add(1, Ordering::AcqRel);
            if self.first {
                self.first = false;
                panic!("injected deferred cleanup panic");
            }
            DeferredDeviceCleanupDisposition::Completed
        }
    }

    struct BlockingCleanup {
        entered: Arc<Barrier>,
        release: Arc<Barrier>,
    }

    impl DeferredDeviceCleanupTask for BlockingCleanup {
        fn try_cleanup(&mut self) -> DeferredDeviceCleanupDisposition {
            self.entered.wait();
            self.release.wait();
            DeferredDeviceCleanupDisposition::Completed
        }
    }

    fn scripted(
        outcomes: impl IntoIterator<Item = DeferredDeviceCleanupDisposition>,
    ) -> (ScriptedCleanup, Arc<AtomicUsize>, Arc<AtomicBool>) {
        let attempts = Arc::new(AtomicUsize::new(0));
        let dropped = Arc::new(AtomicBool::new(false));
        (
            ScriptedCleanup {
                outcomes: outcomes.into_iter().collect(),
                attempts: Arc::clone(&attempts),
                dropped: Arc::clone(&dropped),
            },
            attempts,
            dropped,
        )
    }

    #[test]
    fn encoded_operation_preserves_program_dynamic_compute_and_result_boundaries() {
        let operation = EncodedDeviceOperation::compute("compute")
            .with_program_binding("program-bind")
            .with_dynamic_binding("bind-a")
            .with_dynamic_binding("bind-b")
            .with_result_binding("writeback");
        let mut batch = DeviceCommandBatch::with_capacity(5);
        batch.push_operation(0, operation);

        let entries = batch.into_entries();
        assert_eq!(
            entries
                .iter()
                .map(DeviceCommandEntry::phase)
                .collect::<Vec<_>>(),
            vec![
                DeviceCommandPhase::DynamicBinding,
                DeviceCommandPhase::DynamicBinding,
                DeviceCommandPhase::DynamicBinding,
                DeviceCommandPhase::Compute,
                DeviceCommandPhase::ResultBinding,
            ]
        );
        assert_eq!(
            entries
                .into_iter()
                .map(DeviceCommandEntry::into_parts)
                .map(|(_, _, command)| command)
                .collect::<Vec<_>>(),
            vec!["program-bind", "bind-a", "bind-b", "compute", "writeback",]
        );
    }

    #[test]
    fn reusable_execution_observation_preserves_each_fallback_and_replay_counter() {
        let mut observation = DeviceReusableExecutionObservation::default();
        observation.observe_candidate_segment();
        observation.observe_captured_segment();
        observation.observe_uploaded_segment();
        observation.observe_cache_hit_segment();
        observation.observe_cached_rejected_segment();
        observation.observe_capture_rejection();
        observation.observe_quiescence_deferred_segment();
        observation.observe_capacity_deferred_segment();
        observation.observe_outside_preparation_segment();
        observation.observe_evicted_segment();
        observation.observe_replayed_segment(3);
        observation.observe_eager_command();

        assert_eq!(observation.candidate_segments(), 1);
        assert_eq!(observation.captured_segments(), 1);
        assert_eq!(observation.uploaded_segments(), 1);
        assert_eq!(observation.cache_hit_segments(), 1);
        assert_eq!(observation.cached_rejected_segments(), 1);
        assert_eq!(observation.capture_rejected_segments(), 1);
        assert_eq!(observation.quiescence_deferred_segments(), 1);
        assert_eq!(observation.capacity_deferred_segments(), 1);
        assert_eq!(observation.outside_preparation_segments(), 1);
        assert_eq!(observation.evicted_segments(), 1);
        assert_eq!(observation.replayed_segments(), 1);
        assert_eq!(observation.replayed_commands(), 3);
        assert_eq!(observation.eager_commands(), 1);

        let value = serde_json::to_value(observation).expect("observation serializes");
        for field in [
            "candidate_segments",
            "captured_segments",
            "uploaded_segments",
            "cache_hit_segments",
            "cached_rejected_segments",
            "capture_rejected_segments",
            "quiescence_deferred_segments",
            "capacity_deferred_segments",
            "outside_preparation_segments",
            "evicted_segments",
            "replayed_segments",
            "eager_commands",
        ] {
            assert_eq!(value[field], 1, "counter {field} must remain typed");
        }
        assert_eq!(value["replayed_commands"], 3);
    }

    #[test]
    fn retryable_and_quarantined_cleanup_owners_remain_reachable() {
        for first in [
            DeferredDeviceCleanupDisposition::Retryable,
            DeferredDeviceCleanupDisposition::Quarantined,
        ] {
            let domain = new_deferred_device_cleanup_domain();
            let (task, attempts, dropped) =
                scripted([first, DeferredDeviceCleanupDisposition::Completed]);
            defer_device_cleanup(domain, task);

            let first_receipt = maintain_deferred_device_cleanups(domain, 1);
            assert_eq!(first_receipt.attempted(), 1);
            assert_eq!(first_receipt.completed(), 0);
            assert_eq!(first_receipt.status_after().pending(), 1);
            assert!(!dropped.load(Ordering::Acquire));

            let second_receipt = maintain_deferred_device_cleanups(domain, 1);
            assert_eq!(second_receipt.completed(), 1);
            assert_eq!(second_receipt.status_after().pending(), 0);
            assert_eq!(attempts.load(Ordering::Acquire), 2);
            assert!(dropped.load(Ordering::Acquire));
            assert!(retire_deferred_device_cleanup_domain(domain));
        }
    }

    #[test]
    fn panicking_cleanup_owner_is_retried_in_place() {
        let domain = new_deferred_device_cleanup_domain();
        let attempts = Arc::new(AtomicUsize::new(0));
        defer_device_cleanup(
            domain,
            PanicOnceCleanup {
                first: true,
                attempts: Arc::clone(&attempts),
            },
        );

        let first = maintain_deferred_device_cleanups(domain, 1);
        assert_eq!(first.panicked(), 1);
        assert_eq!(first.status_after().panicked(), 1);
        assert_eq!(first.status_after().pending(), 1);
        let second = maintain_deferred_device_cleanups(domain, 1);
        assert_eq!(second.completed(), 1);
        assert_eq!(second.status_after().pending(), 0);
        assert_eq!(attempts.load(Ordering::Acquire), 2);
        assert!(retire_deferred_device_cleanup_domain(domain));
    }

    #[test]
    fn saturation_keeps_every_owner_and_bounds_each_maintenance_pass() {
        let domain = new_deferred_device_cleanup_domain();
        let task_count = MAX_DEFERRED_DEVICE_CLEANUP_TASKS + 1;
        for _ in 0..task_count {
            let (task, _, _) = scripted([DeferredDeviceCleanupDisposition::Completed]);
            defer_device_cleanup(domain, task);
        }
        let saturated = deferred_device_cleanup_status(domain);
        assert_eq!(saturated.pending(), task_count);
        assert!(saturated.is_saturated());

        let first = maintain_deferred_device_cleanups(
            domain,
            MAX_DEFERRED_DEVICE_CLEANUP_MAINTENANCE_TASKS,
        );
        assert_eq!(first.attempted(), MAX_DEFERRED_DEVICE_CLEANUP_TASKS);
        assert_eq!(first.completed(), MAX_DEFERRED_DEVICE_CLEANUP_TASKS);
        assert_eq!(first.status_after().pending(), 1);
        let second = maintain_deferred_device_cleanups(domain, 1);
        assert_eq!(second.completed(), 1);
        assert_eq!(second.status_after().pending(), 0);
        assert!(retire_deferred_device_cleanup_domain(domain));
    }

    #[test]
    fn blocked_cleanup_does_not_withhold_sibling_task_or_registry() {
        let domain = new_deferred_device_cleanup_domain();
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        defer_device_cleanup(
            domain,
            BlockingCleanup {
                entered: Arc::clone(&entered),
                release: Arc::clone(&release),
            },
        );
        let (ready, _, _) = scripted([DeferredDeviceCleanupDisposition::Completed]);
        defer_device_cleanup(domain, ready);

        let (blocked_receipt, ready_receipt) = std::thread::scope(|scope| {
            let worker = std::thread::Builder::new()
                .name("vnext-cleanup-domain-isolation".to_owned())
                .spawn_scoped(scope, move || maintain_deferred_device_cleanups(domain, 2))
                .expect("the single bounded cleanup isolation worker starts");
            entered.wait();
            let ready_receipt = maintain_deferred_device_cleanups(domain, 1);
            release.wait();
            let blocked_receipt = worker
                .join()
                .expect("the bounded cleanup isolation worker does not panic");
            (blocked_receipt, ready_receipt)
        });

        assert_eq!(ready_receipt.completed(), 1);
        assert_eq!(blocked_receipt.completed(), 1);
        assert_eq!(deferred_device_cleanup_status(domain).pending(), 0);
        assert!(retire_deferred_device_cleanup_domain(domain));
    }
}
