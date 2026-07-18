use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::error::Error;
use std::num::NonZeroU64;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, OnceLock};
use std::time::Duration;

use super::{
    CapabilityId, DeviceAllocationPermit, DeviceId, DynamicStorageProfile, ElementType,
    ExecutionIdentityEnvelope, FailureDomain, FailureEnvelope, IdentifiedFailure, VNextError,
};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BufferUsage {
    Weights,
    Activations,
    State,
    /// Provider/runtime workspace whose lifetime spans operations but is not
    /// model semantic state (for example packed metadata or persistent scratch).
    Persistent,
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
/// readback boundaries.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
#[serde(rename_all = "snake_case")]
pub enum DeviceTimingMode {
    #[default]
    Off,
    Completion,
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

/// Diagnostic-only sink for backend submission attribution.
///
/// `ENABLED = false` is the compile-time off path: a backend must not read a
/// clock or call `record_device_submission` in that specialization. Enabled
/// implementations run on the submission thread and must not block, allocate,
/// or panic.
pub trait DeviceSubmissionTimingSink: Send + Sync {
    const ENABLED: bool;

    fn record_device_submission(&self, stage: DeviceSubmissionStage, elapsed: Duration);
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

/// A terminal and its optional backend clock evidence are inseparable. This
/// prevents timing from being queried before the exact fence proves quiescence.
#[derive(Debug, Serialize)]
#[must_use = "a device terminal receipt owns exact fence timing evidence"]
pub struct DeviceTerminalReceipt<E> {
    terminal: DeviceTerminal<E>,
    execution_timing: DeviceTimingMeasurement<DeviceExecutionTiming>,
}

impl<E> DeviceTerminalReceipt<E> {
    pub fn unprofiled(terminal: DeviceTerminal<E>) -> Self {
        Self {
            terminal,
            execution_timing: DeviceTimingMeasurement::NotRequested,
        }
    }

    pub fn profiled(
        terminal: DeviceTerminal<E>,
        execution_timing: DeviceTimingMeasurement<DeviceExecutionTiming>,
    ) -> Self {
        Self {
            terminal,
            execution_timing,
        }
    }

    pub const fn terminal(&self) -> &DeviceTerminal<E> {
        &self.terminal
    }

    pub const fn execution_timing(&self) -> &DeviceTimingMeasurement<DeviceExecutionTiming> {
        &self.execution_timing
    }

    pub fn into_parts(
        self,
    ) -> (
        DeviceTerminal<E>,
        DeviceTimingMeasurement<DeviceExecutionTiming>,
    ) {
        (self.terminal, self.execution_timing)
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
/// Initialization and dynamic binding commands are explicit eager barriers:
/// replaying either can reset live state or reuse stale request data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeviceCommandPhase {
    Initialization,
    DynamicBinding,
    Compute,
}

/// One command plus the core-issued semantic phase that constrains backend
/// execution optimizations.
pub struct DeviceCommandEntry<C> {
    phase: DeviceCommandPhase,
    command: C,
}

impl<C> DeviceCommandEntry<C> {
    pub const fn phase(&self) -> DeviceCommandPhase {
        self.phase
    }

    pub const fn command(&self) -> &C {
        &self.command
    }

    pub fn into_parts(self) -> (DeviceCommandPhase, C) {
        (self.phase, self.command)
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
            command,
        });
    }

    pub(crate) fn push_dynamic_binding(&mut self, command: C) {
        self.commands.push(DeviceCommandEntry {
            phase: DeviceCommandPhase::DynamicBinding,
            command,
        });
    }

    pub(crate) fn push_compute(&mut self, command: C) {
        self.commands.push(DeviceCommandEntry {
            phase: DeviceCommandPhase::Compute,
            command,
        });
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

    fn create_stream(&self) -> Result<Self::Stream, Self::Error>;

    fn stream_state(&self, stream: &Self::Stream) -> StreamState;

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
