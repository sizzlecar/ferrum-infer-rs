//! Per-transfer evidence from the existing native owner and terminal path.
//! This contains host metadata only; no tensor, stream or resource owner escapes.
use super::{
    invalid_completion, CompletionReaper, CompletionSlotId, StateTransferIdentity,
    StateTransferKind,
};
use crate::vnext::{
    DeviceDescriptor, DeviceExecutionTiming, DeviceId, DeviceRuntime, DeviceTimingMeasurement,
    ExecutionLaneId, PlanHash, ResourceId, SequenceCheckpointBytePlan, StridedCopyRegion,
    VNextError,
};
use sha2::{Digest, Sha256};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum NativeCheckpointTransferKind {
    Capture,
    Restore,
}

pub(crate) fn checkpoint_byte_plan_fingerprint(plan: &SequenceCheckpointBytePlan) -> String {
    super::canonical_completion_fingerprint(plan)
}

/// Reusable statistical domain, excluding single-transfer ownership nonces.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct NativeCheckpointTransferCostDomain {
    plan_hash: PlanHash,
    layout_fingerprint: Arc<str>,
    byte_plan_fingerprint: Arc<str>,
    runtime_implementation_fingerprint: Arc<str>,
    device_id: DeviceId,
    kind: NativeCheckpointTransferKind,
    geometry: NativeCheckpointTransferGeometry,
}
impl NativeCheckpointTransferCostDomain {
    fn from_parts(
        plan_hash: &PlanHash,
        layout: &str,
        byte_plan: &str,
        runtime: &str,
        device: &DeviceId,
        kind: NativeCheckpointTransferKind,
        geometry: NativeCheckpointTransferGeometry,
    ) -> Self {
        Self {
            plan_hash: plan_hash.clone(),
            layout_fingerprint: Arc::from(layout),
            byte_plan_fingerprint: Arc::from(byte_plan),
            runtime_implementation_fingerprint: Arc::from(runtime),
            device_id: device.clone(),
            kind,
            geometry,
        }
    }
    pub(crate) fn from_projection(
        byte_plan: &SequenceCheckpointBytePlan,
        descriptor: &DeviceDescriptor,
        kind: NativeCheckpointTransferKind,
        geometry: NativeCheckpointTransferGeometry,
    ) -> Self {
        Self::from_parts(
            byte_plan.plan_hash(),
            byte_plan.layout_fingerprint(),
            &checkpoint_byte_plan_fingerprint(byte_plan),
            &descriptor.runtime_implementation_fingerprint,
            &descriptor.id,
            kind,
            geometry,
        )
    }
    pub(super) fn from_identity(
        identity: &NativeCheckpointTransferIdentity,
        geometry: NativeCheckpointTransferGeometry,
    ) -> Self {
        Self::from_parts(
            identity.plan_hash(),
            identity.layout_fingerprint(),
            identity.byte_plan_fingerprint(),
            identity.runtime_implementation_fingerprint(),
            identity.device_id(),
            identity.kind(),
            geometry,
        )
    }
    pub fn plan_hash(&self) -> &PlanHash {
        &self.plan_hash
    }
    pub fn layout_fingerprint(&self) -> &str {
        &self.layout_fingerprint
    }
    pub fn byte_plan_fingerprint(&self) -> &str {
        &self.byte_plan_fingerprint
    }
    pub fn runtime_implementation_fingerprint(&self) -> &str {
        &self.runtime_implementation_fingerprint
    }
    pub fn device_id(&self) -> &DeviceId {
        &self.device_id
    }
    pub fn kind(&self) -> NativeCheckpointTransferKind {
        self.kind
    }
    pub fn geometry(&self) -> &NativeCheckpointTransferGeometry {
        &self.geometry
    }
}

/// Host token extents touched by the checkpoint facade. These are work
/// dimensions, not checkpoint ownership or a claim about elapsed time. Cache
/// population and contention are not described by this narrow evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct NativeCheckpointTransferHostWork {
    prefix_tokens: u64,
    full_input_tokens: u64,
}
impl NativeCheckpointTransferHostWork {
    pub(crate) fn from_lengths(
        prefix_tokens: u64,
        full_input_tokens: u64,
    ) -> Result<Self, VNextError> {
        if prefix_tokens == 0 || prefix_tokens > full_input_tokens {
            return Err(invalid_completion(
                "checkpoint host token extents are invalid",
            ));
        }
        Ok(Self {
            prefix_tokens,
            full_input_tokens,
        })
    }
    pub fn prefix_tokens(&self) -> u64 {
        self.prefix_tokens
    }
    pub fn full_input_tokens(&self) -> u64 {
        self.full_input_tokens
    }
}

/// A timestamp minted at an actual caller entry, never a supplied duration.
#[derive(Debug, Clone, Copy)]
pub struct CheckpointTransferObservationStart(Instant, Option<NativeCheckpointTransferHostWork>);
impl CheckpointTransferObservationStart {
    pub fn now() -> Self {
        Self(Instant::now(), None)
    }
    pub(super) fn with_host_work(mut self, work: NativeCheckpointTransferHostWork) -> Self {
        self.1 = Some(work);
        self
    }
    pub(super) fn host_work(self) -> Option<NativeCheckpointTransferHostWork> {
        self.1
    }
}
impl Default for CheckpointTransferObservationStart {
    fn default() -> Self {
        Self::now()
    }
}

/// Exact physical fragmentation, excluding allocator addresses and owner nonces.
/// The fingerprint preserves ordered resource/kind/length within each native
/// phase. Initialization always precedes copies in the checkpoint protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct NativeCheckpointTransferGeometry {
    copy_bytes: u64,
    copy_commands: u64,
    initialization_bytes: u64,
    initialization_commands: u64,
    ordered_fragments_fingerprint: [u8; 32],
}
impl NativeCheckpointTransferGeometry {
    pub fn copy_bytes(&self) -> u64 {
        self.copy_bytes
    }
    pub fn copy_commands(&self) -> u64 {
        self.copy_commands
    }
    pub fn initialization_bytes(&self) -> u64 {
        self.initialization_bytes
    }
    pub fn initialization_commands(&self) -> u64 {
        self.initialization_commands
    }
    pub fn ordered_fragments_fingerprint(&self) -> &[u8; 32] {
        &self.ordered_fragments_fingerprint
    }
}

/// Shared by actual encoding and numeric planning. Two ordered streams allow
/// preparation order to differ while preserving the mandatory submit order.
#[derive(Clone)]
pub(crate) struct NativeCheckpointTransferGeometryBuilder {
    copy: Sha256,
    initialization: Sha256,
    copy_bytes: u64,
    copy_commands: u64,
    initialization_bytes: u64,
    initialization_commands: u64,
    invalid: bool,
}
impl NativeCheckpointTransferGeometryBuilder {
    pub(crate) fn new() -> Self {
        Self {
            copy: Sha256::new(),
            initialization: Sha256::new(),
            copy_bytes: 0,
            copy_commands: 0,
            initialization_bytes: 0,
            initialization_commands: 0,
            invalid: false,
        }
    }
    fn push(
        hash: &mut Sha256,
        bytes: &mut u64,
        commands: &mut u64,
        resource: &ResourceId,
        length: u64,
    ) -> Result<(), VNextError> {
        if length == 0 {
            return Err(invalid_completion(
                "checkpoint geometry contains an empty fragment",
            ));
        }
        let next_bytes = bytes
            .checked_add(length)
            .ok_or_else(|| invalid_completion("checkpoint geometry bytes overflow"))?;
        let next_commands = commands
            .checked_add(1)
            .ok_or_else(|| invalid_completion("checkpoint geometry commands overflow"))?;
        let id = resource.as_str().as_bytes();
        hash.update((id.len() as u64).to_le_bytes());
        hash.update(id);
        hash.update(length.to_le_bytes());
        *bytes = next_bytes;
        *commands = next_commands;
        Ok(())
    }
    pub(crate) fn push_copy(
        &mut self,
        resource: &ResourceId,
        length: u64,
    ) -> Result<(), VNextError> {
        if self.invalid {
            return Err(invalid_completion(
                "checkpoint geometry was already rejected",
            ));
        }
        let result = Self::push(
            &mut self.copy,
            &mut self.copy_bytes,
            &mut self.copy_commands,
            resource,
            length,
        );
        self.invalid |= result.is_err();
        result
    }
    pub(crate) fn push_strided_copy(
        &mut self,
        resource: &ResourceId,
        region: StridedCopyRegion,
    ) -> Result<(), VNextError> {
        let length = region.length_bytes()?;
        self.push_copy(resource, length)?;
        self.copy.update(b"strided-rectangle.v1");
        self.copy.update(region.width_bytes().to_le_bytes());
        self.copy.update(region.height().to_le_bytes());
        self.copy.update(region.source_pitch_bytes().to_le_bytes());
        self.copy
            .update(region.destination_pitch_bytes().to_le_bytes());
        Ok(())
    }
    pub(crate) fn push_initialization(
        &mut self,
        resource: &ResourceId,
        length: u64,
    ) -> Result<(), VNextError> {
        if self.invalid {
            return Err(invalid_completion(
                "checkpoint geometry was already rejected",
            ));
        }
        let result = Self::push(
            &mut self.initialization,
            &mut self.initialization_bytes,
            &mut self.initialization_commands,
            resource,
            length,
        );
        self.invalid |= result.is_err();
        result
    }
    pub(crate) fn finish(self) -> Result<NativeCheckpointTransferGeometry, VNextError> {
        if self.invalid || self.copy_commands == 0 {
            return Err(invalid_completion(
                "checkpoint geometry is invalid or has no copies",
            ));
        }
        let mut hash = Sha256::new();
        hash.update(b"ferrum.native-checkpoint.ordered-fragments.v1");
        hash.update(self.initialization_commands.to_le_bytes());
        hash.update(self.initialization.finalize());
        hash.update(self.copy_commands.to_le_bytes());
        hash.update(self.copy.finalize());
        Ok(NativeCheckpointTransferGeometry {
            copy_bytes: self.copy_bytes,
            copy_commands: self.copy_commands,
            initialization_bytes: self.initialization_bytes,
            initialization_commands: self.initialization_commands,
            ordered_fragments_fingerprint: hash.finalize().into(),
        })
    }
}

/// Opaque exact reservation identity. Slot numbers are local to a reaper;
/// same_transfer compares the original private owner as well as its slot.
#[derive(Debug, Clone)]
pub struct NativeCheckpointTransferIdentity {
    slot_id: CompletionSlotId,
    inner: Arc<StateTransferIdentity>,
}

/// Non-owning comparison authority for one transfer. This keeps only the
/// fixed Arc allocation alive; expired identity strings and native resources
/// are never retained by an observation receipt.
#[derive(Debug)]
pub struct WeakNativeCheckpointTransferIdentity {
    slot_id: CompletionSlotId,
    inner: Weak<StateTransferIdentity>,
}

impl WeakNativeCheckpointTransferIdentity {
    pub fn matches(&self, identity: &NativeCheckpointTransferIdentity) -> bool {
        self.slot_id == identity.slot_id && self.inner.as_ptr() == Arc::as_ptr(&identity.inner)
    }

    /// Additional fixed allocation pinned by this weak owner, excluding the
    /// Weak handle itself and all heap fields dropped with the last strong Arc.
    pub const fn retained_allocation_bytes() -> usize {
        std::mem::size_of::<StateTransferIdentity>() + 2 * std::mem::size_of::<usize>()
    }

    #[cfg(test)]
    pub(crate) fn is_expired(&self) -> bool {
        self.inner.strong_count() == 0
    }
}

impl NativeCheckpointTransferIdentity {
    pub fn downgrade(&self) -> WeakNativeCheckpointTransferIdentity {
        WeakNativeCheckpointTransferIdentity {
            slot_id: self.slot_id,
            inner: Arc::downgrade(&self.inner),
        }
    }
    pub fn slot_id(&self) -> CompletionSlotId {
        self.slot_id
    }
    pub fn checkpoint_authority(&self) -> crate::vnext::CheckpointAuthorityId {
        self.inner.checkpoint_authority()
    }
    pub fn boundary_tokens(&self) -> u64 {
        self.inner.boundary_tokens()
    }
    pub fn sequence_authority(&self) -> crate::vnext::SequenceAuthorityId {
        self.inner.sequence_authority()
    }
    pub fn request_authority(&self) -> crate::vnext::RequestAuthorityId {
        self.inner.request_authority()
    }
    pub fn same_transfer(&self, other: &Self) -> bool {
        self.slot_id == other.slot_id && Arc::ptr_eq(&self.inner, &other.inner)
    }
    pub fn kind(&self) -> NativeCheckpointTransferKind {
        match self.inner.kind() {
            StateTransferKind::Capture => NativeCheckpointTransferKind::Capture,
            StateTransferKind::Restore => NativeCheckpointTransferKind::Restore,
        }
    }
    pub fn plan_hash(&self) -> &PlanHash {
        self.inner.plan_hash()
    }
    pub fn layout_fingerprint(&self) -> &str {
        self.inner.layout_fingerprint()
    }
    pub fn byte_plan_fingerprint(&self) -> &str {
        self.inner.byte_plan_fingerprint()
    }
    pub fn runtime_implementation_fingerprint(&self) -> &str {
        self.inner.runtime_implementation_fingerprint()
    }
    pub fn device_id(&self) -> &DeviceId {
        self.inner.device_id()
    }
    pub fn lane_id(&self) -> ExecutionLaneId {
        self.inner.lane_id()
    }
    pub(super) fn new(slot_id: CompletionSlotId, inner: Arc<StateTransferIdentity>) -> Self {
        Self { slot_id, inner }
    }
}

/// Minted once only after successful model capture publication acknowledgement
/// or successful restore acknowledgement. Missing or failed timing remains typed, never zero.
/// wall_elapsed includes waits/queueing; device time overlaps it and is not added.
#[derive(Debug)]
pub struct NativeCheckpointTransferObservation {
    identity: NativeCheckpointTransferIdentity,
    source_capture_identity: Option<NativeCheckpointTransferIdentity>,
    cost_domain: NativeCheckpointTransferCostDomain,
    host_work: Option<NativeCheckpointTransferHostWork>,
    wall_elapsed: Duration,
    device_timing: DeviceTimingMeasurement<DeviceExecutionTiming>,
}
impl NativeCheckpointTransferObservation {
    pub fn identity(&self) -> &NativeCheckpointTransferIdentity {
        &self.identity
    }
    pub fn kind(&self) -> NativeCheckpointTransferKind {
        self.identity.kind()
    }
    pub fn source_capture_identity(&self) -> Option<&NativeCheckpointTransferIdentity> {
        self.source_capture_identity.as_ref()
    }
    pub fn geometry(&self) -> &NativeCheckpointTransferGeometry {
        self.cost_domain.geometry()
    }
    pub fn cost_domain(&self) -> &NativeCheckpointTransferCostDomain {
        &self.cost_domain
    }
    pub fn host_work(&self) -> Option<&NativeCheckpointTransferHostWork> {
        self.host_work.as_ref()
    }
    pub fn wall_elapsed(&self) -> Duration {
        self.wall_elapsed
    }
    pub fn device_timing(&self) -> DeviceTimingMeasurement<DeviceExecutionTiming> {
        self.device_timing
    }
}

/// Implementations must use a bounded, nonblocking enqueue. Return false when
/// full or unavailable. They must not call the reaper or retain GPU resources.
pub trait NativeCheckpointObservationSink: Send + Sync {
    fn try_record(&self, observation: NativeCheckpointTransferObservation) -> bool;
}

pub(crate) struct PendingCheckpointObservation {
    pub(super) identity: NativeCheckpointTransferIdentity,
    pub(super) source_capture_identity: Option<NativeCheckpointTransferIdentity>,
    pub(super) geometry: NativeCheckpointTransferGeometry,
    pub(super) started: CheckpointTransferObservationStart,
    pub(super) sink: Weak<dyn NativeCheckpointObservationSink>,
    pub(super) device_timing: DeviceTimingMeasurement<DeviceExecutionTiming>,
}
impl PendingCheckpointObservation {
    pub(super) fn publish(self) {
        // Observation must not alter a committed transfer, even if its optional
        // consumer is gone, overloaded, or panics.
        let finished = Instant::now();
        let Some(wall_elapsed) = finished.checked_duration_since(self.started.0) else {
            return;
        };
        let Some(sink) = self.sink.upgrade() else {
            return;
        };
        let cost_domain =
            NativeCheckpointTransferCostDomain::from_identity(&self.identity, self.geometry);
        let observation = NativeCheckpointTransferObservation {
            identity: self.identity,
            source_capture_identity: self.source_capture_identity,
            cost_domain,
            host_work: self.started.host_work(),
            wall_elapsed,
            device_timing: self.device_timing,
        };
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            sink.try_record(observation)
        }));
    }
}
impl<R: DeviceRuntime> CompletionReaper<R> {
    /// Install before transfers start. A weak reference cannot keep a finished
    /// calibrator alive; replacement cannot redirect an in-flight receipt.
    pub fn install_checkpoint_observation_sink(
        &self,
        sink: Weak<dyn NativeCheckpointObservationSink>,
    ) -> Result<(), VNextError> {
        if self
            .checkpoint_observation_sink
            .get()
            .is_some_and(|old| Weak::ptr_eq(old, &sink))
        {
            return Ok(());
        }
        self.checkpoint_observation_sink
            .set(sink)
            .map_err(|_| invalid_completion("checkpoint observation sink is already installed"))
    }
}

#[cfg(test)]
mod tests;
