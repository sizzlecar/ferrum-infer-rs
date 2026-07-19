use super::{
    defer_device_cleanup, invalid_resource, AdmissionDecision, AdmissionDeferred,
    AdmissionFitPolicy, AdmissionPreflightDecision, AdmissionPressureAction, AdmissionRejected,
    AllocationLifetime, Arc, AtomicU64, BTreeMap, BTreeSet, BackingPrepareDecision, BatchStepId,
    CapacityClaimDecision, DeferredDeviceCleanupDisposition, DeferredDeviceCleanupTask,
    DeviceBufferRetention, DeviceRuntime, Digest, DynamicBackingClaimScope, DynamicBackingDeferred,
    DynamicDeferredMaintenanceOutcome, DynamicResourceShape, ExecutionFrameId,
    InitialSequenceAdmissionDecision, LogicalAdmissionCoordinatorId, LogicalAdmissionLease,
    LogicalBackingBufferView, LogicalBackingSliceAuthority, LogicalBackingSliceEvidence,
    LogicalCapacityLease, LogicalRequestLease, ManuallyDrop, Mutex, NonZeroU64, Ordering,
    ParticipantNodeKey, PlanBackingDeferral, PlanCapacityWaitRegistration, PreparedBackingClaim,
    RequestAdmissionDecision, RequestAuthorityId, RequestIdentity, RequestResourceAdmissionRequest,
    ResourceId, ResourceWorkShape, RunId, SequenceAuthorityId, SequenceRecoveryRegistry,
    SequenceResourceAdmissionRequest, Serialize, Sha256, StaticProvisioningLease,
    TrustedPlanRuntimeBinding, TrustedPlanRuntimeEvidence, VNextError, Weak,
    SEQUENCE_DISPATCH_POISONED_BIT,
};
use crate::vnext::CapacityAvailabilitySource;

pub(super) const fn sequence_slot_active(epoch: u64) -> u64 {
    (epoch << 2) | 1
}

pub(super) const fn sequence_slot_poisoned_drained(epoch: u64) -> u64 {
    (epoch << 2) | 2
}

pub(super) const fn sequence_slot_poisoned_undrained(epoch: u64) -> u64 {
    (epoch << 2) | 3
}

pub(super) const fn sequence_slot_is_poisoned(state: u64) -> bool {
    matches!(state & 3, 2 | 3)
}

const SEQUENCE_DISPATCH_COUNT_MASK: u64 = SEQUENCE_DISPATCH_POISONED_BIT - 1;

pub(super) fn sequence_dispatch_is_poisoned(gate: &AtomicU64) -> bool {
    gate.load(Ordering::Acquire) & SEQUENCE_DISPATCH_POISONED_BIT != 0
}

pub(super) struct SequenceDispatchGuard<'a> {
    gate: &'a AtomicU64,
}

impl Drop for SequenceDispatchGuard<'_> {
    fn drop(&mut self) {
        let previous = self.gate.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(previous & SEQUENCE_DISPATCH_COUNT_MASK > 0);
    }
}

pub(super) fn enter_sequence_dispatch(
    gate: &AtomicU64,
) -> Result<SequenceDispatchGuard<'_>, VNextError> {
    gate.fetch_update(Ordering::AcqRel, Ordering::Acquire, |state| {
        if state & SEQUENCE_DISPATCH_POISONED_BIT != 0
            || state & SEQUENCE_DISPATCH_COUNT_MASK == SEQUENCE_DISPATCH_COUNT_MASK
        {
            None
        } else {
            Some(state + 1)
        }
    })
    .map_err(|state| {
        if state & SEQUENCE_DISPATCH_POISONED_BIT != 0 {
            invalid_resource("poisoned resource pool cannot dispatch another operation")
        } else {
            invalid_resource("resource pool dispatch counter is exhausted")
        }
    })?;
    Ok(SequenceDispatchGuard { gate })
}

pub enum RequestResourceAdmissionDecision<R>
where
    R: DeviceRuntime,
{
    Admitted(Arc<AdmittedRequestResources<R>>),
    Deferred(AdmissionDeferred),
    BackingDeferred(RequestBackingDeferral<R>),
    PermanentRejected(AdmissionRejected),
}

pub enum InitialSequenceResourceAdmissionDecision<R>
where
    R: DeviceRuntime,
{
    Admitted(Arc<AdmittedSequenceResources<R>>),
    Deferred(AdmissionDeferred),
    BackingDeferred(InitialSequenceBackingDeferral<R>),
    PermanentRejected(AdmissionRejected),
}

/// Non-cloneable authority for physical maintenance of an uncommitted initial
/// request/sequence bundle. It owns no request or sequence lease.
#[must_use = "initial sequence backing deferral owns its exact bundle attempt"]
pub struct InitialSequenceBackingDeferral<R>
where
    R: DeviceRuntime,
{
    evidence: DynamicBackingDeferred,
    plan: TrustedPlanRuntimeBinding<R>,
}

impl<R> InitialSequenceBackingDeferral<R>
where
    R: DeviceRuntime,
{
    pub fn evidence(&self) -> &DynamicBackingDeferred {
        &self.evidence
    }

    pub fn maintain(&self) -> Result<DynamicDeferredMaintenanceOutcome, VNextError> {
        if self.evidence.scope() != DynamicBackingClaimScope::InitialSequenceBundle {
            return Err(invalid_resource(
                "initial sequence maintenance requires a bundle-scoped deferral",
            ));
        }
        let _lifecycle = self
            .plan
            .resources
            .read_lifecycle("maintain deferred initial sequence backing")?;
        self.plan
            .resources
            .maintenance_controller
            .maintain_for_live_deferred(&self.evidence)
    }

    pub fn register_waiter(&self) -> Result<PlanCapacityWaitRegistration<R>, VNextError> {
        self.plan.register_backing_waiter(&self.evidence)
    }
}

/// Non-cloneable authority for one exact request-admission backing attempt.
/// The embedded evidence can be projected to schedulers and traces, but only
/// this handle can invoke live revalidation.
#[must_use = "request backing deferral owns its exact admission attempt"]
pub struct RequestBackingDeferral<R>
where
    R: DeviceRuntime,
{
    evidence: DynamicBackingDeferred,
    plan: TrustedPlanRuntimeBinding<R>,
    work_shape: ResourceWorkShape,
    run_id: RunId,
    request_id: RequestIdentity,
}

impl<R> RequestBackingDeferral<R>
where
    R: DeviceRuntime,
{
    pub fn evidence(&self) -> &DynamicBackingDeferred {
        &self.evidence
    }

    pub fn work_shape(&self) -> &ResourceWorkShape {
        &self.work_shape
    }

    pub fn run_id(&self) -> &RunId {
        &self.run_id
    }

    pub fn request_id(&self) -> &RequestIdentity {
        &self.request_id
    }

    pub fn maintain(&self) -> Result<DynamicDeferredMaintenanceOutcome, VNextError> {
        self.plan
            .maintain_request_backing_for_deferred(&self.evidence)
    }

    pub fn register_waiter(&self) -> Result<PlanCapacityWaitRegistration<R>, VNextError> {
        self.plan.register_backing_waiter(&self.evidence)
    }
}

impl<R> TrustedPlanRuntimeBinding<R>
where
    R: DeviceRuntime,
{
    /// Maintains a request-lifetime physical deferral while this exact plan
    /// binding proves that no child authority is required for validity.
    pub(super) fn maintain_request_backing_for_deferred(
        &self,
        deferred: &DynamicBackingDeferred,
    ) -> Result<DynamicDeferredMaintenanceOutcome, VNextError> {
        if deferred.scope() != DynamicBackingClaimScope::Request {
            return Err(invalid_resource(
                "request backing maintenance requires a request-lifetime deferral",
            ));
        }
        let _lifecycle = self
            .resources
            .read_lifecycle("maintain deferred request backing")?;
        self.resources
            .maintenance_controller
            .maintain_for_live_deferred(deferred)
    }

    /// Request-scoped capacity is claimed exactly once before any child
    /// sequence, stream, provider encode, or device submission exists.
    pub fn try_admit_request(
        &self,
        request: RequestResourceAdmissionRequest,
        run_id: RunId,
        request_id: RequestIdentity,
    ) -> Result<RequestResourceAdmissionDecision<R>, VNextError> {
        let _lifecycle = self.resources.read_lifecycle("admit a request")?;
        let RequestResourceAdmissionRequest {
            work_shape,
            fit_policy,
            pressure_action,
        } = request;
        // Request-lifetime values own the complete canonical input/output for
        // every prefill chunk. This is live request state, not a reservation
        // for future Sequence/Step/Invocation capacity.
        let request_shape = work_shape.fit_shape();
        let (demand, requested_slices) = self.scoped_demand(
            AllocationLifetime::Request,
            None,
            request_shape,
            request_shape,
            fit_policy,
            pressure_action,
        )?;
        let prepared = match self.prepare_backing_slices(requested_slices)? {
            BackingPrepareDecision::Prepared(prepared) => prepared,
            BackingPrepareDecision::Deferred(deferred) => {
                return Ok(RequestResourceAdmissionDecision::BackingDeferred(
                    RequestBackingDeferral {
                        evidence: deferred,
                        plan: TrustedPlanRuntimeBinding {
                            resources: Arc::clone(&self.resources),
                        },
                        work_shape,
                        run_id,
                        request_id,
                    },
                ));
            }
        };
        match self.logical_admission().try_admit_request(&demand)? {
            RequestAdmissionDecision::Admitted(logical_lease) => {
                if !self.logical_admission().owns_request(&logical_lease) {
                    return Err(invalid_resource(
                        "request admission returned authority from another coordinator",
                    ));
                }
                let slices = prepared.commit();
                Ok(RequestResourceAdmissionDecision::Admitted(Arc::new(
                    AdmittedRequestResources::new(
                        TrustedPlanRuntimeBinding {
                            resources: Arc::clone(&self.resources),
                        },
                        logical_lease,
                        slices,
                        work_shape,
                        run_id,
                        request_id,
                    )?,
                )))
            }
            RequestAdmissionDecision::Deferred(deferred) => {
                Ok(RequestResourceAdmissionDecision::Deferred(deferred))
            }
            RequestAdmissionDecision::PermanentRejected(rejected) => Ok(
                RequestResourceAdmissionDecision::PermanentRejected(rejected),
            ),
        }
    }

    /// Atomically admits one request root and its first child sequence. Elastic
    /// backing may grow before the final logical commit, but no request-only
    /// authority is retained when the sequence cannot be admitted.
    pub fn try_admit_initial_sequence(
        &self,
        request: RequestResourceAdmissionRequest,
        sequence: SequenceResourceAdmissionRequest,
        run_id: RunId,
        request_id: RequestIdentity,
    ) -> Result<InitialSequenceResourceAdmissionDecision<R>, VNextError> {
        let _lifecycle = self
            .resources
            .read_lifecycle("admit an initial request/sequence bundle")?;
        let RequestResourceAdmissionRequest {
            work_shape: request_work_shape,
            fit_policy: request_fit_policy,
            pressure_action: request_pressure_action,
        } = request;
        let SequenceResourceAdmissionRequest {
            work_shape: sequence_work_shape,
            fit_policy: sequence_fit_policy,
            pressure_action: sequence_pressure_action,
        } = sequence;
        if sequence_work_shape.fit_tokens() > request_work_shape.fit_tokens() {
            return Err(invalid_resource(
                "initial sequence token ceiling exceeds its request ceiling",
            ));
        }

        let request_shape = request_work_shape.fit_shape();
        let (request_demand, mut requested_slices) = self.scoped_demand(
            AllocationLifetime::Request,
            None,
            request_shape,
            request_shape,
            request_fit_policy,
            request_pressure_action,
        )?;
        let sequence_immediate_shape = sequence_work_shape.immediate_shape();
        let sequence_fit_shape = match sequence_fit_policy {
            AdmissionFitPolicy::ImmediateOnly => sequence_immediate_shape,
            AdmissionFitPolicy::FullInputMustFit => sequence_work_shape.fit_shape(),
        };
        let (sequence_demand, mut sequence_slices) = self.scoped_demand(
            AllocationLifetime::Sequence,
            None,
            sequence_immediate_shape,
            sequence_fit_shape,
            sequence_fit_policy,
            sequence_pressure_action,
        )?;

        let request_resource_ids = requested_slices
            .iter()
            .flat_map(|request| request.projections.iter())
            .map(|projection| projection.descriptor.base_resource_id().clone())
            .collect::<BTreeSet<_>>();
        let sequence_resource_ids = sequence_slices
            .iter()
            .flat_map(|request| request.projections.iter())
            .map(|projection| projection.descriptor.base_resource_id().clone())
            .collect::<BTreeSet<_>>();
        if !request_resource_ids.is_disjoint(&sequence_resource_ids) {
            return Err(invalid_resource(
                "initial request and sequence backing resources are not disjoint",
            ));
        }
        requested_slices.append(&mut sequence_slices);

        loop {
            match self
                .logical_admission()
                .preflight_initial_sequence(&request_demand, &sequence_demand)?
            {
                AdmissionPreflightDecision::Eligible => {}
                AdmissionPreflightDecision::Deferred(deferred) => {
                    return Ok(InitialSequenceResourceAdmissionDecision::Deferred(deferred));
                }
                AdmissionPreflightDecision::PermanentRejected(rejected) => {
                    return Ok(InitialSequenceResourceAdmissionDecision::PermanentRejected(
                        rejected,
                    ));
                }
            }

            let prepared = match self.prepare_initial_sequence_backing_slices(&requested_slices)? {
                BackingPrepareDecision::Prepared(prepared) => prepared,
                BackingPrepareDecision::Deferred(evidence) => {
                    return Ok(InitialSequenceResourceAdmissionDecision::BackingDeferred(
                        InitialSequenceBackingDeferral {
                            evidence,
                            plan: TrustedPlanRuntimeBinding {
                                resources: Arc::clone(&self.resources),
                            },
                        },
                    ));
                }
            };

            match self
                .logical_admission()
                .try_admit_initial_sequence(&request_demand, &sequence_demand)?
            {
                InitialSequenceAdmissionDecision::Admitted(logical) => {
                    let (request_logical, sequence_logical) = logical.into_parts();
                    if !self.logical_admission().owns_request(&request_logical)
                        || !self.logical_admission().owns(&sequence_logical)
                        || sequence_logical.request() != request_logical.request()
                    {
                        return Err(invalid_resource(
                            "initial bundle admission returned incoherent authority",
                        ));
                    }

                    let mut request_backing = Vec::new();
                    let mut sequence_backing = Vec::new();
                    let mut seen_request = BTreeSet::new();
                    let mut seen_sequence = BTreeSet::new();
                    for authority in prepared.commit() {
                        let resource_id = authority.resource_id().clone();
                        if request_resource_ids.contains(&resource_id) {
                            seen_request.insert(resource_id);
                            request_backing.push(authority);
                        } else if sequence_resource_ids.contains(&resource_id) {
                            seen_sequence.insert(resource_id);
                            sequence_backing.push(authority);
                        } else {
                            return Err(invalid_resource(
                                "initial bundle prepared an unrequested backing resource",
                            ));
                        }
                    }
                    if seen_request != request_resource_ids
                        || seen_sequence != sequence_resource_ids
                    {
                        return Err(invalid_resource(
                            "initial bundle backing did not cover every requested resource",
                        ));
                    }

                    let request = Arc::new(AdmittedRequestResources {
                        backing_slices: request_backing,
                        logical_lease: request_logical,
                        plan: TrustedPlanRuntimeBinding {
                            resources: Arc::clone(&self.resources),
                        },
                        work_shape: request_work_shape,
                        run_id,
                        request_id,
                    });
                    return Ok(InitialSequenceResourceAdmissionDecision::Admitted(
                        Arc::new(AdmittedSequenceResources::new(
                            request,
                            sequence_logical,
                            sequence_backing,
                            sequence_work_shape,
                        )?),
                    ));
                }
                InitialSequenceAdmissionDecision::Deferred => {
                    drop(prepared);
                    match self
                        .logical_admission()
                        .observe_initial_sequence(&request_demand, &sequence_demand)?
                    {
                        AdmissionPreflightDecision::Eligible => continue,
                        AdmissionPreflightDecision::Deferred(deferred) => {
                            return Ok(InitialSequenceResourceAdmissionDecision::Deferred(
                                deferred,
                            ));
                        }
                        AdmissionPreflightDecision::PermanentRejected(rejected) => {
                            return Ok(
                                InitialSequenceResourceAdmissionDecision::PermanentRejected(
                                    rejected,
                                ),
                            );
                        }
                    }
                }
                InitialSequenceAdmissionDecision::PermanentRejected(rejected) => {
                    drop(prepared);
                    return Ok(InitialSequenceResourceAdmissionDecision::PermanentRejected(
                        rejected,
                    ));
                }
            }
        }
    }
}

/// Request root authority. Request-lifetime state is physically and logically
/// claimed once, then shared by exact child sequence authorities through an
/// owning `Arc` parent hold.
#[must_use = "request resources release capacity after their last child sequence"]
pub struct AdmittedRequestResources<R>
where
    R: DeviceRuntime,
{
    backing_slices: Vec<LogicalBackingSliceAuthority>,
    logical_lease: LogicalRequestLease,
    pub(super) plan: TrustedPlanRuntimeBinding<R>,
    work_shape: ResourceWorkShape,
    run_id: RunId,
    request_id: RequestIdentity,
}

impl<R> AdmittedRequestResources<R>
where
    R: DeviceRuntime,
{
    pub(super) fn new(
        plan: TrustedPlanRuntimeBinding<R>,
        logical_lease: LogicalRequestLease,
        backing_slices: Vec<LogicalBackingSliceAuthority>,
        work_shape: ResourceWorkShape,
        run_id: RunId,
        request_id: RequestIdentity,
    ) -> Result<Self, VNextError> {
        if !plan.logical_admission().owns_request(&logical_lease) {
            return Err(invalid_resource(
                "logical request authority belongs to another coordinator",
            ));
        }
        Ok(Self {
            backing_slices,
            logical_lease,
            plan,
            work_shape,
            run_id,
            request_id,
        })
    }

    pub const fn request_authority(&self) -> RequestAuthorityId {
        self.logical_lease.request()
    }

    pub fn coordinator_id(&self) -> LogicalAdmissionCoordinatorId {
        self.logical_lease.coordinator_id()
    }

    pub fn run_id(&self) -> &RunId {
        &self.run_id
    }

    pub fn request_id(&self) -> &RequestIdentity {
        &self.request_id
    }

    pub fn backing_slices(&self) -> &[LogicalBackingSliceAuthority] {
        &self.backing_slices
    }

    pub fn work_shape(&self) -> &ResourceWorkShape {
        &self.work_shape
    }

    pub fn static_provisioning(&self) -> Option<&StaticProvisioningLease<R>> {
        self.plan.static_provisioning()
    }

    /// Maintains a sequence-lifetime physical deferral while this request
    /// authority keeps the exact parent logical and backing claims alive.
    pub(super) fn maintain_sequence_backing_for_deferred(
        &self,
        deferred: &DynamicBackingDeferred,
    ) -> Result<DynamicDeferredMaintenanceOutcome, VNextError> {
        if deferred.scope() != DynamicBackingClaimScope::Sequence {
            return Err(invalid_resource(
                "sequence backing maintenance requires a sequence-lifetime deferral",
            ));
        }
        let _lifecycle = self
            .plan
            .resources
            .read_lifecycle("maintain deferred sequence backing")?;
        self.plan
            .resources
            .maintenance_controller
            .maintain_for_live_deferred(deferred)
    }

    pub fn plan_evidence(&self) -> TrustedPlanRuntimeEvidence {
        self.plan.evidence()
    }

    pub(super) fn backing_view(
        &self,
        resource_id: &ResourceId,
    ) -> Result<LogicalBackingBufferView<'_, R::Buffer>, VNextError> {
        let authority = self
            .backing_slices
            .iter()
            .find(|authority| authority.resource_id() == resource_id)
            .ok_or_else(|| invalid_resource("logical request does not own that backing slice"))?;
        self.plan.dynamic_pools().view(authority)
    }

    /// Sequence-scoped capacity is charged once per exact child sequence.
    pub fn try_admit_sequence(
        self: &Arc<Self>,
        request: SequenceResourceAdmissionRequest,
    ) -> Result<SequenceResourceAdmissionDecision<R>, VNextError> {
        let _lifecycle = self
            .plan
            .resources
            .read_lifecycle("admit a child sequence")?;
        let SequenceResourceAdmissionRequest {
            work_shape,
            fit_policy,
            pressure_action,
        } = request;
        if work_shape.fit_tokens() > self.work_shape.fit_tokens() {
            return Err(invalid_resource(
                "sequence token ceiling exceeds its parent request ceiling",
            ));
        }
        let immediate_shape = work_shape.immediate_shape();
        let fit_shape = match fit_policy {
            AdmissionFitPolicy::ImmediateOnly => immediate_shape,
            AdmissionFitPolicy::FullInputMustFit => work_shape.fit_shape(),
        };
        let (demand, requested_slices) = self.plan.scoped_demand(
            AllocationLifetime::Sequence,
            None,
            immediate_shape,
            fit_shape,
            fit_policy,
            pressure_action,
        )?;
        match self
            .plan
            .logical_admission()
            .preflight_sequence_ceiling_for_request(&self.logical_lease, &demand)?
        {
            AdmissionPreflightDecision::Eligible => {}
            AdmissionPreflightDecision::Deferred(deferred) => {
                return Ok(SequenceResourceAdmissionDecision::Deferred(deferred));
            }
            AdmissionPreflightDecision::PermanentRejected(rejected) => {
                return Ok(SequenceResourceAdmissionDecision::PermanentRejected(
                    rejected,
                ));
            }
        }
        let prepared = match self.plan.prepare_backing_slices(requested_slices)? {
            BackingPrepareDecision::Prepared(prepared) => prepared,
            BackingPrepareDecision::Deferred(deferred) => {
                return Ok(SequenceResourceAdmissionDecision::BackingDeferred(
                    SequenceAdmissionBackingDeferral {
                        evidence: deferred,
                        parent: Arc::clone(self),
                    },
                ));
            }
        };
        match self
            .plan
            .logical_admission()
            .try_admit_sequence_for_request(&self.logical_lease, &demand)?
        {
            AdmissionDecision::Admitted(logical_lease) => {
                if !self.plan.logical_admission().owns(&logical_lease)
                    || logical_lease.request() != self.request_authority()
                {
                    return Err(invalid_resource(
                        "sequence admission returned authority from another request",
                    ));
                }
                let slices = prepared.commit();
                Ok(SequenceResourceAdmissionDecision::Admitted(Arc::new(
                    AdmittedSequenceResources::new(
                        Arc::clone(self),
                        logical_lease,
                        slices,
                        work_shape,
                    )?,
                )))
            }
            AdmissionDecision::Deferred(deferred) => {
                Ok(SequenceResourceAdmissionDecision::Deferred(deferred))
            }
            AdmissionDecision::PermanentRejected(rejected) => Ok(
                SequenceResourceAdmissionDecision::PermanentRejected(rejected),
            ),
        }
    }
}

pub enum SequenceResourceAdmissionDecision<R>
where
    R: DeviceRuntime,
{
    Admitted(Arc<AdmittedSequenceResources<R>>),
    Deferred(AdmissionDeferred),
    BackingDeferred(SequenceAdmissionBackingDeferral<R>),
    PermanentRejected(AdmissionRejected),
}

/// Non-cloneable authority for one sequence-admission backing attempt. Holding
/// it keeps the exact parent request alive; a sibling request cannot maintain
/// or substitute for that parent.
#[must_use = "sequence backing deferral owns its exact request parent"]
pub struct SequenceAdmissionBackingDeferral<R>
where
    R: DeviceRuntime,
{
    evidence: DynamicBackingDeferred,
    parent: Arc<AdmittedRequestResources<R>>,
}

impl<R> SequenceAdmissionBackingDeferral<R>
where
    R: DeviceRuntime,
{
    pub fn evidence(&self) -> &DynamicBackingDeferred {
        &self.evidence
    }

    pub fn parent(&self) -> &Arc<AdmittedRequestResources<R>> {
        &self.parent
    }

    pub fn into_parent(self) -> Arc<AdmittedRequestResources<R>> {
        self.parent
    }

    pub fn maintain(&self) -> Result<DynamicDeferredMaintenanceOutcome, VNextError> {
        self.parent
            .maintain_sequence_backing_for_deferred(&self.evidence)
    }

    pub fn register_waiter(&self) -> Result<PlanCapacityWaitRegistration<R>, VNextError> {
        self.parent.plan.register_backing_waiter(&self.evidence)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct SequenceSessionEpoch(pub(super) NonZeroU64);

impl SequenceSessionEpoch {
    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SequenceSessionFingerprint(pub(super) String);

pub(super) const SEQUENCE_SESSION_FINGERPRINT_DOMAIN: &str = "sequence-session-v1";

#[derive(Serialize)]
pub(super) struct SequenceSessionFingerprintEnvelope<'a, T>
where
    T: Serialize + ?Sized,
{
    pub(super) domain: &'static str,
    pub(super) payload: &'a T,
}

pub(super) fn sequence_session_fingerprint<T>(
    payload: &T,
) -> Result<SequenceSessionFingerprint, VNextError>
where
    T: Serialize + ?Sized,
{
    let envelope = SequenceSessionFingerprintEnvelope {
        domain: SEQUENCE_SESSION_FINGERPRINT_DOMAIN,
        payload,
    };
    let bytes = serde_json::to_vec(&envelope)
        .map_err(|_| invalid_resource("trusted sequence session identity did not serialize"))?;
    Ok(SequenceSessionFingerprint(format!(
        "{:x}",
        Sha256::digest(bytes)
    )))
}

impl SequenceSessionFingerprint {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SequenceSessionTerminalDisposition {
    Completed,
    Aborted,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SequenceSessionTerminalReceipt {
    pub(super) epoch: SequenceSessionEpoch,
    pub(super) fingerprint: SequenceSessionFingerprint,
    pub(super) disposition: SequenceSessionTerminalDisposition,
    pub(super) retired_frames: u64,
}

impl SequenceSessionTerminalReceipt {
    pub const fn epoch(&self) -> SequenceSessionEpoch {
        self.epoch
    }

    pub fn fingerprint(&self) -> &SequenceSessionFingerprint {
        &self.fingerprint
    }

    pub const fn disposition(&self) -> SequenceSessionTerminalDisposition {
        self.disposition
    }

    pub const fn retired_frames(&self) -> u64 {
        self.retired_frames
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SequenceSessionPhase {
    Open,
    CancelRequested,
    Poisoned,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ActiveSequenceFrame {
    pub(super) frame_id: ExecutionFrameId,
    pub(super) batch_step_id: BatchStepId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ParticipantFlightPhase {
    Prepared,
    InFlight,
}

#[derive(Debug, Clone)]
pub(super) struct ActiveSequenceSessionState {
    pub(super) epoch: SequenceSessionEpoch,
    pub(super) fingerprint: SequenceSessionFingerprint,
    pub(super) phase: SequenceSessionPhase,
    pub(super) next_frame: Option<ExecutionFrameId>,
    pub(super) active_frame: Option<ActiveSequenceFrame>,
    pub(super) participant_flights: BTreeMap<ParticipantNodeKey, ParticipantFlightPhase>,
    pub(super) retired_frames: u64,
}

#[derive(Debug, Clone)]
pub(super) enum SequenceSessionSlotState {
    Dormant {
        next_epoch: Option<SequenceSessionEpoch>,
    },
    Active(ActiveSequenceSessionState),
    Terminal(SequenceSessionTerminalReceipt),
    FailClosed,
}

pub(super) struct SequenceSessionSlot {
    pub(super) state: Mutex<SequenceSessionSlotState>,
}

impl SequenceSessionSlot {
    fn new() -> Self {
        Self {
            state: Mutex::new(SequenceSessionSlotState::Dormant {
                next_epoch: Some(SequenceSessionEpoch(
                    NonZeroU64::new(1).expect("one is non-zero"),
                )),
            }),
        }
    }

    fn is_poisoned(&self) -> bool {
        match self.state.lock() {
            Ok(state) => matches!(
                &*state,
                SequenceSessionSlotState::Active(ActiveSequenceSessionState {
                    phase: SequenceSessionPhase::Poisoned,
                    ..
                }) | SequenceSessionSlotState::FailClosed
            ),
            Err(_) => true,
        }
    }
}

/// Process-local proof that one exact sequence session was open when checked.
/// The weak slot reference deliberately does not keep the session or its
/// resources alive, and the private fields prevent sibling modules from
/// forging a witness from copied identity values.
#[derive(Clone)]
pub(crate) struct SequenceSessionLiveWitness {
    pub(super) slot: Weak<SequenceSessionSlot>,
    epoch: SequenceSessionEpoch,
    fingerprint: SequenceSessionFingerprint,
}

impl SequenceSessionLiveWitness {
    pub(crate) fn ensure_live(&self) -> Result<(), VNextError> {
        let slot = self.slot.upgrade().ok_or_else(|| {
            invalid_resource("sequence session live witness owner is no longer available")
        })?;
        let state = slot
            .state
            .lock()
            .map_err(|_| invalid_resource("sequence session state mutex is poisoned"))?;
        match &*state {
            SequenceSessionSlotState::Active(active)
                if active.epoch == self.epoch && active.fingerprint == self.fingerprint =>
            {
                Ok(())
            }
            _ => Err(invalid_resource(
                "sequence session live witness is stale or no longer active",
            )),
        }
    }

    pub(crate) fn ensure_open(&self) -> Result<(), VNextError> {
        let slot = self.slot.upgrade().ok_or_else(|| {
            invalid_resource("sequence session live witness owner is no longer available")
        })?;
        let state = slot
            .state
            .lock()
            .map_err(|_| invalid_resource("sequence session state mutex is poisoned"))?;
        match &*state {
            SequenceSessionSlotState::Active(active)
                if active.epoch == self.epoch
                    && active.fingerprint == self.fingerprint
                    && active.phase == SequenceSessionPhase::Open =>
            {
                Ok(())
            }
            _ => Err(invalid_resource(
                "sequence session live witness is stale or no longer open",
            )),
        }
    }

    pub(crate) fn ensure_identity(
        &self,
        epoch: SequenceSessionEpoch,
        fingerprint: &SequenceSessionFingerprint,
    ) -> Result<(), VNextError> {
        if self.epoch != epoch || self.fingerprint != *fingerprint {
            return Err(invalid_resource(
                "sequence session live witness differs from the expected identity",
            ));
        }
        self.ensure_open()
    }

    pub(crate) fn ensure_live_identity(
        &self,
        epoch: SequenceSessionEpoch,
        fingerprint: &SequenceSessionFingerprint,
    ) -> Result<(), VNextError> {
        if self.epoch != epoch || self.fingerprint != *fingerprint {
            return Err(invalid_resource(
                "sequence session live witness differs from the expected identity",
            ));
        }
        self.ensure_live()
    }
}

/// Core-owned logical sequence lifecycle. It owns sequence resources but no
/// device stream; scheduler-owned execution lanes may serve many sessions.
#[must_use = "a sequence session must reach an explicit terminal disposition"]
pub struct SequenceSession<R>
where
    R: DeviceRuntime,
{
    resources: Arc<AdmittedSequenceResources<R>>,
    pub(super) slot: Arc<SequenceSessionSlot>,
    pub(super) epoch: SequenceSessionEpoch,
    pub(super) fingerprint: SequenceSessionFingerprint,
}

impl<R> SequenceSession<R>
where
    R: DeviceRuntime,
{
    pub fn resources(&self) -> &Arc<AdmittedSequenceResources<R>> {
        &self.resources
    }

    /// Projects the exact scheduler-visible sources advanced by terminally
    /// releasing this sequence authority.
    pub fn write_release_capacity_sources(
        &self,
        sources: &mut Vec<CapacityAvailabilitySource>,
    ) -> Result<(), VNextError> {
        self.ensure_open_identity()?;
        let backing = self.resources.backing_snapshot()?;
        sources.clear();
        sources.push(CapacityAvailabilitySource::ActiveSequenceSlots);
        sources.extend(
            backing
                .backing_slices()
                .iter()
                .map(|slice| CapacityAvailabilitySource::Domain(slice.domain_id())),
        );
        sources.sort_unstable();
        sources.dedup();
        Ok(())
    }

    pub const fn epoch(&self) -> SequenceSessionEpoch {
        self.epoch
    }

    pub fn fingerprint(&self) -> &SequenceSessionFingerprint {
        &self.fingerprint
    }

    pub(crate) fn live_witness(&self) -> Result<SequenceSessionLiveWitness, VNextError> {
        let witness = SequenceSessionLiveWitness {
            slot: Arc::downgrade(&self.slot),
            epoch: self.epoch,
            fingerprint: self.fingerprint.clone(),
        };
        witness.ensure_identity(self.epoch, &self.fingerprint)?;
        Ok(witness)
    }

    pub(crate) fn ensure_open_identity(&self) -> Result<(), VNextError> {
        self.live_witness().map(|_| ())
    }

    pub fn sequence_authority(&self) -> SequenceAuthorityId {
        self.resources.sequence_authority()
    }

    pub fn request_authority(&self) -> RequestAuthorityId {
        self.resources.request_authority()
    }

    pub fn register_admission_waiter(
        &self,
        deferred: &AdmissionDeferred,
    ) -> Result<PlanCapacityWaitRegistration<R>, VNextError> {
        self.resources
            .request
            .plan
            .register_admission_waiter(deferred)
    }

    /// Ensures that the sequence-owned committed backing frontier covers the
    /// requested work without making an in-flight frame observe a different
    /// physical generation. A frontier already wider than the requested work
    /// is returned unchanged. Expensive preparation happens outside the
    /// session/backing locks; publication is one short slot -> backing critical
    /// section.
    pub fn try_ensure_backing_covers(
        self: &Arc<Self>,
        request: SequenceResourceExtensionRequest,
    ) -> Result<SequenceResourceExtensionDecision<R>, VNextError> {
        let _lifecycle = self
            .resources
            .request
            .plan
            .resources
            .read_lifecycle("extend sequence backing")?;
        let target_fingerprint = request.target_work.fingerprint().to_owned();
        let target = request.target_work.fit_shape();
        if target.tokens() > self.resources.request.work_shape().fit_tokens() {
            return Err(invalid_resource(
                "sequence backing extension exceeds its parent request token ceiling",
            ));
        }

        let expected = {
            let slot = self
                .slot
                .state
                .lock()
                .map_err(|_| invalid_resource("sequence session state mutex is poisoned"))?;
            let active = match &*slot {
                SequenceSessionSlotState::Active(active)
                    if active.epoch == self.epoch
                        && active.fingerprint == self.fingerprint
                        && active.phase == SequenceSessionPhase::Open =>
                {
                    active
                }
                SequenceSessionSlotState::Active(active)
                    if active.epoch != self.epoch || active.fingerprint != self.fingerprint =>
                {
                    return Err(invalid_resource("stale sequence session authority"));
                }
                SequenceSessionSlotState::Active(_) => {
                    return Err(invalid_resource(
                        "sequence backing can only grow for an open session",
                    ));
                }
                _ => {
                    return Err(invalid_resource(
                        "inactive or terminal sequence session cannot grow backing",
                    ));
                }
            };
            let backing = self.resources.lock_backing_state()?;
            let current = Arc::clone(&backing.current);
            if target.sequences() != 1 {
                return Err(invalid_resource(
                    "sequence backing coverage target must contain exactly one sequence",
                ));
            }
            if current.committed_tokens() >= target.tokens()
                && current.committed_pages() >= target.pages()
            {
                return Ok(SequenceResourceExtensionDecision::Current(current));
            }
            if target.tokens() < current.committed_tokens()
                || target.pages() < current.committed_pages()
            {
                return Err(invalid_resource(
                    "sequence backing coverage target is incomparable with committed work",
                ));
            }
            if active.active_frame.is_some() || !active.participant_flights.is_empty() {
                return Ok(SequenceResourceExtensionDecision::RetryRequired(current));
            }
            current
        };

        let plan = &self.resources.request.plan;
        let (demand, requested_slices) = plan.sequence_extension_demand(
            expected.committed_shape(),
            target,
            request.pressure_action,
        )?;
        let extension = if demand.immediate_claim().is_empty() {
            if !requested_slices.is_empty() {
                return Err(invalid_resource(
                    "empty sequence extension demand produced physical backing requests",
                ));
            }
            PreparedSequenceExtension::empty()
        } else {
            let prepared = match plan.prepare_backing_slices(requested_slices)? {
                BackingPrepareDecision::Prepared(prepared) => prepared,
                BackingPrepareDecision::Deferred(deferred) => {
                    return Ok(SequenceResourceExtensionDecision::BackingDeferred(
                        SequenceExtensionBackingDeferral {
                            backing: PlanBackingDeferral::new(
                                Arc::clone(&plan.resources),
                                deferred,
                            )?,
                            session: Arc::clone(self),
                            expected_generation: expected.generation(),
                            target_fingerprint,
                        },
                    ));
                }
            };
            let capacity = match plan
                .logical_admission()
                .try_claim_for_sequence(self.resources.logical_lease(), &demand)?
            {
                CapacityClaimDecision::Claimed(capacity) => capacity,
                CapacityClaimDecision::Deferred(deferred) => {
                    return Ok(SequenceResourceExtensionDecision::Deferred(deferred));
                }
                CapacityClaimDecision::PermanentRejected(rejected) => {
                    return Ok(SequenceResourceExtensionDecision::PermanentRejected(
                        rejected,
                    ));
                }
            };
            let extension = PreparedSequenceExtension::claimed(prepared, capacity);
            let capacity = extension
                .capacity()
                .expect("claimed sequence extension owns logical capacity");
            if !plan.logical_admission().owns_capacity_claim(capacity)
                || capacity.sequence() != self.sequence_authority()
                || capacity.request() != self.request_authority()
                || capacity.claims() != demand.immediate_claim()
            {
                return Err(invalid_resource(
                    "sequence extension capacity belongs to another authority or demand",
                ));
            }
            extension
        };

        let slot = self
            .slot
            .state
            .lock()
            .map_err(|_| invalid_resource("sequence session state mutex is poisoned"))?;
        let active = match &*slot {
            SequenceSessionSlotState::Active(active)
                if active.epoch == self.epoch
                    && active.fingerprint == self.fingerprint
                    && active.phase == SequenceSessionPhase::Open =>
            {
                active
            }
            SequenceSessionSlotState::Active(active)
                if active.epoch != self.epoch || active.fingerprint != self.fingerprint =>
            {
                return Err(invalid_resource("stale sequence session authority"));
            }
            SequenceSessionSlotState::Active(_) => {
                return Err(invalid_resource(
                    "sequence backing can only grow for an open session",
                ));
            }
            _ => {
                return Err(invalid_resource(
                    "inactive or terminal sequence session cannot grow backing",
                ));
            }
        };
        let mut backing = self.resources.lock_backing_state()?;
        if !Arc::ptr_eq(&backing.current, &expected) {
            let current = Arc::clone(&backing.current);
            if current.committed_tokens() >= target.tokens()
                && current.committed_pages() >= target.pages()
            {
                return Ok(SequenceResourceExtensionDecision::Current(current));
            }
            return Ok(SequenceResourceExtensionDecision::RetryRequired(current));
        }
        if active.active_frame.is_some() || !active.participant_flights.is_empty() {
            return Ok(SequenceResourceExtensionDecision::RetryRequired(
                Arc::clone(&backing.current),
            ));
        }

        let extension = extension.commit();
        let next = Arc::new(SequenceBackingSnapshot::advanced(
            &expected, extension, target,
        )?);
        backing.current = Arc::clone(&next);
        Ok(SequenceResourceExtensionDecision::Extended(next))
    }

    pub fn request_cancel(&self) -> Result<SequenceSessionCancelSnapshot, VNextError> {
        let mut state = self
            .slot
            .state
            .lock()
            .map_err(|_| invalid_resource("sequence session state mutex is poisoned"))?;
        let active = match &mut *state {
            SequenceSessionSlotState::Active(active)
                if active.epoch == self.epoch && active.fingerprint == self.fingerprint =>
            {
                active
            }
            SequenceSessionSlotState::Active(_) => {
                return Err(invalid_resource("stale sequence session authority"));
            }
            SequenceSessionSlotState::Terminal(_) => {
                return Err(invalid_resource("sequence session is already terminal"));
            }
            SequenceSessionSlotState::Dormant { .. } => {
                return Err(invalid_resource("sequence session is not active"));
            }
            SequenceSessionSlotState::FailClosed => {
                return Err(invalid_resource("sequence session is fail-closed"));
            }
        };
        match active.phase {
            SequenceSessionPhase::Open => active.phase = SequenceSessionPhase::CancelRequested,
            SequenceSessionPhase::CancelRequested => {}
            SequenceSessionPhase::Poisoned => {
                return Err(invalid_resource(
                    "poisoned sequence session cannot be cancelled",
                ));
            }
        }
        Ok(SequenceSessionCancelSnapshot {
            active_frame: active.active_frame.map(|frame| frame.frame_id),
            participant_flights: u64::try_from(active.participant_flights.len())
                .map_err(|_| invalid_resource("participant flight count exceeds u64"))?,
        })
    }

    pub fn try_complete(&self) -> Result<SequenceSessionTerminalReceipt, VNextError> {
        self.terminalize(SequenceSessionTerminalDisposition::Completed)
    }

    pub fn try_abort(&self) -> Result<SequenceSessionTerminalReceipt, VNextError> {
        self.terminalize(SequenceSessionTerminalDisposition::Aborted)
    }

    /// Atomically abort an idle session without publishing an intermediate
    /// cancellation state.
    ///
    /// Capacity preemption uses this transition after a provider reports a
    /// pre-submit deferral. If a frame or participant flight is still live,
    /// the operation fails without changing the session phase so its caller
    /// can reconcile the scheduling transaction without a half-cancelled
    /// sequence.
    pub fn try_abort_if_quiescent(&self) -> Result<SequenceSessionTerminalReceipt, VNextError> {
        let mut state = self
            .slot
            .state
            .lock()
            .map_err(|_| invalid_resource("sequence session state mutex is poisoned"))?;
        let active = match &*state {
            SequenceSessionSlotState::Active(active)
                if active.epoch == self.epoch && active.fingerprint == self.fingerprint =>
            {
                active
            }
            SequenceSessionSlotState::Active(_) => {
                return Err(invalid_resource("stale sequence session authority"));
            }
            SequenceSessionSlotState::Terminal(_) => {
                return Err(invalid_resource("sequence session is already terminal"));
            }
            SequenceSessionSlotState::Dormant { .. } => {
                return Err(invalid_resource("sequence session is not active"));
            }
            SequenceSessionSlotState::FailClosed => {
                return Err(invalid_resource("sequence session is fail-closed"));
            }
        };
        if active.phase == SequenceSessionPhase::Poisoned
            || active.active_frame.is_some()
            || !active.participant_flights.is_empty()
        {
            return Err(invalid_resource(
                "quiescent sequence abort requires an open or cancel-requested phase, no active frame, and no participant flight",
            ));
        }
        let receipt = SequenceSessionTerminalReceipt {
            epoch: active.epoch,
            fingerprint: active.fingerprint.clone(),
            disposition: SequenceSessionTerminalDisposition::Aborted,
            retired_frames: active.retired_frames,
        };
        *state = SequenceSessionSlotState::Terminal(receipt.clone());
        Ok(receipt)
    }

    fn terminalize(
        &self,
        disposition: SequenceSessionTerminalDisposition,
    ) -> Result<SequenceSessionTerminalReceipt, VNextError> {
        let mut state = self
            .slot
            .state
            .lock()
            .map_err(|_| invalid_resource("sequence session state mutex is poisoned"))?;
        let active = match &*state {
            SequenceSessionSlotState::Active(active)
                if active.epoch == self.epoch && active.fingerprint == self.fingerprint =>
            {
                active
            }
            SequenceSessionSlotState::Active(_) => {
                return Err(invalid_resource("stale sequence session authority"));
            }
            SequenceSessionSlotState::Terminal(_) => {
                return Err(invalid_resource("sequence session is already terminal"));
            }
            SequenceSessionSlotState::Dormant { .. } => {
                return Err(invalid_resource("sequence session is not active"));
            }
            SequenceSessionSlotState::FailClosed => {
                return Err(invalid_resource("sequence session is fail-closed"));
            }
        };
        let phase_matches = match disposition {
            SequenceSessionTerminalDisposition::Completed => {
                active.phase == SequenceSessionPhase::Open && active.retired_frames > 0
            }
            SequenceSessionTerminalDisposition::Aborted => matches!(
                active.phase,
                SequenceSessionPhase::CancelRequested | SequenceSessionPhase::Poisoned
            ),
        };
        if !phase_matches || active.active_frame.is_some() || !active.participant_flights.is_empty()
        {
            return Err(invalid_resource(
                "sequence terminalization requires the matching phase, no active frame, and no participant flight",
            ));
        }
        let receipt = SequenceSessionTerminalReceipt {
            epoch: active.epoch,
            fingerprint: active.fingerprint.clone(),
            disposition,
            retired_frames: active.retired_frames,
        };
        *state = SequenceSessionSlotState::Terminal(receipt.clone());
        Ok(receipt)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SequenceSessionCancelSnapshot {
    active_frame: Option<ExecutionFrameId>,
    participant_flights: u64,
}

impl SequenceSessionCancelSnapshot {
    pub const fn active_frame(self) -> Option<ExecutionFrameId> {
        self.active_frame
    }

    pub const fn participant_flights(self) -> u64 {
        self.participant_flights
    }
}

impl<R> Drop for SequenceSession<R>
where
    R: DeviceRuntime,
{
    fn drop(&mut self) {
        let mut state = match self.slot.state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        };
        if matches!(
            &*state,
            SequenceSessionSlotState::Active(active)
                if active.epoch == self.epoch && active.fingerprint == self.fingerprint
        ) {
            *state = SequenceSessionSlotState::FailClosed;
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SequenceExecutionAuthoritySource {
    Unselected,
    LegacyStream,
    SequenceSession,
    FailClosed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub struct SequenceBackingGeneration(NonZeroU64);

impl SequenceBackingGeneration {
    const INITIAL: Self = Self(NonZeroU64::MIN);

    pub const fn get(self) -> u64 {
        self.0.get()
    }
}

/// Shared logical parent for every physical backing generation. Keeping the
/// request after the sequence lease preserves the admission hierarchy even
/// when a fence-owned snapshot outlives `AdmittedSequenceResources`.
struct SequenceLogicalOwner<R>
where
    R: DeviceRuntime,
{
    logical_lease: LogicalAdmissionLease,
    _request: Arc<AdmittedRequestResources<R>>,
}

/// Prepared physical capacity must roll back before its logical claim wakes a
/// waiter. Field order is therefore part of this transaction's contract.
struct PreparedSequenceExtension<R>
where
    R: DeviceRuntime,
{
    prepared: Option<PreparedBackingClaim<R>>,
    capacity: Option<LogicalCapacityLease>,
}

impl<R> PreparedSequenceExtension<R>
where
    R: DeviceRuntime,
{
    fn empty() -> Self {
        Self {
            prepared: None,
            capacity: None,
        }
    }

    fn claimed(prepared: PreparedBackingClaim<R>, capacity: LogicalCapacityLease) -> Self {
        Self {
            prepared: Some(prepared),
            capacity: Some(capacity),
        }
    }

    fn capacity(&self) -> Option<&LogicalCapacityLease> {
        self.capacity.as_ref()
    }

    fn commit(mut self) -> CommittedSequenceExtension {
        let backing_slices = self
            .prepared
            .take()
            .map(PreparedBackingClaim::commit)
            .unwrap_or_default();
        CommittedSequenceExtension {
            backing_slices,
            capacity: self.capacity.take(),
        }
    }
}

/// The same physical-before-logical drop order is preserved after pool commit
/// in case snapshot validation rejects publication.
struct CommittedSequenceExtension {
    backing_slices: Vec<LogicalBackingSliceAuthority>,
    capacity: Option<LogicalCapacityLease>,
}

impl<R> SequenceLogicalOwner<R>
where
    R: DeviceRuntime,
{
    fn lease(&self) -> &LogicalAdmissionLease {
        &self.logical_lease
    }
}

/// Immutable sequence-lifetime backing captured by one scheduler step. Later
/// generations may append capacity, while an in-flight step retains the exact
/// generation it submitted until its completion ownership is released.
#[must_use = "a sequence backing snapshot owns its exact physical extents"]
pub struct SequenceBackingSnapshot<R>
where
    R: DeviceRuntime,
{
    generation: SequenceBackingGeneration,
    backing_slices: Vec<LogicalBackingSliceAuthority>,
    // Extension capacity drops after its physical extents. Arcs let a newer
    // generation share old exact claims without exposing public Clone on a
    // linear logical capacity lease.
    extension_capacity: Vec<Arc<LogicalCapacityLease>>,
    committed_shape: DynamicResourceShape,
    // Declared last so physical extents and child capacity release before the
    // parent sequence lease, request, runtime, and pool owner.
    _logical_owner: Arc<SequenceLogicalOwner<R>>,
}

impl<R> SequenceBackingSnapshot<R>
where
    R: DeviceRuntime,
{
    fn initial(
        backing_slices: Vec<LogicalBackingSliceAuthority>,
        work_shape: ResourceWorkShape,
        logical_owner: Arc<SequenceLogicalOwner<R>>,
    ) -> Result<Self, VNextError> {
        if work_shape.immediate_sequences() != 1
            || work_shape.fit_sequences() != 1
            || backing_slices
                .windows(2)
                .any(|pair| pair[0].resource_id() >= pair[1].resource_id())
        {
            return Err(invalid_resource(
                "initial sequence backing must be canonical and single-sequence",
            ));
        }
        Ok(Self {
            generation: SequenceBackingGeneration::INITIAL,
            backing_slices,
            extension_capacity: Vec::new(),
            committed_shape: work_shape.immediate_shape(),
            _logical_owner: logical_owner,
        })
    }

    fn advanced(
        prior: &Self,
        mut extension: CommittedSequenceExtension,
        committed_shape: DynamicResourceShape,
    ) -> Result<Self, VNextError> {
        let generation = prior
            .generation
            .get()
            .checked_add(1)
            .and_then(NonZeroU64::new)
            .map(SequenceBackingGeneration)
            .ok_or_else(|| invalid_resource("sequence backing generation space is exhausted"))?;
        if committed_shape.sequences() != 1
            || committed_shape.tokens() < prior.committed_shape.tokens()
            || committed_shape.pages() < prior.committed_shape.pages()
            || (committed_shape.tokens() == prior.committed_shape.tokens()
                && committed_shape.pages() == prior.committed_shape.pages())
            || extension.backing_slices.is_empty() != extension.capacity.is_none()
        {
            return Err(invalid_resource(
                "sequence backing generation must monotonically advance with matching capacity",
            ));
        }
        extension
            .backing_slices
            .sort_by(|left, right| left.resource_id().cmp(right.resource_id()));
        if extension
            .backing_slices
            .windows(2)
            .any(|pair| pair[0].resource_id() >= pair[1].resource_id())
        {
            return Err(invalid_resource(
                "sequence backing extension slices must be canonical and unique",
            ));
        }
        let mut backing_slices = prior
            .backing_slices
            .iter()
            .map(LogicalBackingSliceAuthority::retained)
            .collect::<Vec<_>>();
        backing_slices.append(&mut extension.backing_slices);
        backing_slices.sort_by(|left, right| {
            left.resource_id().cmp(right.resource_id()).then_with(|| {
                left.evidence()
                    .segment_generation()
                    .cmp(&right.evidence().segment_generation())
            })
        });
        let mut extension_capacity = prior.extension_capacity.clone();
        if let Some(capacity) = extension.capacity.take() {
            extension_capacity.push(Arc::new(capacity));
        }
        Ok(Self {
            generation,
            backing_slices,
            extension_capacity,
            committed_shape,
            _logical_owner: Arc::clone(&prior._logical_owner),
        })
    }

    pub const fn generation(&self) -> SequenceBackingGeneration {
        self.generation
    }

    pub fn backing_slices(&self) -> &[LogicalBackingSliceAuthority] {
        &self.backing_slices
    }

    pub const fn committed_tokens(&self) -> u64 {
        self.committed_shape.tokens()
    }

    pub const fn committed_pages(&self) -> u64 {
        self.committed_shape.pages()
    }

    pub(crate) const fn committed_shape(&self) -> DynamicResourceShape {
        self.committed_shape
    }

    pub(super) fn backing_slices_for(
        &self,
        resource_id: &ResourceId,
    ) -> &[LogicalBackingSliceAuthority] {
        let start = self
            .backing_slices
            .partition_point(|slice| slice.resource_id() < resource_id);
        let end = self
            .backing_slices
            .partition_point(|slice| slice.resource_id() <= resource_id);
        &self.backing_slices[start..end]
    }
}

pub(super) struct SequenceBackingState<R>
where
    R: DeviceRuntime,
{
    pub(super) current: Arc<SequenceBackingSnapshot<R>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SequenceResourceExtensionRequest {
    target_work: ResourceWorkShape,
    pressure_action: AdmissionPressureAction,
}

impl SequenceResourceExtensionRequest {
    pub fn new(
        target_work: ResourceWorkShape,
        pressure_action: AdmissionPressureAction,
    ) -> Result<Self, VNextError> {
        if target_work.immediate_sequences() != 1 || target_work.fit_sequences() != 1 {
            return Err(invalid_resource(
                "sequence extension target requires single-sequence work evidence",
            ));
        }
        Ok(Self {
            target_work,
            pressure_action,
        })
    }
}

pub enum SequenceResourceExtensionDecision<R>
where
    R: DeviceRuntime,
{
    Current(Arc<SequenceBackingSnapshot<R>>),
    Extended(Arc<SequenceBackingSnapshot<R>>),
    RetryRequired(Arc<SequenceBackingSnapshot<R>>),
    Deferred(AdmissionDeferred),
    BackingDeferred(SequenceExtensionBackingDeferral<R>),
    PermanentRejected(AdmissionRejected),
}

/// Non-cloneable authority for backing growth of one exact open sequence
/// generation and target work shape.
#[must_use = "sequence extension backing must be maintained or explicitly dropped"]
pub struct SequenceExtensionBackingDeferral<R>
where
    R: DeviceRuntime,
{
    backing: PlanBackingDeferral<R>,
    session: Arc<SequenceSession<R>>,
    expected_generation: SequenceBackingGeneration,
    target_fingerprint: String,
}

impl<R> SequenceExtensionBackingDeferral<R>
where
    R: DeviceRuntime,
{
    pub fn evidence(&self) -> &DynamicBackingDeferred {
        self.backing.evidence()
    }

    pub fn expected_generation(&self) -> SequenceBackingGeneration {
        self.expected_generation
    }

    pub fn target_fingerprint(&self) -> &str {
        &self.target_fingerprint
    }

    pub fn maintain(&self) -> Result<DynamicDeferredMaintenanceOutcome, VNextError> {
        self.session.ensure_open_identity()?;
        let current_generation = self
            .session
            .resources
            .lock_backing_state()?
            .current
            .generation();
        if current_generation != self.expected_generation {
            return self.backing.retry_admission();
        }
        self.backing.maintain()
    }

    pub fn register_waiter(&self) -> Result<PlanCapacityWaitRegistration<R>, VNextError> {
        self.backing.register_waiter()
    }
}

/// Sequence authority. There is exactly one state cell for the exact
/// `SequenceAuthorityId` issued by B1; no ceiling-sized slot vector and no
/// caller-selected slot allocator exist here.
#[must_use = "logical sequence resources release capacity when dropped"]
pub struct AdmittedSequenceResources<R>
where
    R: DeviceRuntime,
{
    // Recovery records own undrained raw streams. They must drop before the
    // backing slices and logical lease can make those resources reusable.
    pub(super) sequence_recovery: ManuallyDrop<Arc<SequenceRecoveryRegistry<R>>>,
    backing_state: ManuallyDrop<Mutex<SequenceBackingState<R>>>,
    logical_owner: ManuallyDrop<Arc<SequenceLogicalOwner<R>>>,
    pub(super) request: ManuallyDrop<Arc<AdmittedRequestResources<R>>>,
    pub(super) authority_source: Mutex<SequenceExecutionAuthoritySource>,
    session_slot: Arc<SequenceSessionSlot>,
    pub(super) state: Arc<AtomicU64>,
    pub(super) sequence_dispatch_gate: Arc<AtomicU64>,
    pub(super) next_activation_epoch: AtomicU64,
}

struct DeferredSequenceResourceCleanup<R>
where
    R: DeviceRuntime,
{
    sequence_recovery: ManuallyDrop<Arc<SequenceRecoveryRegistry<R>>>,
    backing_state: ManuallyDrop<Mutex<SequenceBackingState<R>>>,
    logical_owner: ManuallyDrop<Arc<SequenceLogicalOwner<R>>>,
    request: ManuallyDrop<Arc<AdmittedRequestResources<R>>>,
    completed: bool,
}

impl<R> DeferredDeviceCleanupTask for DeferredSequenceResourceCleanup<R>
where
    R: DeviceRuntime,
{
    fn try_cleanup(&mut self) -> DeferredDeviceCleanupDisposition {
        if self.completed {
            return DeferredDeviceCleanupDisposition::Completed;
        }
        let recovery = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.sequence_recovery
                .recover_all_for_owner_drop(self.request.plan.runtime())
        }));
        if !matches!(recovery, Ok(Ok(()))) {
            return DeferredDeviceCleanupDisposition::Retryable;
        }

        // SAFETY: successful recovery removed and destroyed every raw stream.
        // This registry-owned unit is the sole owner of these fields and releases
        // them once, in dependency order.
        unsafe {
            ManuallyDrop::drop(&mut self.sequence_recovery);
            ManuallyDrop::drop(&mut self.backing_state);
            ManuallyDrop::drop(&mut self.logical_owner);
            ManuallyDrop::drop(&mut self.request);
        }
        self.completed = true;
        DeferredDeviceCleanupDisposition::Completed
    }
}

impl<R> AdmittedSequenceResources<R>
where
    R: DeviceRuntime,
{
    fn new(
        request: Arc<AdmittedRequestResources<R>>,
        logical_lease: LogicalAdmissionLease,
        backing_slices: Vec<LogicalBackingSliceAuthority>,
        work_shape: ResourceWorkShape,
    ) -> Result<Self, VNextError> {
        if !request.plan.logical_admission().owns(&logical_lease)
            || logical_lease.request() != request.request_authority()
        {
            return Err(invalid_resource(
                "logical sequence authority belongs to another request",
            ));
        }
        let plan_resources = Arc::clone(&request.plan.resources);
        let logical_owner = Arc::new(SequenceLogicalOwner {
            logical_lease,
            _request: Arc::clone(&request),
        });
        let backing_snapshot = Arc::new(SequenceBackingSnapshot::initial(
            backing_slices,
            work_shape,
            Arc::clone(&logical_owner),
        )?);
        Ok(Self {
            sequence_recovery: ManuallyDrop::new(Arc::new(SequenceRecoveryRegistry::new(
                plan_resources,
            ))),
            backing_state: ManuallyDrop::new(Mutex::new(SequenceBackingState {
                current: backing_snapshot,
            })),
            logical_owner: ManuallyDrop::new(logical_owner),
            request: ManuallyDrop::new(request),
            authority_source: Mutex::new(SequenceExecutionAuthoritySource::Unselected),
            session_slot: Arc::new(SequenceSessionSlot::new()),
            state: Arc::new(AtomicU64::new(0)),
            sequence_dispatch_gate: Arc::new(AtomicU64::new(0)),
            next_activation_epoch: AtomicU64::new(1),
        })
    }

    pub fn sequence_authority(&self) -> SequenceAuthorityId {
        self.logical_lease().sequence()
    }

    pub fn request_authority(&self) -> RequestAuthorityId {
        self.logical_lease().request()
    }

    pub fn coordinator_id(&self) -> LogicalAdmissionCoordinatorId {
        self.logical_lease().coordinator_id()
    }

    pub fn run_id(&self) -> &RunId {
        self.request.run_id()
    }

    pub fn request_id(&self) -> &RequestIdentity {
        self.request.request_id()
    }

    pub(super) fn logical_lease(&self) -> &LogicalAdmissionLease {
        self.logical_owner.lease()
    }

    pub(super) fn lock_backing_state(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, SequenceBackingState<R>>, VNextError> {
        self.backing_state
            .lock()
            .map_err(|_| invalid_resource("sequence backing state mutex is poisoned"))
    }

    pub(crate) fn backing_snapshot(&self) -> Result<Arc<SequenceBackingSnapshot<R>>, VNextError> {
        Ok(Arc::clone(&self.lock_backing_state()?.current))
    }

    pub fn request_resources(&self) -> &Arc<AdmittedRequestResources<R>> {
        &self.request
    }

    pub fn backing_generation(&self) -> Result<SequenceBackingGeneration, VNextError> {
        Ok(self.lock_backing_state()?.current.generation())
    }

    pub fn static_provisioning(&self) -> Option<&StaticProvisioningLease<R>> {
        self.request.static_provisioning()
    }

    pub fn plan_evidence(&self) -> TrustedPlanRuntimeEvidence {
        self.request.plan_evidence()
    }

    pub(crate) fn device_buffer_retention(&self) -> DeviceBufferRetention {
        DeviceBufferRetention::new(Arc::clone(&self.request.plan.resources))
    }

    pub(super) fn lock_authority_source(
        &self,
    ) -> Result<std::sync::MutexGuard<'_, SequenceExecutionAuthoritySource>, VNextError> {
        match self.authority_source.lock() {
            Ok(source) => Ok(source),
            Err(poisoned) => {
                let mut source = poisoned.into_inner();
                *source = SequenceExecutionAuthoritySource::FailClosed;
                Err(invalid_resource(
                    "logical sequence execution authority selector is fail-closed",
                ))
            }
        }
    }

    fn authority_source_is_fail_closed(&self) -> bool {
        match self.authority_source.lock() {
            Ok(source) => *source == SequenceExecutionAuthoritySource::FailClosed,
            Err(poisoned) => {
                *poisoned.into_inner() = SequenceExecutionAuthoritySource::FailClosed;
                true
            }
        }
    }

    pub fn open_session(self: &Arc<Self>) -> Result<Arc<SequenceSession<R>>, VNextError> {
        let _lifecycle = self
            .request
            .plan
            .resources
            .read_lifecycle("open a sequence session")?;
        #[derive(Serialize)]
        struct FingerprintInput<'a> {
            plan: &'a TrustedPlanRuntimeEvidence,
            coordinator_id: LogicalAdmissionCoordinatorId,
            request_authority: RequestAuthorityId,
            sequence_authority: SequenceAuthorityId,
            run_id: &'a RunId,
            request_id: &'a RequestIdentity,
            epoch: SequenceSessionEpoch,
            request_backing: Vec<&'a LogicalBackingSliceEvidence>,
        }

        let mut authority_source = self.lock_authority_source()?;
        let selecting_session = match *authority_source {
            SequenceExecutionAuthoritySource::Unselected => true,
            SequenceExecutionAuthoritySource::SequenceSession => false,
            SequenceExecutionAuthoritySource::LegacyStream => {
                return Err(invalid_resource(
                    "logical sequence execution authority is permanently selected for legacy streams",
                ));
            }
            SequenceExecutionAuthoritySource::FailClosed => {
                return Err(invalid_resource(
                    "logical sequence execution authority selector is fail-closed",
                ));
            }
        };
        let mut state = match self.session_slot.state.lock() {
            Ok(state) => state,
            Err(_) => {
                *authority_source = SequenceExecutionAuthoritySource::FailClosed;
                return Err(invalid_resource("sequence session state mutex is poisoned"));
            }
        };
        let epoch = match &*state {
            SequenceSessionSlotState::Dormant {
                next_epoch: Some(epoch),
            } if selecting_session => *epoch,
            SequenceSessionSlotState::Dormant {
                next_epoch: Some(_),
            } => {
                *authority_source = SequenceExecutionAuthoritySource::FailClosed;
                return Err(invalid_resource(
                    "sequence session authority source lost its active or terminal slot",
                ));
            }
            SequenceSessionSlotState::Dormant { next_epoch: None } => {
                *authority_source = SequenceExecutionAuthoritySource::FailClosed;
                return Err(invalid_resource(
                    "sequence session epoch space is exhausted",
                ));
            }
            SequenceSessionSlotState::Active(_) => {
                return Err(invalid_resource(
                    "logical sequence already has an active session",
                ));
            }
            SequenceSessionSlotState::Terminal(_) => {
                return Err(invalid_resource("logical sequence is already terminal"));
            }
            SequenceSessionSlotState::FailClosed => {
                return Err(invalid_resource(
                    "logical sequence session slot is fail-closed",
                ));
            }
        };
        let plan = self.plan_evidence();
        let request_backing = self
            .request
            .backing_slices
            .iter()
            .map(LogicalBackingSliceAuthority::evidence)
            .collect();
        let input = FingerprintInput {
            plan: &plan,
            coordinator_id: self.coordinator_id(),
            request_authority: self.request_authority(),
            sequence_authority: self.sequence_authority(),
            run_id: self.run_id(),
            request_id: self.request_id(),
            epoch,
            request_backing,
        };
        let fingerprint = match sequence_session_fingerprint(&input) {
            Ok(fingerprint) => fingerprint,
            Err(error) => {
                *authority_source = SequenceExecutionAuthoritySource::FailClosed;
                return Err(error);
            }
        };
        let next_epoch = epoch
            .get()
            .checked_add(1)
            .and_then(NonZeroU64::new)
            .map(SequenceSessionEpoch);
        *state = SequenceSessionSlotState::Active(ActiveSequenceSessionState {
            epoch,
            fingerprint: fingerprint.clone(),
            phase: SequenceSessionPhase::Open,
            next_frame: Some(
                ExecutionFrameId::try_from(1_u64)
                    .expect("the first execution frame id is non-zero"),
            ),
            active_frame: None,
            participant_flights: BTreeMap::new(),
            retired_frames: 0,
        });
        *authority_source = SequenceExecutionAuthoritySource::SequenceSession;
        // A logical sequence is one-shot today. Retaining the checked successor
        // makes epoch exhaustion explicit if recovery later permits reopening.
        let _ = next_epoch;
        drop(state);
        Ok(Arc::new(SequenceSession {
            resources: Arc::clone(self),
            slot: Arc::clone(&self.session_slot),
            epoch,
            fingerprint,
        }))
    }

    pub fn is_poisoned(&self) -> bool {
        self.authority_source_is_fail_closed()
            || self.session_slot.is_poisoned()
            || sequence_dispatch_is_poisoned(&self.sequence_dispatch_gate)
            || sequence_slot_is_poisoned(self.state.load(Ordering::Acquire))
    }
}

impl<R> Drop for AdmittedSequenceResources<R>
where
    R: DeviceRuntime,
{
    fn drop(&mut self) {
        if !self.sequence_recovery.is_empty() {
            self.sequence_dispatch_gate
                .fetch_or(SEQUENCE_DISPATCH_POISONED_BIT, Ordering::AcqRel);
            let cleanup_domain = self.request.plan.resources.deferred_cleanup_domain;
            // SAFETY: this Drop implementation runs once and transfers all four
            // ManuallyDrop fields into one aggregate recovery owner.
            let cleanup = unsafe {
                DeferredSequenceResourceCleanup {
                    sequence_recovery: ManuallyDrop::new(ManuallyDrop::take(
                        &mut self.sequence_recovery,
                    )),
                    backing_state: ManuallyDrop::new(ManuallyDrop::take(&mut self.backing_state)),
                    logical_owner: ManuallyDrop::new(ManuallyDrop::take(&mut self.logical_owner)),
                    request: ManuallyDrop::new(ManuallyDrop::take(&mut self.request)),
                    completed: false,
                }
            };
            defer_device_cleanup(cleanup_domain, cleanup);
            return;
        }

        // SAFETY: an empty registry proves there is no raw stream whose backend
        // quiescence could depend on these resources. This is pure ownership
        // teardown and performs no backend call.
        unsafe {
            ManuallyDrop::drop(&mut self.sequence_recovery);
            ManuallyDrop::drop(&mut self.backing_state);
            ManuallyDrop::drop(&mut self.logical_owner);
            ManuallyDrop::drop(&mut self.request);
        }
    }
}
