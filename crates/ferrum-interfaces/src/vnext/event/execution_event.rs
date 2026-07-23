use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use super::{
    canonical_fingerprint, invalid_event, validate_sha256, ExecutionEventKind, ExecutionFrameId,
    ExecutionIdentityEnvelope, ExecutionIdentityParts, ExecutionPhase, FailureDomain,
    FailureEnvelope, FailureEnvelopeWire, MonotonicTimestamp, NodeId, NodeInvocationId,
    OperationParticipantCompletionReceipt, RequestIdentity, ResourceTransactionIdentity, RunId,
    SpanId, SubmittedOperationReceipt, TrustedAbortedSequenceBinding, TrustedActiveSequenceBinding,
    TrustedCompletedSequenceBinding, TrustedExecutionTopology, UnvalidatedExecutionIdentityParts,
    UnvalidatedFailureEnvelope, VNextError, MAX_EXECUTION_EVENT_WIRE_BYTES,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct IdentifiedFailure {
    identity: ExecutionIdentityEnvelope,
    failure: FailureEnvelope,
}

impl IdentifiedFailure {
    pub fn new(
        identity: ExecutionIdentityEnvelope,
        failure: FailureEnvelope,
    ) -> Result<Self, VNextError> {
        failure.validate()?;
        validate_failure_identity(failure.domain(), identity.parts())?;
        Ok(Self { identity, failure })
    }

    pub fn identity(&self) -> &ExecutionIdentityEnvelope {
        &self.identity
    }

    pub fn failure(&self) -> &FailureEnvelope {
        &self.failure
    }

    pub fn fingerprint(&self) -> String {
        canonical_fingerprint(self)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UnvalidatedIdentifiedFailure {
    identity: UnvalidatedExecutionIdentityParts,
    failure: UnvalidatedFailureEnvelope,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UnvalidatedIdentifiedFailureWire {
    identity: UnvalidatedExecutionIdentityParts,
    failure: FailureEnvelopeWire,
}

impl From<UnvalidatedIdentifiedFailureWire> for UnvalidatedIdentifiedFailure {
    fn from(wire: UnvalidatedIdentifiedFailureWire) -> Self {
        Self {
            identity: wire.identity,
            failure: wire.failure.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionEventDetail {
    None,
    Counters { input: u64, output: u64 },
    Failure(IdentifiedFailure),
    FailureTerminal { first_failure_fingerprint: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UnvalidatedExecutionEventDetail {
    None,
    Counters { input: u64, output: u64 },
    Failure(UnvalidatedIdentifiedFailure),
    FailureTerminal { first_failure_fingerprint: String },
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum UnvalidatedExecutionEventDetailWire {
    None,
    Counters { input: u64, output: u64 },
    Failure(UnvalidatedIdentifiedFailureWire),
    FailureTerminal { first_failure_fingerprint: String },
}

impl From<UnvalidatedExecutionEventDetailWire> for UnvalidatedExecutionEventDetail {
    fn from(wire: UnvalidatedExecutionEventDetailWire) -> Self {
        match wire {
            UnvalidatedExecutionEventDetailWire::None => Self::None,
            UnvalidatedExecutionEventDetailWire::Counters { input, output } => {
                Self::Counters { input, output }
            }
            UnvalidatedExecutionEventDetailWire::Failure(failure) => Self::Failure(failure.into()),
            UnvalidatedExecutionEventDetailWire::FailureTerminal {
                first_failure_fingerprint,
            } => Self::FailureTerminal {
                first_failure_fingerprint,
            },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutionEvent {
    timestamp: MonotonicTimestamp,
    phase: ExecutionPhase,
    kind: ExecutionEventKind,
    identity: ExecutionIdentityEnvelope,
    detail: ExecutionEventDetail,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UnvalidatedExecutionEvent {
    timestamp: MonotonicTimestamp,
    phase: ExecutionPhase,
    kind: ExecutionEventKind,
    identity: UnvalidatedExecutionIdentityParts,
    detail: UnvalidatedExecutionEventDetail,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ExecutionEventWire {
    timestamp: MonotonicTimestamp,
    phase: ExecutionPhase,
    kind: ExecutionEventKind,
    identity: UnvalidatedExecutionIdentityParts,
    detail: UnvalidatedExecutionEventDetailWire,
}

impl From<ExecutionEventWire> for UnvalidatedExecutionEvent {
    fn from(wire: ExecutionEventWire) -> Self {
        Self {
            timestamp: wire.timestamp,
            phase: wire.phase,
            kind: wire.kind,
            identity: wire.identity,
            detail: wire.detail.into(),
        }
    }
}

impl ExecutionEvent {
    pub fn new(
        timestamp: MonotonicTimestamp,
        phase: ExecutionPhase,
        kind: ExecutionEventKind,
        identity: ExecutionIdentityEnvelope,
        detail: ExecutionEventDetail,
    ) -> Result<Self, VNextError> {
        validate_event_shape(phase, kind, identity.parts(), &detail)?;
        Ok(Self {
            timestamp,
            phase,
            kind,
            identity,
            detail,
        })
    }

    pub const fn timestamp(&self) -> MonotonicTimestamp {
        self.timestamp
    }

    pub const fn phase(&self) -> ExecutionPhase {
        self.phase
    }

    pub const fn kind(&self) -> ExecutionEventKind {
        self.kind
    }

    pub fn identity(&self) -> &ExecutionIdentityEnvelope {
        &self.identity
    }

    pub fn detail(&self) -> &ExecutionEventDetail {
        &self.detail
    }

    pub fn decode_untrusted(bytes: &[u8]) -> Result<UnvalidatedExecutionEvent, VNextError> {
        if bytes.len() > MAX_EXECUTION_EVENT_WIRE_BYTES {
            return Err(invalid_event(
                "untrusted execution event exceeds the wire byte limit",
            ));
        }
        let raw = serde_json::from_slice::<serde_json::Value>(bytes).map_err(|error| {
            VNextError::Serialization {
                context: "decode untrusted execution event",
                message: error.to_string(),
            }
        })?;
        let event = serde_json::from_value::<ExecutionEventWire>(raw.clone())
            .map(UnvalidatedExecutionEvent::from)
            .map_err(|error| VNextError::Serialization {
                context: "decode untrusted execution event",
                message: error.to_string(),
            })?;
        let canonical =
            serde_json::to_value(&event).map_err(|error| VNextError::Serialization {
                context: "serialize untrusted execution event",
                message: error.to_string(),
            })?;
        if canonical != raw {
            return Err(invalid_event(
                "execution event wire contains unknown or non-canonical nested fields",
            ));
        }
        Ok(event)
    }
}

pub struct TrustedExecutionEventContext<'a> {
    run_id: &'a RunId,
    request_id: &'a RequestIdentity,
    topology: Option<&'a TrustedExecutionTopology>,
    active: Option<&'a TrustedActiveSequenceBinding>,
    completed: Option<&'a TrustedCompletedSequenceBinding>,
    aborted: Option<&'a TrustedAbortedSequenceBinding>,
    submitted_operation: Option<&'a SubmittedOperationReceipt>,
    retired_operation: Option<&'a OperationParticipantCompletionReceipt>,
    expected_failure: Option<&'a IdentifiedFailure>,
    unsubmitted_recovery_identity: Option<&'a ExecutionIdentityEnvelope>,
}

impl<'a> TrustedExecutionEventContext<'a> {
    pub fn pre_plan(run_id: &'a RunId, request_id: &'a RequestIdentity) -> Self {
        Self {
            run_id,
            request_id,
            topology: None,
            active: None,
            completed: None,
            aborted: None,
            submitted_operation: None,
            retired_operation: None,
            expected_failure: None,
            unsubmitted_recovery_identity: None,
        }
    }

    pub fn bound(
        run_id: &'a RunId,
        request_id: &'a RequestIdentity,
        topology: &'a TrustedExecutionTopology,
    ) -> Self {
        Self {
            run_id,
            request_id,
            topology: Some(topology),
            active: None,
            completed: None,
            aborted: None,
            submitted_operation: None,
            retired_operation: None,
            expected_failure: None,
            unsubmitted_recovery_identity: None,
        }
    }

    pub fn active(
        run_id: &'a RunId,
        request_id: &'a RequestIdentity,
        topology: &'a TrustedExecutionTopology,
        active: &'a TrustedActiveSequenceBinding,
    ) -> Self {
        Self {
            run_id,
            request_id,
            topology: Some(topology),
            active: Some(active),
            completed: None,
            aborted: None,
            submitted_operation: None,
            retired_operation: None,
            expected_failure: None,
            unsubmitted_recovery_identity: None,
        }
    }

    pub fn operation_submitted(
        run_id: &'a RunId,
        request_id: &'a RequestIdentity,
        topology: &'a TrustedExecutionTopology,
        active: &'a TrustedActiveSequenceBinding,
        submitted_operation: &'a SubmittedOperationReceipt,
    ) -> Self {
        Self {
            run_id,
            request_id,
            topology: Some(topology),
            active: Some(active),
            completed: None,
            aborted: None,
            submitted_operation: Some(submitted_operation),
            retired_operation: None,
            expected_failure: None,
            unsubmitted_recovery_identity: None,
        }
    }

    pub(super) fn replay_operation_submitted(
        run_id: &'a RunId,
        request_id: &'a RequestIdentity,
        topology: &'a TrustedExecutionTopology,
        active: &'a TrustedActiveSequenceBinding,
        submitted_operation: &'a SubmittedOperationReceipt,
    ) -> Self {
        Self {
            run_id,
            request_id,
            topology: Some(topology),
            active: Some(active),
            completed: None,
            aborted: None,
            submitted_operation: Some(submitted_operation),
            retired_operation: None,
            expected_failure: None,
            unsubmitted_recovery_identity: None,
        }
    }

    pub fn node_retired(
        run_id: &'a RunId,
        request_id: &'a RequestIdentity,
        topology: &'a TrustedExecutionTopology,
        active: &'a TrustedActiveSequenceBinding,
        retired_operation: &'a OperationParticipantCompletionReceipt,
    ) -> Self {
        Self {
            run_id,
            request_id,
            topology: Some(topology),
            active: Some(active),
            completed: None,
            aborted: None,
            submitted_operation: None,
            retired_operation: Some(retired_operation),
            expected_failure: None,
            unsubmitted_recovery_identity: None,
        }
    }

    pub(super) fn replay_node_retired(
        run_id: &'a RunId,
        request_id: &'a RequestIdentity,
        topology: &'a TrustedExecutionTopology,
        active: &'a TrustedActiveSequenceBinding,
        retired_operation: &'a OperationParticipantCompletionReceipt,
    ) -> Self {
        Self::node_retired(run_id, request_id, topology, active, retired_operation)
    }

    pub fn completed(
        run_id: &'a RunId,
        request_id: &'a RequestIdentity,
        topology: &'a TrustedExecutionTopology,
        active: &'a TrustedActiveSequenceBinding,
        completed: &'a TrustedCompletedSequenceBinding,
    ) -> Self {
        Self {
            run_id,
            request_id,
            topology: Some(topology),
            active: Some(active),
            completed: Some(completed),
            aborted: None,
            submitted_operation: None,
            retired_operation: None,
            expected_failure: None,
            unsubmitted_recovery_identity: None,
        }
    }

    pub fn aborted(
        run_id: &'a RunId,
        request_id: &'a RequestIdentity,
        topology: &'a TrustedExecutionTopology,
        active: &'a TrustedActiveSequenceBinding,
        aborted: &'a TrustedAbortedSequenceBinding,
    ) -> Self {
        Self {
            run_id,
            request_id,
            topology: Some(topology),
            active: Some(active),
            completed: None,
            aborted: Some(aborted),
            submitted_operation: None,
            retired_operation: None,
            expected_failure: None,
            unsubmitted_recovery_identity: None,
        }
    }

    pub fn failure(
        run_id: &'a RunId,
        request_id: &'a RequestIdentity,
        topology: Option<&'a TrustedExecutionTopology>,
        active: Option<&'a TrustedActiveSequenceBinding>,
        expected_failure: &'a IdentifiedFailure,
    ) -> Self {
        Self {
            run_id,
            request_id,
            topology,
            active,
            completed: None,
            aborted: None,
            submitted_operation: None,
            retired_operation: None,
            expected_failure: Some(expected_failure),
            unsubmitted_recovery_identity: None,
        }
    }

    pub(super) fn replay_failure(
        run_id: &'a RunId,
        request_id: &'a RequestIdentity,
        topology: Option<&'a TrustedExecutionTopology>,
        active: Option<&'a TrustedActiveSequenceBinding>,
        expected_failure: &'a IdentifiedFailure,
        unsubmitted_recovery_identity: Option<&'a ExecutionIdentityEnvelope>,
    ) -> Self {
        Self {
            run_id,
            request_id,
            topology,
            active,
            completed: None,
            aborted: None,
            submitted_operation: None,
            retired_operation: None,
            expected_failure: Some(expected_failure),
            unsubmitted_recovery_identity,
        }
    }

    pub fn failure_with_disposition(
        run_id: &'a RunId,
        request_id: &'a RequestIdentity,
        topology: &'a TrustedExecutionTopology,
        active: &'a TrustedActiveSequenceBinding,
        completed: Option<&'a TrustedCompletedSequenceBinding>,
        aborted: Option<&'a TrustedAbortedSequenceBinding>,
        expected_failure: &'a IdentifiedFailure,
    ) -> Self {
        Self {
            run_id,
            request_id,
            topology: Some(topology),
            active: Some(active),
            completed,
            aborted,
            submitted_operation: None,
            retired_operation: None,
            expected_failure: Some(expected_failure),
            unsubmitted_recovery_identity: None,
        }
    }

    pub(super) const fn active_binding(&self) -> Option<&'a TrustedActiveSequenceBinding> {
        self.active
    }
}

impl UnvalidatedExecutionEvent {
    pub fn revalidate(
        self,
        context: &TrustedExecutionEventContext<'_>,
    ) -> Result<ExecutionEvent, VNextError> {
        let identity = ExecutionIdentityEnvelope::new(self.identity.into())?;
        let detail = match self.detail {
            UnvalidatedExecutionEventDetail::None => ExecutionEventDetail::None,
            UnvalidatedExecutionEventDetail::Counters { input, output } => {
                ExecutionEventDetail::Counters { input, output }
            }
            UnvalidatedExecutionEventDetail::Failure(failure) => {
                let expected = context.expected_failure.ok_or_else(|| {
                    invalid_event("wire failure lacks independent trusted failure evidence")
                })?;
                let failure_identity = ExecutionIdentityEnvelope::new(failure.identity.into())?;
                let trusted = IdentifiedFailure::new(
                    failure_identity,
                    failure.failure.revalidate(expected.failure().domain())?,
                )?;
                if &trusted != expected {
                    return Err(invalid_event(
                        "wire failure differs from independent failure evidence",
                    ));
                }
                ExecutionEventDetail::Failure(trusted)
            }
            UnvalidatedExecutionEventDetail::FailureTerminal {
                first_failure_fingerprint,
            } => {
                validate_sha256(&first_failure_fingerprint, "first failure fingerprint")?;
                let expected = context.expected_failure.ok_or_else(|| {
                    invalid_event("failure terminal lacks independent first failure evidence")
                })?;
                if first_failure_fingerprint != expected.fingerprint() {
                    return Err(invalid_event(
                        "failure terminal differs from the first observed failure",
                    ));
                }
                ExecutionEventDetail::FailureTerminal {
                    first_failure_fingerprint,
                }
            }
        };
        let event = ExecutionEvent::new(self.timestamp, self.phase, self.kind, identity, detail)?;
        validate_event_against_context(&event, context)?;
        Ok(event)
    }
}

fn has_pool(ids: &ExecutionIdentityParts) -> bool {
    ids.resource_pool_id.is_some()
}

pub(super) fn has_active(ids: &ExecutionIdentityParts) -> bool {
    ids.active_sequence_slot.is_some()
}

pub(super) fn has_completed(ids: &ExecutionIdentityParts) -> bool {
    ids.completed_sequence_fingerprint.is_some()
}

pub(super) fn has_aborted(ids: &ExecutionIdentityParts) -> bool {
    ids.aborted_sequence_fingerprint.is_some()
}

fn no_resource_item(ids: &ExecutionIdentityParts) -> bool {
    ids.resource_id.is_none()
        && ids.resource_generation.is_none()
        && ids.resource_batch_fingerprint.is_none()
}

fn exact_plan(ids: &ExecutionIdentityParts) -> bool {
    ids.plan_id.is_some() && ids.plan_hash.is_some() && ids.device_id.is_some()
}

pub(super) fn same_operation_authority_except_observation(
    observation: &ExecutionIdentityParts,
    operation: &ExecutionIdentityParts,
) -> bool {
    let mut normalized_observation = observation.clone();
    normalized_observation.sequence = operation.sequence;
    normalized_observation.span_id = operation.span_id.clone();
    normalized_observation.parent_span_id = operation.parent_span_id.clone();
    normalized_observation == *operation
}

fn validate_event_shape(
    phase: ExecutionPhase,
    kind: ExecutionEventKind,
    ids: &ExecutionIdentityParts,
    detail: &ExecutionEventDetail,
) -> Result<(), VNextError> {
    let phase_ok = match kind {
        ExecutionEventKind::RequestAccepted => phase == ExecutionPhase::Resolution,
        ExecutionEventKind::PlanBuilt => phase == ExecutionPhase::Planning,
        ExecutionEventKind::FrameStarted
        | ExecutionEventKind::NodeStarted
        | ExecutionEventKind::OperationSubmitted
        | ExecutionEventKind::NodeRetired
        | ExecutionEventKind::FrameCompleted => phase == ExecutionPhase::Execution,
        ExecutionEventKind::FailureObserved => true,
        ExecutionEventKind::SequenceCompleted
        | ExecutionEventKind::SequenceAborted
        | ExecutionEventKind::RequestCompleted => phase == ExecutionPhase::Completion,
        ExecutionEventKind::RequestFailed => true,
    };
    if !phase_ok {
        return Err(invalid_event(format!(
            "event `{kind:?}` is invalid in phase `{phase:?}`"
        )));
    }
    let no_plan = ids.plan_id.is_none() && ids.plan_hash.is_none() && ids.device_id.is_none();
    let no_frame = ids.frame_id.is_none() && ids.node_invocation_id.is_none();
    let no_node = ids.node_id.is_none() && ids.operation_id.is_none() && ids.provider_id.is_none();
    let no_pool = !has_pool(ids) && !has_active(ids) && no_resource_item(ids);
    let frame_shape = exact_plan(ids)
        && ids.frame_id.is_some()
        && ids.node_invocation_id.is_none()
        && no_node
        && has_active(ids)
        && !has_completed(ids)
        && !has_aborted(ids)
        && no_resource_item(ids);
    let node_shape = exact_plan(ids)
        && ids.frame_id.is_some()
        && ids.node_invocation_id.is_some()
        && ids.node_id.is_some()
        && ids.operation_id.is_some()
        && ids.provider_id.is_some()
        && has_active(ids)
        && !has_completed(ids)
        && !has_aborted(ids)
        && no_resource_item(ids);
    let completed_shape = exact_plan(ids)
        && no_frame
        && no_node
        && has_active(ids)
        && has_completed(ids)
        && !has_aborted(ids)
        && no_resource_item(ids);
    let aborted_shape = exact_plan(ids)
        && no_frame
        && no_node
        && has_active(ids)
        && !has_completed(ids)
        && has_aborted(ids)
        && no_resource_item(ids);
    let identity_ok = match kind {
        ExecutionEventKind::RequestAccepted => no_plan && no_frame && no_node && no_pool,
        ExecutionEventKind::PlanBuilt => exact_plan(ids) && no_frame && no_node && no_pool,
        ExecutionEventKind::FrameStarted | ExecutionEventKind::FrameCompleted => frame_shape,
        ExecutionEventKind::NodeStarted
        | ExecutionEventKind::OperationSubmitted
        | ExecutionEventKind::NodeRetired => node_shape,
        ExecutionEventKind::SequenceCompleted | ExecutionEventKind::RequestCompleted => {
            completed_shape
        }
        ExecutionEventKind::SequenceAborted => aborted_shape,
        ExecutionEventKind::FailureObserved => match detail {
            ExecutionEventDetail::Failure(failure) if has_active(ids) => {
                let failed_operation = failure.identity().parts();
                node_shape
                    && ids.sequence > failed_operation.sequence
                    && ids.parent_span_id.as_ref() == Some(&failed_operation.span_id)
                    && same_operation_authority_except_observation(ids, failed_operation)
            }
            ExecutionEventDetail::Failure(failure) => failure.identity().parts() == ids,
            _ => false,
        },
        ExecutionEventKind::RequestFailed => match detail {
            ExecutionEventDetail::Failure(failure) => failure.identity().parts() == ids,
            ExecutionEventDetail::FailureTerminal { .. } => {
                no_frame
                    && no_node
                    && no_resource_item(ids)
                    && (no_plan || exact_plan(ids))
                    && ((!has_active(ids) && !has_completed(ids) && !has_aborted(ids))
                        || (has_active(ids) && (has_completed(ids) ^ has_aborted(ids))))
            }
            _ => false,
        },
    };
    if !identity_ok {
        return Err(invalid_event(format!(
            "event `{kind:?}` has missing or extraneous identity fields"
        )));
    }
    let detail_ok = match (kind, detail) {
        (ExecutionEventKind::RequestCompleted, ExecutionEventDetail::Counters { .. }) => true,
        (ExecutionEventKind::FailureObserved, ExecutionEventDetail::Failure(_)) => true,
        (
            ExecutionEventKind::RequestFailed,
            ExecutionEventDetail::Failure(_) | ExecutionEventDetail::FailureTerminal { .. },
        ) => true,
        (
            ExecutionEventKind::RequestAccepted
            | ExecutionEventKind::PlanBuilt
            | ExecutionEventKind::FrameStarted
            | ExecutionEventKind::NodeStarted
            | ExecutionEventKind::OperationSubmitted
            | ExecutionEventKind::NodeRetired
            | ExecutionEventKind::FrameCompleted
            | ExecutionEventKind::SequenceCompleted
            | ExecutionEventKind::SequenceAborted,
            ExecutionEventDetail::None,
        ) => true,
        _ => false,
    };
    if !detail_ok {
        return Err(invalid_event(format!(
            "event `{kind:?}` has invalid structured detail"
        )));
    }
    if let ExecutionEventDetail::FailureTerminal {
        first_failure_fingerprint,
    } = detail
    {
        validate_sha256(first_failure_fingerprint, "first failure fingerprint")?;
    }
    Ok(())
}

fn validate_active_identity(
    ids: &ExecutionIdentityParts,
    active: &TrustedActiveSequenceBinding,
) -> Result<(), VNextError> {
    let provisioning = active.static_provisioning_identity();
    let pool_fingerprint = active.static_pool_identity_fingerprint_ref();
    if &ids.run_id != active.run_id()
        || &ids.request_id != active.request_id()
        || ids.resource_pool_id != active.static_pool_id()
        || ids.resource_pool_identity_fingerprint.as_deref() != pool_fingerprint
        || ids.provisioning_run_id.as_ref() != provisioning.map(ResourceTransactionIdentity::run_id)
        || ids.provisioning_request_id.as_ref()
            != provisioning.map(ResourceTransactionIdentity::request_id)
        || ids.transaction_id.as_ref()
            != provisioning.map(ResourceTransactionIdentity::transaction_id)
        || ids.active_sequence_slot != Some(active.sequence_authority().sparse_id())
        || ids.admission_generation != Some(active.sequence_authority().generation())
        || ids.activation_epoch != Some(active.activation_epoch())
        || ids.runtime_implementation_fingerprint.as_deref()
            != Some(active.runtime_implementation_fingerprint())
        || ids.active_sequence_fingerprint.as_deref() != Some(active.fingerprint())
    {
        return Err(invalid_event(
            "event active identity differs from pool, epoch, runtime, or provisioning evidence",
        ));
    }
    Ok(())
}

fn validate_completed_identity(
    ids: &ExecutionIdentityParts,
    completed: &TrustedCompletedSequenceBinding,
    active: &TrustedActiveSequenceBinding,
) -> Result<(), VNextError> {
    if completed.active_sequence_fingerprint() != active.fingerprint()
        || completed.plan() != active.plan()
        || completed.coordinator_id() != active.coordinator_id()
        || completed.sequence_authority() != active.sequence_authority()
        || completed.run_id() != active.run_id()
        || completed.request_id() != active.request_id()
        || completed.activation_epoch() != active.activation_epoch()
        || completed.runtime_implementation_fingerprint()
            != active.runtime_implementation_fingerprint()
        || ids.completed_sequence_fingerprint.as_deref() != Some(completed.fingerprint())
    {
        return Err(invalid_event(
            "event completion identity differs from the synchronized active sequence receipt",
        ));
    }
    Ok(())
}

fn validate_aborted_identity(
    ids: &ExecutionIdentityParts,
    aborted: &TrustedAbortedSequenceBinding,
    active: &TrustedActiveSequenceBinding,
) -> Result<(), VNextError> {
    if !active.matches_abort_disposition(aborted.disposition())
        || aborted.active_sequence_fingerprint() != active.fingerprint()
        || aborted.plan() != active.plan()
        || aborted.coordinator_id() != active.coordinator_id()
        || aborted.sequence_authority() != active.sequence_authority()
        || aborted.run_id() != active.run_id()
        || aborted.request_id() != active.request_id()
        || aborted.activation_epoch() != active.activation_epoch()
        || aborted.runtime_implementation_fingerprint()
            != active.runtime_implementation_fingerprint()
        || ids.aborted_sequence_fingerprint.as_deref() != Some(aborted.fingerprint())
    {
        return Err(invalid_event(
            "event abort identity differs from the poisoned active sequence receipt",
        ));
    }
    Ok(())
}

fn validate_event_against_context(
    event: &ExecutionEvent,
    context: &TrustedExecutionEventContext<'_>,
) -> Result<(), VNextError> {
    let ids = event.identity.parts();
    if &ids.run_id != context.run_id || &ids.request_id != context.request_id {
        return Err(invalid_event(
            "event identity differs from trusted run/request context",
        ));
    }
    if let Some(topology) = context.topology {
        if ids.plan_id.as_ref() != Some(topology.plan_id())
            || ids.plan_hash.as_ref() != Some(topology.plan_hash())
            || ids.device_id.as_ref() != Some(topology.device_id())
            || ids.runtime_implementation_fingerprint.as_deref()
                != Some(topology.device_runtime_implementation_fingerprint())
        {
            return Err(invalid_event(
                "event plan identity differs from trusted topology",
            ));
        }
        if let Some(node_id) = &ids.node_id {
            let node = topology
                .nodes
                .get(node_id)
                .ok_or_else(|| invalid_event("event node is absent from trusted topology"))?;
            if ids.operation_id.as_ref() != Some(&node.operation_id)
                || ids.provider_id.as_ref() != Some(&node.provider_id)
            {
                return Err(invalid_event(
                    "event operation/provider differs from trusted node topology",
                ));
            }
        }
    } else if ids.plan_id.is_some() {
        return Err(invalid_event(
            "plan-bound event lacks trusted topology context",
        ));
    }
    match (has_active(ids), context.active) {
        (true, Some(active)) => {
            validate_active_identity(ids, active)?;
            let topology = context
                .topology
                .ok_or_else(|| invalid_event("active event lacks trusted execution topology"))?;
            if active.runtime_implementation_fingerprint()
                != topology.device_runtime_implementation_fingerprint()
                || active.plan().runtime_implementation_fingerprint()
                    != topology.device_runtime_implementation_fingerprint()
            {
                return Err(invalid_event(
                    "plan, admission, pool, and active runtime implementations differ",
                ));
            }
        }
        (false, None) => {}
        _ => {
            return Err(invalid_event(
                "event active identity presence differs from external active evidence",
            ));
        }
    }
    match (has_completed(ids), context.completed, context.active) {
        (true, Some(completed), Some(active)) => {
            validate_completed_identity(ids, completed, active)?;
        }
        (false, None, _) => {}
        _ => {
            return Err(invalid_event(
                "event completion identity presence differs from external synchronized receipt",
            ));
        }
    }
    match (has_aborted(ids), context.aborted, context.active) {
        (true, Some(aborted), Some(active)) => {
            validate_aborted_identity(ids, aborted, active)?;
        }
        (false, None, _) => {}
        _ => {
            return Err(invalid_event(
                "event abort identity presence differs from external poison receipt",
            ));
        }
    }
    match (event.kind, context.submitted_operation) {
        (ExecutionEventKind::OperationSubmitted, Some(submission))
            if submission
                .participants()
                .iter()
                .any(|participant| participant.identity() == event.identity()) => {}
        (ExecutionEventKind::OperationSubmitted, _) => {
            return Err(invalid_event(
                "OperationSubmitted lacks its exact external dispatch receipt",
            ));
        }
        (_, None) => {}
        (_, Some(_)) => {
            return Err(invalid_event(
                "operation submission receipt supplied for a different event kind",
            ));
        }
    }
    match (event.kind, context.retired_operation) {
        (ExecutionEventKind::NodeRetired, Some(completion))
            if same_operation_authority_except_observation(
                event.identity().parts(),
                completion.submission().identity().parts(),
            ) => {}
        (ExecutionEventKind::NodeRetired, _) => {
            return Err(invalid_event(
                "NodeRetired lacks its exact participant completion projection",
            ));
        }
        (_, None) => {}
        (_, Some(_)) => {
            return Err(invalid_event(
                "operation completion projection supplied for a different event kind",
            ));
        }
    }
    match (&event.detail, context.expected_failure) {
        (ExecutionEventDetail::Failure(failure), Some(expected)) if failure == expected => {}
        (
            ExecutionEventDetail::FailureTerminal {
                first_failure_fingerprint,
            },
            Some(expected),
        ) if first_failure_fingerprint == &expected.fingerprint() => {}
        (ExecutionEventDetail::Failure(_) | ExecutionEventDetail::FailureTerminal { .. }, _) => {
            return Err(invalid_event(
                "event failure differs from independent first failure evidence",
            ));
        }
        (_, None) => {}
        (_, Some(_)) => {
            return Err(invalid_event(
                "trusted failure evidence supplied for a non-failure event",
            ));
        }
    }
    if let Some(recovery_identity) = context.unsubmitted_recovery_identity {
        let ExecutionEventDetail::Failure(failure) = &event.detail else {
            return Err(invalid_event(
                "unsubmitted recovery identity was supplied for a non-failure event",
            ));
        };
        if failure.identity() != recovery_identity {
            return Err(invalid_event(
                "unsubmitted recovery identity differs from the exact observed failure",
            ));
        }
    }
    Ok(())
}

fn validate_failure_identity(
    domain: FailureDomain,
    ids: &ExecutionIdentityParts,
) -> Result<(), VNextError> {
    let operation_shape = exact_plan(ids)
        && ids.frame_id.is_some()
        && ids.node_invocation_id.is_some()
        && ids.node_id.is_some()
        && ids.operation_id.is_some()
        && ids.provider_id.is_some()
        && has_active(ids)
        && !has_completed(ids)
        && !has_aborted(ids)
        && no_resource_item(ids);
    let resource_shape = exact_plan(ids)
        && ids.frame_id.is_none()
        && ids.node_invocation_id.is_none()
        && ids.node_id.is_none()
        && ids.operation_id.is_none()
        && ids.provider_id.is_none()
        && has_pool(ids)
        && !has_active(ids)
        && !has_completed(ids)
        && !has_aborted(ids)
        && (ids.resource_id.is_some() ^ ids.resource_batch_fingerprint.is_some());
    let plan_shape = exact_plan(ids)
        && ids.frame_id.is_none()
        && ids.node_id.is_none()
        && !has_pool(ids)
        && !has_completed(ids)
        && !has_aborted(ids)
        && no_resource_item(ids);
    let device_only = ids.device_id.is_some()
        && ids.plan_id.is_none()
        && ids.frame_id.is_none()
        && ids.node_id.is_none()
        && !has_pool(ids)
        && !has_completed(ids)
        && !has_aborted(ids)
        && no_resource_item(ids);
    let pre_plan = ids.plan_id.is_none()
        && ids.device_id.is_none()
        && ids.frame_id.is_none()
        && ids.node_id.is_none()
        && !has_pool(ids)
        && !has_completed(ids)
        && !has_aborted(ids)
        && no_resource_item(ids);
    let valid = match domain {
        FailureDomain::Operation => operation_shape,
        FailureDomain::Resource => resource_shape,
        FailureDomain::Device => device_only || plan_shape || operation_shape || resource_shape,
        FailureDomain::Planning => plan_shape,
        FailureDomain::ModelResolution | FailureDomain::Product => pre_plan || plan_shape,
        FailureDomain::Event => pre_plan || plan_shape || operation_shape,
    };
    if !valid {
        return Err(invalid_event(format!(
            "failure domain `{domain:?}` has missing or extraneous execution identity fields"
        )));
    }
    Ok(())
}

#[derive(Debug, Clone)]
struct ActiveNodeInvocation {
    invocation_id: NodeInvocationId,
    node_span: SpanId,
    operation_submitted: bool,
}

#[derive(Debug, Clone)]
struct ActiveFrame {
    id: ExecutionFrameId,
    span_id: SpanId,
    active_nodes: BTreeMap<NodeId, ActiveNodeInvocation>,
    completed_nodes: BTreeSet<NodeId>,
}

#[derive(Debug, Clone)]
pub struct ExecutionEventCursor {
    run_id: RunId,
    request_id: RequestIdentity,
    last_sequence: u64,
    last_timestamp: Option<MonotonicTimestamp>,
    last_phase: Option<ExecutionPhase>,
    topology_fingerprint: Option<String>,
    active_fingerprint: Option<String>,
    completion_fingerprint: Option<String>,
    abort_fingerprint: Option<String>,
    observed_failure: Option<IdentifiedFailure>,
    accepted: bool,
    planned: bool,
    terminal: bool,
    root_span: Option<SpanId>,
    seen_spans: BTreeSet<SpanId>,
    next_frame: u64,
    next_invocation: u64,
    completed_frames: u64,
    frame: Option<ActiveFrame>,
}

impl ExecutionEventCursor {
    pub fn new(run_id: RunId, request_id: RequestIdentity) -> Self {
        Self {
            run_id,
            request_id,
            last_sequence: 0,
            last_timestamp: None,
            last_phase: None,
            topology_fingerprint: None,
            active_fingerprint: None,
            completion_fingerprint: None,
            abort_fingerprint: None,
            observed_failure: None,
            accepted: false,
            planned: false,
            terminal: false,
            root_span: None,
            seen_spans: BTreeSet::new(),
            next_frame: 1,
            next_invocation: 1,
            completed_frames: 0,
            frame: None,
        }
    }

    pub fn observe_against(
        &mut self,
        event: &ExecutionEvent,
        context: &TrustedExecutionEventContext<'_>,
    ) -> Result<(), VNextError> {
        let mut next = self.clone();
        next.observe_inner(event, context)?;
        *self = next;
        Ok(())
    }

    pub const fn last_sequence(&self) -> u64 {
        self.last_sequence
    }

    pub const fn is_terminal(&self) -> bool {
        self.terminal
    }

    pub const fn completed_frames(&self) -> u64 {
        self.completed_frames
    }

    fn observe_inner(
        &mut self,
        event: &ExecutionEvent,
        context: &TrustedExecutionEventContext<'_>,
    ) -> Result<(), VNextError> {
        validate_event_against_context(event, context)?;
        let ids = event.identity.parts();
        if ids.run_id != self.run_id
            || ids.request_id != self.request_id
            || ids.sequence != self.last_sequence.saturating_add(1)
            || self
                .last_timestamp
                .is_some_and(|timestamp| event.timestamp <= timestamp)
            || self
                .last_phase
                .is_some_and(|phase| event.phase.rank() < phase.rank())
            || self.terminal
        {
            return Err(invalid_event(
                "request journal run, request, sequence, timestamp, phase, or terminal boundary is invalid",
            ));
        }
        if let Some(topology) = context.topology {
            if let Some(bound) = &self.topology_fingerprint {
                if bound != topology.fingerprint() {
                    return Err(invalid_event("request changed trusted topology"));
                }
            }
        }
        if let Some(active) = context.active {
            if let Some(bound) = &self.active_fingerprint {
                if bound != active.fingerprint() {
                    return Err(invalid_event(
                        "request changed active pool/slot/epoch/runtime binding",
                    ));
                }
            }
        }
        if self.observed_failure.is_some()
            && !matches!(
                event.kind,
                ExecutionEventKind::SequenceCompleted
                    | ExecutionEventKind::SequenceAborted
                    | ExecutionEventKind::RequestFailed
            )
        {
            return Err(invalid_event(
                "only sequence disposition and terminal failure may follow FailureObserved",
            ));
        }

        match event.kind {
            ExecutionEventKind::RequestAccepted => self.accept(ids)?,
            ExecutionEventKind::PlanBuilt => {
                let topology = context
                    .topology
                    .ok_or_else(|| invalid_event("PlanBuilt lacks trusted topology"))?;
                self.plan(ids, topology)?;
            }
            ExecutionEventKind::FrameStarted => {
                let topology = self.require_topology(context)?;
                let active = self.require_active(context)?;
                self.start_frame(ids, topology, active)?;
            }
            ExecutionEventKind::NodeStarted => {
                let topology = self.require_topology(context)?;
                self.require_active(context)?;
                self.start_node(ids, topology)?;
            }
            ExecutionEventKind::OperationSubmitted => self.submit_operation(ids)?,
            ExecutionEventKind::NodeRetired => self.retire_node(ids)?,
            ExecutionEventKind::FrameCompleted => {
                let topology = self.require_topology(context)?;
                self.require_active(context)?;
                self.complete_frame(ids, topology)?;
            }
            ExecutionEventKind::FailureObserved => {
                self.observe_failure(event, context.unsubmitted_recovery_identity)?
            }
            ExecutionEventKind::SequenceCompleted => {
                self.require_topology(context)?;
                self.require_active(context)?;
                let completed = context.completed.ok_or_else(|| {
                    invalid_event("SequenceCompleted lacks synchronized completion evidence")
                })?;
                self.complete_sequence(ids, completed)?;
            }
            ExecutionEventKind::SequenceAborted => {
                self.require_topology(context)?;
                self.require_active(context)?;
                let aborted = context.aborted.ok_or_else(|| {
                    invalid_event("SequenceAborted lacks poisoned abort evidence")
                })?;
                self.abort_sequence(ids, aborted)?;
            }
            ExecutionEventKind::RequestCompleted => {
                self.require_topology(context)?;
                self.require_active(context)?;
                if context.completed.is_none() {
                    return Err(invalid_event(
                        "RequestCompleted lacks synchronized completion evidence",
                    ));
                }
                self.complete_success(ids)?;
            }
            ExecutionEventKind::RequestFailed => self.fail_request(event)?,
        }
        self.last_sequence = ids.sequence;
        self.last_timestamp = Some(event.timestamp);
        self.last_phase = Some(event.phase);
        Ok(())
    }

    fn require_topology<'a>(
        &self,
        context: &'a TrustedExecutionEventContext<'_>,
    ) -> Result<&'a TrustedExecutionTopology, VNextError> {
        if !self.planned {
            return Err(invalid_event("execution event precedes PlanBuilt"));
        }
        context
            .topology
            .ok_or_else(|| invalid_event("execution event lacks trusted topology"))
    }

    fn require_active<'a>(
        &self,
        context: &'a TrustedExecutionEventContext<'_>,
    ) -> Result<&'a TrustedActiveSequenceBinding, VNextError> {
        context
            .active
            .ok_or_else(|| invalid_event("execution event lacks active sequence evidence"))
    }

    fn accept(&mut self, ids: &ExecutionIdentityParts) -> Result<(), VNextError> {
        if self.accepted || self.last_sequence != 0 || ids.parent_span_id.is_some() {
            return Err(invalid_event(
                "RequestAccepted must open the first root span",
            ));
        }
        self.seen_spans.insert(ids.span_id.clone());
        self.root_span = Some(ids.span_id.clone());
        self.accepted = true;
        Ok(())
    }

    fn plan(
        &mut self,
        ids: &ExecutionIdentityParts,
        topology: &TrustedExecutionTopology,
    ) -> Result<(), VNextError> {
        if !self.accepted
            || self.planned
            || ids.parent_span_id.as_ref() != self.root_span.as_ref()
            || !self.seen_spans.insert(ids.span_id.clone())
        {
            return Err(invalid_event(
                "PlanBuilt must uniquely bind one topology under the request root",
            ));
        }
        self.topology_fingerprint = Some(topology.fingerprint().to_owned());
        self.planned = true;
        Ok(())
    }

    fn start_frame(
        &mut self,
        ids: &ExecutionIdentityParts,
        _topology: &TrustedExecutionTopology,
        active: &TrustedActiveSequenceBinding,
    ) -> Result<(), VNextError> {
        let frame_id = ids.frame_id.expect("frame shape validated");
        if self.frame.is_some()
            || self.completion_fingerprint.is_some()
            || self.abort_fingerprint.is_some()
            || self.observed_failure.is_some()
            || frame_id.get() != self.next_frame
            || ids.parent_span_id.as_ref() != self.root_span.as_ref()
            || !self.seen_spans.insert(ids.span_id.clone())
        {
            return Err(invalid_event(
                "frames must start once in strict contiguous order under the request root",
            ));
        }
        self.active_fingerprint
            .get_or_insert_with(|| active.fingerprint().to_owned());
        self.frame = Some(ActiveFrame {
            id: frame_id,
            span_id: ids.span_id.clone(),
            active_nodes: BTreeMap::new(),
            completed_nodes: BTreeSet::new(),
        });
        Ok(())
    }

    fn start_node(
        &mut self,
        ids: &ExecutionIdentityParts,
        topology: &TrustedExecutionTopology,
    ) -> Result<(), VNextError> {
        let node_id = ids.node_id.as_ref().expect("node shape validated");
        let invocation_id = ids
            .node_invocation_id
            .expect("node invocation shape validated");
        let frame = self
            .frame
            .as_mut()
            .ok_or_else(|| invalid_event("node started outside an active frame"))?;
        let node = topology
            .nodes
            .get(node_id)
            .ok_or_else(|| invalid_event("node is absent from trusted topology"))?;
        if ids.frame_id != Some(frame.id)
            || invocation_id.get() != self.next_invocation
            || frame.active_nodes.contains_key(node_id)
            || frame.completed_nodes.contains(node_id)
            || node
                .dependencies
                .iter()
                .any(|dependency| !frame.completed_nodes.contains(dependency))
            || ids.parent_span_id.as_ref() != Some(&frame.span_id)
            || !self.seen_spans.insert(ids.span_id.clone())
        {
            return Err(invalid_event(
                "node invocation is duplicate, non-monotonic, cross-frame, or precedes same-frame dependencies",
            ));
        }
        self.next_invocation = self
            .next_invocation
            .checked_add(1)
            .ok_or_else(|| invalid_event("node invocation id overflow"))?;
        frame.active_nodes.insert(
            node_id.clone(),
            ActiveNodeInvocation {
                invocation_id,
                node_span: ids.span_id.clone(),
                operation_submitted: false,
            },
        );
        Ok(())
    }

    fn submit_operation(&mut self, ids: &ExecutionIdentityParts) -> Result<(), VNextError> {
        let node_id = ids.node_id.as_ref().expect("operation shape validated");
        let frame = self
            .frame
            .as_mut()
            .ok_or_else(|| invalid_event("operation submitted outside an active frame"))?;
        let active = frame
            .active_nodes
            .get_mut(node_id)
            .ok_or_else(|| invalid_event("operation submitted without active node"))?;
        if ids.frame_id != Some(frame.id)
            || ids.node_invocation_id != Some(active.invocation_id)
            || active.operation_submitted
            || ids.parent_span_id.as_ref() != Some(&active.node_span)
            || !self.seen_spans.insert(ids.span_id.clone())
        {
            return Err(invalid_event(
                "operation submission does not match the active node invocation",
            ));
        }
        active.operation_submitted = true;
        Ok(())
    }

    fn retire_node(&mut self, ids: &ExecutionIdentityParts) -> Result<(), VNextError> {
        let node_id = ids.node_id.as_ref().expect("node shape validated");
        let frame = self
            .frame
            .as_mut()
            .ok_or_else(|| invalid_event("node completed outside an active frame"))?;
        let active = frame
            .active_nodes
            .get(node_id)
            .ok_or_else(|| invalid_event("node completed without active invocation"))?;
        if ids.frame_id != Some(frame.id)
            || ids.node_invocation_id != Some(active.invocation_id)
            || ids.span_id != active.node_span
            || ids.parent_span_id.as_ref() != Some(&frame.span_id)
            || !active.operation_submitted
        {
            return Err(invalid_event(
                "node completion requires its exact frame, invocation, span, and operation",
            ));
        }
        frame.active_nodes.remove(node_id);
        frame.completed_nodes.insert(node_id.clone());
        Ok(())
    }

    fn complete_sequence(
        &mut self,
        ids: &ExecutionIdentityParts,
        completed: &TrustedCompletedSequenceBinding,
    ) -> Result<(), VNextError> {
        let failure_cleanup = self.observed_failure.is_some();
        if !self.planned
            || !failure_cleanup && (self.completed_frames == 0 || self.frame.is_some())
            || failure_cleanup && self.active_fingerprint.is_none()
            || self.active_fingerprint.as_deref() != Some(completed.active_sequence_fingerprint())
            || self.completion_fingerprint.is_some()
            || self.abort_fingerprint.is_some()
            || ids.parent_span_id.as_ref() != self.root_span.as_ref()
            || !self.seen_spans.insert(ids.span_id.clone())
        {
            return Err(invalid_event(
                "SequenceCompleted requires submitted frames and one unique synchronized receipt",
            ));
        }
        if failure_cleanup {
            self.frame = None;
        }
        self.completion_fingerprint = Some(completed.fingerprint().to_owned());
        Ok(())
    }

    fn abort_sequence(
        &mut self,
        ids: &ExecutionIdentityParts,
        aborted: &TrustedAbortedSequenceBinding,
    ) -> Result<(), VNextError> {
        if self.observed_failure.is_none()
            || self.active_fingerprint.is_none()
            || self.active_fingerprint.as_deref() != Some(aborted.active_sequence_fingerprint())
            || self.completion_fingerprint.is_some()
            || self.abort_fingerprint.is_some()
            || ids.parent_span_id.as_ref() != self.root_span.as_ref()
            || !self.seen_spans.insert(ids.span_id.clone())
        {
            return Err(invalid_event(
                "SequenceAborted requires one observed failure and one unique poison receipt",
            ));
        }
        self.frame = None;
        self.abort_fingerprint = Some(aborted.fingerprint().to_owned());
        Ok(())
    }

    fn complete_frame(
        &mut self,
        ids: &ExecutionIdentityParts,
        topology: &TrustedExecutionTopology,
    ) -> Result<(), VNextError> {
        let frame = self
            .frame
            .as_ref()
            .ok_or_else(|| invalid_event("FrameCompleted lacks an active frame"))?;
        if ids.frame_id != Some(frame.id)
            || ids.span_id != frame.span_id
            || ids.parent_span_id.as_ref() != self.root_span.as_ref()
            || !frame.active_nodes.is_empty()
            || frame.completed_nodes != topology.node_ids()
        {
            return Err(invalid_event(
                "frame completion requires every trusted node exactly once and no active invocation",
            ));
        }
        self.frame = None;
        self.completed_frames += 1;
        self.next_frame = self
            .next_frame
            .checked_add(1)
            .ok_or_else(|| invalid_event("frame id overflow"))?;
        Ok(())
    }

    fn observe_failure(
        &mut self,
        event: &ExecutionEvent,
        unsubmitted_recovery_identity: Option<&ExecutionIdentityEnvelope>,
    ) -> Result<(), VNextError> {
        if !self.accepted || self.observed_failure.is_some() {
            return Err(invalid_event(
                "FailureObserved requires one accepted non-failed request",
            ));
        }
        let failure = match &event.detail {
            ExecutionEventDetail::Failure(failure) => failure,
            _ => return Err(invalid_event("FailureObserved lacks identified failure")),
        };
        let ids = event.identity.parts();
        if has_active(ids) {
            let failed_operation = failure.identity().parts();
            self.active_fingerprint
                .get_or_insert_with(|| ids.active_sequence_fingerprint.clone().unwrap());
            let frame = self.frame.as_ref().ok_or_else(|| {
                invalid_event("active operation failure lacks its execution frame")
            })?;
            let node_id = ids
                .node_id
                .as_ref()
                .ok_or_else(|| invalid_event("active operation failure lacks its node identity"))?;
            let invocation = frame.active_nodes.get(node_id).ok_or_else(|| {
                invalid_event("active operation failure lacks its node invocation")
            })?;
            let operation_span_was_submitted = self.seen_spans.contains(&failed_operation.span_id);
            let is_unsubmitted_recovery = unsubmitted_recovery_identity
                .is_some_and(|identity| identity == failure.identity());
            if ids.frame_id != Some(frame.id)
                || ids.node_invocation_id != Some(invocation.invocation_id)
                || !same_operation_authority_except_observation(ids, failed_operation)
                || failed_operation.parent_span_id.as_ref() != Some(&invocation.node_span)
                || ids.parent_span_id.as_ref() != Some(&failed_operation.span_id)
                || operation_span_was_submitted == is_unsubmitted_recovery
            {
                return Err(invalid_event(
                    "operation failure does not link one exact submitted operation to its observation span",
                ));
            }
        } else if ids.parent_span_id.as_ref() != self.root_span.as_ref()
            && !(ids.span_id == *self.root_span.as_ref().expect("accepted root")
                && ids.parent_span_id.is_none())
        {
            return Err(invalid_event(
                "non-active failure must be anchored under the request root",
            ));
        }
        if !self.seen_spans.insert(ids.span_id.clone()) {
            return Err(invalid_event("FailureObserved span was already used"));
        }
        self.observed_failure = Some(failure.clone());
        Ok(())
    }

    fn complete_success(&mut self, ids: &ExecutionIdentityParts) -> Result<(), VNextError> {
        if self.observed_failure.is_some()
            || !self.planned
            || self.completed_frames == 0
            || self.frame.is_some()
            || self.abort_fingerprint.is_some()
            || self.completion_fingerprint.as_deref()
                != ids.completed_sequence_fingerprint.as_deref()
        {
            return Err(invalid_event(
                "successful request requires submitted frames and the exact synchronized sequence receipt",
            ));
        }
        if ids.span_id != *self.root_span.as_ref().expect("accepted root")
            || ids.parent_span_id.is_some()
        {
            return Err(invalid_event(
                "terminal request event must close the exact request root",
            ));
        }
        self.terminal = true;
        Ok(())
    }

    fn fail_request(&mut self, event: &ExecutionEvent) -> Result<(), VNextError> {
        let ids = event.identity.parts();
        if !self.accepted {
            if self.last_sequence != 0
                || ids.parent_span_id.is_some()
                || !matches!(event.detail, ExecutionEventDetail::Failure(_))
            {
                return Err(invalid_event(
                    "only first-event pre-plan RequestFailed may precede acceptance",
                ));
            }
            self.terminal = true;
            return Ok(());
        }
        let observed = self
            .observed_failure
            .as_ref()
            .ok_or_else(|| invalid_event("RequestFailed lacks FailureObserved"))?;
        let terminal_fingerprint = match &event.detail {
            ExecutionEventDetail::FailureTerminal {
                first_failure_fingerprint,
            } => first_failure_fingerprint,
            _ => {
                return Err(invalid_event(
                    "post-acceptance RequestFailed requires FailureTerminal",
                ));
            }
        };
        if terminal_fingerprint != &observed.fingerprint() {
            return Err(invalid_event(
                "RequestFailed does not reference the first observed failure",
            ));
        }
        if self.active_fingerprint.is_some() {
            let completed_matches = self.completion_fingerprint.is_some()
                && self.completion_fingerprint.as_deref()
                    == ids.completed_sequence_fingerprint.as_deref();
            let aborted_matches = self.abort_fingerprint.is_some()
                && self.abort_fingerprint.as_deref() == ids.aborted_sequence_fingerprint.as_deref();
            if completed_matches == aborted_matches {
                return Err(invalid_event(
                    "active RequestFailed requires exactly one matching completion or abort disposition",
                ));
            }
            if completed_matches && ids.aborted_sequence_fingerprint.is_some()
                || aborted_matches && ids.completed_sequence_fingerprint.is_some()
            {
                return Err(invalid_event(
                    "active RequestFailed carries an unexpected opposite sequence disposition",
                ));
            }
        } else if self.completion_fingerprint.is_some()
            || self.abort_fingerprint.is_some()
            || has_completed(ids)
            || has_aborted(ids)
        {
            return Err(invalid_event(
                "non-active RequestFailed cannot carry sequence disposition",
            ));
        }
        if ids.span_id != *self.root_span.as_ref().expect("accepted root")
            || ids.parent_span_id.is_some()
        {
            return Err(invalid_event(
                "terminal request event must close the exact request root",
            ));
        }
        self.terminal = true;
        Ok(())
    }
}
