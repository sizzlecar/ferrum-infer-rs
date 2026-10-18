use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

use super::{
    canonical_fingerprint, has_aborted, has_active, has_completed, invalid_event,
    same_operation_authority_except_observation, sha256_bytes, validate_sha256,
    BatchOperationIdentity, CompletionDrainReceipt, CompletionQuarantineReceipt, CompletionSlotId,
    ContractVersion, ExecutionEvent, ExecutionEventCursor, ExecutionEventDetail,
    ExecutionEventKind, ExecutionIdentityEnvelope, IdentifiedFailure, OperationCompletionReceipt,
    OperationParticipantCompletionDisposition, OperationParticipantCompletionReceipt,
    PlanRuntimeCloseReceipt, PlanRuntimeQuarantineReceipt, ResolvedModelPlan, ResourcePoolEvent,
    ResourcePoolEventCursor, ResourcePoolEvidence, ResourcePoolId,
    SubmittedOperationParticipantReceipt, SubmittedOperationReceipt, TrustedAbortedSequenceBinding,
    TrustedActiveSequenceBinding, TrustedCompletedSequenceBinding, TrustedExecutionEventContext,
    TrustedExecutionTopology, UnvalidatedExecutionIdentityParts, VNextError,
    EXECUTION_IDENTITY_VERSION, MAX_REPLAY_IDENTITY_WIRE_BYTES,
};

/// Independent evidence needed to rebuild a replay identity. None of these
/// values are accepted from the serialized replay envelope itself.
pub struct ReplayEvidence<'a> {
    resolved_plan: &'a ResolvedModelPlan,
    request_input: &'a [u8],
    initial_state: &'a [u8],
    random_seed: u64,
    request_journal: &'a [ExecutionEvent],
    active_binding: &'a TrustedActiveSequenceBinding,
    completed_binding: Option<&'a TrustedCompletedSequenceBinding>,
    aborted_binding: Option<&'a TrustedAbortedSequenceBinding>,
    cleanup_requirement: ReplayCleanupRequirement,
    plan_cleanup: ReplayPlanCleanupEvidence<'a>,
    operation_completions: &'a [OperationCompletionReceipt],
    operation_drains: &'a [CompletionDrainReceipt],
    // Quarantine retains invocation ownership and is never terminal replay evidence.
    operation_quarantines: &'a [CompletionQuarantineReceipt],
    pool_evidence: Option<&'a ResourcePoolEvidence>,
    pool_journal: &'a [ResourcePoolEvent],
}

impl<'a> ReplayEvidence<'a> {
    pub fn new(
        resolved_plan: &'a ResolvedModelPlan,
        request_input: &'a [u8],
        initial_state: &'a [u8],
        random_seed: u64,
        request_journal: &'a [ExecutionEvent],
        active_binding: &'a TrustedActiveSequenceBinding,
        completed_binding: Option<&'a TrustedCompletedSequenceBinding>,
        aborted_binding: Option<&'a TrustedAbortedSequenceBinding>,
        cleanup_requirement: ReplayCleanupRequirement,
        plan_cleanup: ReplayPlanCleanupEvidence<'a>,
        operation_completions: &'a [OperationCompletionReceipt],
        operation_drains: &'a [CompletionDrainReceipt],
        operation_quarantines: &'a [CompletionQuarantineReceipt],
        pool_evidence: &'a ResourcePoolEvidence,
        pool_journal: &'a [ResourcePoolEvent],
    ) -> Self {
        Self {
            resolved_plan,
            request_input,
            initial_state,
            random_seed,
            request_journal,
            active_binding,
            completed_binding,
            aborted_binding,
            cleanup_requirement,
            plan_cleanup,
            operation_completions,
            operation_drains,
            operation_quarantines,
            pool_evidence: Some(pool_evidence),
            pool_journal,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_no_static(
        resolved_plan: &'a ResolvedModelPlan,
        request_input: &'a [u8],
        initial_state: &'a [u8],
        random_seed: u64,
        request_journal: &'a [ExecutionEvent],
        active_binding: &'a TrustedActiveSequenceBinding,
        completed_binding: Option<&'a TrustedCompletedSequenceBinding>,
        aborted_binding: Option<&'a TrustedAbortedSequenceBinding>,
        cleanup_requirement: ReplayCleanupRequirement,
        plan_cleanup: ReplayPlanCleanupEvidence<'a>,
        operation_completions: &'a [OperationCompletionReceipt],
        operation_drains: &'a [CompletionDrainReceipt],
        operation_quarantines: &'a [CompletionQuarantineReceipt],
    ) -> Self {
        Self {
            resolved_plan,
            request_input,
            initial_state,
            random_seed,
            request_journal,
            active_binding,
            completed_binding,
            aborted_binding,
            cleanup_requirement,
            plan_cleanup,
            operation_completions,
            operation_drains,
            operation_quarantines,
            pool_evidence: None,
            pool_journal: &[],
        }
    }

    pub fn resolved_plan(&self) -> &ResolvedModelPlan {
        self.resolved_plan
    }

    pub fn request_input(&self) -> &[u8] {
        self.request_input
    }

    pub fn initial_state(&self) -> &[u8] {
        self.initial_state
    }

    pub const fn random_seed(&self) -> u64 {
        self.random_seed
    }

    pub fn request_journal(&self) -> &[ExecutionEvent] {
        self.request_journal
    }

    pub fn active_binding(&self) -> &TrustedActiveSequenceBinding {
        self.active_binding
    }

    pub fn completed_binding(&self) -> Option<&TrustedCompletedSequenceBinding> {
        self.completed_binding
    }

    pub fn aborted_binding(&self) -> Option<&TrustedAbortedSequenceBinding> {
        self.aborted_binding
    }

    pub const fn cleanup_requirement(&self) -> ReplayCleanupRequirement {
        self.cleanup_requirement
    }

    pub const fn plan_cleanup(&self) -> ReplayPlanCleanupEvidence<'a> {
        self.plan_cleanup
    }

    pub fn operation_completions(&self) -> &[OperationCompletionReceipt] {
        self.operation_completions
    }

    pub fn operation_drains(&self) -> &[CompletionDrainReceipt] {
        self.operation_drains
    }

    pub fn operation_quarantines(&self) -> &[CompletionQuarantineReceipt] {
        self.operation_quarantines
    }

    pub fn pool_evidence(&self) -> Option<&ResourcePoolEvidence> {
        self.pool_evidence
    }

    pub fn pool_journal(&self) -> &[ResourcePoolEvent] {
        self.pool_journal
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ReplayOperationTerminalKey {
    Completion(usize),
    Drain(usize),
    Quarantine(usize),
}

#[derive(Clone, Copy)]
enum ReplayOperationTerminalRef<'a> {
    Completion(&'a OperationCompletionReceipt),
    Drain(&'a CompletionDrainReceipt),
    Quarantine(&'a CompletionQuarantineReceipt),
}

impl<'a> ReplayOperationTerminalRef<'a> {
    fn slot_id(self) -> CompletionSlotId {
        match self {
            Self::Completion(receipt) => receipt.submission().slot_id(),
            Self::Drain(receipt) => receipt.slot_id(),
            Self::Quarantine(receipt) => receipt.slot_id(),
        }
    }

    fn batch_identity(self) -> &'a BatchOperationIdentity {
        match self {
            Self::Completion(receipt) => receipt.submission().batch_identity(),
            Self::Drain(receipt) => receipt.batch_identity(),
            Self::Quarantine(receipt) => receipt.batch_identity(),
        }
    }

    fn participant_submission(
        self,
        identity: &ExecutionIdentityEnvelope,
    ) -> Option<&'a SubmittedOperationParticipantReceipt> {
        self.submission()?
            .participants()
            .iter()
            .find(|participant| participant.identity() == identity)
    }

    fn submission(self) -> Option<&'a SubmittedOperationReceipt> {
        match self {
            Self::Completion(receipt) => Some(receipt.submission()),
            Self::Drain(receipt) => receipt.submission(),
            Self::Quarantine(receipt) => receipt.submission(),
        }
    }

    fn participant_completion(
        self,
        identity: &ExecutionIdentityEnvelope,
    ) -> Option<&'a OperationParticipantCompletionReceipt> {
        match self {
            Self::Completion(receipt) => receipt.participants().iter().find(|participant| {
                same_operation_authority_except_observation(
                    identity.parts(),
                    participant.submission().identity().parts(),
                )
            }),
            Self::Drain(_) | Self::Quarantine(_) => None,
        }
    }

    fn contains_identity(self, identity: &ExecutionIdentityEnvelope) -> bool {
        self.batch_identity()
            .participants()
            .iter()
            .any(|participant| participant.identity() == identity)
    }

    fn had_submission_fence(self) -> bool {
        match self {
            Self::Completion(_) => true,
            Self::Drain(receipt) => receipt.had_submission_fence(),
            Self::Quarantine(receipt) => receipt.had_submission_fence(),
        }
    }

    fn exact_failed_completion(
        self,
        identity: &ExecutionIdentityEnvelope,
    ) -> Option<&'a IdentifiedFailure> {
        self.participant_completion(identity)
            .and_then(|participant| match participant.disposition() {
                OperationParticipantCompletionDisposition::FailedButQuiescent(failure) => {
                    Some(failure)
                }
                OperationParticipantCompletionDisposition::Succeeded
                | OperationParticipantCompletionDisposition::ContractFailedButQuiescent(_) => None,
            })
    }

    fn participant_is_success(self, identity: &ExecutionIdentityEnvelope) -> bool {
        self.participant_completion(identity)
            .is_some_and(|participant| {
                matches!(
                    participant.disposition(),
                    OperationParticipantCompletionDisposition::Succeeded
                )
            })
    }
}

#[derive(Serialize)]
struct ReplayOperationTerminalFingerprint<'a> {
    completions: &'a [OperationCompletionReceipt],
    drains: &'a [CompletionDrainReceipt],
    quarantines: &'a [CompletionQuarantineReceipt],
}

fn replay_operation_terminals<'a>(
    evidence: &'a ReplayEvidence<'_>,
) -> Vec<(ReplayOperationTerminalKey, ReplayOperationTerminalRef<'a>)> {
    evidence
        .operation_completions
        .iter()
        .enumerate()
        .map(|(index, receipt)| {
            (
                ReplayOperationTerminalKey::Completion(index),
                ReplayOperationTerminalRef::Completion(receipt),
            )
        })
        .chain(
            evidence
                .operation_drains
                .iter()
                .enumerate()
                .map(|(index, receipt)| {
                    (
                        ReplayOperationTerminalKey::Drain(index),
                        ReplayOperationTerminalRef::Drain(receipt),
                    )
                }),
        )
        .chain(
            evidence
                .operation_quarantines
                .iter()
                .enumerate()
                .map(|(index, receipt)| {
                    (
                        ReplayOperationTerminalKey::Quarantine(index),
                        ReplayOperationTerminalRef::Quarantine(receipt),
                    )
                }),
        )
        .collect()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReplayCleanupRequirement {
    RequireClean,
    AllowPending,
}

/// Independent root-cleanup evidence supplied while rebuilding replay. Pending
/// is explicit and is accepted only when the caller allows pending cleanup.
/// The receipt variants are core-signed outputs and cannot be deserialized or
/// constructed by the replay caller.
#[derive(Clone, Copy)]
pub enum ReplayPlanCleanupEvidence<'a> {
    Pending,
    Closed(&'a PlanRuntimeCloseReceipt),
    Quarantined(&'a PlanRuntimeQuarantineReceipt),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayCleanupStatus {
    Completed,
    SequenceQuiescent,
    Quarantined,
    CleanupPending,
}

/// A replay identity is trusted output. Deserialization always goes through
/// `UnvalidatedReplayIdentity` and reconstruction from independent evidence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReplayIdentity {
    identity_version: ContractVersion,
    terminal_identity: ExecutionIdentityEnvelope,
    resolved_plan_fingerprint: String,
    execution_topology_fingerprint: String,
    request_input_fingerprint: String,
    initial_state_fingerprint: String,
    random_seed: u64,
    request_journal_event_count: u64,
    request_journal_fingerprint: String,
    active_sequence_fingerprint: String,
    completed_sequence_fingerprint: Option<String>,
    aborted_sequence_fingerprint: Option<String>,
    cleanup_status: ReplayCleanupStatus,
    operation_terminal_evidence_count: u64,
    operation_terminal_evidence_fingerprint: String,
    resource_pool_id: Option<ResourcePoolId>,
    resource_pool_identity_fingerprint: Option<String>,
    pool_journal_event_count: u64,
    pool_journal_fingerprint: Option<String>,
    plan_cleanup_fingerprint: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnvalidatedReplayIdentity {
    identity_version: ContractVersion,
    terminal_identity: UnvalidatedExecutionIdentityParts,
    resolved_plan_fingerprint: String,
    execution_topology_fingerprint: String,
    request_input_fingerprint: String,
    initial_state_fingerprint: String,
    random_seed: u64,
    request_journal_event_count: u64,
    request_journal_fingerprint: String,
    active_sequence_fingerprint: String,
    completed_sequence_fingerprint: Option<String>,
    aborted_sequence_fingerprint: Option<String>,
    cleanup_status: ReplayCleanupStatus,
    operation_terminal_evidence_count: u64,
    operation_terminal_evidence_fingerprint: String,
    resource_pool_id: Option<ResourcePoolId>,
    resource_pool_identity_fingerprint: Option<String>,
    pool_journal_event_count: u64,
    pool_journal_fingerprint: Option<String>,
    plan_cleanup_fingerprint: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ReplayIdentityWire {
    identity_version: ContractVersion,
    terminal_identity: UnvalidatedExecutionIdentityParts,
    resolved_plan_fingerprint: String,
    execution_topology_fingerprint: String,
    request_input_fingerprint: String,
    initial_state_fingerprint: String,
    random_seed: u64,
    request_journal_event_count: u64,
    request_journal_fingerprint: String,
    active_sequence_fingerprint: String,
    completed_sequence_fingerprint: Option<String>,
    aborted_sequence_fingerprint: Option<String>,
    cleanup_status: ReplayCleanupStatus,
    operation_terminal_evidence_count: u64,
    operation_terminal_evidence_fingerprint: String,
    resource_pool_id: Option<ResourcePoolId>,
    resource_pool_identity_fingerprint: Option<String>,
    pool_journal_event_count: u64,
    pool_journal_fingerprint: Option<String>,
    plan_cleanup_fingerprint: Option<String>,
}

impl From<ReplayIdentityWire> for UnvalidatedReplayIdentity {
    fn from(wire: ReplayIdentityWire) -> Self {
        Self {
            identity_version: wire.identity_version,
            terminal_identity: wire.terminal_identity,
            resolved_plan_fingerprint: wire.resolved_plan_fingerprint,
            execution_topology_fingerprint: wire.execution_topology_fingerprint,
            request_input_fingerprint: wire.request_input_fingerprint,
            initial_state_fingerprint: wire.initial_state_fingerprint,
            random_seed: wire.random_seed,
            request_journal_event_count: wire.request_journal_event_count,
            request_journal_fingerprint: wire.request_journal_fingerprint,
            active_sequence_fingerprint: wire.active_sequence_fingerprint,
            completed_sequence_fingerprint: wire.completed_sequence_fingerprint,
            aborted_sequence_fingerprint: wire.aborted_sequence_fingerprint,
            cleanup_status: wire.cleanup_status,
            operation_terminal_evidence_count: wire.operation_terminal_evidence_count,
            operation_terminal_evidence_fingerprint: wire.operation_terminal_evidence_fingerprint,
            resource_pool_id: wire.resource_pool_id,
            resource_pool_identity_fingerprint: wire.resource_pool_identity_fingerprint,
            pool_journal_event_count: wire.pool_journal_event_count,
            pool_journal_fingerprint: wire.pool_journal_fingerprint,
            plan_cleanup_fingerprint: wire.plan_cleanup_fingerprint,
        }
    }
}

fn validate_replay_plan_cleanup(
    active: &TrustedActiveSequenceBinding,
    aborted: bool,
    requirement: ReplayCleanupRequirement,
    cleanup: ReplayPlanCleanupEvidence<'_>,
) -> Result<(ReplayCleanupStatus, Option<String>), VNextError> {
    let expected_static_resources = active.static_entries().len();
    match cleanup {
        ReplayPlanCleanupEvidence::Pending => {
            if requirement == ReplayCleanupRequirement::RequireClean {
                return Err(invalid_event(
                    "clean replay requires an exact plan close or quarantine receipt",
                ));
            }
            Ok((ReplayCleanupStatus::CleanupPending, None))
        }
        ReplayPlanCleanupEvidence::Closed(receipt) => {
            if receipt.evidence() != active.plan()
                || receipt.released_static_resources() != expected_static_resources
            {
                return Err(invalid_event(
                    "replay plan close receipt differs from the active plan or static resource set",
                ));
            }
            Ok((
                if aborted {
                    ReplayCleanupStatus::SequenceQuiescent
                } else {
                    ReplayCleanupStatus::Completed
                },
                Some(canonical_fingerprint(receipt)),
            ))
        }
        ReplayPlanCleanupEvidence::Quarantined(receipt) => {
            let accounted_static_resources = receipt
                .released_static_resources()
                .checked_add(receipt.quarantined_static_resources())
                .ok_or_else(|| invalid_event("replay quarantine resource count overflows usize"))?;
            if expected_static_resources == 0
                || receipt.evidence() != active.plan()
                || accounted_static_resources != expected_static_resources
            {
                return Err(invalid_event(
                    "replay quarantine receipt differs from the active static plan resource set",
                ));
            }
            Ok((
                ReplayCleanupStatus::Quarantined,
                Some(canonical_fingerprint(receipt)),
            ))
        }
    }
}

impl ReplayIdentity {
    pub fn from_evidence(evidence: &ReplayEvidence<'_>) -> Result<Self, VNextError> {
        if evidence.request_journal.is_empty() {
            return Err(invalid_event(
                "replay evidence requires a non-empty request journal",
            ));
        }
        if !evidence.operation_quarantines.is_empty() {
            if evidence
                .operation_quarantines
                .iter()
                .any(|receipt| !receipt.is_current())
            {
                return Err(invalid_event(
                    "completion quarantine replay evidence was superseded by a successful drain",
                ));
            }
            if evidence.cleanup_requirement != ReplayCleanupRequirement::AllowPending
                || !matches!(evidence.plan_cleanup, ReplayPlanCleanupEvidence::Pending)
            {
                return Err(invalid_event(
                    "current completion quarantine is pending ownership and requires explicitly pending replay cleanup",
                ));
            }
        }

        let topology =
            TrustedExecutionTopology::from_plan(evidence.resolved_plan.execution_plan())?;
        let active = evidence.active_binding;
        let completed = evidence.completed_binding;
        let aborted = evidence.aborted_binding;
        if completed.is_some() == aborted.is_some() {
            return Err(invalid_event(
                "replay requires exactly one external sequence completion or abort binding",
            ));
        }
        if active.plan().plan_id() != topology.plan_id()
            || active.plan().plan_hash() != topology.plan_hash()
            || active.plan().device_id() != topology.device_id()
            || active.plan().runtime_implementation_fingerprint()
                != topology.device_runtime_implementation_fingerprint()
            || active.runtime_implementation_fingerprint()
                != topology.device_runtime_implementation_fingerprint()
        {
            return Err(invalid_event(
                "replay plan and active binding do not share one authority",
            ));
        }
        if let Some(completed) = completed {
            if completed.active_sequence_fingerprint() != active.fingerprint()
                || completed.sequence_authority() != active.sequence_authority()
                || completed.run_id() != active.run_id()
                || completed.request_id() != active.request_id()
                || completed.activation_epoch() != active.activation_epoch()
                || completed.runtime_implementation_fingerprint()
                    != active.runtime_implementation_fingerprint()
            {
                return Err(invalid_event(
                    "replay completion evidence differs from its active sequence",
                ));
            }
        }
        if let Some(aborted) = aborted {
            if !active.matches_abort_disposition(aborted.disposition())
                || aborted.active_sequence_fingerprint() != active.fingerprint()
                || aborted.sequence_authority() != active.sequence_authority()
                || aborted.run_id() != active.run_id()
                || aborted.request_id() != active.request_id()
                || aborted.activation_epoch() != active.activation_epoch()
                || aborted.runtime_implementation_fingerprint()
                    != active.runtime_implementation_fingerprint()
            {
                return Err(invalid_event(
                    "replay abort evidence differs from its active sequence",
                ));
            }
        }

        let first = evidence
            .request_journal
            .first()
            .expect("non-empty request journal");
        let terminal = evidence
            .request_journal
            .last()
            .expect("non-empty request journal");
        let run_id = &first.identity().parts().run_id;
        let request_id = &first.identity().parts().request_id;
        if run_id != active.run_id() || request_id != active.request_id() {
            return Err(invalid_event(
                "replay request journal differs from the active run/request binding",
            ));
        }

        let operation_terminals = replay_operation_terminals(evidence);
        let mut terminal_slots = BTreeSet::new();
        if operation_terminals
            .iter()
            .any(|(_, terminal)| !terminal_slots.insert(terminal.slot_id()))
        {
            return Err(invalid_event(
                "replay operation terminal evidence reuses a completion slot across terminal types",
            ));
        }

        let mut request_cursor = ExecutionEventCursor::new(run_id.clone(), request_id.clone());
        let mut observed_active_identity = false;
        let mut used_submitted_terminals = BTreeSet::new();
        let mut first_failure: Option<IdentifiedFailure> = None;
        for event in evidence.request_journal {
            observed_active_identity |= has_active(event.identity().parts());
            let context = match event.kind() {
                ExecutionEventKind::RequestAccepted => {
                    TrustedExecutionEventContext::pre_plan(run_id, request_id)
                }
                ExecutionEventKind::PlanBuilt => {
                    TrustedExecutionEventContext::bound(run_id, request_id, &topology)
                }
                ExecutionEventKind::FailureObserved => {
                    let failure = match event.detail() {
                        ExecutionEventDetail::Failure(failure) => failure,
                        _ => unreachable!("trusted FailureObserved shape was validated"),
                    };
                    if first_failure.is_some() {
                        return Err(invalid_event(
                            "replay request journal contains more than one first failure",
                        ));
                    }
                    first_failure = Some(failure.clone());
                    let unsubmitted_recoveries = operation_terminals
                        .iter()
                        .filter(|(_, terminal)| {
                            !terminal.had_submission_fence()
                                && terminal.contains_identity(failure.identity())
                        })
                        .collect::<Vec<_>>();
                    if unsubmitted_recoveries.len() > 1 {
                        return Err(invalid_event(
                            "operation failure matches multiple unsubmitted recovery receipts",
                        ));
                    }
                    TrustedExecutionEventContext::replay_failure(
                        run_id,
                        request_id,
                        event
                            .identity()
                            .parts()
                            .plan_id
                            .is_some()
                            .then_some(&topology),
                        has_active(event.identity().parts()).then_some(active),
                        failure,
                        unsubmitted_recoveries.first().map(|_| failure.identity()),
                    )
                }
                ExecutionEventKind::RequestFailed => match event.detail() {
                    ExecutionEventDetail::Failure(failure) => {
                        TrustedExecutionEventContext::failure(
                            run_id,
                            request_id,
                            event
                                .identity()
                                .parts()
                                .plan_id
                                .is_some()
                                .then_some(&topology),
                            has_active(event.identity().parts()).then_some(active),
                            failure,
                        )
                    }
                    ExecutionEventDetail::FailureTerminal { .. } => {
                        let failure = first_failure.as_ref().ok_or_else(|| {
                            invalid_event(
                                "terminal replay failure lacks its first FailureObserved evidence",
                            )
                        })?;
                        TrustedExecutionEventContext::failure_with_disposition(
                            run_id,
                            request_id,
                            &topology,
                            active,
                            has_completed(event.identity().parts())
                                .then_some(completed)
                                .flatten(),
                            has_aborted(event.identity().parts())
                                .then_some(aborted)
                                .flatten(),
                            failure,
                        )
                    }
                    _ => unreachable!("trusted RequestFailed shape was validated"),
                },
                ExecutionEventKind::OperationSubmitted => {
                    let matches = operation_terminals
                        .iter()
                        .filter_map(|(key, terminal)| {
                            terminal
                                .had_submission_fence()
                                .then(|| {
                                    terminal
                                        .participant_submission(event.identity())
                                        .map(|participant| (*key, participant))
                                })
                                .flatten()
                        })
                        .collect::<Vec<_>>();
                    if matches.len() != 1 || !used_submitted_terminals.insert(matches[0].0) {
                        return Err(invalid_event(
                            "replay operation event lacks one exact terminal receipt proving submission",
                        ));
                    }
                    TrustedExecutionEventContext::replay_operation_submitted(
                        run_id,
                        request_id,
                        &topology,
                        active,
                        operation_terminals
                            .iter()
                            .find(|(key, _)| *key == matches[0].0)
                            .and_then(|(_, terminal)| terminal.submission())
                            .expect("matched submitted participant has a batch receipt"),
                    )
                }
                ExecutionEventKind::NodeRetired => {
                    let matches = operation_terminals
                        .iter()
                        .filter_map(|(key, terminal)| {
                            terminal
                                .participant_completion(event.identity())
                                .map(|participant| (*key, participant))
                        })
                        .collect::<Vec<_>>();
                    if matches.len() != 1 || !used_submitted_terminals.contains(&matches[0].0) {
                        return Err(invalid_event(
                            "replay NodeRetired lacks the exact submitted batch completion projection",
                        ));
                    }
                    TrustedExecutionEventContext::replay_node_retired(
                        run_id,
                        request_id,
                        &topology,
                        active,
                        matches[0].1,
                    )
                }
                ExecutionEventKind::SequenceCompleted | ExecutionEventKind::RequestCompleted => {
                    let completed = completed.ok_or_else(|| {
                        invalid_event(
                            "successful replay journal lacks external sequence completion evidence",
                        )
                    })?;
                    TrustedExecutionEventContext::completed(
                        run_id, request_id, &topology, active, completed,
                    )
                }
                ExecutionEventKind::SequenceAborted => {
                    let aborted = aborted.ok_or_else(|| {
                        invalid_event(
                            "failed replay journal lacks external sequence abort evidence",
                        )
                    })?;
                    TrustedExecutionEventContext::aborted(
                        run_id, request_id, &topology, active, aborted,
                    )
                }
                _ => TrustedExecutionEventContext::active(run_id, request_id, &topology, active),
            };
            request_cursor.observe_against(event, &context)?;
        }
        if !request_cursor.is_terminal()
            || !observed_active_identity
            || !matches!(
                terminal.kind(),
                ExecutionEventKind::RequestCompleted | ExecutionEventKind::RequestFailed
            )
        {
            return Err(invalid_event(
                "replay request journal is incomplete or lacks an exact terminal event",
            ));
        }
        let submitted_terminal_count = operation_terminals
            .iter()
            .filter(|(_, terminal)| terminal.had_submission_fence())
            .count();
        if used_submitted_terminals.len() != submitted_terminal_count {
            return Err(invalid_event(
                "replay contains unused or missing operation terminal evidence for submitted work",
            ));
        }

        let operation_failures = first_failure
            .iter()
            .filter(|failure| has_active(failure.identity().parts()))
            .collect::<Vec<_>>();
        let submitted_identities = evidence
            .request_journal
            .iter()
            .filter(|event| event.kind() == ExecutionEventKind::OperationSubmitted)
            .map(ExecutionEvent::identity)
            .collect::<Vec<_>>();
        let mut used_operation_failures = BTreeSet::new();
        let request_failed = terminal.kind() == ExecutionEventKind::RequestFailed;
        for (_, operation_terminal) in &operation_terminals {
            let mut relevant_identities = submitted_identities
                .iter()
                .copied()
                .filter(|identity| operation_terminal.contains_identity(identity))
                .collect::<Vec<_>>();
            if relevant_identities.is_empty() {
                relevant_identities.extend(
                    operation_failures
                        .iter()
                        .map(|failure| failure.identity())
                        .filter(|identity| operation_terminal.contains_identity(identity)),
                );
            }
            if relevant_identities.len() != 1 {
                return Err(invalid_event(
                    "operation terminal evidence has no unique participant projection in this request journal",
                ));
            }
            let operation_identity = relevant_identities[0];
            let matching_failures = operation_failures
                .iter()
                .enumerate()
                .filter(|(_, failure)| failure.identity() == operation_identity)
                .collect::<Vec<_>>();
            if operation_terminal.participant_is_success(operation_identity) {
                if !matching_failures.is_empty() {
                    return Err(invalid_event(
                        "successful operation completion coexists with a failure for the same submitted operation",
                    ));
                }
                continue;
            }
            if !request_failed || matching_failures.len() != 1 {
                return Err(invalid_event(
                    "non-success operation terminal evidence requires one exact operation failure and RequestFailed",
                ));
            }
            if let Some(expected_failure) =
                operation_terminal.exact_failed_completion(operation_identity)
            {
                if expected_failure.identity() != operation_identity
                    || *matching_failures[0].1 != expected_failure
                {
                    return Err(invalid_event(
                        "failed-but-quiescent completion differs from the exact observed operation failure",
                    ));
                }
            }
            if !used_operation_failures.insert(matching_failures[0].0) {
                return Err(invalid_event(
                    "one operation failure was reused by multiple terminal evidence receipts",
                ));
            }
        }
        if used_operation_failures.len() != operation_failures.len() {
            return Err(invalid_event(
                "operation FailureObserved lacks one exact non-success terminal evidence receipt",
            ));
        }

        let (
            resource_pool_id,
            resource_pool_identity_fingerprint,
            pool_journal_event_count,
            pool_journal_fingerprint,
        ) = match evidence.pool_evidence {
            Some(pool_evidence) => {
                if evidence.pool_journal.is_empty() {
                    return Err(invalid_event(
                        "static replay evidence requires a non-empty pool journal",
                    ));
                }
                let active_static_pool_id = active.static_pool_id().ok_or_else(|| {
                    invalid_event("static pool replay evidence was supplied for a no-static plan")
                })?;
                let active_static_fingerprint = active
                    .static_pool_identity_fingerprint()
                    .ok_or_else(|| invalid_event("static replay pool lacks identity evidence"))?;
                let active_static_provisioning =
                    active.static_provisioning_identity().ok_or_else(|| {
                        invalid_event("static replay pool lacks provisioning identity evidence")
                    })?;
                if active.static_entries().is_empty()
                    || active_static_pool_id != pool_evidence.pool_id()
                    || active_static_fingerprint != pool_evidence.pool_identity_fingerprint()
                    || active_static_provisioning != pool_evidence.provisioning_identity()
                    || pool_evidence.topology_fingerprint() != topology.fingerprint()
                {
                    return Err(invalid_event(
                        "replay active binding and static pool evidence do not share one authority",
                    ));
                }
                let mut pool_cursor = ResourcePoolEventCursor::new(pool_evidence.clone());
                for event in evidence.pool_journal {
                    pool_cursor.observe(event)?;
                }
                if !pool_cursor.has_opened() || !pool_cursor.proves_active_binding(active) {
                    return Err(invalid_event(
                        "replay pool journal does not prove the complete committed active lease",
                    ));
                }
                (
                    Some(active_static_pool_id),
                    Some(active_static_fingerprint),
                    u64::try_from(evidence.pool_journal.len())
                        .map_err(|_| invalid_event("pool journal length exceeds u64"))?,
                    Some(canonical_fingerprint(&evidence.pool_journal)),
                )
            }
            None => {
                if !evidence.pool_journal.is_empty()
                    || active.static_pool_id().is_some()
                    || active.static_provisioning_identity().is_some()
                    || active.plan().static_provisioning_binding().is_some()
                    || active.plan().static_pool_identity().is_some()
                    || !active.static_entries().is_empty()
                {
                    return Err(invalid_event(
                        "no-static replay evidence contains static pool identity or journal state",
                    ));
                }
                (None, None, 0, None)
            }
        };
        let (cleanup_status, plan_cleanup_fingerprint) = validate_replay_plan_cleanup(
            active,
            aborted.is_some(),
            evidence.cleanup_requirement,
            evidence.plan_cleanup,
        )?;

        let request_journal_event_count = u64::try_from(evidence.request_journal.len())
            .map_err(|_| invalid_event("request journal length exceeds u64"))?;
        let operation_terminal_evidence_len = evidence
            .operation_completions
            .len()
            .checked_add(evidence.operation_drains.len())
            .and_then(|count| count.checked_add(evidence.operation_quarantines.len()))
            .ok_or_else(|| invalid_event("operation terminal evidence count exceeds usize"))?;
        let operation_terminal_evidence_count = u64::try_from(operation_terminal_evidence_len)
            .map_err(|_| invalid_event("operation terminal evidence count exceeds u64"))?;
        let identity = Self {
            identity_version: EXECUTION_IDENTITY_VERSION,
            terminal_identity: terminal.identity().clone(),
            resolved_plan_fingerprint: evidence.resolved_plan.fingerprint().to_owned(),
            execution_topology_fingerprint: topology.fingerprint().to_owned(),
            request_input_fingerprint: sha256_bytes(evidence.request_input),
            initial_state_fingerprint: sha256_bytes(evidence.initial_state),
            random_seed: evidence.random_seed,
            request_journal_event_count,
            request_journal_fingerprint: canonical_fingerprint(&evidence.request_journal),
            active_sequence_fingerprint: active.fingerprint().to_owned(),
            completed_sequence_fingerprint: completed
                .map(|binding| binding.fingerprint().to_owned()),
            aborted_sequence_fingerprint: aborted.map(|binding| binding.fingerprint().to_owned()),
            cleanup_status,
            operation_terminal_evidence_count,
            operation_terminal_evidence_fingerprint: canonical_fingerprint(
                &ReplayOperationTerminalFingerprint {
                    completions: evidence.operation_completions,
                    drains: evidence.operation_drains,
                    quarantines: evidence.operation_quarantines,
                },
            ),
            resource_pool_id,
            resource_pool_identity_fingerprint,
            pool_journal_event_count,
            pool_journal_fingerprint,
            plan_cleanup_fingerprint,
        };
        identity.validate_fingerprint_shape()?;
        Ok(identity)
    }

    fn validate_fingerprint_shape(&self) -> Result<(), VNextError> {
        for (value, label) in [
            (&self.resolved_plan_fingerprint, "resolved plan fingerprint"),
            (
                &self.execution_topology_fingerprint,
                "execution topology fingerprint",
            ),
            (&self.request_input_fingerprint, "request input fingerprint"),
            (&self.initial_state_fingerprint, "initial state fingerprint"),
            (
                &self.request_journal_fingerprint,
                "request journal fingerprint",
            ),
            (
                &self.active_sequence_fingerprint,
                "active sequence fingerprint",
            ),
            (
                &self.operation_terminal_evidence_fingerprint,
                "operation terminal evidence fingerprint",
            ),
        ] {
            validate_sha256(value, label)?;
        }
        if let Some(fingerprint) = &self.completed_sequence_fingerprint {
            validate_sha256(fingerprint, "completed sequence fingerprint")?;
        }
        if let Some(fingerprint) = &self.aborted_sequence_fingerprint {
            validate_sha256(fingerprint, "aborted sequence fingerprint")?;
        }
        if let Some(fingerprint) = &self.resource_pool_identity_fingerprint {
            validate_sha256(fingerprint, "resource pool identity fingerprint")?;
        }
        if let Some(fingerprint) = &self.pool_journal_fingerprint {
            validate_sha256(fingerprint, "pool journal fingerprint")?;
        }
        if let Some(fingerprint) = &self.plan_cleanup_fingerprint {
            validate_sha256(fingerprint, "plan cleanup fingerprint")?;
        }
        let has_static_pool = self.resource_pool_id.is_some();
        if self.completed_sequence_fingerprint.is_some()
            == self.aborted_sequence_fingerprint.is_some()
            || self.cleanup_status == ReplayCleanupStatus::Completed
                && self.completed_sequence_fingerprint.is_none()
            || self.cleanup_status == ReplayCleanupStatus::SequenceQuiescent
                && self.aborted_sequence_fingerprint.is_none()
            || self.cleanup_status == ReplayCleanupStatus::CleanupPending
                && self.plan_cleanup_fingerprint.is_some()
            || self.cleanup_status != ReplayCleanupStatus::CleanupPending
                && self.plan_cleanup_fingerprint.is_none()
            || has_static_pool != self.resource_pool_identity_fingerprint.is_some()
            || has_static_pool != self.pool_journal_fingerprint.is_some()
            || has_static_pool != (self.pool_journal_event_count > 0)
        {
            return Err(invalid_event(
                "replay cleanup status differs from its exact sequence disposition",
            ));
        }
        Ok(())
    }

    pub fn terminal_identity(&self) -> &ExecutionIdentityEnvelope {
        &self.terminal_identity
    }

    pub fn resolved_plan_fingerprint(&self) -> &str {
        &self.resolved_plan_fingerprint
    }

    pub fn request_input_fingerprint(&self) -> &str {
        &self.request_input_fingerprint
    }

    pub fn initial_state_fingerprint(&self) -> &str {
        &self.initial_state_fingerprint
    }

    pub const fn random_seed(&self) -> u64 {
        self.random_seed
    }

    pub fn request_journal_fingerprint(&self) -> &str {
        &self.request_journal_fingerprint
    }

    pub fn pool_journal_fingerprint(&self) -> Option<&str> {
        self.pool_journal_fingerprint.as_deref()
    }

    pub fn plan_cleanup_fingerprint(&self) -> Option<&str> {
        self.plan_cleanup_fingerprint.as_deref()
    }

    pub const fn cleanup_status(&self) -> ReplayCleanupStatus {
        self.cleanup_status
    }

    pub fn decode_untrusted(bytes: &[u8]) -> Result<UnvalidatedReplayIdentity, VNextError> {
        if bytes.len() > MAX_REPLAY_IDENTITY_WIRE_BYTES {
            return Err(invalid_event(
                "untrusted replay identity exceeds the wire byte limit",
            ));
        }
        serde_json::from_slice::<ReplayIdentityWire>(bytes)
            .map(Into::into)
            .map_err(|error| VNextError::Serialization {
                context: "decode untrusted replay identity",
                message: error.to_string(),
            })
    }
}

impl UnvalidatedReplayIdentity {
    pub fn revalidate(self, evidence: &ReplayEvidence<'_>) -> Result<ReplayIdentity, VNextError> {
        let rebuilt = ReplayIdentity::from_evidence(evidence)?;
        let supplied_terminal = ExecutionIdentityEnvelope::new(self.terminal_identity.into())?;
        if self.identity_version != rebuilt.identity_version
            || supplied_terminal != rebuilt.terminal_identity
            || self.resolved_plan_fingerprint != rebuilt.resolved_plan_fingerprint
            || self.execution_topology_fingerprint != rebuilt.execution_topology_fingerprint
            || self.request_input_fingerprint != rebuilt.request_input_fingerprint
            || self.initial_state_fingerprint != rebuilt.initial_state_fingerprint
            || self.random_seed != rebuilt.random_seed
            || self.request_journal_event_count != rebuilt.request_journal_event_count
            || self.request_journal_fingerprint != rebuilt.request_journal_fingerprint
            || self.active_sequence_fingerprint != rebuilt.active_sequence_fingerprint
            || self.completed_sequence_fingerprint != rebuilt.completed_sequence_fingerprint
            || self.aborted_sequence_fingerprint != rebuilt.aborted_sequence_fingerprint
            || self.cleanup_status != rebuilt.cleanup_status
            || self.operation_terminal_evidence_count != rebuilt.operation_terminal_evidence_count
            || self.operation_terminal_evidence_fingerprint
                != rebuilt.operation_terminal_evidence_fingerprint
            || self.resource_pool_id != rebuilt.resource_pool_id
            || self.resource_pool_identity_fingerprint != rebuilt.resource_pool_identity_fingerprint
            || self.pool_journal_event_count != rebuilt.pool_journal_event_count
            || self.pool_journal_fingerprint != rebuilt.pool_journal_fingerprint
            || self.plan_cleanup_fingerprint != rebuilt.plan_cleanup_fingerprint
        {
            return Err(invalid_event(
                "serialized replay identity differs from independently rebuilt evidence",
            ));
        }
        rebuilt.validate_fingerprint_shape()?;
        Ok(rebuilt)
    }
}
