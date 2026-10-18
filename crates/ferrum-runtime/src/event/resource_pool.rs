use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

use super::{
    canonical_fingerprint, invalid_event, validate_sha256, FailureDomain, MonotonicTimestamp,
    RequestIdentity, ResourceFailureId, ResourceFailureReceipt, ResourceId, ResourceLeaseState,
    ResourceLeaseTransitionReceipt, ResourceLeaseValidationContext, ResourceLedgerEntrySnapshot,
    ResourceLedgerSnapshot, ResourcePoolId, ResourceTransactionIdentity, ResourceTransactionState,
    ResourceTransitionReceipt, ResourceTransitionValidationContext, RunId,
    StaticProvisioningBinding, TransactionId, TrustedActiveSequenceBinding,
    TrustedExecutionTopology, UnvalidatedResourceLeaseTransitionReceipt,
    UnvalidatedResourceLeaseTransitionReceiptWire, UnvalidatedResourceTransitionReceipt,
    UnvalidatedResourceTransitionReceiptWire, VNextError, MAX_RESOURCE_POOL_EVENT_WIRE_BYTES,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResourcePoolEvidence {
    topology_fingerprint: String,
    pool_id: ResourcePoolId,
    pool_identity_fingerprint: String,
    admission: StaticProvisioningBinding,
    provisioning_identity: ResourceTransactionIdentity,
}

impl ResourcePoolEvidence {
    pub fn from_external(
        topology: &TrustedExecutionTopology,
        admission: &StaticProvisioningBinding,
        provisioning_identity: &ResourceTransactionIdentity,
    ) -> Result<Self, VNextError> {
        let pool_fingerprint = canonical_fingerprint(admission.pool_identity());
        if admission.plan_id() != topology.plan_id()
            || admission.plan_hash() != topology.plan_hash()
            || admission.device_id() != topology.device_id()
            || admission.device_runtime_implementation_fingerprint()
                != topology.device_runtime_implementation_fingerprint()
            || admission
                .pool_identity()
                .device_runtime_implementation_fingerprint()
                != topology.device_runtime_implementation_fingerprint()
            || provisioning_identity.pool_id() != admission.pool_id()
            || provisioning_identity.request_id() != admission.request_id()
        {
            return Err(invalid_event(
                "pool evidence differs from topology, admission, or provisioning identity",
            ));
        }
        Ok(Self {
            topology_fingerprint: topology.fingerprint().to_owned(),
            pool_id: admission.pool_id(),
            pool_identity_fingerprint: pool_fingerprint,
            admission: admission.clone(),
            provisioning_identity: provisioning_identity.clone(),
        })
    }

    pub const fn pool_id(&self) -> ResourcePoolId {
        self.pool_id
    }

    pub fn topology_fingerprint(&self) -> &str {
        &self.topology_fingerprint
    }

    pub fn pool_identity_fingerprint(&self) -> &str {
        &self.pool_identity_fingerprint
    }

    pub fn admission(&self) -> &StaticProvisioningBinding {
        &self.admission
    }

    pub fn provisioning_identity(&self) -> &ResourceTransactionIdentity {
        &self.provisioning_identity
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourcePoolEventKind {
    ResourcePoolOpened,
    ResourceTransition,
    ResourceLeaseTransition,
    ResourceFailed,
    ResourceRecoveryCompleted,
    ResourcePoolClosed,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResourcePoolEventIdentity {
    sequence: u64,
    pool_id: ResourcePoolId,
    pool_identity_fingerprint: String,
    provisioning_run_id: RunId,
    provisioning_request_id: RequestIdentity,
    transaction_id: TransactionId,
    resource_id: Option<ResourceId>,
    resource_generation: Option<u64>,
    resource_batch_fingerprint: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct UnvalidatedResourcePoolEventIdentity {
    sequence: u64,
    pool_id: ResourcePoolId,
    pool_identity_fingerprint: String,
    provisioning_run_id: RunId,
    provisioning_request_id: RequestIdentity,
    transaction_id: TransactionId,
    resource_id: Option<ResourceId>,
    resource_generation: Option<u64>,
    resource_batch_fingerprint: Option<String>,
}

impl ResourcePoolEventIdentity {
    fn for_evidence(
        sequence: u64,
        evidence: &ResourcePoolEvidence,
        resource_id: Option<ResourceId>,
        resource_generation: Option<u64>,
        resource_batch_fingerprint: Option<String>,
    ) -> Result<Self, VNextError> {
        if sequence == 0
            || resource_id.is_some() != resource_generation.is_some()
            || resource_generation == Some(0)
            || resource_id.is_some() && resource_batch_fingerprint.is_some()
        {
            return Err(invalid_event(
                "resource pool event identity shape is invalid",
            ));
        }
        if let Some(fingerprint) = &resource_batch_fingerprint {
            validate_sha256(fingerprint, "resource pool event batch fingerprint")?;
        }
        Ok(Self {
            sequence,
            pool_id: evidence.pool_id,
            pool_identity_fingerprint: evidence.pool_identity_fingerprint.clone(),
            provisioning_run_id: evidence.provisioning_identity.run_id().clone(),
            provisioning_request_id: evidence.provisioning_identity.request_id().clone(),
            transaction_id: evidence.provisioning_identity.transaction_id().clone(),
            resource_id,
            resource_generation,
            resource_batch_fingerprint,
        })
    }

    pub const fn sequence(&self) -> u64 {
        self.sequence
    }

    pub const fn pool_id(&self) -> ResourcePoolId {
        self.pool_id
    }

    pub fn pool_identity_fingerprint(&self) -> &str {
        &self.pool_identity_fingerprint
    }

    pub fn transaction_id(&self) -> &TransactionId {
        &self.transaction_id
    }

    pub fn provisioning_run_id(&self) -> &RunId {
        &self.provisioning_run_id
    }

    pub fn provisioning_request_id(&self) -> &RequestIdentity {
        &self.provisioning_request_id
    }

    pub fn resource_id(&self) -> Option<&ResourceId> {
        self.resource_id.as_ref()
    }

    pub const fn resource_generation(&self) -> Option<u64> {
        self.resource_generation
    }

    pub fn resource_batch_fingerprint(&self) -> Option<&str> {
        self.resource_batch_fingerprint.as_deref()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ResourcePoolEventDetail {
    Opened(StaticProvisioningBinding),
    Transition {
        receipt: ResourceTransitionReceipt,
        context: ResourceTransitionValidationContext,
    },
    LeaseTransition {
        receipt: ResourceLeaseTransitionReceipt,
        context: ResourceLeaseValidationContext,
    },
    Failure(ResourceFailureReceipt),
    Closed {
        ledger: Vec<ResourceLedgerEntrySnapshot>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum UnvalidatedResourcePoolEventDetail {
    Opened(serde_json::Value),
    Transition {
        receipt: UnvalidatedResourceTransitionReceipt,
        context: serde_json::Value,
    },
    LeaseTransition {
        receipt: UnvalidatedResourceLeaseTransitionReceipt,
        context: serde_json::Value,
    },
    Failure(serde_json::Value),
    Closed {
        ledger: serde_json::Value,
    },
}

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum UnvalidatedResourcePoolEventDetailWire {
    Opened(serde_json::Value),
    Transition {
        receipt: UnvalidatedResourceTransitionReceiptWire,
        context: serde_json::Value,
    },
    LeaseTransition {
        receipt: UnvalidatedResourceLeaseTransitionReceiptWire,
        context: serde_json::Value,
    },
    Failure(serde_json::Value),
    Closed {
        ledger: serde_json::Value,
    },
}

impl From<UnvalidatedResourcePoolEventDetailWire> for UnvalidatedResourcePoolEventDetail {
    fn from(wire: UnvalidatedResourcePoolEventDetailWire) -> Self {
        match wire {
            UnvalidatedResourcePoolEventDetailWire::Opened(value) => Self::Opened(value),
            UnvalidatedResourcePoolEventDetailWire::Transition { receipt, context } => {
                Self::Transition {
                    receipt: receipt.into(),
                    context,
                }
            }
            UnvalidatedResourcePoolEventDetailWire::LeaseTransition { receipt, context } => {
                Self::LeaseTransition {
                    receipt: receipt.into(),
                    context,
                }
            }
            UnvalidatedResourcePoolEventDetailWire::Failure(value) => Self::Failure(value),
            UnvalidatedResourcePoolEventDetailWire::Closed { ledger } => Self::Closed { ledger },
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ResourcePoolEvent {
    timestamp: MonotonicTimestamp,
    kind: ResourcePoolEventKind,
    identity: ResourcePoolEventIdentity,
    detail: ResourcePoolEventDetail,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct UnvalidatedResourcePoolEvent {
    timestamp: MonotonicTimestamp,
    kind: ResourcePoolEventKind,
    identity: UnvalidatedResourcePoolEventIdentity,
    detail: UnvalidatedResourcePoolEventDetail,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ResourcePoolEventWire {
    timestamp: MonotonicTimestamp,
    kind: ResourcePoolEventKind,
    identity: UnvalidatedResourcePoolEventIdentity,
    detail: UnvalidatedResourcePoolEventDetailWire,
}

impl From<ResourcePoolEventWire> for UnvalidatedResourcePoolEvent {
    fn from(wire: ResourcePoolEventWire) -> Self {
        Self {
            timestamp: wire.timestamp,
            kind: wire.kind,
            identity: wire.identity,
            detail: wire.detail.into(),
        }
    }
}

pub enum TrustedResourcePoolEventContext<'a> {
    Opened {
        evidence: &'a ResourcePoolEvidence,
    },
    Transition {
        evidence: &'a ResourcePoolEvidence,
        expected: &'a ResourceTransitionValidationContext,
    },
    LeaseTransition {
        evidence: &'a ResourcePoolEvidence,
        expected: &'a ResourceLeaseValidationContext,
    },
    Failure {
        evidence: &'a ResourcePoolEvidence,
        expected: &'a ResourceFailureReceipt,
    },
    Closed {
        evidence: &'a ResourcePoolEvidence,
        expected_snapshot: &'a ResourceLedgerSnapshot,
    },
}

impl<'a> TrustedResourcePoolEventContext<'a> {
    pub fn opened(evidence: &'a ResourcePoolEvidence) -> Self {
        Self::Opened { evidence }
    }

    pub fn transition(
        evidence: &'a ResourcePoolEvidence,
        expected: &'a ResourceTransitionValidationContext,
    ) -> Self {
        Self::Transition { evidence, expected }
    }

    pub fn lease_transition(
        evidence: &'a ResourcePoolEvidence,
        expected: &'a ResourceLeaseValidationContext,
    ) -> Self {
        Self::LeaseTransition { evidence, expected }
    }

    pub fn failure(
        evidence: &'a ResourcePoolEvidence,
        expected: &'a ResourceFailureReceipt,
    ) -> Self {
        Self::Failure { evidence, expected }
    }

    pub fn closed(
        evidence: &'a ResourcePoolEvidence,
        expected_snapshot: &'a ResourceLedgerSnapshot,
    ) -> Self {
        Self::Closed {
            evidence,
            expected_snapshot,
        }
    }
}

fn validate_pool_binding(
    evidence: &ResourcePoolEvidence,
    identity: &ResourceTransactionIdentity,
    admission: &StaticProvisioningBinding,
    label: &str,
) -> Result<(), VNextError> {
    if identity != evidence.provisioning_identity() || admission != evidence.admission() {
        return Err(invalid_event(format!(
            "resource pool {label} identity or admission differs from its exact pool evidence"
        )));
    }
    Ok(())
}

fn validate_failure_pool_binding(
    evidence: &ResourcePoolEvidence,
    failure: &ResourceFailureReceipt,
) -> Result<(), VNextError> {
    validate_pool_binding(evidence, failure.identity(), failure.admission(), "failure")?;
    if failure.failure().domain() != FailureDomain::Resource {
        return Err(invalid_event(
            "resource pool failure must carry the resource failure domain",
        ));
    }
    Ok(())
}

fn validate_terminal_snapshot(
    evidence: &ResourcePoolEvidence,
    snapshot: &ResourceLedgerSnapshot,
) -> Result<(), VNextError> {
    validate_pool_binding(
        evidence,
        snapshot.identity(),
        snapshot.admission(),
        "terminal snapshot",
    )?;
    if snapshot.entries().is_empty()
        || snapshot.entries().iter().any(|entry| {
            entry.entry().generation() != evidence.admission().admission_generation()
                || !matches!(
                    entry.transaction_state(),
                    ResourceTransactionState::RolledBack
                        | ResourceTransactionState::Released
                        | ResourceTransactionState::Quarantined
                )
                || entry.buffer_present()
        })
    {
        return Err(invalid_event(
            "resource pool terminal snapshot must be non-empty, generation-bound, terminal, and buffer-free",
        ));
    }
    Ok(())
}

fn item_or_batch_identity(
    entries: &[ResourceLedgerEntrySnapshot],
) -> (Option<ResourceId>, Option<u64>, Option<String>) {
    if entries.len() == 1 {
        (
            Some(entries[0].entry().resource_id().clone()),
            Some(entries[0].entry().generation()),
            None,
        )
    } else {
        (None, None, Some(canonical_fingerprint(&entries)))
    }
}

fn failure_item_or_batch_identity(
    failure: &ResourceFailureReceipt,
) -> (Option<ResourceId>, Option<u64>, Option<String>) {
    if let Some(point) = failure.failure_point() {
        (
            Some(point.resource_id().clone()),
            Some(point.generation()),
            None,
        )
    } else {
        (
            None,
            None,
            Some(canonical_fingerprint(&(
                failure.failure_id(),
                failure.completed(),
                failure.compensation(),
                failure.recovery_failures(),
                failure.recovery_strategy(),
                failure.ledger_before(),
                failure.ledger_after(),
            ))),
        )
    }
}

fn validate_transition_receipt_context(
    receipt: &ResourceTransitionReceipt,
    context: &ResourceTransitionValidationContext,
) -> Result<(), VNextError> {
    let rebuilt =
        UnvalidatedResourceTransitionReceipt::from(receipt).try_validate_against(context)?;
    if &rebuilt != receipt {
        return Err(invalid_event(
            "resource transition receipt does not match its exact ledger delta",
        ));
    }
    Ok(())
}

fn validate_lease_receipt_context(
    receipt: &ResourceLeaseTransitionReceipt,
    context: &ResourceLeaseValidationContext,
) -> Result<(), VNextError> {
    let rebuilt =
        UnvalidatedResourceLeaseTransitionReceipt::from(receipt).try_validate_against(context)?;
    if &rebuilt != receipt {
        return Err(invalid_event(
            "resource lease receipt does not match its exact ledger delta",
        ));
    }
    Ok(())
}

impl ResourcePoolEvent {
    pub fn opened(
        sequence: u64,
        timestamp: MonotonicTimestamp,
        evidence: &ResourcePoolEvidence,
    ) -> Result<Self, VNextError> {
        Ok(Self {
            timestamp,
            kind: ResourcePoolEventKind::ResourcePoolOpened,
            identity: ResourcePoolEventIdentity::for_evidence(
                sequence, evidence, None, None, None,
            )?,
            detail: ResourcePoolEventDetail::Opened(evidence.admission.clone()),
        })
    }

    pub fn transition(
        sequence: u64,
        timestamp: MonotonicTimestamp,
        evidence: &ResourcePoolEvidence,
        receipt: &ResourceTransitionReceipt,
        context: &ResourceTransitionValidationContext,
    ) -> Result<Self, VNextError> {
        validate_pool_binding(
            evidence,
            context.identity(),
            context.admission(),
            "transition",
        )?;
        validate_transition_receipt_context(receipt, context)?;
        let (resource, generation, batch) = item_or_batch_identity(context.after());
        Ok(Self {
            timestamp,
            kind: ResourcePoolEventKind::ResourceTransition,
            identity: ResourcePoolEventIdentity::for_evidence(
                sequence, evidence, resource, generation, batch,
            )?,
            detail: ResourcePoolEventDetail::Transition {
                receipt: receipt.clone(),
                context: context.clone(),
            },
        })
    }

    pub fn lease_transition(
        sequence: u64,
        timestamp: MonotonicTimestamp,
        evidence: &ResourcePoolEvidence,
        receipt: &ResourceLeaseTransitionReceipt,
        context: &ResourceLeaseValidationContext,
    ) -> Result<Self, VNextError> {
        validate_pool_binding(
            evidence,
            context.identity(),
            context.admission(),
            "lease transition",
        )?;
        validate_lease_receipt_context(receipt, context)?;
        let (resource, generation, batch) = item_or_batch_identity(context.after());
        Ok(Self {
            timestamp,
            kind: ResourcePoolEventKind::ResourceLeaseTransition,
            identity: ResourcePoolEventIdentity::for_evidence(
                sequence, evidence, resource, generation, batch,
            )?,
            detail: ResourcePoolEventDetail::LeaseTransition {
                receipt: receipt.clone(),
                context: context.clone(),
            },
        })
    }

    pub fn failed(
        sequence: u64,
        timestamp: MonotonicTimestamp,
        evidence: &ResourcePoolEvidence,
        failure: &ResourceFailureReceipt,
    ) -> Result<Self, VNextError> {
        validate_failure_pool_binding(evidence, failure)?;
        if failure.recovery_complete() {
            return Err(invalid_event(
                "ResourceFailed requires an incomplete recovery anchor",
            ));
        }
        let (resource, generation, batch) = failure_item_or_batch_identity(failure);
        Ok(Self {
            timestamp,
            kind: ResourcePoolEventKind::ResourceFailed,
            identity: ResourcePoolEventIdentity::for_evidence(
                sequence, evidence, resource, generation, batch,
            )?,
            detail: ResourcePoolEventDetail::Failure(failure.clone()),
        })
    }

    pub fn recovery_completed(
        sequence: u64,
        timestamp: MonotonicTimestamp,
        evidence: &ResourcePoolEvidence,
        recovery: &ResourceFailureReceipt,
    ) -> Result<Self, VNextError> {
        validate_failure_pool_binding(evidence, recovery)?;
        if !recovery.recovery_complete() {
            return Err(invalid_event(
                "ResourceRecoveryCompleted requires completed recovery evidence",
            ));
        }
        let (resource, generation, batch) = failure_item_or_batch_identity(recovery);
        Ok(Self {
            timestamp,
            kind: ResourcePoolEventKind::ResourceRecoveryCompleted,
            identity: ResourcePoolEventIdentity::for_evidence(
                sequence, evidence, resource, generation, batch,
            )?,
            detail: ResourcePoolEventDetail::Failure(recovery.clone()),
        })
    }

    pub fn closed(
        sequence: u64,
        timestamp: MonotonicTimestamp,
        evidence: &ResourcePoolEvidence,
        snapshot: &ResourceLedgerSnapshot,
    ) -> Result<Self, VNextError> {
        validate_terminal_snapshot(evidence, snapshot)?;
        Ok(Self {
            timestamp,
            kind: ResourcePoolEventKind::ResourcePoolClosed,
            identity: ResourcePoolEventIdentity::for_evidence(
                sequence, evidence, None, None, None,
            )?,
            detail: ResourcePoolEventDetail::Closed {
                ledger: snapshot.entries().to_vec(),
            },
        })
    }

    pub const fn timestamp(&self) -> MonotonicTimestamp {
        self.timestamp
    }

    pub const fn kind(&self) -> ResourcePoolEventKind {
        self.kind
    }

    pub fn identity(&self) -> &ResourcePoolEventIdentity {
        &self.identity
    }

    pub fn detail(&self) -> &ResourcePoolEventDetail {
        &self.detail
    }

    pub fn decode_untrusted(bytes: &[u8]) -> Result<UnvalidatedResourcePoolEvent, VNextError> {
        if bytes.len() > MAX_RESOURCE_POOL_EVENT_WIRE_BYTES {
            return Err(invalid_event(
                "untrusted resource pool event exceeds the wire byte limit",
            ));
        }
        let raw = serde_json::from_slice::<serde_json::Value>(bytes).map_err(|error| {
            VNextError::Serialization {
                context: "decode untrusted resource pool event",
                message: error.to_string(),
            }
        })?;
        let event = serde_json::from_value::<ResourcePoolEventWire>(raw.clone())
            .map(UnvalidatedResourcePoolEvent::from)
            .map_err(|error| VNextError::Serialization {
                context: "decode untrusted resource pool event",
                message: error.to_string(),
            })?;
        let canonical =
            serde_json::to_value(&event).map_err(|error| VNextError::Serialization {
                context: "serialize untrusted resource pool event",
                message: error.to_string(),
            })?;
        if canonical != raw {
            return Err(invalid_event(
                "resource pool event wire contains unknown or non-canonical nested fields",
            ));
        }
        Ok(event)
    }
}

impl UnvalidatedResourcePoolEventIdentity {
    fn matches(&self, expected: &ResourcePoolEventIdentity) -> bool {
        self.sequence == expected.sequence
            && self.pool_id == expected.pool_id
            && self.pool_identity_fingerprint == expected.pool_identity_fingerprint
            && self.provisioning_run_id == expected.provisioning_run_id
            && self.provisioning_request_id == expected.provisioning_request_id
            && self.transaction_id == expected.transaction_id
            && self.resource_id == expected.resource_id
            && self.resource_generation == expected.resource_generation
            && self.resource_batch_fingerprint == expected.resource_batch_fingerprint
    }
}

fn require_exact_wire_value(
    supplied: &serde_json::Value,
    expected: &(impl Serialize + ?Sized),
    label: &str,
) -> Result<(), VNextError> {
    let expected = serde_json::to_value(expected).map_err(|error| VNextError::Serialization {
        context: "serialize trusted resource pool event evidence",
        message: error.to_string(),
    })?;
    if supplied != &expected {
        return Err(invalid_event(format!(
            "untrusted resource pool {label} differs from independent evidence"
        )));
    }
    Ok(())
}

impl UnvalidatedResourcePoolEvent {
    pub fn revalidate(
        self,
        context: &TrustedResourcePoolEventContext<'_>,
    ) -> Result<ResourcePoolEvent, VNextError> {
        let rebuilt = match (self.kind, self.detail, context) {
            (
                ResourcePoolEventKind::ResourcePoolOpened,
                UnvalidatedResourcePoolEventDetail::Opened(admission),
                TrustedResourcePoolEventContext::Opened { evidence },
            ) => {
                require_exact_wire_value(&admission, evidence.admission(), "admission")?;
                ResourcePoolEvent::opened(self.identity.sequence, self.timestamp, evidence)?
            }
            (
                ResourcePoolEventKind::ResourceTransition,
                UnvalidatedResourcePoolEventDetail::Transition {
                    receipt,
                    context: supplied_context,
                },
                TrustedResourcePoolEventContext::Transition { evidence, expected },
            ) => {
                validate_pool_binding(
                    evidence,
                    expected.identity(),
                    expected.admission(),
                    "transition",
                )?;
                require_exact_wire_value(&supplied_context, expected, "transition context")?;
                let receipt = receipt.try_validate_against(expected)?;
                ResourcePoolEvent::transition(
                    self.identity.sequence,
                    self.timestamp,
                    evidence,
                    &receipt,
                    expected,
                )?
            }
            (
                ResourcePoolEventKind::ResourceLeaseTransition,
                UnvalidatedResourcePoolEventDetail::LeaseTransition {
                    receipt,
                    context: supplied_context,
                },
                TrustedResourcePoolEventContext::LeaseTransition { evidence, expected },
            ) => {
                validate_pool_binding(
                    evidence,
                    expected.identity(),
                    expected.admission(),
                    "lease transition",
                )?;
                require_exact_wire_value(&supplied_context, expected, "lease context")?;
                let receipt = receipt.try_validate_against(expected)?;
                ResourcePoolEvent::lease_transition(
                    self.identity.sequence,
                    self.timestamp,
                    evidence,
                    &receipt,
                    expected,
                )?
            }
            (
                kind @ (ResourcePoolEventKind::ResourceFailed
                | ResourcePoolEventKind::ResourceRecoveryCompleted),
                UnvalidatedResourcePoolEventDetail::Failure(supplied),
                TrustedResourcePoolEventContext::Failure { evidence, expected },
            ) => {
                validate_failure_pool_binding(evidence, expected)?;
                require_exact_wire_value(&supplied, expected, "failure receipt")?;
                match kind {
                    ResourcePoolEventKind::ResourceFailed => ResourcePoolEvent::failed(
                        self.identity.sequence,
                        self.timestamp,
                        evidence,
                        expected,
                    )?,
                    ResourcePoolEventKind::ResourceRecoveryCompleted => {
                        ResourcePoolEvent::recovery_completed(
                            self.identity.sequence,
                            self.timestamp,
                            evidence,
                            expected,
                        )?
                    }
                    _ => unreachable!(),
                }
            }
            (
                ResourcePoolEventKind::ResourcePoolClosed,
                UnvalidatedResourcePoolEventDetail::Closed { ledger },
                TrustedResourcePoolEventContext::Closed {
                    evidence,
                    expected_snapshot,
                },
            ) => {
                validate_terminal_snapshot(evidence, expected_snapshot)?;
                require_exact_wire_value(&ledger, expected_snapshot.entries(), "terminal ledger")?;
                ResourcePoolEvent::closed(
                    self.identity.sequence,
                    self.timestamp,
                    evidence,
                    expected_snapshot,
                )?
            }
            _ => {
                return Err(invalid_event(
                    "resource pool wire kind, detail, and external evidence do not match",
                ));
            }
        };
        if !self.identity.matches(&rebuilt.identity) {
            return Err(invalid_event(
                "resource pool wire identity differs from independently rebuilt evidence",
            ));
        }
        Ok(rebuilt)
    }
}

#[derive(Debug, Clone)]
pub struct ResourcePoolEventCursor {
    evidence: ResourcePoolEvidence,
    last_sequence: u64,
    last_timestamp: Option<MonotonicTimestamp>,
    opened: bool,
    closed: bool,
    ledgers: BTreeMap<TransactionId, Vec<ResourceLedgerEntrySnapshot>>,
    pending_failures: BTreeMap<TransactionId, ResourceFailureReceipt>,
    seen_failure_ids: BTreeSet<ResourceFailureId>,
    committed_resource_generations: Option<BTreeSet<(ResourceId, u64)>>,
}

impl ResourcePoolEventCursor {
    pub fn new(evidence: ResourcePoolEvidence) -> Self {
        Self {
            evidence,
            last_sequence: 0,
            last_timestamp: None,
            opened: false,
            closed: false,
            ledgers: BTreeMap::new(),
            pending_failures: BTreeMap::new(),
            seen_failure_ids: BTreeSet::new(),
            committed_resource_generations: None,
        }
    }

    pub fn observe(&mut self, event: &ResourcePoolEvent) -> Result<(), VNextError> {
        let mut next = self.clone();
        next.observe_inner(event)?;
        *self = next;
        Ok(())
    }

    pub const fn last_sequence(&self) -> u64 {
        self.last_sequence
    }

    pub const fn is_open(&self) -> bool {
        self.opened && !self.closed
    }

    pub const fn is_closed(&self) -> bool {
        self.closed
    }

    pub(super) const fn has_opened(&self) -> bool {
        self.opened
    }

    pub(super) fn proves_active_binding(&self, active: &TrustedActiveSequenceBinding) -> bool {
        let active_resources = active
            .static_entries()
            .iter()
            .map(|entry| (entry.resource_id().clone(), entry.generation()))
            .collect::<BTreeSet<_>>();
        self.committed_resource_generations.as_ref() == Some(&active_resources)
    }

    fn observe_inner(&mut self, event: &ResourcePoolEvent) -> Result<(), VNextError> {
        if event.identity.sequence != self.last_sequence.saturating_add(1)
            || self
                .last_timestamp
                .is_some_and(|timestamp| event.timestamp <= timestamp)
            || self.closed
            || event.identity.pool_id != self.evidence.pool_id
            || event.identity.pool_identity_fingerprint != self.evidence.pool_identity_fingerprint
            || event.identity.provisioning_run_id != *self.evidence.provisioning_identity.run_id()
            || event.identity.provisioning_request_id
                != *self.evidence.provisioning_identity.request_id()
            || event.identity.transaction_id
                != *self.evidence.provisioning_identity.transaction_id()
        {
            return Err(invalid_event(
                "pool journal sequence, timestamp, lifecycle, or pool identity is invalid",
            ));
        }
        match (&event.kind, &event.detail) {
            (
                ResourcePoolEventKind::ResourcePoolOpened,
                ResourcePoolEventDetail::Opened(admission),
            ) => {
                if self.opened
                    || self.last_sequence != 0
                    || admission != &self.evidence.admission
                    || event.identity.resource_id.is_some()
                    || event.identity.resource_batch_fingerprint.is_some()
                {
                    return Err(invalid_event(
                        "ResourcePoolOpened must be the exact first admission event",
                    ));
                }
                self.opened = true;
            }
            (
                ResourcePoolEventKind::ResourceTransition,
                ResourcePoolEventDetail::Transition { receipt, context },
            ) => {
                self.require_open()?;
                validate_transition_receipt_context(receipt, context)?;
                self.validate_resource_evidence(
                    context.identity(),
                    context.admission(),
                    context.after(),
                    &event.identity,
                )?;
                self.apply_ledger(
                    context.identity().transaction_id(),
                    context.before(),
                    context.after(),
                )?;
            }
            (
                ResourcePoolEventKind::ResourceLeaseTransition,
                ResourcePoolEventDetail::LeaseTransition { receipt, context },
            ) => {
                self.require_open()?;
                validate_lease_receipt_context(receipt, context)?;
                self.validate_resource_evidence(
                    context.identity(),
                    context.admission(),
                    context.after(),
                    &event.identity,
                )?;
                self.apply_ledger(
                    context.identity().transaction_id(),
                    context.before(),
                    context.after(),
                )?;
            }
            (ResourcePoolEventKind::ResourceFailed, ResourcePoolEventDetail::Failure(failure)) => {
                self.require_open()?;
                self.validate_failure(failure, &event.identity)?;
                let transaction = failure.identity().transaction_id().clone();
                if failure.recovery_complete()
                    || self.pending_failures.contains_key(&transaction)
                    || !self.seen_failure_ids.insert(failure.failure_id())
                {
                    return Err(invalid_event(
                        "resource failure is already complete, duplicated, or has a pending anchor",
                    ));
                }
                if let Some(current) = self.ledgers.get(&transaction) {
                    if current != failure.ledger_before() {
                        return Err(invalid_event(
                            "resource failure ledger does not continue pool journal",
                        ));
                    }
                }
                self.ledgers
                    .insert(transaction.clone(), failure.ledger_after().to_vec());
                self.pending_failures.insert(transaction, failure.clone());
            }
            (
                ResourcePoolEventKind::ResourceRecoveryCompleted,
                ResourcePoolEventDetail::Failure(recovery),
            ) => {
                self.require_open()?;
                self.validate_failure(recovery, &event.identity)?;
                let transaction = recovery.identity().transaction_id().clone();
                let anchor = self.pending_failures.get(&transaction).ok_or_else(|| {
                    invalid_event("resource recovery has no exact pending failure anchor")
                })?;
                recovery.validate_recovery_continuation(anchor)?;
                if recovery.failure_id() != anchor.failure_id() || !recovery.recovery_complete() {
                    return Err(invalid_event(
                        "resource recovery failure id or completion flag is invalid",
                    ));
                }
                self.ledgers
                    .insert(transaction.clone(), recovery.ledger_after().to_vec());
                self.pending_failures.remove(&transaction);
            }
            (
                ResourcePoolEventKind::ResourcePoolClosed,
                ResourcePoolEventDetail::Closed { ledger },
            ) => {
                self.require_open()?;
                let current = self
                    .ledgers
                    .get(self.evidence.provisioning_identity.transaction_id())
                    .ok_or_else(|| {
                        invalid_event("pool closure lacks a complete transaction ledger")
                    })?;
                if ledger != current
                    || !self.pending_failures.is_empty()
                    || current.iter().any(|entry| {
                        !matches!(
                            entry.transaction_state(),
                            ResourceTransactionState::RolledBack
                                | ResourceTransactionState::Released
                                | ResourceTransactionState::Quarantined
                        ) || entry.buffer_present()
                    })
                {
                    return Err(invalid_event(
                        "pool closure requires the exact terminal ledger and no pending recovery",
                    ));
                }
                self.closed = true;
            }
            _ => {
                return Err(invalid_event(
                    "resource pool event kind and detail do not match",
                ));
            }
        }
        self.last_sequence = event.identity.sequence;
        self.last_timestamp = Some(event.timestamp);
        Ok(())
    }

    fn require_open(&self) -> Result<(), VNextError> {
        if !self.opened || self.closed {
            return Err(invalid_event("resource pool is not open"));
        }
        Ok(())
    }

    fn validate_resource_evidence(
        &self,
        identity: &ResourceTransactionIdentity,
        admission: &StaticProvisioningBinding,
        after: &[ResourceLedgerEntrySnapshot],
        event_identity: &ResourcePoolEventIdentity,
    ) -> Result<(), VNextError> {
        let expected = item_or_batch_identity(after);
        if identity != &self.evidence.provisioning_identity
            || admission != &self.evidence.admission
            || event_identity.resource_id != expected.0
            || event_identity.resource_generation != expected.1
            || event_identity.resource_batch_fingerprint != expected.2
        {
            return Err(invalid_event(
                "resource event identity differs from receipt/admission/transaction",
            ));
        }
        Ok(())
    }

    fn validate_failure(
        &self,
        failure: &ResourceFailureReceipt,
        event_identity: &ResourcePoolEventIdentity,
    ) -> Result<(), VNextError> {
        let expected = failure_item_or_batch_identity(failure);
        if failure.failure().domain() != FailureDomain::Resource
            || failure.identity() != &self.evidence.provisioning_identity
            || failure.admission() != &self.evidence.admission
            || event_identity.resource_id != expected.0
            || event_identity.resource_generation != expected.1
            || event_identity.resource_batch_fingerprint != expected.2
        {
            return Err(invalid_event(
                "resource failure event differs from its full external anchor",
            ));
        }
        Ok(())
    }

    fn apply_ledger(
        &mut self,
        transaction_id: &TransactionId,
        before: &[ResourceLedgerEntrySnapshot],
        after: &[ResourceLedgerEntrySnapshot],
    ) -> Result<(), VNextError> {
        if before.is_empty()
            || after.is_empty()
            || before.len() != after.len()
            || self
                .ledgers
                .get(transaction_id)
                .is_some_and(|current| current != before)
        {
            return Err(invalid_event(
                "resource ledger before/after does not continue exactly",
            ));
        }
        let mut resources = BTreeSet::new();
        if after.iter().any(|entry| {
            entry.entry().generation() != self.evidence.admission.admission_generation()
                || !resources.insert(entry.entry().resource_id().clone())
        }) {
            return Err(invalid_event(
                "resource ledger contains duplicate or wrong-generation entries",
            ));
        }
        if after.iter().all(|entry| {
            entry.transaction_state() == ResourceTransactionState::Committed
                && entry.buffer_present()
                && entry.entry().state() == ResourceLeaseState::Active
        }) {
            self.committed_resource_generations = Some(
                after
                    .iter()
                    .map(|entry| {
                        (
                            entry.entry().resource_id().clone(),
                            entry.entry().generation(),
                        )
                    })
                    .collect(),
            );
        }
        self.ledgers.insert(transaction_id.clone(), after.to_vec());
        Ok(())
    }
}
