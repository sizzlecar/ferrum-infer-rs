use serde::{Deserialize, Serialize, Serializer};
use std::collections::BTreeSet;
use std::sync::Arc;

use super::{
    invalid_event, validate_sha256, ContractVersion, DeviceId, ExecutionFrameId, NodeId,
    NodeInvocationId, OperationId, PlanHash, PlanId, ProviderId, RequestIdentity, ResourceId,
    ResourcePoolId, RunId, SpanId, TransactionId, VNextError, EXECUTION_IDENTITY_VERSION,
};

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ExecutionIdentityParts {
    pub version: ContractVersion,
    pub run_id: RunId,
    pub request_id: RequestIdentity,
    pub sequence: u64,
    pub plan_id: Option<PlanId>,
    pub plan_hash: Option<PlanHash>,
    pub frame_id: Option<ExecutionFrameId>,
    pub node_invocation_id: Option<NodeInvocationId>,
    pub node_id: Option<NodeId>,
    pub operation_id: Option<OperationId>,
    pub provider_id: Option<ProviderId>,
    pub device_id: Option<DeviceId>,
    pub resource_pool_id: Option<ResourcePoolId>,
    pub resource_pool_identity_fingerprint: Option<String>,
    pub provisioning_run_id: Option<RunId>,
    pub provisioning_request_id: Option<RequestIdentity>,
    pub transaction_id: Option<TransactionId>,
    pub active_sequence_slot: Option<u32>,
    pub admission_generation: Option<u64>,
    pub activation_epoch: Option<u64>,
    pub runtime_implementation_fingerprint: Option<String>,
    pub active_sequence_fingerprint: Option<String>,
    pub completed_sequence_fingerprint: Option<String>,
    pub aborted_sequence_fingerprint: Option<String>,
    pub resource_id: Option<ResourceId>,
    pub resource_generation: Option<u64>,
    pub resource_batch_fingerprint: Option<String>,
    pub span_id: SpanId,
    pub parent_span_id: Option<SpanId>,
    pub async_links: Vec<SpanId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnvalidatedExecutionIdentityParts {
    pub version: ContractVersion,
    pub run_id: RunId,
    pub request_id: RequestIdentity,
    pub sequence: u64,
    pub plan_id: Option<PlanId>,
    pub plan_hash: Option<PlanHash>,
    pub frame_id: Option<ExecutionFrameId>,
    pub node_invocation_id: Option<NodeInvocationId>,
    pub node_id: Option<NodeId>,
    pub operation_id: Option<OperationId>,
    pub provider_id: Option<ProviderId>,
    pub device_id: Option<DeviceId>,
    pub resource_pool_id: Option<ResourcePoolId>,
    pub resource_pool_identity_fingerprint: Option<String>,
    pub provisioning_run_id: Option<RunId>,
    pub provisioning_request_id: Option<RequestIdentity>,
    pub transaction_id: Option<TransactionId>,
    pub active_sequence_slot: Option<u32>,
    pub admission_generation: Option<u64>,
    pub activation_epoch: Option<u64>,
    pub runtime_implementation_fingerprint: Option<String>,
    pub active_sequence_fingerprint: Option<String>,
    pub completed_sequence_fingerprint: Option<String>,
    pub aborted_sequence_fingerprint: Option<String>,
    pub resource_id: Option<ResourceId>,
    pub resource_generation: Option<u64>,
    pub resource_batch_fingerprint: Option<String>,
    pub span_id: SpanId,
    pub parent_span_id: Option<SpanId>,
    pub async_links: Vec<SpanId>,
}

impl From<UnvalidatedExecutionIdentityParts> for ExecutionIdentityParts {
    fn from(parts: UnvalidatedExecutionIdentityParts) -> Self {
        Self {
            version: parts.version,
            run_id: parts.run_id,
            request_id: parts.request_id,
            sequence: parts.sequence,
            plan_id: parts.plan_id,
            plan_hash: parts.plan_hash,
            frame_id: parts.frame_id,
            node_invocation_id: parts.node_invocation_id,
            node_id: parts.node_id,
            operation_id: parts.operation_id,
            provider_id: parts.provider_id,
            device_id: parts.device_id,
            resource_pool_id: parts.resource_pool_id,
            resource_pool_identity_fingerprint: parts.resource_pool_identity_fingerprint,
            provisioning_run_id: parts.provisioning_run_id,
            provisioning_request_id: parts.provisioning_request_id,
            transaction_id: parts.transaction_id,
            active_sequence_slot: parts.active_sequence_slot,
            admission_generation: parts.admission_generation,
            activation_epoch: parts.activation_epoch,
            runtime_implementation_fingerprint: parts.runtime_implementation_fingerprint,
            active_sequence_fingerprint: parts.active_sequence_fingerprint,
            completed_sequence_fingerprint: parts.completed_sequence_fingerprint,
            aborted_sequence_fingerprint: parts.aborted_sequence_fingerprint,
            resource_id: parts.resource_id,
            resource_generation: parts.resource_generation,
            resource_batch_fingerprint: parts.resource_batch_fingerprint,
            span_id: parts.span_id,
            parent_span_id: parts.parent_span_id,
            async_links: parts.async_links,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionIdentityEnvelope {
    parts: Arc<ExecutionIdentityParts>,
}

impl Serialize for ExecutionIdentityEnvelope {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.parts.as_ref().serialize(serializer)
    }
}

impl ExecutionIdentityEnvelope {
    pub fn new(parts: ExecutionIdentityParts) -> Result<Self, VNextError> {
        if parts.version != EXECUTION_IDENTITY_VERSION || parts.sequence == 0 {
            return Err(invalid_event(
                "execution identity version or sequence is invalid",
            ));
        }
        if parts.plan_id.is_some() != parts.plan_hash.is_some()
            || parts.node_invocation_id.is_some() != parts.node_id.is_some()
            || parts.node_id.is_some() && parts.frame_id.is_none()
            || parts.operation_id.is_some() != parts.provider_id.is_some()
            || parts.operation_id.is_some()
                && (parts.node_id.is_none() || parts.device_id.is_none())
            || parts.device_id.is_some() != parts.runtime_implementation_fingerprint.is_some()
        {
            return Err(invalid_event(
                "plan, frame, node invocation, operation, and provider identity shape is invalid",
            ));
        }

        let pool_present = parts.resource_pool_id.is_some();
        let pool_fields = [
            parts.resource_pool_identity_fingerprint.is_some(),
            parts.provisioning_run_id.is_some(),
            parts.provisioning_request_id.is_some(),
            parts.transaction_id.is_some(),
        ];
        if pool_fields.iter().any(|present| *present != pool_present) {
            return Err(invalid_event(
                "pool identity requires fingerprint and exact provisioning transaction",
            ));
        }
        let active_present = parts.active_sequence_slot.is_some();
        let active_fields = [
            parts.admission_generation.is_some(),
            parts.activation_epoch.is_some(),
            parts.active_sequence_fingerprint.is_some(),
        ];
        if active_fields
            .iter()
            .any(|present| *present != active_present)
            || active_present && parts.device_id.is_none()
        {
            return Err(invalid_event(
                "active identity requires slot, admission, epoch, runtime, and binding fingerprint",
            ));
        }
        if parts.admission_generation == Some(0) || parts.activation_epoch == Some(0) {
            return Err(invalid_event(
                "active admission generation and activation epoch must be non-zero",
            ));
        }
        if (parts.completed_sequence_fingerprint.is_some()
            || parts.aborted_sequence_fingerprint.is_some())
            && !active_present
            || parts.completed_sequence_fingerprint.is_some()
                && parts.aborted_sequence_fingerprint.is_some()
        {
            return Err(invalid_event(
                "sequence disposition requires one full active binding and cannot be both completed and aborted",
            ));
        }
        for (value, label) in [
            (
                parts.resource_pool_identity_fingerprint.as_deref(),
                "resource pool identity fingerprint",
            ),
            (
                parts.runtime_implementation_fingerprint.as_deref(),
                "runtime implementation fingerprint",
            ),
            (
                parts.active_sequence_fingerprint.as_deref(),
                "active sequence fingerprint",
            ),
            (
                parts.completed_sequence_fingerprint.as_deref(),
                "completed sequence fingerprint",
            ),
            (
                parts.aborted_sequence_fingerprint.as_deref(),
                "aborted sequence fingerprint",
            ),
            (
                parts.resource_batch_fingerprint.as_deref(),
                "resource batch fingerprint",
            ),
        ] {
            if let Some(value) = value {
                validate_sha256(value, label)?;
            }
        }
        if parts.resource_id.is_some() != parts.resource_generation.is_some()
            || parts.resource_generation == Some(0)
            || parts.resource_id.is_some() && parts.resource_batch_fingerprint.is_some()
            || (parts.resource_id.is_some() || parts.resource_batch_fingerprint.is_some())
                && !pool_present
        {
            return Err(invalid_event(
                "resource item/batch identity is incomplete, ambiguous, or lacks a pool",
            ));
        }
        if parts.parent_span_id.as_ref() == Some(&parts.span_id) {
            return Err(invalid_event("an execution span cannot parent itself"));
        }
        let mut links = BTreeSet::new();
        if parts.async_links.iter().any(|link| {
            link == &parts.span_id
                || parts.parent_span_id.as_ref() == Some(link)
                || !links.insert(link.clone())
        }) {
            return Err(invalid_event(
                "async links must be unique and distinct from span and parent",
            ));
        }
        Ok(Self {
            parts: Arc::new(parts),
        })
    }

    pub fn parts(&self) -> &ExecutionIdentityParts {
        self.parts.as_ref()
    }
}
