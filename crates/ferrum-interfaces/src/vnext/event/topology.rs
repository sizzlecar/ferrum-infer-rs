use serde::Serialize;
use std::collections::{BTreeMap, BTreeSet};

use super::{
    canonical_fingerprint, invalid_event, validate_sha256, DeviceId, ExecutionPlan, NodeId,
    OperationId, PlanHash, PlanId, ProviderId, VNextError,
};
use crate::vnext::ProviderExecutionSemantics;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TrustedNodeTopology {
    pub(super) operation_id: OperationId,
    pub(super) provider_id: ProviderId,
    pub(super) provider_execution_semantics: ProviderExecutionSemantics,
    pub(super) dependencies: BTreeSet<NodeId>,
}

impl TrustedNodeTopology {
    pub fn operation_id(&self) -> &OperationId {
        &self.operation_id
    }

    pub fn provider_id(&self) -> &ProviderId {
        &self.provider_id
    }

    pub const fn provider_execution_semantics(&self) -> ProviderExecutionSemantics {
        self.provider_execution_semantics
    }

    pub fn dependencies(&self) -> &BTreeSet<NodeId> {
        &self.dependencies
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TrustedExecutionTopology {
    plan_id: PlanId,
    plan_hash: PlanHash,
    device_id: DeviceId,
    device_runtime_implementation_fingerprint: String,
    pub(super) nodes: BTreeMap<NodeId, TrustedNodeTopology>,
    #[serde(skip)]
    fingerprint: String,
}

impl TrustedExecutionTopology {
    pub fn from_plan(plan: &ExecutionPlan) -> Result<Self, VNextError> {
        let mut nodes = BTreeMap::new();
        for node in plan.payload().nodes() {
            if nodes
                .insert(
                    node.id().clone(),
                    TrustedNodeTopology {
                        operation_id: node.operation_id().clone(),
                        provider_id: node.selection().selected_provider().clone(),
                        provider_execution_semantics: node.provider_execution_semantics(),
                        dependencies: node.dependencies().iter().cloned().collect(),
                    },
                )
                .is_some()
            {
                return Err(invalid_event("trusted plan has duplicate node ids"));
            }
        }
        if nodes.is_empty() {
            return Err(invalid_event("trusted execution topology is empty"));
        }
        let mut topology = Self {
            plan_id: plan.payload().plan_id().clone(),
            plan_hash: plan.plan_hash().clone(),
            device_id: plan.payload().device_id().clone(),
            device_runtime_implementation_fingerprint: plan
                .payload()
                .device_runtime_implementation_fingerprint()
                .to_owned(),
            nodes,
            fingerprint: String::new(),
        };
        validate_sha256(
            &topology.device_runtime_implementation_fingerprint,
            "topology runtime implementation fingerprint",
        )?;
        topology.fingerprint = canonical_fingerprint(&topology);
        Ok(topology)
    }

    pub fn plan_id(&self) -> &PlanId {
        &self.plan_id
    }

    pub fn plan_hash(&self) -> &PlanHash {
        &self.plan_hash
    }

    pub fn device_id(&self) -> &DeviceId {
        &self.device_id
    }

    pub fn device_runtime_implementation_fingerprint(&self) -> &str {
        &self.device_runtime_implementation_fingerprint
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    pub fn node_ids(&self) -> BTreeSet<NodeId> {
        self.nodes.keys().cloned().collect()
    }

    pub fn node(&self, node_id: &NodeId) -> Option<&TrustedNodeTopology> {
        self.nodes.get(node_id)
    }
}
