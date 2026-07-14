use super::{
    AttributeId, BTreeMap, BTreeSet, CapabilityId, ContractVersion, Deserialize, Deserializer,
    MemoryPlan, ModelFamilyId, NodeId, OperationId, OperationRegistryAuthority, PlanExactAlias,
    PlanHash, PlanId, PlanNode, PlanProviderRejectReason, PlanSchemaVersion, PlanStateEffect,
    ProviderId, ProviderResourcePlan, ProviderSelection, ProviderWorkspaceRequirement,
    QuantizationFormatId, ResolvedValueBinding, ResourceId, SemanticValue, Serialize,
    WeightFormatId,
};

/// Per-node trusted physical resolution. It supplies physical bindings and a
/// provider estimator result, but cannot provide memory totals, compatibility
/// reports, plan identities, or hashes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlanNodeResolution {
    pub(super) operation_registry_authority: OperationRegistryAuthority,
    pub(super) node_id: NodeId,
    pub(super) values: Vec<ResolvedValueBinding>,
    pub(super) required_capabilities: BTreeSet<CapabilityId>,
    pub(super) preferred_provider: Option<ProviderId>,
    pub(super) provider_resource_candidates: Vec<ProviderResourcePlan>,
    pub(super) provider_resolution_rejections: BTreeMap<ProviderId, PlanProviderRejectReason>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionPlanPayload {
    pub(super) schema: PlanSchemaVersion,
    pub(super) plan_id: PlanId,
    pub(super) family_id: ModelFamilyId,
    pub(super) device_id: super::DeviceId,
    pub(super) device_runtime_implementation_fingerprint: String,
    pub(super) prepared_family_fingerprint: String,
    pub(super) program_fingerprint: String,
    pub(super) capability_catalog_fingerprint: String,
    pub(super) policy_version: ContractVersion,
    pub(super) policy_fingerprint: String,
    pub(super) weight_format: WeightFormatId,
    pub(super) quantization_formats: BTreeSet<QuantizationFormatId>,
    pub(super) nodes: Vec<PlanNode>,
    pub(super) memory: MemoryPlan,
}

impl ExecutionPlanPayload {
    pub const fn schema(&self) -> PlanSchemaVersion {
        self.schema
    }

    pub fn plan_id(&self) -> &PlanId {
        &self.plan_id
    }

    pub fn family_id(&self) -> &ModelFamilyId {
        &self.family_id
    }

    pub fn device_id(&self) -> &super::DeviceId {
        &self.device_id
    }

    pub fn device_runtime_implementation_fingerprint(&self) -> &str {
        &self.device_runtime_implementation_fingerprint
    }

    pub fn prepared_family_fingerprint(&self) -> &str {
        &self.prepared_family_fingerprint
    }

    pub fn program_fingerprint(&self) -> &str {
        &self.program_fingerprint
    }

    pub fn capability_catalog_fingerprint(&self) -> &str {
        &self.capability_catalog_fingerprint
    }

    pub const fn policy_version(&self) -> ContractVersion {
        self.policy_version
    }

    pub fn policy_fingerprint(&self) -> &str {
        &self.policy_fingerprint
    }

    pub fn weight_format(&self) -> &WeightFormatId {
        &self.weight_format
    }

    pub fn quantization_formats(&self) -> &BTreeSet<QuantizationFormatId> {
        &self.quantization_formats
    }

    pub fn nodes(&self) -> &[PlanNode] {
        &self.nodes
    }

    pub fn memory(&self) -> &MemoryPlan {
        &self.memory
    }
}

#[derive(Serialize)]
pub(super) struct PlanHashMaterial<'a> {
    pub(super) schema: PlanSchemaVersion,
    pub(super) family_id: &'a ModelFamilyId,
    pub(super) device_id: &'a super::DeviceId,
    pub(super) device_runtime_implementation_fingerprint: &'a str,
    pub(super) prepared_family_fingerprint: &'a str,
    pub(super) program_fingerprint: &'a str,
    pub(super) capability_catalog_fingerprint: &'a str,
    pub(super) policy_version: ContractVersion,
    pub(super) policy_fingerprint: &'a str,
    pub(super) weight_format: &'a WeightFormatId,
    pub(super) quantization_formats: &'a BTreeSet<QuantizationFormatId>,
    pub(super) nodes: &'a [PlanNode],
    pub(super) memory: &'a MemoryPlan,
}

impl<'a> From<&'a ExecutionPlanPayload> for PlanHashMaterial<'a> {
    fn from(payload: &'a ExecutionPlanPayload) -> Self {
        Self {
            schema: payload.schema,
            family_id: &payload.family_id,
            device_id: &payload.device_id,
            device_runtime_implementation_fingerprint: &payload
                .device_runtime_implementation_fingerprint,
            prepared_family_fingerprint: &payload.prepared_family_fingerprint,
            program_fingerprint: &payload.program_fingerprint,
            capability_catalog_fingerprint: &payload.capability_catalog_fingerprint,
            policy_version: payload.policy_version,
            policy_fingerprint: &payload.policy_fingerprint,
            weight_format: &payload.weight_format,
            quantization_formats: &payload.quantization_formats,
            nodes: &payload.nodes,
            memory: &payload.memory,
        }
    }
}

/// A wire payload is deliberately not an executable plan. It must be rebuilt
/// against a typed model family, catalog, and runtime policy before use.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnvalidatedProviderResourcePlan {
    pub(super) provider_id: ProviderId,
    pub(super) estimator_id: String,
    pub(super) estimator_version: ContractVersion,
    pub(super) estimator_implementation_fingerprint: String,
    pub(super) estimator_input_fingerprint: String,
    pub(super) estimate_fingerprint: String,
    pub(super) value_alignment_bytes: u64,
    pub(super) scratch: Option<ProviderWorkspaceRequirement>,
    pub(super) persistent: Option<ProviderWorkspaceRequirement>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UnvalidatedPlanNode {
    pub(super) id: NodeId,
    pub(super) dependencies: Vec<NodeId>,
    pub(super) operation_id: OperationId,
    pub(super) operation_version: ContractVersion,
    pub(super) operation_fingerprint: String,
    pub(super) provider_implementation_fingerprint: String,
    pub(super) required_capabilities: BTreeSet<CapabilityId>,
    pub(super) attributes: BTreeMap<AttributeId, SemanticValue>,
    pub(super) selection: ProviderSelection,
    pub(super) provider_resources: UnvalidatedProviderResourcePlan,
    pub(super) values: Vec<ResolvedValueBinding>,
    pub(super) exact_aliases: Vec<PlanExactAlias>,
    pub(super) state_effects: Vec<PlanStateEffect>,
    pub(super) scratch_resource: Option<ResourceId>,
    pub(super) persistent_resource: Option<ResourceId>,
    pub(super) resources: Vec<ResourceId>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct UnvalidatedExecutionPlanPayload {
    pub(super) schema: PlanSchemaVersion,
    pub(super) plan_id: PlanId,
    pub(super) family_id: ModelFamilyId,
    pub(super) device_id: super::DeviceId,
    pub(super) device_runtime_implementation_fingerprint: String,
    pub(super) prepared_family_fingerprint: String,
    pub(super) program_fingerprint: String,
    pub(super) capability_catalog_fingerprint: String,
    pub(super) policy_version: ContractVersion,
    pub(super) policy_fingerprint: String,
    pub(super) weight_format: WeightFormatId,
    pub(super) quantization_formats: BTreeSet<QuantizationFormatId>,
    pub(super) nodes: Vec<UnvalidatedPlanNode>,
    pub(super) memory: MemoryPlan,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnvalidatedExecutionPlan {
    pub(super) payload: UnvalidatedExecutionPlanPayload,
    pub(super) plan_hash: PlanHash,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub(crate) struct UnvalidatedExecutionPlanWire {
    pub(super) payload: UnvalidatedExecutionPlanPayload,
    pub(super) plan_hash: PlanHash,
}

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct UnvalidatedExecutionPlanWireFields {
    pub(super) payload: UnvalidatedExecutionPlanPayload,
    pub(super) plan_hash: PlanHash,
}

impl<'de> Deserialize<'de> for UnvalidatedExecutionPlanWire {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let raw = serde_json::Value::deserialize(deserializer)?;
        let fields = UnvalidatedExecutionPlanWireFields::deserialize(&raw)
            .map_err(serde::de::Error::custom)?;
        let canonical = serde_json::to_value(&fields).map_err(serde::de::Error::custom)?;
        if canonical != raw {
            return Err(serde::de::Error::custom(
                "execution plan wire contains unknown or non-canonical nested fields",
            ));
        }
        Ok(Self {
            payload: fields.payload,
            plan_hash: fields.plan_hash,
        })
    }
}

impl From<UnvalidatedExecutionPlanWire> for UnvalidatedExecutionPlan {
    fn from(wire: UnvalidatedExecutionPlanWire) -> Self {
        Self {
            payload: wire.payload,
            plan_hash: wire.plan_hash,
        }
    }
}
