use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use super::{
    invalid_plan, AllocationLifetime, ContractVersion, ElementType, ExecutionPlan, NodeId,
    PlanHash, ProgramValueId, ProviderId, ResolvedValueBinding, ResourceId, StateId, TensorAccess,
    TokenSpanWork, VNextError, WeightId,
};
use crate::vnext::{ProviderExecutionContractFingerprint, ProviderReplayEquivalence};

pub const EXECUTION_DETERMINISM_WITNESS_VERSION: ContractVersion = ContractVersion::new(1, 0);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ExecutionDeterminismWitnessKind {
    Output {
        value_id: ProgramValueId,
        output_ordinal: u32,
    },
    StateEffect {
        state_id: StateId,
        state_value_id: ProgramValueId,
        lifetime: AllocationLifetime,
        access: TensorAccess,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionDeterminismWitnessSpec {
    node_id: NodeId,
    provider_id: ProviderId,
    provider_implementation_fingerprint: String,
    provider_execution_contract_fingerprint: ProviderExecutionContractFingerprint,
    kind: ExecutionDeterminismWitnessKind,
    storage_component_ordinal: u32,
    storage_component_id: Option<WeightId>,
    resource_id: ResourceId,
    logical_offset_bytes: u64,
    canonical_length_bytes: u64,
    element_type: ElementType,
    token_bytes_per_token: Option<u64>,
}

impl ExecutionDeterminismWitnessSpec {
    fn from_binding(
        node: &super::PlanNode,
        kind: ExecutionDeterminismWitnessKind,
        binding: &ResolvedValueBinding,
    ) -> Result<Vec<Self>, VNextError> {
        let components = binding.storage().components();
        let single_component_logical_length = (components.len() == 1)
            .then(|| binding.tensor().minimum_storage_bytes())
            .transpose()?;
        components
            .iter()
            .enumerate()
            .map(|(component_ordinal, component)| {
                // A single-component value has a typed logical span. Composite
                // encodings do not expose a lossless logical-to-component byte
                // map, so every declared physical byte is part of the proof.
                let canonical_length_bytes =
                    single_component_logical_length.unwrap_or(component.length_bytes());
                if canonical_length_bytes == 0
                    || canonical_length_bytes > component.length_bytes()
                    || component
                        .offset_bytes()
                        .checked_add(canonical_length_bytes)
                        .is_none()
                {
                    return Err(invalid_plan(format!(
                        "node `{}` determinism witness `{}` component {component_ordinal} has an invalid physical range",
                        node.id(),
                        binding.value_id()
                    )));
                }
                let token_bytes_per_token = node
                    .work()
                    .token_projection(binding.role(), binding.ordinal())
                    .map(|projection| {
                        let canonical_extent = projection.canonical_extent();
                        if canonical_extent == 0
                            || canonical_length_bytes % canonical_extent != 0
                        {
                            return Err(invalid_plan(format!(
                                "node `{}` determinism witness `{}` component {component_ordinal} has a non-integral token projection",
                                node.id(),
                                binding.value_id()
                            )));
                        }
                        canonical_length_bytes
                            .checked_div(canonical_extent)
                            .filter(|bytes| *bytes > 0)
                            .ok_or_else(|| {
                                invalid_plan(format!(
                                    "node `{}` determinism witness `{}` component {component_ordinal} has an invalid token projection",
                                    node.id(),
                                    binding.value_id()
                                ))
                            })
                    })
                    .transpose()?;
                Ok(Self {
                    node_id: node.id().clone(),
                    provider_id: node.selection().selected_provider().clone(),
                    provider_implementation_fingerprint: node
                        .provider_implementation_fingerprint()
                        .to_owned(),
                    provider_execution_contract_fingerprint: node
                        .provider_execution_semantics()
                        .contract_fingerprint(),
                    kind: kind.clone(),
                    storage_component_ordinal: u32::try_from(component_ordinal).map_err(|_| {
                        invalid_plan("determinism witness component ordinal exceeds u32")
                    })?,
                    storage_component_id: component.component_id().cloned(),
                    resource_id: component.resource_id().clone(),
                    logical_offset_bytes: component.offset_bytes(),
                    canonical_length_bytes,
                    element_type: component.element_type(),
                    token_bytes_per_token,
                })
            })
            .collect()
    }

    pub fn node_id(&self) -> &NodeId {
        &self.node_id
    }

    pub fn provider_id(&self) -> &ProviderId {
        &self.provider_id
    }

    pub fn provider_implementation_fingerprint(&self) -> &str {
        &self.provider_implementation_fingerprint
    }

    pub const fn provider_execution_contract_fingerprint(
        &self,
    ) -> ProviderExecutionContractFingerprint {
        self.provider_execution_contract_fingerprint
    }

    pub fn kind(&self) -> &ExecutionDeterminismWitnessKind {
        &self.kind
    }

    pub const fn storage_component_ordinal(&self) -> u32 {
        self.storage_component_ordinal
    }

    pub fn storage_component_id(&self) -> Option<&WeightId> {
        self.storage_component_id.as_ref()
    }

    pub fn resource_id(&self) -> &ResourceId {
        &self.resource_id
    }

    pub const fn logical_offset_bytes(&self) -> u64 {
        self.logical_offset_bytes
    }

    pub const fn canonical_length_bytes(&self) -> u64 {
        self.canonical_length_bytes
    }

    pub const fn element_type(&self) -> ElementType {
        self.element_type
    }

    pub const fn token_bytes_per_token(&self) -> Option<u64> {
        self.token_bytes_per_token
    }

    pub fn active_length_bytes(&self, token_span: &TokenSpanWork) -> Result<u64, VNextError> {
        let bytes = match self.token_bytes_per_token {
            Some(bytes_per_token) => bytes_per_token
                .checked_mul(token_span.immediate_tokens())
                .ok_or_else(|| invalid_plan("determinism witness active byte extent overflows"))?,
            None => self.canonical_length_bytes,
        };
        if bytes == 0 || bytes > self.canonical_length_bytes {
            return Err(invalid_plan(
                "determinism witness active byte extent exceeds its canonical value",
            ));
        }
        Ok(bytes)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderDeterminismCoverageRequirement {
    provider_id: ProviderId,
    provider_implementation_fingerprint: String,
    provider_execution_contract_fingerprint: ProviderExecutionContractFingerprint,
    node_ids: Vec<NodeId>,
}

impl ProviderDeterminismCoverageRequirement {
    pub fn provider_id(&self) -> &ProviderId {
        &self.provider_id
    }

    pub fn provider_implementation_fingerprint(&self) -> &str {
        &self.provider_implementation_fingerprint
    }

    pub const fn provider_execution_contract_fingerprint(
        &self,
    ) -> ProviderExecutionContractFingerprint {
        self.provider_execution_contract_fingerprint
    }

    pub fn node_ids(&self) -> &[NodeId] {
        &self.node_ids
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionDeterminismWitnessPlan {
    schema_version: ContractVersion,
    plan_hash: PlanHash,
    replay_provider_requirements: Vec<ProviderDeterminismCoverageRequirement>,
    witnesses: Vec<ExecutionDeterminismWitnessSpec>,
}

impl ExecutionDeterminismWitnessPlan {
    pub const fn schema_version(&self) -> ContractVersion {
        self.schema_version
    }

    pub fn plan_hash(&self) -> &PlanHash {
        &self.plan_hash
    }

    pub fn replay_provider_requirements(&self) -> &[ProviderDeterminismCoverageRequirement] {
        &self.replay_provider_requirements
    }

    pub fn witnesses(&self) -> &[ExecutionDeterminismWitnessSpec] {
        &self.witnesses
    }
}

impl ExecutionPlan {
    /// Derives the complete same-runtime proof denominator from the trusted
    /// plan. Hardware runners consume this contract rather than maintaining a
    /// second provider/output/state inventory.
    pub fn determinism_witness_plan(&self) -> Result<ExecutionDeterminismWitnessPlan, VNextError> {
        let mut witnesses = Vec::new();
        let mut replay_providers = BTreeMap::<
            ProviderId,
            (
                String,
                ProviderExecutionContractFingerprint,
                BTreeSet<NodeId>,
            ),
        >::new();

        for node in self.payload().nodes() {
            let semantics = node.provider_execution_semantics();
            if semantics.replay_equivalence() == ProviderReplayEquivalence::BitwiseEagerEquivalent {
                match replay_providers.entry(node.selection().selected_provider().clone()) {
                    std::collections::btree_map::Entry::Vacant(entry) => {
                        entry.insert((
                            node.provider_implementation_fingerprint().to_owned(),
                            semantics.contract_fingerprint(),
                            BTreeSet::from([node.id().clone()]),
                        ));
                    }
                    std::collections::btree_map::Entry::Occupied(mut entry) => {
                        let (implementation, contract, nodes) = entry.get_mut();
                        if implementation != node.provider_implementation_fingerprint()
                            || *contract != semantics.contract_fingerprint()
                        {
                            return Err(invalid_plan(format!(
                                "provider `{}` has inconsistent determinism identity in one plan",
                                node.selection().selected_provider()
                            )));
                        }
                        nodes.insert(node.id().clone());
                    }
                }
            }

            for binding in node
                .values()
                .iter()
                .filter(|binding| binding.role() == super::ResolvedValueRole::Output)
            {
                witnesses.extend(ExecutionDeterminismWitnessSpec::from_binding(
                    node,
                    ExecutionDeterminismWitnessKind::Output {
                        value_id: binding.value_id().clone(),
                        output_ordinal: binding.ordinal(),
                    },
                    binding,
                )?);
            }

            for effect in node.state_effects().iter().filter(|effect| {
                matches!(
                    effect.access(),
                    TensorAccess::Write | TensorAccess::ReadWrite
                )
            }) {
                let mut matched_resources = BTreeSet::new();
                for binding in node.values().iter().filter(|binding| {
                    binding.value_id() == effect.state_value_id()
                        && matches!(
                            binding.access(),
                            TensorAccess::Write | TensorAccess::ReadWrite
                        )
                }) {
                    let specs = ExecutionDeterminismWitnessSpec::from_binding(
                        node,
                        ExecutionDeterminismWitnessKind::StateEffect {
                            state_id: effect.state_id().clone(),
                            state_value_id: effect.state_value_id().clone(),
                            lifetime: effect.lifetime(),
                            access: effect.access(),
                        },
                        binding,
                    )?;
                    matched_resources.extend(specs.iter().map(|spec| spec.resource_id().clone()));
                    witnesses.extend(specs);
                }
                let expected_resources = effect
                    .resource_ids()
                    .iter()
                    .cloned()
                    .collect::<BTreeSet<_>>();
                if matched_resources.is_empty() || matched_resources != expected_resources {
                    return Err(invalid_plan(format!(
                        "node `{}` writable state `{}` has no exact determinism witness closure",
                        node.id(),
                        effect.state_id()
                    )));
                }
            }
        }

        if witnesses.is_empty() {
            return Err(invalid_plan(
                "execution determinism witness plan has no declared outputs or writable state",
            ));
        }

        let replay_provider_requirements = replay_providers
            .into_iter()
            .map(
                |(provider_id, (provider_implementation_fingerprint, contract, node_ids))| {
                    ProviderDeterminismCoverageRequirement {
                        provider_id,
                        provider_implementation_fingerprint,
                        provider_execution_contract_fingerprint: contract,
                        node_ids: node_ids.into_iter().collect(),
                    }
                },
            )
            .collect();

        Ok(ExecutionDeterminismWitnessPlan {
            schema_version: EXECUTION_DETERMINISM_WITNESS_VERSION,
            plan_hash: self.plan_hash().clone(),
            replay_provider_requirements,
            witnesses,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vnext::{
        AliasPolicy, BufferUsage, NodeTokenBindingProjection, NodeWorkContract, PlanNode,
        ResolvedTensorLayout, ResolvedTensorSpec, ResolvedValueRole, ResolvedValueStorage,
    };

    #[test]
    fn token_witness_uses_immediate_span_and_rejects_capacity_overrun() {
        let value_id = ProgramValueId::new("value/output").unwrap();
        let projection = NodeTokenBindingProjection {
            value_id: value_id.clone(),
            role: ResolvedValueRole::Output,
            ordinal: 0,
            axis: 0,
            rank: 2,
            canonical_extent: 8,
        };
        let mut node = PlanNode::resource_test_node(NodeId::new("node/token-output").unwrap());
        node.work = NodeWorkContract::Tokens {
            source: projection.clone(),
            projections: vec![projection],
        };
        let binding = ResolvedValueBinding::new(
            value_id.clone(),
            ResolvedValueRole::Output,
            0,
            ResolvedTensorSpec::new(
                vec![8, 8],
                ElementType::F16,
                ResolvedTensorLayout::Contiguous,
            )
            .unwrap(),
            TensorAccess::Write,
            AliasPolicy::NoAlias,
            BufferUsage::Activations,
            None,
            ResolvedValueStorage::single(
                ResourceId::new("resource/token-output").unwrap(),
                32,
                128,
                ElementType::F16,
            )
            .unwrap(),
        )
        .unwrap();

        let witnesses = ExecutionDeterminismWitnessSpec::from_binding(
            &node,
            ExecutionDeterminismWitnessKind::Output {
                value_id,
                output_ordinal: 0,
            },
            &binding,
        )
        .unwrap();
        assert_eq!(witnesses.len(), 1);
        let witness = &witnesses[0];
        assert_eq!(witness.canonical_length_bytes(), 128);
        assert_eq!(witness.token_bytes_per_token(), Some(16));
        assert_eq!(
            witness
                .active_length_bytes(
                    &TokenSpanWork::from_token_ids(&[1, 2, 3, 4, 5, 6, 7, 8], 2..5).unwrap()
                )
                .unwrap(),
            48
        );
        assert!(witness
            .active_length_bytes(
                &TokenSpanWork::from_token_ids(&[1, 2, 3, 4, 5, 6, 7, 8, 9], 0..9).unwrap()
            )
            .is_err());
    }

    #[test]
    fn non_contiguous_witness_retains_the_complete_typed_span() {
        let mut node = PlanNode::resource_test_node(NodeId::new("node/strided-output").unwrap());
        node.work = NodeWorkContract::Fixed;
        let value_id = ProgramValueId::new("value/strided-output").unwrap();
        let binding = ResolvedValueBinding::new(
            value_id.clone(),
            ResolvedValueRole::Output,
            0,
            ResolvedTensorSpec::new(
                vec![2, 2],
                ElementType::F16,
                ResolvedTensorLayout::Strided {
                    byte_strides: vec![8, 2],
                },
            )
            .unwrap(),
            TensorAccess::Write,
            AliasPolicy::NoAlias,
            BufferUsage::Activations,
            None,
            ResolvedValueStorage::single(
                ResourceId::new("resource/strided-output").unwrap(),
                16,
                12,
                ElementType::F16,
            )
            .unwrap(),
        )
        .unwrap();

        let witnesses = ExecutionDeterminismWitnessSpec::from_binding(
            &node,
            ExecutionDeterminismWitnessKind::Output {
                value_id,
                output_ordinal: 0,
            },
            &binding,
        )
        .unwrap();
        assert_eq!(witnesses.len(), 1);
        assert_eq!(witnesses[0].logical_offset_bytes(), 16);
        assert_eq!(witnesses[0].canonical_length_bytes(), 12);
        assert_eq!(witnesses[0].element_type(), ElementType::F16);
    }
}
