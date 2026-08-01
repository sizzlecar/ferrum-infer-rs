use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::is_canonical_sha256;
use crate::vnext::{
    CapabilityCatalog, ContractVersion, DeviceId, ExternalModelMetadataId, NodeId, OperationId,
    PlanHash, ProviderExecutionContractFingerprint, ProviderId, ProviderReplayEquivalence,
    ResolvedModelPlan, VNextError,
};

use super::{invalid_plan, ExecutionDeterminismWitnessPlan};

pub const EXECUTION_DETERMINISM_COVERAGE_VERSION: ContractVersion = ContractVersion::new(1, 0);
pub const EXECUTION_DETERMINISM_EVIDENCE_DENOMINATOR_VERSION: ContractVersion =
    ContractVersion::new(1, 0);

const MAX_COVERAGE_WIRE_BYTES: usize = 16 * 1024 * 1024;
const MAX_EVIDENCE_DENOMINATOR_WIRE_BYTES: usize = 128 * 1024 * 1024;
const MAX_COVERAGE_MODELS: usize = 32;
const MAX_COVERAGE_PROVIDERS: usize = 512;
const MAX_COVERAGE_NODES_PER_MODEL: usize = 65_536;
const MAX_MODEL_KEY_BYTES: usize = 96;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ExecutionDeterminismComparisonKind {
    EagerEager,
    ReplayReplay,
    EagerReplay,
}

impl ExecutionDeterminismComparisonKind {
    fn for_replay_equivalence(
        replay_equivalence: ProviderReplayEquivalence,
    ) -> Vec<ExecutionDeterminismComparisonKind> {
        let mut comparisons = vec![Self::EagerEager];
        if replay_equivalence == ProviderReplayEquivalence::BitwiseEagerEquivalent {
            comparisons.extend([Self::ReplayReplay, Self::EagerReplay]);
        }
        comparisons
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionDeterminismModelPlanIdentity {
    model_key: String,
    external_metadata_id: ExternalModelMetadataId,
    resolved_plan_fingerprint: String,
    plan_hash: PlanHash,
    node_ids: Vec<NodeId>,
}

impl ExecutionDeterminismModelPlanIdentity {
    pub fn model_key(&self) -> &str {
        &self.model_key
    }

    pub fn external_metadata_id(&self) -> &ExternalModelMetadataId {
        &self.external_metadata_id
    }

    pub fn resolved_plan_fingerprint(&self) -> &str {
        &self.resolved_plan_fingerprint
    }

    pub fn plan_hash(&self) -> &PlanHash {
        &self.plan_hash
    }

    pub fn node_ids(&self) -> &[NodeId] {
        &self.node_ids
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionDeterminismModelProviderSelection {
    model_key: String,
    resolved_plan_fingerprint: String,
    plan_hash: PlanHash,
    node_ids: Vec<NodeId>,
}

impl ExecutionDeterminismModelProviderSelection {
    pub fn model_key(&self) -> &str {
        &self.model_key
    }

    pub fn resolved_plan_fingerprint(&self) -> &str {
        &self.resolved_plan_fingerprint
    }

    pub fn plan_hash(&self) -> &PlanHash {
        &self.plan_hash
    }

    pub fn node_ids(&self) -> &[NodeId] {
        &self.node_ids
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionDeterminismCatalogProviderRequirement {
    operation_id: OperationId,
    operation_version: ContractVersion,
    operation_fingerprint: String,
    provider_id: ProviderId,
    provider_version: ContractVersion,
    provider_implementation_fingerprint: String,
    provider_execution_contract_fingerprint: ProviderExecutionContractFingerprint,
    replay_equivalence: ProviderReplayEquivalence,
    required_comparisons: Vec<ExecutionDeterminismComparisonKind>,
    model_selections: Vec<ExecutionDeterminismModelProviderSelection>,
}

impl ExecutionDeterminismCatalogProviderRequirement {
    pub fn operation_id(&self) -> &OperationId {
        &self.operation_id
    }

    pub const fn operation_version(&self) -> ContractVersion {
        self.operation_version
    }

    pub fn operation_fingerprint(&self) -> &str {
        &self.operation_fingerprint
    }

    pub fn provider_id(&self) -> &ProviderId {
        &self.provider_id
    }

    pub const fn provider_version(&self) -> ContractVersion {
        self.provider_version
    }

    pub fn provider_implementation_fingerprint(&self) -> &str {
        &self.provider_implementation_fingerprint
    }

    pub const fn provider_execution_contract_fingerprint(
        &self,
    ) -> ProviderExecutionContractFingerprint {
        self.provider_execution_contract_fingerprint
    }

    pub const fn replay_equivalence(&self) -> ProviderReplayEquivalence {
        self.replay_equivalence
    }

    pub fn required_comparisons(&self) -> &[ExecutionDeterminismComparisonKind] {
        &self.required_comparisons
    }

    pub fn model_selections(&self) -> &[ExecutionDeterminismModelProviderSelection] {
        &self.model_selections
    }

    fn canonical_key(&self) -> (&OperationId, &ProviderId) {
        (&self.operation_id, &self.provider_id)
    }
}

/// Runtime-derived proof denominator for one CUDA catalog and every resolved
/// model plan admitted to the release matrix.
///
/// Providers with no selected model nodes remain present with an empty
/// `model_selections` list. The hardware validator must reject that gap rather
/// than silently shrinking the denominator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionDeterminismCoverageRegistry {
    schema_version: ContractVersion,
    device_id: DeviceId,
    device_runtime_implementation_fingerprint: String,
    capability_catalog_fingerprint: String,
    models: Vec<ExecutionDeterminismModelPlanIdentity>,
    provider_requirements: Vec<ExecutionDeterminismCatalogProviderRequirement>,
}

impl ExecutionDeterminismCoverageRegistry {
    pub fn from_catalog(catalog: &CapabilityCatalog) -> Result<Self, VNextError> {
        let mut provider_requirements = Vec::new();
        for (operation_id, providers) in catalog.providers() {
            let operation = catalog.operations().get(operation_id).ok_or_else(|| {
                invalid_plan(format!(
                    "determinism coverage catalog provider row `{operation_id}` has no operation"
                ))
            })?;
            let operation_fingerprint = operation.fingerprint()?;
            for provider in providers {
                let semantics = provider.execution_semantics();
                provider_requirements.push(ExecutionDeterminismCatalogProviderRequirement {
                    operation_id: operation_id.clone(),
                    operation_version: operation.version,
                    operation_fingerprint: operation_fingerprint.clone(),
                    provider_id: provider.provider_id().clone(),
                    provider_version: provider.version(),
                    provider_implementation_fingerprint: provider
                        .provider_implementation_fingerprint()
                        .to_owned(),
                    provider_execution_contract_fingerprint: semantics.contract_fingerprint(),
                    replay_equivalence: semantics.replay_equivalence(),
                    required_comparisons:
                        ExecutionDeterminismComparisonKind::for_replay_equivalence(
                            semantics.replay_equivalence(),
                        ),
                    model_selections: Vec::new(),
                });
            }
        }
        provider_requirements
            .sort_by(|left, right| left.canonical_key().cmp(&right.canonical_key()));
        let registry = Self {
            schema_version: EXECUTION_DETERMINISM_COVERAGE_VERSION,
            device_id: catalog.device().id.clone(),
            device_runtime_implementation_fingerprint: catalog
                .device()
                .runtime_implementation_fingerprint
                .clone(),
            capability_catalog_fingerprint: catalog.fingerprint()?,
            models: Vec::new(),
            provider_requirements,
        };
        registry.validate_shape(false)?;
        Ok(registry)
    }

    pub fn try_add_resolved_model_plan(
        &mut self,
        model_key: impl Into<String>,
        plan: &ResolvedModelPlan,
    ) -> Result<(), VNextError> {
        let model_key = model_key.into();
        validate_model_key(&model_key)?;
        if self.models.len() >= MAX_COVERAGE_MODELS {
            return Err(invalid_plan(
                "execution determinism coverage model count exceeds its bound",
            ));
        }
        if self.models.iter().any(|model| {
            model.model_key == model_key
                || model.external_metadata_id == plan.parts().external_metadata_id
                || model.resolved_plan_fingerprint == plan.fingerprint()
        }) {
            return Err(invalid_plan(
                "execution determinism coverage cannot reuse a model key, metadata identity, or resolved plan",
            ));
        }
        if plan.parts().device.id != self.device_id
            || plan.parts().device.runtime_implementation_fingerprint
                != self.device_runtime_implementation_fingerprint
            || plan.parts().capabilities.fingerprint()? != self.capability_catalog_fingerprint
        {
            return Err(invalid_plan(
                "resolved model plan differs from the determinism coverage catalog or device runtime",
            ));
        }
        let plan_nodes = plan.execution_plan().payload().nodes();
        if plan_nodes.is_empty() || plan_nodes.len() > MAX_COVERAGE_NODES_PER_MODEL {
            return Err(invalid_plan(
                "resolved model plan is empty or exceeds the determinism coverage node bound",
            ));
        }

        let mut selected_nodes = BTreeMap::<(OperationId, ProviderId), Vec<NodeId>>::new();
        for node in plan_nodes {
            let key = (
                node.operation_id().clone(),
                node.selection().selected_provider().clone(),
            );
            let requirement = self
                .provider_requirements
                .binary_search_by(|candidate| candidate.canonical_key().cmp(&(&key.0, &key.1)))
                .ok()
                .and_then(|index| self.provider_requirements.get(index))
                .ok_or_else(|| {
                    invalid_plan(format!(
                        "resolved node `{}` selected provider `{}` absent from the live determinism catalog",
                        node.id(),
                        node.selection().selected_provider()
                    ))
                })?;
            if node.operation_version() != requirement.operation_version
                || node.operation_fingerprint() != requirement.operation_fingerprint
                || node.provider_implementation_fingerprint()
                    != requirement.provider_implementation_fingerprint
                || node.provider_execution_semantics().contract_fingerprint()
                    != requirement.provider_execution_contract_fingerprint
                || node.provider_execution_semantics().replay_equivalence()
                    != requirement.replay_equivalence
            {
                return Err(invalid_plan(format!(
                    "resolved node `{}` differs from its live catalog determinism identity",
                    node.id()
                )));
            }
            selected_nodes
                .entry(key)
                .or_default()
                .push(node.id().clone());
        }

        let identity = ExecutionDeterminismModelPlanIdentity {
            model_key: model_key.clone(),
            external_metadata_id: plan.parts().external_metadata_id.clone(),
            resolved_plan_fingerprint: plan.fingerprint().to_owned(),
            plan_hash: plan.execution_plan().plan_hash().clone(),
            node_ids: plan_nodes.iter().map(|node| node.id().clone()).collect(),
        };
        let mut next_requirements = self.provider_requirements.clone();
        for requirement in &mut next_requirements {
            if let Some(node_ids) = selected_nodes.remove(&(
                requirement.operation_id.clone(),
                requirement.provider_id.clone(),
            )) {
                requirement
                    .model_selections
                    .push(ExecutionDeterminismModelProviderSelection {
                        model_key: model_key.clone(),
                        resolved_plan_fingerprint: identity.resolved_plan_fingerprint.clone(),
                        plan_hash: identity.plan_hash.clone(),
                        node_ids,
                    });
                requirement
                    .model_selections
                    .sort_by(|left, right| left.model_key.cmp(&right.model_key));
            }
        }
        if !selected_nodes.is_empty() {
            return Err(invalid_plan(
                "resolved model plan left unmatched determinism provider selections",
            ));
        }

        let mut next_models = self.models.clone();
        next_models.push(identity);
        next_models.sort_by(|left, right| left.model_key.cmp(&right.model_key));
        let candidate = Self {
            schema_version: self.schema_version,
            device_id: self.device_id.clone(),
            device_runtime_implementation_fingerprint: self
                .device_runtime_implementation_fingerprint
                .clone(),
            capability_catalog_fingerprint: self.capability_catalog_fingerprint.clone(),
            models: next_models,
            provider_requirements: next_requirements,
        };
        candidate.validate_shape(true)?;
        *self = candidate;
        Ok(())
    }

    pub const fn schema_version(&self) -> ContractVersion {
        self.schema_version
    }

    pub fn device_id(&self) -> &DeviceId {
        &self.device_id
    }

    pub fn device_runtime_implementation_fingerprint(&self) -> &str {
        &self.device_runtime_implementation_fingerprint
    }

    pub fn capability_catalog_fingerprint(&self) -> &str {
        &self.capability_catalog_fingerprint
    }

    pub fn models(&self) -> &[ExecutionDeterminismModelPlanIdentity] {
        &self.models
    }

    pub fn provider_requirements(&self) -> &[ExecutionDeterminismCatalogProviderRequirement] {
        &self.provider_requirements
    }

    pub fn unselected_provider_requirements(
        &self,
    ) -> impl Iterator<Item = &ExecutionDeterminismCatalogProviderRequirement> {
        self.provider_requirements
            .iter()
            .filter(|requirement| requirement.model_selections.is_empty())
    }

    pub fn to_json(&self) -> Result<Vec<u8>, VNextError> {
        self.validate_shape(true)?;
        serde_json::to_vec_pretty(self).map_err(|error| VNextError::Serialization {
            context: "serialize execution determinism coverage registry",
            message: error.to_string(),
        })
    }

    pub fn fingerprint(&self) -> Result<String, VNextError> {
        Ok(format!("{:x}", Sha256::digest(self.to_json()?)))
    }

    pub fn decode_untrusted(bytes: &[u8]) -> Result<Self, VNextError> {
        if bytes.len() > MAX_COVERAGE_WIRE_BYTES {
            return Err(invalid_plan(
                "execution determinism coverage registry exceeds its wire bound",
            ));
        }
        let registry =
            serde_json::from_slice::<Self>(bytes).map_err(|error| VNextError::Serialization {
                context: "decode execution determinism coverage registry",
                message: error.to_string(),
            })?;
        registry.validate_shape(true)?;
        Ok(registry)
    }

    fn validate_shape(&self, require_models: bool) -> Result<(), VNextError> {
        if self.schema_version != EXECUTION_DETERMINISM_COVERAGE_VERSION
            || !is_canonical_sha256(&self.device_runtime_implementation_fingerprint)
            || !is_canonical_sha256(&self.capability_catalog_fingerprint)
            || self.provider_requirements.is_empty()
            || self.provider_requirements.len() > MAX_COVERAGE_PROVIDERS
            || self.models.len() > MAX_COVERAGE_MODELS
            || (require_models && self.models.is_empty())
        {
            return Err(invalid_plan(
                "execution determinism coverage registry identity or cardinality is invalid",
            ));
        }
        if self
            .models
            .windows(2)
            .any(|pair| pair[0].model_key >= pair[1].model_key)
            || self
                .provider_requirements
                .windows(2)
                .any(|pair| pair[0].canonical_key() >= pair[1].canonical_key())
        {
            return Err(invalid_plan(
                "execution determinism coverage rows are not canonical and unique",
            ));
        }

        let mut model_nodes = BTreeMap::<&str, BTreeSet<&NodeId>>::new();
        let mut model_identities = BTreeMap::new();
        let mut metadata_ids = BTreeSet::new();
        let mut resolved_fingerprints = BTreeSet::new();
        for model in &self.models {
            validate_model_key(&model.model_key)?;
            if !is_canonical_sha256(&model.resolved_plan_fingerprint)
                || model.node_ids.is_empty()
                || model.node_ids.len() > MAX_COVERAGE_NODES_PER_MODEL
                || model.node_ids.iter().collect::<BTreeSet<_>>().len() != model.node_ids.len()
                || !metadata_ids.insert(&model.external_metadata_id)
                || !resolved_fingerprints.insert(model.resolved_plan_fingerprint.as_str())
            {
                return Err(invalid_plan(
                    "execution determinism model identity or node denominator is invalid",
                ));
            }
            model_nodes.insert(model.model_key.as_str(), BTreeSet::new());
            model_identities.insert(model.model_key.as_str(), model);
        }

        for requirement in &self.provider_requirements {
            let expected_comparisons = ExecutionDeterminismComparisonKind::for_replay_equivalence(
                requirement.replay_equivalence,
            );
            if requirement.operation_version.major == 0
                || requirement.provider_version.major == 0
                || !is_canonical_sha256(&requirement.operation_fingerprint)
                || !is_canonical_sha256(&requirement.provider_implementation_fingerprint)
                || requirement.required_comparisons != expected_comparisons
                || requirement
                    .model_selections
                    .windows(2)
                    .any(|pair| pair[0].model_key >= pair[1].model_key)
            {
                return Err(invalid_plan(
                    "execution determinism provider requirement is invalid",
                ));
            }
            for selection in &requirement.model_selections {
                let model = model_identities
                    .get(selection.model_key.as_str())
                    .ok_or_else(|| {
                        invalid_plan(
                            "execution determinism provider selection references an unknown model",
                        )
                    })?;
                if selection.resolved_plan_fingerprint != model.resolved_plan_fingerprint
                    || selection.plan_hash != model.plan_hash
                    || selection.node_ids.is_empty()
                    || selection.node_ids.iter().collect::<BTreeSet<_>>().len()
                        != selection.node_ids.len()
                {
                    return Err(invalid_plan(
                        "execution determinism provider selection identity is invalid",
                    ));
                }
                let covered = model_nodes
                    .get_mut(selection.model_key.as_str())
                    .expect("validated model coverage denominator exists");
                for node_id in &selection.node_ids {
                    if !covered.insert(node_id) {
                        return Err(invalid_plan(
                            "execution determinism model node is selected by multiple providers",
                        ));
                    }
                }
            }
        }
        for (model_key, covered) in model_nodes {
            let expected = model_identities[model_key]
                .node_ids
                .iter()
                .collect::<BTreeSet<_>>();
            if covered != expected {
                return Err(invalid_plan(
                    "execution determinism provider selections do not cover the resolved plan exactly",
                ));
            }
        }
        Ok(())
    }
}

/// Exact immutable witness denominator for one provider selection in one
/// resolved model plan.
///
/// The duplicated plan/provider identities are intentional. They make a
/// detached hardware artifact self-describing while `validate_shape` proves
/// that every row is still derived from the nested live coverage registry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionDeterminismProviderEvidenceDenominator {
    model_key: String,
    resolved_plan_fingerprint: String,
    plan_hash: PlanHash,
    operation_id: OperationId,
    operation_fingerprint: String,
    provider_id: ProviderId,
    provider_implementation_fingerprint: String,
    provider_execution_contract_fingerprint: ProviderExecutionContractFingerprint,
    replay_equivalence: ProviderReplayEquivalence,
    required_comparisons: Vec<ExecutionDeterminismComparisonKind>,
    node_ids: Vec<NodeId>,
    witness_plan_fingerprint: String,
    witness_plan: ExecutionDeterminismWitnessPlan,
}

impl ExecutionDeterminismProviderEvidenceDenominator {
    pub fn model_key(&self) -> &str {
        &self.model_key
    }

    pub fn operation_id(&self) -> &OperationId {
        &self.operation_id
    }

    pub fn provider_id(&self) -> &ProviderId {
        &self.provider_id
    }

    pub fn node_ids(&self) -> &[NodeId] {
        &self.node_ids
    }

    pub fn required_comparisons(&self) -> &[ExecutionDeterminismComparisonKind] {
        &self.required_comparisons
    }

    pub fn witness_plan_fingerprint(&self) -> &str {
        &self.witness_plan_fingerprint
    }

    pub fn witness_plan(&self) -> &ExecutionDeterminismWitnessPlan {
        &self.witness_plan
    }

    fn canonical_key(&self) -> (&str, &OperationId, &ProviderId) {
        (&self.model_key, &self.operation_id, &self.provider_id)
    }
}

/// Current-binary source of truth consumed by the CUDA determinism collector.
///
/// Construction requires the live capability catalog and every resolved model
/// plan in the release matrix in the same process. This prevents Python gate
/// code or an artifact author from inventing provider counts, node scopes, or
/// output/state witness ranges.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ExecutionDeterminismEvidenceDenominator {
    schema_version: ContractVersion,
    coverage: ExecutionDeterminismCoverageRegistry,
    provider_evidence: Vec<ExecutionDeterminismProviderEvidenceDenominator>,
}

impl ExecutionDeterminismEvidenceDenominator {
    pub fn from_catalog_and_resolved_plans(
        catalog: &CapabilityCatalog,
        plans: &[(&str, &ResolvedModelPlan)],
    ) -> Result<Self, VNextError> {
        if plans.is_empty() || plans.len() > MAX_COVERAGE_MODELS {
            return Err(invalid_plan(
                "execution determinism evidence model denominator is empty or exceeds its bound",
            ));
        }
        let mut plan_by_key = BTreeMap::new();
        let mut coverage = ExecutionDeterminismCoverageRegistry::from_catalog(catalog)?;
        for (model_key, plan) in plans {
            validate_model_key(model_key)?;
            if plan_by_key.insert((*model_key).to_owned(), *plan).is_some() {
                return Err(invalid_plan(
                    "execution determinism evidence model keys must be unique",
                ));
            }
            coverage.try_add_resolved_model_plan(*model_key, plan)?;
        }
        if coverage.unselected_provider_requirements().next().is_some() {
            return Err(invalid_plan(
                "execution determinism live catalog contains a provider absent from all resolved plans",
            ));
        }

        let mut provider_evidence = Vec::new();
        for requirement in &coverage.provider_requirements {
            for selection in &requirement.model_selections {
                let plan = plan_by_key
                    .get(selection.model_key.as_str())
                    .expect("coverage model came from the exact plan map");
                let witness_plan = plan
                    .execution_plan()
                    .determinism_witness_plan_for_nodes(&selection.node_ids)?;
                let witness_plan_fingerprint = witness_plan.fingerprint()?;
                provider_evidence.push(ExecutionDeterminismProviderEvidenceDenominator {
                    model_key: selection.model_key.clone(),
                    resolved_plan_fingerprint: selection.resolved_plan_fingerprint.clone(),
                    plan_hash: selection.plan_hash.clone(),
                    operation_id: requirement.operation_id.clone(),
                    operation_fingerprint: requirement.operation_fingerprint.clone(),
                    provider_id: requirement.provider_id.clone(),
                    provider_implementation_fingerprint: requirement
                        .provider_implementation_fingerprint
                        .clone(),
                    provider_execution_contract_fingerprint: requirement
                        .provider_execution_contract_fingerprint,
                    replay_equivalence: requirement.replay_equivalence,
                    required_comparisons: requirement.required_comparisons.clone(),
                    node_ids: selection.node_ids.clone(),
                    witness_plan_fingerprint,
                    witness_plan,
                });
            }
        }
        provider_evidence.sort_by(|left, right| left.canonical_key().cmp(&right.canonical_key()));
        let denominator = Self {
            schema_version: EXECUTION_DETERMINISM_EVIDENCE_DENOMINATOR_VERSION,
            coverage,
            provider_evidence,
        };
        denominator.validate_shape()?;
        Ok(denominator)
    }

    pub const fn schema_version(&self) -> ContractVersion {
        self.schema_version
    }

    pub fn coverage(&self) -> &ExecutionDeterminismCoverageRegistry {
        &self.coverage
    }

    pub fn provider_evidence(&self) -> &[ExecutionDeterminismProviderEvidenceDenominator] {
        &self.provider_evidence
    }

    pub fn to_json(&self) -> Result<Vec<u8>, VNextError> {
        self.validate_shape()?;
        serde_json::to_vec_pretty(self).map_err(|error| VNextError::Serialization {
            context: "serialize execution determinism evidence denominator",
            message: error.to_string(),
        })
    }

    pub fn fingerprint(&self) -> Result<String, VNextError> {
        Ok(format!("{:x}", Sha256::digest(self.to_json()?)))
    }

    pub fn decode_untrusted(bytes: &[u8]) -> Result<Self, VNextError> {
        if bytes.len() > MAX_EVIDENCE_DENOMINATOR_WIRE_BYTES {
            return Err(invalid_plan(
                "execution determinism evidence denominator exceeds its wire bound",
            ));
        }
        let denominator =
            serde_json::from_slice::<Self>(bytes).map_err(|error| VNextError::Serialization {
                context: "decode execution determinism evidence denominator",
                message: error.to_string(),
            })?;
        denominator.validate_shape()?;
        Ok(denominator)
    }

    fn validate_shape(&self) -> Result<(), VNextError> {
        self.coverage.validate_shape(true)?;
        if self.schema_version != EXECUTION_DETERMINISM_EVIDENCE_DENOMINATOR_VERSION
            || self.provider_evidence.is_empty()
            || self.provider_evidence.len()
                > MAX_COVERAGE_MODELS.saturating_mul(MAX_COVERAGE_PROVIDERS)
            || self
                .coverage
                .unselected_provider_requirements()
                .next()
                .is_some()
            || self
                .provider_evidence
                .windows(2)
                .any(|pair| pair[0].canonical_key() >= pair[1].canonical_key())
        {
            return Err(invalid_plan(
                "execution determinism evidence denominator identity or cardinality is invalid",
            ));
        }

        let requirements = self
            .coverage
            .provider_requirements
            .iter()
            .map(|requirement| (requirement.canonical_key(), requirement))
            .collect::<BTreeMap<_, _>>();
        let models = self
            .coverage
            .models
            .iter()
            .map(|model| (model.model_key.as_str(), model))
            .collect::<BTreeMap<_, _>>();
        let mut expected = BTreeSet::new();
        for requirement in &self.coverage.provider_requirements {
            for selection in &requirement.model_selections {
                expected.insert((
                    selection.model_key.as_str(),
                    &requirement.operation_id,
                    &requirement.provider_id,
                ));
            }
        }
        let actual = self
            .provider_evidence
            .iter()
            .map(ExecutionDeterminismProviderEvidenceDenominator::canonical_key)
            .collect::<BTreeSet<_>>();
        if actual != expected {
            return Err(invalid_plan(
                "execution determinism provider evidence does not equal the live plan denominator",
            ));
        }

        for evidence in &self.provider_evidence {
            let requirement = requirements
                .get(&(&evidence.operation_id, &evidence.provider_id))
                .expect("validated evidence key exists in coverage");
            let selection = requirement
                .model_selections
                .iter()
                .find(|selection| selection.model_key == evidence.model_key)
                .expect("validated evidence key has a model selection");
            let model = models
                .get(evidence.model_key.as_str())
                .expect("validated evidence model exists in coverage");
            evidence.witness_plan.validate_shape()?;
            if evidence.resolved_plan_fingerprint != selection.resolved_plan_fingerprint
                || evidence.resolved_plan_fingerprint != model.resolved_plan_fingerprint
                || evidence.plan_hash != selection.plan_hash
                || evidence.plan_hash != model.plan_hash
                || evidence.operation_fingerprint != requirement.operation_fingerprint
                || evidence.provider_implementation_fingerprint
                    != requirement.provider_implementation_fingerprint
                || evidence.provider_execution_contract_fingerprint
                    != requirement.provider_execution_contract_fingerprint
                || evidence.replay_equivalence != requirement.replay_equivalence
                || evidence.required_comparisons != requirement.required_comparisons
                || evidence.node_ids != selection.node_ids
                || evidence.witness_plan.plan_hash() != &evidence.plan_hash
                || evidence.witness_plan.node_ids() != evidence.node_ids
                || evidence.witness_plan.fingerprint()? != evidence.witness_plan_fingerprint
                || evidence.witness_plan.witnesses().iter().any(|witness| {
                    witness.provider_id() != &evidence.provider_id
                        || witness.provider_implementation_fingerprint()
                            != evidence.provider_implementation_fingerprint
                        || witness.provider_execution_contract_fingerprint()
                            != evidence.provider_execution_contract_fingerprint
                })
            {
                return Err(invalid_plan(
                    "execution determinism provider evidence differs from its live catalog, plan, or witness denominator",
                ));
            }
        }
        Ok(())
    }
}

fn validate_model_key(model_key: &str) -> Result<(), VNextError> {
    if model_key.is_empty()
        || model_key.len() > MAX_MODEL_KEY_BYTES
        || !model_key
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-' | b'/'))
    {
        return Err(invalid_plan(
            "execution determinism model key is empty or non-canonical",
        ));
    }
    Ok(())
}
