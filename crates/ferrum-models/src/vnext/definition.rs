//! The indexed product source before selecting a numerical execution profile.

use std::collections::BTreeSet;
use std::sync::Arc;

use ferrum_interfaces::vnext::{
    ModelFamilyDefinition, ModelFamilyRegistration, NumericalExecutionPolicy, NumericalProfileId,
    StateCapacityDemand, StateLifetime, WeightComponentSource,
};
use ferrum_types::{FerrumError, ModelCapabilities, Result};

use super::{
    element_type_label, CausalLanguageModelDescriptor, PreparedProductionModel,
    ProductionExecutionKind, ProductionModelSourceBundle,
};

pub struct DefinedProductionModel {
    definition: ModelFamilyDefinition,
    registration: Arc<dyn ModelFamilyRegistration>,
    weights: Arc<dyn WeightComponentSource>,
    descriptor: CausalLanguageModelDescriptor,
    sources: Arc<ProductionModelSourceBundle>,
}

impl std::fmt::Debug for DefinedProductionModel {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DefinedProductionModel")
            .field("definition", &self.definition)
            .field("descriptor", &self.descriptor)
            .field("sources", &self.sources)
            .finish_non_exhaustive()
    }
}

impl DefinedProductionModel {
    pub(super) fn new(
        registration: impl ModelFamilyRegistration + 'static,
        raw: &serde_json::Value,
        weights: impl WeightComponentSource + 'static,
        descriptor: CausalLanguageModelDescriptor,
        sources: Arc<ProductionModelSourceBundle>,
    ) -> Result<Self> {
        let definition = registration
            .define(raw)
            .map_err(|error| FerrumError::model(error.to_string()))?;
        Ok(Self {
            definition,
            registration: Arc::new(registration),
            weights: Arc::new(weights),
            descriptor,
            sources,
        })
    }

    pub fn definition(&self) -> &ModelFamilyDefinition {
        &self.definition
    }
    pub fn sources(&self) -> &Arc<ProductionModelSourceBundle> {
        &self.sources
    }
    pub fn descriptor(&self) -> &CausalLanguageModelDescriptor {
        &self.descriptor
    }

    pub fn product_source_identity(
        &self,
        requested_model: impl Into<String>,
        resolved_model: impl Into<String>,
    ) -> Result<ferrum_interfaces::vnext::ProductModelSourceIdentity> {
        let template = &self.definition.metadata().template;
        self.sources.product_source_identity(
            requested_model,
            resolved_model,
            &template.source_file,
            &template.template,
        )
    }
    pub const fn execution_kind(&self) -> ProductionExecutionKind {
        ProductionExecutionKind::CausalLanguage
    }

    /// This prepares metadata only. Static compilation must still prove the
    /// selected profile against the actual device registries before uploading.
    pub fn prepare(&self, profile: &NumericalProfileId) -> Result<PreparedProductionModel> {
        let family = self
            .registration
            .prepare(&self.definition, profile)
            .map_err(|error| FerrumError::model(error.to_string()))?;
        PreparedProductionModel::new(
            family,
            self.weights.clone(),
            self.descriptor.clone(),
            self.sources.clone(),
        )
    }

    /// Conservative startup facts for the requested candidates. No placeholder
    /// program is constructed, and Require never inherits another profile's
    /// state sizing. Final compilation uses the selected profile's exact state.
    pub fn model_capabilities(
        &self,
        policy: &NumericalExecutionPolicy,
    ) -> Result<ModelCapabilities> {
        let candidates = self
            .definition
            .numerical_profiles()
            .candidates(policy)
            .map_err(|error| FerrumError::model(error.to_string()))?;
        let mut recurrent_bytes = 0_u64;
        let mut supported_dtypes = BTreeSet::new();
        for profile in candidates {
            let bytes = profile
                .states
                .iter()
                .filter(|state| {
                    state.lifetime == StateLifetime::Sequence
                        && state.capacity_demand == StateCapacityDemand::FixedPerScope
                })
                .try_fold(0_u64, |total, state| {
                    let bytes = state
                        .tensor
                        .byte_len()
                        .map_err(|error| FerrumError::model(error.to_string()))?;
                    total.checked_add(bytes).ok_or_else(|| {
                        FerrumError::model("per-sequence numerical state size overflows u64")
                    })
                })?;
            recurrent_bytes = recurrent_bytes.max(bytes);
            supported_dtypes.extend(
                profile
                    .boundaries
                    .values()
                    .copied()
                    .chain(profile.states.iter().map(|state| state.tensor.element_type))
                    .filter_map(element_type_label),
            );
        }
        let formats = self.definition.weight_schema().quantization_formats();
        let weight_bytes = self.sources.weight_payload_bytes()?;
        Ok(ModelCapabilities {
            architecture: self.descriptor.architecture().to_owned(),
            quantization: (!formats.is_empty()).then(|| {
                formats
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()
                    .join("+")
            }),
            moe: self.descriptor.moe.clone(),
            max_context_len: Some(self.descriptor.maximum_sequence_tokens()),
            num_hidden_layers: Some(self.descriptor.layer_count()),
            head_dim: Some(self.descriptor.attention_head_dimension()),
            kv_heads: Some(self.descriptor.kv_head_count()),
            estimated_weight_bytes: (weight_bytes > 0).then_some(weight_bytes),
            recurrent_state_bytes_per_sequence: (recurrent_bytes > 0).then_some(recurrent_bytes),
            supported_dtypes: supported_dtypes.into_iter().collect(),
            graph_safe_moe: false,
        })
    }
}
