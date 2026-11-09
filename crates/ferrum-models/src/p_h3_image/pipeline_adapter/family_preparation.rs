//! Opaque typed access to the existing DiT family provider. The image engine
//! can use its original prepared-family cache without exposing internal block
//! configuration or installing a second cache in the model adapter.

use super::{PH3ImageDiTConfig, PH3ImageDiTFamily};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// The provider emits components in composition order; core prepared families
/// normalize these inventories by ID. Compare their complete physical fields
/// in the same order without changing the public source/plan fingerprint.
/// DiT physical references are Dense, so no segmented-layout normalization is
/// needed here; a different layout still fails the full-schema comparison.
pub(super) fn physical_inventory_fingerprint(schema: &WeightSchema) -> Result<String, VNextError> {
    let mut ordered = schema.clone();
    ordered.components.sort_by(|a, b| a.id.cmp(&b.id));
    ordered.tensors.sort_by(|a, b| a.id.cmp(&b.id));
    ordered.fingerprint()
}

#[derive(Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct PH3ImageDenoiserFamilyConfig(pub(super) PH3ImageDiTConfig);

#[derive(Clone)]
pub struct PH3ImageDenoiserFamilyProvider(pub(super) PH3ImageDiTFamily);

impl ModelFamilyProvider for PH3ImageDenoiserFamilyProvider {
    type Config = PH3ImageDenoiserFamilyConfig;

    fn family_id(&self) -> &ModelFamilyId {
        self.0.family_id()
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        self.0.external_metadata_ids()
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        config: &Self::Config,
    ) -> Result<(), VNextError> {
        self.0.validate_config_identity(raw, &config.0)
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        config: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.0.validated_external_metadata_id(raw, &config.0)
    }
    fn parse_config(&self, raw: &serde_json::Value) -> Result<Self::Config, VNextError> {
        self.0.parse_config(raw).map(PH3ImageDenoiserFamilyConfig)
    }
    fn weights(&self, config: &Self::Config) -> Result<ModelWeights, VNextError> {
        self.0.weights(&config.0)
    }
    fn numerical_profiles(
        &self,
        config: &Self::Config,
    ) -> Result<FamilyNumericalProfiles, VNextError> {
        self.0.numerical_profiles(&config.0)
    }
    fn specialize_weight_schema(
        &self,
        config: &Self::Config,
        source: &WeightSchema,
        profile: &NumericalExecutionProfile,
    ) -> Result<WeightSchema, VNextError> {
        self.0.specialize_weight_schema(&config.0, source, profile)
    }
    fn semantic_program(
        &self,
        config: &Self::Config,
        profile: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        self.0.semantic_program(&config.0, profile)
    }
    fn semantic_metadata(
        &self,
        config: &Self::Config,
    ) -> Result<ModelSemanticMetadata, VNextError> {
        self.0.semantic_metadata(&config.0)
    }
}
