use ferrum_interfaces::vnext::*;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct Config;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
struct WrongConfig;

struct Family {
    id: ModelFamilyId,
}

impl ModelFamilyProvider for Family {
    type Config = Config;

    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }

    fn external_metadata_ids(&self) -> std::collections::BTreeSet<ExternalModelMetadataId> {
        std::collections::BTreeSet::new()
    }

    fn validate_config_identity(
        &self,
        _raw: &serde_json::Value,
        _config: &Config,
    ) -> Result<(), VNextError> {
        Ok(())
    }

    fn validated_external_metadata_id(
        &self,
        _raw: &serde_json::Value,
        _config: &Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        unimplemented!()
    }

    fn parse_config(&self, _raw: &serde_json::Value) -> Result<Config, VNextError> {
        unimplemented!()
    }

    fn weight_schema(&self, _config: &Config) -> Result<WeightSchema, VNextError> {
        unimplemented!()
    }

    fn semantic_program(&self, _config: &Config) -> Result<ModelProgram, VNextError> {
        unimplemented!()
    }

    fn semantic_metadata(&self, _config: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        unimplemented!()
    }
}

fn main() {
    let family = Family {
        id: ModelFamilyId::new("family/typed").unwrap(),
    };
    let _ = family.weight_schema(&WrongConfig);
}
