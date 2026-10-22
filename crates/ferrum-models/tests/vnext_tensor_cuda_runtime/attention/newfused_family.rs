use super::*;

pub(super) struct FusedFamily;

impl ModelFamilyProvider for FusedFamily {
    type Config = AttentionConfig;
    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.cuda.tensor.fused_attention.fixture"))
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.cuda.tensor.fused_attention.fixture")])
    }
    fn validate_config_identity(
        &self,
        raw: &Value,
        config: &AttentionConfig,
    ) -> Result<(), VNextError> {
        // Reuse the existing typed structured-mask/shape validation, while the
        // new family, operation and numerical profile remain separate identities.
        AttentionFamily.validate_config_identity(raw, config)
    }
    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &AttentionConfig,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(id("metadata.cuda.tensor.fused_attention.fixture"))
    }
    fn parse_config(&self, raw: &Value) -> Result<AttentionConfig, VNextError> {
        AttentionFamily.parse_config(raw)
    }
    fn weights(&self, _: &AttentionConfig) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::None)
    }
    fn numerical_profiles(
        &self,
        _: &AttentionConfig,
    ) -> Result<FamilyNumericalProfiles, VNextError> {
        let profile = NumericalProfileId::new(PROFILE).unwrap();
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: profile.clone(),
                family_id: self.family_id().clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: id(OUTPUT_ID),
                boundaries: BTreeMap::from([(id(OUTPUT_ID), ElementType::Bf16)]),
                kv_storage: vec![],
                states: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(STRUCTURED_ATTENTION_FUSED_BF16_OPERATION_ID),
                    version: ContractVersion::new(1, 0),
                    // Vendor SDPA does not promise the old scalar FP32 order
                    // or expose every intermediate storage precision.
                    multiplication_type: None,
                    accumulation_type: None,
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                }],
            }],
            vec![profile],
        )
    }
    fn semantic_program(
        &self,
        config: &AttentionConfig,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        ModelProgram::new(
            self.family_id().clone(),
            INPUTS.into_iter().map(id).collect(),
            vec![ProgramBlock {
                id: "block.tensor.fused_attention".into(),
                nodes: vec![ProgramNode {
                    id: id("node.tensor.fused_attention"),
                    operation_id: id(STRUCTURED_ATTENTION_FUSED_BF16_OPERATION_ID),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: INPUTS.into_iter().map(id).collect(),
                    outputs: vec![id(OUTPUT_ID)],
                    attributes: config.attributes.clone(),
                }],
            }],
            vec![],
            vec![],
            vec![id(OUTPUT_ID)],
        )
    }
    fn semantic_metadata(&self, _: &AttentionConfig) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
