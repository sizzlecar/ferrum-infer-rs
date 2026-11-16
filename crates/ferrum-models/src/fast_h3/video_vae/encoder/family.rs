use super::*;
pub(super) struct Family {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: FastH3VideoEncoderLimits,
}
impl Family {
    pub const ID: &'static str = "family.fast_h3.video_encoder.causal_f32";
    pub const REFERENCE_PROFILE: &'static str =
        "fast_h3.video_encoder.full_causal.reference_f32.v1";
    pub const CUDA_PROFILE: &'static str = "fast_h3.video_encoder.full_causal.cuda_fma_f32.v1";
    pub fn new(limits: FastH3VideoEncoderLimits) -> Result<Self, VNextError> {
        Ok(Self {
            id: id(Self::ID)?,
            metadata: id("metadata.fast_h3.video_encoder.causal_f32")?,
            limits,
        })
    }
}
impl ModelFamilyProvider for Family {
    type Config = Config;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([self.metadata.clone()])
    }
    fn parse_config(&self, raw: &serde_json::Value) -> Result<Config, VNextError> {
        if serde_json::to_vec(raw).map_err(invalid)?.len() > self.limits.maximum_metadata_bytes {
            return Err(invalid("configuration exceeds metadata capacity"));
        }
        let c = serde_json::from_value(raw.clone()).map_err(invalid)?;
        self.validate_config_identity(raw, &c)?;
        Ok(c)
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        c: &Config,
    ) -> Result<(), VNextError> {
        c.validate(self.limits)?;
        if serde_json::to_value(c).map_err(invalid)? != *raw {
            return Err(invalid("noncanonical encoder configuration"));
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        c: &Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        Ok(self.metadata.clone())
    }
    fn weights(&self, c: &Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::Stored(program::build(c, self.limits)?.schema))
    }
    fn numerical_profiles(&self, c: &Config) -> Result<FamilyNumericalProfiles, VNextError> {
        let g = program::build(c, self.limits)?;
        let boundaries = g
            .program
            .blocks()
            .iter()
            .flat_map(|b| &b.nodes)
            .flat_map(|n| &n.outputs)
            .map(|v| (v.clone(), ElementType::F32))
            .collect::<BTreeMap<_, _>>();
        let operations = [
            F32CnnOp::Conv3d,
            F32CnnOp::FrameGroupNormSilu,
            F32CnnOp::VolumeResidualAdd,
        ]
        .into_iter()
        .map(|op| {
            Ok(NumericalOperationContract {
                operation_id: id(op.operation_id())?,
                version: ContractVersion::new(1, 0),
                multiplication_type: (op != F32CnnOp::VolumeResidualAdd)
                    .then_some(ElementType::F32),
                accumulation_type: Some(ElementType::F32),
                staged_arithmetic: None,
                composite_arithmetic: None,
            })
        })
        .collect::<Result<Vec<_>, VNextError>>()?;
        let profile = |name| {
            Ok(NumericalExecutionProfile {
                id: NumericalProfileId::new(name).map_err(invalid)?,
                family_id: self.id.clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: id(PreparedFastH3VideoEncoder::OUTPUT_ID)?,
                boundaries: boundaries.clone(),
                states: vec![],
                kv_storage: vec![],
                operations: operations.clone(),
            })
        };
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            vec![
                profile(Self::REFERENCE_PROFILE)?,
                profile(Self::CUDA_PROFILE)?,
            ],
            vec![NumericalProfileId::new(Self::REFERENCE_PROFILE).map_err(invalid)?],
        )
    }
    fn semantic_program(
        &self,
        c: &Config,
        profile: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        if self.numerical_profiles(c)?.resolve(&profile.id)? != profile {
            return Err(invalid("unknown causal encoder profile"));
        }
        Ok(program::build(c, self.limits)?.program)
    }
    fn semantic_metadata(&self, _: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
