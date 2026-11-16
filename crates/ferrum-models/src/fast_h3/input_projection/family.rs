use super::*;
pub struct FastH3InputProjectionFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: FastH3InputLimits,
}
impl FastH3InputProjectionFamily {
    pub const FAMILY_ID: &'static str = "family.fast_h3.input_projection";
    pub const MEDIA_PROFILE_ID: &'static str =
        "fast_h3.input_projection.media.f32_affine_then_bf16.v2";
    pub const TEXT_PROFILE_ID: &'static str = "fast_h3.input_projection.context.bf16.v1";
    pub const VIDEO_INPUT_ID: &'static str = "value.fast_h3.input_projection.video_f32";
    pub const AUDIO_INPUT_ID: &'static str = "value.fast_h3.input_projection.audio_f32";
    pub const TEXT_INPUT_ID: &'static str = "value.fast_h3.input_projection.text_bf16";
    pub const VIDEO_OUTPUT_ID: &'static str = "value.fast_h3.input_projection.video_bf16";
    pub const AUDIO_OUTPUT_ID: &'static str = "value.fast_h3.input_projection.audio_bf16";
    pub const PROJECTED_TEXT_ID: &'static str =
        "value.fast_h3.input_projection.projected_text_bf16";

    pub fn new(limits: FastH3InputLimits) -> Result<Self, VNextError> {
        limits.validate()?;
        Ok(Self {
            id: id(Self::FAMILY_ID)?,
            metadata: id("metadata.fast_h3.input_projection")?,
            limits,
        })
    }
}
impl ModelFamilyProvider for FastH3InputProjectionFamily {
    type Config = FastH3InputProjectionConfig;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([self.metadata.clone()])
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<(), VNextError> {
        if serde_json::to_value(c).map_err(invalid)? != *raw {
            return Err(invalid("noncanonical input configuration"));
        }
        c.validate(self.limits)
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        Ok(self.metadata.clone())
    }
    fn parse_config(&self, raw: &serde_json::Value) -> Result<Self::Config, VNextError> {
        let c = serde_json::from_value(raw.clone()).map_err(invalid)?;
        self.validate_config_identity(raw, &c)?;
        Ok(c)
    }
    fn weights(&self, c: &Self::Config) -> Result<ModelWeights, VNextError> {
        c.validate(self.limits)?;
        Ok(ModelWeights::Stored(c.schema()?))
    }
    fn numerical_profiles(&self, c: &Self::Config) -> Result<FamilyNumericalProfiles, VNextError> {
        c.validate(self.limits)?;
        profiles(
            &self.id,
            c.phase.profile_id(),
            if c.phase == FastH3InputProjectionPhase::Media {
                Self::VIDEO_OUTPUT_ID
            } else {
                Self::PROJECTED_TEXT_ID
            },
            program::build(c)?,
        )
    }
    fn semantic_program(
        &self,
        c: &Self::Config,
        p: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        if self.numerical_profiles(c)?.resolve(&p.id)? != p {
            return Err(invalid("unknown input profile"));
        }
        Ok((program::build(c)?).0)
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
pub struct FastH3InputPackingFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: FastH3InputLimits,
}
impl FastH3InputPackingFamily {
    pub const FAMILY_ID: &'static str = "family.fast_h3.input_packing";
    pub const PROFILE_ID: &'static str = "fast_h3.input_packing.exact_bf16_modality_rows.v1";
    pub const REFINED_TEXT_ID: &'static str = "value.fast_h3.input_packing.refined_text";
    pub const VIDEO_ID: &'static str = "value.fast_h3.input_packing.video";
    pub const AUDIO_ID: &'static str = "value.fast_h3.input_packing.audio";
    pub const OUTPUT_ID: &'static str = "value.fast_h3.input_packing.packed";

    pub fn new(limits: FastH3InputLimits) -> Result<Self, VNextError> {
        limits.validate()?;
        Ok(Self {
            id: id(Self::FAMILY_ID)?,
            metadata: id("metadata.fast_h3.input_packing")?,
            limits,
        })
    }
}
impl ModelFamilyProvider for FastH3InputPackingFamily {
    type Config = FastH3InputPackingConfig;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([self.metadata.clone()])
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<(), VNextError> {
        if serde_json::to_value(c).map_err(invalid)? != *raw {
            return Err(invalid("noncanonical input configuration"));
        }
        c.validate(self.limits)
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        c: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        Ok(self.metadata.clone())
    }
    fn parse_config(&self, raw: &serde_json::Value) -> Result<Self::Config, VNextError> {
        let c = serde_json::from_value(raw.clone()).map_err(invalid)?;
        self.validate_config_identity(raw, &c)?;
        Ok(c)
    }
    fn weights(&self, c: &Self::Config) -> Result<ModelWeights, VNextError> {
        c.validate(self.limits)?;
        Ok(ModelWeights::None)
    }
    fn numerical_profiles(&self, c: &Self::Config) -> Result<FamilyNumericalProfiles, VNextError> {
        c.validate(self.limits)?;
        profiles(&self.id, Self::PROFILE_ID, Self::OUTPUT_ID, c.program()?)
    }
    fn semantic_program(
        &self,
        c: &Self::Config,
        p: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        if self.numerical_profiles(c)?.resolve(&p.id)? != p {
            return Err(invalid("unknown input profile"));
        }
        Ok((c.program()?).0)
    }
    fn semantic_metadata(&self, _: &Self::Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
fn profiles(
    family: &ModelFamilyId,
    profile: &str,
    output: &str,
    (program, specs): (ModelProgram, BTreeMap<ProgramValueId, ProgramTensorSpec>),
) -> Result<FamilyNumericalProfiles, VNextError> {
    let operations = program
        .blocks()
        .iter()
        .flat_map(|b| &b.nodes)
        .map(|n| n.operation_id.clone())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|operation_id| {
            let (multiplication_type, accumulation_type) = match operation_id.as_str() {
                AFFINE_LINEAR_F32_TO_BF16_OPERATION_ID => {
                    (Some(ElementType::F32), Some(ElementType::F32))
                }
                AFFINE_LINEAR_BF16_OPERATION_ID => {
                    (Some(ElementType::Bf16), Some(ElementType::F32))
                }
                _ => (None, None),
            };
            NumericalOperationContract {
                operation_id,
                version: ContractVersion::new(1, 0),
                multiplication_type,
                accumulation_type,
                staged_arithmetic: None,
                composite_arithmetic: None,
            }
        })
        .collect();
    FamilyNumericalProfiles::new(
        family,
        ContractVersion::new(1, 0),
        vec![NumericalExecutionProfile {
            id: NumericalProfileId::new(profile).map_err(invalid)?,
            family_id: family.clone(),
            version: ContractVersion::new(1, 0),
            primary_activation: id(output)?,
            boundaries: program
                .blocks()
                .iter()
                .flat_map(|b| &b.nodes)
                .flat_map(|n| &n.outputs)
                .map(|v| (v.clone(), specs[v].element_type))
                .collect(),
            states: vec![],
            kv_storage: vec![],
            operations,
        }],
        vec![NumericalProfileId::new(profile).map_err(invalid)?],
    )
}
