use super::*;
use ferrum_quantization::comfy_convrot::{
    ComfyQuantizationMetadata, ComfyQuantizedMatrix, COMFY_CONVROT_CANONICAL_WEIGHT_FORMAT_ID,
};

#[derive(Clone, Debug)]
pub struct PH3ImageTextConditioningFamily {
    id: ModelFamilyId,
    metadata: ExternalModelMetadataId,
    limits: PH3ImageTextConditioningLimits,
}
impl PH3ImageTextConditioningFamily {
    pub const FAMILY_ID: &'static str = "family.p_h3_image.text_conditioning.encoder";
    pub const CUDA_PROFILE_ID: &'static str =
        "p_h3_image.text_conditioning.encoder.cuda.bf16_embedding_f32_decoder.five_taps.v1";
    pub const CUDA_VISION_PROFILE_ID: &'static str =
        "p_h3_image.text_conditioning.encoder.cuda.bf16_embedding_visual_f32_decoder.five_taps.v1";
    pub fn cuda_profile_id(config: &PH3ImageTextConditioningConfig) -> &'static str {
        if config.encoder().visual_inputs().is_some() {
            Self::CUDA_VISION_PROFILE_ID
        } else {
            Self::CUDA_PROFILE_ID
        }
    }
    pub const TOKEN_IDS_ID: &'static str = "value.p_h3_image.text_conditioning.token_ids";
    const TABLE_WEIGHT: &'static str = "weight.p_h3_image.text_conditioning.embedding";
    const SCALE_WEIGHT: &'static str = "weight.p_h3_image.text_conditioning.embedding_scale";
    const TABLE_VALUE: &'static str = "value.p_h3_image.text_conditioning.embedding_table";
    const SCALE_VALUE: &'static str = "value.p_h3_image.text_conditioning.embedding_scale";
    pub fn new(limits: PH3ImageTextConditioningLimits) -> std::result::Result<Self, VNextError> {
        Qwen3VLTextComfyFamily::new(limits.encoder())?;
        Ok(Self {
            id: ModelFamilyId::new(Self::FAMILY_ID)?,
            metadata: ExternalModelMetadataId::new(
                "metadata.p_h3_image.text_conditioning.encoder",
            )?,
            limits,
        })
    }
    fn encoder(&self) -> std::result::Result<Qwen3VLTextComfyFamily, VNextError> {
        Qwen3VLTextComfyFamily::new(self.limits.encoder())
    }
    fn decoder_program(
        &self,
        c: &PH3ImageTextConditioningConfig,
    ) -> std::result::Result<ModelProgram, VNextError> {
        let encoder = self.encoder()?;
        let profiles = encoder.numerical_profiles(c.encoder())?;
        let id = NumericalProfileId::new(Qwen3VLTextComfyFamily::cuda_profile_id(c.encoder()))
            .map_err(invalid)?;
        encoder.semantic_program(c.encoder(), profiles.resolve(&id)?)
    }
    fn schema(
        &self,
        c: &PH3ImageTextConditioningConfig,
    ) -> std::result::Result<WeightSchema, VNextError> {
        let weights = self.encoder()?.weights(c.encoder())?;
        let mut schema = weights.stored_schema()?.clone();
        let a = c.encoder().text().architecture();
        for (external, weight, dimensions, dtype) in [
            (
                Qwen3VLTextEncoderFamily::EMBEDDING_EXTERNAL_NAME.to_owned(),
                Self::TABLE_WEIGHT,
                vec![a.vocabulary_size(), a.hidden_size()],
                ElementType::I8,
            ),
            (
                format!("{}.weight_scale", embedding_module()),
                Self::SCALE_WEIGHT,
                vec![a.vocabulary_size(), 1],
                ElementType::F32,
            ),
        ] {
            let id = WeightId::new(weight)?;
            let component = WeightId::new(format!("component.{weight}"))?;
            schema.components.push(WeightComponentSpec {
                id: component.clone(),
                role: WeightComponentRole::Values,
                external_names: vec![external],
                dimensions: dimensions.clone(),
                encoding: WeightEncoding::Dense {
                    element_type: dtype,
                },
                required: true,
            });
            schema.tensors.push(WeightTensorSpec {
                id,
                dimensions,
                logical_element_type: dtype,
                physical_layout: PhysicalWeightLayout::Dense {
                    component_id: component,
                },
                required: true,
            });
        }
        let bytes = schema.components.iter().try_fold(0u64, |sum, c| {
            let WeightEncoding::Dense { element_type } = &c.encoding else {
                return Err(invalid(
                    "composition requires exact dense physical component ports",
                ));
            };
            let bytes = ProgramTensorSpec {
                dimensions: c.dimensions.clone(),
                element_type: *element_type,
                layout: ResolvedTensorLayout::Contiguous,
            }
            .byte_len()?;
            sum.checked_add(bytes)
                .filter(|n| *n <= self.limits.max_stored_bytes)
                .ok_or_else(|| invalid("combined encoder and embedding exceed Stored capacity"))
        })?;
        if bytes == 0 {
            return Err(invalid("empty Stored inventory"));
        }
        schema.format_id = WeightFormatId::new(COMFY_CONVROT_CANONICAL_WEIGHT_FORMAT_ID)?;
        schema.layout_id =
            WeightLayoutId::new("weight-layout.p_h3_image.text_conditioning.encoder.comfy.v1")?;
        schema.validate(&self.id)?;
        Ok(schema)
    }
    fn program(
        &self,
        c: &PH3ImageTextConditioningConfig,
    ) -> std::result::Result<ModelProgram, VNextError> {
        let decoder = self.decoder_program(c)?;
        let a = c.encoder().text().architecture();
        let mut weights = decoder.weights().to_vec();
        for (weight_id, value_id, dimensions, element_type) in [
            (
                Self::TABLE_WEIGHT,
                Self::TABLE_VALUE,
                vec![a.vocabulary_size(), a.hidden_size()],
                ElementType::I8,
            ),
            (
                Self::SCALE_WEIGHT,
                Self::SCALE_VALUE,
                vec![a.vocabulary_size(), 1],
                ElementType::F32,
            ),
        ] {
            weights.push(WeightReference {
                weight_id: WeightId::new(weight_id)?,
                value_id: ProgramValueId::new(value_id)?,
                tensor: ProgramTensorSpec {
                    dimensions,
                    element_type,
                    layout: ResolvedTensorLayout::Contiguous,
                },
            });
        }
        let mut blocks = vec![ProgramBlock {
            id: "block.p_h3_image.text_conditioning.embedding".into(),
            nodes: vec![ProgramNode {
                id: NodeId::new("node.p_h3_image.text_conditioning.embedding")?,
                operation_id: OperationId::new(COMFY_CONVROT_INT8_EMBEDDING_F32_OPERATION_ID)?,
                required_version: ContractVersion::new(1, 0),
                work: ProgramNodeWorkSpec::Fixed,
                inputs: vec![
                    ProgramValueId::new(Self::TOKEN_IDS_ID)?,
                    ProgramValueId::new(Self::TABLE_VALUE)?,
                    ProgramValueId::new(Self::SCALE_VALUE)?,
                ],
                outputs: vec![ProgramValueId::new(Qwen3VLTextComfyFamily::EMBEDDINGS_ID)?],
                attributes: BTreeMap::from([
                    (
                        AttributeId::new("hidden_size")?,
                        SemanticValue::Unsigned(a.hidden_size()),
                    ),
                    (
                        AttributeId::new("vocab_size")?,
                        SemanticValue::Unsigned(a.vocabulary_size()),
                    ),
                    (
                        AttributeId::new("dequantized_dtype")?,
                        SemanticValue::Text(
                            ComfyEmbeddingDequantizedDtype::Bf16ThenF32.as_str().into(),
                        ),
                    ),
                ]),
            }],
        }];
        blocks.extend_from_slice(decoder.blocks());
        if blocks.iter().map(|b| b.nodes.len()).sum::<usize>() > self.limits.max_plan_nodes {
            return Err(invalid("embedding plus decoder exceed plan capacity"));
        }
        let outputs = c
            .encoder()
            .text()
            .hidden_state_indices()
            .iter()
            .map(|&i| Qwen3VLTextComfyFamily::hidden_state_output_id(i))
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let mut inputs = vec![
            ProgramValueId::new(Self::TOKEN_IDS_ID)?,
            ProgramValueId::new(Qwen3VLTextComfyFamily::COSINE_ID)?,
            ProgramValueId::new(Qwen3VLTextComfyFamily::SINE_ID)?,
        ];
        if c.encoder().visual_inputs().is_some() {
            inputs.push(ProgramValueId::new(
                Qwen3VLTextComfyFamily::VISUAL_EMBEDDINGS_ID,
            )?);
            for id in Qwen3VLTextComfyFamily::DEEPSTACK_IDS {
                inputs.push(ProgramValueId::new(id)?);
            }
        }
        ModelProgram::new(self.id.clone(), inputs, blocks, vec![], weights, outputs)?
            .with_input_tensor_specs(c.input_tensor_specs()?)
    }
    fn check(&self, c: &PH3ImageTextConditioningConfig) -> std::result::Result<(), VNextError> {
        if serde_json::to_vec(c)
            .map_err(|e| invalid(e.to_string()))?
            .len()
            > self.limits.max_metadata_bytes
        {
            return Err(invalid("conditioning metadata exceeds capacity"));
        }
        self.schema(c)?;
        self.program(c)?;
        Ok(())
    }
}
impl ModelFamilyProvider for PH3ImageTextConditioningFamily {
    type Config = PH3ImageTextConditioningConfig;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([self.metadata.clone()])
    }
    fn parse_config(
        &self,
        raw: &serde_json::Value,
    ) -> std::result::Result<Self::Config, VNextError> {
        let c = serde_json::from_value(raw.clone()).map_err(|e| invalid(e.to_string()))?;
        self.validate_config_identity(raw, &c)?;
        Ok(c)
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        c: &Self::Config,
    ) -> std::result::Result<(), VNextError> {
        if serde_json::to_value(c).map_err(|e| invalid(e.to_string()))? != *raw {
            return Err(invalid("conditioning config must be canonical"));
        }
        self.check(c)
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        c: &Self::Config,
    ) -> std::result::Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        Ok(self.metadata.clone())
    }
    fn weights(&self, c: &Self::Config) -> std::result::Result<ModelWeights, VNextError> {
        self.check(c)?;
        Ok(ModelWeights::Stored(self.schema(c)?))
    }
    fn numerical_profiles(
        &self,
        c: &Self::Config,
    ) -> std::result::Result<FamilyNumericalProfiles, VNextError> {
        self.check(c)?;
        let encoder = self.encoder()?;
        let profiles = encoder.numerical_profiles(c.encoder())?;
        let id = NumericalProfileId::new(Qwen3VLTextComfyFamily::cuda_profile_id(c.encoder()))
            .map_err(invalid)?;
        let mut p = profiles.resolve(&id)?.clone();
        p.id = NumericalProfileId::new(Self::cuda_profile_id(c)).map_err(invalid)?;
        p.family_id = self.id.clone();
        p.primary_activation = Qwen3VLTextComfyFamily::hidden_state_output_id(
            c.encoder().text().architecture().layers(),
        )?;
        p.boundaries.insert(
            ProgramValueId::new(Qwen3VLTextComfyFamily::EMBEDDINGS_ID)?,
            ElementType::F32,
        );
        p.operations.push(NumericalOperationContract {
            operation_id: OperationId::new(COMFY_CONVROT_INT8_EMBEDDING_F32_OPERATION_ID)?,
            version: ContractVersion::new(1, 0),
            multiplication_type: None,
            accumulation_type: None,
            staged_arithmetic: Some(StagedNumericalArithmetic::comfy_convrot_int8_embedding_f32(
                ComfyEmbeddingDequantizedDtype::Bf16ThenF32,
            )),
            composite_arithmetic: None,
        });
        FamilyNumericalProfiles::new(
            &self.id,
            ContractVersion::new(1, 0),
            vec![p.clone()],
            vec![p.id],
        )
    }
    fn semantic_program(
        &self,
        c: &Self::Config,
        p: &NumericalExecutionProfile,
    ) -> std::result::Result<ModelProgram, VNextError> {
        if self.numerical_profiles(c)?.resolve(&p.id)? != p {
            return Err(invalid("conditioning profile differs"));
        }
        self.program(c)
    }
    fn semantic_metadata(
        &self,
        _: &Self::Config,
    ) -> std::result::Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}

pub(super) fn embedding_module() -> &'static str {
    "model.language_model.embed_tokens"
}

/// Validates the combined original archive without expanding a projection or
/// the embedding table. Static component initialization borrows this source;
/// the caller retains the mmap/provenance owner and existing cache policy.
pub struct PH3ImageTextConditioningCheckpoint<'a> {
    config: &'a PH3ImageTextConditioningConfig,
    source: &'a ComfyCanonicalWeightSource,
}
impl<'a> PH3ImageTextConditioningCheckpoint<'a> {
    pub fn new(
        config: &'a PH3ImageTextConditioningConfig,
        source: &'a ComfyCanonicalWeightSource,
        limits: PH3ImageTextConditioningLimits,
    ) -> Result<Self> {
        PH3ImageTextConditioningFamily::new(limits)?.check(config)?;
        Qwen3VLTextComfyCheckpoint::new(config.encoder(), source, limits.encoder())?;
        let matrix = ComfyQuantizedMatrix::from_archive(source.archive(), embedding_module())?;
        let a = config.encoder().text().architecture();
        ensure!(
            matrix.metadata()
                == &(ComfyQuantizationMetadata::Int8Tensorwise {
                    convrot: true,
                    convrot_groupsize: 256
                })
                && matrix.rows() as u64 == a.vocabulary_size()
                && matrix.columns() as u64 == a.hidden_size(),
            "embedding metadata or shape differs from the composed encoder"
        );
        Ok(Self { config, source })
    }
    pub fn config(&self) -> &'a PH3ImageTextConditioningConfig {
        self.config
    }
    pub fn source(&self) -> &'a ComfyCanonicalWeightSource {
        self.source
    }
}
impl WeightComponentSource for PH3ImageTextConditioningCheckpoint<'_> {
    fn component<'source>(
        &'source self,
        component: &WeightComponentSpec,
    ) -> std::result::Result<WeightComponentPayload<'source>, VNextError> {
        self.source.component(component)
    }
}
