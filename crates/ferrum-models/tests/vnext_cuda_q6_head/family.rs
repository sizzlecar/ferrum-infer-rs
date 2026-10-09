use super::*;
pub const HIDDEN: u64 = legacy_family::HIDDEN;
pub const OUTPUTS: u64 = 132;
pub const MAX_TOKENS: u64 = 4;
pub const PROFILE: &str = "fixture.q6-head.f32";
const VOCAB: u64 = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "head")]
pub enum Head {
    Q6,
    Dense,
}

pub struct Family {
    id: ModelFamilyId,
    head: Head,
    base: legacy_family::Family,
}
impl Family {
    pub fn new(head: Head) -> Self {
        Self {
            id: id("family.fixture.q6-head"),
            head,
            base: legacy_family::Family::new(AttentionKind::Causal),
        }
    }
    fn states(&self) -> Vec<StateSpec> {
        let mut states = self.base.states();
        for state in &mut states {
            if let StateCapacityDemand::TokenScaled { maximum_tokens, .. } =
                &mut state.capacity_demand
            {
                *maximum_tokens = MAX_TOKENS;
            }
        }
        states
    }
}
impl ModelFamilyProvider for Family {
    type Config = Head;
    fn family_id(&self) -> &ModelFamilyId {
        &self.id
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.fixture.q6-head")])
    }
    fn validate_config_identity(&self, _: &Value, config: &Head) -> Result<(), VNextError> {
        if *config != self.head {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.id.to_string(),
                field: "head".into(),
                reason: "fixture head mismatch".into(),
            });
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Head,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(id("metadata.fixture.q6-head"))
    }
    fn parse_config(&self, raw: &Value) -> Result<Head, VNextError> {
        let value =
            serde_json::from_value(raw.clone()).map_err(|e| VNextError::InvalidModelConfig {
                family_id: self.id.to_string(),
                field: "head".into(),
                reason: e.to_string(),
            })?;
        self.validate_config_identity(raw, &value)?;
        Ok(value)
    }
    fn weight_schema(&self, _: &Head) -> Result<WeightSchema, VNextError> {
        let mut schema = self.base.weight_schema(&AttentionKind::Causal)?;
        schema.layout_id = id("weight-layout.fixture.q6-head");
        schema
            .components
            .iter_mut()
            .find(|c| c.id.as_str() == "component.embedding")
            .unwrap()
            .dimensions[0] = VOCAB;
        schema
            .tensors
            .iter_mut()
            .find(|t| t.id.as_str() == "weight.embedding")
            .unwrap()
            .dimensions[0] = VOCAB;
        let quant = self.head == Head::Q6;
        let cid: WeightId = id("component.head");
        schema.components.push(WeightComponentSpec {
            id: cid.clone(),
            role: if quant {
                WeightComponentRole::PackedValues
            } else {
                WeightComponentRole::Values
            },
            external_names: vec!["head".into()],
            dimensions: vec![OUTPUTS, if quant { HIDDEN / 256 } else { HIDDEN }],
            encoding: if quant {
                WeightEncoding::BlockQuantized(BlockQuantizationSpec {
                    format_id: id("quantization.gguf.q6-k"),
                    logical_values_per_block: 256,
                    bytes_per_block: 210,
                })
            } else {
                WeightEncoding::Dense {
                    element_type: ElementType::F16,
                }
            },
            required: true,
        });
        schema.tensors.push(WeightTensorSpec {
            id: id("weight.head"),
            dimensions: vec![OUTPUTS, HIDDEN],
            logical_element_type: ElementType::F16,
            physical_layout: if quant {
                PhysicalWeightLayout::BlockQuantized {
                    blocks: PhysicalWeightComponentBinding::exact_contiguous(cid),
                    block_axis: 1,
                    block_padding: PhysicalWeightPadding::Exact,
                }
            } else {
                PhysicalWeightLayout::Dense { component_id: cid }
            },
            required: true,
        });
        Ok(schema)
    }
    fn numerical_profiles(&self, _: &Head) -> Result<FamilyNumericalProfiles, VNextError> {
        let catalog = self.base.numerical_profiles(&AttentionKind::Causal)?;
        let mut profile = catalog.resolve(&id(self.base.profile_id()))?.clone();
        profile.id = id(PROFILE);
        profile.family_id = self.id.clone();
        profile.states = self.states();
        profile
            .boundaries
            .insert(id("value.attention"), ElementType::F32);
        profile.operations.push(NumericalOperationContract {
            operation_id: id(LAST_TOKEN_DENSE_LINEAR_Q6_MMQ_F32_OPERATION_ID),
            version: ContractVersion::new(1, 0),
            multiplication_type: None,
            accumulation_type: None,
            staged_arithmetic: Some(Q6MmqF32Policy::new().staged()),
            composite_arithmetic: None,
        });
        FamilyNumericalProfiles::new(&self.id, ContractVersion::new(1, 0), vec![profile], vec![])
    }
    fn semantic_program(
        &self,
        _: &Head,
        profile: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let base = self
            .base
            .semantic_program(&AttentionKind::Causal, profile)?;
        let mut blocks = base.blocks().to_vec();
        let nodes = &mut blocks[0].nodes;
        nodes[0]
            .attributes
            .insert(id("vocab_size"), SemanticValue::Unsigned(VOCAB));
        let attention = nodes
            .iter_mut()
            .find(|n| n.id.as_str() == "node.attention")
            .unwrap();
        attention.outputs = vec![id("value.attention")];
        attention.attributes.insert(
            id("maximum_context_tokens"),
            SemanticValue::Unsigned(MAX_TOKENS),
        );
        nodes.push(ProgramNode {
            id: id("node.head"),
            operation_id: id(LAST_TOKEN_DENSE_LINEAR_Q6_MMQ_F32_OPERATION_ID),
            required_version: ContractVersion::new(1, 0),
            work: ProgramNodeWorkSpec::tokens(id("value.embedding"), 0),
            inputs: vec![id("value.embedding"), id("value.weight.head")],
            outputs: vec![id("value.output")],
            attributes: BTreeMap::from([
                (id("hidden_size"), SemanticValue::Unsigned(HIDDEN)),
                (id("out_features"), SemanticValue::Unsigned(OUTPUTS)),
            ]),
        });
        let mut weights = base.weights().to_vec();
        weights
            .iter_mut()
            .find(|w| w.weight_id.as_str() == "weight.embedding")
            .unwrap()
            .tensor
            .dimensions[0] = VOCAB;
        weights.push(WeightReference {
            weight_id: id("weight.head"),
            value_id: id("value.weight.head"),
            tensor: ProgramTensorSpec {
                dimensions: vec![OUTPUTS, HIDDEN],
                element_type: ElementType::F16,
                layout: ResolvedTensorLayout::Contiguous,
            },
        });
        // The real attention node executes and updates real sequence KV. Its
        // output remains observable; the head deliberately consumes the known
        // embedding so its independent numerical oracle stays exact.
        ModelProgram::new(
            self.id.clone(),
            vec![id("value.tokens")],
            blocks,
            self.states(),
            weights,
            vec![id("value.output"), id("value.attention")],
        )?
        .with_checkpoint_inputs(ProgramCheckpointInputs::new(
            id("value.tokens"),
            BTreeSet::new(),
        )?)
    }
    fn semantic_metadata(&self, _: &Head) -> Result<ModelSemanticMetadata, VNextError> {
        self.base.semantic_metadata(&AttentionKind::Causal)
    }
}

/// Every activation group contains 127/128 and integer multiples of 1/128.
/// This makes D4 codes known independently of the native packer. Weight codes
/// and scales are also exact binary fractions; the F64 dot checks token/row
/// selection and destination extents without approximating a general oracle.
pub fn activation(token: u32, k: usize) -> f32 {
    if k % 32 == 0 {
        127.0 / 128.0
    } else {
        (((token as usize * 31 + k * 17) % 253) as i32 - 126) as f32 / 128.0
    }
}
fn weight_code(n: usize, k: usize) -> i32 {
    ((n * 13 + k * 7 + (k / 256) * 3) % 64) as i32 - 32
}
pub struct Weights {
    values: BTreeMap<WeightId, Vec<u8>>,
    base: legacy_family::Weights,
}
impl Weights {
    pub fn new(schema: &WeightSchema, bad_token: Option<u32>, bad_weight: bool) -> Self {
        let mut values = BTreeMap::new();
        for c in schema
            .components
            .iter()
            .filter(|c| matches!(c.id.as_str(), "component.embedding" | "component.head"))
        {
            let bytes = if c.id.as_str() == "component.embedding" {
                (0..VOCAB as usize * HIDDEN as usize)
                    .flat_map(|i| {
                        let token = (i / HIDDEN as usize) as u32;
                        let value = if bad_token == Some(token) && i % HIDDEN as usize == 0 {
                            f32::NAN
                        } else {
                            activation(token, i % HIDDEN as usize)
                        };
                        f16::from_f32(value).to_le_bytes()
                    })
                    .collect()
            } else if matches!(c.encoding, WeightEncoding::BlockQuantized(_)) {
                let mut bytes = Vec::new();
                for n in 0..OUTPUTS as usize {
                    for block in 0..HIDDEN as usize / 256 {
                        let mut b = [0u8; 210];
                        b[192..208].fill(1);
                        b[208..210].copy_from_slice(
                            &if bad_weight && n == 0 && block == 0 {
                                f16::NAN
                            } else {
                                f16::from_f32(1.0 / 256.0)
                            }
                            .to_le_bytes(),
                        );
                        for k in 0..256 {
                            let q = (weight_code(n, block * 256 + k) + 32) as u8;
                            let half = k / 128;
                            let lane = k % 32;
                            let quarter = (k % 128) / 32;
                            let low = half * 64 + lane + (quarter % 2) * 32;
                            b[low] |= (q & 15) << (4 * (quarter / 2));
                            b[128 + half * 32 + lane] |= (q >> 4) << (2 * quarter);
                        }
                        bytes.extend(b);
                    }
                }
                bytes
            } else {
                (0..OUTPUTS as usize * HIDDEN as usize)
                    .flat_map(|i| {
                        f16::from_f32(
                            weight_code(i / HIDDEN as usize, i % HIDDEN as usize) as f32 / 256.0,
                        )
                        .to_le_bytes()
                    })
                    .collect()
            };
            values.insert(c.id.clone(), bytes);
        }
        let base_schema = legacy_family::Family::new(AttentionKind::Causal)
            .weight_schema(&AttentionKind::Causal)
            .unwrap();
        Self {
            values,
            base: legacy_family::Weights::new(&base_schema),
        }
    }
    pub fn expected(&self, token: u32) -> Vec<f64> {
        (0..OUTPUTS as usize)
            .map(|n| {
                (0..HIDDEN as usize)
                    .map(|k| f64::from(activation(token, k)) * f64::from(weight_code(n, k)) / 256.0)
                    .sum()
            })
            .collect()
    }
}
impl WeightComponentSource for Weights {
    fn component<'s>(
        &'s self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'s>, VNextError> {
        if !self.values.contains_key(&c.id) {
            return self.base.component(c);
        }
        WeightComponentPayload::new(
            c,
            c.external_names[0].clone(),
            "generated-q6-head.bin",
            c.dimensions.clone(),
            c.physical_element_type(),
            self.values[&c.id].as_slice(),
        )
    }
}
