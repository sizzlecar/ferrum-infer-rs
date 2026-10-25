use super::*;

/// Exposes the existing family graph's numerical boundaries for diagnosis.
/// Actual product graphs are executed separately and must produce identical
/// final output with intermediate storage reclaimed normally.
pub(super) struct Inspect<P>(pub P, pub bool);
impl<P: ModelFamilyProvider> ModelFamilyProvider for Inspect<P> {
    type Config = P::Config;
    fn family_id(&self) -> &ModelFamilyId {
        self.0.family_id()
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        self.0.external_metadata_ids()
    }
    fn validate_config_identity(&self, raw: &Value, config: &P::Config) -> Result<(), VNextError> {
        self.0.validate_config_identity(raw, config)
    }
    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &P::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.0.validated_external_metadata_id(raw, config)
    }
    fn parse_config(&self, raw: &Value) -> Result<P::Config, VNextError> {
        self.0.parse_config(raw)
    }
    fn weights(&self, config: &P::Config) -> Result<ModelWeights, VNextError> {
        self.0.weights(config)
    }
    fn numerical_profiles(
        &self,
        config: &P::Config,
    ) -> Result<FamilyNumericalProfiles, VNextError> {
        self.0.numerical_profiles(config)
    }
    fn specialize_weight_schema(
        &self,
        config: &P::Config,
        source: &WeightSchema,
        profile: &NumericalExecutionProfile,
    ) -> Result<WeightSchema, VNextError> {
        self.0.specialize_weight_schema(config, source, profile)
    }
    fn semantic_metadata(&self, config: &P::Config) -> Result<ModelSemanticMetadata, VNextError> {
        self.0.semantic_metadata(config)
    }
    fn semantic_program(
        &self,
        config: &P::Config,
        profile: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let graph = self.0.semantic_program(config, profile)?;
        if !self.1 {
            return Ok(graph);
        }
        assert!(graph.states().is_empty());
        ModelProgram::new(
            graph.family_id().clone(),
            graph.inputs().to_vec(),
            graph.blocks().to_vec(),
            vec![],
            graph.weights().to_vec(),
            graph
                .blocks()
                .iter()
                .flat_map(|b| &b.nodes)
                .flat_map(|n| n.outputs.clone())
                .collect(),
        )?
        .with_input_tensor_specs(graph.input_tensor_specs().clone())
    }
}

#[derive(Clone)]
pub(super) struct Data {
    pub spec: ProgramTensorSpec,
    pub values: Vec<f32>,
}
pub(super) struct Source {
    pub data: BTreeMap<ProgramValueId, Data>,
    bytes: BTreeMap<WeightId, Vec<u8>>,
}
impl Source {
    pub fn new(prepared: &PreparedModelFamily) -> Self {
        let schema = prepared.stored_weight_schema().unwrap();
        let mut data = BTreeMap::new();
        let mut bytes = BTreeMap::new();
        for (seed, weight) in prepared.program().weights().iter().enumerate() {
            let components = schema.physical_component_refs(&weight.weight_id).unwrap();
            let [component] = components.as_slice() else {
                panic!("fixture uses one physical component per weight");
            };
            let count = weight.tensor.dimensions.iter().product::<u64>() as usize;
            let (values, payload) = match &component.encoding {
                WeightEncoding::BlockQuantized(block) => {
                    assert_eq!(block.format_id.as_str(), "quantization.gguf.q8-0");
                    let mut payload = Vec::new();
                    let mut values = Vec::new();
                    for start in (0..count).step_by(32) {
                        let scale = half::f16::from_f32(1.0 / 256.0);
                        payload.extend(scale.to_bits().to_le_bytes());
                        for i in start..start + 32 {
                            let q = ((i * 13 + seed * 7) % 31) as i8 - 15;
                            payload.push(q as u8);
                            values.push(rounds(scale.to_f32() * q as f32));
                        }
                    }
                    (values, payload)
                }
                WeightEncoding::Dense {
                    element_type: ElementType::Bf16,
                } => {
                    let values: Vec<_> = (0..count)
                        .map(|i| {
                            if weight.tensor.dimensions.len() == 1 {
                                rounds(0.75 + (i % 7) as f32 / 16.0)
                            } else {
                                rounds((((i * 13 + seed * 7) % 31) as f32 - 15.0) / 256.0)
                            }
                        })
                        .collect();
                    let payload = values
                        .iter()
                        .flat_map(|v| bf16::from_f32(*v).to_bits().to_le_bytes())
                        .collect();
                    (values, payload)
                }
                other => panic!("unsupported test source {other:?}"),
            };
            bytes.insert(component.id.clone(), payload);
            data.insert(
                weight.value_id.clone(),
                Data {
                    spec: weight.tensor.clone(),
                    values,
                },
            );
        }
        Self { data, bytes }
    }
}
impl WeightComponentSource for Source {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        WeightComponentPayload::new(
            c,
            &c.external_names[0],
            "generated-small-qwen-graph",
            c.dimensions.clone(),
            c.physical_element_type(),
            self.bytes[&c.id].as_slice(),
        )
    }
}

pub(super) fn inputs(
    specs: &BTreeMap<ProgramValueId, ProgramTensorSpec>,
) -> BTreeMap<ProgramValueId, Data> {
    specs
        .iter()
        .enumerate()
        .map(|(port, (name, spec))| {
            let values = (0..spec.dimensions.iter().product::<u64>())
                .map(|i| {
                    let x = ((i * 13 + port as u64 * 19) % 43) as f32 / 32.0 - 0.5;
                    if spec.element_type == ElementType::Bf16 {
                        rounds(x)
                    } else {
                        x
                    }
                })
                .collect();
            (
                name.clone(),
                Data {
                    spec: spec.clone(),
                    values,
                },
            )
        })
        .collect()
}
