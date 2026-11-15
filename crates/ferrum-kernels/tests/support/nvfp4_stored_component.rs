//! Stored NVFP4 operands for the same managed tensor executor as the dynamic
//! fixture. Source declarations stay PackedValues/Quantized and Scales/Dense;
//! raw views neither decode nor allocate another resident weight copy.
use super::fixture::{id, Config, Kind, BIAS, CHANNEL, INPUT, OUTPUT, RELATIVE, WEIGHT};
use ferrum_execution::component::{CompiledTensorComponent, TensorComponentPolicy};
use ferrum_runtime::*;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        atomic::{AtomicUsize, Ordering},
        OnceLock,
    },
};

const PROFILE: &str = "test.tensor_component.stored_nvfp4";
const PACKED: &str = "component.test.nvfp4.packed";
const SCALES: &str = "component.test.nvfp4.scales";

fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}

fn schema(config: &Config) -> Result<WeightSchema, VNextError> {
    let geometry = Nvfp4MatrixLayout::new(config.outputs, config.inputs)?;
    let components = vec![
        WeightComponentSpec {
            id: id(PACKED),
            role: WeightComponentRole::PackedValues,
            external_names: vec!["projection._nvfp4_weight".into()],
            dimensions: geometry.packed_shape().to_vec(),
            encoding: WeightEncoding::Quantized(QuantizationSpec::nvfp4_e2m1_e4m3_128x4()),
            required: true,
        },
        WeightComponentSpec {
            id: id(SCALES),
            role: WeightComponentRole::Scales,
            external_names: vec!["projection._nvfp4_weight_scale".into()],
            // Released checkpoints may flatten this physical scale buffer.
            // The operand view reshapes it without modifying the source spec.
            dimensions: vec![geometry.scale_bytes()],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::U8,
            },
            required: true,
        },
    ];
    Ok(WeightSchema {
        format_id: id("weight-format.safetensors.dense"),
        layout_id: id("weight-layout.test.nvfp4-source"),
        version: ContractVersion::new(1, 0),
        tensors: [
            (WEIGHT, PACKED, geometry.packed_shape()),
            (CHANNEL, SCALES, geometry.scale_shape()),
        ]
        .into_iter()
        .map(|(name, component, shape)| WeightTensorSpec {
            id: id(format!("weight.{name}")),
            dimensions: shape.to_vec(),
            logical_element_type: ElementType::U8,
            physical_layout: PhysicalWeightLayout::PhysicalComponent(
                PhysicalWeightComponentView::new(id(component)),
            ),
            required: true,
        })
        .collect(),
        components,
    })
}

fn inputs(config: Config) -> BTreeMap<ProgramValueId, ProgramTensorSpec> {
    let mut result = config.input_specs();
    result.remove(&id(WEIGHT));
    result.remove(&id(CHANNEL));
    result
}

struct Family;
impl ModelFamilyProvider for Family {
    type Config = Config;
    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.test.stored_nvfp4"))
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.test.stored_nvfp4")])
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        config: &Config,
    ) -> Result<(), VNextError> {
        if !matches!(config.kind, Kind::Nvfp4Matmul)
            || serde_json::to_value(config).unwrap() != *raw
        {
            return Err(invalid(
                "stored NVFP4 fixture requires canonical NVFP4 config",
            ));
        }
        Nvfp4MatrixLayout::new(config.rows, config.inputs)?;
        Nvfp4MatrixLayout::new(config.outputs, config.inputs)?;
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        config: &Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(id("metadata.test.stored_nvfp4"))
    }
    fn parse_config(&self, raw: &serde_json::Value) -> Result<Config, VNextError> {
        let config = serde_json::from_value(raw.clone()).map_err(|e| invalid(format!("{e}")))?;
        self.validate_config_identity(raw, &config)?;
        Ok(config)
    }
    fn weights(&self, config: &Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::Stored(schema(config)?))
    }
    fn numerical_profiles(&self, _: &Config) -> Result<FamilyNumericalProfiles, VNextError> {
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: id(PROFILE),
                family_id: self.family_id().clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: id(OUTPUT),
                boundaries: BTreeMap::from([(id(OUTPUT), ElementType::Bf16)]),
                states: vec![],
                kv_storage: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(NVFP4_MATMUL_BF16_OPERATION_ID),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: None,
                    accumulation_type: Some(ElementType::F32),
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                }],
            }],
            vec![id(PROFILE)],
        )
    }
    fn semantic_program(
        &self,
        config: &Config,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let schema = schema(config)?;
        let weights = schema
            .tensors
            .iter()
            .zip([WEIGHT, CHANNEL])
            .map(|(tensor, name)| WeightReference {
                weight_id: tensor.id.clone(),
                value_id: id(name),
                tensor: ProgramTensorSpec {
                    dimensions: tensor.dimensions.clone(),
                    element_type: tensor.logical_element_type,
                    layout: ResolvedTensorLayout::Contiguous,
                },
            })
            .collect();
        ModelProgram::new(
            self.family_id().clone(),
            [INPUT, RELATIVE, BIAS].into_iter().map(id).collect(),
            vec![ProgramBlock {
                id: "block.test.stored_nvfp4".into(),
                nodes: vec![ProgramNode {
                    id: id("node.test.stored_nvfp4"),
                    operation_id: id(NVFP4_MATMUL_BF16_OPERATION_ID),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: [INPUT, WEIGHT, RELATIVE, CHANNEL, BIAS]
                        .into_iter()
                        .map(id)
                        .collect(),
                    outputs: vec![id(OUTPUT)],
                    attributes: BTreeMap::new(),
                }],
            }],
            vec![],
            weights,
            vec![id(OUTPUT)],
        )?
        .with_input_tensor_specs(inputs(*config))
    }
    fn semantic_metadata(&self, _: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}

/// The test payload is immutable original source data. Initialization obtains
/// it once per exact declared component; the executor admits upload staging.
pub struct Source {
    schema: WeightSchema,
    packed: Vec<u8>,
    scales: Vec<u8>,
    calls: AtomicUsize,
}
impl Source {
    pub fn new(config: Config, packed: Vec<u8>, scales: Vec<u8>) -> Self {
        let schema = schema(&config).unwrap();
        assert_eq!(
            packed.len() as u64,
            schema.components[0].physical_bytes().unwrap()
        );
        assert_eq!(
            scales.len() as u64,
            schema.components[1].physical_bytes().unwrap()
        );
        Self {
            schema,
            packed,
            scales,
            calls: AtomicUsize::new(0),
        }
    }
    pub fn component_reads(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
}
impl WeightComponentSource for Source {
    fn component<'a>(
        &'a self,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        let Some(index) = self
            .schema
            .components
            .iter()
            .position(|spec| spec == component)
        else {
            return Err(invalid(
                "source rejects changed encoding, source identity, role or shape",
            ));
        };
        self.calls.fetch_add(1, Ordering::SeqCst);
        let bytes = if index == 0 {
            &self.packed
        } else {
            &self.scales
        };
        WeightComponentPayload::new(
            component,
            component.external_names[0].clone(),
            "original-nvfp4-fixture.safetensors",
            component.dimensions.clone(),
            ElementType::U8,
            bytes.as_slice(),
        )
    }
}

pub fn compile<R: DeviceRuntime>(
    config: Config,
    runtime: &R,
    registry: &OperationRuntimeRegistry<R>,
    catalog: &CapabilityCatalog,
    materializers: &WeightMaterializerRegistry,
) -> CompiledTensorComponent<R> {
    let family = TypedFamilyRegistration::new(Family)
        .prepare_with_profile(&serde_json::to_value(config).unwrap(), &id(PROFILE))
        .unwrap();
    // The attribution denominator retains the original quantized tensor and
    // format, even though the corresponding port has a packed U8 shape.
    let denominator = QuantizedProviderAttributionDenominator::from_prepared_family(&family)
        .unwrap()
        .unwrap();
    assert_eq!(denominator.quant_tensor_count(), 1);
    let policy = TensorComponentPolicy::new(
        runtime.descriptor().total_memory_bytes,
        runtime
            .descriptor()
            .dynamic_storage_profiles
            .iter()
            .copied()
            .collect(),
    )
    .unwrap();
    CompiledTensorComponent::compile(
        family,
        registry,
        catalog,
        &policy,
        materializers,
        &ProgramPlanCompileOptions::new(inputs(config)).unwrap(),
    )
    .unwrap()
}
