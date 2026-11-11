//! Typed, weightless component fixtures shared by CPU and real CUDA tests.
//! Matrix constants are managed test inputs; every execution uses real plans,
//! admission, uploads, providers, readbacks and retirement.
use ferrum_execution::{
    component::{CompiledTensorComponent, TensorComponentPolicy, TensorComponentSession},
    TensorExecutionOutput,
};
use ferrum_runtime::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::OnceLock,
};

pub const INPUT: &str = "value.fixture.input";
pub const WEIGHT: &str = "value.fixture.weight";
pub const RELATIVE: &str = "value.fixture.relative";
pub const CHANNEL: &str = "value.fixture.channel";
pub const CODEBOOK: &str = "value.fixture.codebook";
pub const BIAS: &str = "value.fixture.bias";
pub const COSINE: &str = "value.fixture.cosine";
pub const SINE: &str = "value.fixture.sine";
pub const OUTPUT: &str = "value.fixture.output";
#[allow(dead_code)] // Shared fixture; each integration test exercises a subset.
pub const F32_EXPECTED: [f32; 6] = [
    0.000244140625,
    0.7501220703125,
    0.875,
    -0.375,
    -3.3125,
    -1.546875,
];

pub fn id<T: TryFrom<String>>(value: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(value.into()).unwrap()
}
fn spec(shape: &[u64], dtype: ElementType) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions: shape.to_vec(),
        element_type: dtype,
        layout: ResolvedTensorLayout::Contiguous,
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Kind {
    F32Dense,
    F32Affine,
    /// Original rank5 patch storage; flattened metadata never replaces its port.
    F32Patch,
    F32ToBf16,
    F32ResidualAdd,
    F32SiluMul,
    HeadRmsF32,
    HalfSplitRopeF32,
    UnitHeadRmsPartialRopeF32,
    ComfyW4A8,
    ComfyW8A8,
    ComfySwiGluF32W4A8,
    ComfySwiGluF32W8A8,
    GeluTanhF32,
    GeluErfF32,
    RowScatterReplaceF32,
    RowScatterAddF32,
}
impl Kind {
    fn operation(self) -> &'static str {
        match self {
            Self::F32Dense => DENSE_LINEAR_F32_OPERATION_ID,
            Self::F32Affine => AFFINE_LINEAR_F32_OPERATION_ID,
            Self::F32Patch => PATCH_PROJECTION_F32_OPERATION_ID,
            Self::F32ToBf16 => CAST_F32_TO_BF16_OPERATION_ID,
            Self::F32ResidualAdd => PLAIN_RESIDUAL_ADD_F32_OPERATION_ID,
            Self::F32SiluMul => SILU_MUL_F32_OPERATION_ID,
            Self::GeluTanhF32 => GELU_TANH_F32_OPERATION_ID,
            Self::GeluErfF32 => GELU_ERF_F32_OPERATION_ID,
            Self::RowScatterReplaceF32 => ROW_SCATTER_REPLACE_F32_OPERATION_ID,
            Self::RowScatterAddF32 => ROW_SCATTER_ADD_F32_OPERATION_ID,
            Self::HeadRmsF32 => HEAD_RMS_NORM_F32_OPERATION_ID,
            Self::HalfSplitRopeF32 => HALF_SPLIT_ROPE_F32_OPERATION_ID,
            Self::UnitHeadRmsPartialRopeF32 => UNIT_HEAD_RMS_PARTIAL_ROPE_F32_OPERATION_ID,
            Self::ComfyW4A8 => COMFY_CONVROT_W4A8_BF16_OPERATION_ID,
            Self::ComfyW8A8 => COMFY_CONVROT_W8A8_BF16_OPERATION_ID,
            Self::ComfySwiGluF32W4A8 => COMFY_CONVROT_SWIGLU_F32_W4A8_BF16_OPERATION_ID,
            Self::ComfySwiGluF32W8A8 => COMFY_CONVROT_SWIGLU_F32_W8A8_BF16_OPERATION_ID,
        }
    }
    fn dtype(self) -> ElementType {
        match self {
            Self::F32Dense
            | Self::F32Affine
            | Self::F32Patch
            | Self::F32ResidualAdd
            | Self::F32SiluMul
            | Self::GeluTanhF32
            | Self::GeluErfF32
            | Self::RowScatterReplaceF32
            | Self::RowScatterAddF32
            | Self::HeadRmsF32
            | Self::HalfSplitRopeF32
            | Self::UnitHeadRmsPartialRopeF32 => ElementType::F32,
            Self::ComfyW4A8
            | Self::F32ToBf16
            | Self::ComfyW8A8
            | Self::ComfySwiGluF32W4A8
            | Self::ComfySwiGluF32W8A8 => ElementType::Bf16,
        }
    }
    fn profile(self) -> &'static str {
        match self {
            Self::F32Dense => "test.tensor_component.f32",
            Self::F32Affine => "test.tensor_component.affine_f32",
            Self::F32Patch => "test.tensor_component.patch_f32",
            Self::F32ToBf16 => "test.tensor_component.cast_f32_to_bf16",
            Self::F32ResidualAdd => "test.tensor_component.residual_add_f32",
            Self::F32SiluMul => "test.tensor_component.silu_mul_f32",
            Self::GeluTanhF32 => "test.tensor_component.gelu_tanh_f32",
            Self::GeluErfF32 => "test.tensor_component.gelu_erf_f32",
            Self::RowScatterReplaceF32 => "test.tensor_component.row_scatter_replace_f32",
            Self::RowScatterAddF32 => "test.tensor_component.row_scatter_add_f32",
            Self::HeadRmsF32 => "test.tensor_component.head_rms_f32",
            Self::HalfSplitRopeF32 => "test.tensor_component.half_split_rope_f32",
            Self::UnitHeadRmsPartialRopeF32 => {
                "test.tensor_component.unit_head_rms_partial_rope_f32"
            }
            Self::ComfyW4A8 => "test.tensor_component.comfy_w4a8",
            Self::ComfyW8A8 => "test.tensor_component.comfy_w8a8",
            Self::ComfySwiGluF32W4A8 => "test.tensor_component.comfy_swiglu_f32_w4a8",
            Self::ComfySwiGluF32W8A8 => "test.tensor_component.comfy_swiglu_f32_w8a8",
        }
    }
}
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    pub rows: u64,
    pub inputs: u64,
    pub outputs: u64,
    pub kind: Kind,
}
impl Config {
    pub fn input_specs(self) -> BTreeMap<ProgramValueId, ProgramTensorSpec> {
        let width = if matches!(
            self.kind,
            Kind::HeadRmsF32 | Kind::HalfSplitRopeF32 | Kind::UnitHeadRmsPartialRopeF32
        ) {
            self.outputs
        } else if matches!(
            self.kind,
            Kind::ComfySwiGluF32W4A8 | Kind::ComfySwiGluF32W8A8
        ) {
            self.inputs.checked_mul(2).unwrap()
        } else {
            self.inputs
        };
        let input_type = if matches!(self.kind, Kind::F32ToBf16) {
            ElementType::F32
        } else {
            self.kind.dtype()
        };
        let mut result = BTreeMap::from([(id(INPUT), spec(&[self.rows, width], input_type))]);
        match self.kind {
            Kind::RowScatterReplaceF32 | Kind::RowScatterAddF32 => {
                result.insert(id(WEIGHT), spec(&[3, self.inputs], ElementType::F32));
            }
            Kind::HeadRmsF32 => {
                result.insert(id(WEIGHT), spec(&[self.inputs], ElementType::F32));
            }
            Kind::HalfSplitRopeF32 | Kind::UnitHeadRmsPartialRopeF32 => {
                let frequency = if matches!(self.kind, Kind::UnitHeadRmsPartialRopeF32) {
                    self.inputs * 3 / 4
                } else {
                    self.inputs
                };
                result.extend([
                    (id(COSINE), spec(&[self.rows, frequency], ElementType::F32)),
                    (id(SINE), spec(&[self.rows, frequency], ElementType::F32)),
                ]);
            }
            Kind::F32ToBf16 | Kind::GeluTanhF32 | Kind::GeluErfF32 => {}
            Kind::F32ResidualAdd | Kind::F32SiluMul => {
                result.insert(
                    id(WEIGHT),
                    spec(&[self.rows, self.inputs], ElementType::F32),
                );
            }
            Kind::F32Patch => {
                let shape = if self.inputs == 1536 {
                    vec![self.outputs, 3, 2, 16, 16]
                } else {
                    vec![self.outputs, 1, 1, 1, self.inputs]
                };
                result.insert(id(WEIGHT), spec(&shape, ElementType::F32));
                result.insert(id(BIAS), spec(&[self.outputs], ElementType::F32));
            }
            Kind::F32Dense | Kind::F32Affine => {
                result.insert(
                    id(WEIGHT),
                    spec(&[self.outputs, self.inputs], ElementType::F32),
                );
                if matches!(self.kind, Kind::F32Affine) {
                    result.insert(id(BIAS), spec(&[self.outputs], ElementType::F32));
                }
            }
            Kind::ComfyW4A8 | Kind::ComfySwiGluF32W4A8 => {
                result.extend([
                    (
                        id(WEIGHT),
                        spec(&[self.outputs, self.inputs / 2], ElementType::I8),
                    ),
                    (
                        id(RELATIVE),
                        spec(&[self.outputs, self.inputs / 16], ElementType::U8),
                    ),
                    (id(CHANNEL), spec(&[self.outputs], ElementType::F32)),
                    (id(CODEBOOK), spec(&[16], ElementType::F32)),
                ]);
            }
            Kind::ComfyW8A8 | Kind::ComfySwiGluF32W8A8 => {
                result.extend([
                    (
                        id(WEIGHT),
                        spec(&[self.outputs, self.inputs], ElementType::I8),
                    ),
                    (id(CHANNEL), spec(&[self.outputs, 1], ElementType::F32)),
                ]);
            }
        }
        result
    }
    pub fn managed(self, host: &HostMemoryAccount, name: &str, bytes: &[u8]) -> ManagedHostTensor {
        let spec = self.input_specs().remove(&id(name)).unwrap();
        let mut backing = host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, bytes.len())
            .unwrap();
        backing.as_mut().copy_from_slice(bytes);
        ManagedHostTensor::new(
            id(name),
            ResolvedTensorSpec::new(spec.dimensions, spec.element_type, spec.layout).unwrap(),
            backing.freeze(),
        )
        .unwrap()
    }
    pub fn compile<R: DeviceRuntime>(
        self,
        runtime: &R,
        registry: &OperationRuntimeRegistry<R>,
        catalog: &CapabilityCatalog,
        materializers: &WeightMaterializerRegistry,
    ) -> CompiledTensorComponent<R> {
        self.try_compile(runtime, registry, catalog, materializers)
            .unwrap()
    }
    pub fn try_compile<R: DeviceRuntime>(
        self,
        runtime: &R,
        registry: &OperationRuntimeRegistry<R>,
        catalog: &CapabilityCatalog,
        materializers: &WeightMaterializerRegistry,
    ) -> Result<CompiledTensorComponent<R>, VNextError> {
        let family = TypedFamilyRegistration::new(Family)
            .prepare_with_profile(
                &serde_json::to_value(self).unwrap(),
                &id(self.kind.profile()),
            )
            .unwrap();
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
            &ProgramPlanCompileOptions::new(self.input_specs()).unwrap(),
        )
    }
}

struct Family;
impl ModelFamilyProvider for Family {
    type Config = Config;
    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.test.tensor_numeric_component"))
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.test.tensor_numeric_component")])
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        config: &Config,
    ) -> Result<(), VNextError> {
        if serde_json::to_value(config).unwrap() != *raw
            || [config.rows, config.inputs, config.outputs].contains(&0)
            || (matches!(
                config.kind,
                Kind::F32ToBf16
                    | Kind::F32ResidualAdd
                    | Kind::F32SiluMul
                    | Kind::GeluTanhF32
                    | Kind::GeluErfF32
                    | Kind::RowScatterReplaceF32
                    | Kind::RowScatterAddF32
            ) && config.inputs != config.outputs)
        {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "noncanonical tensor fixture config".into(),
            });
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        config: &Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(id("metadata.test.tensor_numeric_component"))
    }
    fn parse_config(&self, raw: &serde_json::Value) -> Result<Config, VNextError> {
        let config = serde_json::from_value(raw.clone()).map_err(|error| {
            VNextError::InvalidExecutionPlan {
                reason: format!("{error}"),
            }
        })?;
        self.validate_config_identity(raw, &config)?;
        Ok(config)
    }
    fn weights(&self, _: &Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::None)
    }
    fn numerical_profiles(&self, config: &Config) -> Result<FamilyNumericalProfiles, VNextError> {
        let dense = matches!(
            config.kind,
            Kind::F32Dense
                | Kind::F32Affine
                | Kind::F32Patch
                | Kind::F32ResidualAdd
                | Kind::F32SiluMul
                | Kind::GeluTanhF32
                | Kind::GeluErfF32
                | Kind::HeadRmsF32
                | Kind::HalfSplitRopeF32
                | Kind::UnitHeadRmsPartialRopeF32
        );
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: id(config.kind.profile()),
                family_id: self.family_id().clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: id(OUTPUT),
                boundaries: BTreeMap::from([(id(OUTPUT), config.kind.dtype())]),
                states: vec![],
                kv_storage: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(config.kind.operation()),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: dense.then_some(ElementType::F32),
                    accumulation_type: (dense || matches!(config.kind, Kind::RowScatterAddF32))
                        .then_some(ElementType::F32),
                    staged_arithmetic: match config.kind {
                        Kind::F32Dense
                        | Kind::F32Affine
                        | Kind::F32Patch
                        | Kind::F32ToBf16
                        | Kind::F32ResidualAdd
                        | Kind::F32SiluMul
                        | Kind::GeluTanhF32
                        | Kind::GeluErfF32 => None,
                        Kind::RowScatterReplaceF32 | Kind::RowScatterAddF32 => None,
                        Kind::HeadRmsF32
                        | Kind::HalfSplitRopeF32
                        | Kind::UnitHeadRmsPartialRopeF32 => None,
                        Kind::ComfyW4A8 => {
                            Some(StagedNumericalArithmetic::comfy_convrot_w4a8_bf16())
                        }
                        Kind::ComfyW8A8 => {
                            Some(StagedNumericalArithmetic::comfy_convrot_w8a8_bf16())
                        }
                        Kind::ComfySwiGluF32W4A8 => {
                            Some(StagedNumericalArithmetic::comfy_convrot_swiglu_f32_w4a8_bf16())
                        }
                        Kind::ComfySwiGluF32W8A8 => {
                            Some(StagedNumericalArithmetic::comfy_convrot_swiglu_f32_w8a8_bf16())
                        }
                    },
                    composite_arithmetic: None,
                }],
            }],
            vec![id(config.kind.profile())],
        )
    }
    fn semantic_program(
        &self,
        config: &Config,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let input_specs = config.input_specs();
        let ports = match config.kind {
            Kind::HeadRmsF32 => vec![id(INPUT), id(WEIGHT)],
            Kind::HalfSplitRopeF32 | Kind::UnitHeadRmsPartialRopeF32 => {
                vec![id(INPUT), id(COSINE), id(SINE)]
            }
            Kind::F32Dense => vec![id(INPUT), id(WEIGHT)],
            Kind::F32Affine | Kind::F32Patch => vec![id(INPUT), id(WEIGHT), id(BIAS)],
            Kind::F32ToBf16 | Kind::GeluTanhF32 | Kind::GeluErfF32 => vec![id(INPUT)],
            Kind::F32ResidualAdd | Kind::F32SiluMul => vec![id(INPUT), id(WEIGHT)],
            Kind::RowScatterReplaceF32 | Kind::RowScatterAddF32 => vec![id(INPUT), id(WEIGHT)],
            Kind::ComfyW4A8 | Kind::ComfySwiGluF32W4A8 => {
                [INPUT, WEIGHT, RELATIVE, CHANNEL, CODEBOOK]
                    .into_iter()
                    .map(id)
                    .collect()
            }
            Kind::ComfyW8A8 | Kind::ComfySwiGluF32W8A8 => {
                [INPUT, WEIGHT, CHANNEL].into_iter().map(id).collect()
            }
        };
        ModelProgram::new(
            self.family_id().clone(),
            ports.clone(),
            vec![ProgramBlock {
                id: "block.test.tensor_numeric".into(),
                nodes: vec![ProgramNode {
                    id: id("node.test.tensor_numeric"),
                    operation_id: id(config.kind.operation()),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: ports,
                    outputs: vec![id(OUTPUT)],
                    attributes: if matches!(
                        config.kind,
                        Kind::RowScatterReplaceF32 | Kind::RowScatterAddF32
                    ) {
                        BTreeMap::from([
                            (
                                id("destination_starts"),
                                SemanticValue::Integers(vec![1, 5]),
                            ),
                            (id("run_lengths"), SemanticValue::Integers(vec![2, 1])),
                        ])
                    } else if matches!(config.kind, Kind::UnitHeadRmsPartialRopeF32) {
                        BTreeMap::from([
                            (id("head_dim"), SemanticValue::Unsigned(config.inputs)),
                            (
                                id("rotary_dim"),
                                SemanticValue::Unsigned(config.inputs * 3 / 4),
                            ),
                            (
                                id("epsilon"),
                                SemanticValue::Rational(CanonicalRational::new(1, 4)?),
                            ),
                        ])
                    } else if matches!(config.kind, Kind::HeadRmsF32) {
                        BTreeMap::from([
                            (id("head_dim"), SemanticValue::Unsigned(config.inputs)),
                            (
                                id("epsilon"),
                                SemanticValue::Rational(CanonicalRational::new(1, 4)?),
                            ),
                        ])
                    } else if matches!(config.kind, Kind::HalfSplitRopeF32) {
                        BTreeMap::from([(id("head_dim"), SemanticValue::Unsigned(config.inputs))])
                    } else if matches!(
                        config.kind,
                        Kind::F32ToBf16
                            | Kind::F32Patch
                            | Kind::F32ResidualAdd
                            | Kind::F32SiluMul
                            | Kind::GeluTanhF32
                            | Kind::GeluErfF32
                    ) {
                        BTreeMap::new()
                    } else {
                        BTreeMap::from([
                            (id("in_features"), SemanticValue::Unsigned(config.inputs)),
                            (id("out_features"), SemanticValue::Unsigned(config.outputs)),
                        ])
                    },
                }],
            }],
            vec![],
            vec![],
            vec![id(OUTPUT)],
        )?
        .with_input_tensor_specs(input_specs)
    }
    fn semantic_metadata(&self, _: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}

pub fn host() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            4 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 1 << 20,
                transfer_staging_bytes: 1 << 20,
                output_bytes: 1 << 20,
                weight_backing_bytes: 1 << 20,
            },
        )
        .unwrap(),
    )
}
#[allow(dead_code)] // Shared fixture; each integration test exercises a subset.
pub fn f32_config() -> Config {
    Config {
        rows: 3,
        inputs: 3,
        outputs: 2,
        kind: Kind::F32Dense,
    }
}
#[allow(dead_code)] // Shared fixture; each integration test exercises a subset.
pub fn execute_f32<R: DeviceRuntime>(
    session: &mut TensorComponentSession<R>,
    host: &HostMemoryAccount,
) -> Vec<TensorExecutionOutput> {
    execute_f32_kind(session, host, Kind::F32Dense, 0)
}
#[allow(dead_code)] // Shared fixture; each integration test exercises a subset.
pub fn affine_f32_config() -> Config {
    Config {
        kind: Kind::F32Affine,
        ..f32_config()
    }
}

#[allow(dead_code)] // Shared fixture; each integration test exercises a subset.
pub fn cast_f32_to_bf16_config() -> Config {
    Config {
        rows: 2,
        inputs: 10,
        outputs: 10,
        kind: Kind::F32ToBf16,
    }
}
#[allow(dead_code)] // Shared fixture; each integration test exercises a subset.
pub fn execute_cast_f32_to_bf16<R: DeviceRuntime>(
    session: &mut TensorComponentSession<R>,
    host: &HostMemoryAccount,
    trial: usize,
) -> Vec<TensorExecutionOutput> {
    // Typed bit patterns isolate RNE ties, sign, gradual subnormals, finite
    // overflow, infinities and both quiet/signaling NaN classes independently.
    let bits: [u32; 20] = [
        0x00000000, 0x80000000, 0x3f808000, 0x3f818000, 0xbf808000, 0xbf818000, 0x00008000,
        0x00008001, 0x00010000, 0x00018000, 0x007fffff, 0x00800000, 0x7f7f7fff, 0x7f7f8000,
        0x7f800000, 0xff800000, 0x7fc00001, 0x7f800001, 0xff800001, 0x00000001,
    ];
    let expected: [u16; 20] = [
        0x0000, 0x8000, 0x3f80, 0x3f82, 0xbf80, 0xbf82, 0x0000, 0x0001, 0x0001, 0x0002, 0x0080,
        0x0080, 0x7f7f, 0x7f80, 0x7f80, 0xff80, 0x7fc0, 0x7fc0, 0xffc0, 0x0000,
    ];
    let bytes: Vec<u8> = bits.into_iter().flat_map(u32::to_le_bytes).collect();
    let output = session
        .execute(
            id(format!("run.test.cast_f32_to_bf16.{trial}")),
            id(format!("request.test.cast_f32_to_bf16.{trial}")),
            vec![cast_f32_to_bf16_config().managed(host, INPUT, &bytes)],
        )
        .unwrap();
    assert_eq!(output.len(), 1);
    assert_eq!(output[0].tensor().dimensions(), &[2, 10]);
    assert_eq!(output[0].tensor().element_type(), ElementType::Bf16);
    let actual: Vec<u16> = output[0]
        .bytes()
        .bytes()
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
        .collect();
    for (a, e) in actual.into_iter().zip(expected) {
        if e & 0x7fff > 0x7f80 {
            // Payload is deliberately not contractual.
            assert_eq!(a & 0x7f80, 0x7f80);
            assert_ne!(a & 0x007f, 0);
            assert_ne!(a & 0x0040, 0);
            assert_eq!(a & 0x8000, e & 0x8000);
        } else {
            assert_eq!(a, e);
        }
    }
    output
}
#[allow(dead_code)] // Shared fixture; each integration test exercises a subset.
pub fn execute_affine_f32<R: DeviceRuntime>(
    session: &mut TensorComponentSession<R>,
    host: &HostMemoryAccount,
    trial: usize,
) -> Vec<TensorExecutionOutput> {
    execute_f32_kind(session, host, Kind::F32Affine, trial)
}
#[allow(dead_code)] // Shared fixture; each integration test exercises a subset.
fn execute_f32_kind<R: DeviceRuntime>(
    session: &mut TensorComponentSession<R>,
    host: &HostMemoryAccount,
    kind: Kind,
    trial: usize,
) -> Vec<TensorExecutionOutput> {
    let config = Config {
        kind,
        ..f32_config()
    };
    let x: Vec<u8> = [
        1.000244140625_f32,
        1.0,
        0.0,
        0.125,
        -0.75,
        2.0,
        -3.25,
        0.0625,
        -0.5,
    ]
    .into_iter()
    .flat_map(f32::to_le_bytes)
    .collect();
    let w: Vec<u8> = [1.0_f32, -1.0, 0.0, 0.5, 0.25, -0.125]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect();
    let mut inputs = vec![
        config.managed(host, INPUT, &x),
        config.managed(host, WEIGHT, &w),
    ];
    let bias = [1.000244140625_f32, -0.0001220703125];
    if matches!(kind, Kind::F32Affine) {
        let bytes: Vec<u8> = bias.into_iter().flat_map(f32::to_le_bytes).collect();
        inputs.push(config.managed(host, BIAS, &bytes));
    }
    let output = session
        .execute(
            id(format!("run.test.numeric.f32.execute.{trial}")),
            id(format!("request.test.numeric.f32.execute.{trial}")),
            inputs,
        )
        .unwrap();
    assert_eq!(output.len(), 1);
    assert_eq!(output[0].tensor().dimensions(), &[3, 2]);
    let actual: Vec<_> = output[0]
        .bytes()
        .bytes()
        .chunks_exact(4)
        .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    let expected: Vec<_> = F32_EXPECTED
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            if matches!(kind, Kind::F32Affine) {
                value + bias[index % 2]
            } else {
                value
            }
        })
        .collect();
    assert_eq!(actual, expected);
    output
}
