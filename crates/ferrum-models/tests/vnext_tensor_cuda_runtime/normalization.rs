//! Analytic real-device checks through a managed tensor graph.
use super::*;
use ferrum_kernels::backend::cuda::vnext_ops::{cuda_vnext_tensor_providers, CudaTensorOperation};

const SCALE: &str = "value.tensor.scale";
const GATE: &str = "value.tensor.gate";
const NORMALIZED: &str = "value.tensor.normalized";
const MODULATED: &str = "value.tensor.modulated";
const RESIDUAL: &str = "value.tensor.residual";
const TANH: &str = "value.tensor.tanh";
const HEAD_INPUT: &str = "value.tensor.head_input";
const HEAD_GAMMA: &str = "value.tensor.head_gamma";
const HEAD_OUTPUT: &str = "value.tensor.head_output";
const ZERO_WEIGHT: &str = "value.tensor.zero_centered_weight";
const ZERO_OUTPUT: &str = "value.tensor.zero_centered_output";
const SCALAR_INPUT: &str = "value.tensor.scalar_input";
const SCALAR_WEIGHT: &str = "value.tensor.scalar_weight";
const SCALAR_OUTPUT: &str = "value.tensor.scalar_output";
const GELU_INPUT: &str = "value.tensor.gelu_input";
const GELU_OUTPUT: &str = "value.tensor.gelu_output";
const NORMALIZATION_PROFILE: &str = "fixture.cuda.normalization.activation.bf16.v2";
const OUTPUTS: [&str; 8] = [
    NORMALIZED,
    MODULATED,
    RESIDUAL,
    TANH,
    HEAD_OUTPUT,
    ZERO_OUTPUT,
    SCALAR_OUTPUT,
    GELU_OUTPUT,
];
const SELECTED: [CudaTensorOperation; 9] = [
    CudaTensorOperation::Silu,
    CudaTensorOperation::SiluMul,
    CudaTensorOperation::Tanh,
    CudaTensorOperation::ModulationScale,
    CudaTensorOperation::GatedResidual,
    CudaTensorOperation::LayerNorm,
    CudaTensorOperation::HeadRmsNorm,
    CudaTensorOperation::ZeroCenteredRmsNorm,
    CudaTensorOperation::GeluTanh,
];
// Independent singleton expectations for the explicitly staged FP32 formula.
// GELU(1) ~= .841192, GELU(-1) ~= -.158808: BF16 3f57 / be23.
// For minimum BF16 magnitudes, RN32(1+tanh(argument))=1 and x/2
// is exactly the BF16 midpoint adjacent to signed zero. At +/-2^64,
// x*x overflows FP32 but half*x is finite, giving x / negative zero.
// NaNs are checked by classification; every other entry includes its sign.
const GELU_INPUT_BITS: [u16; 13] = [
    0x0000, 0x8000, 0x3f80, 0xbf80, 0x42b4, 0xc2b4, 0x0001, 0x8001, 0x7f80, 0xff80, 0x7fc1, 0x5f80,
    0xdf80,
];
const GELU_EXPECTED_BITS: [u16; 13] = [
    0x0000, 0x8000, 0x3f57, 0xbe23, 0x42b4, 0x8000, 0x0000, 0x8000, 0x7f80, 0x7fc0, 0x7fc0, 0x5f80,
    0x8000,
];

struct NormalizationFamily;

impl ModelFamilyProvider for NormalizationFamily {
    type Config = Config;

    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.cuda.tensor.normalization.fixture"))
    }

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.cuda.tensor.normalization.fixture")])
    }

    fn validate_config_identity(&self, raw: &Value, config: &Config) -> Result<(), VNextError> {
        if config.rows == 0 || config.width == 0 || serde_json::to_value(config).unwrap() != *raw {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "shape".into(),
                reason: "fixture requires canonical nonempty matrix dimensions".into(),
            });
        }
        Ok(())
    }

    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(id("metadata.cuda.tensor.normalization.fixture"))
    }

    fn parse_config(&self, raw: &Value) -> Result<Config, VNextError> {
        let config = serde_json::from_value(raw.clone()).map_err(|error| {
            VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "config".into(),
                reason: error.to_string(),
            }
        })?;
        self.validate_config_identity(raw, &config)?;
        Ok(config)
    }

    fn weights(&self, _: &Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::None)
    }

    fn numerical_profiles(&self, _: &Config) -> Result<FamilyNumericalProfiles, VNextError> {
        let profile_id = NumericalProfileId::new(NORMALIZATION_PROFILE).unwrap();
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: profile_id.clone(),
                family_id: self.family_id().clone(),
                version: ContractVersion::new(1, 0),
                kv_storage: vec![],
                primary_activation: id(RESIDUAL),
                boundaries: OUTPUTS
                    .into_iter()
                    .map(|name| (id(name), ElementType::Bf16))
                    .collect(),
                states: vec![],
                operations: [
                    LAYER_NORM_BF16_OPERATION_ID,
                    MODULATION_SCALE_BF16_OPERATION_ID,
                    GATED_RESIDUAL_BF16_OPERATION_ID,
                    TANH_BF16_OPERATION_ID,
                    HEAD_RMS_NORM_BF16_OPERATION_ID,
                    ZERO_CENTERED_RMS_NORM_BF16_OPERATION_ID,
                    GELU_TANH_BF16_OPERATION_ID,
                ]
                .into_iter()
                .map(|name| NumericalOperationContract {
                    operation_id: id(name),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: Some(ElementType::F32),
                    accumulation_type: Some(ElementType::F32),
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                })
                .collect(),
            }],
            vec![profile_id],
        )
    }

    fn semantic_program(
        &self,
        config: &Config,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let node =
            |name: &str, operation: &str, inputs: &[&str], output: &str, attributes| ProgramNode {
                id: id(format!("node.tensor.normalization.{name}")),
                operation_id: id(operation),
                required_version: ContractVersion::new(1, 0),
                work: ProgramNodeWorkSpec::Fixed,
                inputs: inputs.iter().map(|&name| id(name)).collect(),
                outputs: vec![id(output)],
                attributes,
            };
        let row_broadcast = BTreeMap::from([(id("broadcast"), SemanticValue::Text("row".into()))]);
        ModelProgram::new(
            self.family_id().clone(),
            vec![
                id(INPUT_ID),
                id(SCALE),
                id(GATE),
                id(HEAD_INPUT),
                id(HEAD_GAMMA),
                id(ZERO_WEIGHT),
                id(SCALAR_INPUT),
                id(SCALAR_WEIGHT),
                id(GELU_INPUT),
            ],
            vec![ProgramBlock {
                id: "block.tensor.normalization".into(),
                nodes: vec![
                    node(
                        "layer_norm",
                        LAYER_NORM_BF16_OPERATION_ID,
                        &[INPUT_ID],
                        NORMALIZED,
                        BTreeMap::from([
                            (id("hidden_size"), SemanticValue::Unsigned(config.width)),
                            (
                                id("epsilon"),
                                SemanticValue::Rational(CanonicalRational::new(1, 1)?),
                            ),
                        ]),
                    ),
                    node(
                        "modulation",
                        MODULATION_SCALE_BF16_OPERATION_ID,
                        &[NORMALIZED, SCALE],
                        MODULATED,
                        row_broadcast.clone(),
                    ),
                    node(
                        "residual",
                        GATED_RESIDUAL_BF16_OPERATION_ID,
                        &[NORMALIZED, GATE, MODULATED],
                        RESIDUAL,
                        row_broadcast,
                    ),
                    node(
                        "tanh",
                        TANH_BF16_OPERATION_ID,
                        &[INPUT_ID],
                        TANH,
                        BTreeMap::new(),
                    ),
                    node(
                        "head_rms_norm",
                        HEAD_RMS_NORM_BF16_OPERATION_ID,
                        &[HEAD_INPUT, HEAD_GAMMA],
                        HEAD_OUTPUT,
                        BTreeMap::from([
                            (id("head_dim"), SemanticValue::Unsigned(128)),
                            (
                                id("epsilon"),
                                SemanticValue::Rational(CanonicalRational::new(112375, 147456)?),
                            ),
                        ]),
                    ),
                    node(
                        "zero_centered_rms_norm",
                        ZERO_CENTERED_RMS_NORM_BF16_OPERATION_ID,
                        &[HEAD_INPUT, ZERO_WEIGHT],
                        ZERO_OUTPUT,
                        BTreeMap::from([(
                            id("epsilon"),
                            SemanticValue::Rational(CanonicalRational::new(112375, 147456)?),
                        )]),
                    ),
                    node(
                        "zero_centered_rms_scalar",
                        ZERO_CENTERED_RMS_NORM_BF16_OPERATION_ID,
                        &[SCALAR_INPUT, SCALAR_WEIGHT],
                        SCALAR_OUTPUT,
                        BTreeMap::from([(
                            id("epsilon"),
                            SemanticValue::Rational(CanonicalRational::new(7, 9)?),
                        )]),
                    ),
                    node(
                        "gelu_tanh",
                        GELU_TANH_BF16_OPERATION_ID,
                        &[GELU_INPUT],
                        GELU_OUTPUT,
                        BTreeMap::new(),
                    ),
                ],
            }],
            vec![],
            vec![],
            OUTPUTS.into_iter().map(id).collect(),
        )
    }

    fn semantic_metadata(&self, _: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}

pub(super) fn managed_input(
    host: &HostMemoryAccount,
    name: &str,
    dimensions: Vec<u64>,
    bits: &[u16],
) -> ManagedHostTensor {
    assert_eq!(dimensions.iter().product::<u64>(), bits.len() as u64);
    let mut allocation = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 2 * bits.len())
        .unwrap();
    for (pair, bits) in allocation.as_mut().chunks_exact_mut(2).zip(bits) {
        pair.copy_from_slice(&bits.to_le_bytes());
    }
    ManagedHostTensor::new(
        id(name),
        input_spec(dimensions, ElementType::Bf16),
        allocation.freeze(),
    )
    .unwrap()
}

#[test]
#[ignore = "requires exclusive CUDA; production managed LayerNorm and pointwise bundle"]
fn managed_cuda_layer_norm_head_rms_modulation_residual_and_tanh() {
    let config = Config {
        rows: 2,
        width: 4096,
    };
    let width = config.width as usize;
    // Row 0 has exact mean 1002 and variance 4; row 1 is constant.
    // epsilon=1 gives +/-2/sqrt(5), rounding to BF16 +/-229/256.
    // B(1+1/256)=1 (even midpoint), so modulation preserves that value.
    // B((229/256)*(129/128))=231/256; their residual sum is 115/64.
    // tanh(1000) and tanh(1004) round to 1. All expectations precede CUDA.
    let mut x = vec![0x447a; 2 * width]; // 1000
    x[width / 2..width].fill(0x447b); // 1004
    let scales = vec![0x3b80; width]; // 1/256
    let gates = vec![0x3f81; width]; // 129/128
                                     // With D=128, x=gamma=129/128 and epsilon=112375/147456,
                                     // the declared FP32 sqrt/div gives reciprocal=3/4. Keeping the first
                                     // BF16 boundary produces 0x3f44; omitting it produces 0x3f43.
    let head_values = vec![0x3f81; 128];
    // The same x with stored weight=1/128 has effective FP32 gamma=129/128.
    // Keeping x*inverse in FP32 yields 3f43; inserting the HeadRMS B gives 3f44.
    let zero_weights = vec![0x3c00; 128];
    // C=1, x=1, epsilon=7/9 gives inverse=3/4. Weight=1/256 must
    // become FP32 257/256; early B(weight+1) instead produces wrong 3f40.
    let scalar_input = [0x3f80];
    let scalar_weight = [0x3b80];
    let expected = |port: &str, index: usize| -> u16 {
        if port == ZERO_OUTPUT {
            return 0x3f43;
        }
        if port == SCALAR_OUTPUT {
            return 0x3f41;
        }
        if port == GELU_OUTPUT {
            return GELU_EXPECTED_BITS[index];
        }
        if port == HEAD_OUTPUT {
            return 0x3f44;
        }
        if port == TANH {
            return 0x3f80;
        }
        if index >= width {
            return 0;
        }
        let magnitude = if port == RESIDUAL { 0x3fe6 } else { 0x3f65 };
        magnitude | if index < width / 2 { 0x8000 } else { 0 }
    };
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
            usable_memory_bytes: USABLE_BYTES,
            bootstrap_memory_bytes: 1 << 30,
            stream_memory_bytes: 64 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 64,
            opaque_memory_profile: "experimental-unqualified-normalization-activation-smoke-v2"
                .into(),
        })
        .unwrap(),
    );
    let contracts: Vec<Box<dyn OperationContract>> = vec![
        Box::new(silu_bf16_contract().unwrap()),
        Box::new(silu_mul_bf16_contract().unwrap()),
        Box::new(tanh_bf16_contract().unwrap()),
        Box::new(modulation_scale_bf16_contract().unwrap()),
        Box::new(gated_residual_bf16_contract().unwrap()),
        Box::new(layer_norm_bf16_contract().unwrap()),
        Box::new(head_rms_norm_bf16_contract().unwrap()),
        Box::new(zero_centered_rms_norm_bf16_contract().unwrap()),
        Box::new(gelu_tanh_bf16_contract().unwrap()),
    ];
    let registry = OperationRuntimeRegistry::new(
        contracts,
        cuda_vnext_tensor_providers(&runtime, &SELECTED).unwrap(),
    )
    .unwrap();
    assert!(cuda_vnext_tensor_providers(&runtime, &SELECTED).is_err());
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.tensor.normalization.fixture"),
        ContractVersion::new(1, 0),
        runtime
            .descriptor()
            .runtime_implementation_fingerprint
            .clone(),
        runtime.descriptor().id.clone(),
        runtime.descriptor().capabilities.clone(),
    )
    .unwrap();
    let catalog = registry
        .capability_catalog(runtime.descriptor().clone(), vec![engine])
        .unwrap();
    let family = TypedFamilyRegistration::new(NormalizationFamily)
        .prepare_with_profile(
            &serde_json::to_value(&config).unwrap(),
            &NumericalProfileId::new(NORMALIZATION_PROFILE).unwrap(),
        )
        .unwrap();
    let mut parameters = config.spec();
    parameters.dimensions[0] = 1;
    let mut options = ProgramPlanCompileOptions::new(BTreeMap::from([
        (id(INPUT_ID), config.spec()),
        (id(SCALE), parameters.clone()),
        (id(GATE), parameters),
        (
            id(HEAD_INPUT),
            ProgramTensorSpec {
                dimensions: vec![1, 128],
                element_type: ElementType::Bf16,
                layout: ResolvedTensorLayout::Contiguous,
            },
        ),
        (
            id(HEAD_GAMMA),
            ProgramTensorSpec {
                dimensions: vec![128],
                element_type: ElementType::Bf16,
                layout: ResolvedTensorLayout::Contiguous,
            },
        ),
        (
            id(ZERO_WEIGHT),
            ProgramTensorSpec {
                dimensions: vec![128],
                element_type: ElementType::Bf16,
                layout: ResolvedTensorLayout::Contiguous,
            },
        ),
        (
            id(SCALAR_INPUT),
            ProgramTensorSpec {
                dimensions: vec![1, 1],
                element_type: ElementType::Bf16,
                layout: ResolvedTensorLayout::Contiguous,
            },
        ),
        (
            id(SCALAR_WEIGHT),
            ProgramTensorSpec {
                dimensions: vec![1],
                element_type: ElementType::Bf16,
                layout: ResolvedTensorLayout::Contiguous,
            },
        ),
        (
            id(GELU_INPUT),
            ProgramTensorSpec {
                dimensions: vec![1, GELU_INPUT_BITS.len() as u64],
                element_type: ElementType::Bf16,
                layout: ResolvedTensorLayout::Contiguous,
            },
        ),
    ]))
    .unwrap();
    for port in OUTPUTS {
        options.retain_completion_value(id(port));
    }
    let compiled = ProgramPlanCompiler::compile(
        &family,
        &catalog,
        &policy(runtime.as_ref()),
        &registry.planning(),
        &options,
    )
    .unwrap();
    let executable = compiled.executable();
    let resources = no_static(runtime.clone(), executable.execution_plan());
    let input_bytes = (x.len()
        + scales.len()
        + gates.len()
        + 2 * head_values.len()
        + zero_weights.len()
        + scalar_input.len()
        + scalar_weight.len()
        + GELU_INPUT_BITS.len())
        * 2;
    let output_bytes = (4 * x.len() + 2 * head_values.len() + 1 + GELU_EXPECTED_BITS.len()) * 2;
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            input_bytes + output_bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: input_bytes,
                output_bytes,
                transfer_staging_bytes: 0,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    );
    let mut executor = PreparedTensorExecutor::new(
        executable.clone(),
        registry.bind_plan(executable).unwrap(),
        resources.clone(),
        host.clone(),
        TensorExecutorConfig::default(),
    )
    .unwrap();
    let outputs = executor
        .execute(
            id("run.cuda.normalization.fixture"),
            id("request.cuda.normalization.fixture"),
            vec![
                managed_input(&host, INPUT_ID, vec![config.rows, config.width], &x),
                managed_input(&host, SCALE, vec![1, config.width], &scales),
                managed_input(&host, GATE, vec![1, config.width], &gates),
                managed_input(&host, HEAD_INPUT, vec![1, 128], &head_values),
                managed_input(&host, HEAD_GAMMA, vec![128], &head_values),
                managed_input(&host, ZERO_WEIGHT, vec![128], &zero_weights),
                managed_input(&host, SCALAR_INPUT, vec![1, 1], &scalar_input),
                managed_input(&host, SCALAR_WEIGHT, vec![1], &scalar_weight),
                managed_input(
                    &host,
                    GELU_INPUT,
                    vec![1, GELU_INPUT_BITS.len() as u64],
                    &GELU_INPUT_BITS,
                ),
            ],
        )
        .unwrap();
    assert_eq!(outputs.len(), OUTPUTS.len());
    for output in &outputs {
        let port = output.value_id().as_str();
        assert!(OUTPUTS.contains(&port));
        assert_eq!(
            output.bytes().len(),
            2 * match port {
                HEAD_OUTPUT | ZERO_OUTPUT => head_values.len(),
                SCALAR_OUTPUT => 1,
                GELU_OUTPUT => GELU_EXPECTED_BITS.len(),
                _ => x.len(),
            }
        );
        assert!(host.owns(output.bytes()));
        for (index, pair) in output.bytes().bytes().chunks_exact(2).enumerate() {
            let actual = u16::from_le_bytes([pair[0], pair[1]]);
            let wanted = expected(port, index);
            if wanted & 0x7fff > 0x7f80 {
                assert_eq!(actual & 0x7f80, 0x7f80, "{port} index {index}");
                assert_ne!(actual & 0x007f, 0, "{port} index {index}");
            } else if wanted == 0 && port != GELU_OUTPUT {
                assert_eq!(actual & 0x7fff, 0, "{port} index {index}");
            } else {
                assert_eq!(actual, wanted, "{port} index {index}");
            }
        }
    }
    drop((outputs, executor, compiled, registry));
    close_resources(resources);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    drop(cuda_vnext_tensor_providers(&runtime, &SELECTED).unwrap());
    eprintln!("CUDA managed LayerNorm H4096, HeadRmsNorm/zero-centered RMS rounding counterexamples, row modulation, gated residual, tanh and GELU(tanh) singleton/classification cases passed with retirement; no full-domain or image claim");
}
