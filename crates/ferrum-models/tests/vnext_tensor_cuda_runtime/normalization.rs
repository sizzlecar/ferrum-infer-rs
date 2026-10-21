//! Analytic real-device checks through a managed tensor graph.
use super::*;
use ferrum_kernels::backend::cuda::vnext_ops::cuda_vnext_tensor_bf16_providers;

const SCALE: &str = "value.tensor.scale";
const GATE: &str = "value.tensor.gate";
const NORMALIZED: &str = "value.tensor.normalized";
const MODULATED: &str = "value.tensor.modulated";
const RESIDUAL: &str = "value.tensor.residual";
const TANH: &str = "value.tensor.tanh";
const HEAD_INPUT: &str = "value.tensor.head_input";
const HEAD_GAMMA: &str = "value.tensor.head_gamma";
const HEAD_OUTPUT: &str = "value.tensor.head_output";
const NORMALIZATION_PROFILE: &str = "fixture.cuda.layer_norm.pointwise.bf16.v1";
const OUTPUTS: [&str; 5] = [NORMALIZED, MODULATED, RESIDUAL, TANH, HEAD_OUTPUT];

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

fn managed_input(
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
    let expected = |port: &str, index: usize| -> u16 {
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
            opaque_memory_profile: "experimental-unqualified-normalization-smoke-v1".into(),
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
    ];
    let registry = OperationRuntimeRegistry::new(
        contracts,
        cuda_vnext_tensor_bf16_providers(&runtime).unwrap(),
    )
    .unwrap();
    assert!(cuda_vnext_tensor_bf16_providers(&runtime).is_err());
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
    let input_bytes = (x.len() + scales.len() + gates.len() + 2 * head_values.len()) * 2;
    let output_bytes = (4 * x.len() + head_values.len()) * 2;
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
            ],
        )
        .unwrap();
    assert_eq!(outputs.len(), OUTPUTS.len());
    for output in &outputs {
        let port = output.value_id().as_str();
        assert!(OUTPUTS.contains(&port));
        assert_eq!(
            output.bytes().len(),
            2 * if port == HEAD_OUTPUT {
                head_values.len()
            } else {
                x.len()
            }
        );
        assert!(host.owns(output.bytes()));
        for (index, pair) in output.bytes().bytes().chunks_exact(2).enumerate() {
            let actual = u16::from_le_bytes([pair[0], pair[1]]);
            let wanted = expected(port, index);
            if wanted == 0 {
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
    drop(cuda_vnext_tensor_bf16_providers(&runtime).unwrap());
    eprintln!("CUDA managed LayerNorm H4096, HeadRmsNorm D128 with first-BF16-boundary counterexample, row modulation, gated residual and tanh: analytic outputs and retirement passed; no full-domain or image claim");
}
