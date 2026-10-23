//! Exact analytic probes for the production HeadRMS launch, without reusing
//! its reduction helper or deriving expectations from a candidate execution.
use super::*;
use half::bf16;

const PROFILE: &str = "fixture.cuda.head_rms.launch.v1";
struct HeadRmsFamily;

fn specs(config: &Config) -> BTreeMap<ProgramValueId, ProgramTensorSpec> {
    BTreeMap::from([
        (
            id(HEAD_INPUT),
            ProgramTensorSpec {
                dimensions: vec![config.rows, 2 * config.width],
                element_type: ElementType::Bf16,
                layout: ResolvedTensorLayout::Contiguous,
            },
        ),
        (
            id(HEAD_GAMMA),
            ProgramTensorSpec {
                dimensions: vec![config.width],
                element_type: ElementType::Bf16,
                layout: ResolvedTensorLayout::Contiguous,
            },
        ),
    ])
}

impl ModelFamilyProvider for HeadRmsFamily {
    type Config = Config;
    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.cuda.head_rms.launch.fixture"))
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.cuda.head_rms.launch.fixture")])
    }
    fn validate_config_identity(&self, raw: &Value, config: &Config) -> Result<(), VNextError> {
        if config.rows == 0
            || !config.width.is_power_of_two()
            || config.width > 4096
            || serde_json::to_value(config).unwrap() != *raw
        {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "shape".into(),
                reason: "requires canonical rows and a supported head width".into(),
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
        Ok(id("metadata.cuda.head_rms.launch.fixture"))
    }
    fn parse_config(&self, raw: &Value) -> Result<Config, VNextError> {
        let config =
            serde_json::from_value(raw.clone()).map_err(|e| VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "shape".into(),
                reason: e.to_string(),
            })?;
        self.validate_config_identity(raw, &config)?;
        Ok(config)
    }
    fn weights(&self, _: &Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::None)
    }
    fn numerical_profiles(&self, _: &Config) -> Result<FamilyNumericalProfiles, VNextError> {
        let profile = NumericalProfileId::new(PROFILE).expect("valid fixture profile ID");
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: profile.clone(),
                family_id: self.family_id().clone(),
                version: ContractVersion::new(1, 0),
                kv_storage: vec![],
                primary_activation: id(HEAD_OUTPUT),
                boundaries: BTreeMap::from([(id(HEAD_OUTPUT), ElementType::Bf16)]),
                states: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(HEAD_RMS_NORM_BF16_OPERATION_ID),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: Some(ElementType::F32),
                    accumulation_type: Some(ElementType::F32),
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                }],
            }],
            vec![profile],
        )
    }
    fn semantic_program(
        &self,
        config: &Config,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        ModelProgram::new(
            self.family_id().clone(),
            vec![id(HEAD_INPUT), id(HEAD_GAMMA)],
            vec![ProgramBlock {
                id: "block.head_rms.launch".into(),
                nodes: vec![ProgramNode {
                    id: id("node.head_rms.launch"),
                    operation_id: id(HEAD_RMS_NORM_BF16_OPERATION_ID),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: vec![id(HEAD_INPUT), id(HEAD_GAMMA)],
                    outputs: vec![id(HEAD_OUTPUT)],
                    attributes: BTreeMap::from([
                        (id("head_dim"), SemanticValue::Unsigned(config.width)),
                        (
                            id("epsilon"),
                            SemanticValue::Rational(CanonicalRational::new(3, 8)?),
                        ),
                    ]),
                }],
            }],
            vec![],
            vec![],
            vec![id(HEAD_OUTPUT)],
        )?
        .with_input_tensor_specs(specs(config))
    }
    fn semantic_metadata(&self, _: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}

fn run_case(
    runtime: &Arc<CudaDeviceRuntime>,
    registry: &OperationRuntimeRegistry<CudaDeviceRuntime>,
    name: &str,
    config: Config,
    input: Vec<u16>,
    gamma: Vec<u16>,
    expected: Vec<u16>,
) {
    assert_eq!(input.len() as u64, config.rows * 2 * config.width);
    assert_eq!(gamma.len() as u64, config.width);
    assert_eq!(input.len(), expected.len());
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.head_rms.launch.fixture"),
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
    let family = TypedFamilyRegistration::new(HeadRmsFamily)
        .prepare_with_profile(
            &serde_json::to_value(&config).unwrap(),
            &NumericalProfileId::new(PROFILE).unwrap(),
        )
        .unwrap();
    let mut options = ProgramPlanCompileOptions::new(specs(&config)).unwrap();
    options.retain_completion_value(id(HEAD_OUTPUT));
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
    let input_bytes = (input.len() + gamma.len()) * 2;
    let output_bytes = expected.len() * 2;
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
            id("run.cuda.head_rms.launch.fixture"),
            id("request.cuda.head_rms.launch.fixture"),
            vec![
                managed_input(
                    &host,
                    HEAD_INPUT,
                    vec![config.rows, 2 * config.width],
                    &input,
                ),
                managed_input(&host, HEAD_GAMMA, vec![config.width], &gamma),
            ],
        )
        .unwrap();
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].bytes().len(), output_bytes);
    assert!(host.owns(outputs[0].bytes()));
    for (index, (bytes, expected)) in outputs[0]
        .bytes()
        .bytes()
        .chunks_exact(2)
        .zip(expected)
        .enumerate()
    {
        let actual = u16::from_le_bytes([bytes[0], bytes[1]]);
        if expected & 0x7fff > 0x7f80 {
            assert!(
                actual & 0x7fff > 0x7f80,
                "{name} index {index}: expected NaN"
            );
        } else {
            assert_eq!(actual, expected, "{name} index {index}");
        }
    }
    drop((outputs, executor, compiled));
    close_resources(resources);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    eprintln!("{name}: exact BF16/classification and Host retirement passed");
}

#[test]
#[ignore = "requires exclusive CUDA; exact HeadRMS head/row, narrow-width and IEEE probes"]
fn managed_cuda_head_rms_widths_preserve_pairwise_boundaries_and_ieee() {
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
            usable_memory_bytes: USABLE_BYTES,
            bootstrap_memory_bytes: 1 << 30,
            stream_memory_bytes: 64 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 64,
            opaque_memory_profile: "experimental-head-rms-launch-test-v1".into(),
        })
        .unwrap(),
    );
    let registry = OperationRuntimeRegistry::new(
        vec![Box::new(head_rms_norm_bf16_contract().unwrap())],
        cuda_vnext_tensor_providers(&runtime, &[CudaTensorOperation::HeadRmsNorm]).unwrap(),
    )
    .unwrap();
    for width in [2_u64, 16, 32, 64, 128, 256, 4096] {
        // Head A: E[x²]=5/8, epsilon=3/8 -> inverse=1.
        // Head B: E[x²]=29/8, same epsilon -> inverse=1/2.
        // All sums and final products are exact dyadics, independently of the
        // kernel tree. Different magnitudes, signs and gamma expose indexing.
        let rows = if width == 2 { 32_769 } else { 2 };
        let mut input = Vec::new();
        let mut expected = Vec::new();
        let gamma_pattern = [2.0_f32, -1.0, 0.5, -2.0];
        for group in 0..2 * rows {
            let (values, output) = if group % 2 == 0 {
                ([0.5_f32, 1.0, -0.5, -1.0], [1.0_f32, -1.0, -0.25, 2.0])
            } else {
                ([2.5_f32, 1.0, -2.5, -1.0], [2.5_f32, -0.5, -0.625, 1.0])
            };
            for column in 0..width as usize {
                input.push(bf16::from_f32(values[column % 4]).to_bits());
                expected.push(bf16::from_f32(output[column % 4]).to_bits());
            }
        }
        let gamma = (0..width as usize)
            .map(|c| bf16::from_f32(gamma_pattern[c % 4]).to_bits())
            .collect();
        // D2 has 65,538 groups: grid-stride shared reuse switches head type,
        // testing the final barrier without a large tensor fixture.
        run_case(
            &runtime,
            &registry,
            &format!("head_d{width}"),
            Config { rows, width },
            input,
            gamma,
            expected,
        );
    }

    // D1: signed zero survives a positive gamma; +/-Inf normalize to NaN.
    // For x=1/2, epsilon=3/8, B(x/sqrt(5/8))=162/256, then gamma2
    // yields 162/128=0x3fa2. No production reduction code supplies this value.
    run_case(
        &runtime,
        &registry,
        "head_d1_ieee",
        Config { rows: 3, width: 1 },
        vec![0x0000, 0x8000, 0x7f80, 0xff80, 0x7fc1, 0x3f00],
        vec![0x4000],
        vec![0x0000, 0x8000, 0x7fc0, 0x7fc0, 0x7fc0, 0x3fa2],
    );

    let mut input = vec![0x3f00; 64];
    input[0] = 0x7fc1; // poison one whole head
    input[16] = 0x7f80; // Inf*zero -> NaN only at this element
    input[32..48].fill(0);
    input[33] = 0x8000;
    input[48] = 0x5f80; // finite 2^64: square overflows, normalized result is 0
    let mut expected = vec![0; 64];
    expected[..16].fill(0x7fc0);
    expected[16] = 0x7fc0;
    expected[33] = 0x8000;
    run_case(
        &runtime,
        &registry,
        "head_d16_ieee_isolation",
        Config { rows: 2, width: 16 },
        input,
        vec![0x3f80; 16],
        expected,
    );
}
