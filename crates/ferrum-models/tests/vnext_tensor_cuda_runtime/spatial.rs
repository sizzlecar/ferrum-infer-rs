//! Explicit test deployment input: FERRUM_TEST_CUDNN_LIBRARY_PATH must name
//! the absolute, immutable cuDNN 9 library selected by the test operator.
//! There is no default path, library search, or unbudgeted discovery load.
//! Reservations below are experimental allowances, not measured capacity proof.
use super::normalization::managed_input;
use super::*;
use ferrum_kernels::backend::cuda::{
    vnext_ops::{cuda_vnext_tensor_providers_with_cudnn, CudaTensorOperation},
    CudaCudnnConfig,
};

const X: &str = "value.spatial.input";
const WEIGHT: &str = "value.spatial.weight";
const BIAS: &str = "value.spatial.bias";
const GAMMA: &str = "value.spatial.gamma";
const CONVOLVED: &str = "value.spatial.convolved";
const NORMALIZED: &str = "value.spatial.normalized";
const ACTIVATED: &str = "value.spatial.activated";
const RESIDUAL: &str = "value.spatial.residual";
const OUTPUTS: [&str; 4] = [CONVOLVED, NORMALIZED, ACTIVATED, RESIDUAL];
const SPATIAL_PROFILE: &str = "fixture.cuda.cudnn9.spatial.bf16.v1";
const WORKSPACE_BYTES: u64 = 1 << 20;
const SELECTED: [CudaTensorOperation; 3] = [
    CudaTensorOperation::ChannelL2Norm,
    CudaTensorOperation::SiluNchw,
    CudaTensorOperation::PlainResidualAddNchw,
];

fn spec(dimensions: &[u64]) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions: dimensions.to_vec(),
        element_type: ElementType::Bf16,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
fn inputs() -> BTreeMap<ProgramValueId, ProgramTensorSpec> {
    BTreeMap::from([
        (id(X), spec(&[1, 1, 1, 2])),
        (id(WEIGHT), spec(&[1, 1, 1, 1])),
        (id(BIAS), spec(&[1])),
        (id(GAMMA), spec(&[1, 1, 1, 1])),
    ])
}

struct SpatialFamily;
impl ModelFamilyProvider for SpatialFamily {
    type Config = Config;
    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.cuda.tensor.spatial.fixture"))
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.cuda.tensor.spatial.fixture")])
    }
    fn validate_config_identity(&self, raw: &Value, config: &Config) -> Result<(), VNextError> {
        if config.rows != 1 || config.width != 2 || serde_json::to_value(config).unwrap() != *raw {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "shape".into(),
                reason: "fixture requires canonical NCHW [1,1,1,2]".into(),
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
        Ok(id("metadata.cuda.tensor.spatial.fixture"))
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
        let profile_id = NumericalProfileId::new(SPATIAL_PROFILE).unwrap();
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
                    CONV2D_BF16_OPERATION_ID,
                    CHANNEL_L2_NORM_BF16_OPERATION_ID,
                    SILU_NCHW_BF16_OPERATION_ID,
                    PLAIN_RESIDUAL_ADD_NCHW_BF16_OPERATION_ID,
                ]
                .into_iter()
                .map(|name| NumericalOperationContract {
                    operation_id: id(name),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: (name != PLAIN_RESIDUAL_ADD_NCHW_BF16_OPERATION_ID)
                        .then_some(ElementType::F32),
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
        _: &Config,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let node = |name: &str, operation: &str, operands: &[&str], output: &str, attributes| {
            ProgramNode {
                id: id(format!("node.spatial.{name}")),
                operation_id: id(operation),
                required_version: ContractVersion::new(1, 0),
                work: ProgramNodeWorkSpec::Fixed,
                inputs: operands.iter().map(|name| id(*name)).collect(),
                outputs: vec![id(output)],
                attributes,
            }
        };
        ModelProgram::new(
            self.family_id().clone(),
            inputs().into_keys().collect(),
            vec![ProgramBlock {
                id: "block.tensor.spatial".into(),
                nodes: vec![
                    node(
                        "conv",
                        CONV2D_BF16_OPERATION_ID,
                        &[X, WEIGHT, BIAS],
                        CONVOLVED,
                        BTreeMap::from([
                            (id("padding"), SemanticValue::Integers(vec![0, 0, 1, 0])),
                            (id("stride"), SemanticValue::Integers(vec![1, 1])),
                            (id("dilation"), SemanticValue::Integers(vec![1, 1])),
                            (id("groups"), SemanticValue::Unsigned(1)),
                            (
                                id("padding_mode"),
                                SemanticValue::Text("explicit_zero".into()),
                            ),
                        ]),
                    ),
                    node(
                        "norm",
                        CHANNEL_L2_NORM_BF16_OPERATION_ID,
                        &[CONVOLVED, GAMMA],
                        NORMALIZED,
                        BTreeMap::from([
                            (
                                id("epsilon"),
                                SemanticValue::Rational(
                                    CanonicalRational::new(1, 1_000_000_000_000).unwrap(),
                                ),
                            ),
                            (
                                id("scale"),
                                SemanticValue::Rational(CanonicalRational::new(1, 1).unwrap()),
                            ),
                        ]),
                    ),
                    node(
                        "silu",
                        SILU_NCHW_BF16_OPERATION_ID,
                        &[NORMALIZED],
                        ACTIVATED,
                        BTreeMap::new(),
                    ),
                    node(
                        "add",
                        PLAIN_RESIDUAL_ADD_NCHW_BF16_OPERATION_ID,
                        &[NORMALIZED, ACTIVATED],
                        RESIDUAL,
                        BTreeMap::new(),
                    ),
                ],
            }],
            vec![],
            vec![],
            OUTPUTS.into_iter().map(id).collect(),
        )?
        .with_input_tensor_specs(inputs())
    }
    fn semantic_metadata(&self, _: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}

fn expected(name: &str) -> [u16; 3] {
    // One-term dot is exact FP32: (129/128)^2 = 16641/16384.
    // B(dot)=130/128; adding bias -130/128 gives +0. Fusing the
    // bias before B would incorrectly give 1/16384 (0x3880).
    // The two padded/zero inputs produce -130/128. C=1 L2, gamma=-1
    // maps these to +1, then SiLU(1) rounds to 187/256 (0x3f3b).
    // B(1+187/256) is the even midpoint endpoint 222/128 (0x3fde).
    match name {
        CONVOLVED => [0xbf82, 0, 0xbf82],
        NORMALIZED => [0x3f80, 0, 0x3f80],
        ACTIVATED => [0x3f3b, 0, 0x3f3b],
        RESIDUAL => [0x3fde, 0, 0x3fde],
        other => panic!("unexpected output {other}"),
    }
}
fn check_bits(bytes: &[u8], expected: [u16; 3]) {
    assert_eq!(bytes.len(), 6);
    for (index, (pair, expected)) in bytes.chunks_exact(2).zip(expected).enumerate() {
        assert_eq!(
            u16::from_le_bytes([pair[0], pair[1]]),
            expected,
            "element {index}"
        );
    }
}

#[test]
#[ignore = "requires exclusive CUDA and explicit FERRUM_TEST_CUDNN_LIBRARY_PATH"]
fn managed_cuda_cudnn_conv_norm_nchw_pointwise_scratch_and_retirement() {
    let library_path = std::env::var_os("FERRUM_TEST_CUDNN_LIBRARY_PATH")
        .map(std::path::PathBuf::from)
        .expect("set FERRUM_TEST_CUDNN_LIBRARY_PATH to the explicit absolute cuDNN 9 library path");
    assert!(
        library_path.is_absolute(),
        "test cuDNN library path must be absolute"
    );
    let cudnn = CudaCudnnConfig {
        library_path,
        dependency_libraries: Vec::new(),
        opaque_memory_bytes: 64 << 20,
        workspace_limit_bytes: WORKSPACE_BYTES,
        opaque_memory_profile: "experimental-unqualified-spatial-test-v1".into(),
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
            opaque_memory_profile: "experimental-unqualified-spatial-test-v1".into(),
        })
        .unwrap(),
    );
    let registry = OperationRuntimeRegistry::new(
        vec![
            Box::new(conv2d_bf16_contract().unwrap()),
            Box::new(channel_l2_norm_bf16_contract().unwrap()),
            Box::new(silu_nchw_bf16_contract().unwrap()),
            Box::new(plain_residual_add_nchw_bf16_contract().unwrap()),
        ],
        cuda_vnext_tensor_providers_with_cudnn(&runtime, &cudnn, &SELECTED).unwrap(),
    )
    .unwrap();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.tensor.spatial.fixture"),
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
    let family = TypedFamilyRegistration::new(SpatialFamily)
        .prepare_with_profile(
            &serde_json::to_value(Config { rows: 1, width: 2 }).unwrap(),
            &NumericalProfileId::new(SPATIAL_PROFILE).unwrap(),
        )
        .unwrap();
    let mut options = ProgramPlanCompileOptions::new(inputs()).unwrap();
    for name in OUTPUTS {
        options
            .insert_tensor_spec(id(name), spec(&[1, 1, 1, 3]))
            .unwrap();
        options.retain_completion_value(id(name));
    }
    let compilation = ProgramPlanCompiler::compile(
        &family,
        &catalog,
        &policy(runtime.as_ref()),
        &registry.planning(),
        &options,
    )
    .unwrap();
    let executable = compilation.executable();
    let nodes = executable.execution_plan().payload().nodes();
    let convolution = nodes
        .iter()
        .find(|node| node.id().as_str() == "node.spatial.conv")
        .unwrap();
    // A real six-byte padded image plus the separately bounded workspace live
    // in one admitted scratch allocation. Neither is a hidden cuDNN allocation.
    let scratch = convolution.provider_resources().scratch().unwrap();
    assert_eq!(scratch.fixed_bytes(), Some(256 + WORKSPACE_BYTES));
    assert_eq!(scratch.minimum_bytes().unwrap(), 256 + WORKSPACE_BYTES);
    assert!(convolution.scratch_resource().is_some());
    for node in nodes.iter().filter(|node| node.id() != convolution.id()) {
        assert!(node.provider_resources().scratch().is_none());
    }
    let resources = no_static(runtime.clone(), executable.execution_plan());
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            34,
            HostMemoryClassLimits {
                input_preprocess_bytes: 10,
                output_bytes: 24,
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
    let mut retained = None;
    for ordinal in 0..2 {
        let outputs = executor
            .execute(
                id(format!("run.cuda.spatial.{ordinal}")),
                id(format!("request.cuda.spatial.{ordinal}")),
                vec![
                    managed_input(&host, X, vec![1, 1, 1, 2], &[0x3f81, 0]),
                    managed_input(&host, WEIGHT, vec![1, 1, 1, 1], &[0x3f81]),
                    managed_input(&host, BIAS, vec![1], &[0xbf82]),
                    managed_input(&host, GAMMA, vec![1, 1, 1, 1], &[0xbf80]),
                ],
            )
            .unwrap();
        assert_eq!(outputs.len(), 4);
        for output in &outputs {
            assert_eq!(output.tensor().dimensions(), [1, 1, 1, 3]);
            assert!(host.owns(output.bytes()));
            check_bits(output.bytes().bytes(), expected(output.value_id().as_str()));
        }
        assert_eq!(host.usage().unwrap().committed_bytes(), 24);
        if ordinal == 1 {
            retained = Some(
                outputs
                    .iter()
                    .find(|output| output.value_id().as_str() == RESIDUAL)
                    .unwrap()
                    .bytes()
                    .clone(),
            );
        }
        drop(outputs);
        assert_eq!(
            host.usage().unwrap().committed_bytes(),
            if ordinal == 1 { 6 } else { 0 }
        );
    }
    drop((executor, compilation, registry));
    close_resources(resources);
    // Reacquisition checks that the retired native bundle and lane no longer
    // consume their single-live-owner slots. It is not an opaque peak measure.
    drop(cuda_vnext_tensor_providers_with_cudnn(&runtime, &cudnn, &SELECTED).unwrap());
    drop(runtime.create_stream().unwrap());
    drop(runtime);
    check_bits(retained.as_ref().unwrap().bytes(), expected(RESIDUAL));
    drop(retained);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    eprintln!("CUDA spatial: staged-bias Conv, explicit padded scratch, channel L2, NCHW SiLU/add, two executions and retained Host output passed; no VAE/Torch parity or capacity qualification claim");
}
