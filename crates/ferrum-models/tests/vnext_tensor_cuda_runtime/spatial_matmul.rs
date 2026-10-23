//! Actual CUDA single-head spatial GEMM probes. The F64 oracle is built before
//! device creation; these bounded samples do not establish full-model parity.
use super::normalization::managed_input;
use super::*;
use ferrum_kernels::backend::cuda::{
    vnext_ops::{cuda_vnext_tensor_providers_with_cudnn_sdpa, CudaTensorOperation},
    CudaCudnnConfig, CudnnSdpaPlanPolicy,
};
use half::bf16;

#[path = "spatial_matmul/oracle.rs"]
mod oracle;

const INPUT: &str = "value.spatial_matmul.qkv";
const OUTPUT: &str = "value.spatial_matmul.output";
const PROFILE: &str = "fixture.cuda.spatial_matmul.bf16_f32.v1";
const ABSOLUTE_TOLERANCE: f64 = 1.0 / 256.0;
const RELATIVE_TOLERANCE: f64 = 1.0 / 128.0;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Shape {
    batch: u64,
    channels: u64,
    spatial: u64,
}

impl Shape {
    fn spec(self, qkv: bool) -> ProgramTensorSpec {
        ProgramTensorSpec {
            dimensions: vec![
                self.batch,
                self.channels * if qkv { 3 } else { 1 },
                1,
                self.spatial,
            ],
            element_type: ElementType::Bf16,
            layout: ResolvedTensorLayout::Contiguous,
        }
    }
    fn inputs(self) -> BTreeMap<ProgramValueId, ProgramTensorSpec> {
        BTreeMap::from([(id(INPUT), self.spec(true))])
    }
    fn elements(self) -> usize {
        (self.batch * self.channels * self.spatial) as usize
    }
    fn at(self, batch: usize, qkv: usize, channel: usize, position: usize) -> usize {
        ((batch * 3 + qkv) * self.channels as usize + channel) * self.spatial as usize + position
    }
}

struct SpatialMatmulFamily;
impl ModelFamilyProvider for SpatialMatmulFamily {
    type Config = Shape;
    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.cuda.spatial_matmul.fixture"))
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.cuda.spatial_matmul.fixture")])
    }
    fn validate_config_identity(&self, raw: &Value, config: &Shape) -> Result<(), VNextError> {
        if config.batch == 0
            || config.channels == 0
            || config.spatial == 0
            || serde_json::to_value(config).unwrap() != *raw
        {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "shape".into(),
                reason: "requires canonical positive spatial dimensions".into(),
            });
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Shape,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(id("metadata.cuda.spatial_matmul.fixture"))
    }
    fn parse_config(&self, raw: &Value) -> Result<Shape, VNextError> {
        let config =
            serde_json::from_value(raw.clone()).map_err(|e| VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "shape".into(),
                reason: e.to_string(),
            })?;
        self.validate_config_identity(raw, &config)?;
        Ok(config)
    }
    fn weights(&self, _: &Shape) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::None)
    }
    fn numerical_profiles(&self, _: &Shape) -> Result<FamilyNumericalProfiles, VNextError> {
        let profile = NumericalProfileId::new(PROFILE).unwrap();
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: profile.clone(),
                family_id: self.family_id().clone(),
                version: ContractVersion::new(1, 0),
                kv_storage: vec![],
                primary_activation: id(OUTPUT),
                boundaries: BTreeMap::from([(id(OUTPUT), ElementType::Bf16)]),
                states: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(SPATIAL_ATTENTION_BF16_OPERATION_ID),
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
        shape: &Shape,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        ModelProgram::new(
            self.family_id().clone(),
            vec![id(INPUT)],
            vec![ProgramBlock {
                id: "block.spatial_matmul".into(),
                nodes: vec![ProgramNode {
                    id: id("node.spatial_matmul"),
                    operation_id: id(SPATIAL_ATTENTION_BF16_OPERATION_ID),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: vec![id(INPUT)],
                    outputs: vec![id(OUTPUT)],
                    attributes: BTreeMap::from([(
                        id("scale"),
                        SemanticValue::Rational(CanonicalRational::new(1, 8).unwrap()),
                    )]),
                }],
            }],
            vec![],
            vec![],
            vec![id(OUTPUT)],
        )?
        .with_input_tensor_specs(shape.inputs())
    }
    fn semantic_metadata(&self, _: &Shape) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}

fn vendor_config() -> (CudaCudnnConfig, CudnnSdpaPlanPolicy) {
    let library_path = std::env::var_os("FERRUM_TEST_CUDNN_LIBRARY_PATH")
        .map(std::path::PathBuf::from)
        .expect("set an explicit immutable cuDNN9 library path");
    assert!(library_path.is_absolute());
    let dependency_libraries = std::env::var_os("FERRUM_TEST_CUDNN_DEPENDENCY_LIBRARIES")
        .map(|paths| std::env::split_paths(&paths).collect())
        .unwrap_or_default();
    (
        CudaCudnnConfig {
            library_path,
            dependency_libraries,
            opaque_memory_bytes: 64 << 20,
            workspace_limit_bytes: 1 << 20,
            opaque_memory_profile: "experimental-spatial-matmul-test-v1".into(),
        },
        CudnnSdpaPlanPolicy {
            maximum_plans: 16,
            maximum_live_bindings: 64,
            opaque_memory_bytes: 128 << 20,
            workspace_limit_bytes: 1 << 20,
        },
    )
}

fn check(name: &str, bytes: &[u8], expected: &[f64]) {
    assert_eq!(bytes.len(), expected.len() * 2);
    let mut maximum_absolute_error = 0.0_f64;
    for (index, (pair, &expected)) in bytes.chunks_exact(2).zip(expected).enumerate() {
        let actual = f64::from(bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]])).to_f32());
        if expected.is_nan() {
            assert!(
                actual.is_nan(),
                "{name} element {index}: expected NaN, got {actual}"
            );
        } else if expected.is_infinite() {
            assert_eq!(actual, expected, "{name} element {index}: signed infinity");
        } else {
            let error = (actual - expected).abs();
            assert!(
                actual.is_finite()
                    && error <= ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * expected.abs(),
                "{name} element {index}: actual={actual}, F64={expected}, error={error}"
            );
            maximum_absolute_error = maximum_absolute_error.max(error);
        }
    }
    eprintln!("{name}: {} elements; maximum absolute error={maximum_absolute_error}; tolerance={ABSOLUTE_TOLERANCE}+{RELATIVE_TOLERANCE}*abs(F64)", expected.len());
}

fn run_case(name: &str, shape: Shape, input: Vec<u16>, expected: Vec<f64>) {
    assert_eq!(input.len(), shape.elements() * 3);
    assert_eq!(expected.len(), shape.elements());
    let (cudnn, sdpa) = vendor_config();
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
            usable_memory_bytes: USABLE_BYTES,
            bootstrap_memory_bytes: 1 << 30,
            stream_memory_bytes: 64 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 64,
            opaque_memory_profile: "experimental-spatial-matmul-test-v1".into(),
        })
        .unwrap(),
    );
    let capacity = DeviceCapacityOwner::open(runtime.descriptor(), USABLE_BYTES).unwrap();
    let registry = OperationRuntimeRegistry::new(
        vec![
            Box::new(conv2d_bf16_contract().unwrap()) as Box<dyn OperationContract>,
            Box::new(structured_attention_fused_bf16_contract().unwrap()),
            Box::new(spatial_attention_bf16_contract().unwrap()),
        ],
        cuda_vnext_tensor_providers_with_cudnn_sdpa(
            &runtime,
            &cudnn,
            sdpa,
            &[CudaTensorOperation::SpatialAttention],
        )
        .unwrap(),
    )
    .unwrap();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.spatial_matmul.fixture"),
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
    let family = TypedFamilyRegistration::new(SpatialMatmulFamily)
        .prepare_with_profile(
            &serde_json::to_value(shape).unwrap(),
            &NumericalProfileId::new(PROFILE).unwrap(),
        )
        .unwrap();
    let mut options = ProgramPlanCompileOptions::new(shape.inputs()).unwrap();
    options
        .insert_tensor_spec(id(OUTPUT), shape.spec(false))
        .unwrap();
    options.retain_completion_value(id(OUTPUT));
    let compiled = ProgramPlanCompiler::compile(
        &family,
        &catalog,
        &policy(runtime.as_ref()),
        &registry.planning(),
        &options,
    )
    .unwrap();
    let executable = compiled.executable();
    let node = &executable.execution_plan().payload().nodes()[0];
    // A declared, real invocation allocation is mandatory on the vendor route.
    // Per-batch workspace is reused; B=2 must not silently double it.
    let align = |n: u64| n.div_ceil(256) * 256;
    let tile = shape.spatial.min(256);
    let expected_scratch = align(
        align(align(4 * shape.channels * shape.spatial) + 4 * shape.spatial * tile)
            + 4 * shape.channels * tile,
    );
    assert_eq!(
        node.provider_resources().scratch().unwrap().fixed_bytes(),
        Some(expected_scratch)
    );
    assert!(node.scratch_resource().is_some());
    let resources = no_static(runtime.clone(), executable.execution_plan());
    let output_bytes = shape.elements() * 2;
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            4 * output_bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: 3 * output_bytes,
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
    let mut retained = None;
    for iteration in 0..2 {
        let outputs = executor
            .execute(
                id(format!("run.spatial_matmul.{name}.{iteration}")),
                id(format!("request.spatial_matmul.{name}.{iteration}")),
                vec![managed_input(
                    &host,
                    INPUT,
                    shape.spec(true).dimensions,
                    &input,
                )],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(
            outputs[0].tensor().dimensions(),
            shape.spec(false).dimensions
        );
        assert!(host.owns(outputs[0].bytes()));
        check(name, outputs[0].bytes().bytes(), &expected);
        if shape.spatial == 1 {
            for (pair, expected) in outputs[0].bytes().bytes().chunks_exact(2).zip(&expected) {
                assert_eq!(
                    u16::from_le_bytes([pair[0], pair[1]]),
                    bf16::from_f64(*expected).to_bits(),
                    "singleton softmax must preserve finite V exactly"
                );
            }
        }
        assert_eq!(host.usage().unwrap().committed_bytes(), output_bytes);
        if iteration == 1 {
            retained = Some(outputs[0].bytes().clone());
        }
        drop(outputs);
        assert_eq!(
            host.usage().unwrap().committed_bytes(),
            if iteration == 1 { output_bytes } else { 0 }
        );
    }
    // Claiming the whole ceiling must fail while the real CUDA resources live.
    assert!(matches!(
        capacity.reserve(DeviceOpaqueMemoryKind::Bootstrap, USABLE_BYTES),
        Err(VNextError::DeviceCapacityUnavailable(_))
    ));
    drop((executor, compiled, registry));
    close_resources(resources);
    drop(runtime);
    // This is a ledger claim, not a native allocation. Exact full-ceiling reuse
    // after close detects leaked dynamic scratch and native opaque owners.
    let all = capacity
        .reserve(DeviceOpaqueMemoryKind::Bootstrap, USABLE_BYTES)
        .unwrap();
    assert!(capacity
        .reserve(DeviceOpaqueMemoryKind::Bootstrap, 1)
        .is_err());
    drop(all);
    let retained = retained.unwrap();
    check(name, retained.bytes(), &expected);
    drop(retained);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
}

#[test]
#[ignore = "requires exclusive CUDA and explicit cuDNN9 library; independent F64 spatial oracle"]
fn managed_cuda_spatial_matmul_full_head_batch_and_query_tails() {
    for (name, shape) in [
        (
            "wide1152",
            Shape {
                batch: 2,
                channels: 1152,
                spatial: 7,
            },
        ),
        (
            "tail257",
            Shape {
                batch: 2,
                channels: 3,
                spatial: 257,
            },
        ),
        (
            "tail513",
            Shape {
                batch: 2,
                channels: 5,
                spatial: 513,
            },
        ),
    ] {
        let input = oracle::nonuniform(shape);
        let expected = oracle::reference(shape, &input);
        run_case(name, shape, input, expected);
    }
    // Scores [0, 1] nearly cancel these exact BF16 values in F64/F32 PV.
    // Narrowing P to BF16 instead produces 0.59375, outside the same tolerance.
    let shape = Shape {
        batch: 1,
        channels: 1,
        spatial: 2,
    };
    let input: Vec<u16> = [1.0, 1.0, 0.0, 8.0, 696.0, -256.0]
        .into_iter()
        .map(|value| bf16::from_f32(value).to_bits())
        .collect();
    let expected = oracle::reference(shape, &input);
    for &value in &expected {
        assert!(value > 0.03 && value < 0.04);
        assert!((0.59375 - value).abs() > ABSOLUTE_TOLERANCE + RELATIVE_TOLERANCE * value.abs());
    }
    run_case("probability_f32_cancellation", shape, input, expected);
}

#[test]
#[ignore = "requires exclusive CUDA and explicit cuDNN9 library; score and value IEEE classes"]
fn managed_cuda_spatial_matmul_nonfinite_rows_values_and_zero_probability() {
    for (name, shape, input, expected) in oracle::nonfinite_cases() {
        run_case(name, shape, input, expected);
    }
}

#[test]
#[ignore = "requires exclusive CUDA and explicit cuDNN9 library; exact singleton and owner retirement"]
fn managed_cuda_spatial_matmul_singleton_preserves_finite_values() {
    let shape = Shape {
        batch: 2,
        channels: 3,
        spatial: 1,
    };
    let input = oracle::nonuniform(shape);
    let expected = (0..2)
        .flat_map(|batch| (0..3).map(move |channel| (batch, channel)))
        .map(|(batch, channel)| {
            f64::from(bf16::from_bits(input[shape.at(batch, 2, channel, 0)]).to_f32())
        })
        .collect();
    run_case("singleton", shape, input, expected);
}
