//! Managed real-device adjacent RoPE: indexing, explicit FMA and signed zero.
use super::*;
use ferrum_kernels::backend::cuda::vnext_ops::{
    cuda_vnext_tensor_component_providers, CUDA_ADJACENT_ROPE_BF16_NUMERICAL_PROFILE_ID,
};

const FREQUENCIES: &str = "value.tensor.frequencies";
struct RotaryFamily;

impl ModelFamilyProvider for RotaryFamily {
    type Config = Config;
    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.cuda.tensor.rotary.fixture"))
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.cuda.tensor.rotary.fixture")])
    }
    fn validate_config_identity(&self, raw: &Value, config: &Config) -> Result<(), VNextError> {
        if config.rows != 4 || config.width != 8 || serde_json::to_value(config).unwrap() != *raw {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "shape".into(),
                reason: "fixture exercises batch=2, sequence=2, heads=2, head_dim=4".into(),
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
        Ok(id("metadata.cuda.tensor.rotary.fixture"))
    }
    fn parse_config(&self, raw: &Value) -> Result<Config, VNextError> {
        let config =
            serde_json::from_value(raw.clone()).map_err(|e| VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "config".into(),
                reason: e.to_string(),
            })?;
        self.validate_config_identity(raw, &config)?;
        Ok(config)
    }
    fn weights(&self, _: &Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::None)
    }
    fn numerical_profiles(&self, _: &Config) -> Result<FamilyNumericalProfiles, VNextError> {
        let profile_id =
            NumericalProfileId::new(CUDA_ADJACENT_ROPE_BF16_NUMERICAL_PROFILE_ID).unwrap();
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: profile_id.clone(),
                family_id: self.family_id().clone(),
                version: ContractVersion::new(1, 0),
                kv_storage: vec![],
                primary_activation: id(OUTPUT_ID),
                boundaries: BTreeMap::from([(id(OUTPUT_ID), ElementType::Bf16)]),
                states: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(ADJACENT_ROPE_BF16_OPERATION_ID),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: Some(ElementType::F32),
                    accumulation_type: Some(ElementType::F32),
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                }],
            }],
            vec![profile_id],
        )
    }
    fn semantic_program(
        &self,
        _: &Config,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        ModelProgram::new(
            self.family_id().clone(),
            vec![id(INPUT_ID), id(FREQUENCIES)],
            vec![ProgramBlock {
                id: "block.tensor.rotary".into(),
                nodes: vec![ProgramNode {
                    id: id("node.tensor.rotary"),
                    operation_id: id(ADJACENT_ROPE_BF16_OPERATION_ID),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: vec![id(INPUT_ID), id(FREQUENCIES)],
                    outputs: vec![id(OUTPUT_ID)],
                    attributes: BTreeMap::from([(id("head_dim"), SemanticValue::Unsigned(4))]),
                }],
            }],
            vec![],
            vec![],
            vec![id(OUTPUT_ID)],
        )
    }
    fn semantic_metadata(&self, _: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}

fn frequencies(host: &HostMemoryAccount, values: &[u32; 8]) -> ManagedHostTensor {
    let mut allocation = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 32)
        .unwrap();
    for (bytes, bits) in allocation.as_mut().chunks_exact_mut(4).zip(values) {
        bytes.copy_from_slice(&bits.to_le_bytes());
    }
    ManagedHostTensor::new(
        id(FREQUENCIES),
        input_spec(vec![2, 2, 2], ElementType::F32),
        allocation.freeze(),
    )
    .unwrap()
}

#[test]
#[ignore = "requires exclusive CUDA; managed adjacent RoPE numerical/indexing probes"]
fn managed_cuda_adjacent_rope_batches_heads_fma_and_signed_zero() {
    let config = Config { rows: 4, width: 8 };
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
            usable_memory_bytes: USABLE_BYTES,
            bootstrap_memory_bytes: 1 << 30,
            stream_memory_bytes: 64 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            opaque_memory_profile: "experimental-unqualified-rotary-smoke-v1".into(),
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
        Box::new(adjacent_rope_bf16_contract().unwrap()),
    ];
    let registry = OperationRuntimeRegistry::new(
        contracts,
        cuda_vnext_tensor_component_providers(&runtime).unwrap(),
    )
    .unwrap();
    assert!(cuda_vnext_tensor_component_providers(&runtime).is_err());
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.tensor.rotary.fixture"),
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
    let family = TypedFamilyRegistration::new(RotaryFamily)
        .prepare_with_profile(
            &serde_json::to_value(&config).unwrap(),
            &NumericalProfileId::new(CUDA_ADJACENT_ROPE_BF16_NUMERICAL_PROFILE_ID).unwrap(),
        )
        .unwrap();
    let mut options = ProgramPlanCompileOptions::new(BTreeMap::from([
        (id(INPUT_ID), config.spec()),
        (
            id(FREQUENCIES),
            ProgramTensorSpec {
                dimensions: vec![2, 2, 2],
                element_type: ElementType::F32,
                layout: ResolvedTensorLayout::Contiguous,
            },
        ),
    ]))
    .unwrap();
    options.retain_completion_value(id(OUTPUT_ID));
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
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            160,
            HostMemoryClassLimits {
                input_preprocess_bytes: 96,
                output_bytes: 64,
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
    // Expectations are literals derived before execution. The second case's
    // non-unit frequency is intentional: this contract accepts arbitrary FP32
    // frequency values; production position synthesis is a separate concern.
    let cases = [
        (
            "quadrants",
            [
                0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0, 0x40e0, 0x4100,
            ]
            .repeat(4),
            [0x3f800000, 0, 0, 0x3f800000, 0xbf800000, 0, 0, 0xbf800000],
            [
                0x3f80, 0x4000, 0xc080, 0x4040, 0x40a0, 0x40c0, 0xc100, 0x40e0, 0xbf80, 0xc000,
                0x4080, 0xc040, 0xc0a0, 0xc0c0, 0x4100, 0xc0e0,
            ]
            .repeat(2),
        ),
        (
            "fma_and_zero",
            [0x3f81, 0x3f80, 0, 0x8000, 0x3f81, 0x3f80, 0, 0x8000].repeat(4),
            [
                0x3f7fffff, 0x3f810000, 0x3f800000, 0, 0x3f7fffff, 0x3f810000, 0x3f800000, 0,
            ],
            [0xb381, 0x4001, 0, 0, 0xb381, 0x4001, 0, 0].repeat(4),
        ),
    ];
    for (name, x, freq, wanted) in cases {
        let outputs = executor
            .execute(
                id(format!("run.cuda.rotary.{name}")),
                id(format!("request.cuda.rotary.{name}")),
                vec![
                    super::normalization::managed_input(&host, INPUT_ID, vec![4, 8], &x),
                    frequencies(&host, &freq),
                ],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert!(host.owns(outputs[0].bytes()));
        let actual: Vec<_> = outputs[0]
            .bytes()
            .bytes()
            .chunks_exact(2)
            .map(|b| u16::from_le_bytes([b[0], b[1]]))
            .collect();
        assert_eq!(actual, wanted, "{name}");
        drop(outputs);
    }
    drop((executor, compiled, registry));
    close_resources(resources);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    drop(cuda_vnext_tensor_component_providers(&runtime).unwrap());
    eprintln!("CUDA managed adjacent RoPE: B2/S2/H2/D4 quadrant indexing, explicit-FMA counterexample and signed zero passed; position synthesis and full-model parity unqualified");
}
