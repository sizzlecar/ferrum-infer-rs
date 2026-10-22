//! Analytic BF16 text primitives through real managed CUDA execution. This
//! covers selected arithmetic/mapping cases, not a text encoder or HF parity.
use super::normalization::managed_input;
use super::*;
use ferrum_kernels::backend::cuda::vnext_ops::{cuda_vnext_tensor_providers, CudaTensorOperation};

const QUERY: &str = "value.text.query";
const KEY: &str = "value.text.key";
const VALUE: &str = "value.text.value";
const UPDATE: &str = "value.text.update";
const ROTARY_INPUT: &str = "value.text.rotary_input";
const COSINE: &str = "value.text.cosine";
const SINE: &str = "value.text.sine";
const ROTATED: &str = "value.text.rotated";
const ATTENDED: &str = "value.text.attended";
const ADDED: &str = "value.text.added";
const TEXT_PROFILE: &str = "fixture.cuda.text_primitives.bf16.v1";
const SELECTED: [CudaTensorOperation; 3] = [
    CudaTensorOperation::PlainResidualAdd,
    CudaTensorOperation::HalfSplitRope,
    CudaTensorOperation::StructuredGqaAttention,
];
const OUTPUTS: [&str; 3] = [ROTATED, ATTENDED, ADDED];

// B2/S3/Hq4/Hkv2/D2. KV is physically [6,4], never [6,8]. Each KV head has
// distinct values so q_head % Hkv is distinguishable from q_head / group_size.
// Batch 0 key 1 and batch 1 key 0 are masked NaN rows. Padding is key-only:
// query 1 of batch 0 must still return values from key 0.
const VALUE_BITS: [u16; 24] = [
    0x3f80, 0x4000, 0x4120, 0x41a0, 0x7fc1, 0xff80, 0x7f80, 0xffc1, 0x4040, 0x4080, 0x41f0, 0x4220,
    0x7fc1, 0xff80, 0x7f80, 0xffc1, 0x40a0, 0x40c0, 0x4248, 0x4270, 0x40e0, 0x4100, 0x428c, 0x42a0,
];
const ATTENDED_BITS: [u16; 48] = [
    0x3f80, 0x4000, 0x3f80, 0x4000, 0x4120, 0x41a0, 0x4120, 0x41a0, 0x3f80, 0x4000, 0x3f80, 0x4000,
    0x4120, 0x41a0, 0x4120, 0x41a0, 0x4000, 0x4040, 0x4000, 0x4040, 0x41a0, 0x41f0, 0x41a0, 0x41f0,
    0, 0, 0, 0, 0, 0, 0, 0, 0x40a0, 0x40c0, 0x40a0, 0x40c0, 0x4248, 0x4270, 0x4248, 0x4270, 0x40c0,
    0x40e0, 0x40c0, 0x40e0, 0x4270, 0x428c, 0x4270, 0x428c,
];
// Add one everywhere except the first two entries: 1+1/256 ties to 1,
// 2+3/128 ties upward to 2.03125. All other sums are exact small integers.
const ADDED_BITS: [u16; 48] = [
    0x3f80, 0x4002, 0x4000, 0x4040, 0x4130, 0x41a8, 0x4130, 0x41a8, 0x4000, 0x4040, 0x4000, 0x4040,
    0x4130, 0x41a8, 0x4130, 0x41a8, 0x4040, 0x4080, 0x4040, 0x4080, 0x41a8, 0x41f8, 0x41a8, 0x41f8,
    0x3f80, 0x3f80, 0x3f80, 0x3f80, 0x3f80, 0x3f80, 0x3f80, 0x3f80, 0x40c0, 0x40e0, 0x40c0, 0x40e0,
    0x424c, 0x4274, 0x424c, 0x4274, 0x40e0, 0x4100, 0x40e0, 0x4100, 0x4274, 0x428e, 0x4274, 0x428e,
];
const ROTARY_BITS: [u16; 48] = [
    0x3f81, 0x3f80, 0x3f82, 0x4000, 0x3f81, 0x4040, 0x3f82, 0x4080, 0x8000, 0x3f80, 0x4000, 0x4040,
    0x8000, 0xbf80, 0xc000, 0xc040, 0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0, 0x40e0, 0x4100,
    0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0, 0x40e0, 0x4100, 0x3f80, 0x4000, 0x4040, 0x4080,
    0x40a0, 0x40c0, 0x40e0, 0x4100, 0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0, 0x40e0, 0x4100,
];
const COSINE_BITS: [u16; 24] = [
    0x3f81, 0, 0, 0x3f80, 0x3f80, 0x3f80, 0x3f80, 0x3f80, 0, 0, 0, 0, 0, 0, 0, 0, 0x3f80, 0x3f80,
    0x3f80, 0x3f80, 0, 0, 0, 0,
];
const SINE_BITS: [u16; 24] = [
    0x3f80, 0x3f80, 0, 0, 0, 0, 0, 0, 0x3f80, 0x3f80, 0x3f80, 0x3f80, 0xbf80, 0xbf80, 0xbf80,
    0xbf80, 0, 0, 0, 0, 0, 0, 0, 0,
];
// First entry: B(B((129/128)^2) - 130/128) = +0. Omitting the first
// product's BF16 boundary would return 1/16384 (0x3880), not zero.
// Row 0 also has different cosine/sine halves; later rows distinguish
// per-row frequencies, half-split pairing, and signed-zero additions.
const ROTATED_BITS: [u16; 48] = [
    0, 0xc000, 0, 0x4000, 0, 0xc080, 0, 0x4080, 0x8000, 0x3f80, 0x4000, 0x4040, 0, 0xbf80, 0xc000,
    0xc040, 0xc040, 0xc080, 0x3f80, 0x4000, 0xc0e0, 0xc100, 0x40a0, 0x40c0, 0x4040, 0x4080, 0xbf80,
    0xc000, 0x40e0, 0x4100, 0xc0a0, 0xc0c0, 0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0, 0x40e0,
    0x4100, 0, 0, 0, 0, 0, 0, 0, 0,
];

fn spec(columns: u64) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions: vec![6, columns],
        element_type: ElementType::Bf16,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
fn input_specs() -> BTreeMap<ProgramValueId, ProgramTensorSpec> {
    [
        (QUERY, 8),
        (KEY, 4),
        (VALUE, 4),
        (UPDATE, 8),
        (ROTARY_INPUT, 8),
        (COSINE, 4),
        (SINE, 4),
    ]
    .into_iter()
    .map(|(name, columns)| (id(name), spec(columns)))
    .collect()
}
fn attention_attributes() -> BTreeMap<AttributeId, SemanticValue> {
    let mut attributes = BTreeMap::from([
        (id("sequence_length"), SemanticValue::Unsigned(3)),
        (id("head_dim"), SemanticValue::Unsigned(2)),
        (
            id("scale"),
            SemanticValue::Rational(CanonicalRational::new(1, 1).unwrap()),
        ),
    ]);
    for (name, values) in [
        ("segment_lengths", vec![3]),
        ("segment_bidirectional", vec![0]),
        ("key_run_counts", vec![3, 2]),
        ("key_run_lengths", vec![1, 1, 1, 1, 2]),
        ("key_run_valid", vec![1, 0, 1, 0, 1]),
    ] {
        attributes.insert(id(name), SemanticValue::Integers(values));
    }
    attributes
}

struct TextFamily;
impl ModelFamilyProvider for TextFamily {
    type Config = Config;
    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.cuda.text_primitives.fixture"))
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.cuda.text_primitives.fixture")])
    }
    fn validate_config_identity(&self, raw: &Value, config: &Config) -> Result<(), VNextError> {
        if config.rows != 6 || config.width != 8 || serde_json::to_value(config).unwrap() != *raw {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "shape".into(),
                reason: "text primitive fixture requires canonical six-row shape".into(),
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
        Ok(id("metadata.cuda.text_primitives.fixture"))
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
        let profile_id = NumericalProfileId::new(TEXT_PROFILE).unwrap();
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: profile_id.clone(),
                family_id: self.family_id().clone(),
                version: ContractVersion::new(1, 0),
                kv_storage: vec![],
                primary_activation: id(ADDED),
                boundaries: OUTPUTS
                    .into_iter()
                    .map(|name| (id(name), ElementType::Bf16))
                    .collect(),
                states: vec![],
                operations: [
                    HALF_SPLIT_ROPE_BF16_OPERATION_ID,
                    STRUCTURED_GQA_ATTENTION_BF16_OPERATION_ID,
                    PLAIN_RESIDUAL_ADD_BF16_OPERATION_ID,
                ]
                .into_iter()
                .map(|name| NumericalOperationContract {
                    operation_id: id(name),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: (name != PLAIN_RESIDUAL_ADD_BF16_OPERATION_ID)
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
        let node =
            |name: &str, operation: &str, inputs: &[&str], output: &str, attributes| ProgramNode {
                id: id(format!("node.text.{name}")),
                operation_id: id(operation),
                required_version: ContractVersion::new(1, 0),
                work: ProgramNodeWorkSpec::Fixed,
                inputs: inputs.iter().map(|name| id(*name)).collect(),
                outputs: vec![id(output)],
                attributes,
            };
        ModelProgram::new(
            self.family_id().clone(),
            input_specs().into_keys().collect(),
            vec![ProgramBlock {
                id: "block.text_primitives".into(),
                nodes: vec![
                    node(
                        "rotary",
                        HALF_SPLIT_ROPE_BF16_OPERATION_ID,
                        &[ROTARY_INPUT, COSINE, SINE],
                        ROTATED,
                        BTreeMap::from([(id("head_dim"), SemanticValue::Unsigned(4))]),
                    ),
                    node(
                        "gqa",
                        STRUCTURED_GQA_ATTENTION_BF16_OPERATION_ID,
                        &[QUERY, KEY, VALUE],
                        ATTENDED,
                        attention_attributes(),
                    ),
                    node(
                        "add",
                        PLAIN_RESIDUAL_ADD_BF16_OPERATION_ID,
                        &[ATTENDED, UPDATE],
                        ADDED,
                        BTreeMap::new(),
                    ),
                ],
            }],
            vec![],
            vec![],
            OUTPUTS.into_iter().map(id).collect(),
        )?
        .with_input_tensor_specs(input_specs())
    }
    fn semantic_metadata(&self, _: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}

#[test]
#[ignore = "requires exclusive CUDA; managed BF16 text primitive arithmetic and GQA mapping"]
fn managed_cuda_half_split_rope_plain_residual_and_grouped_query_attention() {
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
            usable_memory_bytes: USABLE_BYTES,
            bootstrap_memory_bytes: 1 << 30,
            stream_memory_bytes: 64 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 64,
            opaque_memory_profile: "experimental-unqualified-text-primitives-smoke-v1".into(),
        })
        .unwrap(),
    );
    let registry = OperationRuntimeRegistry::new(
        vec![
            Box::new(plain_residual_add_bf16_contract().unwrap()),
            Box::new(half_split_rope_bf16_contract().unwrap()),
            Box::new(structured_gqa_attention_bf16_contract().unwrap()),
        ],
        cuda_vnext_tensor_providers(&runtime, &SELECTED).unwrap(),
    )
    .unwrap();
    assert!(cuda_vnext_tensor_providers(&runtime, &SELECTED).is_err());
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.text_primitives.fixture"),
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
    let family = TypedFamilyRegistration::new(TextFamily)
        .prepare_with_profile(
            &serde_json::to_value(Config { rows: 6, width: 8 }).unwrap(),
            &NumericalProfileId::new(TEXT_PROFILE).unwrap(),
        )
        .unwrap();
    let mut options = ProgramPlanCompileOptions::new(input_specs()).unwrap();
    for name in OUTPUTS {
        options.insert_tensor_spec(id(name), spec(8)).unwrap();
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
    let resources = no_static(runtime.clone(), executable.execution_plan());
    let mut query = [0_u16; 48];
    query[24..32].fill(0x7fc1); // Empty visible set must not read this query.
    let mut key = [0_u16; 24];
    key[4..8].fill(0x7f80);
    key[12..16].fill(0x7fc1);
    let mut update = [0x3f80_u16; 48];
    update[0] = 0x3b80; // 1/256
    update[1] = 0x3cc0; // 3/128
    let input_bytes = 2
        * (query.len()
            + key.len()
            + VALUE_BITS.len()
            + update.len()
            + ROTARY_BITS.len()
            + COSINE_BITS.len()
            + SINE_BITS.len());
    let output_bytes = 2 * (ROTATED_BITS.len() + ATTENDED_BITS.len() + ADDED_BITS.len());
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
    let inputs = [
        (QUERY, 8, &query[..]),
        (KEY, 4, &key[..]),
        (VALUE, 4, &VALUE_BITS[..]),
        (UPDATE, 8, &update[..]),
        (ROTARY_INPUT, 8, &ROTARY_BITS[..]),
        (COSINE, 4, &COSINE_BITS[..]),
        (SINE, 4, &SINE_BITS[..]),
    ]
    .into_iter()
    .map(|(name, columns, bits)| managed_input(&host, name, vec![6, columns], bits))
    .collect();
    let outputs = executor
        .execute(
            id("run.cuda.text_primitives.fixture"),
            id("request.cuda.text_primitives.fixture"),
            inputs,
        )
        .unwrap();
    assert_eq!(outputs.len(), 3);
    for output in &outputs {
        let expected = match output.value_id().as_str() {
            ROTATED => &ROTATED_BITS,
            ATTENDED => &ATTENDED_BITS,
            ADDED => &ADDED_BITS,
            name => panic!("unexpected output {name}"),
        };
        assert!(host.owns(output.bytes()));
        assert_eq!(output.bytes().len(), 2 * expected.len());
        for (index, (pair, expected)) in output
            .bytes()
            .bytes()
            .chunks_exact(2)
            .zip(expected)
            .enumerate()
        {
            assert_eq!(
                u16::from_le_bytes([pair[0], pair[1]]),
                *expected,
                "{} at {index}",
                output.value_id()
            );
        }
    }
    drop((outputs, executor, compilation, registry));
    close_resources(resources);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    drop(cuda_vnext_tensor_providers(&runtime, &SELECTED).unwrap());
    eprintln!("CUDA BF16 staged half-split RoPE, plain residual, B2 causal GQA with original KV extent, masked rows and retirement passed; no full encoder or deployment parity claim");
}
