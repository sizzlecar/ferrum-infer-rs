//! Exact bit transport through actual managed CUDA buffers, including NaN
//! payloads. Expected arrays are literal row mappings, not a copy oracle loop.
use super::normalization::managed_input;
use super::*;
use ferrum_kernels::backend::cuda::vnext_ops::{cuda_vnext_tensor_providers, CudaTensorOperation};

const SOURCE: &str = "value.transport.source";
const RIGHT: &str = "value.transport.right";
const GATHERED: &str = "value.transport.gathered";
const CONCATENATED: &str = "value.transport.concatenated";
const SLICED: &str = "value.transport.sliced";
const TRANSPORT_PROFILE: &str = "fixture.cuda.bitcopy.row_transport.bf16.v1";
const SELECTED: [CudaTensorOperation; 2] = [
    CudaTensorOperation::RowGather,
    CudaTensorOperation::BatchedRowConcat,
];

// Columns 0 and 3 are distinct canaries that must never enter the output.
// Column 2 of row 1 is a negative signaling NaN: decoding/re-encoding it as
// floating point would risk quieting its payload and fail exact-bit comparison.
const SOURCE_BITS: [u16; 24] = [
    0x1111, 0x0000, 0x8000, 0x2222, 0x3333, 0x7fc1, 0xff82, 0x4444, 0x5555, 0x0001, 0x8001, 0x6666,
    0x7777, 0x7f80, 0xff80, 0x8888, 0x9999, 0x3f80, 0xbf80, 0xaaaa, 0xbbbb, 0x4000, 0xc000, 0xcccc,
];
const RIGHT_BITS: [u16; 8] = [
    0x4040, 0xc040, 0x4080, 0xc080, 0x40a0, 0xc0a0, 0x40c0, 0xc0c0,
];
// Batch 0 source rows: 4,5,0,0,1,2. Batch 1: 4,5,3,3,4,5.
const GATHERED_BITS: [u16; 24] = [
    0x3f80, 0xbf80, 0x4000, 0xc000, 0x0000, 0x8000, 0x0000, 0x8000, 0x7fc1, 0xff82, 0x0001, 0x8001,
    0x3f80, 0xbf80, 0x4000, 0xc000, 0x7f80, 0xff80, 0x7f80, 0xff80, 0x3f80, 0xbf80, 0x4000, 0xc000,
];
// Each batch has its own two right-hand rows, rather than one flat append.
const CONCATENATED_BITS: [u16; 32] = [
    0x3f80, 0xbf80, 0x4000, 0xc000, 0x0000, 0x8000, 0x0000, 0x8000, 0x7fc1, 0xff82, 0x0001, 0x8001,
    0x4040, 0xc040, 0x4080, 0xc080, 0x3f80, 0xbf80, 0x4000, 0xc000, 0x7f80, 0xff80, 0x7f80, 0xff80,
    0x3f80, 0xbf80, 0x4000, 0xc000, 0x40a0, 0xc0a0, 0x40c0, 0xc0c0,
];
// Rows 5..8 within each concatenated batch, crossing the left/right seam.
const SLICED_BITS: [u16; 12] = [
    0x0001, 0x8001, 0x4040, 0xc040, 0x4080, 0xc080, 0x4000, 0xc000, 0x40a0, 0xc0a0, 0x40c0, 0xc0c0,
];

fn spec(rows: u64, columns: u64) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions: vec![rows, columns],
        element_type: ElementType::Bf16,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
fn inputs() -> BTreeMap<ProgramValueId, ProgramTensorSpec> {
    BTreeMap::from([(id(SOURCE), spec(6, 4)), (id(RIGHT), spec(4, 2))])
}
fn gather_attributes(
    lengths: Vec<i64>,
    starts: Vec<i64>,
    batch_strides: Vec<i64>,
    row_strides: Vec<i64>,
    column: u64,
) -> BTreeMap<AttributeId, SemanticValue> {
    BTreeMap::from([
        (id("batch"), SemanticValue::Unsigned(2)),
        (id("column_start"), SemanticValue::Unsigned(column)),
        (id("column_count"), SemanticValue::Unsigned(2)),
        (id("run_lengths"), SemanticValue::Integers(lengths)),
        (id("source_starts"), SemanticValue::Integers(starts)),
        (
            id("source_batch_strides"),
            SemanticValue::Integers(batch_strides),
        ),
        (
            id("source_row_strides"),
            SemanticValue::Integers(row_strides),
        ),
    ])
}

struct TransportFamily;
impl ModelFamilyProvider for TransportFamily {
    type Config = Config;
    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.cuda.tensor.row_transport.fixture"))
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.cuda.tensor.row_transport.fixture")])
    }
    fn validate_config_identity(&self, raw: &Value, config: &Config) -> Result<(), VNextError> {
        if config.rows != 6 || config.width != 4 || serde_json::to_value(config).unwrap() != *raw {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "shape".into(),
                reason: "fixture requires canonical source[6,4], right[4,2], batch2".into(),
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
        Ok(id("metadata.cuda.tensor.row_transport.fixture"))
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
        let profile_id = NumericalProfileId::new(TRANSPORT_PROFILE).unwrap();
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: profile_id.clone(),
                family_id: self.family_id().clone(),
                version: ContractVersion::new(1, 0),
                kv_storage: vec![],
                primary_activation: id(SLICED),
                boundaries: [GATHERED, CONCATENATED, SLICED]
                    .into_iter()
                    .map(|name| (id(name), ElementType::Bf16))
                    .collect(),
                states: vec![],
                operations: [
                    ROW_GATHER_BF16_OPERATION_ID,
                    BATCHED_ROW_CONCAT_BF16_OPERATION_ID,
                ]
                .into_iter()
                .map(|name| NumericalOperationContract {
                    operation_id: id(name),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: None,
                    accumulation_type: None,
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
                id: id(format!("node.transport.{name}")),
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
            vec![id(SOURCE), id(RIGHT)],
            vec![ProgramBlock {
                id: "block.tensor.row_transport".into(),
                nodes: vec![
                    node(
                        "gather",
                        ROW_GATHER_BF16_OPERATION_ID,
                        &[SOURCE],
                        GATHERED,
                        gather_attributes(
                            vec![2, 2, 2],
                            vec![4, 0, 1],
                            vec![0, 3, 3],
                            vec![1, 0, 1],
                            1,
                        ),
                    ),
                    node(
                        "concat",
                        BATCHED_ROW_CONCAT_BF16_OPERATION_ID,
                        &[GATHERED, RIGHT],
                        CONCATENATED,
                        BTreeMap::from([(id("batch"), SemanticValue::Unsigned(2))]),
                    ),
                    node(
                        "slice",
                        ROW_GATHER_BF16_OPERATION_ID,
                        &[CONCATENATED],
                        SLICED,
                        gather_attributes(vec![3], vec![5], vec![8], vec![1], 0),
                    ),
                ],
            }],
            vec![],
            vec![],
            [GATHERED, CONCATENATED, SLICED]
                .into_iter()
                .map(id)
                .collect(),
        )?
        .with_input_tensor_specs(inputs())
    }
    fn semantic_metadata(&self, _: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}

#[test]
#[ignore = "requires exclusive CUDA; real managed BF16 bit transport"]
fn managed_cuda_row_gather_batched_concat_and_slice_preserve_exact_bits() {
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
            usable_memory_bytes: USABLE_BYTES,
            bootstrap_memory_bytes: 1 << 30,
            stream_memory_bytes: 64 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 64,
            opaque_memory_profile: "experimental-unqualified-row-transport-smoke-v1".into(),
        })
        .unwrap(),
    );
    let registry = OperationRuntimeRegistry::new(
        vec![
            Box::new(row_gather_bf16_contract().unwrap()),
            Box::new(batched_row_concat_bf16_contract().unwrap()),
        ],
        cuda_vnext_tensor_providers(&runtime, &SELECTED).unwrap(),
    )
    .unwrap();
    assert!(cuda_vnext_tensor_providers(&runtime, &SELECTED).is_err());
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.tensor.transport.fixture"),
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
    let family = TypedFamilyRegistration::new(TransportFamily)
        .prepare_with_profile(
            &serde_json::to_value(Config { rows: 6, width: 4 }).unwrap(),
            &NumericalProfileId::new(TRANSPORT_PROFILE).unwrap(),
        )
        .unwrap();
    let mut options = ProgramPlanCompileOptions::new(inputs()).unwrap();
    // Transport output symbols are explicitly supplied, not guessed from a
    // matching input or filled by a model-name branch in the compiler.
    for (name, shape) in [
        (GATHERED, spec(12, 2)),
        (CONCATENATED, spec(16, 2)),
        (SLICED, spec(6, 2)),
    ] {
        options.insert_tensor_spec(id(name), shape).unwrap();
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
    let input_bytes = 2 * (SOURCE_BITS.len() + RIGHT_BITS.len());
    let output_bytes = 2 * (GATHERED_BITS.len() + CONCATENATED_BITS.len() + SLICED_BITS.len());
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
            id("run.cuda.transport.fixture"),
            id("request.cuda.transport.fixture"),
            vec![
                managed_input(&host, SOURCE, vec![6, 4], &SOURCE_BITS),
                managed_input(&host, RIGHT, vec![4, 2], &RIGHT_BITS),
            ],
        )
        .unwrap();
    assert_eq!(outputs.len(), 3);
    for output in &outputs {
        let expected: &[u16] = match output.value_id().as_str() {
            GATHERED => &GATHERED_BITS,
            CONCATENATED => &CONCATENATED_BITS,
            SLICED => &SLICED_BITS,
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
    eprintln!("CUDA managed B2 row-gather/concat/slice exact-bit mapping and retirement passed; no image or throughput claim");
}
