//! Real managed partial-RoPE execution against the independent H3 CPU oracle.
//! CUDA requires a configured host with admitted fixture capacity; these are operation checks,
//! not evidence of complete P-H3 image generation or Torch model parity.

use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use ferrum_models::p_h3_image::*;
use ferrum_runtime::*;
use half::bf16;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, OnceLock},
};

const INPUT: &str = "value.partial_rope.input";
const COS: &str = "value.partial_rope.cos";
const SIN: &str = "value.partial_rope.sin";
const OUTPUT: &str = "value.partial_rope.output";
const PROFILE: &str = "fixture.partial_rope.staged_bf16.v1";

fn id<T: TryFrom<String>>(value: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(value.into()).unwrap()
}
fn spec(rows: u64, columns: u64) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions: vec![rows, columns],
        element_type: ElementType::Bf16,
        layout: ResolvedTensorLayout::Contiguous,
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    rows: u64,
    heads: u64,
    head_dim: u64,
    rotary_dim: u64,
}
impl Config {
    fn inputs(&self) -> BTreeMap<ProgramValueId, ProgramTensorSpec> {
        BTreeMap::from([
            (id(INPUT), spec(self.rows, self.heads * self.head_dim)),
            (id(COS), spec(self.rows, self.rotary_dim)),
            (id(SIN), spec(self.rows, self.rotary_dim)),
        ])
    }
}
struct Family;
impl ModelFamilyProvider for Family {
    type Config = Config;
    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.test.partial_rope"))
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.test.partial_rope")])
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        config: &Config,
    ) -> Result<(), VNextError> {
        if serde_json::to_value(config).unwrap() != *raw
            || config.heads == 0
            || config.heads.checked_mul(config.head_dim).is_none()
        {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "invalid partial RoPE fixture geometry".into(),
            });
        }
        let specs = config.inputs();
        let resolved = |name| {
            let spec = &specs[&id::<ProgramValueId>(name)];
            ResolvedTensorSpec::new(
                spec.dimensions.clone(),
                spec.element_type,
                spec.layout.clone(),
            )
        };
        PartialHalfSplitRopeBf16Geometry::new(
            &resolved(INPUT)?,
            &resolved(COS)?,
            &resolved(SIN)?,
            &resolved(INPUT)?,
            config.head_dim,
            config.rotary_dim,
        )?;
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        config: &Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(id("metadata.test.partial_rope"))
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
                    operation_id: id(PARTIAL_HALF_SPLIT_ROPE_BF16_OPERATION_ID),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: Some(ElementType::F32),
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
        ModelProgram::new(
            self.family_id().clone(),
            vec![id(INPUT), id(COS), id(SIN)],
            vec![ProgramBlock {
                id: "block.partial_rope".into(),
                nodes: vec![ProgramNode {
                    id: id("node.partial_rope"),
                    operation_id: id(PARTIAL_HALF_SPLIT_ROPE_BF16_OPERATION_ID),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: vec![id(INPUT), id(COS), id(SIN)],
                    outputs: vec![id(OUTPUT)],
                    attributes: BTreeMap::from([
                        (id("head_dim"), SemanticValue::Unsigned(config.head_dim)),
                        (id("rotary_dim"), SemanticValue::Unsigned(config.rotary_dim)),
                    ]),
                }],
            }],
            vec![],
            vec![],
            vec![id(OUTPUT)],
        )?
        .with_input_tensor_specs(config.inputs())
    }
    fn semantic_metadata(&self, _: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}

fn h3_config(head: u64, rotary: u64) -> PH3ImageTransformerConfig {
    PH3ImageTransformerConfig::from_hf_json(&serde_json::to_vec(&serde_json::json!({
        "_class_name":"MiniMaxH3Transformer3DModel","attention_head_dim":head,"audio_in_channels":2,
        "ffn_dim":16,"final_norm_eps":1e-5,"freq_dim":8,"hidden_size":8,"in_channels":2,"norm_eps":1e-5,
        "num_attention_heads":2,"num_layers":1,"num_refiner_layers":1,"patch_size":[1,2,2],
        "qk_norm_eps":1e-5,"rope_freq_dim":rotary/6,"rope_theta":10000.,"text_dim":4,
        "time_embed_dim":4,"time_embed_hidden_dim":8})).unwrap()).unwrap()
}

fn fixture(head: u64, rotary: u64) -> (Config, Vec<u16>, Vec<u16>, Vec<u16>, Vec<u16>) {
    let transformer = h3_config(head, rotary);
    let image = PH3ImageReferenceImage::new(2, 2).unwrap();
    let requests = [
        PH3ImageRequestLayout::new(&transformer, &[PH3ImageModality::Text], image, &[]).unwrap(),
        PH3ImageRequestLayout::new(
            &transformer,
            &[
                PH3ImageModality::Text,
                PH3ImageModality::Image,
                PH3ImageModality::Text,
            ],
            image,
            &[],
        )
        .unwrap(),
    ];
    let batch = PH3ImagePackedBatchLayout::new(&requests).unwrap();
    let config = Config {
        rows: batch.rows() as u64,
        heads: 2,
        head_dim: head,
        rotary_dim: rotary,
    };
    let mut positions = vec![[0.; 3]; batch.rows()];
    batch.write_positions(&mut positions).unwrap();
    let rope = PH3ImageRotaryEmbedding::new(&transformer).unwrap();
    let mut cosine = vec![0.; rope.frequency_elements(batch.rows()).unwrap()];
    let mut sine = cosine.clone();
    rope.write_frequencies(&positions, &mut cosine, &mut sine)
        .unwrap();
    // The device operation receives BF16 frequencies; quantize before the
    // independent oracle so neither side substitutes F32 trigonometry.
    for value in cosine.iter_mut().chain(&mut sine) {
        *value = bf16::from_f32(*value).to_f32();
    }
    cosine[..rotary as usize].fill(0.50390625);
    sine[..rotary as usize].fill(0.50390625);
    let columns = (head * 2) as usize;
    let mut input: Vec<_> = (0..batch.rows() * columns)
        .map(|index| bf16::from_f32((index % 29) as f32 * 0.125 - 1.75).to_bits())
        .collect();
    input[0] = bf16::from_f32(1.0078125).to_bits();
    input[rotary as usize / 2] = bf16::from_f32(1.015625).to_bits();
    let mut expected = vec![0; input.len()];
    rope.apply_bf16(batch.rows(), 2, &input, &cosine, &sine, &mut expected)
        .unwrap();
    assert_eq!(bf16::from_bits(expected[0]).to_f32(), -0.00390625);
    // Packed requests must equal separately rotated documents, including the
    // positions that restart at the second request's own text/media clock.
    for request in 0..batch.request_count() {
        let range = batch.request_range(request).unwrap();
        let mut separate = vec![0; range.len() * columns];
        rope.apply_bf16(
            range.len(),
            2,
            &input[range.start * columns..range.end * columns],
            &cosine[range.start * rotary as usize..range.end * rotary as usize],
            &sine[range.start * rotary as usize..range.end * rotary as usize],
            &mut separate,
        )
        .unwrap();
        assert_eq!(
            &expected[range.start * columns..range.end * columns],
            separate.as_slice()
        );
    }
    for row in 0..batch.rows() {
        for h in 0..2 {
            let index = row * columns + h * head as usize + rotary as usize;
            input[index] = if h == 0 { 0x7fc1 } else { 0x8000 };
            expected[index] = input[index];
        }
    }
    let encode = |values: &[f32]| {
        values
            .iter()
            .map(|value| bf16::from_f32(*value).to_bits())
            .collect()
    };
    (config, input, encode(&cosine), encode(&sine), expected)
}

fn managed(
    host: &HostMemoryAccount,
    name: &str,
    spec: &ProgramTensorSpec,
    bits: &[u16],
) -> ManagedHostTensor {
    let mut allocation = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, bits.len() * 2)
        .unwrap();
    for (destination, value) in allocation.as_mut().chunks_exact_mut(2).zip(bits) {
        destination.copy_from_slice(&value.to_le_bytes());
    }
    ManagedHostTensor::new(
        id(name),
        ResolvedTensorSpec::new(
            spec.dimensions.clone(),
            spec.element_type,
            spec.layout.clone(),
        )
        .unwrap(),
        allocation.freeze(),
    )
    .unwrap()
}

fn run<R: DeviceRuntime>(
    runtime: Arc<R>,
    registry: &OperationRuntimeRegistry<R>,
    catalog: &CapabilityCatalog,
    config: &Config,
    input: &[u16],
    cosine: &[u16],
    sine: &[u16],
    expected: &[u16],
) {
    let family = TypedFamilyRegistration::new(Family)
        .prepare_with_profile(&serde_json::to_value(config).unwrap(), &id(PROFILE))
        .unwrap();
    let options = ProgramPlanCompileOptions::new(config.inputs()).unwrap();
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
    let materializers = WeightMaterializerRegistry::new(vec![]).unwrap();
    let component = CompiledTensorComponent::compile(
        family,
        registry,
        catalog,
        &policy,
        &materializers,
        &options,
    )
    .unwrap();
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            1 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 1 << 18,
                transfer_staging_bytes: 1 << 18,
                output_bytes: 1 << 18,
                weight_backing_bytes: 1 << 18,
            },
        )
        .unwrap(),
    );
    let mut session = component
        .initialize(
            runtime,
            host.clone(),
            ComponentWeightSource::None,
            ComponentInitializationConfig {
                run_id: id("run.partial_rope.init"),
                request_id: id("request.partial_rope.init"),
                transaction_id: id("transaction.partial_rope.init"),
                static_policy: StaticInitializationPolicy::new(1 << 20, 8).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap();
    let inputs = config.inputs();
    let output = session
        .execute(
            id("run.partial_rope.execute"),
            id("request.partial_rope.execute"),
            vec![
                managed(&host, INPUT, &inputs[&id::<ProgramValueId>(INPUT)], input),
                managed(&host, COS, &inputs[&id::<ProgramValueId>(COS)], cosine),
                managed(&host, SIN, &inputs[&id::<ProgramValueId>(SIN)], sine),
            ],
        )
        .unwrap();
    assert_eq!(output.len(), 1);
    assert!(host.owns(output[0].bytes()));
    let actual: Vec<_> = output[0]
        .bytes()
        .bytes()
        .chunks_exact(2)
        .map(|pair| u16::from_le_bytes([pair[0], pair[1]]))
        .collect();
    assert_eq!(actual, expected);
    session
        .close()
        .unwrap_or_else(|failure| panic!("component close failed: {:?}", failure.failure()));
    assert_eq!(output[0].bytes().bytes().len(), expected.len() * 2);
    drop(output);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn managed_reference_partial_rope_matches_h3_oracle_and_retires_owned_storage() {
    let composition =
        ReferenceVNextComposition::create(id("device.reference.partial_rope.component")).unwrap();
    for (head, rotary) in [(8, 6), (128, 96)] {
        let (config, input, cosine, sine, expected) = fixture(head, rotary);
        run(
            composition.runtime().clone(),
            composition.registry(),
            composition.catalog(),
            &config,
            &input,
            &cosine,
            &sine,
            &expected,
        );
        assert_eq!(composition.runtime().snapshot().live_allocations, 0);
    }
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires configured CUDA host with admitted fixture capacity"]
fn managed_cuda_partial_rope_matches_h3_oracle_with_tail_payloads() {
    use ferrum_kernels::backend::cuda::{
        vnext_ops::{cuda_tensor_runtime_config, cuda_vnext_tensor_providers, CudaTensorOperation},
        vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
    };
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
            usable_memory_bytes: 2 << 30,
            bootstrap_memory_bytes: 1 << 30,
            stream_memory_bytes: 64 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 64,
            opaque_memory_profile: "partial-rope-operation-validation".into(),
        })
        .unwrap(),
    );
    let registry = OperationRuntimeRegistry::new(
        vec![Box::new(partial_half_split_rope_bf16_contract().unwrap())],
        cuda_vnext_tensor_providers(&runtime, &[CudaTensorOperation::PartialHalfSplitRope])
            .unwrap(),
    )
    .unwrap();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.partial_rope.fixture"),
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
    for (head, rotary) in [(8, 6), (128, 96)] {
        let (config, input, cosine, sine, expected) = fixture(head, rotary);
        run(
            runtime.clone(),
            &registry,
            &catalog,
            &config,
            &input,
            &cosine,
            &sine,
            &expected,
        );
    }
}
