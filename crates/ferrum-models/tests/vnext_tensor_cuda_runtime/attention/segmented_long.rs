//! Actual long DiT SDPA geometry; synthetic payloads are not model-quality evidence.
//! The 2912-square edit trace has eight committed waves with packed rows 0..17746.
//! Two small independent documents surround that complete noncausal document.
use super::*;
use ferrum_kernels::backend::cuda::{
    vnext_ops::{
        cuda_vnext_cnn_f32_providers_with_segmented_sdpa, CudaTensorOperation,
        CUDA_CUDNN_SEGMENTED_ATTENTION_BF16_NUMERICAL_PROFILE_ID,
    },
    CudaCudnnConfig, CudnnSdpaPlanPolicy,
};

const LENGTHS: [usize; 3] = [11, 17746, 13];
const HEADS: usize = 56;
const DIM: usize = 128;
const COLUMNS: usize = HEADS * DIM;
const ROWS: usize = LENGTHS[0] + LENGTHS[1] + LENGTHS[2];
const BYTES: usize = ROWS * COLUMNS * 2;
const PROFILE: &str = CUDA_CUDNN_SEGMENTED_ATTENTION_BF16_NUMERICAL_PROFILE_ID;
const DEVICE_BYTES: u64 = 4 << 30;
const WORKSPACE_BYTES: u64 = 256 << 20;
// Same engineering limits as the existing newfused F64 oracle; all Q/K/V
// remain in [-1, 1]. These are not an assertion about vendor intermediates.
const ABS: f64 = 1.0 / 256.0;
const REL: f64 = 1.0 / 128.0;

#[derive(Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SegmentedConfig {}

struct SegmentedFamily;
impl ModelFamilyProvider for SegmentedFamily {
    type Config = SegmentedConfig;
    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.cuda.segmented_attention.long_fixture"))
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.cuda.segmented_attention.long_fixture")])
    }
    fn parse_config(&self, raw: &Value) -> Result<SegmentedConfig, VNextError> {
        let config = SegmentedConfig {};
        self.validate_config_identity(raw, &config)?;
        Ok(config)
    }
    fn validate_config_identity(&self, raw: &Value, _: &SegmentedConfig) -> Result<(), VNextError> {
        if !raw.as_object().is_some_and(|fields| fields.is_empty()) {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "config".into(),
                reason: "fixed long-shape fixture accepts only an empty object".into(),
            });
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        c: &SegmentedConfig,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        Ok(id("metadata.cuda.segmented_attention.long_fixture"))
    }
    fn weights(&self, _: &SegmentedConfig) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::None)
    }
    fn numerical_profiles(
        &self,
        _: &SegmentedConfig,
    ) -> Result<FamilyNumericalProfiles, VNextError> {
        let profile = NumericalProfileId::new(PROFILE).unwrap();
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: profile.clone(),
                family_id: self.family_id().clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: id(OUTPUT_ID),
                boundaries: BTreeMap::from([(id(OUTPUT_ID), ElementType::Bf16)]),
                kv_storage: vec![],
                states: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(SEGMENTED_ATTENTION_BF16_OPERATION_ID),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: None,
                    accumulation_type: None,
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                }],
            }],
            vec![profile],
        )
    }
    fn semantic_program(
        &self,
        _: &SegmentedConfig,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        // Exact rational representation of the effective FP32 1/sqrt(D).
        let scale = (1.0_f64 / (DIM as f64).sqrt()) as f32;
        let scale = CanonicalRational::new(
            (f64::from(scale) * (1_u64 << 30) as f64) as i64,
            1_u64 << 30,
        )?;
        ModelProgram::new(
            self.family_id().clone(),
            INPUTS.into_iter().map(id).collect(),
            vec![ProgramBlock {
                id: "block.segmented.long".into(),
                nodes: vec![ProgramNode {
                    id: id("node.segmented.long"),
                    operation_id: id(SEGMENTED_ATTENTION_BF16_OPERATION_ID),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs: INPUTS.into_iter().map(id).collect(),
                    outputs: vec![id(OUTPUT_ID)],
                    attributes: BTreeMap::from([
                        (id("head_dim"), SemanticValue::Unsigned(DIM as u64)),
                        (id("scale"), SemanticValue::Rational(scale)),
                        (
                            id("segment_lengths"),
                            SemanticValue::Integers(LENGTHS.map(|n| n as i64).to_vec()),
                        ),
                    ]),
                }],
            }],
            vec![],
            vec![],
            vec![id(OUTPUT_ID)],
        )
    }
    fn semantic_metadata(&self, _: &SegmentedConfig) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}

fn location(row: usize) -> (usize, usize) {
    if row < LENGTHS[0] {
        (0, row)
    } else if row < LENGTHS[0] + LENGTHS[1] {
        (1, row - LENGTHS[0])
    } else {
        (2, row - LENGTHS[0] - LENGTHS[1])
    }
}

/// The oracle consumes the exact BF16 values uploaded, generated without a
/// second full-sized Host copy. A marked tail makes truncation past 16K visible.
fn payload(port: usize, row: usize, head: usize, channel: usize, changed: bool) -> u16 {
    let (segment, local) = location(row);
    let value = match port {
        // Tail-dominant head: dropping keys >=16K, or emitting zero, is far
        // outside the unchanged tolerance. Other heads retain varying Q/K.
        0 if segment == 1 && head == 0 => 1.0,
        1 if segment == 1 && head == 0 => {
            if local >= 16384 {
                1.0
            } else {
                -1.0
            }
        }
        0 => (((local * 3 + head * 5 + channel * 7) % 31) as i32 - 15) as f32 / 32.0,
        1 => (((local * 7 + head * 3 + channel * 5) % 29) as i32 - 14) as f32 / 32.0,
        2 => {
            let base = (((local * 11 + head * 3 + channel * 7) % 23) as i32 - 11) as f32 / 32.0;
            base + if segment == 1 && local >= 16384 {
                0.5
            } else {
                -0.25
            }
        }
        _ => unreachable!(),
    };
    let value = if changed && segment != 1 {
        match port {
            1 => -value,
            2 => value + 0.5,
            _ => value,
        }
    } else {
        value
    };
    bf16::from_f32(value).to_bits()
}
fn scalar(bits: u16) -> f64 {
    f64::from(bf16::from_bits(bits).to_f32())
}

fn input_tensor(host: &HostMemoryAccount, port: usize, changed: bool) -> ManagedHostTensor {
    let mut bytes = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, BYTES)
        .unwrap();
    for (index, out) in bytes.as_mut().chunks_exact_mut(2).enumerate() {
        out.copy_from_slice(
            &payload(
                port,
                index / COLUMNS,
                index % COLUMNS / DIM,
                index % DIM,
                changed,
            )
            .to_le_bytes(),
        );
    }
    ManagedHostTensor::new(
        id(INPUTS[port]),
        input_spec(vec![ROWS as u64, COLUMNS as u64], ElementType::Bf16),
        bytes.freeze(),
    )
    .unwrap()
}

struct Sample {
    query: usize,
    head: usize,
    values: [f64; DIM],
}
fn reference(changed: bool) -> Vec<Sample> {
    let scale = f64::from((1.0_f64 / (DIM as f64).sqrt()) as f32);
    let mut samples = Vec::new();
    let mut start = 0;
    for (segment, length) in LENGTHS.into_iter().enumerate() {
        let queries = if segment == 1 {
            vec![0, 1, 16383, 16384, length - 2, length - 1]
        } else {
            vec![0, length - 1]
        };
        for local in queries {
            let query = start + local;
            for head in [0, HEADS / 2, HEADS - 1] {
                let q: [f64; DIM] =
                    std::array::from_fn(|c| scalar(payload(0, query, head, c, changed)));
                let mut weights = Vec::with_capacity(length);
                for key in start..start + length {
                    let score = q
                        .iter()
                        .enumerate()
                        .map(|(c, q)| q * scalar(payload(1, key, head, c, changed)))
                        .sum::<f64>()
                        * scale;
                    assert!(score.is_finite());
                    weights.push(score);
                }
                let maximum = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let mut denominator = 0.0;
                for w in &mut weights {
                    *w = libm::exp(*w - maximum);
                    denominator += *w;
                }
                if segment == 1 && head == 0 {
                    let tail_mass = weights[16384..].iter().sum::<f64>() / denominator;
                    assert!(
                        tail_mass > 0.999,
                        "fixture must attend strongly beyond 16K: {tail_mass}"
                    );
                }
                let mut values = [0.0; DIM];
                for (key, w) in (start..start + length).zip(weights) {
                    for (c, v) in values.iter_mut().enumerate() {
                        *v += w * scalar(payload(2, key, head, c, changed));
                    }
                }
                values.iter_mut().for_each(|v| *v /= denominator);
                if segment == 1 && head == 0 {
                    // A zero output fails by >0.45. Truncating before the tail
                    // produces an average near -0.25 instead of +0.5.
                    assert!(values.iter().all(|&v| v > 0.45));
                }
                samples.push(Sample {
                    query,
                    head,
                    values,
                });
            }
        }
        start += length;
    }
    samples
}
fn check(bytes: &[u8], expected: &[Sample]) {
    assert_eq!(bytes.len(), BYTES);
    assert!(bytes
        .chunks_exact(2)
        .all(|b| bf16::from_bits(u16::from_le_bytes([b[0], b[1]])).is_finite()));
    let mut maximum_error = 0.0_f64;
    for sample in expected {
        for (channel, &wanted) in sample.values.iter().enumerate() {
            let at = ((sample.query * HEADS + sample.head) * DIM + channel) * 2;
            let got = scalar(u16::from_le_bytes([bytes[at], bytes[at + 1]]));
            let error = (got - wanted).abs();
            maximum_error = maximum_error.max(error);
            assert!(
                error <= ABS + REL * wanted.abs(),
                "q={} h={} c={channel}: got={got}, F64={wanted}, error={error}",
                sample.query,
                sample.head
            );
        }
    }
    eprintln!("long segmented SDPA: {} sampled scalars, max_abs_error={maximum_error}; all {ROWS}x{COLUMNS} outputs finite; limits={ABS}+{REL}*abs(reference)", expected.len() * DIM);
}

#[test]
#[ignore = "requires exclusive CUDA and explicit FERRUM_TEST_CUDNN_LIBRARY_PATH for cuDNN 9.20"]
fn managed_cuda_segmented_attention_8mp_length_matches_sampled_f64() {
    let expected = [reference(false), reference(true)];
    let library_path = std::env::var_os("FERRUM_TEST_CUDNN_LIBRARY_PATH")
        .map(std::path::PathBuf::from)
        .expect("explicit cuDNN 9.20 library path");
    assert!(library_path.is_absolute());
    let cudnn = CudaCudnnConfig {
        library_path,
        dependency_libraries: std::env::var_os("FERRUM_TEST_CUDNN_DEPENDENCY_LIBRARIES")
            .map(|p| std::env::split_paths(&p).collect())
            .unwrap_or_default(),
        opaque_memory_bytes: 64 << 20,
        workspace_limit_bytes: WORKSPACE_BYTES,
        opaque_memory_profile: "long-segmented-attention-fixture-v1".into(),
    };
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
            usable_memory_bytes: DEVICE_BYTES,
            bootstrap_memory_bytes: 1 << 30,
            stream_memory_bytes: 64 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 64,
            opaque_memory_profile: "long-segmented-attention-fixture-v1".into(),
        })
        .unwrap(),
    );
    let providers = cuda_vnext_cnn_f32_providers_with_segmented_sdpa(
        &runtime,
        &cudnn,
        CudnnSdpaPlanPolicy {
            maximum_plans: 3,
            maximum_live_bindings: 3,
            opaque_memory_bytes: 128 << 20,
            workspace_limit_bytes: WORKSPACE_BYTES,
        },
        &[CudaTensorOperation::SegmentedAttention],
    )
    .unwrap();
    let selected: Vec<_> = providers
        .iter()
        .filter(|p| p.descriptor().operation_id().as_str() == SEGMENTED_ATTENTION_BF16_OPERATION_ID)
        .collect();
    assert_eq!(selected.len(), 1);
    assert_eq!(
        selected[0].descriptor().provider_id().as_str(),
        "provider.cuda.segmented_attention.bf16.cudnn920"
    );
    drop(selected);
    let registry = OperationRuntimeRegistry::new(
        vec![
            Box::new(conv2d_f32_contract().unwrap()),
            Box::new(segmented_attention_bf16_contract().unwrap()),
        ],
        providers,
    )
    .unwrap();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.segmented.long.fixture"),
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
    let family = TypedFamilyRegistration::new(SegmentedFamily)
        .prepare_with_profile(
            &serde_json::json!({}),
            &NumericalProfileId::new(PROFILE).unwrap(),
        )
        .unwrap();
    let spec = ProgramTensorSpec {
        dimensions: vec![ROWS as u64, COLUMNS as u64],
        element_type: ElementType::Bf16,
        layout: ResolvedTensorLayout::Contiguous,
    };
    let mut options = ProgramPlanCompileOptions::new(
        INPUTS
            .into_iter()
            .map(|name| (id(name), spec.clone()))
            .collect(),
    )
    .unwrap();
    options.retain_completion_value(id(OUTPUT_ID));
    let mut policy = policy(runtime.as_ref());
    policy.capacity_bytes = DEVICE_BYTES;
    let compiled =
        ProgramPlanCompiler::compile(&family, &catalog, &policy, &registry.planning(), &options)
            .unwrap();
    let executable = compiled.executable();
    let nodes = executable.execution_plan().payload().nodes();
    assert_eq!(nodes.len(), 1);
    assert_eq!(
        nodes[0].operation_id().as_str(),
        SEGMENTED_ATTENTION_BF16_OPERATION_ID
    );
    assert!(nodes[0].provider_resources().scratch().is_some());
    let resources = no_static(runtime.clone(), executable.execution_plan());
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            5 * BYTES,
            HostMemoryClassLimits {
                input_preprocess_bytes: 3 * BYTES,
                output_bytes: 2 * BYTES,
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
    let mut previous: Option<ManagedHostBytes> = None;
    for (iteration, expected) in expected.iter().enumerate() {
        let inputs = (0..3)
            .map(|port| input_tensor(&host, port, iteration != 0))
            .collect();
        let outputs = executor
            .execute(
                id(format!("run.segmented.long.{iteration}")),
                id(format!("request.segmented.long.{iteration}")),
                inputs,
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(
            outputs[0].tensor().dimensions(),
            [ROWS as u64, COLUMNS as u64]
        );
        assert!(host.owns(outputs[0].bytes()));
        let bytes = outputs[0].bytes().clone();
        check(bytes.bytes(), expected);
        if let Some(previous) = previous.take() {
            let range = LENGTHS[0] * COLUMNS * 2..(LENGTHS[0] + LENGTHS[1]) * COLUMNS * 2;
            assert_eq!(
                &bytes.bytes()[range.clone()],
                &previous.bytes()[range],
                "foreign segment K/V must not affect any long-document output"
            );
            assert_ne!(
                &bytes.bytes()[..LENGTHS[0] * COLUMNS * 2],
                &previous.bytes()[..LENGTHS[0] * COLUMNS * 2]
            );
        }
        drop(outputs);
        previous = Some(bytes);
        assert_eq!(host.usage().unwrap().allocated_bytes(), BYTES);
    }
    drop((executor, compiled, catalog, registry));
    close_resources(resources);
    // Output ownership is independent of the retired device/native plans.
    drop(runtime.create_stream().unwrap());
    drop(runtime);
    check(previous.as_ref().unwrap().bytes(), &expected[1]);
    drop(previous);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
}
