//! Real managed CUDA probes for the spatial decoder contracts. Selected exact
//! fixtures are not a full VAE, SDPA parity, or capacity qualification.
use super::normalization::managed_input;
use super::*;
use ferrum_kernels::backend::cuda::{
    vnext_ops::{
        cuda_vnext_tensor_providers, cuda_vnext_tensor_providers_with_cudnn, CudaTensorOperation,
    },
    CudaCudnnConfig,
};

#[path = "spatial_decoder/channel_norm.rs"]
mod channel_norm;

const PROFILE: &str = "fixture.cuda.spatial_decoder.bf16.v1";
const WORKSPACE_BYTES: u64 = 1 << 20;
const WIDE: &str = "value.decoder.wide_qkv";
const SMALL: &str = "value.decoder.small_qkv";
const NORM: &str = "value.decoder.norm_input";
const GAMMA: &str = "value.decoder.gamma";
const GATHER: &str = "value.decoder.gather_input";
const X: &str = "value.decoder.conv_input";
const WEIGHT: &str = "value.decoder.conv_weight";
const BIAS: &str = "value.decoder.conv_bias";
const WIDE_OUT: &str = "value.decoder.wide_output";
const SMALL_OUT: &str = "value.decoder.small_output";
const NORM_OUT: &str = "value.decoder.norm_output";
const GATHER_OUT: &str = "value.decoder.gather_output";
const IMPLICIT: &str = "value.decoder.implicit_output";
const EXPLICIT: &str = "value.decoder.explicit_output";
const SELECTED: [CudaTensorOperation; 3] = [
    CudaTensorOperation::SpatialAttention,
    CudaTensorOperation::ChannelL2NormImage,
    CudaTensorOperation::SpatialUpsampleGather,
];

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(tag = "probe", rename_all = "snake_case")]
enum Probe {
    OddChannels,
    AlternatingRows,
    Padding,
    ChannelNorm {
        channels: u64,
        batch: u64,
        width: u64,
    },
}
fn spec(d: &[u64]) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions: d.to_vec(),
        element_type: ElementType::Bf16,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
impl Probe {
    fn inputs(self) -> BTreeMap<ProgramValueId, ProgramTensorSpec> {
        if let Self::ChannelNorm {
            channels,
            batch,
            width,
        } = self
        {
            return BTreeMap::from([
                (id(NORM), spec(&[batch, channels, 1, width])),
                (id(GAMMA), spec(&[channels, 1, 1])),
            ]);
        }
        let entries: Vec<(&str, &[u64])> = if matches!(self, Self::Padding) {
            vec![(X, &[1, 1, 2, 3]), (WEIGHT, &[1, 1, 3, 3]), (BIAS, &[1])]
        } else {
            vec![
                (WIDE, &[1, 3456, 1, 2]),
                (SMALL, &[1, 9, 1, 2]),
                (NORM, &[1, 4, 1, 2]),
                (GAMMA, &[4, 1, 1]),
                (GATHER, &[2, 2, 1, 2]),
            ]
        };
        entries
            .into_iter()
            .map(|(name, d)| (id(name), spec(d)))
            .collect()
    }
    fn outputs(self) -> Vec<(&'static str, ProgramTensorSpec)> {
        if let Self::ChannelNorm {
            channels,
            batch,
            width,
        } = self
        {
            return vec![(NORM_OUT, spec(&[batch, channels, 1, width]))];
        }
        if matches!(self, Self::Padding) {
            vec![
                (IMPLICIT, spec(&[1, 1, 2, 3])),
                (EXPLICIT, spec(&[1, 1, 2, 3])),
            ]
        } else {
            vec![
                (WIDE_OUT, spec(&[1, 1152, 1, 2])),
                (SMALL_OUT, spec(&[1, 3, 1, 2])),
                (NORM_OUT, spec(&[1, 4, 1, 2])),
                (GATHER_OUT, spec(&[2, 1, 2, 4])),
            ]
        }
    }
    fn operations(self) -> Vec<&'static str> {
        if matches!(self, Self::ChannelNorm { .. }) {
            return vec![CHANNEL_L2_NORM_IMAGE_BF16_OPERATION_ID];
        }
        if matches!(self, Self::Padding) {
            vec![CONV2D_BF16_OPERATION_ID]
        } else {
            vec![
                SPATIAL_ATTENTION_BF16_OPERATION_ID,
                CHANNEL_L2_NORM_IMAGE_BF16_OPERATION_ID,
                SPATIAL_UPSAMPLE_GATHER_BF16_OPERATION_ID,
            ]
        }
    }
    fn nodes(self) -> Vec<ProgramNode> {
        let node =
            |name: &str, operation: &str, inputs: &[&str], output: &str, attributes| ProgramNode {
                id: id(format!("node.decoder.{name}")),
                operation_id: id(operation),
                required_version: ContractVersion::new(1, 0),
                work: ProgramNodeWorkSpec::Fixed,
                inputs: inputs.iter().map(|n| id(*n)).collect(),
                outputs: vec![id(output)],
                attributes,
            };
        let scale = || {
            BTreeMap::from([(
                id("scale"),
                SemanticValue::Rational(CanonicalRational::new(1, 1).unwrap()),
            )])
        };
        if matches!(self, Self::ChannelNorm { .. }) {
            return vec![node(
                "norm",
                CHANNEL_L2_NORM_IMAGE_BF16_OPERATION_ID,
                &[NORM, GAMMA],
                NORM_OUT,
                BTreeMap::from([
                    (
                        id("scale"),
                        SemanticValue::Rational(CanonicalRational::new(3, 2).unwrap()),
                    ),
                    (
                        id("epsilon"),
                        SemanticValue::Rational(
                            CanonicalRational::new(1, 1_000_000_000_000).unwrap(),
                        ),
                    ),
                ]),
            )];
        }
        if matches!(self, Self::Padding) {
            return [
                ("implicit", IMPLICIT, "implicit_zero"),
                ("explicit", EXPLICIT, "explicit_zero"),
            ]
            .into_iter()
            .map(|(name, output, mode)| {
                node(
                    name,
                    CONV2D_BF16_OPERATION_ID,
                    &[X, WEIGHT, BIAS],
                    output,
                    BTreeMap::from([
                        (id("padding"), SemanticValue::Integers(vec![1; 4])),
                        (id("stride"), SemanticValue::Integers(vec![1; 2])),
                        (id("dilation"), SemanticValue::Integers(vec![1; 2])),
                        (id("groups"), SemanticValue::Unsigned(1)),
                        (id("padding_mode"), SemanticValue::Text(mode.into())),
                    ]),
                )
            })
            .collect();
        }
        let mut norm = scale();
        norm.insert(
            id("scale"),
            SemanticValue::Rational(CanonicalRational::new(2, 1).unwrap()),
        );
        norm.insert(
            id("epsilon"),
            SemanticValue::Rational(CanonicalRational::new(1, 1_000_000_000_000).unwrap()),
        );
        vec![
            node(
                "wide",
                SPATIAL_ATTENTION_BF16_OPERATION_ID,
                &[WIDE],
                WIDE_OUT,
                scale(),
            ),
            node(
                "small",
                SPATIAL_ATTENTION_BF16_OPERATION_ID,
                &[SMALL],
                SMALL_OUT,
                scale(),
            ),
            node(
                "norm",
                CHANNEL_L2_NORM_IMAGE_BF16_OPERATION_ID,
                &[NORM, GAMMA],
                NORM_OUT,
                norm,
            ),
            node(
                "gather",
                SPATIAL_UPSAMPLE_GATHER_BF16_OPERATION_ID,
                &[GATHER],
                GATHER_OUT,
                BTreeMap::from([
                    (id("scale_h"), SemanticValue::Unsigned(2)),
                    (id("scale_w"), SemanticValue::Unsigned(2)),
                    (id("channel_stride"), SemanticValue::Unsigned(2)),
                    (
                        id("channel_offsets"),
                        SemanticValue::Integers(if matches!(self, Self::OddChannels) {
                            vec![1; 4]
                        } else {
                            vec![0, 0, 1, 1]
                        }),
                    ),
                ]),
            ),
        ]
    }
}
struct DecoderFamily;

#[test]
fn typed_decoder_probes_prepare_without_device_creation() {
    for probe in [Probe::OddChannels, Probe::AlternatingRows, Probe::Padding] {
        TypedFamilyRegistration::new(DecoderFamily)
            .prepare_with_profile(
                &serde_json::to_value(probe).unwrap(),
                &NumericalProfileId::new(PROFILE).unwrap(),
            )
            .unwrap();
    }
}

impl ModelFamilyProvider for DecoderFamily {
    type Config = Probe;
    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.cuda.spatial_decoder.fixture"))
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.cuda.spatial_decoder.fixture")])
    }
    fn validate_config_identity(&self, raw: &Value, c: &Probe) -> Result<(), VNextError> {
        if serde_json::to_value(c).unwrap() != *raw {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "probe".into(),
                reason: "requires canonical typed probe".into(),
            });
        }
        Ok(())
    }
    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        c: &Probe,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        Ok(id("metadata.cuda.spatial_decoder.fixture"))
    }
    fn parse_config(&self, raw: &Value) -> Result<Probe, VNextError> {
        let c =
            serde_json::from_value(raw.clone()).map_err(|e| VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "probe".into(),
                reason: e.to_string(),
            })?;
        self.validate_config_identity(raw, &c)?;
        Ok(c)
    }
    fn weights(&self, _: &Probe) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::None)
    }
    fn numerical_profiles(&self, c: &Probe) -> Result<FamilyNumericalProfiles, VNextError> {
        let profile = NumericalProfileId::new(PROFILE).unwrap();
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: profile.clone(),
                family_id: self.family_id().clone(),
                version: ContractVersion::new(1, 0),
                kv_storage: vec![],
                primary_activation: id(c.outputs().last().unwrap().0),
                boundaries: c
                    .outputs()
                    .into_iter()
                    .map(|(name, _)| (id(name), ElementType::Bf16))
                    .collect(),
                states: vec![],
                operations: c
                    .operations()
                    .into_iter()
                    .map(|name| NumericalOperationContract {
                        operation_id: id(name),
                        version: ContractVersion::new(1, 0),
                        multiplication_type: (name != SPATIAL_UPSAMPLE_GATHER_BF16_OPERATION_ID)
                            .then_some(ElementType::F32),
                        accumulation_type: (name != SPATIAL_UPSAMPLE_GATHER_BF16_OPERATION_ID)
                            .then_some(ElementType::F32),
                        staged_arithmetic: None,
                        composite_arithmetic: None,
                    })
                    .collect(),
            }],
            vec![profile],
        )
    }
    fn semantic_program(
        &self,
        c: &Probe,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        ModelProgram::new(
            self.family_id().clone(),
            c.inputs().into_keys().collect(),
            vec![ProgramBlock {
                id: "block.cuda.spatial_decoder".into(),
                nodes: c.nodes(),
            }],
            vec![],
            vec![],
            c.outputs().into_iter().map(|(name, _)| id(name)).collect(),
        )?
        .with_input_tensor_specs(c.inputs())
    }
    fn semantic_metadata(&self, _: &Probe) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}

/// Caller fixture data and hand-derived expectations are prepared before CUDA
/// runtime creation. No candidate output participates in these expectations.
fn prepared(
    c: Probe,
) -> (
    BTreeMap<&'static str, Vec<u16>>,
    BTreeMap<&'static str, Vec<u16>>,
) {
    if let Probe::ChannelNorm {
        channels,
        batch,
        width,
    } = c
    {
        return channel_norm::prepared(channels as usize, batch as usize, width as usize);
    }
    if matches!(c, Probe::Padding) {
        // Asymmetric kernel top-left=1, center=2, bottom-right=-1;
        // cross-correlation of [1,2,3;4,5,6] is [-3,-2,6;8,11,14].
        // Bias .5 is added after B(dot); all these values are exact BF16.
        let inputs = BTreeMap::from([
            (X, vec![0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0]),
            (WEIGHT, vec![0x3f80, 0, 0, 0, 0x4000, 0, 0, 0, 0xbf80]),
            (BIAS, vec![0x3f00]),
        ]);
        let expected = vec![0xc020, 0xbfc0, 0x40d0, 0x4108, 0x4138, 0x4168];
        return (
            inputs,
            BTreeMap::from([(IMPLICIT, expected.clone()), (EXPLICIT, expected)]),
        );
    }
    let channels = 1152;
    let mut wide = vec![0; 3 * channels * 2];
    // Q has first/last coordinates one; K0 only first, K1 only last.
    // Both full-head scores equal one, V=(1,3), so every output is two.
    for p in 0..2 {
        wide[p] = 0x3f80;
        wide[(channels - 1) * 2 + p] = 0x3f80;
    }
    wide[channels * 2] = 0x3f80;
    wide[(2 * channels - 1) * 2 + 1] = 0x3f80;
    for ch in 0..channels {
        wide[(2 * channels + ch) * 2] = 0x3f80;
        wide[(2 * channels + ch) * 2 + 1] = 0x4040;
    }
    // C3: Q0=(1,1,1), Q1=0; K0=(1,-1,0), K1=(0,0,1).
    // Scores are [0,1] and [0,0]. V0=(0,1,-1), V1=(1,0,1).
    // With p=e/(1+e), exact expectations are (p,1-p,2p-1) and (.5,.5,0).
    // 2.71828 < e < 2.71829 encloses all three strictly inside the BF16
    // cells below; the selected precise-libdevice path is checked against
    // these cells, not claimed generally correctly rounded for every input.
    let small = vec![
        0x3f80, 0, 0x3f80, 0, 0x3f80, 0, 0x3f80, 0, 0xbf80, 0, 0, 0x3f80, 0, 0x3f80, 0x3f80, 0,
        0xbf80, 0x3f80,
    ];
    let norm = vec![
        0x3f80, 0xbf80, 0x3f80, 0xbf80, 0x3f80, 0xbf80, 0x3f80, 0xbf80,
    ];
    let gather = vec![0x8000, 0x7fa1, 0x1234, 0x5678, 0xabcd, 0xef01, 0x7fc3, 0];
    let expected_gather = if matches!(c, Probe::OddChannels) {
        vec![
            0x1234, 0x1234, 0x5678, 0x5678, 0x1234, 0x1234, 0x5678, 0x5678, 0x7fc3, 0x7fc3, 0, 0,
            0x7fc3, 0x7fc3, 0, 0,
        ]
    } else {
        vec![
            0x8000, 0x8000, 0x7fa1, 0x7fa1, 0x1234, 0x1234, 0x5678, 0x5678, 0xabcd, 0xabcd, 0xef01,
            0xef01, 0x7fc3, 0x7fc3, 0, 0,
        ]
    };
    (
        BTreeMap::from([
            (WIDE, wide),
            (SMALL, small),
            (NORM, norm),
            (GAMMA, vec![0x3f80, 0x4000, 0xbf80, 0xc000]),
            (GATHER, gather),
        ]),
        BTreeMap::from([
            (WIDE_OUT, vec![0x4000; channels * 2]),
            (SMALL_OUT, vec![0x3f3b, 0x3f00, 0x3e8a, 0x3f00, 0x3eed, 0]),
            (
                NORM_OUT,
                vec![
                    0x3f80, 0xbf80, 0x4000, 0xc000, 0xbf80, 0x3f80, 0xc000, 0x4000,
                ],
            ),
            (GATHER_OUT, expected_gather),
        ]),
    )
}
fn check(bytes: &[u8], expected: &[u16]) {
    assert_eq!(bytes.len(), 2 * expected.len());
    for (i, (pair, expected)) in bytes.chunks_exact(2).zip(expected).enumerate() {
        assert_eq!(
            u16::from_le_bytes([pair[0], pair[1]]),
            *expected,
            "element {i}"
        );
    }
}

fn check_probe(c: Probe, bytes: &[u8], expected: &[u16]) {
    if matches!(c, Probe::ChannelNorm { .. }) {
        channel_norm::check(bytes, expected);
    } else {
        check(bytes, expected);
    }
}

fn execute_probe(c: Probe, cudnn: Option<&CudaCudnnConfig>) {
    let (input_bits, expected) = prepared(c);
    let input_bytes: usize = input_bits.values().map(|v| v.len() * 2).sum();
    let output_bytes: usize = expected.values().map(|v| v.len() * 2).sum();
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
            usable_memory_bytes: USABLE_BYTES,
            bootstrap_memory_bytes: 1 << 30,
            stream_memory_bytes: 64 << 20,
            maximum_streams: 1,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 64,
            opaque_memory_profile: "experimental-unqualified-spatial-decoder-test-v1".into(),
        })
        .unwrap(),
    );
    let (contracts, providers): (Vec<Box<dyn OperationContract>>, _) = if let Some(config) = cudnn {
        (
            vec![Box::new(conv2d_bf16_contract().unwrap())],
            cuda_vnext_tensor_providers_with_cudnn(&runtime, config, &[]).unwrap(),
        )
    } else {
        (
            vec![
                Box::new(spatial_attention_bf16_contract().unwrap()),
                Box::new(channel_l2_norm_image_bf16_contract().unwrap()),
                Box::new(spatial_upsample_gather_bf16_contract().unwrap()),
            ],
            cuda_vnext_tensor_providers(&runtime, &SELECTED).unwrap(),
        )
    };
    let registry = OperationRuntimeRegistry::new(contracts, providers).unwrap();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.cuda.spatial_decoder.fixture"),
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
    let family = TypedFamilyRegistration::new(DecoderFamily)
        .prepare_with_profile(
            &serde_json::to_value(c).unwrap(),
            &NumericalProfileId::new(PROFILE).unwrap(),
        )
        .unwrap();
    let mut options = ProgramPlanCompileOptions::new(c.inputs()).unwrap();
    for (name, tensor) in c.outputs() {
        options.insert_tensor_spec(id(name), tensor).unwrap();
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
    for node in executable.execution_plan().payload().nodes() {
        if matches!(c, Probe::Padding) {
            let padding_bytes = if node.id().as_str() == "node.decoder.explicit" {
                256
            } else {
                0
            };
            let scratch = node.provider_resources().scratch().unwrap();
            assert_eq!(scratch.fixed_bytes(), Some(WORKSPACE_BYTES + padding_bytes));
            assert_eq!(
                scratch.minimum_bytes().unwrap(),
                WORKSPACE_BYTES + padding_bytes
            );
            assert!(node.scratch_resource().is_some());
        } else {
            assert!(node.provider_resources().scratch().is_none());
        }
    }
    let resources = no_static(runtime.clone(), executable.execution_plan());
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
    let tensors = c
        .inputs()
        .into_iter()
        .map(|(name, tensor)| {
            managed_input(
                &host,
                name.as_str(),
                tensor.dimensions,
                &input_bits[name.as_str()],
            )
        })
        .collect();
    let outputs = executor
        .execute(
            id("run.cuda.spatial_decoder"),
            id("request.cuda.spatial_decoder"),
            tensors,
        )
        .unwrap();
    assert_eq!(outputs.len(), expected.len());
    assert!(outputs.len() <= 4);
    for output in &outputs {
        assert!(host.owns(output.bytes()));
        check_probe(
            c,
            output.bytes().bytes(),
            &expected[output.value_id().as_str()],
        );
        let expected_spec = c
            .outputs()
            .into_iter()
            .find(|(name, _)| *name == output.value_id().as_str())
            .unwrap()
            .1;
        assert_eq!(output.tensor().dimensions(), expected_spec.dimensions);
    }
    assert_eq!(host.usage().unwrap().committed_bytes(), output_bytes);
    let name = if matches!(c, Probe::Padding) {
        IMPLICIT
    } else if matches!(c, Probe::ChannelNorm { .. }) {
        NORM_OUT
    } else {
        GATHER_OUT
    };
    let retained = outputs
        .iter()
        .find(|o| o.value_id().as_str() == name)
        .unwrap()
        .bytes()
        .clone();
    drop(outputs);
    assert_eq!(
        host.usage().unwrap().committed_bytes(),
        2 * expected[name].len()
    );
    drop((executor, compilation, registry));
    close_resources(resources);
    if let Some(config) = cudnn {
        drop(cuda_vnext_tensor_providers_with_cudnn(&runtime, config, &[]).unwrap());
    } else {
        drop(cuda_vnext_tensor_providers(&runtime, &SELECTED).unwrap());
    }
    drop(runtime.create_stream().unwrap());
    drop(runtime);
    check_probe(c, retained.bytes(), &expected[name]);
    drop(retained);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
}

#[test]
#[ignore = "requires exclusive CUDA; compares actual managed channel normalization with the original FP32 tree"]
fn managed_cuda_channel_l2_norm_coalesced_tree_tail_and_ieee() {
    for channels in [96, 144, 288, 4096] {
        execute_probe(
            Probe::ChannelNorm {
                channels,
                batch: 2,
                width: if channels == 4096 { 33 } else { 131 },
            },
            None,
        );
    }
}

#[test]
#[ignore = "requires exclusive CUDA; runs the actual managed executor"]
fn managed_cuda_spatial_attention_rank_three_norm_and_subpixel_gather() {
    execute_probe(Probe::OddChannels, None);
    execute_probe(Probe::AlternatingRows, None);
    eprintln!("CUDA spatial decoder: four-output exact budgets, full-head C1152 and nonuniform C3, rank3 gamma, odd/row gather bits, owner retirement; no full VAE or SDPA parity claim");
}
#[test]
#[ignore = "requires exclusive CUDA and explicit FERRUM_TEST_CUDNN_LIBRARY_PATH"]
fn managed_cuda_implicit_and_explicit_symmetric_padding_use_declared_scratch() {
    let library_path = std::env::var_os("FERRUM_TEST_CUDNN_LIBRARY_PATH")
        .map(std::path::PathBuf::from)
        .expect("set explicit immutable cuDNN9 library path");
    assert!(library_path.is_absolute());
    let config = CudaCudnnConfig {
        library_path,
        dependency_libraries: Vec::new(),
        opaque_memory_bytes: 64 << 20,
        workspace_limit_bytes: WORKSPACE_BYTES,
        opaque_memory_profile: "experimental-unqualified-spatial-decoder-test-v1".into(),
    };
    execute_probe(Probe::Padding, Some(&config));
    eprintln!("CUDA Conv: nonzero asymmetric kernel, symmetric implicit and explicit padding exact outputs, workspace-only versus padded+workspace allocation, owner retirement");
}
