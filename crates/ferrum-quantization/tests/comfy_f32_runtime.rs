//! Small typed canonical checkpoints exercise actual Stored component loading.
//! Numeric fixtures establish this declared RN32 profile, not upstream binary
//! parity or full text-encoder support. No model payload is checked into Git.
#[cfg(feature = "cuda")]
use ferrum_execution::{
    component::{
        CompiledTensorComponent, ComponentInitializationConfig, ComponentWeightSource,
        TensorComponentPolicy,
    },
    TensorExecutorConfig,
};
use ferrum_quantization::{
    comfy_convrot::{
        ComfyCanonicalWeightSource, ComfyQuantizationMetadata, ComfyQuantizedMatrix,
        COMFY_CONVROT_CANONICAL_WEIGHT_FORMAT_ID,
    },
    SafetensorsArchive,
};
use ferrum_runtime::*;
use safetensors::{
    tensor::{serialize_to_file, TensorView},
    Dtype,
};
use serde::{Deserialize, Serialize};
#[cfg(feature = "cuda")]
use std::sync::Arc;
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::OnceLock,
};

const X: &str = "value.fixture.comfy_f32.input";
const Y: &str = "value.fixture.comfy_f32.output";
const MODULE: &str = "fixture.q_proj";
const M: u64 = 3;
const K: u64 = 512;
const N: u64 = 4;
#[path = "comfy_f32_runtime/embedding.rs"]
mod embedding;
fn id<T: TryFrom<String>>(s: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
fn spec(dimensions: Vec<u64>, element_type: ElementType) -> ProgramTensorSpec {
    ProgramTensorSpec {
        dimensions,
        element_type,
        layout: ResolvedTensorLayout::Contiguous,
    }
}
fn value(name: &str) -> ProgramValueId {
    id(format!("value.fixture.comfy_f32.{name}"))
}
fn weight(name: &str) -> WeightId {
    id(format!("weight.fixture.comfy_f32.{name}"))
}
fn component(name: &str) -> WeightId {
    id(format!("component.fixture.comfy_f32.{name}"))
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    w4: bool,
    embedding: Option<ComfyEmbeddingDequantizedDtype>,
}
impl Config {
    fn metadata(self) -> ComfyQuantizationMetadata {
        if self.w4 {
            ComfyQuantizationMetadata::CodebookW4A8 {
                group_size: 16,
                convrot: true,
                convrot_groupsize: 256,
            }
        } else {
            ComfyQuantizationMetadata::Int8Tensorwise {
                convrot: true,
                convrot_groupsize: 256,
            }
        }
    }
    fn operation(self) -> &'static str {
        if self.embedding.is_some() {
            COMFY_CONVROT_INT8_EMBEDDING_F32_OPERATION_ID
        } else if self.w4 {
            COMFY_CONVROT_W4A8_F32_OPERATION_ID
        } else {
            COMFY_CONVROT_W8A8_F32_OPERATION_ID
        }
    }
    fn profile(self) -> &'static str {
        if let Some(dtype) = self.embedding {
            match dtype {
                ComfyEmbeddingDequantizedDtype::F32 => "fixture.comfy_f32.embedding.full",
                ComfyEmbeddingDequantizedDtype::Bf16ThenF32 => {
                    "fixture.comfy_f32.embedding.bf16_then_f32"
                }
            }
        } else if self.w4 {
            "fixture.comfy_f32.w4a8"
        } else {
            "fixture.comfy_f32.w8a8"
        }
    }
    fn entries(self) -> Vec<(&'static str, Vec<u64>, ElementType)> {
        if self.w4 {
            vec![
                ("weight", vec![N, K / 2], ElementType::I8),
                ("weight_s_rel", vec![N, K / 16], ElementType::U8),
                ("weight_s_channel", vec![N], ElementType::F32),
                ("weight_codebook", vec![16], ElementType::F32),
            ]
        } else {
            vec![
                ("weight", vec![N, K], ElementType::I8),
                ("weight_scale", vec![N, 1], ElementType::F32),
            ]
        }
    }
    fn input_specs(self) -> BTreeMap<ProgramValueId, ProgramTensorSpec> {
        BTreeMap::from([(
            id(X),
            if self.embedding.is_some() {
                spec(vec![embedding::TOKENS], ElementType::U32)
            } else {
                spec(vec![M, K], ElementType::F32)
            },
        )])
    }
    fn schema(self) -> WeightSchema {
        let mut components = Vec::new();
        let mut tensors = Vec::new();
        for (name, dimensions, dtype) in self.entries() {
            components.push(WeightComponentSpec {
                id: component(name),
                role: WeightComponentRole::Values,
                external_names: vec![format!("{MODULE}.{name}")],
                dimensions: dimensions.clone(),
                encoding: WeightEncoding::Dense {
                    element_type: dtype,
                },
                required: true,
            });
            tensors.push(WeightTensorSpec {
                id: weight(name),
                dimensions,
                logical_element_type: dtype,
                physical_layout: PhysicalWeightLayout::Dense {
                    component_id: component(name),
                },
                required: true,
            });
        }
        WeightSchema {
            format_id: id(COMFY_CONVROT_CANONICAL_WEIGHT_FORMAT_ID),
            layout_id: id("weight-layout.fixture.comfy_f32.canonical.v1"),
            version: ContractVersion::new(1, 0),
            components,
            tensors,
        }
    }
}
struct Family;
impl ModelFamilyProvider for Family {
    type Config = Config;
    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.fixture.comfy_f32_projection"))
    }
    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.fixture.comfy_f32_projection")])
    }
    fn parse_config(&self, raw: &serde_json::Value) -> Result<Config, VNextError> {
        serde_json::from_value(raw.clone()).map_err(|e| invalid(e.to_string()))
    }
    fn validate_config_identity(
        &self,
        raw: &serde_json::Value,
        c: &Config,
    ) -> Result<(), VNextError> {
        if serde_json::to_value(c).unwrap() != *raw || (c.embedding.is_some() && c.w4) {
            Err(invalid("noncanonical fixture config"))
        } else {
            Ok(())
        }
    }
    fn validated_external_metadata_id(
        &self,
        raw: &serde_json::Value,
        c: &Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, c)?;
        Ok(id("metadata.fixture.comfy_f32_projection"))
    }
    fn weights(&self, c: &Config) -> Result<ModelWeights, VNextError> {
        Ok(ModelWeights::Stored(c.schema()))
    }
    fn numerical_profiles(&self, c: &Config) -> Result<FamilyNumericalProfiles, VNextError> {
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: id(c.profile()),
                family_id: self.family_id().clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: id(Y),
                boundaries: BTreeMap::from([(id(Y), ElementType::F32)]),
                states: vec![],
                kv_storage: vec![],
                operations: vec![NumericalOperationContract {
                    operation_id: id(c.operation()),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: None,
                    accumulation_type: None,
                    staged_arithmetic: Some(if let Some(dtype) = c.embedding {
                        StagedNumericalArithmetic::comfy_convrot_int8_embedding_f32(dtype)
                    } else if c.w4 {
                        StagedNumericalArithmetic::comfy_convrot_w4a8_f32()
                    } else {
                        StagedNumericalArithmetic::comfy_convrot_w8a8_f32()
                    }),
                    composite_arithmetic: None,
                }],
            }],
            vec![id(c.profile())],
        )
    }
    fn semantic_program(
        &self,
        c: &Config,
        _: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let mut inputs = vec![id(X)];
        let mut weights = Vec::new();
        for (name, dimensions, dtype) in c.entries() {
            inputs.push(value(name));
            weights.push(WeightReference {
                weight_id: weight(name),
                value_id: value(name),
                tensor: spec(dimensions, dtype),
            });
        }
        ModelProgram::new(
            self.family_id().clone(),
            vec![id(X)],
            vec![ProgramBlock {
                id: "block.fixture.comfy_f32".into(),
                nodes: vec![ProgramNode {
                    id: id("node.fixture.comfy_f32"),
                    operation_id: id(c.operation()),
                    required_version: ContractVersion::new(1, 0),
                    work: ProgramNodeWorkSpec::Fixed,
                    inputs,
                    outputs: vec![id(Y)],
                    attributes: if let Some(dtype) = c.embedding {
                        BTreeMap::from([
                            (id("hidden_size"), SemanticValue::Unsigned(K)),
                            (id("vocab_size"), SemanticValue::Unsigned(N)),
                            (
                                id("dequantized_dtype"),
                                SemanticValue::Text(dtype.as_str().into()),
                            ),
                        ])
                    } else {
                        BTreeMap::from([
                            (id("in_features"), SemanticValue::Unsigned(K)),
                            (id("out_features"), SemanticValue::Unsigned(N)),
                        ])
                    },
                }],
            }],
            vec![],
            weights,
            vec![id(Y)],
        )?
        .with_input_tensor_specs(c.input_specs())
    }
    fn semantic_metadata(&self, _: &Config) -> Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata::TensorComponent)
    }
}
fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}

fn checkpoint(c: Config) -> (tempfile::TempDir, ComfyCanonicalWeightSource) {
    let directory = tempfile::tempdir().unwrap();
    let metadata = serde_json::to_vec(&c.metadata()).unwrap();
    let mut owned = vec![(
        format!("{MODULE}.comfy_quant"),
        Dtype::U8,
        vec![metadata.len()],
        metadata,
    )];
    for (name, dimensions, dtype) in c.entries() {
        let bytes: Vec<u8> = if c.embedding.is_some() {
            embedding::weight_bytes(name)
        } else {
            match name {
                "weight" => {
                    if c.w4 {
                        [0x00_u8, 0x11, 0x22, 0xff]
                            .into_iter()
                            .flat_map(|v| std::iter::repeat_n(v, (K / 2) as usize))
                            .collect()
                    } else {
                        [-128_i8, 127, -2, 0]
                            .into_iter()
                            .flat_map(|v| std::iter::repeat_n(v as u8, K as usize))
                            .collect()
                    }
                }
                "weight_s_rel" => vec![0x38; (N * K / 16) as usize],
                "weight_s_channel" | "weight_scale" => [0.5_f32, 0.25, 0.125, 1.0]
                    .into_iter()
                    .flat_map(f32::to_le_bytes)
                    .collect(),
                "weight_codebook" => {
                    let mut v = [0.0_f32; 16];
                    v[0] = -1.0;
                    v[1] = 1.0;
                    v[2] = 2.0;
                    v[15] = 0.5;
                    v.into_iter().flat_map(f32::to_le_bytes).collect()
                }
                _ => unreachable!(),
            }
        };
        let dtype = match (name, dtype) {
            ("weight_s_rel", _) => Dtype::F8_E4M3,
            (_, ElementType::I8) => Dtype::I8,
            (_, ElementType::F32) => Dtype::F32,
            _ => unreachable!(),
        };
        owned.push((
            format!("{MODULE}.{name}"),
            dtype,
            dimensions.into_iter().map(|v| v as usize).collect(),
            bytes,
        ));
    }
    let views = owned
        .iter()
        .map(|(name, dtype, shape, bytes)| {
            (
                name.as_str(),
                TensorView::new(*dtype, shape.clone(), bytes).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(views, &None, &directory.path().join("model.safetensors")).unwrap();
    let archive = SafetensorsArchive::open(directory.path()).unwrap();
    (directory, ComfyCanonicalWeightSource::new(archive).unwrap())
}
fn input(trial: usize) -> Vec<f32> {
    let a = 1.000244140625_f32;
    let b = a * (63.4_f32 / 127.0);
    [1.0_f32, -0.5, 0.0]
        .into_iter()
        .flat_map(|sign| {
            [a * sign, b * sign]
                .into_iter()
                .flat_map(|v| std::iter::repeat_n(if trial == 0 { v } else { 0.0 }, 256))
        })
        .collect()
}
fn expected(c: Config, trial: usize) -> Vec<f32> {
    let codes = if c.w4 {
        [-1_i32, 1, 2, 0]
    } else {
        [-128, 127, -2, 0]
    };
    let channels = [0.5_f32, 0.25, 0.125, 1.0];
    let scale = 1.000244140625_f32 * (1.0_f32 / 127.0);
    [scale, scale * 0.5, 1.0e-30]
        .into_iter()
        .enumerate()
        .flat_map(|(row, s)| {
            codes.into_iter().zip(channels).map(move |(w, channel)| {
                let acc = if trial != 0 || row == 2 {
                    0
                } else {
                    48640 * w * if row == 0 { 1 } else { -1 }
                };
                ((acc as f32) * s) * channel
            })
        })
        .collect()
}

#[test]
fn canonical_w4_w8_f32_checkpoint_source_and_closed_oracle_are_wired() {
    for w4 in [true, false] {
        let c = Config {
            w4,
            embedding: None,
        };
        let (_directory, source) = checkpoint(c);
        assert_eq!(source.modules().get(MODULE), Some(&c.metadata()));
        let schema = c.schema();
        schema.validate(Family.family_id()).unwrap();
        for component in &schema.components {
            let payload = source.component(component).unwrap();
            assert_eq!(
                payload.bytes().len() as u64,
                spec(
                    component.dimensions.clone(),
                    match component.encoding {
                        WeightEncoding::Dense { element_type } => element_type,
                        _ => unreachable!(),
                    }
                )
                .byte_len()
                .unwrap()
            );
            assert!(payload.retained_host_memory().is_some());
        }
        let prepared = TypedFamilyRegistration::new(Family)
            .prepare_with_profile(&serde_json::to_value(c).unwrap(), &id(c.profile()))
            .unwrap();
        let _ = prepared;
        let result = ComfyQuantizedMatrix::from_archive(source.archive(), MODULE)
            .unwrap()
            .linear_f32_a8_reference(&input(0), M as usize)
            .unwrap();
        assert_eq!(result.output_f32, expected(c, 0));
        assert!(result.activation_i8[256..512].iter().all(|v| *v == 63));
        assert_ne!(
            half::bf16::from_f32(result.output_f32[0]).to_f32(),
            result.output_f32[0]
        );
    }
}

#[cfg(feature = "cuda")]
fn run_cuda(c: Config) {
    use ferrum_kernels::backend::cuda::{
        vnext_ops::{cuda_tensor_runtime_config, cuda_vnext_tensor_providers, CudaTensorOperation},
        vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
    };
    let runtime = Arc::new(
        CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
            device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
            usable_memory_bytes: 1 << 30,
            bootstrap_memory_bytes: 256 << 20,
            stream_memory_bytes: 16 << 20,
            maximum_streams: 2,
            maximum_in_flight_per_stream: 1,
            maximum_commands_per_submission: 64,
            opaque_memory_profile: "unqualified-comfy-f32-component-v1".into(),
        })
        .unwrap(),
    );
    let (contract, operation) = if c.embedding.is_some() {
        (
            comfy_convrot_int8_embedding_f32_contract().unwrap(),
            CudaTensorOperation::ComfyConvRotInt8EmbeddingF32,
        )
    } else if c.w4 {
        (
            comfy_convrot_w4a8_f32_contract().unwrap(),
            CudaTensorOperation::ComfyConvRotW4A8F32,
        )
    } else {
        (
            comfy_convrot_w8a8_f32_contract().unwrap(),
            CudaTensorOperation::ComfyConvRotW8A8F32,
        )
    };
    let registry = OperationRuntimeRegistry::new(
        vec![Box::new(contract)],
        cuda_vnext_tensor_providers(&runtime, &[operation]).unwrap(),
    )
    .unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    let d = runtime.descriptor();
    let engine = EngineProviderDescriptor::new(
        id("provider.engine.fixture.comfy_f32"),
        ContractVersion::new(1, 0),
        d.runtime_implementation_fingerprint.clone(),
        d.id.clone(),
        d.capabilities.clone(),
    )
    .unwrap();
    let catalog = materializers
        .augment_catalog(
            registry
                .capability_catalog(d.clone(), vec![engine])
                .unwrap(),
        )
        .unwrap();
    let family = TypedFamilyRegistration::new(Family)
        .prepare_with_profile(&serde_json::to_value(c).unwrap(), &id(c.profile()))
        .unwrap();
    let policy = TensorComponentPolicy::new(
        d.total_memory_bytes,
        d.dynamic_storage_profiles.iter().copied().collect(),
    )
    .unwrap();
    let component = CompiledTensorComponent::compile(
        family,
        &registry,
        &catalog,
        &policy,
        &materializers,
        &ProgramPlanCompileOptions::new(c.input_specs()).unwrap(),
    )
    .unwrap();
    let (_directory, source) = checkpoint(c);
    let host = HostMemoryAccount::new(
        HostMemoryLimits::new(
            4 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 1 << 20,
                transfer_staging_bytes: 1 << 20,
                output_bytes: 1 << 20,
                weight_backing_bytes: 1 << 20,
            },
        )
        .unwrap(),
    );
    let mut session = component
        .initialize(
            runtime.clone(),
            host.clone(),
            ComponentWeightSource::Stored(&source),
            ComponentInitializationConfig {
                run_id: id("run.comfy_f32.init"),
                request_id: id("request.comfy_f32.init"),
                transaction_id: id("transaction.comfy_f32.init"),
                static_policy: StaticInitializationPolicy::new(1 << 20, 16).unwrap(),
                executor: TensorExecutorConfig::default(),
            },
        )
        .unwrap();
    for trial in 0..2 {
        let bytes: Vec<u8> = if c.embedding.is_some() {
            embedding::ids(trial)
                .into_iter()
                .flat_map(u32::to_le_bytes)
                .collect()
        } else {
            input(trial)
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect()
        };
        let mut backing = host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, bytes.len())
            .unwrap();
        backing.as_mut().copy_from_slice(&bytes);
        let input_spec = c.input_specs().remove(&id(X)).unwrap();
        let x = ManagedHostTensor::new(
            id(X),
            ResolvedTensorSpec::new(
                input_spec.dimensions,
                input_spec.element_type,
                input_spec.layout,
            )
            .unwrap(),
            backing.freeze(),
        )
        .unwrap();
        let outputs = session
            .execute(
                id(format!("run.comfy_f32.{trial}")),
                id(format!("request.comfy_f32.{trial}")),
                vec![x],
            )
            .unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(
            outputs[0].tensor().dimensions(),
            if c.embedding.is_some() {
                &[embedding::TOKENS, K]
            } else {
                &[M, N]
            }
        );
        assert_eq!(outputs[0].tensor().element_type(), ElementType::F32);
        let actual: Vec<u32> = outputs[0]
            .bytes()
            .bytes()
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let expected = if let Some(dtype) = c.embedding {
            ComfyQuantizedMatrix::from_archive(source.archive(), MODULE)
                .unwrap()
                .embedding_f32_reference(&embedding::ids(trial), dtype)
                .unwrap()
        } else {
            expected(c, trial)
        };
        assert_eq!(
            actual,
            expected.into_iter().map(f32::to_bits).collect::<Vec<_>>()
        );
    }
    assert!(matches!(
        session
            .close()
            .unwrap_or_else(|f| panic!("F32 ConvRot close failed: {:?}", f.failure())),
        PlanRuntimeCloseOutcome::Closed(_)
    ));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
#[cfg(feature = "cuda")]
static GPU: std::sync::Mutex<()> = std::sync::Mutex::new(());
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires configured CUDA host; canonical Stored W4 weights and full-F32 A8 arithmetic"]
fn canonical_stored_w4a8_f32_executes_without_bf16_activation_or_output_rounding() {
    let _gpu = GPU
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    run_cuda(Config {
        w4: true,
        embedding: None,
    });
}
#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires configured CUDA host; canonical Stored signed INT8 weights and full-F32 A8 arithmetic"]
fn canonical_stored_w8a8_f32_executes_without_bf16_activation_or_output_rounding() {
    let _gpu = GPU
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    run_cuda(Config {
        w4: false,
        embedding: None,
    });
}
