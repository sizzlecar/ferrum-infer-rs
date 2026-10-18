//! Reusable startup with the real causal-attention provider's program binding.
//! The attention-free readback fixture separately proves that captured segments
//! without such a binding cannot qualify as a reusable language program.

use super::*;
use crate::executor::vnext_executor::sampling_readback::runtime_tests::{
    fixture::resolve_plan, startup::assert_startup_capture_replay,
};
use crate::vnext::{
    CausalLanguageModelDescriptor, PreparedProductionModel, ProductionModelSourceBundle,
    ProductionWeightArtifact,
};
use ferrum_quantization::gguf::GgufWeightComponentSource;
use ferrum_types::{
    CompletionReadbackPolicy, DataType, Device, EngineConfig, KvStorageFormat,
    ReusableExecutionPreparationMode, SamplingReadbackMode,
};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

const VOCAB: u64 = 32;
const TOKENIZER_CONFIG: &[u8] = br#"{"chat_template":"{{ messages }}"}"#;
const GGUF_ARCHITECTURE: &str = "fixture_causal_int8_language";
const SEMANTIC_ARCHITECTURE: &str = "FixtureCausalInt8ForCausalLM";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct LanguageConfig {
    attention: AttentionKind,
    model_type: String,
    architectures: [String; 1],
}

/// Preserve the existing numerical attention fixture, adding only the normal
/// language head and selection operators consumed by VNextModelExecutor.
struct LanguageFamily(Family);

impl ModelFamilyProvider for LanguageFamily {
    type Config = LanguageConfig;

    fn family_id(&self) -> &ModelFamilyId {
        self.0.family_id()
    }

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id(format!("hf.architecture.{SEMANTIC_ARCHITECTURE}"))])
    }

    fn validate_config_identity(
        &self,
        raw: &Value,
        config: &LanguageConfig,
    ) -> Result<(), VNextError> {
        self.0.validate_config_identity(raw, &config.attention)?;
        if config.model_type != GGUF_ARCHITECTURE
            || config.architectures[0] != SEMANTIC_ARCHITECTURE
        {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "architectures".to_owned(),
                reason: "startup fixture semantic and physical architecture identities differ"
                    .to_owned(),
            });
        }
        Ok(())
    }

    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &LanguageConfig,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(id(format!("hf.architecture.{SEMANTIC_ARCHITECTURE}")))
    }

    fn parse_config(&self, raw: &Value) -> Result<LanguageConfig, VNextError> {
        let config = serde_json::from_value(raw.clone()).map_err(|error| {
            VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "config".to_owned(),
                reason: error.to_string(),
            }
        })?;
        self.validate_config_identity(raw, &config)?;
        Ok(config)
    }

    fn weight_schema(&self, config: &LanguageConfig) -> Result<WeightSchema, VNextError> {
        let mut schema = self.0.weight_schema(&config.attention)?;
        schema.components.push(WeightComponentSpec {
            id: id("component.head"),
            role: WeightComponentRole::Values,
            external_names: vec!["head".to_owned()],
            dimensions: vec![VOCAB, HIDDEN],
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F16,
            },
            required: true,
        });
        schema.tensors.push(WeightTensorSpec {
            id: id("weight.head"),
            dimensions: vec![VOCAB, HIDDEN],
            logical_element_type: ElementType::F16,
            physical_layout: PhysicalWeightLayout::Dense {
                component_id: id("component.head"),
            },
            required: true,
        });
        Ok(schema)
    }

    fn numerical_profiles(
        &self,
        config: &LanguageConfig,
    ) -> Result<FamilyNumericalProfiles, VNextError> {
        let original = self.0.numerical_profiles(&config.attention)?;
        let mut profiles = original.profiles().to_vec();
        for profile in &mut profiles {
            profile
                .boundaries
                .insert(id("value.logits"), ElementType::F32);
            profile
                .boundaries
                .insert(id("value.selected"), ElementType::U32);
            for operation in [
                LAST_TOKEN_DENSE_LINEAR_F32_OPERATION_ID,
                LAST_TOKEN_MASKED_ARGMAX_F32_OPERATION_ID,
            ] {
                let linear = operation == LAST_TOKEN_DENSE_LINEAR_F32_OPERATION_ID;
                profile.operations.push(NumericalOperationContract {
                    operation_id: id(operation),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: linear.then_some(ElementType::F32),
                    accumulation_type: linear.then_some(ElementType::F32),
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                });
            }
        }
        FamilyNumericalProfiles::new(
            self.family_id(),
            original.version(),
            profiles,
            original.auto_preference().to_vec(),
        )
    }

    fn semantic_program(
        &self,
        config: &LanguageConfig,
        profile: &NumericalExecutionProfile,
    ) -> Result<ModelProgram, VNextError> {
        let original = self.0.semantic_program(&config.attention, profile)?;
        let mut inputs = original.inputs().to_vec();
        inputs.extend(
            [
                "value.mask",
                "value.repeat_ids",
                "value.repeat_offsets",
                "value.penalty",
            ]
            .map(id),
        );
        let mut blocks = original.blocks().to_vec();
        blocks.last_mut().unwrap().nodes.extend([
            ProgramNode {
                id: id("node.logits"),
                operation_id: id(LAST_TOKEN_DENSE_LINEAR_F32_OPERATION_ID),
                required_version: ContractVersion::new(1, 0),
                work: ProgramNodeWorkSpec::tokens(id("value.output"), 0),
                inputs: vec![id("value.output"), id("value.weight.head")],
                outputs: vec![id("value.logits")],
                attributes: BTreeMap::from([
                    (id("hidden_size"), SemanticValue::Unsigned(HIDDEN)),
                    (id("out_features"), SemanticValue::Unsigned(VOCAB)),
                ]),
            },
            ProgramNode {
                id: id("node.selected"),
                operation_id: id(LAST_TOKEN_MASKED_ARGMAX_F32_OPERATION_ID),
                required_version: ContractVersion::new(1, 0),
                work: ProgramNodeWorkSpec::Fixed,
                inputs: [
                    "value.logits",
                    "value.mask",
                    "value.repeat_ids",
                    "value.repeat_offsets",
                    "value.penalty",
                ]
                .map(id)
                .into(),
                outputs: vec![id("value.selected")],
                attributes: BTreeMap::from([(id("vocab_size"), SemanticValue::Unsigned(VOCAB))]),
            },
        ]);
        let mut weights = original.weights().to_vec();
        weights.push(WeightReference {
            weight_id: id("weight.head"),
            value_id: id("value.weight.head"),
            tensor: ProgramTensorSpec {
                dimensions: vec![VOCAB, HIDDEN],
                element_type: ElementType::F16,
                layout: ResolvedTensorLayout::Contiguous,
            },
        });
        ModelProgram::new(
            self.family_id().clone(),
            inputs,
            blocks,
            original.states().to_vec(),
            weights,
            vec![id("value.logits"), id("value.selected")],
        )?
        .with_checkpoint_inputs(
            ProgramCheckpointInputs::new(id("value.tokens"), BTreeSet::new())?
                // These inputs feed only selection after attention has written
                // KV. Plan derivation still proves they cannot affect state;
                // declaring them here does not grant checkpoint authority.
                .with_output_only_inputs(BTreeSet::from([
                    id("value.mask"),
                    id("value.repeat_ids"),
                    id("value.repeat_offsets"),
                    id("value.penalty"),
                ]))?,
        )
    }

    fn semantic_metadata(
        &self,
        config: &LanguageConfig,
    ) -> Result<ModelSemanticMetadata, VNextError> {
        let mut metadata = self.0.semantic_metadata(&config.attention)?;
        metadata.template.source_file = "tokenizer_config.json".to_owned();
        metadata.template.sha256 = format!("{:x}", Sha256::digest(TOKENIZER_CONFIG));
        Ok(metadata)
    }
}

struct FamilyRegistry(TypedFamilyRegistration<LanguageFamily>);

impl ModelFamilyRegistry for FamilyRegistry {
    fn registrations(&self) -> Vec<&dyn ModelFamilyRegistration> {
        vec![&self.0]
    }
}

fn gguf_string(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

/// Store the existing deterministic fixture bytes unchanged, then use the
/// production mmap-backed GGUF source and frozen source-bundle identity.
fn write_weights(path: &Path, schema: &WeightSchema) {
    let weights = family::Weights::new(schema);
    let components: Vec<_> = schema
        .components
        .iter()
        .map(|component| {
            (
                component,
                weights.component(component).unwrap().bytes().to_vec(),
            )
        })
        .collect();
    let mut bytes = b"GGUF".to_vec();
    bytes.extend_from_slice(&3_u32.to_le_bytes());
    bytes.extend_from_slice(&(components.len() as u64).to_le_bytes());
    // This is a custom typed fixture, not a claim to be a built-in Llama/Qwen
    // architecture. Source identity, descriptor and GGUF use the same model
    // type; the actual state geometry remains in the validated typed family.
    bytes.extend_from_slice(&3_u64.to_le_bytes());
    gguf_string(&mut bytes, "general.architecture");
    bytes.extend_from_slice(&8_u32.to_le_bytes()); // GGUF string.
    gguf_string(&mut bytes, GGUF_ARCHITECTURE);
    for (key, value) in [
        ("general.alignment", 32_u32),
        ("general.quantization_version", 2_u32),
    ] {
        gguf_string(&mut bytes, key);
        bytes.extend_from_slice(&4_u32.to_le_bytes()); // GGUF uint32.
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let mut offset = 0_u64;
    for (component, payload) in &components {
        let name = &component.external_names[0];
        gguf_string(&mut bytes, name);
        bytes.extend_from_slice(&(component.dimensions.len() as u32).to_le_bytes());
        let mut dimensions = component.dimensions.clone();
        let dtype = match &component.encoding {
            WeightEncoding::Dense {
                element_type: ElementType::F16,
            } => 1_u32,
            WeightEncoding::Dense {
                element_type: ElementType::F32,
            } => 0_u32,
            WeightEncoding::BlockQuantized(spec)
                if spec.format_id.as_str() == "quantization.gguf.q4-k" =>
            {
                assert_eq!(spec.logical_values_per_block, 256);
                assert_eq!(spec.bytes_per_block, 144);
                *dimensions.last_mut().unwrap() *= u64::from(spec.logical_values_per_block);
                12_u32 // GGML Q4_K; payload is retained in its original packed layout.
            }
            other => panic!("unexpected startup fixture weight encoding: {other:?}"),
        };
        for dimension in dimensions.iter().rev() {
            bytes.extend_from_slice(&dimension.to_le_bytes());
        }
        bytes.extend_from_slice(&dtype.to_le_bytes());
        bytes.extend_from_slice(&offset.to_le_bytes());
        offset += (payload.len() as u64).div_ceil(32) * 32;
    }
    bytes.resize(bytes.len().div_ceil(32) * 32, 0);
    for (_, payload) in components {
        bytes.extend_from_slice(&payload);
        bytes.resize(bytes.len().div_ceil(32) * 32, 0);
    }
    std::fs::write(path, bytes).unwrap();
}

fn build_executor(directory: &Path) -> VNextModelExecutor<Runtime> {
    let kind = AttentionKind::CausalInt8;
    let family_registry = FamilyRegistry(TypedFamilyRegistration::new(LanguageFamily(
        Family::new(kind),
    )));
    let family_config = LanguageConfig {
        attention: kind,
        model_type: GGUF_ARCHITECTURE.to_owned(),
        architectures: [SEMANTIC_ARCHITECTURE.to_owned()],
    };
    let raw = serde_json::to_value(&family_config).unwrap();
    std::fs::write(
        directory.join("config.json"),
        serde_json::to_vec(&raw).unwrap(),
    )
    .unwrap();
    std::fs::write(directory.join("tokenizer.json"), br#"{"version":"1.0"}"#).unwrap();
    std::fs::write(directory.join("tokenizer_config.json"), TOKENIZER_CONFIG).unwrap();
    let definition = family_registry.0.define(&raw).unwrap();
    let family = family_registry
        .0
        .prepare(&definition, &id(Family::new(kind).profile_id()))
        .unwrap();
    let path = directory.join("startup.gguf");
    write_weights(&path, family.weight_schema());
    let source =
        GgufWeightComponentSource::open_with_schema(&path, family.weight_schema(), None).unwrap();
    assert_eq!(
        source.file().architecture().unwrap(),
        family_config.model_type
    );
    assert_eq!(source.file().quantization_version(), Some(2));
    let local = |kind, path: &Path| OriginalModelSource {
        kind,
        location: path.display().to_string(),
        requested_revision: None,
    };
    let sources = Arc::new(
        ProductionModelSourceBundle::open(
            directory,
            directory,
            ProductionWeightArtifact::gguf_file(&path),
            OriginalModelSources {
                semantic: local(ModelSourceKind::LocalDirectory, directory),
                tokenizer: local(ModelSourceKind::LocalDirectory, directory),
                weights: local(ModelSourceKind::LocalFile, &path),
            },
        )
        .unwrap(),
    );
    assert_eq!(
        &crate::vnext::external_metadata_id_from_bytes(sources.config_json(), "config.json")
            .unwrap(),
        family.external_metadata_id(),
    );
    let parameter_count = family
        .weight_schema()
        .tensors
        .iter()
        .map(|tensor| tensor.dimensions.iter().product::<u64>())
        .sum();
    let descriptor = CausalLanguageModelDescriptor::new(
        GGUF_ARCHITECTURE,
        parameter_count,
        HIDDEN as usize,
        1,
        4,
        2,
        128,
        VOCAB as usize,
        MAX_TOKENS as usize,
        DataType::FP32,
    )
    .unwrap();
    let prepared =
        PreparedProductionModel::new(family, Arc::new(source), descriptor, sources).unwrap();
    let (runtime, registry, materializers, materializer, catalog) =
        composition(kind, prepared.family());
    let info = prepared.model_info("causal-int8-startup-fixture".into(), Device::CUDA(0));
    let mut config = EngineConfig::default();
    config.scheduler.max_running_requests = 2;
    config.scheduler.max_waiting_requests = 2;
    config.scheduler.prefill_step_chunk = Some(4);
    config.scheduler.active_decode_prefill_chunk = Some(1);
    config.batching.max_num_batched_tokens = 16;
    config.runtime.max_model_len = Some(MAX_TOKENS as usize);
    config.runtime.sampling_readback_mode = SamplingReadbackMode::Uniform;
    config.runtime.attention_execution_policy = ferrum_types::AttentionExecutionPolicy::Portable;
    config.runtime.completion_readback_policy = CompletionReadbackPolicy::BatchedU32;
    config.backend.enable_reusable_execution = true;
    config.backend.reusable_execution_capture.preparation =
        ReusableExecutionPreparationMode::Startup;
    // Product stop tokens retain the actual family's EOS declaration. Declare
    // only the roles that these exact IDs alias; model metadata stays intact.
    let special = &prepared.family().metadata().special_tokens;
    let token_ids = special.eos_token_ids.clone();
    let mut allowed_roles = BTreeSet::new();
    if !token_ids.is_empty() {
        allowed_roles.insert(SpecialTokenRole::Eos);
    }
    if special
        .bos_token_id
        .is_some_and(|token| token_ids.contains(&token))
    {
        allowed_roles.insert(SpecialTokenRole::Bos);
    }
    if special
        .pad_token_id
        .is_some_and(|token| token_ids.contains(&token))
    {
        allowed_roles.insert(SpecialTokenRole::Pad);
    }
    let stop = StopPolicy {
        maximum_output_tokens: 8,
        token_ids,
        strings: vec![],
        collision_policy: StopTokenCollisionPolicy::new(allowed_roles).unwrap(),
    };
    VNextModelExecutor::from_runtime_composition(
        prepared.execution_input().unwrap(),
        info,
        &config,
        runtime,
        registry,
        materializers,
        materializer,
        catalog,
        |policy, catalog, compilation| {
            assert!(
                compilation
                    .executable()
                    .execution_plan()
                    .payload()
                    .nodes()
                    .iter()
                    .any(|node| node.binding_resource().is_some()),
                "real causal attention must supply the typed reusable program binding"
            );
            resolve_plan(
                &prepared,
                policy,
                catalog,
                compilation,
                &family_registry,
                &definition,
                KvStorageFormat::Int8PerTokenHeadF32ScaleV1,
                stop,
            )
        },
    )
    .unwrap()
}

#[tokio::test]
#[ignore = "requires actual CUDA device and native attention kernels"]
async fn native_startup_cuda_attention_capture_replay_preserves_exact_descriptors() {
    let directory = tempfile::tempdir().unwrap();
    let executor = build_executor(directory.path());
    assert_startup_capture_replay(&executor).await;
    let resources = Arc::clone(&executor.plan_resources);
    drop(executor);
    assert!(matches!(
        PlanRuntimeResources::close(resources),
        Ok(PlanRuntimeCloseOutcome::Closed(_))
    ));
}
