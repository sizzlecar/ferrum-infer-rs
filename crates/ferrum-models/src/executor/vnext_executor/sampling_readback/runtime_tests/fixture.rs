//! A tiny language program using the production compiler, providers and terminal worker.
use super::*;
use crate::vnext::{CausalLanguageModelDescriptor, ProductionModelSourceBundle};
use ferrum_interfaces::vnext::*;
use ferrum_quantization::SafetensorsArchive;
use half::f16;
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::OnceLock;

pub(super) const VOCAB: usize = 4;
const MAX_TOKENS: usize = 16;
const TOKENIZER_CONFIG: &[u8] = br#"{"chat_template":"{{ messages }}"}"#;

fn id<T>(value: impl Into<String>) -> T
where
    T: TryFrom<String>,
    T::Error: std::fmt::Debug,
{
    T::try_from(value.into()).unwrap()
}

struct LanguageFamily;

impl ModelFamilyProvider for LanguageFamily {
    type Config = Value;

    fn family_id(&self) -> &ModelFamilyId {
        static ID: OnceLock<ModelFamilyId> = OnceLock::new();
        ID.get_or_init(|| id("family.fixture.sampling-readback"))
    }

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.fixture.sampling-readback")])
    }

    fn validate_config_identity(
        &self,
        raw: &Value,
        config: &Value,
    ) -> std::result::Result<(), VNextError> {
        if raw != config || config != &json!({"vocabulary_size": VOCAB}) {
            return Err(VNextError::InvalidModelConfig {
                family_id: self.family_id().to_string(),
                field: "vocabulary_size".into(),
                reason: "fixture vocabulary differs from its weights".into(),
            });
        }
        Ok(())
    }

    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Value,
    ) -> std::result::Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(id("metadata.fixture.sampling-readback"))
    }

    fn parse_config(&self, raw: &Value) -> std::result::Result<Value, VNextError> {
        Ok(raw.clone())
    }

    fn weight_schema(&self, _: &Value) -> std::result::Result<WeightSchema, VNextError> {
        let dimensions = vec![VOCAB as u64, VOCAB as u64];
        let components = ["embedding", "head"].map(|name| WeightComponentSpec {
            id: id(format!("component.{name}")),
            role: WeightComponentRole::Values,
            external_names: vec![format!("{name}.weight")],
            dimensions: dimensions.clone(),
            encoding: WeightEncoding::Dense {
                element_type: ElementType::F16,
            },
            required: true,
        });
        let tensors = ["embedding", "head"].map(|name| WeightTensorSpec {
            id: id(format!("weight.{name}")),
            dimensions: dimensions.clone(),
            logical_element_type: ElementType::F16,
            physical_layout: PhysicalWeightLayout::Dense {
                component_id: id(format!("component.{name}")),
            },
            required: true,
        });
        Ok(WeightSchema {
            format_id: id("weight-format.safetensors.dense"),
            layout_id: id("weight-layout.fixture.sampling-readback"),
            version: ContractVersion::new(1, 0),
            components: components.into(),
            tensors: tensors.into(),
        })
    }

    fn numerical_profiles(
        &self,
        _: &Value,
    ) -> std::result::Result<FamilyNumericalProfiles, VNextError> {
        let profile_id = NumericalProfileId::new("fixture.f32").unwrap();
        FamilyNumericalProfiles::new(
            self.family_id(),
            ContractVersion::new(1, 0),
            vec![NumericalExecutionProfile {
                id: profile_id.clone(),
                family_id: self.family_id().clone(),
                version: ContractVersion::new(1, 0),
                primary_activation: id("value.hidden"),
                boundaries: BTreeMap::from([
                    (id("value.hidden"), ElementType::F32),
                    (id("value.logits"), ElementType::F32),
                    (id("value.selected"), ElementType::U32),
                ]),
                states: vec![],
                kv_storage: vec![],
                operations: [
                    TOKEN_EMBEDDING_F32_MASTER_OPERATION_ID,
                    LAST_TOKEN_DENSE_LINEAR_F32_OPERATION_ID,
                    LAST_TOKEN_MASKED_ARGMAX_F32_OPERATION_ID,
                ]
                .map(|name| NumericalOperationContract {
                    operation_id: id(name),
                    version: ContractVersion::new(1, 0),
                    multiplication_type: (name == LAST_TOKEN_DENSE_LINEAR_F32_OPERATION_ID)
                        .then_some(ElementType::F32),
                    accumulation_type: (name == LAST_TOKEN_DENSE_LINEAR_F32_OPERATION_ID)
                        .then_some(ElementType::F32),
                    staged_arithmetic: None,
                    composite_arithmetic: None,
                })
                .into(),
            }],
            vec![profile_id],
        )
    }

    fn semantic_program(
        &self,
        _: &Value,
        _: &NumericalExecutionProfile,
    ) -> std::result::Result<ModelProgram, VNextError> {
        let attribute = |name, value| (id(name), SemanticValue::Unsigned(value));
        ModelProgram::new(
            self.family_id().clone(),
            [
                "value.tokens",
                "value.mask",
                "value.repeat_ids",
                "value.repeat_offsets",
                "value.penalty",
            ]
            .map(id)
            .into(),
            vec![ProgramBlock {
                id: "block.language".into(),
                nodes: vec![
                    ProgramNode {
                        id: id("node.embedding"),
                        operation_id: id(TOKEN_EMBEDDING_F32_MASTER_OPERATION_ID),
                        required_version: ContractVersion::new(1, 0),
                        work: ProgramNodeWorkSpec::tokens(id("value.tokens"), 0),
                        inputs: vec![id("value.tokens"), id("value.weight.embedding")],
                        outputs: vec![id("value.hidden")],
                        attributes: BTreeMap::from([
                            attribute("hidden_size", VOCAB as u64),
                            attribute("vocab_size", VOCAB as u64),
                        ]),
                    },
                    ProgramNode {
                        id: id("node.logits"),
                        operation_id: id(LAST_TOKEN_DENSE_LINEAR_F32_OPERATION_ID),
                        required_version: ContractVersion::new(1, 0),
                        work: ProgramNodeWorkSpec::tokens(id("value.hidden"), 0),
                        inputs: vec![id("value.hidden"), id("value.weight.head")],
                        outputs: vec![id("value.logits")],
                        attributes: BTreeMap::from([
                            attribute("hidden_size", VOCAB as u64),
                            attribute("out_features", VOCAB as u64),
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
                        attributes: BTreeMap::from([attribute("vocab_size", VOCAB as u64)]),
                    },
                ],
            }],
            vec![],
            ["embedding", "head"]
                .map(|name| WeightReference {
                    weight_id: id(format!("weight.{name}")),
                    value_id: id(format!("value.weight.{name}")),
                    tensor: ProgramTensorSpec {
                        dimensions: vec![VOCAB as u64, VOCAB as u64],
                        element_type: ElementType::F16,
                        layout: ResolvedTensorLayout::Contiguous,
                    },
                })
                .into(),
            vec![id("value.logits"), id("value.selected")],
        )
    }

    fn semantic_metadata(
        &self,
        _: &Value,
    ) -> std::result::Result<ModelSemanticMetadata, VNextError> {
        Ok(ModelSemanticMetadata {
            template: TemplateMetadata {
                template: "{{ messages }}".into(),
                source_file: "tokenizer_config.json".into(),
                sha256: format!("{:x}", Sha256::digest(TOKENIZER_CONFIG)),
            },
            special_tokens: SpecialTokenMetadata {
                bos_token_id: Some(0),
                eos_token_ids: BTreeSet::from([3]),
                pad_token_id: None,
                collision_policy: SpecialTokenCollisionPolicy::require_distinct(),
            },
        })
    }
}

struct FamilyRegistry(TypedFamilyRegistration<LanguageFamily>);
impl ModelFamilyRegistry for FamilyRegistry {
    fn registrations(&self) -> Vec<&dyn ModelFamilyRegistration> {
        vec![&self.0]
    }
}

pub(super) struct Fixture<R: DeviceRuntime> {
    _directory: tempfile::TempDir,
    pub(super) executor: VNextModelExecutor<R>,
}

impl<R: DeviceRuntime> Fixture<R> {
    pub(super) fn new(
        mode: SamplingReadbackMode,
        device: ferrum_types::Device,
        runtime: Arc<R>,
        registry: OperationRuntimeRegistry<R>,
        materializers: WeightMaterializerRegistry,
        catalog: CapabilityCatalog,
        nonfinite_token: Option<usize>,
    ) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let raw = json!({"vocabulary_size": VOCAB});
        std::fs::write(
            directory.path().join("config.json"),
            serde_json::to_vec(&raw).unwrap(),
        )
        .unwrap();
        std::fs::write(
            directory.path().join("tokenizer.json"),
            br#"{"version":"1.0"}"#,
        )
        .unwrap();
        std::fs::write(
            directory.path().join("tokenizer_config.json"),
            TOKENIZER_CONFIG,
        )
        .unwrap();
        // Negative logits, finite ties and a near tie; the identity head avoids
        // attention/normalization arithmetic obscuring the output-selection test.
        let mut embedding = [
            -1.0,
            -1.0,
            -2.0,
            -3.0,
            3.0,
            2.0,
            1.0,
            -4.0,
            0.0,
            0.0,
            -1.0,
            -2.0,
            1.0,
            1.0009765625,
            -1.0,
            -2.0,
        ]
        .map(|value| f16::from_f32(value).to_le_bytes())
        .concat();
        if let Some(token) = nonfinite_token {
            assert!(token < VOCAB);
            for value in embedding[token * VOCAB * 2..(token + 1) * VOCAB * 2].chunks_exact_mut(2) {
                value.copy_from_slice(&f16::NAN.to_bits().to_le_bytes());
            }
        }
        let head = (0..VOCAB * VOCAB)
            .flat_map(|index| {
                f16::from_f32(if index / VOCAB == index % VOCAB {
                    1.0
                } else {
                    0.0
                })
                .to_le_bytes()
            })
            .collect::<Vec<_>>();
        serialize_to_file(
            [
                (
                    "embedding.weight",
                    TensorView::new(Dtype::F16, vec![VOCAB, VOCAB], &embedding).unwrap(),
                ),
                (
                    "head.weight",
                    TensorView::new(Dtype::F16, vec![VOCAB, VOCAB], &head).unwrap(),
                ),
            ],
            &None::<std::collections::HashMap<String, String>>,
            &directory.path().join("model.safetensors"),
        )
        .unwrap();
        let sources = Arc::new(
            ProductionModelSourceBundle::open_colocated_safetensors(directory.path()).unwrap(),
        );
        let family_registry = FamilyRegistry(TypedFamilyRegistration::new(LanguageFamily));
        let definition = family_registry.0.define(&raw).unwrap();
        let family = family_registry
            .0
            .prepare(
                &definition,
                &NumericalProfileId::new("fixture.f32").unwrap(),
            )
            .unwrap();
        let descriptor = CausalLanguageModelDescriptor::new(
            "fixture-language",
            32,
            VOCAB,
            1,
            1,
            1,
            VOCAB,
            VOCAB,
            MAX_TOKENS,
            ferrum_types::DataType::FP32,
        )
        .unwrap();
        let prepared = PreparedProductionModel::new(
            family,
            Arc::new(SafetensorsArchive::open(directory.path()).unwrap()),
            descriptor,
            sources,
        )
        .unwrap();
        let info = prepared.model_info("sampling-readback-fixture".into(), device);
        let mut engine = EngineConfig::default();
        engine.scheduler.max_running_requests = 3;
        engine.scheduler.max_waiting_requests = 3;
        engine.batching.max_num_batched_tokens = MAX_TOKENS;
        engine.runtime.max_model_len = Some(MAX_TOKENS);
        engine.runtime.sampling_readback_mode = mode;
        if matches!(&info.device, Device::CUDA(_)) {
            // This attention-free fixture installs the portable CUDA composition.
            engine.runtime.attention_execution_policy = AttentionExecutionPolicy::Portable;
        }
        engine.backend.enable_reusable_execution = false;
        let executor = VNextModelExecutor::from_runtime_composition(
            &prepared,
            info,
            &engine,
            runtime,
            registry,
            materializers,
            WeightMaterializerSelection::exact(id(IDENTITY_WEIGHT_MATERIALIZER_ID)),
            catalog,
            |prepared, policy, catalog, compilation| {
                resolve_plan(
                    prepared,
                    policy,
                    catalog,
                    compilation,
                    &family_registry,
                    &definition,
                )
            },
        )
        .unwrap();
        Self {
            _directory: directory,
            executor,
        }
    }
}

fn resolve_plan(
    prepared: &PreparedProductionModel,
    runtime: &ResolvedRuntimePolicy,
    catalog: &CapabilityCatalog,
    compilation: &ProgramPlanCompilation,
    registry: &FamilyRegistry,
    definition: &ModelFamilyDefinition,
) -> ferrum_types::Result<ResolvedModelPlan> {
    let plan = compilation.executable().execution_plan();
    let numerical = NumericalProfileResolution::from_static_plan(
        NumericalExecutionPolicy::Auto,
        ferrum_types::KvStorageFormat::F16,
        definition,
        prepared.family(),
        catalog,
        runtime,
        plan,
        vec![],
    )
    .unwrap();
    let source = prepared.sources();
    let provider = catalog.engine_providers().values().next().unwrap();
    let inputs = ResolvedModelPlanInputs {
        original_sources: source.original_sources().clone(),
        resolved_sources: source.resolved_sources().clone(),
        config: ModelConfigFingerprint {
            source_file: "config.json".into(),
            sha256: source
                .fingerprint(ModelArtifactSourceRole::Semantic, "config.json")
                .unwrap()
                .sha256
                .clone(),
            typed_config_sha256: prepared.family().config_fingerprint().into(),
        },
        external_metadata_id: prepared.family().external_metadata_id().clone(),
        prepared_family: prepared.family().clone(),
        numerical_execution: numerical.clone(),
        tokenizer: TokenizerDescriptor {
            tokenizer_id: id("tokenizer.fixture.sampling-readback"),
            source_file: "tokenizer.json".into(),
            sha256: source
                .fingerprint(ModelArtifactSourceRole::Tokenizer, "tokenizer.json")
                .unwrap()
                .sha256
                .clone(),
            vocabulary_size: VOCAB as u64,
        },
        device: catalog.device().clone(),
        capabilities: catalog.clone(),
        runtime: runtime.clone(),
        engine: EngineSelection {
            provider_id: provider.provider_id().clone(),
            contract_version: provider.contract_version(),
            implementation_fingerprint: provider.implementation_fingerprint().into(),
        },
        execution_plan: plan.clone(),
        sampling: SamplingPolicy::new(
            RationalValue::new(0, 1).unwrap(),
            RationalValue::new(1, 1).unwrap(),
            None,
            RationalValue::new(0, 1).unwrap(),
            RationalValue::new(0, 1).unwrap(),
            RationalValue::new(1, 1).unwrap(),
            1,
            TriStatePolicy::ModelDefault,
        )
        .unwrap(),
        stop: StopPolicy {
            maximum_output_tokens: 8,
            token_ids: BTreeSet::from([2]),
            strings: vec![],
            collision_policy: StopTokenCollisionPolicy::require_distinct(),
        },
        structured_output: StructuredOutputPolicy::Disabled,
    };
    // Verified typed evidence is supplied through the same construction API as
    // product composition; no serialized plan or substitute provider is trusted.
    let values = [
        (
            ResolutionField::OriginalSources,
            serde_json::to_value(&inputs.original_sources).unwrap(),
            ResolutionDecisionSource::UserInput,
        ),
        (
            ResolutionField::ResolvedSources,
            serde_json::to_value(&inputs.resolved_sources).unwrap(),
            ResolutionDecisionSource::TypedModelResolution,
        ),
        (
            ResolutionField::Config,
            serde_json::to_value(&inputs.config).unwrap(),
            ResolutionDecisionSource::TypedModelResolution,
        ),
        (
            ResolutionField::ExternalMetadata,
            serde_json::to_value(&inputs.external_metadata_id).unwrap(),
            ResolutionDecisionSource::TypedModelResolution,
        ),
        (
            ResolutionField::Family,
            serde_json::to_value(inputs.prepared_family.family_id()).unwrap(),
            ResolutionDecisionSource::TypedModelResolution,
        ),
        (
            ResolutionField::NumericalExecution,
            serde_json::to_value(&inputs.numerical_execution).unwrap(),
            ResolutionDecisionSource::Planner,
        ),
        (
            ResolutionField::WeightSchema,
            serde_json::to_value(inputs.prepared_family.weight_schema()).unwrap(),
            ResolutionDecisionSource::TypedModelResolution,
        ),
        (
            ResolutionField::WeightFormat,
            serde_json::to_value(&inputs.prepared_family.weight_schema().format_id).unwrap(),
            ResolutionDecisionSource::TypedModelResolution,
        ),
        (
            ResolutionField::Tokenizer,
            serde_json::to_value(&inputs.tokenizer).unwrap(),
            ResolutionDecisionSource::TypedModelResolution,
        ),
        (
            ResolutionField::Template,
            serde_json::to_value(&inputs.prepared_family.metadata().template).unwrap(),
            ResolutionDecisionSource::TypedModelResolution,
        ),
        (
            ResolutionField::SpecialTokens,
            serde_json::to_value(&inputs.prepared_family.metadata().special_tokens).unwrap(),
            ResolutionDecisionSource::TypedModelResolution,
        ),
        (
            ResolutionField::Device,
            serde_json::to_value(&inputs.device).unwrap(),
            ResolutionDecisionSource::CapabilityResolution,
        ),
        (
            ResolutionField::Capabilities,
            serde_json::to_value(&inputs.capabilities).unwrap(),
            ResolutionDecisionSource::CapabilityResolution,
        ),
        (
            ResolutionField::RuntimePreset,
            json!({"policy_id": inputs.runtime.policy_id(), "version": inputs.runtime.version(), "scheduling": inputs.runtime.scheduling()}),
            ResolutionDecisionSource::RuntimePreset,
        ),
        (
            ResolutionField::RuntimeMemory,
            serde_json::to_value(inputs.runtime.memory()).unwrap(),
            ResolutionDecisionSource::RuntimePreset,
        ),
        (
            ResolutionField::Admission,
            serde_json::to_value(inputs.runtime.admission()).unwrap(),
            ResolutionDecisionSource::RuntimePreset,
        ),
        (
            ResolutionField::Engine,
            serde_json::to_value(&inputs.engine).unwrap(),
            ResolutionDecisionSource::CapabilityResolution,
        ),
        (
            ResolutionField::ExecutionPlan,
            json!(inputs.execution_plan.plan_hash().as_str()),
            ResolutionDecisionSource::Planner,
        ),
        (
            ResolutionField::Sampling,
            serde_json::to_value(&inputs.sampling).unwrap(),
            ResolutionDecisionSource::ProductDefault,
        ),
        (
            ResolutionField::Stop,
            serde_json::to_value(&inputs.stop).unwrap(),
            ResolutionDecisionSource::ProductDefault,
        ),
        (
            ResolutionField::StructuredOutput,
            serde_json::to_value(&inputs.structured_output).unwrap(),
            ResolutionDecisionSource::ProductDefault,
        ),
    ];
    let mut evidence = Vec::new();
    let mut bindings = Vec::new();
    for (index, (field, value, source)) in values.into_iter().enumerate() {
        let artifact: ResolutionArtifactId = id(format!("artifact.sampling-readback.{index}"));
        evidence.push(
            ResolutionSourceEvidence::new(
                artifact.clone(),
                source,
                ResolutionSourceProvenance::Upstream {
                    producer_id: "fixture.sampling-readback".into(),
                    producer_version: ContractVersion::new(1, 0),
                    producer_implementation_fingerprint: ResolutionFingerprint::new(format!(
                        "{:x}",
                        Sha256::digest(include_bytes!("fixture.rs"))
                    ))
                    .unwrap(),
                    revision: "fixture-v1".into(),
                    artifact_locator: format!("sampling-readback/{index}"),
                },
                serde_json::to_vec(&json!({"chosen": value})).unwrap(),
                BTreeSet::from(["/chosen".into()]),
                &JSON_RESOLUTION_SOURCE_PARSER,
            )
            .unwrap(),
        );
        bindings.push(
            ResolutionDecisionBinding::new(
                field,
                source,
                id(format!("reason.sampling-readback.{index}")),
                artifact,
                "/chosen",
            )
            .unwrap(),
        );
    }
    let context = ResolvedPlanValidationContext::new(
        registry,
        &evidence,
        compilation.node_resolutions(),
        catalog.device(),
        catalog,
        runtime,
        &numerical,
    )
    .with_completion_retention(compilation.completion_retention().clone());
    ResolvedModelPlan::new(inputs, bindings, &context)
        .map_err(|error| ferrum_types::FerrumError::model(error.to_string()))
}
