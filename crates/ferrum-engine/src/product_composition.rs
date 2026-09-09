//! Product-owned composition of one immutable vNext model plan.

#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use std::collections::BTreeSet;
use std::sync::Arc;

use ferrum_interfaces::vnext::{
    CapabilityCatalog, ContractVersion, DeviceRuntime, EngineSelection, ExecutablePlanView,
    ModelArtifactSourceRole, ModelConfigFingerprint, NumericalProfileRejection,
    NumericalProfileRejectionStage, NumericalProfileResolution, OperationRuntimeRegistry,
    ProgramPlanCompilation, RationalValue, ResolutionArtifactId, ResolutionDecisionBinding,
    ResolutionDecisionSource, ResolutionField, ResolutionFingerprint, ResolutionReasonId,
    ResolutionSourceEvidence, ResolutionSourceProvenance, ResolvedModelPlan,
    ResolvedModelPlanInputs, ResolvedPlanValidationContext, ResolvedRuntimePolicy, SamplingPolicy,
    SpecialTokenRole, StopPolicy, StopTokenCollisionPolicy, StructuredOutputPolicy,
    TokenizerDescriptor, TokenizerId, TriStatePolicy, WeightMaterializerRegistry,
    WeightMaterializerSelection, JSON_RESOLUTION_SOURCE_PARSER,
};
use ferrum_models::vnext::{
    DefinedProductionModel, PreparedProductionModel, ProductionModelFamilyRegistry,
};
use ferrum_models::{VNextExecutorConfig, VNextModelExecutor, VNextRuntimeComposition};
use ferrum_types::{EngineConfig, FerrumError, ModelInfo, ResponseFormat, Result, SamplingParams};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

const PRODUCT_COMPOSITION_VERSION: ContractVersion = ContractVersion::new(2, 0);
const PRODUCT_COMPOSITION_PRODUCER: &str = "ferrum.product-composition";
const DEFAULT_SAMPLER_SEED: u64 = 42;

pub(crate) fn create_vnext_executor<R: DeviceRuntime>(
    engine: &EngineConfig,
    defined: &DefinedProductionModel,
    runtime: Arc<R>,
    operation_registry: OperationRuntimeRegistry<R>,
    weight_materializers: WeightMaterializerRegistry,
    catalog: CapabilityCatalog,
    select_materializer: impl Fn(
        &ferrum_interfaces::vnext::PreparedModelFamily,
    ) -> std::result::Result<
        WeightMaterializerSelection,
        ferrum_interfaces::vnext::VNextError,
    >,
) -> Result<VNextModelExecutor<R>> {
    create_vnext_executor_with_configuration(
        engine,
        defined,
        runtime,
        operation_registry,
        weight_materializers,
        catalog,
        select_materializer,
        |info, runtime| VNextExecutorConfig::from_engine_config(engine, info, runtime),
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn create_vnext_executor_with_configuration<R: DeviceRuntime>(
    engine: &EngineConfig,
    defined: &DefinedProductionModel,
    runtime: Arc<R>,
    operation_registry: OperationRuntimeRegistry<R>,
    weight_materializers: WeightMaterializerRegistry,
    catalog: CapabilityCatalog,
    select_materializer: impl Fn(
        &ferrum_interfaces::vnext::PreparedModelFamily,
    ) -> std::result::Result<
        WeightMaterializerSelection,
        ferrum_interfaces::vnext::VNextError,
    >,
    executor_config: impl Fn(&ModelInfo, &R) -> Result<VNextExecutorConfig>,
) -> Result<VNextModelExecutor<R>> {
    let composition =
        VNextRuntimeComposition::new(runtime, operation_registry, weight_materializers, catalog);
    let candidates = defined
        .definition()
        .numerical_profiles()
        .candidates(&engine.numerical_execution)
        .map_err(|error| FerrumError::config(error.to_string()))?;
    let mut rejected = Vec::new();
    for profile in candidates {
        let prepared = match defined.prepare(&profile.id) {
            Ok(prepared) => prepared,
            Err(error) => {
                rejected.push(NumericalProfileRejection {
                    profile_id: profile.id.clone(),
                    stage: NumericalProfileRejectionStage::FamilyPreparation,
                    reason: error.to_string(),
                });
                continue;
            }
        };
        let model_info =
            prepared.model_info(engine.model.model_id.clone(), engine.backend.device.clone());
        let config = executor_config(&model_info, composition.runtime())?;
        let materializer = match select_materializer(prepared.family()) {
            Ok(selection) => selection,
            Err(error) => {
                rejected.push(NumericalProfileRejection {
                    profile_id: profile.id.clone(),
                    stage: NumericalProfileRejectionStage::WeightMaterializer,
                    reason: error.to_string(),
                });
                continue;
            }
        };
        let compiled =
            match composition.compile_model(&prepared, model_info, engine, config, materializer) {
                Ok(compiled) => compiled,
                Err(error) => {
                    rejected.push(NumericalProfileRejection {
                        profile_id: profile.id.clone(),
                        stage: NumericalProfileRejectionStage::ProgramCompilation,
                        reason: error.to_string(),
                    });
                    continue;
                }
            };
        tracing::info!(requested_numerical_policy = ?engine.numerical_execution,
            numerical_profile = %profile.id, rejected_numerical_profiles = ?rejected,
            family_id = %prepared.family().family_id(),
            "Selected numerical profile using the complete static program");
        return compiled.initialize(|prepared, runtime, catalog, compilation| {
            let numerical_execution = NumericalProfileResolution::from_static_plan(
                engine.numerical_execution.clone(),
                defined.definition(),
                prepared.family(),
                catalog,
                runtime,
                compilation.executable().execution_plan(),
                rejected,
            )
            .map_err(|error| FerrumError::model(error.to_string()))?;
            resolve_model_plan(
                engine,
                prepared,
                catalog,
                runtime,
                compilation,
                &numerical_execution,
            )
        });
    }
    Err(FerrumError::unsupported(format!(
        "no declared numerical profile satisfies {:?}: {}",
        engine.numerical_execution,
        rejected
            .iter()
            .map(|rejection| format!(
                "{} ({:?}): {}",
                rejection.profile_id, rejection.stage, rejection.reason
            ))
            .collect::<Vec<_>>()
            .join("; ")
    )))
}

fn resolve_model_plan(
    engine: &EngineConfig,
    prepared: &PreparedProductionModel,
    catalog: &CapabilityCatalog,
    runtime: &ResolvedRuntimePolicy,
    compilation: &ProgramPlanCompilation,
    numerical_execution: &NumericalProfileResolution,
) -> Result<ResolvedModelPlan> {
    let sources = prepared.sources();
    let config_sha256 = sources
        .fingerprint(ModelArtifactSourceRole::Semantic, "config.json")
        .ok_or_else(|| FerrumError::internal("semantic config.json fingerprint was lost"))?
        .sha256
        .clone();
    let tokenizer_sha256 = sources
        .fingerprint(ModelArtifactSourceRole::Tokenizer, "tokenizer.json")
        .ok_or_else(|| FerrumError::internal("tokenizer.json fingerprint was lost"))?
        .sha256
        .clone();
    let original_sources = sources.original_sources().clone();
    let resolved_sources = sources.resolved_sources().clone();
    let family = prepared.family();

    let engine_provider = match catalog
        .engine_providers()
        .values()
        .collect::<Vec<_>>()
        .as_slice()
    {
        [provider] => *provider,
        providers => {
            return Err(FerrumError::config(format!(
                "vNext product composition requires exactly one engine provider, got {}",
                providers.len()
            )))
        }
    };
    let engine_selection = EngineSelection {
        provider_id: engine_provider.provider_id().clone(),
        contract_version: engine_provider.contract_version(),
        implementation_fingerprint: engine_provider.implementation_fingerprint().to_owned(),
    };
    let (sampling, stop, structured_output) = generation_defaults(engine, prepared)?;

    let inputs = ResolvedModelPlanInputs {
        original_sources,
        resolved_sources,
        config: ModelConfigFingerprint {
            source_file: "config.json".to_owned(),
            sha256: config_sha256,
            typed_config_sha256: family.config_fingerprint().to_owned(),
        },
        external_metadata_id: family.external_metadata_id().clone(),
        prepared_family: family.clone(),
        numerical_execution: numerical_execution.clone(),
        tokenizer: TokenizerDescriptor {
            tokenizer_id: TokenizerId::new("tokenizer.huggingface.json")
                .map_err(|error| FerrumError::tokenizer(error.to_string()))?,
            source_file: "tokenizer.json".to_owned(),
            sha256: tokenizer_sha256,
            vocabulary_size: prepared.descriptor().vocabulary_size() as u64,
        },
        device: catalog.device().clone(),
        capabilities: catalog.clone(),
        runtime: runtime.clone(),
        engine: engine_selection,
        execution_plan: compilation.executable().execution_plan().clone(),
        sampling,
        stop,
        structured_output,
    };
    let (source_evidence, bindings) = resolution_evidence(&inputs)?;
    let registry = ProductionModelFamilyRegistry::new()?;
    let context = ResolvedPlanValidationContext::new(
        &registry,
        &source_evidence,
        compilation.node_resolutions(),
        catalog.device(),
        catalog,
        runtime,
        numerical_execution,
    )
    .with_completion_retention(compilation.completion_retention().clone());
    ResolvedModelPlan::new(inputs, bindings, &context)
        .map_err(|error| FerrumError::model(format!("resolve vNext product plan: {error}")))
}

fn generation_defaults(
    engine: &EngineConfig,
    prepared: &PreparedProductionModel,
) -> Result<(SamplingPolicy, StopPolicy, StructuredOutputPolicy)> {
    let params = &engine.sampling.default_params;
    validate_representable_sampling(params)?;
    let top_k = params
        .top_k
        .map(|value| {
            u32::try_from(value)
                .map_err(|_| FerrumError::config("sampling.top_k exceeds the vNext limit"))
        })
        .transpose()?;
    let sampling = SamplingPolicy::new(
        rational_from_f32(params.temperature, "sampling.temperature")?,
        rational_from_f32(params.top_p, "sampling.top_p")?,
        top_k,
        rational_from_f32(params.min_p.unwrap_or(0.0), "sampling.min_p")?,
        rational_from_f32(params.presence_penalty, "sampling.presence_penalty")?,
        rational_from_f32(params.repetition_penalty, "sampling.repetition_penalty")?,
        params.seed.unwrap_or(DEFAULT_SAMPLER_SEED),
        TriStatePolicy::ModelDefault,
    )
    .map_err(|error| FerrumError::config(error.to_string()))?;
    let maximum_output_tokens = u32::try_from(params.max_tokens)
        .map_err(|_| FerrumError::config("sampling.max_tokens exceeds the vNext limit"))?;
    let mut strings = params.stop_sequences.clone();
    strings.sort();
    strings.dedup();
    let token_ids = prepared
        .family()
        .metadata()
        .special_tokens
        .eos_token_ids
        .clone();
    let special_tokens = &prepared.family().metadata().special_tokens;
    let collision_policy = stop_collision_policy(
        &token_ids,
        special_tokens.bos_token_id,
        special_tokens.pad_token_id,
    )?;
    let stop = StopPolicy {
        maximum_output_tokens,
        token_ids,
        strings,
        collision_policy,
    };
    let structured_output = match &params.response_format {
        ResponseFormat::Text => StructuredOutputPolicy::Disabled,
        ResponseFormat::JsonObject => StructuredOutputPolicy::JsonObject,
        ResponseFormat::JsonSchema(schema) => StructuredOutputPolicy::JsonSchema {
            schema_sha256: format!("{:x}", Sha256::digest(schema.as_bytes())),
        },
    };
    Ok((sampling, stop, structured_output))
}

fn stop_collision_policy(
    token_ids: &BTreeSet<u32>,
    bos_token_id: Option<u32>,
    pad_token_id: Option<u32>,
) -> Result<StopTokenCollisionPolicy> {
    let mut model_roles = BTreeSet::new();
    if !token_ids.is_empty() {
        model_roles.insert(SpecialTokenRole::Eos);
    }
    if bos_token_id.is_some_and(|token_id| token_ids.contains(&token_id)) {
        model_roles.insert(SpecialTokenRole::Bos);
    }
    if pad_token_id.is_some_and(|token_id| token_ids.contains(&token_id)) {
        model_roles.insert(SpecialTokenRole::Pad);
    }
    StopTokenCollisionPolicy::new(model_roles)
        .map_err(|error| FerrumError::config(error.to_string()))
}

fn validate_representable_sampling(params: &SamplingParams) -> Result<()> {
    params.validate()?;
    if params.frequency_penalty != 0.0
        || params.tfs.is_some()
        || params.typical_p.is_some()
        || params.mirostat.is_some()
    {
        return Err(FerrumError::unsupported(
            "the resolved vNext startup policy does not yet represent frequency_penalty, tfs, typical_p, or mirostat",
        ));
    }
    Ok(())
}

fn rational_from_f32(value: f32, field: &'static str) -> Result<RationalValue> {
    if !value.is_finite() {
        return Err(FerrumError::config(format!("{field} must be finite")));
    }
    let rendered = value.to_string();
    let (mantissa, exponent) = rendered
        .split_once(['e', 'E'])
        .map(|(mantissa, exponent)| exponent.parse::<i32>().map(|exponent| (mantissa, exponent)))
        .transpose()
        .map_err(|error| FerrumError::config(format!("parse {field}: {error}")))?
        .unwrap_or((rendered.as_str(), 0));
    let negative = mantissa.starts_with('-');
    let unsigned = mantissa.trim_start_matches(['-', '+']);
    let (whole, fraction) = unsigned.split_once('.').unwrap_or((unsigned, ""));
    let digits = format!("{whole}{fraction}");
    let mut numerator = digits
        .parse::<i128>()
        .map_err(|error| FerrumError::config(format!("parse {field}: {error}")))?;
    if negative {
        numerator = -numerator;
    }
    let decimal_places = i32::try_from(fraction.len())
        .map_err(|_| FerrumError::config(format!("{field} decimal width exceeds i32")))?;
    let scale = decimal_places - exponent;
    let denominator = if scale > 0 {
        10_u128
            .checked_pow(scale as u32)
            .ok_or_else(|| FerrumError::config(format!("{field} denominator overflow")))?
    } else {
        let multiplier = 10_i128
            .checked_pow((-scale) as u32)
            .ok_or_else(|| FerrumError::config(format!("{field} numerator overflow")))?;
        numerator = numerator
            .checked_mul(multiplier)
            .ok_or_else(|| FerrumError::config(format!("{field} numerator overflow")))?;
        1
    };
    let numerator = i64::try_from(numerator)
        .map_err(|_| FerrumError::config(format!("{field} numerator exceeds i64")))?;
    let denominator = u64::try_from(denominator)
        .map_err(|_| FerrumError::config(format!("{field} denominator exceeds u64")))?;
    RationalValue::new(numerator, denominator)
        .map_err(|error| FerrumError::config(error.to_string()))
}

fn resolution_evidence(
    inputs: &ResolvedModelPlanInputs,
) -> Result<(
    Vec<ResolutionSourceEvidence<'static>>,
    Vec<ResolutionDecisionBinding>,
)> {
    const USER_FIELDS: &[ResolutionField] = &[ResolutionField::OriginalSources];
    const MODEL_FIELDS: &[ResolutionField] = &[
        ResolutionField::ResolvedSources,
        ResolutionField::Config,
        ResolutionField::ExternalMetadata,
        ResolutionField::Family,
        ResolutionField::WeightSchema,
        ResolutionField::WeightFormat,
        ResolutionField::Tokenizer,
        ResolutionField::Template,
        ResolutionField::SpecialTokens,
    ];
    const CAPABILITY_FIELDS: &[ResolutionField] = &[
        ResolutionField::Device,
        ResolutionField::Capabilities,
        ResolutionField::Engine,
    ];
    const RUNTIME_FIELDS: &[ResolutionField] = &[
        ResolutionField::RuntimePreset,
        ResolutionField::RuntimeMemory,
        ResolutionField::Admission,
    ];
    const PLANNER_FIELDS: &[ResolutionField] = &[
        ResolutionField::ExecutionPlan,
        ResolutionField::NumericalExecution,
    ];
    const DEFAULT_FIELDS: &[ResolutionField] = &[
        ResolutionField::Sampling,
        ResolutionField::Stop,
        ResolutionField::StructuredOutput,
    ];
    let groups = [
        (ResolutionDecisionSource::UserInput, USER_FIELDS),
        (ResolutionDecisionSource::TypedModelResolution, MODEL_FIELDS),
        (
            ResolutionDecisionSource::CapabilityResolution,
            CAPABILITY_FIELDS,
        ),
        (ResolutionDecisionSource::RuntimePreset, RUNTIME_FIELDS),
        (ResolutionDecisionSource::Planner, PLANNER_FIELDS),
        (ResolutionDecisionSource::ProductDefault, DEFAULT_FIELDS),
    ];
    let implementation_fingerprint = ResolutionFingerprint::new(format!(
        "{:x}",
        Sha256::digest(include_bytes!("product_composition.rs"))
    ))
    .map_err(|error| FerrumError::internal(error.to_string()))?;
    let mut evidence = Vec::with_capacity(groups.len());
    let mut bindings = Vec::new();
    for (source, fields) in groups {
        let source_name = decision_source_name(source);
        let artifact_id = ResolutionArtifactId::new(format!("artifact.product.{source_name}"))
            .map_err(|error| FerrumError::internal(error.to_string()))?;
        let mut document = Map::new();
        let mut field_paths = BTreeSet::new();
        for field in fields {
            let name = field.as_str();
            document.insert(name.to_owned(), resolution_value(inputs, *field)?);
            let path = format!("/{name}");
            field_paths.insert(path.clone());
            bindings.push(
                ResolutionDecisionBinding::new(
                    *field,
                    source,
                    ResolutionReasonId::new(format!("reason.product.{name}"))
                        .map_err(|error| FerrumError::internal(error.to_string()))?,
                    artifact_id.clone(),
                    path,
                )
                .map_err(|error| FerrumError::internal(error.to_string()))?,
            );
        }
        let source_bytes = serde_json::to_vec(&Value::Object(document)).map_err(|error| {
            FerrumError::internal(format!(
                "serialize {source_name} resolution evidence: {error}"
            ))
        })?;
        evidence.push(
            ResolutionSourceEvidence::new(
                artifact_id,
                source,
                ResolutionSourceProvenance::Upstream {
                    producer_id: PRODUCT_COMPOSITION_PRODUCER.to_owned(),
                    producer_version: PRODUCT_COMPOSITION_VERSION,
                    producer_implementation_fingerprint: implementation_fingerprint.clone(),
                    revision: env!("CARGO_PKG_VERSION").to_owned(),
                    artifact_locator: format!("product-composition/{source_name}"),
                },
                source_bytes,
                field_paths,
                &JSON_RESOLUTION_SOURCE_PARSER,
            )
            .map_err(|error| FerrumError::internal(error.to_string()))?,
        );
    }
    Ok((evidence, bindings))
}

fn resolution_value(inputs: &ResolvedModelPlanInputs, field: ResolutionField) -> Result<Value> {
    let value = match field {
        ResolutionField::OriginalSources => serde_json::to_value(&inputs.original_sources),
        ResolutionField::ResolvedSources => serde_json::to_value(&inputs.resolved_sources),
        ResolutionField::Config => serde_json::to_value(&inputs.config),
        ResolutionField::ExternalMetadata => serde_json::to_value(&inputs.external_metadata_id),
        ResolutionField::Family => serde_json::to_value(inputs.prepared_family.family_id()),
        ResolutionField::NumericalExecution => serde_json::to_value(&inputs.numerical_execution),
        ResolutionField::WeightSchema => {
            serde_json::to_value(inputs.prepared_family.weight_schema())
        }
        ResolutionField::WeightFormat => {
            serde_json::to_value(&inputs.prepared_family.weight_schema().format_id)
        }
        ResolutionField::Tokenizer => serde_json::to_value(&inputs.tokenizer),
        ResolutionField::Template => {
            serde_json::to_value(&inputs.prepared_family.metadata().template)
        }
        ResolutionField::SpecialTokens => {
            serde_json::to_value(&inputs.prepared_family.metadata().special_tokens)
        }
        ResolutionField::Device => serde_json::to_value(&inputs.device),
        ResolutionField::Capabilities => serde_json::to_value(&inputs.capabilities),
        ResolutionField::RuntimePreset => Ok(json!({
            "policy_id": inputs.runtime.policy_id(),
            "version": inputs.runtime.version(),
            "scheduling": inputs.runtime.scheduling(),
        })),
        ResolutionField::RuntimeMemory => serde_json::to_value(inputs.runtime.memory()),
        ResolutionField::Admission => serde_json::to_value(inputs.runtime.admission()),
        ResolutionField::Engine => serde_json::to_value(&inputs.engine),
        ResolutionField::ExecutionPlan => Ok(json!(inputs.execution_plan.plan_hash().as_str())),
        ResolutionField::Sampling => serde_json::to_value(&inputs.sampling),
        ResolutionField::Stop => serde_json::to_value(&inputs.stop),
        ResolutionField::StructuredOutput => serde_json::to_value(&inputs.structured_output),
    };
    value.map_err(|error| FerrumError::internal(format!("serialize resolution field: {error}")))
}

const fn decision_source_name(source: ResolutionDecisionSource) -> &'static str {
    match source {
        ResolutionDecisionSource::UserInput => "user-input",
        ResolutionDecisionSource::CommandLine => "command-line",
        ResolutionDecisionSource::ConfigFile => "config-file",
        ResolutionDecisionSource::ModelMetadata => "model-metadata",
        ResolutionDecisionSource::TypedModelResolution => "typed-model",
        ResolutionDecisionSource::ProductDefault => "product-default",
        ResolutionDecisionSource::RuntimePreset => "runtime-preset",
        ResolutionDecisionSource::CapabilityResolution => "capability",
        ResolutionDecisionSource::Planner => "planner",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimal_sampling_values_are_canonical_rationals() {
        let value = rational_from_f32(1.1, "test").unwrap();
        assert_eq!(value.numerator(), 11);
        assert_eq!(value.denominator(), 10);
        let value = rational_from_f32(0.000_001, "test").unwrap();
        assert_eq!(value.numerator(), 1);
        assert_eq!(value.denominator(), 1_000_000);
    }

    #[test]
    fn stop_collision_policy_includes_every_model_role_aliased_by_default_eos() {
        let token_ids = BTreeSet::from([10, 20]);
        let policy = stop_collision_policy(&token_ids, Some(10), Some(20)).unwrap();
        assert_eq!(
            policy.allowed_model_roles(),
            &BTreeSet::from([
                SpecialTokenRole::Bos,
                SpecialTokenRole::Eos,
                SpecialTokenRole::Pad,
            ])
        );

        let distinct = stop_collision_policy(&BTreeSet::new(), Some(10), Some(20)).unwrap();
        assert!(distinct.allowed_model_roles().is_empty());
    }
}
