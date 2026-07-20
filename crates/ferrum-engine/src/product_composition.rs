//! Product-owned composition of one immutable vNext model plan.

#![cfg_attr(not(feature = "cuda"), allow(dead_code))]

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use ferrum_interfaces::vnext::{
    CapabilityCatalog, ContractVersion, DeviceRuntime, EngineSelection, ExecutablePlanView,
    FileFingerprint, ModelConfigFingerprint, ModelSourceKind, OperationRuntimeRegistry,
    OriginalModelSource, OriginalModelSources, ProgramPlanCompilation, RationalValue,
    ResolutionArtifactId, ResolutionDecisionBinding, ResolutionDecisionSource, ResolutionField,
    ResolutionFingerprint, ResolutionReasonId, ResolutionSourceEvidence,
    ResolutionSourceProvenance, ResolvedModelPlan, ResolvedModelPlanInputs, ResolvedModelSource,
    ResolvedModelSources, ResolvedPlanValidationContext, ResolvedRuntimePolicy, SamplingPolicy,
    SpecialTokenRole, StopPolicy, StopTokenCollisionPolicy, StructuredOutputPolicy,
    TokenizerDescriptor, TokenizerId, TriStatePolicy, JSON_RESOLUTION_SOURCE_PARSER,
};
use ferrum_models::vnext::{PreparedProductionModel, ProductionModelFamilyRegistry};
use ferrum_models::VNextModelExecutor;
use ferrum_types::{
    EngineConfig, FerrumError, ModelInfo, ModelSource, ResponseFormat, Result, SamplingParams,
};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

const PRODUCT_COMPOSITION_VERSION: ContractVersion = ContractVersion::new(1, 0);
const PRODUCT_COMPOSITION_PRODUCER: &str = "ferrum.product-composition";
const DEFAULT_SAMPLER_SEED: u64 = 42;

pub(crate) fn create_vnext_executor<R: DeviceRuntime>(
    model_dir: &Path,
    engine: &EngineConfig,
    prepared: PreparedProductionModel,
    model_info: ModelInfo,
    runtime: Arc<R>,
    operation_registry: OperationRuntimeRegistry<R>,
    catalog: CapabilityCatalog,
) -> Result<VNextModelExecutor<R>> {
    VNextModelExecutor::from_runtime_composition(
        prepared,
        model_info,
        engine,
        runtime,
        operation_registry,
        catalog,
        |prepared, runtime, catalog, compilation| {
            resolve_model_plan(model_dir, engine, prepared, catalog, runtime, compilation)
        },
    )
}

fn resolve_model_plan(
    model_dir: &Path,
    engine: &EngineConfig,
    prepared: &PreparedProductionModel,
    catalog: &CapabilityCatalog,
    runtime: &ResolvedRuntimePolicy,
    compilation: &ProgramPlanCompilation,
) -> Result<ResolvedModelPlan> {
    let source_files = fingerprint_source_files(model_dir)?;
    let config_sha256 = source_file(&source_files, "config.json")?.sha256.clone();
    let tokenizer_sha256 = source_file(&source_files, "tokenizer.json")?.sha256.clone();
    let (original_source, resolved_source) =
        resolve_source_identity(model_dir, engine, source_files)?;
    let original_sources = OriginalModelSources {
        semantic: original_source.clone(),
        tokenizer: original_source.clone(),
        weights: original_source,
    };
    let resolved_sources = ResolvedModelSources {
        semantic: resolved_source.clone(),
        tokenizer: resolved_source.clone(),
        weights: resolved_source,
    };
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
    );
    ResolvedModelPlan::new(inputs, bindings, &context)
        .map_err(|error| FerrumError::model(format!("resolve vNext product plan: {error}")))
}

fn fingerprint_source_files(model_dir: &Path) -> Result<Vec<FileFingerprint>> {
    ["config.json", "tokenizer.json", "tokenizer_config.json"]
        .into_iter()
        .map(|relative_path| {
            let path = model_dir.join(relative_path);
            let bytes = fs::read(&path).map_err(|error| {
                FerrumError::model(format!(
                    "read required vNext source file {}: {error}",
                    path.display()
                ))
            })?;
            if bytes.is_empty() {
                return Err(FerrumError::model(format!(
                    "required vNext source file {} is empty",
                    path.display()
                )));
            }
            Ok(FileFingerprint {
                relative_path: relative_path.to_owned(),
                size_bytes: bytes.len() as u64,
                sha256: format!("{:x}", Sha256::digest(bytes)),
            })
        })
        .collect()
}

fn source_file<'a>(files: &'a [FileFingerprint], name: &str) -> Result<&'a FileFingerprint> {
    files
        .iter()
        .find(|file| file.relative_path == name)
        .ok_or_else(|| FerrumError::internal(format!("source fingerprint for {name} was lost")))
}

fn resolve_source_identity(
    model_dir: &Path,
    engine: &EngineConfig,
    files: Vec<FileFingerprint>,
) -> Result<(OriginalModelSource, ResolvedModelSource)> {
    let source = engine.model.source.as_ref().ok_or_else(|| {
        FerrumError::config(
            "registered vNext models require EngineConfig.model.source from the product resolver",
        )
    })?;
    let canonical_path = canonical_path(model_dir)?;
    let manifest_revision = source_manifest_revision(&files)?;
    let (kind, location, requested_revision, canonical_location, resolved_revision) = match source {
        ModelSource::Local(location) => {
            let kind = if model_dir.is_file() {
                ModelSourceKind::LocalFile
            } else {
                ModelSourceKind::LocalDirectory
            };
            (
                kind,
                location.clone(),
                None,
                canonical_path,
                manifest_revision,
            )
        }
        ModelSource::HuggingFace {
            repo_id, revision, ..
        } => (
            ModelSourceKind::Repository,
            repo_id.clone(),
            revision.clone(),
            repo_id.clone(),
            huggingface_snapshot_revision(model_dir)?,
        ),
        ModelSource::Url { .. } | ModelSource::S3 { .. } => {
            return Err(FerrumError::unsupported(
                "URL and S3 sources are not yet accepted by the resolved vNext product plan",
            ))
        }
    };
    Ok((
        OriginalModelSource {
            kind,
            location,
            requested_revision,
        },
        ResolvedModelSource {
            canonical_location,
            resolved_revision,
            files,
        },
    ))
}

fn canonical_path(path: &Path) -> Result<String> {
    path.canonicalize()
        .map_err(|error| FerrumError::model(format!("canonicalize {}: {error}", path.display())))
        .map(|path| path.display().to_string())
}

fn huggingface_snapshot_revision(model_dir: &Path) -> Result<String> {
    let revision = model_dir
        .file_name()
        .and_then(|value| value.to_str())
        .filter(|value| !value.is_empty());
    let snapshot_parent = model_dir
        .parent()
        .and_then(Path::file_name)
        .and_then(|value| value.to_str());
    match (snapshot_parent, revision) {
        (Some("snapshots"), Some(revision)) => Ok(revision.to_owned()),
        _ => Err(FerrumError::config(format!(
            "Hugging Face source resolved outside a snapshots/<revision> directory: {}",
            model_dir.display()
        ))),
    }
}

fn source_manifest_revision(files: &[FileFingerprint]) -> Result<String> {
    let bytes = serde_json::to_vec(files)
        .map_err(|error| FerrumError::internal(format!("serialize source manifest: {error}")))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
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
    let collision_policy = if token_ids.is_empty() {
        StopTokenCollisionPolicy::require_distinct()
    } else {
        StopTokenCollisionPolicy::new(BTreeSet::from([SpecialTokenRole::Eos]))
            .map_err(|error| FerrumError::config(error.to_string()))?
    };
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
    const PLANNER_FIELDS: &[ResolutionField] = &[ResolutionField::ExecutionPlan];
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
    fn local_source_revision_is_content_derived() {
        let files = vec![FileFingerprint {
            relative_path: "config.json".to_owned(),
            size_bytes: 2,
            sha256: format!("{:x}", Sha256::digest(b"{}")),
        }];
        let first = source_manifest_revision(&files).unwrap();
        let second = source_manifest_revision(&files).unwrap();
        assert_eq!(first, second);
        assert_eq!(first.len(), 64);
    }
}
