//! Minimal typed resolver for the tiny model test. All digests and decisions
//! describe the files, family, catalog and compiled plan actually consumed.
use super::*;
use crate::vnext::{PreparedProductionModel, ProductionModelFamilyRegistry};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

pub(super) fn resolve(
    prepared: &PreparedProductionModel,
    runtime: &ResolvedRuntimePolicy,
    catalog: &CapabilityCatalog,
    compilation: &ProgramPlanCompilation,
    numerical: NumericalProfileResolution,
) -> Result<ResolvedModelPlan> {
    let sources = prepared.sources();
    let family = prepared.family();
    let provider = catalog.engine_providers().values().next().unwrap();
    assert_eq!(catalog.engine_providers().len(), 1);
    let inputs = ResolvedModelPlanInputs {
        original_sources: sources.original_sources().clone(),
        resolved_sources: sources.resolved_sources().clone(),
        config: ModelConfigFingerprint {
            source_file: "config.json".into(),
            sha256: sources
                .fingerprint(ModelArtifactSourceRole::Semantic, "config.json")
                .unwrap()
                .sha256
                .clone(),
            typed_config_sha256: family.config_fingerprint().into(),
        },
        external_metadata_id: family.external_metadata_id().clone(),
        prepared_family: family.clone(),
        numerical_execution: numerical.clone(),
        tokenizer: TokenizerDescriptor {
            tokenizer_id: TokenizerId::new("tokenizer.fixture.json").unwrap(),
            source_file: "tokenizer.json".into(),
            sha256: sources
                .fingerprint(ModelArtifactSourceRole::Tokenizer, "tokenizer.json")
                .unwrap()
                .sha256
                .clone(),
            vocabulary_size: prepared.descriptor().vocabulary_size() as u64,
        },
        device: catalog.device().clone(),
        capabilities: catalog.clone(),
        runtime: runtime.clone(),
        engine: EngineSelection {
            provider_id: provider.provider_id().clone(),
            contract_version: provider.contract_version(),
            implementation_fingerprint: provider.implementation_fingerprint().into(),
        },
        execution_plan: compilation.executable().execution_plan().clone(),
        sampling: SamplingPolicy::new(
            RationalValue::new(0, 1).unwrap(),
            RationalValue::new(1, 1).unwrap(),
            None,
            RationalValue::new(0, 1).unwrap(),
            RationalValue::new(0, 1).unwrap(),
            RationalValue::new(1, 1).unwrap(),
            42,
            TriStatePolicy::ModelDefault,
        )
        .unwrap(),
        stop: StopPolicy {
            maximum_output_tokens: 8,
            token_ids: family.metadata().special_tokens.eos_token_ids.clone(),
            strings: Vec::new(),
            // This fixture deliberately uses the model EOS as its product
            // stop token, matching the real product's explicit EOS alias.
            collision_policy: StopTokenCollisionPolicy::new(BTreeSet::from([
                ferrum_interfaces::vnext::SpecialTokenRole::Eos,
            ]))
            .unwrap(),
        },
        structured_output: StructuredOutputPolicy::Disabled,
    };
    let (evidence, bindings) = evidence(&inputs);
    let families = ProductionModelFamilyRegistry::new()?;
    let context = ResolvedPlanValidationContext::new(
        &families,
        &evidence,
        compilation.node_resolutions(),
        catalog.device(),
        catalog,
        runtime,
        &numerical,
    )
    .with_completion_retention(compilation.completion_retention().clone());
    ResolvedModelPlan::new(inputs, bindings, &context)
        .map_err(|error| FerrumError::model(error.to_string()))
}

fn evidence(
    inputs: &ResolvedModelPlanInputs,
) -> (
    Vec<ResolutionSourceEvidence<'static>>,
    Vec<ResolutionDecisionBinding>,
) {
    use ResolutionDecisionSource as S;
    use ResolutionField as F;
    let groups: &[(ResolutionDecisionSource, &[ResolutionField])] = &[
        (S::UserInput, &[F::OriginalSources]),
        (
            S::TypedModelResolution,
            &[
                F::ResolvedSources,
                F::Config,
                F::ExternalMetadata,
                F::Family,
                F::WeightSchema,
                F::WeightFormat,
                F::Tokenizer,
                F::Template,
                F::SpecialTokens,
            ],
        ),
        (
            S::CapabilityResolution,
            &[F::Device, F::Capabilities, F::Engine],
        ),
        (
            S::RuntimePreset,
            &[F::RuntimePreset, F::RuntimeMemory, F::Admission],
        ),
        (S::Planner, &[F::ExecutionPlan, F::NumericalExecution]),
        (
            S::ProductDefault,
            &[F::Sampling, F::Stop, F::StructuredOutput],
        ),
    ];
    let fingerprint = ResolutionFingerprint::new(format!(
        "{:x}",
        Sha256::digest(include_bytes!("resolver.rs"))
    ))
    .unwrap();
    let mut sources = Vec::new();
    let mut bindings = Vec::new();
    for (index, (source, fields)) in groups.iter().enumerate() {
        let artifact = ResolutionArtifactId::new(format!("artifact.guarded-test.{index}")).unwrap();
        let mut document = serde_json::Map::new();
        let mut paths = BTreeSet::new();
        for field in *fields {
            let name = field.as_str();
            document.insert(name.into(), value(inputs, *field));
            let path = format!("/{name}");
            paths.insert(path.clone());
            bindings.push(
                ResolutionDecisionBinding::new(
                    *field,
                    *source,
                    ResolutionReasonId::new(format!("reason.guarded-test.{name}")).unwrap(),
                    artifact.clone(),
                    path,
                )
                .unwrap(),
            );
        }
        sources.push(
            ResolutionSourceEvidence::new(
                artifact,
                *source,
                ResolutionSourceProvenance::Upstream {
                    producer_id: "ferrum.guarded-model-test".into(),
                    producer_version: ContractVersion::new(1, 0),
                    producer_implementation_fingerprint: fingerprint.clone(),
                    revision: env!("CARGO_PKG_VERSION").into(),
                    artifact_locator: format!("guarded-test/{index}"),
                },
                serde_json::to_vec(&document).unwrap(),
                paths,
                &JSON_RESOLUTION_SOURCE_PARSER,
            )
            .unwrap(),
        );
    }
    (sources, bindings)
}

fn value(i: &ResolvedModelPlanInputs, field: ResolutionField) -> Value {
    use ResolutionField as F;
    match field {
        F::OriginalSources => json!(i.original_sources),
        F::ResolvedSources => json!(i.resolved_sources),
        F::Config => json!(i.config),
        F::ExternalMetadata => json!(i.external_metadata_id),
        F::Family => json!(i.prepared_family.family_id()),
        F::NumericalExecution => json!(i.numerical_execution),
        F::WeightSchema => json!(i.prepared_family.weight_schema()),
        F::WeightFormat => json!(i.prepared_family.weight_schema().format_id),
        F::Tokenizer => json!(i.tokenizer),
        F::Template => json!(i.prepared_family.metadata().template),
        F::SpecialTokens => json!(i.prepared_family.metadata().special_tokens),
        F::Device => json!(i.device),
        F::Capabilities => json!(i.capabilities),
        F::RuntimePreset => {
            json!({"policy_id": i.runtime.policy_id(), "version": i.runtime.version(), "scheduling": i.runtime.scheduling()})
        }
        F::RuntimeMemory => json!(i.runtime.memory()),
        F::Admission => json!(i.runtime.admission()),
        F::Engine => json!(i.engine),
        F::ExecutionPlan => json!(i.execution_plan.plan_hash().as_str()),
        F::Sampling => json!(i.sampling),
        F::Stop => json!(i.stop),
        F::StructuredOutput => json!(i.structured_output),
    }
}
