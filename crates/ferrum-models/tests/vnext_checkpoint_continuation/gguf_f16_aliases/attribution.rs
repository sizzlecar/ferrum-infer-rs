//! Calls the same completed-static-initialization witness as product startup.
//! The real CUDA receipt is retained by Fixture; no receipt is reconstructed.
use super::*;
use ferrum_kernels::gguf_f16_projection_materializer::GgufF16ProjectionInventoryV1;
use serde_json::json;

struct Registry<'a>(&'a dyn ModelFamilyRegistration);
impl ModelFamilyRegistry for Registry<'_> {
    fn registrations(&self) -> Vec<&dyn ModelFamilyRegistration> {
        vec![self.0]
    }
}

pub(super) fn assert_product_witness(
    registration: &dyn ModelFamilyRegistration,
    family: &PreparedModelFamily,
    fixture: &Fixture,
    inventory: &GgufF16ProjectionInventoryV1,
) {
    assert_product_witness_inventory(
        registration,
        family,
        fixture,
        inventory.converted_f16_bytes,
        None,
    );
}

/// Reuse the original completed-static-initialization receipt and resolved
/// plan. The packet is extra declared storage for the same source population.
pub(super) fn assert_product_witness_inventory(
    registration: &dyn ModelFamilyRegistration,
    family: &PreparedModelFamily,
    fixture: &Fixture,
    converted_f16_bytes: u64,
    fragment_bytes: Option<u64>,
) {
    let catalog = &fixture._composition._catalog;
    let runtime = &fixture.runtime_policy;
    let compilation = &fixture.compilation;
    let plan = compilation.executable().execution_plan();
    let definition = registration.define(family.canonical_config()).unwrap();
    let numerical = NumericalProfileResolution::from_static_plan(
        ferrum_types::NumericalExecutionPolicy::Require(id(family.numerical_profile().id.as_str())),
        ferrum_types::KvStorageFormat::F16,
        &definition,
        family,
        catalog,
        runtime,
        plan,
        Vec::new(),
    )
    .unwrap();
    let original = OriginalModelSource {
        kind: ModelSourceKind::Repository,
        location: "fixture/rn-f16-attribution".into(),
        requested_revision: Some("typed-fixture".into()),
    };
    let mut resolved = ResolvedModelSource {
        canonical_location: "fixture/rn-f16-attribution".into(),
        resolved_revision: "typed-fixture".into(),
        files: vec![
            FileFingerprint {
                relative_path: "config.json".into(),
                size_bytes: serde_json::to_vec(family.canonical_config()).unwrap().len() as u64,
                sha256: family.config_fingerprint().into(),
            },
            FileFingerprint {
                relative_path: "weights.schema.json".into(),
                size_bytes: serde_json::to_vec(family.weight_schema()).unwrap().len() as u64,
                sha256: family.weight_schema().fingerprint().unwrap(),
            },
            FileFingerprint {
                relative_path: "tokenizer.fixture".into(),
                size_bytes: 1,
                sha256: format!("{:x}", Sha256::digest(b"0")),
            },
            FileFingerprint {
                relative_path: family.metadata().template.source_file.clone(),
                size_bytes: family.metadata().template.template.len() as u64,
                sha256: family.metadata().template.sha256.clone(),
            },
        ],
    };
    // ResolvedModelPlan canonicalizes each role's file list before checking
    // externally parsed field evidence. Bind the same ordered source facts;
    // never relax the production equality check to accommodate this fixture.
    resolved
        .files
        .sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    let engine = catalog.engine_providers().values().next().unwrap();
    assert_eq!(catalog.engine_providers().len(), 1);
    let inputs = ResolvedModelPlanInputs {
        original_sources: OriginalModelSources {
            semantic: original.clone(),
            tokenizer: original.clone(),
            weights: original,
        },
        resolved_sources: ResolvedModelSources {
            semantic: resolved.clone(),
            tokenizer: resolved.clone(),
            weights: resolved,
        },
        config: ModelConfigFingerprint {
            source_file: "config.json".into(),
            sha256: family.config_fingerprint().into(),
            typed_config_sha256: family.config_fingerprint().into(),
        },
        external_metadata_id: family.external_metadata_id().clone(),
        prepared_family: family.clone(),
        numerical_execution: numerical.clone(),
        tokenizer: TokenizerDescriptor {
            tokenizer_id: id("tokenizer.fixture"),
            source_file: "tokenizer.fixture".into(),
            sha256: format!("{:x}", Sha256::digest(b"0")),
            vocabulary_size: family
                .weight_schema()
                .tensor(&id("weight.embedding"))
                .unwrap()
                .dimensions[0],
        },
        device: catalog.device().clone(),
        capabilities: catalog.clone(),
        runtime: runtime.clone(),
        engine: EngineSelection {
            provider_id: engine.provider_id().clone(),
            contract_version: engine.contract_version(),
            implementation_fingerprint: engine.implementation_fingerprint().into(),
        },
        execution_plan: plan.clone(),
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
            token_ids: BTreeSet::new(),
            strings: Vec::new(),
            collision_policy: StopTokenCollisionPolicy::require_distinct(),
        },
        structured_output: StructuredOutputPolicy::Disabled,
    };
    let (evidence, bindings) = evidence(&inputs);
    let registry = Registry(registration);
    let context = ResolvedPlanValidationContext::new(
        &registry,
        &evidence,
        compilation.node_resolutions(),
        catalog.device(),
        catalog,
        runtime,
        &numerical,
    )
    .with_completion_retention(compilation.completion_retention().clone());
    let resolved = ResolvedModelPlan::new(inputs, bindings, &context).unwrap();
    let witness = StaticProviderAttributionWitness::from_completed_static_initialization(
        &resolved,
        &fixture.static_initialization_receipt,
    )
    .unwrap();
    let Some(denominator) =
        QuantizedProviderAttributionDenominator::from_prepared_family(family).unwrap()
    else {
        assert!(witness.is_none());
        return;
    };
    let witness = witness.expect("quantized sources require complete source attribution");
    let json = serde_json::to_value(&witness).unwrap();
    assert_eq!(
        json["schema"],
        if fragment_bytes.is_some() {
            "ferrum.vnext.provider-attribution.v3"
        } else {
            "ferrum.vnext.provider-attribution.v2"
        }
    );
    assert_eq!(
        witness.provider_attribution().denominator_sha256(),
        denominator.sha256()
    );
    assert_eq!(
        witness.provider_attribution().expected_quant_tensor_count() as usize,
        denominator.quant_tensor_count()
    );
    assert_eq!(
        witness
            .provider_attribution()
            .attributed_quant_tensor_count() as usize,
        denominator.quant_tensor_count()
    );
    let declared = &json["declared_dense_execution"];
    assert!(declared["source_quant_tensor_count"].as_u64().unwrap() > 0);
    assert!(declared["dense_f16_component_bytes"].as_u64().unwrap() > 0);
    assert!(declared["dense_f16_component_bytes"].as_u64().unwrap() <= converted_f16_bytes);
    if let Some(bytes) = fragment_bytes {
        let packet = &declared["rn_fragment_execution"];
        assert_eq!(packet["component_bytes"].as_u64(), Some(bytes));
        assert!(packet["component_count"].as_u64().unwrap() > 0);
        assert_eq!(packet["mapping_sha256"].as_str().unwrap().len(), 64);
        let converted = declared["source_quant_tensor_count"].as_u64().unwrap();
        let retained = declared["retained_quantized_source_tensor_count"]
            .as_u64()
            .unwrap();
        assert_eq!(
            converted + retained,
            denominator.quant_tensor_count() as u64,
            "two representations cannot double the original denominator"
        );
    } else {
        assert!(declared.get("rn_fragment_execution").is_none());
    }
    assert_eq!(witness.fallback_counts().silent(), 0);
    assert_eq!(witness.fallback_counts().dense(), 0);
    assert!(witness.binding().execution_contract_fingerprint().is_some());
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
        Sha256::digest(include_bytes!("attribution.rs"))
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
