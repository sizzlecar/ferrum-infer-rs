use super::*;

pub(crate) struct EventModelRegistry {
    pub(crate) registration: TypedFamilyRegistration<TestFamily>,
}

impl EventModelRegistry {
    fn new() -> Self {
        Self {
            registration: TypedFamilyRegistration::new(TestFamily),
        }
    }
}

impl ModelFamilyRegistry for EventModelRegistry {
    fn registrations(&self) -> Vec<&dyn ModelFamilyRegistration> {
        vec![&self.registration]
    }
}

pub(crate) fn plan_resolutions_with_mode(
    suffix: &str,
    family: &PreparedModelFamily,
    catalog: &CapabilityCatalog,
    runtime: &ResolvedRuntimePolicy,
    operation_registry: &OperationRuntimeRegistry<TestRuntime>,
    no_static: bool,
) -> Vec<PlanNodeResolution> {
    let weight = || {
        if no_static {
            binding(
                "value.weight",
                ResolvedValueRole::Input,
                1,
                BufferUsage::Activations,
                &format!("resource.weight.{suffix}"),
            )
        } else {
            ResolvedValueBinding::new(
                id("value.weight"),
                ResolvedValueRole::Input,
                1,
                resolved_tensor(),
                TensorAccess::Read,
                AliasPolicy::NoAlias,
                BufferUsage::Weights,
                ResolvedValueStorage::composite(vec![ResolvedStorageComponent::new(
                    Some(id("weight.component")),
                    id(format!("resource.weight.{suffix}")),
                    0,
                    16,
                    ElementType::F32,
                )
                .unwrap()])
                .unwrap(),
            )
            .unwrap()
        }
    };
    let first_values = vec![
        binding(
            "value.input",
            ResolvedValueRole::Input,
            0,
            BufferUsage::Activations,
            &format!("resource.input.{suffix}"),
        ),
        weight(),
        binding(
            "value.middle",
            ResolvedValueRole::Output,
            0,
            BufferUsage::Activations,
            &format!("resource.middle.{suffix}"),
        ),
    ];
    let second_values = vec![
        binding(
            "value.middle",
            ResolvedValueRole::Input,
            0,
            BufferUsage::Activations,
            &format!("resource.middle.{suffix}"),
        ),
        weight(),
        binding(
            "value.output",
            ResolvedValueRole::Output,
            0,
            BufferUsage::Activations,
            &format!("resource.output.{suffix}"),
        ),
    ];
    let planning = operation_registry.planning();
    [("node.first", first_values), ("node.second", second_values)]
        .into_iter()
        .map(|(node, values)| {
            PlanNodeResolution::resolve(
                family,
                catalog,
                runtime,
                &planning,
                id(node),
                values,
                BTreeSet::new(),
                None,
            )
            .unwrap()
        })
        .collect()
}

pub(crate) const RESOLUTION_FIELDS: [ResolutionField; 20] = [
    ResolutionField::OriginalSource,
    ResolutionField::ResolvedSource,
    ResolutionField::Config,
    ResolutionField::ExternalMetadata,
    ResolutionField::Family,
    ResolutionField::WeightSchema,
    ResolutionField::WeightFormat,
    ResolutionField::Tokenizer,
    ResolutionField::Template,
    ResolutionField::SpecialTokens,
    ResolutionField::Device,
    ResolutionField::Capabilities,
    ResolutionField::RuntimePreset,
    ResolutionField::RuntimeMemory,
    ResolutionField::Admission,
    ResolutionField::Engine,
    ResolutionField::ExecutionPlan,
    ResolutionField::Sampling,
    ResolutionField::Stop,
    ResolutionField::StructuredOutput,
];

pub(crate) fn resolution_source(field: ResolutionField) -> ResolutionDecisionSource {
    match field {
        ResolutionField::OriginalSource => ResolutionDecisionSource::UserInput,
        ResolutionField::ResolvedSource
        | ResolutionField::Config
        | ResolutionField::ExternalMetadata
        | ResolutionField::Family
        | ResolutionField::WeightSchema
        | ResolutionField::WeightFormat
        | ResolutionField::Tokenizer
        | ResolutionField::Template
        | ResolutionField::SpecialTokens => ResolutionDecisionSource::TypedModelResolution,
        ResolutionField::Device | ResolutionField::Capabilities | ResolutionField::Engine => {
            ResolutionDecisionSource::CapabilityResolution
        }
        ResolutionField::RuntimePreset
        | ResolutionField::RuntimeMemory
        | ResolutionField::Admission => ResolutionDecisionSource::RuntimePreset,
        ResolutionField::ExecutionPlan => ResolutionDecisionSource::Planner,
        ResolutionField::Sampling | ResolutionField::Stop | ResolutionField::StructuredOutput => {
            ResolutionDecisionSource::ProductDefault
        }
    }
}

pub(crate) fn resolution_value(inputs: &ResolvedModelPlanInputs, field: ResolutionField) -> Value {
    match field {
        ResolutionField::OriginalSource => serde_json::to_value(&inputs.original_source).unwrap(),
        ResolutionField::ResolvedSource => serde_json::to_value(&inputs.resolved_source).unwrap(),
        ResolutionField::Config => serde_json::to_value(&inputs.config).unwrap(),
        ResolutionField::ExternalMetadata => {
            serde_json::to_value(&inputs.external_metadata_id).unwrap()
        }
        ResolutionField::Family => {
            serde_json::to_value(inputs.prepared_family.family_id()).unwrap()
        }
        ResolutionField::WeightSchema => {
            serde_json::to_value(inputs.prepared_family.weight_schema()).unwrap()
        }
        ResolutionField::WeightFormat => {
            serde_json::to_value(&inputs.prepared_family.weight_schema().format_id).unwrap()
        }
        ResolutionField::Tokenizer => serde_json::to_value(&inputs.tokenizer).unwrap(),
        ResolutionField::Template => {
            serde_json::to_value(&inputs.prepared_family.metadata().template).unwrap()
        }
        ResolutionField::SpecialTokens => {
            serde_json::to_value(&inputs.prepared_family.metadata().special_tokens).unwrap()
        }
        ResolutionField::Device => serde_json::to_value(&inputs.device).unwrap(),
        ResolutionField::Capabilities => serde_json::to_value(&inputs.capabilities).unwrap(),
        ResolutionField::RuntimePreset => json!({
            "policy_id": inputs.runtime.policy_id(),
            "version": inputs.runtime.version(),
            "scheduling": inputs.runtime.scheduling(),
        }),
        ResolutionField::RuntimeMemory => serde_json::to_value(inputs.runtime.memory()).unwrap(),
        ResolutionField::Admission => serde_json::to_value(inputs.runtime.admission()).unwrap(),
        ResolutionField::Engine => serde_json::to_value(&inputs.engine).unwrap(),
        ResolutionField::ExecutionPlan => json!(inputs.execution_plan.plan_hash().as_str()),
        ResolutionField::Sampling => serde_json::to_value(&inputs.sampling).unwrap(),
        ResolutionField::Stop => serde_json::to_value(&inputs.stop).unwrap(),
        ResolutionField::StructuredOutput => {
            serde_json::to_value(&inputs.structured_output).unwrap()
        }
    }
}

pub(crate) fn resolved_model_plan(
    plan: &ExecutionPlan,
    suffix: &str,
    operation_registry: &OperationRuntimeRegistry<TestRuntime>,
) -> ResolvedModelPlan {
    resolved_model_plan_with_mode(plan, suffix, operation_registry, false)
}

pub(crate) fn no_static_resolved_model_plan(
    plan: &ExecutionPlan,
    suffix: &str,
    operation_registry: &OperationRuntimeRegistry<TestRuntime>,
) -> ResolvedModelPlan {
    resolved_model_plan_with_mode(plan, suffix, operation_registry, true)
}

pub(crate) fn resolved_model_plan_with_mode(
    plan: &ExecutionPlan,
    suffix: &str,
    operation_registry: &OperationRuntimeRegistry<TestRuntime>,
    no_static: bool,
) -> ResolvedModelPlan {
    let registry = EventModelRegistry::new();
    let family = registry
        .registration
        .prepare(&json!({"width": 4, "no_static": no_static}))
        .unwrap();
    let catalog = catalog();
    let runtime = policy();
    let resolutions = plan_resolutions_with_mode(
        suffix,
        &family,
        &catalog,
        &runtime,
        operation_registry,
        no_static,
    );
    let rebuilt = ExecutionPlan::build(
        PlanBuildRequest::new(&family, &catalog, &runtime, resolutions.clone()).unwrap(),
    )
    .unwrap();
    assert_eq!(&rebuilt, plan);

    let config_fingerprint = family.config_fingerprint().to_owned();
    let inputs = ResolvedModelPlanInputs {
        original_source: OriginalModelSource {
            kind: ModelSourceKind::Repository,
            location: "repo/event-model".to_owned(),
            requested_revision: Some("main".to_owned()),
        },
        resolved_source: ResolvedModelSource {
            canonical_location: "repo/event-model".to_owned(),
            resolved_revision: "0123456789abcdef".to_owned(),
            files: vec![
                FileFingerprint {
                    relative_path: "config.json".to_owned(),
                    size_bytes: 11,
                    sha256: config_fingerprint.clone(),
                },
                FileFingerprint {
                    relative_path: "template.json".to_owned(),
                    size_bytes: 30,
                    sha256: sha('a'),
                },
                FileFingerprint {
                    relative_path: "tokenizer.json".to_owned(),
                    size_bytes: 20,
                    sha256: sha('b'),
                },
            ],
        },
        config: ModelConfigFingerprint {
            source_file: "config.json".to_owned(),
            sha256: config_fingerprint.clone(),
            typed_config_sha256: config_fingerprint,
        },
        external_metadata_id: id("metadata.event"),
        prepared_family: family.clone(),
        tokenizer: TokenizerDescriptor {
            tokenizer_id: id("tokenizer.event"),
            source_file: "tokenizer.json".to_owned(),
            sha256: sha('b'),
            vocabulary_size: 1024,
        },
        device: catalog.device().clone(),
        capabilities: catalog.clone(),
        runtime: runtime.clone(),
        engine: EngineSelection {
            provider_id: id("provider.engine.event"),
            contract_version: ContractVersion::new(1, 0),
            implementation_fingerprint: sha('d'),
        },
        execution_plan: plan.clone(),
        sampling: SamplingPolicy::new(
            RationalValue::new(0, 1).unwrap(),
            RationalValue::new(1, 1).unwrap(),
            None,
            RationalValue::new(0, 1).unwrap(),
            RationalValue::new(0, 1).unwrap(),
            RationalValue::new(1, 1).unwrap(),
            9271,
            TriStatePolicy::ModelDefault,
        )
        .unwrap(),
        stop: StopPolicy {
            maximum_output_tokens: 32,
            token_ids: BTreeSet::from([3]),
            strings: vec!["stop".to_owned()],
        },
        structured_output: StructuredOutputPolicy::JsonObject,
    };

    let mut bindings = Vec::new();
    let mut evidence = Vec::new();
    for (index, field) in RESOLUTION_FIELDS.into_iter().enumerate() {
        let source = resolution_source(field);
        let artifact_id: ResolutionArtifactId = id(format!("artifact.event.{index}"));
        let path = "/chosen".to_owned();
        evidence.push(
            ResolutionSourceEvidence::new(
                artifact_id.clone(),
                source,
                ResolutionSourceProvenance::Upstream {
                    producer_id: "fixture.event".to_owned(),
                    producer_version: ContractVersion::new(1, 0),
                    producer_implementation_fingerprint: ResolutionFingerprint::new(sha('e'))
                        .unwrap(),
                    revision: "fixture-v1".to_owned(),
                    artifact_locator: format!("event/{index}"),
                },
                serde_json::to_vec(&json!({"chosen": resolution_value(&inputs, field)})).unwrap(),
                BTreeSet::from([path.clone()]),
                &JSON_RESOLUTION_SOURCE_PARSER,
            )
            .unwrap(),
        );
        bindings.push(
            ResolutionDecisionBinding::new(
                field,
                source,
                id(format!("reason.event.{index}")),
                artifact_id,
                path,
            )
            .unwrap(),
        );
    }
    let context = ResolvedPlanValidationContext::new(
        &registry,
        &evidence,
        &resolutions,
        catalog.device(),
        &catalog,
        &runtime,
    );
    ResolvedModelPlan::new(inputs, bindings, &context).unwrap()
}
