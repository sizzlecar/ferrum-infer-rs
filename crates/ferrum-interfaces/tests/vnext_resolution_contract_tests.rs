mod vnext_core_contract;

use vnext_core_contract::*;

fn bytes_sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

#[test]
fn runtime_policy_roundtrip_preserves_sequence_fit_policy() {
    let policy = policy(4096);
    assert_eq!(
        policy.admission().sequence_fit_policy,
        AdmissionFitPolicy::ImmediateOnly
    );
    let wire = serde_json::to_value(&policy).unwrap();
    assert_eq!(wire["admission"]["sequence_fit_policy"], "immediate_only");
    assert_eq!(
        serde_json::from_value::<ResolvedRuntimePolicy>(wire).unwrap(),
        policy
    );
}

fn resolution_test_error(field: &str, reason: &str) -> VNextError {
    VNextError::InvalidResolvedModelPlan {
        field: field.to_owned(),
        reason: reason.to_owned(),
    }
}

fn decision_source(field: ResolutionField) -> ResolutionDecisionSource {
    match field {
        ResolutionField::OriginalSources => ResolutionDecisionSource::UserInput,
        ResolutionField::Config => ResolutionDecisionSource::ModelMetadata,
        ResolutionField::ResolvedSources
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

const RESOLUTION_FIELDS: [ResolutionField; 20] = [
    ResolutionField::OriginalSources,
    ResolutionField::ResolvedSources,
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

const RESOLUTION_CONFIG_BYTES: &[u8] = br#"{"width":4}"#;

struct LockedConfigResolutionParser;

impl ResolutionSourceParser for LockedConfigResolutionParser {
    fn descriptor(&self) -> Result<ResolutionParserDescriptor, VNextError> {
        ResolutionParserDescriptor::new(
            "resolution-parser.test-locked-config",
            ContractVersion::new(1, 0),
            ResolutionFingerprint::new(sha('9')).unwrap(),
        )
    }

    fn parse(
        &self,
        source: ResolutionDecisionSource,
        provenance: &ResolutionSourceProvenance,
        source_bytes: &[u8],
    ) -> Result<Value, VNextError> {
        if source != ResolutionDecisionSource::ModelMetadata
            || !matches!(
                provenance,
                ResolutionSourceProvenance::LockedModelFile { relative_path, .. }
                    if relative_path == "config.json"
            )
        {
            return Err(resolution_test_error(
                "fixture.config_parser",
                "requires model metadata from locked config.json",
            ));
        }
        let typed: Value = serde_json::from_slice(source_bytes)
            .map_err(|error| resolution_test_error("fixture.config_parser", &error.to_string()))?;
        let typed_bytes = serde_json::to_vec(&typed).unwrap();
        Ok(json!({
            "chosen": {
                "source_file": "config.json",
                "sha256": bytes_sha256(source_bytes),
                "typed_config_sha256": bytes_sha256(&typed_bytes),
            }
        }))
    }
}

static LOCKED_CONFIG_RESOLUTION_PARSER: LockedConfigResolutionParser = LockedConfigResolutionParser;

struct WrongImplementationResolutionParser;

impl ResolutionSourceParser for WrongImplementationResolutionParser {
    fn descriptor(&self) -> Result<ResolutionParserDescriptor, VNextError> {
        ResolutionParserDescriptor::new(
            "resolution-parser.core-json",
            ContractVersion::new(1, 0),
            ResolutionFingerprint::new(sha('d')).unwrap(),
        )
    }

    fn parse(
        &self,
        _source: ResolutionDecisionSource,
        _provenance: &ResolutionSourceProvenance,
        source_bytes: &[u8],
    ) -> Result<Value, VNextError> {
        serde_json::from_slice(source_bytes).map_err(|error| {
            resolution_test_error("fixture.wrong_implementation_parser", &error.to_string())
        })
    }
}

static WRONG_IMPLEMENTATION_RESOLUTION_PARSER: WrongImplementationResolutionParser =
    WrongImplementationResolutionParser;

struct NondeterministicResolutionParser {
    calls: AtomicUsize,
}

impl ResolutionSourceParser for NondeterministicResolutionParser {
    fn descriptor(&self) -> Result<ResolutionParserDescriptor, VNextError> {
        ResolutionParserDescriptor::new(
            "resolution-parser.test-nondeterministic",
            ContractVersion::new(1, 0),
            ResolutionFingerprint::new(sha('7')).unwrap(),
        )
    }

    fn parse(
        &self,
        _source: ResolutionDecisionSource,
        _provenance: &ResolutionSourceProvenance,
        source_bytes: &[u8],
    ) -> Result<Value, VNextError> {
        let mut value: Value = serde_json::from_slice(source_bytes).unwrap();
        if self.calls.fetch_add(1, Ordering::SeqCst) % 2 == 1 {
            value["chosen"] = json!("nondeterministic");
        }
        Ok(value)
    }
}

static NONDETERMINISTIC_RESOLUTION_PARSER: NondeterministicResolutionParser =
    NondeterministicResolutionParser {
        calls: AtomicUsize::new(0),
    };

fn upstream_provenance(index: usize) -> ResolutionSourceProvenance {
    ResolutionSourceProvenance::Upstream {
        producer_id: "fixture.resolution-root".to_owned(),
        producer_version: ContractVersion::new(1, 0),
        producer_implementation_fingerprint: ResolutionFingerprint::new(sha('e')).unwrap(),
        revision: "fixture-v1".to_owned(),
        artifact_locator: format!("resolution/input/{index}"),
    }
}

fn resolved_inputs(fixture: &PlanFixture) -> ResolvedModelPlanInputs {
    let config_sha = bytes_sha256(RESOLUTION_CONFIG_BYTES);
    assert_eq!(config_sha, fixture.family.config_fingerprint());
    let original_sources = OriginalModelSources {
        semantic: OriginalModelSource {
            kind: ModelSourceKind::Repository,
            location: "repo/model-semantic".to_owned(),
            requested_revision: Some("semantic-main".to_owned()),
        },
        tokenizer: OriginalModelSource {
            kind: ModelSourceKind::Repository,
            location: "repo/model-tokenizer".to_owned(),
            requested_revision: Some("tokenizer-main".to_owned()),
        },
        weights: OriginalModelSource {
            kind: ModelSourceKind::Repository,
            location: "repo/model-weights".to_owned(),
            requested_revision: Some("weights-main".to_owned()),
        },
    };
    let resolved_sources = ResolvedModelSources {
        semantic: ResolvedModelSource {
            canonical_location: "repo/model-semantic".to_owned(),
            resolved_revision: "semantic-0123456789abcdef".to_owned(),
            files: vec![
                FileFingerprint {
                    relative_path: "config.json".to_owned(),
                    size_bytes: RESOLUTION_CONFIG_BYTES.len() as u64,
                    sha256: config_sha.clone(),
                },
                FileFingerprint {
                    relative_path: "manifest.json".to_owned(),
                    size_bytes: 10,
                    sha256: sha('1'),
                },
            ],
        },
        tokenizer: ResolvedModelSource {
            canonical_location: "repo/model-tokenizer".to_owned(),
            resolved_revision: "tokenizer-0123456789abcdef".to_owned(),
            files: vec![
                FileFingerprint {
                    relative_path: "manifest.json".to_owned(),
                    size_bytes: 11,
                    sha256: sha('2'),
                },
                FileFingerprint {
                    relative_path: "template.json".to_owned(),
                    size_bytes: 30,
                    sha256: sha('c'),
                },
                FileFingerprint {
                    relative_path: "tokenizer.json".to_owned(),
                    size_bytes: 20,
                    sha256: sha('b'),
                },
            ],
        },
        weights: ResolvedModelSource {
            canonical_location: "repo/model-weights".to_owned(),
            resolved_revision: "weights-0123456789abcdef".to_owned(),
            files: vec![
                FileFingerprint {
                    relative_path: "manifest.json".to_owned(),
                    size_bytes: 12,
                    sha256: sha('3'),
                },
                FileFingerprint {
                    relative_path: "model.safetensors".to_owned(),
                    size_bytes: 4096,
                    sha256: sha('d'),
                },
            ],
        },
    };
    ResolvedModelPlanInputs {
        original_sources,
        resolved_sources,
        config: ModelConfigFingerprint {
            source_file: "config.json".to_owned(),
            sha256: config_sha.clone(),
            typed_config_sha256: config_sha,
        },
        external_metadata_id: id("metadata.synthetic"),
        prepared_family: fixture.family.clone(),
        tokenizer: TokenizerDescriptor {
            tokenizer_id: id("tokenizer.synthetic"),
            source_file: "tokenizer.json".to_owned(),
            sha256: sha('b'),
            vocabulary_size: 1000,
        },
        device: fixture.catalog.device().clone(),
        capabilities: fixture.catalog.clone(),
        runtime: fixture.policy.clone(),
        engine: EngineSelection {
            provider_id: id("provider.engine.reference"),
            contract_version: ContractVersion::new(1, 0),
            implementation_fingerprint: sha('8'),
        },
        execution_plan: fixture.plan.clone(),
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
            maximum_output_tokens: 64,
            token_ids: BTreeSet::from([3]),
            strings: vec!["stop".to_owned()],
            collision_policy: StopTokenCollisionPolicy::require_distinct(),
        },
        structured_output: StructuredOutputPolicy::JsonObject,
    }
}

fn fixture_resolution_value(inputs: &ResolvedModelPlanInputs, field: ResolutionField) -> Value {
    match field {
        ResolutionField::OriginalSources => serde_json::to_value(&inputs.original_sources).unwrap(),
        ResolutionField::ResolvedSources => serde_json::to_value(&inputs.resolved_sources).unwrap(),
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

struct ResolvedFixtureEvidence {
    inputs: ResolvedModelPlanInputs,
    bindings: Vec<ResolutionDecisionBinding>,
    source_evidence: Vec<ResolutionSourceEvidence<'static>>,
}

struct AlternateResolvedRegistry {
    registration: TypedFamilyRegistration<OrderedSchemaFamily>,
}

struct DuplicateMetadataFamily;

impl ModelFamilyProvider for DuplicateMetadataFamily {
    type Config = TestConfig;

    fn family_id(&self) -> &ModelFamilyId {
        static FAMILY: std::sync::OnceLock<ModelFamilyId> = std::sync::OnceLock::new();
        FAMILY.get_or_init(|| id("family.duplicate-metadata"))
    }

    fn external_metadata_ids(&self) -> BTreeSet<ExternalModelMetadataId> {
        BTreeSet::from([id("metadata.synthetic")])
    }

    fn validate_config_identity(
        &self,
        _raw: &Value,
        _config: &Self::Config,
    ) -> Result<(), VNextError> {
        Ok(())
    }

    fn validated_external_metadata_id(
        &self,
        raw: &Value,
        config: &Self::Config,
    ) -> Result<ExternalModelMetadataId, VNextError> {
        self.validate_config_identity(raw, config)?;
        Ok(id("metadata.synthetic"))
    }

    fn parse_config(&self, _raw: &Value) -> Result<Self::Config, VNextError> {
        Err(VNextError::InvalidModelConfig {
            family_id: self.family_id().to_string(),
            field: "config".to_owned(),
            reason: "identity-only adversarial registration".to_owned(),
        })
    }

    fn weight_schema(&self, _config: &Self::Config) -> Result<WeightSchema, VNextError> {
        unreachable!("identity-only adversarial registration")
    }

    fn semantic_program(&self, _config: &Self::Config) -> Result<ModelProgram, VNextError> {
        unreachable!("identity-only adversarial registration")
    }

    fn semantic_metadata(
        &self,
        _config: &Self::Config,
    ) -> Result<ModelSemanticMetadata, VNextError> {
        unreachable!("identity-only adversarial registration")
    }
}

struct CrossFamilyResolvedRegistry {
    primary: TypedFamilyRegistration<TestFamily>,
    other: TypedFamilyRegistration<GraphFamily>,
}

impl CrossFamilyResolvedRegistry {
    fn new() -> Self {
        Self {
            primary: TypedFamilyRegistration::new(TestFamily),
            other: TypedFamilyRegistration::new(GraphFamily),
        }
    }
}

impl ModelFamilyRegistry for CrossFamilyResolvedRegistry {
    fn registrations(&self) -> Vec<&dyn ModelFamilyRegistration> {
        vec![&self.primary, &self.other]
    }
}

struct DuplicateMetadataResolvedRegistry {
    primary: TypedFamilyRegistration<TestFamily>,
    duplicate: TypedFamilyRegistration<DuplicateMetadataFamily>,
}

impl DuplicateMetadataResolvedRegistry {
    fn new() -> Self {
        Self {
            primary: TypedFamilyRegistration::new(TestFamily),
            duplicate: TypedFamilyRegistration::new(DuplicateMetadataFamily),
        }
    }
}

impl ModelFamilyRegistry for DuplicateMetadataResolvedRegistry {
    fn registrations(&self) -> Vec<&dyn ModelFamilyRegistration> {
        vec![&self.primary, &self.duplicate]
    }
}

impl AlternateResolvedRegistry {
    fn new() -> Self {
        Self {
            registration: TypedFamilyRegistration::new(OrderedSchemaFamily {
                reverse: false,
                reverse_sources: false,
            }),
        }
    }
}

impl ModelFamilyRegistry for AlternateResolvedRegistry {
    fn registrations(&self) -> Vec<&dyn ModelFamilyRegistration> {
        vec![&self.registration]
    }
}

fn resolved_evidence(fixture: &PlanFixture) -> ResolvedFixtureEvidence {
    let inputs = resolved_inputs(fixture);
    resolved_evidence_for_inputs(inputs)
}

fn resolved_evidence_for_inputs(inputs: ResolvedModelPlanInputs) -> ResolvedFixtureEvidence {
    let mut bindings = Vec::with_capacity(RESOLUTION_FIELDS.len());
    let mut source_evidence = Vec::with_capacity(RESOLUTION_FIELDS.len());
    for (index, field) in RESOLUTION_FIELDS.into_iter().enumerate() {
        let source = decision_source(field);
        let artifact_id: ResolutionArtifactId = id(format!("artifact.{index}"));
        let field_path = "/chosen".to_owned();
        let evidence = if field == ResolutionField::Config {
            ResolutionSourceEvidence::new(
                artifact_id.clone(),
                source,
                ResolutionSourceProvenance::LockedModelFile {
                    source_role: ModelArtifactSourceRole::Semantic,
                    relative_path: "config.json".to_owned(),
                },
                RESOLUTION_CONFIG_BYTES.to_vec(),
                BTreeSet::from([field_path.clone()]),
                &LOCKED_CONFIG_RESOLUTION_PARSER,
            )
            .unwrap()
        } else {
            let document = json!({"chosen": fixture_resolution_value(&inputs, field)});
            ResolutionSourceEvidence::new(
                artifact_id.clone(),
                source,
                upstream_provenance(index),
                serde_json::to_vec(&document).unwrap(),
                BTreeSet::from([field_path.clone()]),
                &JSON_RESOLUTION_SOURCE_PARSER,
            )
            .unwrap()
        };
        source_evidence.push(evidence);
        bindings.push(
            ResolutionDecisionBinding::new(
                field,
                source,
                id(format!("reason.{index}")),
                artifact_id,
                field_path,
            )
            .unwrap(),
        );
    }
    ResolvedFixtureEvidence {
        inputs,
        bindings,
        source_evidence,
    }
}

#[test]
fn resolved_model_plan_closes_all_contract_links() {
    let fixture = plan_fixture(0);
    let evidence = resolved_evidence(&fixture);
    let context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &evidence.source_evidence,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    let plan = ResolvedModelPlan::new(evidence.inputs, evidence.bindings, &context).unwrap();
    let restored =
        ResolvedModelPlan::from_json_validated(&plan.to_json().unwrap(), &context).unwrap();
    assert_eq!(plan, restored);
    assert_eq!(plan.execution_plan(), &fixture.plan);
    assert_eq!(plan.parts().engine.implementation_fingerprint, sha('8'));
    assert_eq!(plan.parts().decisions().len(), RESOLUTION_FIELDS.len());
    assert_ne!(
        plan.parts().resolved_sources.semantic.canonical_location,
        plan.parts().resolved_sources.tokenizer.canonical_location
    );
    for role in ModelArtifactSourceRole::ALL {
        assert!(plan
            .parts()
            .resolved_sources
            .for_role(role)
            .files
            .iter()
            .any(|file| file.relative_path == "manifest.json"));
    }
    let locked = plan
        .parts()
        .source_artifacts()
        .iter()
        .find(|artifact| artifact.source() == ResolutionDecisionSource::ModelMetadata)
        .unwrap();
    assert!(matches!(
        locked.provenance(),
        ResolutionSourceProvenance::LockedModelFile {
            source_role: ModelArtifactSourceRole::Semantic,
            relative_path,
        } if relative_path == "config.json"
    ));
    assert_eq!(
        locked.content_size_bytes(),
        RESOLUTION_CONFIG_BYTES.len() as u64
    );

    for mutate in [
        |value: &mut Value| {
            value["parts"]["prepared_family"]["canonical_config"]["width"] = json!(8)
        },
        |value: &mut Value| {
            value["parts"]["engine"]["provider_id"] = json!("provider.engine.unknown")
        },
        |value: &mut Value| {
            value["parts"]["engine"]["implementation_fingerprint"] = json!(sha('0'))
        },
        |value: &mut Value| value["parts"]["stop"]["token_ids"] = json!([1000]),
        |value: &mut Value| value["parts"]["device"]["unknown_nested_field"] = json!(true),
        |value: &mut Value| {
            value["parts"]["capabilities"]["operations"]["operation.main"]["unknown_nested_field"] =
                json!(true)
        },
        |value: &mut Value| {
            value["parts"]["execution_plan"]["payload"]["memory"]["unknown_nested_field"] =
                json!(true)
        },
    ] {
        let mut tampered = serde_json::to_value(&plan).unwrap();
        mutate(&mut tampered);
        assert!(ResolvedModelPlan::from_json_validated(
            &serde_json::to_vec(&tampered).unwrap(),
            &context,
        )
        .is_err());
    }
}

#[test]
fn resolved_model_plan_requires_trusted_completion_retention() {
    let mut fixture = plan_fixture(0);
    let retention = CompletionRetentionSpec::new(BTreeSet::from([id("value.output")]));
    fixture.plan = ExecutionPlan::build(
        PlanBuildRequest::new(
            &fixture.family,
            &fixture.catalog,
            &fixture.policy,
            fixture.node_resolutions.clone(),
        )
        .unwrap()
        .with_completion_retention(retention.clone())
        .unwrap(),
    )
    .unwrap();
    let evidence = resolved_evidence(&fixture);

    let missing_retention = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &evidence.source_evidence,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    assert!(ResolvedModelPlan::new(
        evidence.inputs.clone(),
        evidence.bindings.clone(),
        &missing_retention,
    )
    .unwrap_err()
    .to_string()
    .contains("not identical to its semantic rebuild"));

    let trusted_retention = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &evidence.source_evidence,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    )
    .with_completion_retention(retention);
    let plan =
        ResolvedModelPlan::new(evidence.inputs, evidence.bindings, &trusted_retention).unwrap();
    let restored =
        ResolvedModelPlan::from_json_validated(&plan.to_json().unwrap(), &trusted_retention)
            .unwrap();
    assert_eq!(restored, plan);
}

#[test]
fn locked_file_provenance_cannot_cross_source_roles() {
    let fixture = plan_fixture(0);
    let mut inputs = resolved_inputs(&fixture);
    inputs
        .resolved_sources
        .tokenizer
        .files
        .push(FileFingerprint {
            relative_path: "config.json".to_owned(),
            size_bytes: RESOLUTION_CONFIG_BYTES.len() as u64,
            sha256: bytes_sha256(RESOLUTION_CONFIG_BYTES),
        });
    inputs
        .resolved_sources
        .tokenizer
        .files
        .sort_by(|left, right| left.relative_path.cmp(&right.relative_path));
    let mut evidence = resolved_evidence_for_inputs(inputs);
    let config_artifact_id = evidence
        .bindings
        .iter()
        .find(|binding| binding.field() == ResolutionField::Config)
        .unwrap()
        .source_artifact_id()
        .clone();
    let config_index = evidence
        .source_evidence
        .iter()
        .position(|source| source.id() == &config_artifact_id)
        .unwrap();
    evidence.source_evidence[config_index] = ResolutionSourceEvidence::new(
        config_artifact_id,
        ResolutionDecisionSource::ModelMetadata,
        ResolutionSourceProvenance::LockedModelFile {
            source_role: ModelArtifactSourceRole::Tokenizer,
            relative_path: "config.json".to_owned(),
        },
        RESOLUTION_CONFIG_BYTES.to_vec(),
        BTreeSet::from(["/chosen".to_owned()]),
        &LOCKED_CONFIG_RESOLUTION_PARSER,
    )
    .unwrap();
    let context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &evidence.source_evidence,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    let error = ResolvedModelPlan::new(evidence.inputs, evidence.bindings, &context)
        .err()
        .expect("semantic config evidence cannot claim the tokenizer source role");
    assert!(matches!(
        error,
        VNextError::InvalidResolvedModelPlan { field, .. }
            if field == "decisions.source_role"
    ));
}

#[test]
fn resolved_stop_alias_requires_exact_product_owned_policy() {
    let fixture = plan_fixture(0);

    let mut missing_inputs = resolved_inputs(&fixture);
    missing_inputs.stop.token_ids = BTreeSet::from([2]);
    let missing = resolved_evidence_for_inputs(missing_inputs);
    let missing_context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &missing.source_evidence,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    let error =
        ResolvedModelPlan::new(missing.inputs, missing.bindings, &missing_context).unwrap_err();
    assert!(matches!(
        error,
        VNextError::InvalidResolvedModelPlan { ref field, .. }
            if field == "stop.collision_policy"
    ));

    let mut exact_inputs = resolved_inputs(&fixture);
    exact_inputs.stop.token_ids = BTreeSet::from([2]);
    exact_inputs.stop.collision_policy =
        StopTokenCollisionPolicy::new(BTreeSet::from([SpecialTokenRole::Eos])).unwrap();
    let exact = resolved_evidence_for_inputs(exact_inputs);
    let exact_context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &exact.source_evidence,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    let plan = ResolvedModelPlan::new(exact.inputs, exact.bindings, &exact_context).unwrap();
    assert_eq!(
        serde_json::to_value(&plan).unwrap()["parts"]["stop"]["collision_policy"]
            ["allowed_model_roles"],
        json!(["eos"])
    );
    assert_eq!(
        ResolvedModelPlan::from_json_validated(&plan.to_json().unwrap(), &exact_context).unwrap(),
        plan
    );

    let mut broad_inputs = resolved_inputs(&fixture);
    broad_inputs.stop.token_ids = BTreeSet::from([2]);
    broad_inputs.stop.collision_policy = StopTokenCollisionPolicy::new(BTreeSet::from([
        SpecialTokenRole::Eos,
        SpecialTokenRole::Pad,
    ]))
    .unwrap();
    let broad = resolved_evidence_for_inputs(broad_inputs);
    let broad_context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &broad.source_evidence,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    let error = ResolvedModelPlan::new(broad.inputs, broad.bindings, &broad_context).unwrap_err();
    assert!(matches!(
        error,
        VNextError::InvalidResolvedModelPlan { ref field, ref reason }
            if field == "stop.collision_policy" && reason.contains("exactly match")
    ));
}

#[test]
fn resolved_model_plan_initial_construction_requires_verified_evidence_context() {
    let fixture = plan_fixture(0);
    let evidence = resolved_evidence(&fixture);

    let missing_sources = &evidence.source_evidence[1..];
    let missing_context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        missing_sources,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    assert!(ResolvedModelPlan::new(
        evidence.inputs.clone(),
        evidence.bindings.clone(),
        &missing_context,
    )
    .is_err());

    let mut extra_sources = evidence.source_evidence.clone();
    extra_sources.push(
        ResolutionSourceEvidence::new(
            id("artifact.unused"),
            ResolutionDecisionSource::ProductDefault,
            upstream_provenance(999),
            serde_json::to_vec(&json!({"chosen": "unused"})).unwrap(),
            BTreeSet::from(["/chosen".to_owned()]),
            &JSON_RESOLUTION_SOURCE_PARSER,
        )
        .unwrap(),
    );
    let extra_context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &extra_sources,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    assert!(ResolvedModelPlan::new(evidence.inputs, evidence.bindings, &extra_context).is_err());
}

#[test]
fn resolved_source_evidence_rejects_raw_bytes_and_provenance_tampering() {
    let fixture = plan_fixture(0);
    let evidence = resolved_evidence(&fixture);
    let context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &evidence.source_evidence,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    let plan = ResolvedModelPlan::new(evidence.inputs.clone(), evidence.bindings.clone(), &context)
        .unwrap();

    let first = &evidence.source_evidence[0];
    let first_document: Value = serde_json::from_slice(first.source_bytes()).unwrap();
    let mut wrong_bytes = evidence.source_evidence.clone();
    wrong_bytes[0] = ResolutionSourceEvidence::new(
        first.id().clone(),
        first.source(),
        first.provenance().clone(),
        serde_json::to_vec_pretty(&first_document).unwrap(),
        first.field_paths().clone(),
        &JSON_RESOLUTION_SOURCE_PARSER,
    )
    .unwrap();
    let wrong_bytes_context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &wrong_bytes,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    assert!(
        ResolvedModelPlan::from_json_validated(&plan.to_json().unwrap(), &wrong_bytes_context,)
            .is_err()
    );

    let mut wrong_provenance = evidence.source_evidence.clone();
    let mut changed_provenance = upstream_provenance(0);
    let ResolutionSourceProvenance::Upstream { revision, .. } = &mut changed_provenance else {
        unreachable!();
    };
    *revision = "fixture-v2".to_owned();
    wrong_provenance[0] = ResolutionSourceEvidence::new(
        first.id().clone(),
        first.source(),
        changed_provenance,
        first.source_bytes().to_vec(),
        first.field_paths().clone(),
        &JSON_RESOLUTION_SOURCE_PARSER,
    )
    .unwrap();
    let wrong_provenance_context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &wrong_provenance,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    assert!(ResolvedModelPlan::from_json_validated(
        &plan.to_json().unwrap(),
        &wrong_provenance_context,
    )
    .is_err());
    let mut invalid_upstream = upstream_provenance(0);
    let ResolutionSourceProvenance::Upstream { revision, .. } = &mut invalid_upstream else {
        unreachable!();
    };
    *revision = "not canonical".to_owned();
    assert!(ResolutionSourceEvidence::new(
        id("artifact.invalid-upstream"),
        first.source(),
        invalid_upstream,
        first.source_bytes().to_vec(),
        first.field_paths().clone(),
        &JSON_RESOLUTION_SOURCE_PARSER,
    )
    .is_err());

    let config_index = RESOLUTION_FIELDS
        .iter()
        .position(|field| *field == ResolutionField::Config)
        .unwrap();
    let config_document = json!({
        "chosen": fixture_resolution_value(&evidence.inputs, ResolutionField::Config)
    });
    let mut unlocked_model_metadata = evidence.source_evidence.clone();
    unlocked_model_metadata[config_index] = ResolutionSourceEvidence::new(
        unlocked_model_metadata[config_index].id().clone(),
        ResolutionDecisionSource::ModelMetadata,
        upstream_provenance(config_index),
        serde_json::to_vec(&config_document).unwrap(),
        BTreeSet::from(["/chosen".to_owned()]),
        &JSON_RESOLUTION_SOURCE_PARSER,
    )
    .unwrap();
    let unlocked_context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &unlocked_model_metadata,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    assert!(ResolvedModelPlan::new(
        evidence.inputs.clone(),
        evidence.bindings.clone(),
        &unlocked_context,
    )
    .is_err());

    let mut missing_locked_file = evidence.source_evidence.clone();
    missing_locked_file[config_index] = ResolutionSourceEvidence::new(
        missing_locked_file[config_index].id().clone(),
        ResolutionDecisionSource::ModelMetadata,
        ResolutionSourceProvenance::LockedModelFile {
            source_role: ModelArtifactSourceRole::Semantic,
            relative_path: "missing.json".to_owned(),
        },
        serde_json::to_vec(&config_document).unwrap(),
        BTreeSet::from(["/chosen".to_owned()]),
        &JSON_RESOLUTION_SOURCE_PARSER,
    )
    .unwrap();
    let missing_locked_context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &missing_locked_file,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    assert!(
        ResolvedModelPlan::new(evidence.inputs, evidence.bindings, &missing_locked_context,)
            .is_err()
    );

    let mut tampered_size = serde_json::to_value(&plan).unwrap();
    let locked_artifact = tampered_size["parts"]["source_artifacts"]
        .as_array_mut()
        .unwrap()
        .iter_mut()
        .find(|artifact| artifact["provenance"]["kind"] == json!("locked_model_file"))
        .unwrap();
    locked_artifact["content_size_bytes"] = json!(RESOLUTION_CONFIG_BYTES.len() as u64 + 1);
    assert!(ResolvedModelPlan::from_json_validated(
        &serde_json::to_vec(&tampered_size).unwrap(),
        &context,
    )
    .is_err());
}

#[test]
fn resolved_source_parser_identity_and_determinism_are_enforced() {
    let fixture = plan_fixture(0);
    let evidence = resolved_evidence(&fixture);
    let first = &evidence.source_evidence[0];
    assert!(ResolutionSourceEvidence::new(
        id("artifact.nondeterministic"),
        first.source(),
        first.provenance().clone(),
        first.source_bytes().to_vec(),
        first.field_paths().clone(),
        &NONDETERMINISTIC_RESOLUTION_PARSER,
    )
    .unwrap()
    .validate()
    .is_err());

    let context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &evidence.source_evidence,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    let plan = ResolvedModelPlan::new(evidence.inputs.clone(), evidence.bindings.clone(), &context)
        .unwrap();
    let mut wrong_implementation = evidence.source_evidence.clone();
    wrong_implementation[0] = ResolutionSourceEvidence::new(
        first.id().clone(),
        first.source(),
        first.provenance().clone(),
        first.source_bytes().to_vec(),
        first.field_paths().clone(),
        &WRONG_IMPLEMENTATION_RESOLUTION_PARSER,
    )
    .unwrap();
    let wrong_implementation_context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &wrong_implementation,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    assert!(ResolvedModelPlan::from_json_validated(
        &plan.to_json().unwrap(),
        &wrong_implementation_context,
    )
    .is_err());

    let mut wrong_parser_id = serde_json::to_value(&plan).unwrap();
    wrong_parser_id["parts"]["source_artifacts"][0]["parser"]["id"] =
        json!("resolution-parser.other");
    assert!(ResolvedModelPlan::from_json_validated(
        &serde_json::to_vec(&wrong_parser_id).unwrap(),
        &context,
    )
    .is_err());
    let mut wrong_parser_version = serde_json::to_value(&plan).unwrap();
    wrong_parser_version["parts"]["source_artifacts"][0]["parser"]["version"]["minor"] = json!(1);
    assert!(ResolvedModelPlan::from_json_validated(
        &serde_json::to_vec(&wrong_parser_version).unwrap(),
        &context,
    )
    .is_err());
    let mut wrong_parser_implementation = serde_json::to_value(&plan).unwrap();
    wrong_parser_implementation["parts"]["source_artifacts"][0]["parser"]
        ["implementation_fingerprint"] = json!(sha('0'));
    assert!(ResolvedModelPlan::from_json_validated(
        &serde_json::to_vec(&wrong_parser_implementation).unwrap(),
        &context,
    )
    .is_err());
}

#[test]
fn resolved_external_device_catalog_runtime_and_node_resolution_are_exact() {
    let fixture = plan_fixture(0);
    let evidence = resolved_evidence(&fixture);
    let context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &evidence.source_evidence,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    let plan = ResolvedModelPlan::new(evidence.inputs.clone(), evidence.bindings.clone(), &context)
        .unwrap();

    let wrong_node_resolutions = vec![node_resolution(
        &fixture.family,
        &fixture.catalog,
        &fixture.policy,
        99,
        &fixture.planning,
    )];
    let wrong_node_context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &evidence.source_evidence,
        &wrong_node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    assert!(ResolvedModelPlan::new(
        evidence.inputs.clone(),
        evidence.bindings.clone(),
        &wrong_node_context,
    )
    .is_err());
    assert!(
        ResolvedModelPlan::from_json_validated(&plan.to_json().unwrap(), &wrong_node_context,)
            .is_err()
    );

    let alternate_registry = AlternateResolvedRegistry::new();
    let wrong_registry_context = ResolvedPlanValidationContext::new(
        &alternate_registry,
        &evidence.source_evidence,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    assert!(ResolvedModelPlan::new(
        evidence.inputs.clone(),
        evidence.bindings.clone(),
        &wrong_registry_context,
    )
    .is_err());

    let mut wrong_device = fixture.catalog.device().clone();
    wrong_device.total_memory_bytes += 1;
    let wrong_device_context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &evidence.source_evidence,
        &fixture.node_resolutions,
        &wrong_device,
        &fixture.catalog,
        &fixture.policy,
    );
    assert!(ResolvedModelPlan::new(
        evidence.inputs.clone(),
        evidence.bindings.clone(),
        &wrong_device_context,
    )
    .is_err());

    let wrong_catalog = catalog_with_secondary_provider();
    let wrong_catalog_context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &evidence.source_evidence,
        &fixture.node_resolutions,
        wrong_catalog.device(),
        &wrong_catalog,
        &fixture.policy,
    );
    assert!(ResolvedModelPlan::new(
        evidence.inputs.clone(),
        evidence.bindings.clone(),
        &wrong_catalog_context,
    )
    .is_err());

    let wrong_runtime = policy(4095);
    let wrong_runtime_context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &evidence.source_evidence,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &wrong_runtime,
    );
    assert!(
        ResolvedModelPlan::new(evidence.inputs, evidence.bindings, &wrong_runtime_context,)
            .is_err()
    );
}

#[test]
fn resolved_model_family_identity_is_unique_and_fail_closed() {
    const EXPECTED: usize = 5;
    let fixture = plan_fixture(0);
    let mut passed = 0usize;

    let mut unknown_inputs = resolved_inputs(&fixture);
    unknown_inputs.external_metadata_id = id("metadata.unknown");
    let unknown = resolved_evidence_for_inputs(unknown_inputs);
    let unknown_context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &unknown.source_evidence,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    assert!(matches!(
        ResolvedModelPlan::new(unknown.inputs, unknown.bindings, &unknown_context),
        Err(VNextError::UnknownExternalModelMetadata { .. })
    ));
    passed += 1;

    let cross_registry = CrossFamilyResolvedRegistry::new();
    let mut wrong_family_inputs = resolved_inputs(&fixture);
    wrong_family_inputs.external_metadata_id = id("metadata.execution-graph");
    let wrong_family = resolved_evidence_for_inputs(wrong_family_inputs);
    let wrong_family_context = ResolvedPlanValidationContext::new(
        &cross_registry,
        &wrong_family.source_evidence,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    assert!(matches!(
        ResolvedModelPlan::new(
            wrong_family.inputs,
            wrong_family.bindings,
            &wrong_family_context,
        ),
        Err(VNextError::InvalidResolvedModelPlan {
            field,
            ..
        }) if field == "external_metadata_id"
    ));
    passed += 1;

    let mut alias_inputs = resolved_inputs(&fixture);
    alias_inputs.external_metadata_id = id("metadata.synthetic.alias");
    let alias = resolved_evidence_for_inputs(alias_inputs);
    let alias_context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &alias.source_evidence,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    assert!(matches!(
        ResolvedModelPlan::new(alias.inputs, alias.bindings, &alias_context),
        Err(VNextError::InvalidResolvedModelPlan {
            field,
            ..
        }) if field == "external_metadata_id"
    ));
    passed += 1;

    let duplicate_registry = DuplicateMetadataResolvedRegistry::new();
    let duplicate = resolved_evidence(&fixture);
    let duplicate_context = ResolvedPlanValidationContext::new(
        &duplicate_registry,
        &duplicate.source_evidence,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    assert!(matches!(
        ResolvedModelPlan::new(
            duplicate.inputs.clone(),
            duplicate.bindings.clone(),
            &duplicate_context,
        ),
        Err(VNextError::AmbiguousModelFamilyRegistration {
            identity_kind: "external metadata",
            matches: 2,
            ..
        })
    ));
    passed += 1;

    let trusted_context = ResolvedPlanValidationContext::new(
        &fixture.registry,
        &duplicate.source_evidence,
        &fixture.node_resolutions,
        fixture.catalog.device(),
        &fixture.catalog,
        &fixture.policy,
    );
    let trusted =
        ResolvedModelPlan::new(duplicate.inputs, duplicate.bindings, &trusted_context).unwrap();
    assert!(ResolvedModelPlan::from_json_validated(
        &trusted.to_json().unwrap(),
        &duplicate_context,
    )
    .is_err());
    passed += 1;

    assert_eq!(passed, EXPECTED);
    println!("\nVNEXT MODEL IDENTITY PASS: {passed}/{EXPECTED}");
}

#[test]
fn resolution_source_matrix_rejects_forbidden_binding_before_plan() {
    assert!(ResolutionDecisionBinding::new(
        ResolutionField::Family,
        ResolutionDecisionSource::ProductDefault,
        id("reason.forbidden"),
        id("artifact.forbidden"),
        "/chosen",
    )
    .is_err());
}

fn rejected_plan_mutation(fixture: &PlanFixture, mutate: impl FnOnce(&mut Value)) {
    let mut value = serde_json::to_value(&fixture.plan).unwrap();
    mutate(&mut value);
    assert!(ExecutionPlan::from_json_validated(
        &serde_json::to_vec(&value).unwrap(),
        &fixture.family,
        &fixture.catalog,
        &fixture.policy,
        fixture.node_resolutions.clone(),
    )
    .is_err());
}

#[test]
fn unknown_inputs_fail_closed() {
    const EXPECTED: usize = 63;
    let fixture = plan_fixture(0);
    let mut passed = 0;
    macro_rules! reject {
        ($body:expr) => {{
            rejected_plan_mutation(&fixture, $body);
            passed += 1;
        }};
    }

    reject!(|value| value["payload"]["schema"]["major"] = json!(9));
    reject!(|value| value["payload"]["family_id"] = json!("family.other"));
    reject!(|value| value["payload"]["device_id"] = json!("device.other"));
    reject!(|value| value["payload"]["prepared_family_fingerprint"] = json!(sha('0')));
    reject!(|value| value["payload"]["program_fingerprint"] = json!(sha('0')));
    reject!(|value| value["payload"]["capability_catalog_fingerprint"] = json!(sha('0')));
    reject!(|value| value["payload"]["policy_version"]["major"] = json!(0));
    reject!(|value| value["payload"]["policy_fingerprint"] = json!(sha('0')));
    reject!(|value| value["payload"]["weight_format"] = json!("weight-format.other"));
    reject!(|value| value["payload"]["quantization_formats"] = json!(["quantization.other"]));
    reject!(|value| value["payload"]["nodes"] = json!([]));
    reject!(|value| value["payload"]["nodes"][0]["operation_id"] = json!("operation.unknown"));
    reject!(|value| value["payload"]["nodes"][0]["operation_fingerprint"] = json!(sha('0')));
    reject!(
        |value| value["payload"]["nodes"][0]["required_capabilities"] =
            json!(["capability.missing"])
    );
    reject!(
        |value| value["payload"]["nodes"][0]["selection"]["selected_provider"] =
            json!("provider.unknown")
    );
    reject!(
        |value| value["payload"]["nodes"][0]["selection"]["selection_reason"] =
            json!("fallback_from_preferred")
    );
    reject!(|value| {
        value["payload"]["nodes"][0]["provider_resources"]["provider_id"] =
            json!("provider.operation.unknown")
    });
    reject!(|value| {
        value["payload"]["nodes"][0]["provider_resources"]["estimator_id"] =
            json!("resource-estimator.other")
    });
    reject!(|value| {
        value["payload"]["nodes"][0]["provider_resources"]["estimator_version"]["major"] = json!(2)
    });
    reject!(|value| {
        value["payload"]["nodes"][0]["provider_resources"]["estimator_implementation_fingerprint"] =
            json!(sha('0'))
    });
    reject!(|value| {
        value["payload"]["nodes"][0]["provider_resources"]["estimator_input_fingerprint"] =
            json!(sha('0'))
    });
    reject!(|value| {
        value["payload"]["nodes"][0]["provider_resources"]["estimate_fingerprint"] = json!(sha('0'))
    });
    reject!(|value| {
        value["payload"]["nodes"][0]["provider_resources"]["value_alignment_bytes"] = json!(8)
    });
    reject!(|value| {
        value["payload"]["nodes"][0]["provider_resources"]["scratch"]["size_formula"]["fixed"]
            ["bytes"] = json!(1)
    });
    reject!(|value| {
        value["payload"]["nodes"][0]["provider_resources"]["scratch"]["alignment_bytes"] = json!(8)
    });
    reject!(|value| {
        value["payload"]["nodes"][0]["provider_resources"]["scratch"]["scope"] = json!("plan")
    });
    reject!(|value| {
        value["payload"]["nodes"][0]["provider_resources"]["persistent"]["size_formula"]["fixed"]
            ["bytes"] = json!(1)
    });
    reject!(|value| {
        value["payload"]["nodes"][0]["provider_resources"]["persistent"]["scope"] =
            json!("sequence")
    });
    reject!(|value| {
        value["payload"]["nodes"][0]["values"]
            .as_array_mut()
            .unwrap()
            .pop();
    });
    reject!(|value| value["payload"]["nodes"][0]["values"][0]["tensor"]["dimensions"] = json!([1]));
    reject!(
        |value| value["payload"]["nodes"][0]["values"][0]["storage"]["components"][0]
            ["length_bytes"] = json!(1)
    );
    reject!(|value| value["payload"]["nodes"][0]["scratch_resource"] = Value::Null);
    reject!(|value| value["payload"]["nodes"][0]["persistent_resource"] = Value::Null);
    reject!(|value| value["payload"]["nodes"][0]["resources"] = json!([]));
    reject!(|value| value["payload"]["memory"]["device_capacity_bytes"] = json!(1));
    reject!(|value| value["payload"]["memory"]["policy_capacity_bytes"] = json!(1));
    reject!(|value| value["payload"]["memory"]["reserve_bytes"] = json!(129));
    reject!(|value| value["payload"]["memory"]["usable_capacity_bytes"] = json!(3967));
    reject!(|value| value["payload"]["memory"]["maximum_active_sequences"] = json!(2));
    reject!(|value| value["payload"]["memory"]["theoretical_ceiling_bytes"] = json!("1"));
    reject!(|value| value["payload"]["memory"]["static_allocations"] = json!([]));
    reject!(|value| {
        value["payload"]["memory"]["static_allocations"][0]["per_instance_bytes"] = json!(1)
    });
    reject!(|value| {
        value["payload"]["memory"]["static_allocations"][0]["instance_stride_bytes"] = json!(1)
    });
    reject!(|value| {
        value["payload"]["memory"]["static_allocations"][0]["instance_count"] = json!(2)
    });
    reject!(|value| value["payload"]["memory"]["static_allocations"][0]["size_bytes"] = json!(1));
    reject!(|value| {
        value["payload"]["memory"]["static_allocations"][0]["alignment_bytes"] = json!(8)
    });
    reject!(
        |value| value["payload"]["memory"]["static_allocations"][0]["usage"] = json!("transfer")
    );
    reject!(|value| {
        value["payload"]["memory"]["static_allocations"][0]["element_type"] = json!("u32")
    });
    reject!(|value| value["payload"]["plan_id"] = json!("plan/forged"));
    reject!(|value| value["plan_hash"] = json!(sha('0')));
    reject!(|value| value["unknown_field"] = json!(true));
    reject!(|value| value["payload"]["memory"]["unknown_nested_field"] = json!(true));
    reject!(|value| {
        value["payload"]["memory"]["static_allocations"][0]["unknown_nested_field"] = json!(true)
    });

    assert!(ExecutionPlan::from_json_validated(
        &fixture.plan.to_json().unwrap(),
        &fixture.family,
        &fixture.catalog,
        &policy(4095),
        fixture.node_resolutions.clone(),
    )
    .is_err());
    passed += 1;
    assert!(ExecutionPlan::from_json_validated(
        &fixture.plan.to_json().unwrap(),
        &fixture.family,
        &fixture.catalog,
        &fixture.policy,
        vec![node_resolution(
            &fixture.family,
            &fixture.catalog,
            &fixture.policy,
            7,
            &fixture.planning,
        )],
    )
    .is_err());
    passed += 1;
    assert!((&fixture.registry as &dyn ModelFamilyRegistry)
        .resolve(&id("family.unknown"))
        .is_err());
    passed += 1;

    assert!(policy_with(1 << 20, 128, 0).is_err());
    passed += 1;

    let mut zero_policy_wire = serde_json::to_value(&fixture.policy).unwrap();
    zero_policy_wire["memory"]["maximum_active_sequences"] = json!(0);
    assert!(serde_json::from_value::<ResolvedRuntimePolicy>(zero_policy_wire).is_err());
    passed += 1;

    let mut zero_token_policy_wire = serde_json::to_value(&fixture.policy).unwrap();
    zero_token_policy_wire["admission"]["maximum_scheduled_tokens"] = json!(0);
    assert!(serde_json::from_value::<ResolvedRuntimePolicy>(zero_token_policy_wire).is_err());
    passed += 1;

    let rejected_planning =
        TestPlanningRegistry::new(&fixture.catalog, 64, 32, EstimateBehavior::Correct);
    let malicious = AdversarialRuntimePolicy {
        maximum_active_sequences: 0,
        maximum_scheduled_tokens: 4096,
        dynamic_storage_profile_order: vec![contiguous_storage_profile()],
    };
    let planning = rejected_planning.planning();
    assert!(PlanNodeResolution::resolve(
        &fixture.family,
        &fixture.catalog,
        &malicious,
        &planning,
        id("node.main"),
        resolved_values(0),
        BTreeSet::new(),
        None,
    )
    .is_err());
    assert_eq!(rejected_planning.estimator_calls.load(Ordering::SeqCst), 0);
    passed += 1;

    let malicious = AdversarialRuntimePolicy {
        maximum_active_sequences: 0,
        maximum_scheduled_tokens: 4096,
        dynamic_storage_profile_order: vec![contiguous_storage_profile()],
    };
    assert!(ExecutionPlan::build(
        PlanBuildRequest::new(
            &fixture.family,
            &fixture.catalog,
            &malicious,
            fixture.node_resolutions.clone(),
        )
        .unwrap(),
    )
    .is_err());
    passed += 1;

    assert!(serde_json::from_value::<StateCapacityDemand>(json!({
        "token_scaled": {"bytes_per_token": 0, "maximum_tokens": 1}
    }))
    .is_err());
    passed += 1;

    let invalid_state = json!({
        "id": "state.invalid-demand",
        "value_id": "value.invalid-demand",
        "tensor": {
            "dimensions": [4],
            "element_type": "u8",
            "layout": "contiguous"
        },
        "lifetime": "sequence",
        "capacity_demand": {
            "page_scaled": {"bytes_per_page": 1, "maximum_pages": 1}
        }
    });
    assert!(serde_json::from_value::<StateSpec>(invalid_state).is_err());
    passed += 1;

    assert_eq!(passed, EXPECTED);
    println!("\nVNEXT FAIL CLOSED PASS: {passed}/{EXPECTED}");
}

#[test]
fn provider_catalog_and_reference_oracle_fail_closed() {
    let base = operation();
    let mut missing = base.clone();
    missing.oracle = OracleSpec::ReferenceOperation {
        operation_id: id("operation.missing"),
        version: ContractVersion::new(1, 0),
    };
    let device = catalog().device().clone();
    assert!(CapabilityCatalog::new(
        device,
        vec![missing.clone()],
        BTreeMap::from([(missing.id.clone(), Vec::new())]),
        vec![EngineProviderDescriptor::new(
            id("provider.engine.reference"),
            ContractVersion::new(1, 0),
            sha('8'),
            id("device.reference.0"),
            BTreeSet::from([id("capability.compute")]),
        )
        .unwrap()],
    )
    .is_err());

    let report = catalog()
        .provider_compatibility(
            ProviderCompatibilityRequest::new(
                base.id.clone(),
                ContractVersion::new(1, 0),
                BTreeSet::new(),
                BTreeSet::from([id("weight-format.unsupported")]),
                BTreeSet::new(),
            )
            .unwrap(),
        )
        .unwrap();
    assert!(report.compatible_provider_ids().is_empty());
    assert_eq!(report.rejected().len(), 1);
    assert!(matches!(
        report.rejected()[0].reasons.as_slice(),
        [ProviderCompatibilityRejectReason::UnsupportedWeightFormats { .. }]
    ));

    let mut reference = base.clone();
    reference.id = id("operation.oracle.reference");
    reference.oracle = OracleSpec::Exact;
    let mut incompatible = base.clone();
    incompatible.id = id("operation.oracle.incompatible");
    incompatible.outputs = vec![tensor_contract(
        ElementType::U8,
        TensorAccess::Write,
        AliasPolicy::NoAlias,
    )];
    incompatible.oracle = OracleSpec::ReferenceOperation {
        operation_id: reference.id.clone(),
        version: ContractVersion::new(1, 0),
    };
    assert!(catalog_from_operations(vec![incompatible, reference]).is_err());

    let mut cycle_a = base.clone();
    cycle_a.id = id("operation.oracle.cycle-a");
    cycle_a.oracle = OracleSpec::ReferenceOperation {
        operation_id: id("operation.oracle.cycle-b"),
        version: ContractVersion::new(1, 0),
    };
    let mut cycle_b = base.clone();
    cycle_b.id = id("operation.oracle.cycle-b");
    cycle_b.oracle = OracleSpec::ReferenceOperation {
        operation_id: cycle_a.id.clone(),
        version: ContractVersion::new(1, 0),
    };
    assert!(catalog_from_operations(vec![cycle_a, cycle_b]).is_err());

    let oracle_chain = |length: usize| {
        (0..length)
            .map(|index| {
                let mut descriptor = base.clone();
                descriptor.id = id(format!("operation.oracle.chain.{index:04}"));
                descriptor.oracle = if index + 1 == length {
                    OracleSpec::Exact
                } else {
                    OracleSpec::ReferenceOperation {
                        operation_id: id(format!("operation.oracle.chain.{:04}", index + 1)),
                        version: ContractVersion::new(1, 0),
                    }
                };
                descriptor
            })
            .collect::<Vec<_>>()
    };
    catalog_from_operations(oracle_chain(MAX_REFERENCE_ORACLE_DEPTH)).unwrap();
    assert!(catalog_from_operations(oracle_chain(MAX_REFERENCE_ORACLE_DEPTH + 1)).is_err());

    let too_many_operations = (0..=MAX_OPERATION_CATALOG_ROWS)
        .map(|index| {
            let mut descriptor = base.clone();
            descriptor.id = id(format!("operation.oracle.row-limit.{index:05}"));
            descriptor
        })
        .collect::<Vec<_>>();
    assert!(CapabilityCatalog::new(
        catalog().device().clone(),
        too_many_operations,
        BTreeMap::from([(id("operation.placeholder"), Vec::new())]),
        vec![EngineProviderDescriptor::new(
            id("provider.engine.row-limit"),
            ContractVersion::new(1, 0),
            sha('8'),
            id("device.reference.0"),
            BTreeSet::from([id("capability.compute")]),
        )
        .unwrap()],
    )
    .is_err());
}

#[test]
fn prepared_family_wire_requires_typed_registry_reconstruction() {
    let registry = TestRegistry::new();
    let family = registry.prepare();
    let bytes = serde_json::to_vec(&family).unwrap();
    let unvalidated = PreparedModelFamily::decode_untrusted(&bytes).unwrap();
    assert_eq!(unvalidated.revalidate(&registry).unwrap(), family);

    let mut value = serde_json::to_value(&family).unwrap();
    value["program"]["outputs"] = json!(["value.input"]);
    let unvalidated =
        PreparedModelFamily::decode_untrusted(&serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(unvalidated.revalidate(&registry).is_err());
}

#[test]
fn mandatory_object_safe_contracts_accept_trait_objects() {
    fn boundaries(
        _runtime: &dyn DeviceRuntime<
            Buffer = Vec<u8>,
            Stream = (),
            Command = (),
            Fence = (),
            Error = std::io::Error,
        >,
        _operation: &dyn OperationContract,
        _estimator: &dyn OperationResourceEstimator,
        _planning: &dyn OperationPlanningRegistry,
        _registration: &dyn ModelFamilyRegistration,
        _registry: &dyn ModelFamilyRegistry,
        _events: &dyn ExecutionEventSink,
    ) {
    }
    let _ = boundaries;
}
