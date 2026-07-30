use super::*;

pub(crate) struct TestModelRegistry {
    pub(crate) registration: TypedFamilyRegistration<TestFamily>,
}

impl TestModelRegistry {
    pub(crate) fn new() -> Self {
        Self {
            registration: TypedFamilyRegistration::new(TestFamily),
        }
    }
}

impl ModelFamilyRegistry for TestModelRegistry {
    fn registrations(&self) -> Vec<&dyn ModelFamilyRegistration> {
        vec![&self.registration]
    }
}

pub(crate) const RESOLUTION_FIELDS: [ResolutionField; 20] = [
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

pub(crate) fn resolution_source(field: ResolutionField) -> ResolutionDecisionSource {
    match field {
        ResolutionField::OriginalSources => ResolutionDecisionSource::UserInput,
        ResolutionField::ResolvedSources
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

pub(crate) fn operation_registry(
    catalog: &CapabilityCatalog,
    behavior: Arc<Mutex<ProviderBehavior>>,
    trace: Arc<Mutex<ProviderTrace>>,
) -> OperationRuntimeRegistry<TestRuntime> {
    OperationRuntimeRegistry::new(
        vec![Box::new(TestOperationContract {
            descriptor: catalog.operation(&id("operation.main")).unwrap().clone(),
        })],
        vec![Box::new(TestProvider {
            descriptor: catalog.providers_for(&id("operation.main")).unwrap()[0].clone(),
            behavior,
            trace,
        })],
    )
    .unwrap()
}

pub(crate) fn node_resolution_for(
    family: &PreparedModelFamily,
    catalog: &CapabilityCatalog,
    runtime_policy: &ResolvedRuntimePolicy,
    registry: &OperationRuntimeRegistry<TestRuntime>,
    node_id: &str,
    values: Vec<ResolvedValueBinding>,
) -> PlanNodeResolution {
    PlanNodeResolution::resolve(
        family,
        catalog,
        runtime_policy,
        &registry.planning(),
        id(node_id),
        values,
        BTreeSet::new(),
        None,
    )
    .unwrap()
}

pub(crate) fn node_resolution(
    family: &PreparedModelFamily,
    catalog: &CapabilityCatalog,
    runtime_policy: &ResolvedRuntimePolicy,
    registry: &OperationRuntimeRegistry<TestRuntime>,
) -> PlanNodeResolution {
    node_resolution_with_zero_state(family, catalog, runtime_policy, registry, false)
}

pub(crate) fn node_resolution_with_zero_state(
    family: &PreparedModelFamily,
    catalog: &CapabilityCatalog,
    runtime_policy: &ResolvedRuntimePolicy,
    registry: &OperationRuntimeRegistry<TestRuntime>,
    zero_state: bool,
) -> PlanNodeResolution {
    node_resolution_for(
        family,
        catalog,
        runtime_policy,
        registry,
        "node.main",
        if zero_state {
            node_values_with_zero_state_for(
                "value.input",
                "resource.input",
                "value.intermediate",
                "resource.intermediate",
            )
        } else {
            node_values()
        },
    )
}

pub(crate) fn tail_node_resolution(
    family: &PreparedModelFamily,
    catalog: &CapabilityCatalog,
    runtime_policy: &ResolvedRuntimePolicy,
    registry: &OperationRuntimeRegistry<TestRuntime>,
) -> PlanNodeResolution {
    tail_node_resolution_with_zero_state(family, catalog, runtime_policy, registry, false)
}

pub(crate) fn tail_node_resolution_with_zero_state(
    family: &PreparedModelFamily,
    catalog: &CapabilityCatalog,
    runtime_policy: &ResolvedRuntimePolicy,
    registry: &OperationRuntimeRegistry<TestRuntime>,
    zero_state: bool,
) -> PlanNodeResolution {
    node_resolution_for(
        family,
        catalog,
        runtime_policy,
        registry,
        "node.tail",
        if zero_state {
            node_values_with_zero_state_for(
                "value.intermediate",
                "resource.intermediate",
                "value.output",
                "resource.output",
            )
        } else {
            tail_node_values()
        },
    )
}

pub(crate) fn resolved_model_plan(
    registry: &OperationRuntimeRegistry<TestRuntime>,
) -> (ResolvedModelPlan, ExecutionPlan) {
    resolved_model_plan_with_zero_state(registry, false)
}

pub(crate) fn resolved_model_plan_with_zero_state(
    registry: &OperationRuntimeRegistry<TestRuntime>,
    zero_state: bool,
) -> (ResolvedModelPlan, ExecutionPlan) {
    let runtime_policy = policy();
    let catalog = catalog_with_zero_state(zero_state);
    resolved_model_plan_with_zero_state_and_policy(
        registry,
        zero_state,
        false,
        &runtime_policy,
        &catalog,
        false,
    )
}

fn test_raw_config(zero_state: bool, token_scaled_state: bool) -> Value {
    match (zero_state, token_scaled_state) {
        (false, false) => json!({"width": 4}),
        (true, false) => json!({"width": 4, "zero_state": true}),
        (true, true) => {
            json!({"width": 4, "zero_state": true, "token_scaled_state": true})
        }
        (false, true) => unreachable!("token-scaled state requires the state binding"),
    }
}

fn resolved_model_plan_with_zero_state_and_policy(
    registry: &OperationRuntimeRegistry<TestRuntime>,
    zero_state: bool,
    token_scaled_state: bool,
    runtime_policy: &ResolvedRuntimePolicy,
    catalog: &CapabilityCatalog,
    retain_determinism_outputs: bool,
) -> (ResolvedModelPlan, ExecutionPlan) {
    let model_registry = TestModelRegistry::new();
    let raw_config = test_raw_config(zero_state, token_scaled_state);
    let family = model_registry.registration.prepare(&raw_config).unwrap();
    let resolutions = vec![
        node_resolution_with_zero_state(&family, catalog, runtime_policy, registry, zero_state),
        tail_node_resolution_with_zero_state(
            &family,
            catalog,
            runtime_policy,
            registry,
            zero_state,
        ),
    ];
    let completion_retention = if retain_determinism_outputs {
        CompletionRetentionSpec::for_determinism_outputs(&family).unwrap()
    } else {
        CompletionRetentionSpec::default()
    };
    let plan = ExecutionPlan::build(
        PlanBuildRequest::new(&family, catalog, runtime_policy, resolutions.clone())
            .unwrap()
            .with_completion_retention(completion_retention.clone())
            .unwrap(),
    )
    .unwrap();
    let config_fingerprint = family.config_fingerprint().to_owned();
    let original_source = OriginalModelSource {
        kind: ModelSourceKind::Repository,
        location: "repo/device-operation-model".to_owned(),
        requested_revision: Some("main".to_owned()),
    };
    let resolved_source = ResolvedModelSource {
        canonical_location: "repo/device-operation-model".to_owned(),
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
    };
    let inputs = ResolvedModelPlanInputs {
        original_sources: OriginalModelSources {
            semantic: original_source.clone(),
            tokenizer: original_source.clone(),
            weights: original_source,
        },
        resolved_sources: ResolvedModelSources {
            semantic: resolved_source.clone(),
            tokenizer: resolved_source.clone(),
            weights: resolved_source,
        },
        config: ModelConfigFingerprint {
            source_file: "config.json".to_owned(),
            sha256: config_fingerprint.clone(),
            typed_config_sha256: config_fingerprint,
        },
        external_metadata_id: id("metadata.device-operation"),
        prepared_family: family,
        tokenizer: TokenizerDescriptor {
            tokenizer_id: id("tokenizer.device-operation"),
            source_file: "tokenizer.json".to_owned(),
            sha256: sha('b'),
            vocabulary_size: 1024,
        },
        device: catalog.device().clone(),
        capabilities: catalog.clone(),
        runtime: runtime_policy.clone(),
        engine: EngineSelection {
            provider_id: id("provider.engine.device-operation"),
            contract_version: ContractVersion::new(1, 0),
            implementation_fingerprint: sha('e'),
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
            collision_policy: StopTokenCollisionPolicy::require_distinct(),
        },
        structured_output: StructuredOutputPolicy::JsonObject,
    };
    let mut bindings = Vec::new();
    let mut evidence = Vec::new();
    for (index, field) in RESOLUTION_FIELDS.into_iter().enumerate() {
        let source = resolution_source(field);
        let artifact_id: ResolutionArtifactId = id(format!("artifact.device-operation.{index}"));
        let path = "/chosen".to_owned();
        evidence.push(
            ResolutionSourceEvidence::new(
                artifact_id.clone(),
                source,
                ResolutionSourceProvenance::Upstream {
                    producer_id: "fixture.device-operation".to_owned(),
                    producer_version: ContractVersion::new(1, 0),
                    producer_implementation_fingerprint: ResolutionFingerprint::new(sha('e'))
                        .unwrap(),
                    revision: "fixture-v1".to_owned(),
                    artifact_locator: format!("device-operation/{index}"),
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
                id(format!("reason.device-operation.{index}")),
                artifact_id,
                path,
            )
            .unwrap(),
        );
    }
    let context = ResolvedPlanValidationContext::new(
        &model_registry,
        &evidence,
        &resolutions,
        catalog.device(),
        &catalog,
        runtime_policy,
    )
    .with_completion_retention(completion_retention);
    (
        ResolvedModelPlan::new(inputs, bindings, &context).unwrap(),
        plan,
    )
}

pub(crate) fn plan_for_registry(registry: &OperationRuntimeRegistry<TestRuntime>) -> ExecutionPlan {
    plan_for_registry_with_zero_state(registry, false)
}

pub(crate) fn plan_for_registry_with_zero_state(
    registry: &OperationRuntimeRegistry<TestRuntime>,
    zero_state: bool,
) -> ExecutionPlan {
    let runtime_policy = policy();
    let catalog = catalog_with_zero_state(zero_state);
    plan_for_registry_with_zero_state_and_policy(
        registry,
        zero_state,
        false,
        &runtime_policy,
        &catalog,
        false,
    )
}

fn plan_for_registry_with_zero_state_and_policy(
    registry: &OperationRuntimeRegistry<TestRuntime>,
    zero_state: bool,
    token_scaled_state: bool,
    runtime_policy: &ResolvedRuntimePolicy,
    catalog: &CapabilityCatalog,
    retain_determinism_outputs: bool,
) -> ExecutionPlan {
    let raw_config = test_raw_config(zero_state, token_scaled_state);
    let family = TypedFamilyRegistration::new(TestFamily)
        .prepare(&raw_config)
        .unwrap();
    let completion_retention = if retain_determinism_outputs {
        CompletionRetentionSpec::for_determinism_outputs(&family).unwrap()
    } else {
        CompletionRetentionSpec::default()
    };
    ExecutionPlan::build(
        PlanBuildRequest::new(
            &family,
            catalog,
            runtime_policy,
            vec![
                node_resolution_with_zero_state(
                    &family,
                    catalog,
                    runtime_policy,
                    registry,
                    zero_state,
                ),
                tail_node_resolution_with_zero_state(
                    &family,
                    catalog,
                    runtime_policy,
                    registry,
                    zero_state,
                ),
            ],
        )
        .unwrap()
        .with_completion_retention(completion_retention)
        .unwrap(),
    )
    .unwrap()
}

pub(crate) fn plan_runtime_resources(
    plan: &ExecutionPlan,
    runtime: Arc<TestRuntime>,
) -> Arc<PlanRuntimeResources<TestRuntime>> {
    let ProvisionedPlanParts { provisioning } = plan
        .provision_static(
            Arc::clone(&runtime),
            id("request.device-operation.provision"),
        )
        .unwrap()
        .into_parts();
    let admission = match provisioning {
        StaticProvisioning::Required(admission) => admission,
        StaticProvisioning::NoStatic(_) => {
            panic!("device operation fixture requires static/backing provisioning")
        }
    };
    let identity = ResourceTransactionIdentity::for_admission(
        admission.binding(),
        id("run.device-operation.provision"),
        id("transaction.device-operation.provision"),
    );
    let driver = TestDriver {
        runtime,
        trace: Arc::new(Mutex::new(DriverTrace::default())),
    };
    let committed = ResourceTransaction::begin(driver, identity, admission)
        .unwrap()
        .reserve()
        .unwrap()
        .commit()
        .unwrap();
    let pool_ids = committed
        .maintenance_controller()
        .pool_ids()
        .cloned()
        .collect::<Vec<_>>();
    for pool_id in pool_ids {
        committed
            .maintenance_controller()
            .initialize_pool(&pool_id)
            .unwrap();
    }
    match committed.into_plan_runtime() {
        Ok(resources) => resources,
        Err(failure) => panic!("plan runtime handoff failed: {}", failure.error()),
    }
}

pub(crate) fn logical_resources(
    plan_resources: &Arc<PlanRuntimeResources<TestRuntime>>,
    run_id: &str,
    request_id: &str,
) -> Arc<AdmittedSequenceResources<TestRuntime>> {
    logical_resources_with_work(plan_resources, run_id, request_id, one_token_span())
}

pub(crate) fn logical_resources_with_work(
    plan_resources: &Arc<PlanRuntimeResources<TestRuntime>>,
    run_id: &str,
    request_id: &str,
    token_span: TokenSpanWork,
) -> Arc<AdmittedSequenceResources<TestRuntime>> {
    let work = ResourceWorkShape::single(token_span).unwrap();
    let request = RequestResourceAdmissionRequest::new(
        work.clone(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let binding = plan_resources.trusted_runtime_binding().unwrap();
    let mut request_maintenance_attempts = 0;
    let request_resources = loop {
        match binding
            .try_admit_request(request.clone(), id(run_id), id(request_id))
            .unwrap()
        {
            RequestResourceAdmissionDecision::Admitted(resources) => break resources,
            RequestResourceAdmissionDecision::BackingDeferred(deferred) => {
                assert!(
                    request_maintenance_attempts < 3,
                    "request admission did not converge after bounded maintenance"
                );
                request_maintenance_attempts += 1;
                deferred.maintain().unwrap();
            }
            RequestResourceAdmissionDecision::Deferred(_) => {
                panic!("device operation fixture request logical admission deferred")
            }
            RequestResourceAdmissionDecision::PermanentRejected(_) => {
                panic!("device operation fixture request admission rejected")
            }
        }
    };
    let sequence = SequenceResourceAdmissionRequest::new(
        work,
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let mut sequence_maintenance_attempts = 0;
    loop {
        match request_resources
            .try_admit_sequence(sequence.clone())
            .unwrap()
        {
            SequenceResourceAdmissionDecision::Admitted(resources) => break resources,
            SequenceResourceAdmissionDecision::BackingDeferred(deferred) => {
                assert!(
                    sequence_maintenance_attempts < 3,
                    "sequence admission did not converge after bounded maintenance"
                );
                sequence_maintenance_attempts += 1;
                deferred.maintain().unwrap();
            }
            SequenceResourceAdmissionDecision::Deferred(_) => {
                panic!("device operation fixture sequence logical admission deferred")
            }
            SequenceResourceAdmissionDecision::PermanentRejected(_) => {
                panic!("device operation fixture sequence admission rejected")
            }
        }
    }
}

pub(crate) fn begin_single_participant_step(
    plan_resources: &Arc<PlanRuntimeResources<TestRuntime>>,
    batch: &ExecutionBatchParticipants<TestRuntime>,
) -> Arc<StepResourceLease<TestRuntime>> {
    let lane = plan_resources.create_execution_lane().unwrap();
    begin_single_participant_step_on_lane_with_bucket(batch, &lane, None)
}

pub(crate) fn begin_single_participant_step_on_lane(
    batch: &ExecutionBatchParticipants<TestRuntime>,
    lane: &Arc<ExecutionLane<TestRuntime>>,
) -> Arc<StepResourceLease<TestRuntime>> {
    begin_single_participant_step_on_lane_with_bucket(batch, lane, None)
}

pub(crate) fn begin_single_participant_step_on_lane_with_bucket(
    batch: &ExecutionBatchParticipants<TestRuntime>,
    lane: &Arc<ExecutionLane<TestRuntime>>,
    bucket: Option<&ReusableExecutionBucketSpec>,
) -> Arc<StepResourceLease<TestRuntime>> {
    begin_single_participant_step_on_lane_with_bucket_and_work(
        batch,
        lane,
        bucket,
        one_token_span(),
    )
}

pub(crate) fn begin_single_participant_step_on_lane_with_bucket_and_work(
    batch: &ExecutionBatchParticipants<TestRuntime>,
    lane: &Arc<ExecutionLane<TestRuntime>>,
    bucket: Option<&ReusableExecutionBucketSpec>,
    token_span: TokenSpanWork,
) -> Arc<StepResourceLease<TestRuntime>> {
    let expected_immediate_tokens = token_span.immediate_tokens();
    let expected_fit_tokens = token_span.fit_input_tokens();
    let request = StepResourceAdmissionRequest::new(
        batch.bind_work_shape(vec![token_span]).unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let request = match bucket {
        Some(bucket) => request.with_reusable_execution_bucket(bucket.bucket_id().clone()),
        None => request,
    };
    for attempt in 0..=3 {
        match batch.try_begin_step(request.clone(), lane).unwrap() {
            StepResourceAdmissionDecision::Admitted(step) => {
                assert_eq!(step.work_shape().participants().len(), 1);
                assert_eq!(step.work_shape().immediate_sequences(), 1);
                assert_eq!(
                    step.work_shape().immediate_tokens(),
                    expected_immediate_tokens
                );
                assert_eq!(step.work_shape().immediate_pages(), 0);
                assert_eq!(step.work_shape().fit_sequences(), 1);
                assert_eq!(step.work_shape().fit_tokens(), expected_fit_tokens);
                assert_eq!(step.work_shape().fit_pages(), 0);
                assert_eq!(step.work_shape().fingerprint().len(), 64);
                assert_eq!(step.claimed_backing().fingerprint().len(), 64);
                match step.claimed_backing().logical_capacity() {
                    Some(capacity) => assert_eq!(
                        step.claimed_backing().demand().immediate_claim(),
                        capacity.claims()
                    ),
                    None => assert!(step.claimed_backing().demand().immediate_claim().is_empty()),
                }
                return step;
            }
            StepResourceAdmissionDecision::BackingDeferred(deferred) if attempt < 3 => {
                deferred.maintain().unwrap();
            }
            StepResourceAdmissionDecision::BackingDeferred(_) => {
                panic!("step backing did not converge after bounded maintenance")
            }
            StepResourceAdmissionDecision::Deferred(_) => {
                panic!("single-participant step unexpectedly deferred")
            }
            StepResourceAdmissionDecision::PermanentRejected(_) => {
                panic!("single-participant step unexpectedly rejected")
            }
        }
    }
    unreachable!("bounded step admission loop always returns or panics")
}

pub(crate) fn admit_single_participant_invocation(
    _plan_resources: &Arc<PlanRuntimeResources<TestRuntime>>,
    step: &Arc<StepResourceLease<TestRuntime>>,
    node_id: &NodeId,
) -> InvocationResourceLease<TestRuntime> {
    let request = InvocationResourceAdmissionRequest::for_all_step_participants(
        node_id.clone(),
        step.bind_all_invocation_work_shape(vec![one_token_span()])
            .unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    for attempt in 0..=3 {
        match step.try_admit_invocation(request.clone()).unwrap() {
            InvocationResourceAdmissionDecision::Admitted(invocation) => {
                assert_eq!(invocation.work_shape(), step.work_shape());
                assert_eq!(invocation.claimed_backing().fingerprint().len(), 64);
                assert_eq!(invocation.work_shape().fingerprint().len(), 64);
                return invocation;
            }
            InvocationResourceAdmissionDecision::BackingDeferred(deferred) if attempt < 3 => {
                deferred.maintain().unwrap();
            }
            InvocationResourceAdmissionDecision::BackingDeferred(_) => {
                panic!("invocation backing did not converge after bounded maintenance")
            }
            InvocationResourceAdmissionDecision::Deferred(_) => {
                panic!("single-participant invocation unexpectedly deferred")
            }
            InvocationResourceAdmissionDecision::PermanentRejected(_) => {
                panic!("single-participant invocation unexpectedly rejected")
            }
        }
    }
    unreachable!("bounded invocation admission loop always returns or panics")
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn encode_and_submit_single(
    provider: &BoundOperationProvider<'_, TestRuntime>,
    resolved: &ResolvedModelPlan,
    identity: &ExecutionIdentityEnvelope,
    frame_id: &ExecutionFrameId,
    node_invocation_id: &NodeInvocationId,
    node_id: &NodeId,
    active: &TrustedActiveSequenceBinding,
    invocation: InvocationResourceLease<TestRuntime>,
    lane: &Arc<ExecutionLane<TestRuntime>>,
    reaper: &Arc<CompletionReaper<TestRuntime>>,
) -> Result<CompletionHandle<TestRuntime>, OperationDispatchError<TestRuntime>> {
    let parts = identity.parts();
    if parts.frame_id != Some(*frame_id)
        || parts.node_invocation_id != Some(*node_invocation_id)
        || parts.node_id.as_ref() != Some(node_id)
    {
        return Err(OperationDispatchError::Contract(
            VNextError::InvalidExecutionPlan {
                reason: "single-participant dispatch arguments disagree".to_owned(),
            },
        ));
    }
    let active_bindings = std::slice::from_ref(active);
    let batch_identity = OperationDispatch::bind_batch_identity(
        resolved,
        vec![identity.clone()],
        active_bindings,
        &invocation,
        lane,
    )
    .map_err(OperationDispatchError::Contract)?;
    OperationDispatch::encode_and_submit(
        provider,
        resolved,
        &batch_identity,
        active_bindings,
        invocation,
        lane,
        reaper,
    )
}

pub(crate) struct Fixture {
    pub(crate) registry: OperationRuntimeRegistry<TestRuntime>,
    pub(crate) impostor_registry: OperationRuntimeRegistry<TestRuntime>,
    pub(crate) resolved: ResolvedModelPlan,
    pub(crate) plan: ExecutionPlan,
    pub(crate) impostor_plan_hash: PlanHash,
    pub(crate) runtime: Arc<TestRuntime>,
    pub(crate) runtime_trace: Arc<Mutex<RuntimeTrace>>,
    pub(crate) provider_behavior: Arc<Mutex<ProviderBehavior>>,
    pub(crate) provider_trace: Arc<Mutex<ProviderTrace>>,
    pub(crate) plan_resources: Arc<PlanRuntimeResources<TestRuntime>>,
    pub(crate) reusable_execution_bucket: Option<ReusableExecutionBucketSpec>,
}

pub(crate) fn fixture() -> Fixture {
    fixture_with_zero_state(false)
}

pub(crate) fn fixture_with_zero_state(zero_state: bool) -> Fixture {
    fixture_with_provider_behavior(zero_state, ProviderBehavior::Success)
}

pub(crate) fn fixture_with_provider_behavior(
    zero_state: bool,
    behavior: ProviderBehavior,
) -> Fixture {
    fixture_with_provider_behavior_and_execution_semantics_and_retention(
        zero_state,
        behavior,
        ProviderExecutionSemantics::bitwise_eager_and_replay(),
        ExecutionDeterminismRequirement::BitwiseSameRuntimeWithReplay,
        false,
    )
}

pub(crate) fn fixture_with_determinism_provider_behavior(
    zero_state: bool,
    behavior: ProviderBehavior,
) -> Fixture {
    fixture_with_provider_behavior_and_execution_semantics_and_retention(
        zero_state,
        behavior,
        ProviderExecutionSemantics::bitwise_eager_and_replay(),
        ExecutionDeterminismRequirement::BitwiseSameRuntimeWithReplay,
        true,
    )
}

pub(crate) fn fixture_with_token_scaled_paged_state() -> Fixture {
    fixture_with_provider_behavior_execution_semantics_retention_and_storage(
        true,
        true,
        ProviderBehavior::Success,
        ProviderExecutionSemantics::bitwise_eager_and_replay(),
        ExecutionDeterminismRequirement::BitwiseSameRuntimeWithReplay,
        true,
    )
}

pub(crate) fn fixture_with_provider_behavior_and_execution_semantics(
    zero_state: bool,
    behavior: ProviderBehavior,
    execution_semantics: ProviderExecutionSemantics,
    execution_determinism: ExecutionDeterminismRequirement,
) -> Fixture {
    fixture_with_provider_behavior_and_execution_semantics_and_retention(
        zero_state,
        behavior,
        execution_semantics,
        execution_determinism,
        false,
    )
}

fn fixture_with_provider_behavior_and_execution_semantics_and_retention(
    zero_state: bool,
    behavior: ProviderBehavior,
    execution_semantics: ProviderExecutionSemantics,
    execution_determinism: ExecutionDeterminismRequirement,
    retain_determinism_outputs: bool,
) -> Fixture {
    fixture_with_provider_behavior_execution_semantics_retention_and_storage(
        zero_state,
        false,
        behavior,
        execution_semantics,
        execution_determinism,
        retain_determinism_outputs,
    )
}

fn fixture_with_provider_behavior_execution_semantics_retention_and_storage(
    zero_state: bool,
    token_scaled_state: bool,
    behavior: ProviderBehavior,
    execution_semantics: ProviderExecutionSemantics,
    execution_determinism: ExecutionDeterminismRequirement,
    retain_determinism_outputs: bool,
) -> Fixture {
    let scratch = if matches!(
        behavior,
        ProviderBehavior::ScratchOverwrite | ProviderBehavior::ScratchZeroed
    ) {
        ResourcePresenceRequirement::Optional
    } else {
        ResourcePresenceRequirement::Forbidden
    };
    let catalog = catalog_with_resource_options_execution_semantics_and_storage(
        zero_state,
        scratch,
        execution_semantics,
        token_scaled_state,
    );
    let (runtime_policy, reusable_execution_bucket) = if behavior.uses_program_binding() {
        let (_, bucket) = reusable_policy();
        (
            policy_with_reusable_execution_determinism_and_storage(
                Some(
                    ReusableExecutionPolicy::new(1, vec![bucket.clone()])
                        .expect("valid reusable execution policy"),
                ),
                execution_determinism,
                token_scaled_state,
            ),
            Some(bucket),
        )
    } else {
        (
            policy_with_reusable_execution_determinism_and_storage(
                None,
                execution_determinism,
                token_scaled_state,
            ),
            None,
        )
    };
    let provider_behavior = Arc::new(Mutex::new(behavior));
    let provider_trace = Arc::new(Mutex::new(ProviderTrace::default()));
    let registry = operation_registry(
        &catalog,
        Arc::clone(&provider_behavior),
        Arc::clone(&provider_trace),
    );
    let derived_catalog = registry
        .capability_catalog(
            catalog.device().clone(),
            catalog.engine_providers().values().cloned().collect(),
        )
        .unwrap();
    assert_eq!(derived_catalog, catalog);
    let planning = registry.planning();
    let registered_contracts = planning.contracts_for(&id("operation.main"));
    assert_eq!(registered_contracts.len(), 1);
    assert_eq!(
        registered_contracts[0].descriptor(),
        catalog.operation(&id("operation.main")).unwrap()
    );
    let (resolved, plan) = resolved_model_plan_with_zero_state_and_policy(
        &registry,
        zero_state,
        token_scaled_state,
        &runtime_policy,
        &catalog,
        retain_determinism_outputs,
    );
    let impostor_registry = operation_registry(
        &catalog,
        Arc::new(Mutex::new(ProviderBehavior::WrongPhase)),
        Arc::new(Mutex::new(ProviderTrace::default())),
    );
    let impostor_plan_hash = plan_for_registry_with_zero_state_and_policy(
        &impostor_registry,
        zero_state,
        token_scaled_state,
        &runtime_policy,
        &catalog,
        retain_determinism_outputs,
    )
    .plan_hash()
    .clone();
    let (runtime, runtime_trace) = runtime(&catalog);
    let plan_resources = plan_runtime_resources(&plan, Arc::clone(&runtime));
    Fixture {
        registry,
        impostor_registry,
        resolved,
        plan,
        impostor_plan_hash,
        runtime,
        runtime_trace,
        provider_behavior,
        provider_trace,
        plan_resources,
        reusable_execution_bucket,
    }
}

pub(crate) fn close_plan_runtime(
    plan_resources: Arc<PlanRuntimeResources<TestRuntime>>,
    passed: &mut usize,
) {
    match PlanRuntimeResources::close(plan_resources) {
        Ok(PlanRuntimeCloseOutcome::Closed(receipt)) => {
            check(passed, receipt.released_static_resources() == 2)
        }
        Ok(PlanRuntimeCloseOutcome::Referenced { strong_count, .. }) => {
            panic!("plan runtime close retained {strong_count} references")
        }
        Err(failure) => panic!("plan runtime close failed: {:?}", failure.failure()),
    }
}

pub(crate) fn device_identity_parts(
    run_id: &str,
    request_id: &str,
    device_id: DeviceId,
    runtime_fingerprint: String,
) -> ExecutionIdentityParts {
    ExecutionIdentityParts {
        version: EXECUTION_IDENTITY_VERSION,
        run_id: id(run_id),
        request_id: id(request_id),
        sequence: 1,
        plan_id: None,
        plan_hash: None,
        frame_id: None,
        node_invocation_id: None,
        node_id: None,
        operation_id: None,
        provider_id: None,
        device_id: Some(device_id),
        resource_pool_id: None,
        resource_pool_identity_fingerprint: None,
        provisioning_run_id: None,
        provisioning_request_id: None,
        transaction_id: None,
        active_sequence_slot: None,
        admission_generation: None,
        activation_epoch: None,
        runtime_implementation_fingerprint: Some(runtime_fingerprint),
        active_sequence_fingerprint: None,
        completed_sequence_fingerprint: None,
        aborted_sequence_fingerprint: None,
        resource_id: None,
        resource_generation: None,
        resource_batch_fingerprint: None,
        span_id: id("span.device-operation"),
        parent_span_id: None,
        async_links: Vec::new(),
    }
}

pub(crate) fn operation_identity(
    plan: &ExecutionPlan,
    active: &TrustedActiveSequenceBinding,
    frame_id: ExecutionFrameId,
    invocation_id: NodeInvocationId,
) -> ExecutionIdentityEnvelope {
    operation_identity_for_node(plan, 0, active, frame_id, invocation_id)
}

pub(crate) fn operation_identity_for_node(
    plan: &ExecutionPlan,
    node_index: usize,
    active: &TrustedActiveSequenceBinding,
    frame_id: ExecutionFrameId,
    invocation_id: NodeInvocationId,
) -> ExecutionIdentityEnvelope {
    let node = &plan.payload().nodes()[node_index];
    let provisioning = active.static_provisioning_identity();
    ExecutionIdentityEnvelope::new(ExecutionIdentityParts {
        version: EXECUTION_IDENTITY_VERSION,
        run_id: active.run_id().clone(),
        request_id: active.request_id().clone(),
        sequence: 1,
        plan_id: Some(plan.payload().plan_id().clone()),
        plan_hash: Some(plan.plan_hash().clone()),
        frame_id: Some(frame_id),
        node_invocation_id: Some(invocation_id),
        node_id: Some(node.id().clone()),
        operation_id: Some(node.operation_id().clone()),
        provider_id: Some(node.selection().selected_provider().clone()),
        device_id: Some(plan.payload().device_id().clone()),
        resource_pool_id: active.static_pool_id(),
        resource_pool_identity_fingerprint: active.static_pool_identity_fingerprint(),
        provisioning_run_id: provisioning.map(|identity| identity.run_id().clone()),
        provisioning_request_id: provisioning.map(|identity| identity.request_id().clone()),
        transaction_id: provisioning.map(|identity| identity.transaction_id().clone()),
        active_sequence_slot: Some(active.sequence_authority().sparse_id()),
        admission_generation: Some(active.sequence_authority().generation()),
        activation_epoch: Some(active.activation_epoch()),
        runtime_implementation_fingerprint: Some(
            active.runtime_implementation_fingerprint().to_owned(),
        ),
        active_sequence_fingerprint: Some(active.fingerprint().to_owned()),
        completed_sequence_fingerprint: None,
        aborted_sequence_fingerprint: None,
        resource_id: None,
        resource_generation: None,
        resource_batch_fingerprint: None,
        span_id: id(format!("span.device-operation.node.{node_index}")),
        parent_span_id: None,
        async_links: Vec::new(),
    })
    .unwrap()
}

pub(crate) fn revalidate_plan_for_registry(
    bytes: &[u8],
    registry: &OperationRuntimeRegistry<TestRuntime>,
) -> ExecutionPlan {
    let family = TypedFamilyRegistration::new(TestFamily)
        .prepare(&json!({"width": 4}))
        .unwrap();
    let catalog = catalog();
    let runtime_policy = policy();
    let resolutions = vec![
        node_resolution(&family, &catalog, &runtime_policy, registry),
        tail_node_resolution(&family, &catalog, &runtime_policy, registry),
    ];
    ExecutionPlan::from_json_validated(bytes, &family, &catalog, &runtime_policy, resolutions)
        .unwrap()
}

pub(crate) fn serialization_message(error: VNextError) -> String {
    match error {
        VNextError::Serialization { message, .. } => message,
        other => panic!("expected serialization error, got {other}"),
    }
}
