//! Real admitted resources exercise sealing and fresh physical preparation.
use super::*;
use crate::vnext::operation::invocation::steady_recipe::SealedNodeRecipe;
use ferrum_types::InvocationPreparationStrategy;
use std::any::Any;

#[path = "../../completion/steady_recipe/cache_tests.rs"]
mod cache_tests;

fn with_recipe_wave(
    run: impl FnOnce(
        &Fixture,
        &PreparedStepSubmissionWave<TestRuntime>,
        &BatchOperationIdentity,
        &[TrustedActiveSequenceBinding],
    ),
) {
    let bucket = ReusableExecutionBucketSpec::new(
        ReusableExecutionClassId::new("execution.device-operation").unwrap(),
        ReusableExecutionCapacity::new(2, 2, 1).unwrap(),
    )
    .unwrap();
    let fixture = fixture_with_retained_dependencies_and_bucket(
        16,
        DependencyMode::Valid,
        Some(bucket.clone()),
    );
    fixture
        .runtime_trace
        .lock()
        .unwrap()
        .immutable_metadata_enabled = true;
    let resources = (0..2)
        .map(|i| {
            logical_resources(
                &fixture.plan_resources,
                &format!("run.steady-admission.{i}"),
                &format!("request.steady-admission.{i}"),
            )
        })
        .collect::<Vec<_>>();
    let sessions = resources
        .iter()
        .map(|r| r.open_session().unwrap())
        .collect::<Vec<_>>();
    let batch = ExecutionBatchParticipants::new(sessions).unwrap();
    let lane = fixture.plan_resources.create_execution_lane().unwrap();
    let request = StepResourceAdmissionRequest::new(
        batch.bind_work_shape(vec![one_token_span(); 2]).unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap()
    .with_reusable_execution_bucket(bucket.bucket_id().clone());
    let step = (0..=3)
        .find_map(
            |attempt| match batch.try_begin_step(request.clone(), &lane).unwrap() {
                StepResourceAdmissionDecision::Admitted(step) => Some(step),
                StepResourceAdmissionDecision::BackingDeferred(deferred) if attempt < 3 => {
                    deferred.maintain().unwrap();
                    None
                }
                _ => panic!("real reusable Step admission did not converge"),
            },
        )
        .unwrap();
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active = batch
        .sessions()
        .iter()
        .map(|s| TrustedActiveSequenceBinding::from_session(s).unwrap())
        .collect::<Vec<_>>();
    let identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    run(&fixture, &wave, &identity, &active);
}

fn dependency() -> SteadyRecipeDependency {
    SteadyRecipeDependency {
        input_ordinal: 1,
        component_id: id("weight.component.left"),
        source_offset_bytes: 0,
        source_length_bytes: 8,
        persistent_offset_bytes: 0,
        persistent_length_bytes: 4,
        alignment_bytes: 4,
        validation_identity: "fixture.exact-weight-validation.v1".into(),
    }
}

fn declaration(state: Arc<dyn Any + Send + Sync>) -> SteadyRecipeDeclaration {
    SteadyRecipeDeclaration::new(
        vec![SteadyRecipeRegionRequest {
            selector: SteadyRecipeRegionSelector::Persistent,
            offset_bytes: 0,
            length_bytes: 4,
            element_type: ElementType::U8,
            alignment_bytes: 4,
        }],
        vec![dependency()],
        state,
    )
    .unwrap()
}

#[allow(clippy::too_many_arguments)]
fn seal_recipe(
    fixture: &Fixture,
    wave: &PreparedStepSubmissionWave<TestRuntime>,
    identity: &BatchOperationIdentity,
    active: &[TrustedActiveSequenceBinding],
    node_index: usize,
    state: Arc<dyn Any + Send + Sync>,
    provider_owner: &Arc<()>,
    entry: DeviceReusableExecutionEntryIdentity,
) -> Result<Option<SealedNodeRecipe<TestRuntime>>, VNextError> {
    let providers = fixture.registry.bind_plan(&fixture.resolved)?;
    let provider = &providers.providers()[node_index];
    let node = identity.materialize_node(node_index)?;
    let invocation = BatchedOperationInvocation::from_wave_node(
        fixture.runtime.as_ref(),
        &fixture.resolved,
        provider.dispatch(),
        identity,
        node,
        wave,
        node_index,
        active.iter(),
    )?;
    let lane = wave.step_resources().execution_lane();
    let program = OperationDispatch::reusable_execution_program_id_for_wave(
        providers.providers(),
        &fixture.resolved,
        wave,
        lane,
    )?
    .expect("real reusable admission");
    SealedNodeRecipe::seal(
        &fixture.runtime,
        &fixture.resolved,
        provider.dispatch(),
        &invocation,
        declaration(state),
        provider_owner,
        entry,
        program,
        lane.id(),
        lane.reusable_execution_epoch(),
        node_index,
    )
}

#[test]
fn steady_recipe_real_two_participant_table_issues_fresh_dependency_scopes() {
    with_recipe_wave(|fixture, wave, identity, active| {
        let owner = Arc::new(());
        let recipe = seal_recipe(
            fixture,
            wave,
            identity,
            active,
            0,
            Arc::new(7u64),
            &owner,
            DeviceReusableExecutionEntryIdentity::new(),
        )
        .unwrap()
        .unwrap();
        let provider = fixture
            .registry
            .bind(&fixture.resolved, wave.nodes()[0].node_id())
            .unwrap();
        let node = identity.materialize_node(0).unwrap();
        let full = BatchedOperationInvocation::from_wave_node(
            fixture.runtime.as_ref(),
            &fixture.resolved,
            provider.dispatch(),
            identity,
            node,
            wave,
            0,
            active.iter(),
        )
        .unwrap();
        let first = recipe
            .prepare(
                &fixture.runtime,
                &fixture.resolved,
                provider.dispatch(),
                identity,
                node,
                wave,
                active.iter(),
            )
            .unwrap()
            .unwrap();
        let second = recipe
            .prepare(
                &fixture.runtime,
                &fixture.resolved,
                provider.dispatch(),
                identity,
                node,
                wave,
                active.iter(),
            )
            .unwrap()
            .unwrap();
        assert_eq!(first.participant_count(), 2);
        assert_eq!(*first.cold_state::<u64>().unwrap(), 7);
        for (index, participant) in full.participants().iter().enumerate() {
            let expected = participant
                .persistent_view()
                .unwrap()
                .translate(0, 4)
                .unwrap();
            let expected = expected.iter().next().unwrap();
            let actual = first.region(index, 0).unwrap().physical_region();
            let (expected_buffer, expected_range, _) = expected.buffer_and_physical_range();
            let (actual_buffer, actual_range, _) = actual.buffer_and_physical_range();
            assert!(std::ptr::eq(expected_buffer, actual_buffer));
            assert_eq!(actual_range, expected_range);
        }
        assert!(first.region(2, 0).is_err());
        assert!(first.region(0, 1).is_err());
        assert!(first.region(usize::MAX, usize::MAX).is_err());
        let spec = dependency();
        let old = full.retained_plan_dependency(spec.as_spec()).unwrap();
        let fresh = first.retained_plan_dependency(spec.as_spec()).unwrap();
        let next = second.retained_plan_dependency(spec.as_spec()).unwrap();
        assert_eq!(old.identity, fresh.identity);
        assert_eq!(fresh.identity, next.identity);
        assert!(!Arc::ptr_eq(&old.scope, &fresh.scope));
        assert!(!Arc::ptr_eq(&fresh.scope, &next.scope));
        let mut identities = Vec::new();
        let mut commands = Vec::new();
        let mut leases = Vec::new();
        assert!(
            crate::vnext::operation::retained_dependency::append_dependencies(
                &first.dependency_scope,
                vec![next.encode(TestCommand::DynamicBinding)],
                &mut identities,
                &mut commands,
                &mut leases
            )
            .is_err()
        );
        assert!(commands.is_empty());
        crate::vnext::operation::retained_dependency::append_dependencies(
            &first.dependency_scope,
            vec![fresh.encode(TestCommand::DynamicBinding)],
            &mut identities,
            &mut commands,
            &mut leases,
        )
        .unwrap();
        assert_eq!(commands.len(), 1);
        let mut undeclared = spec;
        undeclared.validation_identity.push_str(".different");
        assert!(first
            .retained_plan_dependency(undeclared.as_spec())
            .is_err());
    });
}

#[test]
fn steady_recipe_generic_metadata_fallback_preserves_original_getter_failure() {
    with_recipe_wave(|fixture, wave, identity, active| {
        let owner = Arc::new(());
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .immutable_metadata_enabled = false;
        assert!(seal_recipe(
            fixture,
            wave,
            identity,
            active,
            0,
            Arc::new(()),
            &owner,
            DeviceReusableExecutionEntryIdentity::new()
        )
        .unwrap()
        .is_none());
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .immutable_metadata_enabled = true;
        let recipe = seal_recipe(
            fixture,
            wave,
            identity,
            active,
            0,
            Arc::new(()),
            &owner,
            DeviceReusableExecutionEntryIdentity::new(),
        )
        .unwrap()
        .unwrap();
        let provider = fixture
            .registry
            .bind(&fixture.resolved, wave.nodes()[0].node_id())
            .unwrap();
        let node = identity.materialize_node(0).unwrap();
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .immutable_metadata_enabled = false;
        assert!(recipe
            .prepare(
                &fixture.runtime,
                &fixture.resolved,
                provider.dispatch(),
                identity,
                node,
                wave,
                active.iter()
            )
            .unwrap()
            .is_none());
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .tamper_buffer_descriptor = true;
        let fallback = BatchedOperationInvocation::from_wave_node(
            fixture.runtime.as_ref(),
            &fixture.resolved,
            provider.dispatch(),
            identity,
            node,
            wave,
            0,
            active.iter(),
        );
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .tamper_buffer_descriptor = false;
        assert!(
            fallback.is_err(),
            "unsupported capability must retain the generic checks"
        );
    });
}

#[test]
fn steady_recipe_checks_unexported_resources_and_rejects_hot_descriptor_drift() {
    with_recipe_wave(|fixture, wave, identity, active| {
        let owner = Arc::new(());
        let recipe = seal_recipe(
            fixture,
            wave,
            identity,
            active,
            0,
            Arc::new(()),
            &owner,
            DeviceReusableExecutionEntryIdentity::new(),
        )
        .unwrap()
        .unwrap();
        let provider = fixture
            .registry
            .bind(&fixture.resolved, wave.nodes()[0].node_id())
            .unwrap();
        let node = identity.materialize_node(0).unwrap();
        let full = BatchedOperationInvocation::from_wave_node(
            fixture.runtime.as_ref(),
            &fixture.resolved,
            provider.dispatch(),
            identity,
            node,
            wave,
            0,
            active.iter(),
        )
        .unwrap();
        // Only Persistent is exported. The separate program-binding buffer must
        // nevertheless remain inside the complete current resource validation.
        let omitted = full.participants()[0]
            .binding_view()
            .unwrap()
            .translate(0, 1)
            .unwrap();
        let omitted = omitted
            .iter()
            .next()
            .unwrap()
            .buffer_and_physical_range()
            .0
            .descriptor
            .resource_id
            .clone();
        fixture.runtime_trace.lock().unwrap().tamper_buffer_resource = Some(omitted);
        let rejected = recipe.prepare(
            &fixture.runtime,
            &fixture.resolved,
            provider.dispatch(),
            identity,
            node,
            wave,
            active.iter(),
        );
        fixture.runtime_trace.lock().unwrap().tamper_buffer_resource = None;
        assert!(rejected.is_err());
        assert!(recipe
            .prepare(
                &fixture.runtime,
                &fixture.resolved,
                provider.dispatch(),
                identity,
                node,
                wave,
                active.iter()
            )
            .unwrap()
            .is_some());
        fixture
            .runtime
            .use_alternate_descriptor
            .store(true, Ordering::Release);
        let rejected = recipe.prepare(
            &fixture.runtime,
            &fixture.resolved,
            provider.dispatch(),
            identity,
            node,
            wave,
            active.iter(),
        );
        fixture
            .runtime
            .use_alternate_descriptor
            .store(false, Ordering::Release);
        assert!(rejected.is_err());
    });
}

#[test]
fn steady_recipe_rejects_current_participant_order_and_foreign_plan_owners() {
    with_recipe_wave(|fixture, wave, identity, active| {
        let owner = Arc::new(());
        let recipe = seal_recipe(
            fixture,
            wave,
            identity,
            active,
            0,
            Arc::new(()),
            &owner,
            DeviceReusableExecutionEntryIdentity::new(),
        )
        .unwrap()
        .unwrap();
        let provider = fixture
            .registry
            .bind(&fixture.resolved, wave.nodes()[0].node_id())
            .unwrap();
        let node = identity.materialize_node(0).unwrap();
        assert!(recipe
            .prepare(
                &fixture.runtime,
                &fixture.resolved,
                provider.dispatch(),
                identity,
                node,
                wave,
                active.iter().rev()
            )
            .is_err());
        assert!(recipe
            .prepare(
                &fixture.runtime,
                &fixture.resolved,
                provider.dispatch(),
                identity,
                identity.materialize_node(1).unwrap(),
                wave,
                active.iter()
            )
            .is_err());
        with_recipe_wave(|foreign, foreign_wave, foreign_identity, foreign_active| {
            let foreign_provider = foreign
                .registry
                .bind(&foreign.resolved, foreign_wave.nodes()[0].node_id())
                .unwrap();
            let foreign_node = foreign_identity.materialize_node(0).unwrap();
            assert!(recipe
                .prepare(
                    &foreign.runtime,
                    &foreign.resolved,
                    foreign_provider.dispatch(),
                    foreign_identity,
                    foreign_node,
                    foreign_wave,
                    foreign_active.iter()
                )
                .is_err());
            // Equal schema and a valid, independently admitted Plan do not
            // authorize replacing the captured physical Plan allocation.
            assert!(recipe
                .prepare(
                    &fixture.runtime,
                    &foreign.resolved,
                    foreign_provider.dispatch(),
                    foreign_identity,
                    foreign_node,
                    foreign_wave,
                    foreign_active.iter()
                )
                .is_err());
        });
    });
}

struct OffTiming;
impl DeviceSubmissionTimingSink for OffTiming {
    const ENABLED: bool = false;
    fn record_device_submission(&self, _: DeviceSubmissionStage, _: Duration) {
        panic!("Off device callback");
    }
}
impl SubmissionWaveDispatchTimingSink for OffTiming {
    fn record(&self, _: SubmissionWaveDispatchStage, _: Duration) {
        panic!("Off host callback");
    }
}
#[derive(Default)]
struct PreparationCapture(Mutex<Vec<SteadyRecipePreparationStats>>);
impl InvocationPreparationSink for PreparationCapture {
    fn record_preparation(&self, _: InvocationPreparationStats) {}
    fn record_steady_recipe(&self, stats: SteadyRecipePreparationStats) {
        self.0.lock().unwrap().push(stats);
    }
}

#[test]
fn steady_recipe_default_none_complete_dispatch_preserves_off_and_normal_retirement() {
    // The generic runtime/provider deliberately keep their default None hooks.
    let fixture = fixture();
    let resources = (0..2)
        .map(|i| {
            logical_resources(
                &fixture.plan_resources,
                &format!("run.steady-none.{i}"),
                &format!("request.steady-none.{i}"),
            )
        })
        .collect::<Vec<_>>();
    let sessions = resources
        .iter()
        .map(|r| r.open_session().unwrap())
        .collect::<Vec<_>>();
    let batch = ExecutionBatchParticipants::new(sessions.clone()).unwrap();
    let lane = fixture.plan_resources.create_execution_lane().unwrap();
    let step = step_for(&batch, &lane, vec![one_token_span(); 2]);
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active = sessions
        .iter()
        .map(|s| TrustedActiveSequenceBinding::from_session(s).unwrap())
        .collect::<Vec<_>>();
    let topology =
        OperationDispatch::compile_submission_wave_identity(&fixture.resolved, &lane).unwrap();
    let identity = OperationDispatch::bind_compiled_submission_wave_identity_with_preparation(
        &topology,
        active.iter(),
        &wave,
        &lane,
        InvocationPreparationStrategy::SteadyRecipe,
    )
    .unwrap();
    let providers = fixture.registry.bind_plan(&fixture.resolved).unwrap();
    let reaper = CompletionReaper::new();
    let capture = PreparationCapture::default();
    let (completion, _) = OperationDispatch::encode_and_submit_wave_with_inputs_and_preparation(
        providers.providers(),
        &fixture.resolved,
        &identity,
        active.iter(),
        DeviceTimingMode::Off,
        &[],
        SubmissionExecutionPolicy::adaptive(),
        InvocationPreparationStrategy::SteadyRecipe,
        &OffTiming,
        &capture,
        wave,
        &lane,
        &reaper,
    )
    .unwrap()
    .into_parts();
    assert!(matches!(
        completion.poll().unwrap(),
        CompletionObservation::Terminal(_)
    ));
    assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 1);
    assert!(fixture.provider_trace.lock().unwrap().encode_calls > 0);
    assert_eq!(identity.preparation_snapshot().parts_materialized, 0);
    assert_eq!(lane.in_flight_count(), 0);
    assert_eq!(reaper.retained_count(), 0);
    let records = capture.0.lock().unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].prepared_nodes, 0);
    assert_eq!(records[0].invalid_nodes, 0);
    drop(records);
    drop(completion);
    drop(providers);
    drop(active);
    drop(reaper);
    step.try_retire_normal().unwrap();
    for session in &sessions {
        session.try_complete().unwrap();
    }
    drop(batch);
    drop(sessions);
    drop(resources);
    drop(lane);
    drop(fixture.registry);
    drop(fixture.impostor_registry);
    drop(fixture.runtime);
    assert!(matches!(
        PlanRuntimeResources::close(fixture.plan_resources),
        Ok(PlanRuntimeCloseOutcome::Closed(_))
    ));
}
