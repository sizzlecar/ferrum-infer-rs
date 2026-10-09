//! Actual admitted multi-node waves exercise the private agreement boundary.
use super::*;
use ferrum_types::InvocationPreparationStrategy;
use std::cell::Cell;
use vnext_device_operation_wave_contract::{setup, teardown, wave_active_bindings};

fn agreement_identity(
    fixture: &Fixture,
    wave: &PreparedStepSubmissionWave<TestRuntime>,
    active: &[TrustedActiveSequenceBinding],
) -> BatchOperationIdentity {
    let lane = wave.step_resources().execution_lane();
    let topology =
        OperationDispatch::compile_submission_wave_identity(&fixture.resolved, lane).unwrap();
    OperationDispatch::bind_compiled_submission_wave_identity_with_preparation(
        &topology,
        active.iter(),
        wave,
        lane,
        InvocationPreparationStrategy::WaveAgreement,
    )
    .unwrap()
}

fn materialize<'a, 'binding, I>(
    fixture: &'a Fixture,
    resolved: &'a dyn ExecutablePlanView,
    wave: &'a PreparedStepSubmissionWave<TestRuntime>,
    identity: &'a BatchOperationIdentity,
    node_index: usize,
    active: I,
    agreements: Option<&mut WaveInvocationAgreements<'a, 'binding, TestRuntime>>,
) -> Result<Vec<serde_json::Value>, VNextError>
where
    I: ExactSizeIterator<Item = &'binding TrustedActiveSequenceBinding>,
{
    // The originally bound provider stays fixed while the current plan operand
    // can change. A replacement plan must not silently replace that provider.
    let provider = fixture
        .registry
        .bind(&fixture.resolved, wave.nodes()[node_index].node_id())?;
    let node = identity.materialize_node(node_index)?;
    let invocation = BatchedOperationInvocation::from_wave_node_with_agreements(
        fixture.runtime.as_ref(),
        resolved,
        provider.dispatch(),
        identity,
        node,
        wave,
        node_index,
        active,
        false,
        agreements,
    )?;
    Ok(snapshot(&invocation))
}

#[test]
fn wave_agreement_matches_full_for_distinct_nodes_and_unequal_participant_windows() {
    with_live_wave_spans(vec![1, 3], |fixture, wave, reference, active| {
        let identity = agreement_identity(fixture, wave, active);
        let mut agreements = WaveInvocationAgreements::new(&identity, wave);
        for index in 0..wave.node_count() {
            let expected = materialize(
                fixture,
                &fixture.resolved,
                wave,
                reference,
                index,
                active.iter(),
                None,
            )
            .unwrap();
            let actual = materialize(
                fixture,
                &fixture.resolved,
                wave,
                &identity,
                index,
                active.iter(),
                Some(&mut agreements),
            )
            .unwrap();
            assert_eq!(actual, expected);
        }
        let stats = agreements.snapshot();
        assert_eq!(stats.agreement_builds, active.len() as u64);
        assert!(
            stats.agreement_reuses > 0,
            "a later real node must reuse agreement"
        );
        assert_eq!(stats.agreement_fallbacks, 0);
        assert_eq!(identity.preparation_snapshot().parts_materialized, 0);
        assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 0);
    });
}

#[test]
fn wave_agreement_later_node_accepts_equal_active_clones_but_rejects_reordering() {
    with_live_wave_spans(vec![1, 3], |fixture, wave, _, active| {
        let identity = agreement_identity(fixture, wave, active);
        let cloned = active.to_vec();
        let mut agreements = WaveInvocationAgreements::new(&identity, wave);
        materialize(
            fixture,
            &fixture.resolved,
            wave,
            &identity,
            0,
            active.iter(),
            Some(&mut agreements),
        )
        .unwrap();
        let before = agreements.snapshot();
        let expected = materialize(
            fixture,
            &fixture.resolved,
            wave,
            &identity,
            1,
            cloned.iter(),
            None,
        )
        .unwrap();
        let actual = materialize(
            fixture,
            &fixture.resolved,
            wave,
            &identity,
            1,
            cloned.iter(),
            Some(&mut agreements),
        )
        .unwrap();
        assert_eq!(actual, expected);
        let after = agreements.snapshot();
        assert_eq!(after.agreement_reuses, before.agreement_reuses);
        assert!(after.agreement_fallbacks > before.agreement_fallbacks);
        assert!(materialize(
            fixture,
            &fixture.resolved,
            wave,
            &identity,
            1,
            cloned.iter().rev(),
            Some(&mut agreements)
        )
        .is_err());
        // Rejection cannot poison a subsequent valid construction on this wave.
        assert_eq!(
            materialize(
                fixture,
                &fixture.resolved,
                wave,
                &identity,
                1,
                cloned.iter(),
                Some(&mut agreements)
            )
            .unwrap(),
            expected
        );
    });
}

struct ChangingPlan<'a> {
    original: &'a ResolvedModelPlan,
    alternate: &'a ExecutionPlan,
    switched: Cell<bool>,
}

impl ExecutablePlanView for ChangingPlan<'_> {
    fn execution_plan(&self) -> &ExecutionPlan {
        if self.switched.get() {
            self.alternate
        } else {
            self.original.execution_plan()
        }
    }
    fn device(&self) -> &DeviceDescriptor {
        ExecutablePlanView::device(self.original)
    }
    fn capabilities(&self) -> &CapabilityCatalog {
        self.original.capabilities()
    }
}

#[test]
fn wave_agreement_later_node_rechecks_changed_execution_plan_operand() {
    let foreign = fixture_with_zero_state(true);
    with_live_wave(2, |fixture, wave, _, active| {
        let identity = agreement_identity(fixture, wave, active);
        let equal_plan = fixture.resolved.execution_plan().clone();
        for (alternate, valid) in [(&equal_plan, true), (&foreign.plan, false)] {
            let current = ChangingPlan {
                original: &fixture.resolved,
                alternate,
                switched: Cell::new(false),
            };
            let mut agreements = WaveInvocationAgreements::new(&identity, wave);
            materialize(
                fixture,
                &current,
                wave,
                &identity,
                0,
                active.iter(),
                Some(&mut agreements),
            )
            .unwrap();
            let before = agreements.snapshot();
            current.switched.set(true);
            let expected = materialize(fixture, &current, wave, &identity, 1, active.iter(), None);
            let actual = materialize(
                fixture,
                &current,
                wave,
                &identity,
                1,
                active.iter(),
                Some(&mut agreements),
            );
            assert_eq!(expected.is_ok(), valid);
            assert_eq!(actual.is_ok(), valid);
            if let (Ok(actual), Ok(expected)) = (actual, expected) {
                assert_eq!(actual, expected);
            }
            assert_eq!(
                agreements.snapshot().agreement_reuses,
                before.agreement_reuses
            );
            if valid {
                assert!(agreements.snapshot().agreement_fallbacks > before.agreement_fallbacks);
            }
        }
    });
}

#[test]
fn wave_agreement_later_node_rechecks_runtime_descriptor_and_buffer_coverage() {
    let configured = fixture_with_runtime_configuration(|runtime| {
        runtime.alternate_descriptor = runtime.descriptor.clone();
        runtime.alternate_descriptor.total_memory_bytes += 1;
    });
    with_live_wave_fixture(configured, vec![1, 3], |fixture, wave, _, active| {
        let identity = agreement_identity(fixture, wave, active);
        let mut agreements = WaveInvocationAgreements::new(&identity, wave);
        materialize(
            fixture,
            &fixture.resolved,
            wave,
            &identity,
            0,
            active.iter(),
            Some(&mut agreements),
        )
        .unwrap();
        fixture
            .runtime
            .use_alternate_descriptor
            .store(true, Ordering::Release);
        assert!(materialize(
            fixture,
            &fixture.resolved,
            wave,
            &identity,
            1,
            active.iter(),
            None
        )
        .is_err());
        assert!(materialize(
            fixture,
            &fixture.resolved,
            wave,
            &identity,
            1,
            active.iter(),
            Some(&mut agreements)
        )
        .is_err());
        fixture
            .runtime
            .use_alternate_descriptor
            .store(false, Ordering::Release);
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .tamper_buffer_descriptor = true;
        assert!(materialize(
            fixture,
            &fixture.resolved,
            wave,
            &identity,
            1,
            active.iter(),
            None
        )
        .is_err());
        assert!(materialize(
            fixture,
            &fixture.resolved,
            wave,
            &identity,
            1,
            active.iter(),
            Some(&mut agreements)
        )
        .is_err());
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .tamper_buffer_descriptor = false;
        materialize(
            fixture,
            &fixture.resolved,
            wave,
            &identity,
            1,
            active.iter(),
            Some(&mut agreements),
        )
        .unwrap();
    });
}

#[test]
fn wave_agreement_failed_participant_does_not_publish_partial_node_proof() {
    with_live_wave(2, |fixture, wave, _, active| {
        let identity = agreement_identity(fixture, wave, active);
        let wrong_second = [&active[0], &active[0]];
        let mut agreements = WaveInvocationAgreements::new(&identity, wave);
        assert!(materialize(
            fixture,
            &fixture.resolved,
            wave,
            &identity,
            0,
            wrong_second.into_iter(),
            Some(&mut agreements)
        )
        .is_err());
        assert_eq!(agreements.snapshot().agreement_builds, 0);
        assert_eq!(agreements.snapshot().agreement_reuses, 0);
        materialize(
            fixture,
            &fixture.resolved,
            wave,
            &identity,
            0,
            active.iter(),
            Some(&mut agreements),
        )
        .unwrap();
        assert_eq!(agreements.snapshot().agreement_builds, active.len() as u64);
        materialize(
            fixture,
            &fixture.resolved,
            wave,
            &identity,
            1,
            active.iter(),
            Some(&mut agreements),
        )
        .unwrap();
        assert!(agreements.snapshot().agreement_reuses > 0);
        assert_eq!(fixture.provider_trace.lock().unwrap().encode_calls, 0);
        assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 0);
    });
}

#[test]
fn wave_agreement_cannot_authorize_another_admitted_wave_or_recipe() {
    with_live_wave(2, |fixture, wave, _, active| {
        let identity = agreement_identity(fixture, wave, active);
        let equal_recipe = agreement_identity(fixture, wave, active);
        with_live_wave(2, |other_fixture, other_wave, _, other_active| {
            let other_identity = agreement_identity(other_fixture, other_wave, other_active);
            let mut agreements = WaveInvocationAgreements::new(&identity, wave);
            materialize(
                fixture,
                &fixture.resolved,
                wave,
                &identity,
                0,
                active.iter(),
                Some(&mut agreements),
            )
            .unwrap();
            let before = agreements.snapshot();
            // Equal metadata from a distinct private compiled owner is legal,
            // but it cannot borrow the first owner's cached agreement.
            materialize(
                fixture,
                &fixture.resolved,
                wave,
                &equal_recipe,
                1,
                active.iter(),
                Some(&mut agreements),
            )
            .unwrap();
            assert_eq!(
                agreements.snapshot().agreement_reuses,
                before.agreement_reuses
            );
            // The other admitted wave remains legal through full validation.
            materialize(
                other_fixture,
                &other_fixture.resolved,
                other_wave,
                &other_identity,
                1,
                other_active.iter(),
                Some(&mut agreements),
            )
            .unwrap();
            assert_eq!(
                agreements.snapshot().agreement_reuses,
                before.agreement_reuses
            );
            // The first wave identity itself cannot authorize the other wave.
            assert!(materialize(
                other_fixture,
                &other_fixture.resolved,
                other_wave,
                &identity,
                1,
                other_active.iter(),
                Some(&mut agreements)
            )
            .is_err());
        });
    });
}

struct AgreementOffTiming;
impl DeviceSubmissionTimingSink for AgreementOffTiming {
    const ENABLED: bool = false;
    fn record_device_submission(&self, _: DeviceSubmissionStage, _: Duration) {
        panic!("Off must not call timing hooks");
    }
}
impl SubmissionWaveDispatchTimingSink for AgreementOffTiming {
    fn record(&self, _: SubmissionWaveDispatchStage, _: Duration) {
        panic!("Off must not call timing hooks");
    }
}
#[derive(Default)]
struct AgreementCapture(Mutex<Vec<InvocationPreparationStats>>);
impl InvocationPreparationSink for AgreementCapture {
    fn record_preparation(&self, stats: InvocationPreparationStats) {
        self.0.lock().unwrap().push(stats);
    }
}

#[test]
fn wave_agreement_actual_dispatch_observes_changed_clone_iterator_after_first_encode() {
    let (fixture, sequence, session, batch, step) = setup();
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active = wave_active_bindings(&wave, &session);
    let cloned = active.clone();
    let identity = agreement_identity(&fixture, &wave, &active);
    let lane = Arc::clone(step.execution_lane());
    let providers = fixture.registry.bind_plan(&fixture.resolved).unwrap();
    let reaper = CompletionReaper::new();
    let capture = AgreementCapture::default();
    // Clone of this iterator retains the selection logic, not a pre-collected
    // immutable list. The second actual provider consumes equal new bindings.
    let current = active.iter().enumerate().map(|(index, original)| {
        if fixture.provider_trace.lock().unwrap().encode_calls > 0 {
            &cloned[index]
        } else {
            original
        }
    });
    let (handle, _) = OperationDispatch::encode_and_submit_wave_with_inputs_and_preparation(
        providers.providers(),
        &fixture.resolved,
        &identity,
        current,
        DeviceTimingMode::Off,
        &[],
        SubmissionExecutionPolicy::adaptive(),
        InvocationPreparationStrategy::WaveAgreement,
        &AgreementOffTiming,
        &capture,
        wave,
        &lane,
        &reaper,
    )
    .unwrap()
    .into_parts();
    let records = capture.0.lock().unwrap();
    assert_eq!(records.len(), 1);
    assert!(records[0].agreement_fallbacks > 0);
    assert_eq!(records[0].agreement_reuses, 0);
    assert_eq!(records[0].parts_materialized, 0);
    drop(records);
    assert_eq!(
        fixture.provider_trace.lock().unwrap().encode_calls,
        fixture.plan.payload().nodes().len() as u64
    );
    assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 1);
    assert!(matches!(
        handle.poll().unwrap(),
        CompletionObservation::Terminal(_)
    ));
    assert_eq!(lane.in_flight_count(), 0);
    drop(handle);
    drop(providers);
    drop(active);
    drop(cloned);
    drop(lane);
    drop(reaper);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn wave_agreement_definitely_not_submitted_retry_builds_fresh_attempt_proofs() {
    let (fixture, sequence, session, batch, step) = setup();
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let prior_attempt = wave.batch_invocation_id();
    let active = wave_active_bindings(&wave, &session);
    let first_identity = agreement_identity(&fixture, &wave, &active);
    let lane = Arc::clone(step.execution_lane());
    let providers = fixture.registry.bind_plan(&fixture.resolved).unwrap();
    let reaper = CompletionReaper::new();
    let capture = AgreementCapture::default();
    fixture.runtime_trace.lock().unwrap().submit_behavior = SubmitBehavior::DefinitelyNotSubmitted;
    let retry = match OperationDispatch::encode_and_submit_wave_with_inputs_and_preparation(
        providers.providers(),
        &fixture.resolved,
        &first_identity,
        active.iter(),
        DeviceTimingMode::Off,
        &[],
        SubmissionExecutionPolicy::adaptive(),
        InvocationPreparationStrategy::WaveAgreement,
        &AgreementOffTiming,
        &capture,
        wave,
        &lane,
        &reaper,
    ) {
        Err(SubmissionWaveDispatchError::DefinitelyNotSubmitted { retry, .. }) => retry,
        Err(error) => panic!("expected real typed runtime rejection, got {error:?}"),
        Ok(_) => panic!("expected real typed runtime rejection, got successful submission"),
    };
    assert_eq!(lane.in_flight_count(), 0);
    assert_eq!(reaper.retained_count(), 0);
    let wave = retry.retry().unwrap();
    assert_ne!(wave.batch_invocation_id(), prior_attempt);
    let identity = agreement_identity(&fixture, &wave, &active);
    // Old batch attempt cannot substitute for the newly admitted attempt.
    {
        let mut old = WaveInvocationAgreements::new(&first_identity, &wave);
        assert!(materialize(
            &fixture,
            &fixture.resolved,
            &wave,
            &first_identity,
            0,
            active.iter(),
            Some(&mut old)
        )
        .is_err());
        assert_eq!(old.snapshot().agreement_builds, 0);
        assert_eq!(old.snapshot().agreement_reuses, 0);
    }
    fixture.runtime_trace.lock().unwrap().submit_behavior = SubmitBehavior::Success;
    let (handle, _) = OperationDispatch::encode_and_submit_wave_with_inputs_and_preparation(
        providers.providers(),
        &fixture.resolved,
        &identity,
        active.iter(),
        DeviceTimingMode::Off,
        &[],
        SubmissionExecutionPolicy::adaptive(),
        InvocationPreparationStrategy::WaveAgreement,
        &AgreementOffTiming,
        &capture,
        wave,
        &lane,
        &reaper,
    )
    .unwrap()
    .into_parts();
    let records = capture.0.lock().unwrap();
    assert_eq!(records.len(), 2);
    for record in records.iter() {
        assert_eq!(record.agreement_builds, active.len() as u64);
        assert!(record.agreement_reuses > 0);
        assert_eq!(record.agreement_fallbacks, 0);
    }
    drop(records);
    assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 2);
    assert!(matches!(
        handle.poll().unwrap(),
        CompletionObservation::Terminal(_)
    ));
    assert_eq!(lane.in_flight_count(), 0);
    assert_eq!(reaper.retained_count(), 0);
    drop(handle);
    drop(providers);
    drop(active);
    drop(lane);
    drop(reaper);
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn wave_agreement_warm_identity_proof_does_not_keep_cancelled_session_open() {
    let (fixture, sequence, session, batch, step) = setup();
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active = wave_active_bindings(&wave, &session);
    let identity = agreement_identity(&fixture, &wave, &active);
    let lane = Arc::clone(step.execution_lane());
    let topology =
        OperationDispatch::compile_submission_wave_identity(&fixture.resolved, &lane).unwrap();
    let foreign_lane = fixture.plan_resources.create_execution_lane().unwrap();
    assert!(
        OperationDispatch::bind_compiled_submission_wave_identity_with_preparation(
            &topology,
            active.iter(),
            &wave,
            &foreign_lane,
            InvocationPreparationStrategy::WaveAgreement,
        )
        .is_err()
    );
    {
        let mut agreements = WaveInvocationAgreements::new(&identity, &wave);
        materialize(
            &fixture,
            &fixture.resolved,
            &wave,
            &identity,
            0,
            active.iter(),
            Some(&mut agreements),
        )
        .unwrap();
        materialize(
            &fixture,
            &fixture.resolved,
            &wave,
            &identity,
            1,
            active.iter(),
            Some(&mut agreements),
        )
        .unwrap();
        assert!(agreements.snapshot().agreement_reuses > 0);
        session.request_cancel().unwrap();
        assert!(active[0].ensure_open_for_emission().is_err());
        assert!(
            OperationDispatch::bind_compiled_submission_wave_identity_with_preparation(
                &topology,
                active.iter(),
                &wave,
                &lane,
                InvocationPreparationStrategy::WaveAgreement,
            )
            .is_err()
        );
    }
    let providers = fixture.registry.bind_plan(&fixture.resolved).unwrap();
    let reaper = CompletionReaper::new();
    assert!(OperationDispatch::encode_and_submit_wave(
        providers.providers(),
        &fixture.resolved,
        &identity,
        active.iter(),
        DeviceTimingMode::Off,
        wave,
        &lane,
        &reaper,
    )
    .is_err());
    assert_eq!(fixture.provider_trace.lock().unwrap().encode_calls, 0);
    assert_eq!(fixture.runtime_trace.lock().unwrap().submit_calls, 0);
    assert_eq!(lane.in_flight_count(), 0);
    assert_eq!(reaper.retained_count(), 0);
    drop(providers);
    drop(active);
    drop(lane);
    drop(foreign_lane);
    drop(reaper);
    let retired = step.try_retire_normal().unwrap();
    assert_eq!(
        retired.participants()[0].disposition(),
        StepParticipantRetirementDisposition::DiscardedCancelled
    );
    drop(retired);
    drop(batch);
    session.try_abort().unwrap();
    drop(session);
    drop(sequence);
    drop(fixture.registry);
    drop(fixture.impostor_registry);
    drop(fixture.runtime);
    assert!(matches!(
        PlanRuntimeResources::close(fixture.plan_resources),
        Ok(PlanRuntimeCloseOutcome::Closed(_))
    ));
}
