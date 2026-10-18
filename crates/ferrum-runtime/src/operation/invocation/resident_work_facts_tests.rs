//! The original operation dispatcher publishes a real cold recipe, compiles a
//! resident program from its reserved bootstrap, then submits fresh hot uses.
//! TestRuntime fences exercise ownership and work evidence, not device numerics.
use super::*;
use crate::operation::dispatch_contract::DisabledSubmissionWaveDispatchTimingSink;
use ferrum_types::InvocationPreparationStrategy;
use vnext_device_operation_wave_contract::test_reusable_program;

struct Preparation;
impl InvocationPreparationSink for Preparation {
    fn record_preparation(&self, _: InvocationPreparationStats) {}
}

#[derive(Default)]
struct FactsSink(AtomicU64);
impl ExecutionEventSink for FactsSink {
    fn is_enabled(&self, _: ExecutionEventKind) -> bool {
        false
    }
    fn record(&self, _: EventEmissionPermit) -> Result<(), ExecutionEventSinkError> {
        Ok(())
    }
    fn execution_facts_capture(&self) -> ExecutionFactsCapture {
        ExecutionFactsCapture::On
    }
    fn record_execution_facts(
        &self,
        facts: TerminalExecutionFacts<'_>,
    ) -> Result<(), ExecutionEventSinkError> {
        assert_eq!(
            facts.completion().disposition(),
            &OperationCompletionDisposition::Succeeded
        );
        assert!(facts.attribution().is_none());
        assert_eq!(facts.work().submission(), facts.completion().submission());
        self.0.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

fn span(start: usize) -> TokenSpanWork {
    TokenSpanWork::from_token_ids(&[11, 12, 13, 14], start..start + 1).unwrap()
}

fn request(
    batch: &ExecutionBatchParticipants<TestRuntime>,
    bucket: &ReusableExecutionBucketSpec,
    start: usize,
) -> StepResourceAdmissionRequest {
    StepResourceAdmissionRequest::new(
        batch.bind_work_shape(vec![span(start)]).unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap()
    .with_reusable_execution_bucket(bucket.bucket_id().clone())
}

#[allow(clippy::too_many_arguments)]
fn bootstrap(
    fixture: &Fixture,
    batch: &ExecutionBatchParticipants<TestRuntime>,
    lane: &Arc<ExecutionLane<TestRuntime>>,
    reaper: &Arc<CompletionReaper<TestRuntime>>,
    topology: &CompiledSubmissionWaveIdentity,
    replay: &DeviceReusableExecutionProgram,
    step: Arc<StepResourceLease<TestRuntime>>,
    wave: PreparedStepSubmissionWave<TestRuntime>,
    publish: bool,
) -> OperationCompletionReceipt {
    let weak = Arc::downgrade(&step);
    let active = batch
        .sessions()
        .iter()
        .map(|s| TrustedActiveSequenceBinding::from_session(s).unwrap())
        .collect::<Vec<_>>();
    let identity = OperationDispatch::bind_compiled_submission_wave_identity_with_preparation(
        topology,
        active.iter(),
        &wave,
        lane,
        InvocationPreparationStrategy::DecodeSegmentProgram,
    )
    .unwrap();
    let providers = fixture.registry.bind_plan(&fixture.resolved).unwrap();
    let (handle, observation) =
        OperationDispatch::encode_and_submit_reusable_wave_with_inputs_and_preparation(
            providers.providers(),
            &fixture.resolved,
            &identity,
            active.iter(),
            DeviceTimingMode::Off,
            ExecutionFactsCapture::Off,
            &[],
            replay,
            SubmissionExecutionPolicy::adaptive(),
            InvocationPreparationStrategy::DecodeSegmentProgram,
            &DisabledSubmissionWaveDispatchTimingSink,
            &Preparation,
            wave,
            lane,
            reaper,
        )
        .unwrap()
        .into_parts();
    assert!(observation.work().is_none());
    let publication = handle.take_segment_binding_publication();
    assert_eq!(publication.is_some(), publish);
    let CompletionObservation::Terminal(completion) = handle.wait().unwrap() else {
        panic!("real bootstrap must terminalize");
    };
    assert_eq!(
        completion.disposition(),
        &OperationCompletionDisposition::Succeeded
    );
    drop(handle);
    let retirement = step.try_retire_normal().unwrap();
    assert!(
        weak.upgrade().is_none(),
        "resident cold state must not retain the bootstrap Step"
    );
    if let Some(publication) = publication {
        assert!(publication
            .bind_completion(&completion)
            .unwrap()
            .publish(&retirement)
            .unwrap());
    }
    assert_eq!(reaper.retained_count(), 0);
    assert_eq!(lane.in_flight_count(), 0);
    completion
}

fn resident_facts(post_submit_panic: bool) {
    let bucket = ReusableExecutionBucketSpec::new(
        ReusableExecutionClassId::new("execution.resident-work-facts").unwrap(),
        ReusableExecutionCapacity::new(1, 1, 1).unwrap(),
    )
    .unwrap();
    let fixture = fixture_with_token_scaled_paged_state_and_bucket(bucket.clone());
    {
        let mut trace = fixture.runtime_trace.lock().unwrap();
        trace.segment_metadata_enabled = true;
        trace.segment_encoder_enabled = true;
        trace.resident_enabled = true;
        trace.segment_entry = Some(DeviceReusableExecutionEntryIdentity::new());
    }
    // Establish actual physical coverage for every later source window, while
    // each submitted resident use still has exactly one immediate token.
    let sequence = logical_resources_with_work(
        &fixture.plan_resources,
        "run.resident-work-facts",
        "request.resident-work-facts",
        TokenSpanWork::from_token_ids(&[11, 12, 13, 14], 0..4).unwrap(),
    );
    let sequence_weak = Arc::downgrade(&sequence);
    let session = sequence.open_session().unwrap();
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&session)]).unwrap();
    let lane = fixture.plan_resources.create_execution_lane().unwrap();
    let reaper = CompletionReaper::new();
    let topology =
        OperationDispatch::compile_submission_wave_identity(&fixture.resolved, &lane).unwrap();
    let step = begin_single_participant_step_on_lane_with_bucket_and_work(
        &batch,
        &lane,
        Some(&bucket),
        span(0),
    );
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let declarations = super::segment_hot_tests::compile(&fixture, &wave);
    for node in &declarations.nodes {
        fixture
            .provider_trace
            .lock()
            .unwrap()
            .segment_regions_by_node
            .insert(
                fixture.plan.payload().nodes()[node.node_index].id().clone(),
                node.declaration.regions().to_vec(),
            );
    }
    let count = u32::try_from(wave.node_count()).unwrap();
    let replay = test_reusable_program(
        declarations.program_id.clone(),
        count,
        vec![],
        vec![DeviceReusableExecutionSegment::new(0, 0, count, count).unwrap()],
        (0..count).collect(),
        vec![],
    );
    drop(declarations);
    let cold_completion = bootstrap(
        &fixture, &batch, &lane, &reaper, &topology, &replay, step, wave, true,
    );
    let first = request(&batch, &bucket, 0);
    let reservation = batch
        .try_reserve_resident_decode(&first, &lane)
        .unwrap()
        .expect("the actual warmed bucket reserves resident storage");
    let reservation_weak = Arc::downgrade(&reservation);
    let (step, wave) = match reservation.prepare_bootstrap(&first).unwrap() {
        ResidentBootstrapDecision::Prepared { step, wave } => (step, wave),
        _ => panic!("the original reserved bootstrap must remain admitted"),
    };
    let program = OperationDispatch::try_compile_resident_decode(
        Arc::clone(&reservation),
        &fixture.resolved,
        &wave,
        &topology,
        &replay,
        &lane,
    )
    .unwrap()
    .expect("complete original cached recipe must compile a resident program");
    let bootstrap_completion = bootstrap(
        &fixture, &batch, &lane, &reaper, &topology, &replay, step, wave, false,
    );
    assert_ne!(
        cold_completion.submission().slot_id(),
        bootstrap_completion.submission().slot_id()
    );
    assert!(reservation.is_idle());
    let (cold_providers, cold_encodes, cold_allocations) = {
        let providers = fixture.provider_trace.lock().unwrap().encode_calls;
        let trace = fixture.runtime_trace.lock().unwrap();
        assert_eq!(trace.resident_compile_calls, 1);
        (
            providers,
            trace.segment_program_encode_calls,
            trace.allocation_calls,
        )
    };
    let mut observations = Vec::new();
    let sink = FactsSink::default();
    let mut prior_frame = bootstrap_completion
        .submission()
        .batch_identity()
        .batch_step_id();
    let mut off_preparation = None;
    for (start, capture) in [
        (1, ExecutionFactsCapture::Off),
        (2, ExecutionFactsCapture::On),
        (3, ExecutionFactsCapture::On),
    ] {
        let req = request(&batch, &bucket, start);
        let (use_, retirement) = match reservation.try_begin_use(&batch, &req).unwrap() {
            ResidentDecodeDecision::Ready { use_, retirement } => (use_, retirement),
            _ => panic!("fresh work must reuse the original resident program"),
        };
        assert_ne!(use_.batch_step_id(), prior_frame);
        prior_frame = use_.batch_step_id();
        let active = vec![TrustedActiveSequenceBinding::from_session(&session).unwrap()];
        let identity =
            OperationDispatch::bind_resident_decode_identity(&program, active.iter(), &use_, &lane)
                .unwrap();
        let panic_now = post_submit_panic && start == 3;
        fixture
            .runtime_trace
            .lock()
            .unwrap()
            .submission_attribution_panics = panic_now;
        let submitted = OperationDispatch::encode_and_submit_resident_decode_with_inputs_and_facts(
            &program,
            &fixture.resolved,
            &identity,
            &[],
            use_,
            &lane,
            &reaper,
            DeviceTimingMode::Off,
            capture,
            &DisabledSubmissionWaveDispatchTimingSink,
        );
        let (handle, observation) = if panic_now {
            match submitted {
                Err(ResidentDecodeDispatchError::PostSubmitContract {
                    completion,
                    observation,
                    ..
                }) => (completion, observation),
                _ => panic!(
                    "real post-submit attribution failure must retain the exact resident outcome"
                ),
            }
        } else {
            submitted.unwrap().into_parts()
        };
        assert!(!handle.receipt().has_materialized_participant_receipts());
        let preparation = identity.preparation_snapshot();
        assert_eq!(preparation.parts_materialized, 0);
        assert_eq!(preparation.segment_hits, 1);
        if let Some(off) = off_preparation {
            assert_eq!(preparation, off);
        } else {
            off_preparation = Some(preparation);
        }
        let CompletionObservation::Terminal(completion) = handle.wait().unwrap() else {
            panic!("actual resident fence must terminalize");
        };
        assert_eq!(
            completion.disposition(),
            &OperationCompletionDisposition::Succeeded
        );
        assert_eq!(
            observation.work().is_some(),
            capture == ExecutionFactsCapture::On
        );
        assert!(observation.attribution().is_none());
        if let Some(work) = observation.work() {
            assert!(TerminalExecutionFacts::new(&bootstrap_completion, work, None).is_err());
            let facts = TerminalExecutionFacts::new(&completion, work, None).unwrap();
            assert_eq!(
                facts.device_coverage(),
                DeviceExecutionFactsCoverage::Unavailable(
                    DeviceExecutionFactsUnavailableReason::NotCaptured
                )
            );
            sink.record_execution_facts(facts).unwrap();
            assert_eq!(work.nodes().len(), count as usize);
            for (index, node) in work.nodes().iter().enumerate() {
                assert_eq!(node.node_index() as usize, index);
                let rows = node.participants();
                assert_eq!(rows.len(), 1);
                assert_eq!(rows[0].participant(), req.work_shape().participants()[0]);
                assert_eq!(rows[0].immediate_token_range(), 0..1);
                assert_eq!(rows[0].source_token_range(), start as u64..start as u64 + 1);
                assert_eq!(rows[0].full_input_tokens(), 4);
            }
        }
        assert!(!completion
            .submission()
            .has_materialized_participant_receipts());
        drop(handle);
        retirement.try_retire_normal().unwrap();
        assert!(reservation.is_idle());
        assert_eq!(reaper.retained_count(), 0);
        assert_eq!(lane.in_flight_count(), 0);
        observations.push((completion, observation));
    }
    {
        let trace = fixture.runtime_trace.lock().unwrap();
        assert_eq!(trace.resident_compile_calls, 1);
        assert_eq!(trace.resident_encode_calls, 3);
        assert_eq!(
            trace.resident_source_ranges,
            vec![vec![1..2], vec![2..3], vec![3..4]]
        );
        assert_eq!(trace.submit_calls, 5);
        assert_eq!(trace.segment_program_encode_calls, cold_encodes);
        assert_eq!(trace.allocation_calls, cold_allocations);
    }
    assert_eq!(
        fixture.provider_trace.lock().unwrap().encode_calls,
        cold_providers
    );
    assert_eq!(sink.0.load(Ordering::Relaxed), 2);
    drop(program);
    drop(reservation);
    assert!(reservation_weak.upgrade().is_none());
    drop(batch);
    session.try_complete().unwrap();
    drop(session);
    drop(sequence);
    assert!(
        sequence_weak.upgrade().is_none(),
        "retained terminal work must not own the sequence"
    );
    drop(reaper);
    drop(lane);
    drop(fixture.registry);
    drop(fixture.impostor_registry);
    drop(fixture.runtime);
    assert!(matches!(
        PlanRuntimeResources::close(fixture.plan_resources),
        Ok(PlanRuntimeCloseOutcome::Closed(_))
    ));
    assert_eq!(
        observations.len(),
        3,
        "facts remain readable after actual resource closure"
    );
}

#[test]
fn execution_work_facts_direct_resident_keeps_cache_and_fresh_source_windows() {
    resident_facts(false);
}

#[test]
fn execution_work_facts_direct_resident_retains_post_submit_contract_terminal() {
    resident_facts(true);
}
