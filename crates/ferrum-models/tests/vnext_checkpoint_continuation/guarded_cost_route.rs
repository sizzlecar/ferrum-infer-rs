//! Real selected providers and native encoding, followed by a rejected final
//! host gate. No fake receipt or permissive executor substitutes for rollback.
use super::*;
use ferrum_interfaces::execution_cost::{
    CoreReadbackRoute, GuardedNotSubmittedReason, HostSubmissionRejection,
    StructuredCostSampleDemand,
};
use std::sync::atomic::{AtomicU64, Ordering};

struct ExpiredAfterEncoding {
    calls: AtomicU64,
}
impl PreparedWaveSubmissionGuard for ExpiredAfterEncoding {
    fn check(
        &self,
        actual: &DeviceSubmissionAttribution,
        readback: CoreReadbackRoute,
    ) -> Result<(), GuardedNotSubmittedReason> {
        assert!(!actual.commands().is_empty());
        assert!(actual
            .commands()
            .iter()
            .any(|command| command.node_index().is_some() && command.compute_dispatch_count() > 0));
        assert!(actual
            .commands()
            .iter()
            .any(
                |command| command.command_phase() == DeviceCommandPhase::DynamicBinding
                    && command.native_op_id() == HOST_UPLOAD_NATIVE_OPERATION_ID.as_str()
            ));
        assert_eq!(readback, CoreReadbackRoute::SubmissionStaged);
        self.calls.fetch_add(1, Ordering::Relaxed);
        Err(GuardedNotSubmittedReason::HostRejected(
            HostSubmissionRejection::WitnessExpired,
        ))
    }
}

fn reject_encoded(
    fixture: &Fixture,
    session: &Arc<SequenceSession<Runtime>>,
    tokens: Arc<[u32]>,
) -> (PendingGuardedWaveRejection, Arc<StepResourceLease<Runtime>>) {
    reject_encoded_with_sample_demand(
        fixture,
        session,
        0..tokens.len(),
        tokens,
        StructuredCostSampleDemand::RuntimePolicy,
    )
}

pub(super) fn reject_encoded_with_sample_demand(
    fixture: &Fixture,
    session: &Arc<SequenceSession<Runtime>>,
    range: Range<usize>,
    tokens: Arc<[u32]>,
    demand: StructuredCostSampleDemand,
) -> (PendingGuardedWaveRejection, Arc<StepResourceLease<Runtime>>) {
    reject_encoded_output_with_sample_demand(
        fixture,
        session,
        range,
        tokens,
        demand,
        "node.attention",
        false,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn reject_encoded_output_with_sample_demand(
    fixture: &Fixture,
    session: &Arc<SequenceSession<Runtime>>,
    range: Range<usize>,
    tokens: Arc<[u32]>,
    demand: StructuredCostSampleDemand,
    output_node_id: &str,
    resident: bool,
) -> (PendingGuardedWaveRejection, Arc<StepResourceLease<Runtime>>) {
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(session)]).unwrap();
    let mut request = StepResourceAdmissionRequest::new(
        batch
            .bind_work_shape(vec![token_span(Arc::clone(&tokens), range.clone())])
            .unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    if resident {
        request = request.with_reusable_execution_bucket(
            fixture
                .reusable_bucket
                .clone()
                .expect("resident guard requires declared slots"),
        );
    }
    let step = loop {
        match batch
            .try_begin_step(request.clone(), &fixture.lane)
            .unwrap()
        {
            StepResourceAdmissionDecision::Admitted(step) => break step,
            StepResourceAdmissionDecision::BackingDeferred(deferred) => {
                require_progress(deferred.maintain().unwrap())
            }
            StepResourceAdmissionDecision::Deferred(reason) => require_progress(
                fixture
                    .resources
                    .maintain_for_admission_deferred(&reason)
                    .unwrap(),
            ),
            _ => panic!("fixture Step unavailable"),
        }
    };
    let wave = loop {
        match step
            .try_prepare_full_plan_submission_wave(
                Arc::new(step.work_shape().clone()),
                AdmissionFitPolicy::ImmediateOnly,
                AdmissionPressureAction::WaitForRelease,
            )
            .unwrap()
        {
            StepSubmissionWaveAdmissionDecision::Prepared(wave) => break wave,
            StepSubmissionWaveAdmissionDecision::BackingDeferred(deferred) => {
                require_progress(deferred.maintain().unwrap())
            }
            StepSubmissionWaveAdmissionDecision::Deferred(reason) => require_progress(
                fixture
                    .resources
                    .maintain_for_admission_deferred(&reason)
                    .unwrap(),
            ),
            _ => panic!("fixture wave unavailable"),
        }
    };
    let executable = fixture.compilation.executable();
    let output_node = executable
        .execution_plan()
        .payload()
        .nodes()
        .iter()
        .find(|node| node.id().as_str() == output_node_id)
        .unwrap();
    let output = output_node
        .values()
        .iter()
        .find(|value| value.role() == ResolvedValueRole::Output && value.ordinal() == 0)
        .unwrap();
    let component = &output.storage().components()[0];
    let readbacks = CompletionReadbackBatchRequest::new(vec![CompletionReadbackRequest::new(
        output_node.id().clone(),
        0,
        component.resource_id().clone(),
        component.offset_bytes(),
        HostTransferLayout::new(fixture.output_type, range.len() as u64 * HIDDEN).unwrap(),
    )
    .unwrap()])
    .unwrap();
    let wave = wave.with_submission_readbacks(readbacks).unwrap();
    let active = TrustedActiveSequenceBinding::from_session(session).unwrap();
    let identity = OperationDispatch::bind_submission_wave_identity(
        executable,
        std::iter::once(&active),
        &wave,
        &fixture.lane,
    )
    .unwrap();
    let upload = SubmissionWaveInputUpload::new(
        id("node.embedding"),
        0,
        0,
        range.start as u64 * ElementType::U32.size_bytes(),
        HostTransferLayout::new(ElementType::U32, range.len() as u64).unwrap(),
        tokens[range]
            .iter()
            .flat_map(|token| token.to_le_bytes())
            .collect(),
    )
    .unwrap();
    // Resolve the current wave against the real warmed catalog. An exact
    // guard cannot authorize cold on-demand graph capture; it must not mutate
    // cache state before returning a definite NotSubmitted receipt.
    let catalog = resident.then(|| fixture.lane.reusable_execution_catalog().unwrap());
    let program = catalog.as_ref().map(|catalog| {
        let program_id = OperationDispatch::reusable_execution_program_id_for_wave(
            fixture.providers.providers(),
            executable,
            &wave,
            &fixture.lane,
        )
        .unwrap()
        .expect("resident guard requires a current reusable topology");
        let program = catalog
            .programs()
            .iter()
            .find(|program| program.program_id() == &program_id)
            .expect("resident guard must match the current wave, without eager fallback");
        assert!(program.is_determinism_ready());
        program
    });
    let guard = ExpiredAfterEncoding {
        calls: AtomicU64::new(0),
    };
    let result = OperationDispatch::encode_and_submit_guarded_wave_with_cost_evidence_demand(
        fixture.providers.providers(),
        executable,
        &identity,
        std::iter::once(&active),
        &[upload],
        program,
        &guard,
        ferrum_interfaces::vnext::DeviceCostObservationDemand::Required,
        demand,
        &ActualSampleGateNoTiming,
        wave,
        &fixture.lane,
        &fixture.reaper,
    );
    let rejection = match result {
        GuardedWaveSubmissionOutcome::NotSubmitted(rejection) => rejection,
        GuardedWaveSubmissionOutcome::Dispatch(Err(error)) => {
            panic!("encoded guard was not reached: {error}")
        }
        GuardedWaveSubmissionOutcome::Dispatch(Ok(_)) => {
            panic!("expired guard unexpectedly submitted the wave")
        }
    };
    assert_eq!(
        guard.calls.load(Ordering::Relaxed),
        1,
        "guarded wave cannot retry or reject before the requested host check: {rejection:?}"
    );
    assert_eq!(
        fixture.reaper.retained_count(),
        0,
        "no fence/reaper record may remain after rejection"
    );
    assert_eq!(
        fixture.lane.cost_readback_available_bytes(),
        Some(1 << 20),
        "failed wave must release staged readback bytes"
    );
    (rejection, step)
}

#[test]
fn guarded_native_wave_rejection_rolls_back_and_same_request_still_completes() {
    let fixture = Fixture::new(AttentionKind::GatedDelta);
    fixture
        .lane
        .configure_submission_readback_staging(1 << 20)
        .unwrap();
    let tokens: Arc<[u32]> = Arc::from([3, 7]);
    let baseline_session = fixture.admit("guard-baseline", Arc::clone(&tokens));
    let baseline = fixture.execute(&baseline_session, Arc::clone(&tokens), 0..tokens.len());
    baseline_session.try_abort_if_quiescent().unwrap();
    let session = fixture.admit("guard-rejected", Arc::clone(&tokens));
    let (pending, step) = reject_encoded(&fixture, &session, Arc::clone(&tokens));
    let receipt = pending
        .reconcile_step(step)
        .unwrap_or_else(|(error, _step)| {
            panic!("guard rejection must reconcile its own Step: {error}")
        });
    assert_eq!(
        receipt.reason(),
        GuardedNotSubmittedReason::HostRejected(HostSubmissionRejection::WitnessExpired)
    );
    let completed = fixture.execute(&session, tokens, 0..2);
    completed.assert_same(
        &baseline,
        "native guard rejection must not update recurrent state",
    );
    session.try_abort_if_quiescent().unwrap();
}

#[test]
fn guarded_native_rejection_receipt_cannot_reconcile_another_step() {
    let fixture = Fixture::new(AttentionKind::GatedDelta);
    fixture
        .lane
        .configure_submission_readback_staging(1 << 20)
        .unwrap();
    let tokens: Arc<[u32]> = Arc::from([2]);
    let session = fixture.admit("guard-original", Arc::clone(&tokens));
    let (pending, original_step) = reject_encoded(&fixture, &session, Arc::clone(&tokens));
    let other = Fixture::new(AttentionKind::GatedDelta);
    let other_session = other.admit("guard-other", Arc::clone(&tokens));
    let batch = ExecutionBatchParticipants::new(vec![Arc::clone(&other_session)]).unwrap();
    let request = StepResourceAdmissionRequest::new(
        batch
            .bind_work_shape(vec![token_span(Arc::clone(&tokens), 0..1)])
            .unwrap(),
        AdmissionFitPolicy::ImmediateOnly,
        AdmissionPressureAction::WaitForRelease,
    )
    .unwrap();
    let wrong_step = loop {
        match batch.try_begin_step(request.clone(), &other.lane).unwrap() {
            StepResourceAdmissionDecision::Admitted(step) => break step,
            StepResourceAdmissionDecision::BackingDeferred(deferred) => {
                require_progress(deferred.maintain().unwrap())
            }
            StepResourceAdmissionDecision::Deferred(reason) => require_progress(
                other
                    .resources
                    .maintain_for_admission_deferred(&reason)
                    .unwrap(),
            ),
            _ => panic!("second fixture Step unavailable"),
        }
    };
    let (error, wrong_step) = pending.reconcile_step(wrong_step).unwrap_err();
    assert!(error.to_string().contains("another Step"));
    wrong_step.try_rollback_unsubmitted().unwrap();
    original_step.try_rollback_unsubmitted().unwrap();
    session.try_abort_if_quiescent().unwrap();
    other_session.try_abort_if_quiescent().unwrap();
}
