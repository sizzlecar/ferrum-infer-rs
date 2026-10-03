//! CompleteRequests enters cold capture through a separate one-call gate.
//! A cost witness is never inferred from this fallback execution.
use super::*;
use ferrum_interfaces::execution_cost::{
    host_history_cost_signature, ActualRowWork, ActualWaveGraphState, ActualWaveKind,
    ActualWavePath, ActualWaveRowOrder, CanonicalCostError, CanonicalCostRow,
    CanonicalWaveCostBuilder, CostPhysicalCommand, CostProductOutput, CostRowOutput,
};

fn captured_original_command_structure(
    f: &Fixture,
    actual: &DeviceSubmissionAttribution,
) -> Result<(), CanonicalCostError> {
    let direct_operation = f
        .runtime
        .cost_direct_graph_replay_operation()
        .and_then(DeviceNativeOperationId::new)
        .unwrap();
    assert!(actual.replayed_segments().is_empty());
    assert_eq!(actual.graph_evidence().unwrap().replayed_segments(), 1);
    let mut original_replays = 0;
    let mut canonical = CanonicalWaveCostBuilder::new_exact(0, CostProductOutput::FullLogits);
    for command in actual.commands() {
        if command.execution_path() == DeviceExecutionPath::Replayed {
            original_replays += 1;
            assert_ne!(command.native_op_id(), direct_operation.as_str());
            assert_eq!(command.command_phase(), DeviceCommandPhase::Compute);
            assert!(command.node_index().is_some());
            assert_eq!(command.participant_count(), 1);
            assert_eq!(command.token_count(), 1);
        }
        let provider = command
            .node_index()
            .map(|index| f.cost_provider_at(index).expect("original bound provider"));
        let projected = CostPhysicalCommand::from_attribution(command, provider);
        if command.execution_path() == DeviceExecutionPath::Replayed {
            canonical.original_replay_command(projected, direct_operation)?;
        } else {
            canonical.physical_command(projected)?;
        }
    }
    assert!(
        original_replays > 0,
        "the actual cold attempt launched its graph"
    );
    // The same readback route was checked by CompleteGuard before submission.
    canonical.core_readback_route(CoreReadbackRoute::SubmissionStaged)?;
    let work = span();
    let range = work.immediate_token_range();
    // A declared projection row for this one-token, full-output kernel fixture.
    // There is no sampler/host-settlement authority here and no source8 proof
    // is minted. The actual commands and provider identities above are retained.
    canonical.row(CanonicalCostRow {
        work: ActualRowWork::Prefill {
            offset: u32::try_from(range.start).unwrap(),
            count: u32::try_from(
                range
                    .end
                    .checked_sub(range.start)
                    .expect("ordered admitted token span"),
            )
            .unwrap(),
            total_prompt_tokens: u32::try_from(work.full_input_tokens()).unwrap(),
        },
        host_policy_signature: host_history_cost_signature(
            Sha256::digest(b"cuda-guard-fixture-full-output-projection").into(),
            0,
        ),
        host_features: None,
        mask_upload_required: false,
        output: CostRowOutput::Prefill {
            final_logits: range.end == work.full_input_tokens(),
        },
    })?;
    canonical.validate_physical_structure(ActualWaveKind::Prefill)?;
    assert_eq!(
        canonical.finish(
            ActualWaveKind::Prefill,
            ActualWavePath::PlanRuntime,
            ActualWaveGraphState::Warm,
            ActualWaveRowOrder::Ordered,
            0,
        ),
        Err(CanonicalCostError::InvalidRoute),
        "physical settlement cannot invent numerical graph eligibility"
    );
    Ok(())
}

struct CompleteGuard {
    reject: bool,
    before_enqueues: u64,
    enqueues: Arc<AtomicU64>,
    calls: AtomicU64,
}
impl CompleteGuard {
    fn new(f: &Fixture, reject: bool) -> Self {
        Self {
            reject,
            before_enqueues: f.enqueues.load(Ordering::Relaxed),
            enqueues: Arc::clone(&f.enqueues),
            calls: AtomicU64::new(0),
        }
    }
}
impl PreparedWaveSubmissionGuard for CompleteGuard {
    fn submission_mode(&self) -> GuardedSubmissionMode {
        GuardedSubmissionMode::CompleteRequestsAdaptive
    }
    fn check_adaptive_preparation(
        &self,
        intent: &DeviceAdaptiveSubmissionIntent<'_>,
        readback: CoreReadbackRoute,
    ) -> Result<(), GuardedNotSubmittedReason> {
        assert_eq!(
            self.calls.fetch_add(1, Ordering::Relaxed),
            0,
            "one-use call, no post-capture recheck"
        );
        assert_eq!(
            self.enqueues.load(Ordering::Relaxed),
            self.before_enqueues,
            "capture callbacks have not run before authorization"
        );
        assert_eq!(readback, CoreReadbackRoute::SubmissionStaged);
        assert_eq!(
            intent.graph_before().configuration(),
            DeviceCostGraphConfiguration::OnDemand
        );
        assert!(intent.encoded_work().graph_evidence().is_none());
        assert!(intent.encoded_work().replayed_segments().is_empty());
        assert!(
            intent.capture().is_some(),
            "real compiled program identity is retained"
        );
        if self.reject {
            Err(GuardedNotSubmittedReason::HostRejected(
                HostSubmissionRejection::WitnessExpired,
            ))
        } else {
            Ok(())
        }
    }
    fn check(
        &self,
        _: &DeviceSubmissionAttribution,
        _: CoreReadbackRoute,
    ) -> Result<(), GuardedNotSubmittedReason> {
        panic!("cold intent must not be labelled complete actual eager/warm attribution")
    }
}

#[test]
#[ignore = "requires an actual CUDA device and installed native operator artifacts"]
fn guarded_cuda_cold_complete_rejects_without_preparing_then_retries_and_becomes_warm() {
    let f = Fixture::new_graph();
    let before_catalog = f.lane.reusable_execution_catalog().unwrap().into_parts();
    let before_preparation = f.lane.reusable_executable_preparation().unwrap();
    let (step, wave) = f.prepare();
    let guard = CompleteGuard::new(&f, true);
    let pending = match f.dispatch_program(wave, Some(&guard), None) {
        GuardedWaveSubmissionOutcome::NotSubmitted(value) => value,
        _ => panic!("cold authorization rejection must precede all preparation"),
    };
    assert_eq!(guard.calls.load(Ordering::Relaxed), 1);
    assert_eq!(f.enqueues.load(Ordering::Relaxed), 0);
    assert_eq!(f.lane.in_flight_count(), 0);
    assert_eq!(f.reaper.retained_count(), 0);
    assert_eq!(
        before_catalog,
        f.lane.reusable_execution_catalog().unwrap().into_parts()
    );
    assert_eq!(
        before_preparation,
        f.lane.reusable_executable_preparation().unwrap()
    );
    pending
        .reconcile_step(step)
        .unwrap_or_else(|(e, _)| panic!("cold rollback: {e}"));

    // Two real submitted attempts: first warmup, second capture+upload+execute.
    // First retry must still be cold: the rejected attempt did not advance the cache.
    let mut captured_structure = None;
    for index in 0..2 {
        let (step, wave) = f.prepare();
        let guard = CompleteGuard::new(&f, false);
        let (handle, actual) =
            warm_graph::dispatched(f.dispatch_program(wave, Some(&guard), None)).into_parts();
        assert_eq!(guard.calls.load(Ordering::Relaxed), 1);
        let evidence = actual.as_ref().unwrap().device().graph_evidence().unwrap();
        assert!(!evidence.proves_warm_direct_replay());
        assert!(!evidence.proves_unconfigured_eager());
        warm_graph::finish(&f, handle); // state doubles exactly once, capture never executes it twice
        step.try_retire_normal().unwrap();
        let catalog = f.lane.reusable_execution_catalog().unwrap();
        if index == 0 {
            // OnDemand removes empty program tombstones: warmup alone must
            // not advertise a resident program or authorize warm replay.
            assert!(catalog.programs().is_empty());
        } else {
            assert_eq!(catalog.programs().len(), 1);
            assert!(catalog.programs()[0].has_resident_segments());
            captured_structure = Some(captured_original_command_structure(
                &f,
                actual.as_ref().unwrap().device(),
            ));
        }
    }
    let catalog = f.lane.reusable_execution_catalog().unwrap();
    let program = catalog.programs()[0].clone();
    assert!(program.is_determinism_ready());
    let (step, wave) = f.prepare();
    let guard = warm_graph::WarmGuard::new(&program, false);
    let (handle, actual) =
        warm_graph::dispatched(f.dispatch_program(wave, Some(&guard), Some(&program))).into_parts();
    assert!(actual
        .as_ref()
        .unwrap()
        .device()
        .graph_evidence()
        .unwrap()
        .proves_warm_direct_replay());
    warm_graph::finish(&f, handle);
    step.try_retire_normal().unwrap();
    f.close(true);
    assert_eq!(
        captured_structure,
        Some(Ok(())),
        "actual capture/replay retained complete original commands; it was not a compact invocation"
    );
}

#[test]
#[ignore = "requires an actual CUDA device and installed native operator artifacts"]
fn guarded_cuda_cold_complete_drop_keeps_owned_submission_until_fence() {
    let f = Fixture::new_graph();
    let (step, wave) = f.prepare();
    let guard = CompleteGuard::new(&f, false);
    let (handle, _) =
        warm_graph::dispatched(f.dispatch_program(wave, Some(&guard), None)).into_parts();
    assert_eq!(guard.calls.load(Ordering::Relaxed), 1);
    let step = step
        .try_rollback_unsubmitted()
        .expect_err("authorized cold attempt was submitted")
        .into_step();
    drop(handle);
    assert_eq!(f.reaper.retained_count(), 1);
    assert_eq!(f.lane.in_flight_count(), 1);
    assert!(f.session.try_abort_if_quiescent().is_err());
    f.runtime.context.synchronize().unwrap();
    let _ = f.reaper.poll_bounded(1).unwrap();
    assert_eq!(f.lane.in_flight_count(), 0);
    assert_eq!(f.reaper.retained_count(), 0);
    step.try_retire_normal().unwrap();
    f.close(true);
}
