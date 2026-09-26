//! Real core admission/encoding and the existing on-demand CUDA graph cache.
//! Cold warmup/capture is deliberately performed by the ordinary path here;
//! this tests the warm guard, not the later cold-maintenance product bridge.
use super::*;

pub(super) fn dispatched(
    outcome: GuardedWaveSubmissionOutcome<CudaDeviceRuntime>,
) -> ProfiledSubmissionHandle<CudaDeviceRuntime> {
    match outcome {
        GuardedWaveSubmissionOutcome::Dispatch(Ok(value)) => value,
        GuardedWaveSubmissionOutcome::Dispatch(Err(error)) => panic!("dispatch failed: {error}"),
        GuardedWaveSubmissionOutcome::NotSubmitted(_) => panic!("unexpected guard rejection"),
    }
}

pub(super) fn finish(f: &Fixture, handle: CompletionHandle<CudaDeviceRuntime>) {
    let receipt = match handle.wait_with_readbacks(f.readback()).unwrap() {
        CompletionReadbackBatchObservation::Terminal(value) => value,
        _ => panic!("real CUDA fence did not become terminal"),
    };
    assert!(matches!(
        receipt.completion().submission_timing(),
        DeviceTimingMeasurement::NotRequested
    ));
    let CompletionReadbackDisposition::Succeeded(output) = &receipt.dispositions()[0] else {
        panic!("scale output failed")
    };
    let expected = [2.0_f32, -4.0, 1.0, 8.0]
        .into_iter()
        .flat_map(|v| f16::from_f32(v).to_le_bytes())
        .collect::<Vec<_>>();
    assert_eq!(output.bytes(), expected);
    drop((receipt, handle));
}

fn warmed(f: &Fixture) -> DeviceReusableExecutionProgram {
    // OnDemand's documented two real submissions: eager warmup, then
    // capture/upload plus execution. No synthetic cache or capture descriptor.
    for _ in 0..2 {
        let (step, wave) = f.prepare();
        let (handle, _) = dispatched(f.dispatch_program(wave, None, None)).into_parts();
        finish(f, handle);
        step.try_retire_normal().unwrap();
    }
    let catalog = f.lane.reusable_execution_catalog().unwrap();
    assert_eq!(catalog.programs().len(), 1);
    let program = catalog.programs()[0].clone();
    assert_eq!(program.segments().len(), 1);
    assert!(program.gaps().is_empty());
    assert_eq!(program.program_id().lane_id(), f.lane.id());
    program
}

pub(super) struct WarmGuard {
    reject: bool,
    program: DeviceReusableExecutionProgram,
    actual: Mutex<Option<DeviceSubmissionAttribution>>,
}
impl WarmGuard {
    pub(super) fn new(program: &DeviceReusableExecutionProgram, reject: bool) -> Self {
        Self {
            reject,
            program: program.clone(),
            actual: Mutex::new(None),
        }
    }
}
impl PreparedWaveSubmissionGuard for WarmGuard {
    fn check(
        &self,
        actual: &DeviceSubmissionAttribution,
        readback: CoreReadbackRoute,
    ) -> Result<(), GuardedNotSubmittedReason> {
        assert_eq!(readback, CoreReadbackRoute::SubmissionStaged);
        assert!(actual.graph_evidence().unwrap().proves_warm_direct_replay());
        assert!(!actual.graph_evidence().unwrap().proves_unconfigured_eager());
        assert_eq!(actual.replayed_segments().len(), 1);
        let replay = &actual.replayed_segments()[0];
        assert_eq!(replay.program_id(), self.program.program_id());
        assert_eq!(replay.segment(), &self.program.segments()[0]);
        let [logical] = replay.logical_commands() else {
            panic!("one real captured node")
        };
        assert_eq!(logical.native_op_id(), "cuda_guard_fixture_scale");
        assert_eq!(logical.node_index(), 0);
        assert_eq!(logical.participant_count(), 1);
        assert_eq!(logical.token_count(), 1);
        assert_eq!(logical.compute_dispatch_count(), 1);
        assert_eq!(logical.transfer_command_count(), 0);
        assert_eq!(logical.reusable_graph_node_count(), 1);
        let mut slot = self.actual.lock().unwrap();
        assert!(slot.is_none(), "final host guard is called once");
        *slot = Some(actual.clone());
        if self.reject {
            Err(GuardedNotSubmittedReason::HostRejected(
                HostSubmissionRejection::WitnessExpired,
            ))
        } else {
            Ok(())
        }
    }
}

#[test]
#[ignore = "requires an actual CUDA device and installed native operator artifacts"]
fn guarded_cuda_warm_direct_replay_rejects_before_execution_then_retries_same_owner() {
    warm_rejected_then_retry(&NoHostTiming);
}

#[test]
#[ignore = "requires an actual CUDA device and installed native operator artifacts"]
fn guarded_cuda_host_timing_preserves_warm_graph_rejection_and_exact_output() {
    let timing = HostTiming::default();
    warm_rejected_then_retry(&timing);
    assert!(timing.backend_stages.load(Ordering::Relaxed) > 0);
    assert_eq!(timing.completion_arms.load(Ordering::Relaxed), 1);
}

fn warm_rejected_then_retry<S: SubmissionWaveDispatchTimingSink>(timing: &S) {
    warm_rejected_then_retry_fixture(Fixture::new_graph(), timing, false);
}

#[test]
#[ignore = "requires an actual CUDA device and installed native operator artifacts"]
fn guarded_cuda_structured_replay_binds_actual_capture_and_current_work_without_reencoding() {
    warm_rejected_then_retry_fixture(
        Fixture::new_graph_with_structured_capture(),
        &NoHostTiming,
        true,
    );
}

fn assert_selected_replay(attribution: &DeviceSubmissionAttribution, expected: bool) {
    let logical = &attribution.replayed_segments()[0].logical_commands()[0];
    let selected = logical.statistical_evidence();
    assert_eq!(selected.is_some(), expected);
    if let Some(selected) = selected {
        selected.validate_command(1, 1, 0).unwrap();
        assert_eq!(selected.work().logical_units, 4);
        assert_eq!(selected.work().padded_units, 32);
        assert_eq!(selected.work().inner_work_units, 4);
        selected
            .algorithm_work()
            .unwrap()
            .unwrap()
            .validate_command(selected)
            .unwrap();
    }
}

fn warm_rejected_then_retry_fixture<S: SubmissionWaveDispatchTimingSink>(
    f: Fixture,
    timing: &S,
    structured: bool,
) {
    let program = warmed(&f);
    let before_output = f.output_bytes();
    let before_preparation = f.lane.reusable_executable_preparation().unwrap();
    let before_catalog = f.lane.reusable_execution_catalog().unwrap().into_parts();
    let encoded = f.encoded.load(Ordering::Relaxed);
    let enqueues = f.enqueues.load(Ordering::Relaxed);
    let (step, wave) = f.prepare();
    let guard = WarmGuard::new(&program, true);
    let pending = match f.dispatch_program_with_timing(wave, Some(&guard), Some(&program), timing) {
        GuardedWaveSubmissionOutcome::NotSubmitted(value) => value,
        _ => panic!("warm final guard must reject before enqueue or graph launch"),
    };
    assert!(guard.actual.lock().unwrap().is_some());
    assert_selected_replay(guard.actual.lock().unwrap().as_ref().unwrap(), structured);
    assert_eq!(
        f.output_bytes(),
        before_output,
        "rejection cannot execute the stateful graph"
    );
    assert_eq!(
        f.encoded.load(Ordering::Relaxed),
        encoded,
        "reuse original graph, no provider re-encode"
    );
    assert_eq!(
        f.enqueues.load(Ordering::Relaxed),
        enqueues,
        "no eager/capture callback on warm rejection"
    );
    assert_eq!(f.lane.in_flight_count(), 0);
    assert_eq!(f.reaper.retained_count(), 0);
    assert_eq!(f.lane.cost_readback_available_bytes(), Some(8));
    let receipt = pending
        .reconcile_step(step)
        .unwrap_or_else(|(error, _)| panic!("exact warm rollback: {error}"));
    assert_eq!(
        receipt.reason(),
        GuardedNotSubmittedReason::HostRejected(HostSubmissionRejection::WitnessExpired)
    );
    assert_eq!(
        f.lane.reusable_executable_preparation().unwrap(),
        before_preparation
    );
    assert_eq!(
        f.lane.reusable_execution_catalog().unwrap().into_parts(),
        before_catalog
    );

    // Same request/session and lane, newly admitted Step: no receipt reuse.
    let (step, wave) = f.prepare();
    let guard = WarmGuard::new(&program, false);
    let (handle, attribution) =
        dispatched(f.dispatch_program_with_timing(wave, Some(&guard), Some(&program), timing))
            .into_parts();
    let attribution = attribution.unwrap();
    assert_selected_replay(attribution.device(), structured);
    assert_eq!(
        attribution.device(),
        guard.actual.lock().unwrap().as_ref().unwrap()
    );
    assert_selected_replay(guard.actual.lock().unwrap().as_ref().unwrap(), structured);
    assert_eq!(f.encoded.load(Ordering::Relaxed), encoded);
    assert_eq!(f.enqueues.load(Ordering::Relaxed), enqueues);
    finish(&f, handle);
    step.try_retire_normal().unwrap();
    f.close(true);
}

#[test]
#[ignore = "requires an actual CUDA device and installed native operator artifacts"]
fn guarded_cuda_warm_replay_drop_retains_resources_until_actual_fence() {
    let f = Fixture::new_graph();
    let program = warmed(&f);
    let (step, wave) = f.prepare();
    let guard = WarmGuard::new(&program, false);
    let (handle, _) =
        dispatched(f.dispatch_program(wave, Some(&guard), Some(&program))).into_parts();
    let step = step
        .try_rollback_unsubmitted()
        .expect_err("accepted graph is genuinely submitted")
        .into_step();
    drop(handle);
    assert_eq!(f.reaper.retained_count(), 1);
    assert_eq!(f.lane.in_flight_count(), 1);
    assert!(f.session.try_abort_if_quiescent().is_err());
    f.runtime.context.synchronize().unwrap();
    f.reaper.poll_bounded(1).unwrap();
    assert_eq!(f.lane.in_flight_count(), 0);
    assert_eq!(f.reaper.retained_count(), 0);
    step.try_retire_normal().unwrap();
    f.close(true);
}
