use super::*;

struct CostGuard {
    cost: bool,
    inner: Guard,
}
impl PreparedWaveSubmissionGuard for CostGuard {
    fn relies_on_cost_witness(&self) -> bool {
        self.cost
    }
    fn check(
        &self,
        actual: &DeviceSubmissionAttribution,
        readback: CoreReadbackRoute,
    ) -> Result<(), GuardedNotSubmittedReason> {
        self.inner.check(actual, readback)
    }
}

#[test]
#[ignore = "requires an actual CUDA device and installed native operator artifacts"]
fn guarded_cuda_missing_library_contract_rejects_cost_but_preserves_completion_retry() {
    let f = Fixture::new_missing_library_contract();
    let (step, wave) = f.prepare();
    let guard = CostGuard {
        cost: true,
        inner: Guard::new(&f, false),
    };
    let pending = match f.dispatch(wave, &guard) {
        GuardedWaveSubmissionOutcome::NotSubmitted(value) => value,
        _ => panic!("missing required library evidence cannot authorize a cost witness"),
    };
    assert_eq!(f.enqueues.load(Ordering::Relaxed), 0);
    assert_eq!(guard.inner.calls.load(Ordering::Relaxed), 0);
    assert_eq!(f.lane.in_flight_count(), 0);
    assert_eq!(f.reaper.retained_count(), 0);
    let receipt = pending
        .reconcile_step(step)
        .unwrap_or_else(|(e, _)| panic!("exact rollback: {e}"));
    assert_eq!(
        receipt.reason(),
        GuardedNotSubmittedReason::AttributionUnavailable
    );

    // The same declared missing evidence cannot reject ordinary completion.
    // Original owner/Step guards, output, fence and readback remain mandatory.
    let (step, wave) = f.prepare();
    let guard = CostGuard {
        cost: false,
        inner: Guard::new(&f, false),
    };
    let handle = accepted(f.dispatch(wave, &guard));
    let receipt = match handle.wait_with_readbacks(f.readback()).unwrap() {
        CompletionReadbackBatchObservation::Terminal(value) => value,
        _ => panic!("completion readback did not terminate"),
    };
    assert!(matches!(
        receipt.completion().submission_timing(),
        DeviceTimingMeasurement::NotRequested
    ));
    let CompletionReadbackDisposition::Succeeded(output) = &receipt.dispositions()[0] else {
        panic!("real scale output failed")
    };
    let expected = [2.0_f32, -4.0, 1.0, 8.0]
        .into_iter()
        .flat_map(|x| f16::from_f32(x).to_le_bytes())
        .collect::<Vec<_>>();
    assert_eq!(output.bytes(), expected);
    drop((receipt, handle));
    step.try_retire_normal().unwrap();
    f.close(true);
}
