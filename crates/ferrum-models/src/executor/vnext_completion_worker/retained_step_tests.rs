use super::{VNextCompletionTaskKind, VNextCompletionWorker};

#[path = "../../../../ferrum-interfaces/tests/vnext_device_operation_contract/mod.rs"]
mod vnext_device_operation_contract;
#[path = "../../../../ferrum-interfaces/tests/vnext_device_operation_wave_contract/mod.rs"]
mod vnext_device_operation_wave_contract;

use vnext_device_operation_contract::*;
use vnext_device_operation_wave_contract::*;

/// Reports release only after the real Step lease has been dropped. An
/// abandoned result deliberately does not commit that Step's logical frontier.
struct AbandonedOutput {
    step: Option<Arc<StepResourceLease<TestRuntime>>>,
    released: Option<tokio::sync::oneshot::Sender<()>>,
}

impl Drop for AbandonedOutput {
    fn drop(&mut self) {
        drop(self.step.take());
        let _ = self.released.take().unwrap().send(());
    }
}

#[tokio::test]
async fn abandoned_ticket_retains_real_step_and_reaper_until_fence_terminal() {
    let (fixture, sequence, session, batch, step) = setup();
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, &step);
    let active = wave_active_bindings(&wave, &session);
    let lane = Arc::clone(step.execution_lane());
    let providers = fixture.registry.bind_plan(&fixture.resolved).unwrap();
    let identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active.iter(),
        &wave,
        &lane,
    )
    .unwrap();
    let reaper = CompletionReaper::new();
    let worker = VNextCompletionWorker::new().unwrap();
    let reservation = worker.reserve().await.unwrap();
    let completion = OperationDispatch::encode_and_submit_wave_with_inputs(
        providers.providers(),
        &fixture.resolved,
        &identity,
        active.iter(),
        DeviceTimingMode::Off,
        &[],
        wave,
        &lane,
        &reaper,
    )
    .unwrap();
    let entered = Arc::new(std::sync::Barrier::new(2));
    let release = Arc::new(std::sync::Barrier::new(2));
    fixture.runtime_trace.lock().unwrap().wait_fence_block =
        Some((Arc::clone(&entered), Arc::clone(&release)));
    let weak_step = Arc::downgrade(&step);
    let weak_reaper = Arc::downgrade(&reaper);
    let retained_reaper = Arc::clone(&reaper);
    let (released_sender, released_receiver) = tokio::sync::oneshot::channel();
    let ticket = reservation.submit(VNextCompletionTaskKind::WaveReadback, move || {
        assert!(matches!(
            completion.wait().unwrap(),
            CompletionObservation::Terminal(_)
        ));
        drop(retained_reaper);
        AbandonedOutput {
            step: Some(step),
            released: Some(released_sender),
        }
    });
    entered.wait();
    assert!(weak_step.upgrade().unwrap().try_retire_normal().is_err());
    drop(ticket);
    drop(worker);
    drop(reaper);
    assert!(weak_step.upgrade().is_some());
    assert_eq!(weak_reaper.upgrade().unwrap().retained_count(), 1);
    assert_eq!(lane.in_flight_count(), 1);

    release.wait();
    released_receiver.await.unwrap();
    assert!(weak_step.upgrade().is_none());
    assert!(weak_reaper.upgrade().is_none());
    assert_eq!(lane.in_flight_count(), 0);
    // Discarding an uncommitted Step intentionally leaves its frame poisoned:
    // a terminal fence proves memory safety, not a committed token frontier.
    // Releasing the sequence owners below must still release plan capacity.
    assert!(sequence.is_poisoned());
    assert!(session.try_complete().is_err());
    assert!(session.try_abort().is_err());
    assert!(sequence.open_session().is_err());
    drop(active);
    drop(providers);
    drop(batch);
    drop(session);
    drop(sequence);
    drop(lane);
    drop(fixture.registry);
    drop(fixture.impostor_registry);
    drop(fixture.runtime);
    assert!(matches!(
        PlanRuntimeResources::close(fixture.plan_resources),
        Ok(PlanRuntimeCloseOutcome::Closed(_))
    ));
}
