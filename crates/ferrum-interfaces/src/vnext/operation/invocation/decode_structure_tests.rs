use super::*;
use crate::vnext::{
    DeviceSubmissionStage, DeviceSubmissionTimingSink, SubmissionWaveDispatchStage,
    SubmissionWaveStructureObservation,
};
use std::sync::OnceLock;
use std::time::Duration;

#[derive(Default)]
struct Observe<const ENABLED: bool> {
    value: OnceLock<SubmissionWaveStructureObservation>,
}
impl<const ENABLED: bool> DeviceSubmissionTimingSink for Observe<ENABLED> {
    const ENABLED: bool = ENABLED;
    fn record_device_submission(&self, _: DeviceSubmissionStage, _: Duration) {
        assert!(ENABLED, "Off must not call the observer");
    }
}
impl<const ENABLED: bool> SubmissionWaveDispatchTimingSink for Observe<ENABLED> {
    fn record(&self, _: SubmissionWaveDispatchStage, _: Duration) {
        assert!(ENABLED);
    }
    fn wants_submission_wave_structure(&self) -> bool {
        assert!(ENABLED);
        true
    }
    fn record_submission_wave_structure(&self, value: SubmissionWaveStructureObservation) {
        assert!(ENABLED);
        assert!(
            self.value.set(value).is_ok(),
            "one observation per dispatch"
        );
    }
}

fn submit<const ENABLED: bool>(
    fixture: &Fixture,
    step: &Arc<StepResourceLease<TestRuntime>>,
    session: &Arc<SequenceSession<TestRuntime>>,
    sink: &Observe<ENABLED>,
) -> ExecutionFrameId {
    let wave = prepare_wave(&fixture.plan_resources, &fixture.plan, step);
    let frame = wave.nodes()[0].participant_frames()[0].frame_id();
    let active = vnext_device_operation_wave_contract::wave_active_bindings(&wave, session);
    let providers = fixture.registry.bind_plan(&fixture.resolved).unwrap();
    let lane = step.execution_lane();
    let identity = OperationDispatch::bind_submission_wave_identity(
        &fixture.resolved,
        active.iter(),
        &wave,
        lane,
    )
    .unwrap();
    let reaper = CompletionReaper::new();
    let (completion, _) = OperationDispatch::encode_and_submit_wave_with_inputs_and_timing(
        providers.providers(),
        &fixture.resolved,
        &identity,
        active.iter(),
        DeviceTimingMode::Off,
        &[],
        SubmissionExecutionPolicy::adaptive(),
        sink,
        wave,
        lane,
        &reaper,
    )
    .unwrap()
    .into_parts();
    assert!(matches!(
        completion.poll().unwrap(),
        CompletionObservation::Terminal(_)
    ));
    drop(completion);
    assert_eq!(reaper.retained_count(), 0);
    frame
}

#[test]
fn decode_structure_real_fresh_frames_keep_claim_structure_without_retaining_resources() {
    use vnext_device_operation_wave_contract::{setup, teardown};
    let (fixture, sequence, session, batch, step) = setup();
    let lane = Arc::clone(step.execution_lane());
    let first = Observe::<true>::default();
    let first_frame = submit(&fixture, &step, &session, &first);
    step.try_retire_normal().unwrap();
    let next_step = step_for(
        &batch,
        &lane,
        vec![TokenSpanWork::from_token_ids(&[1, 2], 1..2).unwrap()],
    );
    let next = Observe::<true>::default();
    let next_frame = submit(&fixture, &next_step, &session, &next);
    assert_ne!(first_frame, next_frame);
    let mask = next
        .value
        .get()
        .unwrap()
        .change_mask(first.value.get().unwrap());
    assert_eq!(
        mask, 0,
        "fresh frame/position/claim issuance are not physical structure"
    );
    drop(lane);
    // Both snapshots remain alive while the actual plan closes.
    teardown(fixture, sequence, session, batch, next_step);
    assert!(first.value.get().is_some() && next.value.get().is_some());
}

#[test]
fn decode_structure_disabled_dispatch_never_queries_or_calls_observer() {
    use vnext_device_operation_wave_contract::{setup, teardown};
    let (fixture, sequence, session, batch, step) = setup();
    let sink = Observe::<false>::default();
    submit(&fixture, &step, &session, &sink);
    assert!(sink.value.get().is_none());
    teardown(fixture, sequence, session, batch, step);
}

#[test]
fn decode_structure_real_membership_and_width_changes_are_joint() {
    let mut prior = None;
    with_live_wave(1, |_, wave, _, _| {
        prior = Some(wave.structure_observation(None, 1, false));
    });
    with_live_wave(2, |_, wave, _, _| {
        let current = wave.structure_observation(None, 1, false);
        let mask = current.change_mask(prior.as_ref().unwrap());
        assert_ne!(mask & SubmissionWaveStructureObservation::DIMENSIONS, 0);
        assert_ne!(mask & SubmissionWaveStructureObservation::PARTICIPANTS, 0);
        assert_ne!(
            current.change_mask(&wave.structure_observation(None, 2, false))
                & SubmissionWaveStructureObservation::PROGRAM_OR_LANE,
            0
        );
    });
}
