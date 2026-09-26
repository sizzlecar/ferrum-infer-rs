//! Opt-in assertion on a real resident graph launch in the existing fixture.
use super::*;
use ferrum_interfaces::execution_cost::SelectedReplayAlgorithmTemplateV1;
use std::num::NonZeroU64;

struct NoTiming;
impl DeviceSubmissionTimingSink for NoTiming {
    const ENABLED: bool = false;
    fn record_device_submission(&self, _: DeviceSubmissionStage, _: std::time::Duration) {
        panic!("selected replay must not enable timing")
    }
}
impl SubmissionWaveDispatchTimingSink for NoTiming {
    fn record(&self, _: SubmissionWaveDispatchStage, _: std::time::Duration) {
        panic!("selected replay must not enable timing")
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn submit(
    fixture: &Fixture,
    executable: &dyn ExecutablePlanView,
    identity: &BatchOperationIdentity,
    active: &TrustedActiveSequenceBinding,
    input: SubmissionWaveInputUpload,
    program: &DeviceReusableExecutionProgram,
    wave: PreparedStepSubmissionWave<Runtime>,
    range: Range<usize>,
) -> CompletionHandle<Runtime> {
    submit_for_nodes(
        fixture,
        executable,
        identity,
        active,
        input,
        program,
        wave,
        range,
        &["node.embedding", "node.attention"],
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn submit_for_nodes(
    fixture: &Fixture,
    executable: &dyn ExecutablePlanView,
    identity: &BatchOperationIdentity,
    active: &TrustedActiveSequenceBinding,
    input: SubmissionWaveInputUpload,
    program: &DeviceReusableExecutionProgram,
    wave: PreparedStepSubmissionWave<Runtime>,
    range: Range<usize>,
    node_ids: &[&str],
) -> CompletionHandle<Runtime> {
    let rows = [OperationCostWorkRow {
        offset: range.start as u64,
        count: NonZeroU64::new(range.len() as u64).unwrap(),
        full_input_tokens: NonZeroU64::new(range.end as u64).unwrap(),
    }];
    let predictions = node_ids
        .iter()
        .map(|&node_id| {
            let index = executable
                .execution_plan()
                .payload()
                .nodes()
                .iter()
                .position(|node| node.id().as_str() == node_id)
                .unwrap();
            let route = fixture.providers.providers()[index]
                .eager_cost_route(executable, &rows)
                .unwrap()
                .expect("current native route before graph launch");
            let commands = route
                .commands()
                .iter()
                .filter(|command| command.phase() == DeviceCommandPhase::Compute)
                .collect::<Vec<_>>();
            let [command] = commands.as_slice() else {
                panic!("one native compute command")
            };
            let predicted = command
                .statistical_evidence()
                .expect("future current native evidence")
                .clone();
            (index, predicted)
        })
        .collect::<Vec<_>>();
    let (handle, attribution) = OperationDispatch::encode_and_submit_wave_with_cost_observation(
        fixture.providers.providers(),
        executable,
        identity,
        std::iter::once(active),
        DeviceTimingMode::Off,
        &[input],
        SubmissionExecutionPolicy::determinism_replayed(1),
        Some(program),
        &NoTiming,
        wave,
        &fixture.lane,
        &fixture.reaper,
    )
    .unwrap()
    .into_parts();
    let attribution = attribution.expect("real graph attribution");
    for (index, predicted) in predictions {
        let observed = attribution
            .device()
            .replayed_segments()
            .iter()
            .flat_map(|segment| segment.logical_commands())
            .filter(|command| command.node_index() == index as u32)
            .collect::<Vec<_>>();
        let [logical] = observed.as_slice() else {
            panic!("native node must actually replay, no eager fallback")
        };
        let selected = logical
            .statistical_evidence()
            .expect("current native work bound to sealed capture");
        assert_eq!(selected, &predicted);
        assert_eq!(
            selected.algorithm_work().unwrap().unwrap(),
            predicted.algorithm_work().unwrap().unwrap()
        );
        SelectedReplayAlgorithmTemplateV1::from_selected(
            &predicted,
            logical.token_count(),
            logical.compute_dispatch_count(),
            logical.transfer_command_count(),
        )
        .unwrap()
        .validate_binding(selected)
        .unwrap();
        selected
            .algorithm_work()
            .unwrap()
            .unwrap()
            .validate_command(selected)
            .unwrap();
    }
    handle
}
