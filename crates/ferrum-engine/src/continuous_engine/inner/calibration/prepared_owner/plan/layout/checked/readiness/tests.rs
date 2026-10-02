use super::*;
use crate::continuous_engine::inner::calibration::geometry_projection::tests::fixture;
use crate::AutomaticCostProbeOutput;
use ferrum_types::InferenceRequest;
use std::{sync::atomic::Ordering, time::Duration};

fn case(maximum_output: usize) -> Case {
    Case {
        product: OpportunityProduct::Prefill,
        template: 0,
        width: 1,
        maximum_output: NonZeroUsize::new(maximum_output).unwrap(),
        release_generated: 0,
        suffix_tokens: maximum_output,
        preset: SloAutomaticCostProbeSamplingPresetV1::Configured,
        prefix: PrefixKind::Ordinary,
        route: CalibrationDecodeRoute::Actual,
        reset: true,
        acquisition: None,
    }
}

#[tokio::test]
async fn readiness_output_groups_execute_one_original_longest_cohort() {
    let (mut session, executor) = fixture(3).await;
    let chunk = NonZeroU32::new(3).unwrap();
    let mut request = InferenceRequest::new(
        "test test test",
        session.configuration().model.model_id.clone(),
    );
    request.stream = true;
    request.sampling_params.temperature = 1.0;
    request.sampling_params.repetition_penalty = 1.0;
    let prompt_tokens = session
        .engine
        .inner
        .tokenizer
        .encode(&request.prompt, true)
        .unwrap()
        .len();
    assert_eq!(prompt_tokens, 3);
    let templates =
        [AutomaticCostProbeTemplate::new(request, AutomaticCostProbeOutput::CliText).unwrap()];
    let mut cases = [case(1), case(2), case(4)];
    cases[1].prefix = PrefixKind::Pending;
    cases[1].release_generated = 1;
    cases[1].suffix_tokens = 1;
    cases[1].reset = false;
    cases[1].acquisition = work::declared_plan(
        &cases[1],
        work::PrefixBlueprint {
            prompt_tokens,
            boundary: 2,
            span: ferrum_interfaces::vnext::CheckpointTokenSpanConstraint::any_positive(),
            input_tokens_sha256: [3; 32],
        },
        chunk,
        None,
    )
    .unwrap();
    let original_acquisition = cases[1].acquisition.unwrap();
    let original_work = cases[1].waves(prompt_tokens, chunk.get() as usize).unwrap();
    assert_eq!(original_work, (2, 3));
    cases[2].prefix = PrefixKind::Pending;
    cases[2].release_generated = 3;
    cases[2].suffix_tokens = 1;
    cases[2].reset = false;
    let mut attempts = Attempts::new(2 * cases.len(), 4096).unwrap();
    let mut budget = ProbeExecutionBudget::new(
        Instant::now() + Duration::from_secs(10),
        NonZeroUsize::new(1).unwrap(),
        NonZeroUsize::new(4).unwrap(),
    );
    let mut completed = 0;
    // An on-demand route starts from the native case, but readiness executes
    // the complete cold prompt with the longest original output allowance.
    for original in [&cases[1], &cases[0], &cases[2]] {
        let readiness = longest(original, &cases, &[NonZeroUsize::new(4).unwrap()]).unwrap();
        assert_eq!(readiness.maximum_output.get(), 4);
        assert!(matches!(readiness.prefix, PrefixKind::Ordinary));
        assert!(!readiness.reset);
        let (offers, actions) = readiness
            .waves_with_row_ceiling(prompt_tokens, chunk.get() as usize, None)
            .unwrap();
        assert_eq!((offers, actions), (4, 4));
        assert!(readiness.acquisition.is_none());
        if !attempts
            .claim(&readiness, CalibrationDecodeRoute::Actual)
            .unwrap()
        {
            continue;
        }
        let (requests, settings) =
            inventory::readiness_requests(&readiness, &templates, chunk).unwrap();
        budget.reserve_readiness_waves(actions).unwrap();
        let summary = session
            .run_readiness_probe_cohort(requests, settings, &mut budget)
            .await
            .unwrap();
        assert_eq!(summary.completed_requests, 1);
        assert_eq!(summary.completed_output_tokens, 4);
        assert_eq!(summary.wave_attempts, 4);
        completed += summary.completed_requests;
    }
    assert_eq!(completed, 1);
    assert_eq!(executor.physical.load(Ordering::Acquire), 4);
    assert_eq!(budget.requests_remaining(), 0);
    assert_eq!(budget.attempts_remaining(), 0);
    assert_eq!(budget.selection_attempts_remaining(), 0);
    assert_eq!(budget.preflight_charge().admitted_requests, 1);
    assert_eq!(cases[1].acquisition, Some(original_acquisition));
    assert!(matches!(cases[1].prefix, PrefixKind::Pending));
    assert_eq!(cases[1].maximum_output.get(), 2);
    assert_eq!(
        cases[1].waves(prompt_tokens, chunk.get() as usize).unwrap(),
        original_work
    );
    session.completed_owner_boundary().unwrap();
    assert!(session.prepared_owner_capture.is_none());
    session.shutdown().await.unwrap();
}

#[test]
fn readiness_preserves_original_output_cap_and_distinct_routes() {
    let cases = [case(1), case(2), case(4)];
    assert!(longest(&cases[0], &cases, &[NonZeroUsize::new(3).unwrap()]).is_err());
    let shorter = longest(&cases[0], &cases[..2], &[NonZeroUsize::new(32).unwrap()]).unwrap();
    assert_eq!(shorter.maximum_output.get(), 2);
    let mut attempts = Attempts::new(8, 4096).unwrap();
    assert!(attempts
        .claim(&shorter, CalibrationDecodeRoute::Actual)
        .unwrap());
    assert!(!attempts
        .claim(&shorter, CalibrationDecodeRoute::Actual)
        .unwrap());
    assert!(attempts
        .claim(&shorter, CalibrationDecodeRoute::FullLogits)
        .unwrap());
    assert!(!attempts
        .claim(&cases[0], CalibrationDecodeRoute::Actual)
        .unwrap());
    assert!(attempts
        .claim(&cases[2], CalibrationDecodeRoute::Actual)
        .unwrap());
    let mut distinct = shorter.clone();
    distinct.preset = SloAutomaticCostProbeSamplingPresetV1::GreedyLength;
    assert!(attempts
        .claim(&distinct, CalibrationDecodeRoute::Actual)
        .unwrap());
    distinct = shorter.clone();
    distinct.width = 2;
    assert!(attempts
        .claim(&distinct, CalibrationDecodeRoute::Actual)
        .unwrap());
    distinct = shorter;
    distinct.template = 1;
    assert!(attempts
        .claim(&distinct, CalibrationDecodeRoute::Actual)
        .unwrap());
}

#[test]
fn warm_ordinary_filter_preserves_prefix_members_in_the_same_group() {
    let mut warm = case(2);
    warm.reset = false;
    warm.width = 3;
    let mut prefix = warm.clone();
    prefix.width = 2;
    prefix.prefix = PrefixKind::Clean;
    prefix.release_generated = 1;
    prefix.suffix_tokens = 1;
    assert!(inventory::same_group(&warm, &prefix));
    assert!(!needs_inventory(&warm));
    assert!(needs_inventory(&prefix));
    assert_eq!(
        initial_inventory_admissions(&[warm.clone(), prefix.clone()]).unwrap(),
        2
    );
    assert_eq!(initial_inventory_admissions(&[warm]).unwrap(), 0);
    assert_eq!(initial_inventory_admissions(&[prefix, case(1)]).unwrap(), 3);
}
