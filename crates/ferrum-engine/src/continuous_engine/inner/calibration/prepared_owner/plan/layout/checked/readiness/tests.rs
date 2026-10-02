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
    let (mut session, executor) = fixture(1).await;
    let mut request = InferenceRequest::new("test", session.configuration().model.model_id.clone());
    request.stream = true;
    request.sampling_params.temperature = 1.0;
    request.sampling_params.repetition_penalty = 1.0;
    let templates =
        [AutomaticCostProbeTemplate::new(request, AutomaticCostProbeOutput::CliText).unwrap()];
    let mut cases = [case(1), case(2), case(4)];
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
    for original in &cases {
        let readiness = longest(original, &cases, &[NonZeroUsize::new(4).unwrap()]).unwrap();
        assert_eq!(readiness.maximum_output.get(), 4);
        assert!(matches!(readiness.prefix, PrefixKind::Ordinary));
        assert!(!readiness.reset);
        if !attempts
            .claim(&readiness, CalibrationDecodeRoute::Actual)
            .unwrap()
        {
            continue;
        }
        let (requests, settings) =
            inventory::readiness_requests(&readiness, &templates, NonZeroU32::MIN).unwrap();
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
    assert_eq!(budget.preflight_charge().admitted_requests, 1);
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
