//! Selected production GDN query, physical attribution and output/state parity.
use super::attention_cost_route::{compare, ExpectedRoute};
use super::*;

#[test]
fn selected_metal_gated_delta_cost_route_matches_actual_scalar_f32_master() {
    compare(AttentionKind::GatedDelta, &[4], ExpectedRoute::GatedDelta);
}

#[test]
fn selected_metal_gated_delta_cost_route_matches_actual_packed_multiple_rows() {
    compare(
        AttentionKind::GatedDelta,
        &[2, 3],
        ExpectedRoute::GatedDelta,
    );
}

#[test]
fn selected_metal_gated_delta_cost_route_unknown_hadamard_still_executes() {
    for kind in [
        AttentionKind::GatedDeltaHadamardF16,
        AttentionKind::GatedDeltaHadamardF32,
    ] {
        compare(kind, &[2, 3], ExpectedRoute::GatedDelta);
    }
}

// Metal currently has no graph replay implementation. A consumer's omission
// only gates resident numeric reconstruction: real eager evidence remains.
pub(super) fn predict_sample_demand(
    fixture: &Fixture,
    range: &Range<usize>,
) -> (usize, OperationCostRoute) {
    let executable = fixture.compilation.executable();
    let index = executable
        .execution_plan()
        .payload()
        .nodes()
        .iter()
        .position(|node| node.id().as_str() == "node.attention")
        .unwrap();
    let rows = [OperationCostWorkRow {
        offset: range.start as u64,
        count: std::num::NonZeroU64::new(range.len() as u64).unwrap(),
        full_input_tokens: std::num::NonZeroU64::new(range.end as u64).unwrap(),
    }];
    let route = fixture.providers.providers()[index]
        .eager_cost_route(executable, &rows)
        .unwrap()
        .expect("real Metal GDN future route remains supported");
    (index, route)
}

pub(super) fn assert_sample_demand_actual(
    (index, route): (usize, OperationCostRoute),
    attribution: &BoundDeviceSubmissionAttribution,
) {
    // The caller has joined actual terminal completion. Resolve once at the
    // test worker boundary, then share that result across every comparison.
    let resolved = attribution
        .device()
        .clone()
        .resolve_observation()
        .expect("actual completed Metal numeric projection");
    assert!(resolved.replayed_segments().is_empty());
    let actual = resolved
        .commands()
        .iter()
        .filter(|command| {
            command.node_index() == Some(index as u32)
                && command.command_phase() == DeviceCommandPhase::Compute
        })
        .collect::<Vec<_>>();
    let [actual] = actual.as_slice() else {
        panic!("one real Metal GDN compute command")
    };
    let [predicted] = route.commands() else {
        panic!("one future Metal GDN compute command")
    };
    assert_eq!(actual.execution_path(), DeviceExecutionPath::Eager);
    assert_eq!(actual.reusable_graph_node_count(), None);
    assert_eq!(actual.native_op_id(), predicted.native_operation());
    assert_eq!(actual.participant_start(), predicted.participant_start());
    assert_eq!(actual.participant_count(), predicted.participant_count());
    assert_eq!(actual.token_count(), predicted.token_count());
    assert_eq!(actual.batching_form(), predicted.batching());
    assert_eq!(
        actual.compute_dispatch_count(),
        predicted.compute_dispatch_count()
    );
    assert_eq!(
        actual.transfer_command_count(),
        predicted.transfer_command_count()
    );
    assert_metal_algorithm_work(
        actual
            .statistical_evidence()
            .expect("eager sample is retained for every demand"),
        predicted
            .statistical_evidence()
            .expect("future selected evidence is retained"),
        ferrum_types::SloStructuredCostCapture::HostSettledV1,
    );
}

#[test]
fn consumer_demand_preserves_metal_eager_gdn_evidence_and_state() {
    use ferrum_interfaces::execution_cost::{
        GuardedNotSubmittedReason, HostSubmissionRejection, StructuredCostSampleDemand,
    };
    use ferrum_types::{SloStructuredActualCapturePolicy, SloStructuredCostCapture};
    const ROWS: usize = 2;
    const WAVES: usize = 3;
    let build = |capture| {
        let kind = AttentionKind::GatedDelta;
        let definition = Family::new(kind);
        let states = definition.states();
        let profile = definition.profile_id();
        let family = TypedFamilyRegistration::new(definition)
            .prepare_with_profile(&serde_json::to_value(kind).unwrap(), &id(profile))
            .unwrap();
        let composition = composition_with_capture(kind, capture);
        assert_eq!(
            composition.0.cost_graph_capture_capability(),
            DeviceCostGraphCaptureCapability::Unsupported
        );
        Fixture::from_prepared_family_with_composition(
            kind,
            family,
            states,
            FixtureExecutionMode::Eager,
            None,
            ROWS as u64,
            BTreeMap::new(),
            composition,
        )
    };
    let eager = build(SloStructuredCostCapture::Disabled);
    let observed = build(SloStructuredCostCapture::HostSettledV1);
    let tokens: Arc<[u32]> = (0..WAVES * ROWS)
        .map(|i| 1 + ((i * 7 + i / ROWS) % 29) as u32)
        .collect();
    let expected_owner = eager.admit("metal-demand-baseline", Arc::clone(&tokens));
    let owner = observed.admit("metal-demand-observed", Arc::clone(&tokens));
    let demand = |consumer| {
        StructuredCostSampleDemand::for_call(
            SloStructuredActualCapturePolicy::ConsumerDrivenV1,
            consumer,
            false,
        )
    };
    assert_eq!(demand(true), StructuredCostSampleDemand::Requested);
    assert_eq!(demand(false), StructuredCostSampleDemand::NotRequested);
    observed
        .lane
        .configure_submission_readback_staging(1 << 20)
        .unwrap();
    let (pending, step) = guarded_cost_route::reject_encoded_with_sample_demand(
        &observed,
        &owner,
        0..ROWS,
        Arc::clone(&tokens),
        demand(false),
    );
    let rejected = pending
        .reconcile_step(step)
        .unwrap_or_else(|(error, _)| panic!("sample demand changed Metal rollback: {error}"));
    assert_eq!(
        rejected.reason(),
        GuardedNotSubmittedReason::HostRejected(HostSubmissionRejection::WitnessExpired)
    );
    drop(rejected);
    let mut previous: Option<Observation> = None;
    for (index, consumer) in [true, false, true].into_iter().enumerate() {
        let range = index * ROWS..(index + 1) * ROWS;
        let expected = eager.execute(&expected_owner, Arc::clone(&tokens), range.clone());
        let actual = observed
            .execute_checked_output_node_with_sample_demand(
                &owner,
                Arc::clone(&tokens),
                range,
                false,
                false,
                true,
                "node.attention",
                Some(demand(consumer)),
            )
            .unwrap();
        expected.assert_same(
            &actual,
            "demand preserves real Metal output and complete recurrent states",
        );
        actual.assert_state_nonzero();
        if let Some(prior) = &previous {
            assert_ne!(
                actual.values["output"], prior.values["output"],
                "new tokens must not reuse old output"
            );
            actual.assert_state_changed(prior, "new tokens must update every recurrent state");
        }
        previous = Some(actual);
    }
    owner.try_complete().unwrap();
    expected_owner.try_complete().unwrap();
}
