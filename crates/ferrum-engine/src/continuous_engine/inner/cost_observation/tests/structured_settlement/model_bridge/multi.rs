use super::super::super::super::trainer::structured_v2::validate_capture_v2;
use super::*;
use crate::continuous_engine::inner::cost_observation::{
    PreparedRowBindingV2, PreparedStructuredFactsV2,
};
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::{
    windows::{PreparedRowFactsV2, PreparedWorkV2},
    StructuredMemberBindingV2, StructuredOwnerFactsV2, StructuredPhaseV2, StructuredUnknownV2,
};

fn original_prepared(
    actual: &ActualWaveShape,
    exact: ferrum_interfaces::execution_cost::CanonicalWaveCostShape,
) -> PreparedStructuredFactsV2 {
    let selected = actual.statistical_evidence.clone().unwrap();
    let recipe = Arc::clone(selected.structured_capture().unwrap().unwrap());
    let owner = StructuredOwnerFactsV2::from_prepared(&exact, &selected, &recipe).unwrap();
    let rows = actual
        .rows
        .iter()
        .enumerate()
        .map(|(i, row)| {
            let numeric = &exact.numeric_features.as_ref().unwrap().rows[i];
            PreparedRowBindingV2 {
                request_id: row.request_id.clone(),
                owner_incarnation: row.owner_incarnation,
                work_generation: row.work_generation,
                frontier: PreparedRowFactsV2 {
                    physical_position: i as u32,
                    work: PreparedWorkV2::Decode { kv_tokens: 7 },
                    generated_before: numeric.generated_tokens_before,
                    maximum_output: numeric.maximum_output_tokens,
                    context_before: 7,
                },
            }
        })
        .collect();
    let prepared = PreparedStructuredFactsV2 {
        exact,
        selected,
        recipe,
        owner,
        rows,
    };
    prepared.validate().unwrap();
    prepared
}

fn member() -> StructuredMemberBindingV2 {
    StructuredMemberBindingV2 {
        rule_signature: [33; 32],
        offered_ordinal: 7,
        member_ordinal: 2,
        phase: StructuredPhaseV2::Fit,
    }
}

#[test]
fn structured_group_real_wall_has_only_pre_execution_bound_sources() {
    let a = session_at([11; 32], 0);
    let b = session_at([12; 32], 0);
    let capture = Arc::new(
        CostCalibrationCapture::for_structured_sessions(vec![Arc::clone(&a), Arc::clone(&b)])
            .unwrap(),
    );
    assert!(
        capture.structured_session().is_none(),
        "legacy single-source bridge stays closed"
    );
    let (actual, hosts, exact) = algorithm_parts_graph(&[3], 8, None);
    // Both source identities and the original Prepared are fixed before call
    // attach/execute. The writer creates the only private settlement receipt.
    let prepared = original_prepared(&actual, exact);
    let stages = stages_for_actual(actual, hosts, Some(terminal()), Some(Arc::clone(&capture)));
    let validated = validate_capture_v2(&capture, true, &prepared).unwrap();
    let first = validated.member_for(&a, member()).unwrap();
    let second = validated.member_for(&b, member()).unwrap();
    assert_eq!(first.source, a.identity());
    assert_eq!(second.source, b.identity());
    assert_ne!(first.source, second.source);
    assert_ne!(first.protocol, second.protocol);
    assert_eq!(first.call_id, stages.call_id);
    assert_eq!(
        (first.call_id, first.ordinal),
        (second.call_id, second.ordinal)
    );
    assert_eq!(
        (first.wall_ns, first.observed_at_ns),
        (second.wall_ns, second.observed_at_ns)
    );
    assert_eq!(first.wall_ns, stages.full_wall_ns.unwrap());
    assert_eq!(first.wall_ns, 12);
    assert_eq!(first.observed_at_ns, 100);
    assert_eq!(first.input, second.input);
    // Even an equal protocol/fingerprint opened afterwards cannot acquire the
    // private call's source authority; nor can an unreconciled call qualify.
    let foreign = session_at([11; 32], 0);
    assert!(matches!(
        validated.member_for(&foreign, member()),
        Err(StructuredUnknownV2::WrongSource)
    ));
    assert!(matches!(
        validate_capture_v2(&capture, false, &prepared),
        Err(StructuredUnknownV2::InvalidSample)
    ));
    assert!(matches!(
        validated.into_member(member()),
        Err(StructuredUnknownV2::WrongSource)
    ));
}

#[test]
fn structured_group_rejects_duplicate_binding_and_any_child_opening_after_prepare() {
    let first = session();
    assert!(
        CostCalibrationCapture::for_structured_sessions(vec![Arc::clone(&first), first]).is_err()
    );
    assert!(CostCalibrationCapture::for_structured_sessions(vec![]).is_err());
    let capture = Arc::new(
        CostCalibrationCapture::for_structured_sessions(vec![
            session(),
            session_at([21; 32], 1_000),
        ])
        .unwrap(),
    );
    let (actual, hosts, exact) = algorithm_parts_graph(&[3], 8, None);
    let prepared = original_prepared(&actual, exact);
    stages_for_actual(actual, hosts, Some(terminal()), Some(Arc::clone(&capture)));
    assert!(matches!(
        validate_capture_v2(&capture, true, &prepared),
        Err(StructuredUnknownV2::Clock)
    ));
}
