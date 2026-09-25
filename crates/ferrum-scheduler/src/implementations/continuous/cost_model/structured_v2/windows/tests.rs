use super::*;
use crate::implementations::continuous::cost_model::structured_v2::{
    StructuredProductV2, StructuredTemplateV2, StructuredWaveRoleV2,
};
use ferrum_interfaces::execution_cost::CoreReadbackRoute;

fn owner(rows: u32) -> StructuredOwnerKeyV2 {
    StructuredOwnerKeyV2 {
        rows,
        role: StructuredWaveRoleV2::OrdinaryDecode,
        product: StructuredProductV2::FullLogits,
        readback: CoreReadbackRoute::HostSynchronized,
        provider_template: StructuredTemplateV2::Ordered([1; 32]),
        algorithm_domain: [2; 32],
        installed_policy: [3; 32],
    }
}
fn range(minimum: u64, maximum: u64) -> ClosedRangeV2 {
    ClosedRangeV2 { minimum, maximum }
}
fn decode_window(generated: ClosedRangeV2) -> RowWindowV2 {
    RowWindowV2 {
        generated_before: generated,
        remaining_output: ClosedRangeV2::ALL,
        context_before: ClosedRangeV2::ALL,
        work: WorkWindowV2::Decode {
            kv_tokens: ClosedRangeV2::ALL,
        },
    }
}
fn row(physical_position: u32, generated_before: u64) -> PreparedRowFactsV2 {
    PreparedRowFactsV2 {
        physical_position,
        work: PreparedWorkV2::Decode { kv_tokens: 100 },
        generated_before,
        maximum_output: 10,
        context_before: 100,
    }
}
fn make_rule(windows: Vec<Vec<RowWindowV2>>) -> MembershipRuleV2 {
    MembershipRuleV2 {
        owner: owner(windows[0].len() as u32),
        windows: windows
            .into_iter()
            .map(|rows| FrontierWindowV2 { rows })
            .collect(),
    }
}

#[test]
fn structured_v2_windows_overlap_reserves_once_and_preserves_gaps() {
    let rule = make_rule(vec![
        vec![decode_window(range(2, 4))],
        vec![decode_window(range(3, 5))],
    ]);
    let classified: Vec<_> = (1..=6)
        .map(|n| rule.classify(&owner(1), &[row(0, n)]).unwrap())
        .collect();
    assert_eq!(
        classified,
        vec![None, Some(0), Some(0), Some(0), Some(1), None]
    );
    // A later budget boundary can be selected without admitting the complete
    // intervening prefix into the numerical population.
    let mut terminal = decode_window(ClosedRangeV2::ALL);
    terminal.remaining_output = range(1, 1);
    let terminal_rule = make_rule(vec![vec![terminal]]);
    assert_eq!(terminal_rule.classify(&owner(1), &[row(0, 8)]), Ok(None));
    assert_eq!(terminal_rule.classify(&owner(1), &[row(0, 9)]), Ok(Some(0)));
}

#[test]
fn structured_v2_windows_keep_physical_positions_and_reject_invalid_outside_facts() {
    let rule = make_rule(vec![vec![
        decode_window(range(1, 1)),
        decode_window(range(4, 4)),
    ]]);
    let mut rows = [row(0, 1), row(1, 4)];
    assert_eq!(rule.classify(&owner(2), &rows), Ok(Some(0)));
    rows.swap(0, 1);
    assert_eq!(
        rule.classify(&owner(2), &rows),
        Err(StructuredUnknownV2::InvalidInput)
    );
    rows = [row(0, 4), row(1, 1)];
    assert_eq!(rule.classify(&owner(2), &rows), Ok(None));
    let mut another = owner(2);
    another.algorithm_domain = [4; 32];
    assert_eq!(rule.classify(&another, &rows), Ok(None));
    rows[0].maximum_output = rows[0].generated_before;
    assert_eq!(
        rule.classify(&another, &rows),
        Err(StructuredUnknownV2::InvalidInput)
    );
}

#[test]
fn structured_v2_windows_prefill_finality_is_derived_from_checked_work() {
    let mut rule = make_rule(vec![vec![RowWindowV2 {
        generated_before: range(0, 0),
        remaining_output: ClosedRangeV2::ALL,
        context_before: ClosedRangeV2::ALL,
        work: WorkWindowV2::Prefill {
            offset: ClosedRangeV2::ALL,
            count: range(2, 2),
            total_prompt_tokens: range(4, 4),
            emits_token: Some(true),
        },
    }]]);
    rule.owner.role = StructuredWaveRoleV2::Prefill;
    let mut prepared = row(0, 0);
    prepared.context_before = 0;
    prepared.work = PreparedWorkV2::Prefill {
        offset: 0,
        count: 2,
        total_prompt_tokens: 4,
    };
    assert_eq!(rule.classify(&rule.owner, &[prepared]), Ok(None));
    prepared.work = PreparedWorkV2::Prefill {
        offset: 2,
        count: 2,
        total_prompt_tokens: 4,
    };
    assert_eq!(rule.classify(&rule.owner, &[prepared]), Ok(Some(0)));
    prepared.work = PreparedWorkV2::Prefill {
        offset: u32::MAX,
        count: 2,
        total_prompt_tokens: 4,
    };
    assert_eq!(
        rule.classify(&rule.owner, &[prepared]),
        Err(StructuredUnknownV2::InvalidInput)
    );
    prepared.work = PreparedWorkV2::Prefill {
        offset: 2,
        count: 0,
        total_prompt_tokens: 4,
    };
    assert_eq!(
        rule.classify(&rule.owner, &[prepared]),
        Err(StructuredUnknownV2::InvalidInput)
    );
}

#[test]
fn structured_v2_windows_rule_signature_binds_full_predicates_and_owner() {
    let rule = make_rule(vec![vec![decode_window(range(2, 4))]]);
    let encoded = serde_json::to_vec(&rule).unwrap();
    let decoded: MembershipRuleV2 = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(rule.signature(), decoded.signature());
    let mut changed = rule.clone();
    changed.windows[0].rows[0].generated_before.maximum += 1;
    assert_ne!(rule.signature(), changed.signature());
    changed = rule.clone();
    changed.owner.installed_policy = [7; 32];
    assert_ne!(rule.signature(), changed.signature());
    changed.windows[0].rows[0].generated_before = range(4, 2);
    assert_eq!(
        changed.signature(),
        Err(StructuredUnknownV2::InvalidSettings)
    );
}

fn cohort_plan() -> CohortPlanV2 {
    CohortPlanV2 {
        phases: std::array::from_fn(|_| {
            vec![CohortV2 {
                manifest_case: 0,
                repetition: 0,
                requests: vec![
                    CohortRequestV2 {
                        manifest_prompt: 0,
                        maximum_output: 12,
                    },
                    CohortRequestV2 {
                        manifest_prompt: 0,
                        maximum_output: 12,
                    },
                ],
            }]
        }),
    }
}

#[test]
fn structured_v2_cohort_contract_preserves_duplicate_requests_and_all_three_phases() {
    let plan = cohort_plan();
    plan.validate().unwrap();
    let payload: serde_json::Value = serde_json::from_str(r#"{"a":1,"b":{"d":4,"c":3}}"#).unwrap();
    let reordered: serde_json::Value =
        serde_json::from_str(r#"{"b":{"c":3,"d":4},"a":1}"#).unwrap();
    assert_eq!(plan.signature(&payload), plan.signature(&reordered));
    let mut fewer = plan.clone();
    fewer.phases[1][0].requests.pop();
    assert_ne!(plan.signature(&payload), fewer.signature(&payload));
    let mut changed = plan.clone();
    changed.phases[2][0].requests[1].maximum_output += 1;
    assert_ne!(plan.signature(&payload), changed.signature(&payload));
    assert_ne!(
        plan.signature(&payload),
        plan.signature(&serde_json::json!({"a":2,"b":{"c":3,"d":4}}))
    );
    changed.phases[2].clear();
    assert_eq!(
        changed.validate(),
        Err(StructuredUnknownV2::InvalidSettings)
    );
    changed = plan;
    changed.phases[0][0].repetition = 1;
    assert_eq!(
        changed.validate(),
        Err(StructuredUnknownV2::InvalidSettings)
    );
}
