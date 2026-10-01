use super::*;
use crate::implementations::continuous::cost_model::structured_v2::windows::{
    CohortRequestV2, CohortV2,
};
use crate::implementations::continuous::cost_profile::structured_v10::tests::fixture as old;
use ferrum_interfaces::execution_cost::{PlainTextPolicyCapabilityV2, PlainTextSamplingRouteV2};

fn policy(eos: bool, stop: bool) -> HostCostPolicyV2 {
    let (p, _, _) = old::prepared("policy", 1, 0);
    let mut policy = p.recipe.physical_host_rows[0].installed_policy;
    policy.empirical_content_domain = Some(HostContentDomainV1::PlainTextInstalledV2(
        PlainTextPolicyCapabilityV2 {
            sampling: PlainTextSamplingRouteV2::FullLogits,
            model_eos: eos,
            user_stop: stop,
        },
    ));
    policy
}
fn terminal(reason: ferrum_types::FinishReason, generated: u64) -> Terminal {
    let (p, _, _) = old::prepared("terminal", 3, 2);
    let mut terminal = old::stages(&old::header(), &p, 1, 1000)
        .rows
        .remove(0)
        .terminal
        .unwrap();
    terminal.finish_reason = reason;
    terminal.generated_tokens = generated;
    terminal.through_output_ordinal = generated;
    terminal
}
#[test]
fn source8_installed_terminals_close_actual_budget_while_legacy_keeps_length_only() {
    let mode = LifecycleMode::OriginalInstalledPlainText;
    for reason in [
        ferrum_types::FinishReason::EOS,
        ferrum_types::FinishReason::Stop,
    ] {
        let t = terminal(reason, 1);
        mode.terminal(Some(policy(true, true)), &t, 1, 3).unwrap();
        assert!(LifecycleMode::LegacyLengthOnly
            .terminal(Some(policy(true, true)), &t, 1, 3)
            .is_err());
        assert!(mode.terminal(Some(policy(false, false)), &t, 1, 3).is_err());
        let mut bad = t.clone();
        bad.terminal_handoff_succeeded = false;
        assert!(mode.terminal(Some(policy(true, true)), &bad, 1, 3).is_err());
    }
    assert!(mode
        .terminal(
            Some(policy(true, true)),
            &terminal(ferrum_types::FinishReason::Length, 1),
            1,
            3
        )
        .is_err());
    mode.terminal(
        Some(policy(true, true)),
        &terminal(ferrum_types::FinishReason::Length, 3),
        3,
        3,
    )
    .unwrap();
    assert!(mode
        .terminal(
            Some(policy(true, true)),
            &terminal(ferrum_types::FinishReason::EOS, 4),
            4,
            3
        )
        .is_err());
}
#[test]
fn source8_outside_settlement_closes_cohort_only_with_original_installed_terminal_receipt() {
    let plan = CohortPlanV2 {
        phases: std::array::from_fn(|_| {
            vec![CohortV2 {
                manifest_case: 0,
                repetition: 0,
                requests: vec![CohortRequestV2 {
                    manifest_prompt: 0,
                    maximum_output: 3,
                }],
            }]
        }),
    };
    let mut lifecycle = Lifecycle::with_mode(plan, LifecycleMode::OriginalInstalledPlainText);
    lifecycle.begin(0, 0, 0, 0).unwrap();
    lifecycle.admit(0, 0, 0, "outside".into(), 3, 128).unwrap();
    let (p, _, _) = old::prepared("outside", 1, 0);
    let mut stages = old::stages(&old::header(), &p, 1, 1000);
    stages.rows[0].terminal = Some(terminal(ferrum_types::FinishReason::EOS, 1));
    let row = OriginalCohortRow {
        request_id: "outside",
        owner_incarnation: 1,
        work_generation: 1,
        frontier: p.rows[0].frontier,
        policy: policy(true, false),
        pending_decoded_utf8: false,
    };
    lifecycle
        .outside_completed(StructuredProfilePhaseV10::Fit, 0, &[row], &stages, 1)
        .unwrap();
    assert!(lifecycle.end(0, 0, 1, 1).is_err());
    lifecycle
        .request_completed(CompletedRequest {
            phase: StructuredProfilePhaseV10::Fit,
            cohort: 0,
            slot: 0,
            request_id: "outside".into(),
            owner_incarnation: 1,
            call_id: 1,
            fifo: 1,
            generated_tokens: 1,
            terminal: stages.rows[0].terminal.clone().unwrap(),
        })
        .unwrap();
    lifecycle.end(0, 0, 1, 1).unwrap();
    lifecycle.freeze(0).unwrap();
}
