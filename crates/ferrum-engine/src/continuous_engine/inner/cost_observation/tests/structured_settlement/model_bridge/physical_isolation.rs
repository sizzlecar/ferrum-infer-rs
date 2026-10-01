//! Same private recorder/settlement and worker FIFO as ordinary calibration.
//! Scope rejection must not replace the old receipt or Source3 projection.
use super::*;
use crate::continuous_engine::inner::cost_observation::{
    resolved::ResolvedCostEntry, trainer::structured_v2::validate_capture_v2, PreparedRowBindingV2,
    PreparedStructuredFactsV2,
};
use ferrum_interfaces::execution_cost::{
    CostWorkloadDomainV1, CostWorkloadLimitsV1, ExecutorCostIdentity, EXECUTOR_COST_IDENTITY_SCHEMA,
};
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::{
    windows::{PreparedRowFactsV2, PreparedWorkV2},
    StructuredInputV2, StructuredMemberBindingV2, StructuredOwnerFactsV2, StructuredPhaseV2,
    StructuredUnknownV2,
};
use std::num::{NonZeroU32, NonZeroU64};

fn domain(context: u32) -> CostWorkloadDomainV1 {
    CostWorkloadDomainV1::new_vnext(
        &ExecutorCostIdentity {
            schema_version: EXECUTOR_COST_IDENTITY_SCHEMA,
            model_weights: [1; 32],
            numerical_policy: [2; 32],
            device_runtime: [3; 32],
            execution_config: [4; 32],
        },
        CostWorkloadLimitsV1 {
            maximum_rows: NonZeroU32::new(1).unwrap(),
            maximum_context_tokens: NonZeroU32::new(context).unwrap(),
            maximum_scheduled_tokens_per_wave: NonZeroU64::new(1).unwrap(),
            output_vocabulary_elements: NonZeroU64::new(32).unwrap(),
            repetition_slot_capacity: 32,
            fixed_state_bytes_per_row: 64,
        },
    )
    .unwrap()
}
fn recorded(
    maximum: u64,
    result: Option<HostTerminalStageV1>,
    domain: CostWorkloadDomainV1,
) -> (
    ResolvedCostEntry,
    Arc<CostCalibrationCapture>,
    PreparedStructuredFactsV2,
) {
    let (actual, hosts, exact) = algorithm_parts_graph(&[maximum], 8, None);
    let selected = actual.statistical_evidence.as_ref().unwrap().clone();
    let recipe = selected.structured_capture().unwrap().unwrap().clone();
    let owner = StructuredOwnerFactsV2::from_prepared(&exact, &selected, &recipe).unwrap();
    let row = &actual.rows[0];
    let prepared = PreparedStructuredFactsV2 {
        exact,
        selected,
        recipe,
        owner,
        rows: vec![PreparedRowBindingV2 {
            request_id: row.request_id.clone(),
            owner_incarnation: row.owner_incarnation,
            work_generation: row.work_generation,
            frontier: PreparedRowFactsV2 {
                physical_position: 0,
                work: PreparedWorkV2::Decode { kv_tokens: 7 },
                generated_before: 2,
                maximum_output: maximum,
                context_before: 7,
            },
        }],
    };
    prepared.validate().unwrap();
    let queue = sink(8, 256);
    queue.install_workload_domain(domain).unwrap();
    let capture = Arc::new(CostCalibrationCapture::for_structured_session(session()));
    let (call, clock) = begin_with_retained_capacity(&actual, &queue, 64);
    let mut call = call.with_structured_capture(true);
    for (participant, host) in call.participants.iter_mut().zip(&hosts) {
        participant.host_features = Some(*host);
    }
    call.attach_calibration_capture(capture.clone());
    execute(&mut call, &clock, actual.clone());
    settle(&mut call, &clock, &actual.rows[0], 10, result);
    call.reject(CostCallRejection::Composite);
    clock.set(100);
    assert_eq!(call.finish(), CostCallDisposition::Queued);
    let (fifo, resolved, ticket, _generation, _proof) = queue.pop_resolved_bound().unwrap();
    assert_eq!(fifo, 1);
    assert!(ticket.is_none());
    assert!(queue.pop_resolved_bound().is_none());
    let resolved = resolved.unwrap();
    assert!(matches!(
        capture.status(),
        CostCalibrationStatus::Complete(_)
    ));
    let stages = capture.host_stages().unwrap();
    stages
        .structured_evidence
        .as_ref()
        .unwrap()
        .as_ref()
        .unwrap()
        .validate_host_stages(&stages)
        .unwrap();
    assert!(Arc::ptr_eq(
        &capture.structured_projection(&stages).unwrap().unwrap(),
        &resolved.shared_projection().unwrap()
    ));
    (resolved, capture, prepared)
}
fn source3_input(
    capture: &CostCalibrationCapture,
    prepared: &PreparedStructuredFactsV2,
) -> Result<StructuredInputV2, StructuredUnknownV2> {
    validate_capture_v2(capture, true, prepared)?
        .into_member(StructuredMemberBindingV2 {
            rule_signature: [8; 32],
            offered_ordinal: 1,
            member_ordinal: 1,
            phase: StructuredPhaseV2::Fit,
        })
        .map(|sample| sample.input)
}

#[test]
fn physical_scope_outside_domain_preserves_private_source3_and_legacy_consumers() {
    // Decode at KV 7 consumes one further context token: capacity 7 excludes
    // this sample without making the original completed call invalid.
    let (resolved, capture, prepared) = recorded(4, None, domain(7));
    let shared = resolved.structured().unwrap();
    let unscoped = ResolvedCostEntry::project(resolved.entry()).unwrap();
    assert!(matches!(
        shared.physical_scope,
        Err(StructuredUnknownV2::WrongDomain)
    ));
    assert_eq!(shared.base_input, unscoped.base_input);
    assert_eq!(shared.query.input(), unscoped.query.input());
    assert_eq!(shared.query.input().physical_domain_signature(), None);
    assert_eq!(shared.query.input().settled_terminal_causes(), None);
    assert_eq!(
        source3_input(&capture, &prepared).unwrap(),
        shared.base_input
    );
    let actual = resolved.shared_actual().unwrap();
    assert!(Arc::ptr_eq(&actual, &shared.actual));
    assert!(Arc::ptr_eq(&resolved.selected_actual().unwrap(), &actual));
    assert_eq!(actual.observed_at_ns, 100);
    assert_eq!(actual.wall_ns, unscoped.actual.wall_ns);
    let stages = capture.host_stages().unwrap();
    assert!(Arc::ptr_eq(
        &capture.actual_projection(&stages).unwrap().unwrap(),
        &actual
    ));
}

#[test]
fn physical_scope_valid_domain_carries_real_causes_without_changing_source3_base() {
    let expected_domain = domain(8);
    let (resolved, capture, prepared) = recorded(3, Some(terminal()), expected_domain.clone());
    let shared = resolved.structured().unwrap();
    assert!(shared.physical_scope.is_ok());
    assert_eq!(
        shared.query.input().physical_domain_signature(),
        Some(expected_domain.sha256())
    );
    assert_eq!(
        shared.query.input().settled_terminal_causes(),
        Some([(0, FinishReason::Length)].as_slice())
    );
    assert_eq!(shared.base_input.physical_domain_signature(), None);
    assert_eq!(shared.base_input.settled_terminal_causes(), None);
    assert_eq!(
        source3_input(&capture, &prepared).unwrap(),
        shared.base_input
    );
    let old = ResolvedCostEntry::project(resolved.entry()).unwrap();
    assert_eq!(shared.base_input, old.base_input);
    assert_eq!(
        shared.query.input().regression_axes(),
        old.query.input().regression_axes()
    );
    assert_eq!(shared.actual.observed_at_ns, old.actual.observed_at_ns);
    assert_eq!(shared.actual.wall_ns, old.actual.wall_ns);
    let (continued, _, _) = recorded(4, None, domain(8));
    let continued = continued.structured().unwrap();
    assert!(continued.physical_scope.is_ok());
    assert_eq!(
        continued.query.input().settled_terminal_causes(),
        Some([].as_slice())
    );
}

#[test]
fn physical_scope_invalid_actual_cause_stays_unknown_without_poisoning_old_result() {
    // The legacy PlainTextGreedyV1 recipe lacks an installed EOS declaration.
    // Produce the terminal through the actual private publication/settlement
    // fixture; never mutate a receipt or forge its binding after execution.
    let mut eos = terminal();
    eos.finish_reason = FinishReason::EOS;
    let (resolved, capture, prepared) = recorded(3, Some(eos), domain(8));
    let shared = resolved.structured().unwrap();
    let old = ResolvedCostEntry::project(resolved.entry()).unwrap();
    assert!(matches!(
        shared.physical_scope,
        Err(StructuredUnknownV2::InvalidInput)
    ));
    assert_eq!(shared.base_input, old.base_input);
    assert_eq!(shared.query.input(), old.query.input());
    assert_eq!(shared.query.input().physical_domain_signature(), None);
    assert_eq!(shared.query.input().settled_terminal_causes(), None);
    assert!(Arc::ptr_eq(
        &resolved.shared_actual().unwrap(),
        &shared.actual
    ));
    assert!(resolved.selected_actual().is_ok());
    // Original Source3 rejects EOS at a Length frontier; scope isolation keeps
    // that original rejection instead of broadening its protocol.
    assert!(matches!(
        source3_input(&capture, &prepared),
        Err(StructuredUnknownV2::UnsupportedScope)
    ));
}
