use super::*;

fn prepaid(window: u32, remaining: u32, wire_bytes: u64) -> OutputCreditView {
    OutputCreditView {
        available_token_commands: window,
        byte_backing: OutputByteBacking::PrepaidLifetime {
            remaining_token_commands: remaining,
            remaining_wire_bytes: wire_bytes,
        },
    }
}

fn tight_decode() -> RequestSchedulingView {
    let mut request = decode(1);
    request.timing.maximum_output_tokens = n32(4);
    request.timing.budgets.itl_ns = n64(15);
    request.timing.budgets.tpot_ns = n64(15);
    request
}

fn propose(snapshot: &SchedulerSnapshot) -> PlanningDecision {
    planner(4).propose(
        snapshot,
        &Model(|_: &WaveExecutionShape| Some(4)),
        &TestResolver,
        &mut Clock(100),
    )
}

fn blocked(decision: PlanningDecision) {
    assert!(matches!(
        decision,
        PlanningDecision::Unknown {
            reason: PlanningUnknownReason::OutputOrResourceBlocked,
            ..
        }
    ));
}

#[test]
fn prepaid_lifetime_supports_multiple_commands_with_no_free_pool_bytes() {
    let mut request = tight_decode();
    request.output_credit = prepaid(3, 3, 4096);
    let mut snapshot = snapshot(vec![request]);
    snapshot.capacity.available_output_bytes = 0;
    snapshot.scope.horizon_end_ns = 120;
    let original = snapshot.clone();
    let (_, witness, _) = feasible(propose(&snapshot));
    assert!(witness.predicted_output_tokens >= 2);
    assert_eq!(
        snapshot, original,
        "planning cannot spend live reservations"
    );
}

#[test]
fn prepaid_bytes_do_not_invent_wire_or_event_slots() {
    let mut request = tight_decode();
    request.output_credit = prepaid(1, 3, u64::MAX);
    let mut snapshot = snapshot(vec![request]);
    snapshot.capacity.available_output_bytes = u64::MAX;
    snapshot.scope.horizon_end_ns = 120;
    blocked(propose(&snapshot));
}

#[test]
fn prepaid_command_coverage_does_not_grow_to_match_queue_capacity() {
    let mut request = tight_decode();
    request.output_credit = prepaid(3, 1, u64::MAX);
    let mut snapshot = snapshot(vec![request]);
    snapshot.scope.horizon_end_ns = 120;
    blocked(propose(&snapshot));
}

#[test]
fn prepaid_zero_wire_capacity_can_cover_a_proven_no_wire_command() {
    let mut request = decode(1);
    request.output_credit = prepaid(1, 1, 0);
    let mut snapshot = snapshot(vec![request]);
    snapshot.capacity.available_output_bytes = 0;
    feasible(propose(&snapshot));
}

#[test]
fn mixed_backing_charges_only_incremental_bytes_to_shared_pool() {
    let mut owner = decode(1);
    owner.output_credit = prepaid(1, 1, 4096);
    let mut snapshot = snapshot(vec![owner, decode(2)]);
    snapshot.capacity.available_output_bytes = 31;
    blocked(propose(&snapshot));
    snapshot.capacity.available_output_bytes = 32;
    feasible(propose(&snapshot));
}

#[test]
fn two_prepaid_owners_do_not_borrow_each_others_command_capacity() {
    let mut one = decode(1);
    one.output_credit = prepaid(2, 1, 4096);
    let mut two = decode(2);
    two.output_credit = prepaid(0, 1, 4096);
    let mut snapshot = snapshot(vec![one, two]);
    snapshot.capacity.available_output_bytes = 0;
    blocked(propose(&snapshot));
    snapshot.requests[1].output_credit.available_token_commands = 1;
    feasible(propose(&snapshot));
}

#[test]
fn incremental_backing_still_requires_a_proved_byte_bound() {
    let mut request = decode(1);
    request.output_credit.byte_backing = OutputByteBacking::Incremental {
        available_bytes: u64::MAX,
        bytes_per_token_upper_bound: None,
    };
    let snapshot = snapshot(vec![request]);
    blocked(propose(&snapshot));
}
