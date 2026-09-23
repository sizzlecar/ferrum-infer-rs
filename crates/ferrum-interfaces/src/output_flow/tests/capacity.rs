use super::*;

fn open_window(plan: RequestOutputPlan, events: usize) -> (OutputCreditPool, RequestOutputBudget) {
    let mut limits = limits(&plan);
    limits.maximum.events = limits.terminal.events + events;
    let pool = OutputCreditPool::new(OutputPoolLimits {
        maximum: limits.maximum,
        max_total_bytes: limits.maximum.total_bytes().unwrap(),
        max_open_accounts: 1,
    })
    .unwrap();
    let budget = RequestOutputBudget::open(&pool, limits, plan).unwrap();
    (pool, budget)
}

fn frame(budget: &mut RequestOutputBudget) -> OutputFramePermit {
    let OutputFrameAttempt::Reserved(frame) = budget.try_begin_frame().unwrap() else {
        panic!("backed frame")
    };
    frame
}

#[test]
fn output_flow_future_capacity_requires_real_event_escrow_and_wire_slots() {
    let plan = plan(OutputProjectionContract::cli_text(), 4);
    let (pool, mut budget) = open_window(plan, 3);
    let first = frame(&mut budget);
    assert_eq!(
        budget
            .future_capacity_view(&first, 0, 100)
            .unwrap()
            .no_drain_token_commands(),
        1
    );
    budget.return_unsubmitted_frame(first).unwrap();
    budget.reserve_future_event_window(3).unwrap();
    let first = frame(&mut budget);
    let used = pool.snapshot().data_used;
    let capacity = budget.future_capacity_view(&first, 0, 2).unwrap();
    assert_eq!(capacity.remaining_token_commands(), 4);
    assert_eq!(
        capacity.remaining_wire_bytes(),
        budget.plan().lifetime_wire_bytes()
    );
    assert_eq!(capacity.no_drain_token_commands(), 3);
    assert_eq!(
        budget
            .future_capacity_view(&first, 0, 0)
            .unwrap()
            .no_drain_token_commands(),
        1
    );
    assert_eq!(
        budget
            .future_capacity_view(&first, 3, 2)
            .unwrap()
            .no_drain_token_commands(),
        1
    );
    assert_eq!(
        pool.snapshot().data_used,
        used,
        "copying views acquires no credit"
    );
    budget.return_unsubmitted_frame(first).unwrap();
    assert_eq!(
        pool.snapshot().data_used,
        used,
        "Deferred keeps actual future event escrow"
    );
    drop(budget);
    assert_eq!(pool.snapshot().data_used, OutputCreditAmount::ZERO);
    assert_eq!(pool.snapshot().retained_accounts, 0);
}

#[test]
fn output_flow_future_capacity_rejects_same_generation_foreign_pool_and_keeps_spent_bytes_spent() {
    let plan = plan(OutputProjectionContract::cli_text(), 4);
    let (pool, mut budget) = open_window(plan.clone(), 2);
    let (other_pool, mut other) = open_window(plan, 2);
    budget.reserve_future_event_window(2).unwrap();
    let first = frame(&mut budget);
    let foreign = frame(&mut other);
    assert_eq!(first.generation(), foreign.generation());
    assert!(matches!(
        budget.future_capacity_view(&foreign, 0, 1),
        Err(OutputFlowError::Credit(
            OutputCreditError::ForeignReservation
        ))
    ));
    let initial = first.credit().bytes;
    let wire = budget.encode_data_frame(first, "abc", 0).unwrap();
    let spent = wire.credit().bytes;
    drop(wire);
    let next = frame(&mut budget);
    assert_eq!(
        budget
            .future_capacity_view(&next, 1, 1)
            .unwrap()
            .remaining_wire_bytes(),
        initial - spent
    );
    drop((next, foreign, budget, other));
    assert_eq!(pool.snapshot().data_used, OutputCreditAmount::ZERO);
    assert_eq!(other_pool.snapshot().data_used, OutputCreditAmount::ZERO);
}
