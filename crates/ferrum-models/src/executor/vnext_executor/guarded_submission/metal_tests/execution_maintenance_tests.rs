//! Real cold resource attempts, separated maintenance, and no warm-up work.
use super::completion_work_tests::complete;
use super::*;
use ferrum_interfaces::model_executor::{
    ExecutorExecutionMaintenanceOutcome as Maintenance, ExecutorExecutionMaintenanceTicket,
};

async fn cold_ticket(
    fixture: &Fixture,
    input: &PlanRuntimePrefillInput,
) -> ExecutorExecutionMaintenanceTicket {
    let before = fixture.submissions();
    let expected = complete(fixture, std::slice::from_ref(input), &[]);
    let gate = Gate::new(false);
    let ticket = match fixture
        .executor
        .plan_runtime_batch_prefill_guarded_work_observed(
            std::slice::from_ref(input),
            &expected,
            &gate,
            None,
        )
        .await
    {
        GuardedDispatchOutcome::MaintenanceDeferred { deferral, ticket } => {
            assert_eq!(ticket.stage(), deferral.stage());
            assert_eq!(
                ticket.request_ids(),
                std::slice::from_ref(&input.request_id)
            );
            ticket
        }
        _ => panic!("cold product must retain its real maintenance continuation"),
    };
    assert_eq!(fixture.submissions(), before);
    assert_eq!(gate.calls.load(Ordering::Relaxed), 0);
    fixture.assert_ready(input, input.chunk.tokens_processed());
    ticket
}

fn residency(fixture: &Fixture) -> Vec<(DynamicBackingPoolId, u64, u64)> {
    fixture
        .executor
        .plan_resources
        .dynamic_pool_status()
        .unwrap()
        .pools()
        .iter()
        .map(|pool| {
            (
                pool.pool_id().clone(),
                pool.resident_bytes(),
                pool.pending_growth_bytes(),
            )
        })
        .collect()
}

#[tokio::test]
async fn guarded_metal_execution_maintenance_cold_turns_then_same_work_matches_ordinary() {
    let fixture = Fixture::new(8, false).await;
    let baseline = Fixture::new(8, false).await;
    let input = prompt(&[0, 1, 2, 0], 4);
    let reference = prompt(&[0, 1, 2, 0], 4);
    fixture.admit(&input);
    baseline.admit(&reference);
    let initial_submissions = fixture.submissions();
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut maintained = 0;
    let gate = Gate::new(false);
    let result = loop {
        assert!(
            Instant::now() < deadline,
            "cold exact work failed to make bounded progress"
        );
        let expected = complete(&fixture, std::slice::from_ref(&input), &[]);
        let before = residency(&fixture);
        let attempt = fixture
            .executor
            .plan_runtime_batch_prefill_guarded_work_observed(
                std::slice::from_ref(&input),
                &expected,
                &gate,
                None,
            )
            .await;
        match attempt {
            GuardedDispatchOutcome::MaintenanceDeferred { deferral, ticket } => {
                assert_eq!(
                    residency(&fixture),
                    before,
                    "guarded attempt must not grow physical pools"
                );
                assert_eq!(fixture.submissions(), initial_submissions);
                fixture.assert_ready(&input, 0);
                assert_eq!(ticket.stage(), deferral.stage());
                // Distinct engine turn: no model work or ordinary warm path.
                match fixture
                    .executor
                    .maintain_execution_capacity_once(ticket, &gate)
                    .unwrap()
                {
                    Maintenance::Recapture {
                        progress: Some(progress),
                        ..
                    } => {
                        assert_eq!(progress.attempts(), 1);
                        assert!(!progress.mutations().is_empty());
                        assert_ne!(residency(&fixture), before);
                    }
                    other => panic!("uncontended cold growth must report real mutation: {other:?}"),
                }
                maintained += 1;
                assert_eq!(fixture.submissions(), initial_submissions);
                fixture.assert_ready(&input, 0);
            }
            other => break submitted(other),
        }
    };
    assert!(maintained > 0);
    assert_eq!(gate.calls.load(Ordering::Relaxed), maintained + 1);
    assert_eq!(fixture.submissions() - initial_submissions, 1);
    let reference = baseline.prefill(&reference).await;
    assert_prefill_same(&result[0], &reference);
    assert_eq!(result[0].output().committed_tokens(), 4);
    assert_eq!(result[0].capacity_probe_count(), 0);
}

#[tokio::test]
async fn guarded_metal_execution_maintenance_cancel_reuse_and_host_rejection_do_not_grow() {
    let fixture = Fixture::new(8, false).await;
    let input = prompt(&[0, 1, 2, 0], 4);
    fixture.admit(&input);
    let ticket = cold_ticket(&fixture, &input).await;
    let before = residency(&fixture);
    let gate = Gate::new(true);
    assert!(matches!(
        fixture
            .executor
            .maintain_execution_capacity_once(ticket, &gate)
            .unwrap(),
        Maintenance::Rejected(HostSubmissionRejection::WitnessExpired)
    ));
    assert_eq!(residency(&fixture), before);
    fixture.assert_ready(&input, 0);

    let ticket = cold_ticket(&fixture, &input).await;
    assert!(fixture.executor.cancel_prefill_admission(&input.request_id));
    fixture.admit(&input); // same RequestId, new real SequenceAuthority
    let before = residency(&fixture);
    let gate = Gate::new(false);
    assert!(matches!(
        fixture
            .executor
            .maintain_execution_capacity_once(ticket, &gate)
            .unwrap(),
        Maintenance::Rejected(HostSubmissionRejection::Cancelled)
    ));
    assert_eq!(gate.calls.load(Ordering::Relaxed), 0);
    assert_eq!(residency(&fixture), before);
    fixture.assert_ready(&input, 0);
}

#[tokio::test]
async fn guarded_metal_execution_maintenance_drop_and_other_runtime_cannot_allocate() {
    let fixture = Fixture::new(8, false).await;
    let other = Fixture::new(8, false).await;
    let input = prompt(&[0, 1, 2, 0], 4);
    fixture.admit(&input);
    let ticket = cold_ticket(&fixture, &input).await;
    let before = residency(&fixture);
    drop(ticket);
    assert_eq!(residency(&fixture), before);
    let ticket = cold_ticket(&fixture, &input).await;
    let owners = fixture.rows(std::slice::from_ref(&input), &[]);
    let busy = owners[0].0.operation.try_lock().unwrap();
    let gate = Gate::new(false);
    assert!(matches!(
        fixture
            .executor
            .maintain_execution_capacity_once(ticket, &gate)
            .unwrap(),
        Maintenance::Rejected(HostSubmissionRejection::Busy)
    ));
    assert_eq!(gate.calls.load(Ordering::Relaxed), 0);
    assert_eq!(residency(&fixture), before);
    drop(busy);
    drop(owners);
    // Busy consumed its ticket, but a new exact acquire can request maintenance.
    let ticket = cold_ticket(&fixture, &input).await;
    let other_before = residency(&other);
    let gate = Gate::new(false);
    assert!(matches!(
        other
            .executor
            .maintain_execution_capacity_once(ticket, &gate)
            .unwrap(),
        Maintenance::Unsupported
    ));
    assert_eq!(gate.calls.load(Ordering::Relaxed), 0);
    assert_eq!(residency(&other), other_before);
    assert_eq!(residency(&fixture), before);
    fixture.assert_ready(&input, 0);
}
