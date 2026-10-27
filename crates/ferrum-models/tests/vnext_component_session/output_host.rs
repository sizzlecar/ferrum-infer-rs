//! Completion readbacks consume pre-admitted, same-parent Host scope credits.
use super::*;

#[test]
fn output_scope_cannot_change_parent_and_remains_rebindable_after_close() {
    let fixture = Fixture::new("device.component.output-scope");
    let mut session = fixture.initialize("output-scope");
    let scope = fixture
        .host
        .reserve_scope(&[(HostMemoryClass::Output, 2 * OUTPUT_BYTES)])
        .unwrap();
    session.set_output_host_account(scope.account()).unwrap();
    let before = fixture.composition.runtime().snapshot();
    let foreign = HostMemoryAccount::new(fixture.host.limits());
    assert!(session.set_output_host_account(foreign.clone()).is_err());
    assert_eq!(
        fixture.composition.runtime().snapshot().submissions,
        before.submissions
    );
    assert_eq!(foreign.usage().unwrap().committed_bytes(), 0);
    assert!(fixture.host.reserve(HostMemoryClass::Output, 1).is_err());

    // The parent has no spare Output capacity. Both executions must consume
    // the view's already committed credit rather than double-charge it.
    let first = execute(&mut session, &fixture, "scope-first", 1.);
    let second = execute(&mut session, &fixture, "scope-second", 2.);
    assert_eq!(
        fixture.host.usage().unwrap().allocated_bytes(),
        2 * OUTPUT_BYTES
    );
    drop(scope);
    session
        .set_output_host_account(fixture.host.clone())
        .unwrap();
    drop(first);
    let next = execute(&mut session, &fixture, "parent-rebound", 3.);
    assert_eq!(
        fixture.host.usage().unwrap().allocated_bytes(),
        2 * OUTPUT_BYTES
    );
    drop((second, next));
    assert_eq!(fixture.host.usage().unwrap().committed_bytes(), 0);
    match session
        .close()
        .unwrap_or_else(|_| panic!("session close failed"))
    {
        PlanRuntimeCloseOutcome::Closed(_) => (),
        PlanRuntimeCloseOutcome::Referenced { .. } => panic!("unexpected retained session"),
    }
    fixture.assert_device_reusable();
}

#[test]
fn closed_output_scope_fails_before_submission_and_parent_rebinding_recovers() {
    let fixture = Fixture::new("device.component.closed-output-scope");
    let mut session = fixture.initialize("closed-output-scope");
    let scope = fixture
        .host
        .reserve_scope(&[(HostMemoryClass::Output, OUTPUT_BYTES)])
        .unwrap();
    session.set_output_host_account(scope.account()).unwrap();
    drop(scope);
    let before = fixture.composition.runtime().snapshot();
    assert!(session
        .execute(
            id("run.closed-scope"),
            id("request.closed-scope"),
            vec![fixture.input(1.)]
        )
        .is_err());
    assert_eq!(
        fixture.composition.runtime().snapshot().submissions,
        before.submissions
    );
    assert_eq!(fixture.host.usage().unwrap().committed_bytes(), 0);
    session
        .set_output_host_account(fixture.host.clone())
        .unwrap();
    drop(execute(&mut session, &fixture, "after-closed-scope", 1.));
    match session
        .close()
        .unwrap_or_else(|_| panic!("session close failed"))
    {
        PlanRuntimeCloseOutcome::Closed(_) => (),
        PlanRuntimeCloseOutcome::Referenced { .. } => panic!("unexpected retained session"),
    }
    fixture.assert_device_reusable();
}
