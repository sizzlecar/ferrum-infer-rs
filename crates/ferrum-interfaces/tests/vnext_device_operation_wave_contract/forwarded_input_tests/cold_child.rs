use super::*;

/// A cold successor needs its own physical graph slot. Preparing it before
/// observing the parent must keep that slot and its forwarded source valid
/// while the parent fence becomes quiescent, without retiring the parent early.
#[test]
fn prepared_forwarded_child_dispatches_after_parent_observation_before_retirement() {
    let setup = TokenFixture::new(&[37, 91]);
    assert_eq!(setup.lane.in_flight_count(), 1);
    let child = setup.child();
    let wave = prepare_wave(&setup.fixture.plan_resources, &setup.fixture.plan, &child)
        .with_forwarded_inputs(setup.forwards())
        .unwrap();

    // This consumes the parent's completion slot but deliberately keeps its
    // Step unretired. There is no reconstruction of the already prepared child.
    setup.finish_parent(&[38, 92]);
    assert_eq!(setup.lane.in_flight_count(), 0);
    assert!(setup.parent_handle.wait().is_err());
    drop(setup.parent_handle);

    let child_handle = submit(
        &setup.fixture,
        &setup.sessions,
        &setup.lane,
        &setup.reaper,
        wave,
        &[],
    )
    .unwrap();
    assert_eq!(setup.lane.in_flight_count(), 1);
    assert!(matches!(
        child_handle.poll().unwrap(),
        CompletionObservation::Pending
    ));

    // Fence observation alone never authorizes the child's logical commit.
    setup.parent.try_retire_normal().unwrap();
    assert_tokens(
        child_handle.wait_with_readbacks(token_sources(2)).unwrap(),
        &[39, 93],
    );
    assert_eq!(setup.lane.in_flight_count(), 0);
    drop(child_handle);
    child.try_retire_normal().unwrap();
    drop(setup.predecessor);
    drop(setup.batch);
    for session in &setup.sessions {
        session.try_complete().unwrap();
    }
    drop(setup.sessions);
    drop(setup.resources);
    drop(setup.lane);
    drop(setup.reaper);
    drop(setup.fixture.registry);
    drop(setup.fixture.impostor_registry);
    drop(setup.fixture.runtime);
    assert!(matches!(
        PlanRuntimeResources::close(setup.fixture.plan_resources),
        Ok(PlanRuntimeCloseOutcome::Closed(_))
    ));
}
