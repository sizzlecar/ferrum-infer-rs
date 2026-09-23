use super::*;

fn ready_view(port: &OutputFlowPort) -> ReadyOutputCreditView {
    let OutputPlanningCreditView::Ready(view) = port.planning_credit_view() else {
        panic!("expected a physically retained ready grant")
    };
    view
}

#[test]
fn output_owner_planning_view_readiness_requires_token_and_event_capacity() {
    let view = |tokens, events| {
        OutputPlanningCreditView::Ready(ReadyOutputCreditView {
            future_tokens: tokens,
            reserved: OutputCreditAmount {
                events,
                bytes: 0,
                projection_bytes: 0,
            },
        })
    };
    assert_eq!(
        view(0, 1).readiness_state(),
        OutputReadinessState::ProjectionBusy
    );
    assert_eq!(
        view(1, 0).readiness_state(),
        OutputReadinessState::ProjectionBusy
    );
    assert_eq!(
        view(1, 1).readiness_state(),
        OutputReadinessState::Ready,
        "zero wire bytes do not forbid a committed token with no text delta"
    );
}

#[tokio::test]
async fn output_owner_planning_view_is_read_only_and_one_future_token() {
    let pool = pool(4);
    let (port, session) = start(&pool, 4);
    let before = pool.snapshot();
    let changed = port.subscribe();
    let version = *changed.borrow();
    assert_eq!(
        port.planning_credit_view(),
        OutputPlanningCreditView::ProjectionBusy
    );
    assert_eq!(pool.snapshot().data_used, before.data_used);
    assert_eq!(*changed.borrow(), version);

    state(&port, OutputReadinessState::Ready).await;
    let before = pool.snapshot();
    let version = *changed.borrow();
    let view = ready_view(&port);
    assert_eq!(view.future_tokens, 1);
    assert_eq!(view.reserved.events, 1);
    assert_eq!(view.reserved.bytes, plan().lifetime_wire_bytes());
    assert_eq!(view.reserved.projection_bytes, 0);
    assert_eq!(ready_view(&port), view);
    assert_eq!(pool.snapshot().data_used, before.data_used);
    assert_eq!(pool.snapshot().terminal_held, before.terminal_held);
    assert_eq!(*changed.borrow(), version);

    let grant = match port.try_take() {
        OutputReadiness::Ready(grant) => grant,
        _ => panic!("the view must not consume the actual grant"),
    };
    assert_eq!(grant.parts.as_ref().unwrap().frame.credit(), view.reserved);
    assert_eq!(
        port.planning_credit_view(),
        OutputPlanningCreditView::ProjectionBusy
    );
    grant.return_unsubmitted();
    state(&port, OutputReadinessState::Ready).await;
    assert_eq!(
        ready_view(&port),
        view,
        "Deferred restores the identical reservoir"
    );
    drop(session);
    drop(port);
    drained(&pool).await;
}

#[tokio::test]
async fn output_owner_planning_view_requires_ready_state_and_real_permit() {
    let pool = pool(1);
    let (port, session) = start(&pool, 1);
    state(&port, OutputReadinessState::Ready).await;
    let view = ready_view(&port);
    let mut mailbox = port.shared.mailbox.lock();
    assert_eq!(
        port.planning_credit_view(),
        OutputPlanningCreditView::ProjectionBusy,
        "a contended read never waits for or borrows owner authority"
    );
    let parts = mailbox.ready.take().unwrap();
    drop(mailbox);
    assert_eq!(
        port.planning_credit_view(),
        OutputPlanningCreditView::ProjectionBusy,
        "Ready state alone cannot invent reserved storage"
    );
    assert_eq!(port.readiness(), OutputReadinessState::ProjectionBusy);
    {
        let mut mailbox = port.shared.mailbox.lock();
        mailbox.ready = Some(parts);
        mailbox.state = PortState::ProjectionBusy;
    }
    assert_eq!(
        port.planning_credit_view(),
        OutputPlanningCreditView::ProjectionBusy,
        "a retained permit alone does not publish a new token opportunity"
    );
    port.shared.mailbox.lock().state = PortState::Ready;
    assert_eq!(ready_view(&port), view);
    drop(session);
    drop(port);
    drained(&pool).await;
}

#[tokio::test]
async fn output_owner_planning_view_tracks_actual_spent_bytes_and_event_pressure() {
    let pool = pool(1);
    let (port, mut session) = start(&pool, 1);
    state(&port, OutputReadinessState::Ready).await;
    let initial = ready_view(&port);
    ready(&port).await.committed(delta("abc", 1));
    let frame = bounded(session.frames.next()).await.unwrap();
    let spent = frame.wire().credit().bytes;
    state(
        &port,
        OutputReadinessState::OutputBlocked(OutputBlockReason::EventCredit),
    )
    .await;
    assert_eq!(
        port.planning_credit_view(),
        OutputPlanningCreditView::OutputBlocked(OutputBlockReason::EventCredit)
    );
    let used = pool.snapshot().data_used;
    assert_eq!(
        port.planning_credit_view(),
        OutputPlanningCreditView::OutputBlocked(OutputBlockReason::EventCredit)
    );
    assert_eq!(pool.snapshot().data_used, used);
    drop(frame);
    state(&port, OutputReadinessState::Ready).await;
    let next = ready_view(&port);
    assert_eq!(next.future_tokens, 1);
    assert_eq!(next.reserved.events, 1);
    assert_eq!(
        next.reserved.bytes,
        initial.reserved.bytes - spent,
        "releasing a consumed frame never replenishes the lifetime reservoir"
    );
    drop(session);
    drop(port);
    drained(&pool).await;
}

#[tokio::test]
async fn output_owner_planning_view_preserves_wire_pressure_and_healthy_readiness() {
    let pool = pool(2);
    let (slow_port, slow) = start(&pool, 2);
    ready(&slow_port).await.committed(delta("a", 1));
    state(
        &slow_port,
        OutputReadinessState::OutputBlocked(OutputBlockReason::WireQueue),
    )
    .await;
    assert_eq!(
        slow_port.planning_credit_view(),
        OutputPlanningCreditView::OutputBlocked(OutputBlockReason::WireQueue)
    );
    let (healthy_port, healthy) = start(&pool, 2);
    state(&healthy_port, OutputReadinessState::Ready).await;
    assert_eq!(ready_view(&healthy_port).future_tokens, 1);
    let changed = slow_port.subscribe();
    let version = *changed.borrow();
    assert_eq!(
        slow_port.planning_credit_view(),
        OutputPlanningCreditView::OutputBlocked(OutputBlockReason::WireQueue)
    );
    assert_eq!(*changed.borrow(), version);
    drop((slow, healthy));
    drop((slow_port, healthy_port));
    drained(&pool).await;
}

#[tokio::test]
async fn output_owner_planning_view_closing_revokes_future_token_observation() {
    for reason in [
        OutputCloseReason::Complete,
        OutputCloseReason::Cancelled,
        OutputCloseReason::Disconnected,
    ] {
        let pool = pool(1);
        let (mut port, session) = start(&pool, 1);
        state(&port, OutputReadinessState::Ready).await;
        match reason {
            OutputCloseReason::Complete => assert!(port.terminal(terminal(0, 0, "")).is_ok()),
            OutputCloseReason::Cancelled => port.cancel(),
            OutputCloseReason::Disconnected => port.shared.consumer_dropped(),
            _ => unreachable!(),
        }
        // The physical grant can still be in the mailbox until its actor
        // runs. Closing must override it without releasing its storage here.
        assert!(port.shared.mailbox.lock().ready.is_some());
        let before = pool.snapshot().data_used;
        assert_eq!(
            port.planning_credit_view(),
            OutputPlanningCreditView::Closing(reason)
        );
        assert_eq!(pool.snapshot().data_used, before);
        drop(session);
        drop(port);
        drained(&pool).await;
    }
}
