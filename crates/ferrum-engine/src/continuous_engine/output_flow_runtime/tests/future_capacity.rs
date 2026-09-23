use super::*;
use ferrum_scheduler::implementations::continuous::slo_planner::{
    OutputByteBacking, OutputCreditView,
};

fn window(
    plan: RequestOutputPlan,
    events: usize,
) -> (OutputCreditPool, OutputFlowPort, CreditedOutputSession) {
    let limits = limits(&plan, events);
    let pool = OutputCreditPool::new(OutputPoolLimits {
        maximum: limits.maximum,
        max_total_bytes: limits.maximum.total_bytes().unwrap(),
        max_open_accounts: 1,
    })
    .unwrap();
    let config = SloOutputConfig {
        max_queued_events_per_request: NonZeroUsize::new(limits.maximum.events).unwrap(),
        ..SloOutputConfig::default()
    };
    let budget = RequestOutputBudget::open(&pool, limits, plan).unwrap();
    let (port, session) = spawn_output_flow_runtime(
        budget,
        Arc::new(Notify::new()),
        OutputFlowRuntimeOptions::from_config(&config),
    );
    (pool, port, session)
}

#[tokio::test]
async fn output_owner_future_capacity_commits_whole_window_without_consumer_read() {
    let (pool, mut port, mut session) = window(plan(), 3);
    state(&port, OutputReadinessState::Ready).await;
    let capacity = port.planning_snapshot().future_capacity.unwrap();
    assert_eq!(capacity.remaining_token_commands(), 4);
    assert_eq!(capacity.no_drain_token_commands(), 3);
    let planner = OutputCreditView::try_from(capacity).unwrap();
    assert_eq!(planner.available_token_commands, 3);
    assert_eq!(
        planner.byte_backing,
        OutputByteBacking::PrepaidLifetime {
            remaining_token_commands: 4,
            remaining_wire_bytes: capacity.remaining_wire_bytes() as u64,
        }
    );
    assert_eq!(
        pool.snapshot().data_used.events,
        3,
        "future window is actually escrowed"
    );
    let initial_bytes = capacity.remaining_wire_bytes();
    for generated in 1..=3 {
        let grant = ready(&port).await;
        assert!(
            port.planning_snapshot().future_capacity.is_none(),
            "in-flight is not future Ready"
        );
        grant.committed(delta("a", generated));
    }
    // No frames receiver poll occurred. Three real commands filled three slots.
    state(
        &port,
        OutputReadinessState::OutputBlocked(OutputBlockReason::WireQueue),
    )
    .await;
    assert!(port.planning_snapshot().future_capacity.is_none());
    assert_eq!(pool.snapshot().data_used.bytes, initial_bytes);
    assert!(port.terminal(terminal(3, 3, "")).is_ok());
    assert!(port.planning_snapshot().future_capacity.is_none());
    drop(port);
    let completion = bounded(session.completion).await.unwrap();
    for generated in 1..=3 {
        let frame = bounded(session.frames.next()).await.unwrap();
        assert_eq!(frame.metadata().generated_tokens, generated);
        assert_eq!(frame.wire().payload(), b"a");
        drop(frame);
    }
    let finish = bounded(session.frames.next()).await.unwrap();
    assert!(
        finish.metadata().terminal,
        "reserved terminal fits beside the full data queue"
    );
    drop((finish, completion, session.frames));
    drained(&pool).await;
}

#[tokio::test]
async fn output_owner_future_capacity_chat_accounts_for_two_frames_and_terminal_flush() {
    let plan = super::chat::chat_plan(4, true, true);
    let (pool, mut port, mut session) = window(plan, 5);
    state(&port, OutputReadinessState::Ready).await;
    assert_eq!(
        port.planning_snapshot()
            .future_capacity
            .unwrap()
            .no_drain_token_commands(),
        3,
        "five data slots guarantee three commits, not five, for a two-frame Chat command"
    );
    ready(&port).await.committed(delta("r</think>a", 1));
    state(&port, OutputReadinessState::Ready).await;
    assert_eq!(
        port.planning_snapshot()
            .future_capacity
            .unwrap()
            .no_drain_token_commands(),
        2
    );
    ready(&port).await.committed(delta("b", 2));
    ready(&port).await.committed(delta("c", 3));
    assert!(port.terminal(terminal(3, 3, "d")).is_ok());
    drop(port);
    // Both semantic channels, two further commands and the final flush all fit
    // without polling the receiver; the fixed usage/DONE slot remains separate.
    let completion = bounded(session.completion).await.unwrap();
    let mut metadata = Vec::new();
    for _ in 0..5 {
        let frame = bounded(session.frames.next()).await.unwrap();
        assert!(!frame.metadata().terminal);
        metadata.push((frame.metadata().ordinal, frame.metadata().token));
        drop(frame);
    }
    assert_eq!(
        metadata,
        vec![
            (1, Some(TokenId(1))),
            (1, None),
            (2, Some(TokenId(2))),
            (3, Some(TokenId(3))),
            (3, None)
        ]
    );
    let terminal = bounded(session.frames.next()).await.unwrap();
    assert!(terminal.metadata().terminal);
    assert!(String::from_utf8_lossy(terminal.wire().payload()).contains("\"completion_tokens\":3"));
    drop((terminal, completion, session.frames));
    drained(&pool).await;
}

#[tokio::test]
async fn output_owner_future_capacity_cancel_releases_escrow_but_keeps_projection_until_cleanup() {
    let (pool, port, session) = window(plan(), 3);
    ready(&port).await.committed(delta("a", 1));
    state(&port, OutputReadinessState::Ready).await;
    let capacity = port.planning_snapshot().future_capacity.unwrap();
    assert_eq!(capacity.no_drain_token_commands(), 2);
    let CreditedOutputSession { frames, completion } = session;
    drop(frames);
    assert!(port.planning_snapshot().future_capacity.is_none());
    bounded(async {
        let mut wake = pool.subscribe();
        while pool.snapshot().data_used.events != 0 {
            wake.changed().await.unwrap();
        }
    })
    .await;
    assert_eq!(
        pool.snapshot().data_used.projection_bytes,
        plan().retained_projection_bytes()
    );
    drop(port);
    drop(bounded(completion).await.unwrap());
    drained(&pool).await;
}

#[tokio::test]
async fn output_owner_future_capacity_full_global_event_escrows_cannot_steal_terminals() {
    let pool = pool(2);
    let mut owners = Vec::new();
    for _ in 0..2 {
        let plan = plan();
        let account_limits = limits(&plan, 2);
        let config = SloOutputConfig {
            max_queued_events_per_request: NonZeroUsize::new(account_limits.maximum.events)
                .unwrap(),
            ..SloOutputConfig::default()
        };
        let budget = RequestOutputBudget::open(&pool, account_limits, plan).unwrap();
        owners.push(spawn_output_flow_runtime(
            budget,
            Arc::new(Notify::new()),
            OutputFlowRuntimeOptions::from_config(&config),
        ));
    }
    for (port, _) in &owners {
        state(port, OutputReadinessState::Ready).await;
        assert_eq!(
            port.planning_snapshot()
                .future_capacity
                .unwrap()
                .no_drain_token_commands(),
            2
        );
    }
    let snapshot = pool.snapshot();
    assert_eq!(
        snapshot.data_used.events + snapshot.terminal_held.events,
        snapshot.limits.maximum.events
    );
    for (port, _) in &owners {
        ready(port).await.committed(delta("a", 1));
        ready(port).await.committed(delta("b", 2));
        state(
            port,
            OutputReadinessState::OutputBlocked(OutputBlockReason::WireQueue),
        )
        .await;
    }
    for (mut port, mut session) in owners {
        assert!(port.terminal(terminal(2, 2, "")).is_ok());
        drop(port);
        let completion = bounded(session.completion).await.unwrap();
        for text in [b"a", b"b"] {
            let frame = bounded(session.frames.next()).await.unwrap();
            assert_eq!(frame.wire().payload(), text);
            drop(frame);
        }
        let terminal = bounded(session.frames.next()).await.unwrap();
        assert!(terminal.metadata().terminal);
        drop((terminal, completion, session.frames));
    }
    drained(&pool).await;
}
