use super::*;
use std::sync::atomic::AtomicUsize;

struct VirtualTimerState {
    now: Mutex<Instant>,
    revision: watch::Sender<u64>,
    registrations: AtomicUsize,
}
struct VirtualTimer(Arc<VirtualTimerState>);
impl VirtualTimer {
    fn new() -> Arc<Self> {
        let (revision, _) = watch::channel(0);
        Arc::new(Self(Arc::new(VirtualTimerState {
            now: Mutex::new(Instant::now()),
            revision,
            registrations: AtomicUsize::new(0),
        })))
    }
    fn advance(&self, duration: Duration) {
        let mut now = self.0.now.lock();
        *now = now.checked_add(duration).unwrap();
        drop(now);
        self.0.revision.send_modify(|revision| *revision += 1);
    }
    async fn registered(&self, minimum: usize) {
        let mut changes = self.0.revision.subscribe();
        bounded(async {
            while self.0.registrations.load(Ordering::Acquire) < minimum {
                changes.changed().await.unwrap();
            }
        })
        .await;
    }
}
impl OutputFlowTimer for VirtualTimer {
    fn now(&self) -> Instant {
        *self.0.now.lock()
    }
    fn sleep_until(&self, deadline: Instant) -> BoxFuture<'static, ()> {
        let state = self.0.clone();
        state.registrations.fetch_add(1, Ordering::Release);
        state.revision.send_modify(|revision| *revision += 1);
        Box::pin(async move {
            let mut changes = state.revision.subscribe();
            loop {
                if *state.now.lock() >= deadline {
                    return;
                }
                changes.changed().await.unwrap();
            }
        })
    }
}

fn start_timed(
    pool: &OutputCreditPool,
    timer: Arc<VirtualTimer>,
    timeout: Duration,
) -> (OutputFlowPort, CreditedOutputSession) {
    let plan = plan();
    let max_queued_events = plan.terminal_credit().events + 1;
    let budget = RequestOutputBudget::open(pool, limits(&plan, 1), plan).unwrap();
    spawn_output_flow_runtime(
        budget,
        Arc::new(Notify::new()),
        OutputFlowRuntimeOptions {
            slow_consumer_timeout: timeout,
            max_queued_events,
            timer,
        },
    )
}

async fn timeout_state(port: &OutputFlowPort) {
    state(
        port,
        OutputReadinessState::Closing(OutputCloseReason::SlowConsumer),
    )
    .await;
}

#[tokio::test]
async fn output_owner_timeout_excludes_inflight_and_keeps_projection_until_engine_cleanup() {
    let pool = pool(1);
    let timer = VirtualTimer::new();
    let (port, mut session) = start_timed(&pool, timer.clone(), Duration::from_secs(10));
    let grant = ready(&port).await;
    timer.advance(Duration::from_secs(100));
    assert_eq!(port.readiness(), OutputReadinessState::ProjectionBusy);
    assert_eq!(timer.0.registrations.load(Ordering::Acquire), 0);
    grant.return_unsubmitted();
    ready(&port).await.committed(delta("a", 1));
    state(
        &port,
        OutputReadinessState::OutputBlocked(OutputBlockReason::WireQueue),
    )
    .await;
    timer.registered(1).await;
    timer.advance(Duration::from_secs(10));
    timeout_state(&port).await;
    assert_eq!(
        pool.snapshot().data_used.projection_bytes,
        plan().retained_projection_bytes()
    );
    let mut completion = session.completion;
    assert!(futures::poll!(&mut completion).is_pending());
    let data = bounded(session.frames.next()).await.unwrap();
    let terminal = bounded(session.frames.next()).await.unwrap();
    assert!(terminal.metadata().terminal);
    assert!(String::from_utf8_lossy(terminal.wire().payload()).contains("blocking timeout"));
    drop(port);
    drop(bounded(completion).await.unwrap());
    drop(terminal);
    assert_eq!(pool.snapshot().retained_accounts, 1);
    assert_eq!(
        pool.snapshot().data_used.events,
        1,
        "delivered payload still owns its event"
    );
    drop((data, session.frames));
    drained(&pool).await;
}

#[tokio::test]
async fn output_owner_timeout_does_not_restart_when_wire_pressure_becomes_event_pressure() {
    let pool = pool(1);
    let timer = VirtualTimer::new();
    let (port, mut session) = start_timed(&pool, timer.clone(), Duration::from_secs(10));
    ready(&port).await.committed(delta("a", 1));
    state(
        &port,
        OutputReadinessState::OutputBlocked(OutputBlockReason::WireQueue),
    )
    .await;
    timer.advance(Duration::from_secs(6));
    let retained = bounded(session.frames.next()).await.unwrap();
    state(
        &port,
        OutputReadinessState::OutputBlocked(OutputBlockReason::EventCredit),
    )
    .await;
    timer.registered(2).await;
    timer.advance(Duration::from_secs(4));
    timeout_state(&port).await;
    drop(port);
    let terminal = bounded(session.frames.next()).await.unwrap();
    assert!(terminal.metadata().terminal);
    drop((
        terminal,
        bounded(session.completion).await.unwrap(),
        session.frames,
    ));
    assert_eq!(pool.snapshot().data_used.events, 1);
    drop(retained);
    drained(&pool).await;
}

#[tokio::test]
async fn output_owner_timeout_recovery_starts_a_new_blocking_interval() {
    let pool = pool(1);
    let timer = VirtualTimer::new();
    let (port, mut session) = start_timed(&pool, timer.clone(), Duration::from_secs(10));
    ready(&port).await.committed(delta("a", 1));
    state(
        &port,
        OutputReadinessState::OutputBlocked(OutputBlockReason::WireQueue),
    )
    .await;
    timer.advance(Duration::from_secs(8));
    drop(bounded(session.frames.next()).await.unwrap());
    ready(&port).await.committed(delta("b", 2));
    state(
        &port,
        OutputReadinessState::OutputBlocked(OutputBlockReason::WireQueue),
    )
    .await;
    timer.registered(2).await;
    timer.advance(Duration::from_secs(2));
    assert_eq!(
        port.readiness(),
        OutputReadinessState::OutputBlocked(OutputBlockReason::WireQueue)
    );
    timer.advance(Duration::from_secs(8));
    timeout_state(&port).await;
    drop(port);
    drop(bounded(session.frames.next()).await.unwrap());
    drop(bounded(session.frames.next()).await.unwrap());
    drop((bounded(session.completion).await.unwrap(), session.frames));
    drained(&pool).await;
}

#[tokio::test]
async fn output_owner_terminal_flush_inherits_existing_blocked_deadline() {
    let pool = pool(1);
    let timer = VirtualTimer::new();
    let (mut port, mut session) = start_timed(&pool, timer.clone(), Duration::from_secs(10));
    ready(&port).await.committed(delta("a", 1));
    state(
        &port,
        OutputReadinessState::OutputBlocked(OutputBlockReason::WireQueue),
    )
    .await;
    timer.advance(Duration::from_secs(6));
    assert!(port.terminal(terminal(1, 1, "tail")).is_ok());
    timer.registered(2).await;
    timer.advance(Duration::from_secs(4));
    timeout_state(&port).await;
    drop(port);
    let completion = bounded(session.completion).await.unwrap();
    assert!(matches!(completion.payload(), OutputCompletion::Failed(_)));
    let data = bounded(session.frames.next()).await.unwrap();
    assert_eq!(data.wire().payload(), b"a");
    let terminal = bounded(session.frames.next()).await.unwrap();
    assert!(
        terminal.metadata().terminal,
        "expired tail is replaced by the reserved error frame"
    );
    drop((data, terminal, completion, session.frames));
    drained(&pool).await;
}

#[tokio::test]
async fn output_owner_terminal_flush_starts_deadline_only_when_it_waits() {
    let pool = pool(1);
    let timer = VirtualTimer::new();
    let (mut port, mut session) = start_timed(&pool, timer.clone(), Duration::from_secs(10));
    let grant = ready(&port).await;
    timer.advance(Duration::from_secs(100));
    grant.committed(delta("a", 1));
    assert!(port.terminal(terminal(1, 1, "tail")).is_ok());
    timer.registered(1).await;
    timer.advance(Duration::from_secs(9));
    assert_eq!(
        port.readiness(),
        OutputReadinessState::Closing(OutputCloseReason::Complete)
    );
    timer.advance(Duration::from_secs(1));
    timeout_state(&port).await;
    drop(port);
    drop(bounded(session.frames.next()).await.unwrap());
    drop(bounded(session.frames.next()).await.unwrap());
    drop((bounded(session.completion).await.unwrap(), session.frames));
    drained(&pool).await;
}

#[tokio::test]
async fn output_owner_same_tick_release_reprobes_real_capacity_before_timing_out() {
    let pool = pool(1);
    let timer = VirtualTimer::new();
    let (mut port, mut session) = start_timed(&pool, timer.clone(), Duration::from_secs(10));
    ready(&port).await.committed(delta("a", 1));
    let retained = bounded(session.frames.next()).await.unwrap();
    state(
        &port,
        OutputReadinessState::OutputBlocked(OutputBlockReason::EventCredit),
    )
    .await;
    timer.registered(1).await;
    // Both notifications become ready without giving the actor a poll between
    // them. A real reservation, not this release notification, clears timeout.
    timer.advance(Duration::from_secs(10));
    drop(retained);
    ready(&port).await.return_unsubmitted();
    assert!(port.terminal(terminal(1, 1, "")).is_ok());
    drop(port);
    drop(bounded(session.frames.next()).await.unwrap());
    let completion = bounded(session.completion).await.unwrap();
    assert!(matches!(
        completion.payload(),
        OutputCompletion::Succeeded { .. }
    ));
    drop((completion, session.frames));
    drained(&pool).await;
}

#[tokio::test]
async fn output_owner_unrelated_release_does_not_clear_original_timeout() {
    let pool = pool(1);
    let timer = VirtualTimer::new();
    let (port, mut session) = start_timed(&pool, timer.clone(), Duration::from_secs(10));
    ready(&port).await.committed(delta("a", 1));
    let retained = bounded(session.frames.next()).await.unwrap();
    state(
        &port,
        OutputReadinessState::OutputBlocked(OutputBlockReason::EventCredit),
    )
    .await;
    timer.registered(1).await;
    timer.advance(Duration::from_secs(6));
    let other_plan = plan();
    let other = RequestOutputBudget::open(&pool, limits(&other_plan, 1), other_plan).unwrap();
    drop(other);
    timer.registered(2).await;
    assert_eq!(
        port.readiness(),
        OutputReadinessState::OutputBlocked(OutputBlockReason::EventCredit)
    );
    timer.advance(Duration::from_secs(4));
    timeout_state(&port).await;
    drop(port);
    drop(bounded(session.frames.next()).await.unwrap());
    drop((
        bounded(session.completion).await.unwrap(),
        session.frames,
        retained,
    ));
    drained(&pool).await;
}

#[tokio::test]
async fn output_owner_invalid_or_overflowed_deadline_closes_instead_of_waiting_forever() {
    for timeout in [Duration::ZERO, Duration::MAX] {
        let pool = pool(1);
        let timer = VirtualTimer::new();
        let (port, mut session) = start_timed(&pool, timer, timeout);
        if !timeout.is_zero() {
            ready(&port).await.committed(delta("a", 1));
        }
        state(
            &port,
            OutputReadinessState::Closing(OutputCloseReason::InvalidOutput),
        )
        .await;
        drop(port);
        if !timeout.is_zero() {
            drop(bounded(session.frames.next()).await.unwrap());
        }
        let terminal = bounded(session.frames.next()).await.unwrap();
        assert!(terminal.metadata().terminal);
        drop((
            terminal,
            bounded(session.completion).await.unwrap(),
            session.frames,
        ));
        drained(&pool).await;
    }
}

#[tokio::test]
async fn output_owner_chat_terminal_projection_timeout_keeps_delivered_bytes_charged() {
    let pool = super::chat::chat_pool();
    let timer = VirtualTimer::new();
    let (mut port, mut session) = super::chat::start_chat(
        &pool,
        super::chat::chat_plan(4, true, false),
        OutputFlowRuntimeOptions {
            slow_consumer_timeout: Duration::from_secs(10),
            max_queued_events: SloOutputConfig::default()
                .max_queued_events_per_request
                .get(),
            timer: timer.clone(),
        },
    );
    ready(&port).await.committed(delta("reason", 1));
    let delivered = bounded(session.frames.next()).await.unwrap();
    assert!(port.terminal(terminal(1, 1, "</think>answer")).is_ok());
    // The first frame is out of the channel but still owns the only data
    // event. Final content must wait for it and must not evade the deadline.
    timer.registered(1).await;
    timer.advance(Duration::from_secs(10));
    timeout_state(&port).await;
    let failed = bounded(session.frames.next()).await.unwrap();
    assert!(failed.metadata().terminal);
    assert!(String::from_utf8_lossy(failed.wire().payload()).contains("blocking timeout"));
    drop(port);
    drop(bounded(session.completion).await.unwrap());
    drop(failed);
    assert_eq!(pool.snapshot().data_used.events, 1);
    assert!(pool.snapshot().data_used.bytes > 0);
    drop((delivered, session.frames));
    drained(&pool).await;
}
