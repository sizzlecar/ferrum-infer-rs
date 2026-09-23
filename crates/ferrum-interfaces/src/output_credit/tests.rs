use super::*;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Barrier,
};

fn amount(events: usize, bytes: usize, projection_bytes: usize) -> OutputCreditAmount {
    OutputCreditAmount {
        events,
        bytes,
        projection_bytes,
    }
}

fn account_limits() -> OutputAccountLimits {
    OutputAccountLimits {
        maximum: amount(5, 100, 50),
        terminal: amount(1, 20, 0),
    }
}

fn pool_limits() -> OutputPoolLimits {
    OutputPoolLimits {
        maximum: amount(10, 200, 100),
        max_total_bytes: 300,
        max_open_accounts: 2,
    }
}

fn reserve(
    account: &OutputCreditAccount,
    lane: OutputCreditLane,
    amount: OutputCreditAmount,
) -> OutputReservation {
    match account.try_reserve(lane, amount).unwrap() {
        OutputCreditAttempt::Reserved(lease) => lease,
        OutputCreditAttempt::Full(_) => panic!("unexpected capacity pressure"),
    }
}

fn full(
    account: &OutputCreditAccount,
    lane: OutputCreditLane,
    amount: OutputCreditAmount,
) -> OutputCreditWake {
    match account.try_reserve(lane, amount).unwrap() {
        OutputCreditAttempt::Full(wake) => wake,
        OutputCreditAttempt::Reserved(_) => panic!("unexpected reservation"),
    }
}

#[test]
fn output_credit_byte_event_and_projection_boundaries_are_independent() {
    for (held, denied) in [
        (amount(4, 1, 0), amount(1, 1, 0)),
        (amount(1, 80, 0), amount(1, 1, 0)),
        (amount(0, 0, 50), amount(0, 0, 1)),
    ] {
        let pool = OutputCreditPool::new(pool_limits()).unwrap();
        let account = pool
            .open_request(RequestId::new(), account_limits())
            .unwrap();
        let lease = reserve(&account, OutputCreditLane::Data, held);
        let _wake = full(&account, OutputCreditLane::Data, denied);
        assert_eq!(account.snapshot().data_used, held);
        assert_eq!(pool.snapshot().data_used, held);
        drop(lease);
        assert_eq!(pool.snapshot().data_used, OutputCreditAmount::ZERO);
    }
}

#[test]
fn output_credit_global_total_couples_payload_projection_and_terminal_escrow() {
    let mut limits = pool_limits();
    limits.max_total_bytes = 180;
    let pool = OutputCreditPool::new(limits).unwrap();
    let first = pool
        .open_request(RequestId::new(), account_limits())
        .unwrap();
    let second = pool
        .open_request(RequestId::new(), account_limits())
        .unwrap();
    let a = reserve(&first, OutputCreditLane::Data, amount(1, 60, 20));
    let b = reserve(&second, OutputCreditLane::Data, amount(1, 30, 30));
    // 60+20 + 30+30 + two 20-byte terminal escrows = exactly 180.
    let _wake = full(&second, OutputCreditLane::Data, amount(0, 1, 0));
    let terminal_a = reserve(&first, OutputCreditLane::Terminal, amount(1, 20, 0));
    let terminal_b = reserve(&second, OutputCreditLane::Terminal, amount(1, 20, 0));
    assert_eq!(pool.snapshot().terminal_held, amount(2, 40, 0));
    assert_eq!(pool.snapshot().terminal_used, amount(2, 40, 0));
    drop((a, b, terminal_a, terminal_b, first, second));
    assert_eq!(pool.snapshot().retained_accounts, 0);
    assert_eq!(pool.snapshot().terminal_held, OutputCreditAmount::ZERO);
}

#[test]
fn output_credit_full_data_queue_cannot_take_reserved_terminal_capacity() {
    let pool = OutputCreditPool::new(pool_limits()).unwrap();
    let account = pool
        .open_request(RequestId::new(), account_limits())
        .unwrap();
    let data = reserve(&account, OutputCreditLane::Data, amount(4, 80, 50));
    let _wake = full(&account, OutputCreditLane::Data, amount(1, 1, 0));
    let terminal = reserve(&account, OutputCreditLane::Terminal, amount(1, 20, 0));
    assert!(
        account.snapshot().blocked_since.is_some(),
        "terminal progress must not reset data blockage"
    );
    assert_eq!(pool.snapshot().data_used, data.amount());
    assert_eq!(pool.snapshot().terminal_used, terminal.amount());
}

#[test]
fn output_credit_close_preserves_inflight_charges_and_reopened_generation() {
    let pool = OutputCreditPool::new(pool_limits()).unwrap();
    let request = RequestId::new();
    let old = pool
        .open_request(request.clone(), account_limits())
        .unwrap();
    let data = reserve(&old, OutputCreditLane::Data, amount(1, 40, 10));
    let terminal = reserve(&old, OutputCreditLane::Terminal, amount(1, 7, 0));
    old.close();
    assert!(matches!(
        old.try_reserve(OutputCreditLane::Data, amount(1, 1, 0)),
        Err(OutputCreditError::Closed)
    ));
    assert_eq!(pool.snapshot().terminal_held, amount(1, 7, 0));
    assert_eq!(pool.snapshot().data_used, amount(1, 40, 10));
    let new = pool.open_request(request, account_limits()).unwrap();
    assert_ne!(old.generation(), new.generation());
    let new_data = reserve(&new, OutputCreditLane::Data, amount(1, 10, 0));
    drop(old); // Must not close the new incarnation's active index.
    assert!(!new.snapshot().closed);
    drop((data, terminal));
    assert_eq!(pool.snapshot().data_used, new_data.amount());
    assert_eq!(pool.snapshot().terminal_held, account_limits().terminal);
    assert_eq!(pool.snapshot().retained_accounts, 1);
}

#[test]
fn output_credit_last_control_drop_closes_but_payload_keeps_charge() {
    let pool = OutputCreditPool::new(pool_limits()).unwrap();
    let account = pool
        .open_request(RequestId::new(), account_limits())
        .unwrap();
    let peer = account.clone();
    let lease = reserve(&account, OutputCreditLane::Data, amount(1, 40, 10));
    drop(account);
    assert_eq!(pool.snapshot().open_accounts, 1);
    drop(peer);
    assert_eq!(pool.snapshot().open_accounts, 0);
    assert_eq!(pool.snapshot().terminal_held, OutputCreditAmount::ZERO);
    assert_eq!(pool.snapshot().data_used, lease.amount());
    assert_eq!(pool.snapshot().retained_accounts, 1);
    drop(lease);
    assert_eq!(pool.snapshot().retained_accounts, 0);
}

#[test]
fn output_credit_split_shrink_merge_and_move_preserve_exact_ownership() {
    let pool = OutputCreditPool::new(pool_limits()).unwrap();
    let account = pool
        .open_request(RequestId::new(), account_limits())
        .unwrap();
    let mut lease = reserve(&account, OutputCreditLane::Data, amount(4, 80, 40));
    let mut split = lease.split(amount(1, 30, 10)).unwrap();
    assert_eq!(pool.snapshot().data_used, amount(4, 80, 40));
    split.shrink_to(amount(1, 20, 5)).unwrap();
    assert_eq!(pool.snapshot().data_used, amount(4, 70, 35));
    assert_eq!(
        lease.split(amount(4, 1, 0)).err(),
        Some(OutputCreditError::ExceedsReservation)
    );
    assert_eq!(
        lease.shrink_to(amount(4, 80, 40)).err(),
        Some(OutputCreditError::ExceedsReservation)
    );
    lease.try_merge(&mut split).unwrap();
    assert_eq!(split.amount(), OutputCreditAmount::ZERO);
    assert_eq!(lease.amount(), amount(4, 70, 35));
    drop(split);
    let payload = lease.into_output(vec![7u8; 70]);
    let projected = payload.map(|payload| payload.into_boxed_slice());
    assert_eq!(projected.payload().len(), 70);
    assert_eq!(pool.snapshot().data_used, projected.credit());
    drop(projected);
    assert_eq!(pool.snapshot().data_used, OutputCreditAmount::ZERO);
}

#[test]
fn output_credit_cross_pool_request_generation_and_lane_merge_do_not_mint_credit() {
    let pool = OutputCreditPool::new(OutputPoolLimits {
        max_open_accounts: 3,
        ..pool_limits()
    })
    .unwrap();
    let other_pool = OutputCreditPool::new(pool_limits()).unwrap();
    let request = RequestId::new();
    let first = pool
        .open_request(request.clone(), account_limits())
        .unwrap();
    let mut target = reserve(&first, OutputCreditLane::Data, amount(1, 10, 0));
    let other = pool
        .open_request(RequestId::new(), account_limits())
        .unwrap();
    let foreign_pool = other_pool
        .open_request(request.clone(), account_limits())
        .unwrap();
    let mut foreign_request_lease = reserve(&other, OutputCreditLane::Data, amount(1, 10, 0));
    let mut foreign_pool_lease = reserve(&foreign_pool, OutputCreditLane::Data, amount(1, 10, 0));
    let mut terminal = reserve(&first, OutputCreditLane::Terminal, amount(1, 10, 0));
    for foreign in [
        &mut foreign_request_lease,
        &mut foreign_pool_lease,
        &mut terminal,
    ] {
        assert_eq!(
            target.try_merge(foreign).err(),
            Some(OutputCreditError::ForeignReservation)
        );
        assert_eq!(foreign.amount(), amount(1, 10, 0));
    }
    first.close();
    let reopened = pool.open_request(request, account_limits()).unwrap();
    let mut new_generation = reserve(&reopened, OutputCreditLane::Data, amount(1, 10, 0));
    assert_eq!(
        target.try_merge(&mut new_generation).err(),
        Some(OutputCreditError::ForeignReservation)
    );
    assert_eq!(target.amount(), amount(1, 10, 0));
    assert_eq!(pool.snapshot().data_used, amount(3, 30, 0));
}

#[tokio::test]
async fn output_credit_release_and_close_before_waiting_cannot_lose_wakes() {
    let pool = OutputCreditPool::new(pool_limits()).unwrap();
    let account = pool
        .open_request(RequestId::new(), account_limits())
        .unwrap();
    let lease = reserve(&account, OutputCreditLane::Data, amount(4, 80, 0));
    let mut wake = full(&account, OutputCreditLane::Data, amount(1, 1, 0));
    drop(lease);
    assert!(futures::poll!(Box::pin(wake.changed())).is_ready());
    let lease = reserve(&account, OutputCreditLane::Data, amount(4, 80, 0));
    let mut wake = full(&account, OutputCreditLane::Data, amount(1, 1, 0));
    account.close();
    assert!(futures::poll!(Box::pin(wake.changed())).is_ready());
    assert!(matches!(
        account.try_reserve(OutputCreditLane::Data, amount(1, 1, 0)),
        Err(OutputCreditError::Closed)
    ));
    assert_eq!(pool.snapshot().data_used, lease.amount());
}

#[test]
fn output_credit_simultaneous_requests_and_cancellation_never_exceed_global_capacity() {
    let pool = OutputCreditPool::new(OutputPoolLimits {
        maximum: amount(16, 180, 100),
        max_total_bytes: 180,
        max_open_accounts: 4,
    })
    .unwrap();
    let accounts = (0..4)
        .map(|_| {
            pool.open_request(RequestId::new(), account_limits())
                .unwrap()
        })
        .collect::<Vec<_>>();
    let start = Arc::new(Barrier::new(5));
    let reserved = Arc::new(Barrier::new(5));
    let release = Arc::new(Barrier::new(5));
    let successes = Arc::new(AtomicUsize::new(0));
    let threads = accounts
        .into_iter()
        .map(|account| {
            let (start, reserved, release, successes) = (
                start.clone(),
                reserved.clone(),
                release.clone(),
                successes.clone(),
            );
            std::thread::spawn(move || {
                start.wait();
                let lease = match account
                    .try_reserve(OutputCreditLane::Data, amount(1, 40, 10))
                    .unwrap()
                {
                    OutputCreditAttempt::Reserved(lease) => {
                        successes.fetch_add(1, Ordering::SeqCst);
                        Some(lease)
                    }
                    OutputCreditAttempt::Full(_) => None,
                };
                reserved.wait();
                release.wait();
                account.close();
                drop(lease);
            })
        })
        .collect::<Vec<_>>();
    start.wait();
    reserved.wait();
    assert_eq!(successes.load(Ordering::SeqCst), 2); // 80 escrow + 2 * 50 data.
    let snapshot = pool.snapshot();
    assert_eq!(snapshot.data_used.total_bytes(), Some(100));
    assert_eq!(snapshot.terminal_held.bytes, 80);
    release.wait();
    for thread in threads {
        thread.join().unwrap();
    }
    let snapshot = pool.snapshot();
    assert_eq!(snapshot.data_used, OutputCreditAmount::ZERO);
    assert_eq!(snapshot.terminal_held, OutputCreditAmount::ZERO);
    assert_eq!(snapshot.retained_accounts, 0);
}

struct ManualClock {
    start: Instant,
    elapsed_ms: AtomicUsize,
}
impl OutputCreditClock for ManualClock {
    fn now(&self) -> Instant {
        self.start + Duration::from_millis(self.elapsed_ms.load(Ordering::SeqCst) as u64)
    }
}

#[test]
fn output_credit_slow_consumer_clock_tracks_blocking_without_canceling_or_resetting_on_terminal() {
    let clock = Arc::new(ManualClock {
        start: Instant::now(),
        elapsed_ms: AtomicUsize::new(0),
    });
    let pool = OutputCreditPool::with_clock(pool_limits(), clock.clone()).unwrap();
    let account = pool
        .open_request(RequestId::new(), account_limits())
        .unwrap();
    let mut lease = reserve(&account, OutputCreditLane::Data, amount(4, 80, 0));
    let _wake = full(&account, OutputCreditLane::Data, amount(1, 1, 0));
    clock.elapsed_ms.store(40, Ordering::SeqCst);
    let _wake = full(&account, OutputCreditLane::Data, amount(1, 1, 0));
    let terminal = reserve(&account, OutputCreditLane::Terminal, amount(1, 1, 0));
    assert_eq!(
        account.snapshot().blocked_for(),
        Some(Duration::from_millis(40))
    );
    lease.shrink_to(amount(3, 79, 0)).unwrap();
    assert_eq!(account.snapshot().last_release_at, clock.now());
    let new_data = reserve(&account, OutputCreditLane::Data, amount(1, 1, 0));
    assert_eq!(account.snapshot().blocked_for(), None);
    assert!(!account.snapshot().closed);
    drop((lease, new_data, terminal));
}

#[test]
fn output_credit_configuration_and_arithmetic_reject_before_mutating_ledger() {
    let config = SloOutputConfig::default();
    let account = OutputAccountLimits::from_slo(&config, NonZeroUsize::new(3).unwrap()).unwrap();
    assert_eq!(
        account.maximum.bytes,
        config.max_queued_bytes_per_request.get()
    );
    assert_eq!(account.terminal.events, 3);
    assert_eq!(
        account.data().bytes + account.terminal.bytes,
        account.maximum.bytes
    );
    assert!(OutputAccountLimits::from_slo(&config, config.max_queued_events_per_request).is_err());
    assert!(OutputPoolLimits::from_slo(&config, NonZeroUsize::new(usize::MAX).unwrap()).is_err());
    let mut insufficient = config.clone();
    insufficient.max_total_buffer_bytes = NonZeroUsize::new(
        config.max_queued_bytes_per_request.get() + config.max_projection_bytes_per_request.get(),
    )
    .unwrap();
    assert!(matches!(
        OutputPoolLimits::from_slo(&insufficient, NonZeroUsize::new(2).unwrap()),
        Err(OutputCreditError::InvalidConfiguration(_))
    ));
    let pool = OutputCreditPool::new(pool_limits()).unwrap();
    let account = pool
        .open_request(RequestId::new(), account_limits())
        .unwrap();
    assert!(matches!(
        account.try_reserve(OutputCreditLane::Data, amount(1, usize::MAX, 1)),
        Err(OutputCreditError::Overflow)
    ));
    assert!(matches!(
        account.try_reserve(OutputCreditLane::Data, amount(1, 81, 0)),
        Err(OutputCreditError::ExceedsLimit)
    ));
    assert!(matches!(
        account.try_reserve(OutputCreditLane::Data, OutputCreditAmount::ZERO),
        Err(OutputCreditError::EmptyAmount)
    ));
    assert_eq!(pool.snapshot().data_used, OutputCreditAmount::ZERO);
}

#[tokio::test]
async fn output_credit_closed_generation_retains_admission_slot_and_healthy_share() {
    let pool = OutputCreditPool::new(pool_limits()).unwrap();
    let request = RequestId::new();
    let slow = pool
        .open_request(request.clone(), account_limits())
        .unwrap();
    let healthy = pool
        .open_request(RequestId::new(), account_limits())
        .unwrap();
    let slow_data = reserve(&slow, OutputCreditLane::Data, amount(4, 80, 50));
    let slow_terminal = reserve(&slow, OutputCreditLane::Terminal, amount(1, 20, 0));
    slow.close();
    assert_eq!(pool.snapshot().open_accounts, 1);
    assert_eq!(pool.snapshot().retained_accounts, 2);
    // A new terminal escrow would fit now, but it could steal the healthy
    // request's configured data share. Closing cannot recycle this slot yet.
    assert!(matches!(
        pool.open_request(request.clone(), account_limits()),
        Err(OutputCreditError::AdmissionFull)
    ));
    let healthy_data = reserve(&healthy, OutputCreditLane::Data, amount(4, 80, 50));
    let healthy_terminal = reserve(&healthy, OutputCreditLane::Terminal, amount(1, 20, 0));
    assert_eq!(pool.snapshot().data_used, amount(8, 160, 100));
    assert_eq!(pool.snapshot().terminal_used, amount(2, 40, 0));
    drop(slow_data);
    assert!(
        matches!(
            pool.open_request(request.clone(), account_limits()),
            Err(OutputCreditError::AdmissionFull)
        ),
        "terminal ownership alone must retain the old slot"
    );
    let mut wake = pool.subscribe();
    drop(slow_terminal);
    assert!(futures::poll!(Box::pin(wake.changed())).is_ready());
    let replacement = pool.open_request(request, account_limits()).unwrap();
    let replacement_data = reserve(&replacement, OutputCreditLane::Data, amount(4, 80, 50));
    assert_eq!(pool.snapshot().data_used, amount(8, 160, 100));
    assert_eq!(healthy.snapshot().data_used, healthy_data.amount());
    drop((
        healthy_data,
        healthy_terminal,
        replacement_data,
        slow,
        healthy,
        replacement,
    ));
    assert_eq!(pool.snapshot().retained_accounts, 0);
    assert_eq!(pool.snapshot().data_used, OutputCreditAmount::ZERO);
    assert_eq!(pool.snapshot().terminal_held, OutputCreditAmount::ZERO);
}

#[test]
fn output_credit_global_event_and_projection_limits_restrict_individually_valid_requests() {
    for (global, first_amount, second_amount) in [
        (amount(6, 200, 100), amount(3, 1, 0), amount(2, 1, 0)),
        (amount(10, 200, 60), amount(0, 0, 40), amount(0, 0, 21)),
    ] {
        let pool = OutputCreditPool::new(OutputPoolLimits {
            maximum: global,
            ..pool_limits()
        })
        .unwrap();
        let first = pool
            .open_request(RequestId::new(), account_limits())
            .unwrap();
        let second = pool
            .open_request(RequestId::new(), account_limits())
            .unwrap();
        let lease = reserve(&first, OutputCreditLane::Data, first_amount);
        let _wake = full(&second, OutputCreditLane::Data, second_amount);
        assert_eq!(pool.snapshot().data_used, first_amount);
        drop(lease);
        drop(reserve(&second, OutputCreditLane::Data, second_amount));
    }
}

#[test]
fn output_credit_close_racing_reservation_keeps_successful_ownership_charged() {
    let pool = OutputCreditPool::new(pool_limits()).unwrap();
    let account = pool
        .open_request(RequestId::new(), account_limits())
        .unwrap();
    let closer = account.clone();
    let start = Arc::new(Barrier::new(2));
    let peer_start = start.clone();
    let thread = std::thread::spawn(move || {
        peer_start.wait();
        closer.close();
    });
    start.wait();
    let attempt = account.try_reserve(OutputCreditLane::Data, amount(1, 40, 10));
    thread.join().unwrap();
    match attempt {
        Ok(OutputCreditAttempt::Reserved(lease)) => {
            assert_eq!(pool.snapshot().data_used, lease.amount());
            assert!(account.snapshot().closed);
            drop(lease);
        }
        Err(OutputCreditError::Closed) => {}
        _ => panic!("close race produced a result other than atomic reserve or rejection"),
    }
    assert_eq!(pool.snapshot().data_used, OutputCreditAmount::ZERO);
    assert_eq!(pool.snapshot().terminal_held, OutputCreditAmount::ZERO);
    assert_eq!(pool.snapshot().retained_accounts, 0);
}

#[test]
fn output_credit_terminal_shrink_returns_unused_escrow_only_after_close() {
    let pool = OutputCreditPool::new(pool_limits()).unwrap();
    let account = pool
        .open_request(RequestId::new(), account_limits())
        .unwrap();
    let mut terminal = reserve(&account, OutputCreditLane::Terminal, amount(1, 20, 0));
    terminal.shrink_to(amount(1, 5, 0)).unwrap();
    assert_eq!(pool.snapshot().terminal_used, amount(1, 5, 0));
    assert_eq!(pool.snapshot().terminal_held, amount(1, 20, 0));
    account.close();
    assert_eq!(pool.snapshot().terminal_held, amount(1, 5, 0));
    terminal.shrink_to(OutputCreditAmount::ZERO).unwrap();
    assert_eq!(pool.snapshot().terminal_held, OutputCreditAmount::ZERO);
    assert_eq!(pool.snapshot().retained_accounts, 0);
    drop(terminal); // Empty ownership must not access an already drained record.
}

#[test]
fn output_credit_payload_destructor_runs_before_its_capacity_is_returned() {
    struct Payload {
        pool: OutputCreditPool,
        observed: Arc<AtomicUsize>,
    }
    impl Drop for Payload {
        fn drop(&mut self) {
            self.observed
                .store(self.pool.snapshot().data_used.bytes, Ordering::SeqCst);
        }
    }
    let pool = OutputCreditPool::new(pool_limits()).unwrap();
    let account = pool
        .open_request(RequestId::new(), account_limits())
        .unwrap();
    let observed = Arc::new(AtomicUsize::new(0));
    let payload =
        reserve(&account, OutputCreditLane::Data, amount(1, 10, 0)).into_output(Payload {
            pool: pool.clone(),
            observed: observed.clone(),
        });
    drop(payload);
    assert_eq!(observed.load(Ordering::SeqCst), 10);
    assert_eq!(pool.snapshot().data_used, OutputCreditAmount::ZERO);
}
