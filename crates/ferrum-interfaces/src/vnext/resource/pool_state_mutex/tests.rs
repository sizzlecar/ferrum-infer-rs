use super::*;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::mpsc;

#[test]
fn clean_reads_preserve_proof_and_foreign_mutex_never_matches() {
    let pool = PoolStateMutex::new(vec![7_u64, 11]);
    let foreign = PoolStateMutex::new(vec![7_u64, 11]);
    let original = {
        let guard = pool.lock().unwrap();
        assert_eq!(&*guard, &[7, 11]);
        guard.read_stamp().unwrap()
    };
    assert!(original.matches(&pool));
    assert!(!original.matches(&foreign));
    for _ in 0..3 {
        let guard = pool.lock().unwrap();
        // The shadow predicate and stamp observe the same locked state.
        assert!(original.matches_guard(&guard));
        assert_eq!(&*guard, &[7, 11]);
        assert!(guard.read_stamp().unwrap().matches(&pool));
    }
    assert!(original.matches(&pool));
}

#[test]
fn concurrent_first_mutable_borrow_invalidates_before_any_state_change() {
    let pool = PoolStateMutex::new(vec![7_u64]);
    let original = pool.lock().unwrap().read_stamp().unwrap();
    std::thread::scope(|scope| {
        let pool = &pool;
        let (locked_tx, locked_rx) = mpsc::channel();
        let (mutate_tx, mutate_rx) = mpsc::channel();
        let (dirty_tx, dirty_rx) = mpsc::channel();
        let (finish_tx, finish_rx) = mpsc::channel();
        let writer = scope.spawn(move || {
            let mut guard = pool.lock().unwrap();
            locked_tx.send(()).unwrap();
            if mutate_rx.recv().is_err() {
                return;
            }
            {
                let state: &mut Vec<u64> = &mut guard;
                // Merely obtaining the writable reference must invalidate.
                assert_eq!(state.as_slice(), &[7]);
            }
            assert!(guard.read_stamp().is_none());
            dirty_tx.send(()).unwrap();
            if finish_rx.recv().is_err() {
                return;
            }
            guard.push(13);
        });
        locked_rx.recv().unwrap();
        // A held clean read guard alone is not a mutation.
        assert!(original.matches(&pool));
        mutate_tx.send(()).unwrap();
        dirty_rx.recv().unwrap();
        // The writer still holds the mutex and has not changed the vector.
        assert!(!original.matches(&pool));
        finish_tx.send(()).unwrap();
        writer.join().unwrap();
    });
    let guard = pool.lock().unwrap();
    assert_eq!(&*guard, &[7, 13]);
    assert!(!original.matches_guard(&guard));
    assert!(guard.read_stamp().unwrap().matches_guard(&guard));
}

#[test]
fn restoring_equal_state_never_resurrects_a_pre_mutation_stamp() {
    let pool = PoolStateMutex::new(vec![3_u64, 5]);
    let mut guard = pool.lock().unwrap();
    let original = guard.read_stamp().unwrap();
    guard.push(17);
    assert!(guard.read_stamp().is_none());
    assert!(!original.matches(&pool));
    assert_eq!(guard.pop(), Some(17));
    drop(guard);
    let restored = pool.lock().unwrap();
    assert_eq!(&*restored, &[3, 5]);
    // Equality of the current value is not provenance for the old stamp.
    assert!(!original.matches_guard(&restored));
    assert!(restored.read_stamp().unwrap().matches_guard(&restored));
}

#[test]
fn panic_of_read_or_write_guard_disables_reuse_and_poison_recovery_stays_wrapped() {
    for mutate in [false, true] {
        let pool = PoolStateMutex::new(vec![7_u64]);
        let original = pool.lock().unwrap().read_stamp().unwrap();
        let unwind = catch_unwind(AssertUnwindSafe(|| {
            let mut guard = pool.lock().unwrap();
            if mutate {
                guard.push(11);
            } else {
                assert_eq!(&*guard, &[7]);
            }
            panic!("exercise pool guard unwind");
        }));
        assert!(unwind.is_err());
        assert!(!original.matches(&pool));
        assert_eq!(pool.version.load(Ordering::Acquire), DISABLED);
        let mut recovered = match pool.lock() {
            Err(poisoned) => poisoned.into_inner(),
            Ok(_) => panic!("the real mutex must retain its panic poison"),
        };
        assert!(recovered.read_stamp().is_none());
        recovered.clear();
        recovered.push(7);
        assert!(recovered.read_stamp().is_none());
        drop(recovered);
        assert!(!original.matches(&pool));
        assert_eq!(pool.version.load(Ordering::Acquire), DISABLED);
        assert!(pool.lock().is_err());
    }
}

#[test]
fn saturated_version_permanently_disables_reuse_without_aba() {
    let pool = PoolStateMutex::new(vec![1_u64]);
    // Exercise the actual arithmetic boundary without a production setter.
    pool.version.store(DISABLED - 3, Ordering::Relaxed);
    let previous = pool.lock().unwrap().read_stamp().unwrap();
    {
        let mut guard = pool.lock().unwrap();
        guard.push(2);
        assert!(guard.read_stamp().is_none());
        assert!(!previous.matches(&pool));
    }
    let last = {
        let guard = pool.lock().unwrap();
        assert_eq!(pool.version.load(Ordering::Acquire), DISABLED - 1);
        guard.read_stamp().unwrap()
    };
    assert!(last.matches(&pool));
    {
        let mut guard = pool.lock().unwrap();
        guard.pop();
        assert_eq!(pool.version.load(Ordering::Acquire), DISABLED);
        assert!(guard.read_stamp().is_none());
    }
    for _ in 0..2 {
        let mut guard = pool.lock().unwrap();
        assert!(guard.read_stamp().is_none());
        guard.push(3);
        guard.pop();
    }
    assert_eq!(pool.version.load(Ordering::Acquire), DISABLED);
    assert!(!previous.matches(&pool));
    assert!(!last.matches(&pool));
    assert_eq!(&*pool.lock().unwrap(), &[1]);
}

#[test]
fn returned_error_after_mutation_publishes_a_new_observation() {
    fn mutate_then_fail(pool: &PoolStateMutex<Vec<u64>>) -> Result<(), ()> {
        let mut guard = pool.lock().unwrap();
        guard.push(19);
        Err(())
    }

    let pool = PoolStateMutex::new(vec![7_u64]);
    let original = pool.lock().unwrap().read_stamp().unwrap();
    assert!(mutate_then_fail(&pool).is_err());
    let guard = pool.lock().unwrap();
    assert_eq!(&*guard, &[7, 19]);
    assert!(!original.matches_guard(&guard));
    assert!(guard.read_stamp().unwrap().matches_guard(&guard));
}
