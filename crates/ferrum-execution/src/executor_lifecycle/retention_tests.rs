use super::*;
use ferrum_runtime::{HostMemoryAccount, HostMemoryClass, HostMemoryClassLimits, HostMemoryLimits};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Barrier,
};

fn host() -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            128,
            HostMemoryClassLimits {
                input_preprocess_bytes: 128,
                output_bytes: 128,
                transfer_staging_bytes: 128,
                weight_backing_bytes: 128,
            },
        )
        .unwrap(),
        1,
    )
    .unwrap()
}

struct DropProbe(Arc<AtomicUsize>);
impl Drop for DropProbe {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

#[test]
fn retirement_retention_survives_all_observers_and_retained_reports() {
    let host = host();
    let scope = host
        .reserve_scope(&[(HostMemoryClass::Output, 32)])
        .unwrap();
    let view = scope.account();
    let (reporter, observer) = ExecutorShutdownObserver::channel(Arc::new(()));
    observer.retain_until_retired(scope);
    let second = observer.clone();
    drop(observer);
    drop(second);
    reporter.retained(FerrumError::backend("original cleanup is retained"));
    assert_eq!(host.usage().unwrap().committed_bytes(), 32);
    // The original granted view remains usable while cleanup retains custody.
    let allocation = view.allocate_zeroed(HostMemoryClass::Output, 8).unwrap();
    assert_eq!(host.usage().unwrap().allocated_bytes(), 8);
    drop(allocation);
    assert_eq!(host.usage().unwrap().committed_bytes(), 32);
    reporter.retired(None);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert!(view.allocate_zeroed(HostMemoryClass::Output, 1).is_err());
}

#[tokio::test]
async fn retirement_retention_build_cleanup_survives_cancelled_wait_and_error_drop() {
    let host = host();
    let scope = host
        .reserve_scope(&[(HostMemoryClass::WeightBacking, 48)])
        .unwrap();
    let (reporter, observer) = ExecutorShutdownObserver::channel(Arc::new(()));
    let error = crate::ExecutorBuildError::with_cleanup(
        FerrumError::backend("original initialization failed"),
        observer.clone(),
    );
    error.cleanup().unwrap().retain_until_retired(scope);
    let mut waiting = Box::pin(observer.wait());
    assert!(futures::poll!(&mut waiting).is_pending());
    drop(waiting);
    reporter.retained(FerrumError::backend("unknown native completion"));
    assert_eq!(
        observer.wait().await.phase(),
        ExecutorShutdownPhase::Retained
    );
    drop(error);
    drop(observer);
    assert_eq!(host.usage().unwrap().committed_bytes(), 48);
    reporter.retired(Some(FerrumError::backend(
        "execution failed but safely retired",
    )));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn retirement_retention_retired_with_prior_error_and_late_registration_release_once() {
    let drops = Arc::new(AtomicUsize::new(0));
    let (reporter, observer) = ExecutorShutdownObserver::channel(Arc::new(()));
    reporter.retain_until_retired(DropProbe(Arc::clone(&drops)));
    reporter.retained(FerrumError::backend("original error"));
    assert_eq!(drops.load(Ordering::SeqCst), 0);
    reporter.retired(None);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert_eq!(observer.snapshot().phase(), ExecutorShutdownPhase::Retired);
    assert!(observer.snapshot().result().is_err());
    observer.retain_until_retired(DropProbe(Arc::clone(&drops)));
    assert_eq!(drops.load(Ordering::SeqCst), 2);
    reporter.retired(None);
    reporter.retained(FerrumError::backend("late failed notification"));
    drop(observer);
    drop(reporter);
    assert_eq!(drops.load(Ordering::SeqCst), 2);
}

#[test]
fn retirement_retention_registration_and_retirement_are_linearized() {
    let host = host();
    let scope = host
        .reserve_scope(&[(HostMemoryClass::InputPreprocess, 64)])
        .unwrap();
    let drops = Arc::new(AtomicUsize::new(0));
    let (reporter, observer) = ExecutorShutdownObserver::channel(Arc::new(()));
    let registering = observer.clone();
    let barrier = Arc::new(Barrier::new(3));
    let register_barrier = Arc::clone(&barrier);
    let drop_probe = DropProbe(Arc::clone(&drops));
    let register = std::thread::spawn(move || {
        register_barrier.wait();
        registering.retain_until_retired((scope, drop_probe));
    });
    let retire_barrier = Arc::clone(&barrier);
    let retire = std::thread::spawn(move || {
        retire_barrier.wait();
        reporter.retired(None);
    });
    barrier.wait();
    register.join().unwrap();
    retire.join().unwrap();
    assert_eq!(observer.snapshot().phase(), ExecutorShutdownPhase::Retired);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn retirement_retention_concurrent_terminal_error_is_preserved_during_release() {
    struct BlockingDrop {
        entered: Arc<Barrier>,
        release: Arc<Barrier>,
        _scope: ferrum_runtime::HostMemoryScope,
    }
    impl Drop for BlockingDrop {
        fn drop(&mut self) {
            self.entered.wait();
            self.release.wait();
        }
    }
    let host = host();
    let scope = host
        .reserve_scope(&[(HostMemoryClass::Output, 32)])
        .unwrap();
    let entered = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let (reporter, observer) = ExecutorShutdownObserver::channel(Arc::new(()));
    observer.retain_until_retired(BlockingDrop {
        entered: Arc::clone(&entered),
        release: Arc::clone(&release),
        _scope: scope,
    });
    let first = reporter.clone();
    let retiring = std::thread::spawn(move || first.retired(None));
    entered.wait();
    assert_eq!(host.usage().unwrap().committed_bytes(), 32);
    let error = FerrumError::backend("original execution failure reported during resource release");
    reporter.retired(Some(error.clone()));
    let during = observer.snapshot();
    assert_ne!(during.phase(), ExecutorShutdownPhase::Retired);
    assert_eq!(during.error().unwrap().to_string(), error.to_string());
    release.wait();
    retiring.join().unwrap();
    let terminal = observer.snapshot();
    assert_eq!(terminal.phase(), ExecutorShutdownPhase::Retired);
    assert_eq!(
        terminal.result().unwrap_err().to_string(),
        error.to_string()
    );
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn retirement_retention_user_destructors_can_observe_and_register_without_lock_cycle() {
    struct RegisterOnDrop {
        observer: ExecutorShutdownObserver,
        drops: Arc<AtomicUsize>,
    }
    impl Drop for RegisterOnDrop {
        fn drop(&mut self) {
            let _actual_report = self.observer.snapshot();
            self.observer
                .retain_until_retired(DropProbe(Arc::clone(&self.drops)));
        }
    }
    let drops = Arc::new(AtomicUsize::new(0));
    let (reporter, observer) = ExecutorShutdownObserver::channel(Arc::new(()));
    observer.retain_until_retired(RegisterOnDrop {
        observer: observer.clone(),
        drops: Arc::clone(&drops),
    });
    reporter.retired(None);
    assert_eq!(drops.load(Ordering::SeqCst), 1);
    assert_eq!(observer.snapshot().phase(), ExecutorShutdownPhase::Retired);
}

#[test]
fn retirement_retention_missing_publisher_never_manufactures_release() {
    // Deliberate per-instance quarantine: no original publisher remains to
    // prove retirement. A tiny probe avoids retaining a model or native buffer.
    let drops = Arc::new(AtomicUsize::new(0));
    let (reporter, observer) = ExecutorShutdownObserver::channel(Arc::new(()));
    observer.retain_until_retired(DropProbe(Arc::clone(&drops)));
    drop(reporter);
    assert_eq!(observer.snapshot().phase(), ExecutorShutdownPhase::Retained);
    assert!(observer.snapshot().result().is_err());
    drop(observer);
    assert_eq!(drops.load(Ordering::SeqCst), 0);
}
