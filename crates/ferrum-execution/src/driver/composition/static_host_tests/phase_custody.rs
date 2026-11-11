//! Real Stored static transfers using the same construction-phase guard as the
//! public retained initializer. Raw-panic quarantine intentionally cannot claim
//! recovery/zero usage; normal success/error paths must return every claim.
use super::*;
use crate::driver::composition::phase_custody::PhaseCustody;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn granted(host: &HostMemoryAccount, backing: usize, staging: usize) -> HostMemoryScope {
    host.reserve_scope(&[
        (HostMemoryClass::WeightBacking, backing),
        (HostMemoryClass::TransferStaging, staging),
    ])
    .unwrap()
}

#[test]
fn construction_phase_custody_pre_native_panic_refunds_the_original_scope() {
    let host = account(16, 8);
    let (family, plan, runtime) = setup(&host);
    let scope = granted(&host, 16, 8);
    let panic = catch_unwind(AssertUnwindSafe(|| {
        let _custody = PhaseCustody::new(scope);
        // The public constructor may still be resolving metadata/providers;
        // no original static provision/transaction has been acquired yet.
        assert_eq!(runtime.live.load(Ordering::SeqCst), 0);
        panic!("actual preparation callback panicked before native acquisition");
    }));
    assert!(panic.is_err());
    assert_eq!(runtime.live.load(Ordering::SeqCst), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    drop((family, plan));
}

#[tokio::test]
async fn construction_phase_custody_success_handoff_waits_for_original_transaction_retirement() {
    let host = account(16, 8);
    let scope = granted(&host, 16, 8);
    let scoped = scope.account();
    let source = source(scoped.clone());
    let (family, plan, runtime) = setup(&host);
    let custody = PhaseCustody::new(scope);
    custody.native_started().set(true);
    let initialized = initialize_static_with_host(
        committed(&plan, &runtime),
        &family,
        &plan,
        &source,
        Some(&scoped),
        policy(),
    )
    .unwrap();
    assert_eq!(initialized.receipt().uploaded_bytes(), 16);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 24);
    // The low-level Stored fixture has no causal execution operators. Retire
    // this actual successful installation via its original initialization
    // owner, using the same handoff to an authentic cleanup observer.
    let original = initialized.abort_activation(VNextError::InvalidExecutionPlan {
        reason: "fixture closes the successfully initialized original transaction".into(),
    });
    let cleanup = shutdown::begin_initialization(original, FerrumError::device("fixture shutdown"));
    custody.handoff(&cleanup);
    let report = tokio::time::timeout(Duration::from_secs(5), async {
        let mut cleanup = cleanup;
        loop {
            let report = cleanup.snapshot();
            if report.phase() == crate::ExecutorShutdownPhase::Retired {
                break report;
            }
            cleanup.changed().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(report.phase(), crate::ExecutorShutdownPhase::Retired);
    assert_eq!(runtime.live.load(Ordering::SeqCst), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert!(scoped
        .allocate_zeroed(HostMemoryClass::WeightBacking, 1)
        .is_err());
}

#[test]
fn construction_phase_custody_quiescent_error_releases_after_actual_synchronous_cleanup() {
    let host = account(15, 8);
    let scope = granted(&host, 15, 8);
    let scoped = scope.account();
    let source = source(scoped.clone());
    let (family, plan, runtime) = setup(&host);
    let custody = PhaseCustody::new(scope);
    custody.native_started().set(true);
    let failure = initialize_static_with_host(
        committed(&plan, &runtime),
        &family,
        &plan,
        &source,
        Some(&scoped),
        policy(),
    )
    .err()
    .expect("actual converter exceeds the granted backing by one byte");
    assert!(!failure.is_indeterminate());
    let error = shutdown::initialization_error(
        failure,
        FerrumError::device("actual conversion quota rejected"),
    );
    assert!(error.cleanup().is_none());
    custody.finish_error(&error);
    assert_eq!(runtime.live.load(Ordering::SeqCst), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[tokio::test]
async fn construction_phase_custody_unknown_fence_keeps_full_grant_until_original_cleanup() {
    let host = account(16, 8);
    let scope = granted(&host, 16, 8);
    let scoped = scope.account();
    let source = source(scoped.clone());
    let (family, plan, runtime) = setup(&host);
    runtime.unknown_upload_fence.store(true, Ordering::SeqCst);
    runtime.allow_cleanup.store(false, Ordering::SeqCst);
    let custody = PhaseCustody::new(scope);
    custody.native_started().set(true);
    let failure = initialize_static_with_host(
        committed(&plan, &runtime),
        &family,
        &plan,
        &source,
        Some(&scoped),
        policy(),
    )
    .err()
    .expect("real managed upload fence is indeterminate");
    assert!(failure.is_indeterminate());
    let error =
        shutdown::initialization_error(failure, FerrumError::device("original upload unknown"));
    let mut cleanup = error.cleanup().unwrap().clone();
    custody.finish_error(&error);
    let report = tokio::time::timeout(Duration::from_secs(5), cleanup.wait())
        .await
        .unwrap();
    assert_eq!(report.phase(), crate::ExecutorShutdownPhase::Retained);
    drop(error);
    assert!(runtime.live.load(Ordering::SeqCst) > 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 8);
    assert_eq!(host.usage().unwrap().committed_bytes(), 24);
    let submissions = runtime.trace.lock().unwrap().submissions;
    runtime.allow_cleanup.store(true, Ordering::SeqCst);
    cleanup.retry();
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if cleanup.snapshot().phase() == crate::ExecutorShutdownPhase::Retired {
                break;
            }
            cleanup.changed().await;
        }
    })
    .await
    .unwrap();
    assert_eq!(runtime.trace.lock().unwrap().submissions, submissions);
    assert_eq!(runtime.live.load(Ordering::SeqCst), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn construction_phase_custody_real_source_panic_preserves_original_grant_and_unwind() {
    struct PanicSource;
    impl WeightComponentSource for PanicSource {
        fn component<'a>(
            &'a self,
            _: &WeightComponentSpec,
        ) -> std::result::Result<WeightComponentPayload<'a>, VNextError> {
            panic!("actual Stored source callback panicked");
        }
    }
    let host = account(16, 8);
    let scope = granted(&host, 16, 8);
    let scoped = scope.account();
    let (family, plan, runtime) = setup(&host);
    let panic = catch_unwind(AssertUnwindSafe(|| {
        let custody = PhaseCustody::new(scope);
        custody.native_started().set(true);
        let _actual = initialize_static_with_host(
            committed(&plan, &runtime),
            &family,
            &plan,
            &PanicSource,
            Some(&scoped),
            policy(),
        );
    }));
    assert!(
        panic.is_err(),
        "the original source panic must not be swallowed"
    );
    // Runtime's actual native-abandonment quarantine owns the original buffer
    // and capacity. There is no cleanup observer or fabricated retirement.
    assert!(runtime.live.load(Ordering::SeqCst) > 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 24);
    assert!(host
        .reserve_scope(&[(HostMemoryClass::WeightBacking, 1)])
        .is_err());
}
