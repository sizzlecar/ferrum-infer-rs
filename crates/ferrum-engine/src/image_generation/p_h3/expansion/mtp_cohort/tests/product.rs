//! The shared product wrapper owns the same RuntimeComposition from declared
//! preparation through original retirement, rather than only a ready machine.
use super::*;
use crate::image_generation::p_h3::expansion::product::ProductExpansionCohort;
use ferrum_execution::executor_lifecycle::ExecutorShutdownPhase;

#[tokio::test]
async fn mtp_product_wrapper_runs_original_assets_and_reports_real_retirement() {
    let mut fixture = Fixture::new();
    let (composition, runtime) = composition();
    let host = host();
    let (_, answers) = ordinary(&fixture, &host, &composition).await;
    // Configure the original source owner before publishing any Arc clones.
    // Neither the verified mapping nor the tokenizer/template is reconstructed.
    let assets =
        Arc::get_mut(&mut fixture.assets).expect("fixture has not shared its expansion assets");
    assets.configure_mtp(PH3ImagePromptExpansionMtp::OneDraft);
    let prepared = fixture
        .assets
        .prepare_text_cohort(fixture.requests())
        .unwrap();
    let requirements = fixture
        .assets
        .cohort_host_requirements(&composition, &prepared, &host)
        .unwrap();
    let phase = host.reserve_scope(&requirements.claims()).unwrap();
    let mut product = ProductExpansionCohort::initialize_with_diagnostics(
        Arc::clone(&fixture.assets),
        composition,
        prepared,
        phase,
        vec![None, None],
    )
    .unwrap();
    assert!(matches!(&product, ProductExpansionCohort::Mtp(_)));
    assert_eq!(
        runtime.resident_bytes(),
        0,
        "native initialization is polled by the original owner"
    );
    let observer = product.cleanup();
    assert_eq!(observer.snapshot().phase(), ExecutorShutdownPhase::Draining);
    observer.retry();
    assert_eq!(observer.snapshot().phase(), ExecutorShutdownPhase::Draining);
    let mut retried_live_native = false;
    let results = tokio::time::timeout(
        Duration::from_secs(10),
        std::future::poll_fn(|cx| {
            if runtime.resident_bytes() > 0 && !retried_live_native {
                observer.retry();
                assert_eq!(observer.snapshot().phase(), ExecutorShutdownPhase::Draining);
                assert!(runtime.resident_bytes() > 0);
                retried_live_native = true;
            }
            match product.advance(&[false, false], cx.waker()).unwrap() {
                ExpansionCohortProgress::Pending { made_progress } => {
                    if made_progress {
                        cx.waker().wake_by_ref();
                    }
                    Poll::Pending
                }
                ExpansionCohortProgress::Retired(results) => Poll::Ready(results),
            }
        }),
    )
    .await
    .expect("the product wrapper must notify real Core progress");
    assert!(retried_live_native);
    assert_eq!(results.len(), answers.len());
    for (result, expected) in results.into_iter().zip(answers) {
        assert!(matches!(result, ExpansionParticipantResult::Expanded(text) if text == expected));
    }
    observer.snapshot().result().unwrap();
    assert_eq!(observer.snapshot().phase(), ExecutorShutdownPhase::Retired);
    assert_eq!(runtime.resident_bytes(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    drop(product);
    observer.snapshot().result().unwrap();
}

#[test]
fn mtp_product_wrapper_cancel_before_initialization_refunds_original_phase() {
    let mut fixture = Fixture::new();
    Arc::get_mut(&mut fixture.assets)
        .unwrap()
        .configure_mtp(PH3ImagePromptExpansionMtp::OneDraft);
    let (composition, runtime) = composition();
    let host = host();
    let prepared = fixture
        .assets
        .prepare_text_cohort(fixture.requests())
        .unwrap();
    let requirements = fixture
        .assets
        .cohort_host_requirements(&composition, &prepared, &host)
        .unwrap();
    let phase = host.reserve_scope(&requirements.claims()).unwrap();
    assert!(host.usage().unwrap().committed_bytes() > 0);
    let mut product = ProductExpansionCohort::initialize_with_diagnostics(
        Arc::clone(&fixture.assets),
        composition,
        prepared,
        phase,
        vec![None, None],
    )
    .unwrap();
    let observer = product.cleanup();
    assert_eq!(observer.snapshot().phase(), ExecutorShutdownPhase::Draining);
    assert_eq!(runtime.resident_bytes(), 0);
    observer.retry();
    assert_eq!(observer.snapshot().phase(), ExecutorShutdownPhase::Draining);
    assert!(host.usage().unwrap().committed_bytes() > 0);
    let ExpansionCohortProgress::Retired(results) = product
        .advance(&[true, true], &futures::task::noop_waker())
        .unwrap()
    else {
        panic!("no native resource has been acquired; cancellation can retire now")
    };
    assert_eq!(results.len(), 2);
    assert!(results
        .into_iter()
        .all(|result| matches!(result, ExpansionParticipantResult::Cancelled)));
    assert_eq!(observer.snapshot().phase(), ExecutorShutdownPhase::Retired);
    observer.snapshot().result().unwrap();
    assert_eq!(runtime.resident_bytes(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    drop(product);
    observer.snapshot().result().unwrap();
}

#[tokio::test]
async fn mtp_product_wrapper_latches_partial_cancel_during_initialization() {
    let mut fixture = Fixture::new();
    let (composition, runtime) = composition();
    let host = host();
    let (_, answers) = ordinary(&fixture, &host, &composition).await;
    Arc::get_mut(&mut fixture.assets)
        .unwrap()
        .configure_mtp(PH3ImagePromptExpansionMtp::OneDraft);
    let prepared = fixture
        .assets
        .prepare_text_cohort(fixture.requests())
        .unwrap();
    let requirements = fixture
        .assets
        .cohort_host_requirements(&composition, &prepared, &host)
        .unwrap();
    let phase = host.reserve_scope(&requirements.claims()).unwrap();
    let mut product = ProductExpansionCohort::initialize_with_diagnostics(
        Arc::clone(&fixture.assets),
        composition,
        prepared,
        phase,
        vec![None, None],
    )
    .unwrap();
    let observer = product.cleanup();
    assert!(matches!(
        product
            .advance(&[true, false], &futures::task::noop_waker())
            .unwrap(),
        ExpansionCohortProgress::Pending {
            made_progress: true
        }
    ));
    assert_eq!(
        runtime.resident_bytes(),
        0,
        "only input preparation has happened"
    );
    // A one-poll pulse precedes creation of the paired machine. Its subsequent
    // initialization and compute polls deliberately receive no cancellation.
    let mut results = tokio::time::timeout(
        Duration::from_secs(10),
        std::future::poll_fn(
            |cx| match product.advance(&[false, false], cx.waker()).unwrap() {
                ExpansionCohortProgress::Pending { made_progress } => {
                    if made_progress {
                        cx.waker().wake_by_ref();
                    }
                    Poll::Pending
                }
                ExpansionCohortProgress::Retired(results) => Poll::Ready(results),
            },
        ),
    )
    .await
    .expect("survivor must retain actual Core progress after the initialization pulse");
    assert_eq!(results.len(), 2);
    assert!(matches!(
        results.remove(0),
        ExpansionParticipantResult::Cancelled
    ));
    assert!(
        matches!(results.remove(0), ExpansionParticipantResult::Expanded(text) if text == answers[1])
    );
    assert_eq!(observer.snapshot().phase(), ExecutorShutdownPhase::Retired);
    observer.snapshot().result().unwrap();
    assert_eq!(runtime.resident_bytes(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
}

#[tokio::test]
async fn mtp_product_wrapper_invalid_membership_retires_partial_installation_with_first_error() {
    let mut fixture = Fixture::new();
    Arc::get_mut(&mut fixture.assets)
        .unwrap()
        .configure_mtp(PH3ImagePromptExpansionMtp::OneDraft);
    let (composition, runtime) = composition();
    let host = host();
    let prepared = fixture
        .assets
        .prepare_text_cohort(fixture.requests())
        .unwrap();
    let requirements = fixture
        .assets
        .cohort_host_requirements(&composition, &prepared, &host)
        .unwrap();
    let phase = host.reserve_scope(&requirements.claims()).unwrap();
    let mut product = ProductExpansionCohort::initialize_with_diagnostics(
        Arc::clone(&fixture.assets),
        composition,
        prepared,
        phase,
        vec![None, None],
    )
    .unwrap();
    let observer = product.cleanup();
    tokio::time::timeout(
        Duration::from_secs(10),
        std::future::poll_fn(|cx| {
            if runtime.resident_bytes() > 0 {
                return Poll::Ready(());
            }
            match product.advance(&[false, false], cx.waker()).unwrap() {
                ExpansionCohortProgress::Pending { made_progress } => {
                    if made_progress {
                        cx.waker().wake_by_ref();
                    }
                    Poll::Pending
                }
                ExpansionCohortProgress::Retired(_) => {
                    panic!("must acquire the real union before any request completes")
                }
            }
        }),
    )
    .await
    .expect("actual union installation must become resident");
    assert!(host.usage().unwrap().committed_bytes() > 0);
    let first = match product.advance(&[], &futures::task::noop_waker()) {
        Err(error) => error,
        Ok(_) => panic!("foreign cancellation membership must fail before another transition"),
    };
    assert_eq!(observer.snapshot().phase(), ExecutorShutdownPhase::Retired);
    assert_eq!(
        observer.snapshot().result().unwrap_err().to_string(),
        first.to_string()
    );
    assert_eq!(runtime.resident_bytes(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    assert!(product
        .advance(&[false, false], &futures::task::noop_waker())
        .is_err());
    observer.retry();
    drop(product);
    assert_eq!(observer.snapshot().phase(), ExecutorShutdownPhase::Retired);
    assert_eq!(
        observer.snapshot().result().unwrap_err().to_string(),
        first.to_string()
    );
}
