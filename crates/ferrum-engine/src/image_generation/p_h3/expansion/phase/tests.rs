use super::*;
use crate::image_generation::p_h3::expansion::tests::fixture::*;
mod diagnostics;

async fn complete(
    phase: &mut NativeExpansionPhase,
    cancelled: bool,
) -> Result<ExpansionPhaseProgress> {
    tokio::time::timeout(
        std::time::Duration::from_secs(10),
        std::future::poll_fn(|cx| match phase.advance(cancelled, cx.waker()) {
            Ok(ExpansionPhaseProgress::Pending { made_progress }) => {
                if made_progress {
                    cx.waker().wake_by_ref();
                }
                Poll::Pending
            }
            result => Poll::Ready(result),
        }),
    )
    .await
    .expect("original native cleanup must notify its actual waiter")
}

#[tokio::test]
async fn owned_native_phase_yields_each_actual_qwen_wave_and_publishes_only_after_retirement() {
    let mut fixture = Fixture::new();
    let assets = fixture.take_expansion();
    let prepared = prepared(&assets, 2);
    let prefill_waves = prepared
        .tokens
        .len()
        .div_ceil(prepared.prefill_chunk_tokens);
    let (composition, runtime) = composition();
    let host = host();
    let requirements = assets
        .host_requirements(&composition, &prepared, &host)
        .unwrap();
    let grant = host.reserve_scope(&requirements.claims()).unwrap();
    let mut phase =
        NativeExpansionPhase::initialize(assets, &composition, prepared, grant).unwrap();
    let held_scope = phase.cohort.owner.scope().unwrap();
    assert!(held_scope.status().is_ready);
    assert_eq!(
        held_scope.cache_metrics_snapshot().unwrap()["counters"]["submitted_waves"],
        0
    );
    let wake = futures::task::noop_waker();
    // Real backing maintenance also returns to the caller before retrying
    // admission. No prefill wave may be consumed during these transitions.
    let admission_deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while matches!(
        phase.cohort.participants[0].state,
        Some(State::Admission(_))
    ) {
        assert!(std::time::Instant::now() < admission_deadline);
        assert!(matches!(
            phase.advance(false, &wake).unwrap(),
            ExpansionPhaseProgress::Pending {
                made_progress: true
            }
        ));
    }
    assert!(matches!(
        phase.cohort.participants[0].state,
        Some(State::Tokens(_))
    ));
    // The actual token geometry fixes the number of prefill forwards; no
    // advance is allowed to consume the subsequent decode wave as well.
    for _ in 0..prefill_waves {
        assert!(matches!(
            phase.advance(false, &wake).unwrap(),
            ExpansionPhaseProgress::Pending {
                made_progress: true
            }
        ));
        assert!(matches!(
            phase.cohort.participants[0].state,
            Some(State::Tokens(_))
        ));
    }
    assert!(matches!(
        phase.advance(false, &wake).unwrap(),
        ExpansionPhaseProgress::Pending {
            made_progress: true
        }
    ));
    assert!(matches!(
        phase.cohort.state,
        Some(CohortState::Retiring { .. })
    ));
    assert!(matches!(
        phase.advance(false, &wake).unwrap(),
        ExpansionPhaseProgress::Pending {
            made_progress: false
        }
    ));
    assert!(host.usage().unwrap().committed_bytes() > 0);
    assert!(runtime.resident_bytes() > 0);
    drop(held_scope);
    let ExpansionPhaseProgress::Expanded(text) = complete(&mut phase, false).await.unwrap() else {
        panic!("valid EOS must publish expanded text")
    };
    assert_eq!(text, "A small red cat sits quietly.");
    assert_eq!(
        phase.cleanup().snapshot().phase(),
        ExecutorShutdownPhase::Retired
    );
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(runtime.resident_bytes(), 0);
}

#[tokio::test]
async fn owned_native_phase_cancel_waits_for_original_scope_retirement() {
    let mut fixture = Fixture::new();
    let assets = fixture.take_expansion();
    let prepared = prepared(&assets, 2);
    let (composition, runtime) = composition();
    let host = host();
    let requirements = assets
        .host_requirements(&composition, &prepared, &host)
        .unwrap();
    let grant = host.reserve_scope(&requirements.claims()).unwrap();
    let mut phase =
        NativeExpansionPhase::initialize(assets, &composition, prepared, grant).unwrap();
    let held_scope = phase.cohort.owner.scope().unwrap();
    let wake = futures::task::noop_waker();
    assert!(matches!(
        phase.advance(true, &wake).unwrap(),
        ExpansionPhaseProgress::Pending {
            made_progress: true
        }
    ));
    assert!(matches!(
        phase.advance(true, &wake).unwrap(),
        ExpansionPhaseProgress::Pending {
            made_progress: false
        }
    ));
    assert!(host.usage().unwrap().committed_bytes() > 0);
    drop(held_scope);
    assert!(matches!(
        complete(&mut phase, true).await.unwrap(),
        ExpansionPhaseProgress::Cancelled
    ));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(runtime.resident_bytes(), 0);
}

#[tokio::test]
async fn dropped_owned_native_phase_preserves_grant_until_original_cleanup_finishes() {
    let mut fixture = Fixture::new();
    let assets = fixture.take_expansion();
    let prepared = prepared(&assets, 2);
    let (composition, runtime) = composition();
    let host = host();
    let requirements = assets
        .host_requirements(&composition, &prepared, &host)
        .unwrap();
    let grant = host.reserve_scope(&requirements.claims()).unwrap();
    let phase = NativeExpansionPhase::initialize(assets, &composition, prepared, grant).unwrap();
    let held_scope = phase.cohort.owner.scope().unwrap();
    let cleanup = phase.cleanup();
    drop(phase);
    assert!(host.usage().unwrap().committed_bytes() > 0);
    assert_ne!(cleanup.snapshot().phase(), ExecutorShutdownPhase::Retired);
    drop(held_scope);
    tokio::time::timeout(std::time::Duration::from_secs(10), cleanup.wait())
        .await
        .unwrap()
        .result()
        .unwrap();
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(runtime.resident_bytes(), 0);
}

#[tokio::test]
async fn owned_native_initialization_failure_at_actual_conversion_boundary_refunds_phase() {
    let mut fixture = Fixture::new();
    let assets = fixture.take_expansion();
    let prepared = prepared(&assets, 2);
    let (composition, runtime) = composition();
    let host = host();
    let requirements = assets
        .host_requirements(&composition, &prepared, &host)
        .unwrap();
    let mut claims = requirements.claims();
    let (_, bytes) = claims
        .iter_mut()
        .find(|(class, _)| *class == ferrum_runtime::HostMemoryClass::WeightBacking)
        .unwrap();
    assert!(*bytes > 0);
    *bytes -= 1;
    let grant = host.reserve_scope(&claims).unwrap();
    let error = match NativeExpansionPhase::initialize(assets, &composition, prepared, grant) {
        Ok(_) => panic!("conversion must consume its actual original phase quota"),
        Err(error) => error,
    };
    if let Some(cleanup) = error.cleanup() {
        let report = tokio::time::timeout(std::time::Duration::from_secs(10), cleanup.wait())
            .await
            .unwrap();
        assert_eq!(report.phase(), ExecutorShutdownPhase::Retired);
    }
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(runtime.resident_bytes(), 0);
}
