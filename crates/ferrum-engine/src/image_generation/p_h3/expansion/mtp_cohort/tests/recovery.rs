use super::*;

async fn first_prefill_wave(cohort: &mut MtpExpansionCohort<CpuDeviceRuntime>, with_frame: bool) {
    tokio::time::timeout(
        Duration::from_secs(10),
        std::future::poll_fn(|cx| {
            if let Some(active) = &cohort.active {
                // Both requests were sealed before admission. Maintenance of
                // one must not turn their first main wave into a Ready-only B1.
                assert_eq!(active.kind, WaveKind::Prefill);
                assert_eq!(active.participants, vec![0, 1]);
                if active.native.has_frame() == with_frame {
                    return Poll::Ready(());
                }
            }
            assert!(cohort.settling.is_none());
            match cohort.advance(&[false, false], cx.waker()).unwrap() {
                ExpansionCohortProgress::Pending { made_progress } => {
                    if made_progress {
                        cx.waker().wake_by_ref();
                    }
                    Poll::Pending
                }
                ExpansionCohortProgress::Retired(_) => {
                    panic!("cohort retired before its first prefill wave")
                }
            }
        }),
    )
    .await
    .expect("real admission maintenance must wake the first shared wave");
}

async fn cancel_at_first_wave(with_frame: bool) {
    let fixture = Fixture::new();
    let (composition, runtime) = composition();
    let host = host();
    let (golden, answers) = ordinary(&fixture, &host, &composition).await;
    let mut cohort = fixture.machine(&composition, &host);
    first_prefill_wave(&mut cohort, with_frame).await;
    let wake = futures::task::noop_waker();
    assert!(matches!(
        cohort.advance(&[false, true], &wake).unwrap(),
        ExpansionCohortProgress::Pending { .. }
    ));
    if with_frame {
        // An admitted frame cannot be aborted without poisoning its survivor.
        // Cancellation is latched, but the actual session stays Open until
        // compute has returned and the next poll performs selected settlement.
        assert!(cohort.cursors[1].as_ref().unwrap().finish.is_none());
    }
    // The external flag is deliberately a one-poll pulse. It must still be
    // honored if admission or computation needed more than that one poll.
    let (results, observed) = complete(&mut cohort, false).await;
    assert_eq!(observed.tokens[0], golden[0]);
    assert!(observed.tokens[1].is_empty());
    let mut results = results.into_iter();
    assert!(matches!(
        results.next().unwrap(),
        ExpansionParticipantResult::Expanded(text) if text == answers[0]
    ));
    assert!(matches!(
        results.next().unwrap(),
        ExpansionParticipantResult::Cancelled
    ));
    assert_eq!(runtime.resident_bytes(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[tokio::test]
async fn mtp_cohort_cancel_during_extension_reforms_wave_without_poisoning_survivor() {
    cancel_at_first_wave(false).await;
}

#[tokio::test]
async fn mtp_cohort_cancel_after_frame_admission_settles_original_frame_and_survivor() {
    cancel_at_first_wave(true).await;
}

#[test]
fn mtp_cohort_first_error_is_latched_until_original_owners_close() {
    let fixture = Fixture::new();
    let (composition, runtime) = composition();
    let host = host();
    let mut cohort = fixture.machine(&composition, &host);
    let wake = futures::task::noop_waker();
    let first = match cohort.advance(&[], &wake) {
        Err(error) => error,
        Ok(_) => panic!("wrong cancellation membership must fail"),
    };
    let repeated = match cohort.advance(&[false, false], &wake) {
        Err(error) => error,
        Ok(_) => panic!("a corrected call cannot restart a failed cohort"),
    };
    assert_eq!(repeated.to_string(), first.to_string());
    assert!(cohort.cursors.iter().all(Option::is_none));
    assert!(cohort.try_close().unwrap());
    assert_eq!(runtime.resident_bytes(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert!(cohort.advance(&[false, false], &wake).is_err());
}

#[test]
fn mtp_cohort_partial_external_close_cannot_resume_execution_or_publish_normal_results() {
    let fixture = Fixture::new();
    let (composition, runtime) = composition();
    let host = host();
    let mut cohort = fixture.machine(&composition, &host);
    // A real external plan owner prevents checked close; no fabricated native
    // fault, replacement resource account, or poisoned survivor is involved.
    let retained = Arc::clone(cohort.programs.main().unwrap().resources());
    assert!(!cohort.try_close().unwrap());
    assert!(cohort
        .advance(&[false, false], &futures::task::noop_waker())
        .is_err());
    assert!(cohort.results.iter().all(Option::is_none));
    drop(retained);
    assert!(cohort.try_close().unwrap());
    assert!(cohort.try_close().unwrap());
    assert_eq!(runtime.resident_bytes(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[tokio::test]
async fn mtp_cohort_partial_normal_retirement_preserves_results_until_real_close() {
    let fixture = Fixture::new();
    let (composition, runtime) = composition();
    let host = host();
    let (_, answers) = ordinary(&fixture, &host, &composition).await;
    let mut cohort = fixture.machine(&composition, &host);
    let retained = Arc::clone(cohort.programs.main().unwrap().resources());
    tokio::time::timeout(
        Duration::from_secs(10),
        std::future::poll_fn(
            |cx| match cohort.advance(&[false, false], cx.waker()).unwrap() {
                ExpansionCohortProgress::Pending { made_progress } => {
                    if cohort.results.iter().all(Option::is_some) {
                        return Poll::Ready(());
                    }
                    if made_progress {
                        cx.waker().wake_by_ref();
                    }
                    Poll::Pending
                }
                ExpansionCohortProgress::Retired(_) => {
                    panic!("external original plan owner still prevents close")
                }
            },
        ),
    )
    .await
    .expect("real requests must complete before their referenced plan closes");
    let wake = futures::task::noop_waker();
    assert!(matches!(
        cohort.advance(&[false, false], &wake).unwrap(),
        ExpansionCohortProgress::Pending { .. }
    ));
    drop(retained);
    let results = match cohort.advance(&[false, false], &wake).unwrap() {
        ExpansionCohortProgress::Retired(results) => results,
        ExpansionCohortProgress::Pending { .. } => panic!("all original owners are now releasable"),
    };
    for (result, expected) in results.into_iter().zip(answers) {
        assert!(matches!(result, ExpansionParticipantResult::Expanded(text) if text == expected));
    }
    assert!(cohort.advance(&[false, false], &wake).is_err());
    assert_eq!(runtime.resident_bytes(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
