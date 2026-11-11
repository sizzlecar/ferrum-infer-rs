//! Real CPU Qwen execution and resource lifetime tests. The separate fault
//! module exercises contract failures; it makes no numerical performance claim.
use super::*;
use crate::image_generation::p_h3::expansion::tests::fixture::*;
use ferrum_types::TokenId;

mod batching;
mod faults;
mod timing;
mod visual;

fn requests(assets: &NativeTextExpansion, first_max: u32) -> Vec<PreparedTextExpansion> {
    let mut first = prepared(assets, first_max);
    let mut second_engine = first.engine.clone();
    second_engine.runtime.max_model_len = Some(64);
    let mut second = assets
        .prepare_text("draw a dog", 1024, 1024, 2, 1, second_engine)
        .unwrap();
    // Explicit tiny-model numerical inputs: the two genuine embedding rows
    // select different answer tokens. Unequal prompt extents exercise separate
    // prefill frontiers; the original protocol still extracts actual EOS output.
    first.tokens = vec![TokenId::new(3); 3];
    second.tokens = vec![TokenId::new(0); 7];
    for request in [&mut first, &mut second] {
        request.engine.runtime.max_model_len =
            Some(request.tokens.len() + request.generation.max_new_tokens as usize);
    }
    vec![first, second]
}

async fn complete(
    cohort: &mut NativeExpansionCohort,
    cancelled: &[bool],
) -> Result<Vec<ExpansionParticipantResult>> {
    tokio::time::timeout(
        std::time::Duration::from_secs(10),
        std::future::poll_fn(|cx| match cohort.advance(cancelled, cx.waker()) {
            Ok(ExpansionCohortProgress::Pending { made_progress }) => {
                if made_progress {
                    cx.waker().wake_by_ref();
                }
                Poll::Pending
            }
            Ok(ExpansionCohortProgress::Retired(results)) => Poll::Ready(Ok(results)),
            Err(error) => Poll::Ready(Err(error)),
        }),
    )
    .await
    .expect("original cohort retirement must notify its actual waiter")
}

fn advance_until(
    cohort: &mut NativeExpansionCohort,
    cancelled: &[bool],
    condition: impl Fn(&NativeExpansionCohort) -> bool,
) {
    let deadline = Instant::now() + std::time::Duration::from_secs(10);
    while !condition(cohort) {
        assert!(Instant::now() < deadline);
        assert!(matches!(
            cohort
                .advance(cancelled, &futures::task::noop_waker())
                .unwrap(),
            ExpansionCohortProgress::Pending {
                made_progress: true
            }
        ));
    }
}

#[tokio::test]
async fn two_real_qwen_requests_share_one_installation_and_keep_independent_eos_histories() {
    check_two_real_requests(PH3ImagePromptExpansionSourceBacking::Fresh).await;
}

#[tokio::test]
async fn reusable_single_slot_cohort_keeps_distinct_real_eos_histories() {
    check_two_real_requests(PH3ImagePromptExpansionSourceBacking::ReusableSingleSlot).await;
}

async fn check_two_real_requests(policy: PH3ImagePromptExpansionSourceBacking) {
    let mut fixture = Fixture::new_two_answers();
    let mut assets = fixture.take_expansion();
    Arc::get_mut(&mut assets)
        .unwrap()
        .configure_source_backing(policy)
        .unwrap();
    let prepared = assets.prepare_text_cohort(requests(&assets, 2)).unwrap();
    assert_eq!(prepared.len(), 2);
    let (composition, runtime) = composition();
    let host = host();
    let requirements = assets
        .cohort_host_requirements(&composition, &prepared, &host)
        .unwrap();
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(runtime.resident_bytes(), 0);
    let phase = host.reserve_scope(&requirements.claims()).unwrap();
    let mut cohort =
        NativeExpansionCohort::initialize(assets, &composition, prepared, phase).unwrap();
    let original = cohort.owner.scope().unwrap();
    assert!(original.status().is_ready);
    assert_eq!(
        original.cache_metrics_snapshot().unwrap()["counters"]["submitted_waves"],
        0
    );
    advance_until(&mut cohort, &[false, false], |c| {
        c.participants
            .iter()
            .all(|p| matches!(p.state, Some(ParticipantState::Tokens(_))))
    });
    let before = original.cache_metrics_snapshot().unwrap()["counters"]["submitted_waves"]
        .as_u64()
        .unwrap();
    cohort
        .advance(&[false, false], &futures::task::noop_waker())
        .unwrap();
    let frontiers: Vec<_> = cohort
        .participants
        .iter()
        .map(|p| match &p.state {
            Some(ParticipantState::Tokens(cursor)) => cursor.consumed_tokens(),
            _ => panic!("both first prefills must remain live"),
        })
        .collect();
    assert_eq!(frontiers, [2, 1]);
    assert_eq!(
        original.cache_metrics_snapshot().unwrap()["counters"]["submitted_waves"]
            .as_u64()
            .unwrap(),
        before + 1
    );
    advance_until(&mut cohort, &[false, false], |c| {
        c.participants[0].ended.is_some()
    });
    assert!(matches!(cohort.state, Some(CohortState::Running)));
    assert!(cohort.participants[1].ended.is_none());
    assert_eq!(
        cohort.participants[0]
            .completed
            .as_ref()
            .unwrap()
            .history
            .ids(),
        &[ANSWER_ID, EOS_ID]
    );
    // Both requests use this same actual executor installation. Completing A
    // neither seals its owner nor releases its phase while B remains active.
    assert!(std::ptr::eq(
        original.as_ref(),
        cohort.scope.as_ref().unwrap().as_ref()
    ));
    drop(cohort.owner.scope().unwrap());
    assert!(runtime.resident_bytes() > 0);
    assert!(host.usage().unwrap().committed_bytes() > 0);
    advance_until(&mut cohort, &[false, false], |c| {
        matches!(c.state, Some(CohortState::Retiring { .. }))
    });
    assert_eq!(
        cohort.participants[1]
            .completed
            .as_ref()
            .unwrap()
            .history
            .ids(),
        &[SECOND_ANSWER_ID, EOS_ID]
    );
    assert!(cohort.participants.iter().all(|p| p
        .ended
        .as_ref()
        .unwrap()
        .sequence()
        .observed_end()
        .is_some()));
    assert!(matches!(
        cohort
            .advance(&[false, false], &futures::task::noop_waker())
            .unwrap(),
        ExpansionCohortProgress::Pending {
            made_progress: false
        }
    ));
    assert!(host.usage().unwrap().allocated_bytes() > 0);
    drop(original);
    let results = complete(&mut cohort, &[false, false]).await.unwrap();
    assert!(
        matches!(&results[0], ExpansionParticipantResult::Expanded(text) if text == "A small red cat sits quietly.")
    );
    assert!(
        matches!(&results[1], ExpansionParticipantResult::Expanded(text) if text == "A blue dog runs quickly.")
    );
    assert_eq!(
        cohort.cleanup().snapshot().phase(),
        ExecutorShutdownPhase::Retired
    );
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(runtime.resident_bytes(), 0);
}

#[tokio::test]
async fn cancelling_one_request_before_or_after_admission_preserves_the_real_survivor() {
    check_cancellation(PH3ImagePromptExpansionSourceBacking::Fresh).await;
}

#[tokio::test]
async fn reusable_single_slot_cohort_cancellation_preserves_the_real_survivor() {
    check_cancellation(PH3ImagePromptExpansionSourceBacking::ReusableSingleSlot).await;
}

async fn check_cancellation(policy: PH3ImagePromptExpansionSourceBacking) {
    for after_admission in [false, true] {
        let mut fixture = Fixture::new_two_answers();
        let mut assets = fixture.take_expansion();
        Arc::get_mut(&mut assets)
            .unwrap()
            .configure_source_backing(policy)
            .unwrap();
        let prepared = assets.prepare_text_cohort(requests(&assets, 2)).unwrap();
        let (composition, runtime) = composition();
        let host = host();
        let requirements = assets
            .cohort_host_requirements(&composition, &prepared, &host)
            .unwrap();
        let phase = host.reserve_scope(&requirements.claims()).unwrap();
        let mut cohort =
            NativeExpansionCohort::initialize(assets, &composition, prepared, phase).unwrap();
        if after_admission {
            advance_until(&mut cohort, &[false, false], |c| {
                matches!(c.participants[0].state, Some(ParticipantState::Tokens(_)))
            });
            // Consume a real prefill wave before cancellation; no completion is
            // injected, and cancellation must close the original native state.
            advance_until(&mut cohort, &[false, false], |c| {
                c.scope.as_ref().unwrap().status().prefill_operations > 0
            });
        }
        cohort
            .advance(&[true, false], &futures::task::noop_waker())
            .unwrap();
        assert!(matches!(
            cohort.participants[0].result,
            Some(ExpansionParticipantResult::Cancelled)
        ));
        assert!(matches!(cohort.state, Some(CohortState::Running)));
        assert_eq!(cohort.participants[0].ended.is_some(), after_admission);
        assert!(runtime.resident_bytes() > 0);
        let results = complete(&mut cohort, &[true, false]).await.unwrap();
        assert!(matches!(results[0], ExpansionParticipantResult::Cancelled));
        assert!(
            matches!(&results[1], ExpansionParticipantResult::Expanded(text) if text == "A blue dog runs quickly.")
        );
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        assert_eq!(runtime.resident_bytes(), 0);
    }
}

#[tokio::test]
async fn one_length_failure_does_not_replace_another_requests_eos_or_history() {
    let mut fixture = Fixture::new_two_answers();
    let assets = fixture.take_expansion();
    let prepared = assets.prepare_text_cohort(requests(&assets, 1)).unwrap();
    let (composition, runtime) = composition();
    let host = host();
    let requirements = assets
        .cohort_host_requirements(&composition, &prepared, &host)
        .unwrap();
    let phase = host.reserve_scope(&requirements.claims()).unwrap();
    let mut cohort =
        NativeExpansionCohort::initialize(assets, &composition, prepared, phase).unwrap();
    advance_until(&mut cohort, &[false, false], |c| {
        c.participants[0].ended.is_some()
    });
    let failure = cohort.participants[0].failure.as_ref().unwrap();
    assert_eq!(failure.finish, FinishReason::Length);
    assert_eq!(failure.history.as_ref().unwrap().ids(), &[ANSWER_ID]);
    assert!(matches!(cohort.state, Some(CohortState::Running)));
    let results = complete(&mut cohort, &[false, false]).await.unwrap();
    assert!(matches!(results[0], ExpansionParticipantResult::Failed(_)));
    assert!(
        matches!(&results[1], ExpansionParticipantResult::Expanded(text) if text == "A blue dog runs quickly.")
    );
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(runtime.resident_bytes(), 0);
}

#[test]
fn cohort_rejects_incompatible_sources_policy_and_unbound_context_before_resources() {
    let fixture = Fixture::new_two_answers();
    let assets = fixture.expansion();
    assert!(assets.prepare_text_cohort(vec![]).is_err());
    let foreign = Fixture::new_two_answers();
    assert!(!assets
        .text_cohort_compatible(&prepared(assets, 2), &prepared(foreign.expansion(), 2))
        .unwrap());
    assert!(assets
        .prepare_text_cohort(vec![prepared(assets, 2), prepared(foreign.expansion(), 2)])
        .is_err());
    let mut requests = requests(assets, 2);
    assert!(assets
        .text_cohort_compatible(&requests[0], &requests[1])
        .unwrap());
    requests[1].engine.backend.device = ferrum_types::Device::CUDA(0);
    assert!(!assets
        .text_cohort_compatible(&requests[0], &requests[1])
        .unwrap());
    assert!(assets.prepare_text_cohort(requests).is_err());
    let mut requests = self::requests(assets, 2);
    requests[1].engine.runtime.max_model_len = Some(1);
    assert!(assets.prepare_text_cohort(requests).is_err());
}

#[tokio::test]
async fn both_real_admissions_require_the_aggregate_physical_history_grant() {
    for insufficient in [true, false] {
        let mut fixture = Fixture::new_two_answers();
        let assets = fixture.take_expansion();
        let prepared = assets.prepare_text_cohort(requests(&assets, 2)).unwrap();
        let (composition, runtime) = composition();
        let host = host();
        let history_bytes: usize = prepared
            .requests
            .iter()
            .map(|r| execute::history_backing_bytes(&host, r.generation.max_new_tokens).unwrap())
            .sum();
        let requirements = assets
            .cohort_host_requirements(&composition, &prepared, &host)
            .unwrap();
        assert!(requirements.budget.output >= history_bytes);
        let mut claims = requirements.claims();
        let (_, output) = claims
            .iter_mut()
            .find(|(class, _)| *class == ferrum_runtime::HostMemoryClass::Output)
            .unwrap();
        // Isolate the retained histories at real admission: no token wave or
        // transient readback runs in this test. Other classes keep their actual
        // compiled requirements, so the negative case reaches Output allocation.
        *output = history_bytes - usize::from(insufficient);
        let phase = host.reserve_scope(&claims).unwrap();
        let mut cohort =
            NativeExpansionCohort::initialize(assets, &composition, prepared, phase).unwrap();
        let deadline = Instant::now() + std::time::Duration::from_secs(10);
        while cohort.participants[1].failure.is_none()
            && cohort
                .participants
                .iter()
                .any(|p| matches!(p.state, Some(ParticipantState::Admission(_))))
        {
            assert!(Instant::now() < deadline);
            // Advance admission only. A ready participant is not given a token
            // wave until the other history has passed its capacity boundary.
            let index = cohort
                .participants
                .iter()
                .position(|p| matches!(p.state, Some(ParticipantState::Admission(_))))
                .unwrap();
            cohort.next = index;
            cohort
                .advance(&[false, false], &futures::task::noop_waker())
                .unwrap();
        }
        if insufficient {
            let failure = cohort.participants[1].failure.as_ref().unwrap();
            assert_eq!(
                failure.admission_probe,
                execute::AdmissionProbe::NeverEntered
            );
            assert!(failure.history.is_none());
            assert!(matches!(
                failure.error,
                FerrumError::ResourceExhausted { .. }
            ));
        } else {
            assert!(cohort
                .participants
                .iter()
                .all(|p| matches!(p.state, Some(ParticipantState::Tokens(_)))));
            assert!(cohort.participants.iter().all(|p| p.failure.is_none()));
        }
        complete(&mut cohort, &[true, true]).await.unwrap();
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
        assert_eq!(runtime.resident_bytes(), 0);
    }
}
