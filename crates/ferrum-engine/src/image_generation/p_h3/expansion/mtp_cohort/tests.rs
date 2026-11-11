//! Actual paired engine execution over the existing tiny BF16 source/template.
//! No synthetic completion permits, fixed-pool preprovisioning, or replacement
//! arithmetic. The plain main route supplies independent generated-token goldens.
use super::*;
use ferrum_kernels::backend::cpu::vnext_runtime::CpuDeviceRuntime;
use ferrum_types::TokenId;
use std::{task::Poll, time::Duration};

mod diagnostics;
mod fixture;
mod product;
mod recovery;
mod replay;
use fixture::*;

async fn ordinary(
    fixture: &Fixture,
    host: &HostMemoryAccount,
    composition: &RuntimeComposition<CpuDeviceRuntime>,
) -> (Vec<Vec<u32>>, Vec<String>) {
    let mut tokens = Vec::new();
    let mut answers = Vec::new();
    for prepared in fixture.requests() {
        let budget = fixture
            .assets
            .host_requirements(composition, &prepared, host)
            .unwrap();
        let phase = host.reserve_scope(&budget.claims()).unwrap();
        let phase_host = phase.account();
        let executor = fixture
            .assets
            .initialize(composition, &prepared, &phase_host)
            .unwrap();
        let owner = ferrum_execution::executor_lifecycle::ExecutorOwner::new(Box::new(executor));
        owner.shutdown_observer().retain_until_retired(phase);
        let scope = owner.scope().unwrap();
        let result = tokio::time::timeout(
            Duration::from_secs(10),
            execute::execute(
                &scope,
                &ferrum_types::RequestId::new(),
                &prepared.tokens,
                prepared.generation,
                prepared.prefill_chunk_tokens,
                &phase_host,
                &|| false,
            ),
        )
        .await
        .expect("ordinary Core waiter must make progress")
        .unwrap();
        tokens.push(result.history.ids().to_vec());
        answers.push(
            fixture
                .assets
                .extract(&prepared, result.history.ids(), result.finish)
                .unwrap(),
        );
        drop(result);
        drop(scope);
        owner.begin_shutdown().wait().await.result().unwrap();
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        assert_eq!(composition.runtime().resident_bytes(), 0);
    }
    (tokens, answers)
}

#[derive(Default)]
struct Observed {
    tokens: Vec<Vec<u32>>,
    prefill: usize,
    draft: usize,
    verify: usize,
    last: usize,
    b2: bool,
    cancelled_between_compute_and_commit: bool,
}

async fn complete(
    cohort: &mut MtpExpansionCohort<CpuDeviceRuntime>,
    cancel: bool,
) -> (Vec<ExpansionParticipantResult>, Observed) {
    let mut observed = Observed {
        tokens: vec![vec![]; 2],
        ..Observed::default()
    };
    let mut cancelled = [false, false];
    let result = tokio::time::timeout(
        Duration::from_secs(10),
        std::future::poll_fn(|cx| {
            // This is the actual completed native wave, before selected copy and
            // publication. Cancellation must not publish the cancelled member's h.
            if let Some(settling) = &cohort.settling {
                observed.b2 |= settling.indices.len() == 2;
                match settling.kind {
                    WaveKind::Prefill => observed.prefill += 1,
                    WaveKind::Draft => observed.draft += 1,
                    WaveKind::Verify => observed.verify += 1,
                    WaveKind::Last => observed.last += 1,
                }
                if cancel
                    && !observed.cancelled_between_compute_and_commit
                    && settling.indices.len() == 2
                {
                    cancelled[0] = true;
                    observed.cancelled_between_compute_and_commit = true;
                }
            }
            let progress = cohort.advance(&cancelled, cx.waker()).unwrap();
            for (index, cursor) in cohort.cursors.iter().enumerate() {
                if let Some(cursor) = cursor {
                    observed.tokens[index] = cursor.output.ids().to_vec();
                }
            }
            match progress {
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
    .expect("actual elastic Core admission/retirement must notify its waiter");
    (result, observed)
}

#[tokio::test]
async fn mtp_cohort_real_shared_pair_matches_ordinary_b2_multichunk_draft_and_last_slot() {
    let fixture = Fixture::new();
    let (composition, runtime) = composition();
    let host = host();
    let (golden, answers) = ordinary(&fixture, &host, &composition).await;
    assert_eq!(
        golden,
        [vec![ANSWER_ID, EOS_ID], vec![SECOND_ANSWER_ID, EOS_ID]]
    );
    let mut cohort = fixture.machine(&composition, &host);
    let (results, observed) = complete(&mut cohort, false).await;
    assert_eq!(observed.tokens, golden);
    assert!(observed.b2 && observed.prefill > 1);
    assert!(observed.draft > 0 && observed.verify > 0 && observed.last > 0);
    for (result, expected) in results.into_iter().zip(answers) {
        assert!(matches!(result,ExpansionParticipantResult::Expanded(text) if text == expected));
    }
    assert_eq!(runtime.resident_bytes(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
}

#[tokio::test]
async fn mtp_cohort_cancel_after_real_b2_compute_keeps_survivor_and_shared_owner() {
    let fixture = Fixture::new();
    let (composition, runtime) = composition();
    let host = host();
    let (golden, answers) = ordinary(&fixture, &host, &composition).await;
    let mut cohort = fixture.machine(&composition, &host);
    let (mut results, observed) = complete(&mut cohort, true).await;
    assert!(observed.cancelled_between_compute_and_commit && observed.b2);
    assert!(observed.tokens[0].is_empty());
    assert_eq!(observed.tokens[1], golden[1]);
    assert!(matches!(
        results.remove(0),
        ExpansionParticipantResult::Cancelled
    ));
    assert!(
        matches!(results.remove(0),ExpansionParticipantResult::Expanded(text) if text == answers[1])
    );
    assert_eq!(runtime.resident_bytes(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
}
