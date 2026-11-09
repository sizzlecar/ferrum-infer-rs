//! Distinct visual/RoPE inputs through real native cohort execution.
use super::*;
#[cfg(feature = "cuda")]
mod cuda;
mod fixture;
mod metadata;
mod replay;
use ferrum_execution::{ManagedCausalIoWork, ManagedCausalReadback};
use ferrum_runtime::{
    HostMemoryClass, HostMemoryClassLimits, HostMemoryError, HostMemoryLimits,
    HostMemoryPressureScope,
};
use fixture::*;

#[derive(Debug, PartialEq, Eq)]
struct Observed {
    tokens: Vec<Vec<u32>>,
    text: Vec<Option<String>>,
}

async fn run(
    fixture: &VisualFixture,
    cases: &[VisualCase],
    cancel: Option<(usize, bool)>,
) -> Observed {
    run_with_host(fixture, cases, cancel, host()).await
}

async fn run_with_host(
    fixture: &VisualFixture,
    cases: &[VisualCase],
    cancel: Option<(usize, bool)>,
    host: HostMemoryAccount,
) -> Observed {
    let (composition, runtime) = composition();
    let result = run_on(
        fixture,
        cases,
        cancel,
        host,
        &composition,
        ferrum_types::Device::CPU,
        || assert_eq!(runtime.resident_bytes(), 0),
    )
    .await;
    assert_eq!(runtime.resident_bytes(), 0);
    result
}

async fn run_on<R: DeviceRuntime>(
    fixture: &VisualFixture,
    cases: &[VisualCase],
    cancel: Option<(usize, bool)>,
    host: HostMemoryAccount,
    composition: &RuntimeComposition<R>,
    device: ferrum_types::Device,
    observe_retired: impl FnOnce(),
) -> Observed {
    let mut prepared = cohort(
        fixture,
        cases
            .iter()
            .map(|&c| fixture.prepare_on(c, device.clone()))
            .collect(),
    );
    let requirements = fixture
        .assets
        .cohort_host_requirements(composition, &prepared, &host)
        .unwrap();
    let persistent: usize = prepared
        .requests
        .iter()
        .map(|r| {
            r.visual
                .as_ref()
                .unwrap()
                .retained_input_bytes(&host)
                .unwrap()
        })
        .sum();
    let phase = host.reserve_scope(&requirements.claims()).unwrap();
    for (request, &case) in prepared.requests.iter_mut().zip(cases) {
        bind(request, &phase.account(), case);
    }
    assert_eq!(
        host.usage()
            .unwrap()
            .class_allocated_bytes(HostMemoryClass::InputPreprocess),
        persistent
    );
    let mut cohort = NativeExpansionCohort::initialize(
        Arc::clone(&fixture.assets),
        composition,
        prepared,
        phase,
    )
    .unwrap();
    let identities: std::collections::HashSet<_> = cohort
        .participants
        .iter()
        .map(|p| p.request.clone())
        .collect();
    assert_eq!(identities.len(), cases.len());
    let mut cancelled = vec![false; cases.len()];
    if !matches!(cancel, Some((_, false))) {
        advance_until_observed(&mut cohort, &cancelled, |c| {
            c.participants
                .iter()
                .all(|p| matches!(p.state, Some(ParticipantState::Tokens(_))))
        })
        .await;
        let original = cohort.owner.scope().unwrap();
        let before = original.cache_metrics_snapshot().unwrap()["counters"]["submitted_waves"]
            .as_u64()
            .unwrap();
        cohort
            .advance(&cancelled, &futures::task::noop_waker())
            .unwrap();
        assert_eq!(
            original.cache_metrics_snapshot().unwrap()["counters"]["submitted_waves"]
                .as_u64()
                .unwrap(),
            before + 1
        );
        for (participant, case) in cohort.participants.iter().zip(cases) {
            assert!(
                matches!(&participant.state, Some(ParticipantState::Tokens(cursor)) if cursor.consumed_tokens() == case.chunk)
            );
        }
        drop(original);
        if matches!(cancel, Some((_, true))) {
            // Both members must have consumed their own image rows into KV,
            // not merely been admitted or executed leading ordinary tokens.
            advance_until_observed(&mut cohort, &cancelled, |c| {
                c.participants.iter().all(|p| {
                    let run = &p.prepared.visual.as_ref().unwrap().tokens().image_runs()[0];
                    let end = run.token_start() + run.rows();
                    match &p.state {
                        Some(ParticipantState::Tokens(cursor)) => cursor.consumed_tokens() >= end,
                        _ => false,
                    }
                })
            })
            .await;
        }
    }
    if let Some((index, after_kv)) = cancel {
        cancelled[index] = true;
        cohort
            .advance(&cancelled, &futures::task::noop_waker())
            .unwrap();
        assert!(matches!(
            cohort.participants[index].result,
            Some(ExpansionParticipantResult::Cancelled)
        ));
        if after_kv {
            // CUDA may return a held close. Observe its original terminal
            // proof through the real wake path before asserting retirement.
            advance_until_observed(&mut cohort, &cancelled, |c| {
                c.participants[index].ended.is_some()
            })
            .await;
        } else {
            assert!(cohort.participants[index].ended.is_none());
        }
        assert!(matches!(cohort.state, Some(CohortState::Running)));
    }
    advance_until_observed(&mut cohort, &cancelled, |c| {
        matches!(c.state, Some(CohortState::Retiring { .. }))
    })
    .await;
    let tokens = cohort
        .participants
        .iter()
        .map(|p| {
            p.completed
                .as_ref()
                .map(|done| done.history.ids().to_vec())
                .unwrap_or_default()
        })
        .collect();
    for (index, participant) in cohort.participants.iter().enumerate() {
        if !cancelled[index] || matches!(cancel, Some((_, true))) {
            assert!(participant
                .ended
                .as_ref()
                .unwrap()
                .sequence()
                .observed_end()
                .is_some());
        }
    }
    let results = complete(&mut cohort, &cancelled).await.unwrap();
    let text = results
        .into_iter()
        .enumerate()
        .map(|(index, result)| match result {
            ExpansionParticipantResult::Expanded(text) => {
                assert!(!cancelled[index]);
                Some(text)
            }
            ExpansionParticipantResult::Cancelled => {
                assert!(cancelled[index]);
                None
            }
            ExpansionParticipantResult::Failed(error) => {
                panic!("native visual member {index} failed: {error}")
            }
        })
        .collect();
    assert_eq!(
        cohort.cleanup().snapshot().phase(),
        ExecutorShutdownPhase::Retired
    );
    assert!(cohort.owner.scope().is_err());
    observe_retired();
    // The prepared requests still own their immutable Host feature tables.
    // Dropping the retired adapter must return these original grants as well.
    drop(cohort);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    Observed { tokens, text }
}

async fn advance_until_observed(
    cohort: &mut NativeExpansionCohort,
    cancelled: &[bool],
    condition: impl Fn(&NativeExpansionCohort) -> bool,
) {
    tokio::time::timeout(
        std::time::Duration::from_secs(60),
        std::future::poll_fn(|cx| {
            if condition(cohort) {
                return Poll::Ready(());
            }
            match cohort.advance(cancelled, cx.waker()).unwrap() {
                ExpansionCohortProgress::Pending { made_progress } => {
                    if condition(cohort) {
                        Poll::Ready(())
                    } else {
                        if made_progress {
                            cx.waker().wake_by_ref();
                        }
                        Poll::Pending
                    }
                }
                ExpansionCohortProgress::Retired(_) => {
                    panic!("native cohort retired before its required observed boundary")
                }
            }
        }),
    )
    .await
    .expect("original native boundary must progress or wake its waiter");
}

#[tokio::test]
async fn visual_native_cohort_matches_independent_members_and_reversed_order() {
    let fixture = VisualFixture::new();
    let a = run(&fixture, &[MIXED], None).await;
    let b = run(&fixture, &[OTHER], None).await;
    // These are numerical oracles, not equality against an equally broken B1:
    // zero attention, ignored feature rows or ignored explicit RoPE fail here.
    assert_eq!(a.tokens, [vec![SECOND_ANSWER_ID, EOS_ID]]);
    assert_eq!(b.tokens, [vec![ANSWER_ID, EOS_ID]]);
    assert_eq!(a.text, [Some(NEGATIVE_TEXT.to_owned())]);
    assert_eq!(b.text, [Some(POSITIVE_TEXT.to_owned())]);
    let pair = run(&fixture, &[MIXED, OTHER], None).await;
    assert_eq!(pair.tokens, [a.tokens[0].clone(), b.tokens[0].clone()]);
    assert_eq!(pair.text, [a.text[0].clone(), b.text[0].clone()]);
    let reverse = run(&fixture, &[OTHER, MIXED], None).await;
    assert_eq!(reverse.tokens, [b.tokens[0].clone(), a.tokens[0].clone()]);
    assert_eq!(reverse.text, [b.text[0].clone(), a.text[0].clone()]);
}

#[tokio::test]
async fn visual_native_cohort_one_feature_or_rope_change_cannot_contaminate_its_peer() {
    let fixture = VisualFixture::new();
    let baseline = run(&fixture, &[MIXED, OTHER], None).await;
    assert_eq!(
        baseline.tokens,
        [vec![SECOND_ANSWER_ID, EOS_ID], vec![ANSWER_ID, EOS_ID]]
    );
    for changed in [
        VisualCase {
            signs: &[1., 1.],
            ..MIXED
        },
        VisualCase {
            quarter_turn: -1.,
            ..MIXED
        },
    ] {
        let single = run(&fixture, &[changed], None).await;
        assert_eq!(single.tokens, [vec![ANSWER_ID, EOS_ID]]);
        assert_ne!(single.tokens[0], baseline.tokens[0]);
        let pair = run(&fixture, &[changed, OTHER], None).await;
        assert_eq!(pair.tokens[0], single.tokens[0]);
        assert_eq!(pair.text[0], single.text[0]);
        assert_eq!(pair.tokens[1], baseline.tokens[1]);
        assert_eq!(pair.text[1], baseline.text[1]);
    }
}

#[tokio::test]
async fn visual_native_cohort_cancel_before_admission_or_after_visual_kv_retires_original_owners() {
    let fixture = VisualFixture::new();
    // Matching chunks keep both members live after the longer image run;
    // unequal prompt extents still exercise different physical KV frontiers.
    let cases = [MIXED, VisualCase { chunk: 2, ..OTHER }];
    let baseline = run(&fixture, &cases, None).await;
    assert_eq!(
        baseline.tokens,
        [vec![SECOND_ANSWER_ID, EOS_ID], vec![ANSWER_ID, EOS_ID]]
    );
    for index in 0..2 {
        for after_kv in [false, true] {
            let actual = run(&fixture, &cases, Some((index, after_kv))).await;
            let survivor = 1 - index;
            assert!(actual.tokens[index].is_empty());
            assert_eq!(actual.text[index], None);
            assert_eq!(actual.tokens[survivor], baseline.tokens[survivor]);
            assert_eq!(actual.text[survivor], baseline.text[survivor]);
        }
    }
}

#[tokio::test]
async fn visual_native_cohort_budget_counts_each_original_input_and_rejects_one_byte_short() {
    let fixture = VisualFixture::new();
    let cases = [MIXED, OTHER];
    let mut prepared = cohort(
        &fixture,
        cases.iter().map(|&c| fixture.prepare(c)).collect(),
    );
    let (composition, runtime) = composition();
    let accounting = host();
    let requirements = fixture
        .assets
        .cohort_host_requirements(&composition, &prepared, &accounting)
        .unwrap();

    // Measure actual original allocations, not the production helper's sum.
    // The second image has a different row count and continuation-table size.
    bind(&mut prepared.requests[0], &accounting, cases[0]);
    let first_bytes = accounting
        .usage()
        .unwrap()
        .class_allocated_bytes(HostMemoryClass::InputPreprocess);
    bind(&mut prepared.requests[1], &accounting, cases[1]);
    let retained_bytes = accounting
        .usage()
        .unwrap()
        .class_allocated_bytes(HostMemoryClass::InputPreprocess);
    assert!(retained_bytes > first_bytes && first_bytes > 0);

    let wave_input = {
        let source = fixture
            .assets
            .selected_model(&prepared.requests[0])
            .unwrap()
            .weight_source(&accounting);
        let compiled = fixture
            .assets
            .compile_with_engine(
                &composition,
                &source,
                &prepared.requests[0],
                &prepared.engine,
            )
            .unwrap();
        let query = |work: &[ManagedCausalIoWork]| {
            compiled
                .managed_io_requirements(
                    &accounting,
                    ManagedCausalIoProfile::ManagedGreedyF16V1,
                    work,
                )
                .unwrap()
                .input_bytes
        };
        let prefill = cases.map(|case| ManagedCausalIoWork {
            immediate_tokens: case.chunk,
            mask_upload: true,
            output: ManagedCausalReadback::F16Logits,
        });
        let decode = cases.map(|_| ManagedCausalIoWork {
            immediate_tokens: 1,
            mask_upload: true,
            output: ManagedCausalReadback::GreedyToken,
        });
        query(&prefill).max(query(&decode))
    };
    assert_eq!(requirements.budget.input, wave_input + retained_bytes);
    assert_ne!(requirements.budget.input, wave_input + first_bytes);
    assert_eq!(runtime.resident_bytes(), 0);
    drop(prepared);
    assert_eq!(accounting.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(accounting.usage().unwrap().committed_bytes(), 0);

    let budget = requirements.budget;
    let bounded_host = |input| {
        HostMemoryAccount::with_allocation_alignment(
            HostMemoryLimits::new(
                budget.input + budget.output + budget.staging + budget.weights,
                HostMemoryClassLimits {
                    input_preprocess_bytes: input,
                    output_bytes: budget.output,
                    transfer_staging_bytes: budget.staging,
                    weight_backing_bytes: budget.weights,
                },
            )
            .unwrap(),
            16,
        )
        .unwrap()
    };
    // Exact declared class ceilings run the real two-member executor to EOS.
    let exact = bounded_host(budget.input);
    let actual = run_with_host(&fixture, &cases, None, exact.clone()).await;
    assert_eq!(
        actual.tokens,
        [vec![SECOND_ANSWER_ID, EOS_ID], vec![ANSWER_ID, EOS_ID]]
    );
    assert_eq!(exact.usage().unwrap().committed_bytes(), 0);

    let short = bounded_host(budget.input - 1);
    let prepared = cohort(
        &fixture,
        cases.iter().map(|&c| fixture.prepare(c)).collect(),
    );
    let required = fixture
        .assets
        .cohort_host_requirements(&composition, &prepared, &short)
        .unwrap();
    assert_eq!(required.budget, budget);
    // This is a declared admission boundary, not a claim that every wave
    // reaches its conservative peak. Refusal precedes any native upload.
    assert!(matches!(
        short.reserve_scope(&required.claims()),
        Err(HostMemoryError::Capacity {
            scope: HostMemoryPressureScope::Class(HostMemoryClass::InputPreprocess),
            requested,
            committed: 0,
            limit,
        }) if requested == budget.input && limit == budget.input - 1
    ));
    assert_eq!(short.usage().unwrap().committed_bytes(), 0);
    assert_eq!(short.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(runtime.resident_bytes(), 0);
}
