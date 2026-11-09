//! Current visual inputs and KV through repeated typed decode work. The real
//! EOS suite remains separate: teacher forcing makes enough live work for a
//! capture and subsequent hit without weakening its EOS assertions.
use super::*;
use ferrum_execution::{
    model_executor::{
        ExecutorPrefillAdmission, ExecutorSamplingOutput, LogitsReturnPolicy, PrefillChunk,
    },
    token_execution::*,
};

pub(super) async fn run<R: DeviceRuntime>(
    fixture: &VisualFixture,
    composition: &RuntimeComposition<R>,
    device: ferrum_types::Device,
    cases: &[VisualCase],
    cancel_after_four: Option<usize>,
) -> (Vec<Vec<u32>>, serde_json::Value) {
    let mut prepared = cohort(
        fixture,
        cases
            .iter()
            .map(|&case| {
                let mut request = fixture.prepare_with_limit(case, device.clone(), 12);
                // One actual full prefill, then current one-token decode inputs.
                request.prefill_chunk_tokens = request.tokens.len();
                request.engine.batching.max_num_batched_tokens = request.tokens.len();
                request
            })
            .collect(),
    );
    let host = host();
    let requirements = fixture
        .assets
        .cohort_host_requirements(composition, &prepared, &host)
        .unwrap();
    let phase = host.reserve_scope(&requirements.claims()).unwrap();
    for (request, &case) in prepared.requests.iter_mut().zip(cases) {
        bind_varying_positions(request, &phase.account(), case);
    }
    let owner = fixture
        .assets
        .initialize_owned_with_engine(composition, &prepared.requests[0], &prepared.engine, phase)
        .unwrap();
    let scope = owner.scope().unwrap();
    assert!(scope.status().is_ready);
    let mut frontiers = Vec::new();
    for request in &prepared.requests {
        let id = RequestId::new();
        let frontier = tokio::time::timeout(std::time::Duration::from_secs(10), async {
            loop {
                let context = ExecutorPrefillAdmission::for_product_request(
                    &id, &request.tokens, request.tokens.len() + 12, request.tokens.len(), 0,
                ).unwrap();
                match scope.admit_tokens_with_inputs(
                    ExecutionAdmission { context, adapter: None },
                    Arc::clone(request.causal_inputs.as_ref().unwrap()),
                ).await.unwrap() {
                    TokenAdmissionOutcome::Admitted(frontier) => break frontier,
                    TokenAdmissionOutcome::MaintenanceDeferred(_) => {
                        assert!(matches!(scope.maintain_prefill_backing(&id).unwrap(),
                            ferrum_execution::model_executor::ExecutorPrefillMaintenanceOutcome::Maintained { .. }
                            | ferrum_execution::model_executor::ExecutorPrefillMaintenanceOutcome::RetryAdmission { .. }));
                    }
                    other => panic!("visual admission failed: {other:?}"),
                }
            }
        }).await.expect("original visual admission must complete");
        frontiers.push(Some(frontier));
    }
    let policy = LogitsReturnPolicy::GreedyArgmax {
        token_mask: None,
        repetition_penalty: None,
    };
    let mut tokens = vec![Vec::new(); cases.len()];
    let decode_tokens = [0, 1, 2, 3, 0, 1, 2, 3, 0];
    for step in 0..=decode_tokens.len() {
        if step == 5 {
            if let Some(index) = cancel_after_four {
                if prepared.engine.backend.enable_reusable_execution {
                    assert!(
                        scope.cache_metrics_snapshot().unwrap()["counters"]["reusable_execution"]
                            ["direct_waves"]
                            .as_u64()
                            .unwrap()
                            > 0,
                        "the cancellation fixture must reach a real graph hit first"
                    );
                }
                let frontier = frontiers[index].take().unwrap();
                assert!(matches!(
                    scope
                        .cancel_tokens(frontier.sequence())
                        .unwrap()
                        .disposition,
                    ExecutionCloseDisposition::Ended(_)
                ));
            }
        }
        let live: Vec<_> = frontiers
            .iter()
            .enumerate()
            .filter_map(|(i, f)| f.as_ref().map(|f| (i, f)))
            .collect();
        let work: Vec<_> = live
            .iter()
            .map(|&(index, frontier)| {
                let prompt = prepared.requests[index].tokens.len();
                let work = if step == 0 {
                    TokenWork::Prefill(PrefillChunk::new(0, prompt, prompt).unwrap())
                } else {
                    TokenWork::Decode {
                        token: TokenId::new(decode_tokens[step - 1]),
                        position: prompt + step - 1,
                    }
                };
                ExecutionWork::new(frontier, work, &policy).unwrap()
            })
            .collect();
        let ExecutionBatchOutcome::Completed(batch) = scope.execute_tokens(&work).await.unwrap()
        else {
            panic!("visual replay work must complete at its original frontier")
        };
        let completions = batch.into_completions();
        assert_eq!(completions.len(), live.len());
        let indices: Vec<_> = live.iter().map(|&(i, _)| i).collect();
        drop(work);
        drop(live);
        for (index, completed) in indices.into_iter().zip(completions) {
            let ExecutionProduct::Sampling(ExecutorSamplingOutput::GreedyToken(token)) =
                completed.product()
            else {
                panic!("original managed greedy readback required")
            };
            tokens[index].push(token.get());
            frontiers[index] = Some(completed.next_frontier().clone());
        }
    }
    let metrics = scope.cache_metrics_snapshot().unwrap();
    for frontier in frontiers.into_iter().flatten() {
        assert!(matches!(
            scope
                .cancel_tokens(frontier.sequence())
                .unwrap()
                .disposition,
            ExecutionCloseDisposition::Ended(_)
        ));
    }
    drop(scope);
    owner.begin_shutdown().wait().await.result().unwrap();
    drop(prepared);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    (tokens, metrics)
}

#[test]
fn visual_decode_replay_preparation_keeps_original_inputs_and_precision() {
    let eager = VisualFixture::new();
    let replay = VisualFixture::new().with_decode_replay();
    let a = eager.prepare(MIXED);
    let b = replay.prepare(MIXED);
    assert!(!a.engine.backend.enable_reusable_execution);
    assert!(b.engine.backend.enable_reusable_execution);
    assert_eq!(a.tokens, b.tokens);
    assert_eq!(a.generation.eos_token_id, b.generation.eos_token_id);
    assert_eq!(a.engine.numerical_execution, b.engine.numerical_execution);
    assert_eq!(
        a.visual.as_ref().unwrap().schema(),
        b.visual.as_ref().unwrap().schema()
    );
}

#[tokio::test]
async fn visual_multi_decode_retains_original_inputs_and_survivor_after_cancellation() {
    let fixture = VisualFixture::new();
    let (composition, runtime) = super::composition();
    let pair = run(
        &fixture,
        &composition,
        ferrum_types::Device::CPU,
        &[MIXED, OTHER],
        None,
    )
    .await
    .0;
    let reverse = run(
        &fixture,
        &composition,
        ferrum_types::Device::CPU,
        &[OTHER, MIXED],
        None,
    )
    .await
    .0;
    assert_eq!(pair, [reverse[1].clone(), reverse[0].clone()]);
    assert_ne!(
        pair[0], pair[1],
        "different visual inputs must affect actual greedy outputs"
    );
    for index in 0..2 {
        let cancelled = run(
            &fixture,
            &composition,
            ferrum_types::Device::CPU,
            &[MIXED, OTHER],
            Some(index),
        )
        .await
        .0;
        assert_eq!(cancelled[index], pair[index][..5]);
        assert_eq!(cancelled[1 - index], pair[1 - index]);
    }
    assert_eq!(runtime.resident_bytes(), 0);
}
