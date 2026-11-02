//! Before-readiness preparation must remain useful after synthetic owners end.

use super::*;
use crate::model_executor::TokenSelectionMask;

async fn admit_product<R: DeviceRuntime>(
    executor: &ProgramExecutor<R>,
    tokens: &[TokenId],
    maximum_tokens: usize,
) -> ExecutionFrontier {
    let request = RequestId::new();
    let admission = ExecutionAdmission {
        context: ExecutorPrefillAdmission::for_product_request(
            &request,
            tokens,
            maximum_tokens,
            tokens.len(),
            0,
        )
        .unwrap(),
        adapter: None,
    };
    loop {
        match executor.admit_tokens(admission).await.unwrap() {
            TokenAdmissionOutcome::Admitted(frontier) => return frontier,
            TokenAdmissionOutcome::MaintenanceDeferred(_) => {
                assert!(matches!(
                    executor.maintain_prefill_backing(&request).unwrap(),
                    ExecutorPrefillMaintenanceOutcome::Maintained { .. }
                        | ExecutorPrefillMaintenanceOutcome::RetryAdmission { .. }
                ));
            }
            other => panic!("fresh product admission failed: {other:?}"),
        }
    }
}

async fn select_only<R: DeviceRuntime>(
    executor: &ProgramExecutor<R>,
    frontier: &ExecutionFrontier,
    work: TokenWork,
    allowed: TokenId,
) -> ExecutionFrontier {
    let mut mask = vec![0; executor.info.vocab_size];
    mask[allowed.get() as usize] = 1;
    let policy = LogitsReturnPolicy::GreedyArgmax {
        token_mask: Some(TokenSelectionMask::new(mask)),
        repetition_penalty: None,
    };
    let input = ExecutionWork::new(frontier, work, &policy).unwrap();
    let mut completions = sampling::completed(
        executor
            .execute_tokens(std::slice::from_ref(&input))
            .await
            .unwrap(),
    );
    assert_eq!(completions.len(), 1);
    let completion = completions.pop().unwrap();
    completion
        .validate_for(0, &input, executor.info.vocab_size)
        .unwrap();
    assert_eq!(
        sampling::sampling(&completion),
        &ExecutorSamplingOutput::GreedyToken(allowed),
        "replayed work must consume this request's current mask"
    );
    completion.next_frontier().clone()
}

pub async fn assert_startup_on_demand_product_reuse<R: DeviceRuntime>(
    executor: &ProgramExecutor<R>,
) {
    executor.attach_execution_event_sink(Arc::new(StartupEvents(
        Mutex::new(Vec::new()),
        DeviceTimingMode::Replay,
    )));
    executor.prepare_startup().await.unwrap();
    let plan = executor.reusable_execution_startup_plan.as_ref().unwrap();
    assert!(plan.program_policy.prepares_before_readiness());
    assert_eq!(
        plan.program_policy.catalog_lifetime(),
        ReusableExecutionCatalogLifetime::OnDemandBounded
    );
    let inventory = executor.lane.reusable_executable_preparation().unwrap();
    assert!(inventory.resident_executables() > 0);
    assert_eq!(
        inventory.state(),
        DeviceReusableExecutionPreparationState::Ready
    );
    settle(executor).await;
    assert_eq!(executor.metrics.submitted_waves.load(Ordering::Relaxed), 0);

    // Startup used FullLogits. A new Product owner must reuse those graphs
    // while selecting Greedy tokens, and a later mask must replace this one.
    let mut frontier = admit_product(
        executor,
        &[TokenId::new(0)],
        plan.maximum_decode_sequence_tokens,
    )
    .await;
    let replay = &executor.metrics.wave_timing.reusable_execution;
    let before = replay.replayed_segments.load(Ordering::Relaxed);
    frontier = select_only(
        executor,
        &frontier,
        TokenWork::Prefill(PrefillChunk::new(0, 1, 1).unwrap()),
        TokenId::new(1),
    )
    .await;
    assert!(replay.replayed_segments.load(Ordering::Relaxed) > before);
    for allowed in [TokenId::new(2), TokenId::new(3)] {
        let before = replay.replayed_segments.load(Ordering::Relaxed);
        frontier = select_only(
            executor,
            &frontier,
            TokenWork::Decode {
                token: TokenId::new(0),
                position: frontier.consumed_tokens(),
            },
            allowed,
        )
        .await;
        assert!(replay.replayed_segments.load(Ordering::Relaxed) > before);
    }
    sampling::close(executor, frontier.sequence()).await;
    settle(executor).await;
    assert!(reusable_executable_inventory_matches(
        inventory,
        executor.lane.reusable_executable_preparation().unwrap()
    ));

    // Exercise a legal prefill shape that was not seeded. It must still warm,
    // capture, then replay with another fresh owner, within the same budget.
    let maximum_chunk = plan
        .prefill_chunks()
        .into_iter()
        .map(PrefillChunk::tokens_to_process)
        .max()
        .unwrap();
    let chunk = (2..=maximum_chunk)
        .map(|tokens| PrefillChunk::new(0, tokens, tokens).unwrap())
        .find(|chunk| {
            !plan
                .descriptors
                .contains(&VNextReusableExecutionDescriptor::prefill(*chunk))
        })
        .expect("fixture needs an admitted shape outside the startup seeds");
    let tokens = vec![TokenId::new(0); chunk.total_prompt_tokens()];
    for phase in ["warm", "capture", "replay"] {
        let initial = admit_product(executor, &tokens, plan.maximum_decode_sequence_tokens).await;
        let before = replay.replayed_segments.load(Ordering::Relaxed);
        let done = select_only(
            executor,
            &initial,
            TokenWork::Prefill(chunk),
            TokenId::new(1),
        )
        .await;
        if phase == "replay" {
            assert!(replay.replayed_segments.load(Ordering::Relaxed) > before);
        }
        sampling::close(executor, done.sequence()).await;
        settle(executor).await;
    }
    let after = executor.lane.reusable_executable_preparation().unwrap();
    assert!(after.captured_executables() > inventory.captured_executables());
    assert_eq!(after.maximum_executables(), inventory.maximum_executables());
    assert!(after.resident_executables() <= after.maximum_executables());
    assert!(executor.on_demand_reusable_execution_enabled());
}
