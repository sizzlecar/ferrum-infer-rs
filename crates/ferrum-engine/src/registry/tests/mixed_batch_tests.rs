//! Real CPU executor coverage for heterogeneous product frontiers.

use super::*;
use ferrum_execution::model_executor::{
    ExecutorPrefillAdmission, ExecutorPrefillMaintenanceOutcome, ExecutorSamplingOutput,
    LogitsReturnPolicy, PrefillChunk,
};
use ferrum_execution::token_execution::*;
use ferrum_kernels::backend::cpu::vnext_ops::CpuVNextComposition;
use ferrum_kernels::backend::cpu::vnext_runtime::CpuDeviceRuntime;
use ferrum_models::VNextModelExecutor;
use ferrum_runtime::{DeviceId, WeightMaterializerSelection};
use ferrum_types::{RequestId, TokenId};

mod greedy_readback;

struct CpuFixture {
    executor: VNextModelExecutor<CpuDeviceRuntime>,
    directory: PathBuf,
}

impl CpuFixture {
    async fn new(maximum_batch_tokens: usize) -> Self {
        let directory = write_qwen35_fixture_model_dir();
        std::fs::write(directory.join("tokenizer.json"), br#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<unk>":0,"hello":1,"<eos>":2},"unk_token":"<unk>"}}"#).unwrap();
        std::fs::write(directory.join("tokenizer_config.json"), br#"{"chat_template":"{% for message in messages %}{{ message['content'] }}{% endfor %}","eos_token_id":2,"unk_token":"<unk>"}"#).unwrap();
        let defined = ferrum_models::vnext::qwen35::define_from_model_dir(&directory).unwrap();
        let mut config = qwen35_fixture_component_config(&directory).engine_config;
        config.scheduler.max_running_requests = 4;
        config.batching.max_num_batched_tokens = maximum_batch_tokens;
        config.backend.enable_reusable_execution = false;
        let composition = CpuVNextComposition::create(
            DeviceId::new(format!("device.cpu.mixed-{}", uuid::Uuid::new_v4())).unwrap(),
            16 * 1024 * 1024,
        )
        .unwrap();
        let (runtime, operations, materializers, materializer, catalog) =
            composition.into_parts().unwrap();
        let executor = crate::product_composition::create_vnext_executor(
            &config,
            &defined,
            runtime,
            operations,
            materializers,
            catalog,
            |_| Ok(WeightMaterializerSelection::exact(materializer.clone())),
            None,
        )
        .unwrap();
        executor.prepare_startup().await.unwrap();
        Self {
            executor,
            directory,
        }
    }

    async fn admit(&self, tokens: &[u32]) -> ExecutionFrontier {
        let request_id = RequestId::new();
        let tokens: Vec<_> = tokens.iter().copied().map(TokenId::new).collect();
        let input = ExecutionAdmission {
            context: ExecutorPrefillAdmission::for_product_request(
                &request_id,
                &tokens,
                16,
                tokens.len(),
                0,
            )
            .unwrap(),
            adapter: None,
        };
        loop {
            match self.executor.admit_tokens(input).await.unwrap() {
                TokenAdmissionOutcome::Admitted(frontier) => {
                    assert_eq!(frontier.consumed_tokens(), 0);
                    return frontier;
                }
                TokenAdmissionOutcome::MaintenanceDeferred(_) => {
                    assert!(matches!(
                        self.executor.maintain_prefill_backing(&request_id).unwrap(),
                        ExecutorPrefillMaintenanceOutcome::Maintained { .. }
                            | ExecutorPrefillMaintenanceOutcome::RetryAdmission { .. }
                    ));
                }
                other => panic!("tiny CPU admission did not become ready: {other:?}"),
            }
        }
    }

    async fn execute(&self, inputs: &[ExecutionWork<'_>]) -> Vec<ExecutionCompletion> {
        let outcome = self.executor.execute_tokens(inputs).await.unwrap();
        let ExecutionBatchOutcome::Completed(batch) = outcome else {
            panic!("tiny CPU work did not complete: {outcome:?}");
        };
        let completed = batch.into_completions();
        assert_eq!(completed.len(), inputs.len());
        for (index, (output, input)) in completed.iter().zip(inputs).enumerate() {
            output
                .validate_for(index, input, self.executor.info().vocab_size)
                .unwrap();
        }
        completed
    }

    async fn prefill(
        &self,
        frontier: &ExecutionFrontier,
        chunk: PrefillChunk,
    ) -> ExecutionCompletion {
        self.execute(&[ExecutionWork::new(
            frontier,
            TokenWork::Prefill(chunk),
            &LogitsReturnPolicy::FullLogits,
        )
        .unwrap()])
            .await
            .pop()
            .unwrap()
    }

    async fn seed_decode(&self, tokens: &[u32]) -> ExecutionFrontier {
        let frontier = self.admit(tokens).await;
        self.prefill(
            &frontier,
            PrefillChunk::new(0, tokens.len(), tokens.len()).unwrap(),
        )
        .await
        .next_frontier()
        .clone()
    }

    async fn close(&self, frontier: &ExecutionFrontier) {
        let closed = self.executor.cancel_tokens(frontier.sequence()).unwrap();
        assert!(closed.error.is_none(), "{:?}", closed.error);
        let ended = match closed.disposition {
            ExecutionCloseDisposition::Ended(ended) => ended,
            ExecutionCloseDisposition::Held(held) => {
                let terminal = held.wait().await.unwrap();
                assert!(terminal.error.is_none(), "{:?}", terminal.error);
                terminal.ended
            }
        };
        assert_eq!(ended.sequence(), frontier.sequence());
    }

    fn assert_no_live_sequences(&self) {
        let snapshot = self.executor.cache_metrics_snapshot().unwrap();
        assert_eq!(snapshot["active_sequences"], 0);
        assert_eq!(snapshot["pending_sequences"], 0);
    }

    fn submissions(&self) -> u64 {
        self.executor.cache_metrics_snapshot().unwrap()["counters"]["submitted_waves"]
            .as_u64()
            .unwrap()
    }
}

impl Drop for CpuFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

fn decode_work<'a>(
    frontier: &'a ExecutionFrontier,
    token: u32,
    policy: &'a LogitsReturnPolicy,
) -> ExecutionWork<'a> {
    ExecutionWork::new(
        frontier,
        TokenWork::Decode {
            token: TokenId::new(token),
            position: frontier.consumed_tokens(),
        },
        policy,
    )
    .unwrap()
}

fn sampling(output: &ExecutionCompletion) -> &ExecutorSamplingOutput {
    let ExecutionProduct::Sampling(sampling) = output.product() else {
        panic!("expected a sampleable product: {output:?}");
    };
    sampling
}

fn assert_logits_close(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    assert!(
        actual.iter().zip(expected).all(|(actual, expected)| {
            actual.is_finite() && expected.is_finite() && (actual - expected).abs() <= 1e-5
        }),
        "mixed logits {actual:?} differ from split logits {expected:?}"
    );
}

fn assert_prefill_matches(actual: &ExecutionCompletion, expected: &ExecutionCompletion) {
    let (
        ExecutedTokenWork::Prefill {
            chunk: actual_chunk,
            ..
        },
        ExecutedTokenWork::Prefill {
            chunk: expected_chunk,
            ..
        },
    ) = (actual.executed(), expected.executed())
    else {
        panic!("expected prefill completions");
    };
    assert_eq!(actual_chunk, expected_chunk);
    assert_eq!(
        actual.next_frontier().consumed_tokens(),
        expected.next_frontier().consumed_tokens()
    );
    match (actual.product(), expected.product()) {
        (ExecutionProduct::Intermediate, ExecutionProduct::Intermediate) => {}
        (
            ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(actual)),
            ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(expected)),
        ) => assert_logits_close(actual, expected),
        other => panic!("mixed and split prefill product kinds differ: {other:?}"),
    }
}

fn assert_decode_matches(actual: &ExecutionCompletion, expected: &ExecutionCompletion) {
    assert!(matches!(
        actual.executed(),
        ExecutedTokenWork::Decode { .. }
    ));
    assert_eq!(actual.executed(), expected.executed());
    assert_eq!(
        actual.next_frontier().consumed_tokens(),
        expected.next_frontier().consumed_tokens()
    );
    match (sampling(actual), sampling(expected)) {
        (
            ExecutorSamplingOutput::FullLogits(actual),
            ExecutorSamplingOutput::FullLogits(expected),
        ) => assert_logits_close(actual, expected),
        _ => panic!("full-logits CPU fixture returned an unauthorized selected token"),
    }
}

#[tokio::test]
async fn real_cpu_mixed_prefill_decode_matches_split_through_final_and_intermediate_chunks() {
    let mixed = CpuFixture::new(8).await;
    let split = CpuFixture::new(8).await;
    let mixed_decode = mixed.seed_decode(&[0, 1]).await;
    let split_decode = split.seed_decode(&[0, 1]).await;
    let mixed_prefills = [mixed.admit(&[1, 0, 1, 2]).await, mixed.admit(&[2, 1]).await];
    let split_prefills = [split.admit(&[1, 0, 1, 2]).await, split.admit(&[2, 1]).await];
    let chunks = [
        PrefillChunk::new(0, 2, 4).unwrap(),
        PrefillChunk::new(0, 2, 2).unwrap(),
    ];
    let policy = LogitsReturnPolicy::FullLogits;
    let before = mixed.submissions();
    let first = mixed
        .execute(&[
            ExecutionWork::new(&mixed_prefills[0], TokenWork::Prefill(chunks[0]), &policy).unwrap(),
            ExecutionWork::new(&mixed_prefills[1], TokenWork::Prefill(chunks[1]), &policy).unwrap(),
            decode_work(&mixed_decode, 1, &policy),
        ])
        .await;
    assert_eq!(mixed.submissions() - before, 1);
    assert_eq!(first.len(), 3);
    assert_eq!(first[2].next_frontier().consumed_tokens(), 3);
    let expected_prefills = [
        split.prefill(&split_prefills[0], chunks[0]).await,
        split.prefill(&split_prefills[1], chunks[1]).await,
    ];
    let expected_decodes = split
        .execute(&[decode_work(&split_decode, 1, &policy)])
        .await;
    for (actual, expected) in first[..2].iter().zip(&expected_prefills) {
        assert_prefill_matches(actual, expected);
    }
    assert_decode_matches(&first[2], &expected_decodes[0]);

    // The first request finishes prefill while both peers decode. Product
    // order reverses their admission order, and the protocol must preserve it.
    let tail = PrefillChunk::new(2, 2, 4).unwrap();
    let before = mixed.submissions();
    let next = mixed
        .execute(&[
            ExecutionWork::new(first[0].next_frontier(), TokenWork::Prefill(tail), &policy)
                .unwrap(),
            decode_work(first[1].next_frontier(), 0, &policy),
            decode_work(first[2].next_frontier(), 2, &policy),
        ])
        .await;
    assert_eq!(mixed.submissions() - before, 1);
    let expected_tail = split
        .prefill(expected_prefills[0].next_frontier(), tail)
        .await;
    let expected_next = split
        .execute(&[
            decode_work(expected_prefills[1].next_frontier(), 0, &policy),
            decode_work(expected_decodes[0].next_frontier(), 2, &policy),
        ])
        .await;
    assert_prefill_matches(&next[0], &expected_tail);
    for ((actual, expected), prior) in next[1..].iter().zip(&expected_next).zip(&first[1..]) {
        assert_decode_matches(actual, expected);
        assert_eq!(
            actual.next_frontier().sequence(),
            prior.next_frontier().sequence()
        );
        assert_eq!(
            actual.next_frontier().consumed_tokens(),
            prior.next_frontier().consumed_tokens() + 1
        );
    }
    for output in &next {
        mixed.close(output.next_frontier()).await;
    }
    split.close(expected_tail.next_frontier()).await;
    for output in &expected_next {
        split.close(output.next_frontier()).await;
    }
    mixed.assert_no_live_sequences();
    split.assert_no_live_sequences();
}

#[tokio::test]
async fn real_cpu_mixed_work_beyond_compiled_token_bound_fails_and_releases_participants() {
    let fixture = CpuFixture::new(2).await;
    let decode = fixture.seed_decode(&[0]).await;
    let prefill = fixture.admit(&[1, 2]).await;
    let policy = LogitsReturnPolicy::FullLogits;
    let inputs = [
        ExecutionWork::new(
            &prefill,
            TokenWork::Prefill(PrefillChunk::new(0, 2, 2).unwrap()),
            &policy,
        )
        .unwrap(),
        decode_work(&decode, 1, &policy),
    ];
    let before = fixture.submissions();
    // Each phase fits alone, but the cohort exceeds the immutable two-token
    // bound. Preserve the real per-participant interruption, not a bare error.
    let outcome = fixture.executor.execute_tokens(&inputs).await.unwrap();
    let ExecutionBatchOutcome::Interrupted(interruption) = outcome else {
        panic!("bounded mixed work must interrupt its cohort: {outcome:?}");
    };
    assert!(interruption.completed().is_empty());
    assert_eq!(interruption.remaining().len(), inputs.len());
    assert!(interruption.causes().iter().any(|error| matches!(
        error,
        FerrumError::Backend { message }
            if message.contains("exceeds its bounded resource formula")
    )));
    let (_, remaining, causes) = interruption.into_parts();
    for (index, participant) in remaining.into_iter().enumerate() {
        let (actual_index, sequence, progress, continuation, cause_index) =
            participant.into_parts();
        assert_eq!(actual_index, index);
        assert_eq!(&sequence, inputs[index].sequence());
        assert!(cause_index < causes.len());
        assert_eq!(
            progress,
            ExecutionProgress::NotSubmitted {
                stage: NoSubmissionStage::BeforeSubmit
            }
        );
        let ended = match continuation {
            ContinuationDisposition::Ended(ended) => ended,
            ContinuationDisposition::Held(held) => {
                let terminal = held.wait().await.unwrap();
                assert!(terminal.error.is_none(), "{:?}", terminal.error);
                terminal.ended
            }
            ContinuationDisposition::Retryable(_) => {
                panic!("permanent plan bound is not retryable")
            }
        };
        assert_eq!(ended.sequence(), &sequence);
    }
    assert_eq!(fixture.submissions(), before);
    assert_eq!(decode.consumed_tokens(), 1);
    fixture.assert_no_live_sequences();
    // Neither original frontier can be replayed through a split fallback.
    for input in &inputs {
        assert!(fixture
            .executor
            .execute_tokens(std::slice::from_ref(input))
            .await
            .is_err());
    }
    assert_eq!(fixture.submissions(), before);

    // Real released ownership permits a fresh request under the same plan.
    let replacement = fixture.seed_decode(&[1, 2]).await;
    let decoded = fixture
        .execute(&[decode_work(&replacement, 1, &policy)])
        .await;
    assert_eq!(decoded[0].next_frontier().consumed_tokens(), 3);
    fixture.close(decoded[0].next_frontier()).await;
    fixture.assert_no_live_sequences();
}
