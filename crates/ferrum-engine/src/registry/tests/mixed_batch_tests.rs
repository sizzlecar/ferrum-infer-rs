//! Real CPU executor coverage for heterogeneous product frontiers.

use super::*;
use ferrum_interfaces::model_executor::{
    ExecutorPrefillAdmission, ExecutorPrefillAdmissionDecision, ExecutorPrefillMaintenanceOutcome,
    ExecutorSamplingOutput, PlanRuntimeBatchDecodeOutcome, PlanRuntimeDecodeInput,
    PlanRuntimeDecodeOutput, PlanRuntimeMixedBatchOutcome, PlanRuntimePrefillCompletion,
    PlanRuntimePrefillInput, PlanRuntimePrefillOutcome, PlanRuntimePrefillProduct, PrefillChunk,
};
use ferrum_interfaces::vnext::{DeviceId, WeightMaterializerSelection};
use ferrum_kernels::backend::cpu::vnext_ops::CpuVNextComposition;
use ferrum_kernels::backend::cpu::vnext_runtime::CpuDeviceRuntime;
use ferrum_models::VNextModelExecutor;
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
        )
        .unwrap();
        executor.prepare_startup().await.unwrap();
        Self {
            executor,
            directory,
        }
    }

    fn admit(&self, input: &PlanRuntimePrefillInput) {
        loop {
            let admission = ExecutorPrefillAdmission::for_product_request(
                &input.request_id,
                &input.input_tokens,
                input.maximum_sequence_tokens,
                input.input_tokens.len(),
                0,
            )
            .unwrap();
            match self.executor.try_admit_prefill(admission).unwrap() {
                ExecutorPrefillAdmissionDecision::Admitted(_) => return,
                ExecutorPrefillAdmissionDecision::MaintenanceDeferred(_) => {
                    assert!(matches!(
                        self.executor
                            .maintain_prefill_backing(&input.request_id)
                            .unwrap(),
                        ExecutorPrefillMaintenanceOutcome::Maintained { .. }
                            | ExecutorPrefillMaintenanceOutcome::RetryAdmission { .. }
                    ));
                }
                other => panic!("tiny CPU admission did not become ready: {other:?}"),
            }
        }
    }

    async fn prefill(&self, input: &PlanRuntimePrefillInput) -> PlanRuntimePrefillCompletion {
        match self
            .executor
            .plan_runtime_prefill_with_capacity(input)
            .await
            .unwrap()
        {
            PlanRuntimePrefillOutcome::Completed(output) => output,
            PlanRuntimePrefillOutcome::Deferred(deferral) => {
                panic!("tiny CPU prefill was deferred: {deferral:?}")
            }
        }
    }

    async fn seed_decode(&self, tokens: &[u32]) -> PlanRuntimeDecodeInput {
        let input = prompt(tokens, tokens.len());
        self.admit(&input);
        let output = self.prefill(&input).await;
        PlanRuntimeDecodeInput::new(
            input.request_id,
            TokenId::new(1),
            Arc::clone(output.output().kv_cache()),
        )
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

fn prompt(tokens: &[u32], chunk_tokens: usize) -> PlanRuntimePrefillInput {
    PlanRuntimePrefillInput::new(
        RequestId::new(),
        tokens.iter().copied().map(TokenId::new).collect::<Vec<_>>(),
        16,
        PrefillChunk::new(0, chunk_tokens, tokens.len()).unwrap(),
    )
    .unwrap()
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

fn assert_prefill_matches(
    actual: &PlanRuntimePrefillCompletion,
    expected: &PlanRuntimePrefillCompletion,
) {
    assert_eq!(actual.completed_chunk(), expected.completed_chunk());
    assert_eq!(
        actual.output().committed_tokens(),
        expected.output().committed_tokens()
    );
    match (actual.output().product(), expected.output().product()) {
        (PlanRuntimePrefillProduct::Intermediate, PlanRuntimePrefillProduct::Intermediate) => {}
        (
            PlanRuntimePrefillProduct::FinalLogits(actual),
            PlanRuntimePrefillProduct::FinalLogits(expected),
        ) => assert_logits_close(actual, expected),
        other => panic!("mixed and split prefill product kinds differ: {other:?}"),
    }
}

async fn split_decode(
    fixture: &CpuFixture,
    inputs: &[PlanRuntimeDecodeInput],
) -> Vec<PlanRuntimeDecodeOutput> {
    match fixture
        .executor
        .plan_runtime_batch_decode_with_capacity(inputs)
        .await
        .unwrap()
    {
        PlanRuntimeBatchDecodeOutcome::Completed(outputs) => outputs,
        PlanRuntimeBatchDecodeOutcome::Deferred(deferral) => {
            panic!("tiny CPU decode was deferred: {deferral:?}")
        }
    }
}

fn assert_decode_matches(actual: &PlanRuntimeDecodeOutput, expected: &PlanRuntimeDecodeOutput) {
    assert_eq!(actual.kv_cache.num_tokens(), expected.kv_cache.num_tokens());
    match (&actual.sampling_output, &expected.sampling_output) {
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
    let split_decode_input = split.seed_decode(&[0, 1]).await;
    let mixed_prefills = [prompt(&[1, 0, 1, 2], 2), prompt(&[2, 1], 2)];
    let split_prefills = [prompt(&[1, 0, 1, 2], 2), prompt(&[2, 1], 2)];
    for input in &mixed_prefills {
        mixed.admit(input);
    }
    for input in &split_prefills {
        split.admit(input);
    }
    let before = mixed.submissions();
    let (prefills, decodes) = match mixed
        .executor
        .plan_runtime_mixed_batch_with_capacity(
            &mixed_prefills,
            std::slice::from_ref(&mixed_decode),
        )
        .await
        .unwrap()
    {
        PlanRuntimeMixedBatchOutcome::Completed { prefills, decodes } => (prefills, decodes),
        _ => panic!("tiny mixed wave did not execute"),
    };
    assert_eq!(mixed.submissions() - before, 1);
    assert_eq!(prefills.len(), 2);
    assert_eq!(decodes.len(), 1);
    for (output, input) in prefills.iter().zip(&mixed_prefills) {
        output
            .validate_for(&input.request_id, input.chunk, 3)
            .unwrap();
    }
    assert_eq!(
        decodes[0].kv_cache.cache_id(),
        mixed_decode.kv_cache.cache_id()
    );
    assert_eq!(decodes[0].kv_cache.num_tokens(), 3);
    let expected_prefills = [
        split.prefill(&split_prefills[0]).await,
        split.prefill(&split_prefills[1]).await,
    ];
    let expected_decodes = split_decode(&split, std::slice::from_ref(&split_decode_input)).await;
    for (actual, expected) in prefills.iter().zip(&expected_prefills) {
        assert_prefill_matches(actual, expected);
    }
    assert_decode_matches(&decodes[0], &expected_decodes[0]);

    // The first request finishes its prefill, while the newly completed peer
    // and incumbent both decode. Reverse their admission order in the input.
    let mut mixed_tail = mixed_prefills[0].clone();
    mixed_tail.chunk = PrefillChunk::new(2, 2, 4).unwrap();
    let mut split_tail = split_prefills[0].clone();
    split_tail.chunk = mixed_tail.chunk;
    let mixed_next = [
        PlanRuntimeDecodeInput::new(
            mixed_prefills[1].request_id.clone(),
            TokenId::new(0),
            Arc::clone(prefills[1].output().kv_cache()),
        ),
        PlanRuntimeDecodeInput::new(
            mixed_decode.request_id,
            TokenId::new(2),
            Arc::clone(&decodes[0].kv_cache),
        ),
    ];
    let split_next = [
        PlanRuntimeDecodeInput::new(
            split_prefills[1].request_id.clone(),
            TokenId::new(0),
            Arc::clone(expected_prefills[1].output().kv_cache()),
        ),
        PlanRuntimeDecodeInput::new(
            split_decode_input.request_id,
            TokenId::new(2),
            Arc::clone(&expected_decodes[0].kv_cache),
        ),
    ];
    let (tail_outputs, next_outputs) = match mixed
        .executor
        .plan_runtime_mixed_batch_with_capacity(&[mixed_tail], &mixed_next)
        .await
        .unwrap()
    {
        PlanRuntimeMixedBatchOutcome::Completed { prefills, decodes } => (prefills, decodes),
        _ => panic!("mixed continuation wave did not execute"),
    };
    let expected_tail = split.prefill(&split_tail).await;
    let expected_next = split_decode(&split, &split_next).await;
    assert_prefill_matches(&tail_outputs[0], &expected_tail);
    for ((actual, expected), input) in next_outputs.iter().zip(&expected_next).zip(&mixed_next) {
        assert_decode_matches(actual, expected);
        assert_eq!(actual.kv_cache.cache_id(), input.kv_cache.cache_id());
        assert_eq!(
            actual.kv_cache.num_tokens(),
            input.kv_cache.num_tokens() + 1
        );
    }
    for output in &next_outputs {
        mixed.executor.release_cache(&output.kv_cache.cache_id());
    }
    mixed
        .executor
        .release_cache(&tail_outputs[0].output().kv_cache().cache_id());
    for output in &expected_next {
        split.executor.release_cache(&output.kv_cache.cache_id());
    }
    split
        .executor
        .release_cache(&expected_tail.output().kv_cache().cache_id());
    assert_eq!(
        mixed.executor.cache_metrics_snapshot().unwrap()["active_sequences"],
        0
    );
    assert_eq!(
        mixed.executor.cache_metrics_snapshot().unwrap()["pending_sequences"],
        0
    );
}

#[tokio::test]
async fn real_cpu_mixed_work_beyond_compiled_token_bound_fails_and_releases_participants() {
    let fixture = CpuFixture::new(2).await;
    let decode = fixture.seed_decode(&[0]).await;
    let prefill = prompt(&[1, 2], 2);
    fixture.admit(&prefill);
    let before = fixture.submissions();
    // Each phase fits alone, but their combined work exceeds the immutable
    // plan's two-token bound. This is a permanent error, not retryable pressure.
    let outcome = fixture
        .executor
        .plan_runtime_mixed_batch_with_capacity(
            std::slice::from_ref(&prefill),
            std::slice::from_ref(&decode),
        )
        .await;
    assert!(matches!(
        outcome,
        Err(FerrumError::Backend { ref message })
            if message.contains("exceeds its bounded resource formula")
    ));
    assert_eq!(fixture.submissions(), before);
    assert_eq!(decode.kv_cache.num_tokens(), 1);
    assert_eq!(
        fixture.executor.cache_metrics_snapshot().unwrap()["active_sequences"],
        0
    );
    assert_eq!(
        fixture.executor.cache_metrics_snapshot().unwrap()["pending_sequences"],
        0
    );
    // A permanent mixed error invalidates its product frontiers. The engine
    // must never replay them through the split fallback.
    assert!(fixture
        .executor
        .plan_runtime_batch_decode_with_capacity(std::slice::from_ref(&decode))
        .await
        .is_err());
    assert!(fixture
        .executor
        .plan_runtime_prefill_with_capacity(&prefill)
        .await
        .is_err());
    assert_eq!(fixture.submissions(), before);

    // Failed participants released their physical and logical ownership: a
    // fresh request can still complete under the same bounded plan.
    let replacement = fixture.seed_decode(&[1, 2]).await;
    let decoded = split_decode(&fixture, std::slice::from_ref(&replacement)).await;
    assert_eq!(decoded[0].kv_cache.num_tokens(), 3);
    fixture
        .executor
        .release_cache(&decoded[0].kv_cache.cache_id());
}
