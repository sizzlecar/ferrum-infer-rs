use super::*;
use ferrum_execution::model_executor::{GreedyRepetitionPenalty, TokenSelectionMask};
use ferrum_execution::token_execution::*;
use ferrum_interfaces::sampler::{
    GreedySampler, LogitsProcessor, RepetitionPenaltyProcessor, Sampler, SamplingContext,
    SamplingRng,
};
use ferrum_kernels::backend::cpu::vnext_ops::CpuVNextComposition;

pub(in crate::executor::vnext_executor) mod fixture;
mod lifecycle;
pub(in crate::executor::vnext_executor) mod startup;
mod token_execution;
use fixture::{Fixture, FixtureOptions, VOCAB};

fn cpu(
    mode: SamplingReadbackMode,
) -> Fixture<ferrum_kernels::backend::cpu::vnext_runtime::CpuDeviceRuntime> {
    cpu_with_nonfinite(mode, None)
}

fn cpu_with_nonfinite(
    mode: SamplingReadbackMode,
    nonfinite: Option<usize>,
) -> Fixture<ferrum_kernels::backend::cpu::vnext_runtime::CpuDeviceRuntime> {
    cpu_with_precision(mode, nonfinite, ElementType::F32)
}

fn cpu_with_precision(
    mode: SamplingReadbackMode,
    nonfinite: Option<usize>,
    logits_type: ElementType,
) -> Fixture<ferrum_kernels::backend::cpu::vnext_runtime::CpuDeviceRuntime> {
    let (runtime, registry, materializers, _, catalog) = CpuVNextComposition::create(
        DeviceId::new("device.fixture.sampling-readback").unwrap(),
        256 << 20,
    )
    .unwrap()
    .into_parts()
    .unwrap();
    Fixture::new(
        mode,
        Device::CPU,
        runtime,
        registry,
        materializers,
        catalog,
        FixtureOptions {
            logits_type,
            nonfinite_token: nonfinite,
            ..FixtureOptions::default()
        },
    )
}

fn greedy(mask: Option<[i8; VOCAB]>) -> LogitsReturnPolicy {
    LogitsReturnPolicy::GreedyArgmax {
        token_mask: mask.map(|mask| TokenSelectionMask::new(mask.into())),
        repetition_penalty: None,
    }
}

fn repeated() -> LogitsReturnPolicy {
    LogitsReturnPolicy::GreedyArgmax {
        token_mask: None,
        repetition_penalty: Some(GreedyRepetitionPenalty::new(2.0, vec![0])),
    }
}

/// Use product host processing, not a second copy of the device comparator.
fn host_token(logits: &[f32], policy: &LogitsReturnPolicy) -> ferrum_types::Result<TokenId> {
    let mut logits = logits.to_vec();
    if let LogitsReturnPolicy::GreedyArgmax {
        token_mask,
        repetition_penalty,
    } = policy
    {
        if let Some(mask) = token_mask {
            for (value, valid) in logits.iter_mut().zip(mask.valid_token_mask.iter()) {
                if *valid == 0 {
                    *value = f32::NEG_INFINITY;
                }
            }
        }
        if let Some(repetition) = repetition_penalty {
            let frequencies = repetition
                .token_ids()
                .iter()
                .map(|&token| (TokenId::new(token), 1))
                .collect();
            let params = ferrum_types::SamplingParams::default();
            let mut context =
                SamplingContext::new(0, &params, &mut logits, &[], &frequencies, VOCAB);
            RepetitionPenaltyProcessor::new(repetition.penalty()).process(&mut context)?;
        }
    }
    GreedySampler.sample(&logits, &mut SamplingRng::seeded(1))
}

fn readback<R: DeviceRuntime>(executor: &VNextModelExecutor<R>) -> serde_json::Value {
    executor.cache_metrics_snapshot().unwrap()["counters"]["product_readback"].clone()
}

fn assert_delta(before: &serde_json::Value, after: &serde_json::Value, field: &str, value: u64) {
    assert_eq!(
        after[field].as_u64().unwrap() - before[field].as_u64().unwrap(),
        value,
        "{field}"
    );
}

async fn reserve_prefill<R: DeviceRuntime>(
    fixture: &Fixture<R>,
    request_id: &RequestId,
) -> ExecutionFrontier {
    let tokens = [TokenId::new(0)];
    token_execution::admit_tokens(fixture, request_id, &tokens).await
}

fn sampling(completion: &ExecutionCompletion) -> &ExecutorSamplingOutput {
    let ExecutionProduct::Sampling(value) = completion.product() else {
        panic!("final prefill/decode omitted its sampling product")
    };
    value
}

fn completed(outcome: ExecutionBatchOutcome) -> Vec<ExecutionCompletion> {
    let ExecutionBatchOutcome::Completed(batch) = outcome else {
        panic!("tiny native wave did not complete: {outcome:?}")
    };
    batch.into_completions()
}

async fn close<R: DeviceRuntime>(executor: &VNextModelExecutor<R>, sequence: &ExecutorSequence) {
    let outcome = executor.cancel_tokens(sequence).unwrap();
    assert!(outcome.error.is_none(), "{:?}", outcome.error);
    let ended = match outcome.disposition {
        ExecutionCloseDisposition::Ended(ended) => ended,
        ExecutionCloseDisposition::Held(held) => {
            let terminal = held.wait().await.unwrap();
            assert!(terminal.error.is_none(), "{:?}", terminal.error);
            terminal.ended
        }
    };
    assert_eq!(ended.sequence(), sequence);
}

async fn admit<R: DeviceRuntime>(
    fixture: &Fixture<R>,
    request_id: &RequestId,
) -> ExecutionFrontier {
    let frontier = reserve_prefill(fixture, request_id).await;
    let executor = &fixture.executor;
    let policy = LogitsReturnPolicy::FullLogits;
    let input = ExecutionWork::new(
        &frontier,
        TokenWork::Prefill(PrefillChunk::new(0, 1, 1).unwrap()),
        &policy,
    )
    .unwrap();
    let before = readback(executor);
    let mut outputs = completed(
        executor
            .execute_tokens(std::slice::from_ref(&input))
            .await
            .unwrap(),
    );
    assert_eq!(outputs.len(), 1);
    let output = outputs.pop().unwrap();
    output.validate_for(0, &input, VOCAB).unwrap();
    let ExecutorSamplingOutput::FullLogits(logits) = sampling(&output) else {
        panic!("final prefill lost full logits")
    };
    assert_eq!(logits, &[-1.0, -1.0, -2.0, -3.0]);
    let after = readback(executor);
    assert_delta(&before, &after, "full_logits_participants", 1);
    assert_delta(&before, &after, "greedy_token_participants", 0);
    assert_delta(
        &before,
        &after,
        "product_bytes",
        executor.io.output_layout.byte_len().unwrap(),
    );
    output.next_frontier().clone()
}

fn assert_no_adapter_execution<R: DeviceRuntime>(
    executor: &VNextModelExecutor<R>,
    before: &serde_json::Value,
) {
    let after = executor.cache_metrics_snapshot().unwrap();
    for field in ["active_sequences", "pending_sequences"] {
        assert_eq!(
            after[field].as_u64().expect(field),
            before[field].as_u64().expect(field),
            "{field}"
        );
    }
    for field in ["submitted_waves", "prefill_operations", "decode_operations"] {
        assert_eq!(
            after["counters"][field].as_u64().expect(field),
            before["counters"][field].as_u64().expect(field),
            "{field}"
        );
    }
}

#[tokio::test]
async fn lora_token_admission_rejects_before_any_native_submission() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    let first = RequestId::new();
    let rejected = RequestId::new();
    let frontier = reserve_prefill(&fixture, &first).await;
    let before = fixture.executor.cache_metrics_snapshot().unwrap();
    let tokens = [TokenId::new(0)];
    let result = fixture
        .executor
        .admit_tokens(ExecutionAdmission {
            context: ExecutorPrefillAdmission::for_diagnostic(&rejected, &tokens, 16),
            adapter: Some(ferrum_interfaces::adapter::LoraAdapterSelection {
                name: "adapter",
                path: "unused-adapter-path",
            }),
        })
        .await;
    assert!(
        matches!(result, Err(FerrumError::Unsupported { .. })),
        "{result:?}"
    );
    assert_no_adapter_execution(&fixture.executor, &before);
    let output = token_execution::execute_one(
        &fixture,
        &frontier,
        TokenWork::Prefill(PrefillChunk::new(0, 1, 1).unwrap()),
    )
    .await;
    assert_eq!(
        sampling(&output),
        &ExecutorSamplingOutput::FullLogits(vec![-1.0, -1.0, -2.0, -3.0])
    );
    close(&fixture.executor, frontier.sequence()).await;
}

#[tokio::test]
async fn lora_rejected_admission_cannot_rebind_live_decode_incarnation() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    let id = RequestId::new();
    let frontier = admit(&fixture, &id).await;
    let before = fixture.executor.cache_metrics_snapshot().unwrap();
    let tokens = [TokenId::new(0)];
    let result = fixture
        .executor
        .admit_tokens(ExecutionAdmission {
            context: ExecutorPrefillAdmission::for_diagnostic(&id, &tokens, 16),
            adapter: Some(ferrum_interfaces::adapter::LoraAdapterSelection {
                name: "adapter",
                path: "unused-adapter-path",
            }),
        })
        .await;
    assert!(
        matches!(result, Err(FerrumError::Unsupported { .. })),
        "{result:?}"
    );
    assert_no_adapter_execution(&fixture.executor, &before);
    let owner = fixture
        .executor
        .sequences
        .lock()
        .token_index
        .get(frontier.sequence())
        .unwrap()
        .upgrade()
        .unwrap();
    assert_eq!(*owner.tokens.lock(), [0]);
    let output = token_execution::execute_one(
        &fixture,
        &frontier,
        TokenWork::Decode {
            token: TokenId::new(1),
            position: 1,
        },
    )
    .await;
    assert_eq!(
        sampling(&output),
        &ExecutorSamplingOutput::FullLogits(vec![3.0, 2.0, 1.0, -4.0])
    );
    drop(owner);
    close(&fixture.executor, frontier.sequence()).await;
}

struct Wave {
    order: Vec<usize>,
    tokens: Vec<u32>,
    policies: Vec<LogitsReturnPolicy>,
    participant_scalar: Vec<bool>,
    uniform_scalar: bool,
}

async fn compare<R: DeviceRuntime>(uniform: Fixture<R>, participant: Fixture<R>) {
    let fixtures = [&uniform, &participant];
    let ids = [RequestId::new(), RequestId::new(), RequestId::new()];
    let mut caches = Vec::new();
    for (fixture, mode) in fixtures.iter().zip([
        SamplingReadbackMode::Uniform,
        SamplingReadbackMode::Participant,
    ]) {
        fixture.executor.prepare_startup().await.unwrap();
        assert_eq!(fixture.executor.sampling_readback_mode, mode);
        assert_eq!(
            readback(&fixture.executor)["mode"],
            serde_json::to_value(mode).unwrap()
        );
        let mut requests = Vec::new();
        for id in &ids {
            requests.push(admit(fixture, id).await);
        }
        caches.push(requests);
    }
    let waves = [
        // F16's last-token linear rounds a negative half-ulp value to -0. The Full
        // arm reads those real logits and supplies the host-order oracle.
        Wave {
            order: vec![2, 1, 0],
            tokens: vec![2, 0, 3],
            policies: vec![greedy(None), LogitsReturnPolicy::FullLogits, greedy(None)],
            participant_scalar: vec![true, false, true],
            uniform_scalar: false,
        },
        // Caller order differs from admission order; negative ties, masks and
        // near ties exercise distinct selected-token owners in the same terminal.
        Wave {
            order: vec![2, 0, 1],
            tokens: vec![1, 0, 3],
            policies: vec![
                greedy(Some([0, 1, 1, 1])),
                LogitsReturnPolicy::FullLogits,
                greedy(None),
            ],
            participant_scalar: vec![true, false, true],
            uniform_scalar: false,
        },
        // Existing homogeneous greedy behavior includes its repetition path.
        Wave {
            order: vec![1, 2, 0],
            tokens: vec![1, 0, 2],
            policies: vec![repeated(), greedy(Some([0, 1, 1, 1])), greedy(None)],
            participant_scalar: vec![true, true, true],
            uniform_scalar: true,
        },
        // Active repetition must retain host logits when mixing policies.
        Wave {
            order: vec![0, 2, 1],
            tokens: vec![1, 3, 0],
            policies: vec![
                repeated(),
                LogitsReturnPolicy::FullLogits,
                greedy(Some([0, 1, 1, 1])),
            ],
            participant_scalar: vec![false, false, true],
            uniform_scalar: false,
        },
        // The same owners transition from full to selected and back; masks are
        // changed rather than merely replayed from a previous Step slot.
        Wave {
            order: vec![1, 0, 2],
            tokens: vec![3, 1, 0],
            policies: vec![
                LogitsReturnPolicy::FullLogits,
                greedy(Some([0, 0, 1, 1])),
                greedy(None),
            ],
            participant_scalar: vec![false, true, true],
            uniform_scalar: false,
        },
        // A smaller live participant set still restores the caller's order.
        Wave {
            order: vec![2, 0],
            tokens: vec![0, 1],
            policies: vec![greedy(Some([0, 1, 1, 1])), LogitsReturnPolicy::FullLogits],
            participant_scalar: vec![true, false],
            uniform_scalar: false,
        },
        // A statically empty mask keeps its request on the old full-logits
        // fallback without aborting an independently valid selected row.
        Wave {
            order: vec![0, 1, 2],
            tokens: vec![1, 3, 0],
            policies: vec![
                greedy(Some([0, 0, 0, 0])),
                LogitsReturnPolicy::FullLogits,
                greedy(None),
            ],
            participant_scalar: vec![false, false, true],
            uniform_scalar: false,
        },
    ];
    for (index, wave) in waves.iter().enumerate() {
        let mut outputs = Vec::new();
        for (arm, fixture) in fixtures.iter().enumerate() {
            let input = wave
                .order
                .iter()
                .zip(&wave.tokens)
                .zip(&wave.policies)
                .map(|((&request, &token), policy)| {
                    ExecutionWork::new(
                        &caches[arm][request],
                        TokenWork::Decode {
                            token: TokenId::new(token),
                            position: caches[arm][request].consumed_tokens(),
                        },
                        policy,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
            let before = readback(&fixture.executor);
            let completed = completed(fixture.executor.execute_tokens(&input).await.unwrap());
            assert_eq!(completed.len(), input.len());
            let input_len = input.len();
            let scalar = if arm == 0 {
                vec![wave.uniform_scalar; input.len()]
            } else {
                wave.participant_scalar.clone()
            };
            for (input_index, ((output, input), (&request, scalar))) in completed
                .iter()
                .zip(&input)
                .zip(wave.order.iter().zip(&scalar))
                .enumerate()
            {
                output.validate_for(input_index, input, VOCAB).unwrap();
                assert_eq!(
                    output.next_frontier().sequence(),
                    input.sequence(),
                    "caller request order"
                );
                assert_eq!(
                    output.next_frontier().consumed_tokens(),
                    input.expected_frontier().consumed_tokens() + 1,
                    "committed frontier"
                );
                sampling(output)
                    .validate_for_policy(input.logits_policy(), VOCAB)
                    .unwrap();
                assert_eq!(
                    matches!(sampling(output), ExecutorSamplingOutput::GreedyToken(_)),
                    *scalar,
                    "arm {arm}, wave {index}, request {request}"
                );
            }
            // Work borrows the old frontiers until all outputs were validated.
            for (output, &request) in completed.iter().zip(&wave.order) {
                caches[arm][request] = output.next_frontier().clone();
            }
            let greedy = scalar.iter().filter(|&&scalar| scalar).count() as u64;
            let full = input_len as u64 - greedy;
            let after = readback(&fixture.executor);
            assert_delta(&before, &after, "full_logits_participants", full);
            assert_delta(&before, &after, "greedy_token_participants", greedy);
            assert_delta(
                &before,
                &after,
                "product_bytes",
                full * fixture.executor.io.output_layout.byte_len().unwrap() + greedy * 4,
            );
            assert_delta(
                &before,
                &after,
                "hybrid_waves",
                u64::from(greedy != 0 && full != 0),
            );
            assert_delta(
                &before,
                &after,
                "sparse_repetition_participants",
                u64::from(wave.uniform_scalar),
            );
            outputs.push(completed);
        }
        if uniform.executor.io.output_element_type == ElementType::F16 && wave.tokens[0] == 2 {
            let ExecutorSamplingOutput::FullLogits(logits) = sampling(&outputs[0][0]) else {
                panic!("signed-zero oracle requires the original Full route")
            };
            assert_eq!(logits[0].to_bits(), (-0.0_f32).to_bits());
            assert_eq!(logits[1].to_bits(), 0.0_f32.to_bits());
            assert_eq!(
                host_token(logits, &wave.policies[0]).unwrap(),
                TokenId::new(1)
            );
        }
        for ((baseline, selected), policy) in outputs[0].iter().zip(&outputs[1]).zip(&wave.policies)
        {
            match (sampling(baseline), sampling(selected)) {
                (ExecutorSamplingOutput::FullLogits(a), ExecutorSamplingOutput::FullLogits(b)) => {
                    assert_eq!(
                        a.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                        b.iter().map(|value| value.to_bits()).collect::<Vec<_>>()
                    )
                }
                (
                    ExecutorSamplingOutput::FullLogits(logits),
                    ExecutorSamplingOutput::GreedyToken(token),
                ) => assert_eq!(*token, host_token(logits, policy).unwrap()),
                (
                    ExecutorSamplingOutput::GreedyToken(a),
                    ExecutorSamplingOutput::GreedyToken(b),
                ) => assert_eq!(a, b),
                other => panic!("unexpected sampling route: {other:?}"),
            }
            if matches!(policy, LogitsReturnPolicy::GreedyArgmax { token_mask: Some(mask), .. } if mask.valid_token_mask.iter().all(|&valid| valid == 0))
            {
                let ExecutorSamplingOutput::FullLogits(logits) = sampling(selected) else {
                    panic!("all-masked row must preserve host failure scope")
                };
                assert!(host_token(logits, policy).is_err());
            }
        }
    }
    for (fixture, caches) in fixtures.iter().zip(caches) {
        for cache in caches {
            close(&fixture.executor, cache.sequence()).await;
        }
        assert_eq!(
            fixture.executor.cache_metrics_snapshot().unwrap()["active_sequences"],
            0
        );
        assert_eq!(
            fixture.executor.cache_metrics_snapshot().unwrap()["counters"]["failed_waves"],
            0
        );
    }
}

#[tokio::test]
async fn sampling_readback_cpu_terminal_preserves_mixed_outputs_masks_order_and_bytes() {
    compare(
        cpu(SamplingReadbackMode::Uniform),
        cpu(SamplingReadbackMode::Participant),
    )
    .await;
}

#[tokio::test]
async fn sampling_readback_cpu_f16_terminal_preserves_mixed_outputs_signed_zero_and_order() {
    compare(
        cpu_with_precision(SamplingReadbackMode::Uniform, None, ElementType::F16),
        cpu_with_precision(SamplingReadbackMode::Participant, None, ElementType::F16),
    )
    .await;
}

async fn compare_no_finite<R: DeviceRuntime>(uniform: Fixture<R>, participant: Fixture<R>) {
    let fixtures = [&uniform, &participant];
    let ids = [RequestId::new(), RequestId::new(), RequestId::new()];
    let policies = [
        greedy(None),
        LogitsReturnPolicy::FullLogits,
        greedy(Some([0, 1, 1, 1])),
    ];
    let mut outputs = Vec::new();
    for (arm, fixture) in fixtures.iter().enumerate() {
        fixture.executor.prepare_startup().await.unwrap();
        let mut caches = Vec::new();
        for id in &ids {
            caches.push(admit(fixture, id).await);
        }
        // Failure and healthy outputs belong to different admitted owners,
        // presented in a different order from admission.
        let input = [2, 0, 1]
            .into_iter()
            .zip([3, 1, 0])
            .zip(&policies)
            .map(|((owner, token), policy)| {
                ExecutionWork::new(
                    &caches[owner],
                    TokenWork::Decode {
                        token: TokenId::new(token),
                        position: caches[owner].consumed_tokens(),
                    },
                    policy,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let before = readback(&fixture.executor);
        let completed = completed(fixture.executor.execute_tokens(&input).await.unwrap());
        assert_eq!(completed.len(), 3);
        for (index, (output, input)) in completed.iter().zip(&input).enumerate() {
            output.validate_for(index, input, VOCAB).unwrap();
            assert_eq!(output.next_frontier().sequence(), input.sequence());
            assert_eq!(output.next_frontier().consumed_tokens(), 2);
            sampling(output)
                .validate_for_policy(input.logits_policy(), VOCAB)
                .unwrap();
        }
        let after = readback(&fixture.executor);
        assert_delta(
            &before,
            &after,
            "product_bytes",
            if arm == 0 {
                3 * fixture.executor.io.output_layout.byte_len().unwrap()
            } else {
                fixture.executor.io.output_layout.byte_len().unwrap() + 8
            },
        );
        assert_delta(
            &before,
            &after,
            "full_logits_participants",
            if arm == 0 { 3 } else { 1 },
        );
        assert_delta(
            &before,
            &after,
            "greedy_token_participants",
            if arm == 0 { 0 } else { 2 },
        );
        assert_delta(&before, &after, "hybrid_waves", arm as u64);
        assert_eq!(
            fixture.executor.cache_metrics_snapshot().unwrap()["counters"]["failed_waves"],
            0
        );
        outputs.push(completed);
        for cache in caches {
            close(&fixture.executor, cache.sequence()).await;
        }
    }
    let ExecutorSamplingOutput::FullLogits(failed_logits) = sampling(&outputs[0][0]) else {
        panic!("baseline failure must remain host sampling")
    };
    assert!(failed_logits.iter().all(|value| !value.is_finite()));
    assert!(host_token(failed_logits, &policies[0]).is_err());
    assert_eq!(
        *sampling(&outputs[1][0]),
        ExecutorSamplingOutput::NoSelectableFiniteLogit
    );
    assert_eq!(*sampling(&outputs[0][1]), *sampling(&outputs[1][1]));
    let ExecutorSamplingOutput::FullLogits(healthy_logits) = sampling(&outputs[0][2]) else {
        panic!("baseline healthy row must remain full")
    };
    assert_eq!(
        *sampling(&outputs[1][2]),
        ExecutorSamplingOutput::GreedyToken(host_token(healthy_logits, &policies[2]).unwrap())
    );
}

#[tokio::test]
async fn sampling_readback_cpu_terminal_keeps_nonfinite_failure_request_local() {
    compare_no_finite(
        cpu_with_nonfinite(SamplingReadbackMode::Uniform, Some(3)),
        cpu_with_nonfinite(SamplingReadbackMode::Participant, Some(3)),
    )
    .await;
}

#[tokio::test]
async fn sampling_readback_cpu_f16_terminal_keeps_nonfinite_failure_request_local() {
    compare_no_finite(
        cpu_with_precision(SamplingReadbackMode::Uniform, Some(3), ElementType::F16),
        cpu_with_precision(SamplingReadbackMode::Participant, Some(3), ElementType::F16),
    )
    .await;
}

#[cfg(feature = "cuda")]
#[tokio::test]
#[ignore = "requires exclusive CUDA and real F32 language providers"]
async fn sampling_readback_cuda_terminal_preserves_mixed_outputs_masks_order_and_bytes() {
    use ferrum_kernels::backend::cuda::vnext_ops::CudaVNextComposition;
    let make = |mode, nonfinite| {
        let (runtime, registry, materializers, catalog) = CudaVNextComposition::create(
            0,
            DeviceId::new("device.fixture.sampling-readback").unwrap(),
            AttentionExecutionPolicy::Portable,
        )
        .unwrap()
        .into_parts();
        Fixture::new(
            mode,
            Device::CUDA(0),
            runtime,
            registry,
            materializers,
            catalog,
            FixtureOptions {
                logits_type: ElementType::F32,
                nonfinite_token: nonfinite,
                ..FixtureOptions::default()
            },
        )
    };
    compare(
        make(SamplingReadbackMode::Uniform, None),
        make(SamplingReadbackMode::Participant, None),
    )
    .await;
    compare_no_finite(
        make(SamplingReadbackMode::Uniform, Some(3)),
        make(SamplingReadbackMode::Participant, Some(3)),
    )
    .await;
}

#[cfg(feature = "cuda")]
#[tokio::test]
#[ignore = "requires exclusive CUDA and real F16 language providers"]
async fn sampling_readback_cuda_f16_terminal_preserves_mixed_outputs_and_failure_scope() {
    use ferrum_kernels::backend::cuda::vnext_ops::CudaVNextComposition;
    let make = |mode, nonfinite| {
        let (runtime, registry, materializers, catalog) = CudaVNextComposition::create(
            0,
            DeviceId::new("device.fixture.sampling-readback-f16").unwrap(),
            AttentionExecutionPolicy::Portable,
        )
        .unwrap()
        .into_parts();
        Fixture::new(
            mode,
            Device::CUDA(0),
            runtime,
            registry,
            materializers,
            catalog,
            FixtureOptions {
                logits_type: ElementType::F16,
                nonfinite_token: nonfinite,
                ..FixtureOptions::default()
            },
        )
    };
    compare(
        make(SamplingReadbackMode::Uniform, None),
        make(SamplingReadbackMode::Participant, None),
    )
    .await;
    compare_no_finite(
        make(SamplingReadbackMode::Uniform, Some(3)),
        make(SamplingReadbackMode::Participant, Some(3)),
    )
    .await;
}
