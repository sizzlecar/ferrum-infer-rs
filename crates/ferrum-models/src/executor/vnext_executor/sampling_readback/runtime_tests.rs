use super::*;
use ferrum_interfaces::model_executor::{GreedyRepetitionPenalty, TokenSelectionMask};
use ferrum_interfaces::sampler::{
    GreedySampler, LogitsProcessor, RepetitionPenaltyProcessor, Sampler, SamplingContext,
    SamplingRng,
};
use ferrum_kernels::backend::cpu::vnext_ops::CpuVNextComposition;
use serde_json::json;

mod fixture;
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

fn reserve_prefill<R: DeviceRuntime>(fixture: &Fixture<R>, request_id: &RequestId) {
    let executor = &fixture.executor;
    let tokens = [TokenId::new(0)];
    loop {
        match executor
            .try_admit_prefill(ExecutorPrefillAdmission::for_diagnostic(
                request_id, &tokens, 16,
            ))
            .unwrap()
        {
            ExecutorPrefillAdmissionDecision::Admitted(_) => break,
            ExecutorPrefillAdmissionDecision::MaintenanceDeferred(_) => {
                assert!(matches!(
                    executor.maintain_prefill_backing(request_id).unwrap(),
                    ExecutorPrefillMaintenanceOutcome::Maintained { .. }
                        | ExecutorPrefillMaintenanceOutcome::RetryAdmission { .. }
                ));
            }
            other => panic!("tiny admitted fixture unexpectedly deferred: {other:?}"),
        }
    }
}

async fn admit<R: DeviceRuntime>(
    fixture: &Fixture<R>,
    request_id: &RequestId,
) -> Arc<dyn KvCacheHandle> {
    reserve_prefill(fixture, request_id);
    let executor = &fixture.executor;
    let tokens = [TokenId::new(0)];
    let input = PlanRuntimePrefillInput::new(
        request_id.clone(),
        Arc::<[TokenId]>::from(tokens),
        16,
        PrefillChunk::new(0, 1, 1).unwrap(),
    )
    .unwrap();
    let before = readback(executor);
    let completion = match executor
        .plan_runtime_prefill_with_capacity(&input)
        .await
        .unwrap()
    {
        PlanRuntimePrefillOutcome::Completed(completion) => completion,
        PlanRuntimePrefillOutcome::Deferred(_) => panic!("tiny admitted prefill did not execute"),
    };
    completion
        .validate_for(request_id, input.chunk, VOCAB)
        .unwrap();
    let output = completion.output();
    let PlanRuntimePrefillProduct::FinalLogits(logits) = output.product() else {
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
    Arc::clone(output.kv_cache())
}

fn legacy_tokens(tokens: &[u32]) -> TensorRef {
    common::wrap_tensor(common::tokens_to_tensor(tokens, &candle_core::Device::Cpu).unwrap())
}

fn lora_metadata() -> std::collections::HashMap<String, serde_json::Value> {
    std::collections::HashMap::from([
        ("ferrum_lora_adapter".into(), json!("adapter")),
        ("ferrum_lora_path".into(), json!("unused-adapter-path")),
    ])
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
async fn lora_legacy_prefill_rejects_before_any_plan_runtime_submission() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    let first_id = RequestId::new();
    let second_id = RequestId::new();
    reserve_prefill(&fixture, &first_id);
    reserve_prefill(&fixture, &second_id);
    let first = PrefillInput::new(legacy_tokens(&[0])).with_request_context(first_id, 16);
    let second = PrefillInput::new(legacy_tokens(&[0]))
        .with_request_context(second_id, 16)
        .with_metadata(lora_metadata());
    let executor = &fixture.executor;
    let before = executor.cache_metrics_snapshot().unwrap();

    let results = [
        executor.prefill(&second).await.map(|_| ()),
        executor.prefill_with_capacity(&second).await.map(|_| ()),
        executor
            .batch_prefill(&[first.clone(), second.clone()])
            .await
            .map(|_| ()),
        executor
            .batch_prefill_with_capacity(&[first.clone(), second.clone()])
            .await
            .map(|_| ()),
    ];
    for result in results {
        assert!(
            matches!(result, Err(FerrumError::Unsupported { .. })),
            "{result:?}"
        );
    }
    assert_no_adapter_execution(executor, &before);

    // The valid participant remains executable after rejecting the whole batch.
    executor.prefill(&first).await.unwrap();
    assert_eq!(
        executor.metrics.prefill_operations.load(Ordering::Relaxed),
        1
    );
}

#[tokio::test]
async fn lora_legacy_decode_and_verify_reject_the_whole_batch_before_progress() {
    let fixture = cpu(SamplingReadbackMode::Uniform);
    let first_id = RequestId::new();
    let second_id = RequestId::new();
    let first_cache = admit(&fixture, &first_id).await;
    let second_cache = admit(&fixture, &second_id).await;
    let first =
        DecodeInput::new(legacy_tokens(&[1]), first_cache.clone()).with_request_id(first_id);
    let second = DecodeInput::new(legacy_tokens(&[1]), second_cache.clone())
        .with_request_id(second_id)
        .with_metadata(lora_metadata());
    let executor = &fixture.executor;
    let before = executor.cache_metrics_snapshot().unwrap();
    let results = [
        executor.decode(&second).await.map(|_| ()),
        executor
            .batch_decode(&[first.clone(), second.clone()])
            .await
            .map(|_| ()),
        executor
            .batch_decode_with_capacity(&[first.clone(), second.clone()])
            .await
            .map(|_| ()),
        executor
            .forward_verify(&[first.clone(), second.clone()])
            .await
            .map(|_| ()),
    ];
    for result in results {
        assert!(
            matches!(result, Err(FerrumError::Unsupported { .. })),
            "{result:?}"
        );
    }
    assert_no_adapter_execution(executor, &before);
    for cache in [&first_cache, &second_cache] {
        let sequence = executor.sequence_for_cache(&cache.cache_id()).unwrap();
        assert_eq!(sequence.tokens.lock().len(), 1);
    }
    executor.decode(&first).await.unwrap();
    assert_eq!(
        executor.metrics.decode_operations.load(Ordering::Relaxed),
        1
    );
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
                    PlanRuntimeDecodeInput::new(
                        ids[request].clone(),
                        TokenId::new(token),
                        Arc::clone(&caches[arm][request]),
                    )
                    .with_logits_policy(policy.clone())
                })
                .collect::<Vec<_>>();
            let before = readback(&fixture.executor);
            let completed = match fixture
                .executor
                .plan_runtime_batch_decode_with_capacity(&input)
                .await
                .unwrap()
            {
                PlanRuntimeBatchDecodeOutcome::Completed(output) => output,
                PlanRuntimeBatchDecodeOutcome::Deferred(_) => {
                    panic!("tiny decode wave {index} did not execute")
                }
            };
            assert_eq!(completed.len(), input.len());
            let scalar = if arm == 0 {
                vec![wave.uniform_scalar; input.len()]
            } else {
                wave.participant_scalar.clone()
            };
            for ((output, input), (&request, scalar)) in completed
                .iter()
                .zip(&input)
                .zip(wave.order.iter().zip(&scalar))
            {
                assert_eq!(
                    output.kv_cache.cache_id(),
                    input.kv_cache.cache_id(),
                    "caller request order"
                );
                assert_eq!(
                    output.kv_cache.num_tokens(),
                    input.kv_cache.num_tokens() + 1,
                    "committed frontier"
                );
                output
                    .sampling_output
                    .validate_for_policy(&input.logits_policy, VOCAB)
                    .unwrap();
                assert_eq!(
                    matches!(
                        output.sampling_output,
                        ExecutorSamplingOutput::GreedyToken(_)
                    ),
                    *scalar,
                    "arm {arm}, wave {index}, request {request}"
                );
                caches[arm][request] = Arc::clone(&output.kv_cache);
            }
            let greedy = scalar.iter().filter(|&&scalar| scalar).count() as u64;
            let full = input.len() as u64 - greedy;
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
            let ExecutorSamplingOutput::FullLogits(logits) = &outputs[0][0].sampling_output else {
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
            match (&baseline.sampling_output, &selected.sampling_output) {
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
                let ExecutorSamplingOutput::FullLogits(logits) = &selected.sampling_output else {
                    panic!("all-masked row must preserve host failure scope")
                };
                assert!(host_token(logits, policy).is_err());
            }
        }
    }
    for (fixture, caches) in fixtures.iter().zip(caches) {
        for cache in caches {
            fixture.executor.release_cache(&cache.cache_id());
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
                PlanRuntimeDecodeInput::new(
                    ids[owner].clone(),
                    TokenId::new(token),
                    Arc::clone(&caches[owner]),
                )
                .with_logits_policy(policy.clone())
            })
            .collect::<Vec<_>>();
        let before = readback(&fixture.executor);
        let completed = match fixture
            .executor
            .plan_runtime_batch_decode_with_capacity(&input)
            .await
            .unwrap()
        {
            PlanRuntimeBatchDecodeOutcome::Completed(completed) => completed,
            PlanRuntimeBatchDecodeOutcome::Deferred(_) => {
                panic!("nonfinite fixture did not execute")
            }
        };
        assert_eq!(completed.len(), 3);
        for (output, input) in completed.iter().zip(&input) {
            assert_eq!(output.kv_cache.cache_id(), input.kv_cache.cache_id());
            assert_eq!(output.kv_cache.num_tokens(), 2);
            output
                .sampling_output
                .validate_for_policy(&input.logits_policy, VOCAB)
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
            fixture.executor.release_cache(&cache.cache_id());
        }
    }
    let ExecutorSamplingOutput::FullLogits(failed_logits) = &outputs[0][0].sampling_output else {
        panic!("baseline failure must remain host sampling")
    };
    assert!(failed_logits.iter().all(|value| !value.is_finite()));
    assert!(host_token(failed_logits, &policies[0]).is_err());
    assert_eq!(
        outputs[1][0].sampling_output,
        ExecutorSamplingOutput::NoSelectableFiniteLogit
    );
    assert_eq!(outputs[0][1].sampling_output, outputs[1][1].sampling_output);
    let ExecutorSamplingOutput::FullLogits(healthy_logits) = &outputs[0][2].sampling_output else {
        panic!("baseline healthy row must remain full")
    };
    assert_eq!(
        outputs[1][2].sampling_output,
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
