//! Actual stored Qwen3.5 CPU execution through both original and managed I/O.
//! This validates the new transport/ownership path against the existing full
//! logits path, not the whole H3 pipeline or CUDA numerical qualification.

use super::*;
use ferrum_execution::{
    executor_lifecycle::{ExecutorOwner, ExecutorScope},
    model_executor::{
        ExecutorPrefillAdmission, ExecutorSamplingOutput, LogitsReturnPolicy, PrefillChunk,
    },
    token_execution::{
        ExecutionAdmission, ExecutionBatchOutcome, ExecutionCloseDisposition, ExecutionCompletion,
        ExecutionEndReason, ExecutionFrontier, ExecutionProduct, ExecutionWork,
        TokenAdmissionOutcome, TokenWork,
    },
    ManagedCausalIoProfile, ManagedCausalIoWork, ManagedCausalReadback,
};
use ferrum_kernels::backend::cpu::{
    vnext_ops::CpuVNextComposition, vnext_runtime::CpuDeviceRuntime,
};
use ferrum_runtime::{HostMemoryAccount, HostMemoryClass, HostMemoryClassLimits, HostMemoryLimits};
use ferrum_types::{RequestId, TokenId};

fn fixture() -> (
    std::path::PathBuf,
    DefinedProductionModel,
    PreparedProductionModel,
    EngineConfig,
) {
    fixture_for(ManagedCausalIoProfile::ManagedGreedyF16V1)
}

fn fixture_for(
    profile: ManagedCausalIoProfile,
) -> (
    std::path::PathBuf,
    DefinedProductionModel,
    PreparedProductionModel,
    EngineConfig,
) {
    let numerical_profile = match profile {
        ManagedCausalIoProfile::ManagedGreedyF16V1 => {
            ferrum_models::vnext::qwen35::F16_NUMERICAL_PROFILE_ID
        }
        ManagedCausalIoProfile::ManagedGreedyF32V1 => {
            ferrum_models::vnext::qwen35::F32_MASTER_NUMERICAL_PROFILE_ID
        }
    };
    prepared_execution_fixture_with(numerical_profile.parse().unwrap(), |directory| {
        // Preserve the shared nonzero GDN/attention fixture, but use a head
        // whose winners distinguish token rows. Install before source identity
        // or mmap preparation, so the test never mutates a retained checkpoint.
        let path = directory.join("model.safetensors");
        let bytes = std::fs::read(&path).unwrap();
        let archive = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let head = [1.0f32, -1.0, -1.0, 1.0, -1.0, -1.0];
        let head_bytes = head
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>();
        let mut tensors = archive.tensors();
        tensors
            .iter_mut()
            .find(|(name, _)| name == "model.lm_head.weight")
            .unwrap()
            .1 =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![3, 2], &head_bytes)
                .unwrap();
        safetensors::tensor::serialize_to_file(tensors, &None, &path).unwrap();
    })
}

fn host(output_bytes: usize, alignment: usize) -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            1 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 1 << 18,
                transfer_staging_bytes: 1 << 18,
                output_bytes,
                weight_backing_bytes: 1 << 18,
            },
        )
        .unwrap(),
        alignment,
    )
    .unwrap()
}

fn executor(
    defined: &DefinedProductionModel,
    prepared: &PreparedProductionModel,
    engine: &EngineConfig,
    host: Option<&HostMemoryAccount>,
) -> (ProgramExecutor<CpuDeviceRuntime>, Arc<CpuDeviceRuntime>) {
    executor_with_profile(
        defined,
        prepared,
        engine,
        host,
        ManagedCausalIoProfile::ManagedGreedyF16V1,
    )
}

fn executor_with_profile(
    defined: &DefinedProductionModel,
    prepared: &PreparedProductionModel,
    engine: &EngineConfig,
    host: Option<&HostMemoryAccount>,
    profile: ManagedCausalIoProfile,
) -> (ProgramExecutor<CpuDeviceRuntime>, Arc<CpuDeviceRuntime>) {
    let (runtime, operations, materializers, materializer, catalog) = CpuVNextComposition::create(
        ferrum_runtime::DeviceId::new("device.cpu.managed-causal-io").unwrap(),
        16 << 20,
    )
    .unwrap()
    .into_parts()
    .unwrap();
    let composition =
        RuntimeComposition::new(Arc::clone(&runtime), operations, materializers, catalog);
    let info = prepared.model_info(engine.model.model_id.clone(), ferrum_types::Device::CPU);
    let config =
        ProgramExecutorConfig::from_engine_config(engine, &info, runtime.as_ref()).unwrap();
    let compiled = composition
        .compile_model(
            prepared.execution_input().unwrap(),
            info,
            engine,
            config,
            WeightMaterializerSelection::exact(materializer),
        )
        .unwrap();
    let readback = match profile {
        ManagedCausalIoProfile::ManagedGreedyF16V1 => ManagedCausalReadback::F16Logits,
        ManagedCausalIoProfile::ManagedGreedyF32V1 => ManagedCausalReadback::F32Logits,
    };
    let planning_work = [ManagedCausalIoWork {
        immediate_tokens: 1,
        mask_upload: true,
        output: readback,
    }];
    let planned = host.map(|host| {
        let before = host.usage().unwrap().committed_bytes();
        let planned = compiled
            .managed_io_requirements(host, profile, &planning_work)
            .unwrap();
        assert_eq!(
            planned.output_bytes,
            host.backing_bytes(
                prepared.descriptor().vocabulary_size()
                    * profile.logits_element_type().size_bytes() as usize,
            )
            .unwrap()
        );
        let foreign = match profile {
            ManagedCausalIoProfile::ManagedGreedyF16V1 => {
                ManagedCausalIoProfile::ManagedGreedyF32V1
            }
            ManagedCausalIoProfile::ManagedGreedyF32V1 => {
                ManagedCausalIoProfile::ManagedGreedyF16V1
            }
        };
        assert!(compiled
            .managed_io_requirements(host, foreign, &planning_work)
            .is_err());
        assert_eq!(host.usage().unwrap().committed_bytes(), before);
        planned
    });
    let resolve = |policy: &ResolvedRuntimePolicy,
                   catalog: &CapabilityCatalog,
                   compilation: &ProgramPlanCompilation| {
        let numerical = NumericalProfileResolution::from_static_plan(
            engine.numerical_execution.clone(),
            ferrum_types::KvStorageFormat::F16,
            defined.definition(),
            prepared.family(),
            catalog,
            policy,
            compilation.executable().execution_plan(),
            vec![],
        )
        .map_err(|error| FerrumError::model(error.to_string()))?;
        resolve_model_plan(engine, prepared, catalog, policy, compilation, &numerical)
    };
    let executor = match host {
        Some(host) => compiled
            .initialize_with_managed_io(host, profile, resolve)
            .unwrap(),
        None => compiled.initialize(resolve).unwrap(),
    };
    if let Some(planned) = planned {
        assert_eq!(
            executor.managed_io_requirements(&planning_work).unwrap(),
            planned
        );
    }
    (executor, runtime)
}

fn greedy() -> LogitsReturnPolicy {
    LogitsReturnPolicy::GreedyArgmax {
        token_mask: None,
        repetition_penalty: None,
    }
}

fn submissions(executor: &ExecutorScope) -> u64 {
    executor.cache_metrics_snapshot().unwrap()["counters"]["submitted_waves"]
        .as_u64()
        .unwrap()
}

async fn admit(executor: &ExecutorScope, tokens: &[TokenId]) -> ExecutionFrontier {
    let request = RequestId::new();
    let admission = ExecutionAdmission {
        context: ExecutorPrefillAdmission::for_diagnostic(&request, tokens, 16),
        adapter: None,
    };
    loop {
        match executor.admit_tokens(admission).await.unwrap() {
            TokenAdmissionOutcome::Admitted(frontier) => return frontier,
            TokenAdmissionOutcome::MaintenanceDeferred(_) => {
                executor.maintain_prefill_backing(&request).unwrap();
            }
            other => panic!("fresh stored-model admission failed: {other:?}"),
        }
    }
}

async fn execute(
    executor: &ExecutorScope,
    frontier: &ExecutionFrontier,
    work: TokenWork,
    policy: &LogitsReturnPolicy,
) -> ExecutionCompletion {
    let input = ExecutionWork::new(frontier, work, policy).unwrap();
    let ExecutionBatchOutcome::Completed(batch) = executor
        .execute_tokens(std::slice::from_ref(&input))
        .await
        .unwrap()
    else {
        panic!("admitted CPU wave must finish");
    };
    assert_eq!(batch.completions().len(), 1);
    let completion = batch.into_completions().pop().unwrap();
    completion
        .validate_for(0, &input, executor.info().vocab_size)
        .unwrap();
    completion
}

fn selected(completion: &ExecutionCompletion) -> TokenId {
    match completion.product() {
        ExecutionProduct::Sampling(ExecutorSamplingOutput::GreedyToken(token)) => *token,
        ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(logits)) => {
            assert!(!logits.is_empty());
            assert!(logits.iter().all(|value| value.is_finite()));
            let mut index = 0;
            for candidate in 1..logits.len() {
                if logits[candidate].total_cmp(&logits[index]).is_gt() {
                    index = candidate;
                }
            }
            TokenId::new(index as u32)
        }
        other => panic!("expected actual sampling output, got {other:?}"),
    }
}

async fn retire(owner: ExecutorOwner, scope: ExecutorScope) {
    drop(scope);
    owner.begin_shutdown().wait().await.result().unwrap();
}

#[tokio::test]
async fn managed_qwen35_partial_prefill_and_decode_match_original_full_logits() {
    let (directory, defined, prepared, engine) = fixture();
    let parent = host(1 << 18, 16);
    // H3 hands the causal phase an existing admitted scope, not an unrelated
    // unlimited account. Both static and dynamic I/O must consume its credit.
    let phase = parent
        .reserve_scope(&[
            (HostMemoryClass::InputPreprocess, 1 << 18),
            (HostMemoryClass::TransferStaging, 1 << 18),
            (HostMemoryClass::Output, 1 << 18),
            (HostMemoryClass::WeightBacking, 1 << 18),
        ])
        .unwrap();
    let host = phase.account();
    let (managed, managed_runtime) = executor(&defined, &prepared, &engine, Some(&host));
    let (legacy, legacy_runtime) = executor(&defined, &prepared, &engine, None);
    let managed_owner = ExecutorOwner::new(Box::new(managed));
    let legacy_owner = ExecutorOwner::new(Box::new(legacy));
    let managed = managed_owner.scope().unwrap();
    let legacy = legacy_owner.scope().unwrap();
    let prompt = [TokenId::new(0), TokenId::new(1)];
    let mut actual = admit(&managed, &prompt).await;
    let mut expected = admit(&legacy, &prompt).await;
    let first = TokenWork::Prefill(PrefillChunk::new(0, 1, 2).unwrap());
    let a = execute(&managed, &actual, first, &greedy()).await;
    let e = execute(&legacy, &expected, first, &LogitsReturnPolicy::FullLogits).await;
    assert!(matches!(a.product(), ExecutionProduct::Intermediate));
    assert!(matches!(e.product(), ExecutionProduct::Intermediate));
    actual = a.next_frontier().clone();
    expected = e.next_frontier().clone();
    let final_prompt = TokenWork::Prefill(PrefillChunk::new(1, 1, 2).unwrap());
    let a = execute(&managed, &actual, final_prompt, &greedy()).await;
    let e = execute(
        &legacy,
        &expected,
        final_prompt,
        &LogitsReturnPolicy::FullLogits,
    )
    .await;
    assert!(matches!(
        a.product(),
        ExecutionProduct::Sampling(ExecutorSamplingOutput::GreedyToken(_))
    ));
    assert!(matches!(
        e.product(),
        ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(_))
    ));
    assert_eq!(selected(&a), selected(&e));
    actual = a.next_frontier().clone();
    expected = e.next_frontier().clone();
    let mut token = selected(&e);
    // Distinct decode positions exercise both recurrent state and paged KV.
    for position in [2, 3] {
        let work = TokenWork::Decode { token, position };
        let a = execute(&managed, &actual, work, &greedy()).await;
        let e = execute(&legacy, &expected, work, &LogitsReturnPolicy::FullLogits).await;
        assert_eq!(selected(&a), selected(&e));
        actual = a.next_frontier().clone();
        expected = e.next_frontier().clone();
        token = selected(&e);
    }
    for (executor, frontier) in [(&managed, &actual), (&legacy, &expected)] {
        assert!(matches!(
            executor
                .finish_tokens(
                    frontier.sequence(),
                    ExecutionEndReason::Completed {
                        input_tokens: prompt.len(),
                        output_tokens: 3,
                    }
                )
                .await
                .unwrap()
                .disposition,
            ExecutionCloseDisposition::Ended(_)
        ));
    }
    retire(managed_owner, managed).await;
    retire(legacy_owner, legacy).await;
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    phase.close().unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
    assert_eq!(managed_runtime.resident_bytes(), 0);
    assert_eq!(legacy_runtime.resident_bytes(), 0);
    std::fs::remove_dir_all(directory).unwrap();
}

#[tokio::test]
async fn managed_qwen35_two_participants_keep_original_token_rows_and_outputs() {
    let (directory, defined, prepared, engine) = fixture();
    let host = host(1 << 18, 16);
    let (managed, managed_runtime) = executor(&defined, &prepared, &engine, Some(&host));
    let (legacy, legacy_runtime) = executor(&defined, &prepared, &engine, None);
    let managed_owner = ExecutorOwner::new(Box::new(managed));
    let legacy_owner = ExecutorOwner::new(Box::new(legacy));
    let managed = managed_owner.scope().unwrap();
    let legacy = legacy_owner.scope().unwrap();
    let prompts = [
        [TokenId::new(0), TokenId::new(1)],
        [TokenId::new(2), TokenId::new(0)],
    ];
    let mut actual = Vec::new();
    let mut expected = Vec::new();
    for prompt in &prompts {
        actual.push(admit(&managed, prompt).await);
        expected.push(admit(&legacy, prompt).await);
    }
    let work = TokenWork::Prefill(PrefillChunk::new(0, 2, 2).unwrap());
    let greedy = greedy();
    let full = LogitsReturnPolicy::FullLogits;
    let inputs = actual
        .iter()
        .map(|frontier| ExecutionWork::new(frontier, work, &greedy).unwrap())
        .collect::<Vec<_>>();
    let ExecutionBatchOutcome::Completed(batch) = managed.execute_tokens(&inputs).await.unwrap()
    else {
        panic!("two admitted managed participants must finish");
    };
    assert_eq!(batch.completions().len(), prompts.len());
    assert_ne!(
        selected(&batch.completions()[0]),
        selected(&batch.completions()[1]),
        "the original prompts must distinguish swapped participant outputs"
    );
    for (index, completion) in batch.completions().iter().enumerate() {
        completion
            .validate_for(index, &inputs[index], managed.info().vocab_size)
            .unwrap();
        // Independent execution of each original request is the transport
        // oracle; the reference executor has no shared packed participant row.
        let reference = execute(&legacy, &expected[index], work, &full).await;
        assert_eq!(selected(completion), selected(&reference));
        expected[index] = reference.next_frontier().clone();
    }
    drop(inputs);
    for completion in batch.completions() {
        assert!(matches!(
            managed
                .cancel_tokens(completion.next_frontier().sequence())
                .unwrap()
                .disposition,
            ExecutionCloseDisposition::Ended(_)
        ));
    }
    for frontier in &expected {
        assert!(matches!(
            legacy
                .cancel_tokens(frontier.sequence())
                .unwrap()
                .disposition,
            ExecutionCloseDisposition::Ended(_)
        ));
    }
    drop(batch);
    retire(managed_owner, managed).await;
    retire(legacy_owner, legacy).await;
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(managed_runtime.resident_bytes(), 0);
    assert_eq!(legacy_runtime.resident_bytes(), 0);
    std::fs::remove_dir_all(directory).unwrap();
}

#[tokio::test]
async fn managed_qwen35_mixed_partial_and_final_prefill_use_actual_readback_modes() {
    let (directory, defined, prepared, engine) = fixture();
    let host = host(1 << 18, 16);
    let (managed, managed_runtime) = executor(&defined, &prepared, &engine, Some(&host));
    let (legacy, legacy_runtime) = executor(&defined, &prepared, &engine, None);
    let managed_owner = ExecutorOwner::new(Box::new(managed));
    let legacy_owner = ExecutorOwner::new(Box::new(legacy));
    let managed = managed_owner.scope().unwrap();
    let legacy = legacy_owner.scope().unwrap();
    let policy = greedy();
    let full = LogitsReturnPolicy::FullLogits;
    let first = admit(&managed, &[TokenId::new(0)]).await;
    let reference_first = admit(&legacy, &[TokenId::new(0)]).await;
    let prefill = TokenWork::Prefill(PrefillChunk::new(0, 1, 1).unwrap());
    let a = execute(&managed, &first, prefill, &policy).await;
    let e = execute(&legacy, &reference_first, prefill, &full).await;
    assert_eq!(selected(&a), selected(&e));
    let mut active = a.next_frontier().clone();
    let mut reference_active = e.next_frontier().clone();
    let mut next_token = selected(&e);
    let mut joining = admit(&managed, &[TokenId::new(2), TokenId::new(1)]).await;
    let mut reference_joining = admit(&legacy, &[TokenId::new(2), TokenId::new(1)]).await;
    for start in [0, 1] {
        let prefill = TokenWork::Prefill(PrefillChunk::new(start, 1, 2).unwrap());
        let decode = TokenWork::Decode {
            token: next_token,
            position: active.consumed_tokens(),
        };
        let inputs = [
            ExecutionWork::new(&joining, prefill, &policy).unwrap(),
            ExecutionWork::new(&active, decode, &policy).unwrap(),
        ];
        let ExecutionBatchOutcome::Completed(batch) =
            managed.execute_tokens(&inputs).await.unwrap()
        else {
            panic!("managed mixed token wave must complete");
        };
        assert_eq!(batch.completions().len(), 2);
        for (index, completion) in batch.completions().iter().enumerate() {
            completion
                .validate_for(index, &inputs[index], managed.info().vocab_size)
                .unwrap();
        }
        let expected_join = execute(&legacy, &reference_joining, prefill, &full).await;
        let expected_active = execute(&legacy, &reference_active, decode, &full).await;
        if start == 0 {
            assert!(matches!(
                batch.completions()[0].product(),
                ExecutionProduct::Intermediate
            ));
            assert!(matches!(
                expected_join.product(),
                ExecutionProduct::Intermediate
            ));
        } else {
            assert_eq!(selected(&batch.completions()[0]), selected(&expected_join));
        }
        assert_eq!(
            selected(&batch.completions()[1]),
            selected(&expected_active)
        );
        joining = batch.completions()[0].next_frontier().clone();
        active = batch.completions()[1].next_frontier().clone();
        reference_joining = expected_join.next_frontier().clone();
        reference_active = expected_active.next_frontier().clone();
        next_token = selected(&expected_active);
    }
    for frontier in [&joining, &active] {
        assert!(matches!(
            managed
                .cancel_tokens(frontier.sequence())
                .unwrap()
                .disposition,
            ExecutionCloseDisposition::Ended(_)
        ));
    }
    for frontier in [&reference_joining, &reference_active] {
        assert!(matches!(
            legacy
                .cancel_tokens(frontier.sequence())
                .unwrap()
                .disposition,
            ExecutionCloseDisposition::Ended(_)
        ));
    }
    retire(managed_owner, managed).await;
    retire(legacy_owner, legacy).await;
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(managed_runtime.resident_bytes(), 0);
    assert_eq!(legacy_runtime.resident_bytes(), 0);
    std::fs::remove_dir_all(directory).unwrap();
}

#[tokio::test]
async fn managed_f16_prefill_requires_full_logits_budget_before_submission_and_retries() {
    prefill_requires_full_logits_budget(ManagedCausalIoProfile::ManagedGreedyF16V1, 2).await;
}

#[tokio::test]
async fn managed_f32_prefill_requires_full_logits_budget_before_submission_and_retries() {
    prefill_requires_full_logits_budget(ManagedCausalIoProfile::ManagedGreedyF32V1, 4).await;
}

async fn prefill_requires_full_logits_budget(
    profile: ManagedCausalIoProfile,
    element_bytes: usize,
) {
    let (directory, defined, prepared, engine) = fixture_for(profile);
    assert_eq!(prepared.descriptor().vocabulary_size(), 3);
    let host = host(3 * element_bytes, 1);
    let (executor, runtime) =
        executor_with_profile(&defined, &prepared, &engine, Some(&host), profile);
    let owner = ExecutorOwner::new(Box::new(executor));
    let executor = owner.scope().unwrap();
    let initial = admit(&executor, &[TokenId::new(0), TokenId::new(1)]).await;
    let work = TokenWork::Prefill(PrefillChunk::new(0, 2, 2).unwrap());
    let policy = greedy();
    let hold = host.allocate_zeroed(HostMemoryClass::Output, 1).unwrap();
    let input = ExecutionWork::new(&initial, work, &policy).unwrap();
    let before = submissions(&executor);
    assert!(executor
        .execute_tokens(std::slice::from_ref(&input))
        .await
        .is_err());
    assert_eq!(
        submissions(&executor),
        before,
        "Host refusal precedes native submission"
    );
    assert_eq!(
        host.usage()
            .unwrap()
            .class_committed_bytes(HostMemoryClass::Output),
        1
    );
    assert_eq!(
        host.usage()
            .unwrap()
            .class_committed_bytes(HostMemoryClass::InputPreprocess),
        0
    );
    drop(hold);
    let done = execute(&executor, &initial, work, &policy).await;
    assert!(submissions(&executor) > before);
    assert!(matches!(
        done.product(),
        ExecutionProduct::Sampling(ExecutorSamplingOutput::GreedyToken(_))
    ));
    assert!(matches!(
        executor
            .cancel_tokens(done.next_frontier().sequence())
            .unwrap()
            .disposition,
        ExecutionCloseDisposition::Ended(_)
    ));
    retire(owner, executor).await;
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(runtime.resident_bytes(), 0);
    std::fs::remove_dir_all(directory).unwrap();
}

#[tokio::test]
async fn managed_greedy_profile_rejects_full_logits_without_advancing_frontier() {
    let (directory, defined, prepared, engine) = fixture();
    let host = host(1 << 18, 16);
    let (executor, runtime) = executor(&defined, &prepared, &engine, Some(&host));
    let owner = ExecutorOwner::new(Box::new(executor));
    let executor = owner.scope().unwrap();
    let initial = admit(&executor, &[TokenId::new(1)]).await;
    let work = TokenWork::Prefill(PrefillChunk::new(0, 1, 1).unwrap());
    let unsupported = LogitsReturnPolicy::FullLogits;
    let input = ExecutionWork::new(&initial, work, &unsupported).unwrap();
    let before = submissions(&executor);
    assert!(executor.execute_tokens(&[input]).await.is_err());
    assert_eq!(submissions(&executor), before);
    let done = execute(&executor, &initial, work, &greedy()).await;
    assert_eq!(done.next_frontier().consumed_tokens(), 1);
    assert!(matches!(
        executor
            .cancel_tokens(initial.sequence())
            .unwrap()
            .disposition,
        ExecutionCloseDisposition::Ended(_)
    ));
    retire(owner, executor).await;
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(runtime.resident_bytes(), 0);
    std::fs::remove_dir_all(directory).unwrap();
}
