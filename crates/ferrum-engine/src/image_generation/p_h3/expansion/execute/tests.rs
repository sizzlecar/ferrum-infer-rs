//! Protocol/ownership fixtures using the existing Testkit executor. These are
//! not model numerical or GPU qualifications; the parent adds real Qwen runs.
use super::*;
use ferrum_execution::executor_lifecycle::ExecutorOwner;
use ferrum_runtime::{HostMemoryClassLimits, HostMemoryLimits};
use ferrum_testkit::{ConfigurableModelExecutor, MockModelExecutor};

mod interrupted;
mod maintenance;

fn host() -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            4096,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                transfer_staging_bytes: 0,
                output_bytes: 4096,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
        64,
    )
    .unwrap()
}
fn generation(max_new_tokens: u32) -> PH3ImagePromptExpansionGeneration {
    PH3ImagePromptExpansionGeneration {
        max_new_tokens,
        do_sample: false,
        eos_token_id: 13,
        pad_token_id: 0,
    }
}
fn prompt() -> [TokenId; 5] {
    [1, 4, 3, 6, 8].map(TokenId::new)
}

#[test]
fn borrowed_greedy_logits_preserve_signed_zero_and_finite_contract() {
    let product = ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(vec![
        f32::NAN,
        -0.0,
        0.0,
        f32::INFINITY,
    ]));
    assert_eq!(selected(&product).unwrap(), Some(TokenId::new(2)));
    let reverse =
        ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(vec![0.0, -0.0, 0.0]));
    assert_eq!(selected(&reverse).unwrap(), Some(TokenId::new(0)));
    let invalid = ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(vec![
        f32::NAN,
        f32::NEG_INFINITY,
    ]));
    assert!(selected(&invalid).is_err());
}
async fn retire(owner: ExecutorOwner) {
    let observer = owner.begin_shutdown();
    let report = observer.wait().await;
    report.result().unwrap();
}

#[test]
fn managed_history_uses_checked_alignment_native_words_and_one_physical_owner() {
    let host = host();
    let mut history = ManagedTokenHistory::new(&host, 4).unwrap();
    let physical = history_backing_bytes(&host, 4).unwrap();
    assert_eq!(host.usage().unwrap().allocated_bytes(), physical);
    for id in [0, 0x8000_0000, 0x0000_0001, u32::MAX] {
        history.push(TokenId::new(id)).unwrap();
    }
    assert_eq!(history.ids(), &[0, 0x8000_0000, 1, u32::MAX]);
    assert_eq!(
        (history.ids().as_ptr() as usize) % std::mem::align_of::<u32>(),
        0
    );
    assert!(history.push(TokenId::new(5)).is_err());
    assert_eq!(history.ids(), &[0, 0x8000_0000, 1, u32::MAX]);
    assert_eq!(history.backing_bytes(), physical);
    drop(history);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert!(ManagedTokenHistory::new(&host, 0).is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[tokio::test]
async fn protocol_eos_and_chunked_prefill_keep_the_real_eos_in_managed_history() {
    // Token 2 is an ordinary generated token. Only caller protocol EOS=13
    // stops this phase; no common/model-default EOS is substituted.
    let model = ConfigurableModelExecutor::with_token_sequence(64, vec![2, 7, 13]);
    let observed = model.observer();
    let owner = ExecutorOwner::new(Box::new(model));
    let scope = owner.scope().unwrap();
    let host = host();
    let required = history_backing_bytes(&host, 6).unwrap();
    let phase = host
        .reserve_scope(&[(HostMemoryClass::Output, required)])
        .unwrap();
    let output = execute(
        &scope,
        &RequestId::new(),
        &prompt(),
        generation(6),
        2,
        &phase.account(),
        &|| false,
    )
    .await
    .unwrap();
    assert_eq!(output.finish, FinishReason::EOS);
    assert_eq!(output.history.ids(), &[2, 2, 7, 13]);
    assert_eq!(output.frontier.consumed_tokens(), prompt().len() + 3);
    assert_eq!(output.ended.sequence(), output.frontier.sequence());
    assert!(output.frontier.sequence().observed_end().is_some());
    assert_eq!(observed.decode_count(), 3);
    phase.close().unwrap();
    assert_eq!(host.usage().unwrap().allocated_bytes(), required);
    drop(output);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    drop(scope);
    retire(owner).await;
}

#[tokio::test]
async fn one_missing_physical_output_byte_rejects_before_any_model_work_and_can_retry() {
    let model = MockModelExecutor::instant(64);
    let observed = model.observer();
    let owner = ExecutorOwner::new(Box::new(model));
    let scope = owner.scope().unwrap();
    let host = host();
    let policy = PH3ImagePromptExpansionGeneration {
        eos_token_id: 42,
        ..generation(1)
    };
    let required = history_backing_bytes(&host, 1).unwrap();
    let short = host
        .reserve_scope(&[(HostMemoryClass::Output, required - 1)])
        .unwrap();
    let request = RequestId::new();
    let error = execute(
        &scope,
        &request,
        &prompt(),
        policy,
        2,
        &short.account(),
        &|| false,
    )
    .await
    .unwrap_err();
    assert!(error.history.is_none());
    assert!(error.frontier.is_none());
    assert!(error.admission.is_none());
    assert_eq!(observed.prefill_count(), 0);
    assert_eq!(observed.decode_count(), 0);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    short.close().unwrap();
    let exact = host
        .reserve_scope(&[(HostMemoryClass::Output, required)])
        .unwrap();
    let output = execute(
        &scope,
        &request,
        &prompt(),
        policy,
        2,
        &exact.account(),
        &|| false,
    )
    .await
    .unwrap();
    assert_eq!(output.history.ids(), &[42]);
    assert_eq!(observed.prefill_count(), 3);
    assert_eq!(observed.decode_count(), 0);
    exact.close().unwrap();
    drop(output);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    drop(scope);
    retire(owner).await;
}

#[tokio::test]
async fn missing_eos_returns_length_with_original_history_and_real_terminal_identity() {
    let model = ConfigurableModelExecutor::with_token_sequence(64, vec![7]);
    let observed = model.observer();
    let owner = ExecutorOwner::new(Box::new(model));
    let scope = owner.scope().unwrap();
    let host = host();
    let error = execute(
        &scope,
        &RequestId::new(),
        &prompt(),
        generation(2),
        3,
        &host,
        &|| false,
    )
    .await
    .unwrap_err();
    assert_eq!(error.finish, FinishReason::Length);
    assert_eq!(error.history.as_ref().unwrap().ids(), &[7, 7]);
    assert_eq!(observed.decode_count(), 1);
    let ExecutionCloseDisposition::Ended(ended) = &error.close.as_ref().unwrap().disposition else {
        panic!("real Testkit close must end");
    };
    assert_eq!(
        ended.sequence(),
        error.frontier.as_ref().unwrap().sequence()
    );
    assert!(error.close_error.is_none());
    drop(error);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    drop(scope);
    retire(owner).await;
}

#[tokio::test]
async fn boundary_cancellation_preserves_completed_tokens_and_never_runs_another_decode() {
    let model = ConfigurableModelExecutor::with_token_sequence(64, vec![7]);
    let observed = model.observer();
    let owner = ExecutorOwner::new(Box::new(model));
    let scope = owner.scope().unwrap();
    let host = host();
    let error = execute(
        &scope,
        &RequestId::new(),
        &prompt(),
        generation(8),
        2,
        &host,
        &|| observed.decode_count() == 1,
    )
    .await
    .unwrap_err();
    assert_eq!(error.finish, FinishReason::Cancelled);
    assert_eq!(error.history.as_ref().unwrap().ids(), &[7, 7]);
    assert_eq!(
        error.frontier.as_ref().unwrap().consumed_tokens(),
        prompt().len() + 1
    );
    assert!(error.outcome.is_some());
    assert_eq!(observed.decode_count(), 1);
    assert!(matches!(
        error.close.as_ref().unwrap().disposition,
        ExecutionCloseDisposition::Ended(_)
    ));
    drop(error);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    drop(scope);
    retire(owner).await;
}

#[tokio::test]
async fn invalid_sampling_and_pre_admission_cancel_never_execute_the_model() {
    let model = MockModelExecutor::instant(64);
    let observed = model.observer();
    let owner = ExecutorOwner::new(Box::new(model));
    let scope = owner.scope().unwrap();
    let host = host();
    let invalid = PH3ImagePromptExpansionGeneration {
        do_sample: true,
        ..generation(8)
    };
    let error = execute(
        &scope,
        &RequestId::new(),
        &prompt(),
        invalid,
        2,
        &host,
        &|| false,
    )
    .await
    .unwrap_err();
    assert!(error.history.is_none());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    let cancelled = execute(
        &scope,
        &RequestId::new(),
        &prompt(),
        generation(8),
        2,
        &host,
        &|| true,
    )
    .await
    .unwrap_err();
    assert_eq!(cancelled.finish, FinishReason::Cancelled);
    assert!(cancelled.frontier.is_none());
    assert!(cancelled.history.as_ref().unwrap().ids().is_empty());
    assert_eq!(observed.prefill_count(), 0);
    assert_eq!(observed.decode_count(), 0);
    drop(cancelled);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    drop(scope);
    retire(owner).await;
}
