use super::*;
use ferrum_execution::executor_lifecycle::ExecutorOwner;
use ferrum_runtime::HostMemoryClass;
use ferrum_types::{FinishReason, RequestId};
pub(super) mod fixture;
use fixture::*;
mod process_mapped;
mod source_backing;

#[tokio::test]
async fn native_bf16_source_f16_phase_generates_and_extracts_the_protocol_eos_answer() {
    let fixture = Fixture::new();
    let expansion = fixture.expansion();
    let prepared = prepared(expansion, 2);
    assert_eq!(prepared.generation.eos_token_id, EOS_ID);
    assert_ne!(
        prepared.generation.eos_token_id,
        prepared.generation.pad_token_id
    );
    assert_eq!(
        prepared.engine.runtime.max_model_len,
        Some(prepared.tokens.len() + 2)
    );
    let (composition, runtime) = composition();
    let host = host();
    let requirements = expansion
        .host_requirements(&composition, &prepared, &host)
        .unwrap();
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(
        runtime.resident_bytes(),
        0,
        "metadata compilation must not allocate device memory"
    );
    assert!(requirements.startup_device_bytes > 0);
    let phase = host.reserve_scope(&requirements.claims()).unwrap();
    let phase_host = phase.account();
    let executor = expansion
        .initialize(&composition, &prepared, &phase_host)
        .unwrap();
    let owner = ExecutorOwner::new(Box::new(executor));
    owner.shutdown_observer().retain_until_retired(phase);
    let scope = owner.scope().unwrap();
    let result = execute::execute(
        &scope,
        &RequestId::new(),
        &prepared.tokens,
        prepared.generation,
        prepared.prefill_chunk_tokens,
        &phase_host,
        &|| false,
    )
    .await
    .unwrap();
    assert_eq!(result.history.ids(), &[ANSWER_ID, EOS_ID]);
    assert_eq!(result.finish, FinishReason::EOS);
    assert_eq!(
        expansion
            .extract(&prepared, result.history.ids(), result.finish)
            .unwrap(),
        "A small red cat sits quietly."
    );
    assert!(
        phase_host
            .usage()
            .unwrap()
            .class_allocated_bytes(HostMemoryClass::Output)
            > 0
    );
    drop(result);
    drop(scope);
    owner.begin_shutdown().wait().await.result().unwrap();
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(runtime.resident_bytes(), 0);
}

#[tokio::test]
async fn native_expansion_length_never_publishes_an_answer_and_retires_original_owners() {
    let fixture = Fixture::new();
    let expansion = fixture.expansion();
    let prepared = prepared(expansion, 1);
    let (composition, runtime) = composition();
    let host = host();
    let requirements = expansion
        .host_requirements(&composition, &prepared, &host)
        .unwrap();
    let phase = host.reserve_scope(&requirements.claims()).unwrap();
    let phase_host = phase.account();
    let executor = expansion
        .initialize(&composition, &prepared, &phase_host)
        .unwrap();
    let owner = ExecutorOwner::new(Box::new(executor));
    let scope = owner.scope().unwrap();
    let failed = execute::execute(
        &scope,
        &RequestId::new(),
        &prepared.tokens,
        prepared.generation,
        prepared.prefill_chunk_tokens,
        &phase_host,
        &|| false,
    )
    .await
    .unwrap_err();
    assert_eq!(failed.finish, FinishReason::Length);
    let ids = failed.history.as_ref().unwrap().ids();
    assert_eq!(ids, &[ANSWER_ID]);
    assert!(expansion.extract(&prepared, ids, failed.finish).is_err());
    // Preserve the actual failed request and its managed history with the
    // original cleanup authority even after all public observers are dropped.
    owner
        .shutdown_observer()
        .retain_until_retired((phase, failed));
    assert!(host.usage().unwrap().allocated_bytes() > 0);
    drop(scope);
    owner.begin_shutdown().wait().await.result().unwrap();
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(runtime.resident_bytes(), 0);
}

#[test]
fn native_preparation_stays_source_bound_and_checks_context_before_installation() {
    let first = Fixture::new();
    let second = Fixture::new();
    let prepared = prepared(first.expansion(), 2);
    let (composition, runtime) = composition();
    let host = host();
    assert!(second
        .expansion()
        .host_requirements(&composition, &prepared, &host)
        .is_err());
    let mut too_short = EngineConfig::default();
    too_short.runtime.max_model_len = Some(1);
    assert!(first
        .expansion()
        .prepare_text("draw a cat", 1024, 1024, 2, 2, too_short)
        .is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(runtime.resident_bytes(), 0);
}
