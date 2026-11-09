use super::tests::fixture::Fixture;
use super::*;

#[cfg(feature = "cuda")]
pub(super) mod cuda;
mod nonzero;

#[test]
fn decode_replay_preparation_is_opt_in_and_preserves_original_tokens() {
    let fixture = Fixture::new();
    let assets = fixture.expansion();
    let default = EngineConfig::default();
    let eager = assets
        .prepare_text("A cat", 32, 32, 2, 4, default.clone())
        .unwrap();
    assert!(!eager.engine.backend.enable_reusable_execution);
    let mut engine = default;
    engine.backend.reusable_execution_capture.scope =
        ferrum_types::ReusableExecutionCaptureScope::DecodeOnly;
    engine.backend.reusable_execution_capture.preparation =
        ferrum_types::ReusableExecutionPreparationMode::OnDemand;
    let replay = assets.prepare_text("A cat", 32, 32, 2, 4, engine).unwrap();
    assert!(replay.engine.backend.enable_reusable_execution);
    assert_eq!(eager.tokens, replay.tokens);
    assert_eq!(
        eager.engine.numerical_execution,
        replay.engine.numerical_execution
    );
    assert_eq!(
        eager.generation.eos_token_id,
        replay.generation.eos_token_id
    );
    assert_eq!(eager.prefill_chunk_tokens, replay.prefill_chunk_tokens);
}
