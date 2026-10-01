use super::*;
use ferrum_types::{DataType, Device, EngineConfig, ModelInfo, ModelType};

#[test]
fn vnext_workload_domain_exports_real_configuration_and_compiled_limits() {
    let fixture = super::super::identity_contract::fixture();
    let info = ModelInfo {
        model_id: "domain-contract".into(),
        model_type: ModelType::Custom("domain-contract".into()),
        num_parameters: 0,
        hidden_size: 4,
        num_layers: 1,
        num_heads: 1,
        num_kv_heads: 1,
        vocab_size: 32,
        max_sequence_length: 64,
        dtype: DataType::FP16,
        device: Device::CPU,
        version: None,
        license: None,
        metadata: Default::default(),
    };
    let mut engine = EngineConfig::default();
    engine.backend.device = Device::CPU;
    engine.backend.enable_reusable_execution = false;
    engine.scheduler.max_running_requests = 3;
    engine.batching.max_num_batched_tokens = 5;
    engine.runtime.max_model_len = Some(17);
    let mut config =
        VNextExecutorConfig::from_engine_config(&engine, &info, fixture.runtime.as_ref()).unwrap();
    let compiled = fixture
        .resolved
        .execution_plan()
        .payload()
        .maximum_scheduled_tokens();
    let state = TypedSequenceStateMemory {
        kv_bytes_per_token: 128,
        other_token_scaled_bytes_per_token: 0,
        fixed_bytes_per_sequence: 64,
    };
    let exported = limits(&config, compiled, 32, 7, state).unwrap();
    assert_eq!(
        exported.maximum_rows.get(),
        config.runtime_policy.memory().maximum_active_sequences
    );
    assert_eq!(exported.maximum_rows.get(), 3);
    assert_eq!(exported.maximum_context_tokens.get(), 17);
    assert_eq!(
        exported.maximum_scheduled_tokens_per_wave.get(),
        compiled.min(5)
    );
    assert_eq!(exported.output_vocabulary_elements.get(), 32);
    assert_eq!(exported.repetition_slot_capacity, 7);
    assert_eq!(exported.fixed_state_bytes_per_row, 64);
    assert_eq!(
        limits(&config, 2, 32, 7, state)
            .unwrap()
            .maximum_scheduled_tokens_per_wave
            .get(),
        2
    );
    assert!(limits(&config, 0, 32, 7, state).is_none());
    assert!(limits(&config, compiled, 0, 7, state).is_none());
    config.maximum_model_tokens = 0;
    assert!(limits(&config, compiled, 32, 7, state).is_none());
    if let Some(too_large) = (u32::MAX as usize).checked_add(1) {
        config.maximum_model_tokens = too_large;
        assert!(limits(&config, compiled, 32, 7, state).is_none());
    }
    let mut checks = 0;
    super::super::identity_contract::close_plan_runtime(fixture.plan_resources, &mut checks);
}
