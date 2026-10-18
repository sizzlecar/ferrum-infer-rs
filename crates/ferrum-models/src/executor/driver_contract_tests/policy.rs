//! Public driver policy projection using the actual reference composition.
use ferrum_execution::ProgramExecutorConfig;
use ferrum_runtime::{DeviceId, SchedulingDiscipline};
use ferrum_types::{Device, EngineConfig, InvocationPreparationStrategy, ModelInfo};

#[test]
fn product_flag_resolves_to_existing_memory_budget_without_reserving_slots() {
    use ferrum_kernels::backend::reference::ReferenceVNextComposition;
    use ferrum_types::{DataType, ModelType};
    let composition =
        ReferenceVNextComposition::create(DeviceId::new("device.prefix-policy").unwrap()).unwrap();
    let info = ModelInfo {
        model_id: "policy-fixture".into(),
        model_type: ModelType::Custom("policy-fixture".into()),
        num_parameters: 0,
        hidden_size: 4,
        num_layers: 1,
        num_heads: 1,
        num_kv_heads: 1,
        vocab_size: 16,
        max_sequence_length: 64,
        dtype: DataType::FP16,
        device: Device::CPU,
        version: None,
        license: None,
        metadata: Default::default(),
    };
    let mut engine = EngineConfig::default();
    // Native checkpoint capacity follows its own explicit state-cache policy.
    engine.runtime.prefix_state_cache_enabled = false;
    let disabled =
        ProgramExecutorConfig::from_engine_config(&engine, &info, composition.runtime().as_ref())
            .unwrap();
    assert!(disabled
        .runtime_policy
        .memory()
        .checkpoint_capacity
        .is_none());
    engine.runtime.prefix_state_cache_enabled = true;
    let enabled =
        ProgramExecutorConfig::from_engine_config(&engine, &info, composition.runtime().as_ref())
            .unwrap();
    let memory = enabled.runtime_policy.memory();
    assert_eq!(
        memory.checkpoint_capacity.unwrap().maximum_retained_bytes(),
        memory.capacity_bytes - memory.reserve_bytes
    );
    assert_eq!(
        memory.maximum_active_sequences,
        disabled.runtime_policy.memory().maximum_active_sequences
    );
    assert_eq!(
        enabled.runtime_policy.admission(),
        disabled.runtime_policy.admission()
    );
}

#[test]
fn invocation_preparation_reaches_shared_executor_without_changing_device_policy() {
    use ferrum_kernels::backend::reference::ReferenceVNextComposition;
    use ferrum_types::{DataType, ModelType, RuntimeConfigSnapshot};
    let composition = ReferenceVNextComposition::create(
        DeviceId::new("device.invocation-preparation-policy").unwrap(),
    )
    .unwrap();
    let info = ModelInfo {
        model_id: "preparation-policy-fixture".into(),
        model_type: ModelType::Custom("preparation-policy-fixture".into()),
        num_parameters: 0,
        hidden_size: 4,
        num_layers: 1,
        num_heads: 1,
        num_kv_heads: 1,
        vocab_size: 16,
        max_sequence_length: 64,
        dtype: DataType::FP16,
        device: Device::CPU,
        version: None,
        license: None,
        metadata: Default::default(),
    };
    let mut engine = EngineConfig::default();
    let default =
        ProgramExecutorConfig::from_engine_config(&engine, &info, composition.runtime().as_ref())
            .unwrap();
    assert_eq!(
        default.runtime_policy.scheduling(),
        SchedulingDiscipline::FirstReady
    );
    assert_eq!(
        default.invocation_preparation_strategy,
        InvocationPreparationStrategy::Full
    );
    for strategy in [
        InvocationPreparationStrategy::IdentityProjection,
        InvocationPreparationStrategy::DecodeSegment,
        InvocationPreparationStrategy::Full,
    ] {
        engine
            .apply_runtime_config_snapshot(&RuntimeConfigSnapshot::from_env_vars([(
                "FERRUM_INVOCATION_PREPARATION_STRATEGY",
                strategy.as_runtime_value(),
            )]))
            .unwrap();
        let config = ProgramExecutorConfig::from_engine_config(
            &engine,
            &info,
            composition.runtime().as_ref(),
        )
        .unwrap();
        assert_eq!(config.invocation_preparation_strategy, strategy);
        assert_eq!(
            config.runtime_policy.fingerprint_str(),
            default.runtime_policy.fingerprint_str()
        );
        assert_eq!(config.maximum_model_tokens, default.maximum_model_tokens);
        assert_eq!(
            config.device_reusable_execution_enabled,
            default.device_reusable_execution_enabled
        );
    }
}
