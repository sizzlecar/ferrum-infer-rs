//! The real hybrid CPU program exercises the product startup allocation hook.
use super::*;
use ferrum_execution::model_executor::{
    PersistentCapacityMember, PersistentCapacityQuery, PersistentCapacityState,
};
use ferrum_interfaces::request_observation::RequestLifecycleId;
use ferrum_kernels::backend::cpu::vnext_ops::CpuVNextComposition;
use ferrum_runtime::{AllocationLifetime, DeviceId, WeightMaterializerSelection};
use ferrum_types::{PersistentPoolPreparation, RequestId};

#[test]
fn persistent_startup_direct_legacy_factory_rejects_before_loading() {
    let mut engine = EngineConfig::default();
    engine.memory.persistent_pool_preparation = PersistentPoolPreparation::ConfiguredEnvelope;
    let config = ComponentConfig::from_engine_config(&engine);
    for error in [
        tokio_test::block_on(LlmExecutorFactory.create(&config))
            .err()
            .unwrap(),
        tokio_test::block_on(StubExecutorFactory.create(&config))
            .err()
            .unwrap(),
    ] {
        assert!(
            error.to_string().contains("compiled-plan executor"),
            "{error}"
        );
    }
}

#[tokio::test]
async fn persistent_startup_real_hybrid_cpu_allocation_default_and_budget_boundaries() {
    let directory = write_qwen35_fixture_model_dir();
    std::fs::write(directory.join("tokenizer.json"), br#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<unk>":0,"hello":1,"<eos>":2},"unk_token":"<unk>"}}"#).unwrap();
    std::fs::write(directory.join("tokenizer_config.json"), br#"{"chat_template":"{% for message in messages %}{{ message['content'] }}{% endfor %}","eos_token_id":2,"unk_token":"<unk>"}"#).unwrap();
    let defined = ferrum_models::vnext::qwen35::define_from_model_dir(&directory).unwrap();
    let mut config = qwen35_fixture_component_config(&directory).engine_config;
    config.runtime.max_model_len = Some(16);
    config.scheduler.max_running_requests = 4;
    config.batching.max_batch_size = 4;
    config.batching.max_num_batched_tokens = 8;
    config.backend.enable_reusable_execution = false;
    let build = |config: &EngineConfig| {
        let (runtime, operations, materializers, materializer, catalog) =
            CpuVNextComposition::create(
                DeviceId::new(format!(
                    "device.cpu.persistent-startup-{}",
                    uuid::Uuid::new_v4()
                ))
                .unwrap(),
                16 << 20,
            )
            .unwrap()
            .into_parts()
            .unwrap();
        let observed = Arc::clone(&runtime);
        let result = crate::product_composition::create_program_executor(
            config,
            &defined,
            runtime,
            operations,
            materializers,
            catalog,
            |_| Ok(WeightMaterializerSelection::exact(materializer.clone())),
            None,
        );
        (observed, result)
    };
    let (default_runtime, baseline) = build(&config);
    let baseline = baseline.unwrap();
    let memory = baseline
        .resolved_model_plan()
        .unwrap()
        .execution_plan()
        .payload()
        .memory();
    assert!(memory
        .dynamic_descriptors()
        .iter()
        .any(|d| d.lifetime() == AllocationLifetime::Sequence));
    let plan = memory.persistent_preallocation_plan(16, 4, 8).unwrap();
    assert!(plan.persistent_bytes() > 0);
    let baseline_health = baseline.cache_metrics_snapshot().unwrap();
    assert_eq!(
        baseline_health["startup_persistent_preparation"]["mode"],
        "demand-driven"
    );
    assert!(baseline_health["startup_persistent_preparation"]["actual_growth"].is_null());
    assert!(baseline_health["dynamic_pools"]["pools"]
        .as_array()
        .unwrap()
        .iter()
        .all(|pool| pool["resident_bytes"] == 0));
    let static_resident = default_runtime.resident_bytes();
    let owner = ExecutorOwner::new(Box::new(baseline));
    assert_eq!(
        owner.begin_shutdown().wait().await.phase(),
        ferrum_execution::executor_lifecycle::ExecutorShutdownPhase::Retired
    );
    assert_eq!(default_runtime.resident_bytes(), 0);

    config.memory.persistent_pool_preparation = PersistentPoolPreparation::ConfiguredEnvelope;
    let (runtime, prepared) = build(&config);
    let prepared = prepared.unwrap();
    let health = prepared.cache_metrics_snapshot().unwrap();
    let receipt = &health["startup_persistent_preparation"];
    assert_eq!(receipt["mode"], "configured-envelope");
    assert_eq!(receipt["plan"]["context_tokens"], 16);
    assert_eq!(receipt["plan"]["active_sequences"], 4);
    assert_eq!(receipt["plan"]["step_tokens"], 8);
    assert_eq!(
        receipt["plan"]["checked_peak_bytes"],
        plan.checked_peak_bytes()
    );
    let growths = receipt["actual_growth"]["growths"].as_array().unwrap();
    assert_eq!(
        growths
            .iter()
            .map(|g| g["chunk_bytes"].as_u64().unwrap())
            .sum::<u64>(),
        plan.persistent_bytes()
    );
    assert_eq!(
        runtime.resident_bytes(),
        static_resident + plan.persistent_bytes()
    );
    let members: Vec<_> = (1..=8)
        .map(|admission| PersistentCapacityMember {
            request_id: RequestId::new(),
            lifecycle: RequestLifecycleId {
                producer: 1,
                admission,
            },
            state: PersistentCapacityState::Waiting,
            through_tokens: 16,
        })
        .collect();
    prepared
        .persistent_cohort_capacity(&PersistentCapacityQuery {
            members: &members,
            max_live_sequences: 4,
            max_wave_sequences: 4,
            inspection_budget: Default::default(),
        })
        .expect("actual full envelope covers waiting work through original four-slot owner");
    let owner = ExecutorOwner::new(Box::new(prepared));
    assert_eq!(
        owner.begin_shutdown().wait().await.phase(),
        ferrum_execution::executor_lifecycle::ExecutorShutdownPhase::Retired
    );
    assert_eq!(runtime.resident_bytes(), 0);

    config.memory.usable_capacity_bytes = Some((plan.checked_peak_bytes() - 1) as usize);
    let (runtime, rejected) = build(&config);
    assert!(rejected.is_err(), "full configured peak must be rejected");
    assert_eq!(
        runtime.peak_resident_bytes(),
        0,
        "budget rejection precedes weights and persistent allocations"
    );
    std::fs::remove_dir_all(directory).unwrap();
}
