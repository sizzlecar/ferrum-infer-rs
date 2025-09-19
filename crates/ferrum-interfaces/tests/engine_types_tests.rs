use ferrum_interfaces::engine::{
    EngineStatus, ComponentStatus, ComponentHealth, ComponentHealthStatus,
};
use ferrum_types::{ModelId, MemoryUsage};
use serde_json as json;

#[test]
fn engine_status_serde_roundtrip() {
    let st = EngineStatus {
        is_ready: true,
        loaded_models: vec![ModelId::from("m")],
        active_requests: 1,
        queued_requests: 0,
        memory_usage: MemoryUsage { total_bytes: 1_000, used_bytes: 100, free_bytes: 900, gpu_memory_bytes: None, cpu_memory_bytes: Some(100), cache_memory_bytes: 0, utilization_percent: 10.0 },
        uptime_seconds: 5,
        last_heartbeat: chrono::Utc::now(),
        version: "0.1".into(),
        component_status: ComponentStatus {
            scheduler: ComponentHealth { status: ComponentHealthStatus::Healthy, message: "ok".into(), last_check: chrono::Utc::now(), metrics: Default::default() },
            model_executor: ComponentHealth { status: ComponentHealthStatus::Healthy, message: "ok".into(), last_check: chrono::Utc::now(), metrics: Default::default() },
            tokenizer: ComponentHealth { status: ComponentHealthStatus::Healthy, message: "ok".into(), last_check: chrono::Utc::now(), metrics: Default::default() },
            kv_cache: ComponentHealth { status: ComponentHealthStatus::Healthy, message: "ok".into(), last_check: chrono::Utc::now(), metrics: Default::default() },
            memory_manager: ComponentHealth { status: ComponentHealthStatus::Healthy, message: "ok".into(), last_check: chrono::Utc::now(), metrics: Default::default() },
            backend: ComponentHealth { status: ComponentHealthStatus::Healthy, message: "ok".into(), last_check: chrono::Utc::now(), metrics: Default::default() },
        },
    };
    let s = json::to_string(&st).unwrap();
    let back: EngineStatus = json::from_str(&s).unwrap();
    assert_eq!(back.is_ready, true);
}
