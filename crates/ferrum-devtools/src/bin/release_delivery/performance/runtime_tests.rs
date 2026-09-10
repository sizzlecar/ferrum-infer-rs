use super::*;
use ferrum_types::ModelOutputProtocol;
use serde_json::json;

fn target(path: &str) -> ExecutionTarget {
    ExecutionTarget {
        architecture: "declared-dense".into(),
        protocol: ModelOutputProtocol::Text,
        precision: "gguf-q4_k_m".into(),
        backend: Backend::Metal,
        execution_path: path.into(),
    }
}
fn identity(target: &ExecutionTarget, role: ServerRole) -> ServerIdentity<'_> {
    ServerIdentity {
        version: "0.8.9",
        target,
        role,
        memory_budget_bytes: 8192,
        workload: Workload {
            input_tokens: 32,
            output_tokens: 8,
            measured_requests: 4,
            warmup_requests: 2,
            repeats: 3,
            seed: 7,
            max_model_len: 1024,
            concurrency: 2,
            max_num_batched_tokens: Some(512),
        },
    }
}
pub(super) fn health() -> Value {
    json!({"status":"healthy","version":"0.8.9","auto_config":{
        "hardware_capabilities":{"backend":"Metal"},
        "execution_resource_authority":"plan_runtime",
        "selected_kv_capacity":null,"selected_max_model_len":1024,
        "selected_max_sequences":2,"selected_max_batched_tokens":512,
        "admission":{"kv_capacity_tokens":32768}
    },"cache":{"prefix_cache":{
        "schema":"ferrum.runtime-vnext.executor-trace.v1",
        "device_id":"device.metal.test",
        "plan_hash":"a".repeat(64),"family_fingerprint":"b".repeat(64),
        "program_fingerprint":"c".repeat(64),"runtime_fingerprint":"d".repeat(64),
        "maximum_model_tokens":1024,"static_bytes":2048,
        "runtime_memory_policy":{"capacity_bytes":16384,"reserve_bytes":8192,"maximum_active_sequences":2},
        "runtime_admission_policy":{"maximum_scheduled_tokens":512},
        "numerical_execution":{
            "requested":"auto","selected_profile":"dense.f32-master",
            "selected_version":{"major":1,"minor":0},
            "qualification_version":{"major":1,"minor":0},
            "profile_fingerprint":"e".repeat(64),"definition_fingerprint":"f".repeat(64),
            "prepared_family_fingerprint":"b".repeat(64),
            "capability_catalog_fingerprint":"1".repeat(64),
            "runtime_policy_fingerprint":"2".repeat(64),"execution_plan_hash":"a".repeat(64)
        }
    }}})
}

#[test]
fn vnext_requires_observed_backend_capacities_plan_and_numerical_selection() {
    let target = target("production-plan-runtime");
    let identity = identity(&target, ServerRole::Candidate);
    identity.verify(&health()).unwrap();
    for (pointer, value) in [
        ("/version", json!("0.8.8")),
        ("/auto_config/hardware_capabilities/backend", json!("cpu")),
        ("/auto_config/execution_resource_authority", json!("legacy")),
        ("/auto_config/selected_max_sequences", json!(1)),
        ("/auto_config/selected_max_model_len", json!(512)),
        ("/auto_config/selected_max_batched_tokens", Value::Null),
        ("/cache/prefix_cache/device_id", json!("device.cuda.test")),
        ("/cache/prefix_cache/maximum_model_tokens", json!(2048)),
        (
            "/cache/prefix_cache/runtime_memory_policy/maximum_active_sequences",
            json!(1),
        ),
        (
            "/cache/prefix_cache/runtime_admission_policy/maximum_scheduled_tokens",
            json!(1024),
        ),
        ("/cache/prefix_cache/plan_hash", json!("unbound")),
        ("/cache/prefix_cache/numerical_execution", Value::Null),
        (
            "/cache/prefix_cache/numerical_execution/selected_profile",
            json!(""),
        ),
        (
            "/cache/prefix_cache/numerical_execution/selected_version/major",
            json!(-1),
        ),
        (
            "/cache/prefix_cache/numerical_execution/execution_plan_hash",
            json!("0".repeat(64)),
        ),
        (
            "/cache/prefix_cache/numerical_execution/prepared_family_fingerprint",
            json!("0".repeat(64)),
        ),
    ] {
        let mut invalid = health();
        *invalid.pointer_mut(pointer).unwrap() = value;
        assert!(identity.verify(&invalid).is_err(), "accepted {pointer}");
    }
}

#[test]
fn baseline_without_named_profile_remains_unknown_and_cannot_mask_invalid_observations() {
    let target = target("production-plan-runtime");
    let mut value = health();
    value["cache"]["prefix_cache"]["numerical_execution"] = Value::Null;
    identity(&target, ServerRole::Baseline)
        .verify(&value)
        .unwrap();
    assert!(identity(&target, ServerRole::Candidate)
        .verify(&value)
        .is_err());
    value["cache"]["prefix_cache"]["numerical_execution"] = json!({"selected_profile":"invented"});
    assert!(identity(&target, ServerRole::Baseline)
        .verify(&value)
        .is_err());
}

#[test]
fn memory_checks_use_reserved_capacity_and_actual_static_residency() {
    let target = target("production-plan-runtime");
    let identity = identity(&target, ServerRole::Candidate);
    for (capacity, reserve, static_bytes) in [
        (16384, 8191, 2048), // exceeds the predeclared ceiling
        (8192, 8193, 2048),
        (8192, 8192, 0), // underflow or no capacity
        (16384, 8192, 8193),
        (16384, 8192, 0), // invalid resident allocation
    ] {
        let mut value = health();
        let trace = &mut value["cache"]["prefix_cache"];
        trace["runtime_memory_policy"]["capacity_bytes"] = json!(capacity);
        trace["runtime_memory_policy"]["reserve_bytes"] = json!(reserve);
        trace["static_bytes"] = json!(static_bytes);
        assert!(identity.verify(&value).is_err());
    }
}

#[test]
fn measured_plan_cannot_change_between_health_observations() {
    let target = target("production-plan-runtime");
    let identity = identity(&target, ServerRole::Candidate);
    let before = health();
    identity.verify_after(&before, &before).unwrap();
    let mut after = health();
    after["cache"]["prefix_cache"]["plan_hash"] = json!("0".repeat(64));
    after["cache"]["prefix_cache"]["numerical_execution"]["execution_plan_hash"] =
        json!("0".repeat(64));
    identity.verify(&after).unwrap();
    assert!(identity.verify_after(&before, &after).is_err());
}

#[test]
fn legacy_requires_selected_kv_capacity_and_rejects_a_plan_runtime() {
    let target = target("legacy-model-executor");
    let identity = identity(&target, ServerRole::Baseline);
    let mut value = health();
    assert!(identity.verify(&value).is_err());
    value["auto_config"]["execution_resource_authority"] = Value::Null;
    value["auto_config"]["selected_kv_capacity"] = json!(1024);
    identity.verify(&value).unwrap();
    for capacity in [Value::Null, json!(0), json!(512), json!(32768)] {
        value["auto_config"]["selected_kv_capacity"] = capacity;
        assert!(identity.verify(&value).is_err());
    }
}
