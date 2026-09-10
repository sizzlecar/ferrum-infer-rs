//! Validate the executor and capacity actually used by a measured server.
use ferrum_bench_core::release_regression::{performance::Workload, Backend, ExecutionTarget};
use serde_json::Value;

#[derive(Clone, Copy)]
pub(super) enum ServerRole {
    Baseline,
    Candidate,
}

pub(super) struct ServerIdentity<'a> {
    pub version: &'a str,
    pub target: &'a ExecutionTarget,
    pub workload: Workload,
    pub memory_budget_bytes: u64,
    pub role: ServerRole,
}

pub(super) fn backend_name(backend: Backend) -> &'static str {
    match backend {
        Backend::Cpu => "cpu",
        Backend::Metal => "metal",
        Backend::Cuda => "cuda",
    }
}

pub(super) fn is_plan_runtime(target: &ExecutionTarget) -> bool {
    target.execution_path == "production-plan-runtime"
}

fn equal_count(parent: &Value, field: &str, expected: u64) -> Result<(), String> {
    if expected == 0 || parent[field].as_u64() != Some(expected) {
        return Err(format!(
            "performance server actual capacity mismatch: {field} expected {expected}, observed {}",
            parent[field]
        ));
    }
    Ok(())
}

fn fingerprint(value: &Value) -> bool {
    value
        .as_str()
        .is_some_and(|s| s.len() == 64 && s.bytes().all(|byte| byte.is_ascii_hexdigit()))
}

fn numerical_observation(trace: &Value, role: ServerRole) -> Result<(), String> {
    let numerical = &trace["numerical_execution"];
    // Published baselines can predate named numerical profiles. Preserve the
    // missing observation in the raw health document; never invent a profile.
    if numerical.is_null() && matches!(role, ServerRole::Baseline) {
        return Ok(());
    }
    if numerical["requested"] != "auto"
        || numerical["selected_profile"]
            .as_str()
            .is_none_or(|id| ferrum_types::NumericalProfileId::new(id).is_err())
        || ["selected_version", "qualification_version"]
            .iter()
            .any(|key| {
                ["major", "minor"].iter().any(|part| {
                    numerical[key][part]
                        .as_u64()
                        .is_none_or(|n| u32::try_from(n).is_err())
                })
            })
        || [
            "profile_fingerprint",
            "definition_fingerprint",
            "prepared_family_fingerprint",
            "capability_catalog_fingerprint",
            "runtime_policy_fingerprint",
            "execution_plan_hash",
        ]
        .iter()
        .any(|field| !fingerprint(&numerical[field]))
        || numerical["execution_plan_hash"] != trace["plan_hash"]
        || numerical["prepared_family_fingerprint"] != trace["family_fingerprint"]
    {
        return Err("missing or inconsistent vNext numerical selection/plan observation".into());
    }
    Ok(())
}

impl ServerIdentity<'_> {
    pub(super) fn verify_after(&self, before: &Value, after: &Value) -> Result<(), String> {
        self.verify(before)?;
        self.verify(after)?;
        if is_plan_runtime(self.target) {
            for field in [
                "plan_hash",
                "family_fingerprint",
                "program_fingerprint",
                "runtime_fingerprint",
                "numerical_execution",
                "runtime_memory_policy",
                "runtime_admission_policy",
            ] {
                if before["cache"]["prefix_cache"][field] != after["cache"]["prefix_cache"][field] {
                    return Err(format!("running vNext {field} changed during measurement"));
                }
            }
        }
        Ok(())
    }

    pub(super) fn verify(&self, health: &Value) -> Result<(), String> {
        let auto = &health["auto_config"];
        let backend = backend_name(self.target.backend);
        if health["status"] != "healthy"
            || health["version"].as_str() != Some(self.version)
            || auto["hardware_capabilities"]["backend"]
                .as_str()
                .is_none_or(|value| !value.eq_ignore_ascii_case(backend))
        {
            return Err(
                "performance server health omitted/mismatched actual backend or version".into(),
            );
        }
        for (field, value) in [
            ("selected_max_model_len", self.workload.max_model_len),
            ("selected_max_sequences", self.workload.concurrency),
            (
                "selected_max_batched_tokens",
                self.workload.batched_tokens(),
            ),
        ] {
            equal_count(auto, field, u64::from(value))?;
        }
        if self.target.execution_path == "legacy-model-executor" {
            if auto["execution_resource_authority"] == "plan_runtime" {
                return Err("legacy performance task observed a plan runtime executor".into());
            }
            return equal_count(
                auto,
                "selected_kv_capacity",
                u64::from(self.workload.max_model_len),
            );
        }
        if !is_plan_runtime(self.target) {
            return Err("unsupported performance executor path".into());
        }
        let trace = &health["cache"]["prefix_cache"];
        if auto["execution_resource_authority"] != "plan_runtime"
            || trace["schema"] != "ferrum.runtime-vnext.executor-trace.v1"
            || trace["device_id"].as_str().is_none_or(|id| {
                id.strip_prefix(&format!("device.{backend}."))
                    .is_none_or(|suffix| suffix.is_empty())
            })
            || !fingerprint(&trace["plan_hash"])
            || !fingerprint(&trace["family_fingerprint"])
            || !fingerprint(&trace["program_fingerprint"])
            || !fingerprint(&trace["runtime_fingerprint"])
        {
            return Err("performance task omitted/mismatched the running vNext executor".into());
        }
        equal_count(
            trace,
            "maximum_model_tokens",
            u64::from(self.workload.max_model_len),
        )?;
        equal_count(
            &trace["runtime_memory_policy"],
            "maximum_active_sequences",
            u64::from(self.workload.concurrency),
        )?;
        equal_count(
            &trace["runtime_admission_policy"],
            "maximum_scheduled_tokens",
            u64::from(self.workload.batched_tokens()),
        )?;
        let memory = &trace["runtime_memory_policy"];
        let usable = memory["capacity_bytes"]
            .as_u64()
            .and_then(|capacity| capacity.checked_sub(memory["reserve_bytes"].as_u64()?));
        if usable.is_none_or(|bytes| bytes == 0 || bytes > self.memory_budget_bytes)
            || trace["static_bytes"]
                .as_u64()
                .is_none_or(|bytes| bytes == 0 || bytes > usable.unwrap_or(0))
        {
            return Err("vNext server memory ceiling or static residency is invalid".into());
        }
        numerical_observation(trace, self.role)
    }
}

#[cfg(test)]
#[path = "runtime_tests.rs"]
mod tests;

#[cfg(test)]
pub(super) fn fixture_health() -> Value {
    tests::health()
}
