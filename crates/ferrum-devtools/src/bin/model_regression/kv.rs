//! Actual KV selection with separate compiled-plan and model-owned evidence.
use anyhow::{ensure, Context, Result};
use ferrum_types::KvStorageFormat;
use serde::Serialize;
use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum, Serialize)]
#[serde(rename_all = "lowercase")]
pub(super) enum KvDtype {
    Fp16,
    Int8,
}

impl KvDtype {
    pub(super) fn cli_name(self) -> &'static str {
        match self {
            Self::Fp16 => "fp16",
            Self::Int8 => "int8",
        }
    }

    fn storage(self) -> KvStorageFormat {
        match self {
            Self::Fp16 => KvStorageFormat::F16,
            Self::Int8 => KvStorageFormat::Int8PerTokenHeadF32ScaleV1,
        }
    }
}

const MODEL_KV_SOURCE: &str = "llama-family-paged-block-prefix-cache";

pub(super) fn validate_storage(dtype: Option<KvDtype>, observed: &Value) -> Result<()> {
    let Some(dtype) = dtype else { return Ok(()) };
    if let Some(storage) = observed.get("kv_storage").filter(|value| !value.is_null()) {
        // A present plan must satisfy its entire original contract. A malformed
        // plan cannot pass by falling back to unrelated model-cache evidence.
        ensure!(
            observed["model_kv_storage"].is_null()
                && observed["cache"]["prefix_cache"]["source"] != MODEL_KV_SOURCE,
            "KV evidence mixes compiled-plan and model-owned source domains"
        );
        return validate_plan_storage(dtype, storage);
    }
    let model = observed
        .get("model_kv_storage")
        .unwrap_or(&observed["cache"]["prefix_cache"]);
    ensure!(
        model["source"] == MODEL_KV_SOURCE,
        "KV evidence is neither a resolved plan nor an actual model cache: {model}"
    );
    ensure!(
        model["kv_dtype"].as_str() == Some(dtype.cli_name()),
        "actual model KV dtype was {}, expected {}",
        model["kv_dtype"],
        dtype.cli_name()
    );
    Ok(())
}

fn validate_plan_storage(dtype: KvDtype, storage: &Value) -> Result<()> {
    ensure!(
        storage["source"] == "resolved_model_plan",
        "KV evidence must come from the resolved model plan: {storage}"
    );
    for field in ["requested", "selected"] {
        let actual: KvStorageFormat = serde_json::from_value(storage[field].clone())
            .with_context(|| format!("KV evidence is missing a typed {field} format"))?;
        ensure!(
            actual == dtype.storage(),
            "KV {field} was {actual}, expected {}",
            dtype.storage()
        );
    }
    ensure!(
        storage["numerical_profile"]
            .as_str()
            .is_some_and(|profile| !profile.is_empty()),
        "KV evidence has no selected numerical profile"
    );
    Ok(())
}

/// Public product evidence cannot inspect every internal floating-point value.
/// It must nevertheless reject any failed request or reported executor failure,
/// including the INT8 encoder's nonfinite-input error propagated by its fence.
pub(super) fn validate_completed_health(health: &Value) -> Result<()> {
    ensure!(
        health["scheduler"]["failed_requests"].as_u64() == Some(0),
        "server reported failed requests: {}",
        health["scheduler"]
    );
    let prefix = &health["cache"]["prefix_cache"];
    match prefix["source"].as_str() {
        Some("vnext-native-sequence-checkpoint-cache") => {
            ensure!(
                prefix.get("last_failure").is_some_and(Value::is_null),
                "executor reported a failure or omitted its failure status: {}",
                prefix["last_failure"]
            );
        }
        Some(MODEL_KV_SOURCE) => {
            ensure!(
                health["kv_storage"].is_null(),
                "model-owned KV health cannot masquerade as a compiled plan"
            );
            // The retained Llama model reports its compile-time KV type, not a
            // vNext profile or last_failure field. Its real errors propagate to
            // the scheduler and the checked product request/shutdown outcomes.
            ensure!(
                matches!(prefix["kv_dtype"].as_str(), Some("fp16" | "int8")),
                "model-owned KV evidence has no supported actual dtype"
            );
            ensure!(health["status"] == "healthy", "model server is not healthy");
            ensure!(
                health["scheduler"]["successful_requests"]
                    .as_u64()
                    .is_some_and(|count| count > 0),
                "model-owned KV acceptance requires actual completed requests"
            );
        }
        _ => anyhow::bail!("KV acceptance requires a recognized actual executor source"),
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    use serde_json::json;

    #[test]
    fn kv_selection_reaches_both_entrypoints_and_cannot_override_a_prepared_task() {
        let base = [
            "model-regression",
            "--ferrum-bin",
            "fixture",
            "--model",
            "fixture-model",
            "--backend",
            "metal",
            "--report-dir",
            "fixture-report",
        ];
        for dtype in ["fp16", "int8"] {
            let args =
                super::super::Args::try_parse_from(base.into_iter().chain(["--kv-dtype", dtype]))
                    .unwrap();
            for entrypoint in ["run", "serve"] {
                assert!(args
                    .common_args(entrypoint)
                    .windows(2)
                    .any(|words| words == ["--kv-dtype", dtype]));
            }
        }
        assert!(super::super::Args::try_parse_from(base.into_iter().chain([
            "--kv-dtype",
            "int8",
            "--expected-task",
            "task.json"
        ]))
        .is_err());
    }

    #[test]
    fn kv_evidence_rejects_echoed_configuration_and_silent_storage_fallback() {
        for dtype in [KvDtype::Fp16, KvDtype::Int8] {
            let evidence = json!({"kv_storage": {"source": "resolved_model_plan",
                "requested": dtype.storage(), "selected": dtype.storage(),
                "numerical_profile": "test.profile"}});
            validate_storage(Some(dtype), &evidence).unwrap();
            let mut mismatch = evidence.clone();
            mismatch["kv_storage"]["selected"] = json!(match dtype {
                KvDtype::Fp16 => KvStorageFormat::Int8PerTokenHeadF32ScaleV1,
                KvDtype::Int8 => KvStorageFormat::F16,
            });
            assert!(validate_storage(Some(dtype), &mismatch).is_err());
            for field in ["source", "requested", "selected", "numerical_profile"] {
                let mut missing = evidence.clone();
                missing["kv_storage"].as_object_mut().unwrap().remove(field);
                assert!(validate_storage(Some(dtype), &missing).is_err());
            }
        }
        validate_storage(None, &Value::Null).unwrap();
    }

    #[test]
    fn kv_model_owned_evidence_uses_actual_source_and_dtype_for_both_entrypoints() {
        for dtype in [KvDtype::Fp16, KvDtype::Int8] {
            let snapshot = json!({"source": MODEL_KV_SOURCE, "kv_dtype": dtype.cli_name()});
            let run = json!({"model_kv_storage": snapshot});
            let serve = json!({"kv_storage": null, "cache": {"prefix_cache": snapshot}});
            validate_storage(Some(dtype), &run).unwrap();
            validate_storage(Some(dtype), &serve).unwrap();
            let other = match dtype {
                KvDtype::Fp16 => KvDtype::Int8,
                KvDtype::Int8 => KvDtype::Fp16,
            };
            assert!(validate_storage(Some(other), &run).is_err());
            assert!(validate_storage(Some(other), &serve).is_err());
            for field in ["source", "kv_dtype"] {
                let mut missing = run.clone();
                missing["model_kv_storage"]
                    .as_object_mut()
                    .unwrap()
                    .remove(field);
                assert!(validate_storage(Some(dtype), &missing).is_err());
            }
            let echo = json!({"runtime": {"kv_dtype": dtype.cli_name()},
                "numerical_execution": {"requested": "auto"}});
            assert!(validate_storage(Some(dtype), &echo).is_err());
            let echo_source = json!({"model_kv_storage": {
                "source": "requested_configuration", "kv_dtype": dtype.cli_name()}});
            assert!(validate_storage(Some(dtype), &echo_source).is_err());
        }
    }

    #[test]
    fn kv_model_evidence_cannot_bypass_or_relabel_a_present_compiled_plan() {
        let model = json!({"source": MODEL_KV_SOURCE, "kv_dtype": "int8"});
        for plan in [
            json!({}),
            json!({"source": "requested_configuration", "selected": "int8"}),
            json!({"source": "resolved_model_plan", "requested": KvStorageFormat::Int8PerTokenHeadF32ScaleV1,
                "selected": KvStorageFormat::Int8PerTokenHeadF32ScaleV1}),
            json!({"source": "resolved_model_plan", "requested": KvStorageFormat::Int8PerTokenHeadF32ScaleV1,
                "selected": KvStorageFormat::Int8PerTokenHeadF32ScaleV1, "numerical_profile": "test.profile"}),
        ] {
            for evidence in [
                json!({"kv_storage": plan, "model_kv_storage": model}),
                json!({"kv_storage": plan, "cache": {"prefix_cache": model}}),
            ] {
                assert!(validate_storage(Some(KvDtype::Int8), &evidence).is_err());
            }
        }
    }

    #[test]
    fn kv_final_health_preserves_each_executors_actual_failure_contract() {
        let model = json!({"status": "healthy",
            "scheduler": {"failed_requests": 0, "successful_requests": 2},
            "kv_storage": null,
            "cache": {"prefix_cache": {"source": MODEL_KV_SOURCE, "kv_dtype": "int8"}}});
        let plan = json!({"scheduler": {"failed_requests": 0},
            "cache": {"prefix_cache": {"source": "vnext-native-sequence-checkpoint-cache",
                "last_failure": null}}});
        for good in [&model, &plan] {
            validate_completed_health(good).unwrap();
            let mut failed = good.clone();
            failed["scheduler"]["failed_requests"] = json!(1);
            assert!(validate_completed_health(&failed).is_err());
            failed["scheduler"]
                .as_object_mut()
                .unwrap()
                .remove("failed_requests");
            assert!(validate_completed_health(&failed).is_err());
        }
        let mut failed = plan.clone();
        failed["cache"]["prefix_cache"]["last_failure"] = json!("encoded failure");
        assert!(validate_completed_health(&failed).is_err());
        failed["cache"]["prefix_cache"]
            .as_object_mut()
            .unwrap()
            .remove("last_failure");
        assert!(validate_completed_health(&failed).is_err());
        let mut empty = model.clone();
        empty["scheduler"]["successful_requests"] = json!(0);
        assert!(validate_completed_health(&empty).is_err());
        empty["scheduler"]
            .as_object_mut()
            .unwrap()
            .remove("successful_requests");
        assert!(validate_completed_health(&empty).is_err());
        let mut unhealthy = model.clone();
        unhealthy["status"] = json!("unhealthy");
        assert!(validate_completed_health(&unhealthy).is_err());
        let mut mixed = model.clone();
        mixed["kv_storage"] = json!({"source": "resolved_model_plan"});
        assert!(validate_completed_health(&mixed).is_err());
        let mut unknown = model;
        unknown["cache"]["prefix_cache"]["source"] = json!("requested_configuration");
        assert!(validate_completed_health(&unknown).is_err());
    }
}
