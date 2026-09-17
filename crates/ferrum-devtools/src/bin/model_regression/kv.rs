//! Explicit KV selection and evidence from the executing model plan.
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

pub(super) fn validate_storage(dtype: Option<KvDtype>, observed: &Value) -> Result<()> {
    let Some(dtype) = dtype else { return Ok(()) };
    let storage = &observed["kv_storage"];
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
}
