//! Real-model paired acceptance through the shared Rust model_regression runner.
//!
//! Set FERRUM_KV_PRODUCT_CONFIG to an explicit JSON configuration, then run:
//! cargo test -p ferrum-cli --test vnext_kv_product -- --ignored --test-threads=1
//! Both executables, models, hardware label and an external report directory are
//! selected by that configuration. No model name selects product behavior.
//! See docs/vnext-8bit-kv-validation.zh.md for calibration and evidence limits.

use anyhow::{ensure, Context, Result};
use ferrum_bench_core::release_regression::model_tasks::ModelCheck;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::fs::{self, File};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

#[derive(Debug, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
enum Backend {
    Metal,
    Cuda,
}

impl Backend {
    fn as_str(&self) -> &'static str {
        match self {
            Self::Metal => "metal",
            Self::Cuda => "cuda",
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct LongContext {
    records: u32,
    min_prompt_tokens: u32,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Model {
    id: String,
    model: PathBuf,
    source_label: String,
    precision_label: String,
    #[serde(default = "default_checks")]
    checks: Vec<ModelCheck>,
    #[serde(default)]
    expect_prefix_restore: bool,
    long_context: Option<LongContext>,
}

fn default_checks() -> Vec<ModelCheck> {
    vec![ModelCheck::Basic, ModelCheck::State]
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields, default)]
struct Timeouts {
    startup_secs: u64,
    request_secs: u64,
    run_secs: u64,
}

impl Default for Timeouts {
    fn default() -> Self {
        Self {
            startup_secs: 600,
            request_secs: 300,
            run_secs: 1200,
        }
    }
}

fn one() -> u32 {
    1
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Configuration {
    ferrum_bin: PathBuf,
    runner_bin: PathBuf,
    backend: Backend,
    hardware_label: String,
    report_dir: PathBuf,
    context_tokens: u32,
    max_num_seqs: u32,
    runtime_memory_budget_bytes: Option<u64>,
    max_tokens: u32,
    #[serde(default)]
    disable_thinking: bool,
    #[serde(default = "one")]
    repetitions: u32,
    #[serde(default)]
    timeouts: Timeouts,
    models: Vec<Model>,
}

impl Configuration {
    fn validate(&self) -> Result<()> {
        ensure!(
            !self.hardware_label.trim().is_empty(),
            "hardware_label is required for latency evidence"
        );
        ensure!(!self.models.is_empty(), "models must not be empty");
        ensure!(
            self.context_tokens > self.max_tokens && self.max_tokens > 0 && self.max_num_seqs > 0,
            "positive output budget must fit inside the selected context"
        );
        ensure!(
            self.runtime_memory_budget_bytes != Some(0),
            "memory budget must be positive"
        );
        ensure!(
            self.repetitions > 0
                && self.timeouts.startup_secs > 0
                && self.timeouts.request_secs > 0
                && self.timeouts.run_secs > 0,
            "repetitions and timeouts must be positive"
        );
        let mut ids = BTreeSet::new();
        for model in &self.models {
            ensure!(
                !model.checks.is_empty()
                    && model.checks.iter().copied().collect::<BTreeSet<_>>().len()
                        == model.checks.len(),
                "model checks must be nonempty and unique"
            );
            ensure!(
                !model.id.is_empty()
                    && model
                        .id
                        .bytes()
                        .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
                    && ids.insert(&model.id),
                "model IDs must be unique portable directory components"
            );
            ensure!(
                !model.source_label.trim().is_empty() && !model.precision_label.trim().is_empty(),
                "each model requires explicit source and weight precision labels"
            );
            if let Some(long) = &model.long_context {
                ensure!(
                    long.records > 0
                        && long.min_prompt_tokens > 0
                        && long.min_prompt_tokens < self.context_tokens - self.max_tokens,
                    "long-context minimum plus output budget must fit the selected context"
                );
            }
        }
        Ok(())
    }

    fn runner_args(&self, model: &Model, dtype: &str, report_dir: &Path) -> Vec<String> {
        let mut args = vec![
            "--ferrum-bin".into(),
            self.ferrum_bin.to_string_lossy().into_owned(),
            "--model".into(),
            model.model.to_string_lossy().into_owned(),
            "--backend".into(),
            self.backend.as_str().into(),
            "--kv-dtype".into(),
            dtype.into(),
            "--report-dir".into(),
            report_dir.to_string_lossy().into_owned(),
            "--checks".into(),
            model
                .checks
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(","),
            "--context-tokens".into(),
            self.context_tokens.to_string(),
            "--max-num-seqs".into(),
            self.max_num_seqs.to_string(),
            "--max-tokens".into(),
            self.max_tokens.to_string(),
            "--startup-timeout-secs".into(),
            self.timeouts.startup_secs.to_string(),
            "--request-timeout-secs".into(),
            self.timeouts.request_secs.to_string(),
            "--run-timeout-secs".into(),
            self.timeouts.run_secs.to_string(),
            "--source-label".into(),
            model.source_label.clone(),
            "--precision-label".into(),
            model.precision_label.clone(),
        ];
        if self.disable_thinking {
            args.push("--disable-thinking".into());
        }
        if let Some(budget) = self.runtime_memory_budget_bytes {
            args.extend(["--runtime-memory-budget-bytes".into(), budget.to_string()]);
        }
        if let Some(long) = &model.long_context {
            args.extend([
                "--long-context-records".into(),
                long.records.to_string(),
                "--long-context-min-prompt-tokens".into(),
                long.min_prompt_tokens.to_string(),
            ]);
        }
        args
    }
}

fn case<'a>(report: &'a Value, name: &str) -> Result<&'a Value> {
    let cases = report["cases"]
        .as_array()
        .context("runner report has no cases")?;
    let matches: Vec<_> = cases.iter().filter(|case| case["case"] == name).collect();
    ensure!(
        matches.len() == 1,
        "runner must record exactly one {name} case"
    );
    ensure!(
        matches[0]["status"] == "passed",
        "{name} failed: {}",
        matches[0]
    );
    Ok(&matches[0]["evidence"])
}

fn prefix_observation(report: &Value, require_restore: bool) -> Result<Value> {
    let before = &case(report, "serve-startup")?["cache"]["prefix_cache"];
    let after = &case(report, "serve-kv-final")?["cache"]["prefix_cache"];
    for snapshot in [before, after] {
        ensure!(
            snapshot["source"] == "vnext-native-sequence-checkpoint-cache"
                && snapshot["hits_scope"] == "successfully-acknowledged-native-restores",
            "text-prefix observations cannot establish native KV reuse"
        );
    }
    let delta = |field: &str| -> Result<u64> {
        let a = after[field]
            .as_u64()
            .with_context(|| format!("missing final {field}"))?;
        let b = before[field]
            .as_u64()
            .with_context(|| format!("missing initial {field}"))?;
        a.checked_sub(b)
            .with_context(|| format!("native cache counter {field} went backwards"))
    };
    let hits = delta("hits")?;
    let restored_tokens = delta("saved_prefill_tokens")?;
    if require_restore {
        ensure!(after["enabled"] == true && hits > 0 && restored_tokens > 0,
            "this configured checkpoint-capable sample did not demonstrate native prefix restore: {after}");
    }
    Ok(
        json!({"required": require_restore, "enabled": after["enabled"], "hits": hits,
        "restored_tokens": restored_tokens, "unsupported_reasons": after["unsupported_reasons"]}),
    )
}

fn validate_report(report: &Value, model: &Model, dtype: &str) -> Result<Value> {
    ensure!(
        report["status"] == "passed",
        "runner failed; inspect its report and raw logs"
    );
    ensure!(
        report["options"]["kv_dtype"] == dtype,
        "runner did not record the requested KV format"
    );
    ensure!(
        report["options"]["checks"] == json!(model.checks),
        "runner checks differ from the model's calibrated selection"
    );
    for name in [
        "binary-version",
        "serve-startup",
        "serve-kv-final",
        "binary-unchanged",
    ] {
        case(report, name)?;
    }
    for check in &model.checks {
        for name in check.cases() {
            case(report, name)?;
        }
    }
    if model.long_context.is_some() {
        case(report, "run-long-context")?;
        case(report, "serve-long-context")?;
    }
    let actual = &case(report, "serve-kv-final")?["kv_storage"];
    let expected = match dtype {
        "fp16" => ferrum_types::KvStorageFormat::F16,
        "int8" => ferrum_types::KvStorageFormat::Int8PerTokenHeadF32ScaleV1,
        _ => anyhow::bail!("unsupported paired KV format"),
    };
    ensure!(
        actual["source"] == "resolved_model_plan"
            && actual["selected"] == json!(expected)
            && actual["requested"] == json!(expected),
        "observed KV storage differs from requested format"
    );
    Ok(
        json!({"kv_storage": actual, "prefix": prefix_observation(report, model.expect_prefix_restore)?}),
    )
}

fn write_json(path: &Path, value: &impl Serialize) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)
        .with_context(|| format!("write {}", path.display()))
}

#[test]
#[ignore = "requires an explicit real-model configuration and prebuilt Ferrum/model_regression binaries"]
fn fp16_and_int8_run_and_serve_preserve_product_behavior() -> Result<()> {
    let path = std::env::var_os("FERRUM_KV_PRODUCT_CONFIG")
        .context("set FERRUM_KV_PRODUCT_CONFIG to the explicit JSON test configuration")?;
    let mut config: Configuration = serde_json::from_slice(&fs::read(path)?)?;
    config.validate()?;
    config.ferrum_bin = fs::canonicalize(&config.ferrum_bin).context("resolve Ferrum binary")?;
    config.runner_bin =
        fs::canonicalize(&config.runner_bin).context("resolve shared Rust model runner")?;
    for model in &mut config.models {
        model.model = fs::canonicalize(&model.model).context("resolve local model")?;
    }
    fs::create_dir_all(&config.report_dir)?;
    ensure!(
        fs::read_dir(&config.report_dir)?.next().is_none(),
        "report directory must be empty to preserve existing evidence"
    );
    config.report_dir = fs::canonicalize(&config.report_dir)?;
    write_json(&config.report_dir.join("configuration.json"), &config)?;
    let mut aggregate = json!({"schema_version": 1, "status": "running", "configuration": config,
        "started_at": chrono::Utc::now().to_rfc3339(), "host_os": std::env::consts::OS,
        "host_arch": std::env::consts::ARCH, "observations": [],
        "timing_scope": "observed subprocess/case/HTTP wall time; first HTTP body chunk is not a generated-token TTFT",
        "quality_scope": "shared basic/state semantic oracles and optional exact long-context fact retrieval; no numerical-quality or performance generalization"});
    let aggregate_path = config.report_dir.join("paired.json");
    write_json(&aggregate_path, &aggregate)?;
    let mut failed = false;
    for model in &config.models {
        for repetition in 0..config.repetitions {
            for dtype in ["fp16", "int8"] {
                let stem = format!("{}-{repetition}-{dtype}", model.id);
                let directory = config.report_dir.join(&stem);
                let argv = config.runner_args(model, dtype, &directory);
                write_json(
                    &config.report_dir.join(format!("{stem}.command.json")),
                    &json!({"program": config.runner_bin, "args": argv}),
                )?;
                let started = Instant::now();
                let result = (|| -> Result<Value> {
                    let status = Command::new(&config.runner_bin)
                        .args(&argv)
                        .stdin(Stdio::null())
                        .stdout(Stdio::from(File::create(
                            config.report_dir.join(format!("{stem}.stdout.txt")),
                        )?))
                        .stderr(Stdio::from(File::create(
                            config.report_dir.join(format!("{stem}.stderr.txt")),
                        )?))
                        .status()
                        .context("run shared model regression")?;
                    let report: Value = serde_json::from_slice(
                        &fs::read(directory.join("report.json"))
                            .context("runner did not produce its report")?,
                    )?;
                    ensure!(
                        status.success(),
                        "runner exited {status}; inspect {}/report.json",
                        directory.display()
                    );
                    validate_report(&report, model, dtype)
                })();
                let mut observation = json!({"model_id": model.id, "repetition": repetition,
                    "kv_dtype": dtype, "report": directory.join("report.json"),
                    "elapsed_ms": started.elapsed().as_secs_f64() * 1000.0});
                match result {
                    Ok(evidence) => {
                        observation["status"] = json!("passed");
                        observation["evidence"] = evidence;
                    }
                    Err(error) => {
                        failed = true;
                        observation["status"] = json!("failed");
                        observation["error"] = json!(format!("{error:#}"));
                    }
                }
                eprintln!("KV_PRODUCT_PROGRESS {observation}");
                aggregate["observations"]
                    .as_array_mut()
                    .unwrap()
                    .push(observation);
                write_json(&aggregate_path, &aggregate)?;
            }
        }
    }
    aggregate["status"] = json!(if failed { "failed" } else { "passed" });
    aggregate["finished_at"] = json!(chrono::Utc::now().to_rfc3339());
    write_json(&aggregate_path, &aggregate)?;
    ensure!(
        !failed,
        "paired acceptance failed; see {}",
        aggregate_path.display()
    );
    Ok(())
}

#[test]
fn prefix_evidence_distinguishes_native_restore_from_text_overlap_and_disabled_support() {
    let native = json!({"source": "vnext-native-sequence-checkpoint-cache", "hits_scope": "successfully-acknowledged-native-restores",
        "enabled": true, "hits": 0, "saved_prefill_tokens": 0});
    let mut final_native = native.clone();
    final_native["hits"] = json!(1);
    final_native["saved_prefill_tokens"] = json!(128);
    let mut report = json!({"cases": [
        {"case": "serve-startup", "status": "passed", "evidence": {"cache": {"prefix_cache": native}}},
        {"case": "serve-kv-final", "status": "passed", "evidence": {"cache": {"prefix_cache": final_native}}}
    ]});
    assert_eq!(
        prefix_observation(&report, true).unwrap()["restored_tokens"],
        128
    );
    report["cases"][1]["evidence"]["cache"]["prefix_cache"]["source"] =
        json!("server-prompt-lcp-observability");
    assert!(prefix_observation(&report, true).is_err());
    report["cases"][1]["evidence"]["cache"]["prefix_cache"] = native.clone();
    for case in report["cases"].as_array_mut().unwrap() {
        case["evidence"]["cache"]["prefix_cache"]["enabled"] = json!(false);
    }
    assert_eq!(
        prefix_observation(&report, false).unwrap()["restored_tokens"],
        0
    );
    assert!(prefix_observation(&report, true).is_err());
}
