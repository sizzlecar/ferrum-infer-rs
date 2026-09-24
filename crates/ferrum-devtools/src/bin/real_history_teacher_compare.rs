//! Strict full-vocabulary comparison of production real-history captures.
use anyhow::{ensure, Context, Result};
use clap::Parser;
use ferrum_bench_core::teacher_metrics::{
    compare_full_vocabulary, compensated_sum, FullVocabularyMetrics,
};
use ferrum_types::teacher_capture::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
};

#[path = "real_history_teacher_compare/capture.rs"]
mod capture;
#[path = "real_history_teacher_compare/files.rs"]
mod files;
#[path = "real_history_teacher_compare/native_work.rs"]
mod native_work;
#[path = "real_history_teacher_compare/numerical_transition.rs"]
mod numerical_transition;
#[path = "real_history_teacher_compare/receipt.rs"]
mod receipt;
#[cfg(test)]
#[path = "real_history_teacher_compare/tests.rs"]
mod tests;
#[path = "real_history_teacher_compare/transition.rs"]
mod transition;

#[derive(Clone, Copy, Debug, clap::Args, Serialize)]
struct Budgets {
    /// Upper bound on mean candidate-minus-reference target NLL (nats).
    #[arg(long)]
    mean_delta_nll_limit: f64,
    #[arg(long)]
    max_delta_nll_limit: f64,
    /// Upper bound on mean full-vocabulary KL(reference || candidate), nats.
    #[arg(long)]
    mean_kl_limit: f64,
    #[arg(long)]
    max_kl_limit: f64,
}

impl Budgets {
    fn validate(self) -> Result<()> {
        ensure!(
            [
                self.mean_delta_nll_limit,
                self.max_delta_nll_limit,
                self.mean_kl_limit,
                self.max_kl_limit
            ]
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0),
            "all four quality budgets must be finite and nonnegative"
        );
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, clap::ValueEnum, Serialize)]
#[serde(rename_all = "snake_case")]
enum ComparisonShape {
    #[default]
    SerialToBatched,
    BatchedToBatched,
}

#[derive(Parser)]
#[command(about = "Validate and compare real fixed-history full-vocabulary captures")]
struct Args {
    #[arg(long)]
    reference_dir: PathBuf,
    #[arg(long)]
    candidate_dir: PathBuf,
    /// Local immutable binary copy when the capture's recorded absolute path is unavailable.
    #[arg(long)]
    reference_binary: Option<PathBuf>,
    #[arg(long)]
    candidate_binary: Option<PathBuf>,
    /// Exact history JSON shared by both captures, overriding their recorded absolute paths.
    #[arg(long)]
    history_file: Option<PathBuf>,
    /// Exact, declared binary/plan/provider migration; omitted means strict fixed plan.
    #[arg(long, conflicts_with = "numerical_profile_transition")]
    implementation_transition: Option<PathBuf>,
    /// Exact numerical profile/node migration; does not relax any quality budget.
    #[arg(long, conflicts_with = "implementation_transition")]
    numerical_profile_transition: Option<PathBuf>,
    /// Actual capture modes to compare; all physical wave/owner checks remain required.
    #[arg(long, value_enum, default_value = "serial-to-batched")]
    comparison_shape: ComparisonShape,
    /// Additionally require every full-vocabulary F32 logit bit to match, including signed zero.
    /// This never replaces the declared KL/NLL budgets or raw readback validation.
    #[arg(long)]
    require_bitwise_logits: bool,
    /// Optional explicit, pinned actual-native inventory audit; omitted preserves the original comparison.
    #[arg(long)]
    native_work_audit: Option<PathBuf>,
    /// New JSON file, including every error and every available per-decision metric.
    #[arg(long)]
    output_file: PathBuf,
    #[command(flatten)]
    budgets: Budgets,
}

#[derive(Debug, Clone, Serialize)]
struct Issue {
    scope: String,
    message: String,
}

fn issue(issues: &mut Vec<Issue>, scope: impl Into<String>, error: impl std::fmt::Display) {
    issues.push(Issue {
        scope: scope.into(),
        message: error.to_string(),
    });
}

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn token_digest(tokens: &[u32]) -> String {
    let mut hash = Sha256::new();
    for token in tokens {
        hash.update(token.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

#[derive(Serialize)]
struct DecisionComparison {
    owner_id: String,
    decision_index: usize,
    phase: &'static str,
    teacher_token_id: u32,
    history_tokens: usize,
    history_sha256: String,
    metrics: FullVocabularyMetrics,
    logit_count: usize,
    bitwise_mismatch_count: usize,
}

fn aggregate<'a>(rows: impl Iterator<Item = &'a DecisionComparison>, budgets: Budgets) -> Value {
    let rows: Vec<_> = rows.collect();
    if rows.is_empty() {
        return json!({"decision_count":0,"quality_passed":false,"checks":[]});
    }
    let mean = |values: Vec<f64>| compensated_sum(values.into_iter()) / rows.len() as f64;
    fn nll(row: &DecisionComparison) -> &ferrum_bench_core::teacher_metrics::TeacherNll {
        row.metrics
            .teacher_forced
            .as_ref()
            .expect("target was provided")
    }
    let mean_delta = mean(rows.iter().map(|row| nll(row).delta_nll_nats).collect());
    let max_delta = rows
        .iter()
        .map(|row| nll(row).delta_nll_nats)
        .fold(f64::NEG_INFINITY, f64::max);
    let mean_kl = mean(
        rows.iter()
            .map(|row| row.metrics.kl_reference_to_candidate_nats)
            .collect(),
    );
    let max_kl = rows
        .iter()
        .map(|row| row.metrics.kl_reference_to_candidate_nats)
        .fold(0.0, f64::max);
    let checks = [
        ("mean_delta_nll_nats", mean_delta, budgets.mean_delta_nll_limit),
        ("max_delta_nll_nats", max_delta, budgets.max_delta_nll_limit),
        ("mean_kl_reference_to_candidate_nats", mean_kl, budgets.mean_kl_limit),
        ("max_kl_reference_to_candidate_nats", max_kl, budgets.max_kl_limit),
    ].map(|(metric, observed, limit)| json!({"metric":metric,"observed":observed,"limit":limit,"passed":observed.is_finite() && observed <= limit}));
    json!({"decision_count":rows.len(),"quality_passed":checks.iter().all(|check| check["passed"]==true),"checks":checks,
        "mean_reference_nll_nats":mean(rows.iter().map(|row| nll(row).reference_nll_nats).collect()),
        "mean_candidate_nll_nats":mean(rows.iter().map(|row| nll(row).candidate_nll_nats).collect()),
        "argmax_agreement_count":rows.iter().filter(|row| row.metrics.argmax_agrees).count(),
        "max_abs_logit_error":rows.iter().map(|row| row.metrics.max_abs).fold(0.0,f64::max),
        "delta_nll_direction":"candidate_minus_reference_upper_bound"})
}

fn compare(args: &Args) -> (Value, i32) {
    let mut errors = Vec::new();
    if args.implementation_transition.is_some() && args.numerical_profile_transition.is_some() {
        issue(
            &mut errors,
            "transition_mode",
            "implementation and numerical profile transitions are mutually exclusive",
        );
    }
    let transition = args.implementation_transition.as_deref().and_then(|path| {
        match transition::DeclaredTransition::read(path) {
            Ok(declaration) => Some(declaration),
            Err(error) => {
                issue(&mut errors, "implementation_transition", error);
                None
            }
        }
    });
    let numerical_transition = args
        .numerical_profile_transition
        .as_deref()
        .and_then(
            |path| match numerical_transition::DeclaredNumericalTransition::read(path) {
                Ok(declaration) => Some(declaration),
                Err(error) => {
                    issue(&mut errors, "numerical_profile_transition", error);
                    None
                }
            },
        );
    if let Err(error) = args.budgets.validate() {
        issue(&mut errors, "budgets", error);
    }
    let reference = capture::validate(
        &args.reference_dir,
        "reference",
        args.reference_binary.as_deref(),
        args.history_file.as_deref(),
        &mut errors,
    );
    let candidate = capture::validate(
        &args.candidate_dir,
        "candidate",
        args.candidate_binary.as_deref(),
        args.history_file.as_deref(),
        &mut errors,
    );
    let mut comparisons = Vec::new();
    let mut provenance = Value::Null;
    if let (Some(reference), Some(candidate)) = (&reference, &candidate) {
        match capture::compare_identity(
            reference,
            candidate,
            transition.as_ref(),
            numerical_transition.as_ref(),
            args.comparison_shape,
        ) {
            Ok(identity) => provenance = identity,
            Err(error) => issue(&mut errors, "paired_identity", error),
        }
        let keys: BTreeSet<_> = reference
            .decisions
            .keys()
            .chain(candidate.decisions.keys())
            .cloned()
            .collect();
        for (owner_id, index) in keys {
            let key = (owner_id.clone(), index);
            let scope = format!("decision/{owner_id}/{index}");
            let result = (|| -> Result<DecisionComparison> {
                let a = reference
                    .decisions
                    .get(&key)
                    .context("reference decision missing or invalid")?;
                let b = candidate
                    .decisions
                    .get(&key)
                    .context("candidate decision missing or invalid")?;
                ensure!(
                    a.evidence.teacher_token_id == b.evidence.teacher_token_id
                        && a.evidence.history_tokens == b.evidence.history_tokens
                        && a.evidence.history_sha256 == b.evidence.history_sha256,
                    "paired decisions have different target/history"
                );
                let x = files::logits(&reference.directory, &a.logits)?;
                let y = files::logits(&candidate.directory, &b.logits)?;
                let metrics = compare_full_vocabulary(&x, &y, Some(a.evidence.teacher_token_id))
                    .map_err(anyhow::Error::msg)?;
                let bitwise_mismatch_count = x
                    .iter()
                    .zip(&y)
                    .filter(|(a, b)| a.to_bits() != b.to_bits())
                    .count();
                ensure!(
                    metrics.nmse.is_finite()
                        && metrics.max_abs.is_finite()
                        && metrics.kl_reference_to_candidate_nats.is_finite(),
                    "non-finite distribution metrics"
                );
                Ok(DecisionComparison {
                    owner_id,
                    decision_index: index,
                    phase: if index == 0 { "prefill" } else { "decode" },
                    teacher_token_id: a.evidence.teacher_token_id,
                    history_tokens: a.evidence.history_tokens,
                    history_sha256: a.evidence.history_sha256.clone(),
                    metrics,
                    logit_count: x.len(),
                    bitwise_mismatch_count,
                })
            })();
            match result {
                Ok(row) => comparisons.push(row),
                Err(error) => issue(&mut errors, scope, error),
            }
        }
    }
    let mut native_audit_report = None;
    if let Some(path) = &args.native_work_audit {
        if errors.is_empty() {
            let result = reference
                .as_ref()
                .zip(candidate.as_ref())
                .context("native audit requires two fully validated captures")
                .and_then(|(a, b)| native_work::audit(args, path, a, b));
            match result {
                Ok(report) => native_audit_report = Some(report),
                Err(error) => {
                    native_audit_report = Some(json!({"passed":false,"reason":error.to_string()}));
                    issue(&mut errors, "native_work_audit", error);
                }
            }
        } else {
            native_audit_report = Some(
                json!({"passed":false,"reason":"original capture or paired identity validation failed"}),
            );
        }
    }
    let all = aggregate(comparisons.iter(), args.budgets);
    let prefill = aggregate(
        comparisons.iter().filter(|row| row.decision_index == 0),
        args.budgets,
    );
    let decode = aggregate(
        comparisons.iter().filter(|row| row.decision_index > 0),
        args.budgets,
    );
    // A pooled mean must not hide a failing decode-only mean.
    let quality = all["quality_passed"] == true
        && prefill["quality_passed"] == true
        && decode["quality_passed"] == true;
    let bitwise_equal = !comparisons.is_empty()
        && comparisons
            .iter()
            .all(|row| row.bitwise_mismatch_count == 0);
    let bitwise_passed = errors.is_empty() && bitwise_equal;
    let comparison_passed =
        errors.is_empty() && quality && (!args.require_bitwise_logits || bitwise_passed);
    let (status, exit) = if !errors.is_empty() {
        ("evidence_insufficient", 2)
    } else if !quality {
        ("quality_failed", 1)
    } else if args.require_bitwise_logits && !bitwise_equal {
        ("bitwise_failed", 1)
    } else {
        ("passed", 0)
    };
    let mut report = json!({"schema_version":1,"artifact_type":"ferrum.real_history_teacher_comparison","status":status,
        "evidence_complete":errors.is_empty(),"quality_passed":errors.is_empty()&&quality,"release_approved":false,
        "comparison_shape":args.comparison_shape,"comparison_passed":comparison_passed,
        "bitwise_logits":{"required":args.require_bitwise_logits,"passed":bitwise_passed,
            "scope":"complete_f32_logits_independently_verified_against_each_raw_readback_including_signed_zero",
            "checked_decision_count":comparisons.len(),
            "checked_logit_count":comparisons.iter().map(|row|row.logit_count as u64).sum::<u64>(),
            "mismatch_count":comparisons.iter().map(|row|row.bitwise_mismatch_count as u64).sum::<u64>()},
        "scope":"fixed_history_full_vocabulary_finite_sample_only","performance_evidence":false,
        "reference_manifest":reference.as_ref().map(|capture|json!({"path":capture.directory.join("manifest.json"),"sha256":capture.manifest_sha256})),
        "candidate_manifest":candidate.as_ref().map(|capture|json!({"path":capture.directory.join("manifest.json"),"sha256":capture.manifest_sha256})),
        "provenance":provenance,"budgets":args.budgets,"errors":errors,
        "summary":{"all":all,"prefill":prefill,"decode":decode},"decisions":comparisons});
    if let Some(audit) = native_audit_report {
        report["native_work_audit"] = audit;
    }
    (report, exit)
}

fn main() -> Result<()> {
    let args = Args::parse();
    let (report, exit) = compare(&args);
    let mut output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args.output_file)
        .with_context(|| format!("create {}", args.output_file.display()))?;
    output.write_all(&serde_json::to_vec_pretty(&report)?)?;
    output.flush()?;
    println!(
        "{}: {}",
        report["status"].as_str().unwrap_or("evidence_insufficient"),
        args.output_file.display()
    );
    if exit != 0 {
        std::process::exit(exit);
    }
    Ok(())
}
