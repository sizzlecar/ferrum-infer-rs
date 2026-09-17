//! Explicit sample budgets, evaluated from every teacher-forced distribution.
use anyhow::{ensure, Context, Result};
use serde::Serialize;
use serde_json::{json, Value};

#[derive(Clone, Copy, Debug, clap::Args, Serialize)]
pub(super) struct Budgets {
    /// Upper bound on mean candidate-minus-reference NLL, in nats.
    #[arg(long)]
    pub mean_delta_nll_limit: f64,
    #[arg(long)]
    pub max_delta_nll_limit: f64,
    /// Upper bound on mean KL(reference || candidate), in nats.
    #[arg(long)]
    pub mean_kl_limit: f64,
    #[arg(long)]
    pub max_kl_limit: f64,
}

impl Budgets {
    pub fn validate(self) -> Result<()> {
        ensure!(
            [
                self.mean_delta_nll_limit,
                self.max_delta_nll_limit,
                self.mean_kl_limit,
                self.max_kl_limit
            ]
            .iter()
            .all(|n| n.is_finite() && *n >= 0.0),
            "all quality budgets must be finite and nonnegative"
        );
        Ok(())
    }
}

fn finite(value: &Value, name: &str) -> Result<f64> {
    value
        .as_f64()
        .filter(|n| n.is_finite())
        .with_context(|| format!("missing or non-finite {name}"))
}

fn mean(values: &[f64]) -> Result<f64> {
    let (mut sum, mut correction) = (0.0, 0.0);
    for value in values {
        let adjusted = value - correction;
        let next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
    }
    let mean = sum / values.len() as f64;
    ensure!(mean.is_finite(), "metric mean overflowed");
    Ok(mean)
}

pub(super) fn evaluate(report: &Value, tokens: &[u32], budgets: Budgets) -> Result<Value> {
    budgets.validate()?;
    ensure!(!tokens.is_empty(), "empty teacher history");
    ensure!(
        report["schema_version"] == 1
            && report["scope"] == "checkpoint_directory_diagnostic"
            && report.get("error").is_none(),
        "checkpoint_diff did not produce a complete directory comparison"
    );
    let teacher = &report["teacher_forcing"];
    ensure!(
        teacher["mode"] == "canonical-history"
            && teacher["encoding"] == "u32-le"
            && teacher["token_count"] == tokens.len()
            && teacher["token_ids_sha256"] == super::evidence::token_digest(tokens),
        "checkpoint history differs from the complete canonical seed"
    );
    let waves = report["waves"]
        .as_array()
        .context("missing compared waves")?;
    ensure!(
        waves.len() == tokens.len() && report["wave_count"] == tokens.len(),
        "incomplete teacher wave inventory"
    );
    let (mut kl, mut delta, mut reference, mut candidate) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for (index, (wave, token)) in waves.iter().zip(tokens).enumerate() {
        let report = &wave["report"];
        let decision = &report["teacher_forced_decision"];
        ensure!(
            decision["token_index"] == index && decision["token_id"] == *token,
            "wave {index} targets another teacher decision"
        );
        ensure!(
            report["participant_count"] == 1
                && report["reference_schema_version"] == 4
                && report["candidate_schema_version"] == 4,
            "wave {index} lacks paired teacher evidence"
        );
        let comparisons = report["comparisons"]
            .as_array()
            .context("missing comparisons")?;
        ensure!(
            comparisons.len() == 1,
            "expected exactly one full-vocabulary output per decision"
        );
        let comparison = &comparisons[0];
        for field in ["nmse", "max_abs"] {
            finite(&comparison[field], field)?;
        }
        let distribution = &comparison["distribution"];
        ensure!(
            distribution["vocabulary_size"]
                .as_u64()
                .is_some_and(|n| n > u64::from(*token)),
            "invalid full vocabulary"
        );
        let target = &distribution["teacher_forced"];
        ensure!(
            &target["decision"] == decision,
            "distribution teacher decision mismatch"
        );
        let k = finite(&distribution["kl_reference_to_candidate_nats"], "KL")?;
        let r = finite(&target["reference_nll_nats"], "reference NLL")?;
        let c = finite(&target["candidate_nll_nats"], "candidate NLL")?;
        let d = finite(&target["delta_nll_nats"], "delta NLL")?;
        ensure!(
            k >= 0.0 && r >= 0.0 && c >= 0.0 && ((c - r) - d).abs() <= 1e-10 * (1.0 + d.abs()),
            "invalid distribution metrics at wave {index}"
        );
        kl.push(k);
        delta.push(d);
        reference.push(r);
        candidate.push(c);
    }
    let mean_kl = mean(&kl)?;
    let mean_delta = mean(&delta)?;
    let aggregate = &report["aggregate"];
    for count in ["distribution_count", "teacher_forced_target_count"] {
        ensure!(
            aggregate[count] == tokens.len(),
            "aggregate {count} is incomplete"
        );
    }
    for (field, computed) in [
        ("mean_kl_reference_to_candidate_nats", mean_kl),
        ("mean_delta_nll_nats", mean_delta),
        ("mean_reference_nll_nats", mean(&reference)?),
        ("mean_candidate_nll_nats", mean(&candidate)?),
    ] {
        let recorded = finite(&aggregate[field], field)?;
        ensure!(
            (recorded - computed).abs() <= 1e-10 * (1.0 + computed.abs()),
            "aggregate {field} differs from its complete waves"
        );
    }
    let checks = [
        ("mean_delta_nll_nats", mean_delta, budgets.mean_delta_nll_limit),
        ("max_delta_nll_nats", delta.iter().copied().fold(f64::NEG_INFINITY, f64::max), budgets.max_delta_nll_limit),
        ("mean_kl_reference_to_candidate_nats", mean_kl, budgets.mean_kl_limit),
        ("max_kl_reference_to_candidate_nats", kl.iter().copied().fold(0.0, f64::max), budgets.max_kl_limit),
    ].map(|(metric, observed, limit)| json!({"metric":metric,"observed":observed,"limit":limit,"passed":observed <= limit}));
    Ok(
        json!({"passed":checks.iter().all(|c| c["passed"] == true),"checks":checks,
        "teacher_token_count":tokens.len(),"delta_nll_direction":"candidate_minus_reference_upper_bound",
        "release_approved":false,"scope":"finite_sample_only"}),
    )
}
