use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};

pub(super) fn reference_log_probabilities(
    body: &Value,
    vocabulary: usize,
    prompt_len: usize,
) -> Result<Vec<f64>> {
    ensure!(
        body.get("error").is_none() && body["truncated"] == false,
        "reference failed or truncated the history"
    );
    ensure!(
        body["tokens_evaluated"] == prompt_len && body["tokens_predicted"] == 1,
        "reference token usage differs from the complete requested history/one decision"
    );
    let records = body["completion_probabilities"]
        .as_array()
        .context("missing pre-sampling probabilities")?;
    ensure!(
        records.len() == 1,
        "reference must return exactly one distribution"
    );
    let probabilities = records[0]["top_logprobs"]
        .as_array()
        .context("missing reference top_logprobs")?;
    ensure!(
        vocabulary > 0 && probabilities.len() == vocabulary,
        "reference did not return the complete vocabulary"
    );
    let mut output = vec![None; vocabulary];
    for probability in probabilities {
        let id = probability["id"]
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .context("invalid reference token ID")?;
        ensure!(
            id < vocabulary,
            "reference token ID is outside the vocabulary"
        );
        let log_probability = probability["logprob"]
            .as_f64()
            .filter(|value| value.is_finite() && *value <= 0.0)
            .context("invalid reference log probability")?;
        ensure!(
            output[id].replace(log_probability).is_none(),
            "duplicate reference token ID"
        );
    }
    let output: Vec<_> = output
        .into_iter()
        .map(|value| value.expect("complete unique vocabulary"))
        .collect();
    let mass: f64 = output.iter().map(|p| p.exp()).sum();
    // The native endpoint accumulates softmax in F32. Reject missing mass and
    // post-sampling renormalization instead of comparing a top-N subset.
    ensure!(
        (mass - 1.0).abs() < 1e-3,
        "reference probability mass is {mass}, expected one"
    );
    Ok(output)
}

pub(super) fn compare(reference: &[f64], logits: &[f32], teacher: u32) -> Result<Value> {
    ensure!(
        !logits.is_empty() && reference.len() == logits.len() && (teacher as usize) < logits.len(),
        "distribution geometry differs"
    );
    ensure!(
        logits.iter().all(|x| x.is_finite()) && reference.iter().all(|x| x.is_finite()),
        "non-finite distribution value"
    );
    let maximum = f64::from(logits.iter().copied().fold(f32::NEG_INFINITY, f32::max));
    let partition = logits
        .iter()
        .map(|x| (f64::from(*x) - maximum).exp())
        .sum::<f64>()
        .ln();
    let candidate: Vec<_> = logits
        .iter()
        .map(|x| f64::from(*x) - maximum - partition)
        .collect();
    let reference_mass = reference.iter().map(|x| x.exp()).sum::<f64>();
    ensure!(
        reference_mass.is_finite() && (reference_mass - 1.0).abs() < 1e-3,
        "invalid reference normalization"
    );
    let reference_offset = reference_mass.ln();
    let normalized: Vec<_> = reference.iter().map(|x| x - reference_offset).collect();
    // Native llama.cpp serializes a zero F32 softmax probability as f32::MIN
    // to avoid JSON null for -inf. This remains a zero-mass vocabulary entry,
    // not a measurable finite NLL of approximately 3.4e38.
    let zero_probability_token_count = normalized.iter().filter(|p| p.exp() == 0.0).count();
    ensure!(
        normalized[teacher as usize].exp() > 0.0,
        "reference teacher token {teacher} has zero probability after F32 softmax (including its serialized floor); NLL and delta NLL are not measurable ({zero_probability_token_count} zero-probability vocabulary entries)"
    );
    let kl = normalized
        .iter()
        .zip(&candidate)
        .map(|(p, q)| {
            let probability = p.exp();
            if probability == 0.0 {
                0.0
            } else {
                probability * (p - q)
            }
        })
        .sum::<f64>()
        .max(0.0);
    let reference_nll = -normalized[teacher as usize];
    let candidate_nll = -candidate[teacher as usize];
    let argmax = |values: &[f64]| {
        values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
            .unwrap()
            .0
    };
    ensure!(
        [kl, reference_nll, candidate_nll]
            .iter()
            .all(|x| x.is_finite()),
        "distribution metric overflow"
    );
    Ok(
        json!({"vocabulary_size":logits.len(),"reference_probability_mass":reference_mass,
        "reference_zero_probability_token_count":zero_probability_token_count,
        "reference_log_normalization_correction":reference_offset,
        "kl_reference_to_candidate_nats":kl,"reference_nll_nats":reference_nll,
        "candidate_nll_nats":candidate_nll,"delta_nll_nats":candidate_nll-reference_nll,
        "reference_argmax":argmax(&normalized),"candidate_argmax":argmax(&candidate)}),
    )
}

pub(super) fn aggregate(decisions: &[Value]) -> Result<Value> {
    ensure!(!decisions.is_empty(), "empty teacher comparison");
    let mut result = serde_json::Map::new();
    for key in [
        "kl_reference_to_candidate_nats",
        "reference_nll_nats",
        "candidate_nll_nats",
        "delta_nll_nats",
    ] {
        let values = decisions
            .iter()
            .map(|decision| {
                decision["comparison"][key]
                    .as_f64()
                    .filter(|value| value.is_finite())
                    .context("missing complete per-decision metric")
            })
            .collect::<Result<Vec<_>>>()?;
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        ensure!(mean.is_finite(), "aggregate metric overflow");
        result.insert(format!("mean_{key}"), json!(mean));
        let (position, maximum) = values
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap();
        result.insert(
            format!("max_{key}"),
            json!({"value":maximum,"teacher_token_index":position}),
        );
    }
    result.insert("distribution_count".into(), json!(decisions.len()));
    let zero_counts = decisions
        .iter()
        .map(|decision| {
            decision["comparison"]["reference_zero_probability_token_count"]
                .as_u64()
                .context("missing reference zero-probability count")
        })
        .collect::<Result<Vec<_>>>()?;
    let total = zero_counts
        .iter()
        .try_fold(0_u64, |total, count| total.checked_add(*count))
        .context("reference zero-probability count overflow")?;
    result.insert(
        "reference_zero_probability_token_count_total".into(),
        json!(total),
    );
    result.insert(
        "distributions_with_reference_zero_probability".into(),
        json!(zero_counts.iter().filter(|&&count| count != 0).count()),
    );
    Ok(Value::Object(result))
}
