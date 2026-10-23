use super::{
    artifacts,
    dyadic::{self, Dot},
    source::Artifact,
};
use anyhow::{ensure, Context, Result};
use ferrum_models::qwen_image21::{
    QwenImage21Block0AttentionFamily as Attention, QwenImage21Block0Family as Block,
    QwenImage21Block0MlpFamily as Mlp,
};
use ferrum_quantization::SafetensorsArchive;
use serde_json::{json, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    Query,
    Value,
    MlpOutput,
    MlpUp,
}
impl Stage {
    pub fn name(self) -> &'static str {
        match self {
            Self::Query => "query",
            Self::Value => "value",
            Self::MlpOutput => "mlp_output",
            Self::MlpUp => "mlp_up",
        }
    }
    fn ports(self) -> (&'static str, &'static str, &'static str) {
        match self {
            Self::Query => (
                Block::MODULATED1_ID,
                Attention::QUERY_ID,
                Block::WEIGHT_NAMES[0],
            ),
            Self::Value => (
                Block::MODULATED1_ID,
                Attention::VALUE_ID,
                Block::WEIGHT_NAMES[4],
            ),
            Self::MlpOutput => (Mlp::ACTIVATION_ID, Mlp::OUTPUT_ID, Block::WEIGHT_NAMES[8]),
            Self::MlpUp => (Block::MODULATED2_ID, Mlp::UP_ID, Block::WEIGHT_NAMES[7]),
        }
    }
}
pub fn analyze(
    stage: Stage,
    channels: &[usize],
    reference: &Artifact,
    candidate: &Artifact,
    weights: &SafetensorsArchive,
    artifacts: &artifacts::Artifacts,
) -> Result<Value> {
    let (input_id, output_id, weight_name) = stage.ports();
    let (input_meta, xr) = reference.output(input_id)?;
    let (candidate_meta, xc) = candidate.output(input_id)?;
    ensure!(
        input_meta.shape == candidate_meta.shape,
        "local input shapes differ"
    );
    if matches!(stage, Stage::Query | Stage::Value) {
        ensure!(
            xr == xc,
            "earliest Q/V inputs must be identical to isolate GEMM"
        );
    }
    let output_meta = reference.output_metadata(output_id)?;
    let candidate_output = candidate.output_metadata(output_id)?;
    ensure!(
        output_meta.shape == candidate_output.shape
            && output_meta.dtype == "BF16"
            && candidate_output.dtype == "BF16"
            && output_meta.shape.len() == 2,
        "output geometry mismatch"
    );
    let rows = usize::try_from(input_meta.shape[0])?;
    let k = usize::try_from(input_meta.shape[1])?;
    ensure!(
        (1..=16).contains(&rows) && (1..=65536).contains(&k),
        "diagnostic supports at most 16 rows and K<=65536"
    );
    ensure!(
        output_meta.shape[0] == input_meta.shape[0],
        "dense input/output row count differs"
    );
    let n = usize::try_from(output_meta.shape[1])?;
    ensure!(
        !channels.is_empty() && channels.len() <= 64 && channels.iter().all(|c| *c < n),
        "selected channels outside extent/limit"
    );
    let weight = reference.weight(weights, weight_name, k, n)?;
    let stride = k.checked_mul(2).context("row byte count overflow")?;
    let mut prepared = Vec::new();
    for &channel in channels {
        let row = &weight[channel * stride..(channel + 1) * stride];
        artifacts.bytes(&format!("{}-weight-row-{channel}.bf16", stage.name()), row)?;
        for r in 0..rows {
            prepared.push((
                r,
                channel,
                Dot::prepare(&xr[r * stride..(r + 1) * stride], row)?,
                Dot::prepare(&xc[r * stride..(r + 1) * stride], row)?,
                artifacts::sha256(row),
            ));
        }
    }
    // Candidate outputs are read only after every exact target and domain
    // bound has been prepared from the two explicit, immutable local inputs.
    let (_, yr) = reference.output(output_id)?;
    let (_, yc) = candidate.output(output_id)?;
    let mut records = Vec::new();
    let mut incompatible = 0;
    let mut reference_replay_mismatches = 0;
    for (row, channel, dr, dc, weight_sha) in prepared {
        let a = dyadic::bits(&yr, row * n + channel);
        let b = dyadic::bits(&yc, row * n + channel);
        ensure!(
            dyadic::value(a).is_finite() && dyadic::value(b).is_finite(),
            "nonfinite observed output"
        );
        let reference_replay = a == dr.sequential_bf16_bits;
        if !reference_replay {
            reference_replay_mismatches += 1;
        }
        let compatible_reference = dr.compatible(a);
        let compatible_candidate = dc.compatible(b);
        incompatible += usize::from(!compatible_reference) + usize::from(!compatible_candidate);
        let counterfactual = dyadic::from_bf16(dc.sequential_bf16_bits)?;
        let observed = dyadic::from_bf16(b)?.subtract(dyadic::from_bf16(a)?)?;
        let local = dyadic::from_bf16(b)?.subtract(counterfactual)?;
        let upstream = counterfactual.subtract(dyadic::from_bf16(a)?)?;
        let exact_input_effect = dc.exact.subtract(dr.exact)?;
        let midpoint = if dyadic::adjacent(a, b) {
            Some(dyadic::midpoint_distance(dr.exact, a, b)?)
        } else {
            None
        };
        records.push(json!({"row":row,"channel":channel,"weight_row_sha256":weight_sha,
            "reference_output_bits":a,"candidate_output_bits":b,"reference_output":dyadic::value(a),"candidate_output":dyadic::value(b),
            "reference_input_dot":dr,"candidate_input_dot":dc,"reference_sequential_replay_matches":reference_replay,
            "reference_within_declared_dot_bound":compatible_reference,"candidate_within_declared_dot_bound":compatible_candidate,
            "observed_adjacent_bf16":dyadic::adjacent(a,b),"exact_reference_dot_minus_observed_midpoint":midpoint,
            "exact_observed_difference":observed,"reference_fold_on_candidate_input_bits":dc.sequential_bf16_bits,
            "exact_local_profile_residual":local,"exact_upstream_effect_under_reference_fold":upstream,
            "exact_mathematical_input_effect":exact_input_effect,
            "decomposition":"Ycuda-Yref = [Ycuda-B(F32fold(Xcuda,W))] + [B(F32fold(Xcuda,W))-Yref]; this is node-local, not a whole-block oracle"}));
    }
    Ok(
        json!({"stage":stage.name(),"input_port":input_id,"output_port":output_id,"weight":weight_name,"input_shapes":input_meta.shape,
        "reference_input_sha256":input_meta.sha256,"candidate_input_sha256":candidate_meta.sha256,"input_bits_identical":xr==xc,
        "selected_channels":channels,"rows":rows,"K":k,"records":records,
        "dot_bound_incompatible_count":incompatible,"reference_replay_mismatch_count":reference_replay_mismatches,
        "original_block_criterion":"abs(candidate-reference)<=0.02+0.02*abs(reference), unchanged; original result remains independent",
        "scope":"selected real weight rows only; compatibility supports a rounding/propagation explanation, does not prove CUDA or the model correct"}),
    )
}
