//! Predeclared Single / two-token basis oracle. All intermediate expectations
//! originate from weight bytes and independent arithmetic enclosures, never
//! from candidate Q/K, attention, or projected outputs.
#[path = "arithmetic.rs"]
mod arithmetic;
#[path = "dense.rs"]
mod dense;
#[path = "../verify_qwen_image21_qk_norm/oracle.rs"]
mod qk;
#[path = "../../../ferrum-kernels/src/backend/reference/transcendental/oracle/interval.rs"]
mod real_interval;
#[path = "../../../ferrum-kernels/src/backend/reference/transcendental/oracle/rounding.rs"]
mod rounding;
#[cfg(test)]
#[path = "oracle_tests.rs"]
mod tests;

use super::Probe;
use anyhow::{bail, Context, Result};
use arithmetic::attention_choices;
use ferrum_execution::TensorExecutionOutput;
use ferrum_models::qwen_image21::{QwenImage21Block0AttentionFamily as Family, QwenImage21Config};
use serde_json::{json, Value};

pub(super) fn bits(bytes: &[u8], index: usize) -> u16 {
    u16::from_le_bytes([bytes[index * 2], bytes[index * 2 + 1]])
}
pub(super) fn value(raw: u16) -> f64 {
    f64::from(rounding::decode_bf16(raw))
}
pub(super) fn finite(raw: u16) -> bool {
    raw & 0x7fff < 0x7f80
}
fn signed_column(raw: u16, negative: bool) -> Vec<u16> {
    if raw & 0x7fff == 0 {
        vec![0, 0x8000]
    } else {
        vec![raw ^ if negative { 0x8000 } else { 0 }]
    }
}

pub struct Prepared {
    input: Vec<u8>,
    frequencies: Vec<u8>,
    expected: Vec<Vec<Vec<u16>>>,
    samples: Vec<dense::Sample>,
    hidden: usize,
    head_dim: usize,
    probe: Probe,
    scale: f32,
    epsilon: f32,
}
impl Prepared {
    pub fn new(config: &QwenImage21Config, probe: Probe, weights: [&[u8]; 6]) -> Result<Self> {
        let hidden = usize::try_from(config.hidden_size()).context("hidden size exceeds usize")?;
        let head_dim =
            usize::try_from(config.head_dim()).context("head dimension exceeds usize")?;
        let epsilon = config.epsilon();
        let epsilon = (epsilon.numerator() as f64 / epsilon.denominator() as f64) as f32;
        let scale = Family::scale_for_head_dim(config.head_dim())?;
        let scale = (scale.numerator() as f64 / scale.denominator() as f64) as f32;
        Self::prepare(hidden, head_dim, epsilon, scale, probe, weights)
    }
    fn prepare(
        hidden: usize,
        head_dim: usize,
        epsilon: f32,
        scale: f32,
        probe: Probe,
        weights: [&[u8]; 6],
    ) -> Result<Self> {
        if !scale.is_finite()
            || scale <= 0.0
            || head_dim < 2
            || !head_dim.is_power_of_two()
            || head_dim > 256
        {
            bail!("unresolved: unsupported attention geometry or scale");
        }
        let matrix_bytes = hidden
            .checked_mul(hidden)
            .and_then(|x| x.checked_mul(2))
            .context("matrix byte extent overflows")?;
        if weights[4].len() != matrix_bytes || weights[5].len() != matrix_bytes {
            bail!("attention oracle needs complete exact V/out OI matrices");
        }
        let prepared_qk = qk::Prepared::new(
            hidden,
            head_dim,
            epsilon,
            [weights[0], weights[1], weights[2], weights[3]],
        )?;
        let mut expected = prepared_qk.expected_bf16().to_vec();
        if probe.rows() == 2 {
            // Every nonzero stage before RoPE is odd: products/squares share
            // the same RMS denominator and RN-even conversion is sign symmetric.
            // Dense exact-zero signs remain independent alternatives.
            for values in &mut expected {
                let negative = values
                    .iter()
                    .map(|set| {
                        let mut v = set.iter().map(|x| x ^ 0x8000).collect::<Vec<_>>();
                        v.sort_unstable();
                        v.dedup();
                        v
                    })
                    .collect::<Vec<_>>();
                values.extend(negative);
            }
        }
        let rotary = qk::RotaryProbe::Identity;
        expected.extend([
            rotary.choices(&expected[2], head_dim)?,
            rotary.choices(&expected[3], head_dim)?,
        ]);
        // A basis GEMM is an exact weight-column extraction only in this
        // predeclared finite normal-or-zero matrix domain; no FTZ assumption.
        for pair in weights[4].chunks_exact(2) {
            let raw = u16::from_le_bytes([pair[0], pair[1]]) & 0x7fff;
            if raw >= 0x7f80 || (raw != 0 && raw < 0x0080) {
                bail!("unresolved: V basis projection requires finite normal-or-zero weights");
            }
        }
        let mut values = Vec::with_capacity(hidden * probe.rows());
        for row in 0..probe.rows() {
            for channel in 0..hidden {
                values.push(signed_column(bits(weights[4], channel * hidden), row == 1));
            }
        }
        let attention = attention_choices(
            &expected[4],
            &expected[5],
            &values,
            hidden,
            head_dim,
            scale,
            probe,
        )?;
        let samples = dense::prepare(&attention, hidden, probe.rows(), weights[5])?;
        expected.extend([values, attention]);
        let mut input = vec![
            0;
            hidden
                .checked_mul(probe.rows())
                .and_then(|x| x.checked_mul(2))
                .context("input bytes overflow")?
        ];
        input[..2].copy_from_slice(&0x3f80_u16.to_le_bytes());
        if probe.rows() == 2 {
            input[hidden * 2..hidden * 2 + 2].copy_from_slice(&0xbf80_u16.to_le_bytes());
        }
        let frequencies = rotary.frequencies(head_dim)?.repeat(probe.rows());
        Ok(Self {
            input,
            frequencies,
            expected,
            samples,
            hidden,
            head_dim,
            probe,
            scale,
            epsilon,
        })
    }
    pub fn input(&self) -> &[u8] {
        &self.input
    }
    pub fn frequencies(&self) -> &[u8] {
        &self.frequencies
    }
    pub fn declaration(&self) -> Value {
        json!({
            "probe":self.probe,"input":"S1: e0; S2: [e0,-e0]; already normalized/modulated attention entrance", "rows":self.probe.rows(),"hidden":self.hidden,"head_dim":self.head_dim,"effective_scale_f32_bits":self.scale.to_bits(),"effective_epsilon_f32_bits":self.epsilon.to_bits(),
            "mask":"Single: one valid key; CausalPair: k<=q; FullPair: one length-2 bidirectional image block; no padding", "frequencies":"identity [S,D/2,2], analytic FMA signed-zero propagation",
            "qk":"reused candidate-independent grouped RMS bounds with both BF16 conversions; negative basis via finite sign symmetry, identity RoPE recomputed over all allowed branches",
            "score":"all prior BF16 Q/K choices enclosed; exact BF16 product then FP32 RN; bounded balanced pairwise tree, RN32 explicit scale; never candidate-derived",
            "attention":"ascending-key online FP32 operations; no intermediate BF16 probabilities; all allowed input branches propagated by unions/outward hulls, final RN32 multiply then integer BF16 RNE",
            "exp_criterion":"predeclared 4 finite FP32-neighbor envelope around independent Taylor real-exp enclosures at every possible FP32 input; exp(+/-0)=1 exactly; endpoint hull is a conservative union across different allowed inputs; no intersection between distinct input branches",
            "exp_scope":"composition acceptance criterion only: prior BF16-input SiLU qualification does not qualify general FP32 exp/libdevice, and composed output acceptance does not prove every internal exp call obeyed this envelope",
            "unresolved_domain":"finite FP32 arithmetic; exp arguments within (-256,0], strictly positive finite rounded real-exp classification; at most 256 BF16 alternatives per attention channel; sampled dense products exact FP32 and lattice >=2^-126; overflow or unsupported classification fails before execution",
            "all_channel_ports": &Family::OUTPUT_IDS[..8],"nonunique_counts":self.expected.iter().map(|xs|xs.iter().filter(|x|x.len()>1).count()).collect::<Vec<_>>(),
            "projected_samples":self.samples,"projected_scope":"predeclared up to 7 coordinates per row, derived only from hidden dimension; not all projected channels",
            "dense_bound":"outward F64 exact-product interval over every BF16 attention choice, gamma(2K,2^-24)*absolute_sum_upper plus BF16 relative and subnormal rounding terms",
            "candidate_independence":"all input bytes, all expected sets and sampled output bounds prepared before runtime creation; output data only enters compare", "scope":"single and pair attention subgraphs; no complete block, general-input qualification, Torch/vLLM parity or performance claim"
        })
    }
    pub fn compare(&self, outputs: &[TensorExecutionOutput]) -> Result<Value> {
        let records = outputs
            .iter()
            .map(|output| (output.value_id().as_str(), output.bytes().bytes()))
            .collect::<Vec<_>>();
        self.compare_records(&records)
    }
    fn compare_records(&self, outputs: &[(&str, &[u8])]) -> Result<Value> {
        if outputs.len() != 9 {
            bail!("attention diagnostic expects exactly nine output ports");
        }
        let mut reports = Vec::new();
        let mut failures = 0;
        for (id, expected) in Family::OUTPUT_IDS[..8].iter().zip(&self.expected) {
            let raw = output(outputs, id, expected.len() * 2)?;
            let errors=expected.iter().enumerate().filter(|(index,set)|!set.contains(&bits(raw,*index))).map(|(index,set)|json!({"row":index/self.hidden,"channel":index%self.hidden,"actual_bits":bits(raw,index),"allowed_bits":set})).collect::<Vec<_>>();
            failures += errors.len();
            reports.push(json!({"port":id,"samples":expected.len(),"failures":errors}));
        }
        let raw = output(
            outputs,
            Family::OUTPUT_ID,
            self.hidden * self.probe.rows() * 2,
        )?;
        let projected = dense::compare(&self.samples, raw, self.hidden);
        failures += projected.len();
        reports.push(json!({"port":Family::OUTPUT_ID,"samples":self.samples.len(),"failures":projected,"scope":"predeclared coordinates only"}));
        Ok(
            json!({"passed":failures==0,"failures":failures,"ports":reports,"probe":self.probe,"scope":"eight complete intermediate ports; sampled projected coordinates only; independent declared finite-domain composition bounds"}),
        )
    }
}
fn output<'a>(outputs: &[(&str, &'a [u8])], id: &str, length: usize) -> Result<&'a [u8]> {
    let matching = outputs
        .iter()
        .filter(|(name, _)| *name == id)
        .collect::<Vec<_>>();
    if matching.len() != 1 || matching[0].1.len() != length {
        bail!("attention output {id} missing, repeated or wrong size");
    }
    Ok(matching[0].1)
}
