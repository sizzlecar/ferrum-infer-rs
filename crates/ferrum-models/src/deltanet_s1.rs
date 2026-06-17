//! W3-S1 deterministic Gated DeltaNet single-layer reference harness.
//!
//! This module is intentionally small and CPU-only. It gives the W3 release
//! goal a Ferrum-owned single-layer implementation that can emit the same dump
//! schema consumed by `scripts/release/w3_deltanet_s1_layer_compare.py`.
//! Product model integration still belongs to the real W3 DeltaNet model path.

use std::fs;
use std::io::Write;
use std::path::Path;

use serde_json::json;

use crate::moe::router::route;

pub const DUMP_MANIFEST_NAME: &str = "w3_deltanet_s1_dump_manifest.json";

const FLOAT_TENSORS: &[&str] = &[
    "input",
    "delta_q",
    "delta_k",
    "delta_v",
    "delta_beta",
    "delta_core",
    "delta_gate",
    "delta_output",
    "router_logits",
    "router_topk_weights",
    "routed_expert_output",
    "shared_expert_output",
    "moe_output",
    "layer_output",
];

const INT_TENSORS: &[&str] = &["router_topk_indices"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct W3DeltaNetS1Shape {
    pub tokens: usize,
    pub hidden_dim: usize,
    pub heads: usize,
    pub key_dim: usize,
    pub value_dim: usize,
    pub experts: usize,
    pub top_k: usize,
    pub expert_hidden_dim: usize,
}

impl Default for W3DeltaNetS1Shape {
    fn default() -> Self {
        Self {
            tokens: 6,
            hidden_dim: 8,
            heads: 2,
            key_dim: 3,
            value_dim: 4,
            experts: 5,
            top_k: 2,
            expert_hidden_dim: 6,
        }
    }
}

impl W3DeltaNetS1Shape {
    pub fn validate(self) -> Result<(), String> {
        for (name, value) in [
            ("tokens", self.tokens),
            ("hidden_dim", self.hidden_dim),
            ("heads", self.heads),
            ("key_dim", self.key_dim),
            ("value_dim", self.value_dim),
            ("experts", self.experts),
            ("top_k", self.top_k),
            ("expert_hidden_dim", self.expert_hidden_dim),
        ] {
            if value == 0 {
                return Err(format!("{name} must be positive"));
            }
        }
        if self.top_k > self.experts {
            return Err(format!(
                "top_k {} exceeds experts {}",
                self.top_k, self.experts
            ));
        }
        Ok(())
    }

    fn qk_dim(self) -> usize {
        self.heads * self.key_dim
    }

    fn value_total_dim(self) -> usize {
        self.heads * self.value_dim
    }
}

#[derive(Debug, Clone)]
struct ExpertWeights {
    gate: Vec<f64>,
    up: Vec<f64>,
    down: Vec<f64>,
}

#[derive(Debug, Clone)]
struct Weights {
    input: Vec<f64>,
    w_q: Vec<f64>,
    w_k: Vec<f64>,
    w_v: Vec<f64>,
    w_beta: Vec<f64>,
    w_delta_gate: Vec<f64>,
    w_o: Vec<f64>,
    w_router: Vec<f64>,
    experts: Vec<ExpertWeights>,
    shared_gate: Vec<f64>,
    shared_up: Vec<f64>,
    shared_down: Vec<f64>,
}

#[derive(Debug, Clone)]
pub struct W3DeltaNetS1Dump {
    pub input: Vec<f32>,
    pub delta_q: Vec<f32>,
    pub delta_k: Vec<f32>,
    pub delta_v: Vec<f32>,
    pub delta_beta: Vec<f32>,
    pub delta_core: Vec<f32>,
    pub delta_gate: Vec<f32>,
    pub delta_output: Vec<f32>,
    pub router_logits: Vec<f32>,
    pub router_topk_indices: Vec<i64>,
    pub router_topk_weights: Vec<f32>,
    pub routed_expert_output: Vec<f32>,
    pub shared_expert_output: Vec<f32>,
    pub moe_output: Vec<f32>,
    pub layer_output: Vec<f32>,
}

struct Lcg {
    state: u32,
}

impl Lcg {
    fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u32 {
        self.state = self
            .state
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.state
    }
}

fn rand_float(gen: &mut Lcg, scale: f64) -> f64 {
    let raw = gen.next() as f64;
    let centered = (raw / u32::MAX as f64) * 2.0 - 1.0;
    centered * scale
}

fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

fn silu(x: f64) -> f64 {
    x * sigmoid(x)
}

fn softmax(values: &[f64]) -> Vec<f64> {
    let peak = values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, |a, b| a.max(b));
    let exp_values: Vec<f64> = values.iter().map(|value| (*value - peak).exp()).collect();
    let denom: f64 = exp_values.iter().sum();
    exp_values.into_iter().map(|value| value / denom).collect()
}

fn make_weights(shape: W3DeltaNetS1Shape, seed: u32) -> Weights {
    let mut gen = Lcg::new(seed);
    let mut floats =
        |count: usize, scale: f64| (0..count).map(|_| rand_float(&mut gen, scale)).collect();

    let experts = (0..shape.experts)
        .map(|_| ExpertWeights {
            gate: floats(shape.hidden_dim * shape.expert_hidden_dim, 0.10),
            up: floats(shape.hidden_dim * shape.expert_hidden_dim, 0.10),
            down: floats(shape.expert_hidden_dim * shape.hidden_dim, 0.10),
        })
        .collect();
    Weights {
        input: floats(shape.tokens * shape.hidden_dim, 0.30),
        w_q: floats(shape.hidden_dim * shape.qk_dim(), 0.12),
        w_k: floats(shape.hidden_dim * shape.qk_dim(), 0.12),
        w_v: floats(shape.hidden_dim * shape.value_total_dim(), 0.12),
        w_beta: floats(shape.hidden_dim * shape.heads, 0.10),
        w_delta_gate: floats(shape.hidden_dim * shape.value_total_dim(), 0.10),
        w_o: floats(shape.value_total_dim() * shape.hidden_dim, 0.10),
        w_router: floats(shape.hidden_dim * shape.experts, 0.10),
        experts,
        shared_gate: floats(shape.hidden_dim * shape.expert_hidden_dim, 0.10),
        shared_up: floats(shape.hidden_dim * shape.expert_hidden_dim, 0.10),
        shared_down: floats(shape.expert_hidden_dim * shape.hidden_dim, 0.10),
    }
}

fn matmul_token_major(
    x: &[f64],
    weights: &[f64],
    tokens: usize,
    in_dim: usize,
    out_dim: usize,
) -> Vec<f64> {
    let mut out = vec![0.0; tokens * out_dim];
    for t in 0..tokens {
        for o in 0..out_dim {
            let mut acc = 0.0;
            for i in 0..in_dim {
                acc += x[t * in_dim + i] * weights[i * out_dim + o];
            }
            out[t * out_dim + o] = acc;
        }
    }
    out
}

fn qk_index(shape: W3DeltaNetS1Shape, t: usize, h: usize, kk: usize) -> usize {
    (t * shape.heads + h) * shape.key_dim + kk
}

fn value_index(shape: W3DeltaNetS1Shape, t: usize, h: usize, vv: usize) -> usize {
    (t * shape.heads + h) * shape.value_dim + vv
}

fn delta_rule(shape: W3DeltaNetS1Shape, q: &[f64], k: &[f64], v: &[f64], beta: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; shape.tokens * shape.value_total_dim()];
    for h in 0..shape.heads {
        let mut state = vec![0.0; shape.key_dim * shape.value_dim];
        for t in 0..shape.tokens {
            let bt = beta[t * shape.heads + h];
            for vv in 0..shape.value_dim {
                let mut pred = 0.0;
                for kk in 0..shape.key_dim {
                    pred += k[qk_index(shape, t, h, kk)] * state[kk * shape.value_dim + vv];
                }
                let delta = bt * (v[value_index(shape, t, h, vv)] - pred);
                for kk in 0..shape.key_dim {
                    state[kk * shape.value_dim + vv] += k[qk_index(shape, t, h, kk)] * delta;
                }
            }
            for vv in 0..shape.value_dim {
                let mut acc = 0.0;
                for kk in 0..shape.key_dim {
                    acc += q[qk_index(shape, t, h, kk)] * state[kk * shape.value_dim + vv];
                }
                out[value_index(shape, t, h, vv)] = acc;
            }
        }
    }
    out
}

fn expert_mlp(
    token: &[f64],
    gate: &[f64],
    up: &[f64],
    down: &[f64],
    hidden_dim: usize,
    expert_hidden_dim: usize,
) -> Vec<f64> {
    let mut fused = vec![0.0; expert_hidden_dim];
    for j in 0..expert_hidden_dim {
        let mut gate_acc = 0.0;
        let mut up_acc = 0.0;
        for i in 0..hidden_dim {
            gate_acc += token[i] * gate[i * expert_hidden_dim + j];
            up_acc += token[i] * up[i * expert_hidden_dim + j];
        }
        fused[j] = silu(gate_acc) * up_acc;
    }
    let mut out = vec![0.0; hidden_dim];
    for o in 0..hidden_dim {
        let mut acc = 0.0;
        for j in 0..expert_hidden_dim {
            acc += fused[j] * down[j * hidden_dim + o];
        }
        out[o] = acc;
    }
    out
}

fn to_f32(values: Vec<f64>) -> Vec<f32> {
    values.into_iter().map(|value| value as f32).collect()
}

pub fn compute_w3_deltanet_s1_dump(
    shape: W3DeltaNetS1Shape,
    seed: u32,
) -> Result<W3DeltaNetS1Dump, String> {
    shape.validate()?;
    let weights = make_weights(shape, seed);
    let x = weights.input;
    let q = matmul_token_major(
        &x,
        &weights.w_q,
        shape.tokens,
        shape.hidden_dim,
        shape.qk_dim(),
    );
    let k = matmul_token_major(
        &x,
        &weights.w_k,
        shape.tokens,
        shape.hidden_dim,
        shape.qk_dim(),
    );
    let v = matmul_token_major(
        &x,
        &weights.w_v,
        shape.tokens,
        shape.hidden_dim,
        shape.value_total_dim(),
    );
    let beta_raw = matmul_token_major(
        &x,
        &weights.w_beta,
        shape.tokens,
        shape.hidden_dim,
        shape.heads,
    );
    let beta: Vec<f64> = beta_raw.into_iter().map(sigmoid).collect();
    let delta_core = delta_rule(shape, &q, &k, &v, &beta);
    let delta_gate_raw = matmul_token_major(
        &x,
        &weights.w_delta_gate,
        shape.tokens,
        shape.hidden_dim,
        shape.value_total_dim(),
    );
    let delta_gate: Vec<f64> = delta_gate_raw.into_iter().map(sigmoid).collect();
    let gated_delta: Vec<f64> = delta_core
        .iter()
        .zip(&delta_gate)
        .map(|(core, gate)| core * gate)
        .collect();
    let delta_output = matmul_token_major(
        &gated_delta,
        &weights.w_o,
        shape.tokens,
        shape.value_total_dim(),
        shape.hidden_dim,
    );
    let router_logits = matmul_token_major(
        &x,
        &weights.w_router,
        shape.tokens,
        shape.hidden_dim,
        shape.experts,
    );

    // Use Ferrum's MoE router for selected expert ids. Recompute the top-k
    // weights in f64 over the selected logits so the dump aligns with the S1
    // Python reference while still checking Ferrum's routing tie/order.
    let router_logits_f32 = to_f32(router_logits.clone());
    let routed = route(
        &router_logits_f32,
        shape.tokens,
        shape.experts,
        shape.top_k,
        true,
    );

    let mut router_topk_indices = Vec::with_capacity(shape.tokens * shape.top_k);
    let mut router_topk_weights = Vec::with_capacity(shape.tokens * shape.top_k);
    let mut routed_expert_output = vec![0.0; shape.tokens * shape.hidden_dim];
    let mut shared_expert_output = vec![0.0; shape.tokens * shape.hidden_dim];
    for t in 0..shape.tokens {
        let token = &x[t * shape.hidden_dim..(t + 1) * shape.hidden_dim];
        let selected_ids = &routed.expert_ids[t * shape.top_k..(t + 1) * shape.top_k];
        let selected_logits: Vec<f64> = selected_ids
            .iter()
            .map(|id| router_logits[t * shape.experts + *id as usize])
            .collect();
        let selected_weights = softmax(&selected_logits);
        let mut routed_row = vec![0.0; shape.hidden_dim];
        for (&expert_id, &weight) in selected_ids.iter().zip(&selected_weights) {
            let expert = &weights.experts[expert_id as usize];
            let expert_out = expert_mlp(
                token,
                &expert.gate,
                &expert.up,
                &expert.down,
                shape.hidden_dim,
                shape.expert_hidden_dim,
            );
            for i in 0..shape.hidden_dim {
                routed_row[i] += weight * expert_out[i];
            }
            router_topk_indices.push(expert_id as i64);
            router_topk_weights.push(weight);
        }
        let shared = expert_mlp(
            token,
            &weights.shared_gate,
            &weights.shared_up,
            &weights.shared_down,
            shape.hidden_dim,
            shape.expert_hidden_dim,
        );
        for i in 0..shape.hidden_dim {
            routed_expert_output[t * shape.hidden_dim + i] = routed_row[i];
            shared_expert_output[t * shape.hidden_dim + i] = shared[i];
        }
    }
    let moe_output: Vec<f64> = routed_expert_output
        .iter()
        .zip(&shared_expert_output)
        .map(|(routed, shared)| routed + shared)
        .collect();
    let layer_output: Vec<f64> = x
        .iter()
        .zip(&delta_output)
        .zip(&moe_output)
        .map(|((input, delta), moe)| input + delta + moe)
        .collect();

    Ok(W3DeltaNetS1Dump {
        input: to_f32(x),
        delta_q: to_f32(q),
        delta_k: to_f32(k),
        delta_v: to_f32(v),
        delta_beta: to_f32(beta),
        delta_core: to_f32(delta_core),
        delta_gate: to_f32(delta_gate),
        delta_output: to_f32(delta_output),
        router_logits: router_logits_f32,
        router_topk_indices,
        router_topk_weights: to_f32(router_topk_weights),
        routed_expert_output: to_f32(routed_expert_output),
        shared_expert_output: to_f32(shared_expert_output),
        moe_output: to_f32(moe_output),
        layer_output: to_f32(layer_output),
    })
}

fn tensor_shapes(shape: W3DeltaNetS1Shape) -> serde_json::Value {
    json!({
        "input": [shape.tokens, shape.hidden_dim],
        "delta_q": [shape.tokens, shape.heads, shape.key_dim],
        "delta_k": [shape.tokens, shape.heads, shape.key_dim],
        "delta_v": [shape.tokens, shape.heads, shape.value_dim],
        "delta_beta": [shape.tokens, shape.heads],
        "delta_core": [shape.tokens, shape.heads, shape.value_dim],
        "delta_gate": [shape.tokens, shape.value_total_dim()],
        "delta_output": [shape.tokens, shape.hidden_dim],
        "router_logits": [shape.tokens, shape.experts],
        "router_topk_indices": [shape.tokens, shape.top_k],
        "router_topk_weights": [shape.tokens, shape.top_k],
        "routed_expert_output": [shape.tokens, shape.hidden_dim],
        "shared_expert_output": [shape.tokens, shape.hidden_dim],
        "moe_output": [shape.tokens, shape.hidden_dim],
        "layer_output": [shape.tokens, shape.hidden_dim],
    })
}

fn write_f32(path: &Path, values: &[f32]) -> Result<(), String> {
    let mut file = fs::File::create(path).map_err(|err| format!("create {path:?}: {err}"))?;
    for value in values {
        file.write_all(&value.to_le_bytes())
            .map_err(|err| format!("write {path:?}: {err}"))?;
    }
    Ok(())
}

fn write_json(path: &Path, value: &serde_json::Value) -> Result<(), String> {
    let mut file = fs::File::create(path).map_err(|err| format!("create {path:?}: {err}"))?;
    serde_json::to_writer_pretty(&mut file, value)
        .map_err(|err| format!("write json {path:?}: {err}"))?;
    file.write_all(b"\n")
        .map_err(|err| format!("write newline {path:?}: {err}"))?;
    Ok(())
}

pub fn write_w3_deltanet_s1_dump(
    out_dir: &Path,
    shape: W3DeltaNetS1Shape,
    seed: u32,
    producer: &str,
) -> Result<(), String> {
    shape.validate()?;
    fs::create_dir_all(out_dir).map_err(|err| format!("create {out_dir:?}: {err}"))?;
    let dump = compute_w3_deltanet_s1_dump(shape, seed)?;
    let tensors: &[(&str, &[f32])] = &[
        ("input", &dump.input),
        ("delta_q", &dump.delta_q),
        ("delta_k", &dump.delta_k),
        ("delta_v", &dump.delta_v),
        ("delta_beta", &dump.delta_beta),
        ("delta_core", &dump.delta_core),
        ("delta_gate", &dump.delta_gate),
        ("delta_output", &dump.delta_output),
        ("router_logits", &dump.router_logits),
        ("router_topk_weights", &dump.router_topk_weights),
        ("routed_expert_output", &dump.routed_expert_output),
        ("shared_expert_output", &dump.shared_expert_output),
        ("moe_output", &dump.moe_output),
        ("layer_output", &dump.layer_output),
    ];
    for (name, values) in tensors {
        write_f32(&out_dir.join(format!("{name}.bin")), values)?;
    }
    write_json(
        &out_dir.join("router_topk_indices.json"),
        &json!(dump.router_topk_indices),
    )?;
    let float_tensors = FLOAT_TENSORS
        .iter()
        .map(|name| (name.to_string(), json!(format!("{name}.bin"))))
        .collect::<serde_json::Map<String, serde_json::Value>>();
    let int_tensors = INT_TENSORS
        .iter()
        .map(|name| (name.to_string(), json!(format!("{name}.json"))))
        .collect::<serde_json::Map<String, serde_json::Value>>();
    write_json(
        &out_dir.join(DUMP_MANIFEST_NAME),
        &json!({
            "schema_version": 1,
            "producer": producer,
            "shape": {
                "tokens": shape.tokens,
                "hidden_dim": shape.hidden_dim,
                "heads": shape.heads,
                "key_dim": shape.key_dim,
                "value_dim": shape.value_dim,
                "experts": shape.experts,
                "top_k": shape.top_k,
                "expert_hidden_dim": shape.expert_hidden_dim,
            },
            "seed": seed,
            "float_tensors": float_tensors,
            "int_tensors": int_tensors,
            "tensor_shapes": tensor_shapes(shape),
            "layout": "token-major contiguous float32 unless noted",
            "semantics": {
                "delta_rule": "S_t = S_{t-1} + beta_t * k_t^T * (v_t - k_t @ S_{t-1}); core_t = q_t @ S_t",
                "delta_output": "matmul(delta_core * sigmoid(x @ w_delta_gate), w_o)",
                "router": "Ferrum MoE route() stable top-k ids; dump weights are selected-logit softmax for reference alignment",
                "expert": "down(silu(x @ gate) * (x @ up))",
                "moe_merge": "routed_topk_expert_sum + shared_expert_output",
                "layer_output": "input + delta_output + moe_output",
            },
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_shape_dump_has_expected_tensor_sizes() {
        let shape = W3DeltaNetS1Shape::default();
        let dump = compute_w3_deltanet_s1_dump(shape, 9271).unwrap();
        assert_eq!(dump.input.len(), shape.tokens * shape.hidden_dim);
        assert_eq!(
            dump.delta_q.len(),
            shape.tokens * shape.heads * shape.key_dim
        );
        assert_eq!(
            dump.delta_core.len(),
            shape.tokens * shape.heads * shape.value_dim
        );
        assert_eq!(dump.router_topk_indices.len(), shape.tokens * shape.top_k);
        assert_eq!(dump.layer_output.len(), shape.tokens * shape.hidden_dim);
    }

    #[test]
    fn shared_expert_merge_is_explicitly_materialized() {
        let shape = W3DeltaNetS1Shape::default();
        let dump = compute_w3_deltanet_s1_dump(shape, 9271).unwrap();
        for i in 0..dump.moe_output.len() {
            let expected = dump.routed_expert_output[i] + dump.shared_expert_output[i];
            assert!(
                (dump.moe_output[i] - expected).abs() < 1e-7,
                "moe_output[{i}] mismatch"
            );
        }
    }

    #[test]
    fn top_k_indices_are_in_range() {
        let shape = W3DeltaNetS1Shape::default();
        let dump = compute_w3_deltanet_s1_dump(shape, 9271).unwrap();
        for &expert_id in &dump.router_topk_indices {
            assert!(expert_id >= 0);
            assert!((expert_id as usize) < shape.experts);
        }
    }
}
