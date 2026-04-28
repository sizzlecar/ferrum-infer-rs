//! Expert dispatch tests — `ExpertStack` construction (from raw stacks
//! and from a synthesized GGUF) plus `moe_forward_cpu` numerical checks.
//!
//! All hand-computed expected values use `silu(x) = x * sigmoid(x)` —
//! the same SwiGLU formulation `Backend::fused_silu_mul_split` implements.

use std::io::{Cursor, Write};

use candle_core::quantized::gguf_file::{self, Value};
use candle_core::quantized::{GgmlDType, QTensor};
use candle_core::{Device, Tensor};
use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_models::moe::{moe_forward_cpu, route, ExpertStack, RouterOutput};

fn silu(x: f32) -> f32 {
    x * (1.0 / (1.0 + (-x).exp()))
}

/// Identity matrix flattened row-major: `[1 0 0; 0 1 0; 0 0 1]` etc.
fn identity_rows(n: usize) -> Vec<f32> {
    let mut v = vec![0.0_f32; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    v
}

/// `scale * I_n` flattened row-major.
fn scaled_identity_rows(n: usize, scale: f32) -> Vec<f32> {
    let mut v = vec![0.0_f32; n * n];
    for i in 0..n {
        v[i * n + i] = scale;
    }
    v
}

#[test]
fn from_dense_stacks_builds_correct_linear_count() {
    // 3 experts, hidden=4, expert_inter=8.
    let n_experts = 3;
    let hidden = 4;
    let ffn = 8;

    let gate = vec![0.5_f32; n_experts * ffn * hidden];
    let up = vec![0.6_f32; n_experts * ffn * hidden];
    let down = vec![0.7_f32; n_experts * hidden * ffn];

    let stack: ExpertStack<CpuBackend> =
        ExpertStack::from_dense_stacks(&gate, &up, &down, n_experts, hidden, ffn).unwrap();
    assert_eq!(stack.num_experts(), n_experts);
    for e in 0..n_experts {
        // gate_up Linear: out=2*ffn, in=hidden
        assert_eq!(stack.gate_up[e].in_features(), hidden);
        assert_eq!(stack.gate_up[e].out_features(), 2 * ffn);
        // down Linear: out=hidden, in=ffn
        assert_eq!(stack.down[e].in_features(), ffn);
        assert_eq!(stack.down[e].out_features(), hidden);
    }
}

#[test]
fn from_dense_stacks_rejects_size_mismatch() {
    let n_experts = 2;
    let hidden = 2;
    let ffn = 2;
    let good = vec![0.0_f32; n_experts * ffn * hidden];
    let bad_short = vec![0.0_f32; 4]; // wrong size
    let result = ExpertStack::<CpuBackend>::from_dense_stacks(
        &bad_short, &good, &good, n_experts, hidden, ffn,
    );
    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(err.contains("gate_stack"), "{err}");
}

#[test]
fn moe_forward_single_expert_gives_that_experts_output() {
    // Two experts. Expert 0 is identity; expert 1 is 2× identity.
    // hidden=2, expert_inter=2, num_experts=2. Input x = [1, 0].
    // Router selects only expert 0 (top_k=1, weight 1.0).
    // Expected: silu(1)*1=silu(1), and second component=0 (since x[1]=0).
    let hidden = 2;
    let ffn = 2;
    let n_experts = 2;

    let mut gate = Vec::new();
    let mut up = Vec::new();
    let mut down = Vec::new();
    // Expert 0: identity
    gate.extend_from_slice(&identity_rows(2));
    up.extend_from_slice(&identity_rows(2));
    down.extend_from_slice(&identity_rows(2));
    // Expert 1: 2× identity
    gate.extend_from_slice(&scaled_identity_rows(2, 2.0));
    up.extend_from_slice(&scaled_identity_rows(2, 2.0));
    down.extend_from_slice(&scaled_identity_rows(2, 2.0));

    let stack =
        ExpertStack::<CpuBackend>::from_dense_stacks(&gate, &up, &down, n_experts, hidden, ffn)
            .unwrap();

    let x = vec![1.0_f32, 0.0];
    let router = RouterOutput {
        expert_ids: vec![0],
        expert_weights: vec![1.0],
    };
    let mut out = Vec::new();
    moe_forward_cpu(&x, 1, hidden, ffn, 1, &router, &stack, &mut out).unwrap();

    // x=[1,0]; expert 0 (identity): gate=[1,0], up=[1,0]
    // silu(gate)*up = [silu(1)*1, silu(0)*0] = [silu(1), 0]
    // down(identity) = [silu(1), 0]
    // weight 1.0 → out = [silu(1), 0]
    let expected = vec![silu(1.0), 0.0];
    for (got, exp) in out.iter().zip(expected.iter()) {
        assert!(
            (got - exp).abs() < 1e-5,
            "expected {exp}, got {got} (diff {})",
            (got - exp).abs()
        );
    }
}

#[test]
fn moe_forward_two_experts_combines_weighted() {
    // hidden=2, expert_inter=2, num_experts=2. Input x=[1, 0].
    // Expert 0: identity. Expert 1: 2× identity.
    // Router: top_k=2, weights [0.3, 0.7].
    let hidden = 2;
    let ffn = 2;
    let n_experts = 2;

    let mut gate = Vec::new();
    let mut up = Vec::new();
    let mut down = Vec::new();
    gate.extend_from_slice(&identity_rows(2));
    up.extend_from_slice(&identity_rows(2));
    down.extend_from_slice(&identity_rows(2));
    gate.extend_from_slice(&scaled_identity_rows(2, 2.0));
    up.extend_from_slice(&scaled_identity_rows(2, 2.0));
    down.extend_from_slice(&scaled_identity_rows(2, 2.0));

    let stack =
        ExpertStack::<CpuBackend>::from_dense_stacks(&gate, &up, &down, n_experts, hidden, ffn)
            .unwrap();

    let x = vec![1.0_f32, 0.0];
    let router = RouterOutput {
        expert_ids: vec![0, 1],
        expert_weights: vec![0.3, 0.7],
    };
    let mut out = Vec::new();
    moe_forward_cpu(&x, 1, hidden, ffn, 2, &router, &stack, &mut out).unwrap();

    // Expert 0 forward (computed above): [silu(1), 0]
    // Expert 1: gate=[2, 0], up=[2, 0]; silu(2)*2 = [silu(2)*2, 0]
    //   down (2× identity) → [2 * silu(2)*2, 0] = [4*silu(2), 0]
    // Combined: 0.3 * [silu(1), 0] + 0.7 * [4*silu(2), 0]
    let expected = vec![0.3 * silu(1.0) + 0.7 * 4.0 * silu(2.0), 0.0];

    for (got, exp) in out.iter().zip(expected.iter()) {
        assert!(
            (got - exp).abs() < 1e-4,
            "expected {exp}, got {got} (diff {})",
            (got - exp).abs()
        );
    }
}

#[test]
fn moe_forward_handles_batch_with_independent_routing() {
    // Two tokens, top_k=1 each. Token 0 routed to expert 0, token 1 to expert 1.
    let hidden = 2;
    let ffn = 2;
    let n_experts = 2;

    let mut gate = Vec::new();
    let mut up = Vec::new();
    let mut down = Vec::new();
    gate.extend_from_slice(&identity_rows(2));
    up.extend_from_slice(&identity_rows(2));
    down.extend_from_slice(&identity_rows(2));
    gate.extend_from_slice(&scaled_identity_rows(2, 2.0));
    up.extend_from_slice(&scaled_identity_rows(2, 2.0));
    down.extend_from_slice(&scaled_identity_rows(2, 2.0));

    let stack =
        ExpertStack::<CpuBackend>::from_dense_stacks(&gate, &up, &down, n_experts, hidden, ffn)
            .unwrap();

    // Token 0: x=[1, 0]; Token 1: x=[1, 0] (same input, different expert)
    let x = vec![1.0_f32, 0.0, 1.0, 0.0];
    let router = RouterOutput {
        expert_ids: vec![0, 1],
        expert_weights: vec![1.0, 1.0],
    };
    let mut out = Vec::new();
    moe_forward_cpu(&x, 2, hidden, ffn, 1, &router, &stack, &mut out).unwrap();

    // Token 0 (expert 0): [silu(1), 0]
    // Token 1 (expert 1): [4*silu(2), 0]
    let expected = vec![silu(1.0), 0.0, 4.0 * silu(2.0), 0.0];
    for (i, (got, exp)) in out.iter().zip(expected.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-4, "[{i}] expected {exp}, got {got}");
    }
}

#[test]
fn moe_forward_rejects_invalid_expert_id() {
    let hidden = 2;
    let ffn = 2;
    let n_experts = 2;

    let zeros = vec![0.0_f32; n_experts * ffn * hidden];
    let stack = ExpertStack::<CpuBackend>::from_dense_stacks(
        &zeros, &zeros, &zeros, n_experts, hidden, ffn,
    )
    .unwrap();

    let x = vec![1.0_f32; hidden];
    let router = RouterOutput {
        expert_ids: vec![99], // out of range
        expert_weights: vec![1.0],
    };
    let mut out = Vec::new();
    let result = moe_forward_cpu(&x, 1, hidden, ffn, 1, &router, &stack, &mut out);
    assert!(result.is_err());
    let err = result.err().unwrap().to_string();
    assert!(err.contains("99"), "{err}");
}

#[test]
fn moe_forward_rejects_shape_mismatch() {
    let hidden = 2;
    let ffn = 2;
    let n_experts = 2;
    let zeros = vec![0.0_f32; n_experts * ffn * hidden];
    let stack = ExpertStack::<CpuBackend>::from_dense_stacks(
        &zeros, &zeros, &zeros, n_experts, hidden, ffn,
    )
    .unwrap();

    let x = vec![1.0_f32; 5]; // wrong size
    let router = RouterOutput {
        expert_ids: vec![0],
        expert_weights: vec![1.0],
    };
    let mut out = Vec::new();
    let r = moe_forward_cpu(&x, 1, hidden, ffn, 1, &router, &stack, &mut out);
    assert!(r.is_err());
}

#[test]
fn router_plus_dispatch_end_to_end_softmax_to_combine() {
    // The integration the production code will use: hand-built logits
    // → route(...) → moe_forward_cpu(...). Verifies the two pieces
    // compose with consistent strides.
    let hidden = 2;
    let ffn = 2;
    let n_experts = 2;
    let top_k = 1;

    // Identical experts (both identity) so we don't depend on which is picked.
    let weights = identity_rows(2);
    let gate = [&weights[..], &weights[..]].concat();
    let up = gate.clone();
    let down = gate.clone();
    let stack =
        ExpertStack::<CpuBackend>::from_dense_stacks(&gate, &up, &down, n_experts, hidden, ffn)
            .unwrap();

    // Router logits favour expert 0 strongly.
    let logits = vec![5.0_f32, 0.0];
    let router = route(&logits, 1, n_experts, top_k, true);
    assert_eq!(router.expert_ids, vec![0]);

    let x = vec![1.5_f32, -0.5];
    let mut out = Vec::new();
    moe_forward_cpu(&x, 1, hidden, ffn, top_k, &router, &stack, &mut out).unwrap();

    // Both experts identity: silu(x[i])*x[i] elementwise; weight 1.0
    let expected = vec![silu(1.5) * 1.5, silu(-0.5) * -0.5];
    for (got, exp) in out.iter().zip(expected.iter()) {
        assert!(
            (got - exp).abs() < 1e-4,
            "end-to-end mismatch: expected {exp}, got {got}"
        );
    }
}

// ── End-to-end via synthesized GGUF ────────────────────────────────────

fn ramp_3d(d0: usize, d1: usize, d2: usize, base: f32) -> QTensor {
    let device = Device::Cpu;
    let n = d0 * d1 * d2;
    let raw: Vec<f32> = (0..n).map(|i| base + (i as f32) * 0.001).collect();
    let t = Tensor::from_vec(raw, (d0, d1, d2), &device).unwrap();
    QTensor::quantize(&t, GgmlDType::F32).unwrap()
}

fn build_minimal_moe_gguf(n_experts: usize, hidden: usize, ffn: usize) -> tempfile::NamedTempFile {
    let gate_exps = ramp_3d(n_experts, ffn, hidden, 0.1);
    let up_exps = ramp_3d(n_experts, ffn, hidden, 0.2);
    let down_exps = ramp_3d(n_experts, hidden, ffn, 0.3);

    let arch_v = Value::String("qwen3moe".to_string());
    let metadata: Vec<(&str, &Value)> = vec![("general.architecture", &arch_v)];
    let tensors: Vec<(&str, &QTensor)> = vec![
        ("blk.0.ffn_gate_exps.weight", &gate_exps),
        ("blk.0.ffn_up_exps.weight", &up_exps),
        ("blk.0.ffn_down_exps.weight", &down_exps),
    ];

    let mut buf: Vec<u8> = Vec::new();
    {
        let mut cursor = Cursor::new(&mut buf);
        gguf_file::write(&mut cursor, &metadata, &tensors).unwrap();
    }
    let mut tmp = tempfile::NamedTempFile::new().unwrap();
    tmp.write_all(&buf).unwrap();
    tmp.flush().unwrap();
    tmp
}

#[test]
fn expert_stack_loads_from_synthesized_gguf() {
    let n_experts = 3;
    let hidden = 4;
    let ffn = 6;
    let tmp = build_minimal_moe_gguf(n_experts, hidden, ffn);

    let stack: ExpertStack<CpuBackend> =
        ExpertStack::open_and_load(tmp.path(), 0, n_experts, hidden, ffn).unwrap();

    assert_eq!(stack.num_experts(), n_experts);
    for e in 0..n_experts {
        assert_eq!(stack.gate_up[e].in_features(), hidden);
        assert_eq!(stack.gate_up[e].out_features(), 2 * ffn);
        assert_eq!(stack.down[e].in_features(), ffn);
        assert_eq!(stack.down[e].out_features(), hidden);
    }
}

#[test]
fn full_pipeline_synthesized_gguf_router_dispatch() {
    // The reason the GGUF series matters: load real-shaped MoE weights
    // and run forward end-to-end. Numerical correctness is harder to
    // assert with ramp tensors (would need to reproduce silu+gemm in
    // the test) so we just check shape + finiteness.
    let n_experts = 4;
    let hidden = 4;
    let ffn = 8;
    let top_k = 2;

    let tmp = build_minimal_moe_gguf(n_experts, hidden, ffn);
    let stack: ExpertStack<CpuBackend> =
        ExpertStack::open_and_load(tmp.path(), 0, n_experts, hidden, ffn).unwrap();

    // Random-ish logits; route with norm.
    let logits = vec![0.1_f32, 1.5, -0.3, 0.8];
    let router = route(&logits, 1, n_experts, top_k, true);

    let x = vec![0.5_f32, -0.25, 0.1, 0.0];
    let mut out = Vec::new();
    moe_forward_cpu(&x, 1, hidden, ffn, top_k, &router, &stack, &mut out).unwrap();

    assert_eq!(out.len(), hidden);
    for (i, &v) in out.iter().enumerate() {
        assert!(v.is_finite(), "out[{i}] = {v} is not finite");
    }
}
