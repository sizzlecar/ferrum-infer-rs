//! ModelRunner CPU integration tests.
//!
//! Tests the full pipeline: embedding → N × layer_forward → norm → lm_head.
//! Uses synthetic weights with tiny dimensions for fast, deterministic testing.

use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_kernels::backend::runner::ModelRunner;
use ferrum_kernels::backend::{
    AttnType, LayerWeights, MlpType, ModelWeights, RopeConfig, TransformerConfig,
};

// ── Test helpers ─────────────────────────────────────────────────────────

fn pseudo_random(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
        })
        .collect()
}

/// Create a tiny transformer config for testing.
fn tiny_config() -> TransformerConfig {
    TransformerConfig {
        num_layers: 2,
        hidden_size: 16,
        intermediate_size: 32,
        num_heads: 2,
        num_kv_heads: 2,
        head_dim: 8,
        vocab_size: 32,
        max_seq_len: 64,
        rms_norm_eps: 1e-5,
        rope: RopeConfig {
            theta: 10000.0,
            head_dim: 8,
            max_seq_len: 64,
        },
        has_qk_norm: false,
        attn_type: AttnType::Gqa,
        mlp_type: MlpType::SwiGlu,
    }
}

/// Create synthetic weights for the tiny config.
fn tiny_weights(cfg: &TransformerConfig) -> ModelWeights<CpuBackend> {
    let h = cfg.hidden_size;
    let nh = cfg.num_heads;
    let nkv = cfg.num_kv_heads;
    let hd = cfg.head_dim;
    let im = cfg.intermediate_size;
    let vocab = cfg.vocab_size;
    let q_dim = nh * hd;
    let kv_dim = nkv * hd;
    let qkv_dim = q_dim + 2 * kv_dim;

    let mut seed = 1000u64;
    let mut next_w = |size: usize| -> Vec<f32> {
        seed += 1;
        pseudo_random(seed, size)
    };

    let embed = next_w(vocab * h);
    let final_norm_w = vec![1.0f32; h]; // unit norm weights for predictable behavior
    let lm_head_w = next_w(vocab * h);

    let layers = (0..cfg.num_layers)
        .map(|_| LayerWeights {
            input_ln_w: vec![1.0f32; h],
            qkv_proj_w: next_w(qkv_dim * h),
            o_proj_w: next_w(h * q_dim),
            post_ln_w: vec![1.0f32; h],
            gate_up_proj_w: next_w(2 * im * h),
            down_proj_w: next_w(h * im),
            q_norm_w: None,
            k_norm_w: None,
        })
        .collect();

    ModelWeights {
        embed,
        layers,
        final_norm_w,
        lm_head_w,
    }
}

fn create_runner() -> ModelRunner<CpuBackend> {
    let cfg = tiny_config();
    let weights = tiny_weights(&cfg);
    ModelRunner::new(cfg, weights)
}

// ── Tests ────────────────────────────────────────────────────────────────

#[test]
fn test_decode_single_token() {
    let mut runner = create_runner();
    let logits = runner.decode("seq-0", 1, 0);
    assert_eq!(logits.len(), 32, "logits should be vocab_size");
    assert!(
        logits.iter().all(|x| x.is_finite()),
        "logits should be finite"
    );
    assert!(
        logits.iter().any(|x| *x != 0.0),
        "logits should be non-zero"
    );
}

#[test]
fn test_decode_multiple_tokens_sequential() {
    let mut runner = create_runner();
    let logits0 = runner.decode("seq-0", 5, 0);
    let logits1 = runner.decode("seq-0", 10, 1);
    let logits2 = runner.decode("seq-0", 15, 2);

    // All should produce valid logits
    for (i, logits) in [&logits0, &logits1, &logits2].iter().enumerate() {
        assert_eq!(logits.len(), 32, "step {i}: wrong logits length");
        assert!(
            logits.iter().all(|x| x.is_finite()),
            "step {i}: non-finite logits"
        );
    }

    // Each step should produce different logits (different token + KV context)
    let diff01: f32 = logits0
        .iter()
        .zip(&logits1)
        .map(|(a, b)| (a - b).abs())
        .sum();
    let diff12: f32 = logits1
        .iter()
        .zip(&logits2)
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(diff01 > 0.01, "step 0 vs 1 should differ, diff={diff01}");
    assert!(diff12 > 0.01, "step 1 vs 2 should differ, diff={diff12}");
}

#[test]
fn test_prefill() {
    let mut runner = create_runner();
    let logits = runner.prefill("seq-0", &[1, 2, 3, 4, 5]);

    assert_eq!(logits.len(), 32, "logits should be vocab_size");
    assert!(
        logits.iter().all(|x| x.is_finite()),
        "logits should be finite"
    );
    assert!(
        logits.iter().any(|x| *x != 0.0),
        "logits should be non-zero"
    );
}

#[test]
fn test_prefill_then_decode() {
    let mut runner = create_runner();

    // Prefill with 3 tokens
    let prefill_logits = runner.prefill("seq-0", &[1, 2, 3]);
    assert_eq!(prefill_logits.len(), 32);

    // Decode next token (pos=3 since 3 tokens already in cache)
    let decode_logits = runner.decode("seq-0", 7, 3);
    assert_eq!(decode_logits.len(), 32);
    assert!(decode_logits.iter().all(|x| x.is_finite()));

    // Decode another (pos=4)
    let decode_logits2 = runner.decode("seq-0", 12, 4);
    assert_eq!(decode_logits2.len(), 32);
    assert!(decode_logits2.iter().all(|x| x.is_finite()));
}

#[test]
fn test_deterministic() {
    // Same config, same weights, same tokens → same logits
    let cfg = tiny_config();
    let weights1 = tiny_weights(&cfg);
    let weights2 = tiny_weights(&cfg);
    let mut runner1 = ModelRunner::<CpuBackend>::new(cfg.clone(), weights1);
    let mut runner2 = ModelRunner::<CpuBackend>::new(cfg, weights2);

    let logits1 = runner1.decode("s", 5, 0);
    let logits2 = runner2.decode("s", 5, 0);

    let max_diff: f32 = logits1
        .iter()
        .zip(&logits2)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max_diff < 1e-6, "determinism: max diff = {max_diff}");
}

#[test]
fn test_reset_reproduces() {
    let mut runner = create_runner();

    // First run
    let logits1 = runner.decode("s", 5, 0);

    // Reset and run again
    runner.reset();
    let logits2 = runner.decode("s", 5, 0);

    let max_diff: f32 = logits1
        .iter()
        .zip(&logits2)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-6,
        "reset should reproduce, max diff = {max_diff}"
    );
}

#[test]
fn test_release_sequence() {
    let mut runner = create_runner();
    runner.decode("seq-a", 1, 0);
    runner.decode("seq-b", 2, 0);

    // Release seq-a, seq-b should still work
    runner.release("seq-a");
    let logits = runner.decode("seq-b", 3, 1);
    assert!(logits.iter().all(|x| x.is_finite()));
}

#[test]
fn test_multiple_sequences_independent() {
    let mut runner = create_runner();

    // Two sequences with different tokens
    let logits_a = runner.decode("seq-a", 1, 0);
    let logits_b = runner.decode("seq-b", 2, 0);

    // They should produce different results (different input tokens)
    let diff: f32 = logits_a
        .iter()
        .zip(&logits_b)
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff > 1e-6,
        "different tokens should produce different logits, diff={diff}"
    );
}

#[test]
fn test_gqa_config() {
    // GQA: 4 Q heads, 2 KV heads
    let cfg = TransformerConfig {
        num_layers: 1,
        hidden_size: 16,
        intermediate_size: 32,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 4,
        vocab_size: 16,
        max_seq_len: 32,
        rms_norm_eps: 1e-5,
        rope: RopeConfig {
            theta: 10000.0,
            head_dim: 4,
            max_seq_len: 32,
        },
        has_qk_norm: false,
        attn_type: AttnType::Gqa,
        mlp_type: MlpType::SwiGlu,
    };

    let h = cfg.hidden_size;
    let nh = cfg.num_heads;
    let nkv = cfg.num_kv_heads;
    let hd = cfg.head_dim;
    let im = cfg.intermediate_size;
    let vocab = cfg.vocab_size;
    let q_dim = nh * hd;
    let kv_dim = nkv * hd;
    let qkv_dim = q_dim + 2 * kv_dim;

    let weights = ModelWeights {
        embed: pseudo_random(1, vocab * h),
        layers: vec![LayerWeights {
            input_ln_w: vec![1.0; h],
            qkv_proj_w: pseudo_random(2, qkv_dim * h),
            o_proj_w: pseudo_random(3, h * q_dim),
            post_ln_w: vec![1.0; h],
            gate_up_proj_w: pseudo_random(4, 2 * im * h),
            down_proj_w: pseudo_random(5, h * im),
            q_norm_w: None,
            k_norm_w: None,
        }],
        final_norm_w: vec![1.0; h],
        lm_head_w: pseudo_random(6, vocab * h),
    };

    let mut runner = ModelRunner::<CpuBackend>::new(cfg, weights);
    let logits = runner.decode("s", 3, 0);
    assert_eq!(logits.len(), 16);
    assert!(
        logits.iter().all(|x| x.is_finite()),
        "GQA logits should be finite"
    );
    assert!(
        logits.iter().any(|x| *x != 0.0),
        "GQA logits should be non-zero"
    );
}
