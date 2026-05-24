//! Quantization KL-divergence gate (PLAYBOOK § 3 L2).
//!
//! Compares ferrum's per-token logit distribution between two builds
//! of the **same model in different quantizations** — typically FP16
//! baseline vs GPTQ-INT4 (Marlin path). Drift > 0.01 mean-KL across a
//! 100-prompt × 50-token sample is the gate.
//!
//! llama.cpp's `tools/perplexity/README.md` is explicit that
//! perplexity-vs-FP16 is too coarse to catch real quant regressions
//! (e.g. a Marlin tile change that drops a token in 1% of prompts);
//! KL divergence on the raw logits is the recommended replacement.
//!
//! # Current status: SKELETON
//!
//! This test is structurally in place but waits on:
//!
//! 1. **Logits-capture API** — ferrum's `InferenceResponse.tokens` is
//!    sampled `TokenId`s only. KL needs the full per-step logit vector
//!    (or at least top-k). Adding this is a small change to
//!    `InferenceResponse` + the sampler path, but it's a separate PR
//!    (touches the engine surface).
//! 2. **A paired model variant** — qwen3:0.6b ships FP16; the GPTQ-INT4
//!    sibling isn't in the local cache by default. The test resolves
//!    both via aliases (FP16: `qwen3:0.6b`, INT4: `qwen3:0.6b-gptq-int4`)
//!    and skips with a clear message if either is missing.
//!
//! Once both land, populate `run_pair()` to capture logits and compute
//! the KL.
//!
//! # Tolerance
//!
//! Mean KL across 100×50 token samples should be < 0.01 nats. Empirical
//! baseline on Qwen3-0.6B fp16 vs GPTQ-INT4 in vLLM is ~0.003; ferrum
//! should be in the same band once Marlin path is stable.
//!
//! Run:
//!   cargo test --release -p ferrum-cli --features metal \
//!       --test quant_kl -- --ignored --test-threads=1

#[test]
#[ignore = "requires logits-capture API + paired FP16/INT4 model variants"]
fn marlin_int4_kl_within_tolerance() {
    eprintln!(
        "\n⚠ quant_kl test is a SKELETON — see crates/ferrum-cli/tests/quant_kl.rs \
         module docs.\n\
         Required to enable:\n  \
         1. InferenceResponse expose per-token logits (or top-k)\n  \
         2. Paired FP16/INT4 model variants in HF cache\n"
    );
    // The test framework recognizes this as ignored; remaining body left
    // unimplemented intentionally. Replace with the real assertion once
    // the two prerequisites land:
    //
    //   let fp16 = run_pair("qwen3:0.6b", &prompts).expect("FP16 logits");
    //   let int4 = run_pair("qwen3:0.6b-gptq-int4", &prompts).expect("INT4 logits");
    //   let mean_kl = mean_kl_divergence(&fp16, &int4);
    //   assert!(mean_kl < 0.01, "INT4 drift {} > 0.01 nats", mean_kl);
}
