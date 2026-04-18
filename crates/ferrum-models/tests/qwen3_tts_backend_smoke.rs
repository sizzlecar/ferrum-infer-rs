//! Phase F smoke test: Qwen3-TTS Talker on `Backend<B>`.
//!
//! Verifies that the new `Qwen3TtsTalker<B>` + `Qwen3TtsSubTalker<B>` can
//! load real Qwen3-TTS-0.6B weights via `NativeSafetensorsLoader` and
//! execute a prefill + a handful of decode steps without panicking. Logs
//! the first-codebook argmax token per step so a caller can eyeball
//! whether the CUDA path matches the known-good Metal path.
//!
//! Run on a GPU box:
//!     HF_HOME=/workspace/.hf_home cargo test --release --features cuda \
//!         -p ferrum-models --test qwen3_tts_backend_smoke \
//!         -- --ignored --nocapture
//!
//! Locally on CPU (no CUDA):
//!     HF_HOME=/path/to/cache cargo test --release \
//!         -p ferrum-models --test qwen3_tts_backend_smoke \
//!         -- --ignored --nocapture cpu_smoke

#![allow(clippy::needless_range_loop)]

use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_models::architectures::qwen3_tts::TalkerConfig;
use ferrum_models::architectures::qwen3_tts_backend::{
    Qwen3TtsSubTalker, Qwen3TtsTalker,
};
use ferrum_quantization::NativeSafetensorsLoader;
use std::path::PathBuf;

/// Resolve the cached Qwen3-TTS model dir from $HF_HOME, falling back to
/// `~/.cache/huggingface`. Returns None if the checkpoint snapshot isn't
/// present — tests skip with a helpful message instead of failing.
fn resolve_tts_dir() -> Option<PathBuf> {
    let hf_home = std::env::var("HF_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
            PathBuf::from(home).join(".cache/huggingface")
        });
    let repo = hf_home.join("hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/snapshots");
    let snap_dir = std::fs::read_dir(&repo).ok()?.find_map(|e| {
        let p = e.ok()?.path();
        p.is_dir().then_some(p)
    })?;
    if snap_dir.join("config.json").exists() {
        Some(snap_dir)
    } else {
        None
    }
}

fn load_talker_config(model_dir: &PathBuf) -> TalkerConfig {
    let cfg_path = model_dir.join("config.json");
    let data =
        std::fs::read_to_string(&cfg_path).expect("read config.json");
    let v: serde_json::Value =
        serde_json::from_str(&data).expect("parse config.json");
    TalkerConfig::from_json(&v).expect("TalkerConfig::from_json")
}

/// Shared flow for both CPU and CUDA tests: load weights, run prefill +
/// 4 decode steps, then exercise SubTalker.predict_greedy. Returns
/// `(first_5_codec_tokens, subtalker_tokens)` for caller to compare
/// across backends.
fn run_full_chain<B: ferrum_kernels::backend::Backend>(
    dir: &PathBuf,
    cfg: &TalkerConfig,
    tag: &str,
) -> (Vec<u32>, Vec<u32>) {
    let loader: NativeSafetensorsLoader<B> =
        NativeSafetensorsLoader::open(dir).expect("NativeSafetensorsLoader::open");

    let mut talker =
        Qwen3TtsTalker::<B>::new(cfg.clone(), &loader).expect("Qwen3TtsTalker::new");
    let mut sub =
        Qwen3TtsSubTalker::<B>::new(cfg.clone(), &loader).expect("SubTalker::new");

    let text_tok = |id: u32| (id, true);
    let codec_tok = |id: u32| (id, false);
    let prompt = vec![
        text_tok(9707),
        text_tok(11),
        text_tok(1879),
        codec_tok(cfg.codec_bos_id),
        codec_tok(0),
    ];

    let logits = talker.prefill(tag, &prompt);
    assert_eq!(logits.len(), cfg.vocab_size);
    let (first, val) = logits
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap();
    assert!(val.is_finite());
    eprintln!("{tag} prefill ok: top={first} val={val:.4}");

    let mut codec_tokens = vec![first as u32];
    let mut last = first as u32;
    for step in 0..4 {
        let logits = talker.decode_codec(tag, last);
        let (top, val) = logits
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        assert!(val.is_finite());
        eprintln!("  {tag} decode step={step} top={top} val={val:.4}");
        codec_tokens.push(top as u32);
        last = top as u32;
    }

    // SubTalker predict: takes talker's post-norm hidden + first codec
    // token's embedding. Returns num_code_groups - 1 codec tokens.
    let talker_hidden = talker.last_hidden_normed(tag);
    let first_embed = talker.codec_embed_lookup(codec_tokens[0]);
    let sub_tokens = sub.predict_greedy(&format!("{tag}-sub"), &talker_hidden, &first_embed);
    eprintln!(
        "{tag} subtalker predicted {} tokens; first 5: {:?}",
        sub_tokens.len(),
        &sub_tokens[..sub_tokens.len().min(5)]
    );
    assert_eq!(sub_tokens.len(), cfg.num_code_groups - 1, "sub token count");

    (codec_tokens, sub_tokens)
}

// On macOS the CpuBackend GEMM routes through Accelerate cblas_sgemm;
// on Linux it falls back to a naive triple-loop in `cpu.rs::gemm`.
// Both paths are exercised; Linux surfacing any shape mismatches via
// `assert!` that the macOS path would paper over.
#[test]
#[ignore = "requires Qwen3-TTS weights cached under $HF_HOME"]
fn cpu_smoke() {
    let dir = match resolve_tts_dir() {
        Some(d) => d,
        None => {
            eprintln!(
                "SKIP: Qwen3-TTS weights not found under $HF_HOME — run `ferrum pull qwen3-tts` first"
            );
            return;
        }
    };
    eprintln!("TTS model dir: {dir:?}");

    let cfg = load_talker_config(&dir);
    eprintln!(
        "talker: hidden={} layers={} heads={}/{} vocab={}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.vocab_size,
    );

    let (codec_tokens, sub_tokens) = run_full_chain::<CpuBackend>(&dir, &cfg, "CPU");
    assert!(!codec_tokens.is_empty());
    assert!(!sub_tokens.is_empty());
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires CUDA + Qwen3-TTS weights"]
fn cuda_smoke() {
    use ferrum_kernels::backend::cuda::CudaBackend;

    let dir = match resolve_tts_dir() {
        Some(d) => d,
        None => {
            eprintln!(
                "SKIP: Qwen3-TTS weights not found under $HF_HOME — run `ferrum pull qwen3-tts` first"
            );
            return;
        }
    };
    eprintln!("TTS model dir: {dir:?}");

    let cfg = load_talker_config(&dir);
    let (codec_tokens, sub_tokens) = run_full_chain::<CudaBackend>(&dir, &cfg, "CUDA");
    assert!(!codec_tokens.is_empty());
    assert!(!sub_tokens.is_empty());
}
