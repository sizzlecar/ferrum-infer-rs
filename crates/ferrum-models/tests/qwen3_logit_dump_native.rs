//! Dump ferrum's last-position prefill logits using the PRODUCTION loader path
//! (`NativeSafetensorsLoader` → `LlamaFamilyModel<MetalBackend>`), so they can
//! be compared bitwise-ish against `transformers` reference logits dumped to
//! `/tmp/ferrum-logits-diff/hf_logits.npy`.
//!
//! The existing `qwen3_model_parity_test` uses `CandleShimLoader` (candle's
//! VarBuilder wrapped to expose the `WeightLoader<B>` trait); it does NOT
//! exercise `NativeSafetensorsLoader`, which is what production
//! `ferrum run` / `serve` / `bench` actually use. This test fills that gap.
//!
//! Run:
//!   cargo test -p ferrum-models --test qwen3_logit_dump_native \
//!       --features metal -- --ignored qwen3_4b_prefill_logits_metal --nocapture
//!
//! Writes `/tmp/ferrum-logits-diff/ferrum_logits.bin` as raw little-endian f32
//! of length `vocab_size` (151936 for Qwen3-4B).

#![cfg(all(target_os = "macos", feature = "metal"))]

use ferrum_kernels::backend::metal::MetalBackend;
use ferrum_models::models::{LlamaFamilyConfig, LlamaFamilyModel};
use ferrum_models::DecoderOnlyLLM;
use ferrum_quantization::NativeSafetensorsLoader;
use std::io::Write;
use std::path::PathBuf;

fn qwen3_4b_path() -> Option<PathBuf> {
    let snapshots =
        dirs::home_dir()?.join(".cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots");
    std::fs::read_dir(&snapshots).ok()?.find_map(|e| {
        let p = e.ok()?.path();
        p.join("config.json").exists().then_some(p)
    })
}

fn load_model_def(mp: &std::path::Path) -> ferrum_models::definition::ModelDefinition {
    let cj: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(mp.join("config.json")).unwrap()).unwrap();
    ferrum_models::definition::ModelDefinition {
        architecture: ferrum_models::registry::Architecture::Qwen3,
        hidden_size: cj["hidden_size"].as_u64().unwrap() as usize,
        intermediate_size: cj["intermediate_size"].as_u64().unwrap() as usize,
        vocab_size: cj["vocab_size"].as_u64().unwrap() as usize,
        num_hidden_layers: cj["num_hidden_layers"].as_u64().unwrap() as usize,
        num_attention_heads: cj["num_attention_heads"].as_u64().unwrap() as usize,
        num_key_value_heads: cj
            .get("num_key_value_heads")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize),
        max_position_embeddings: cj["max_position_embeddings"].as_u64().unwrap() as usize,
        rope_theta: cj.get("rope_theta").and_then(|v| v.as_f64()),
        // Qwen3 sets explicit head_dim=128 even though hidden/heads = 64
        extra_params: cj.clone(),
        ..Default::default()
    }
}

#[test]
#[ignore = "needs Qwen3-4B in HF cache + Metal feature; produces /tmp/ferrum-logits-diff/ferrum_logits.bin"]
fn qwen3_4b_prefill_logits_metal() {
    let mp = qwen3_4b_path().expect("Qwen3-4B not in HF cache");
    eprintln!("model path: {mp:?}");

    let def = load_model_def(&mp);
    let cfg: LlamaFamilyConfig = LlamaFamilyConfig::qwen3_from_def(&def);
    eprintln!(
        "cfg: hidden={} heads={} kv={} head_dim={} layers={} vocab={} rope_theta={}",
        cfg.hidden_size,
        cfg.num_heads,
        cfg.num_kv_heads,
        cfg.head_dim,
        cfg.num_layers,
        cfg.vocab_size,
        cfg.rope_theta,
    );
    let vocab_size = cfg.vocab_size;

    let t_load = std::time::Instant::now();
    let loader: NativeSafetensorsLoader<MetalBackend> =
        NativeSafetensorsLoader::open(&mp).expect("NativeSafetensorsLoader::open");
    let mut model =
        LlamaFamilyModel::<MetalBackend>::new(cfg, &loader).expect("LlamaFamilyModel::new");
    eprintln!("model loaded in {:.2}s", t_load.elapsed().as_secs_f32());

    // Same 13-token chat-template prompt as dump_hf.py (verified upstream).
    let prompt_ids: Vec<u32> = vec![
        151644, 872, 198, 108386, 151645, 198, 151644, 77091, 198, 151667, 271, 151668, 271,
    ];
    eprintln!("prompt ids: {prompt_ids:?}");

    let out_dir = std::path::PathBuf::from("/tmp/ferrum-logits-diff");
    std::fs::create_dir_all(&out_dir).expect("mkdir");

    let dump_logits = |path: &std::path::Path, logits: &[f32]| {
        let mut buf: Vec<u8> = Vec::with_capacity(logits.len() * 4);
        for v in logits {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        std::fs::File::create(path)
            .and_then(|mut f| f.write_all(&buf))
            .expect("dump logits");
    };

    let top1 = |logits: &[f32]| -> (usize, f32) {
        logits
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap()
    };

    // ── prefill ──
    let t_pf = std::time::Instant::now();
    let logits0 = model.prefill("dump", &prompt_ids);
    eprintln!(
        "prefill returned {} logits in {:.3}s",
        logits0.len(),
        t_pf.elapsed().as_secs_f32()
    );
    assert_eq!(logits0.len(), vocab_size, "logit length mismatch");
    assert!(logits0.iter().all(|v| v.is_finite()), "non-finite logit");
    let (id0, lg0) = top1(&logits0);
    eprintln!("decode_0 (prefill last-pos): top1 id={id0} logit={lg0:.6}");
    dump_logits(&out_dir.join("ferrum_decode_0.bin"), &logits0);

    // ── 5 decode steps via `decode_batch` (engine production path on Metal) ──
    // The engine's `LlmExecutor::unified_decode` fallback (triggered on Metal
    // because `supports_varlen_qkv=false`) routes ALL decode items through
    // `model.decode_batch`, not `model.decode`. We exercise the same path to
    // catch any divergence vs the single-item `model.decode` route.
    let mut last_id: u32 = id0 as u32;
    let mut pos: u32 = prompt_ids.len() as u32;
    for step in 1..=5u32 {
        let t = std::time::Instant::now();
        let batch = vec![("dump".to_string(), last_id, pos)];
        let mut logits_vec = model.decode_batch(&batch);
        let logits = logits_vec.pop().expect("decode_batch result");
        let dt = t.elapsed().as_secs_f32();
        assert_eq!(logits.len(), vocab_size);
        assert!(logits.iter().all(|v| v.is_finite()), "non-finite logit");
        let (id, lg) = top1(&logits);
        eprintln!(
            "decode_batch_{step}: fed token={last_id} at pos={pos}, top1 id={id} logit={lg:.6} ({dt:.3}s)"
        );
        dump_logits(
            &out_dir.join(format!("ferrum_decode_batch_{step}.bin")),
            &logits,
        );
        last_id = id as u32;
        pos += 1;
    }
    eprintln!("\ndump dir: {out_dir:?}");
}
