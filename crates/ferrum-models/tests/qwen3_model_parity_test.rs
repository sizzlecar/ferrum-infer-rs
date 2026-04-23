//! LlamaFamilyModel self-parity — Cpu↔Metal on real Qwen3-0.6B weights.
//!
//! Verifies that both backend specialisations of LlamaFamilyModel produce identical
//! logits for prefill + multi-step decode on the Model-as-Code path.
//!
//! Run: cargo test -p ferrum-models --features metal --release \
//!          --test qwen3_model_parity_test -- --ignored --nocapture

#![cfg(all(target_os = "macos", feature = "metal"))]

use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_kernels::backend::metal::MetalBackend;
use ferrum_models::common::DecoderOnlyLLM;
use ferrum_models::models::llama_family::{LlamaFamilyConfig, LlamaFamilyModel};
use ferrum_quantization::{DenseLinear, Linear, WeightLoader};
use ferrum_types::{FerrumError, Result};

// A minimal WeightLoader<B> that wraps the existing candle VarBuilder-backed
// weights we already pull in for the legacy ModelRunner tests. This is a
// shim for B4 parity only — Phase B2 SafeTensorsLoader impl will replace it.
struct CandleShimLoader<'a, B: ferrum_kernels::backend::Backend> {
    vb: &'a candle_nn::VarBuilder<'a>,
    _m: std::marker::PhantomData<B>,
}

impl<'a, B: ferrum_kernels::backend::Backend> CandleShimLoader<'a, B> {
    fn new(vb: &'a candle_nn::VarBuilder<'a>) -> Self {
        Self {
            vb,
            _m: std::marker::PhantomData,
        }
    }

    fn load_f32_rows(&self, name: &str) -> Result<Vec<f32>> {
        let t = self
            .vb
            .get_unchecked(name)
            .map_err(|e| FerrumError::model(format!("weight '{name}': {e}")))?;
        let t = t
            .to_dtype(candle_core::DType::F32)
            .map_err(|e| FerrumError::model(format!("to_f32: {e}")))?;
        let t = t
            .flatten_all()
            .map_err(|e| FerrumError::model(format!("flatten: {e}")))?;
        t.to_vec1::<f32>()
            .map_err(|e| FerrumError::model(format!("to_vec: {e}")))
    }
}

impl<'a, B: ferrum_kernels::backend::Backend> CandleShimLoader<'a, B> {
    /// Fetch tensor shape + f32 data. Accepts either a single fused tensor or
    /// a list of component names to concat along dim 0 (for Qwen3's separate
    /// q_proj/k_proj/v_proj and gate_proj/up_proj).
    fn fetch_2d(&self, name_key: &str) -> Option<(usize, usize, Vec<f32>)> {
        let t = self.vb.get_unchecked(name_key).ok()?;
        let shape = t.shape().dims().to_vec();
        if shape.len() != 2 {
            return None;
        }
        let (rows, cols) = (shape[0], shape[1]);
        let data = self.load_f32_rows(name_key).ok()?;
        Some((rows, cols, data))
    }

    fn fetch_2d_concat(&self, keys: &[String]) -> Option<(usize, usize, Vec<f32>)> {
        let mut total_rows = 0usize;
        let mut cols = 0usize;
        let mut acc: Vec<f32> = Vec::new();
        for k in keys {
            let (r, c, data) = self.fetch_2d(k)?;
            if cols == 0 {
                cols = c;
            } else if cols != c {
                return None;
            }
            total_rows += r;
            acc.extend_from_slice(&data);
        }
        Some((total_rows, cols, acc))
    }
}

impl<'a, B: ferrum_kernels::backend::Backend> WeightLoader<B> for CandleShimLoader<'a, B> {
    fn load_tensor(&self, name: &str) -> Result<B::Buffer> {
        let data = self.load_f32_rows(name)?;
        Ok(B::from_slice(&data))
    }

    fn load_linear(&self, name: &str) -> Result<Box<dyn Linear<B>>> {
        // Try direct fused `<name>.weight` first.
        let direct = format!("{name}.weight");
        if let Some((r, c, data)) = self.fetch_2d(&direct) {
            return Ok(Box::new(DenseLinear::<B>::from_rows(&data, r, c)));
        }

        // Fallback: Qwen3 0.6B (and Llama family) checkpoints store projections
        // split as q/k/v and gate/up — fuse them by concatenating on dim 0.
        if let Some(prefix) = name.strip_suffix("qkv_proj") {
            let keys = vec![
                format!("{prefix}q_proj.weight"),
                format!("{prefix}k_proj.weight"),
                format!("{prefix}v_proj.weight"),
            ];
            if let Some((r, c, data)) = self.fetch_2d_concat(&keys) {
                return Ok(Box::new(DenseLinear::<B>::from_rows(&data, r, c)));
            }
        }
        if let Some(prefix) = name.strip_suffix("gate_up_proj") {
            let keys = vec![
                format!("{prefix}gate_proj.weight"),
                format!("{prefix}up_proj.weight"),
            ];
            if let Some((r, c, data)) = self.fetch_2d_concat(&keys) {
                return Ok(Box::new(DenseLinear::<B>::from_rows(&data, r, c)));
            }
        }

        Err(FerrumError::model(format!(
            "could not load linear '{name}' — no direct weight, no split components"
        )))
    }

    fn has_tensor(&self, name: &str) -> bool {
        self.vb.get_unchecked(name).is_ok()
    }

    fn quant_config(&self) -> Option<&ferrum_quantization::QuantConfig> {
        None
    }
}

fn qwen3_path() -> Option<std::path::PathBuf> {
    let p = dirs::home_dir()?.join(".cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots");
    std::fs::read_dir(&p).ok()?.find_map(|e| {
        let e = e.ok()?;
        e.path().join("config.json").exists().then(|| e.path())
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
        // Keep the raw config.json around so downstream cfg builders can read
        // explicit `head_dim` (Qwen3 sets it to 128 even though hidden/heads=64).
        extra_params: cj.clone(),
        ..Default::default()
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b) {
        dot += *x as f64 * *y as f64;
        na += (*x as f64).powi(2);
        nb += (*y as f64).powi(2);
    }
    dot / (na.sqrt() * nb.sqrt() + 1e-10)
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max)
}

fn def_to_qwen3_cfg(def: &ferrum_models::definition::ModelDefinition) -> LlamaFamilyConfig {
    LlamaFamilyConfig::qwen3_from_def(def)
}

/// LlamaFamilyModel<CpuBackend> vs LlamaFamilyModel<MetalBackend> on the same weights.
///
/// This is the post-Phase-B regression guard for the Model-as-Code path:
/// if either backend's LlamaFamilyModel-facing Backend methods (qk_norm_rope,
/// kv_cache_append_head_major, etc.) diverge, this catches it immediately.
#[test]
#[ignore]
fn qwen3model_cpu_vs_metal() {
    let mp = qwen3_path().expect("Qwen3-0.6B not in HF cache");
    let def = load_model_def(&mp);
    let qcfg = def_to_qwen3_cfg(&def);

    let loader = ferrum_models::SafeTensorsLoader::new(mp.to_str().unwrap());
    let vb = loader
        .load_varbuilder(&candle_core::Device::Cpu, candle_core::DType::F32)
        .unwrap();

    let cpu_loader = CandleShimLoader::<CpuBackend>::new(&vb);
    let mut cpu_model = LlamaFamilyModel::<CpuBackend>::new(qcfg.clone(), &cpu_loader).unwrap();
    let mtl_loader = CandleShimLoader::<MetalBackend>::new(&vb);
    let mut mtl_model = LlamaFamilyModel::<MetalBackend>::new(qcfg, &mtl_loader).unwrap();

    let prompt: Vec<u32> = vec![872, 111, 248, 104715, 0, 56568, 53481, 5048];
    eprintln!("\n=== Prefill {} tokens ===", prompt.len());

    let c_logits = cpu_model.prefill("t", &prompt);
    let m_logits = mtl_model.prefill("t", &prompt);
    let (ca, ma) = (argmax(&c_logits), argmax(&m_logits));
    let cos = cosine(&c_logits, &m_logits);
    let mad = max_abs_diff(&c_logits, &m_logits);
    eprintln!(
        "prefill  CPU argmax={ca} ({:.4})  Metal argmax={ma} ({:.4})  cos={cos:.6}  max_diff={mad:.4}",
        c_logits[ca], m_logits[ma]
    );
    assert_eq!(ca, ma, "prefill argmax mismatch");
    assert!(cos > 0.9999, "prefill cosine too low: {cos}");

    let mut pos = prompt.len() as u32;
    let mut tok = ca as u32;
    for step in 0..5 {
        let c = cpu_model.decode("t", tok, pos);
        let m = mtl_model.decode("t", tok, pos);
        let (ca, ma) = (argmax(&c), argmax(&m));
        let cos = cosine(&c, &m);
        let mad = max_abs_diff(&c, &m);
        eprintln!(
            "decode {step} pos={pos} tok={tok}  CPU argmax={ca}  Metal argmax={ma}  cos={cos:.6}  max_diff={mad:.4}",
        );
        assert_eq!(ca, ma, "decode step {step} argmax mismatch");
        assert!(cos > 0.9999, "decode step {step} cosine too low: {cos}");
        tok = ca as u32;
        pos += 1;
    }
    eprintln!("✅ LlamaFamilyModel Cpu↔Metal parity pass");
}
