//! LlamaFamilyModel self-parity — Cpu↔Cuda on real Qwen3-0.6B weights.
//!
//! Phase E regression guard: CPU is the numerical reference, CUDA must match
//! it on prefill + multi-step decode for the Model-as-Code LLM path.
//!
//! Run: cargo test -p ferrum-models --features cuda --release \
//!          --test qwen3_cuda_parity_test -- --ignored --nocapture
//!
//! Requires: ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/ populated.

#![cfg(feature = "cuda")]

use ferrum_kernels::backend::cpu::CpuBackend;
use ferrum_kernels::backend::cuda::CudaBackend;
use ferrum_models::common::DecoderOnlyLLM;
use ferrum_models::models::llama_family::{LlamaFamilyConfig, LlamaFamilyModel};
use ferrum_quantization::{DenseLinear, Linear, WeightLoader};
use ferrum_types::{FerrumError, Result};

// Same CandleShimLoader used by the Metal parity test — fp32 tensors from
// HF checkpoint via candle VarBuilder, fused into split q/k/v + gate/up.
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
        let direct = format!("{name}.weight");
        if let Some((r, c, data)) = self.fetch_2d(&direct) {
            return Ok(Box::new(DenseLinear::<B>::from_rows(&data, r, c)));
        }
        if name.ends_with("qkv_proj") {
            let prefix = &name[..name.len() - "qkv_proj".len()];
            let keys = vec![
                format!("{prefix}q_proj.weight"),
                format!("{prefix}k_proj.weight"),
                format!("{prefix}v_proj.weight"),
            ];
            if let Some((r, c, data)) = self.fetch_2d_concat(&keys) {
                return Ok(Box::new(DenseLinear::<B>::from_rows(&data, r, c)));
            }
        }
        if name.ends_with("gate_up_proj") {
            let prefix = &name[..name.len() - "gate_up_proj".len()];
            let keys = vec![
                format!("{prefix}gate_proj.weight"),
                format!("{prefix}up_proj.weight"),
            ];
            if let Some((r, c, data)) = self.fetch_2d_concat(&keys) {
                return Ok(Box::new(DenseLinear::<B>::from_rows(&data, r, c)));
            }
        }
        Err(FerrumError::model(format!(
            "could not load linear '{name}'"
        )))
    }

    fn has_tensor(&self, name: &str) -> bool {
        self.vb.get_unchecked(name).is_ok()
    }

    fn quant_config(&self) -> Option<&ferrum_quantization::QuantConfig> {
        None
    }
}

fn model_path(hf_id: &str) -> Option<std::path::PathBuf> {
    // Honour HF_HOME (vast.ai etc often remap it to /workspace or similar) +
    // fallback to the standard ~/.cache/huggingface location.
    let base = std::env::var_os("HF_HOME")
        .map(std::path::PathBuf::from)
        .or_else(|| dirs::home_dir().map(|d| d.join(".cache/huggingface")))?;
    let dir_name = format!("models--{}", hf_id.replace('/', "--"));
    let p = base.join("hub").join(dir_name).join("snapshots");
    std::fs::read_dir(&p).ok()?.find_map(|e| {
        let e = e.ok()?;
        e.path().join("config.json").exists().then(|| e.path())
    })
}

fn qwen3_path() -> Option<std::path::PathBuf> {
    model_path("Qwen/Qwen3-0.6B")
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

/// LlamaFamilyModel<CpuBackend> vs LlamaFamilyModel<CudaBackend>.
///
/// CPU is the reference (fp32 throughout). CUDA runs fp16 — expect minor
/// float drift but argmax must agree and cosine similarity ≥ 0.999.
#[test]
#[ignore]
fn qwen3model_cpu_vs_cuda() {
    let mp = qwen3_path().expect("Qwen3-0.6B not in HF cache");
    let def = load_model_def(&mp);
    let qcfg = LlamaFamilyConfig::qwen3_from_def(&def);

    let loader = ferrum_models::SafeTensorsLoader::new(mp.to_str().unwrap());
    let vb = loader
        .load_varbuilder(&candle_core::Device::Cpu, candle_core::DType::F32)
        .unwrap();

    let cpu_loader = CandleShimLoader::<CpuBackend>::new(&vb);
    let mut cpu_model = LlamaFamilyModel::<CpuBackend>::new(qcfg.clone(), &cpu_loader).unwrap();
    let cuda_loader = CandleShimLoader::<CudaBackend>::new(&vb);
    let mut cuda_model = LlamaFamilyModel::<CudaBackend>::new(qcfg, &cuda_loader).unwrap();

    // Chinese prompt: "你好，欢迎使用"
    let prompt: Vec<u32> = vec![872, 111, 248, 104715, 0, 56568, 53481, 5048];
    eprintln!("\n=== Prefill {} tokens ===", prompt.len());

    let c_logits = cpu_model.prefill("t", &prompt);
    let u_logits = cuda_model.prefill("t", &prompt);
    let (ca, ua) = (argmax(&c_logits), argmax(&u_logits));
    let cos = cosine(&c_logits, &u_logits);
    let mad = max_abs_diff(&c_logits, &u_logits);
    eprintln!(
        "prefill  CPU argmax={ca} ({:.4})  CUDA argmax={ua} ({:.4})  cos={cos:.6}  max_diff={mad:.4}",
        c_logits[ca], u_logits[ua]
    );
    assert_eq!(ca, ua, "prefill argmax mismatch");
    assert!(cos > 0.999, "prefill cosine too low: {cos}");

    let mut pos = prompt.len() as u32;
    let mut tok = ca as u32;
    for step in 0..5 {
        let c = cpu_model.decode("t", tok, pos);
        let u = cuda_model.decode("t", tok, pos);
        let (ca, ua) = (argmax(&c), argmax(&u));
        let cos = cosine(&c, &u);
        let mad = max_abs_diff(&c, &u);
        eprintln!(
            "decode {step} pos={pos} tok={tok}  CPU argmax={ca}  CUDA argmax={ua}  cos={cos:.6}  max_diff={mad:.4}",
        );
        assert_eq!(ca, ua, "decode step {step} argmax mismatch");
        assert!(cos > 0.999, "decode step {step} cosine too low: {cos}");
        tok = ca as u32;
        pos += 1;
    }
    eprintln!("✅ LlamaFamilyModel Cpu↔Cuda parity pass");
}

/// Same parity check on Qwen3-4B. Bigger model stresses more kernel paths
/// (larger hidden_size, different rope dims, more layers).
///
/// Run: cargo test -p ferrum-models --features cuda --release \
///          --test qwen3_cuda_parity_test qwen3_4b -- --ignored --nocapture
#[test]
#[ignore]
fn qwen3_4b_cpu_vs_cuda() {
    let Some(mp) = model_path("Qwen/Qwen3-4B") else {
        eprintln!("SKIP: Qwen3-4B not in HF cache — run `ferrum pull qwen3:4b` first");
        return;
    };
    let def = load_model_def(&mp);
    let qcfg = LlamaFamilyConfig::qwen3_from_def(&def);

    let loader = ferrum_models::SafeTensorsLoader::new(mp.to_str().unwrap());
    let vb = loader
        .load_varbuilder(&candle_core::Device::Cpu, candle_core::DType::F32)
        .unwrap();

    let cpu_loader = CandleShimLoader::<CpuBackend>::new(&vb);
    let mut cpu_model = LlamaFamilyModel::<CpuBackend>::new(qcfg.clone(), &cpu_loader).unwrap();
    let cuda_loader = CandleShimLoader::<CudaBackend>::new(&vb);
    let mut cuda_model = LlamaFamilyModel::<CudaBackend>::new(qcfg, &cuda_loader).unwrap();

    let prompt: Vec<u32> = vec![872, 111, 248, 104715, 0, 56568, 53481, 5048];
    eprintln!("\n=== Qwen3-4B Prefill {} tokens ===", prompt.len());

    let c = cpu_model.prefill("t", &prompt);
    let u = cuda_model.prefill("t", &prompt);
    let (ca, ua) = (argmax(&c), argmax(&u));
    let cos = cosine(&c, &u);
    let mad = max_abs_diff(&c, &u);
    eprintln!(
        "prefill  CPU argmax={ca} ({:.4})  CUDA argmax={ua} ({:.4})  cos={cos:.6}  max_diff={mad:.4}",
        c[ca], u[ua]
    );
    assert_eq!(ca, ua, "4B prefill argmax mismatch");
    assert!(cos > 0.999, "4B prefill cosine too low: {cos}");

    let mut pos = prompt.len() as u32;
    let mut tok = ca as u32;
    for step in 0..5 {
        let c = cpu_model.decode("t", tok, pos);
        let u = cuda_model.decode("t", tok, pos);
        let (ca, ua) = (argmax(&c), argmax(&u));
        let cos = cosine(&c, &u);
        let mad = max_abs_diff(&c, &u);
        eprintln!(
            "decode {step} pos={pos} tok={tok}  CPU argmax={ca}  CUDA argmax={ua}  cos={cos:.6}  max_diff={mad:.4}",
        );
        assert_eq!(ca, ua, "4B decode step {step} argmax mismatch");
        assert!(cos > 0.999, "4B decode step {step} cosine too low: {cos}");
        tok = ca as u32;
        pos += 1;
    }
    eprintln!("✅ Qwen3-4B Cpu↔Cuda parity pass");
}

/// End-to-end text correctness: generate N tokens greedily on both
/// CPU and CUDA with the SAME chat-formatted prompt, decode to string,
/// assert the two strings are identical.
///
/// This catches bugs that argmax-parity doesn't: tokenizer glitches,
/// seq-len boundary issues, wrong KV append offsets after decode N+1,
/// RoPE pos drift, etc.
///
/// Uses the tokenizer from the same HF snapshot as the model.
#[test]
#[ignore]
fn qwen3_generate_text_cpu_vs_cuda() {
    use tokenizers::Tokenizer;

    let mp = qwen3_path().expect("Qwen3-0.6B not in HF cache");
    let def = load_model_def(&mp);
    let qcfg = LlamaFamilyConfig::qwen3_from_def(&def);

    let tok_path = mp.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(&tok_path).expect("tokenizer.json");

    let loader = ferrum_models::SafeTensorsLoader::new(mp.to_str().unwrap());
    let vb = loader
        .load_varbuilder(&candle_core::Device::Cpu, candle_core::DType::F32)
        .unwrap();
    let cpu_loader = CandleShimLoader::<CpuBackend>::new(&vb);
    let mut cpu_model = LlamaFamilyModel::<CpuBackend>::new(qcfg.clone(), &cpu_loader).unwrap();
    let cuda_loader = CandleShimLoader::<CudaBackend>::new(&vb);
    let mut cuda_model = LlamaFamilyModel::<CudaBackend>::new(qcfg, &cuda_loader).unwrap();

    let prompt_text = "The capital of France is";
    let enc = tokenizer.encode(prompt_text, false).expect("encode");
    let prompt: Vec<u32> = enc.get_ids().to_vec();
    eprintln!("\n=== Text generation: '{}' ({} tokens) ===", prompt_text, prompt.len());

    const N_STEPS: usize = 20;

    let mut cpu_out: Vec<u32> = Vec::with_capacity(N_STEPS);
    let mut cuda_out: Vec<u32> = Vec::with_capacity(N_STEPS);

    let first_c = argmax(&cpu_model.prefill("c", &prompt)) as u32;
    let first_u = argmax(&cuda_model.prefill("u", &prompt)) as u32;
    cpu_out.push(first_c);
    cuda_out.push(first_u);

    let mut pos = prompt.len() as u32;
    let (mut tc, mut tu) = (first_c, first_u);
    for _ in 1..N_STEPS {
        tc = argmax(&cpu_model.decode("c", tc, pos)) as u32;
        tu = argmax(&cuda_model.decode("u", tu, pos)) as u32;
        cpu_out.push(tc);
        cuda_out.push(tu);
        pos += 1;
    }

    let cpu_text = tokenizer.decode(&cpu_out, true).expect("decode cpu");
    let cuda_text = tokenizer.decode(&cuda_out, true).expect("decode cuda");
    eprintln!("CPU  → '{}'", cpu_text);
    eprintln!("CUDA → '{}'", cuda_text);
    eprintln!("CPU  ids: {:?}", cpu_out);
    eprintln!("CUDA ids: {:?}", cuda_out);

    assert_eq!(
        cpu_out, cuda_out,
        "generated token sequence diverged — text mismatch"
    );
    eprintln!("✅ End-to-end text parity: {N_STEPS} tokens identical");
}
