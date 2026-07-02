//! W3-S1 Qwen3.5 first-layer CPU replay harness.
//!
//! This module is diagnostic evidence plumbing, not the product model path.  It
//! reads real HF safetensors weights and a matching HF layer dump, replays the
//! first Qwen3.5 linear-attention decoder layer with Ferrum-owned Rust code, and
//! writes a dump that can be compared tensor-by-tensor against the HF reference.

use std::collections::HashMap;
use std::fs;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use serde_json::{json, Value};

use crate::qwen35_config::Qwen35TextConfig;

pub const FERRUM_MANIFEST_NAME: &str = "w3_qwen35_ferrum_layer_dump_manifest.json";

pub const QWEN35_LAYER_TENSORS: &[&str] = &[
    "layer_input",
    "input_norm",
    "mixed_qkv_raw",
    "z_raw",
    "b_raw",
    "a_raw",
    "mixed_qkv_conv",
    "delta_q",
    "delta_k",
    "delta_v",
    "delta_beta",
    "delta_g",
    "delta_core",
    "delta_norm",
    "delta_output",
    "residual_after_mixer",
    "post_attention_norm",
    "mlp_output",
    "layer_output",
];

#[derive(Debug, Clone)]
pub struct Qwen35S1Dump {
    pub tensors: HashMap<String, Vec<f32>>,
    pub tensor_shapes: HashMap<String, Vec<usize>>,
    pub input_ids: Vec<usize>,
    pub model_id: Option<String>,
    pub layer_idx: usize,
}

#[derive(Debug, Clone)]
struct TensorMeta {
    dtype: Dtype,
    shape: Vec<usize>,
    data_start: usize,
    data_end: usize,
}

struct Shard {
    mmap: Mmap,
    meta: HashMap<String, TensorMeta>,
}

impl Shard {
    fn open(path: &Path) -> Result<Self, String> {
        let file = File::open(path).map_err(|err| format!("open {path:?}: {err}"))?;
        let mmap = unsafe { Mmap::map(&file).map_err(|err| format!("mmap {path:?}: {err}"))? };
        let st = SafeTensors::deserialize(&mmap).map_err(|err| format!("parse {path:?}: {err}"))?;
        let mut meta = HashMap::new();
        for name in st.names() {
            let view = st
                .tensor(name)
                .map_err(|err| format!("tensor {name:?} in {path:?}: {err}"))?;
            let start = view.data().as_ptr() as usize - mmap.as_ptr() as usize;
            let end = start + view.data().len();
            meta.insert(
                name.to_string(),
                TensorMeta {
                    dtype: view.dtype(),
                    shape: view.shape().to_vec(),
                    data_start: start,
                    data_end: end,
                },
            );
        }
        Ok(Self { mmap, meta })
    }

    fn contains(&self, name: &str) -> bool {
        self.meta.contains_key(name)
    }

    fn read_f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
        let meta = self
            .meta
            .get(name)
            .ok_or_else(|| format!("tensor {name:?} not found"))?;
        let raw = &self.mmap[meta.data_start..meta.data_end];
        Ok((dtype_to_f32(meta.dtype, raw)?, meta.shape.clone()))
    }
}

struct TensorStore {
    shards: Vec<Shard>,
}

impl TensorStore {
    fn open(model_dir: &Path) -> Result<Self, String> {
        let mut shard_paths = discover_safetensors(model_dir)?;
        shard_paths.sort();
        let shards = shard_paths
            .iter()
            .map(|path| Shard::open(path))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { shards })
    }

    fn contains(&self, name: &str) -> bool {
        self.shards.iter().any(|shard| shard.contains(name))
    }

    fn read_f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>), String> {
        for shard in &self.shards {
            if shard.contains(name) {
                return shard.read_f32(name);
            }
        }
        Err(format!(
            "tensor {name:?} not found in any safetensors shard"
        ))
    }
}

fn discover_safetensors(model_dir: &Path) -> Result<Vec<PathBuf>, String> {
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }
    let index = model_dir.join("model.safetensors.index.json");
    if index.exists() {
        let raw = fs::read_to_string(&index).map_err(|err| format!("read {index:?}: {err}"))?;
        let value: Value =
            serde_json::from_str(&raw).map_err(|err| format!("parse {index:?}: {err}"))?;
        let map = value
            .get("weight_map")
            .and_then(Value::as_object)
            .ok_or_else(|| format!("{index:?} missing weight_map"))?;
        let mut files = map
            .values()
            .filter_map(Value::as_str)
            .map(|name| model_dir.join(name))
            .collect::<Vec<_>>();
        files.sort();
        files.dedup();
        return Ok(files);
    }
    let mut files = fs::read_dir(model_dir)
        .map_err(|err| format!("read_dir {model_dir:?}: {err}"))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "safetensors"))
        .collect::<Vec<_>>();
    files.sort();
    if files.is_empty() {
        Err(format!("no safetensors files found in {model_dir:?}"))
    } else {
        Ok(files)
    }
}

fn dtype_to_f32(dtype: Dtype, raw: &[u8]) -> Result<Vec<f32>, String> {
    match dtype {
        Dtype::F32 => {
            if raw.len() % 4 != 0 {
                return Err("f32 tensor byte length is not divisible by 4".to_string());
            }
            Ok(raw
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().unwrap()))
                .collect())
        }
        Dtype::F16 => {
            if raw.len() % 2 != 0 {
                return Err("f16 tensor byte length is not divisible by 2".to_string());
            }
            Ok(raw
                .chunks_exact(2)
                .map(|bytes| f16::from_le_bytes(bytes.try_into().unwrap()).to_f32())
                .collect())
        }
        Dtype::BF16 => {
            if raw.len() % 2 != 0 {
                return Err("bf16 tensor byte length is not divisible by 2".to_string());
            }
            Ok(raw
                .chunks_exact(2)
                .map(|bytes| bf16::from_le_bytes(bytes.try_into().unwrap()).to_f32())
                .collect())
        }
        other => Err(format!("unsupported safetensors dtype {other:?}")),
    }
}

pub fn write_qwen35_s1_dump(
    model_dir: &Path,
    hf_dump_dir: &Path,
    out_dir: &Path,
    producer: &str,
) -> Result<(), String> {
    fs::create_dir_all(out_dir).map_err(|err| format!("create {out_dir:?}: {err}"))?;
    let tensors_dir = out_dir.join("tensors");
    fs::create_dir_all(&tensors_dir).map_err(|err| format!("create {tensors_dir:?}: {err}"))?;
    let dump = compute_qwen35_s1_dump(model_dir, hf_dump_dir)?;
    for name in QWEN35_LAYER_TENSORS {
        let values = dump
            .tensors
            .get(*name)
            .ok_or_else(|| format!("computed dump missing tensor {name}"))?;
        write_f32(&tensors_dir.join(format!("{name}.bin")), values)?;
    }
    let tensor_manifest = QWEN35_LAYER_TENSORS
        .iter()
        .map(|name| {
            let shape = dump
                .tensor_shapes
                .get(*name)
                .cloned()
                .unwrap_or_else(Vec::new);
            (
                name.to_string(),
                json!({
                    "file": format!("{name}.bin"),
                    "shape": shape,
                    "dtype": "float32",
                    "numel": dump.tensors.get(*name).map_or(0, Vec::len),
                }),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    let pass_line = format!("W3 QWEN35 FERRUM LAYER DUMP PASS: {}", out_dir.display());
    write_json(
        &out_dir.join(FERRUM_MANIFEST_NAME),
        &json!({
            "schema_version": 1,
            "status": "pass",
            "mode": "ferrum-qwen35-s1-replay",
            "producer": producer,
            "pass_line": pass_line,
            "command_line": std::env::args().collect::<Vec<_>>(),
            "git": git_summary(),
            "model_id": dump.model_id,
            "selected_layer_idx": dump.layer_idx,
            "selected_layer_type": "linear_attention",
            "prompt_input_ids": [dump.input_ids],
            "hf_reference_dump": hf_dump_dir.display().to_string(),
            "model_dir": model_dir.display().to_string(),
            "tensor_dir": tensors_dir.display().to_string(),
            "tensors": tensor_manifest,
            "note": "Ferrum-owned CPU replay of Qwen3.5 first linear-attention layer from HF safetensors weights",
        }),
    )?;
    println!("{pass_line}");
    Ok(())
}

pub fn compute_qwen35_s1_dump(
    model_dir: &Path,
    hf_dump_dir: &Path,
) -> Result<Qwen35S1Dump, String> {
    let config_raw = fs::read_to_string(model_dir.join("config.json"))
        .map_err(|err| format!("read config.json: {err}"))?;
    let cfg = Qwen35TextConfig::from_hf_config_str(&config_raw)?;
    if cfg.is_moe() {
        return Err("Qwen3.5 dense replay expected qwen3_5_text, got MoE".to_string());
    }
    let raw_cfg: Value =
        serde_json::from_str(&config_raw).map_err(|err| format!("parse config.json: {err}"))?;
    let text_cfg = raw_cfg.get("text_config").unwrap_or(&raw_cfg);
    let eps = text_cfg
        .get("rms_norm_eps")
        .and_then(Value::as_f64)
        .unwrap_or(1e-6) as f32;
    let hidden_act = text_cfg
        .get("hidden_act")
        .and_then(Value::as_str)
        .unwrap_or("silu");
    if hidden_act != "silu" {
        return Err(format!("unsupported hidden_act {hidden_act:?}"));
    }
    let intermediate = cfg
        .dense_intermediate_size
        .ok_or_else(|| "dense Qwen3.5 config missing intermediate_size".to_string())?;
    let layer_idx = read_hf_layer_idx(hf_dump_dir)?;
    if cfg.first_linear_attention_layer() != Some(layer_idx) {
        return Err(format!(
            "HF dump selected layer {layer_idx}, config first linear layer {:?}",
            cfg.first_linear_attention_layer()
        ));
    }
    let input_ids = read_hf_input_ids(hf_dump_dir)?;
    let tokens = input_ids.len();
    let hidden = cfg.hidden_size;
    let heads = cfg.linear_attention.num_value_heads;
    let key_heads = cfg.linear_attention.num_key_heads;
    let key_dim = cfg.linear_attention.key_head_dim;
    let value_dim = cfg.linear_attention.value_head_dim;
    if heads != key_heads {
        return Err(format!(
            "first replay only supports equal key/value heads, got key={key_heads} value={heads}"
        ));
    }
    let qk_total = key_heads * key_dim;
    let value_total = heads * value_dim;
    let conv_dim = qk_total * 2 + value_total;
    if qk_total != value_total {
        return Err(format!(
            "first replay expects q/k/v total dims to match, got qk={qk_total} v={value_total}"
        ));
    }

    let store = TensorStore::open(model_dir)?;
    let prefix = detect_prefix(&store, layer_idx)?;
    let layer_prefix = format!("{prefix}.layers.{layer_idx}");
    let embed = read_checked(
        &store,
        &format!("{prefix}.embed_tokens.weight"),
        &[usize::MAX, hidden],
    )?;
    let input = gather_embeddings(&embed.0, embed.1[1], &input_ids)?;
    let input_norm_weight = read_checked(
        &store,
        &format!("{layer_prefix}.input_layernorm.weight"),
        &[hidden],
    )?
    .0;
    let input_norm = rms_norm_plus_one(&input, &input_norm_weight, tokens, hidden, eps);

    let mixed_qkv_raw = linear(
        &input_norm,
        &read_checked(
            &store,
            &format!("{layer_prefix}.linear_attn.in_proj_qkv.weight"),
            &[conv_dim, hidden],
        )?
        .0,
        tokens,
        hidden,
        conv_dim,
    );
    let z_raw = linear(
        &input_norm,
        &read_checked(
            &store,
            &format!("{layer_prefix}.linear_attn.in_proj_z.weight"),
            &[value_total, hidden],
        )?
        .0,
        tokens,
        hidden,
        value_total,
    );
    let b_raw = linear(
        &input_norm,
        &read_checked(
            &store,
            &format!("{layer_prefix}.linear_attn.in_proj_b.weight"),
            &[heads, hidden],
        )?
        .0,
        tokens,
        hidden,
        heads,
    );
    let a_raw = linear(
        &input_norm,
        &read_checked(
            &store,
            &format!("{layer_prefix}.linear_attn.in_proj_a.weight"),
            &[heads, hidden],
        )?
        .0,
        tokens,
        hidden,
        heads,
    );
    let conv_weight = read_checked(
        &store,
        &format!("{layer_prefix}.linear_attn.conv1d.weight"),
        &[conv_dim, 1, cfg.linear_attention.conv_kernel_dim],
    )?
    .0;
    let mixed_qkv_conv = depthwise_causal_conv_silu(
        &mixed_qkv_raw,
        &conv_weight,
        tokens,
        conv_dim,
        cfg.linear_attention.conv_kernel_dim,
    );
    let delta_q = split_features(&mixed_qkv_conv, tokens, conv_dim, 0, qk_total);
    let delta_k = split_features(&mixed_qkv_conv, tokens, conv_dim, qk_total, qk_total);
    let delta_v = split_features(&mixed_qkv_conv, tokens, conv_dim, qk_total * 2, value_total);
    let delta_beta = b_raw
        .iter()
        .map(|value| sigmoid(*value))
        .collect::<Vec<_>>();
    let a_log = read_checked(
        &store,
        &format!("{layer_prefix}.linear_attn.A_log"),
        &[heads],
    )?
    .0;
    let dt_bias = read_checked(
        &store,
        &format!("{layer_prefix}.linear_attn.dt_bias"),
        &[heads],
    )?
    .0;
    let delta_g = compute_g(&a_raw, &a_log, &dt_bias, tokens, heads);
    let delta_core = chunk_gated_delta_rule_single_chunk(
        &delta_q,
        &delta_k,
        &delta_v,
        &delta_g,
        &delta_beta,
        tokens,
        heads,
        key_dim,
        value_dim,
    )?;
    let norm_weight = read_checked(
        &store,
        &format!("{layer_prefix}.linear_attn.norm.weight"),
        &[value_dim],
    )?
    .0;
    let delta_norm = rms_norm_gated(
        &delta_core,
        &z_raw,
        &norm_weight,
        tokens,
        heads,
        value_dim,
        eps,
    );
    let delta_output = linear(
        &delta_norm,
        &read_checked(
            &store,
            &format!("{layer_prefix}.linear_attn.out_proj.weight"),
            &[hidden, value_total],
        )?
        .0,
        tokens,
        value_total,
        hidden,
    );
    let residual_after_mixer = add(&input, &delta_output)?;
    let post_norm_weight = read_checked(
        &store,
        &format!("{layer_prefix}.post_attention_layernorm.weight"),
        &[hidden],
    )?
    .0;
    let post_attention_norm = rms_norm_plus_one(
        &residual_after_mixer,
        &post_norm_weight,
        tokens,
        hidden,
        eps,
    );
    let gate = linear(
        &post_attention_norm,
        &read_checked(
            &store,
            &format!("{layer_prefix}.mlp.gate_proj.weight"),
            &[intermediate, hidden],
        )?
        .0,
        tokens,
        hidden,
        intermediate,
    );
    let up = linear(
        &post_attention_norm,
        &read_checked(
            &store,
            &format!("{layer_prefix}.mlp.up_proj.weight"),
            &[intermediate, hidden],
        )?
        .0,
        tokens,
        hidden,
        intermediate,
    );
    let fused = gate
        .iter()
        .zip(&up)
        .map(|(gate, up)| silu(*gate) * up)
        .collect::<Vec<_>>();
    let mlp_output = linear(
        &fused,
        &read_checked(
            &store,
            &format!("{layer_prefix}.mlp.down_proj.weight"),
            &[hidden, intermediate],
        )?
        .0,
        tokens,
        intermediate,
        hidden,
    );
    let layer_output = add(&residual_after_mixer, &mlp_output)?;

    let mut tensors = HashMap::new();
    let mut shapes = HashMap::new();
    insert_tensor(
        &mut tensors,
        &mut shapes,
        "layer_input",
        input,
        &[1, tokens, hidden],
    );
    insert_tensor(
        &mut tensors,
        &mut shapes,
        "input_norm",
        input_norm,
        &[1, tokens, hidden],
    );
    insert_tensor(
        &mut tensors,
        &mut shapes,
        "mixed_qkv_raw",
        mixed_qkv_raw,
        &[1, tokens, conv_dim],
    );
    insert_tensor(
        &mut tensors,
        &mut shapes,
        "z_raw",
        z_raw,
        &[1, tokens, value_total],
    );
    insert_tensor(
        &mut tensors,
        &mut shapes,
        "b_raw",
        b_raw,
        &[1, tokens, heads],
    );
    insert_tensor(
        &mut tensors,
        &mut shapes,
        "a_raw",
        a_raw,
        &[1, tokens, heads],
    );
    insert_tensor(
        &mut tensors,
        &mut shapes,
        "mixed_qkv_conv",
        mixed_qkv_conv,
        &[1, tokens, conv_dim],
    );
    insert_tensor(
        &mut tensors,
        &mut shapes,
        "delta_q",
        delta_q,
        &[1, tokens, heads, key_dim],
    );
    insert_tensor(
        &mut tensors,
        &mut shapes,
        "delta_k",
        delta_k,
        &[1, tokens, heads, key_dim],
    );
    insert_tensor(
        &mut tensors,
        &mut shapes,
        "delta_v",
        delta_v,
        &[1, tokens, heads, value_dim],
    );
    insert_tensor(
        &mut tensors,
        &mut shapes,
        "delta_beta",
        delta_beta,
        &[1, tokens, heads],
    );
    insert_tensor(
        &mut tensors,
        &mut shapes,
        "delta_g",
        delta_g,
        &[1, tokens, heads],
    );
    insert_tensor(
        &mut tensors,
        &mut shapes,
        "delta_core",
        delta_core,
        &[1, tokens, heads, value_dim],
    );
    insert_tensor(
        &mut tensors,
        &mut shapes,
        "delta_norm",
        delta_norm,
        &[tokens * heads, value_dim],
    );
    insert_tensor(
        &mut tensors,
        &mut shapes,
        "delta_output",
        delta_output,
        &[1, tokens, hidden],
    );
    insert_tensor(
        &mut tensors,
        &mut shapes,
        "residual_after_mixer",
        residual_after_mixer,
        &[1, tokens, hidden],
    );
    insert_tensor(
        &mut tensors,
        &mut shapes,
        "post_attention_norm",
        post_attention_norm,
        &[1, tokens, hidden],
    );
    insert_tensor(
        &mut tensors,
        &mut shapes,
        "mlp_output",
        mlp_output,
        &[1, tokens, hidden],
    );
    insert_tensor(
        &mut tensors,
        &mut shapes,
        "layer_output",
        layer_output,
        &[1, tokens, hidden],
    );

    Ok(Qwen35S1Dump {
        tensors,
        tensor_shapes: shapes,
        input_ids,
        model_id: read_hf_model_id(hf_dump_dir),
        layer_idx,
    })
}

fn detect_prefix(store: &TensorStore, layer_idx: usize) -> Result<String, String> {
    for prefix in ["model.language_model", "model"] {
        if store.contains(&format!(
            "{prefix}.layers.{layer_idx}.linear_attn.in_proj_qkv.weight"
        )) {
            return Ok(prefix.to_string());
        }
    }
    Err(format!(
        "could not find Qwen3.5 layer {layer_idx} weight prefix"
    ))
}

fn read_checked(
    store: &TensorStore,
    name: &str,
    expected_shape: &[usize],
) -> Result<(Vec<f32>, Vec<usize>), String> {
    let (data, shape) = store.read_f32(name)?;
    if shape.len() != expected_shape.len() {
        return Err(format!(
            "{name} shape {shape:?} != expected {expected_shape:?}"
        ));
    }
    for (actual, expected) in shape.iter().zip(expected_shape) {
        if *expected != usize::MAX && actual != expected {
            return Err(format!(
                "{name} shape {shape:?} != expected {expected_shape:?}"
            ));
        }
    }
    Ok((data, shape))
}

fn insert_tensor(
    tensors: &mut HashMap<String, Vec<f32>>,
    shapes: &mut HashMap<String, Vec<usize>>,
    name: &str,
    values: Vec<f32>,
    shape: &[usize],
) {
    tensors.insert(name.to_string(), values);
    shapes.insert(name.to_string(), shape.to_vec());
}

fn read_hf_manifest(hf_dump_dir: &Path) -> Result<Value, String> {
    let manifest = hf_dump_dir.join("w3_qwen35_hf_layer_dump_manifest.json");
    let raw = fs::read_to_string(&manifest).map_err(|err| format!("read {manifest:?}: {err}"))?;
    serde_json::from_str(&raw).map_err(|err| format!("parse {manifest:?}: {err}"))
}

fn read_hf_layer_idx(hf_dump_dir: &Path) -> Result<usize, String> {
    read_hf_manifest(hf_dump_dir)?
        .get("selected_layer_idx")
        .and_then(Value::as_u64)
        .map(|value| value as usize)
        .ok_or_else(|| "HF manifest missing selected_layer_idx".to_string())
}

fn read_hf_model_id(hf_dump_dir: &Path) -> Option<String> {
    read_hf_manifest(hf_dump_dir).ok().and_then(|value| {
        value
            .get("model_id")
            .and_then(Value::as_str)
            .map(str::to_string)
    })
}

fn read_hf_input_ids(hf_dump_dir: &Path) -> Result<Vec<usize>, String> {
    let value = read_hf_manifest(hf_dump_dir)?;
    let rows = value
        .get("prompt_input_ids")
        .and_then(Value::as_array)
        .ok_or_else(|| "HF manifest missing prompt_input_ids".to_string())?;
    if rows.len() != 1 {
        return Err(format!(
            "expected one prompt_input_ids row, got {}",
            rows.len()
        ));
    }
    rows[0]
        .as_array()
        .ok_or_else(|| "prompt_input_ids[0] must be an array".to_string())?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .map(|value| value as usize)
                .ok_or_else(|| format!("invalid token id {value:?}"))
        })
        .collect()
}

fn gather_embeddings(
    weight: &[f32],
    hidden: usize,
    input_ids: &[usize],
) -> Result<Vec<f32>, String> {
    let vocab = weight.len() / hidden;
    let mut out = Vec::with_capacity(input_ids.len() * hidden);
    for &token in input_ids {
        if token >= vocab {
            return Err(format!("token id {token} exceeds vocab {vocab}"));
        }
        out.extend_from_slice(&weight[token * hidden..(token + 1) * hidden]);
    }
    Ok(out)
}

fn rms_norm_plus_one(x: &[f32], weight: &[f32], rows: usize, dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0; rows * dim];
    for row in 0..rows {
        let base = row * dim;
        let mean = x[base..base + dim]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            / dim as f32;
        let inv = (mean + eps).sqrt().recip();
        for i in 0..dim {
            out[base + i] = x[base + i] * inv * (1.0 + weight[i]);
        }
    }
    out
}

fn rms_norm_gated(
    core: &[f32],
    z: &[f32],
    weight: &[f32],
    tokens: usize,
    heads: usize,
    dim: usize,
    eps: f32,
) -> Vec<f32> {
    let rows = tokens * heads;
    let mut out = vec![0.0; rows * dim];
    for row in 0..rows {
        let base = row * dim;
        let mean = core[base..base + dim]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            / dim as f32;
        let inv = (mean + eps).sqrt().recip();
        for i in 0..dim {
            out[base + i] = core[base + i] * inv * weight[i] * silu(z[base + i]);
        }
    }
    out
}

fn linear(x: &[f32], weight: &[f32], rows: usize, in_dim: usize, out_dim: usize) -> Vec<f32> {
    let mut out = vec![0.0; rows * out_dim];
    for row in 0..rows {
        for out_col in 0..out_dim {
            let mut acc = 0.0;
            for in_col in 0..in_dim {
                acc += x[row * in_dim + in_col] * weight[out_col * in_dim + in_col];
            }
            out[row * out_dim + out_col] = acc;
        }
    }
    out
}

fn depthwise_causal_conv_silu(
    x: &[f32],
    weight: &[f32],
    tokens: usize,
    channels: usize,
    kernel: usize,
) -> Vec<f32> {
    let pad = kernel - 1;
    let mut out = vec![0.0; tokens * channels];
    for t in 0..tokens {
        for c in 0..channels {
            let mut acc = 0.0;
            for k in 0..kernel {
                let padded_idx = t + k;
                if padded_idx >= pad {
                    let src_t = padded_idx - pad;
                    if src_t < tokens {
                        acc += x[src_t * channels + c] * weight[(c * kernel) + k];
                    }
                }
            }
            out[t * channels + c] = silu(acc);
        }
    }
    out
}

fn split_features(x: &[f32], rows: usize, width: usize, start: usize, len: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(rows * len);
    for row in 0..rows {
        out.extend_from_slice(&x[row * width + start..row * width + start + len]);
    }
    out
}

fn compute_g(a: &[f32], a_log: &[f32], dt_bias: &[f32], tokens: usize, heads: usize) -> Vec<f32> {
    let mut out = vec![0.0; tokens * heads];
    for t in 0..tokens {
        for h in 0..heads {
            out[t * heads + h] = -a_log[h].exp() * softplus(a[t * heads + h] + dt_bias[h]);
        }
    }
    out
}

fn chunk_gated_delta_rule_single_chunk(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    g: &[f32],
    beta: &[f32],
    tokens: usize,
    heads: usize,
    key_dim: usize,
    value_dim: usize,
) -> Result<Vec<f32>, String> {
    const CHUNK: usize = 64;
    if tokens > CHUNK {
        return Err(format!(
            "single-chunk replay supports <= {CHUNK} tokens, got {tokens}"
        ));
    }
    let mut q = vec![0.0; heads * CHUNK * key_dim];
    let mut k = vec![0.0; heads * CHUNK * key_dim];
    let mut v = vec![0.0; heads * CHUNK * value_dim];
    let mut b = vec![0.0; heads * CHUNK];
    let mut gg = vec![0.0; heads * CHUNK];
    let scale = (key_dim as f32).sqrt().recip();
    for t in 0..tokens {
        for h in 0..heads {
            let mut q_norm = 0.0;
            let mut k_norm = 0.0;
            for d in 0..key_dim {
                q_norm += query[qkv_idx(tokens, heads, key_dim, t, h, d)].powi(2);
                k_norm += key[qkv_idx(tokens, heads, key_dim, t, h, d)].powi(2);
            }
            let q_inv = (q_norm + 1e-6).sqrt().recip();
            let k_inv = (k_norm + 1e-6).sqrt().recip();
            for d in 0..key_dim {
                q[padded_idx(heads, CHUNK, key_dim, h, t, d)] =
                    query[qkv_idx(tokens, heads, key_dim, t, h, d)] * q_inv * scale;
                k[padded_idx(heads, CHUNK, key_dim, h, t, d)] =
                    key[qkv_idx(tokens, heads, key_dim, t, h, d)] * k_inv;
            }
            for d in 0..value_dim {
                v[padded_idx(heads, CHUNK, value_dim, h, t, d)] =
                    value[qkv_idx(tokens, heads, value_dim, t, h, d)];
            }
            b[h * CHUNK + t] = beta[t * heads + h];
            gg[h * CHUNK + t] = g[t * heads + h];
        }
    }

    let mut out = vec![0.0; tokens * heads * value_dim];
    for h in 0..heads {
        let mut g_cum = [0.0f32; CHUNK];
        let mut acc = 0.0;
        for t in 0..CHUNK {
            acc += gg[h * CHUNK + t];
            g_cum[t] = acc;
        }
        let mut decay = vec![0.0; CHUNK * CHUNK];
        for i in 0..CHUNK {
            for j in 0..=i {
                decay[i * CHUNK + j] = (g_cum[i] - g_cum[j]).exp();
            }
        }
        let mut attn = vec![0.0; CHUNK * CHUNK];
        for i in 0..CHUNK {
            for j in 0..i {
                let mut dot = 0.0;
                for d in 0..key_dim {
                    let kb = k[padded_idx(heads, CHUNK, key_dim, h, i, d)] * b[h * CHUNK + i];
                    dot += kb * k[padded_idx(heads, CHUNK, key_dim, h, j, d)];
                }
                attn[i * CHUNK + j] = -(dot * decay[i * CHUNK + j]);
            }
        }
        for i in 1..CHUNK {
            let row = (0..i).map(|j| attn[i * CHUNK + j]).collect::<Vec<_>>();
            for j in 0..i {
                let mut correction = 0.0;
                for l in 0..i {
                    correction += row[l] * attn[l * CHUNK + j];
                }
                attn[i * CHUNK + j] = row[j] + correction;
            }
        }
        for i in 0..CHUNK {
            attn[i * CHUNK + i] += 1.0;
        }
        let mut value_prime = vec![0.0; CHUNK * value_dim];
        for i in 0..CHUNK {
            for d in 0..value_dim {
                let mut sum = 0.0;
                for j in 0..CHUNK {
                    let v_beta = v[padded_idx(heads, CHUNK, value_dim, h, j, d)] * b[h * CHUNK + j];
                    sum += attn[i * CHUNK + j] * v_beta;
                }
                value_prime[i * value_dim + d] = sum;
            }
        }
        for t in 0..tokens {
            for d in 0..value_dim {
                let mut sum = 0.0;
                for j in 0..CHUNK {
                    let mut qk = 0.0;
                    for kd in 0..key_dim {
                        qk += q[padded_idx(heads, CHUNK, key_dim, h, t, kd)]
                            * k[padded_idx(heads, CHUNK, key_dim, h, j, kd)];
                    }
                    sum += qk * decay[t * CHUNK + j] * value_prime[j * value_dim + d];
                }
                out[qkv_idx(tokens, heads, value_dim, t, h, d)] = sum;
            }
        }
    }
    Ok(out)
}

fn qkv_idx(tokens: usize, heads: usize, dim: usize, t: usize, h: usize, d: usize) -> usize {
    debug_assert!(t < tokens);
    debug_assert!(h < heads);
    (t * heads + h) * dim + d
}

fn padded_idx(heads: usize, chunk: usize, dim: usize, h: usize, t: usize, d: usize) -> usize {
    debug_assert!(h < heads);
    debug_assert!(t < chunk);
    (h * chunk + t) * dim + d
}

fn add(a: &[f32], b: &[f32]) -> Result<Vec<f32>, String> {
    if a.len() != b.len() {
        return Err(format!("add length mismatch {} != {}", a.len(), b.len()));
    }
    Ok(a.iter().zip(b).map(|(a, b)| a + b).collect())
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn write_f32(path: &Path, values: &[f32]) -> Result<(), String> {
    let mut file = File::create(path).map_err(|err| format!("create {path:?}: {err}"))?;
    for value in values {
        file.write_all(&value.to_le_bytes())
            .map_err(|err| format!("write {path:?}: {err}"))?;
    }
    Ok(())
}

fn write_json(path: &Path, value: &Value) -> Result<(), String> {
    let mut file = File::create(path).map_err(|err| format!("create {path:?}: {err}"))?;
    serde_json::to_writer_pretty(&mut file, value)
        .map_err(|err| format!("write json {path:?}: {err}"))?;
    file.write_all(b"\n")
        .map_err(|err| format!("write newline {path:?}: {err}"))?;
    Ok(())
}

fn git_summary() -> Value {
    fn git(args: &[&str]) -> String {
        Command::new("git")
            .args(args)
            .output()
            .ok()
            .filter(|output| output.status.success())
            .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
            .unwrap_or_else(|| "unknown".to_string())
    }
    let tracked = git(&["status", "--short", "--untracked-files=no"])
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    let untracked = git(&["ls-files", "--others", "--exclude-standard"])
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(str::to_string)
        .collect::<Vec<_>>();
    json!({
        "sha": git(&["rev-parse", "HEAD"]),
        "is_dirty": !tracked.is_empty() || !untracked.is_empty(),
        "tracked_status_short": tracked,
        "untracked_count": untracked.len(),
        "untracked_sample": untracked.into_iter().take(20).collect::<Vec<_>>(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depthwise_causal_conv_uses_left_context_only() {
        let x = vec![1.0, 2.0, 3.0];
        let w = vec![10.0, 1.0];
        let out = depthwise_causal_conv_silu(&x, &w, 3, 1, 2);
        let expected_raw = [1.0, 12.0, 23.0];
        for (got, expected) in out.iter().zip(expected_raw) {
            assert!((*got - silu(expected)).abs() < 1e-6);
        }
    }

    #[test]
    fn rms_norm_plus_one_matches_qwen35_semantics() {
        let x = vec![3.0, 4.0];
        let w = vec![0.0, 1.0];
        let out = rms_norm_plus_one(&x, &w, 1, 2, 0.0);
        let inv = ((3.0f32 * 3.0 + 4.0 * 4.0) / 2.0).sqrt().recip();
        assert!((out[0] - 3.0 * inv).abs() < 1e-6);
        assert!((out[1] - 4.0 * inv * 2.0).abs() < 1e-6);
    }
}
