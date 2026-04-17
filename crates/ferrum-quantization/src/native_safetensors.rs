//! Native safetensors `WeightLoader<B>` — mmap + `safetensors` crate, no
//! candle dependency on the LLM hot path.
//!
//! What this owns:
//!   - Discovering `model.safetensors` vs sharded `model.safetensors.index.json`.
//!   - Mmapping each shard file.
//!   - Per-tensor lookup: returns shape + dtype + a byte slice into the mmap.
//!   - f32 materialisation for Dense weights (bf16 / f16 / f32 accepted).
//!   - The Qwen3 / Llama fusion trick: `qkv_proj` / `gate_up_proj` synthesised
//!     on the fly from split `q_proj`+`k_proj`+`v_proj` etc.
//!
//! What it deliberately doesn't do:
//!   - GPTQ / AWQ / GGUF packed weights. Those need `B::from_slice_i32` /
//!     `B::from_slice_f16` which aren't on the Backend trait yet. A dedicated
//!     loader per quant format lands in Phase E.

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use ferrum_kernels::backend::Backend;
use ferrum_types::{FerrumError, Result};
use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

use crate::config::QuantConfig;
use crate::dense::DenseLinear;
use crate::loader::WeightLoader;
use crate::traits::Linear;

/// A single shard file: mmap + name→(shape, dtype, byte-offset-in-mmap).
struct Shard {
    mmap: Mmap,
    /// Parsed entries. Safetensors' `SafeTensors` type borrows from the mmap,
    /// so we can't store it directly — instead we pre-extract name → metadata
    /// and rebuild a `SafeTensors` view on demand via `SafeTensors::deserialize`.
    names: Vec<String>,
}

impl Shard {
    fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| FerrumError::io(format!("open {path:?}: {e}")))?;
        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| FerrumError::io(format!("mmap {path:?}: {e}")))?
        };
        // Parse just to validate and extract names; the SafeTensors view is
        // rebuilt on each read (cheap — it's a header reparse).
        let st = SafeTensors::deserialize(&mmap)
            .map_err(|e| FerrumError::model(format!("parse {path:?}: {e}")))?;
        let names = st.names().iter().map(|s| s.to_string()).collect();
        Ok(Self { mmap, names })
    }

    fn get<'a>(&'a self, name: &str) -> Result<safetensors::tensor::TensorView<'a>> {
        let st = SafeTensors::deserialize(&self.mmap)
            .map_err(|e| FerrumError::model(format!("reparse: {e}")))?;
        st.tensor(name)
            .map_err(|e| FerrumError::model(format!("tensor '{name}': {e}")))
    }
}

/// Native safetensors loader. Generic over `Backend` so every tensor is
/// materialised directly into backend-native buffers.
pub struct NativeSafetensorsLoader<B: Backend> {
    /// All shards keyed by file; each tensor's name maps to its shard here.
    shards: Vec<Shard>,
    /// Name → shard index. Populated once at construction.
    index: HashMap<String, usize>,
    /// Optional `quantize_config.json` contents.
    quant_config: Option<QuantConfig>,
    _m: std::marker::PhantomData<B>,
}

impl<B: Backend> NativeSafetensorsLoader<B> {
    /// Discover shards under `model_dir` and build the name → shard index.
    pub fn open(model_dir: impl AsRef<Path>) -> Result<Self> {
        let dir = model_dir.as_ref();

        let shard_paths = if dir.join("model.safetensors").exists() {
            vec![dir.join("model.safetensors")]
        } else if dir.join("model.safetensors.index.json").exists() {
            Self::parse_sharded_index(&dir.join("model.safetensors.index.json"))?
                .into_iter()
                .map(|name| dir.join(name))
                .collect()
        } else {
            return Err(FerrumError::model(format!(
                "no safetensors files in {dir:?}"
            )));
        };

        let mut shards = Vec::with_capacity(shard_paths.len());
        let mut index: HashMap<String, usize> = HashMap::new();
        for (i, p) in shard_paths.iter().enumerate() {
            let shard = Shard::open(p)?;
            for name in &shard.names {
                index.insert(name.clone(), i);
            }
            shards.push(shard);
        }

        let quant_config = load_quantize_config(dir)?;

        Ok(Self {
            shards,
            index,
            quant_config,
            _m: std::marker::PhantomData,
        })
    }

    fn parse_sharded_index(index_path: &Path) -> Result<Vec<String>> {
        let data = std::fs::read_to_string(index_path)
            .map_err(|e| FerrumError::io(format!("read {index_path:?}: {e}")))?;
        let json: serde_json::Value = serde_json::from_str(&data)
            .map_err(|e| FerrumError::serialization(format!("index json: {e}")))?;
        let weight_map = json
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| FerrumError::model("index missing weight_map"))?;
        let mut files: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect();
        files.sort();
        files.dedup();
        Ok(files)
    }

    /// Read a tensor as f32 (converting from bf16 / f16 / f32) + its shape.
    fn read_f32(&self, name: &str) -> Result<(Vec<f32>, Vec<usize>)> {
        let shard_idx = *self
            .index
            .get(name)
            .ok_or_else(|| FerrumError::model(format!("tensor '{name}' not in index")))?;
        let view = self.shards[shard_idx].get(name)?;
        let shape = view.shape().to_vec();
        let data = dtype_to_f32(view.dtype(), view.data())?;
        Ok((data, shape))
    }

    fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }
}

impl<B: Backend> WeightLoader<B> for NativeSafetensorsLoader<B> {
    fn load_tensor(&self, name: &str) -> Result<B::Buffer> {
        let (data, _) = self.read_f32(name)?;
        Ok(B::from_slice(&data))
    }

    fn load_linear(&self, name: &str) -> Result<Box<dyn Linear<B>>> {
        // Direct fused `<name>.weight` first.
        let direct = format!("{name}.weight");
        if self.has(&direct) {
            let (data, shape) = self.read_f32(&direct)?;
            if shape.len() != 2 {
                return Err(FerrumError::model(format!(
                    "linear '{name}': expected 2D weight, got {shape:?}"
                )));
            }
            return Ok(Box::new(DenseLinear::<B>::from_rows(
                &data, shape[0], shape[1],
            )));
        }

        // Llama-family fusion shims: synthesise qkv_proj / gate_up_proj from
        // split q_proj+k_proj+v_proj / gate_proj+up_proj if present.
        if name.ends_with("qkv_proj") {
            let prefix = &name[..name.len() - "qkv_proj".len()];
            let parts = [
                format!("{prefix}q_proj.weight"),
                format!("{prefix}k_proj.weight"),
                format!("{prefix}v_proj.weight"),
            ];
            if parts.iter().all(|p| self.has(p)) {
                let (rows, cols, data) = self.cat_rows(&parts)?;
                return Ok(Box::new(DenseLinear::<B>::from_rows(&data, rows, cols)));
            }
        }
        if name.ends_with("gate_up_proj") {
            let prefix = &name[..name.len() - "gate_up_proj".len()];
            let parts = [
                format!("{prefix}gate_proj.weight"),
                format!("{prefix}up_proj.weight"),
            ];
            if parts.iter().all(|p| self.has(p)) {
                let (rows, cols, data) = self.cat_rows(&parts)?;
                return Ok(Box::new(DenseLinear::<B>::from_rows(&data, rows, cols)));
            }
        }

        Err(FerrumError::model(format!(
            "could not load linear '{name}' — no direct `.weight`, no split components"
        )))
    }

    fn has_tensor(&self, name: &str) -> bool {
        self.has(name)
    }

    fn quant_config(&self) -> Option<&QuantConfig> {
        self.quant_config.as_ref()
    }
}

impl<B: Backend> NativeSafetensorsLoader<B> {
    /// Read each name, assert shape width matches, concatenate along dim 0.
    fn cat_rows(&self, names: &[String]) -> Result<(usize, usize, Vec<f32>)> {
        let mut total_rows = 0usize;
        let mut cols = 0usize;
        let mut out: Vec<f32> = Vec::new();
        for n in names {
            let (data, shape) = self.read_f32(n)?;
            if shape.len() != 2 {
                return Err(FerrumError::model(format!(
                    "cat_rows: '{n}' is {shape:?}, need 2D"
                )));
            }
            if cols == 0 {
                cols = shape[1];
            } else if cols != shape[1] {
                return Err(FerrumError::model(format!(
                    "cat_rows: col mismatch {cols} vs {}",
                    shape[1]
                )));
            }
            total_rows += shape[0];
            out.extend_from_slice(&data);
        }
        Ok((total_rows, cols, out))
    }
}

fn dtype_to_f32(dtype: Dtype, raw: &[u8]) -> Result<Vec<f32>> {
    match dtype {
        Dtype::F32 => {
            debug_assert_eq!(raw.len() % 4, 0);
            let n = raw.len() / 4;
            let mut out = vec![0.0f32; n];
            for i in 0..n {
                let bytes = [raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]];
                out[i] = f32::from_le_bytes(bytes);
            }
            Ok(out)
        }
        Dtype::F16 => {
            debug_assert_eq!(raw.len() % 2, 0);
            let n = raw.len() / 2;
            let mut out = vec![0.0f32; n];
            for i in 0..n {
                let bytes = [raw[i * 2], raw[i * 2 + 1]];
                out[i] = f16::from_le_bytes(bytes).to_f32();
            }
            Ok(out)
        }
        Dtype::BF16 => {
            debug_assert_eq!(raw.len() % 2, 0);
            let n = raw.len() / 2;
            let mut out = vec![0.0f32; n];
            for i in 0..n {
                let bytes = [raw[i * 2], raw[i * 2 + 1]];
                out[i] = bf16::from_le_bytes(bytes).to_f32();
            }
            Ok(out)
        }
        other => Err(FerrumError::model(format!(
            "dtype {other:?} not supported by NativeSafetensorsLoader's f32 path; \
             use a format-specific loader (GPTQ / AWQ / GGUF)",
        ))),
    }
}

fn load_quantize_config(dir: &Path) -> Result<Option<QuantConfig>> {
    let p = dir.join("quantize_config.json");
    if !p.exists() {
        return Ok(None);
    }
    let data =
        std::fs::read_to_string(&p).map_err(|e| FerrumError::io(format!("read {p:?}: {e}")))?;
    let qc: QuantConfig = serde_json::from_str(&data)
        .map_err(|e| FerrumError::serialization(format!("parse quantize_config.json: {e}")))?;
    Ok(Some(qc))
}
