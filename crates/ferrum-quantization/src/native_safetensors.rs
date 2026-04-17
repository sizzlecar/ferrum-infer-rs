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

use crate::config::{QuantConfig, QuantMethod};
use crate::dense::DenseLinear;
use crate::gptq::GptqLinear;
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

    /// Read a tensor as i32 (for GPTQ qweight / qzeros / g_idx).
    fn read_i32(&self, name: &str) -> Result<(Vec<i32>, Vec<usize>)> {
        let shard_idx = *self
            .index
            .get(name)
            .ok_or_else(|| FerrumError::model(format!("tensor '{name}' not in index")))?;
        let view = self.shards[shard_idx].get(name)?;
        let shape = view.shape().to_vec();
        if view.dtype() != Dtype::I32 {
            return Err(FerrumError::model(format!(
                "'{name}': expected I32, got {:?}",
                view.dtype()
            )));
        }
        let bytes = view.data();
        debug_assert_eq!(bytes.len() % 4, 0);
        let mut out = vec![0i32; bytes.len() / 4];
        out.as_mut_slice()
            .iter_mut()
            .zip(bytes.chunks_exact(4))
            .for_each(|(d, chunk)| *d = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
        Ok((out, shape))
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
        // GPTQ first: `<name>.qweight` + `<name>.scales` + `<name>.qzeros`.
        let qw_key = format!("{name}.qweight");
        if self.has(&qw_key) {
            return self.load_gptq_linear(name);
        }
        // GPTQ fusion shims: synthesise qkv_proj / gate_up_proj from split
        // components — same pattern as Dense but concatenating the GPTQ
        // tensors (qweight/scales/qzeros) along the N dim.
        if name.ends_with("qkv_proj") {
            let prefix = &name[..name.len() - "qkv_proj".len()];
            let parts = [
                format!("{prefix}q_proj"),
                format!("{prefix}k_proj"),
                format!("{prefix}v_proj"),
            ];
            if parts.iter().all(|p| self.has(&format!("{p}.qweight"))) {
                return self.load_gptq_linear_fused(&parts);
            }
        }
        if name.ends_with("gate_up_proj") {
            let prefix = &name[..name.len() - "gate_up_proj".len()];
            let parts = [format!("{prefix}gate_proj"), format!("{prefix}up_proj")];
            if parts.iter().all(|p| self.has(&format!("{p}.qweight"))) {
                return self.load_gptq_linear_fused(&parts);
            }
        }

        // Direct fused `<name>.weight` next.
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
    /// Load a GPTQ-packed linear projection: reads `<name>.qweight`,
    /// `<name>.scales`, `<name>.qzeros`, optionally `<name>.g_idx`, and
    /// hands the raw host-side tensors to `Backend::load_gptq` which
    /// repacks + uploads per its own strategy.
    fn load_gptq_linear(&self, name: &str) -> Result<Box<dyn Linear<B>>> {
        let qcfg = self.quant_config.as_ref().ok_or_else(|| {
            FerrumError::model(format!(
                "'{name}.qweight' present but no quantize_config.json — \
                 can't determine bits/group_size"
            ))
        })?;
        if qcfg.method != QuantMethod::Gptq {
            return Err(FerrumError::model(format!(
                "'{name}.qweight' present but quant_method={:?} (expected GPTQ)",
                qcfg.method
            )));
        }

        let (qweight, qw_shape) = self.read_i32(&format!("{name}.qweight"))?;
        let (scales_f32, sc_shape) = self.read_f32(&format!("{name}.scales"))?;
        let (qzeros, _qz_shape) = self.read_i32(&format!("{name}.qzeros"))?;
        let g_idx = if self.has(&format!("{name}.g_idx")) {
            Some(self.read_i32(&format!("{name}.g_idx"))?.0)
        } else {
            None
        };

        // Shape inference: qweight is [K/8, N]; scales is [K/group, N].
        // → K = qw_shape[0] * 8, N = qw_shape[1].
        if qw_shape.len() != 2 {
            return Err(FerrumError::model(format!(
                "'{name}.qweight' expected 2D, got {qw_shape:?}"
            )));
        }
        let in_features = qw_shape[0] * 8;
        let out_features = qw_shape[1];
        if sc_shape.len() != 2 || sc_shape[1] != out_features {
            return Err(FerrumError::model(format!(
                "'{name}.scales' {sc_shape:?} incompatible with qweight {qw_shape:?}"
            )));
        }

        let linear = GptqLinear::<B>::from_raw(
            &qweight,
            &scales_f32,
            &qzeros,
            g_idx.as_deref(),
            qcfg.bits,
            qcfg.group_size,
            in_features,
            out_features,
        )?;
        Ok(Box::new(linear))
    }

    /// Fuse multiple GPTQ projections by concatenating qweight/scales/qzeros
    /// along the output (N) dim. Matches the Dense fusion shim used for
    /// non-quantized models: q_proj + k_proj + v_proj → qkv_proj.
    ///
    /// All parts must share:
    /// - in_features (K)
    /// - bits, group_size
    /// - qzeros N-packing (which the GPTQ format always honours: qzeros[-1]
    ///   = N/8, concat along that axis works)
    ///
    /// g_idx: only present when desc_act=true. When present, all parts
    /// share it (same K rows, same activation permutation).
    fn load_gptq_linear_fused(&self, parts: &[String]) -> Result<Box<dyn Linear<B>>> {
        let qcfg = self.quant_config.as_ref().ok_or_else(|| {
            FerrumError::model("GPTQ fusion requires quantize_config.json".to_string())
        })?;
        if qcfg.method != QuantMethod::Gptq {
            return Err(FerrumError::model(format!(
                "GPTQ fusion but quant_method={:?}",
                qcfg.method
            )));
        }

        let mut qw_acc: Vec<i32> = Vec::new();
        let mut sc_acc: Vec<f32> = Vec::new();
        let mut qz_acc: Vec<i32> = Vec::new();
        let mut qw_rows = 0usize;
        let mut sc_rows = 0usize;
        let mut qz_rows = 0usize;
        let mut total_n = 0usize;
        let mut total_n_scales = 0usize;
        let mut total_n_zeros = 0usize;
        let mut g_idx: Option<Vec<i32>> = None;
        // Segments: (qw_slice, sc_slice, qz_slice) per part, needed for N-major layout concat
        let mut qw_parts: Vec<(Vec<i32>, usize, usize)> = Vec::new(); // (data, rows, cols)
        let mut sc_parts: Vec<(Vec<f32>, usize, usize)> = Vec::new();
        let mut qz_parts: Vec<(Vec<i32>, usize, usize)> = Vec::new();

        for p in parts {
            let (qw, qw_sh) = self.read_i32(&format!("{p}.qweight"))?;
            let (sc, sc_sh) = self.read_f32(&format!("{p}.scales"))?;
            let (qz, qz_sh) = self.read_i32(&format!("{p}.qzeros"))?;
            if qw_sh.len() != 2 || sc_sh.len() != 2 || qz_sh.len() != 2 {
                return Err(FerrumError::model(format!(
                    "GPTQ fusion '{p}': expected 2D tensors, got qw {qw_sh:?} sc {sc_sh:?} qz {qz_sh:?}"
                )));
            }
            if qw_rows == 0 {
                qw_rows = qw_sh[0];
                sc_rows = sc_sh[0];
                qz_rows = qz_sh[0];
            } else if qw_sh[0] != qw_rows || sc_sh[0] != sc_rows || qz_sh[0] != qz_rows {
                return Err(FerrumError::model(format!(
                    "GPTQ fusion row mismatch on '{p}'"
                )));
            }
            total_n += qw_sh[1];
            total_n_scales += sc_sh[1];
            total_n_zeros += qz_sh[1];
            qw_parts.push((qw, qw_sh[0], qw_sh[1]));
            sc_parts.push((sc, sc_sh[0], sc_sh[1]));
            qz_parts.push((qz, qz_sh[0], qz_sh[1]));

            // g_idx optional; if first part has it, use that
            if g_idx.is_none() {
                if self.has(&format!("{p}.g_idx")) {
                    g_idx = Some(self.read_i32(&format!("{p}.g_idx"))?.0);
                }
            }
        }

        // Interleave row-major concatenation: for each row, write all parts' cols.
        qw_acc.reserve(qw_rows * total_n);
        for r in 0..qw_rows {
            for (part, _rows, cols) in &qw_parts {
                qw_acc.extend_from_slice(&part[r * cols..r * cols + cols]);
            }
        }
        sc_acc.reserve(sc_rows * total_n_scales);
        for r in 0..sc_rows {
            for (part, _rows, cols) in &sc_parts {
                sc_acc.extend_from_slice(&part[r * cols..r * cols + cols]);
            }
        }
        qz_acc.reserve(qz_rows * total_n_zeros);
        for r in 0..qz_rows {
            for (part, _rows, cols) in &qz_parts {
                qz_acc.extend_from_slice(&part[r * cols..r * cols + cols]);
            }
        }

        let in_features = qw_rows * 8;
        let out_features = total_n;
        let linear = GptqLinear::<B>::from_raw(
            &qw_acc,
            &sc_acc,
            &qz_acc,
            g_idx.as_deref(),
            qcfg.bits,
            qcfg.group_size,
            in_features,
            out_features,
        )?;
        Ok(Box::new(linear))
    }

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
