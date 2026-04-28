//! `GgufLoader<B>`: implements `WeightLoader<B>` against a GGUF file.
//!
//! Bridges the model layer (which addresses weights by ferrum's HuggingFace-
//! style names) to the on-disk GGUF format (llama.cpp's `blk.{i}.attn_q.weight`
//! shorthand). Three responsibilities:
//!
//!   1. **Name translation** — delegates to `gguf::names::ferrum_to_gguf`
//!   2. **Tensor materialisation** — uses Phase 1A's `GgufFile::read_tensor`
//!      then dequant on CPU into `B::Buffer` for `load_tensor`, or wraps
//!      the QTensor in `GgufLinear<B>` for `load_linear`.
//!   3. **Fusion** — reproduces the `qkv_proj` / `gate_up_proj` shims the
//!      model expects: q/k/v split tensors are concatenated row-wise into
//!      a single fused weight before the Linear is built.
//!
//! All paths go through eager dequant-to-fp32 (Phase 1B's strategy).
//! Phase 1D will add a quant-aware shortcut so Q4_K_M weights can stay
//! quantised in backend memory; the public `WeightLoader<B>` API stays
//! the same.

use std::path::Path;
use std::sync::Arc;

use candle_core::Device;
use ferrum_kernels::backend::Backend;
use ferrum_types::{FerrumError, Result};

use crate::config::QuantConfig;
use crate::gguf::file::GgufFile;
use crate::gguf::linear::GgufLinear;
use crate::gguf::names::{ferrum_to_gguf, gate_up_split_parts, qkv_split_parts};
use crate::loader::WeightLoader;
use crate::traits::Linear;

/// Backend-generic weight loader for GGUF files.
///
/// Build with [`GgufLoader::open`]. The underlying file stays mmap'd for
/// the lifetime of the loader so per-tensor reads only do byte slicing,
/// not file I/O.
pub struct GgufLoader<B: Backend> {
    gguf: Arc<GgufFile>,
    /// Decode device for `QTensor::dequantize`. We always use CPU here:
    /// the dequant is followed by `B::from_slice`, which uploads to the
    /// backend's preferred memory. Going through Metal/CUDA candle paths
    /// would add a cross-allocator hop with no benefit (Phase 1D revisits).
    decode_device: Device,
    _marker: std::marker::PhantomData<B>,
}

impl<B: Backend> GgufLoader<B> {
    /// Open and parse a `.gguf` file. Tensor payloads stay on disk (mmap'd)
    /// until each `load_tensor` / `load_linear` call.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let gguf = GgufFile::open(path).map_err(candle_to_ferrum)?;
        Ok(Self {
            gguf: Arc::new(gguf),
            decode_device: Device::Cpu,
            _marker: std::marker::PhantomData,
        })
    }

    /// Build from an already-opened [`GgufFile`] (test helper, also useful
    /// when several loaders share the same mmap).
    pub fn from_file(gguf: Arc<GgufFile>) -> Self {
        Self {
            gguf,
            decode_device: Device::Cpu,
            _marker: std::marker::PhantomData,
        }
    }

    /// Direct access to the underlying file — exposes metadata + tensor
    /// descriptor lookups for callers that need them (e.g. a config helper
    /// that reads `general.architecture` and `<arch>.block_count`).
    pub fn gguf(&self) -> &GgufFile {
        &self.gguf
    }

    // ── Internals ────────────────────────────────────────────────────────

    /// Look up a ferrum-named tensor in the GGUF, returning the GGUF tensor
    /// name on success.
    fn locate(&self, ferrum_name: &str) -> Result<String> {
        let gguf_name = ferrum_to_gguf(ferrum_name).ok_or_else(|| {
            FerrumError::model(format!(
                "GgufLoader: unrecognised tensor name '{ferrum_name}' (no GGUF mapping)"
            ))
        })?;
        if !self.gguf.has_tensor(&gguf_name) {
            return Err(FerrumError::model(format!(
                "GgufLoader: tensor '{ferrum_name}' (mapped to '{gguf_name}') not present in GGUF"
            )));
        }
        Ok(gguf_name)
    }

    /// Read a quantized tensor and dequantize to fp32 row-major. Used by
    /// both `load_tensor` (raw buffer) and the fusion path (concat sources).
    fn read_dequant(&self, gguf_name: &str) -> Result<Vec<f32>> {
        let qt = self
            .gguf
            .read_tensor(gguf_name, &self.decode_device)
            .map_err(candle_to_ferrum)?;
        let dense = qt
            .dequantize(&self.decode_device)
            .map_err(candle_to_ferrum)?;
        let flat = dense.flatten_all().map_err(candle_to_ferrum)?;
        flat.to_vec1::<f32>().map_err(candle_to_ferrum)
    }

    /// Look up a tensor's `[rows, cols]` (2-D) without reading the payload.
    /// Errors if the tensor isn't 2-D — fusion needs row counts to compute
    /// the combined output dim.
    fn rows_cols(&self, gguf_name: &str) -> Result<(usize, usize)> {
        let info = self
            .gguf
            .tensor_info(gguf_name)
            .ok_or_else(|| FerrumError::model(format!("tensor info missing for '{gguf_name}'")))?;
        let dims = info.shape.dims();
        if dims.len() != 2 {
            return Err(FerrumError::model(format!(
                "expected 2-D tensor for '{gguf_name}', got rank {}",
                dims.len()
            )));
        }
        Ok((dims[0], dims[1]))
    }

    /// Build a fused `Linear<B>` by row-concatenating several sub-tensors.
    /// All parts must share `cols` (in_features); rows (out_features) sum.
    fn load_fused(&self, parts: &[String]) -> Result<Box<dyn Linear<B>>> {
        let mut fused: Vec<f32> = Vec::new();
        let mut total_rows = 0usize;
        let mut cols_check: Option<usize> = None;

        for stem in parts {
            let weight_name = format!("{stem}.weight");
            let gguf_name = ferrum_to_gguf(&weight_name).ok_or_else(|| {
                FerrumError::model(format!(
                    "GgufLoader: fusion source '{weight_name}' has no GGUF mapping"
                ))
            })?;
            if !self.gguf.has_tensor(&gguf_name) {
                return Err(FerrumError::model(format!(
                    "GgufLoader: fusion source '{weight_name}' (gguf '{gguf_name}') missing"
                )));
            }
            let (rows, cols) = self.rows_cols(&gguf_name)?;
            match cols_check {
                Some(c) if c != cols => {
                    return Err(FerrumError::model(format!(
                        "GgufLoader: fusion in_features mismatch ({c} vs {cols} for '{stem}')"
                    )))
                }
                _ => cols_check = Some(cols),
            }
            let data = self.read_dequant(&gguf_name)?;
            debug_assert_eq!(data.len(), rows * cols);
            fused.extend_from_slice(&data);
            total_rows += rows;
        }

        let cols = cols_check.ok_or_else(|| FerrumError::model("fusion: no parts"))?;
        Ok(Box::new(GgufLinear::<B>::from_dense_rows(
            &fused, total_rows, cols,
        )))
    }
}

impl<B: Backend> WeightLoader<B> for GgufLoader<B> {
    fn load_tensor(&self, name: &str) -> Result<B::Buffer> {
        let gguf_name = self.locate(name)?;
        let raw = self.read_dequant(&gguf_name)?;
        Ok(B::from_slice(&raw))
    }

    fn load_linear(&self, name: &str) -> Result<Box<dyn Linear<B>>> {
        // 1) Direct path: <name>.weight exists as a single GGUF tensor.
        if let Some(gguf_weight) = ferrum_to_gguf(&format!("{name}.weight")) {
            if self.gguf.has_tensor(&gguf_weight) {
                let qt = self
                    .gguf
                    .read_tensor(&gguf_weight, &self.decode_device)
                    .map_err(candle_to_ferrum)?;
                // Optional bias (Qwen2.5 / Bert variants)
                if let Some(gguf_bias) = ferrum_to_gguf(&format!("{name}.bias")) {
                    if self.gguf.has_tensor(&gguf_bias) {
                        let bqt = self
                            .gguf
                            .read_tensor(&gguf_bias, &self.decode_device)
                            .map_err(candle_to_ferrum)?;
                        let linear = GgufLinear::<B>::from_qtensor_with_bias(&qt, &bqt)
                            .map_err(candle_to_ferrum)?;
                        return Ok(Box::new(linear));
                    }
                }
                let linear = GgufLinear::<B>::from_qtensor(&qt).map_err(candle_to_ferrum)?;
                return Ok(Box::new(linear));
            }
        }

        // 2) Fusion path: qkv_proj from q_proj/k_proj/v_proj
        if let Some(layer_prefix) = name.strip_suffix("self_attn.qkv_proj") {
            let parts = qkv_split_parts(layer_prefix);
            return self.load_fused(&parts);
        }
        // 3) Fusion path: gate_up_proj from gate_proj/up_proj
        if let Some(layer_prefix) = name.strip_suffix("mlp.gate_up_proj") {
            let parts = gate_up_split_parts(layer_prefix);
            return self.load_fused(&parts);
        }

        Err(FerrumError::model(format!(
            "GgufLoader: could not load Linear '{name}' — no direct weight, no split components"
        )))
    }

    fn has_tensor(&self, name: &str) -> bool {
        match ferrum_to_gguf(name) {
            Some(g) => self.gguf.has_tensor(&g),
            None => false,
        }
    }

    fn quant_config(&self) -> Option<&QuantConfig> {
        // Phase 1C doesn't surface a QuantConfig — every tensor in a GGUF
        // declares its own dtype (`GgmlDType`) per descriptor, so the
        // model's existing branching on QuantConfig::method isn't useful
        // here. Phase 1D may add a derived config if downstream code grows
        // a need for it.
        None
    }
}

fn candle_to_ferrum(e: candle_core::Error) -> FerrumError {
    FerrumError::model(format!("candle: {e}"))
}
