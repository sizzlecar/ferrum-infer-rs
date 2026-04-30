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
    ///
    /// Two paths:
    ///   1. **Fast (quant-fused)** — every part is Q4_K with no bias. The
    ///      raw super-block bytes are byte-concatenated and handed to
    ///      `QuantLinear::from_gguf_bytes`, so weights stay quantised in
    ///      backend memory.
    ///   2. **Eager (dense-fused)** — fallback. Each part is dequanted to
    ///      fp32 and concatenated; the result wraps a dense fp16 weight
    ///      via `GgufLinear::from_dense_rows`.
    ///
    /// Why the dual path: an 8B Qwen3 has 36 layers × (qkv + gate_up) of
    /// ~140M weights apiece — eager-fp32-fusing them inflates 5 GB on disk
    /// to 25+ GB in RAM, defeating Q4_K_M entirely. The fast path only
    /// works for Q4K-without-bias which is the vast majority of dense
    /// transformers; bias-bearing fusions (rare) take the eager hit.
    fn load_fused(&self, parts: &[String]) -> Result<Box<dyn Linear<B>>> {
        if let Some(fast) = self.try_load_fused_q4k(parts)? {
            if std::env::var("FERRUM_GGUF_LOAD_TRACE").is_ok() {
                eprintln!("[gguf-load] {:?} → fused-Q4 (homogeneous)", parts);
            }
            return Ok(fast);
        }
        if let Some(multi) = self.try_load_fused_multi_quant(parts)? {
            if std::env::var("FERRUM_GGUF_LOAD_TRACE").is_ok() {
                eprintln!("[gguf-load] {:?} → MultiQuant (mixed dtype)", parts);
            }
            return Ok(multi);
        }
        if std::env::var("FERRUM_GGUF_LOAD_TRACE").is_ok() {
            eprintln!("[gguf-load] {:?} → eager fp32 fallback ⚠", parts);
        }
        self.load_fused_eager(parts)
    }

    /// Multi-quant fused fast path: each part is a Q4_K or Q6_K tensor
    /// with no bias. Parts may have **different** quant types (e.g.
    /// Qwen3 qkv_proj where q+k are Q4_K but v is Q6_K). Builds a
    /// `MetalQuantStore::Fused` (or whatever the backend's `Fused`
    /// variant is) so each part stays compact in backend memory and
    /// gemv dispatches per part with output offsets.
    fn try_load_fused_multi_quant(&self, parts: &[String]) -> Result<Option<Box<dyn Linear<B>>>> {
        let mut spec: Vec<(ferrum_kernels::backend::GgufQuantType, &[u8], usize)> = Vec::new();
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

            // Bias on a fused part disqualifies the whole multi-quant
            // path; fall back to eager fusion which already handles bias.
            let has_bias = ferrum_to_gguf(&format!("{stem}.bias"))
                .map(|n| self.gguf.has_tensor(&n))
                .unwrap_or(false);
            if has_bias {
                return Ok(None);
            }

            let info = self.gguf.tensor_info(&gguf_name).ok_or_else(|| {
                FerrumError::model(format!("tensor_info missing for '{gguf_name}'"))
            })?;
            let kind = match info.ggml_dtype {
                candle_core::quantized::GgmlDType::Q4K => {
                    ferrum_kernels::backend::GgufQuantType::Q4K
                }
                candle_core::quantized::GgmlDType::Q6K => {
                    ferrum_kernels::backend::GgufQuantType::Q6K
                }
                _ => return Ok(None), // unsupported quant in this part
            };

            let dims = info.shape.dims();
            if dims.len() != 2 {
                return Ok(None);
            }
            let (rows, cols) = (dims[0], dims[1]);
            if cols % 256 != 0 {
                return Ok(None);
            }
            match cols_check {
                Some(c) if c != cols => {
                    return Err(FerrumError::model(format!(
                        "GgufLoader: fusion in_features mismatch ({c} vs {cols} for '{stem}')"
                    )))
                }
                _ => cols_check = Some(cols),
            }

            // Slice the mmap directly. The slice's lifetime is tied to
            // `&self.gguf`, which outlives this scope, so the backend
            // can read the bytes safely without us owning a copy.
            let bytes = self.gguf.tensor_byte_slice(&gguf_name).ok_or_else(|| {
                FerrumError::model(format!(
                    "GgufLoader: tensor_byte_slice failed for '{gguf_name}'"
                ))
            })?;
            spec.push((kind, bytes, rows));
        }

        let cols = cols_check.ok_or_else(|| FerrumError::model("fusion: no parts"))?;
        let parts_view: Vec<(_, &[u8], _)> = spec
            .iter()
            .map(|(kind, bytes, rows)| (*kind, *bytes, *rows))
            .collect();
        let quant = match crate::QuantLinear::<B>::from_gguf_fused(&parts_view, cols) {
            Ok(q) => q,
            Err(_) => return Ok(None), // backend doesn't support Fused
        };
        Ok(Some(Box::new(quant)))
    }

    /// Q4_K fast path for `load_fused`. Returns `Ok(None)` if any part
    /// disqualifies (non-Q4K dtype, rank != 2, has bias, cols mismatch).
    fn try_load_fused_q4k(&self, parts: &[String]) -> Result<Option<Box<dyn Linear<B>>>> {
        let mut fused_bytes: Vec<u8> = Vec::new();
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

            // Disqualifier 1: bias on this part — can't byte-concat that
            // into a single QuantLinear.
            let bias_name = ferrum_to_gguf(&format!("{stem}.bias"))
                .map(|n| self.gguf.has_tensor(&n))
                .unwrap_or(false);
            if bias_name {
                return Ok(None);
            }

            let info = self.gguf.tensor_info(&gguf_name).ok_or_else(|| {
                FerrumError::model(format!("tensor_info missing for '{gguf_name}'"))
            })?;

            // Disqualifier 2: not Q4K dtype.
            if !matches!(info.ggml_dtype, candle_core::quantized::GgmlDType::Q4K) {
                return Ok(None);
            }

            let dims = info.shape.dims();
            if dims.len() != 2 {
                return Ok(None);
            }
            let (rows, cols) = (dims[0], dims[1]);

            // Disqualifier 3: cols not a multiple of 256 (Q4K super-block
            // boundary) — should not happen for Q4K tensors, but guard
            // anyway so byte-concat produces a valid block stream.
            if cols % 256 != 0 {
                return Ok(None);
            }

            match cols_check {
                Some(c) if c != cols => {
                    return Err(FerrumError::model(format!(
                        "GgufLoader: fusion in_features mismatch ({c} vs {cols} for '{stem}')"
                    )))
                }
                _ => cols_check = Some(cols),
            }

            // Read raw block bytes directly from the mmap (no candle
            // QTensor intermediate copy). Fused tensors must still be
            // byte-concatenated into a single buffer, so the fused
            // payload itself remains a heap allocation — but it's
            // a one-shot total ≪ MoE expert weights, so the
            // consequence is negligible.
            let bytes = self.gguf.tensor_byte_slice(&gguf_name).ok_or_else(|| {
                FerrumError::model(format!(
                    "GgufLoader: tensor_byte_slice failed for '{gguf_name}'"
                ))
            })?;
            // Sanity: 144 bytes per super-block, super-blocks = rows * (cols / 256).
            let expected = rows * (cols / 256) * 144;
            debug_assert_eq!(
                bytes.len(),
                expected,
                "Q4K byte count mismatch for '{gguf_name}': got {} expected {}",
                bytes.len(),
                expected
            );

            fused_bytes.extend_from_slice(bytes);
            total_rows += rows;
        }

        let cols = cols_check.ok_or_else(|| FerrumError::model("fusion: no parts"))?;
        let quant = crate::QuantLinear::<B>::from_gguf_bytes(
            ferrum_kernels::backend::GgufQuantType::Q4K,
            &fused_bytes,
            total_rows,
            cols,
        )?;
        Ok(Some(Box::new(quant)))
    }

    /// Eager (dequant-to-fp32 then concat) fusion. Used for non-Q4K parts
    /// or parts with bias. See `load_fused` doc for the trade-off.
    fn load_fused_eager(&self, parts: &[String]) -> Result<Box<dyn Linear<B>>> {
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
                // Inspect the on-disk dtype before reading the payload.
                // Q4_K_M (and future k-quant flavours) get the QuantLinear
                // path that keeps weights quantised in backend memory;
                // F16 / F32 / non-Q4-K dtypes fall through to GgufLinear's
                // eager-dequant DenseLinear path.
                let info = self.gguf.tensor_info(&gguf_weight).ok_or_else(|| {
                    FerrumError::model(format!("tensor_info missing for '{gguf_weight}'"))
                })?;
                let dims = info.shape.dims();
                if dims.len() != 2 {
                    return Err(FerrumError::model(format!(
                        "GgufLoader::load_linear '{name}': expected rank-2 weight, got rank {}",
                        dims.len()
                    )));
                }
                let (n_rows, n_cols) = (dims[0], dims[1]);

                let quant_kind = match info.ggml_dtype {
                    candle_core::quantized::GgmlDType::Q4K => {
                        Some(ferrum_kernels::backend::GgufQuantType::Q4K)
                    }
                    candle_core::quantized::GgmlDType::Q6K => {
                        Some(ferrum_kernels::backend::GgufQuantType::Q6K)
                    }
                    _ => None,
                };
                if let Some(kind) = quant_kind {
                    // Read raw block bytes and hand to QuantLinear.
                    // Bias on quantised projections is rare in GGUF
                    // (Qwen2.5 attention biases land as F32), so we
                    // currently take the bias path only when the bias
                    // tensor is present AND the weight is non-quantised.
                    // For quantised weights with bias, fall back to
                    // eager dequant so Phase 1B's bias support keeps
                    // working.
                    let has_bias = ferrum_to_gguf(&format!("{name}.bias"))
                        .map(|n| self.gguf.has_tensor(&n))
                        .unwrap_or(false);
                    if !has_bias {
                        // Zero-copy: slice the mmap directly. The
                        // backend's registry (`register_gguf_mmap`)
                        // recognises the slice as belonging to the
                        // shared file buffer and returns a `QuantStore`
                        // that bind-references the big buffer with an
                        // offset, instead of allocating a fresh device
                        // copy. Falls back to copy if no registration
                        // covers this slice.
                        let bytes = self.gguf.tensor_byte_slice(&gguf_weight).ok_or_else(|| {
                            FerrumError::model(format!(
                                "GgufLoader: tensor_byte_slice failed for '{gguf_weight}'"
                            ))
                        })?;
                        let quant =
                            crate::QuantLinear::<B>::from_gguf_bytes(kind, bytes, n_rows, n_cols)?;
                        return Ok(Box::new(quant));
                    }
                    // else fall through to eager-dequant bias path below
                }

                let qt = self
                    .gguf
                    .read_tensor(&gguf_weight, &self.decode_device)
                    .map_err(candle_to_ferrum)?;
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
