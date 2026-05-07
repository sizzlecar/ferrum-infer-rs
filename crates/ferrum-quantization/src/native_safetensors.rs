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
use std::path::Path;

use ferrum_kernels::backend::{Backend, SrcDtype};
use ferrum_types::{FerrumError, Result};
use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

/// Map a safetensors Dtype to ferrum's SrcDtype.
fn map_src_dtype(dtype: Dtype) -> Result<SrcDtype> {
    match dtype {
        Dtype::F32 => Ok(SrcDtype::F32),
        Dtype::F16 => Ok(SrcDtype::F16),
        Dtype::BF16 => Ok(SrcDtype::BF16),
        other => Err(FerrumError::model(format!(
            "dtype {other:?} not supported; Dense path expects F32/F16/BF16"
        ))),
    }
}

use crate::config::{QuantConfig, QuantMethod};
use crate::dense::DenseLinear;
use crate::gptq::GptqLinear;
use crate::loader::WeightLoader;
use crate::traits::Linear;

/// Tensor metadata extracted ONCE from the safetensors header at open time.
/// Avoids the per-tensor `SafeTensors::deserialize(&mmap)` re-parse that
/// previously dominated cold-load on stacked-MoE (>= 18 000 calls per
/// model, ~30 ms each on a 7 000-tensor header → 9+ minutes of header
/// re-parse alone).
struct TensorMeta {
    dtype: Dtype,
    shape: Vec<usize>,
    /// Byte range in the shard's mmap that holds the raw tensor data.
    data_start: usize,
    data_end: usize,
}

/// A single shard file: mmap + name→TensorMeta cache.
struct Shard {
    mmap: Mmap,
    names: Vec<String>,
    /// Pre-extracted tensor metadata. Looked up by name; no header
    /// re-parse on every read.
    meta: HashMap<String, TensorMeta>,
}

impl Shard {
    fn open(path: &Path) -> Result<Self> {
        let file = File::open(path).map_err(|e| FerrumError::io(format!("open {path:?}: {e}")))?;
        let mmap = unsafe {
            Mmap::map(&file).map_err(|e| FerrumError::io(format!("mmap {path:?}: {e}")))?
        };
        // Parse the header ONCE and cache (offset, len, dtype, shape) for
        // every tensor — re-deriving the data slice from the cache is a
        // simple `&mmap[start..end]` rather than a full header re-parse.
        let st = SafeTensors::deserialize(&mmap)
            .map_err(|e| FerrumError::model(format!("parse {path:?}: {e}")))?;
        // SafeTensors stores: 8-byte little-endian header_len + header_json
        // + data_blob. The TensorView's `data()` returns a slice into the
        // data_blob region. We compute the data_blob base by reading the
        // 8-byte header_len.
        debug_assert!(mmap.len() >= 8, "safetensors smaller than 8 bytes");
        let header_len = u64::from_le_bytes(
            mmap[0..8]
                .try_into()
                .expect("8-byte header len read failed"),
        ) as usize;
        let data_base = 8 + header_len;
        let names: Vec<String> = st.names().iter().map(|s| s.to_string()).collect();
        let mut meta = HashMap::with_capacity(names.len());
        for name in &names {
            let view = st.tensor(name).map_err(|e| {
                FerrumError::model(format!("tensor '{name}' missing during preindex: {e}"))
            })?;
            // TensorView::data() is &[u8] into the mmap; we recompute its
            // [start, end) byte range relative to the mmap base via
            // pointer arithmetic.
            let view_data = view.data();
            let start = view_data.as_ptr() as usize - mmap.as_ptr() as usize;
            let end = start + view_data.len();
            debug_assert!(start >= data_base);
            meta.insert(
                name.clone(),
                TensorMeta {
                    dtype: view.dtype(),
                    shape: view.shape().to_vec(),
                    data_start: start,
                    data_end: end,
                },
            );
        }
        let _ = data_base;
        Ok(Self { mmap, names, meta })
    }

    /// Returns (data_bytes, dtype, shape) for the named tensor without
    /// re-parsing the safetensors header.
    fn get_cached(&self, name: &str) -> Result<(&[u8], Dtype, &[usize])> {
        let m = self
            .meta
            .get(name)
            .ok_or_else(|| FerrumError::model(format!("tensor '{name}' not in shard")))?;
        Ok((&self.mmap[m.data_start..m.data_end], m.dtype, &m.shape))
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
        let (data_bytes, dtype, shape) = self.shards[shard_idx].get_cached(name)?;
        let data = dtype_to_f32(dtype, data_bytes)?;
        Ok((data, shape.to_vec()))
    }

    /// Read the raw on-disk byte slice plus dtype and shape. Zero-copy into
    /// the mmap — used to hand weights straight to `B::from_weight_bytes` so
    /// a fp16-preferring backend can skip the transient f32 Vec.
    fn read_bytes_typed(&self, name: &str) -> Result<(&[u8], SrcDtype, Vec<usize>)> {
        let shard_idx = *self
            .index
            .get(name)
            .ok_or_else(|| FerrumError::model(format!("tensor '{name}' not in index")))?;
        let (data_bytes, st_dtype, shape) = self.shards[shard_idx].get_cached(name)?;
        let dtype = map_src_dtype(st_dtype)?;
        Ok((data_bytes, dtype, shape.to_vec()))
    }

    /// Concatenate several tensors along dim 0 at the byte level. All parts
    /// must share the same dtype and trailing-dim width. Returns the fused
    /// raw bytes + common dtype + `(total_rows, cols)` shape.
    fn cat_rows_bytes(&self, names: &[String]) -> Result<(Vec<u8>, SrcDtype, (usize, usize))> {
        let mut total_rows = 0usize;
        let mut cols = 0usize;
        let mut dtype: Option<SrcDtype> = None;
        let mut bytes: Vec<u8> = Vec::new();
        for n in names {
            let (raw, d, shape) = self.read_bytes_typed(n)?;
            if shape.len() != 2 {
                return Err(FerrumError::model(format!(
                    "cat_rows_bytes: '{n}' is {shape:?}, need 2D"
                )));
            }
            match dtype {
                Some(prev) if prev != d => {
                    return Err(FerrumError::model(format!(
                        "cat_rows_bytes: dtype mismatch on '{n}'"
                    )))
                }
                _ => dtype = Some(d),
            }
            if cols == 0 {
                cols = shape[1];
            } else if cols != shape[1] {
                return Err(FerrumError::model(format!(
                    "cat_rows_bytes: col mismatch {cols} vs {}",
                    shape[1]
                )));
            }
            total_rows += shape[0];
            bytes.extend_from_slice(raw);
        }
        Ok((bytes, dtype.expect("at least one part"), (total_rows, cols)))
    }

    /// Read a tensor as i32 (for GPTQ qweight / qzeros / g_idx).
    /// Bulk memcpy from the LE-stored bytes (safetensors guarantees LE)
    /// — the previous per-element `from_le_bytes` was 4 ms for a single
    /// 768 KB tensor and dominated stacked-MoE load.
    fn read_i32(&self, name: &str) -> Result<(Vec<i32>, Vec<usize>)> {
        let shard_idx = *self
            .index
            .get(name)
            .ok_or_else(|| FerrumError::model(format!("tensor '{name}' not in index")))?;
        let (bytes, dtype, shape) = self.shards[shard_idx].get_cached(name)?;
        if dtype != Dtype::I32 {
            return Err(FerrumError::model(format!(
                "'{name}': expected I32, got {:?}",
                dtype
            )));
        }
        debug_assert_eq!(bytes.len() % 4, 0);
        let count = bytes.len() / 4;
        let mut out = Vec::<i32>::with_capacity(count);
        // SAFETY: Vec<i32>'s buffer is 4-byte aligned by allocator
        // contract. `bytes` is a raw u8 slice; copy_nonoverlapping
        // doesn't require src alignment. We're on x86_64 LE, and
        // safetensors stores LE i32 — bit pattern is identical.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                out.as_mut_ptr() as *mut u8,
                bytes.len(),
            );
            out.set_len(count);
        }
        Ok((out, shape.to_vec()))
    }

    fn has(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// Read the four raw GPTQ tensors for a named projection without
    /// triggering a Backend repack. Used by MoE batch loading: callers
    /// stack many experts host-side then issue a single `B::load_gptq`,
    /// avoiding the 12 288× per-expert Marlin repack overhead.
    ///
    /// Returns `(qweight, scales, qzeros, g_idx, k, n)`.
    /// `g_idx` is `None` when desc_act=false (no act-order perm needed).
    pub fn read_gptq_raw(
        &self,
        name: &str,
    ) -> Result<(Vec<i32>, Vec<f32>, Vec<i32>, Option<Vec<i32>>, usize, usize)> {
        let (qweight, qw_shape) = self.read_i32(&format!("{name}.qweight"))?;
        let (scales, _) = self.read_f32(&format!("{name}.scales"))?;
        let (qzeros, _) = self.read_i32(&format!("{name}.qzeros"))?;
        let g_idx = if self.has(&format!("{name}.g_idx")) {
            Some(self.read_i32(&format!("{name}.g_idx"))?.0)
        } else {
            None
        };
        if qw_shape.len() != 2 {
            return Err(FerrumError::model(format!(
                "'{name}.qweight' expected 2D, got {qw_shape:?}"
            )));
        }
        let k = qw_shape[0] * 8;
        let n = qw_shape[1];
        Ok((qweight, scales, qzeros, g_idx, k, n))
    }

    pub fn quant_config_ref(&self) -> Option<&crate::config::QuantConfig> {
        self.quant_config.as_ref()
    }

    /// Load a STACKED GPTQ tile that concatenates `num_experts` experts'
    /// raw GPTQ tensors along the N (column) axis and runs ONE backend
    /// repack — instead of `num_experts × proj_names.len()` repacks.
    ///
    /// Layout: per row `r`, the cols are emitted in expert-major order:
    /// `expert_0[proj_0|proj_1|...] | expert_1[...] | ... | expert_{N-1}[...]`.
    /// Caller can therefore index expert `e` at column offset
    /// `e * n_per_expert`, where `n_per_expert = Σ n(proj)` across the
    /// `proj_names` for one expert.
    ///
    /// `expert_prefix_fmt` should be a closure-style `&str` that contains
    /// `"{e}"` placeholder (replaced by the expert index) and ends *just
    /// before* the proj name — e.g. `"model.layers.5.mlp.experts.{e}."`.
    /// The full tensor name probed is `{expert_prefix}{proj}`.
    ///
    /// Returns `(store, n_per_expert, k)` where `n_per_expert` is the
    /// per-expert column width and `k = in_features` (shared by all).
    pub fn load_stacked_gptq_experts(
        &self,
        expert_prefix_fmt: &str,
        num_experts: usize,
        proj_names: &[&str],
    ) -> Result<(B::GptqStore, usize, usize)> {
        let qcfg = self.quant_config.as_ref().ok_or_else(|| {
            FerrumError::model(
                "load_stacked_gptq_experts requires quantize_config.json".to_string(),
            )
        })?;
        if qcfg.method != QuantMethod::Gptq {
            return Err(FerrumError::model(format!(
                "stacked GPTQ load but quant_method={:?}",
                qcfg.method
            )));
        }

        let mut qw_rows = 0usize;
        let mut sc_rows = 0usize;
        let mut qz_rows = 0usize;
        let mut n_per_expert = 0usize;
        let mut n_per_expert_scales = 0usize;
        let mut n_per_expert_zeros = 0usize;
        let mut k_shared = 0usize;
        let mut g_idx_first: Option<Vec<i32>> = None;

        // Per (expert, proj) raw slices — row-major (rows × cols).
        let total_pairs = num_experts * proj_names.len();
        let mut qw_parts: Vec<(Vec<i32>, usize)> = Vec::with_capacity(total_pairs); // (data, cols)
        let mut sc_parts: Vec<(Vec<f32>, usize)> = Vec::with_capacity(total_pairs);
        let mut qz_parts: Vec<(Vec<i32>, usize)> = Vec::with_capacity(total_pairs);

        for e in 0..num_experts {
            let prefix = expert_prefix_fmt.replace("{e}", &e.to_string());
            let mut e_n = 0usize;
            let mut e_n_scales = 0usize;
            let mut e_n_zeros = 0usize;
            for proj in proj_names {
                let name = format!("{prefix}{proj}");
                let (qw, qw_sh) = self.read_i32(&format!("{name}.qweight"))?;
                let (sc, sc_sh) = self.read_f32(&format!("{name}.scales"))?;
                let (qz, qz_sh) = self.read_i32(&format!("{name}.qzeros"))?;
                if qw_sh.len() != 2 || sc_sh.len() != 2 || qz_sh.len() != 2 {
                    return Err(FerrumError::model(format!(
                        "stacked GPTQ '{name}': expected 2D, got qw {qw_sh:?} sc {sc_sh:?} qz {qz_sh:?}"
                    )));
                }
                if qw_rows == 0 {
                    qw_rows = qw_sh[0];
                    sc_rows = sc_sh[0];
                    qz_rows = qz_sh[0];
                    k_shared = qw_sh[0] * 8;
                } else if qw_sh[0] != qw_rows
                    || sc_sh[0] != sc_rows
                    || qz_sh[0] != qz_rows
                {
                    return Err(FerrumError::model(format!(
                        "stacked GPTQ '{name}': row mismatch qw {} sc {} qz {} vs ref {qw_rows}/{sc_rows}/{qz_rows}",
                        qw_sh[0], sc_sh[0], qz_sh[0]
                    )));
                }
                e_n += qw_sh[1];
                e_n_scales += sc_sh[1];
                e_n_zeros += qz_sh[1];
                qw_parts.push((qw, qw_sh[1]));
                sc_parts.push((sc, sc_sh[1]));
                qz_parts.push((qz, qz_sh[1]));

                // g_idx is a permutation over K — Marlin assumes ONE g_idx
                // for the whole stacked tile. Validate all experts share
                // identical g_idx if any has it (which they should, since
                // K = hidden_size is the same across experts and GPTQ's
                // act-order is computed on the input distribution).
                let g_key = format!("{name}.g_idx");
                if self.has(&g_key) {
                    let (gx, _) = self.read_i32(&g_key)?;
                    match &g_idx_first {
                        None => g_idx_first = Some(gx),
                        Some(prev) => {
                            if prev.len() != gx.len() || prev.iter().zip(&gx).any(|(a, b)| a != b)
                            {
                                return Err(FerrumError::model(format!(
                                    "stacked GPTQ '{name}': g_idx mismatch with first \
                                     expert — Marlin requires identical act-order across \
                                     experts in the same stacked tile"
                                )));
                            }
                        }
                    }
                }
            }
            if e == 0 {
                n_per_expert = e_n;
                n_per_expert_scales = e_n_scales;
                n_per_expert_zeros = e_n_zeros;
            } else if e_n != n_per_expert
                || e_n_scales != n_per_expert_scales
                || e_n_zeros != n_per_expert_zeros
            {
                return Err(FerrumError::model(format!(
                    "stacked GPTQ expert {e} N mismatch: qw {e_n} sc {e_n_scales} qz {e_n_zeros} vs expert 0 {n_per_expert}/{n_per_expert_scales}/{n_per_expert_zeros}"
                )));
            }
        }

        let total_n = num_experts * n_per_expert;
        let total_n_scales = num_experts * n_per_expert_scales;
        let total_n_zeros = num_experts * n_per_expert_zeros;
        let proj_count = proj_names.len();
        let pairs_per_expert = proj_count;
        debug_assert_eq!(total_pairs, num_experts * pairs_per_expert);

        // GROUP-MAJOR layout for ALL THREE (qweight + scales + qzeros).
        // Layout: out[g * total_n + e * n_per_expert + c]. Marlin's
        // CUDA kernel uses prob_n_full as the stride for both b_gl_stride
        // and s_gl_stride — i.e. the kernel walks `prob_n_full` cols
        // between K-tile rows / between groups. With group-major,
        // group g of expert e is at byte offset
        // `(g * total_n + e * n_per) * sizeof`, exactly where the
        // kernel reads when given pointer offset = `e * n_per * sizeof`
        // and stride = `total_n`.
        let _ = pairs_per_expert; // silence unused-var
        let mut qw_acc: Vec<i32> = Vec::with_capacity(qw_rows * total_n);
        for r in 0..qw_rows {
            for pair_idx in 0..total_pairs {
                let (data, cols) = &qw_parts[pair_idx];
                qw_acc.extend_from_slice(&data[r * cols..r * cols + cols]);
            }
        }
        let mut sc_acc: Vec<f32> = Vec::with_capacity(sc_rows * total_n_scales);
        for r in 0..sc_rows {
            for pair_idx in 0..total_pairs {
                let (data, cols) = &sc_parts[pair_idx];
                sc_acc.extend_from_slice(&data[r * cols..r * cols + cols]);
            }
        }
        let mut qz_acc: Vec<i32> = Vec::with_capacity(qz_rows * total_n_zeros);
        for r in 0..qz_rows {
            for pair_idx in 0..total_pairs {
                let (data, cols) = &qz_parts[pair_idx];
                qz_acc.extend_from_slice(&data[r * cols..r * cols + cols]);
            }
        }

        // Drop intermediates eagerly — these can be ~hundreds of MB on
        // 30B-A3B and we still have to feed B::load_gptq below.
        drop(qw_parts);
        drop(sc_parts);
        drop(qz_parts);

        let store = B::load_gptq(
            &qw_acc,
            &sc_acc,
            &qz_acc,
            g_idx_first.as_deref(),
            qcfg.bits,
            qcfg.group_size,
            k_shared,
            total_n,
        )?;
        Ok((store, n_per_expert, k_shared))
    }
}

impl<B: Backend> WeightLoader<B> for NativeSafetensorsLoader<B> {
    fn load_tensor(&self, name: &str) -> Result<B::Buffer> {
        // Route through `from_weight_bytes` so fp16-preferring backends can
        // materialise big tensors (embed table) directly as half-precision
        // without the transient f32 Vec. Tiny tensors (norm weights) still
        // end up as f32 because backends size-threshold inside the override.
        let (raw, src_dtype, _) = self.read_bytes_typed(name)?;
        Ok(B::from_weight_bytes(raw, src_dtype))
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
        if let Some(prefix) = name.strip_suffix("qkv_proj") {
            let parts = [
                format!("{prefix}q_proj"),
                format!("{prefix}k_proj"),
                format!("{prefix}v_proj"),
            ];
            if parts.iter().all(|p| self.has(&format!("{p}.qweight"))) {
                return self.load_gptq_linear_fused(&parts);
            }
        }
        if let Some(prefix) = name.strip_suffix("gate_up_proj") {
            let parts = [format!("{prefix}gate_proj"), format!("{prefix}up_proj")];
            if parts.iter().all(|p| self.has(&format!("{p}.qweight"))) {
                return self.load_gptq_linear_fused(&parts);
            }
        }

        // Direct fused `<name>.weight` next. Load straight from raw bytes
        // so fp16-preferring backends can skip the f32 Vec intermediate.
        let direct = format!("{name}.weight");
        if self.has(&direct) {
            let (raw, src_dtype, shape) = self.read_bytes_typed(&direct)?;
            if shape.len() != 2 {
                return Err(FerrumError::model(format!(
                    "linear '{name}': expected 2D weight, got {shape:?}"
                )));
            }
            let weight = B::from_weight_bytes(raw, src_dtype);
            return Ok(Box::new(DenseLinear::<B>::from_buffer(
                weight, shape[0], shape[1],
            )));
        }

        // Llama-family fusion shims: synthesise qkv_proj / gate_up_proj from
        // split q_proj+k_proj+v_proj / gate_proj+up_proj if present. The cat
        // happens at the byte level so fused-weight memory is the same size
        // as the per-part weights — no expansion to f32.
        if let Some(prefix) = name.strip_suffix("qkv_proj") {
            let parts = [
                format!("{prefix}q_proj.weight"),
                format!("{prefix}k_proj.weight"),
                format!("{prefix}v_proj.weight"),
            ];
            if parts.iter().all(|p| self.has(p)) {
                let (bytes, dtype, (rows, cols)) = self.cat_rows_bytes(&parts)?;
                let weight = B::from_weight_bytes(&bytes, dtype);
                return Ok(Box::new(DenseLinear::<B>::from_buffer(weight, rows, cols)));
            }
        }
        if let Some(prefix) = name.strip_suffix("gate_up_proj") {
            let parts = [
                format!("{prefix}gate_proj.weight"),
                format!("{prefix}up_proj.weight"),
            ];
            if parts.iter().all(|p| self.has(p)) {
                let (bytes, dtype, (rows, cols)) = self.cat_rows_bytes(&parts)?;
                let weight = B::from_weight_bytes(&bytes, dtype);
                return Ok(Box::new(DenseLinear::<B>::from_buffer(weight, rows, cols)));
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

        // desc_act=true detection. AutoGPTQ writes g_idx[k] = k/group_size
        // for desc_act=false (trivial). Non-monotonic values → act-order.
        let is_desc_act = g_idx.as_ref().map_or(false, |gx| {
            !gx.iter()
                .enumerate()
                .all(|(i, &g)| g == (i as i32) / qcfg.group_size as i32)
        });

        // Act-order GPTQ. CUDA backend has perm-aware Marlin (load_gptq
        // builds perm = argsort(g_idx) + permutes qweight rows at load;
        // gemm_gptq gathers input columns before the standard Marlin call).
        // CPU/Metal still need the dequant→DenseLinear fallback.
        #[cfg(not(feature = "cuda"))]
        if is_desc_act {
            let dequant_f32 = dequantize_gptq_with_g_idx(
                &qweight,
                &scales_f32,
                &qzeros,
                g_idx.as_ref().expect("desc_act=true requires g_idx"),
                qcfg.group_size,
                in_features,
                out_features,
            );
            let mut linear =
                crate::dense::DenseLinear::<B>::from_rows(&dequant_f32, out_features, in_features);
            let bias_key = format!("{name}.bias");
            if self.has(&bias_key) {
                let (bias, _) = self.read_f32(&bias_key)?;
                linear = linear.with_bias(B::from_slice(&bias));
            }
            tracing::info!(
                "GPTQ load (desc_act dequant→DenseLinear, non-cuda): name={name} K={in_features} N={out_features}"
            );
            return Ok(Box::new(linear));
        }
        #[cfg(feature = "cuda")]
        let _ = is_desc_act; // CUDA: g_idx threaded through to GptqLinear below
        if sc_shape.len() != 2 || sc_shape[1] != out_features {
            return Err(FerrumError::model(format!(
                "'{name}.scales' {sc_shape:?} incompatible with qweight {qw_shape:?}"
            )));
        }

        let mut linear = GptqLinear::<B>::from_raw(
            &qweight,
            &scales_f32,
            &qzeros,
            g_idx.as_deref(),
            qcfg.bits,
            qcfg.group_size,
            in_features,
            out_features,
        )?;

        // Bias (Qwen2.5 attention projections, some Llama variants).
        let bias_key = format!("{name}.bias");
        if self.has(&bias_key) {
            let (bias, bias_shape) = self.read_f32(&bias_key)?;
            if bias_shape != [out_features] {
                return Err(FerrumError::model(format!(
                    "'{bias_key}' {bias_shape:?} != [{out_features}]"
                )));
            }
            linear = linear.with_bias(&bias);
        }
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
            if g_idx.is_none() && self.has(&format!("{p}.g_idx")) {
                g_idx = Some(self.read_i32(&format!("{p}.g_idx"))?.0);
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

        // desc_act detection — same as load_gptq_linear. Q/K/V (or
        // gate/up) share K, so g_idx from first part covers all.
        let is_desc_act = g_idx.as_ref().map_or(false, |gx| {
            !gx.iter()
                .enumerate()
                .all(|(i, &g)| g == (i as i32) / qcfg.group_size as i32)
        });
        // CUDA: perm-aware Marlin via load_gptq. CPU/Metal: dequant→Dense.
        #[cfg(not(feature = "cuda"))]
        if is_desc_act {
            let dequant_f32 = dequantize_gptq_with_g_idx(
                &qw_acc,
                &sc_acc,
                &qz_acc,
                g_idx.as_ref().expect("desc_act=true requires g_idx"),
                qcfg.group_size,
                in_features,
                out_features,
            );
            let mut linear =
                crate::dense::DenseLinear::<B>::from_rows(&dequant_f32, out_features, in_features);
            let mut bias_acc: Vec<f32> = Vec::new();
            let mut any_bias = false;
            for p in parts {
                let bk = format!("{p}.bias");
                if self.has(&bk) {
                    any_bias = true;
                    bias_acc.extend_from_slice(&self.read_f32(&bk)?.0);
                } else if any_bias {
                    return Err(FerrumError::model(format!(
                        "GPTQ fusion bias mix: '{p}' has no bias but earlier part did"
                    )));
                }
            }
            if any_bias {
                linear = linear.with_bias(B::from_slice(&bias_acc));
            }
            tracing::info!(
                "GPTQ fused load (desc_act dequant→DenseLinear, non-cuda): K={in_features} N={out_features} parts={}",
                parts.len()
            );
            return Ok(Box::new(linear));
        }
        #[cfg(feature = "cuda")]
        let _ = is_desc_act;

        let mut linear = GptqLinear::<B>::from_raw(
            &qw_acc,
            &sc_acc,
            &qz_acc,
            g_idx.as_deref(),
            qcfg.bits,
            qcfg.group_size,
            in_features,
            out_features,
        )?;

        // Biases: concatenate `<part>.bias` across parts in the same order as
        // qweights. All-or-none; if any part has a bias, all must.
        let bias_keys: Vec<String> = parts.iter().map(|p| format!("{p}.bias")).collect();
        let any = bias_keys.iter().any(|k| self.has(k));
        let all = bias_keys.iter().all(|k| self.has(k));
        if any && !all {
            return Err(FerrumError::model(
                "GPTQ fusion: inconsistent bias presence across parts".to_string(),
            ));
        }
        if all {
            let mut fused: Vec<f32> = Vec::with_capacity(out_features);
            for k in &bias_keys {
                let (b, _) = self.read_f32(k)?;
                fused.extend_from_slice(&b);
            }
            if fused.len() != out_features {
                return Err(FerrumError::model(format!(
                    "GPTQ fusion bias length {} != out_features {out_features}",
                    fused.len()
                )));
            }
            linear = linear.with_bias(&fused);
        }
        Ok(Box::new(linear))
    }

    /// Read each name, assert shape width matches, concatenate along dim 0.
    /// Kept for diagnostic / fallback paths; DenseLinear fusion prefers the
    /// byte-level `cat_rows_bytes` above.
    #[allow(dead_code)]
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

/// Dequantise GPTQ INT4 weights with desc_act=true (act-order) g_idx and
/// return original-order f32 weights laid out `[N, K]` row-major (matches
/// `DenseLinear::from_rows`).
///
/// Key insight: in AutoGPTQ desc_act format, qweight rows are NOT
/// permuted from original-K order. The act-order trick is encoded purely
/// in `g_idx[k]` — which records the QUANTISATION GROUP (not column
/// position) chosen for disk row k. Different rows that originally
/// belonged to far-apart positions can share a group via g_idx.
///
/// Verified against vLLM's exllama path (gptq.py:368): for desc_act it
/// runs `g_idx ← argsort(g_idx)` then `gptq_shuffle(qweight, g_idx)`,
/// which physically reorders qweight by argsort and gathers x by argsort
/// at GEMM. Net effect: y[n] = Σⱼ x[j] · dequant(qweight[j, n],
/// scales[g_idx_orig[j], n], qzeros[g_idx_orig[j], n]).
/// → disk_k IS original_k; only the (scale, zero) LOOKUP differs.
#[cfg(not(feature = "cuda"))]
fn dequantize_gptq_with_g_idx(
    qweight: &[i32], // [K/8, N] packed int4
    scales: &[f32],  // [num_groups, N]
    qzeros: &[i32],  // [num_groups, N/8] packed int4
    g_idx: &[i32],   // [K]
    _group_size: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    debug_assert_eq!(g_idx.len(), k);

    // Output: [N, K] row-major → out[col * k + k_idx] = value.
    let mut w = vec![0.0f32; n * k];
    let packed_rows = k / 8;
    for pr in 0..packed_rows {
        for col in 0..n {
            let packed = qweight[pr * n + col] as u32;
            for bi in 0..8 {
                let ki = pr * 8 + bi;
                let q = ((packed >> (bi * 4)) & 0xF) as i32;
                let g = g_idx[ki] as usize;
                let scale = scales[g * n + col];
                let z_packed = qzeros[g * (n / 8) + (col / 8)] as u32;
                let zero = (((z_packed >> ((col % 8) * 4)) & 0xF) as i32) + 1;
                w[col * k + ki] = (q - zero) as f32 * scale;
            }
        }
    }
    w
}

fn dtype_to_f32(dtype: Dtype, raw: &[u8]) -> Result<Vec<f32>> {
    match dtype {
        Dtype::F32 => {
            // Bulk memcpy from LE-stored bytes (safetensors is LE; we're
            // on x86_64 LE). Per-element from_le_bytes was the bottleneck
            // for stacked-MoE load (4-5 ms per call * 384 calls/layer *
            // 48 layers = ~80 sec just for f32 reads).
            debug_assert_eq!(raw.len() % 4, 0);
            let n = raw.len() / 4;
            let mut out = Vec::<f32>::with_capacity(n);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    raw.as_ptr(),
                    out.as_mut_ptr() as *mut u8,
                    raw.len(),
                );
                out.set_len(n);
            }
            Ok(out)
        }
        Dtype::F16 => {
            debug_assert_eq!(raw.len() % 2, 0);
            let n = raw.len() / 2;
            // Reinterpret raw bytes as f16, then convert. This avoids
            // the per-element from_le_bytes byte-array construction.
            let mut tmp = Vec::<f16>::with_capacity(n);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    raw.as_ptr(),
                    tmp.as_mut_ptr() as *mut u8,
                    raw.len(),
                );
                tmp.set_len(n);
            }
            let mut out = Vec::with_capacity(n);
            for h in &tmp {
                out.push(h.to_f32());
            }
            Ok(out)
        }
        Dtype::BF16 => {
            debug_assert_eq!(raw.len() % 2, 0);
            let n = raw.len() / 2;
            let mut tmp = Vec::<bf16>::with_capacity(n);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    raw.as_ptr(),
                    tmp.as_mut_ptr() as *mut u8,
                    raw.len(),
                );
                tmp.set_len(n);
            }
            let mut out = Vec::with_capacity(n);
            for h in &tmp {
                out.push(h.to_f32());
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
    // AutoGPTQ / gptq-for-llama format: separate quantize_config.json.
    let p = dir.join("quantize_config.json");
    if p.exists() {
        let data =
            std::fs::read_to_string(&p).map_err(|e| FerrumError::io(format!("read {p:?}: {e}")))?;
        let qc: QuantConfig = serde_json::from_str(&data)
            .map_err(|e| FerrumError::serialization(format!("parse quantize_config.json: {e}")))?;
        return Ok(Some(qc));
    }
    // Qwen GPTQ / transformers-style: embedded in config.json under
    // "quantization_config": { "quant_method": "gptq", "bits": 4, ... }.
    let cfg = dir.join("config.json");
    if cfg.exists() {
        let data = std::fs::read_to_string(&cfg)
            .map_err(|e| FerrumError::io(format!("read {cfg:?}: {e}")))?;
        let root: serde_json::Value = serde_json::from_str(&data)
            .map_err(|e| FerrumError::serialization(format!("parse config.json: {e}")))?;
        if let Some(qc_val) = root.get("quantization_config") {
            // The embedded block has "quant_method" (not "method"); remap.
            let method = qc_val
                .get("quant_method")
                .and_then(|v| v.as_str())
                .unwrap_or("none");
            let method = match method.to_lowercase().as_str() {
                "gptq" => QuantMethod::Gptq,
                "awq" => QuantMethod::Awq,
                "gguf" => QuantMethod::Gguf,
                _ => QuantMethod::None,
            };
            let bits = qc_val.get("bits").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
            let group_size = qc_val
                .get("group_size")
                .and_then(|v| v.as_i64())
                .unwrap_or(128)
                .max(0) as usize;
            let desc_act = qc_val
                .get("desc_act")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let sym = qc_val.get("sym").and_then(|v| v.as_bool()).unwrap_or(false);
            if method != QuantMethod::None {
                return Ok(Some(QuantConfig {
                    method,
                    bits,
                    group_size,
                    desc_act,
                    sym,
                }));
            }
        }
    }
    Ok(None)
}
