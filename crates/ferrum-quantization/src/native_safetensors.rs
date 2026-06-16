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

use ferrum_kernels::backend::{Backend, BackendQuantMarlin, SrcDtype};
use ferrum_kernels::LinearMetadata;
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
pub struct NativeSafetensorsLoader<B: Backend + BackendQuantMarlin> {
    /// All shards keyed by file; each tensor's name maps to its shard here.
    shards: Vec<Shard>,
    /// Name → shard index. Populated once at construction.
    index: HashMap<String, usize>,
    /// Optional `quantize_config.json` contents.
    quant_config: Option<QuantConfig>,
    _m: std::marker::PhantomData<B>,
}

impl<B: Backend + BackendQuantMarlin> NativeSafetensorsLoader<B> {
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

    /// Concatenate optional projection biases in the same order as split
    /// weight fusion. Biases must be all-or-none; silently dropping one part's
    /// bias corrupts Qwen2.5 attention logits.
    fn cat_optional_biases(
        &self,
        weight_names: &[String],
        out_features: usize,
    ) -> Result<Option<Vec<f32>>> {
        let bias_names: Vec<String> = weight_names
            .iter()
            .map(|name| {
                name.strip_suffix(".weight")
                    .map(|stem| format!("{stem}.bias"))
                    .unwrap_or_else(|| format!("{name}.bias"))
            })
            .collect();
        let any_bias = bias_names.iter().any(|name| self.has(name));
        if !any_bias {
            return Ok(None);
        }
        if let Some(missing) = bias_names.iter().find(|name| !self.has(name)) {
            return Err(FerrumError::model(format!(
                "dense fusion bias mix: '{missing}' missing while another fused part has bias"
            )));
        }
        let mut fused = Vec::new();
        for name in &bias_names {
            let (bias, shape) = self.read_f32(name)?;
            if shape.len() != 1 {
                return Err(FerrumError::model(format!(
                    "dense fusion bias '{name}': expected 1D, got {shape:?}"
                )));
            }
            fused.extend_from_slice(&bias);
        }
        if fused.len() != out_features {
            return Err(FerrumError::model(format!(
                "dense fusion bias length {} != out_features {out_features}",
                fused.len()
            )));
        }
        Ok(Some(fused))
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
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, bytes.len());
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
    ) -> Result<(
        std::sync::Arc<dyn ferrum_kernels::MarlinExpertStack<B>>,
        usize,
        usize,
    )> {
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
                } else if qw_sh[0] != qw_rows || sc_sh[0] != sc_rows || qz_sh[0] != qz_rows {
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
                            if prev.len() != gx.len() || prev.iter().zip(&gx).any(|(a, b)| a != b) {
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

        let proj_count = proj_names.len();
        let pairs_per_expert = proj_count;
        debug_assert_eq!(total_pairs, num_experts * pairs_per_expert);

        // PER-EXPERT layout: build num_experts independent
        // `[K/8, n_per_expert]` qweight tiles + scales + qzeros, each
        // a row-major concat of the proj_names within that expert.
        // Hand them to `B::load_gptq_stacked` which repacks PER-EXPERT
        // and concats the resulting Marlin-format tiles into one
        // contiguous buffer. Each expert's packed bytes are then
        // contiguous, so the offset GEMM dispatches correctly via
        // pointer arithmetic alone.
        //
        // Without per-expert repack, a single concat-then-repack of
        // the stacked tile mangles per-expert tile boundaries (Marlin
        // permutes in K-tile-major order across the whole tile).
        let mut per_expert_qw: Vec<Vec<i32>> = Vec::with_capacity(num_experts);
        let mut per_expert_sc: Vec<Vec<f32>> = Vec::with_capacity(num_experts);
        let mut per_expert_qz: Vec<Vec<i32>> = Vec::with_capacity(num_experts);
        for e in 0..num_experts {
            let mut qw: Vec<i32> = Vec::with_capacity(qw_rows * n_per_expert);
            let mut sc: Vec<f32> = Vec::with_capacity(sc_rows * n_per_expert_scales);
            let mut qz: Vec<i32> = Vec::with_capacity(qz_rows * n_per_expert_zeros);
            for r in 0..qw_rows {
                for j in 0..pairs_per_expert {
                    let pair_idx = e * pairs_per_expert + j;
                    let (data, cols) = &qw_parts[pair_idx];
                    qw.extend_from_slice(&data[r * cols..(r + 1) * cols]);
                }
            }
            for r in 0..sc_rows {
                for j in 0..pairs_per_expert {
                    let pair_idx = e * pairs_per_expert + j;
                    let (data, cols) = &sc_parts[pair_idx];
                    sc.extend_from_slice(&data[r * cols..(r + 1) * cols]);
                }
            }
            for r in 0..qz_rows {
                for j in 0..pairs_per_expert {
                    let pair_idx = e * pairs_per_expert + j;
                    let (data, cols) = &qz_parts[pair_idx];
                    qz.extend_from_slice(&data[r * cols..(r + 1) * cols]);
                }
            }
            per_expert_qw.push(qw);
            per_expert_sc.push(sc);
            per_expert_qz.push(qz);
        }

        // Drop the original part buffers — we own copies in per_expert_*.
        drop(qw_parts);
        drop(sc_parts);
        drop(qz_parts);

        let qw_refs: Vec<&[i32]> = per_expert_qw.iter().map(|v| v.as_slice()).collect();
        let sc_refs: Vec<&[f32]> = per_expert_sc.iter().map(|v| v.as_slice()).collect();
        let qz_refs: Vec<&[i32]> = per_expert_qz.iter().map(|v| v.as_slice()).collect();

        let is_desc_act = validate_gptq_g_idx(
            "stacked GPTQ experts",
            qcfg,
            g_idx_first.as_deref(),
            k_shared,
        )?;
        #[cfg(feature = "cuda")]
        if is_desc_act {
            validate_cuda_marlin_desc_act_g_idx(
                "stacked GPTQ experts",
                qcfg,
                g_idx_first
                    .as_deref()
                    .expect("desc_act=true requires g_idx"),
                k_shared,
            )?;
        }
        #[cfg(not(feature = "cuda"))]
        let _ = is_desc_act;

        let store = B::load_gptq_stacked(
            &qw_refs,
            &sc_refs,
            &qz_refs,
            g_idx_first.as_deref(),
            qcfg.bits,
            qcfg.group_size,
            k_shared,
            n_per_expert,
        )?;
        Ok((store, n_per_expert, k_shared))
    }
}

impl<B: Backend + BackendQuantMarlin> WeightLoader<B> for NativeSafetensorsLoader<B> {
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
            return Ok(Box::new(
                DenseLinear::<B>::from_buffer(weight, shape[0], shape[1])
                    .with_metadata(LinearMetadata::from_name(name)),
            ));
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
                let mut linear = DenseLinear::<B>::from_buffer(weight, rows, cols).with_metadata(
                    LinearMetadata::from_fused_names(parts.iter().map(String::as_str)),
                );
                if let Some(bias) = self.cat_optional_biases(&parts, rows)? {
                    linear = linear.with_bias(B::from_slice(&bias));
                }
                return Ok(Box::new(linear));
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
                let mut linear = DenseLinear::<B>::from_buffer(weight, rows, cols).with_metadata(
                    LinearMetadata::from_fused_names(parts.iter().map(String::as_str)),
                );
                if let Some(bias) = self.cat_optional_biases(&parts, rows)? {
                    linear = linear.with_bias(B::from_slice(&bias));
                }
                return Ok(Box::new(linear));
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

impl<B: Backend + BackendQuantMarlin> NativeSafetensorsLoader<B> {
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

        let is_desc_act = validate_gptq_g_idx(name, qcfg, g_idx.as_deref(), in_features)?;
        trace_gptq_g_idx_if_requested(name, qcfg, g_idx.as_deref(), in_features, is_desc_act);
        trace_gptq_qzeros_if_requested(name, qcfg, &qzeros, out_features);

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
                crate::dense::DenseLinear::<B>::from_rows(&dequant_f32, out_features, in_features)
                    .with_metadata(LinearMetadata::from_name(name));
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
        if is_desc_act {
            validate_cuda_marlin_desc_act_g_idx(
                name,
                qcfg,
                g_idx.as_deref().expect("desc_act=true requires g_idx"),
                in_features,
            )?;
        }
        if sc_shape.len() != 2 || sc_shape[1] != out_features {
            return Err(FerrumError::model(format!(
                "'{name}.scales' {sc_shape:?} incompatible with qweight {qw_shape:?}"
            )));
        }

        // Read optional bias FIRST (Qwen2.5 attention projections, some
        // Llama variants). Phase 3e/2: load_gptq takes the bias eagerly
        // because the boxed Linear bakes it in.
        let bias_key = format!("{name}.bias");
        let bias_vec = if self.has(&bias_key) {
            let (bias, bias_shape) = self.read_f32(&bias_key)?;
            if bias_shape != [out_features] {
                return Err(FerrumError::model(format!(
                    "'{bias_key}' {bias_shape:?} != [{out_features}]"
                )));
            }
            Some(bias)
        } else {
            None
        };

        let linear = GptqLinear::<B>::from_raw_with_metadata(
            &qweight,
            &scales_f32,
            &qzeros,
            g_idx.as_deref(),
            bias_vec.as_deref(),
            qcfg.bits,
            qcfg.group_size,
            in_features,
            out_features,
            LinearMetadata::from_name(name),
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
        let mut g_idx_presence: Vec<(String, bool)> = Vec::with_capacity(parts.len());
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

            let g_key = format!("{p}.g_idx");
            if self.has(&g_key) {
                let (gx, gx_shape) = self.read_i32(&g_key)?;
                if gx_shape != [qw_rows * 8] {
                    return Err(FerrumError::model(format!(
                        "GPTQ fusion '{p}': g_idx shape {gx_shape:?} incompatible with K={}",
                        qw_rows * 8
                    )));
                }
                match &g_idx {
                    None => g_idx = Some(gx),
                    Some(prev) => {
                        if prev.len() != gx.len() || prev.iter().zip(&gx).any(|(a, b)| a != b) {
                            return Err(FerrumError::model(format!(
                                "GPTQ fusion '{p}': g_idx mismatch with first part; \
                                 fused qkv/gate_up requires identical act-order across parts"
                            )));
                        }
                    }
                }
                g_idx_presence.push((p.clone(), true));
            } else {
                g_idx_presence.push((p.clone(), false));
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

        if g_idx.is_some() {
            let missing = g_idx_presence
                .iter()
                .filter_map(|(part, present)| (!present).then_some(part.as_str()))
                .collect::<Vec<_>>();
            if !missing.is_empty() {
                return Err(FerrumError::model(format!(
                    "GPTQ fusion requires all parts to carry g_idx when any part does; \
                     missing g_idx for {missing:?}"
                )));
            }
        }
        let fused_name = format!("GPTQ fusion {}", parts.join("+"));
        let is_desc_act = validate_gptq_g_idx(&fused_name, qcfg, g_idx.as_deref(), in_features)?;
        trace_gptq_g_idx_if_requested(
            &fused_name,
            qcfg,
            g_idx.as_deref(),
            in_features,
            is_desc_act,
        );
        trace_gptq_qzeros_if_requested(&fused_name, qcfg, &qz_acc, out_features);
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
                crate::dense::DenseLinear::<B>::from_rows(&dequant_f32, out_features, in_features)
                    .with_metadata(LinearMetadata::from_fused_names(
                        parts.iter().map(String::as_str),
                    ));
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
        if is_desc_act {
            validate_cuda_marlin_desc_act_g_idx(
                &fused_name,
                qcfg,
                g_idx.as_deref().expect("desc_act=true requires g_idx"),
                in_features,
            )?;
        }

        // Biases: concatenate `<part>.bias` across parts in the same
        // order as qweights. All-or-none; if any part has a bias, all
        // must. Phase 3e/2: read first, pass into load_gptq.
        let bias_keys: Vec<String> = parts.iter().map(|p| format!("{p}.bias")).collect();
        let any = bias_keys.iter().any(|k| self.has(k));
        let all = bias_keys.iter().all(|k| self.has(k));
        if any && !all {
            return Err(FerrumError::model(
                "GPTQ fusion: inconsistent bias presence across parts".to_string(),
            ));
        }
        let fused_bias = if all {
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
            Some(fused)
        } else {
            None
        };

        let linear = GptqLinear::<B>::from_raw_with_metadata(
            &qw_acc,
            &sc_acc,
            &qz_acc,
            g_idx.as_deref(),
            fused_bias.as_deref(),
            qcfg.bits,
            qcfg.group_size,
            in_features,
            out_features,
            LinearMetadata::from_fused_names(parts.iter().map(String::as_str)),
        )?;

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

fn gptq_g_idx_is_desc_act(g_idx: &[i32], group_size: usize) -> bool {
    g_idx
        .iter()
        .enumerate()
        .any(|(i, &g)| g != (i as i32) / group_size as i32)
}

fn trace_gptq_g_idx_if_requested(
    name: &str,
    qcfg: &QuantConfig,
    g_idx: Option<&[i32]>,
    in_features: usize,
    is_desc_act: bool,
) {
    if !gptq_gidx_trace_enabled() {
        return;
    }

    let Some(g_idx) = g_idx else {
        tracing::info!(
            "GPTQ g_idx trace: name={name} K={in_features} desc_act_config={} no_g_idx",
            qcfg.desc_act
        );
        return;
    };
    if qcfg.group_size == 0 {
        tracing::info!("GPTQ g_idx trace: name={name} K={in_features} group_size=0 invalid");
        return;
    }

    let expected_groups = in_features.div_ceil(qcfg.group_size);
    let mut counts = vec![0usize; expected_groups];
    for &group in g_idx {
        if group >= 0 {
            let group = group as usize;
            if group < counts.len() {
                counts[group] += 1;
            }
        }
    }
    let nonzero_groups = counts.iter().filter(|&&count| count > 0).count();
    let count_min = counts.iter().copied().min().unwrap_or(0);
    let count_max = counts.iter().copied().max().unwrap_or(0);
    let unbalanced_groups = counts
        .iter()
        .filter(|&&count| count != qcfg.group_size)
        .count();
    let preview_len = g_idx.len().min(16);
    tracing::info!(
        "GPTQ g_idx trace: name={name} desc_act={is_desc_act} K={in_features} \
         group_size={} groups={expected_groups} nonzero_groups={nonzero_groups} \
         count_min={count_min} count_max={count_max} unbalanced_groups={unbalanced_groups} \
         first{preview_len}={:?}",
        qcfg.group_size,
        &g_idx[..preview_len]
    );
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GptqQzeroStats {
    words: usize,
    total_codes: usize,
    min_code: u8,
    max_code: u8,
    code7_count: usize,
    histogram: [usize; 16],
}

impl GptqQzeroStats {
    fn all_code7(&self) -> bool {
        self.total_codes > 0 && self.code7_count == self.total_codes
    }
}

fn gptq_qzero_stats(qzeros: &[i32]) -> GptqQzeroStats {
    let mut histogram = [0usize; 16];
    for &word in qzeros {
        let word = word as u32;
        for nibble in 0..8 {
            let code = ((word >> (nibble * 4)) & 0xF) as usize;
            histogram[code] += 1;
        }
    }

    let mut min_code = 0u8;
    let mut max_code = 0u8;
    let mut found = false;
    for (code, &count) in histogram.iter().enumerate() {
        if count == 0 {
            continue;
        }
        if !found {
            min_code = code as u8;
            found = true;
        }
        max_code = code as u8;
    }

    GptqQzeroStats {
        words: qzeros.len(),
        total_codes: qzeros.len() * 8,
        min_code,
        max_code,
        code7_count: histogram[7],
        histogram,
    }
}

fn trace_gptq_qzeros_if_requested(
    name: &str,
    qcfg: &QuantConfig,
    qzeros: &[i32],
    out_features: usize,
) {
    if !gptq_gidx_trace_enabled() {
        return;
    }

    let stats = gptq_qzero_stats(qzeros);
    let expected_words_per_group = out_features.div_ceil(8);
    tracing::info!(
        "GPTQ qzeros trace: name={name} sym={} N={out_features} \
         expected_words_per_group={expected_words_per_group} words={} total_codes={} \
         min_code={} max_code={} code7={}/{} all_code7={} histogram={:?}",
        qcfg.sym,
        stats.words,
        stats.total_codes,
        stats.min_code,
        stats.max_code,
        stats.code7_count,
        stats.total_codes,
        stats.all_code7(),
        stats.histogram
    );
}

fn gptq_gidx_trace_enabled() -> bool {
    runtime_snapshot_value("FERRUM_GPTQ_GIDX_TRACE").as_deref() == Some("1")
}

fn runtime_snapshot_value(key: &str) -> Option<String> {
    ferrum_types::active_runtime_snapshot()
        .entries
        .iter()
        .find(|entry| entry.key == key)
        .map(|entry| entry.effective_value.clone())
}

fn validate_gptq_g_idx(
    name: &str,
    qcfg: &QuantConfig,
    g_idx: Option<&[i32]>,
    in_features: usize,
) -> Result<bool> {
    if qcfg.desc_act && g_idx.is_none() {
        return Err(FerrumError::model(format!(
            "{name}: quantize_config desc_act=true but no g_idx tensor was found"
        )));
    }

    let Some(g_idx) = g_idx else {
        return Ok(false);
    };
    if qcfg.group_size == 0 {
        return Err(FerrumError::model(format!(
            "{name}: GPTQ g_idx present but group_size is 0"
        )));
    }
    if g_idx.len() != in_features {
        return Err(FerrumError::model(format!(
            "{name}: g_idx length {} must match K={in_features}",
            g_idx.len()
        )));
    }
    let expected_groups = in_features.div_ceil(qcfg.group_size);
    for (idx, &group) in g_idx.iter().enumerate() {
        if group < 0 || group as usize >= expected_groups {
            return Err(FerrumError::model(format!(
                "{name}: g_idx[{idx}]={group} outside expected group range 0..{}",
                expected_groups.saturating_sub(1)
            )));
        }
    }
    Ok(gptq_g_idx_is_desc_act(g_idx, qcfg.group_size))
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
fn validate_cuda_marlin_desc_act_g_idx(
    name: &str,
    qcfg: &QuantConfig,
    g_idx: &[i32],
    in_features: usize,
) -> Result<()> {
    if qcfg.group_size == 0 {
        return Err(FerrumError::model(format!(
            "{name}: CUDA Marlin desc_act requires non-zero group_size"
        )));
    }
    if in_features % qcfg.group_size != 0 {
        return Err(FerrumError::unsupported(format!(
            "{name}: CUDA Marlin desc_act requires K={in_features} to be divisible by \
             group_size={}",
            qcfg.group_size
        )));
    }

    let expected_groups = in_features / qcfg.group_size;
    let mut counts = vec![0usize; expected_groups];
    for (idx, &group) in g_idx.iter().enumerate() {
        if group < 0 || group as usize >= expected_groups {
            return Err(FerrumError::model(format!(
                "{name}: g_idx[{idx}]={group} outside expected group range 0..{}",
                expected_groups.saturating_sub(1)
            )));
        }
        counts[group as usize] += 1;
    }

    if let Some((group, count)) = counts
        .iter()
        .copied()
        .enumerate()
        .find(|&(_, count)| count != qcfg.group_size)
    {
        return Err(FerrumError::unsupported(format!(
            "{name}: CUDA Marlin desc_act requires balanced full groups; \
             group {group} has {count} rows, expected {}",
            qcfg.group_size
        )));
    }

    Ok(())
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
                std::ptr::copy_nonoverlapping(raw.as_ptr(), out.as_mut_ptr() as *mut u8, raw.len());
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
                std::ptr::copy_nonoverlapping(raw.as_ptr(), tmp.as_mut_ptr() as *mut u8, raw.len());
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
                std::ptr::copy_nonoverlapping(raw.as_ptr(), tmp.as_mut_ptr() as *mut u8, raw.len());
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

#[cfg(test)]
mod tests {
    use super::*;

    fn gptq_config(desc_act: bool) -> QuantConfig {
        QuantConfig {
            method: QuantMethod::Gptq,
            bits: 4,
            group_size: 2,
            desc_act,
            sym: true,
        }
    }

    #[test]
    fn validate_gptq_g_idx_requires_tensor_when_desc_act_configured() {
        let err = validate_gptq_g_idx("proj", &gptq_config(true), None, 4)
            .unwrap_err()
            .to_string();

        assert!(err.contains("desc_act=true"));
        assert!(err.contains("no g_idx"));
    }

    #[test]
    fn validate_gptq_g_idx_accepts_trivial_non_desc_act_order() {
        let is_desc_act =
            validate_gptq_g_idx("proj", &gptq_config(false), Some(&[0, 0, 1, 1]), 4).unwrap();

        assert!(!is_desc_act);
    }

    #[test]
    fn validate_gptq_g_idx_detects_nontrivial_act_order() {
        let is_desc_act =
            validate_gptq_g_idx("proj", &gptq_config(false), Some(&[1, 1, 0, 0]), 4).unwrap();

        assert!(is_desc_act);
    }

    #[test]
    fn validate_gptq_g_idx_rejects_invalid_shape_and_group() {
        let short = validate_gptq_g_idx("proj", &gptq_config(false), Some(&[0, 0, 1]), 4)
            .unwrap_err()
            .to_string();
        assert!(short.contains("must match K=4"));

        let out_of_range = validate_gptq_g_idx("proj", &gptq_config(false), Some(&[0, 0, 2, 1]), 4)
            .unwrap_err()
            .to_string();
        assert!(out_of_range.contains("outside expected group range"));
    }

    #[test]
    fn validate_cuda_marlin_desc_act_accepts_balanced_full_groups() {
        validate_cuda_marlin_desc_act_g_idx("proj", &gptq_config(true), &[1, 1, 0, 0], 4).unwrap();
    }

    #[test]
    fn validate_cuda_marlin_desc_act_rejects_unbalanced_groups() {
        let err = validate_cuda_marlin_desc_act_g_idx("proj", &gptq_config(true), &[0, 0, 0, 1], 4)
            .unwrap_err()
            .to_string();

        assert!(err.contains("balanced full groups"));
        assert!(err.contains("group 0 has 3 rows"));
    }

    #[test]
    fn validate_cuda_marlin_desc_act_rejects_partial_last_group() {
        let err = validate_cuda_marlin_desc_act_g_idx("proj", &gptq_config(true), &[0, 0, 1], 3)
            .unwrap_err()
            .to_string();

        assert!(err.contains("K=3"));
        assert!(err.contains("group_size=2"));
    }

    #[test]
    fn qzero_stats_detects_symmetric_code7_packing() {
        let stats = gptq_qzero_stats(&[0x7777_7777u32 as i32, 0x7777_7777u32 as i32]);

        assert_eq!(stats.words, 2);
        assert_eq!(stats.total_codes, 16);
        assert_eq!(stats.min_code, 7);
        assert_eq!(stats.max_code, 7);
        assert_eq!(stats.code7_count, 16);
        assert!(stats.all_code7());
    }

    #[test]
    fn qzero_stats_reports_mixed_codes() {
        let stats = gptq_qzero_stats(&[0x0123_4567]);

        assert_eq!(stats.total_codes, 8);
        assert_eq!(stats.min_code, 0);
        assert_eq!(stats.max_code, 7);
        assert_eq!(stats.code7_count, 1);
        assert!(!stats.all_code7());
    }
}
