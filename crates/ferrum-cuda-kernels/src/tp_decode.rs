//! Tensor Parallel Decode Group.
//!
//! Coordinates N CudaDecodeRunners (one per GPU) for tensor-parallel decode.
//! Each GPU holds 1/N of attention heads and MLP width.
//! NCCL all-reduce after O projection and down projection (Megatron-LM pattern).
//!
//! Architecture: one thread per GPU, NCCL handles inter-GPU sync.
//! The caller thread drives rank 0; other ranks run on spawned threads.

#[cfg(feature = "tensor-parallel")]
use crate::cuda_decode::CudaDecodeRunner;
#[cfg(feature = "tensor-parallel")]
use crate::nccl_comm::NcclRank;
#[cfg(feature = "tensor-parallel")]
use cudarc::driver::CudaSlice;

/// Tensor parallel decode group.
///
/// Holds one CudaDecodeRunner + NcclRank per GPU.
/// Orchestrates the decode pipeline with all-reduce at the right points.
#[cfg(feature = "tensor-parallel")]
pub struct TpDecodeGroup {
    /// Per-rank runners (index = rank)
    runners: Vec<CudaDecodeRunner>,
    /// Per-rank NCCL communicators
    nccl: Vec<NcclRank>,
    /// Number of layers
    num_layers: usize,
}

#[cfg(feature = "tensor-parallel")]
impl TpDecodeGroup {
    pub fn new(runners: Vec<CudaDecodeRunner>, nccl: Vec<NcclRank>) -> candle_core::Result<Self> {
        let world_size = runners.len();
        if nccl.len() != world_size || world_size == 0 {
            return Err(candle_core::Error::Msg(format!(
                "runners ({}) != nccl ({}) or zero",
                world_size,
                nccl.len()
            )));
        }
        let num_layers = runners[0].weight_layers().len();
        Ok(Self {
            runners,
            nccl,
            num_layers,
        })
    }

    pub fn world_size(&self) -> usize {
        self.runners.len()
    }

    /// Tensor-parallel decode step for a single token.
    ///
    /// All ranks execute each sub-phase in lockstep on their own stream.
    /// NCCL all-reduce synchronizes the partial results after O-proj and down-proj.
    /// Returns logits from rank 0.
    ///
    /// NOTE: This is a single-threaded implementation where the caller drives
    /// all ranks sequentially. For production, each rank should be on its own
    /// thread with NCCL providing the synchronization. The sequential version
    /// is correct (NCCL ops are stream-ordered) but doesn't overlap GPU work
    /// across ranks.
    pub fn decode_step(
        &mut self,
        token_id: u32,
        position: usize,
        cache_key: &str,
    ) -> candle_core::Result<CudaSlice<half::f16>> {
        let n = self.num_layers;
        let ws = self.runners.len();

        // Helper: bind CUDA context before operating on each rank
        macro_rules! for_each_rank {
            ($body:expr) => {
                for r in 0..ws {
                    self.runners[r].bind_context()?;
                    $body(r)?;
                }
            };
        }

        // Embed (replicated)
        for_each_rank!(|r: usize| self.runners[r].tp_embed(token_id));

        // First layer norm (replicated)
        for_each_rank!(|r: usize| self.runners[r].tp_first_norm());

        // Per-layer pipeline
        for li in 0..n {
            // 1. QKV + norm + RoPE + attention (local heads)
            for_each_rank!(|r: usize| self.runners[r].tp_pre_o_proj(li, position, cache_key));

            // 2. O projection (row-parallel: partial output)
            for_each_rank!(|r: usize| self.runners[r].tp_o_proj(li));

            // 3. ALL-REDUCE o_proj_out
            for r in 0..ws {
                self.runners[r].bind_context()?;
                self.nccl[r].all_reduce_f16_inplace(self.runners[r].o_proj_out_mut())?;
            }

            // 4. Post-attention residual + norm
            for_each_rank!(|r: usize| self.runners[r].tp_post_attn_norm(li));

            // 5. MLP
            for_each_rank!(|r: usize| self.runners[r].tp_mlp(li));

            // 6. ALL-REDUCE down_out
            for r in 0..ws {
                self.runners[r].bind_context()?;
                self.nccl[r].all_reduce_f16_inplace(self.runners[r].down_out_mut())?;
            }

            // 7. Post-MLP residual + next layer norm
            for_each_rank!(|r: usize| self.runners[r].tp_post_mlp_norm(li));
        }

        // LM head (replicated)
        for_each_rank!(|r: usize| self.runners[r].tp_lm_head());

        // Sync rank 0 and return its logits
        self.runners[0].bind_context()?;
        self.runners[0].sync_stream()?;
        self.runners[0].clone_logits()
    }

    /// Initialize KV cache on all ranks (with context binding).
    pub fn init_kv_cache(
        &mut self,
        cache_key: &str,
        kv_data_per_rank: Vec<Vec<(CudaSlice<half::f16>, CudaSlice<half::f16>)>>,
        prefill_len: usize,
        max_len: usize,
    ) -> candle_core::Result<()> {
        for (r, kv_data) in kv_data_per_rank.into_iter().enumerate() {
            self.runners[r].bind_context()?;
            self.runners[r].init_kv_cache(cache_key, kv_data, prefill_len, max_len)?;
        }
        Ok(())
    }

    /// Release KV cache on all ranks.
    pub fn release_kv_cache(&mut self, cache_key: &str) {
        for r in &mut self.runners {
            r.release_kv_cache(cache_key);
        }
    }

    /// Check if KV cache exists on rank 0.
    pub fn has_kv_cache(&self, cache_key: &str) -> bool {
        self.runners[0].has_kv_cache(cache_key)
    }
}
