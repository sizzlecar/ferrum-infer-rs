//! Metal `BackendPagedKv` impl. Extracted from `metal.rs` (Audit #8).

use super::{st, MetalBackend};
use ferrum_types::Result;

impl crate::backend::BackendPagedKv for MetalBackend {
    fn supports_paged_kv() -> bool {
        true
    }
    #[allow(clippy::too_many_arguments)]
    fn split_qkv_norm_rope_into_paged_cache(
        ctx: &mut Self::Context,
        qkv: &Self::Buffer,
        qkv_byte_offset: u64,
        q_norm_w: &Self::Buffer,
        k_norm_w: &Self::Buffer,
        cos: &Self::Buffer,
        sin: &Self::Buffer,
        q_out: &mut Self::Buffer,
        q_out_byte_offset: u64,
        cache_k: &mut Self::Buffer,
        cache_v: &mut Self::Buffer,
        block_table: &Self::Buffer,
        tokens: usize,
        q_heads: usize,
        kv_heads: usize,
        head_dim: usize,
        pos_offset: usize,
        eps: f32,
        qk_mode: i32,
        cache_len: usize,
        block_size: usize,
        max_num_blocks_per_seq: usize,
    ) -> Result<()> {
        let qkv = qkv.expect_f32("split_qkv_norm_rope_paged qkv");
        let q_norm_w = q_norm_w.expect_f32("split_qkv_norm_rope_paged q_norm_w");
        let k_norm_w = k_norm_w.expect_f32("split_qkv_norm_rope_paged k_norm_w");
        let cos = cos.expect_f32("split_qkv_norm_rope_paged cos");
        let sin = sin.expect_f32("split_qkv_norm_rope_paged sin");
        let q_out = q_out.expect_f32_mut("split_qkv_norm_rope_paged q_out");
        let cache_k = cache_k.expect_f32_mut("split_qkv_norm_rope_paged cache_k");
        let cache_v = cache_v.expect_f32_mut("split_qkv_norm_rope_paged cache_v");
        let bt = &block_table.raw;
        let enc = ctx.compute_encoder();
        st().pipes.split_qkv_norm_rope_into_paged_cache(
            enc,
            qkv,
            qkv_byte_offset,
            q_norm_w,
            k_norm_w,
            cos,
            sin,
            q_out,
            q_out_byte_offset,
            cache_k,
            cache_v,
            bt,
            tokens,
            q_heads,
            kv_heads,
            head_dim,
            pos_offset,
            eps,
            qk_mode,
            cache_len,
            block_size,
            max_num_blocks_per_seq,
        );
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn paged_decode_attention(
        ctx: &mut Self::Context,
        q: &Self::Buffer,
        k_pool: &Self::Buffer,
        v_pool: &Self::Buffer,
        out: &mut Self::Buffer,
        block_tables: &Self::Buffer,
        context_lens: &Self::Buffer,
        num_seqs: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        block_size: usize,
        max_num_blocks_per_seq: usize,
        q_len: usize,
    ) -> Result<()> {
        let q = q.expect_f32("paged_decode_attention q");
        let k_pool = k_pool.expect_f32("paged_decode_attention k_pool");
        let v_pool = v_pool.expect_f32("paged_decode_attention v_pool");
        let out = out.expect_f32_mut("paged_decode_attention out");
        let bt = &block_tables.raw;
        let cl = &context_lens.raw;
        let enc = ctx.compute_encoder();
        // q_len=1 (decode): token-major layout matches scratch.q_head_major
        // when tokens=1 (the head and token dims collapse).
        // q_len>1 (prefill): scratch.q_head_major is `[num_heads, q_len,
        // head_dim]` head-major from `split_qkv_norm_rope_into_paged_cache`.
        let q_layout = if q_len == 1 {
            crate::attention::metal::pipelines::PagedAttnQLayout::TokenMajor
        } else {
            crate::attention::metal::pipelines::PagedAttnQLayout::HeadMajor
        };
        st().pipes.paged_decode_attention_on_encoder(
            enc,
            q,
            k_pool,
            v_pool,
            out,
            bt,
            cl,
            &crate::attention::metal::pipelines::PagedAttnDispatchParams {
                num_seqs,
                num_heads,
                num_kv_heads,
                head_dim,
                block_size,
                max_num_blocks_per_seq,
                q_len,
                q_layout,
            },
        );
        Ok(())
    }
}
