//! Metal `BackendMoeFused` impl. Extracted from `metal.rs` (Audit #8).

use super::{st, MetalBackend};
use ferrum_types::Result;

impl crate::backend::BackendMoeFused for MetalBackend {
    fn supports_batched_moe_gemv() -> bool {
        true
    }
    fn supports_batched_moe_gate_up_silu() -> bool {
        true
    }
    fn route_topk_softmax(
        ctx: &mut Self::Context,
        logits: &Self::Buffer,
        out_ids: &mut Self::Buffer,
        out_weights: &mut Self::Buffer,
        batch: usize,
        num_experts: usize,
        top_k: usize,
        norm_topk_prob: bool,
    ) -> Result<()> {
        let logits_buf = logits.expect_f32("route_topk_softmax logits");
        let ids_buf = &out_ids.raw;
        let weights_buf = out_weights.expect_f32_mut("route_topk_softmax out_weights");
        let enc = ctx.compute_encoder();
        crate::moe_router::dispatch_route_topk_softmax(
            &st().pipes.device,
            enc,
            logits_buf,
            ids_buf,
            weights_buf,
            batch,
            num_experts,
            top_k,
            norm_topk_prob,
        );
        Ok(())
    }
    fn silu_mul_batched(
        ctx: &mut Self::Context,
        gate: &Self::Buffer,
        up: &Self::Buffer,
        out: &mut Self::Buffer,
        total_pairs: usize,
        ffn: usize,
    ) -> Result<()> {
        let gate_buf = gate.expect_f32("silu_mul_batched gate");
        let up_buf = up.expect_f32("silu_mul_batched up");
        let out_buf = out.expect_f32_mut("silu_mul_batched out");
        let enc = ctx.compute_encoder();
        crate::moe_post_ops_batched::dispatch_silu_mul_batched(
            &st().pipes.device,
            enc,
            gate_buf,
            up_buf,
            out_buf,
            total_pairs,
            ffn,
        );
        Ok(())
    }
    fn compute_ids_tpe_gpu(
        ctx: &mut Self::Context,
        selected_ids: &Self::Buffer,
        tpe: &mut Self::Buffer,
        ids: &mut Self::Buffer,
        gate_up_args: &mut Self::Buffer,
        down_args: &mut Self::Buffer,
        batch: usize,
        num_experts: usize,
        top_k: usize,
        m_gate_up: usize,
        m_down: usize,
    ) -> Result<()> {
        let sel_buf = &selected_ids.raw;
        let tpe_buf = &tpe.raw;
        let ids_buf = &ids.raw;
        let gate_up_args_buf = &gate_up_args.raw;
        let down_args_buf = &down_args.raw;
        let enc = ctx.compute_encoder();
        crate::moe_router::dispatch_compute_ids_tpe(
            &st().pipes.device,
            enc,
            sel_buf,
            tpe_buf,
            ids_buf,
            gate_up_args_buf,
            down_args_buf,
            batch,
            num_experts,
            top_k,
            m_gate_up,
            m_down,
        );
        Ok(())
    }
    fn weighted_sum_residual_stacked(
        ctx: &mut Self::Context,
        slots: &Self::Buffer,
        weights: &Self::Buffer,
        residual: &mut Self::Buffer,
        n_slots: usize,
        hidden: usize,
    ) -> Result<()> {
        let slots_buf = slots.expect_f32("weighted_sum_residual_stacked slots");
        let weights_buf = weights.expect_f32("weighted_sum_residual_stacked weights");
        let residual_buf = residual.expect_f32_mut("weighted_sum_residual_stacked residual");
        let enc = ctx.compute_encoder();
        crate::moe_post_ops::dispatch_weighted_sum_residual_stacked(
            &st().pipes.device,
            enc,
            slots_buf,
            weights_buf,
            residual_buf,
            n_slots,
            hidden,
        );
        Ok(())
    }
    fn weighted_sum_residual_norm_stacked(
        ctx: &mut Self::Context,
        slots: &Self::Buffer,
        weights: &Self::Buffer,
        residual: &mut Self::Buffer,
        next_norm_w: &Self::Buffer,
        normed_out: &mut Self::Buffer,
        n_slots: usize,
        hidden: usize,
        eps: f32,
    ) -> Result<()> {
        let slots_buf = slots.expect_f32("weighted_sum_residual_norm_stacked slots");
        let weights_buf = weights.expect_f32("weighted_sum_residual_norm_stacked weights");
        let residual_buf = residual.expect_f32_mut("weighted_sum_residual_norm_stacked residual");
        let nw_buf = next_norm_w.expect_f32("weighted_sum_residual_norm_stacked next_norm_w");
        let normed_buf = normed_out.expect_f32_mut("weighted_sum_residual_norm_stacked normed_out");
        let enc = ctx.compute_encoder();
        crate::moe_post_ops::dispatch_weighted_sum_residual_norm_stacked(
            &st().pipes.device,
            enc,
            slots_buf,
            weights_buf,
            residual_buf,
            nw_buf,
            normed_buf,
            n_slots,
            hidden,
            eps,
        );
        Ok(())
    }
    fn weighted_sum_batched(
        ctx: &mut Self::Context,
        slots: &Self::Buffer,
        weights: &Self::Buffer,
        out: &mut Self::Buffer,
        batch: usize,
        top_k: usize,
        hidden: usize,
    ) -> Result<()> {
        let slots_buf = slots.expect_f32("weighted_sum_batched slots");
        let weights_buf = weights.expect_f32("weighted_sum_batched weights");
        let out_buf = out.expect_f32_mut("weighted_sum_batched out");
        let enc = ctx.compute_encoder();
        crate::moe_post_ops_batched::dispatch_weighted_sum_batched(
            &st().pipes.device,
            enc,
            slots_buf,
            weights_buf,
            out_buf,
            batch,
            top_k,
            hidden,
        );
        Ok(())
    }
    fn weighted_sum_batched_offset(
        ctx: &mut Self::Context,
        slots: &Self::Buffer,
        weights: &Self::Buffer,
        weights_offset: usize,
        out: &mut Self::Buffer,
        out_offset: usize,
        batch: usize,
        top_k: usize,
        hidden: usize,
    ) -> Result<()> {
        let slots_buf = slots.expect_f32("weighted_sum_batched_offset slots");
        let weights_buf = weights.expect_f32("weighted_sum_batched_offset weights");
        let out_buf = out.expect_f32_mut("weighted_sum_batched_offset out");
        let enc = ctx.compute_encoder();
        // weights/out are f32; multiply by 4 for byte offset.
        let weights_byte_offset = (weights_offset * std::mem::size_of::<f32>()) as u64;
        let out_byte_offset = (out_offset * std::mem::size_of::<f32>()) as u64;
        crate::moe_post_ops_batched::dispatch_weighted_sum_batched_offset(
            &st().pipes.device,
            enc,
            slots_buf,
            0,
            weights_buf,
            weights_byte_offset,
            out_buf,
            out_byte_offset,
            batch,
            top_k,
            hidden,
        );
        Ok(())
    }
    fn silu_mul_stacked(
        ctx: &mut Self::Context,
        gate: &Self::Buffer,
        up: &Self::Buffer,
        out: &mut Self::Buffer,
        n_slots: usize,
        ffn: usize,
    ) -> Result<()> {
        let gate_buf = gate.expect_f32("silu_mul_stacked gate");
        let up_buf = up.expect_f32("silu_mul_stacked up");
        let out_buf = out.expect_f32_mut("silu_mul_stacked out");
        let enc = ctx.compute_encoder();
        crate::moe_post_ops::dispatch_silu_mul_stacked(
            &st().pipes.device,
            enc,
            gate_buf,
            up_buf,
            out_buf,
            n_slots,
            ffn,
        );
        Ok(())
    }
    fn supports_fused_moe_gate_up_silu() -> bool {
        true
    }
    fn weighted_sum_stacked(
        ctx: &mut Self::Context,
        slots: &Self::Buffer,
        weights: &Self::Buffer,
        out: &mut Self::Buffer,
        n_slots: usize,
        hidden: usize,
    ) -> Result<()> {
        let slots_buf = slots.expect_f32("weighted_sum_stacked slots");
        let weights_buf = weights.expect_f32("weighted_sum_stacked weights");
        let out_buf = out.expect_f32_mut("weighted_sum_stacked out");
        let enc = ctx.compute_encoder();
        crate::moe_post_ops::dispatch_weighted_sum_stacked(
            &st().pipes.device,
            enc,
            slots_buf,
            weights_buf,
            out_buf,
            n_slots,
            hidden,
        );
        Ok(())
    }
}
