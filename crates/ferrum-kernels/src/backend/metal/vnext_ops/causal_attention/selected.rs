//! Actual F16-KV attention PSO chain, including immutable specializations.
//! No page proof is inferred here: caller supplies the result of its original
//! retained-page/physical-range proof. INT8 and transforms remain unavailable.
use super::*;
use ferrum_interfaces::execution_cost::{
    KernelNumericWorkV1, SelectedAlgorithmClassV1, SelectedCommandCostBuilderV1,
    SelectedCommandCostEvidenceV1,
};
use sha2::{Digest, Sha256};
use std::sync::OnceLock;

#[derive(Clone, Copy)]
pub(super) struct Projection {
    pub tokens: u32,
    pub hidden: u32,
    pub epsilon: f32,
    pub launches: [LinearLaunch; 4],
}
#[derive(Clone, Copy)]
pub(super) struct Row {
    pub params: CausalAttentionParams,
    pub projection: Projection,
}

fn selected_entry(
    p: &MetalCausalAttentionPipelines,
    params: &CausalAttentionParams,
    kind: AttentionDispatchKind,
    reduce: bool,
) -> Option<(&'static str, u32)> {
    if p.kv_type != ElementType::F16 {
        return None;
    }
    if reduce {
        let actual = p.grouped_reduce_pipeline(params);
        if std::ptr::eq(actual, &p.grouped_decode_reduce_attention) {
            return Some((GROUPED_DECODE_REDUCE_ATTENTION_KERNEL, 0));
        }
        let special = &p.specialization(params)?.grouped.as_ref()?.1;
        return std::ptr::eq(actual, special)
            .then_some((GROUPED_DECODE_REDUCE_ATTENTION_KERNEL, params.head_dim));
    }
    let actual = p.attention_pipeline(params, kind);
    let (base, entry, special) = match kind {
        AttentionDispatchKind::General => (&p.attention, ATTENTION_KERNEL, None),
        AttentionDispatchKind::DirectDecode => (
            &p.direct_decode_attention,
            DIRECT_DECODE_ATTENTION_KERNEL,
            None,
        ),
        AttentionDispatchKind::GroupedDecode => (
            &p.grouped_decode_partial_attention,
            GROUPED_DECODE_PARTIAL_ATTENTION_KERNEL,
            p.specialization(params)
                .and_then(|s| s.grouped.as_ref().map(|v| &v.0)),
        ),
        AttentionDispatchKind::TiledPrefill => (
            &p.tiled_prefill_attention,
            TILED_PREFILL_ATTENTION_KERNEL,
            p.specialization(params).and_then(|s| s.tiled.as_ref()),
        ),
        AttentionDispatchKind::GqaTiledPrefill => (
            &p.gqa_tiled_prefill_attention,
            GQA_TILED_PREFILL_ATTENTION_KERNEL,
            p.specialization(params).and_then(|s| s.gqa.as_ref()),
        ),
    };
    if std::ptr::eq(actual, base) {
        Some((entry, 0))
    } else {
        special
            .filter(|s| std::ptr::eq(actual, *s))
            .map(|_| (entry, params.head_dim))
    }
}
#[allow(clippy::too_many_arguments)]
fn kernel(
    b: &mut SelectedCommandCostBuilderV1,
    entry: &'static str,
    special: u32,
    p: &CausalAttentionParams,
    logical: u64,
    padded: u64,
    inner: u64,
    grid: [u64; 3],
    threads: [u64; 3],
    memory: [u64; 2],
    scratch: u64,
) -> Option<()> {
    static NUMERIC: OnceLock<[u8; 32]> = OnceLock::new();
    let numeric = *NUMERIC.get_or_init(|| {
        let mut h = Sha256::new();
        h.update(b"metal.causal.fp16kv.default-options.v1");
        h.update(SHADER_SOURCE.as_bytes());
        h.update(include_str!("head_dim_specialization.rs").as_bytes());
        h.update(include_str!("selected.rs").as_bytes());
        h.finalize().into()
    });
    let mut layout = Sha256::new();
    layout.update(b"causal.paged.params.v1");
    for n in [
        special,
        p.page_elements,
        p.query_heads,
        p.key_value_heads,
        p.head_dim,
        p.rope_dim,
        p.query_projection_stride,
        p.query_head_stride,
        p.kv_projection_stride,
        p.output_gate,
        p.rope_interleaved,
        p.epsilon.to_bits(),
        p.rope_theta.to_bits(),
    ] {
        layout.update(n.to_le_bytes());
    }
    for n in threads.into_iter().chain(memory) {
        layout.update(n.to_le_bytes());
    }
    b.kernel(
        SelectedAlgorithmClassV1::new(entry, 1, numeric, layout.finalize().into()).ok()?,
        KernelNumericWorkV1 {
            logical_units: logical,
            padded_units: padded,
            inner_units_per_logical_unit: inner,
            grid: [
                u32::try_from(grid[0]).ok()?,
                u32::try_from(grid[1]).ok()?,
                u32::try_from(grid[2]).ok()?,
            ],
            scratch_bytes: scratch,
            staged_weight_bytes: 0,
        },
    )
    .ok()
}
fn prepare(
    b: &mut SelectedCommandCostBuilderV1,
    p: &CausalAttentionParams,
    scratch: u64,
) -> Option<()> {
    let heads =
        u64::from(p.query_heads).checked_add(u64::from(p.key_value_heads).checked_mul(2)?)?;
    let n = u64::from(p.tokens).checked_mul(heads)?;
    kernel(
        b,
        PREPARE_KERNEL,
        0,
        p,
        n,
        n,
        u64::from(p.head_dim),
        [u64::from(p.tokens), heads, 1],
        [SIMD_THREADS, 1, 1],
        [0, 0],
        scratch,
    )
}
/// Legal causal pairs are a numeric work proxy, not a claim of exact FLOPs.
/// It includes every prefix row; positions must never be replaced by tokens.
fn pairs(p: &CausalAttentionParams) -> Option<u64> {
    let t = u64::from(p.tokens);
    let start = u64::from(p.position_start);
    t.checked_mul(start)?
        .checked_add(t.checked_mul(t.checked_add(1)?)?.checked_div(2)?)?
        .checked_mul(u64::from(p.query_heads))
}
fn rectangular(p: &CausalAttentionParams, kind: AttentionDispatchKind) -> Option<u64> {
    let t = u64::from(p.tokens);
    let end = u64::from(p.position_start).checked_add(t)?;
    let (qt, kt) = match kind {
        AttentionDispatchKind::TiledPrefill => {
            (u64::from(TILED_PREFILL_QUERY_TILE), TILED_PREFILL_KEY_TILE)
        }
        AttentionDispatchKind::GqaTiledPrefill => (
            u64::from(TILED_PREFILL_QUERY_TILE),
            GQA_TILED_PREFILL_KEY_TILE,
        ),
        AttentionDispatchKind::GroupedDecode => (1, TILED_PREFILL_KEY_TILE),
        _ => (1, 1),
    };
    t.div_ceil(qt)
        .checked_mul(qt)?
        .checked_mul(end.div_ceil(kt).checked_mul(kt)?)?
        .checked_mul(u64::from(p.query_heads))
}
fn attention(
    b: &mut SelectedCommandCostBuilderV1,
    a: &MetalCausalAttentionPipelines,
    p: &CausalAttentionParams,
    scratch: u64,
) -> Option<()> {
    let plan = a.dispatch_plan(p);
    let (entry, special) = selected_entry(a, p, plan.kind, false)?;
    kernel(
        b,
        entry,
        special,
        p,
        pairs(p)?,
        rectangular(p, plan.kind)?,
        u64::from(p.head_dim),
        plan.threadgroups,
        plan.threads_per_threadgroup,
        plan.threadgroup_memory_bytes,
        scratch,
    )?;
    if plan.kind == AttentionDispatchKind::GroupedDecode {
        let (entry, special) = selected_entry(a, p, plan.kind, true)?;
        let n = u64::from(p.query_heads).checked_mul(u64::from(p.head_dim))?;
        kernel(
            b,
            entry,
            special,
            p,
            n,
            n,
            grouped_decode_partitions(p),
            [u64::from(p.query_heads), 1, 1],
            [SIMD_THREADS, 1, 1],
            [grouped_decode_reduce_threadgroup_memory_bytes(), 0],
            scratch,
        )?;
    }
    Some(())
}
fn batched<'a>(
    b: &mut SelectedCommandCostBuilderV1,
    a: &MetalCausalAttentionPipelines,
    rows: impl ExactSizeIterator<Item = &'a CausalAttentionParams> + Clone,
    scratch: u64,
) -> Option<()> {
    if rows.len() == 0 || rows.len() > GROUPED_BATCH_ROWS {
        return None;
    }
    let p = rows.clone().next()?;
    let n = u64::try_from(rows.len()).ok()?;
    a.specialization(p)?.batched_grouped.as_ref()?;
    let mut logical = 0_u64;
    let mut padded = 0_u64;
    let mut partitions = 0_u64;
    let mut reduce_work = 0_u64;
    for v in rows {
        if a.dispatch_plan(v).kind != AttentionDispatchKind::GroupedDecode
            || v.head_dim != p.head_dim
            || v.query_heads != p.query_heads
            || v.key_value_heads != p.key_value_heads
        {
            return None;
        }
        logical = logical.checked_add(pairs(v)?)?;
        padded = padded.checked_add(rectangular(v, AttentionDispatchKind::GroupedDecode)?)?;
        partitions = partitions.max(grouped_decode_partitions(v));
        reduce_work = reduce_work.checked_add(
            u64::from(v.query_heads)
                .checked_mul(u64::from(v.head_dim))?
                .checked_mul(grouped_decode_partitions(v))?,
        )?;
    }
    let plan = grouped_decode_attention_dispatch_plan(p);
    kernel(
        b,
        "vnext_causal_attention_decode_batched_partial_f16",
        p.head_dim,
        p,
        logical,
        padded,
        u64::from(p.head_dim),
        [partitions, u64::from(p.key_value_heads), n],
        [SIMD_THREADS, TILED_PREFILL_SIMDGROUPS, 1],
        plan.threadgroup_memory_bytes,
        scratch,
    )?;
    // Heterogeneous per-row partitions are summed, not multiplied by one row's extent.
    kernel(
        b,
        "vnext_causal_attention_decode_batched_reduce_f16",
        p.head_dim,
        p,
        reduce_work,
        reduce_work,
        1,
        [u64::from(p.query_heads), 1, n],
        [SIMD_THREADS, 1, 1],
        [grouped_decode_reduce_threadgroup_memory_bytes(), 0],
        scratch,
    )
}
fn input(
    b: &mut SelectedCommandCostBuilderV1,
    l: &MetalLinearPipelines,
    p: &MetalPrimitivePipelines,
    hidden: ElementType,
    v: Projection,
    scratch: u64,
) -> Option<()> {
    super::super::primitives::append_selected_rms(
        b,
        p,
        hidden,
        ElementType::F16,
        v.tokens,
        v.hidden,
        v.epsilon,
        scratch,
    )?;
    for launch in &v.launches[..3] {
        super::super::linear::append_selected_projection(b, l, *launch, None, scratch)?;
    }
    Some(())
}
fn output(
    b: &mut SelectedCommandCostBuilderV1,
    l: &MetalLinearPipelines,
    p: &MetalPrimitivePipelines,
    hidden: ElementType,
    v: Projection,
    scratch: u64,
) -> Option<()> {
    super::super::linear::append_selected_projection(b, l, v.launches[3], None, scratch)?;
    super::super::primitives::append_selected_residual(
        b,
        p,
        hidden,
        v.tokens.checked_mul(v.hidden)?,
        scratch,
    )
}
#[allow(clippy::too_many_arguments)]
pub(super) fn evidence(
    a: &MetalCausalAttentionPipelines,
    l: &MetalLinearPipelines,
    p: &MetalPrimitivePipelines,
    hidden: ElementType,
    tokens: u64,
    scratch: u64,
    packed: Option<Projection>,
    batched_grouped: bool,
    rows: &[Row],
) -> Option<SelectedCommandCostEvidenceV1> {
    if a.kv_type != ElementType::F16
        || rows.is_empty()
        || rows
            .iter()
            .try_fold(0_u64, |n, v| n.checked_add(u64::from(v.params.tokens)))?
            != tokens
    {
        return None;
    }
    if batched_grouped && (packed.is_none() || rows.len() < 2) {
        return None;
    }
    let mut b = SelectedCommandCostBuilderV1::new(tokens);
    if let Some(v) = packed {
        if u64::from(v.tokens) != tokens {
            return None;
        }
        input(&mut b, l, p, hidden, v, scratch)?;
        for row in rows {
            prepare(&mut b, &row.params, scratch)?;
            if !batched_grouped {
                attention(&mut b, a, &row.params, scratch)?;
            }
        }
        if batched_grouped {
            for chunk in rows.chunks(GROUPED_BATCH_ROWS) {
                batched(&mut b, a, chunk.iter().map(|row| &row.params), scratch)?;
            }
        }
        output(&mut b, l, p, hidden, v, scratch)?;
    } else {
        for row in rows {
            input(&mut b, l, p, hidden, row.projection, scratch)?;
            prepare(&mut b, &row.params, scratch)?;
            attention(&mut b, a, &row.params, scratch)?;
            output(&mut b, l, p, hidden, row.projection, scratch)?;
        }
    }
    b.finish().ok()
}
#[cfg(test)]
mod tests;
