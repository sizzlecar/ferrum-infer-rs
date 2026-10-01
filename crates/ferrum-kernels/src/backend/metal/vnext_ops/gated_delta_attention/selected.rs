//! Passive selected-kernel evidence for the actual recurrent GDN command.
//! Chunked scans and transformed projections remain explicitly unavailable.
use super::*;
use ferrum_interfaces::execution_cost::{
    KernelNumericWorkV1, SelectedAlgorithmClassV1, SelectedCommandCostBuilderV1,
    SelectedCommandCostEvidenceV1,
};
use std::sync::OnceLock;

mod classes;
pub(super) use classes::PreparedKernelClasses;

mod observation;
pub(super) use observation::freeze as observation;

struct CommandBuilder<'a> {
    selected: SelectedCommandCostBuilderV1,
    classes: Option<&'a PreparedKernelClasses>,
    linear_classes: Option<&'a super::cost_route::PreparedLinearProjectionClasses>,
}

impl<'a> CommandBuilder<'a> {
    fn new(
        capture: ferrum_types::SloStructuredCostCapture,
        tokens: u64,
        classes: Option<&'a PreparedKernelClasses>,
    ) -> Self {
        Self {
            selected: crate::backend::metal::vnext_runtime::selected_cost_builder(capture, tokens),
            classes,
            linear_classes: None,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct Projection<'a> {
    pub params: GatedDeltaParams,
    pub input: &'a [LinearLaunch],
    pub output: LinearLaunch,
    pub staged: bool,
}
#[derive(Clone, Copy)]
pub(super) struct Row<'a> {
    pub projection: Projection<'a>,
    pub form: GatedDeltaExecutionForm,
}

/// Used by execution itself as well as evidence; no selector is guessed from a
/// native operation label or historical trace.
pub(super) fn recurrent<'a>(
    p: &'a MetalGatedDeltaPipelines,
    params: &GatedDeltaParams,
) -> (&'a ComputePipelineState, &'static str, u64, u64) {
    if supports_simd_delta(
        params,
        p.simd_delta.thread_execution_width(),
        p.simd_delta
            .max_total_threads_per_threadgroup()
            .try_into()
            .unwrap_or(0),
    ) {
        (
            &p.simd_delta,
            SIMD_DELTA_KERNEL,
            SIMD_DELTA_ROWS_PER_GROUP,
            SIMD_DELTA_THREADS,
        )
    } else {
        (&p.delta, DELTA_KERNEL, VALUE_TILE, THREADS_PER_GROUP)
    }
}

#[allow(clippy::too_many_arguments)]
fn kernel(
    b: &mut CommandBuilder<'_>,
    entry: &'static str,
    params: &GatedDeltaParams,
    logical: u64,
    padded: u64,
    inner: u64,
    grid: [u64; 3],
    threads: u64,
    scratch: u64,
) -> Option<()> {
    let class = match b.classes {
        Some(classes) => classes.get(entry, threads)?,
        None => classes::fresh(entry, params, threads)?,
    };
    b.selected
        .kernel(
            class,
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
fn elements(
    b: &mut CommandBuilder<'_>,
    entry: &'static str,
    p: &GatedDeltaParams,
    n: u64,
    inner: u64,
    scratch: u64,
) -> Option<()> {
    let groups = n.div_ceil(THREADS_PER_GROUP);
    kernel(
        b,
        entry,
        p,
        n,
        groups.checked_mul(THREADS_PER_GROUP)?,
        inner,
        [groups, 1, 1],
        THREADS_PER_GROUP,
        scratch,
    )
}
fn norm(b: &mut CommandBuilder<'_>, p: &GatedDeltaParams, gated: bool, scratch: u64) -> Option<()> {
    let heads = if gated { p.value_heads } else { p.key_heads };
    let n = u64::from(p.tokens).checked_mul(u64::from(heads))?;
    kernel(
        b,
        if gated {
            GATED_NORM_KERNEL
        } else {
            QK_NORM_KERNEL
        },
        p,
        n,
        n,
        u64::from(if gated { p.value_dim } else { p.key_dim }),
        [n, 1, 1],
        THREADS_PER_GROUP,
        scratch,
    )
}
fn conv(b: &mut CommandBuilder<'_>, p: &GatedDeltaParams, scratch: u64) -> Option<()> {
    let n = u64::from(p.tokens).checked_mul(u64::from(p.qkvz_features))?;
    // Work units are physical element domains, not a FLOP claim (Z lanes copy).
    elements(b, PREPARE_CONV_KERNEL, p, n, 1, scratch)?;
    let state = u64::from(p.qkv_features).checked_mul(u64::from(p.conv_kernel.checked_sub(1)?))?;
    elements(b, COLLECT_CONV_STATE_KERNEL, p, state, 1, scratch)?;
    elements(b, COPY_F16_KERNEL, p, state, 1, scratch)
}
fn gates(b: &mut CommandBuilder<'_>, p: &GatedDeltaParams, scratch: u64) -> Option<()> {
    elements(
        b,
        PREPARE_GATES_KERNEL,
        p,
        u64::from(p.tokens).checked_mul(u64::from(p.value_heads))?,
        1,
        scratch,
    )
}
fn delta(
    b: &mut CommandBuilder<'_>,
    a: &MetalGatedDeltaPipelines,
    p: &GatedDeltaParams,
    scratch: u64,
) -> Option<()> {
    let (_, entry, tile, threads) = recurrent(a, p);
    let groups = u64::from(p.value_dim).div_ceil(tile);
    let heads = u64::from(p.value_heads);
    kernel(
        b,
        entry,
        p,
        heads.checked_mul(u64::from(p.value_dim))?,
        heads.checked_mul(groups)?.checked_mul(tile)?,
        u64::from(p.tokens).checked_mul(u64::from(p.key_dim))?,
        [groups, heads, 1],
        threads,
        scratch,
    )
}
fn input(
    b: &mut CommandBuilder<'_>,
    l: &MetalLinearPipelines,
    p: &MetalPrimitivePipelines,
    hidden: ElementType,
    v: Projection<'_>,
    scratch: u64,
) -> Option<()> {
    super::super::primitives::append_selected_rms(
        &mut b.selected,
        p,
        hidden,
        ElementType::F16,
        v.params.tokens,
        v.params.hidden_size,
        v.params.epsilon,
        scratch,
    )?;
    if b.linear_classes
        .is_some_and(|classes| classes.input.len() != v.input.len())
    {
        return None;
    }
    for (index, &launch) in v.input.iter().enumerate() {
        super::super::linear::append_selected_projection_with_classes(
            &mut b.selected,
            l,
            launch,
            v.staged
                .then_some(staged_prefill::StagingPolicy::GatedDelta),
            scratch,
            b.linear_classes.map(|classes| &classes.input[index]),
        )?;
    }
    Some(())
}
fn output(
    b: &mut CommandBuilder<'_>,
    l: &MetalLinearPipelines,
    p: &MetalPrimitivePipelines,
    hidden: ElementType,
    v: Projection<'_>,
    scratch: u64,
) -> Option<()> {
    norm(b, &v.params, true, scratch)?;
    super::super::linear::append_selected_projection_with_classes(
        &mut b.selected,
        l,
        v.output,
        None,
        scratch,
        b.linear_classes.map(|classes| &classes.output),
    )?;
    let n = v.params.tokens.checked_mul(v.params.hidden_size)?;
    super::super::primitives::append_selected_residual(&mut b.selected, p, hidden, n, scratch)
}
#[allow(clippy::too_many_arguments)]
pub(super) fn evidence<'a>(
    a: &MetalGatedDeltaPipelines,
    l: &MetalLinearPipelines,
    p: &MetalPrimitivePipelines,
    hidden: ElementType,
    tokens: u64,
    scratch: u64,
    packed: Option<Projection<'a>>,
    rows: impl Iterator<Item = Row<'a>> + Clone,
) -> Option<SelectedCommandCostEvidenceV1> {
    evidence_with_classes(a, l, p, hidden, tokens, scratch, packed, rows, None)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn evidence_with_classes<'a>(
    a: &MetalGatedDeltaPipelines,
    l: &MetalLinearPipelines,
    p: &MetalPrimitivePipelines,
    hidden: ElementType,
    tokens: u64,
    scratch: u64,
    packed: Option<Projection<'a>>,
    rows: impl Iterator<Item = Row<'a>> + Clone,
    classes: Option<&PreparedKernelClasses>,
) -> Option<SelectedCommandCostEvidenceV1> {
    evidence_with_prepared_classes(
        a, l, p, hidden, tokens, scratch, packed, rows, classes, None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn evidence_with_prepared_classes<'a>(
    a: &MetalGatedDeltaPipelines,
    l: &MetalLinearPipelines,
    p: &MetalPrimitivePipelines,
    hidden: ElementType,
    tokens: u64,
    scratch: u64,
    packed: Option<Projection<'a>>,
    rows: impl Iterator<Item = Row<'a>> + Clone,
    classes: Option<&PreparedKernelClasses>,
    linear_classes: Option<&super::cost_route::PreparedLinearProjectionClasses>,
) -> Option<SelectedCommandCostEvidenceV1> {
    let mut total = 0_u64;
    let mut count = 0_u64;
    for row in rows.clone() {
        if !matches!(row.form, GatedDeltaExecutionForm::RecurrentScan)
            || classes.is_some_and(|classes| !classes.matches(&row.projection.params))
        {
            return None;
        }
        total = total.checked_add(u64::from(row.projection.params.tokens))?;
        count = count.checked_add(1)?;
    }
    if count == 0 || total != tokens {
        return None;
    }
    let mut b = CommandBuilder::new(l.structured_capture(), tokens, classes);
    b.linear_classes = linear_classes;
    if let Some(v) = packed {
        if u64::from(v.params.tokens) != tokens
            || classes.is_some_and(|classes| !classes.matches(&v.params))
        {
            return None;
        }
        input(&mut b, l, p, hidden, v, scratch)?;
        gates(&mut b, &v.params, scratch)?;
        for row in rows.clone() {
            conv(&mut b, &row.projection.params, scratch)?;
        }
        norm(&mut b, &v.params, false, scratch)?;
        for row in rows {
            delta(&mut b, a, &row.projection.params, scratch)?;
        }
        output(&mut b, l, p, hidden, v, scratch)?;
    } else {
        for row in rows {
            let v = row.projection;
            input(&mut b, l, p, hidden, v, scratch)?;
            conv(&mut b, &v.params, scratch)?;
            gates(&mut b, &v.params, scratch)?;
            norm(&mut b, &v.params, false, scratch)?;
            delta(&mut b, a, &v.params, scratch)?;
            output(&mut b, l, p, hidden, v, scratch)?;
        }
    }
    b.selected.finish().ok()
}

#[cfg(test)]
mod tests;

pub(super) use observation::payload_upper as observation_payload_upper;
