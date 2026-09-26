//! One ordered table for the real FFN leaf plan. Missing evidence stays passive.
use super::*;
use crate::backend::cuda::vnext_runtime::selected_cost;
use ferrum_interfaces::execution_cost::{
    KernelNumericWorkV1, KernelReplayGeometryV1, SelectedAlgorithmClassV1,
    SelectedCommandCostBuilderV1, SelectedCommandCostEvidenceV1,
};
use ferrum_interfaces::vnext::OperationCostCommand;
use ferrum_types::SloStructuredCostCapture;
use sha2::{Digest, Sha256};
use std::sync::OnceLock;

pub(super) fn attach(
    command: OperationCostCommand,
    evidence: Option<SelectedCommandCostEvidenceV1>,
) -> OperationCostCommand {
    match evidence {
        Some(evidence) => command
            .clone()
            .with_statistical_evidence(evidence)
            .unwrap_or(command),
        None => command,
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn swiglu(
    gate_up: &[weights::MatrixPart],
    down: &[weights::MatrixPart],
    rows: impl IntoIterator<Item = u32>,
    tokens: u64,
    hidden: u32,
    intermediate: u32,
    transform_bytes: u64,
    q8_bytes: u64,
    q8: Option<Q8SumPolicy>,
    mmq: Option<&StreamMmq>,
    mmq_hit: bool,
    capture: SloStructuredCostCapture,
) -> Option<SelectedCommandCostEvidenceV1> {
    if capture.is_disabled() {
        return None;
    }
    let layouts = MmqLayouts::from_selected(mmq, mmq_hit, down, hidden, intermediate)?;
    swiglu_planned(
        gate_up,
        down,
        rows,
        tokens,
        hidden,
        intermediate,
        transform_bytes,
        q8_bytes,
        q8,
        layouts,
        capture,
    )
}

/// The actual selector's numeric workspace/CTA/precision plan. It retains no
/// function, stream, buffer or live device authority.
#[derive(Clone, Copy, Debug)]
pub(super) struct MmqLayouts {
    gate: Option<stream_mmq::Workspace>,
    down: Option<stream_mmq::Workspace>,
}

impl MmqLayouts {
    pub(super) fn from_selected(
        mmq: Option<&StreamMmq>,
        mmq_hit: bool,
        down: &[weights::MatrixPart],
        hidden: u32,
        intermediate: u32,
    ) -> Option<Self> {
        let down = if mmq_hit
            && mmq?.is_residual2()
            && stream_mmq::eligible_q4_down(down, 8, intermediate, hidden)
        {
            Some(mmq?.workspace(intermediate, hidden).ok()?)
        } else {
            None
        };
        let gate = if mmq_hit {
            Some(mmq?.workspace(hidden, intermediate).ok()?)
        } else {
            None
        };
        Some(Self { gate, down })
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn swiglu_planned(
    gate_up: &[weights::MatrixPart],
    down: &[weights::MatrixPart],
    rows: impl IntoIterator<Item = u32>,
    tokens: u64,
    hidden: u32,
    intermediate: u32,
    transform_bytes: u64,
    q8_bytes: u64,
    q8: Option<Q8SumPolicy>,
    layouts: MmqLayouts,
    capture: SloStructuredCostCapture,
) -> Option<SelectedCommandCostEvidenceV1> {
    let mut builder = selected_cost::builder(capture, tokens)?;
    let mmq = layouts.gate;
    let mmq_down = layouts.down;
    let doubled = intermediate.checked_mul(2)?;
    let inventory_matches = |parts: &[weights::MatrixPart], columns: u32, outputs: u32| {
        parts.iter().try_fold(0_u32, |end, part| {
            if part.columns != columns || part.rows == 0 || part.output_offset != end {
                return None;
            }
            end.checked_add(part.rows)
        }) == Some(outputs)
            && columns > 0
            && outputs > 0
    };
    if !inventory_matches(gate_up, hidden, doubled)
        || !inventory_matches(down, intermediate, hidden)
    {
        return None;
    }
    if q8.is_some() && mmq.is_some() {
        return None;
    }
    let scratch = ScratchLayout::new(tokens, u64::from(intermediate)).ok()?;
    let mut seen = 0_u64;
    for rows in rows {
        seen = seen.checked_add(u64::from(rows))?;
        if seen > tokens {
            return None;
        }
        if let Some(layout) = mmq {
            stream_mmq::selected::append_gate_up(
                &mut builder,
                gate_up,
                rows,
                hidden,
                intermediate,
                layout,
            )?;
        } else if let Some(policy) = q8 {
            q8_f32scale::selected::append_projection(
                &mut builder,
                gate_up,
                tokens,
                rows,
                doubled,
                policy,
                q8_bytes,
            )?;
        } else {
            for part in gate_up {
                super::super::super::native_blocks::selected::append_transformed_linear(
                    &mut builder,
                    part,
                    rows,
                    doubled,
                    ElementType::F16,
                    transform_bytes,
                )?;
            }
        }
        append_silu(&mut builder, rows, intermediate, scratch.total_bytes)?;
        if let Some(layout) = mmq_down {
            stream_mmq::selected::append_residual_down(
                &mut builder,
                down,
                rows,
                intermediate,
                hidden,
                layout,
            )?;
        } else if let Some(policy) = q8 {
            q8_f32scale::selected::append_projection(
                &mut builder,
                down,
                tokens,
                rows,
                hidden,
                policy,
                q8_bytes,
            )?;
        } else {
            for part in down {
                super::super::super::native_blocks::selected::append_transformed_linear(
                    &mut builder,
                    part,
                    rows,
                    hidden,
                    ElementType::F16,
                    transform_bytes,
                )?;
            }
        }
    }
    if seen != tokens {
        return None;
    }
    builder.finish().ok()
}

pub(in crate::backend::cuda::vnext_ops::transformer) fn append_silu(
    builder: &mut SelectedCommandCostBuilderV1,
    rows: u32,
    intermediate: u32,
    scratch: u64,
) -> Option<()> {
    static NUMERICAL: OnceLock<[u8; 32]> = OnceLock::new();
    let intermediate = i32::try_from(intermediate).ok()?;
    let elements = u64::from(rows).checked_mul(intermediate as u64)?;
    let (config, total) = super::super::silu_mul_launch_config(elements).ok()?;
    builder
        .kernel_with_replay_geometry(
            SelectedAlgorithmClassV1::new(
                SILU_MUL_FUNCTION_NAME,
                1,
                *NUMERICAL
                    .get_or_init(|| Sha256::digest(crate::ptx::FUSED_SILU_MUL.as_bytes()).into()),
                Sha256::digest(b"cuda.silu_mul.interleaved.f16.v1").into(),
            )
            .ok()?,
            KernelNumericWorkV1 {
                logical_units: elements,
                padded_units: u64::from(config.grid_dim.0)
                    .checked_mul(u64::from(config.block_dim.0))?,
                inner_units_per_logical_unit: 1,
                grid: [config.grid_dim.0, config.grid_dim.1, config.grid_dim.2],
                scratch_bytes: scratch,
                staged_weight_bytes: 0,
            },
            KernelReplayGeometryV1 {
                block: [config.block_dim.0, config.block_dim.1, config.block_dim.2],
                dynamic_shared_bytes: u64::from(config.shared_mem_bytes),
                fixed_parameters: &[intermediate as u64, total as u64],
            },
        )
        .ok()
}

#[cfg(test)]
mod tests;
