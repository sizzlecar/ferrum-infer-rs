//! Shared numerical selection for actual encoding and its eager cost route.
//! No address, allocation, sequence or submission authority is held here.
use super::*;
use ferrum_interfaces::vnext::OperationCostCommand;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Arithmetic {
    Strict,
    Q8,
    StreamMmq,
    Residual2M2To8,
}

#[derive(Clone, Debug)]
pub(super) struct Selection {
    pub packed_rows: Option<u32>,
    pub mmq_hit: bool,
    pub mmq_down_hit: bool,
    pub launches: u64,
    pub command: OperationCostCommand,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn select(
    gate_up: &[weights::MatrixPart],
    down: &[weights::MatrixPart],
    hidden: u64,
    intermediate: u64,
    tokens: u64,
    participants: u32,
    input_packed: bool,
    output_packed: bool,
    arithmetic: Arithmetic,
) -> Result<Selection, String> {
    if participants == 0 || u64::from(participants) > tokens {
        return Err("native SwiGLU selection needs nonempty participant rows".into());
    }
    let hidden = checked_u32(hidden, "native SwiGLU hidden")?;
    let intermediate_u32 = checked_u32(intermediate, "native SwiGLU intermediate")?;
    let packed_rows = ScratchLayout::new(tokens, intermediate)?
        .packed_rows(tokens)
        .filter(|_| input_packed && output_packed);
    let mmq_hit = match arithmetic {
        Arithmetic::StreamMmq => packed_rows == Some(8),
        Arithmetic::Residual2M2To8 => (2..=8).contains(&tokens),
        _ => false,
    } && stream_mmq::eligible(gate_up, 8, hidden, intermediate_u32);
    let mmq_down_hit = mmq_hit
        && arithmetic == Arithmetic::Residual2M2To8
        && stream_mmq::eligible_q4_down(down, 8, intermediate_u32, hidden);
    let native_operation = match arithmetic {
        Arithmetic::Strict | Arithmetic::Q8 => "vnext_native_swiglu",
        Arithmetic::StreamMmq if mmq_hit => "vnext_native_swiglu_stream_mmq_hit",
        Arithmetic::StreamMmq => "vnext_native_swiglu_stream_mmq_strict_fallback",
        Arithmetic::Residual2M2To8 if mmq_down_hit => "vnext_native_swiglu_residual2_m2to8_ffn_hit",
        Arithmetic::Residual2M2To8 if mmq_hit => "vnext_native_swiglu_residual2_m2to8_gate_up_hit",
        Arithmetic::Residual2M2To8 => "vnext_native_swiglu_residual2_m2to8_strict_fallback",
    };
    let launches = if packed_rows.is_some() {
        1
    } else {
        u64::from(participants)
    };
    let per_launch = if mmq_hit {
        // One pack, two MMQ projections, two fixups, original SiLU and down.
        5 + if mmq_down_hit {
            3
        } else {
            weights::dispatches(down)
        } + 1
    } else {
        dispatches_per_launch(gate_up, down, arithmetic == Arithmetic::Q8)?
    };
    let dispatches = launches
        .checked_mul(per_launch)
        .ok_or("native SwiGLU dispatch count overflows")?;
    let command = super::super::super::cost_route::compute(
        native_operation,
        if participants == 1 {
            DeviceBatchingForm::Scalar
        } else if packed_rows.is_some() {
            DeviceBatchingForm::Packed
        } else {
            DeviceBatchingForm::ParticipantLoop
        },
        participants,
        tokens,
        dispatches,
    )
    .map_err(|error| error.to_string())?;
    Ok(Selection {
        packed_rows,
        mmq_hit,
        mmq_down_hit,
        launches,
        command,
    })
}

#[cfg(test)]
mod tests;
