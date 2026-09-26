//! Passive work from the actual selection launch ABI. Penalty values and
//! history offsets live on device, so the work describes the declared bounded
//! input, not an assertion that the no-penalty branch will execute.
use super::*;
use crate::backend::cuda::vnext_runtime::selected_cost;
use ferrum_interfaces::execution_cost::{
    KernelNumericWorkV1, KernelReplayGeometryV1, SelectedAlgorithmClassV1,
    SelectedCommandCostBuilderV1, SelectedCommandCostEvidenceV1,
};
use ferrum_types::SloStructuredCostCapture;
use sha2::{Digest, Sha256};
use std::sync::OnceLock;

pub(in crate::backend::cuda::vnext_ops) fn evidence(
    precision: ArgmaxPrecision,
    participants: u32,
    rows: impl IntoIterator<Item = (i32, i32)>,
    capture: SloStructuredCostCapture,
) -> Option<SelectedCommandCostEvidenceV1> {
    let mut builder = selected_cost::builder(capture, u64::from(participants))?;
    if participants == 0 {
        return None;
    }
    let mut count = 0_u32;
    for (vocabulary, repetition_capacity) in rows {
        if vocabulary <= 0 || repetition_capacity <= 0 {
            return None;
        }
        count = count.checked_add(1)?;
        let vocabulary = u64::try_from(vocabulary).ok()?;
        let repetition_capacity = u64::try_from(repetition_capacity).ok()?;
        let parallel = argmax_dispatches(vocabulary as i32) == 2;
        let first = first_launch_config(parallel);
        let threads = u64::from(first.grid_dim.0) * u64::from(first.block_dim.0);
        // These are input extents, not instruction counts. Penalty scratch
        // copies and reductions are part of this kernel's learned cost;
        // actual history offsets and penalty remain runtime data.
        let logical = vocabulary.checked_add(repetition_capacity)?;
        let padded = vocabulary
            .div_ceil(threads)
            .checked_mul(threads)?
            .checked_add(repetition_capacity)?;
        append(
            &mut builder,
            if parallel {
                precision.partitioned_kernel()
            } else {
                precision.kernel()
            },
            first,
            logical,
            padded,
            masked_argmax_scratch_stride(vocabulary, precision.element()).ok()?,
            &[vocabulary, vocabulary, repetition_capacity],
        )?;
        if parallel {
            append(
                &mut builder,
                FINALIZE_ENTRY,
                finalize_launch_config(),
                u64::from(ARGMAX_PARTITIONS),
                u64::from(ARGMAX_PARTITIONS),
                u64::from(ARGMAX_PARTITIONS) * 8,
                &[repetition_capacity],
            )?;
        }
    }
    if count != participants {
        return None;
    }
    builder.finish().ok()
}

fn append(
    builder: &mut SelectedCommandCostBuilderV1,
    entry: &'static str,
    config: LaunchConfig,
    logical: u64,
    padded: u64,
    scratch_bytes: u64,
    fixed_parameters: &[u64],
) -> Option<()> {
    static NUMERICAL: OnceLock<[u8; 32]> = OnceLock::new();
    let algorithm = SelectedAlgorithmClassV1::new(
        entry,
        1,
        *NUMERICAL.get_or_init(|| Sha256::digest(crate::ptx::ARGMAX_ROWS.as_bytes()).into()),
        Sha256::digest(b"cuda.masked_argmax.contiguous_logits_and_bounded_history.v1").into(),
    )
    .ok()?;
    builder
        .kernel_with_replay_geometry(
            algorithm,
            KernelNumericWorkV1 {
                logical_units: logical,
                padded_units: padded,
                inner_units_per_logical_unit: 1,
                grid: [config.grid_dim.0, config.grid_dim.1, config.grid_dim.2],
                scratch_bytes,
                staged_weight_bytes: 0,
            },
            KernelReplayGeometryV1 {
                block: [config.block_dim.0, config.block_dim.1, config.block_dim.2],
                dynamic_shared_bytes: u64::from(config.shared_mem_bytes),
                fixed_parameters,
            },
        )
        .ok()
}

#[cfg(test)]
mod tests;
