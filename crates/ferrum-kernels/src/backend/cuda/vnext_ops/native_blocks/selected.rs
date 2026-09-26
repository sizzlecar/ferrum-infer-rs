//! Checked native matrix work, shared by actual encoder and future route.
//! No tensor decode, device operation or selected table is created when Off.
use super::{hadamard, linear_launch, weights};
use crate::backend::cuda::vnext_runtime::selected_cost;
use ferrum_interfaces::execution_cost::{
    KernelNumericWorkV1, KernelReplayGeometryV1, SelectedAlgorithmClassV1,
    SelectedCommandCostBuilderV1, SelectedCommandCostEvidenceV1,
};
use ferrum_interfaces::vnext::{ElementType, HadamardApplication, HadamardSigns};
use ferrum_types::SloStructuredCostCapture;
use sha2::{Digest, Sha256};
use std::sync::OnceLock;

pub(in crate::backend::cuda::vnext_ops) fn linear(
    parts: &[weights::MatrixPart],
    rows: impl IntoIterator<Item = u32>,
    tokens: u64,
    output_stride: u32,
    activation: ElementType,
    transform_scratch_bytes: u64,
    capture: SloStructuredCostCapture,
) -> Option<SelectedCommandCostEvidenceV1> {
    let mut builder = selected_cost::builder(capture, tokens)?;
    if parts.is_empty() {
        return None;
    }
    for rows in rows {
        for part in parts {
            append_transformed_linear(
                &mut builder,
                part,
                rows,
                output_stride,
                activation,
                transform_scratch_bytes,
            )?;
        }
    }
    builder.finish().ok()
}

pub(in crate::backend::cuda::vnext_ops) fn append_transformed_linear(
    builder: &mut SelectedCommandCostBuilderV1,
    part: &weights::MatrixPart,
    rows: u32,
    output_stride: u32,
    activation: ElementType,
    transform_scratch_bytes: u64,
) -> Option<()> {
    if part.rows == 0 || part.columns == 0 || rows == 0 || rows > u16::MAX as u32 {
        return None;
    }
    let [_, block_values, _] = part.format.parameters();
    if !part.columns.is_multiple_of(block_values) {
        return None;
    }
    let (input, scratch_bytes) = if let Some(spec) = &part.transform {
        hadamard::validate(spec, u64::from(part.columns)).ok()?;
        let HadamardApplication::BeforeMatmul { input_permutation } = spec.application else {
            return None;
        };
        if matches!(spec.signs, HadamardSigns::Explicit(_)) != part.signs_region.is_some() {
            return None;
        }
        let entry = hadamard::launch_entry(activation, ElementType::F32).ok()?;
        let elements = u64::from(rows).checked_mul(u64::from(part.columns))?;
        let required = elements.checked_mul(4)?;
        if transform_scratch_bytes < required {
            return None;
        }
        let scratch = transform_scratch_bytes;
        let block = spec.block_size.get();
        let mut layout = Sha256::new();
        layout.update(b"cuda.native_hadamard.before_matmul.v1");
        layout.update(block.to_le_bytes());
        layout.update([u8::from(part.signs_region.is_some())]);
        if let Some(permutation) = input_permutation {
            layout.update([1]);
            layout.update(permutation.inner_extent.to_le_bytes());
            layout.update(permutation.first_outer_extent.to_le_bytes());
            layout.update(permutation.second_outer_extent.to_le_bytes());
        } else {
            layout.update([0]);
        }
        let permutation = input_permutation.map_or([0, 0, 0], |p| {
            [p.inner_extent, p.first_outer_extent, p.second_outer_extent]
        });
        let fixed = [
            u64::from(part.columns),
            u64::from(block),
            0,
            permutation[0],
            permutation[1],
            permutation[2],
        ];
        builder
            .kernel_with_replay_geometry(
                algorithm(entry, layout.finalize().into())?,
                KernelNumericWorkV1 {
                    logical_units: elements,
                    padded_units: elements,
                    inner_units_per_logical_unit: u64::from(block.ilog2()) + 1,
                    grid: [part.columns / block, rows, 1],
                    scratch_bytes: scratch,
                    staged_weight_bytes: 0,
                },
                KernelReplayGeometryV1 {
                    block: [256, 1, 1],
                    dynamic_shared_bytes: u64::from(block) * 4,
                    fixed_parameters: &fixed,
                },
            )
            .ok()?;
        (ElementType::F32, scratch)
    } else {
        if part.signs_region.is_some() {
            return None;
        }
        (activation, 0)
    };
    let selected = linear_launch::select(part, rows, output_stride, input, activation).ok()?;
    let logical = u64::from(rows).checked_mul(u64::from(part.rows))?;
    let config = selected.config;
    let padded = u64::from(config.grid_dim.0)
        .checked_mul(u64::from(selected.tile[0]))?
        .checked_mul(u64::from(config.grid_dim.1))?
        .checked_mul(u64::from(selected.tile[1]))?;
    let mut layout = Sha256::new();
    layout.update(b"cuda.native_matrix.row_major.strided_output.v1");
    for parameter in part.format.parameters() {
        layout.update(parameter.to_le_bytes());
    }
    for tile in selected.tile {
        layout.update(tile.to_le_bytes());
    }
    for dimension in [config.block_dim.0, config.block_dim.1, config.block_dim.2] {
        layout.update(dimension.to_le_bytes());
    }
    // M/N/K and output offset/stride are numeric/binding facts, not an
    // algorithm family. The exact physical weight/row binding stays in the
    // original selected recipe and is validated before calling this helper.
    let [format, values, bytes] = part.format.parameters();
    let fixed = [
        rows,
        part.columns,
        part.rows,
        output_stride,
        part.output_offset,
        format,
        values,
        bytes,
    ]
    .map(u64::from);
    builder
        .kernel_with_replay_geometry(
            algorithm(selected.kernel.entry(), layout.finalize().into())?,
            KernelNumericWorkV1 {
                logical_units: logical,
                padded_units: padded,
                inner_units_per_logical_unit: u64::from(part.columns),
                grid: [config.grid_dim.0, config.grid_dim.1, config.grid_dim.2],
                scratch_bytes,
                // These kernels read original native weights; none creates a
                // staged weight tensor. Kernel-local shared tiles are not a
                // second global staging allocation.
                staged_weight_bytes: 0,
            },
            KernelReplayGeometryV1 {
                block: [config.block_dim.0, config.block_dim.1, config.block_dim.2],
                dynamic_shared_bytes: u64::from(config.shared_mem_bytes),
                fixed_parameters: &fixed,
            },
        )
        .ok()
}

fn algorithm(entry: &'static str, layout: [u8; 32]) -> Option<SelectedAlgorithmClassV1> {
    static NUMERICAL: OnceLock<[u8; 32]> = OnceLock::new();
    SelectedAlgorithmClassV1::new(
        entry,
        1,
        *NUMERICAL.get_or_init(|| Sha256::digest(crate::ptx::VNEXT_GGUF.as_bytes()).into()),
        layout,
    )
    .ok()
}

#[cfg(test)]
mod tests;
