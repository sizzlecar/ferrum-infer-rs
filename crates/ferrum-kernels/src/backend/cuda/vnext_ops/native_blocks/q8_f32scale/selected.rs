//! Passive Q8 projection work from the same MatrixPlan and launch descriptors.
use super::{launch_plan, matrix_plan, weights, Q8SumPolicy};
use ferrum_interfaces::execution_cost::{
    KernelNumericWorkV1, KernelReplayGeometryV1, SelectedAlgorithmClassV1,
    SelectedCommandCostBuilderV1,
};
use ferrum_interfaces::vnext::ElementType;
use sha2::{Digest, Sha256};
use std::sync::OnceLock;

pub(in crate::backend::cuda::vnext_ops) fn append_projection(
    builder: &mut SelectedCommandCostBuilderV1,
    parts: &[weights::MatrixPart],
    whole_rows: u64,
    rows: u32,
    output_stride: u32,
    policy: Q8SumPolicy,
    scratch_bytes: u64,
) -> Option<()> {
    let columns = parts.first()?.columns;
    let plan = matrix_plan(parts, whole_rows, columns, output_stride, policy).ok()?;
    let leaf = plan.leaf(u64::from(rows)).ok()?;
    if let Some(pack) = leaf.pack {
        if scratch_bytes < pack.total_bytes {
            return None;
        }
        let config = launch_plan::pack_config(rows, columns).ok()?;
        let values = u64::from(rows).checked_mul(u64::from(columns))?;
        let fixed = [
            u64::from(rows),
            u64::from(columns),
            pack.scales_bytes,
            pack.words_offset,
        ];
        push(
            builder,
            launch_plan::pack_entry(policy),
            policy,
            0,
            KernelNumericWorkV1 {
                logical_units: values,
                padded_units: u64::from(config.grid_dim.0).checked_mul(128)?,
                inner_units_per_logical_unit: 1,
                grid: [config.grid_dim.0, config.grid_dim.1, config.grid_dim.2],
                scratch_bytes,
                staged_weight_bytes: 0,
            },
            config,
            &fixed,
        )?;
    }
    for part in parts {
        let format = match part.format {
            weights::MatrixFormat::Block(format) => Some(format),
            weights::MatrixFormat::DenseF16 => None,
        };
        if let Some((format, kernel)) = format
            .zip(leaf.quantized_kernel)
            .filter(|(format, _)| launch_plan::entries(*format, policy).is_some())
        {
            let config = launch_plan::project_config(kernel, rows, part.rows);
            let (row_tile, column_tile) = kernel.tiles();
            let logical = u64::from(rows).checked_mul(u64::from(part.rows))?;
            let padded = u64::from(config.grid_dim.0)
                .checked_mul(u64::from(column_tile))?
                .checked_mul(u64::from(config.grid_dim.1))?
                .checked_mul(u64::from(row_tile))?;
            let pack = leaf.pack?;
            let fixed = [
                u64::from(rows),
                u64::from(columns),
                u64::from(part.rows),
                u64::from(output_stride),
                u64::from(part.output_offset),
                pack.scales_bytes,
                pack.words_offset,
            ];
            push(
                builder,
                launch_plan::entry(format, policy, kernel)?,
                policy,
                part.format.parameters()[0],
                KernelNumericWorkV1 {
                    logical_units: logical,
                    padded_units: padded,
                    inner_units_per_logical_unit: u64::from(columns),
                    grid: [config.grid_dim.0, config.grid_dim.1, config.grid_dim.2],
                    scratch_bytes,
                    staged_weight_bytes: 0,
                },
                config,
                &fixed,
            )?;
        } else {
            // MatrixPlan rejects transforms. Non-Q4/Q5/Q6 parts follow the
            // original strict F16 launcher, even in a mixed Q8 matrix.
            super::super::selected::append_transformed_linear(
                builder,
                part,
                rows,
                output_stride,
                ElementType::F16,
                0,
            )?;
        }
    }
    Some(())
}

fn push(
    builder: &mut SelectedCommandCostBuilderV1,
    entry: &'static str,
    policy: Q8SumPolicy,
    format: u32,
    work: KernelNumericWorkV1,
    config: cudarc::driver::LaunchConfig,
    fixed: &[u64],
) -> Option<()> {
    static NUMERICAL: OnceLock<[u8; 32]> = OnceLock::new();
    let mut layout = Sha256::new();
    layout.update(b"cuda.q8.f32scale.f16.row_major.v1");
    layout.update([match policy {
        Q8SumPolicy::Quantized => 0,
        Q8SumPolicy::Input => 1,
    }]);
    layout.update(format.to_le_bytes());
    builder
        .kernel_with_replay_geometry(
            SelectedAlgorithmClassV1::new(
                entry,
                1,
                *NUMERICAL.get_or_init(|| Sha256::digest(crate::ptx::VNEXT_GGUF.as_bytes()).into()),
                layout.finalize().into(),
            )
            .ok()?,
            work,
            KernelReplayGeometryV1 {
                block: [config.block_dim.0, config.block_dim.1, config.block_dim.2],
                dynamic_shared_bytes: u64::from(config.shared_mem_bytes),
                fixed_parameters: fixed,
            },
        )
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gguf_blocks::GgufBlockFormat;
    use ferrum_interfaces::vnext::WeightId;

    #[test]
    fn cuda_selected_q8_matrix_uses_actual_entries_and_launches_across_scalar_tiled_mma() {
        for (format, name) in [
            (GgufBlockFormat::Q4K, "q4k"),
            (GgufBlockFormat::Q5K, "q5k"),
            (GgufBlockFormat::Q6K, "q6k"),
        ] {
            for policy in [Q8SumPolicy::Quantized, Q8SumPolicy::Input] {
                for (rows, suffix, expected_grid) in [
                    (1, "dp4a_lane_f16_prototype", [65, 1, 1]),
                    (7, "dp4a_lane_tiled_f16_prototype", [65, 1, 1]),
                    (8, "mma_f16_prototype", [17, 1, 1]),
                    (33, "mma_f16_prototype", [17, 2, 1]),
                ] {
                    let part = weights::MatrixPart {
                        component_id: WeightId::new("weight.q8.selected").unwrap(),
                        format: weights::MatrixFormat::Block(format),
                        rows: 260,
                        columns: 256,
                        output_offset: 1,
                        transform: None,
                        signs_region: None,
                    };
                    let plan =
                        matrix_plan(std::slice::from_ref(&part), rows, 256, 261, policy).unwrap();
                    let leaf = plan.leaf(rows).unwrap();
                    let kernel = leaf.quantized_kernel.unwrap();
                    let sum = if policy == Q8SumPolicy::Input && format != GgufBlockFormat::Q6K {
                        "_input_sum"
                    } else {
                        ""
                    };
                    assert_eq!(
                        launch_plan::entry(format, policy, kernel).unwrap(),
                        format!("vnext_gguf_{name}_q8_f32scale{sum}_{suffix}")
                    );
                    let config = launch_plan::project_config(kernel, rows as u32, 260);
                    assert_eq!(
                        [config.grid_dim.0, config.grid_dim.1, config.grid_dim.2],
                        expected_grid
                    );
                    let mut builder = SelectedCommandCostBuilderV1::new_with_algorithm_work(rows);
                    append_projection(
                        &mut builder,
                        &[part],
                        rows,
                        rows as u32,
                        261,
                        policy,
                        plan.workspace_bytes(rows).unwrap(),
                    )
                    .unwrap();
                    let evidence = builder.finish().unwrap();
                    evidence.validate_command(rows, 2, 0).unwrap();
                    evidence
                        .algorithm_work()
                        .unwrap()
                        .unwrap()
                        .validate_command(&evidence)
                        .unwrap();
                }
            }
        }
    }
}
