use super::{
    eligible, eligible_q4_down, launch_plan, weights, ActivationPrecision, Workspace, PTX,
};
use ferrum_interfaces::execution_cost::{
    KernelNumericWorkV1, KernelReplayGeometryV1, SelectedAlgorithmClassV1,
    SelectedCommandCostBuilderV1,
};
use sha2::{Digest, Sha256};
use std::sync::OnceLock;

pub(in crate::backend::cuda::vnext_ops) fn append_gate_up(
    builder: &mut SelectedCommandCostBuilderV1,
    parts: &[weights::MatrixPart],
    rows: u32,
    hidden: u32,
    intermediate: u32,
    layout: Workspace,
) -> Option<()> {
    let qualified_rows =
        if layout.precision == ActivationPrecision::Residual2Q8 && (1..=8).contains(&rows) {
            8
        } else {
            rows
        };
    if !eligible(parts, qualified_rows, hidden, intermediate) {
        return None;
    }
    append_parts(
        builder,
        parts,
        rows,
        hidden,
        intermediate,
        intermediate.checked_mul(2)?,
        layout,
    )
}

pub(in crate::backend::cuda::vnext_ops) fn append_residual_down(
    builder: &mut SelectedCommandCostBuilderV1,
    parts: &[weights::MatrixPart],
    rows: u32,
    inputs: u32,
    outputs: u32,
    layout: Workspace,
) -> Option<()> {
    if layout.precision != ActivationPrecision::Residual2Q8
        || !(1..=8).contains(&rows)
        || !eligible_q4_down(parts, 8, inputs, outputs)
    {
        return None;
    }
    append_parts(builder, parts, rows, inputs, outputs, outputs, layout)
}

fn append_parts(
    builder: &mut SelectedCommandCostBuilderV1,
    parts: &[weights::MatrixPart],
    rows: u32,
    hidden: u32,
    intermediate: u32,
    stride: u32,
    layout: Workspace,
) -> Option<()> {
    let [pack, project, fixup] = launch_plan::configs(rows, hidden, intermediate, layout).ok()?;
    let fixed = [
        u64::from(rows),
        u64::from(hidden),
        layout.words_bytes,
        layout.scales_bytes,
        layout.partial_offset,
    ];
    push(
        builder,
        layout.precision,
        layout.precision.pack_name(),
        pack,
        u64::from(rows).checked_mul(u64::from(hidden))?,
        u64::from(pack.grid_dim.0).checked_mul(256)?,
        layout.precision.terms(),
        layout.total_bytes,
        &fixed,
    )?;
    for part in parts {
        let logical = u64::from(rows).checked_mul(u64::from(intermediate))?;
        // The MMA always computes its full eight-row tile, including zero
        // lanes for shorter participants; fixup stores only the live rows.
        let padded = 8_u64
            .checked_mul(u64::from(intermediate).div_ceil(128))?
            .checked_mul(128)?;
        let fixed = [
            u64::from(rows),
            u64::from(hidden),
            u64::from(intermediate),
            u64::from(layout.ctas),
            layout.words_bytes,
            layout.scales_bytes,
            layout.partial_offset,
        ];
        push(
            builder,
            layout.precision,
            layout.precision.project_name(),
            project,
            logical,
            padded,
            u64::from(hidden).checked_mul(layout.precision.terms())?,
            layout.total_bytes,
            &fixed,
        )?;
        let fixed = [
            u64::from(rows),
            u64::from(hidden),
            u64::from(intermediate),
            u64::from(stride),
            u64::from(part.output_offset),
            u64::from(layout.ctas),
            layout.partial_offset,
        ];
        push(
            builder,
            layout.precision,
            launch_plan::FIXUP,
            fixup,
            logical,
            u64::from(fixup.grid_dim.0).checked_mul(u64::from(fixup.block_dim.0))?,
            u64::from(layout.ctas),
            layout.total_bytes,
            &fixed,
        )?;
    }
    Some(())
}

#[allow(clippy::too_many_arguments)]
fn push(
    builder: &mut SelectedCommandCostBuilderV1,
    precision: ActivationPrecision,
    entry: &'static str,
    config: cudarc::driver::LaunchConfig,
    logical: u64,
    padded: u64,
    inner: u64,
    scratch: u64,
    fixed: &[u64],
) -> Option<()> {
    static NUMERICAL: OnceLock<[u8; 32]> = OnceLock::new();
    builder
        .kernel_with_replay_geometry(
            SelectedAlgorithmClassV1::new(
                entry,
                1,
                *NUMERICAL.get_or_init(|| Sha256::digest(PTX.as_bytes()).into()),
                Sha256::digest(if precision == ActivationPrecision::SingleQ8 {
                    b"cuda.stream_mmq.q4.f16.b8.v1".as_slice()
                } else {
                    b"cuda.stream_mmq.q4.f16.residual2.m2to8.v1".as_slice()
                })
                .into(),
            )
            .ok()?,
            KernelNumericWorkV1 {
                logical_units: logical,
                padded_units: padded,
                inner_units_per_logical_unit: inner,
                grid: [config.grid_dim.0, config.grid_dim.1, config.grid_dim.2],
                scratch_bytes: scratch,
                staged_weight_bytes: 0,
            },
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
    use ferrum_interfaces::execution_cost::SelectedReplayAlgorithmTemplateV1;
    use ferrum_interfaces::vnext::WeightId;

    #[test]
    fn cuda_selected_stream_mmq_binds_pack_two_projects_fixups_and_resident_geometry() {
        let parts = [0, 256].map(|offset| weights::MatrixPart {
            component_id: WeightId::new(format!("weight.mmq.{offset}")).unwrap(),
            format: weights::MatrixFormat::Block(GgufBlockFormat::Q4K),
            rows: 256,
            columns: 512,
            output_offset: offset,
            transform: None,
            signs_region: None,
        });
        let make = |ctas| {
            let mut builder = SelectedCommandCostBuilderV1::new_with_algorithm_work(8);
            append_gate_up(
                &mut builder,
                &parts,
                8,
                512,
                256,
                Workspace::new(512, 256, ctas).unwrap(),
            )
            .unwrap();
            builder.finish().unwrap()
        };
        let original = make(1);
        original.validate_command(8, 5, 0).unwrap();
        original
            .algorithm_work()
            .unwrap()
            .unwrap()
            .validate_command(&original)
            .unwrap();
        let template =
            SelectedReplayAlgorithmTemplateV1::from_selected(&original, 8, 5, 0).unwrap();
        template.validate_binding(&make(1)).unwrap();
        assert!(template.validate_binding(&make(2)).is_err());
        let mut builder = SelectedCommandCostBuilderV1::new_with_algorithm_work(7);
        assert!(append_gate_up(
            &mut builder,
            &parts,
            7,
            512,
            256,
            Workspace::new(512, 256, 1).unwrap()
        )
        .is_none());
    }
    #[test]
    fn cuda_selected_residual2_preserves_leaf_math_and_rejects_single_q8_replay() {
        let parts = [0, 256].map(|offset| weights::MatrixPart {
            component_id: WeightId::new(format!("weight.residual.{offset}")).unwrap(),
            format: weights::MatrixFormat::Block(GgufBlockFormat::Q4K),
            rows: 256,
            columns: 512,
            output_offset: offset,
            transform: None,
            signs_region: None,
        });
        let build = |rows, precision| {
            let mut builder =
                SelectedCommandCostBuilderV1::new_with_algorithm_work(u64::from(rows));
            append_gate_up(
                &mut builder,
                &parts,
                rows,
                512,
                256,
                Workspace::with_precision(512, 256, 3, precision).unwrap(),
            )
            .unwrap();
            builder.finish().unwrap()
        };
        let single = build(8, ActivationPrecision::SingleQ8);
        let residual = build(8, ActivationPrecision::Residual2Q8);
        assert_ne!(single, residual);
        let template =
            SelectedReplayAlgorithmTemplateV1::from_selected(&residual, 8, 5, 0).unwrap();
        assert!(template.validate_binding(&single).is_err());
        template
            .validate_binding(&build(8, ActivationPrecision::Residual2Q8))
            .unwrap();
        for rows in 1..=8 {
            let evidence = build(rows, ActivationPrecision::Residual2Q8);
            evidence.validate_command(u64::from(rows), 5, 0).unwrap();
            evidence
                .algorithm_work()
                .unwrap()
                .unwrap()
                .validate_command(&evidence)
                .unwrap();
            let mut down = parts[0].clone();
            down.rows = 512;
            down.columns = 256;
            let mut b = SelectedCommandCostBuilderV1::new_with_algorithm_work(u64::from(rows));
            append_residual_down(
                &mut b,
                &[down.clone()],
                rows,
                256,
                512,
                Workspace::with_precision(256, 512, 3, ActivationPrecision::Residual2Q8).unwrap(),
            )
            .unwrap();
            b.finish()
                .unwrap()
                .validate_command(u64::from(rows), 3, 0)
                .unwrap();
            down.format = weights::MatrixFormat::Block(GgufBlockFormat::Q6K);
            let mut b = SelectedCommandCostBuilderV1::new_with_algorithm_work(u64::from(rows));
            assert!(append_residual_down(
                &mut b,
                &[down],
                rows,
                256,
                512,
                Workspace::with_precision(256, 512, 3, ActivationPrecision::Residual2Q8).unwrap()
            )
            .is_none());
        }
    }
}
