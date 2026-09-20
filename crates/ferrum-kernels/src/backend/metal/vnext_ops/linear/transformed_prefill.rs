//! Bound projection plans for a shared, completed Hadamard workspace.

use super::*;

#[derive(Debug, Clone, Copy)]
pub(super) enum TransformedLinearPlan {
    Single,
    Pq2M64Prefix {
        rows: u32,
        tail_input_offset_bytes: u64,
        tail_output_offset_bytes: u64,
        tail_workspace_offset_bytes: u64,
    },
}

impl TransformedLinearPlan {
    pub(super) fn for_launch(
        pipelines: &MetalLinearPipelines,
        regions: &[MetalBufferRegion],
        launch: LinearLaunch,
    ) -> Self {
        Self::split(pipelines, regions, launch).unwrap_or(Self::Single)
    }

    fn split(
        pipelines: &MetalLinearPipelines,
        regions: &[MetalBufferRegion],
        launch: LinearLaunch,
    ) -> Option<Self> {
        if launch.format != LinearPhysicalFormat::Native(GgufBlockFormat::Pq2_0)
            || launch.activation_type != ElementType::F16
            || launch.transform.is_none()
        {
            return None;
        }
        let rows = launch.params.rows / 64 * 64;
        // Amortize the second projection dispatch with at least two complete
        // M64 tiles. A one-tile prefix regressed with a nearly full M32 tail.
        if rows < 128 || rows == launch.params.rows {
            return None;
        }
        let (workspace_region, workspace_offset) = launch.transform_workspace?;
        let workspace = regions.get(workspace_region)?;
        let input_elements =
            u64::from(launch.params.rows).checked_mul(u64::from(launch.params.in_features))?;
        validate_region_span(
            workspace,
            workspace_offset,
            input_elements.checked_mul(4)?,
            "Metal split transformed input",
        )
        .ok()?;
        // These activations can be views of typed F16 buffers or declared raw
        // scratch. Their dtype authorization remains in the caller's launch
        // validation; the partition must still fit each retained byte span.
        validate_region_span(
            regions.get(launch.input_region)?,
            launch.input_offset_bytes,
            input_elements.checked_mul(2)?,
            "Metal split original input",
        )
        .ok()?;
        let output_elements =
            u64::from(launch.params.rows).checked_mul(u64::from(launch.params.output_stride))?;
        validate_region_span(
            regions.get(launch.output_region)?,
            launch.output_offset_bytes,
            output_elements.checked_mul(2)?,
            "Metal split output",
        )
        .ok()?;
        if launch
            .params
            .output_column_offset
            .checked_add(launch.params.out_features)?
            > launch.params.output_stride
        {
            return None;
        }
        pipelines.pq2_vector_input_prefill_pipeline(
            LinearParams {
                rows,
                ..launch.params
            },
            workspace,
            workspace_offset,
        )?;
        let prefix_inputs = u64::from(rows).checked_mul(u64::from(launch.params.in_features))?;
        let prefix_outputs = u64::from(rows).checked_mul(u64::from(launch.params.output_stride))?;
        Some(Self::Pq2M64Prefix {
            rows,
            tail_input_offset_bytes: launch
                .input_offset_bytes
                .checked_add(prefix_inputs.checked_mul(2)?)?,
            tail_output_offset_bytes: launch
                .output_offset_bytes
                .checked_add(prefix_outputs.checked_mul(2)?)?,
            tail_workspace_offset_bytes: workspace_offset
                .checked_add(prefix_inputs.checked_mul(4)?)?,
        })
    }

    pub(super) fn projection_dispatch_count(self) -> u64 {
        match self {
            Self::Single => 1,
            Self::Pq2M64Prefix { .. } => 2,
        }
    }

    pub(super) fn parts(self, launch: LinearLaunch) -> Option<[LinearLaunch; 2]> {
        let Self::Pq2M64Prefix {
            rows,
            tail_input_offset_bytes,
            tail_output_offset_bytes,
            tail_workspace_offset_bytes,
        } = self
        else {
            return None;
        };
        let mut head = launch;
        head.params.rows = rows;
        head.transformed_plan = Self::Single;
        let mut tail = launch;
        tail.params.rows -= rows;
        tail.input_offset_bytes = tail_input_offset_bytes;
        tail.output_offset_bytes = tail_output_offset_bytes;
        let (workspace_region, _) = launch
            .transform_workspace
            .expect("bound split projection has a Hadamard workspace");
        tail.transform_workspace = Some((workspace_region, tail_workspace_offset_bytes));
        tail.transformed_plan = Self::Single;
        Some([head, tail])
    }
}

pub(super) fn dispatch_tail(
    pipelines: &MetalLinearPipelines,
    encoder: &ComputeCommandEncoderRef,
    regions: &[MetalBufferRegion],
    launch: LinearLaunch,
) {
    let (workspace_region, workspace_offset) = launch
        .transform_workspace
        .expect("bound split tail has a Hadamard workspace");
    // Even a one-row tail keeps M32's F32 MMA reduction order. Selecting GEMV
    // by the sliced row count would change the original projection's math.
    encoder.set_compute_pipeline_state(&pipelines.native.pq2_gemm_input_f32_output_f16);
    set_region_offset(encoder, 0, &regions[workspace_region], workspace_offset);
    set_region_offset(encoder, 1, &regions[launch.weight_region], 0);
    set_region_offset(
        encoder,
        2,
        &regions[launch.output_region],
        launch.output_offset_bytes,
    );
    bind_linear_params(
        encoder,
        launch.params,
        launch.format,
        launch.activation_type,
    );
    dispatch_linear_grid(encoder, launch.params, LinearDispatchKind::NativeTiledGemm);
}
