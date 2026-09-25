//! Explicit Q6 head contract: half-rounded operands, F32 sums and logits.

use super::*;
use ferrum_interfaces::vnext::{
    last_token_dense_linear_f32_f16_operands_contract,
    LAST_TOKEN_DENSE_LINEAR_F32_F16_OPERANDS_CAPABILITY_ID,
    LAST_TOKEN_DENSE_LINEAR_F32_F16_OPERANDS_OPERATION_ID,
};

const PROVIDER_ID: &str = "provider.metal.last_token_dense_linear.f32.f16-operands.q6";
const ESTIMATOR_ID: &str = "resource-estimator.metal.last_token_dense_linear.f32.f16-operands.q6";
const SHARED_BYTES: u64 = 8192;
const SMALL_SOURCE: &str = include_str!("half_head.metal");
const TILED_SOURCE: &str = include_str!("half_head_tiled.metal");

pub(in super::super) struct MetalHalfHeadProvider {
    descriptor: OperationProviderDescriptor,
    pipelines: Arc<HalfHeadPipelines>,
}

impl MetalHalfHeadProvider {
    pub(in super::super) fn new(
        runtime: &MetalDeviceRuntime,
    ) -> Result<Self, MetalDeviceRuntimeError> {
        let contract = last_token_dense_linear_f32_f16_operands_contract()
            .map_err(super::super::contract_error)?;
        let descriptor = provider_descriptor(
            runtime,
            &contract,
            PROVIDER_ID,
            LAST_TOKEN_DENSE_LINEAR_F32_F16_OPERANDS_CAPABILITY_ID,
            ESTIMATOR_ID,
            contiguous_bindings(2),
            &[GGUF_NATIVE_BLOCK_FORMAT_ID],
            &[Q6_K_FORMAT_ID],
            implementation_fingerprint(&[FINGERPRINT_SOURCE.as_bytes(), PROVIDER_ID.as_bytes()]),
        )?
        .with_checkpoint_capability(ProviderCheckpointCapability::CompletedBoundary(
            ProviderCheckpointContract::new(
                CheckpointInputDependency::ExactTokenPrefix,
                CheckpointBoundaryConstraint::any_positive(),
                CheckpointPartitionNumerics::CapturedExecutionContinuation,
            )
            .with_completed_input_capture(CheckpointCompletedInputCapture::Supported),
        ));
        Ok(Self {
            descriptor,
            pipelines: Arc::new(
                HalfHeadPipelines::new(runtime.device())?
                    .with_structured_capture(runtime.structured_capture()),
            ),
        })
    }
}

impl OperationResourceEstimator for MetalHalfHeadProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        if request.operation().id.as_str() != LAST_TOKEN_DENSE_LINEAR_F32_F16_OPERANDS_OPERATION_ID
            || request.operation().fingerprint()? != self.descriptor.operation_fingerprint()
        {
            return Err(invalid_plan(
                "Metal half-head estimator received another operation",
            ));
        }
        let hidden =
            unsigned_attribute(request.attributes(), "hidden_size").map_err(invalid_plan)?;
        let outputs =
            unsigned_attribute(request.attributes(), "out_features").map_err(invalid_plan)?;
        let shape = LinearParams {
            rows: 1,
            in_features: checked_u32(hidden, "half-head hidden size").map_err(invalid_plan)?,
            out_features: checked_u32(outputs, "half-head output size").map_err(invalid_plan)?,
            output_stride: checked_u32(outputs, "half-head output stride").map_err(invalid_plan)?,
            output_column_offset: 0,
        };
        half_params(shape).map_err(invalid_plan)?;
        if hadamard::workspace_bytes_per_token(request.values()).map_err(invalid_plan)? != 0 {
            return Err(invalid_plan(
                "Metal half-head does not support transformed weights",
            ));
        }
        let bytes = last_token_scratch_bytes_per_sequence(hidden, outputs, ElementType::F32)
            .map_err(invalid_plan)?;
        // Only invocation-local F32 gather/scatter storage. Q6 weights remain
        // quantized; half operands live in registers/threadgroup memory.
        let scratch = ProviderWorkspaceRequirement::from_formula(
            ProviderWorkspaceSizeFormula::affine(LAST_TOKEN_SCRATCH_PADDING_BYTES, bytes, 0)?,
            VALUE_ALIGNMENT_BYTES,
            ProviderWorkspaceScope::Invocation,
            ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
            DynamicStorageRequirement::contiguous(),
        )?;
        Ok(OperationResourceEstimate::new(
            self.descriptor.resource_estimator_id(),
            self.descriptor.resource_estimator_version(),
            self.descriptor
                .resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            VALUE_ALIGNMENT_BYTES,
            Some(scratch),
            None,
        ))
    }
}

impl OperationProvider<MetalDeviceRuntime> for MetalHalfHeadProvider {
    fn eager_cost_route(
        &self,
        request: ferrum_interfaces::vnext::OperationCostRouteRequest<'_>,
    ) -> Result<Option<ferrum_interfaces::vnext::OperationCostRoute>, VNextError> {
        head_cost_route::route(
            request,
            LAST_TOKEN_DENSE_LINEAR_F32_F16_OPERANDS_OPERATION_ID,
            ElementType::F32,
            &LastTokenProjection::Half(Arc::clone(&self.pipelines)),
        )
    }

    fn reusable_execution_topology(
        &self,
        _request: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        authorize_reusable_topology(self.descriptor.execution_semantics(), || {
            Ok(ReusableExecutionTopology::Static)
        })
    }

    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, MetalDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<MetalDeviceCommand>, OperationFailure> {
        let identity = invocation.participants()[0].identity().clone();
        encode_last_token_dense_linear(
            LastTokenProjection::Half(Arc::clone(&self.pipelines)),
            LAST_TOKEN_DENSE_LINEAR_F32_F16_OPERANDS_OPERATION_ID,
            ElementType::F32,
            invocation,
        )
        .map(EncodedDeviceOperation::compute)
        .map_err(|message| {
            provider_failure(identity, "metal.last_token.f16_operands.encode", message)
        })
    }
}

/// Shares final-row selection, gather/scatter and resource lifetimes without
/// allowing a different operation to fall back to strict F32 multiplication.
pub(super) enum LastTokenProjection {
    Strict(Arc<MetalLinearPipelines>),
    Half(Arc<HalfHeadPipelines>),
}

#[derive(Clone, Copy, Debug)]
pub(super) enum LastTokenProjectionKind {
    Strict,
    Half,
}

impl LastTokenProjectionKind {
    pub(super) fn validate_part(self, part: PreparedLinearPart) -> Result<(), String> {
        if matches!(self, Self::Half)
            && (part.format != LinearPhysicalFormat::Q6K || part.transform.is_some())
        {
            return Err("Metal half-head requires one untransformed native Q6K matrix".into());
        }
        Ok(())
    }

    pub(super) fn dispatch_count(self, launch: LinearLaunch) -> u64 {
        match self {
            Self::Strict => launch.dispatch_count(),
            Self::Half => half_dispatch_count(launch.params.rows),
        }
    }

    pub(super) fn operation_label(self, dtype: ElementType) -> &'static str {
        match self {
            Self::Half => "vnext_last_token_dense_linear_f32_f16_operands",
            Self::Strict if dtype == ElementType::F32 => "vnext_last_token_dense_linear_f32",
            Self::Strict => "vnext_last_token_dense_linear",
        }
    }

    pub(super) fn validate_numeric_launch(self, launch: LinearLaunch) -> Result<(), String> {
        launch.activation_bytes()?;
        if matches!(self, Self::Half) {
            half_params(launch.params)?;
            if launch.activation_type != ElementType::F32 {
                return Err("Metal half-head requires F32 activation storage".into());
            }
        }
        Ok(())
    }
}

impl LastTokenProjection {
    pub(super) fn structured_capture(&self) -> ferrum_types::SloStructuredCostCapture {
        match self {
            Self::Strict(pipelines) => pipelines.structured_capture(),
            Self::Half(pipelines) => pipelines.structured_capture(),
        }
    }

    pub(super) fn kind(&self) -> LastTokenProjectionKind {
        match self {
            Self::Strict(_) => LastTokenProjectionKind::Strict,
            Self::Half(_) => LastTokenProjectionKind::Half,
        }
    }

    pub(super) fn validate_part(&self, part: PreparedLinearPart) -> Result<(), String> {
        self.kind().validate_part(part)
    }

    pub(super) fn bind_hadamard_workspace(
        &self,
        launch: &mut LinearLaunch,
        regions: &[MetalBufferRegion],
        index: usize,
        offset: u64,
    ) -> Result<(), String> {
        match self {
            Self::Strict(pipelines) => {
                launch.bind_hadamard_workspace(pipelines, regions, index, offset)
            }
            Self::Half(_) if launch.transform.is_none() => Ok(()),
            Self::Half(_) => Err("Metal half-head cannot bind a transform".into()),
        }
    }

    pub(super) fn validate_launch(
        &self,
        regions: &[MetalBufferRegion],
        launch: LinearLaunch,
        raw: &[usize],
    ) -> Result<(), String> {
        if let Self::Half(_) = self {
            validate_half_launch(regions, launch, raw)?;
        }
        Ok(())
    }

    pub(super) fn dispatch_count(&self, launch: LinearLaunch) -> u64 {
        self.kind().dispatch_count(launch)
    }

    pub(super) fn operation_label(&self, dtype: ElementType) -> &'static str {
        self.kind().operation_label(dtype)
    }

    pub(super) fn dispatch(
        &self,
        encoder: &ComputeCommandEncoderRef,
        regions: &[MetalBufferRegion],
        launch: LinearLaunch,
    ) {
        match self {
            Self::Strict(pipelines) => dispatch_linear(pipelines, encoder, regions, launch),
            Self::Half(pipelines) => pipelines.dispatch(encoder, regions, launch),
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct HalfParams {
    out_features: i32,
    rows: i32,
    in_features: i32,
    weight_row_bytes: i32,
    output_stride: i32,
}

fn half_params(p: LinearParams) -> Result<HalfParams, String> {
    if p.rows == 0 || p.out_features == 0 || p.in_features == 0 || p.in_features % 256 != 0 {
        return Err("Metal half-head requires positive M/N and K divisible by 256".into());
    }
    if u64::from(p.output_column_offset) + u64::from(p.out_features) > u64::from(p.output_stride) {
        return Err("Metal half-head output columns exceed stride".into());
    }
    let row_bytes = u64::from(p.in_features / 256) * 210;
    for span in [
        u64::from(p.rows) * u64::from(p.in_features),
        u64::from(p.out_features) * row_bytes,
        u64::from(p.rows - 1) * u64::from(p.output_stride) + u64::from(p.out_features),
    ] {
        if span > i32::MAX as u64 {
            return Err("Metal half-head shader relative address exceeds i32".into());
        }
    }
    let checked =
        |x: u64| i32::try_from(x).map_err(|_| "Metal half-head parameter exceeds i32".to_owned());
    Ok(HalfParams {
        out_features: checked(u64::from(p.out_features))?,
        rows: checked(u64::from(p.rows))?,
        in_features: checked(u64::from(p.in_features))?,
        weight_row_bytes: checked(row_bytes)?,
        output_stride: checked(u64::from(p.output_stride))?,
    })
}

pub(super) fn validate_half_launch(
    regions: &[MetalBufferRegion],
    launch: LinearLaunch,
    raw: &[usize],
) -> Result<(), String> {
    if launch.activation_type != ElementType::F32
        || launch.format != LinearPhysicalFormat::Q6K
        || launch.transform.is_some()
    {
        return Err("Metal half-head launch requires Q6K and untransformed F32 boundaries".into());
    }
    validate_launch_regions_with_raw_workspace(regions, &[launch], raw)?;
    let p = half_params(launch.params)?;
    let input = &regions[launch.input_region];
    let output = &regions[launch.output_region];
    let weight = &regions[launch.weight_region];
    let input_offset = input
        .offset_bytes()
        .checked_add(launch.input_offset_bytes)
        .ok_or("half-head input offset overflow")?;
    let output_offset = output
        .offset_bytes()
        .checked_add(launch.output_offset_bytes)
        .and_then(|x| x.checked_add(u64::from(launch.params.output_column_offset) * 4))
        .ok_or("half-head output offset overflow")?;
    // The tiled path loads float2x4. Keep this uniform input contract across
    // cohort widths; K256 also keeps every selected row and tile aligned.
    if input_offset % 16 != 0 || output_offset % 4 != 0 || weight.offset_bytes() % 2 != 0 {
        return Err("Metal half-head requires input16/output4/Q6K2 byte alignment".into());
    }
    let weight_bytes = u64::from(launch.params.out_features)
        .checked_mul(p.weight_row_bytes as u64)
        .ok_or("half-head weight span overflow")?;
    validate_region_span(weight, 0, weight_bytes, "Metal half-head Q6K weights")?;
    Ok(())
}

pub(super) fn half_dispatch_count(rows: u32) -> u64 {
    if rows < 8 {
        u64::from(rows.div_ceil(4))
    } else {
        1
    }
}

pub(super) struct HalfHeadPipelines {
    structured_capture: ferrum_types::SloStructuredCostCapture,
    small: [ComputePipelineState; 4],
    tiled: ComputePipelineState,
}

impl HalfHeadPipelines {
    pub(super) fn with_structured_capture(
        mut self,
        capture: ferrum_types::SloStructuredCostCapture,
    ) -> Self {
        self.structured_capture = capture;
        self
    }

    pub(super) fn structured_capture(&self) -> ferrum_types::SloStructuredCostCapture {
        self.structured_capture
    }

    pub(super) fn new(device: &Device) -> Result<Self, MetalDeviceRuntimeError> {
        // The public half-operands contract includes Inf/NaN classification.
        // MSL fast math leaves those values undefined; conversion itself is RNE.
        let options = CompileOptions::new();
        options.set_fast_math_enabled(false);
        let small = device
            .new_library_with_source(SMALL_SOURCE, &options)
            .map_err(|e| {
                MetalDeviceRuntimeError::contract(format!("compile half-head small: {e}"))
            })?;
        let tiled = device
            .new_library_with_source(TILED_SOURCE, &options)
            .map_err(|e| {
                MetalDeviceRuntimeError::contract(format!("compile half-head tiled: {e}"))
            })?;
        let build = |library: &metal::Library, name: &str, threads: u64, shared: u64| {
            let function = library.get_function(name, None).map_err(|e| {
                MetalDeviceRuntimeError::contract(format!("load half-head {name}: {e}"))
            })?;
            let pipeline = device
                .new_compute_pipeline_state_with_function(&function)
                .map_err(|e| {
                    MetalDeviceRuntimeError::contract(format!("build half-head {name}: {e}"))
                })?;
            if pipeline.thread_execution_width() != 32
                || pipeline.max_total_threads_per_threadgroup() < threads
                || pipeline
                    .static_threadgroup_memory_length()
                    .checked_add(shared)
                    .is_none_or(|x| x > device.max_threadgroup_memory_length())
                || (threads == 128 && device.max_threads_per_threadgroup().width < 128)
                || (threads == 64
                    && (device.max_threads_per_threadgroup().width < 32
                        || device.max_threads_per_threadgroup().height < 2))
            {
                return Err(MetalDeviceRuntimeError::contract(
                    "unsupported Metal half-head threadgroup resources",
                ));
            }
            Ok(pipeline)
        };
        Ok(Self {
            structured_capture: ferrum_types::SloStructuredCostCapture::Disabled,
            small: [
                build(&small, "q6_half_head_b1", 64, 0)?,
                build(&small, "q6_half_head_b2", 64, 0)?,
                build(&small, "q6_half_head_b3", 64, 0)?,
                build(&small, "q6_half_head_b4", 64, 0)?,
            ],
            tiled: build(&tiled, "gemm_q6kw_f32a_f32o", 128, SHARED_BYTES)?,
        })
    }

    pub(super) fn dispatch(
        &self,
        encoder: &ComputeCommandEncoderRef,
        regions: &[MetalBufferRegion],
        launch: LinearLaunch,
    ) {
        if launch.params.rows >= 8 {
            let params =
                half_params(launch.params).expect("half-head launch validated before encoding");
            encoder.set_compute_pipeline_state(
                self.selected_pipeline(launch.params)
                    .expect("validated half-head")
                    .0,
            );
            set_region_offset(encoder, 0, &regions[launch.weight_region], 0);
            set_region_offset(
                encoder,
                1,
                &regions[launch.input_region],
                launch.input_offset_bytes,
            );
            set_region_offset(
                encoder,
                2,
                &regions[launch.output_region],
                launch.output_offset_bytes + u64::from(launch.params.output_column_offset) * 4,
            );
            encoder.set_bytes(
                3,
                std::mem::size_of::<HalfParams>() as u64,
                (&params as *const HalfParams).cast::<c_void>(),
            );
            encoder.set_threadgroup_memory_length(0, SHARED_BYTES);
            encoder.dispatch_thread_groups(
                MTLSize::new(
                    u64::from(launch.params.rows).div_ceil(32),
                    u64::from(launch.params.out_features).div_ceil(64),
                    1,
                ),
                MTLSize::new(128, 1, 1),
            );
        } else {
            for (first, rows) in small_parts(launch.params.rows).into_iter().flatten() {
                let params = LinearParams {
                    rows,
                    ..launch.params
                };
                encoder.set_compute_pipeline_state(
                    self.selected_pipeline(params)
                        .expect("validated half-head small")
                        .0,
                );
                set_region_offset(
                    encoder,
                    0,
                    &regions[launch.input_region],
                    launch.input_offset_bytes
                        + u64::from(first) * u64::from(params.in_features) * 4,
                );
                set_region_offset(encoder, 1, &regions[launch.weight_region], 0);
                set_region_offset(
                    encoder,
                    2,
                    &regions[launch.output_region],
                    launch.output_offset_bytes
                        + u64::from(first) * u64::from(params.output_stride) * 4,
                );
                encoder.set_bytes(
                    3,
                    std::mem::size_of::<LinearParams>() as u64,
                    (&params as *const LinearParams).cast::<c_void>(),
                );
                encoder.dispatch_thread_groups(
                    MTLSize::new(u64::from(params.out_features).div_ceil(4), 1, 1),
                    MTLSize::new(32, 2, 1),
                );
            }
        }
    }
}

#[cfg(test)]
mod tests;

fn small_parts(rows: u32) -> [Option<(u32, u32)>; 2] {
    debug_assert!((1..8).contains(&rows));
    [Some((0, rows.min(4))), (rows > 4).then(|| (4, rows - 4))]
}
impl HalfHeadPipelines {
    fn selected_pipeline(
        &self,
        params: LinearParams,
    ) -> Option<(&ComputePipelineState, &'static str)> {
        if params.rows >= 8 {
            Some((&self.tiled, "gemm_q6kw_f32a_f32o"))
        } else {
            let at = params.rows.checked_sub(1)? as usize;
            let entry = [
                "q6_half_head_b1",
                "q6_half_head_b2",
                "q6_half_head_b3",
                "q6_half_head_b4",
            ]
            .get(at)?;
            Some((self.small.get(at)?, *entry))
        }
    }
    pub(super) fn append_statistical(
        &self,
        b: &mut ferrum_interfaces::execution_cost::SelectedCommandCostBuilderV1,
        launch: LinearLaunch,
        scratch: u64,
    ) -> Option<()> {
        use ferrum_interfaces::execution_cost::{KernelNumericWorkV1, SelectedAlgorithmClassV1};
        use sha2::{Digest, Sha256};
        use std::sync::OnceLock;
        LastTokenProjectionKind::Half
            .validate_part(PreparedLinearPart {
                region: launch.weight_region,
                format: launch.format,
                out_features: launch.params.out_features,
                output_offset: 0,
                transform: launch.transform,
            })
            .ok()?;
        LastTokenProjectionKind::Half
            .validate_numeric_launch(launch)
            .ok()?;
        static NUMERIC: OnceLock<[u8; 32]> = OnceLock::new();
        let numeric = *NUMERIC.get_or_init(|| {
            let mut h = Sha256::new();
            h.update(b"metal.half-head.fast-math-disabled.v1");
            h.update(SMALL_SOURCE.as_bytes());
            h.update(TILED_SOURCE.as_bytes());
            h.update(include_str!("half_head.rs").as_bytes());
            h.finalize().into()
        });
        let tiled = launch.params.rows >= 8;
        let parts = if tiled {
            [Some((0, launch.params.rows)), None]
        } else {
            small_parts(launch.params.rows)
        };
        for (_, rows) in parts.into_iter().flatten() {
            let params = LinearParams {
                rows,
                ..launch.params
            };
            let (_, entry) = self.selected_pipeline(params)?;
            let (grid, padded, threads, shared) = if tiled {
                let x = u64::from(rows).div_ceil(32);
                let y = u64::from(params.out_features).div_ceil(64);
                (
                    [u32::try_from(x).ok()?, u32::try_from(y).ok()?, 1],
                    x.checked_mul(32)?.checked_mul(y)?.checked_mul(64)?,
                    [128u32, 1, 1],
                    SHARED_BYTES,
                )
            } else {
                let x = u64::from(params.out_features).div_ceil(4);
                (
                    [u32::try_from(x).ok()?, 1, 1],
                    x.checked_mul(4)?.checked_mul(u64::from(rows))?,
                    [32u32, 2, 1],
                    0,
                )
            };
            let mut layout = Sha256::new();
            layout.update(b"half-head.q6-f32-boundary.params.v1");
            for n in [
                params.output_stride,
                params.output_column_offset,
                threads[0],
                threads[1],
                threads[2],
            ] {
                layout.update(n.to_le_bytes());
            }
            layout.update(shared.to_le_bytes());
            let class =
                SelectedAlgorithmClassV1::new(entry, 1, numeric, layout.finalize().into()).ok()?;
            b.kernel(
                class,
                KernelNumericWorkV1 {
                    logical_units: u64::from(rows).checked_mul(u64::from(params.out_features))?,
                    padded_units: padded,
                    inner_units_per_logical_unit: u64::from(params.in_features),
                    grid,
                    scratch_bytes: scratch,
                    staged_weight_bytes: 0,
                },
            )
            .ok()?;
        }
        Some(())
    }
}
