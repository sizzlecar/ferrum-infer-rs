use super::*;

pub(super) fn direct_square_pipeline(
    runtime: &MetalDeviceRuntime,
    library: &metal::LibraryRef,
    kernel_size: u32,
) -> Result<Option<ComputePipelineState>, MetalDeviceRuntimeError> {
    let constants = metal::FunctionConstantValues::new();
    constants.set_constant_value_at_index(
        (&kernel_size as *const u32).cast(),
        metal::MTLDataType::UInt,
        102,
    );
    specialized_pipeline(runtime, library, constants)
}

pub(super) fn masked_3x3_pipeline(
    runtime: &MetalDeviceRuntime,
    library: &metal::LibraryRef,
) -> Result<Option<ComputePipelineState>, MetalDeviceRuntimeError> {
    let constants = metal::FunctionConstantValues::new();
    let masked = true;
    constants.set_constant_value_at_index(
        (&masked as *const bool).cast(),
        metal::MTLDataType::Bool,
        103,
    );
    specialized_pipeline(runtime, library, constants)
}

fn specialized_pipeline(
    runtime: &MetalDeviceRuntime,
    library: &metal::LibraryRef,
    constants: metal::FunctionConstantValues,
) -> Result<Option<ComputePipelineState>, MetalDeviceRuntimeError> {
    let function = library
        .get_function("ferrum_tensor_conv2d_bf16", Some(constants))
        .map_err(MetalDeviceRuntimeError::contract)?;
    let pipeline = runtime
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(MetalDeviceRuntimeError::contract)?;
    Ok((pipeline.thread_execution_width() == 32
        && pipeline.max_total_threads_per_threadgroup() >= 128
        && pipeline.static_threadgroup_memory_length()
            <= runtime.device().max_threadgroup_memory_length())
    .then_some(pipeline))
}

pub(super) struct Shape {
    g: Conv2dBf16Geometry,
    bytes: Vec<u64>,
    pub scratch_bytes: u64,
    virtual_padding: bool,
    jobs: u64,
}
pub(super) fn geometry(
    values: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
    masked_3x3_available: bool,
) -> Result<Shape, String> {
    let bytes = dense_ports(MetalTensorOperation::Conv2d, values)?;
    let input = |i| binding(values, ResolvedValueRole::Input, i).map(|v| v.tensor());
    let g = Conv2dBf16Geometry::new(
        input(0)?,
        input(1)?,
        input(2)?,
        binding(values, ResolvedValueRole::Output, 0)?.tensor(),
        attrs,
    )
    .map_err(|e| e.to_string())?;
    // The bounded loader presents the same conceptual external +0 padding
    // and still multiplies every padded tap by its original weight. Only
    // physical materialization is omitted; dot order and BF16 boundaries stay.
    // Estimation and encoding use this same actual-PSO capability decision.
    let virtual_padding = masked_3x3_available
        && g.padding_mode() == Conv2dPaddingMode::ExplicitZero
        && g.padding() != [0; 4]
        && g.kernel_height() == 3
        && g.kernel_width() == 3
        && g.stride() == [1, 1]
        && g.dilation() == [1, 1];
    let scratch_bytes = if g.padding_mode() == Conv2dPaddingMode::ExplicitZero
        && g.padding() != [0; 4]
        && !virtual_padding
    {
        g.padded_input_bytes()
    } else {
        0
    };
    if scratch_bytes > isize::MAX as u64 {
        return Err("Metal padded convolution exceeds address space".into());
    }
    let jobs = g
        .batch()
        .checked_mul(g.groups())
        .and_then(|n| n.checked_mul((g.output_height() * g.output_width()).div_ceil(32)))
        .and_then(|n| n.checked_mul((g.output_channels() / g.groups()).div_ceil(64)))
        .ok_or("Metal convolution dispatch overflows")?;
    Ok(Shape {
        g,
        bytes,
        scratch_bytes,
        virtual_padding,
        jobs,
    })
}
#[repr(C)]
struct Params {
    jobs: u64,
    grid_groups: u64,
    spatial_tiles: u64,
    output_tiles: u64,
    groups: u64,
    input_channels: u64,
    input_height: u64,
    input_width: u64,
    output_channels: u64,
    output_height: u64,
    output_width: u64,
    kernel_height: u64,
    kernel_width: u64,
    stride_h: u64,
    stride_w: u64,
    dilation_h: u64,
    dilation_w: u64,
    pad_top: u64,
    pad_left: u64,
}
#[repr(C)]
struct PadParams {
    elements: u64,
    grid_threads: u64,
    channels: u64,
    height: u64,
    width: u64,
    padded_height: u64,
    padded_width: u64,
    top: u64,
    left: u64,
}
const _: () =
    assert!(std::mem::size_of::<Params>() == 152 && std::mem::size_of::<PadParams>() == 72);

pub(super) fn encode(
    participant: &TensorParticipant<'_>,
    pipelines: Arc<super::super::Pipelines>,
) -> Result<MetalDeviceCommand, String> {
    let shape = geometry(
        participant.bindings(),
        participant.attributes(),
        pipelines.spatial.conv_3x3_masked.is_some(),
    )?;
    let g = shape.g;
    let mut regions = regions(participant, &shape.bytes)?;
    if shape.scratch_bytes != 0 {
        append_scratch(participant, shape.scratch_bytes, &mut regions)?;
    }
    let [top, bottom, left, right] = g.padding();
    let padding = PadParams {
        elements: shape.scratch_bytes / 2,
        grid_threads: (shape.scratch_bytes / 2).min(1 << 20),
        channels: g.input_channels(),
        height: g.input_height(),
        width: g.input_width(),
        padded_height: g.input_height() + top + bottom,
        padded_width: g.input_width() + left + right,
        top,
        left,
    };
    let params = Params {
        jobs: shape.jobs,
        grid_groups: shape.jobs.min(65535),
        spatial_tiles: (g.output_height() * g.output_width()).div_ceil(32),
        output_tiles: (g.output_channels() / g.groups()).div_ceil(64),
        groups: g.groups(),
        input_channels: g.input_channels(),
        input_height: if shape.scratch_bytes == 0 {
            g.input_height()
        } else {
            padding.padded_height
        },
        input_width: if shape.scratch_bytes == 0 {
            g.input_width()
        } else {
            padding.padded_width
        },
        output_channels: g.output_channels(),
        output_height: g.output_height(),
        output_width: g.output_width(),
        kernel_height: g.kernel_height(),
        kernel_width: g.kernel_width(),
        stride_h: g.stride()[0],
        stride_w: g.stride()[1],
        dilation_h: g.dilation()[0],
        dilation_w: g.dilation()[1],
        pad_top: if shape.scratch_bytes == 0 { top } else { 0 },
        pad_left: if shape.scratch_bytes == 0 { left } else { 0 },
    };
    let work =
        DeviceCommandLogicalWork::tensor(g.element_count(), g.logical_arithmetic_operations())
            .map_err(|e| e.to_string())?;
    // Padding has already been materialized when scratch is present. The
    // specialization only omits bounds checks proven by the resolved geometry.
    let direct_square = matches!(params.kernel_height, 1 | 3)
        && params.kernel_height == params.kernel_width
        && params.stride_h == 1
        && params.stride_w == 1
        && params.dilation_h == 1
        && params.dilation_w == 1
        && params.pad_top == 0
        && params.pad_left == 0
        && params.output_height + params.kernel_height - 1 <= params.input_height
        && params.output_width + params.kernel_width - 1 <= params.input_width;
    let masked_3x3 = !direct_square
        && (g.padding_mode() == Conv2dPaddingMode::ImplicitZero || shape.virtual_padding)
        && params.kernel_height == 3
        && params.kernel_width == 3
        && params.stride_h == 1
        && params.stride_w == 1
        && params.dilation_h == 1
        && params.dilation_w == 1;
    MetalDeviceCommand::tensor_operation(
        "metal.conv2d.bf16",
        regions,
        work,
        move |encoder, regions| {
            let source = if shape.scratch_bytes != 0 {
                encoder.begin_compute_subwork("metal.conv2d.materialize_padding");
                let compute = encoder.compute_encoder();
                let p = pipelines
                    .spatial
                    .pad
                    .as_ref()
                    .expect("selected pad pipeline");
                compute.set_compute_pipeline_state(p);
                set_region(compute, 0, &regions[0]);
                set_region(compute, 1, &regions[4]);
                set_params(compute, 2, &padding);
                compute.dispatch_threads(
                    MTLSize::new(padding.grid_threads, 1, 1),
                    MTLSize::new(p.max_total_threads_per_threadgroup().min(256), 1, 1),
                );
                encoder.record_compute_dispatches(1);
                encoder.end_compute();
                4
            } else {
                0
            };
            encoder.begin_compute_subwork("metal.conv2d.convolution");
            let compute = encoder.compute_encoder();
            let (pipeline, implementation) = direct_square
                .then(|| match params.kernel_width {
                    1 => pipelines.spatial.conv_1x1.as_ref(),
                    3 => pipelines.spatial.conv_3x3.as_ref(),
                    _ => None,
                })
                .flatten()
                .map(|p| (p, "direct_square"))
                .or_else(|| {
                    masked_3x3
                        .then_some(pipelines.spatial.conv_3x3_masked.as_ref())
                        .flatten()
                        .map(|p| (p, "masked_3x3"))
                })
                .or_else(|| pipelines.spatial.conv.as_ref().map(|p| (p, "generic")))
                .expect("selected convolution pipeline");
            tracing::info!(
                implementation,
                padding_mode = g.padding_mode().as_str(),
                virtual_padding = shape.virtual_padding,
                scratch_bytes = shape.scratch_bytes,
                input_height = params.input_height,
                input_width = params.input_width,
                output_height = params.output_height,
                output_width = params.output_width,
                input_channels = params.input_channels,
                output_channels = params.output_channels,
                simd_width = pipeline.thread_execution_width(),
                max_threads = pipeline.max_total_threads_per_threadgroup(),
                static_threadgroup_bytes = pipeline.static_threadgroup_memory_length(),
                "Metal convolution actual dispatch"
            );
            compute.set_compute_pipeline_state(pipeline);
            set_region(compute, 0, &regions[source]);
            for port in 1..4 {
                set_region(compute, port, &regions[port as usize]);
            }
            set_params(compute, 4, &params);
            compute.dispatch_thread_groups(
                MTLSize::new(params.grid_groups, 1, 1),
                MTLSize::new(128, 1, 1),
            );
            encoder.record_compute_dispatches(1);
            Ok(())
        },
    )
    .map_err(|e| e.to_string())
}
