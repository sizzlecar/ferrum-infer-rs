use super::*;
pub(super) fn geometry(
    values: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<SpatialGroupMeanBf16Geometry, String> {
    dense_ports(MetalTensorOperation::SpatialGroupMean, values)?;
    SpatialGroupMeanBf16Geometry::new(
        binding(values, ResolvedValueRole::Input, 0)?.tensor(),
        binding(values, ResolvedValueRole::Output, 0)?.tensor(),
        attrs,
    )
    .map_err(|e| e.to_string())
}
#[repr(C)]
struct Params {
    elements: u64,
    grid_threads: u64,
    input_channels: u64,
    input_height: u64,
    input_width: u64,
    output_channels: u64,
    output_height: u64,
    output_width: u64,
    scale_h: u64,
    scale_w: u64,
    frames: u64,
    group_size: u64,
}
const _: () = assert!(std::mem::size_of::<Params>() == 96);
pub(super) fn encode(
    participant: &TensorParticipant<'_>,
    pipelines: Arc<super::super::Pipelines>,
) -> Result<MetalDeviceCommand, String> {
    let g = geometry(participant.bindings(), participant.attributes())?;
    let regions = regions(
        participant,
        &dense_ports(
            MetalTensorOperation::SpatialGroupMean,
            participant.bindings(),
        )?,
    )?;
    let params = Params {
        elements: g.element_count(),
        grid_threads: g.element_count().min(1 << 20),
        input_channels: g.input_channels(),
        input_height: g.input_height(),
        input_width: g.input_width(),
        output_channels: g.output_channels(),
        output_height: g.output_height(),
        output_width: g.output_width(),
        scale_h: g.scale_h(),
        scale_w: g.scale_w(),
        frames: g.frame_factor(),
        group_size: g.group_size(),
    };
    let work =
        DeviceCommandLogicalWork::tensor(g.element_count(), g.logical_arithmetic_operations())
            .map_err(|e| e.to_string())?;
    MetalDeviceCommand::tensor_operation(
        "metal.spatial_group_mean.bf16",
        regions,
        work,
        move |encoder, regions| {
            let compute = encoder.compute_encoder();
            let pipeline = pipelines.spatial.mean.as_ref().unwrap();
            compute.set_compute_pipeline_state(pipeline);
            set_region(compute, 0, &regions[0]);
            set_region(compute, 1, &regions[1]);
            set_params(compute, 2, &params);
            compute.dispatch_threads(
                MTLSize::new(params.grid_threads, 1, 1),
                MTLSize::new(pipeline.max_total_threads_per_threadgroup().min(256), 1, 1),
            );
            Ok(())
        },
    )
    .map_err(|e| e.to_string())
}
