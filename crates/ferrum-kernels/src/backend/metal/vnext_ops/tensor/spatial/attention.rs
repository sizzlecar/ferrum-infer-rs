use super::*;
pub(super) struct Shape {
    g: SpatialAttentionBf16Geometry,
    bytes: Vec<u64>,
    pub scratch_bytes: u64,
}
pub(super) fn geometry(
    values: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<Shape, String> {
    let bytes = dense_ports(MetalTensorOperation::SpatialAttention, values)?;
    let g = SpatialAttentionBf16Geometry::new(
        binding(values, ResolvedValueRole::Input, 0)?.tensor(),
        binding(values, ResolvedValueRole::Output, 0)?.tensor(),
        attrs,
    )
    .map_err(|e| e.to_string())?;
    let scratch_bytes = g
        .element_count()
        .checked_mul(4)
        .filter(|&n| n <= isize::MAX as u64)
        .ok_or("Metal spatial attention accumulator exceeds address space")?;
    Ok(Shape {
        g,
        bytes,
        scratch_bytes,
    })
}
#[repr(C)]
struct Params {
    jobs: u64,
    grid_groups: u64,
    query_tiles: u64,
    spatial: u64,
    channels: u32,
    scale: f32,
}
const _: () = assert!(std::mem::size_of::<Params>() == 40);
pub(super) fn encode(
    participant: &TensorParticipant<'_>,
    pipelines: Arc<super::super::Pipelines>,
) -> Result<MetalDeviceCommand, String> {
    let shape = geometry(participant.bindings(), participant.attributes())?;
    let g = shape.g;
    let mut regions = regions(participant, &shape.bytes)?;
    append_scratch(participant, shape.scratch_bytes, &mut regions)?;
    let query_tiles = g.spatial().div_ceil(16);
    let jobs = g.batch() * query_tiles; // <= output element count, checked by geometry.
    let params = Params {
        jobs,
        grid_groups: jobs.min(65535),
        query_tiles,
        spatial: g.spatial(),
        channels: g.channels() as u32,
        scale: g.effective_scale(),
    };
    let work =
        DeviceCommandLogicalWork::tensor(g.element_count(), g.logical_arithmetic_operations())
            .map_err(|e| e.to_string())?;
    MetalDeviceCommand::tensor_operation(
        "metal.spatial_attention.bf16",
        regions,
        work,
        move |encoder, regions| {
            let compute = encoder.compute_encoder();
            compute.set_compute_pipeline_state(pipelines.spatial.attention.as_ref().unwrap());
            set_region(compute, 0, &regions[0]);
            set_region(compute, 1, &regions[2]);
            set_region(compute, 2, &regions[1]);
            set_params(compute, 3, &params);
            compute.dispatch_thread_groups(
                MTLSize::new(params.grid_groups, 1, 1),
                MTLSize::new(128, 1, 1),
            );
            Ok(())
        },
    )
    .map_err(|e| e.to_string())
}
