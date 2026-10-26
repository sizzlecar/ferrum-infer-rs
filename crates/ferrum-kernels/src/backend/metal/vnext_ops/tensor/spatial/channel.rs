use super::*;

pub(super) struct Shape {
    g: ChannelL2NormBf16Geometry,
    bytes: Vec<u64>,
    padded: u32,
    shared_bytes: u64,
}
pub(super) fn geometry(
    kind: MetalTensorOperation,
    values: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
    shared_limit: u64,
) -> Result<Shape, String> {
    let bytes = dense_ports(kind, values)?;
    let x = binding(values, ResolvedValueRole::Input, 0)?.tensor();
    let gamma = binding(values, ResolvedValueRole::Input, 1)?.tensor();
    let y = binding(values, ResolvedValueRole::Output, 0)?.tensor();
    let g = match kind {
        MetalTensorOperation::ChannelL2Norm => ChannelL2NormBf16Geometry::new(x, gamma, y, attrs),
        MetalTensorOperation::ChannelL2NormImage => {
            ChannelL2NormImageBf16Geometry::new(x, gamma, y, attrs).map(|g| g.arithmetic_geometry())
        }
        _ => return Err("Metal channel normalization kind differs".into()),
    }
    .map_err(|e| e.to_string())?;
    let padded = g
        .channels()
        .checked_next_power_of_two()
        .ok_or("Metal channel reduction overflows")?;
    let shared_bytes = padded
        .checked_mul(4)
        .map(|n| n.max(16))
        .filter(|&n| n <= shared_limit)
        .ok_or("Metal channel reduction exceeds on-chip memory")?;
    Ok(Shape {
        g,
        bytes,
        padded: u32::try_from(padded).map_err(|e| e.to_string())?,
        shared_bytes,
    })
}
#[repr(C)]
struct Params {
    spatial: u64,
    pixels: u64,
    grid_groups: u64,
    channels: u32,
    padded: u32,
    threads: u32,
    reserved: u32,
    epsilon: f32,
    scale: f32,
}
const _: () = assert!(std::mem::size_of::<Params>() == 48);
pub(super) fn encode(
    kind: MetalTensorOperation,
    participant: &TensorParticipant<'_>,
    pipelines: Arc<super::super::Pipelines>,
) -> Result<MetalDeviceCommand, String> {
    let shape = geometry(
        kind,
        participant.bindings(),
        participant.attributes(),
        pipelines.spatial.channel_shared_limit,
    )?;
    let g = shape.g;
    let regions = regions(participant, &shape.bytes)?;
    let threads = pipelines
        .spatial
        .channel
        .as_ref()
        .unwrap()
        .max_total_threads_per_threadgroup()
        .min(256)
        .min(u64::from(shape.padded));
    let pixels = g.batch() * g.height() * g.width();
    let params = Params {
        spatial: g.height() * g.width(),
        pixels,
        grid_groups: pixels.min(65535),
        channels: g.channels() as u32,
        padded: shape.padded,
        threads: threads as u32,
        reserved: 0,
        epsilon: g.effective_epsilon(),
        scale: g.effective_scale(),
    };
    let work =
        DeviceCommandLogicalWork::tensor(g.element_count(), g.logical_arithmetic_operations())
            .map_err(|e| e.to_string())?;
    MetalDeviceCommand::tensor_operation(
        "metal.channel_l2.bf16",
        regions,
        work,
        move |encoder, regions| {
            let compute = encoder.compute_encoder();
            compute.set_compute_pipeline_state(pipelines.spatial.channel.as_ref().unwrap());
            for (i, region) in regions.iter().enumerate() {
                set_region(compute, i as u64, region);
            }
            set_params(compute, 3, &params);
            compute.set_threadgroup_memory_length(0, shape.shared_bytes);
            compute.dispatch_thread_groups(
                MTLSize::new(params.grid_groups, 1, 1),
                MTLSize::new(threads, 1, 1),
            );
            Ok(())
        },
    )
    .map_err(|e| e.to_string())
}
