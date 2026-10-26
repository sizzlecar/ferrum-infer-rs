//! Ordered, nonoverlapping row updates. One writer per element, no atomics or scratch.
use super::*;
use ferrum_runtime::{PhysicalWeightLayout, TensorPrimitiveBf16, TensorPrimitiveBf16Geometry};

#[repr(C)]
#[derive(Default)]
struct Params {
    elements: u64,
    grid_threads: u64,
    columns: u64,
    starts: [u64; 32],
    ends: [u64; 32],
    update_starts: [u64; 32],
    run_count: u32,
    reserved: u32,
}
// Scalar fields only, initialized including the explicit alignment field.
const _: () = assert!(std::mem::size_of::<Params>() == 800);

pub(super) fn geometry(
    values: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<TensorPrimitiveBf16Geometry, String> {
    let g = TensorPrimitiveBf16Geometry::new(TensorPrimitiveBf16::RowScatterAdd, values, attrs)
        .map_err(|e| e.to_string())?;
    if g.input_bytes
        .iter()
        .chain([&g.output_bytes])
        .any(|&n| n > isize::MAX as u64)
    {
        return Err("Metal row scatter byte extent exceeds address space".into());
    }
    for port in 0..2 {
        let input = binding(values, ResolvedValueRole::Input, port)?;
        if let Some(weight) = input.weight() {
            weight
                .validate_logical(input.tensor().dimensions(), ElementType::Bf16)
                .map_err(|e| e.to_string())?;
            if !matches!(weight.physical_layout(), PhysicalWeightLayout::Dense { .. }) {
                return Err("Metal row scatter requires dense BF16 weight storage".into());
            }
        }
    }
    Ok(g)
}

pub(super) fn encode(
    participant: &TensorParticipant<'_>,
    pipelines: Arc<Pipelines>,
) -> Result<MetalDeviceCommand, String> {
    let g = geometry(participant.bindings(), participant.attributes())?;
    let output = tensor_port_region(
        participant,
        ResolvedValueRole::Output,
        0,
        ElementType::Bf16,
        g.output_bytes,
    )?;
    let mut regions = Vec::with_capacity(3);
    for port in 0..2 {
        let input = tensor_port_region(
            participant,
            ResolvedValueRole::Input,
            port,
            ElementType::Bf16,
            g.input_bytes[port as usize],
        )?;
        disjoint(&input, &output)?;
        regions.push(input);
    }
    regions.push(output);
    let elements = g.output_bytes / 2;
    let mut params = Params {
        elements,
        grid_threads: elements.min(1 << 20),
        columns: g.columns,
        run_count: g.starts.len() as u32,
        ..Params::default()
    };
    let mut updates = 0;
    for (i, (&start, &length)) in g.starts.iter().zip(&g.lengths).enumerate() {
        params.starts[i] = start;
        params.ends[i] = start + length;
        params.update_starts[i] = updates;
        updates += length; // Extents and update sum already checked by core geometry.
    }
    let work = DeviceCommandLogicalWork::tensor(elements, g.input_bytes[1] / 2)
        .map_err(|e| e.to_string())?;
    MetalDeviceCommand::tensor_operation(
        "metal.row_scatter_add.bf16",
        regions,
        work,
        move |encoder, regions| {
            let compute = encoder.compute_encoder();
            let pipeline = pipelines
                .scatter
                .as_ref()
                .expect("selected scatter pipeline");
            compute.set_compute_pipeline_state(pipeline);
            for (index, region) in regions.iter().enumerate() {
                set_region(compute, index as u64, region);
            }
            set_params(compute, 3, &params);
            compute.dispatch_threads(
                MTLSize::new(params.grid_threads, 1, 1),
                MTLSize::new(pipeline.max_total_threads_per_threadgroup().min(256), 1, 1),
            );
            Ok(())
        },
    )
    .map_err(|e| e.to_string())
}
