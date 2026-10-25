//! Exact row transport over core-validated windows/runs, without scratch.
use super::*;
use ferrum_runtime::{BatchedRowConcatBf16Geometry, PhysicalWeightLayout, RowGatherBf16Geometry};

#[repr(C)]
#[derive(Default)]
struct Params {
    elements: u64,
    grid_threads: u64,
    rows_per_batch: u64,
    source_width: u64,
    column_start: u64,
    columns: u64,
    left_rows: u64,
    right_rows: u64,
    run_ends: [u64; 32],
    source_starts: [u64; 32],
    batch_strides: [u64; 32],
    row_strides: [u64; 32],
    run_count: u32,
    kind: u32,
}
// Initialized integer fields, no implicit padding; one bounded set_bytes packet.
const _: () = assert!(std::mem::size_of::<Params>() == 1096);
pub(super) struct Shape {
    params: Params,
    input_bytes: [u64; 2],
    output_bytes: u64,
}

pub(super) fn geometry(
    kind: MetalTensorOperation,
    values: &[ResolvedValueBinding],
    attrs: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<Shape, String> {
    if !kind.is_transport() || values.len() != kind.inputs() as usize + 1 {
        return Err("Metal transport port count differs".into());
    }
    kind.contract()
        .map_err(|e| e.to_string())?
        .descriptor()
        .validate_attributes(attrs)
        .map_err(|e| e.to_string())?;
    let mut input_bytes = [0; 2];
    for port in 0..kind.inputs() {
        let input = binding(values, ResolvedValueRole::Input, port)?;
        if let Some(weight) = input.weight() {
            weight
                .validate_logical(input.tensor().dimensions(), ElementType::Bf16)
                .map_err(|e| e.to_string())?;
            if !matches!(weight.physical_layout(), PhysicalWeightLayout::Dense { .. }) {
                return Err("Metal transport requires dense BF16 weight storage".into());
            }
        }
        input_bytes[port as usize] = input
            .tensor()
            .minimum_storage_bytes()
            .map_err(|e| e.to_string())?;
    }
    let x = binding(values, ResolvedValueRole::Input, 0)?.tensor();
    let y = binding(values, ResolvedValueRole::Output, 0)?.tensor();
    let output_bytes = y.minimum_storage_bytes().map_err(|e| e.to_string())?;
    if input_bytes
        .iter()
        .chain([&output_bytes])
        .any(|&n| n > isize::MAX as u64)
    {
        return Err("Metal transport byte extent exceeds address space".into());
    }
    let mut params = Params::default();
    match kind {
        MetalTensorOperation::RowGather => {
            let g = RowGatherBf16Geometry::new(x, y, attrs).map_err(|e| e.to_string())?;
            params.elements = g.element_count();
            params.rows_per_batch = g.rows_per_batch();
            params.source_width = g.source_width();
            params.column_start = g.column_start();
            params.columns = g.column_count();
            params.run_count = g.run_lengths().len() as u32;
            let mut end = 0;
            for (i, &length) in g.run_lengths().iter().enumerate() {
                end += length; // Sum already checked by core geometry.
                params.run_ends[i] = end;
                params.source_starts[i] = g.source_starts()[i];
                params.batch_strides[i] = g.source_batch_strides()[i];
                params.row_strides[i] = g.source_row_strides()[i];
            }
        }
        MetalTensorOperation::BatchedRowConcat => {
            let right = binding(values, ResolvedValueRole::Input, 1)?.tensor();
            let g =
                BatchedRowConcatBf16Geometry::new(x, right, y, unsigned_attribute(attrs, "batch")?)
                    .map_err(|e| e.to_string())?;
            params.elements = g.element_count();
            params.left_rows = g.left_rows_per_batch();
            params.right_rows = g.right_rows_per_batch();
            params.rows_per_batch = params.left_rows + params.right_rows;
            params.columns = g.columns();
            params.kind = 1;
        }
        _ => unreachable!("transport kind checked"),
    }
    params.grid_threads = params.elements.min(1 << 20);
    Ok(Shape {
        params,
        input_bytes,
        output_bytes,
    })
}

pub(super) fn encode(
    kind: MetalTensorOperation,
    participant: &TensorParticipant<'_>,
    pipelines: Arc<Pipelines>,
) -> Result<MetalDeviceCommand, String> {
    let shape = geometry(kind, participant.bindings(), participant.attributes())?;
    let mut regions = Vec::with_capacity(kind.inputs() as usize + 1);
    for port in 0..kind.inputs() {
        regions.push(tensor_port_region(
            participant,
            ResolvedValueRole::Input,
            port,
            ElementType::Bf16,
            shape.input_bytes[port as usize],
        )?);
    }
    let output = tensor_port_region(
        participant,
        ResolvedValueRole::Output,
        0,
        ElementType::Bf16,
        shape.output_bytes,
    )?;
    for input in &regions {
        disjoint(input, &output)?;
    }
    regions.push(output);
    let work =
        DeviceCommandLogicalWork::tensor(shape.params.elements, 0).map_err(|e| e.to_string())?;
    MetalDeviceCommand::tensor_operation(
        "metal.row_transport.bf16",
        regions,
        work,
        move |encoder, regions| {
            let compute = encoder.compute_encoder();
            let pipeline = pipelines
                .transport
                .as_ref()
                .expect("selected transport pipeline");
            compute.set_compute_pipeline_state(pipeline);
            set_region(compute, 0, &regions[0]);
            set_region(compute, 1, &regions[if kind.inputs() == 2 { 1 } else { 0 }]);
            set_region(compute, 2, &regions[kind.inputs() as usize]);
            set_params(compute, 3, &shape.params);
            compute.dispatch_threads(
                MTLSize::new(shape.params.grid_threads, 1, 1),
                MTLSize::new(pipeline.max_total_threads_per_threadgroup().min(256), 1, 1),
            );
            Ok(())
        },
    )
    .map_err(|e| e.to_string())
}
