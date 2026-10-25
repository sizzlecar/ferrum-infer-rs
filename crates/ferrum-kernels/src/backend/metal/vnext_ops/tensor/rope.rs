//! Adjacent F32-frequency FMA and staged BF16 half-split rotary are separate.
use super::*;
use ferrum_runtime::{AdjacentRopeBf16Geometry, HalfSplitRopeBf16Geometry};

pub(super) struct Shape {
    columns: u64,
    sequence: u64,
    head_dim: u64,
    pairs: u64,
    bytes: u64,
    frequency_bytes: u64,
    operations: u64,
}
pub(super) fn geometry(
    kind: MetalTensorOperation,
    values: &[ResolvedValueBinding],
    attributes: &BTreeMap<AttributeId, SemanticValue>,
) -> Result<Shape, String> {
    if !kind.is_rope() || values.len() != kind.inputs() as usize + 1 || attributes.len() != 1 {
        return Err("Metal rotary port or attribute count differs".into());
    }
    let head_dim = unsigned_attribute(attributes, "head_dim")?;
    let input = binding(values, ResolvedValueRole::Input, 0)?.tensor();
    let frequencies = binding(values, ResolvedValueRole::Input, 1)?.tensor();
    let output = binding(values, ResolvedValueRole::Output, 0)?.tensor();
    let (rows, columns, sequence, frequency_element) = if kind == MetalTensorOperation::AdjacentRope
    {
        let g = AdjacentRopeBf16Geometry::new(input, frequencies, output, head_dim)
            .map_err(|e| e.to_string())?;
        (g.rows(), g.columns(), g.sequence(), 4_u64)
    } else {
        let sine = binding(values, ResolvedValueRole::Input, 2)?.tensor();
        let g = HalfSplitRopeBf16Geometry::new(input, frequencies, sine, output, head_dim)
            .map_err(|e| e.to_string())?;
        (g.rows(), g.columns(), g.rows(), 2)
    };
    let elements = rows
        .checked_mul(columns)
        .ok_or("Metal rotary elements overflow")?;
    let bytes = elements
        .checked_mul(2)
        .filter(|v| *v <= isize::MAX as u64)
        .ok_or("Metal rotary extent exceeds address space")?;
    let frequency_bytes = sequence
        .checked_mul(head_dim)
        .and_then(|v| v.checked_mul(frequency_element))
        .filter(|v| *v <= isize::MAX as u64)
        .ok_or("Metal rotary frequencies exceed address space")?;
    let pairs = elements / 2;
    let operations = pairs.checked_mul(7).ok_or("Metal rotary work overflows")?;
    Ok(Shape {
        columns,
        sequence,
        head_dim,
        pairs,
        bytes,
        frequency_bytes,
        operations,
    })
}

pub(super) fn encode(
    kind: MetalTensorOperation,
    participant: &TensorParticipant<'_>,
    pipelines: Arc<Pipelines>,
) -> Result<MetalDeviceCommand, String> {
    let shape = geometry(kind, participant.bindings(), participant.attributes())?;
    let adjacent = kind == MetalTensorOperation::AdjacentRope;
    let mut regions = Vec::new();
    for port in 0..kind.inputs() {
        let element = if adjacent && port == 1 {
            ElementType::F32
        } else {
            ElementType::Bf16
        };
        regions.push(tensor_port_region(
            participant,
            ResolvedValueRole::Input,
            port,
            element,
            if port == 0 {
                shape.bytes
            } else {
                shape.frequency_bytes
            },
        )?);
    }
    let output = tensor_port_region(
        participant,
        ResolvedValueRole::Output,
        0,
        ElementType::Bf16,
        shape.bytes,
    )?;
    for input in &regions {
        disjoint(input, &output)?;
    }
    regions.push(output);
    let work = DeviceCommandLogicalWork::tensor(shape.bytes / 2, shape.operations)
        .map_err(|e| e.to_string())?;
    let params = Params {
        pairs: shape.pairs,
        grid_width: shape.pairs.min(1 << 20),
        columns: shape.columns,
        sequence: shape.sequence,
        head_dim: shape.head_dim,
    };
    MetalDeviceCommand::tensor_operation(
        if adjacent {
            "metal.adjacent_rope.bf16"
        } else {
            "metal.half_split_rope.bf16"
        },
        regions,
        work,
        move |encoder, regions| {
            let pipeline = if adjacent {
                &pipelines.adjacent_rope
            } else {
                &pipelines.half_split_rope
            }
            .as_ref()
            .expect("selected rotary pipeline");
            let compute = encoder.compute_encoder();
            compute.set_compute_pipeline_state(pipeline);
            for (index, region) in regions.iter().enumerate() {
                set_region(compute, index as u64, region);
            }
            set_params(compute, regions.len() as u64, &params);
            compute.dispatch_threads(
                MTLSize::new(params.grid_width, 1, 1),
                MTLSize::new(pipeline.max_total_threads_per_threadgroup().min(256), 1, 1),
            );
            Ok(())
        },
    )
    .map_err(|e| e.to_string())
}

#[repr(C)]
struct Params {
    pairs: u64,
    grid_width: u64,
    columns: u64,
    sequence: u64,
    head_dim: u64,
}
