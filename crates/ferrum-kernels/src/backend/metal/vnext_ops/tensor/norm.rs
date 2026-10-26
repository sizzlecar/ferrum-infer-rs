//! Two-pass LayerNorm and the distinct head/zero-centered RMS contracts.
use super::*;
use ferrum_runtime::{
    HeadRmsNormBf16Geometry, LayerNormBf16Geometry, PhysicalWeightLayout,
    ZeroCenteredRmsNormBf16Geometry,
};

pub(super) struct Shape {
    groups: u64,
    width: u32,
    padded: u32,
    shared_bytes: u64,
    bytes: u64,
    operations: u64,
    epsilon: f32,
    kind: u32,
}

pub(super) fn geometry(
    kind: MetalTensorOperation,
    values: &[ResolvedValueBinding],
    attributes: &BTreeMap<AttributeId, SemanticValue>,
    shared_limit: u64,
) -> Result<Shape, String> {
    use MetalTensorOperation::*;
    let expected_attributes = if kind == ZeroCenteredRmsNorm { 1 } else { 2 };
    if !kind.is_norm()
        || values.len() != kind.inputs() as usize + 1
        || attributes.len() != expected_attributes
    {
        return Err("Metal normalization port or attribute count differs".into());
    }
    let epsilon = match attributes.iter().find(|(key, _)| key.as_str() == "epsilon") {
        Some((_, SemanticValue::Rational(value))) => *value,
        _ => return Err("Metal normalization requires rational epsilon".into()),
    };
    let input = binding(values, ResolvedValueRole::Input, 0)?.tensor();
    let output = binding(values, ResolvedValueRole::Output, 0)?.tensor();
    let gamma = if kind != LayerNorm {
        let gamma = binding(values, ResolvedValueRole::Input, 1)?;
        if let Some(weight) = gamma.weight() {
            weight
                .validate_logical(gamma.tensor().dimensions(), ElementType::Bf16)
                .map_err(|e| e.to_string())?;
            if !matches!(weight.physical_layout(), PhysicalWeightLayout::Dense { .. }) {
                return Err("Metal normalization requires dense BF16 scale storage".into());
            }
        }
        Some(gamma.tensor())
    } else {
        None
    };
    let (rows, columns, width, epsilon, index, per_width) = match kind {
        AffineLayerNorm => {
            let g = primitives::geometry(kind, values, attributes)?;
            (g.rows, g.columns, g.columns, g.scalar, 3, 8_u64)
        }
        LayerNorm => {
            let g = LayerNormBf16Geometry::new(
                input,
                output,
                unsigned_attribute(attributes, "hidden_size")?,
                epsilon,
            )
            .map_err(|e| e.to_string())?;
            (
                g.rows(),
                g.hidden_size(),
                g.hidden_size(),
                g.effective_epsilon(),
                0,
                6_u64,
            )
        }
        HeadRmsNorm => {
            let g = HeadRmsNormBf16Geometry::new(
                input,
                gamma.unwrap(),
                output,
                unsigned_attribute(attributes, "head_dim")?,
                epsilon,
            )
            .map_err(|e| e.to_string())?;
            (
                g.rows(),
                g.columns(),
                g.head_dim(),
                g.effective_epsilon(),
                1,
                4,
            )
        }
        ZeroCenteredRmsNorm => {
            let g = ZeroCenteredRmsNormBf16Geometry::new(input, gamma.unwrap(), output, epsilon)
                .map_err(|e| e.to_string())?;
            (
                g.rows(),
                g.columns(),
                g.columns(),
                g.effective_epsilon(),
                2,
                5,
            )
        }
        _ => unreachable!("normalization operation checked above"),
    };
    let padded = width
        .checked_next_power_of_two()
        .ok_or("Metal normalization width overflows")?;
    // Metal dynamic threadgroup lengths must be multiples of 16 bytes.
    let shared = padded
        .checked_mul(4)
        .ok_or("Metal normalization shared bytes overflow")?
        .max(16);
    if shared > shared_limit {
        return Err("Metal normalization width exceeds device threadgroup storage".into());
    }
    let elements = rows
        .checked_mul(columns)
        .ok_or("Metal normalization elements overflow")?;
    let bytes = elements
        .checked_mul(2)
        .filter(|v| *v <= isize::MAX as u64)
        .ok_or("Metal normalization byte extent exceeds address space")?;
    let groups = elements / width;
    let operations = width
        .checked_mul(per_width)
        .and_then(|v| v.checked_add(3))
        .and_then(|v| v.checked_mul(groups))
        .ok_or("Metal normalization work overflows")?;
    Ok(Shape {
        groups,
        width: u32::try_from(width).map_err(|e| e.to_string())?,
        padded: u32::try_from(padded).map_err(|e| e.to_string())?,
        shared_bytes: shared,
        bytes,
        operations,
        epsilon,
        kind: index,
    })
}

pub(super) fn encode(
    kind: MetalTensorOperation,
    participant: &TensorParticipant<'_>,
    pipelines: Arc<Pipelines>,
) -> Result<MetalDeviceCommand, String> {
    let shape = geometry(
        kind,
        participant.bindings(),
        participant.attributes(),
        pipelines.norm_shared_limit,
    )?;
    let mut regions = Vec::new();
    for port in 0..kind.inputs() {
        regions.push(tensor_port_region(
            participant,
            ResolvedValueRole::Input,
            port,
            ElementType::Bf16,
            if port == 0 {
                shape.bytes
            } else {
                u64::from(shape.width) * 2
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
    let output_index = kind.inputs() as usize;
    let threads = u64::from(shape.padded).min(256).min(
        pipelines
            .norm
            .as_ref()
            .expect("selected normalization pipeline")
            .max_total_threads_per_threadgroup(),
    );
    if threads == 0 {
        return Err("Metal normalization has no available threads".into());
    }
    let params = Params {
        groups: shape.groups,
        grid_groups: shape.groups.min(65535),
        width: shape.width,
        padded: shape.padded,
        kind: shape.kind,
        threads: threads as u32,
        epsilon: shape.epsilon,
        reserved: 0,
    };
    MetalDeviceCommand::tensor_operation(
        "metal.normalization.bf16",
        regions,
        work,
        move |encoder, regions| {
            let compute = encoder.compute_encoder();
            compute.set_compute_pipeline_state(
                pipelines
                    .norm
                    .as_ref()
                    .expect("selected normalization pipeline"),
            );
            set_region(compute, 0, &regions[0]);
            // LayerNorm never reads this placeholder. No implicit scale allocation.
            set_region(compute, 1, &regions[if output_index == 1 { 0 } else { 1 }]);
            set_region(compute, 2, &regions[output_index]);
            set_params(compute, 3, &params);
            set_region(
                compute,
                4,
                &regions[if kind == MetalTensorOperation::AffineLayerNorm {
                    2
                } else {
                    0
                }],
            );
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

#[repr(C)]
struct Params {
    groups: u64,
    grid_groups: u64,
    width: u32,
    padded: u32,
    kind: u32,
    threads: u32,
    epsilon: f32,
    reserved: u32,
}
