//! Tiled online attention: bounded on-chip state, no score matrix or KV copies.
use super::*;
use ferrum_runtime::{
    StructuredAttentionBf16Geometry, StructuredAttentionBf16Mask,
    StructuredGqaAttentionBf16Geometry,
};

// All fields initialized, no padding; each set_bytes packet is below 4 KiB.
#[repr(C)]
struct NativeMask {
    segment_count: u32,
    segment_ends: [u32; 32],
    bidirectional_mask: u32,
    key_run_counts: [u32; 16],
    key_run_ends: [u32; 512],
    key_valid_masks: [u32; 16],
    first_valid_keys: [u32; 16],
}
const _: () = assert!(std::mem::size_of::<NativeMask>() == 2376);

impl NativeMask {
    fn new(mask: &StructuredAttentionBf16Mask, sequence: u32) -> Self {
        let mut result = Self {
            segment_count: mask.segment_ends().len() as u32,
            segment_ends: [0; 32],
            bidirectional_mask: 0,
            key_run_counts: [0; 16],
            key_run_ends: [0; 512],
            key_valid_masks: [0; 16],
            first_valid_keys: [sequence; 16],
        };
        for (i, (&end, &bidirectional)) in mask
            .segment_ends()
            .iter()
            .zip(mask.segment_bidirectional())
            .enumerate()
        {
            result.segment_ends[i] = end;
            result.bidirectional_mask |= u32::from(bidirectional) << i;
        }
        for batch in 0..mask.batch() {
            let ends = mask.key_run_ends(batch).expect("checked mask batch");
            let valid = mask.key_run_valid(batch).expect("checked mask batch");
            result.key_run_counts[batch] = ends.len() as u32;
            let mut start = 0;
            for (i, (&end, &valid)) in ends.iter().zip(valid).enumerate() {
                result.key_run_ends[batch * 32 + i] = end;
                result.key_valid_masks[batch] |= u32::from(valid) << i;
                if valid {
                    result.first_valid_keys[batch] = result.first_valid_keys[batch].min(start);
                }
                start = end;
            }
        }
        result
    }
}

#[repr(C)]
struct Params {
    jobs: u64,
    grid_groups: u64,
    query_tiles: u64,
    sequence: u64,
    query_heads: u64,
    key_value_heads: u64,
    head_dim: u32,
    scale: f32,
}
const _: () = assert!(std::mem::size_of::<Params>() == 56);

pub(super) struct Shape {
    params: Params,
    mask: NativeMask,
    query_bytes: u64,
    key_value_bytes: u64,
    shared_bytes: u64,
    operations: u64,
}

pub(super) fn geometry(
    kind: MetalTensorOperation,
    values: &[ResolvedValueBinding],
    attributes: &BTreeMap<AttributeId, SemanticValue>,
    shared_limit: u64,
) -> Result<Shape, String> {
    if !kind.is_attention() || values.len() != 4 {
        return Err("Metal attention requires Q/K/V and one output".into());
    }
    let q = binding(values, ResolvedValueRole::Input, 0)?.tensor();
    let k = binding(values, ResolvedValueRole::Input, 1)?.tensor();
    let v = binding(values, ResolvedValueRole::Input, 2)?.tensor();
    let y = binding(values, ResolvedValueRole::Output, 0)?.tensor();
    // Validate MHA's stricter equal-width contract before unifying the geometry.
    if kind == MetalTensorOperation::StructuredAttention {
        StructuredAttentionBf16Geometry::new(q, k, v, y, attributes).map_err(|e| e.to_string())?;
    }
    let checked = StructuredGqaAttentionBf16Geometry::new(q, k, v, y, attributes)
        .map_err(|e| e.to_string())?;
    let query_bytes = q.minimum_storage_bytes().map_err(|e| e.to_string())?;
    let key_value_bytes = k.minimum_storage_bytes().map_err(|e| e.to_string())?;
    if query_bytes > isize::MAX as u64 || key_value_bytes > isize::MAX as u64 {
        return Err("Metal attention byte extent exceeds address space".into());
    }
    let shared_bytes = (2048 + 16 * checked.head_dim() + 80) * 4;
    if shared_bytes > shared_limit {
        return Err("Metal attention exceeds device threadgroup storage".into());
    }
    let query_tiles = checked.sequence_length().div_ceil(16);
    let jobs = checked
        .batch()
        .checked_mul(checked.query_heads())
        .and_then(|v| v.checked_mul(query_tiles))
        .ok_or("Metal attention grid overflows")?;
    Ok(Shape {
        params: Params {
            jobs,
            grid_groups: jobs.min(65535),
            query_tiles,
            sequence: checked.sequence_length(),
            query_heads: checked.query_heads(),
            key_value_heads: checked.key_value_heads(),
            head_dim: checked.head_dim() as u32,
            scale: checked.effective_scale(),
        },
        mask: NativeMask::new(checked.mask(), checked.sequence_length() as u32),
        query_bytes,
        key_value_bytes,
        shared_bytes,
        operations: checked.logical_arithmetic_operations(),
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
        pipelines.attention_shared_limit,
    )?;
    let mut regions = Vec::with_capacity(4);
    for port in 0..3 {
        regions.push(tensor_port_region(
            participant,
            ResolvedValueRole::Input,
            port,
            ElementType::Bf16,
            if port == 0 {
                shape.query_bytes
            } else {
                shape.key_value_bytes
            },
        )?);
    }
    let output = tensor_port_region(
        participant,
        ResolvedValueRole::Output,
        0,
        ElementType::Bf16,
        shape.query_bytes,
    )?;
    for input in &regions {
        disjoint(input, &output)?;
    }
    regions.push(output);
    let work = DeviceCommandLogicalWork::tensor(shape.query_bytes / 2, shape.operations)
        .map_err(|e| e.to_string())?;
    MetalDeviceCommand::tensor_operation(
        "metal.structured_attention.bf16",
        regions,
        work,
        move |encoder, regions| {
            let compute = encoder.compute_encoder();
            compute.set_compute_pipeline_state(
                pipelines
                    .attention
                    .as_ref()
                    .expect("selected attention pipeline"),
            );
            for (i, region) in regions.iter().enumerate() {
                set_region(compute, i as u64, region);
            }
            set_params(compute, 4, &shape.params);
            set_params(compute, 5, &shape.mask);
            compute.set_threadgroup_memory_length(0, shape.shared_bytes);
            compute.dispatch_thread_groups(
                MTLSize::new(shape.params.grid_groups, 1, 1),
                MTLSize::new(128, 1, 1),
            );
            Ok(())
        },
    )
    .map_err(|e| e.to_string())
}
