//! Tiled online attention: bounded on-chip state, no score matrix or KV copies.
use super::*;
use ferrum_runtime::{
    StructuredAttentionBf16Geometry, StructuredAttentionBf16Mask,
    StructuredGqaAttentionBf16Geometry,
};

pub(super) mod cross;
pub(super) mod dense_guarded;
pub(super) mod mpp;
pub(super) mod register128;

const CACHED_SHARED_BYTES: u64 = (2048 + 96) * 4;

pub(super) fn head128_pipeline(
    runtime: &MetalDeviceRuntime,
    library: &metal::LibraryRef,
) -> Result<Option<ComputePipelineState>, MetalDeviceRuntimeError> {
    let constants = metal::FunctionConstantValues::new();
    constants.set_constant_value_at_index(
        (&128_u32 as *const u32).cast(),
        metal::MTLDataType::UInt,
        101,
    );
    let function = library
        .get_function("ferrum_tensor_attention_bf16", Some(constants))
        .map_err(MetalDeviceRuntimeError::contract)?;
    let pipeline = runtime
        .device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(MetalDeviceRuntimeError::contract)?;
    let dynamic_bytes = CACHED_SHARED_BYTES;
    Ok((pipeline.thread_execution_width() == 32
        && pipeline.max_total_threads_per_threadgroup() >= 128
        && pipeline
            .static_threadgroup_memory_length()
            .checked_add(dynamic_bytes)
            .is_some_and(|n| n <= runtime.device().max_threadgroup_memory_length()))
    .then_some(pipeline))
}

// All fields initialized, no padding; each set_bytes packet is below 4 KiB.
#[repr(C)]
struct NativeMask {
    isolated_segments: u32,
    segment_count: u32,
    segment_ends: [u32; 32],
    bidirectional_mask: u32,
    key_run_counts: [u32; 16],
    key_run_ends: [u32; 512],
    key_valid_masks: [u32; 16],
    first_valid_keys: [u32; 16],
}
const _: () = assert!(std::mem::size_of::<NativeMask>() == 2380);

impl NativeMask {
    fn new(mask: &StructuredAttentionBf16Mask, sequence: u32) -> Self {
        let mut result = Self {
            isolated_segments: 0,
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
    key_visibility: register128::KeyVisibility,
    query_bytes: u64,
    key_value_bytes: u64,
    shared_bytes: u64,
    operations: u64,
}

impl Shape {
    pub(in crate::backend::metal::vnext_ops::tensor) fn dense_schedule(
        &self,
    ) -> Result<Option<dense_guarded::Schedule>, String> {
        dense_guarded::Schedule::new(dense_guarded::GuardGeometry {
            batch_heads: self.params.jobs / self.params.query_tiles,
            query: self.params.sequence,
            key: self.params.sequence,
            query_heads: self.params.query_heads,
            key_value_heads: self.params.key_value_heads,
            head_dim: self.params.head_dim,
            scale: self.params.scale,
            all_keys_valid: self.key_visibility == register128::KeyVisibility::AllValid,
            dense_start: dense_guarded::dense_start(&self.mask, self.params.sequence),
        })
    }
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
    if kind == MetalTensorOperation::SegmentedAttention {
        let g = primitives::geometry(kind, values, attributes)?;
        let sequence = u32::try_from(g.rows).map_err(|_| "Metal attention sequence exceeds u32")?;
        let shared_bytes = (2048 + 16 * g.head_dim + 96) * 4;
        if shared_bytes > shared_limit {
            return Err("Metal attention exceeds device threadgroup storage".into());
        }
        let heads = g.columns / g.head_dim;
        let query_tiles = g.rows.div_ceil(16);
        let jobs = heads
            .checked_mul(query_tiles)
            .ok_or("Metal attention grid overflows")?;
        let pairs = g
            .lengths
            .iter()
            .try_fold(0_u64, |sum, &length| {
                length.checked_mul(length).and_then(|n| sum.checked_add(n))
            })
            .ok_or("Metal segmented attention pair count overflows")?;
        let operations = pairs
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(4 * g.head_dim + 5))
            .ok_or("Metal segmented attention work overflows")?;
        let mut mask = NativeMask {
            isolated_segments: 1,
            segment_count: g.lengths.len() as u32,
            segment_ends: [0; 32],
            bidirectional_mask: u32::MAX,
            key_run_counts: [0; 16],
            key_run_ends: [0; 512],
            key_valid_masks: [0; 16],
            first_valid_keys: [sequence; 16],
        };
        for (i, (&start, &length)) in g.starts.iter().zip(&g.lengths).enumerate() {
            mask.segment_ends[i] = (start + length) as u32;
        }
        mask.key_run_counts[0] = 1;
        mask.key_run_ends[0] = sequence;
        mask.key_valid_masks[0] = 1;
        mask.first_valid_keys[0] = 0;
        return Ok(Shape {
            params: Params {
                jobs,
                grid_groups: jobs.min(65535),
                query_tiles,
                sequence: g.rows,
                query_heads: heads,
                key_value_heads: heads,
                head_dim: g.head_dim as u32,
                scale: g.scalar,
            },
            mask,
            key_visibility: register128::KeyVisibility::AllValid,
            query_bytes: g.output_bytes,
            key_value_bytes: g.output_bytes,
            shared_bytes,
            operations,
        });
    }
    // Validate MHA's stricter equal-width contract before unifying the geometry.
    if matches!(
        kind,
        MetalTensorOperation::StructuredAttention | MetalTensorOperation::StructuredAttentionFused
    ) {
        StructuredAttentionBf16Geometry::new(q, k, v, y, attributes).map_err(|e| e.to_string())?;
    }
    let checked = StructuredGqaAttentionBf16Geometry::new(q, k, v, y, attributes)
        .map_err(|e| e.to_string())?;
    let query_bytes = q.minimum_storage_bytes().map_err(|e| e.to_string())?;
    let key_value_bytes = k.minimum_storage_bytes().map_err(|e| e.to_string())?;
    if query_bytes > isize::MAX as u64 || key_value_bytes > isize::MAX as u64 {
        return Err("Metal attention byte extent exceeds address space".into());
    }
    let shared_bytes = (2048 + 16 * checked.head_dim() + 96) * 4;
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
        key_visibility: register128::KeyVisibility::from_mask(
            checked.mask(),
            checked.sequence_length() as u32,
        ),
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
    if kind == MetalTensorOperation::StructuredAttentionFused && pipelines.dense_attention.is_some()
    {
        if let Some(plan) = shape.dense_schedule()?.filter(|plan| {
            pipelines
                .dense_attention
                .as_ref()
                .is_some_and(|dense| dense.supports(plan))
        }) {
            dense_guarded::append_flags(participant, &plan, &mut regions)?;
            let work = DeviceCommandLogicalWork::tensor(shape.query_bytes / 2, shape.operations)
                .map_err(|e| e.to_string())?;
            return MetalDeviceCommand::tensor_operation(
                "metal.structured_attention.fused_dense_guarded",
                regions,
                work,
                move |encoder, regions| {
                    dense_guarded::encode(
                        encoder,
                        regions,
                        &plan,
                        &shape.mask,
                        pipelines
                            .dense_attention
                            .as_ref()
                            .expect("selected dense pipelines"),
                        false,
                    );
                    Ok(())
                },
            )
            .map_err(|e| e.to_string());
        }
    }
    let use_all_valid_register = shape.key_visibility == register128::KeyVisibility::AllValid
        && pipelines.attention_register128_all_valid.is_some()
        && register128::supports_shape(shape.params.sequence, shape.params.head_dim);
    let use_register = use_all_valid_register
        || (pipelines.attention_register128.is_some()
            && register128::supports_shape(shape.params.sequence, shape.params.head_dim));
    let use_mpp = !use_register
        && pipelines.attention_mpp_head128.is_some()
        && mpp::supports_shape(
            shape.params.sequence,
            shape.params.query_heads,
            shape.params.key_value_heads,
            shape.params.head_dim,
        );
    let register_params = if use_register {
        Some(register128::RegisterParams::new(
            shape.params.jobs / shape.params.query_tiles,
            shape.params.sequence,
            shape.params.sequence,
            shape.params.query_heads,
            shape.params.key_value_heads,
            shape.params.scale,
        )?)
    } else {
        None
    };
    let kernel = if use_all_valid_register {
        "metal.structured_attention.bf16.register32x16.all_valid_keys"
    } else if use_register {
        "metal.structured_attention.bf16.register32x16"
    } else if use_mpp {
        "metal.structured_attention.bf16.mpp_qk"
    } else {
        "metal.structured_attention.bf16"
    };
    let shared_bytes = if use_register {
        register128::SHARED_BYTES
    } else if use_mpp || (shape.params.head_dim == 128 && pipelines.attention_head128.is_some()) {
        CACHED_SHARED_BYTES
    } else {
        shape.shared_bytes
    };
    tracing::debug!(
        kernel,
        query_tile_rows = if use_register {
            register128::QUERY_ROWS
        } else {
            16
        },
        shared_bytes,
        sequence = shape.params.sequence,
        query_heads = shape.params.query_heads,
        "Metal attention dispatch selected"
    );
    let work = DeviceCommandLogicalWork::tensor(shape.query_bytes / 2, shape.operations)
        .map_err(|e| e.to_string())?;
    MetalDeviceCommand::tensor_operation(kernel, regions, work, move |encoder, regions| {
        let compute = encoder.compute_encoder();
        compute.set_compute_pipeline_state(
            (if use_all_valid_register {
                pipelines.attention_register128_all_valid.as_ref()
            } else if use_register {
                pipelines.attention_register128.as_ref()
            } else if use_mpp {
                pipelines.attention_mpp_head128.as_ref()
            } else if shape.params.head_dim == 128 {
                pipelines
                    .attention_head128
                    .as_ref()
                    .or(pipelines.attention.as_ref())
            } else {
                pipelines.attention.as_ref()
            })
            .expect("selected attention pipeline"),
        );
        for (i, region) in regions.iter().enumerate() {
            set_region(compute, i as u64, region);
        }
        let grid_groups = if let Some(params) = &register_params {
            set_params(compute, 4, params);
            params.grid_groups
        } else {
            set_params(compute, 4, &shape.params);
            shape.params.grid_groups
        };
        set_params(compute, 5, &shape.mask);
        compute.set_threadgroup_memory_length(0, shared_bytes);
        compute.dispatch_thread_groups(MTLSize::new(grid_groups, 1, 1), MTLSize::new(128, 1, 1));
        Ok(())
    })
    .map_err(|e| e.to_string())
}
