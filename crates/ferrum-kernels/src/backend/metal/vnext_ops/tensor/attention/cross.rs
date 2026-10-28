//! Rectangular attention over the shared bounded online-attention shader.
use super::super::*;
use super::{dense_guarded, register128, NativeMask, CACHED_SHARED_BYTES};
use ferrum_runtime::CrossAttentionBf16Geometry;

pub(in crate::backend::metal::vnext_ops::tensor) const PROFILE: &str =
    "metal.register_guarded_accum32x16_or_mpp_simd16x32.f32.cross_attention.bf16.online.v3";
pub(in crate::backend::metal::vnext_ops::tensor) const FINGERPRINT_SOURCE: &str =
    concat!(include_str!("cross.rs"), include_str!("cross.metal"));

pub(in crate::backend::metal::vnext_ops::tensor) struct CrossPipelines {
    generic: ComputePipelineState,
    cached: Option<ComputePipelineState>,
    mpp: Option<ComputePipelineState>,
    register: Option<ComputePipelineState>,
    register_all_valid: Option<ComputePipelineState>,
    pub(in crate::backend::metal::vnext_ops::tensor) shared_limit: u64,
}

pub(in crate::backend::metal::vnext_ops::tensor) fn pipelines(
    runtime: &MetalDeviceRuntime,
) -> Result<CrossPipelines, MetalDeviceRuntimeError> {
    use linear::mpp::{compile_options, MppCompileOptions};
    let source = concat!(
        include_str!("../bf16.metal"),
        "\n",
        include_str!("cross.metal"),
        "\n",
        include_str!("../attention.metal")
    );
    let options = CompileOptions::new();
    options.set_fast_math_enabled(false);
    let library = runtime
        .device()
        .new_library_with_source(source, &options)
        .map_err(MetalDeviceRuntimeError::contract)?;
    let pipeline = |library: &metal::LibraryRef, cached: bool| {
        let constants = metal::FunctionConstantValues::new();
        if cached {
            constants.set_constant_value_at_index(
                (&128_u32 as *const u32).cast(),
                metal::MTLDataType::UInt,
                101,
            );
        }
        let function = library
            .get_function("ferrum_tensor_cross_attention_bf16", Some(constants))
            .map_err(MetalDeviceRuntimeError::contract)?;
        runtime
            .device()
            .new_compute_pipeline_state_with_function(&function)
            .map_err(MetalDeviceRuntimeError::contract)
    };
    let generic = pipeline(&library, false)?;
    if generic.thread_execution_width() != 32 || generic.max_total_threads_per_threadgroup() < 128 {
        return Err(MetalDeviceRuntimeError::contract(
            "Metal cross attention needs four 32-lane SIMD groups",
        ));
    }
    let shared_limit = runtime
        .device()
        .max_threadgroup_memory_length()
        .checked_sub(generic.static_threadgroup_memory_length())
        .ok_or_else(|| {
            MetalDeviceRuntimeError::contract("Metal cross attention exceeds threadgroup memory")
        })?;
    let supports_cached = |p: &ComputePipelineState| {
        p.thread_execution_width() == 32
            && p.max_total_threads_per_threadgroup() >= 128
            && p.static_threadgroup_memory_length()
                .checked_add(CACHED_SHARED_BYTES)
                .is_some_and(|n| n <= runtime.device().max_threadgroup_memory_length())
    };
    let cached = pipeline(&library, true)?;
    let cached = supports_cached(&cached).then_some(cached);
    let mpp = match compile_options(runtime.device()).map_err(MetalDeviceRuntimeError::contract)? {
        MppCompileOptions::Unsupported { reason } => {
            tracing::info!(%reason, "Metal cross attention MPP unavailable; using SIMD-group fallback");
            None
        }
        MppCompileOptions::Ready(options) => {
            let source = concat!(
                include_str!("../bf16.metal"),
                "\n",
                include_str!("cross.metal"),
                "\n",
                include_str!("mpp.metal"),
                "\n",
                include_str!("../attention.metal")
            );
            let library = runtime
                .device()
                .new_library_with_source(source, &options)
                .map_err(|e| {
                    MetalDeviceRuntimeError::contract(format!(
                        "Metal cross attention MPP compilation failed: {e}"
                    ))
                })?;
            let mpp = pipeline(&library, true)?;
            if supports_cached(&mpp) {
                Some(mpp)
            } else {
                tracing::info!(static_bytes = mpp.static_threadgroup_memory_length(),
                    maximum_threads = mpp.max_total_threads_per_threadgroup(),
                    "Metal cross attention MPP exceeds SIMD/threadgroup limits; using SIMD-group fallback");
                None
            }
        }
    };
    let [register, register_all_valid] = register128::compile_cross_variants(runtime)?;
    Ok(CrossPipelines {
        generic,
        cached,
        mpp,
        register,
        register_all_valid,
        shared_limit,
    })
}

// The shared shader adds key_sequence only when FERRUM_ATTENTION_CROSS is set.
// Existing square-attention packets retain their original layout unchanged.
#[repr(C)]
struct Params {
    jobs: u64,
    grid_groups: u64,
    query_tiles: u64,
    sequence: u64,
    query_heads: u64,
    key_value_heads: u64,
    key_sequence: u64,
    head_dim: u32,
    scale: f32,
}
const _: () = assert!(std::mem::size_of::<Params>() == 64);

pub(in crate::backend::metal::vnext_ops::tensor) struct Shape {
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
            key: self.params.key_sequence,
            query_heads: self.params.query_heads,
            key_value_heads: self.params.key_value_heads,
            head_dim: self.params.head_dim,
            scale: self.params.scale,
            all_keys_valid: self.key_visibility == register128::KeyVisibility::AllValid,
            dense_start: Some(0),
        })
    }
}

pub(in crate::backend::metal::vnext_ops::tensor) fn geometry(
    values: &[ResolvedValueBinding],
    attributes: &BTreeMap<AttributeId, SemanticValue>,
    shared_limit: u64,
) -> Result<Shape, String> {
    if values.len() != 4 {
        return Err("Metal cross attention requires Q/K/V and one output".into());
    }
    let checked = CrossAttentionBf16Geometry::new(
        binding(values, ResolvedValueRole::Input, 0)?.tensor(),
        binding(values, ResolvedValueRole::Input, 1)?.tensor(),
        binding(values, ResolvedValueRole::Input, 2)?.tensor(),
        binding(values, ResolvedValueRole::Output, 0)?.tensor(),
        attributes,
    )
    .map_err(|e| e.to_string())?;
    if checked.batch() > 16 {
        return Err("Metal attention batch exceeds this provider's native packet capacity".into());
    }
    let shared_bytes = (2048 + 16 * checked.head_dim() + 96) * 4;
    if shared_bytes > shared_limit {
        return Err("Metal cross attention exceeds device threadgroup storage".into());
    }
    let query_tiles = checked.query_length().div_ceil(16);
    let jobs = checked
        .batch()
        .checked_mul(checked.heads())
        .and_then(|n| n.checked_mul(query_tiles))
        .ok_or("Metal cross attention grid overflows")?;
    let mut mask = NativeMask {
        isolated_segments: 0,
        segment_count: 0,
        segment_ends: [0; 32],
        bidirectional_mask: 0,
        key_run_counts: [0; 16],
        key_run_ends: [0; 512],
        key_valid_masks: [0; 16],
        first_valid_keys: [checked.key_value_length() as u32; 16],
    };
    for batch in 0..checked.batch() as usize {
        let ends = checked.key_run_ends(batch).expect("validated mask batch");
        let valid = checked.key_run_valid(batch).expect("validated mask batch");
        mask.key_run_counts[batch] = ends.len() as u32;
        let mut start = 0;
        for (index, (&end, &valid)) in ends.iter().zip(valid).enumerate() {
            mask.key_run_ends[batch * 32 + index] = end;
            mask.key_valid_masks[batch] |= u32::from(valid) << index;
            if valid {
                mask.first_valid_keys[batch] = mask.first_valid_keys[batch].min(start);
            }
            start = end;
        }
    }
    Ok(Shape {
        params: Params {
            jobs,
            grid_groups: jobs.min(65535),
            query_tiles,
            sequence: checked.query_length(),
            key_sequence: checked.key_value_length(),
            query_heads: checked.heads(),
            key_value_heads: checked.heads(),
            head_dim: checked.head_dim() as u32,
            scale: checked.effective_scale(),
        },
        mask,
        key_visibility: register128::KeyVisibility::from_cross_mask(&checked),
        query_bytes: checked.query_bytes(),
        key_value_bytes: checked.key_value_bytes(),
        shared_bytes,
        operations: checked.logical_arithmetic_operations(),
    })
}

fn supports_mpp(q: u64, kv: u64, heads: u64, width: u32) -> bool {
    width == 128
        && q >= 16
        && kv >= 32
        && heads > 0
        && q.max(kv)
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(128))
            .is_some_and(|n| n <= i32::MAX as u64)
}

pub(in crate::backend::metal::vnext_ops::tensor) fn encode(
    kind: MetalTensorOperation,
    participant: &TensorParticipant<'_>,
    pipelines: Arc<Pipelines>,
) -> Result<MetalDeviceCommand, String> {
    let cross = pipelines
        .cross_attention
        .as_ref()
        .expect("selected cross attention pipelines");
    let shape = geometry(
        participant.bindings(),
        participant.attributes(),
        cross.shared_limit,
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
    if kind == MetalTensorOperation::CrossAttentionFused && pipelines.dense_attention.is_some() {
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
                "metal.cross_attention.fused_dense_guarded",
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
                        true,
                    );
                    Ok(())
                },
            )
            .map_err(|e| e.to_string());
        }
    }
    let register_supported = register128::supports_cross_shape(
        shape.params.sequence,
        shape.params.key_sequence,
        shape.params.head_dim,
    );
    let use_all_valid_register = register_supported
        && shape.key_visibility == register128::KeyVisibility::AllValid
        && cross.register_all_valid.is_some();
    let use_register = use_all_valid_register || (register_supported && cross.register.is_some());
    let register_params = if use_register {
        Some(register128::RegisterParams::new(
            shape.params.jobs / shape.params.query_tiles,
            shape.params.sequence,
            shape.params.key_sequence,
            shape.params.query_heads,
            shape.params.key_value_heads,
            shape.params.scale,
        )?)
    } else {
        None
    };
    let use_mpp = !use_register
        && cross.mpp.is_some()
        && supports_mpp(
            shape.params.sequence,
            shape.params.key_sequence,
            shape.params.query_heads,
            shape.params.head_dim,
        );
    let cached =
        !use_register && (use_mpp || (shape.params.head_dim == 128 && cross.cached.is_some()));
    let kernel = if use_all_valid_register {
        "metal.cross_attention.bf16.register32x16.all_valid_keys"
    } else if use_register {
        "metal.cross_attention.bf16.register32x16"
    } else if use_mpp {
        "metal.cross_attention.bf16.mpp_qk"
    } else {
        "metal.cross_attention.bf16"
    };
    let shared_bytes = if use_register {
        register128::SHARED_BYTES
    } else if cached {
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
        query_sequence = shape.params.sequence,
        key_sequence = shape.params.key_sequence,
        query_heads = shape.params.query_heads,
        "Metal cross attention dispatch selected"
    );
    let work = DeviceCommandLogicalWork::tensor(shape.query_bytes / 2, shape.operations)
        .map_err(|e| e.to_string())?;
    MetalDeviceCommand::tensor_operation(kernel, regions, work, move |encoder, regions| {
        let cross = pipelines
            .cross_attention
            .as_ref()
            .expect("selected cross attention pipelines");
        let compute = encoder.compute_encoder();
        compute.set_compute_pipeline_state(if use_all_valid_register {
            cross.register_all_valid.as_ref().unwrap()
        } else if use_register {
            cross.register.as_ref().unwrap()
        } else if use_mpp {
            cross.mpp.as_ref().unwrap()
        } else if cached {
            cross.cached.as_ref().unwrap()
        } else {
            &cross.generic
        });
        for (index, region) in regions.iter().enumerate() {
            set_region(compute, index as u64, region);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::metal::vnext_ops::tensor::tests::value;
    use ferrum_runtime::CanonicalRational;

    #[test]
    fn rectangular_geometry_preserves_mask_strides_and_exact_threadgroup_capacity() {
        let id = |name: &str| AttributeId::new(name).unwrap();
        let attributes = BTreeMap::from([
            (id("batch"), SemanticValue::Unsigned(2)),
            (
                id("scale"),
                SemanticValue::Rational(CanonicalRational::new(1, 8).unwrap()),
            ),
            (id("key_run_counts"), SemanticValue::Integers(vec![3, 1])),
            (
                id("key_run_lengths"),
                SemanticValue::Integers(vec![1, 31, 1, 33]),
            ),
            (
                id("key_run_valid"),
                SemanticValue::Integers(vec![0, 1, 0, 0]),
            ),
        ]);
        for width in [2, 72, 128, 256] {
            let mut attributes = attributes.clone();
            attributes.insert(id("head_dim"), SemanticValue::Unsigned(width));
            let mut values = vec![
                value(
                    "q",
                    ResolvedValueRole::Input,
                    0,
                    &[34, 2 * width],
                    ElementType::Bf16,
                ),
                value(
                    "k",
                    ResolvedValueRole::Input,
                    1,
                    &[66, 2 * width],
                    ElementType::Bf16,
                ),
                value(
                    "v",
                    ResolvedValueRole::Input,
                    2,
                    &[66, 2 * width],
                    ElementType::Bf16,
                ),
                value(
                    "y",
                    ResolvedValueRole::Output,
                    0,
                    &[34, 2 * width],
                    ElementType::Bf16,
                ),
            ];
            let required = (2048 + 16 * width + 96) * 4;
            assert!(geometry(&values, &attributes, required - 1).is_err());
            let shape = geometry(&values, &attributes, required).unwrap();
            assert_eq!(shape.params.sequence, 17);
            assert_eq!(shape.params.key_sequence, 33);
            assert_eq!(shape.params.jobs, 8);
            assert_eq!(shape.query_bytes, 34 * 2 * width * 2);
            assert_eq!(shape.key_value_bytes, 66 * 2 * width * 2);
            assert_eq!(&shape.mask.first_valid_keys[..2], &[1, 33]);
            assert_eq!(&shape.mask.key_run_ends[..3], &[1, 32, 33]);
            assert_eq!(shape.mask.key_run_ends[32], 33);
            assert_eq!(&shape.mask.key_valid_masks[..2], &[2, 0]);
            assert_eq!(shape.key_visibility, register128::KeyVisibility::MaskRuns);
            // The proof must cover KV=33, not Q=17, and every batch slot.
            let mut all_valid = attributes.clone();
            all_valid.insert(id("key_run_counts"), SemanticValue::Integers(vec![1, 1]));
            all_valid.insert(id("key_run_lengths"), SemanticValue::Integers(vec![33, 33]));
            all_valid.insert(id("key_run_valid"), SemanticValue::Integers(vec![1, 1]));
            assert_eq!(
                geometry(&values, &all_valid, required)
                    .unwrap()
                    .key_visibility,
                register128::KeyVisibility::AllValid
            );
            all_valid.insert(id("key_run_valid"), SemanticValue::Integers(vec![1, 0]));
            assert_eq!(
                geometry(&values, &all_valid, required)
                    .unwrap()
                    .key_visibility,
                register128::KeyVisibility::MaskRuns
            );
            // Cross remains the same-head contract, not rectangular GQA.
            let mut unequal_heads = values.clone();
            for (port, name) in [(1, "k"), (2, "v")] {
                unequal_heads[port] = value(
                    name,
                    ResolvedValueRole::Input,
                    port as u32,
                    &[66, width],
                    ElementType::Bf16,
                );
            }
            assert!(geometry(&unequal_heads, &attributes, required).is_err());
            // V cannot silently inherit Q's shorter sequence or bypass the
            // standard contract just because its storage region is valid.
            values[2] = value(
                "v",
                ResolvedValueRole::Input,
                2,
                &[34, 2 * width],
                ElementType::Bf16,
            );
            assert!(geometry(&values, &attributes, required).is_err());
        }
    }

    #[test]
    fn cross_mpp_checks_both_rectangular_extents() {
        assert!(supports_mpp(16, 65, 4, 128));
        assert!(supports_mpp(65, 32, 4, 128));
        assert!(!supports_mpp(15, 65, 4, 128));
        assert!(!supports_mpp(65, 31, 4, 128));
        assert!(!supports_mpp(32, 32, 0, 128));
        assert!(!supports_mpp(32, 32, 4, 72));
        let maximum = i32::MAX as u64 / 128;
        assert!(supports_mpp(maximum, maximum, 1, 128));
        assert!(!supports_mpp(maximum + 1, 32, 1, 128));
        assert!(!supports_mpp(16, maximum + 1, 1, 128));
        assert!(!supports_mpp(16, u64::MAX, 4, 128));
    }
}
