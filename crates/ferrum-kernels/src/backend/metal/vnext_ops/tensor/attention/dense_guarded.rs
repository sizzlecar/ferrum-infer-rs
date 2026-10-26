//! Explicit fused profile: GPU qualification, dense suffix, full-grid fallback.
use super::*;
use crate::backend::metal::vnext_runtime::MetalSubmissionEncoder;
use ferrum_runtime::{
    ProviderWorkspaceRequirement, ProviderWorkspaceReusePolicy, ProviderWorkspaceScope,
    ProviderWorkspaceSizeFormula,
};

pub(in crate::backend::metal::vnext_ops::tensor) const FINGERPRINT_SOURCE: &str = concat!(
    include_str!("dense_guarded.rs"),
    include_str!("dense_guarded.metal"),
    include_str!("dense_core.metal")
);
pub(in crate::backend::metal::vnext_ops::tensor) const NUMERICAL_IDENTITY: &str =
    "steel-v0.32.2-d128-bq32-bk16-native-bfloat-f32-fragments;scale-log2e-fast-exp2-final-div;dense-suffix-only;all-valid-original-specialization;key-runs-masked-KV-load-zero-masked-P-zero-empty-key-tiles-skipped;visible-input-abs<=2^20-scale<=2^20-K<=65536;empty-key-batches-strict;actual-PSO-coordinate-check;final-finite-positive-denominator-output<=2^21-check;per-original-job-flags;matching-key-visibility-strict-full-grid-fallback;independent-fused-profile-not-IEEE-bitwise.v2";

#[repr(C)]
struct DenseParams {
    batch: i32,
    heads: i32,
    width: i32,
    query: i32,
    key: i32,
    gqa: i32,
    scale: f32,
    query_tiles: i32,
    key_tiles: i32,
    aligned_query_tiles: i32,
    aligned_key_tiles: i32,
    query_remainder: i32,
    key_remainder: i32,
    query_offset: i32,
    query_strides: [i64; 3],
    key_strides: [i64; 3],
    value_strides: [i64; 3],
    output_strides: [i64; 3],
}
const _: () = assert!(std::mem::size_of::<DenseParams>() == 152);

#[repr(C)]
struct GateParams {
    original_query_tiles: u64,
    first_fast_tile: u64,
}

pub(super) struct GuardGeometry {
    pub batch_heads: u64,
    pub query: u64,
    pub key: u64,
    pub query_heads: u64,
    pub key_value_heads: u64,
    pub head_dim: u32,
    pub scale: f32,
    pub all_keys_valid: bool,
    pub dense_start: Option<u64>,
}

pub(in crate::backend::metal::vnext_ops::tensor) struct Schedule {
    strict: register128::RegisterParams,
    dense: DenseParams,
    gate: GateParams,
    query_offset_bytes: u64,
    flag_bytes: u64,
    masked_keys: bool,
}

impl Schedule {
    pub(super) fn new(g: GuardGeometry) -> Result<Option<Self>, String> {
        let Some(start) = g.dense_start else {
            return Ok(None);
        };
        if g.head_dim != 128
            || g.query_heads != g.key_value_heads
            || g.query_heads == 0
            || g.query_heads > i32::MAX as u64
            || g.batch_heads == 0
            || !g.batch_heads.is_multiple_of(g.query_heads)
            || g.query < 32
            || g.query > i32::MAX as u64
            || g.key == 0
            || g.key > 65536
            || start >= g.query
            || !g.scale.is_finite()
            || g.scale <= 0.0
            || g.scale > 1_048_576.0
        {
            return Ok(None);
        }
        let batch = g.batch_heads / g.query_heads;
        if batch > 16 {
            return Ok(None);
        }
        let first_fast_tile = start.div_ceil(32);
        let first_query = first_fast_tile * 32;
        if first_query >= g.query {
            return Ok(None);
        }
        let query = g.query - first_query;
        let row_stride = g
            .query_heads
            .checked_mul(128)
            .ok_or("dense row stride overflow")?;
        let query_stride = g
            .query
            .checked_mul(row_stride)
            .ok_or("dense query stride overflow")?;
        let key_stride = g
            .key
            .checked_mul(row_stride)
            .ok_or("dense key stride overflow")?;
        for stride in [query_stride, key_stride] {
            if stride
                .checked_mul(batch)
                .and_then(|n| n.checked_mul(2))
                .is_none_or(|n| n > isize::MAX as u64)
            {
                return Err("dense attention extent exceeds address space".into());
            }
        }
        let strict = register128::RegisterParams::new(
            g.batch_heads,
            g.query,
            g.key,
            g.query_heads,
            g.key_value_heads,
            g.scale,
        )?;
        let flag_bytes = checked_flag_bytes(strict.jobs)?;
        let query_strides = [query_stride as i64, 128, row_stride as i64];
        let key_strides = [key_stride as i64, 128, row_stride as i64];
        Ok(Some(Self {
            gate: GateParams {
                original_query_tiles: strict.query_tiles,
                first_fast_tile,
            },
            query_offset_bytes: first_query * row_stride * 2,
            dense: DenseParams {
                batch: batch as i32,
                heads: g.query_heads as i32,
                width: 128,
                query: query as i32,
                key: g.key as i32,
                gqa: 1,
                scale: g.scale,
                query_tiles: query.div_ceil(32) as i32,
                key_tiles: g.key.div_ceil(16) as i32,
                aligned_query_tiles: (query / 32) as i32,
                aligned_key_tiles: (g.key / 16) as i32,
                query_remainder: (query % 32) as i32,
                key_remainder: (g.key % 16) as i32,
                query_offset: 0,
                query_strides,
                key_strides,
                value_strides: key_strides,
                output_strides: query_strides,
            },
            strict,
            flag_bytes,
            masked_keys: !g.all_keys_valid,
        }))
    }

    pub(in crate::backend::metal::vnext_ops::tensor) fn flag_bytes(&self) -> u64 {
        self.flag_bytes
    }
}

fn checked_flag_bytes(jobs: u64) -> Result<u64, String> {
    jobs.checked_mul(4)
        .and_then(|n| n.checked_add(15))
        .map(|n| n & !15)
        .filter(|&n| n > 0 && n <= isize::MAX as u64)
        .ok_or_else(|| "dense attention flags exceed address space".into())
}

pub(super) fn dense_start(mask: &NativeMask, sequence: u64) -> Option<u64> {
    let last = mask.segment_count.checked_sub(1)? as usize;
    if mask.isolated_segments != 0
        || last >= mask.segment_ends.len()
        || u64::from(mask.segment_ends[last]) != sequence
        || mask.bidirectional_mask & (1 << last) == 0
    {
        return None;
    }
    Some(if last == 0 {
        0
    } else {
        u64::from(mask.segment_ends[last - 1])
    })
}

pub(in crate::backend::metal::vnext_ops::tensor) struct DensePipelines {
    prepass: ComputePipelineState,
    fast: Vec<ComputePipelineState>,
    strict: [ComputePipelineState; 2],
    masked: Option<MaskedPipelines>,
}

struct MaskedPipelines {
    prepass: ComputePipelineState,
    fast: Vec<ComputePipelineState>,
    strict: [ComputePipelineState; 2],
}

impl DensePipelines {
    pub(in crate::backend::metal::vnext_ops::tensor) fn supports(&self, plan: &Schedule) -> bool {
        !plan.masked_keys || self.masked.is_some()
    }
}

pub(in crate::backend::metal::vnext_ops::tensor) fn compile(
    runtime: &MetalDeviceRuntime,
) -> Option<DensePipelines> {
    let result = compile_checked(runtime);
    match result {
        Ok(pipelines) => pipelines,
        Err(error) => {
            tracing::warn!(%error, "Metal fused dense pipeline unavailable; keeping strict attention");
            None
        }
    }
}

fn compile_checked(
    runtime: &MetalDeviceRuntime,
) -> Result<Option<DensePipelines>, MetalDeviceRuntimeError> {
    let options = CompileOptions::new();
    options.set_fast_math_enabled(false);
    options.set_language_version(metal::MTLLanguageVersion::V3_1);
    let source = concat!(
        include_str!("dense_core.metal"),
        "\n",
        include_str!("dense_guarded.metal")
    );
    let library = runtime
        .device()
        .new_library_with_source(source, &options)
        .map_err(MetalDeviceRuntimeError::contract)?;
    let create = |name: &str, constants: metal::FunctionConstantValues, dynamic: u64| {
        let function = library
            .get_function(name, Some(constants))
            .map_err(MetalDeviceRuntimeError::contract)?;
        let pipeline = runtime
            .device()
            .new_compute_pipeline_state_with_function(&function)
            .map_err(MetalDeviceRuntimeError::contract)?;
        let supported = pipeline.thread_execution_width() == 32
            && pipeline.max_total_threads_per_threadgroup() >= 128
            && pipeline
                .static_threadgroup_memory_length()
                .checked_add(dynamic)
                .is_some_and(|n| n <= runtime.device().max_threadgroup_memory_length());
        tracing::info!(
            kernel = name,
            supported,
            static_threadgroup_bytes = pipeline.static_threadgroup_memory_length(),
            dynamic_threadgroup_bytes = dynamic,
            "Metal fused dense pipeline capability; rejected capability retains strict path"
        );
        Ok::<_, MetalDeviceRuntimeError>(supported.then_some(pipeline))
    };
    let Some(prepass) = create(
        "ferrum_dense_attention_prepass",
        metal::FunctionConstantValues::new(),
        16,
    )?
    else {
        return Ok(None);
    };
    let mut fast = Vec::with_capacity(4);
    for aligned in 0..4 {
        let constants = metal::FunctionConstantValues::new();
        for (index, value) in [
            (200, aligned & 2 != 0),
            (201, aligned & 1 != 0),
            (202, true),
        ] {
            constants.set_constant_value_at_index(
                (&value as *const bool).cast(),
                metal::MTLDataType::Bool,
                index,
            );
        }
        let Some(pipeline) = create("ferrum_dense_attention_bf16_bq32_bk16_d128", constants, 0)?
        else {
            return Ok(None);
        };
        fast.push(pipeline);
    }
    let Some(structured) =
        register128::compile_gated(runtime, false, register128::KeyVisibility::AllValid)?
    else {
        return Ok(None);
    };
    let Some(cross) =
        register128::compile_gated(runtime, true, register128::KeyVisibility::AllValid)?
    else {
        return Ok(None);
    };
    // A masked specialization failure must not remove the original all-valid
    // pipelines. The containing provider retains both sets under its admitted
    // opaque native owner; no independent device allocation is introduced.
    let masked = (|| -> Result<Option<MaskedPipelines>, MetalDeviceRuntimeError> {
        let constants = metal::FunctionConstantValues::new();
        constants.set_constant_value_at_index(
            (&true as *const bool).cast(),
            metal::MTLDataType::Bool,
            203,
        );
        let Some(prepass) = create("ferrum_dense_attention_prepass", constants, 16)? else {
            return Ok(None);
        };
        let mut fast = Vec::with_capacity(4);
        for aligned in 0..4 {
            let constants = metal::FunctionConstantValues::new();
            for (index, value) in [
                (200, aligned & 2 != 0),
                (201, aligned & 1 != 0),
                (202, true),
                (203, true),
            ] {
                constants.set_constant_value_at_index(
                    (&value as *const bool).cast(),
                    metal::MTLDataType::Bool,
                    index,
                );
            }
            let Some(pipeline) =
                create("ferrum_dense_attention_bf16_bq32_bk16_d128", constants, 0)?
            else {
                return Ok(None);
            };
            fast.push(pipeline);
        }
        // Prefix rows and rejected suffix jobs still consume the original
        // mask. An all-valid specialization would load masked NaN/Inf and
        // produce nonzero/NaN output for rows with no visible valid keys.
        let Some(structured) =
            register128::compile_gated(runtime, false, register128::KeyVisibility::MaskRuns)?
        else {
            return Ok(None);
        };
        let Some(cross) =
            register128::compile_gated(runtime, true, register128::KeyVisibility::MaskRuns)?
        else {
            return Ok(None);
        };
        Ok(Some(MaskedPipelines {
            prepass,
            fast,
            strict: [structured, cross],
        }))
    })();
    let masked = match masked {
        Ok(pipelines) => pipelines,
        Err(error) => {
            tracing::warn!(%error, "Metal masked dense pipeline unavailable; masked attention keeps strict path");
            None
        }
    };
    Ok(Some(DensePipelines {
        prepass,
        fast,
        strict: [structured, cross],
        masked,
    }))
}

pub(in crate::backend::metal::vnext_ops::tensor) fn estimate_workspace(
    descriptor: &OperationProviderDescriptor,
    request: &OperationResourceEstimateRequest<'_>,
    bytes: u64,
) -> Result<OperationResourceEstimate, VNextError> {
    Ok(OperationResourceEstimate::new(
        descriptor.resource_estimator_id(),
        descriptor.resource_estimator_version(),
        descriptor.resource_estimator_implementation_fingerprint(),
        request.input_fingerprint(),
        16,
        Some(ProviderWorkspaceRequirement::from_formula(
            ProviderWorkspaceSizeFormula::fixed(bytes)?,
            16,
            ProviderWorkspaceScope::Invocation,
            ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
            DynamicStorageRequirement::contiguous(),
        )?),
        None,
    ))
}

pub(super) fn append_flags(
    participant: &TensorParticipant<'_>,
    plan: &Schedule,
    regions: &mut Vec<MetalBufferRegion>,
) -> Result<(), String> {
    let flags = contiguous_scratch_region(participant, plan.flag_bytes)?;
    if !flags.offset_bytes().is_multiple_of(16) {
        return Err("dense flags require 16-byte alignment".into());
    }
    for region in regions.iter() {
        disjoint(region, &flags)?;
    }
    regions.push(flags);
    Ok(())
}

pub(super) fn encode(
    encoder: &mut MetalSubmissionEncoder,
    regions: &[MetalBufferRegion],
    plan: &Schedule,
    mask: &NativeMask,
    pipelines: &DensePipelines,
    cross: bool,
) {
    let (prepass, fast, strict) = if plan.masked_keys {
        let masked = pipelines
            .masked
            .as_ref()
            .expect("selected masked dense pipelines");
        (&masked.prepass, &masked.fast, &masked.strict)
    } else {
        (&pipelines.prepass, &pipelines.fast, &pipelines.strict)
    };
    // Tracked buffers + encoder boundaries publish flags and output on-device.
    // Retained full regions remain authoritative even when Q/O start at a suffix.
    for phase in 0..3 {
        encoder.begin_compute_subwork(match phase {
            0 => "metal.attention.dense_guard",
            1 => "metal.attention.dense_suffix",
            _ => "metal.attention.strict_fallback",
        });
        let compute = encoder.compute_encoder();
        for (slot, region) in regions[..4].iter().enumerate() {
            set_region(compute, slot as u64, region);
        }
        set_region(compute, 6, &regions[4]);
        if plan.masked_keys && phase != 2 {
            // Same checked 2380-byte mask packet as the strict fallback. This
            // bounded inline constant copy adds no provider workspace buffer.
            set_params(compute, 7, mask);
        }
        if phase == 1 {
            let variant = usize::from(plan.dense.query_remainder == 0) * 2
                + usize::from(plan.dense.key_remainder == 0);
            compute.set_compute_pipeline_state(&fast[variant]);
            for slot in [0_usize, 3] {
                compute.set_buffer(
                    slot as u64,
                    Some(regions[slot].buffer()),
                    regions[slot].offset_bytes() + plan.query_offset_bytes,
                );
            }
            set_params(compute, 4, &plan.dense);
            set_params(compute, 5, &plan.gate);
            compute.dispatch_thread_groups(
                MTLSize::new(
                    plan.dense.query_tiles as u64,
                    plan.dense.heads as u64,
                    plan.dense.batch as u64,
                ),
                MTLSize::new(32, 4, 1),
            );
        } else {
            compute.set_compute_pipeline_state(if phase == 0 {
                prepass
            } else {
                &strict[usize::from(cross)]
            });
            set_params(compute, 4, &plan.strict);
            if phase == 0 {
                set_params(compute, 5, &plan.gate);
            } else {
                set_params(compute, 5, mask);
            }
            compute.set_threadgroup_memory_length(
                0,
                if phase == 0 {
                    16
                } else {
                    register128::SHARED_BYTES
                },
            );
            let groups = if phase == 0 {
                (plan.strict.jobs / plan.strict.query_tiles).min(65535)
            } else {
                plan.strict.grid_groups
            };
            compute.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(128, 1, 1));
        }
        encoder.record_compute_dispatches(1);
        encoder.end_compute();
    }
}

#[cfg(test)]
mod tests;
