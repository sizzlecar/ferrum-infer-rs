//! Optional Head128 QK/PV: borrowed BF16 Q/K and FP32 probability/value tiles.
//! The shared shader keeps its online softmax and visible-only nonfinite PV.
use crate::backend::metal::vnext_ops::tensor::linear::mpp::{
    compile_options, MppCompileOptions, MppPipelineAvailability,
};
use metal::DeviceRef;

pub(in crate::backend::metal::vnext_ops::tensor) const FINGERPRINT_SOURCE: &str =
    concat!(include_str!("mpp.rs"), include_str!("mpp.metal"));
pub(in crate::backend::metal::vnext_ops::tensor) const NUMERICAL_IDENTITY: &str =
    "metal.attention.qk-mpp-bf16-bf16-f32-m16n32k128-full-visible;simdgroup-f32-qk-fallback-staged-query32;head128-output-cache-without-q-fragment-cache;pv-mpp-f32-f32-f32-m16n32k32-finite-threadgroup-tiles;unchanged-f32-softmax-and-visible-only-nonfinite-pv;relaxed=false.v3";

pub(in crate::backend::metal::vnext_ops::tensor) fn compile(
    device: &DeviceRef,
) -> Result<MppPipelineAvailability, String> {
    let options = match compile_options(device)? {
        MppCompileOptions::Unsupported { reason } => {
            return Ok(MppPipelineAvailability::Unsupported { reason });
        }
        MppCompileOptions::Ready(options) => options,
    };
    let source = concat!(
        include_str!("../bf16.metal"),
        "\n",
        include_str!("mpp.metal"),
        "\n",
        include_str!("../attention.metal")
    );
    let library = device
        .new_library_with_source(source, &options)
        .map_err(|error| format!("MPP attention MSL 4 library compilation failed: {error}"))?;
    let constants = metal::FunctionConstantValues::new();
    constants.set_constant_value_at_index(
        (&128_u32 as *const u32).cast(),
        metal::MTLDataType::UInt,
        101,
    );
    let function = library
        .get_function("ferrum_tensor_attention_bf16", Some(constants))
        .map_err(|error| format!("MPP attention kernel lookup failed: {error}"))?;
    let pipeline = device
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|error| format!("MPP attention pipeline creation failed: {error}"))?;
    let required = pipeline
        .static_threadgroup_memory_length()
        .checked_add(super::CACHED_SHARED_BYTES);
    if pipeline.thread_execution_width() != 32
        || pipeline.max_total_threads_per_threadgroup() < 128
        || required.is_none_or(|bytes| bytes > device.max_threadgroup_memory_length())
    {
        return Ok(MppPipelineAvailability::Unsupported {
            reason: format!(
                "MPP attention requires four 32-thread SIMD groups and static+dynamic threadgroup memory within the device limit; width={}, maximum_threads={}, static_bytes={}, dynamic_bytes={}, limit={}",
                pipeline.thread_execution_width(),
                pipeline.max_total_threads_per_threadgroup(),
                pipeline.static_threadgroup_memory_length(),
                super::CACHED_SHARED_BYTES,
                device.max_threadgroup_memory_length()
            ),
        });
    }
    Ok(MppPipelineAvailability::Ready(pipeline))
}

/// A tensor view uses signed-int extents and strides. Batch/head pointer bases
/// stay in the original checked ulong domain; each strided sequence must fit int.
pub(super) fn supports_shape(
    sequence: u64,
    query_heads: u64,
    key_value_heads: u64,
    head_dim: u32,
) -> bool {
    let fits = |heads: u64| {
        sequence
            .checked_mul(heads)
            .and_then(|n| n.checked_mul(128))
            .is_some_and(|n| n <= i32::MAX as u64)
    };
    head_dim == 128
        && sequence >= 32
        && query_heads > 0
        && key_value_heads > 0
        && query_heads % key_value_heads == 0
        && fits(query_heads)
        && fits(key_value_heads)
}

#[cfg(test)]
mod tests {
    use super::supports_shape;

    #[test]
    fn qk_mpp_requires_full_key_tile_and_checked_head_strides() {
        assert!(supports_shape(32, 32, 32, 128));
        assert!(supports_shape(65, 8, 2, 128));
        assert!(!supports_shape(31, 8, 2, 128));
        assert!(!supports_shape(32, 8, 2, 64));
        assert!(!supports_shape(32, 0, 2, 128));
        assert!(!supports_shape(32, 8, 0, 128));
        assert!(!supports_shape(32, 7, 2, 128));
        let maximum_sequence = i32::MAX as u64 / 128;
        assert!(supports_shape(maximum_sequence, 1, 1, 128));
        assert!(!supports_shape(maximum_sequence + 1, 1, 1, 128));
        assert!(!supports_shape(u64::MAX, 2, 1, 128));
        assert!(!supports_shape(32, u64::MAX, 1, 128));
    }
}
