//! Existing Q8 entrypoints and launch geometry, shared with passive evidence.
use super::{GgufBlockFormat, LaunchConfig, Q8SumPolicy, QuantizedKernel};

pub(super) fn pack_entry(policy: Q8SumPolicy) -> &'static str {
    match policy {
        Q8SumPolicy::Quantized => "vnext_gguf_q8_f32scale_pack_f16_prototype",
        Q8SumPolicy::Input => "vnext_gguf_q8_f32scale_input_sum_pack_f16_prototype",
    }
}

pub(super) fn entries(format: GgufBlockFormat, policy: Q8SumPolicy) -> Option<[&'static str; 3]> {
    Some(match (format, policy) {
        (GgufBlockFormat::Q4K, Q8SumPolicy::Quantized) => [
            "vnext_gguf_q4k_q8_f32scale_dp4a_lane_f16_prototype",
            "vnext_gguf_q4k_q8_f32scale_dp4a_lane_tiled_f16_prototype",
            "vnext_gguf_q4k_q8_f32scale_mma_f16_prototype",
        ],
        (GgufBlockFormat::Q4K, Q8SumPolicy::Input) => [
            "vnext_gguf_q4k_q8_f32scale_input_sum_dp4a_lane_f16_prototype",
            "vnext_gguf_q4k_q8_f32scale_input_sum_dp4a_lane_tiled_f16_prototype",
            "vnext_gguf_q4k_q8_f32scale_input_sum_mma_f16_prototype",
        ],
        (GgufBlockFormat::Q5K, Q8SumPolicy::Quantized) => [
            "vnext_gguf_q5k_q8_f32scale_dp4a_lane_f16_prototype",
            "vnext_gguf_q5k_q8_f32scale_dp4a_lane_tiled_f16_prototype",
            "vnext_gguf_q5k_q8_f32scale_mma_f16_prototype",
        ],
        (GgufBlockFormat::Q5K, Q8SumPolicy::Input) => [
            "vnext_gguf_q5k_q8_f32scale_input_sum_dp4a_lane_f16_prototype",
            "vnext_gguf_q5k_q8_f32scale_input_sum_dp4a_lane_tiled_f16_prototype",
            "vnext_gguf_q5k_q8_f32scale_input_sum_mma_f16_prototype",
        ],
        // Q6 has no affine minimum and retains its original ABI for both policies.
        (GgufBlockFormat::Q6K, _) => [
            "vnext_gguf_q6k_q8_f32scale_dp4a_lane_f16_prototype",
            "vnext_gguf_q6k_q8_f32scale_dp4a_lane_tiled_f16_prototype",
            "vnext_gguf_q6k_q8_f32scale_mma_f16_prototype",
        ],
        _ => return None,
    })
}

pub(super) fn entry(
    format: GgufBlockFormat,
    policy: Q8SumPolicy,
    kernel: QuantizedKernel,
) -> Option<&'static str> {
    let index = match kernel {
        QuantizedKernel::Scalar => 0,
        QuantizedKernel::Tiled => 1,
        QuantizedKernel::Mma => 2,
    };
    Some(entries(format, policy)?[index])
}

pub(super) fn pack_config(rows: u32, columns: u32) -> Result<LaunchConfig, String> {
    let groups = u64::from(rows) * u64::from(columns / 32);
    Ok(LaunchConfig {
        grid_dim: (
            u32::try_from(groups.div_ceil(4)).map_err(|_| "Q8 pack grid overflows")?,
            1,
            1,
        ),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    })
}

pub(super) fn project_config(kernel: QuantizedKernel, rows: u32, outputs: u32) -> LaunchConfig {
    let (row_tile, column_tile) = kernel.tiles();
    LaunchConfig {
        grid_dim: (outputs.div_ceil(column_tile), rows.div_ceil(row_tile), 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    }
}
