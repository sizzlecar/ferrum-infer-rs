//! M8×N64 MMA qualification. Original M32 math/order is the equality oracle;
//! the unchanged raw-coefficient numerical catalog is reported separately.
use super::b8_two_b4::{
    exact_bits, run, Arm, Halves, Pipelines, Projection, HALF_GUARD, PREFIX, ROWS, WEIGHT_PREFIX,
};
use super::q4_b4_ffn_cooperative::oracle::metrics;
use super::*;
use crate::backend::metal::k_quant_gemm;

mod ffn;
mod gdn;
mod product;
mod tests;

const CANDIDATE_SOURCE: &str = include_str!("../../../k_quant_gemm_m8.metal");

fn pipelines(device: &Device) -> Pipelines {
    let production = MetalLinearPipelines::new(device).unwrap();
    let candidate = [
        production.k_quant_gemm.q4_k_m8.clone(),
        production.k_quant_gemm.q5_k_m8.clone(),
        production.k_quant_gemm.q6_k_m8.clone(),
    ];
    for pso in &candidate {
        assert_eq!(pso.thread_execution_width(), 32);
        assert!(pso.max_total_threads_per_threadgroup() >= 128);
    }
    Pipelines {
        production,
        candidate,
    }
}

fn identity() -> serde_json::Value {
    serde_json::json!({
        "candidate":"M8_N64_four_simdgroups_two_accumulators_each",
        "reference":"original_TiledGemm_M32_N64_before_M8_adoption",
        "source_sha256":format!("{:x}",Sha256::digest(k_quant_gemm::SHADER_SOURCE.as_bytes())),
        "candidate_source_sha256":format!("{:x}",Sha256::digest(CANDIDATE_SOURCE.as_bytes())),
        "necessary_admission":"all_stage_elements_bitwise_equal_to_original_M32_MMA",
        "absolute_numerical_quality_claimed":false,"release_approved":false,
        "q8_route":"unchanged_actual_B8_TiledGemm",
        "threadgroup_bytes":8192,"threads":128,"rows":ROWS,
        "input_prefix_half_elements":PREFIX,"weight_prefix_bytes":WEIGHT_PREFIX
    })
}

fn order(round: usize) -> [Arm; 2] {
    if round % 2 == 0 {
        [Arm::ProductionMma, Arm::M8Mma]
    } else {
        [Arm::M8Mma, Arm::ProductionMma]
    }
}

fn assert_finite_bits(actual: &[f16], expected: &[f16], scope: &str) {
    assert_eq!(actual.len(), expected.len(), "{scope}");
    for (index, (a, b)) in actual.iter().zip(expected).enumerate() {
        assert!(
            a.is_finite() && b.is_finite(),
            "{scope}[{index}] nonfinite: {a}/{b}"
        );
        assert_eq!(a.to_bits(), b.to_bits(), "{scope}[{index}]: {a}/{b}");
    }
}
