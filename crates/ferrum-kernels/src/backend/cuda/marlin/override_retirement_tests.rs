//! Call the linked native entrypoints before any native static configuration
//! can be cached. Product configuration validation is deliberately not used.

use super::profile_retirement_tests::{run_fresh_children, ChildCase};

#[path = "override_retirement_tests/fixtures.rs"]
mod fixtures;

fn classic_cases() -> [ChildCase<'static>; 2] {
    [
        ChildCase {
            name: "baseline",
            environment: &[],
            profile_canary: None,
        },
        // N=128 is legal for the original small-M automatic 128x128 tile,
        // but the retired override selects N=256 and rejects that same shape.
        ChildCase {
            name: "retired-tile",
            environment: &[("FERRUM_MARLIN_TILE", "64x256")],
            profile_canary: None,
        },
    ]
}

#[cfg(feature = "vllm-moe-marlin")]
fn moe_cases() -> [ChildCase<'static>; 3] {
    [
        ChildCase {
            name: "baseline",
            environment: &[],
            profile_canary: None,
        },
        ChildCase {
            name: "retired-thread-shape",
            environment: &[
                ("FERRUM_VLLM_MOE_THREAD_K", "17"),
                ("FERRUM_VLLM_MOE_THREAD_N", "19"),
            ],
            profile_canary: None,
        },
        ChildCase {
            name: "retired-block-bound",
            environment: &[("FERRUM_VLLM_MOE_BLOCKS_PER_SM", "5")],
            profile_canary: None,
        },
    ]
}

#[test]
#[ignore = "requires CUDA and the classic Marlin artifact; does not bypass SM120 failures"]
fn native_classic_dense_tile_retirement_preserves_cpu_reference() {
    run_fresh_children(
        module_path!(),
        "native_classic_dense_tile_retirement_preserves_cpu_reference",
        &classic_cases(),
        fixtures::classic_dense,
    );
}

#[test]
#[ignore = "requires CUDA and the classic fused Marlin artifact; does not bypass SM120 failures"]
fn native_classic_fused_tile_retirement_preserves_cpu_reference() {
    run_fresh_children(
        module_path!(),
        "native_classic_fused_tile_retirement_preserves_cpu_reference",
        &classic_cases(),
        fixtures::classic_fused,
    );
}

#[cfg(feature = "vllm-moe-marlin")]
#[test]
#[ignore = "requires CUDA and the native U4B8/F16 Marlin-MoE artifact"]
fn native_moe_f16_override_retirement_preserves_cpu_reference() {
    run_fresh_children(
        module_path!(),
        "native_moe_f16_override_retirement_preserves_cpu_reference",
        &moe_cases(),
        fixtures::vllm_f16,
    );
}

#[cfg(feature = "vllm-moe-marlin")]
#[test]
#[ignore = "requires CUDA and the native FP8 Marlin-MoE artifact"]
fn native_moe_fp8_override_retirement_preserves_cpu_reference() {
    run_fresh_children(
        module_path!(),
        "native_moe_fp8_override_retirement_preserves_cpu_reference",
        &moe_cases(),
        super::tests::qwen36_a3_fp8_marlin_moe_ffi_matches_cpu_reference_for_four_cases,
    );
}

#[cfg(feature = "vllm-moe-marlin")]
#[test]
#[ignore = "requires CUDA and the native MXFP4 Marlin-MoE artifact"]
fn native_moe_mxfp4_override_retirement_preserves_source_reference() {
    run_fresh_children(
        module_path!(),
        "native_moe_mxfp4_override_retirement_preserves_source_reference",
        &moe_cases(),
        super::tests::gpt_oss_mxfp4_marlin_moe_bf16_ffi_matches_source_reference_for_four_cases,
    );
}
