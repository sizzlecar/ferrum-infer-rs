#![cfg(all(feature = "metal", target_os = "macos"))]

use ferrum_interfaces::vnext::*;
use ferrum_kernels::backend::metal::{
    vnext_ops::MetalVNextComposition, vnext_runtime::MetalDeviceRuntime,
};
use half::f16;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

#[path = "vnext_checkpoint_continuation/family.rs"]
mod family;
#[path = "vnext_checkpoint_continuation/runtime.rs"]
mod runtime;

use family::{AttentionKind, Family, HIDDEN, MAX_TOKENS};
use runtime::Fixture;
#[path = "vnext_checkpoint_continuation/checks.rs"]
mod checks;
use checks::{verify, verify_completed_input, verify_with_timing};

type Runtime = MetalDeviceRuntime;

fn composition(kind: AttentionKind, _family: &PreparedModelFamily) -> runtime::Composition {
    let (runtime, registry, materializers, materializer_id, catalog) =
        MetalVNextComposition::create(id(format!("device.metal.checkpoint.{kind:?}")))
            .unwrap()
            .into_parts();
    (
        runtime,
        registry,
        materializers,
        WeightMaterializerSelection::exact(materializer_id),
        catalog,
    )
}

fn id<T>(value: impl Into<String>) -> T
where
    T: TryFrom<String>,
    T::Error: std::fmt::Debug,
{
    T::try_from(value.into()).unwrap()
}

#[test]
fn causal_f32_master_q4k_provider_resumes_public_native_checkpoint() {
    verify(AttentionKind::Causal, &[0..64, 64..65], &[65..73, 73..74]);
    verify(AttentionKind::Causal, &[0..17, 17..65], &[65..73, 73..74]);
}

#[test]
fn causal_int8_kv_q4k_provider_resumes_payload_and_scales_across_a_page_boundary() {
    // The I8 payload row is 512 bytes: position 128 crosses its 64 KiB page,
    // while the independent F32 scales remain inside their own page.
    verify(
        AttentionKind::CausalInt8,
        &[0..128, 128..129],
        &[129..137, 137..138],
    );
    verify(
        AttentionKind::CausalInt8,
        &[0..17, 17..129],
        &[129..137, 137..138],
    );
}

#[test]
fn causal_int8_kv_captures_completed_input_and_restores_an_appended_suffix() {
    verify_completed_input(
        AttentionKind::CausalInt8,
        &[0..128, 128..129],
        &[129..130, 130..138],
    );
}

#[test]
fn causal_int8_kv_checkpoint_rejects_a_same_token_f16_target() {
    let quantized = Fixture::new(AttentionKind::CausalInt8);
    let f16 = Fixture::new(AttentionKind::Causal);
    let tokens: Arc<[u32]> = (0..10).map(|index| ((index * 7 + 3) % 32) as u32).collect();
    let source = quantized.admit("int8-source", Arc::clone(&tokens));
    quantized
        .execute(&source, Arc::clone(&tokens), 0..8)
        .assert_state_nonzero();
    let checkpoint = quantized.capture(&source);
    let target = f16.admit("f16-target", Arc::clone(&tokens));
    f16.assert_restore_rejected(&target, &checkpoint, tokens);
    target.try_abort_if_quiescent().unwrap();
    source.try_abort_if_quiescent().unwrap();
}

#[test]
fn gated_delta_f32_master_q4k_provider_resumes_public_native_checkpoint() {
    verify(AttentionKind::GatedDelta, &[0..2, 2..5], &[5..6, 6..9]);
    verify(AttentionKind::GatedDelta, &[0..5], &[5..6, 6..9]);
}

#[test]
fn gated_delta_hadamard_pq2_shared_signs_and_mixed_projections_resume_public_checkpoint() {
    for kind in [
        AttentionKind::GatedDeltaHadamardF16,
        AttentionKind::GatedDeltaHadamardF32,
    ] {
        verify(kind, &[0..2, 2..5], &[5..6, 6..9]);
        verify_completed_input(kind, &[0..4, 4..5], &[5..6, 6..9]);
    }
}

#[test]
fn native_checkpoint_completion_timing_preserves_metal_continuation() {
    verify_with_timing(
        AttentionKind::Causal,
        &[0..64, 64..65],
        &[65..73, 73..74],
        DeviceTimingMode::Completion,
    );
    verify_with_timing(
        AttentionKind::GatedDelta,
        &[0..2, 2..5],
        &[5..6, 6..9],
        DeviceTimingMode::Completion,
    );
}

#[test]
fn causal_f32_master_captures_growing_decode_input_end() {
    verify_completed_input(AttentionKind::Causal, &[0..64, 64..65], &[65..66, 66..74]);
}

#[test]
fn gated_delta_f32_master_captures_growing_decode_input_end() {
    verify_completed_input(AttentionKind::GatedDelta, &[0..4, 4..5], &[5..6, 6..9]);
}

#[test]
fn causal_int8_kv_eager_failure_does_not_poison_the_execution_lane() {
    checks::verify_eager_numerical_failure();
}
