#![cfg(feature = "cuda")]

use ferrum_interfaces::vnext::*;
use ferrum_kernels::backend::cuda::{
    vnext_ops::{cuda_weight_materializer_selection, CudaVNextComposition},
    vnext_runtime::CudaDeviceRuntime,
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
use checks::{verify, verify_completed_input};

type Runtime = CudaDeviceRuntime;

fn composition(kind: AttentionKind, family: &PreparedModelFamily) -> runtime::Composition {
    let (runtime, registry, materializers, catalog) = CudaVNextComposition::create(
        0,
        id(format!("device.cuda.checkpoint.{kind:?}")),
        ferrum_types::AttentionExecutionPolicy::Portable,
    )
    .unwrap()
    .into_parts();
    let materializer = cuda_weight_materializer_selection(family).unwrap();
    (runtime, registry, materializers, materializer, catalog)
}

fn id<T>(value: impl Into<String>) -> T
where
    T: TryFrom<String>,
    T::Error: std::fmt::Debug,
{
    T::try_from(value.into()).unwrap()
}

#[test]
fn causal_int8_kv_q4k_provider_resumes_payload_and_scales_across_a_page_boundary() {
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
fn causal_int8_kv_typed_slots_replay_and_recover_after_nonfinite_input() {
    let fixture = Fixture::with_execution_options(AttentionKind::CausalInt8, true, Some(31));
    let tokens: Arc<[u32]> = Arc::from([3, 4, 5, 6]);
    let source = fixture.admit("replay-warmup", Arc::clone(&tokens));
    // The first quiescent shape warms the native libraries; the next execution
    // captures it. Each wave updates token, state and output typed slot pointers.
    let first = fixture.execute(&source, Arc::clone(&tokens), 0..1);
    let second = fixture.execute(&source, Arc::clone(&tokens), 1..2);
    let third = fixture.execute_replayed(&source, Arc::clone(&tokens), 2..3);
    third.assert_state_nonzero();
    let checkpoint = fixture.capture(&source);
    assert_eq!(checkpoint.logical_bytes(), 3 * (512 + 16));
    source.try_abort_if_quiescent().unwrap();
    drop(source);
    let restored = fixture.admit("replay-restored", Arc::clone(&tokens));
    fixture.restore(&restored, &checkpoint, Arc::clone(&tokens));
    let suffix = fixture.execute_replayed(&restored, Arc::clone(&tokens), 3..4);
    restored.try_complete().unwrap();
    drop((restored, checkpoint));

    // Real provider work injects NaN through one embedding row. Exercise both
    // the typed binding prelude and direct graph invocation, then reuse the
    // same lane and resident slot allocation with a different sequence owner.
    let bad_tokens: Arc<[u32]> = Arc::from([31, 4]);
    let bad = fixture.admit("replay-nonfinite", Arc::clone(&bad_tokens));
    fixture.execute_numerical_failure(&bad, bad_tokens, 0..1, true);
    drop(bad);
    let good = fixture.admit("replay-after-failure", Arc::clone(&tokens));
    first.assert_same(
        &fixture.execute_replayed(&good, Arc::clone(&tokens), 0..1),
        "replay after failure first",
    );
    second.assert_same(
        &fixture.execute_replayed(&good, Arc::clone(&tokens), 1..2),
        "replay after failure second",
    );
    third.assert_same(
        &fixture.execute_replayed(&good, Arc::clone(&tokens), 2..3),
        "replay after failure third",
    );
    suffix.assert_same(
        &fixture.execute_replayed(&good, tokens, 3..4),
        "replay restored suffix",
    );
    good.try_complete().unwrap();
}

#[test]
fn causal_int8_kv_eager_failure_does_not_poison_the_execution_lane() {
    checks::verify_eager_numerical_failure();
}

#[test]
fn gated_delta_q8_provider_charges_pack_and_replays_changed_inputs_with_isolated_state() {
    for rows in [1, 8] {
        verify_gated_delta_q8_provider_replay(rows);
    }
}

fn verify_gated_delta_q8_provider_replay(rows: usize) {
    let eager = Fixture::new(AttentionKind::GatedDeltaQ8Projections);
    let replay =
        Fixture::with_replay_token_capacity(AttentionKind::GatedDeltaQ8Projections, rows as u64);
    {
        // The real compiler invokes both selected estimators. This comparison
        // checks the additional pack allocation without inventing an invocation
        // or substituting a provider descriptor in the test.
        let strict = Fixture::new(AttentionKind::GatedDelta);
        eager.assert_q8_projection_workspace(&strict);
        replay.assert_q8_projection_workspace(&strict);
    }
    // Each mapping has its own same-width eager oracle. Lane and MMA may use
    // different permitted accumulation orders; this is not a cross-width
    // equality assertion. Inputs vary within and between all four windows.
    let tokens_a: Arc<[u32]> = (0..4 * rows)
        .map(|index| 3 + ((index + index / rows) % 8) as u32)
        .collect();
    let tokens_b: Arc<[u32]> = (0..4 * rows)
        .map(|index| 11 + ((index + index / rows) % 8) as u32)
        .collect();
    let window = |index| index * rows..(index + 1) * rows;
    let baseline = |name, tokens: &Arc<[u32]>| {
        let session = eager.admit(name, Arc::clone(tokens));
        let observations = (0..4)
            .map(|index| eager.execute(&session, Arc::clone(tokens), window(index)))
            .collect::<Vec<_>>();
        session.try_complete().unwrap();
        observations
    };
    let expected_a = baseline("q8-eager-a", &tokens_a);
    let expected_b = baseline("q8-eager-b", &tokens_b);
    expected_a[0].assert_different_output(&expected_b[0]);

    let session_a = replay.admit("q8-replay-a", Arc::clone(&tokens_a));
    // First execution warms native libraries; the second captures the actual
    // provider command. Explicit replay below must find the published program
    // and its typed binding nodes: the helper rejects an eager fallback.
    for index in 0..2 {
        expected_a[index].assert_same(
            &replay.execute(&session_a, Arc::clone(&tokens_a), window(index)),
            "Q8 provider warm/capture",
        );
    }
    expected_a[2].assert_same(
        &replay.execute_replayed(&session_a, Arc::clone(&tokens_a), window(2)),
        "Q8 provider changed-token replay",
    );
    let session_b = replay.admit("q8-replay-b", Arc::clone(&tokens_b));
    for index in 0..3 {
        let actual = replay.execute_replayed(&session_b, Arc::clone(&tokens_b), window(index));
        actual.assert_state_nonzero();
        expected_b[index].assert_same(&actual, "Q8 provider rebound sequence");
    }
    // A remains live while B overwrites the shared invocation scratch. Returning
    // to A must preserve its separate F16 convolution and F32 delta state.
    expected_a[3].assert_same(
        &replay.execute_replayed(&session_a, Arc::clone(&tokens_a), window(3)),
        "Q8 provider parked peer state",
    );
    expected_b[3].assert_same(
        &replay.execute_replayed(&session_b, Arc::clone(&tokens_b), window(3)),
        "Q8 provider second peer state",
    );
    session_a.try_complete().unwrap();
    session_b.try_complete().unwrap();
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
