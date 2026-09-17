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
