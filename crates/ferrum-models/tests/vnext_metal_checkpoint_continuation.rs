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

#[path = "vnext_metal_checkpoint_continuation/family.rs"]
mod family;
#[path = "vnext_metal_checkpoint_continuation/runtime.rs"]
mod runtime;

use family::{AttentionKind, Family, HIDDEN, MAX_TOKENS};
use runtime::Fixture;

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
fn gated_delta_f32_master_q4k_provider_resumes_public_native_checkpoint() {
    verify(AttentionKind::GatedDelta, &[0..2, 2..5], &[5..6, 6..9]);
    verify(AttentionKind::GatedDelta, &[0..5], &[5..6, 6..9]);
}

fn verify(
    kind: AttentionKind,
    prefix: &[std::ops::Range<usize>],
    suffix: &[std::ops::Range<usize>],
) {
    let fixture = Fixture::new(kind);
    let total = suffix.last().unwrap().end;
    let tokens: Arc<[u32]> = (0..total)
        .map(|index| ((index * 7 + 3) % 32) as u32)
        .collect();
    let source = fixture.admit("source", Arc::clone(&tokens));
    let cold = fixture.admit("cold", Arc::clone(&tokens));
    let restored = fixture.admit("restored", Arc::clone(&tokens));
    eprintln!("{kind:?}: F32-master hidden={HIDDEN}, native Q4_K projections, single participant, prefix={prefix:?}, suffix={suffix:?}; real embedding+attention FullPlan, native public capture/restore");
    let mut previous = None;
    for span in prefix {
        let expected = fixture.execute(&source, Arc::clone(&tokens), span.clone());
        let actual = fixture.execute(&cold, Arc::clone(&tokens), span.clone());
        expected.assert_same(&actual, "cold prefix");
        expected.assert_state_nonzero();
        previous = Some(expected);
    }
    let checkpoint = fixture.capture(&source);
    assert_eq!(checkpoint.completed_tokens(), prefix.last().unwrap().end);
    assert_eq!(
        checkpoint.token_prefix(),
        &tokens[..checkpoint.completed_tokens()]
    );
    let expected = suffix
        .iter()
        .map(|span| fixture.execute(&source, Arc::clone(&tokens), span.clone()))
        .collect::<Vec<_>>();
    expected
        .last()
        .unwrap()
        .assert_state_changed(previous.as_ref().unwrap(), "suffix");
    source.try_complete().unwrap();
    drop(source);
    // The source has overwritten its boundary values/appended KV, then ended.
    // The expected suffix is a host snapshot; only the independently owned
    // checkpoint can initialize this fresh target after its source is gone.
    fixture.restore(&restored, &checkpoint, Arc::clone(&tokens));
    for (span, expected) in suffix.iter().zip(expected) {
        expected.assert_same(
            &fixture.execute(&cold, Arc::clone(&tokens), span.clone()),
            "cold suffix",
        );
        expected.assert_same(
            &fixture.execute(&restored, Arc::clone(&tokens), span.clone()),
            "restored suffix",
        );
    }
    drop(checkpoint);
    cold.try_complete().unwrap();
    restored.try_complete().unwrap();
    drop((cold, restored));
    // Change a real token at the same prefix frontier: nonzero constant state
    // would otherwise make a continuation comparison vacuous.
    let mut changed = tokens.to_vec();
    let index = prefix.last().unwrap().end - 1;
    changed[index] = (changed[index] + 1) % 32;
    let changed: Arc<[u32]> = changed.into();
    let control = fixture.admit("changed-prefix", Arc::clone(&changed));
    let mut changed_observation = None;
    for span in prefix {
        changed_observation = Some(fixture.execute(&control, Arc::clone(&changed), span.clone()));
    }
    changed_observation
        .unwrap()
        .assert_state_changed(previous.as_ref().unwrap(), "changed prefix");
    control.try_complete().unwrap();
}
