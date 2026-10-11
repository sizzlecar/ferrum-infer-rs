#![cfg(feature = "cuda")]
//! Real provider gate for the explicit attention-Q8 policy, never a checkpoint
//! qualification substitute or an Auto-policy mutation.

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
mod legacy_family;
use legacy_family::{AttentionKind, HIDDEN, MAX_TOKENS};
#[path = "vnext_cuda_q8act_attention/family.rs"]
mod family;
use family::Family;
#[path = "vnext_cuda_q8act_attention/checks.rs"]
mod checks;
#[path = "vnext_cuda_q8act_attention/runtime.rs"]
mod runtime;

type Runtime = CudaDeviceRuntime;

fn composition(kind: AttentionKind, family: &PreparedModelFamily) -> runtime::Composition {
    let (runtime, registry, materializers, catalog) = CudaVNextComposition::create(
        0,
        id(format!("device.cuda.q8act-attention.{kind:?}")),
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
#[ignore = "requires exclusive CUDA and real attention providers"]
fn q8act_gated_delta_mixed_leaves_preserve_state_across_eager_and_replay() {
    checks::verify(AttentionKind::GatedDelta);
}

#[test]
#[ignore = "requires exclusive CUDA and real FP16-KV causal providers"]
fn q8act_causal_fp16_kv_mixed_leaves_preserve_valid_positions_across_eager_and_replay() {
    checks::verify(AttentionKind::Causal);
}
