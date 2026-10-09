//! MarkerV2 graph fixture. CPU schema tests do not qualify the native artifact;
//! ignored CUDA gates use actual providers, admitted flags and real state.

use ferrum_interfaces::vnext::*;
use half::f16;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
#[cfg(feature = "cuda")]
use std::sync::Arc;

#[path = "vnext_checkpoint_continuation/family.rs"]
mod legacy_family;
use legacy_family::{AttentionKind, HIDDEN, MAX_TOKENS};
#[path = "vnext_cuda_q8act_attention/family.rs"]
mod attention_family;
#[path = "vnext_cuda_upstream_marker/family.rs"]
mod family;
use family::Family;

#[cfg(feature = "cuda")]
#[path = "vnext_cuda_upstream_marker/checks.rs"]
mod checks;
#[cfg(feature = "cuda")]
#[path = "vnext_cuda_upstream_marker/runtime.rs"]
mod runtime;
#[cfg(feature = "cuda")]
use ferrum_kernels::backend::cuda::{
    vnext_ops::{cuda_weight_materializer_selection, CudaVNextComposition},
    vnext_runtime::CudaDeviceRuntime,
};
#[cfg(feature = "cuda")]
type Runtime = CudaDeviceRuntime;

#[cfg(feature = "cuda")]
fn composition(kind: AttentionKind, family: &PreparedModelFamily) -> runtime::Composition {
    let (runtime, registry, materializers, catalog) = CudaVNextComposition::create(
        0,
        id(format!("device.cuda.upstream-marker.{kind:?}")),
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

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires exclusive CUDA and the actual MarkerV2 native artifact"]
fn upstream_marker_gated_delta_and_swiglu_preserve_state_across_eager_and_replay() {
    checks::verify(AttentionKind::GatedDelta);
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires exclusive CUDA, MarkerV2 artifact and real FP16-KV providers"]
fn upstream_marker_causal_and_swiglu_preserve_valid_kv_across_eager_and_replay() {
    checks::verify(AttentionKind::Causal);
}
