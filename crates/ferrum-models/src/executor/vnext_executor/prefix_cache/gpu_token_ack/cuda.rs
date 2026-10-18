use super::*;
use ferrum_kernels::backend::cuda::{
    vnext_ops::{cuda_weight_materializer_selection, CudaVNextComposition},
    vnext_runtime::CudaDeviceRuntime,
};
use ferrum_runtime::*;
use half::f16;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::result::Result;

#[allow(
    dead_code,
    reason = "Shared numerical fixture also serves broader backend contract tests"
)]
#[path = "../../../../../tests/vnext_checkpoint_continuation/family.rs"]
mod family;
#[allow(
    dead_code,
    reason = "Shared numerical fixture also serves broader backend contract tests"
)]
#[path = "../../../../../tests/vnext_checkpoint_continuation/runtime.rs"]
mod runtime;
use family::{AttentionKind, Family, HIDDEN, MAX_TOKENS};
use runtime::Fixture;

type Runtime = CudaDeviceRuntime;

fn composition(kind: AttentionKind, family: &PreparedModelFamily) -> runtime::Composition {
    let (runtime, registry, materializers, catalog) = CudaVNextComposition::create(
        0,
        id(format!("device.cuda.prefix-token-ack.{kind:?}")),
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

fn device() -> Device {
    Device::CUDA(0)
}

#[path = "checks.rs"]
mod checks;
