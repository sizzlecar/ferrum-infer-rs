use ferrum_kernels::backend::metal::{
    vnext_ops::MetalVNextComposition, vnext_runtime::MetalDeviceRuntime,
};
use ferrum_runtime::*;
use half::f16;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::result::Result;
use std::sync::Arc;

#[allow(
    dead_code,
    reason = "Shared numerical fixture also serves broader backend contract tests"
)]
#[path = "../../../../tests/vnext_checkpoint_continuation/family.rs"]
mod family;
#[allow(
    dead_code,
    reason = "Shared numerical fixture also serves broader backend contract tests"
)]
#[path = "../../../../tests/vnext_checkpoint_continuation/runtime.rs"]
mod runtime;
use family::{AttentionKind, Family, HIDDEN, MAX_TOKENS};
use runtime::Fixture;

type Runtime = MetalDeviceRuntime;

fn composition(kind: AttentionKind, _family: &PreparedModelFamily) -> runtime::Composition {
    let (runtime, registry, materializers, materializer, catalog) =
        MetalVNextComposition::create(id(format!("device.metal.prefix-token-ack.{kind:?}")))
            .unwrap()
            .into_parts();
    (
        runtime,
        registry,
        materializers,
        WeightMaterializerSelection::exact(materializer),
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

#[path = "checks.rs"]
mod checks;
