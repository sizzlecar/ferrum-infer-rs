//! Metadata reuse and owner lifetime checks. Actual admitted tensor execution
//! and foreign-runtime rejection use the existing Reference component tests.
use super::*;
use crate::component::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[allow(dead_code)]
#[path = "../../../tests/capacity_boundary/fixture.rs"]
mod resource_fixture;
use resource_fixture as f;

#[allow(dead_code)]
#[path = "../../component/tests/fixture.rs"]
mod fixture;

fn compile(
    config: &fixture::Config,
    registry: &OperationRuntimeRegistry<f::TestRuntime>,
    catalog: &CapabilityCatalog,
) -> CompiledTensorComponent<f::TestRuntime> {
    CompiledTensorComponent::compile(
        fixture::family(config),
        registry,
        catalog,
        &fixture::policy(),
        &WeightMaterializerRegistry::identity_only().unwrap(),
        &fixture::options(),
    )
    .unwrap()
}

#[test]
fn same_lane_reuses_metadata_but_new_plan_and_lane_compile_their_own_identity() {
    let config = fixture::Config {
        stored: false,
        persistent: false,
    };
    let (registry, catalog) = fixture::registry(&config);
    let first = compile(&config, &registry, &catalog);
    let second = compile(
        &fixture::Config {
            stored: false,
            persistent: true,
        },
        &registry,
        &catalog,
    );
    let runtime = f::runtime(first.executable().execution_plan());
    let lane = ExecutionLane::create(runtime.clone()).unwrap();
    let mut cache = TensorWaveIdentityCache::default();
    let fingerprint = cache
        .get_or_compile(first.executable(), &lane)
        .unwrap()
        .fingerprint()
        .to_owned();
    assert_eq!(
        cache
            .get_or_compile(first.executable(), &lane)
            .unwrap()
            .fingerprint(),
        fingerprint
    );
    assert_eq!(
        cache.compilations, 1,
        "warm identity must avoid recompiling nodes"
    );
    assert_ne!(
        first.executable().execution_plan().plan_hash(),
        second.executable().execution_plan().plan_hash()
    );
    let changed = cache.get_or_compile(second.executable(), &lane).unwrap();
    assert_eq!(
        changed.plan_hash(),
        second.executable().execution_plan().plan_hash()
    );
    assert_ne!(changed.fingerprint(), fingerprint);
    assert_eq!(cache.compilations, 2);

    let new_lane = ExecutionLane::create(runtime).unwrap();
    let changed = cache
        .get_or_compile(second.executable(), &new_lane)
        .unwrap();
    assert_eq!(changed.lane_id(), new_lane.id());
    assert_ne!(changed.lane_id(), lane.id());
    assert_eq!(cache.compilations, 3);
}

#[test]
fn identical_foreign_runtime_descriptors_cannot_hit_a_cached_lane() {
    let compiled = fixture::compile(&fixture::Config {
        stored: false,
        persistent: false,
    });
    let first_runtime = f::runtime(compiled.executable().execution_plan());
    let other_runtime = f::runtime(compiled.executable().execution_plan());
    assert_eq!(first_runtime.descriptor(), other_runtime.descriptor());
    assert!(!Arc::ptr_eq(&first_runtime, &other_runtime));
    let first = ExecutionLane::create(first_runtime).unwrap();
    let other = ExecutionLane::create(other_runtime).unwrap();
    let mut cache = TensorWaveIdentityCache::default();
    let original = cache
        .get_or_compile(compiled.executable(), &first)
        .unwrap()
        .fingerprint()
        .to_owned();
    let independent = cache.get_or_compile(compiled.executable(), &other).unwrap();
    assert_ne!(independent.fingerprint(), original);
    assert_eq!(independent.lane_id(), other.id());
    assert_eq!(cache.compilations, 2);
    // This helper caches topology only. Ordinary admission and compiled bind
    // still reject a foreign runtime when attaching actual plan resources.
    let incompatible = fixture::compile(&fixture::Config {
        stored: false,
        persistent: false,
    });
    assert!(cache
        .get_or_compile(incompatible.executable(), &other)
        .is_err());
    assert_eq!(cache.compilations, 2);
    cache.get_or_compile(compiled.executable(), &other).unwrap();
    assert_eq!(cache.compilations, 2);
}

#[test]
fn cached_topology_does_not_keep_a_retired_lane_or_failed_actor_alive() {
    let compiled = fixture::compile(&fixture::Config {
        stored: false,
        persistent: false,
    });
    let runtime = f::runtime(compiled.executable().execution_plan());
    let mut cache = TensorWaveIdentityCache::default();
    let lane = ExecutionLane::create(runtime.clone()).unwrap();
    let weak = Arc::downgrade(&lane);
    cache.get_or_compile(compiled.executable(), &lane).unwrap();
    drop(lane);
    assert!(
        weak.upgrade().is_none(),
        "metadata must not retain a one-shot lane"
    );

    for unwind in [false, true] {
        let mut actor = TensorSerialActor::new(runtime.clone()).unwrap();
        let weak = actor
            .execute(|lane| {
                cache.get_or_compile(compiled.executable(), lane)?;
                Ok(Arc::downgrade(lane))
            })
            .unwrap();
        let compilations = cache.compilations;
        if unwind {
            assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = actor.execute::<()>(|_| panic!("injected actor unwind"));
            }))
            .is_err());
        } else {
            assert!(actor
                .execute::<()>(|_| Err(invalid("injected actor failure").into()))
                .is_err());
        }
        assert!(actor.is_failed());
        assert!(actor
            .execute::<()>(|lane| {
                cache.get_or_compile(compiled.executable(), lane)?;
                panic!("failed actor must reject before reaching its identity cache")
            })
            .is_err());
        assert_eq!(cache.compilations, compilations);
        drop(actor);
        assert!(
            weak.upgrade().is_none(),
            "metadata must not retain a failed actor lane"
        );
    }
}
