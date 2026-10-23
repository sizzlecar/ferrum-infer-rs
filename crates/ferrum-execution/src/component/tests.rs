//! Lifecycle tests use the existing metadata-only device fixture. They do not
//! pretend to validate tensor numerics or advertise managed IO on that device.
use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

#[allow(dead_code)]
#[path = "../../tests/capacity_boundary/fixture.rs"]
mod resource_fixture;
use resource_fixture as f;

mod fixture;
use fixture::*;

struct UnreadSource(AtomicUsize);

#[test]
fn shared_weight_publication_rejects_none_and_mutable_static_before_source_access() {
    let source = UnreadSource(AtomicUsize::new(0));
    for (stored, persistent) in [(false, false), (false, true), (true, true)] {
        let compiled = compile(&Config { stored, persistent });
        let runtime = f::runtime(compiled.executable().execution_plan());
        let weak = Arc::downgrade(&runtime);
        let error = compiled
            .initialize_shared_weights(
                runtime,
                host(),
                if stored {
                    ComponentWeightSource::Stored(&source)
                } else {
                    ComponentWeightSource::None
                },
                init_config(),
            )
            .err()
            .unwrap();
        assert!(matches!(error, ComponentInitializationError::Contract(_)));
        assert!(
            weak.upgrade().is_none(),
            "rejected sharing must retain no new runtime owner"
        );
    }
    assert_eq!(source.0.load(Ordering::Relaxed), 0);
}
impl WeightComponentSource for UnreadSource {
    fn component<'a>(
        &'a self,
        _: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        self.0.fetch_add(1, Ordering::Relaxed);
        Err(invalid(
            "test source must not be read before capability/source validation",
        ))
    }
}

#[test]
fn component_policy_is_fixed_tensor_and_validates_its_finite_capacity() {
    let storage = f::contiguous_storage_profile();
    assert!(TensorComponentPolicy::new(0, vec![storage]).is_err());
    assert!(TensorComponentPolicy::new(4096, vec![]).is_err());
    assert!(TensorComponentPolicy::new(4096, vec![storage, storage]).is_err());
    let policy = TensorComponentPolicy::new(4096, vec![storage]).unwrap();
    assert_eq!(policy.work_limits(), RuntimeWorkLimits::FixedTensor);
    assert_eq!(policy.maximum_active_sequences(), 1);
    assert_eq!(policy.memory_reserve_bytes(), 0);
    assert!(policy.reusable_execution_policy().is_none());
    let wire = serde_json::to_value(&policy).unwrap();
    let restored: TensorComponentPolicy = serde_json::from_value(wire.clone()).unwrap();
    assert_eq!(restored, policy);
    let mut bad = wire;
    bad["capacity_bytes"] = serde_json::json!(0);
    assert!(serde_json::from_value::<TensorComponentPolicy>(bad.clone()).is_err());
    bad["capacity_bytes"] = serde_json::json!(4096);
    bad["maximum_scheduled_tokens"] = serde_json::json!(1);
    assert!(serde_json::from_value::<TensorComponentPolicy>(bad).is_err());
}

#[test]
fn component_compile_rejects_language_and_explicit_materializer_for_none() {
    let config = Config {
        stored: false,
        persistent: false,
    };
    let (registry, catalog) = registry(&config);
    let language = TypedFamilyRegistration::new(f::TestFamily)
        .prepare_with_profile(
            &serde_json::json!({"width":4}),
            &NumericalProfileId::new("fixture.f32").unwrap(),
        )
        .unwrap();
    let materializers = WeightMaterializerRegistry::identity_only().unwrap();
    assert!(CompiledTensorComponent::compile(
        language,
        &registry,
        &catalog,
        &policy(),
        &materializers,
        &options()
    )
    .is_err());
    let mut explicit = options();
    explicit.require_weight_materializer(f::id(IDENTITY_WEIGHT_MATERIALIZER_ID));
    assert!(CompiledTensorComponent::compile(
        family(&config),
        &registry,
        &catalog,
        &policy(),
        &materializers,
        &explicit
    )
    .is_err());
}

#[test]
fn source_kind_mismatch_is_rejected_before_source_access_or_provisioning() {
    let source = UnreadSource(AtomicUsize::new(0));
    for stored in [false, true] {
        let compiled = compile(&Config {
            stored,
            persistent: false,
        });
        let runtime = f::runtime(compiled.executable().execution_plan());
        let weak = Arc::downgrade(&runtime);
        let source_mode = if stored {
            ComponentWeightSource::None
        } else {
            ComponentWeightSource::Stored(&source)
        };
        let error = compiled
            .initialize(runtime, host(), source_mode, init_config())
            .err()
            .unwrap();
        assert!(matches!(error, ComponentInitializationError::Contract(_)));
        assert!(
            weak.upgrade().is_none(),
            "preflight failure must not strand a root"
        );
    }
    assert_eq!(source.0.load(Ordering::Relaxed), 0);
}

#[test]
fn no_static_constructor_failure_preserves_root_until_explicit_close() {
    let compiled = compile(&Config {
        stored: false,
        persistent: false,
    });
    assert!(compiled
        .executable()
        .execution_plan()
        .payload()
        .memory()
        .static_allocations()
        .is_empty());
    let runtime = f::runtime(compiled.executable().execution_plan());
    let weak = Arc::downgrade(&runtime);
    let error = compiled
        .initialize(runtime, host(), ComponentWeightSource::None, init_config())
        .err()
        .unwrap();
    // The existing fixture intentionally declines managed IO; this tests the
    // actual executor-construction failure after successful NoStatic handoff.
    let ComponentInitializationError::Executor {
        source,
        resources,
        receipt,
    } = error
    else {
        panic!("expected typed executor error")
    };
    assert!(matches!(source, TensorExecutionError::Runtime(_)));
    assert!(matches!(receipt, ComponentInitializationReceipt::NoStatic));
    assert!(weak.upgrade().is_some());
    let alias = resources.clone();
    let resources =
        match PlanRuntimeResources::close(resources).unwrap_or_else(|_| panic!("close failed")) {
            PlanRuntimeCloseOutcome::Referenced { resources, .. } => resources,
            PlanRuntimeCloseOutcome::Closed(_) => {
                panic!("an actual root alias must prevent final close")
            }
        };
    assert!(resources.is_closing());
    drop(alias);
    assert_eq!(close(resources), 0);
    assert!(weak.upgrade().is_none());
}

#[test]
fn weightless_persistent_resources_are_initialized_not_treated_as_no_static() {
    let compiled = compile(&Config {
        stored: false,
        persistent: true,
    });
    assert!(!compiled
        .executable()
        .execution_plan()
        .payload()
        .memory()
        .static_allocations()
        .is_empty());
    let runtime = f::runtime(compiled.executable().execution_plan());
    let weak = Arc::downgrade(&runtime);
    let error = compiled
        .initialize(runtime, host(), ComponentWeightSource::None, init_config())
        .err()
        .unwrap();
    let ComponentInitializationError::Executor {
        resources, receipt, ..
    } = error
    else {
        panic!("expected initialized root before IO refusal")
    };
    let ComponentInitializationReceipt::Initialized { receipt } = receipt else {
        panic!("persistent memory must have an initialization receipt")
    };
    assert_eq!(receipt.initialized_resource_count(), 1);
    assert_eq!(receipt.uploaded_bytes(), 0);
    assert!(receipt.submission_batch_count() > 0);
    assert_eq!(close(resources), 1);
    assert!(weak.upgrade().is_none());
}

#[test]
fn managed_static_failure_retains_original_recoverable_transaction() {
    let compiled = compile(&Config {
        stored: true,
        persistent: false,
    });
    let runtime = f::runtime(compiled.executable().execution_plan());
    let weak = Arc::downgrade(&runtime);
    let source = UnreadSource(AtomicUsize::new(0));
    let error = compiled
        .initialize(
            runtime,
            host(),
            ComponentWeightSource::Stored(&source),
            init_config(),
        )
        .err()
        .unwrap();
    let ComponentInitializationError::Static(mut failure) = error else {
        panic!("managed upload support must fail with the static owner")
    };
    assert!(!failure.is_indeterminate());
    assert!(!failure.is_retired());
    assert!(weak.upgrade().is_some());
    assert_eq!(source.0.load(Ordering::Relaxed), 0);
    failure.retire_in_place().unwrap();
    assert!(failure.is_retired());
    assert!(weak.upgrade().is_none());
}

#[test]
fn repeated_component_initialization_admits_independent_roots() {
    let compiled = compile(&Config {
        stored: false,
        persistent: true,
    });
    let runtime = f::runtime(compiled.executable().execution_plan());
    let mut roots = Vec::new();
    for index in 0..2 {
        let mut config = init_config();
        config.transaction_id = f::id(&format!("transaction.component.repeat.{index}"));
        config.request_id = f::id(&format!("request.component.repeat.{index}"));
        config.run_id = f::id(&format!("run.component.repeat.{index}"));
        let failure = compiled
            .initialize(runtime.clone(), host(), ComponentWeightSource::None, config)
            .err()
            .unwrap();
        let ComponentInitializationError::Executor {
            resources, receipt, ..
        } = failure
        else {
            panic!("fixture must reach managed IO refusal after its own static initialization");
        };
        assert!(matches!(
            receipt,
            ComponentInitializationReceipt::Initialized { .. }
        ));
        roots.push(resources);
    }
    assert!(!Arc::ptr_eq(&roots[0], &roots[1]));
    let second = roots.pop().unwrap();
    assert_eq!(close(roots.pop().unwrap()), 1);
    assert!(!second.is_closing());
    assert_eq!(close(second), 1);
}

fn close(resources: Arc<PlanRuntimeResources<f::TestRuntime>>) -> usize {
    match PlanRuntimeResources::close(resources).unwrap_or_else(|_| panic!("close failed")) {
        PlanRuntimeCloseOutcome::Closed(receipt) => receipt.released_static_resources(),
        PlanRuntimeCloseOutcome::Referenced { .. } => panic!("unexpected live root owner"),
    }
}

fn host() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            64,
            HostMemoryClassLimits {
                input_preprocess_bytes: 16,
                output_bytes: 16,
                transfer_staging_bytes: 32,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
    )
}
fn init_config() -> ComponentInitializationConfig {
    ComponentInitializationConfig {
        request_id: f::id("request.component.test"),
        run_id: f::id("run.component.test"),
        transaction_id: f::id("transaction.component.test"),
        static_policy: StaticInitializationPolicy::new(32, 8).unwrap(),
        executor: TensorExecutorConfig::default(),
    }
}
