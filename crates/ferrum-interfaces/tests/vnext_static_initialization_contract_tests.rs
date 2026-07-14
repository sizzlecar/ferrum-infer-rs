mod vnext_device_operation_contract;

use vnext_device_operation_contract::*;

struct ZeroWeightSource;

impl WeightComponentSource for ZeroWeightSource {
    fn component<'source>(
        &'source self,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'source>, VNextError> {
        let byte_len = usize::try_from(component.physical_bytes()?).map_err(|_| {
            VNextError::InvalidExecutionPlan {
                reason: format!(
                    "test weight component `{}` exceeds the host address space",
                    component.id
                ),
            }
        })?;
        WeightComponentPayload::new(
            component,
            component
                .external_names
                .first()
                .expect("test component has an external name")
                .clone(),
            "model.safetensors",
            component.dimensions.clone(),
            component.physical_element_type(),
            vec![0_u8; byte_len],
        )
    }
}

fn committed_transaction(
    plan: &ExecutionPlan,
    runtime: Arc<TestRuntime>,
    suffix: &str,
) -> ResourceTransaction<TestDriver, TransactionCommitted> {
    let ProvisionedPlanParts { provisioning } = plan
        .provision_static(
            Arc::clone(&runtime),
            id(format!("request.static-initialization.{suffix}")),
        )
        .unwrap()
        .into_parts();
    let admission = match provisioning {
        StaticProvisioning::Required(admission) => admission,
        StaticProvisioning::NoStatic(_) => {
            panic!("static initialization fixture requires plan-static resources")
        }
    };
    let identity = ResourceTransactionIdentity::for_admission(
        admission.binding(),
        id(format!("run.static-initialization.{suffix}")),
        id(format!("transaction.static-initialization.{suffix}")),
    );
    ResourceTransaction::begin(
        TestDriver {
            runtime,
            trace: Arc::new(Mutex::new(DriverTrace::default())),
        },
        identity,
        admission,
    )
    .unwrap()
    .reserve()
    .unwrap()
    .commit()
    .unwrap()
}

fn test_plan() -> (
    ResolvedModelPlan,
    ExecutionPlan,
    Arc<TestRuntime>,
    Arc<Mutex<RuntimeTrace>>,
) {
    let catalog = catalog();
    let registry = operation_registry(
        &catalog,
        Arc::new(Mutex::new(ProviderBehavior::Success)),
        Arc::new(Mutex::new(ProviderTrace::default())),
    );
    let (resolved, plan) = resolved_model_plan(&registry);
    let (runtime, trace) = runtime(&catalog);
    (resolved, plan, runtime, trace)
}

fn close(resources: Arc<PlanRuntimeResources<TestRuntime>>) {
    let mut passed = 0;
    close_plan_runtime(resources, &mut passed);
    assert_eq!(passed, 1);
}

fn handoff(
    initialized: InitializedResourceTransaction<TestDriver>,
) -> Arc<PlanRuntimeResources<TestRuntime>> {
    match initialized.into_plan_runtime() {
        Ok(resources) => resources,
        Err(failure) => panic!("plan runtime handoff failed: {}", failure.error()),
    }
}

#[test]
fn static_initialization_uploads_schema_components_before_handoff() {
    let (resolved, plan, runtime, trace) = test_plan();
    let family = &resolved.parts().prepared_family;
    let expected_uploaded_bytes = family
        .weight_schema()
        .components
        .iter()
        .map(WeightComponentSpec::physical_bytes)
        .try_fold(0_u64, |total, bytes| {
            total
                .checked_add(bytes?)
                .ok_or(VNextError::InvalidExecutionPlan {
                    reason: "test schema byte length overflow".to_owned(),
                })
        })
        .unwrap();
    let initialized = committed_transaction(&plan, Arc::clone(&runtime), "success")
        .initialize_static(
            family,
            &plan,
            &ZeroWeightSource,
            StaticInitializationPolicy::new(8, 2).unwrap(),
        )
        .unwrap();
    let receipt = initialized.receipt();
    assert_eq!(
        receipt.initialized_resource_count(),
        plan.payload().memory().static_allocations().len()
    );
    assert_eq!(
        receipt.uploaded_component_count(),
        family.weight_schema().components.len()
    );
    assert_eq!(receipt.uploaded_bytes(), expected_uploaded_bytes);
    assert_eq!(receipt.upload_command_count(), 2);
    assert_eq!(
        receipt.source_files(),
        &BTreeSet::from(["model.safetensors".to_owned()])
    );
    {
        let trace = trace.lock().unwrap();
        assert_eq!(
            trace.submit_calls as usize,
            receipt.submission_batch_count()
        );
        assert_eq!(trace.wait_fence_calls, trace.submit_calls);
        assert!(trace
            .submitted_command_counts
            .iter()
            .all(|count| *count <= 2));
        assert_eq!(trace.synchronize_calls, 0);
    }
    close(handoff(initialized));
}

#[test]
fn fence_wait_panic_requires_stream_recovery_before_retry() {
    let (resolved, plan, runtime, trace) = test_plan();
    trace.lock().unwrap().fence_behavior = FenceBehavior::Panic;
    let failed = suppress_expected_panic_hook(|| {
        committed_transaction(&plan, Arc::clone(&runtime), "recovery").initialize_static(
            &resolved.parts().prepared_family,
            &plan,
            &ZeroWeightSource,
            StaticInitializationPolicy::new(8, 2).unwrap(),
        )
    });
    let failure = match failed {
        Ok(_) => panic!("fence wait panic must not permit runtime handoff"),
        Err(failure) => failure,
    };
    assert!(failure.is_indeterminate());
    assert_eq!(failure.failure().code(), "static_fence_wait_panic");
    assert_eq!(trace.lock().unwrap().synchronize_calls, 0);

    trace.lock().unwrap().fence_behavior = FenceBehavior::Succeeded;
    let committed = match failure.recover() {
        Ok(transaction) => transaction,
        Err(failure) => panic!("stream recovery failed: {failure}"),
    };
    assert_eq!(trace.lock().unwrap().synchronize_calls, 1);
    let initialized = committed
        .initialize_static(
            &resolved.parts().prepared_family,
            &plan,
            &ZeroWeightSource,
            StaticInitializationPolicy::new(8, 2).unwrap(),
        )
        .unwrap();
    close(handoff(initialized));
}

#[test]
fn error_classifier_panic_preserves_indeterminate_recovery_ownership() {
    let (resolved, plan, runtime, trace) = test_plan();
    {
        let mut trace = trace.lock().unwrap();
        trace.fence_behavior = FenceBehavior::Indeterminate;
        trace.describe_error_panics = true;
    }
    let failed = suppress_expected_panic_hook(|| {
        committed_transaction(&plan, Arc::clone(&runtime), "classifier-panic").initialize_static(
            &resolved.parts().prepared_family,
            &plan,
            &ZeroWeightSource,
            StaticInitializationPolicy::new(8, 2).unwrap(),
        )
    });
    let failure = match failed {
        Ok(_) => panic!("an indeterminate fence must not permit runtime handoff"),
        Err(failure) => failure,
    };
    assert!(failure.is_indeterminate());
    assert_eq!(failure.failure().code(), "static_fence_indeterminate");
    assert!(failure
        .failure()
        .message()
        .contains("classification panicked"));

    {
        let mut trace = trace.lock().unwrap();
        trace.fence_behavior = FenceBehavior::Succeeded;
        trace.describe_error_panics = false;
    }
    let committed = match failure.recover() {
        Ok(transaction) => transaction,
        Err(failure) => panic!("stream recovery failed: {failure}"),
    };
    assert_eq!(trace.lock().unwrap().synchronize_calls, 1);
    let initialized = committed
        .initialize_static(
            &resolved.parts().prepared_family,
            &plan,
            &ZeroWeightSource,
            StaticInitializationPolicy::new(8, 2).unwrap(),
        )
        .unwrap();
    close(handoff(initialized));
}

#[test]
fn dropped_indeterminate_failure_defers_backend_cleanup_until_maintenance() {
    let (resolved, plan, runtime, trace) = test_plan();
    trace.lock().unwrap().fence_behavior = FenceBehavior::Indeterminate;
    let status_before = static_initialization_cleanup_status();
    let failed = committed_transaction(&plan, Arc::clone(&runtime), "deferred-drop")
        .initialize_static(
            &resolved.parts().prepared_family,
            &plan,
            &ZeroWeightSource,
            StaticInitializationPolicy::new(8, 2).unwrap(),
        );
    let failure = match failed {
        Ok(_) => panic!("an indeterminate fence must retain cleanup ownership"),
        Err(failure) => failure,
    };
    assert!(failure.is_indeterminate());
    drop(failure);

    assert_eq!(trace.lock().unwrap().synchronize_calls, 0);
    let status_deferred = static_initialization_cleanup_status();
    assert_eq!(
        status_deferred.submitted_total(),
        status_before.submitted_total() + 1
    );
    assert_eq!(status_deferred.pending(), status_before.pending() + 1);

    trace.lock().unwrap().fence_behavior = FenceBehavior::Succeeded;
    let maintenance = maintain_static_initialization_cleanups(1).unwrap();
    assert_eq!(maintenance.attempted(), 1);
    assert_eq!(maintenance.completed(), 1);
    assert_eq!(
        maintenance.status_after().pending(),
        status_before.pending()
    );
    assert_eq!(trace.lock().unwrap().synchronize_calls, 1);
}
