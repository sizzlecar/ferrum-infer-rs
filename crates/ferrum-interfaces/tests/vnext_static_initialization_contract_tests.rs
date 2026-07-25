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

struct StrictSourceWeightSource {
    requested: Arc<Mutex<Vec<WeightId>>>,
}

impl WeightComponentSource for StrictSourceWeightSource {
    fn component<'source>(
        &'source self,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'source>, VNextError> {
        if !matches!(
            component.id.as_str(),
            "weight.component.left" | "weight.component.right"
        ) {
            return Err(VNextError::InvalidExecutionPlan {
                reason: format!(
                    "strict source rejects derived execution component `{}`",
                    component.id
                ),
            });
        }
        self.requested.lock().unwrap().push(component.id.clone());
        ZeroWeightSource.component(component)
    }
}

struct DerivedComponentMaterializer {
    descriptor: WeightMaterializerDescriptor,
}

impl DerivedComponentMaterializer {
    fn new() -> Self {
        Self {
            descriptor: WeightMaterializerDescriptor::new(
                id("weight-materializer.test.derived-components"),
                ContractVersion::new(1, 0),
                sha('8'),
                BTreeSet::from([id("capability.compute")]),
            )
            .unwrap(),
        }
    }
}

impl WeightMaterializer for DerivedComponentMaterializer {
    fn descriptor(&self) -> &WeightMaterializerDescriptor {
        &self.descriptor
    }

    fn execution_schema(
        &self,
        family: &PreparedModelFamily,
        _device: &DeviceDescriptor,
    ) -> Result<WeightSchema, VNextError> {
        let mut schema = family.weight_schema().clone();
        schema.layout_id = id("weight-layout.device-operation-derived");
        schema.components[0].id = id("weight.execution.left");
        schema.components[0].external_names = vec!["derived.left.bin".to_owned()];
        schema.components[1].id = id("weight.execution.right");
        schema.components[1].external_names = vec!["derived.right.bin".to_owned()];
        let PhysicalWeightLayout::Composite { parts } = &mut schema.tensors[0].physical_layout
        else {
            panic!("fixture weight must remain composite");
        };
        let PhysicalWeightLayout::Dense { component_id } = parts[0].layout.as_mut() else {
            panic!("fixture left weight must remain dense");
        };
        *component_id = id("weight.execution.left");
        let PhysicalWeightLayout::Dense { component_id } = parts[1].layout.as_mut() else {
            panic!("fixture right weight must remain dense");
        };
        *component_id = id("weight.execution.right");
        Ok(schema)
    }

    fn component_sources(
        &self,
        _family: &PreparedModelFamily,
        _execution_schema: &WeightSchema,
    ) -> Result<BTreeMap<WeightId, Vec<WeightId>>, VNextError> {
        Ok(BTreeMap::from([
            (
                id("weight.execution.left"),
                vec![id("weight.component.left")],
            ),
            (
                id("weight.execution.right"),
                vec![id("weight.component.right")],
            ),
        ]))
    }

    fn materialize_component<'source>(
        &self,
        source: &'source dyn WeightComponentSource,
        source_components: &[&WeightComponentSpec],
        execution_component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'source>, VNextError> {
        let [source_component] = source_components else {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "derived test materializer requires one source component".to_owned(),
            });
        };
        let source_payload = source.component(source_component)?;
        let fill = match execution_component.id.as_str() {
            "weight.execution.left" => 0xa1,
            "weight.execution.right" => 0xb2,
            other => {
                return Err(VNextError::InvalidExecutionPlan {
                    reason: format!("unexpected derived execution component `{other}`"),
                })
            }
        };
        let byte_len = usize::try_from(execution_component.physical_bytes()?).map_err(|_| {
            VNextError::InvalidExecutionPlan {
                reason: "derived test component exceeds host address space".to_owned(),
            }
        })?;
        WeightComponentPayload::from_ordered_sources(
            execution_component,
            execution_component.external_names.clone(),
            source_payload.source_files().to_vec(),
            execution_component.dimensions.clone(),
            execution_component.physical_element_type(),
            vec![fill; byte_len],
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

fn derived_component_test_plan() -> (
    PreparedModelFamily,
    ExecutionPlan,
    Arc<TestRuntime>,
    Arc<Mutex<RuntimeTrace>>,
) {
    let materializers =
        WeightMaterializerRegistry::new(vec![Box::new(DerivedComponentMaterializer::new())])
            .unwrap();
    let catalog = materializers.augment_catalog(catalog()).unwrap();
    let registry = operation_registry(
        &catalog,
        Arc::new(Mutex::new(ProviderBehavior::Success)),
        Arc::new(Mutex::new(ProviderTrace::default())),
    );
    let family = TestModelRegistry::new()
        .registration
        .prepare(&json!({"width": 4}))
        .unwrap();
    let runtime_policy = policy();
    let materializer_id = id("weight-materializer.test.derived-components");
    let mut options = ProgramPlanCompileOptions::new(BTreeMap::from([(
        id("value.input"),
        ProgramTensorSpec {
            dimensions: vec![4],
            element_type: ElementType::F32,
            layout: ResolvedTensorLayout::Contiguous,
        },
    )]))
    .unwrap();
    options.require_weight_materializer(materializer_id);
    let plan = ProgramPlanCompiler::compile_with_weight_materializers(
        &family,
        &catalog,
        &runtime_policy,
        &registry.planning(),
        &materializers,
        &options,
    )
    .unwrap()
    .executable()
    .execution_plan()
    .clone();
    let (runtime, trace) = runtime(&catalog);
    (family, plan, runtime, trace)
}

fn close(resources: Arc<PlanRuntimeResources<TestRuntime>>) {
    close_with_expected_static_resources(resources, 2);
}

fn close_with_expected_static_resources(
    resources: Arc<PlanRuntimeResources<TestRuntime>>,
    expected_static_resources: usize,
) {
    match PlanRuntimeResources::close(resources) {
        Ok(PlanRuntimeCloseOutcome::Closed(receipt)) => assert_eq!(
            receipt.released_static_resources(),
            expected_static_resources
        ),
        Ok(PlanRuntimeCloseOutcome::Referenced { strong_count, .. }) => {
            panic!("plan runtime close retained {strong_count} references")
        }
        Err(failure) => panic!("plan runtime close failed: {:?}", failure.failure()),
    }
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
    let execution_weight_schema = plan.payload().execution_weights().schema();
    assert_eq!(execution_weight_schema, family.weight_schema());
    let expected_uploaded_bytes = execution_weight_schema
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
        execution_weight_schema.components.len()
    );
    assert_eq!(receipt.uploaded_bytes(), expected_uploaded_bytes);
    assert_eq!(receipt.upload_command_count(), 2);
    assert_eq!(receipt.imported_component_count(), 0);
    assert_eq!(receipt.device_import_duration_us(), 0);
    assert_eq!(receipt.import_seal_duration_us(), 0);
    assert!(receipt.total_duration_us() >= receipt.source_materialization_duration_us());
    assert!(receipt.total_duration_us() >= receipt.device_encode_duration_us());
    assert!(receipt.total_duration_us() >= receipt.submission_wait_duration_us());
    assert!(receipt.slowest_component_id().is_some());
    assert!(
        receipt.source_materialization_duration_us()
            >= receipt.slowest_component_materialization_duration_us()
    );
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
fn static_initialization_materializes_derived_components_through_trusted_plan_authority() {
    let (family, plan, runtime, trace) = derived_component_test_plan();
    assert_eq!(
        plan.payload().execution_weights().component_sources(),
        &BTreeMap::from([
            (
                id("weight.execution.left"),
                vec![id("weight.component.left")],
            ),
            (
                id("weight.execution.right"),
                vec![id("weight.component.right")],
            ),
        ])
    );
    let requested = Arc::new(Mutex::new(Vec::new()));
    let source = StrictSourceWeightSource {
        requested: Arc::clone(&requested),
    };
    let initialized = committed_transaction(&plan, Arc::clone(&runtime), "derived-materializer")
        .initialize_static(
            &family,
            &plan,
            &source,
            StaticInitializationPolicy::new(64, 8).unwrap(),
        )
        .unwrap();
    assert_eq!(
        *requested.lock().unwrap(),
        vec![id("weight.component.left"), id("weight.component.right")]
    );
    assert_eq!(
        trace.lock().unwrap().uploaded_payloads,
        vec![vec![0xa1; 8], vec![0xb2; 8]]
    );
    let expected_static_resources = plan.payload().memory().static_allocations().len();
    close_with_expected_static_resources(handoff(initialized), expected_static_resources);
}

#[test]
fn static_initialization_seals_backend_weight_import_without_upload_commands() {
    let (resolved, plan, runtime, trace) = test_plan();
    trace.lock().unwrap().static_weight_import_enabled = true;
    let family = &resolved.parts().prepared_family;
    let execution_weight_schema = plan.payload().execution_weights().schema();
    let expected_imported_bytes = execution_weight_schema
        .components
        .iter()
        .map(|component| component.physical_bytes().unwrap())
        .try_fold(0_u64, u64::checked_add)
        .unwrap();

    let initialized = committed_transaction(&plan, Arc::clone(&runtime), "import")
        .initialize_static(
            family,
            &plan,
            &ZeroWeightSource,
            StaticInitializationPolicy::new(8, 2).unwrap(),
        )
        .unwrap();
    let receipt = initialized.receipt();
    assert_eq!(receipt.uploaded_component_count(), 0);
    assert_eq!(receipt.uploaded_bytes(), 0);
    assert_eq!(receipt.upload_command_count(), 0);
    assert_eq!(
        receipt.imported_component_count(),
        execution_weight_schema.components.len()
    );
    assert_eq!(receipt.imported_bytes(), expected_imported_bytes);
    assert_eq!(receipt.submission_batch_count(), 0);
    assert_eq!(receipt.submission_wait_duration_us(), 0);
    assert!(receipt.slowest_component_id().is_some());
    assert!(receipt.total_duration_us() >= receipt.device_import_duration_us());
    assert!(receipt.total_duration_us() >= receipt.import_seal_duration_us());
    {
        let trace = trace.lock().unwrap();
        assert_eq!(trace.static_weight_import_begin_calls, 1);
        assert_eq!(trace.static_weight_import_seal_calls, 1);
        assert_eq!(
            trace.imported_component_count,
            execution_weight_schema.components.len()
        );
        assert_eq!(trace.imported_bytes, expected_imported_bytes);
        assert_eq!(trace.submit_calls, 0);
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
