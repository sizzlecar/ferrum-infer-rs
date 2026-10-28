use super::*;

struct ManagedDerivedMaterializer {
    base: DerivedComponentMaterializer,
    parent: HostMemoryAccount,
    retained: Arc<Mutex<Vec<ManagedHostBytes>>>,
    queries: Arc<Mutex<Vec<bool>>>,
}

impl WeightMaterializer for ManagedDerivedMaterializer {
    fn descriptor(&self) -> &WeightMaterializerDescriptor {
        self.base.descriptor()
    }

    fn execution_schema(
        &self,
        family: &PreparedModelFamily,
        device: &DeviceDescriptor,
    ) -> Result<WeightSchema, VNextError> {
        self.base.execution_schema(family, device)
    }

    fn component_sources(
        &self,
        family: &PreparedModelFamily,
        schema: &WeightSchema,
    ) -> Result<BTreeMap<WeightId, Vec<WeightId>>, VNextError> {
        self.base.component_sources(family, schema)
    }

    fn additional_host_backing_bytes(
        &self,
        family: &PreparedModelFamily,
        source: Option<&dyn WeightComponentSource>,
    ) -> Result<usize, VNextError> {
        self.queries.lock().unwrap().push(source.is_some());
        let bytes = family
            .stored_weight_schema()?
            .components
            .iter()
            .map(|component| {
                self.parent
                    .backing_bytes(component.physical_bytes()? as usize)
                    .map_err(host_error)
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .sum::<usize>();
        // Distinct values prove that the optional source reaches this exact
        // retained authority through a cloned execution plan.
        Ok(if source.is_some() { bytes } else { 2 * bytes })
    }

    fn materialize_component<'source>(
        &self,
        _source: &'source dyn WeightComponentSource,
        _source_components: &[&WeightComponentSpec],
        _execution_component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'source>, VNextError> {
        panic!("managed static initialization must supply invocation Host authority")
    }

    fn materialize_components_with_host<'source>(
        &self,
        source: &'source dyn WeightComponentSource,
        source_components: &[&WeightComponentSpec],
        execution_components: &[&WeightComponentSpec],
        host: &HostMemoryAccount,
    ) -> Result<Vec<WeightComponentPayload<'source>>, VNextError> {
        if !host.same_parent(&self.parent) {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "test materializer rejected foreign invocation Host account".to_owned(),
            });
        }
        let payloads =
            self.base
                .materialize_components(source, source_components, execution_components)?;
        let backing = payloads
            .iter()
            .map(|payload| {
                host.allocate_zeroed(HostMemoryClass::WeightBacking, payload.bytes().len())
                    .map(|mut buffer| {
                        buffer.as_mut().copy_from_slice(payload.bytes());
                        buffer.freeze()
                    })
                    .map_err(host_error)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let managed = execution_components
            .iter()
            .zip(&payloads)
            .zip(&backing)
            .map(|((component, payload), bytes)| {
                assert!(self.parent.owns(bytes));
                WeightComponentPayload::from_managed_host(
                    component,
                    payload.external_names().to_vec(),
                    payload.source_files().to_vec(),
                    component.dimensions.clone(),
                    component.physical_element_type(),
                    bytes.clone(),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.retained.lock().unwrap().extend(backing);
        Ok(managed)
    }
}

fn host_error(error: HostMemoryError) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: error.to_string(),
    }
}

fn backing_host(bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                transfer_staging_bytes: 0,
                output_bytes: 0,
                weight_backing_bytes: bytes,
            },
        )
        .unwrap(),
    )
}

#[test]
fn managed_materializer_consumes_invocation_scope_and_keeps_live_cache_charged() {
    let parent = backing_host(16);
    let retained = Arc::new(Mutex::new(Vec::new()));
    let queries = Arc::new(Mutex::new(Vec::new()));
    let (family, plan, runtime, trace) =
        derived_component_test_plan_with(Box::new(ManagedDerivedMaterializer {
            base: DerivedComponentMaterializer::new(),
            parent: parent.clone(),
            retained: Arc::clone(&retained),
            queries: Arc::clone(&queries),
        }));
    trace.lock().unwrap().managed_static_import = ManagedStaticImportBehavior::Import;
    assert!(!runtime.requires_managed_static_staging());
    let requested = Arc::new(Mutex::new(Vec::new()));
    let source = StrictSourceWeightSource {
        requested: Arc::clone(&requested),
    };
    let plan = plan.clone();
    assert_eq!(
        plan.additional_host_backing_bytes(&family, None).unwrap(),
        32
    );
    let preparation = plan.prepare_host_backing(&family, Some(&source)).unwrap();
    let bytes = preparation.additional_bytes();
    assert_eq!(bytes, 16);
    assert_eq!(*queries.lock().unwrap(), vec![false, true]);
    assert!(requested.lock().unwrap().is_empty());
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
    let scope = parent
        .reserve_scope(&[(HostMemoryClass::WeightBacking, bytes)])
        .unwrap();
    assert!(matches!(
        parent.reserve(HostMemoryClass::WeightBacking, 1),
        Err(HostMemoryError::Capacity { .. })
    ));
    let initialized = committed_transaction(&plan, runtime, "managed-materializer-scope")
        .initialize_static_managed(
            &family,
            &plan,
            &source,
            &scope.account(),
            StaticInitializationPolicy::new(8, 2).unwrap(),
        )
        .unwrap();
    assert_eq!(initialized.receipt().imported_bytes(), 16);
    assert_eq!(trace.lock().unwrap().submit_calls, 0);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 16);
    assert_eq!(parent.usage().unwrap().allocated_bytes(), 16);
    assert_eq!(retained.lock().unwrap()[0].bytes(), &[0xa1; 8]);
    assert_eq!(retained.lock().unwrap()[1].bytes(), &[0xb2; 8]);
    drop(scope);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 16);
    close_with_expected_static_resources(
        handoff(initialized),
        plan.payload().memory().static_allocations().len(),
    );
    drop(preparation);
    retained.lock().unwrap().clear();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn default_materializer_remains_compatible_with_managed_import() {
    let (family, plan, runtime, trace) = derived_component_test_plan();
    trace.lock().unwrap().managed_static_import = ManagedStaticImportBehavior::Import;
    let host = host_without_staging();
    assert_eq!(
        plan.additional_host_backing_bytes(&family, Some(&ZeroWeightSource))
            .unwrap(),
        0
    );
    let initialized = committed_transaction(&plan, runtime, "legacy-materializer-scope")
        .initialize_static_managed(
            &family,
            &plan,
            &ZeroWeightSource,
            &host,
            StaticInitializationPolicy::new(8, 2).unwrap(),
        )
        .unwrap();
    assert_eq!(initialized.receipt().imported_bytes(), 16);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    close_with_expected_static_resources(
        handoff(initialized),
        plan.payload().memory().static_allocations().len(),
    );
}

#[test]
fn managed_materializer_rejects_foreign_account_before_payload_allocation() {
    let parent = backing_host(16);
    let foreign = backing_host(16);
    let retained = Arc::new(Mutex::new(Vec::new()));
    let (family, plan, runtime, trace) =
        derived_component_test_plan_with(Box::new(ManagedDerivedMaterializer {
            base: DerivedComponentMaterializer::new(),
            parent: parent.clone(),
            retained: Arc::clone(&retained),
            queries: Arc::new(Mutex::new(Vec::new())),
        }));
    trace.lock().unwrap().managed_static_import = ManagedStaticImportBehavior::Import;
    let scope = foreign
        .reserve_scope(&[(HostMemoryClass::WeightBacking, 16)])
        .unwrap();
    let mut failure = committed_transaction(&plan, runtime, "foreign-materializer-scope")
        .initialize_static_managed(
            &family,
            &plan,
            &ZeroWeightSource,
            &scope.account(),
            StaticInitializationPolicy::new(8, 2).unwrap(),
        )
        .err()
        .expect("materializer must reject unrelated Host authority");
    assert!(failure
        .failure()
        .message()
        .contains("foreign invocation Host"));
    failure.retire_in_place().unwrap();
    assert!(retained.lock().unwrap().is_empty());
    assert_eq!(parent.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(foreign.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(trace.lock().unwrap().submit_calls, 0);
    drop(scope);
    assert_eq!(foreign.usage().unwrap().committed_bytes(), 0);
}
