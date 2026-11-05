//! Real static initialization, Host scopes and fence retention over the shared
//! contract backend. Diagnostic snapshots never retain an upload allocation.
use super::*;

fn host_error(error: HostMemoryError) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: error.to_string(),
    }
}

fn host(weight: usize, staging: usize) -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            weight + staging,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                transfer_staging_bytes: staging,
                output_bytes: 0,
                weight_backing_bytes: weight,
            },
        )
        .unwrap(),
        8,
    )
    .unwrap()
}

fn enable(trace: &Arc<Mutex<RuntimeTrace>>, direct: bool) {
    let mut trace = trace.lock().unwrap();
    trace.managed_static_uploads_enabled = true;
    trace.managed_static_weight_sources_enabled = direct;
}

fn payload_bytes(component: &WeightComponentSpec) -> [u8; 8] {
    if component.id.as_str().ends_with("left") {
        // F32 low bits and negative zero must reach the device unchanged.
        [1, 0, 0x80, 0x3f, 0, 0, 0, 0x80]
    } else {
        [0xff, 0xff, 0x7f, 0x3f, 1, 0, 0, 0]
    }
}

struct Source {
    host: HostMemoryAccount,
    before: Mutex<Vec<(usize, usize)>>,
    fail_first_fence: Option<Arc<Mutex<RuntimeTrace>>>,
    failed_once: AtomicBool,
}
impl Source {
    fn new(host: HostMemoryAccount) -> Self {
        Self {
            host,
            before: Mutex::new(Vec::new()),
            fail_first_fence: None,
            failed_once: AtomicBool::new(false),
        }
    }
}
impl WeightComponentSource for Source {
    fn component<'a>(
        &'a self,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        let usage = self.host.usage().unwrap();
        self.before.lock().unwrap().push((
            usage.class_allocated_bytes(HostMemoryClass::WeightBacking),
            usage.class_allocated_bytes(HostMemoryClass::TransferStaging),
        ));
        if let Some(trace) = &self.fail_first_fence {
            if !self.failed_once.swap(true, Ordering::SeqCst) {
                // This changes only subsequent fences: zero initialization has
                // already completed before materialization asks for a payload.
                trace.lock().unwrap().fence_behavior = FenceBehavior::Indeterminate;
            }
        }
        let bytes = payload_bytes(component);
        let mut backing = self
            .host
            .allocate_zeroed(HostMemoryClass::WeightBacking, bytes.len())
            .map_err(host_error)?;
        backing.as_mut().copy_from_slice(&bytes);
        WeightComponentPayload::from_managed_host(
            component,
            component.external_names.clone(),
            vec!["model.safetensors".into(); component.external_names.len()],
            component.dimensions.clone(),
            component.physical_element_type(),
            backing.freeze(),
        )
    }
}

#[test]
fn direct_upload_exact_scope_and_distinct_groups_use_only_one_source_peak() {
    let (resolved, plan, runtime, trace) = test_plan();
    enable(&trace, true);
    // Logical8B at alignment8 occupies15B. There is only one such source
    // allocation's allowance, although the plan has two independent groups.
    let parent = host(15, 15);
    let scope = parent
        .reserve_scope(&[
            (HostMemoryClass::WeightBacking, 15),
            (HostMemoryClass::TransferStaging, 15),
        ])
        .unwrap();
    let account = scope.account();
    let source = Source::new(account.clone());
    let initialized = committed_transaction(&plan, runtime, "direct-exact-groups")
        .initialize_static_managed(
            &resolved.parts().prepared_family,
            &plan,
            &source,
            &account,
            StaticInitializationPolicy::new(8, 2).unwrap(),
        )
        .unwrap();
    assert_eq!(initialized.receipt().retained_source_upload_bytes(), 16);
    assert_eq!(initialized.receipt().uploaded_bytes(), 16);
    assert_eq!(initialized.receipt().staging_copy_duration_us(), 0);
    assert_eq!(*source.before.lock().unwrap(), vec![(0, 0), (0, 0)]);
    let t = trace.lock().unwrap();
    assert_eq!(
        t.managed_upload_classes,
        vec![HostMemoryClass::WeightBacking; 2]
    );
    assert_eq!(t.managed_upload_backing_bytes, vec![15, 15]);
    assert_eq!(t.uploaded_ranges.len(), 2);
    let components = &plan
        .payload()
        .execution_weights()
        .stored()
        .unwrap()
        .schema()
        .components;
    for ((managed, _, uploaded), component) in t.uploaded_ranges.iter().zip(components) {
        assert!(*managed);
        assert_eq!(uploaded.as_slice(), &payload_bytes(component)[..]);
    }
    drop(t);
    assert_eq!(parent.usage().unwrap().allocated_bytes(), 0);
    close(handoff(initialized));
    drop(source);
    drop(account);
    drop(scope);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}

/// Both outputs share one declared materialization group. Alias mode gives
/// both components the same immutable physical allocation; mixed mode puts a
/// small managed component before a differently sized ordinary component.
struct GroupMaterializer {
    base: DerivedComponentMaterializer,
    mixed_sizes: bool,
    seen_backing: Arc<Mutex<Vec<usize>>>,
}
impl WeightMaterializer for GroupMaterializer {
    fn descriptor(&self) -> &WeightMaterializerDescriptor {
        self.base.descriptor()
    }
    fn execution_schema(
        &self,
        family: &PreparedModelFamily,
        device: &DeviceDescriptor,
    ) -> Result<WeightSchema, VNextError> {
        let mut schema = self.base.execution_schema(family, device)?;
        if self.mixed_sizes {
            schema.components[0].dimensions = vec![1];
            schema.components[1].dimensions = vec![3];
            let PhysicalWeightLayout::Composite { parts } = &mut schema.tensors[0].physical_layout
            else {
                unreachable!()
            };
            parts[0].extents = vec![1];
            parts[1].extents = vec![3];
            parts[1].logical_offsets = vec![1];
        }
        Ok(schema)
    }
    fn component_sources(
        &self,
        family: &PreparedModelFamily,
        schema: &WeightSchema,
    ) -> Result<BTreeMap<WeightId, Vec<WeightId>>, VNextError> {
        self.base.component_sources(family, schema)
    }
    fn materialize_component<'a>(
        &self,
        _source: &'a dyn WeightComponentSource,
        _components: &[&WeightComponentSpec],
        _execution: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        panic!("managed group requires the actual invocation Host account")
    }
    fn materialize_components_with_host<'a>(
        &self,
        source: &'a dyn WeightComponentSource,
        source_components: &[&WeightComponentSpec],
        execution: &[&WeightComponentSpec],
        host: &HostMemoryAccount,
    ) -> Result<Vec<WeightComponentPayload<'a>>, VNextError> {
        // Use actual declared source bytes; this fixture is supplied by the
        // existing ZeroWeightSource, so concatenation/partition preserves them.
        let sources = source_components
            .iter()
            .map(|c| source.component(c))
            .collect::<Result<Vec<_>, _>>()?;
        assert!(sources.iter().all(|p| p.bytes().iter().all(|&b| b == 0)));
        let first_bytes = execution[0].physical_bytes()? as usize;
        let backing = host
            .allocate_zeroed(HostMemoryClass::WeightBacking, first_bytes)
            .map_err(host_error)?
            .freeze();
        let payloads = execution
            .iter()
            .enumerate()
            .map(|(i, component)| {
                let names = component.external_names.clone();
                let files = vec!["model.safetensors".into(); names.len()];
                if self.mixed_sizes && i == 1 {
                    WeightComponentPayload::from_ordered_sources(
                        component,
                        names,
                        files,
                        component.dimensions.clone(),
                        component.physical_element_type(),
                        vec![0; component.physical_bytes()? as usize],
                    )
                } else {
                    WeightComponentPayload::from_managed_host(
                        component,
                        names,
                        files,
                        component.dimensions.clone(),
                        component.physical_element_type(),
                        backing.clone(),
                    )
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.seen_backing.lock().unwrap().push(
            host.usage()
                .unwrap()
                .class_allocated_bytes(HostMemoryClass::WeightBacking),
        );
        Ok(payloads)
    }
}

#[test]
fn direct_same_group_aliases_keep_one_physical_charge() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let (family, plan, runtime, trace) =
        derived_component_test_plan_with(Box::new(GroupMaterializer {
            base: DerivedComponentMaterializer::new(),
            mixed_sizes: false,
            seen_backing: Arc::clone(&seen),
        }));
    enable(&trace, true);
    let account = host(15, 15);
    let initialized = committed_transaction(&plan, runtime, "direct-alias")
        .initialize_static_managed(
            &family,
            &plan,
            &ZeroWeightSource,
            &account,
            StaticInitializationPolicy::new(4, 8).unwrap(),
        )
        .unwrap();
    assert_eq!(*seen.lock().unwrap(), vec![15]);
    assert_eq!(initialized.receipt().retained_source_upload_bytes(), 16);
    assert_eq!(initialized.receipt().staging_copy_duration_us(), 0);
    let t = trace.lock().unwrap();
    assert_eq!(
        t.managed_upload_classes,
        vec![HostMemoryClass::WeightBacking; 4]
    );
    assert!(t.managed_upload_backing_bytes.iter().all(|&n| n == 15));
    assert!(t
        .uploaded_ranges
        .iter()
        .all(|(_, _, bytes)| bytes == &[0; 4]));
    drop(t);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    close_with_expected_static_resources(
        handoff(initialized),
        plan.payload().memory().static_allocations().len(),
    );
}

#[test]
fn direct_first_component_releases_unused_reservation_before_different_sized_fallback() {
    let (family, plan, runtime, trace) =
        derived_component_test_plan_with(Box::new(GroupMaterializer {
            base: DerivedComponentMaterializer::new(),
            mixed_sizes: true,
            seen_backing: Arc::new(Mutex::new(Vec::new())),
        }));
    enable(&trace, true);
    let account = host(15, 15);
    let initialized = committed_transaction(&plan, runtime, "direct-then-staged")
        .initialize_static_managed(
            &family,
            &plan,
            &ZeroWeightSource,
            &account,
            StaticInitializationPolicy::new(8, 8).unwrap(),
        )
        .unwrap();
    assert_eq!(initialized.receipt().retained_source_upload_bytes(), 4);
    assert_eq!(initialized.receipt().uploaded_bytes(), 16);
    assert_eq!(
        trace.lock().unwrap().managed_upload_classes,
        vec![
            HostMemoryClass::WeightBacking,
            HostMemoryClass::TransferStaging,
            HostMemoryClass::TransferStaging
        ]
    );
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    close_with_expected_static_resources(
        handoff(initialized),
        plan.payload().memory().static_allocations().len(),
    );
}

#[test]
fn unsupported_direct_source_uses_original_accounted_staging() {
    let (resolved, plan, runtime, trace) = test_plan();
    enable(&trace, false);
    let account = host(15, 15);
    let source = Source::new(account.clone());
    let initialized = committed_transaction(&plan, runtime, "direct-unsupported")
        .initialize_static_managed(
            &resolved.parts().prepared_family,
            &plan,
            &source,
            &account,
            StaticInitializationPolicy::new(8, 2).unwrap(),
        )
        .unwrap();
    assert_eq!(initialized.receipt().retained_source_upload_bytes(), 0);
    assert_eq!(
        trace.lock().unwrap().managed_upload_classes,
        vec![HostMemoryClass::TransferStaging; 2]
    );
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    close(handoff(initialized));
}

#[test]
fn foreign_parent_sibling_and_unscoped_source_never_take_direct_route() {
    for origin in ["foreign", "sibling", "parent"] {
        let (resolved, plan, runtime, trace) = test_plan();
        enable(&trace, true);
        let parent = host(30, 15);
        let target = parent
            .reserve_scope(&[
                (HostMemoryClass::WeightBacking, 15),
                (HostMemoryClass::TransferStaging, 15),
            ])
            .unwrap();
        let sibling = parent
            .reserve_scope(&[(HostMemoryClass::WeightBacking, 15)])
            .unwrap();
        let source_account = match origin {
            "foreign" => host(15, 0),
            "sibling" => sibling.account(),
            "parent" => {
                drop(sibling);
                parent.clone()
            }
            _ => unreachable!(),
        };
        let source = Source::new(source_account.clone());
        let initialized = committed_transaction(&plan, runtime, origin)
            .initialize_static_managed(
                &resolved.parts().prepared_family,
                &plan,
                &source,
                &target.account(),
                StaticInitializationPolicy::new(8, 2).unwrap(),
            )
            .unwrap();
        assert_eq!(
            initialized.receipt().retained_source_upload_bytes(),
            0,
            "{origin}"
        );
        assert!(
            trace
                .lock()
                .unwrap()
                .managed_upload_classes
                .iter()
                .all(|&class| class == HostMemoryClass::TransferStaging),
            "{origin}"
        );
        assert_eq!(
            source_account.usage().unwrap().allocated_bytes(),
            0,
            "{origin}"
        );
        close(handoff(initialized));
    }
}

#[test]
fn indeterminate_fence_retains_original_weight_charge_until_successful_recovery() {
    let (resolved, plan, runtime, trace) = test_plan();
    enable(&trace, true);
    let account = host(15, 15);
    let mut source = Source::new(account.clone());
    source.fail_first_fence = Some(Arc::clone(&trace));
    let failure = match committed_transaction(&plan, Arc::clone(&runtime), "direct-unknown")
        .initialize_static_managed(
            &resolved.parts().prepared_family,
            &plan,
            &source,
            &account,
            StaticInitializationPolicy::new(8, 2).unwrap(),
        ) {
        Ok(_) => panic!("indeterminate upload cannot become initialized"),
        Err(failure) => failure,
    };
    assert!(failure.is_indeterminate());
    assert_eq!(
        account
            .usage()
            .unwrap()
            .class_allocated_bytes(HostMemoryClass::WeightBacking),
        15
    );
    assert_eq!(
        account
            .usage()
            .unwrap()
            .class_allocated_bytes(HostMemoryClass::TransferStaging),
        0
    );
    assert!(account
        .allocate_zeroed(HostMemoryClass::WeightBacking, 1)
        .is_err());
    trace.lock().unwrap().synchronize_fails = true;
    let failure = match failure.recover() {
        Ok(_) => panic!("failed synchronization cannot release upload owner"),
        Err(failure) => failure,
    };
    assert_eq!(
        account
            .usage()
            .unwrap()
            .class_allocated_bytes(HostMemoryClass::WeightBacking),
        15
    );
    {
        let mut t = trace.lock().unwrap();
        t.synchronize_fails = false;
        t.fence_behavior = FenceBehavior::Succeeded;
    }
    let transaction = failure
        .recover()
        .unwrap_or_else(|_| panic!("quiescent recovery failed"));
    assert_eq!(account.usage().unwrap().allocated_bytes(), 0);
    let initialized = transaction
        .initialize_static_managed(
            &resolved.parts().prepared_family,
            &plan,
            &source,
            &account,
            StaticInitializationPolicy::new(8, 2).unwrap(),
        )
        .unwrap();
    assert_eq!(initialized.receipt().retained_source_upload_bytes(), 16);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    close(handoff(initialized));
}

#[test]
fn conservative_first_staging_admission_one_byte_short_submits_nothing() {
    let (resolved, plan, runtime, trace) = test_plan();
    enable(&trace, true);
    let account = host(15, 14);
    let source = Source::new(account.clone());
    let failure = match committed_transaction(&plan, runtime, "direct-short-preflight")
        .initialize_static_managed(
            &resolved.parts().prepared_family,
            &plan,
            &source,
            &account,
            StaticInitializationPolicy::new(8, 2).unwrap(),
        ) {
        Ok(_) => panic!("one-byte-short conservative admission must reject"),
        Err(failure) => failure,
    };
    assert!(!failure.is_indeterminate());
    assert!(source.before.lock().unwrap().is_empty());
    assert_eq!(trace.lock().unwrap().submit_calls, 0);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    drop(failure);
}
