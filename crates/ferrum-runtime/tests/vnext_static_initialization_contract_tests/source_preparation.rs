//! The core supplies actual route, plan and scope facts before initialization
//! writes; a reusable source never infers them from a backend or model name.
use super::*;
use std::sync::atomic::AtomicUsize;

fn payload_bytes(component: &WeightComponentSpec) -> [u8; 8] {
    if component.id.as_str().ends_with("left") {
        [1, 0, 0x80, 0x3f, 0, 0, 0, 0x80]
    } else {
        [0xff, 0xff, 0x7f, 0x3f, 1, 0, 0, 0]
    }
}

fn host(weight: usize, staging: usize) -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            weight + staging,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                output_bytes: 0,
                transfer_staging_bytes: staging,
                weight_backing_bytes: weight,
            },
        )
        .unwrap(),
        8,
    )
    .unwrap()
}

fn invalid(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}

struct Source {
    host: HostMemoryAccount,
    expected_plan: PlanHash,
    expected_family: ModelFamilyId,
    trace: Arc<Mutex<RuntimeTrace>>,
    owner: Mutex<Option<ManagedHostBytes>>,
    addresses: Mutex<Vec<usize>>,
    components: AtomicUsize,
    unknown: AtomicBool,
}

impl Source {
    fn new(
        host: HostMemoryAccount,
        family: &PreparedModelFamily,
        plan: &ExecutionPlan,
        trace: Arc<Mutex<RuntimeTrace>>,
    ) -> Self {
        Self {
            host,
            expected_plan: plan.plan_hash().clone(),
            expected_family: family.family_id().clone(),
            trace,
            owner: Mutex::new(None),
            addresses: Mutex::new(Vec::new()),
            components: AtomicUsize::new(0),
            unknown: AtomicBool::new(false),
        }
    }
}

impl WeightComponentSource for Source {
    fn validate_static_initialization(
        &self,
        context: &StaticWeightSourceContext<'_>,
    ) -> Result<(), VNextError> {
        let trace = self.trace.lock().unwrap();
        assert_eq!(trace.submit_calls, 0);
        assert_eq!(trace.managed_upload_encode_calls, 0);
        assert!(trace.submitted_commands.is_empty());
        drop(trace);
        if context.plan().plan_hash() != &self.expected_plan
            || context.family().family_id() != &self.expected_family
            || !context.matches_host_account(&self.host)
            || context.source_retention() != StaticWeightSourceRetention::FencedCopy
        {
            return Err(invalid("actual initialization source authority differs"));
        }
        let owner = self
            .host
            .allocate_zeroed(HostMemoryClass::WeightBacking, 8)
            .map_err(|error| invalid(error.to_string()))?
            .freeze();
        let mut slot = self.owner.lock().unwrap();
        if slot.is_some() {
            return Err(invalid("installation is already claimed"));
        }
        *slot = Some(owner);
        Ok(())
    }

    fn component<'a>(
        &'a self,
        spec: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        let mut slot = self.owner.lock().unwrap();
        let owner = slot
            .take()
            .ok_or_else(|| invalid("source is not prepared"))?;
        let mut buffer = match owner.try_thaw() {
            Ok(buffer) => buffer,
            Err(owner) => {
                *slot = Some(owner);
                return Err(invalid("previous source owner remains retained"));
            }
        };
        self.components.fetch_add(1, Ordering::SeqCst);
        buffer.as_mut().copy_from_slice(&payload_bytes(spec));
        let owner = buffer.freeze();
        self.addresses
            .lock()
            .unwrap()
            .push(owner.as_ref().as_ptr() as usize);
        *slot = Some(owner.clone());
        if self.unknown.swap(false, Ordering::SeqCst) {
            self.trace.lock().unwrap().fence_behavior = FenceBehavior::Indeterminate;
        }
        WeightComponentPayload::from_managed_host(
            spec,
            spec.external_names.clone(),
            vec!["model.safetensors".into(); spec.external_names.len()],
            spec.dimensions.clone(),
            spec.physical_element_type(),
            owner,
        )
    }
}

fn enable(trace: &Arc<Mutex<RuntimeTrace>>) {
    let mut trace = trace.lock().unwrap();
    trace.managed_static_uploads_enabled = true;
    trace.managed_static_weight_sources_enabled = true;
    trace.static_source_retention = StaticWeightSourceRetention::FencedCopy;
}

#[test]
fn source_preparation_uses_exact_clone_scope_and_reuses_one_real_owner() {
    let (resolved, plan, runtime, trace) = test_plan();
    enable(&trace);
    let parent = host(15, 15);
    let scope = parent
        .reserve_scope(&[
            (HostMemoryClass::WeightBacking, 15),
            (HostMemoryClass::TransferStaging, 15),
        ])
        .unwrap();
    let account = scope.account();
    let family = &resolved.parts().prepared_family;
    let source = Source::new(account.clone(), family, &plan, trace.clone());
    let initialized = committed_transaction(&plan, runtime, "prepared-reuse")
        .initialize_static_managed(
            family,
            &plan,
            &source,
            &account.clone(),
            StaticInitializationPolicy::new(8, 2).unwrap(),
        )
        .unwrap();
    assert_eq!(initialized.receipt().retained_source_upload_bytes(), 16);
    let addresses = source.addresses.lock().unwrap();
    assert_eq!(addresses.len(), 2);
    assert_eq!(addresses[0], addresses[1]);
    drop(addresses);
    assert_eq!(parent.usage().unwrap().allocated_bytes(), 15);
    close(handoff(initialized));
    scope.close().unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 15);
    drop(source);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn source_preparation_rejects_parent_sibling_and_foreign_before_writes() {
    for origin in ["parent", "sibling", "foreign"] {
        let (resolved, plan, runtime, trace) = test_plan();
        enable(&trace);
        let parent = host(60, 60);
        let target = parent
            .reserve_scope(&[
                (HostMemoryClass::WeightBacking, 15),
                (HostMemoryClass::TransferStaging, 15),
            ])
            .unwrap();
        let sibling = parent
            .reserve_scope(&[(HostMemoryClass::WeightBacking, 15)])
            .unwrap();
        let foreign = host(15, 15);
        let source_host = match origin {
            "parent" => parent.clone(),
            "sibling" => sibling.account(),
            _ => foreign.clone(),
        };
        let family = &resolved.parts().prepared_family;
        let source = Source::new(source_host, family, &plan, trace.clone());
        let result = committed_transaction(&plan, runtime, origin).initialize_static_managed(
            family,
            &plan,
            &source,
            &target.account(),
            StaticInitializationPolicy::new(8, 2).unwrap(),
        );
        assert!(result.is_err(), "{origin}");
        assert_eq!(trace.lock().unwrap().submit_calls, 0, "{origin}");
        assert_eq!(source.components.load(Ordering::SeqCst), 0, "{origin}");
        assert_eq!(parent.usage().unwrap().allocated_bytes(), 0, "{origin}");
        assert_eq!(foreign.usage().unwrap().allocated_bytes(), 0, "{origin}");
    }
}

#[test]
fn source_preparation_rejects_unknown_and_actual_import_even_with_copy_declaration() {
    for import in [false, true] {
        let (resolved, plan, runtime, trace) = test_plan();
        enable(&trace);
        if import {
            trace.lock().unwrap().managed_static_import = ManagedStaticImportBehavior::Import;
        } else {
            trace.lock().unwrap().static_source_retention =
                StaticWeightSourceRetention::UnknownOrPlanRetained;
        }
        let account = host(15, 15);
        let family = &resolved.parts().prepared_family;
        let source = Source::new(account.clone(), family, &plan, trace.clone());
        let result = committed_transaction(&plan, runtime, "unsafe-source-route")
            .initialize_static_managed(
                family,
                &plan,
                &source,
                &account,
                StaticInitializationPolicy::new(8, 2).unwrap(),
            );
        assert!(result.is_err());
        assert_eq!(trace.lock().unwrap().submit_calls, 0);
        assert_eq!(source.components.load(Ordering::SeqCst), 0);
        assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
fn source_preparation_first_staging_and_slot_capacity_fail_before_submission() {
    for (weight, staging) in [(14, 15), (15, 14)] {
        let (resolved, plan, runtime, trace) = test_plan();
        enable(&trace);
        let account = host(weight, staging);
        let family = &resolved.parts().prepared_family;
        let source = Source::new(account.clone(), family, &plan, trace.clone());
        let result = committed_transaction(&plan, runtime, "source-short")
            .initialize_static_managed(
                family,
                &plan,
                &source,
                &account,
                StaticInitializationPolicy::new(8, 2).unwrap(),
            );
        assert!(result.is_err());
        assert_eq!(trace.lock().unwrap().submit_calls, 0);
        assert_eq!(source.components.load(Ordering::SeqCst), 0);
        assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
fn source_preparation_unknown_fence_retains_alias_until_original_recovery() {
    let (resolved, plan, runtime, trace) = test_plan();
    enable(&trace);
    let account = host(15, 15);
    let family = &resolved.parts().prepared_family;
    let source = Source::new(account.clone(), family, &plan, trace.clone());
    source.unknown.store(true, Ordering::SeqCst);
    let failure = match committed_transaction(&plan, runtime, "source-unknown")
        .initialize_static_managed(
            family,
            &plan,
            &source,
            &account,
            StaticInitializationPolicy::new(8, 2).unwrap(),
        ) {
        Ok(_) => panic!("unknown fence cannot publish initialization"),
        Err(failure) => failure,
    };
    assert!(failure.is_indeterminate());
    assert_eq!(source.components.load(Ordering::SeqCst), 1);
    let spec = &family.stored_weight_schema().unwrap().components[0];
    assert!(source.component(spec).is_err());
    assert_eq!(account.usage().unwrap().allocated_bytes(), 15);
    trace.lock().unwrap().synchronize_fails = true;
    let failure = match failure.recover() {
        Ok(_) => panic!("failed synchronization cannot release original owner"),
        Err(failure) => failure,
    };
    assert_eq!(account.usage().unwrap().allocated_bytes(), 15);
    {
        let mut trace = trace.lock().unwrap();
        trace.synchronize_fails = false;
        trace.fence_behavior = FenceBehavior::Succeeded;
    }
    let transaction = failure
        .recover()
        .unwrap_or_else(|_| panic!("recovery failed"));
    let payload = source.component(spec).unwrap();
    assert_eq!(payload.bytes(), payload_bytes(spec));
    drop(payload);
    drop(transaction);
    drop(source);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}
