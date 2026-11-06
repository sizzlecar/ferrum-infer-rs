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
    host_aligned(weight, staging, 8)
}

fn host_aligned(weight: usize, staging: usize, alignment: usize) -> HostMemoryAccount {
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
        alignment,
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
    streaming: bool,
    range_requests: Mutex<Vec<(WeightId, usize, usize)>>,
    fail_after_ranges: Option<usize>,
    hold_range: bool,
    held: Mutex<Option<ManagedHostBytes>>,
    submit_panic: bool,
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
            streaming: false,
            range_requests: Mutex::new(Vec::new()),
            fail_after_ranges: None,
            hold_range: false,
            held: Mutex::new(None),
            submit_panic: false,
        }
    }
}

impl WeightComponentSource for Source {
    fn streaming_component(
        &self,
        component: &WeightComponentSpec,
    ) -> Result<Option<WeightComponentStreamSpec>, VNextError> {
        self.streaming
            .then(|| {
                WeightComponentStreamSpec::new(
                    component,
                    vec!["model.safetensors".into(); component.external_names.len()],
                    4,
                )
            })
            .transpose()
    }

    fn component_range(
        &self,
        request: &WeightComponentRangeRequest,
    ) -> Result<WeightComponentRange, VNextError> {
        let mut requests = self.range_requests.lock().unwrap();
        if self.fail_after_ranges == Some(requests.len()) {
            return Err(invalid("late range conversion failed"));
        }
        let mut slot = self.owner.lock().unwrap();
        let owner = slot.take().ok_or_else(|| invalid("source not prepared"))?;
        let mut buffer = match owner.try_thaw() {
            Ok(buffer) => buffer,
            Err(owner) => {
                *slot = Some(owner);
                return Err(invalid("previous range still retained"));
            }
        };
        let begin = request.offset_bytes();
        let end = begin + request.length_bytes();
        buffer.as_mut()[..request.length_bytes()]
            .copy_from_slice(&payload_bytes(request.component())[begin..end]);
        let owner = buffer.freeze();
        self.addresses
            .lock()
            .unwrap()
            .push(owner.bytes().as_ptr() as usize);
        if self.hold_range && requests.is_empty() {
            *self.held.lock().unwrap() = Some(owner.clone());
        }
        *slot = Some(owner.clone());
        if self.unknown.swap(false, Ordering::SeqCst) {
            self.trace.lock().unwrap().fence_behavior = FenceBehavior::Indeterminate;
        }
        if self.submit_panic {
            self.trace.lock().unwrap().submit_behavior = SubmitBehavior::Panic;
        }
        requests.push((
            request.component().id.clone(),
            begin,
            request.length_bytes(),
        ));
        WeightComponentRange::new(request, owner.slice(0..request.length_bytes()).unwrap())
    }

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
        if self.streaming {
            assert!(context.supports_managed_source_upload());
            assert!(context.maximum_source_chunk_bytes() >= 4);
        }
        let owner = self
            .host
            .allocate_zeroed(
                HostMemoryClass::WeightBacking,
                if self.streaming { 4 } else { 8 },
            )
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

#[test]
fn streaming_source_reuses_one_bounded_owner_with_contiguous_exact_device_bytes() {
    let (resolved, plan, runtime, trace) = test_plan();
    enable(&trace);
    // Four logical bytes occupy seven physical bytes at alignment four.
    // Neither of the eight-byte complete components fits the source budget.
    let account = host_aligned(7, 7, 4);
    let family = &resolved.parts().prepared_family;
    let mut source = Source::new(account.clone(), family, &plan, trace.clone());
    source.streaming = true;
    let initialized = committed_transaction(&plan, runtime, "streamed-ranges")
        .initialize_static_managed(
            family,
            &plan,
            &source,
            &account,
            StaticInitializationPolicy::new(4, 8).unwrap(),
        )
        .unwrap();
    assert_eq!(source.components.load(Ordering::SeqCst), 0);
    let expected = family
        .stored_weight_schema()
        .unwrap()
        .components
        .iter()
        .flat_map(|c| [(c.id.clone(), 0, 4), (c.id.clone(), 4, 4)])
        .collect::<Vec<_>>();
    assert_eq!(*source.range_requests.lock().unwrap(), expected);
    let pointers = source.addresses.lock().unwrap();
    assert!(pointers.windows(2).all(|pair| pair[0] == pair[1]));
    let t = trace.lock().unwrap();
    assert_eq!(t.managed_upload_backing_bytes, vec![7; 4]);
    for (uploads, component) in t
        .uploaded_ranges
        .chunks_exact(2)
        .zip(&family.stored_weight_schema().unwrap().components)
    {
        let actual = uploads
            .iter()
            .flat_map(|(_, _, bytes)| bytes.clone())
            .collect::<Vec<_>>();
        assert_eq!(actual, payload_bytes(component));
    }
    drop(t);
    drop(pointers);
    assert_eq!(initialized.receipt().uploaded_bytes(), 16);
    assert_eq!(initialized.receipt().retained_source_upload_bytes(), 16);
    close(handoff(initialized));
    assert_eq!(account.usage().unwrap().allocated_bytes(), 7);
    drop(source);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn streaming_source_capacity_route_and_nonidentity_fail_before_source_allocation_or_writes() {
    for case in 0..6 {
        let (resolved, plan, runtime, trace) = test_plan();
        enable(&trace);
        let account = host_aligned(
            if case == 0 { 6 } else { 7 },
            if case == 1 { 6 } else { 7 },
            4,
        );
        let family = &resolved.parts().prepared_family;
        let mut source = Source::new(account.clone(), family, &plan, trace.clone());
        source.streaming = true;
        match case {
            2 => trace.lock().unwrap().managed_static_weight_sources_enabled = false,
            3 => {
                trace.lock().unwrap().static_source_retention =
                    StaticWeightSourceRetention::UnknownOrPlanRetained
            }
            4 => trace.lock().unwrap().managed_static_import = ManagedStaticImportBehavior::Import,
            _ => {}
        }
        let tx = committed_transaction(&plan, runtime, "streamed-preflight-reject");
        let result = if case == 5 {
            tx.initialize_static(
                family,
                &plan,
                &source,
                StaticInitializationPolicy::new(4, 8).unwrap(),
            )
        } else {
            tx.initialize_static_managed(
                family,
                &plan,
                &source,
                &account,
                StaticInitializationPolicy::new(4, 8).unwrap(),
            )
        };
        assert!(result.is_err());
        assert_eq!(trace.lock().unwrap().submit_calls, 0);
        assert!(source.range_requests.lock().unwrap().is_empty());
        assert_eq!(account.usage().unwrap().allocated_bytes(), 0);
    }
    let (family, plan, runtime, trace) =
        derived_component_test_plan_with(Box::new(DerivedComponentMaterializer::new()));
    enable(&trace);
    let account = host_aligned(7, 7, 4);
    let mut source = Source::new(account.clone(), &family, &plan, trace.clone());
    source.streaming = true;
    assert!(
        committed_transaction(&plan, runtime, "streamed-derived-reject")
            .initialize_static_managed(
                &family,
                &plan,
                &source,
                &account,
                StaticInitializationPolicy::new(4, 8).unwrap()
            )
            .is_err()
    );
    assert_eq!(trace.lock().unwrap().submit_calls, 0);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn streaming_late_conversion_or_retained_alias_aborts_without_overwrite_or_executor() {
    for hold_alias in [false, true] {
        let (resolved, plan, runtime, trace) = test_plan();
        enable(&trace);
        let account = host_aligned(7, 7, 4);
        let family = &resolved.parts().prepared_family;
        let mut source = Source::new(account.clone(), family, &plan, trace.clone());
        source.streaming = true;
        source.hold_range = hold_alias;
        source.fail_after_ranges = (!hold_alias).then_some(1);
        let failed = committed_transaction(&plan, runtime, "streamed-late-failure")
            .initialize_static_managed(
                family,
                &plan,
                &source,
                &account,
                StaticInitializationPolicy::new(4, 8).unwrap(),
            );
        assert!(failed.is_err());
        assert!(!failed.as_ref().err().unwrap().is_indeterminate());
        assert_eq!(source.range_requests.lock().unwrap().len(), 1);
        assert_eq!(trace.lock().unwrap().uploaded_ranges.len(), 1);
        if hold_alias {
            assert_eq!(
                source.held.lock().unwrap().as_ref().unwrap().bytes(),
                &payload_bytes(&family.stored_weight_schema().unwrap().components[0])[..4]
            );
        }
        drop(failed);
        drop(source);
        assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
fn streaming_unknown_wait_or_submit_panic_retains_original_owner_after_source_drop() {
    for submit_panic in [false, true] {
        let (resolved, plan, runtime, trace) = test_plan();
        enable(&trace);
        let account = host_aligned(7, 7, 4);
        let family = &resolved.parts().prepared_family;
        let mut source = Source::new(account.clone(), family, &plan, trace.clone());
        source.streaming = true;
        source.submit_panic = submit_panic;
        source.unknown.store(!submit_panic, Ordering::SeqCst);
        let failure = match committed_transaction(&plan, runtime, "streamed-unknown")
            .initialize_static_managed(
                family,
                &plan,
                &source,
                &account,
                StaticInitializationPolicy::new(4, 8).unwrap(),
            ) {
            Ok(_) => panic!("unknown completion cannot publish resources"),
            Err(failure) => failure,
        };
        assert!(failure.is_indeterminate());
        assert_eq!(source.range_requests.lock().unwrap().len(), 1);
        drop(source);
        assert_eq!(account.usage().unwrap().allocated_bytes(), 7);
        trace.lock().unwrap().synchronize_fails = true;
        let failure = failure
            .recover()
            .err()
            .expect("original recovery must retain unknown work");
        assert_eq!(account.usage().unwrap().allocated_bytes(), 7);
        {
            let mut t = trace.lock().unwrap();
            t.synchronize_fails = false;
            t.submit_behavior = SubmitBehavior::Success;
            t.fence_behavior = FenceBehavior::Succeeded;
        }
        drop(
            failure
                .recover()
                .unwrap_or_else(|_| panic!("original recovery failed")),
        );
        assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
fn full_managed_submit_panic_retains_original_source_until_recovery() {
    // Exercise the existing full-component route. Inject only after producing
    // its actual payload, so an earlier zero submission cannot consume fault.
    struct PanicAfterPayload(Source);
    impl WeightComponentSource for PanicAfterPayload {
        fn validate_static_initialization(
            &self,
            context: &StaticWeightSourceContext<'_>,
        ) -> Result<(), VNextError> {
            self.0.validate_static_initialization(context)
        }

        fn component<'a>(
            &'a self,
            component: &WeightComponentSpec,
        ) -> Result<WeightComponentPayload<'a>, VNextError> {
            let payload = self.0.component(component)?;
            self.0.trace.lock().unwrap().submit_behavior = SubmitBehavior::Panic;
            Ok(payload)
        }
    }

    let (resolved, plan, runtime, trace) = test_plan();
    enable(&trace);
    let account = host(15, 15);
    let family = &resolved.parts().prepared_family;
    let source = PanicAfterPayload(Source::new(account.clone(), family, &plan, trace.clone()));
    let failure = committed_transaction(&plan, runtime, "full-source-submit-panic")
        .initialize_static_managed(
            family,
            &plan,
            &source,
            &account,
            StaticInitializationPolicy::new(8, 8).unwrap(),
        )
        .err()
        .expect("submit panic cannot publish initialized resources");
    assert!(failure.is_indeterminate());
    assert_eq!(failure.failure().code(), "static_submit_indeterminate");
    assert_eq!(source.0.components.load(Ordering::SeqCst), 1);
    assert!(source.0.range_requests.lock().unwrap().is_empty());
    drop(source);
    let after_source_drop = account.usage().unwrap().allocated_bytes();
    trace.lock().unwrap().synchronize_fails = true;
    let failure = failure
        .recover()
        .err()
        .expect("failed original synchronization retains custody");
    let after_failed_sync = account.usage().unwrap().allocated_bytes();
    {
        let mut trace = trace.lock().unwrap();
        trace.synchronize_fails = false;
        trace.submit_behavior = SubmitBehavior::Success;
    }
    drop(
        failure
            .recover()
            .unwrap_or_else(|_| panic!("original recovery failed")),
    );
    // Evaluate the captured intermediate observations after actual cleanup, so
    // a failing regression does not leave an unrelated deferred transaction.
    assert_eq!(after_source_drop, account.backing_bytes(8).unwrap());
    assert_eq!(after_failed_sync, account.backing_bytes(8).unwrap());
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}
