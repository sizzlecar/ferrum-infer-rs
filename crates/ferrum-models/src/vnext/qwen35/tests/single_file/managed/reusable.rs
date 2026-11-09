//! Real BF16 archive -> F16 physical components through the production CPU
//! static initializer. CPU copies through charged TransferStaging; these tests
//! do not claim to validate the CUDA direct-source path.
use super::*;
mod streaming;
use ferrum_kernels::backend::cpu::vnext_runtime::CpuDeviceRuntime;
use ferrum_runtime::{
    DeviceRuntime, ExecutionPlan, PlanRuntimeCloseOutcome, PlanRuntimeResources,
    ResourceTransaction, ResourceTransactionIdentity, RuntimeResourceDriver,
    StaticInitializationPolicy, StaticProvisioning, TransactionCommitted,
};

fn compile(prepared: &PreparedProductionModel) -> (Arc<CpuDeviceRuntime>, ExecutionPlan) {
    let (runtime, registry, materializers, materializer, catalog) = CpuVNextComposition::create(
        DeviceId::new("device.cpu.reusable-source").unwrap(),
        64 << 20,
    )
    .unwrap()
    .into_parts()
    .unwrap();
    let composition = RuntimeComposition::new(runtime.clone(), registry, materializers, catalog);
    let mut engine = ferrum_types::EngineConfig::default();
    engine.backend.device = ferrum_types::Device::CPU;
    engine.backend.enable_reusable_execution = false;
    engine.scheduler.max_running_requests = 2;
    engine.batching.max_num_batched_tokens = 8;
    engine.runtime.max_model_len = Some(128);
    let info = prepared.model_info(engine.model.model_id.clone(), ferrum_types::Device::CPU);
    let config =
        ProgramExecutorConfig::from_engine_config(&engine, &info, runtime.as_ref()).unwrap();
    let compiled = composition
        .compile_model(
            prepared.execution_input().unwrap(),
            info,
            &engine,
            config,
            WeightMaterializerSelection::exact(materializer),
        )
        .unwrap();
    (
        runtime,
        compiled.compilation().executable().execution_plan().clone(),
    )
}

fn committed(
    plan: &ExecutionPlan,
    runtime: Arc<CpuDeviceRuntime>,
) -> ResourceTransaction<RuntimeResourceDriver<CpuDeviceRuntime>, TransactionCommitted> {
    let provisioning = plan
        .provision_static(
            runtime.clone(),
            "request.reusable-source".to_owned().try_into().unwrap(),
        )
        .unwrap()
        .into_provisioning();
    let StaticProvisioning::Required(permit) = provisioning else {
        panic!("the actual Stored Qwen graph requires static resources")
    };
    let identity = ResourceTransactionIdentity::for_admission(
        permit.binding(),
        "run.reusable-source".to_owned().try_into().unwrap(),
        "transaction.reusable-source".to_owned().try_into().unwrap(),
    );
    let transaction = ResourceTransaction::begin(
        RuntimeResourceDriver::new(runtime).unwrap(),
        identity,
        permit,
    )
    .unwrap()
    .reserve()
    .unwrap()
    .commit()
    .unwrap_or_else(|error| match error {
        ferrum_runtime::ResourceCommitTransitionError::Recoverable(failure) => {
            panic!("actual CPU static commit failed: {:?}", failure.failure())
        }
        ferrum_runtime::ResourceCommitTransitionError::Poisoned(failure) => {
            panic!(
                "actual CPU static commit is indeterminate: {:?}",
                failure.failure()
            )
        }
    });
    let pools = transaction
        .maintenance_controller()
        .pool_ids()
        .cloned()
        .collect::<Vec<_>>();
    for pool in pools {
        transaction
            .maintenance_controller()
            .initialize_pool(&pool)
            .unwrap();
    }
    transaction
}

fn close(resources: Arc<PlanRuntimeResources<CpuDeviceRuntime>>) {
    match PlanRuntimeResources::close(resources) {
        Ok(PlanRuntimeCloseOutcome::Closed(_)) => {}
        Ok(PlanRuntimeCloseOutcome::Referenced { strong_count, .. }) => {
            panic!("unexpected references: {strong_count}")
        }
        Err(failure) => panic!(
            "actual CPU static retirement failed: {:?}",
            failure.failure()
        ),
    }
}

fn policy() -> StaticInitializationPolicy {
    StaticInitializationPolicy::new(64, 2).unwrap()
}

#[test]
fn reusable_static_source_actual_qwen_installation_keeps_one_exact_slot_until_drop() {
    let fixture = Fixture::new();
    let prepared = prepare(fixture.open().unwrap());
    let mapped = prepared.managed_safetensors().unwrap();
    let (runtime, plan) = compile(&prepared);
    assert!(!runtime.supports_managed_static_weight_sources());
    let parent = host(1 << 18);
    let peak = mapped
        .weight_source(&parent)
        .maximum_group_backing_bytes(&plan)
        .unwrap();
    let staging = parent.backing_bytes(64).unwrap();
    let scope = parent
        .reserve_scope(&[
            (HostMemoryClass::WeightBacking, peak),
            (HostMemoryClass::TransferStaging, staging),
        ])
        .unwrap();
    let account = scope.account();
    let source = mapped.weight_source(&account);
    source.prepare_reusable_static_backing(&plan).unwrap();
    assert_eq!(parent.usage().unwrap().allocated_bytes(), 0);
    let specs = &prepared.family().stored_weight_schema().unwrap().components;
    assert!(
        source.component(&specs[0]).is_err(),
        "armed is not yet authorized"
    );
    assert!(source.prepare_reusable_static_backing(&plan).is_err());
    let initialized = committed(&plan, runtime)
        .initialize_static_managed(
            prepared.family(),
            &plan,
            &source,
            &account.clone(),
            policy(),
        )
        .unwrap();
    assert_eq!(
        initialized.receipt().retained_source_upload_bytes(),
        0,
        "CPU uses charged staging, not direct managed-source uploads"
    );
    assert!(initialized.receipt().uploaded_bytes() > 0);
    assert_eq!(parent.usage().unwrap().allocated_bytes(), peak);
    let embed = component(&prepared, "model.language_model.embed_tokens.weight");
    let payload = source.component(embed).unwrap();
    let retained = payload.retained_host_memory().unwrap().clone();
    drop(payload);
    assert!(
        source.component(embed).is_err(),
        "an actual retained owner forbids checkout"
    );
    assert_eq!(parent.usage().unwrap().allocated_bytes(), peak);
    drop(retained);
    let mut addresses = std::collections::BTreeSet::new();
    for spec in specs {
        let payload = source.component(spec).unwrap();
        let expected = prepared.weights().component(spec).unwrap();
        assert_eq!(payload.bytes(), expected.bytes(), "{}", spec.id);
        assert_eq!(payload.dimensions(), expected.dimensions());
        assert_eq!(payload.element_type(), expected.element_type());
        assert_eq!(payload.external_names(), expected.external_names());
        if source.component_backing_bytes(spec).unwrap() != 0 {
            addresses.insert(payload.bytes().as_ptr() as usize);
            assert_eq!(parent.usage().unwrap().allocated_bytes(), peak);
        }
    }
    assert_eq!(
        addresses.len(),
        1,
        "differently sized conversions share the actual owner"
    );
    let mut changed = specs[0].clone();
    changed.external_names[0].push_str(".foreign");
    assert!(source.component(&changed).is_err());
    let mut changed = specs[0].clone();
    changed.dimensions[0] += 1;
    assert!(source.component(&changed).is_err());
    close(initialized.into_plan_runtime().unwrap());
    scope.close().unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), peak);
    drop(source);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn reusable_static_source_short_real_scope_fails_without_ready_or_fresh_fallback() {
    let fixture = Fixture::new();
    let prepared = prepare(fixture.open().unwrap());
    let mapped = prepared.managed_safetensors().unwrap();
    let (runtime, plan) = compile(&prepared);
    let parent = host(1 << 18);
    let peak = mapped
        .weight_source(&parent)
        .maximum_group_backing_bytes(&plan)
        .unwrap();
    let scope = parent
        .reserve_scope(&[
            (HostMemoryClass::WeightBacking, peak - 1),
            (
                HostMemoryClass::TransferStaging,
                parent.backing_bytes(64).unwrap(),
            ),
        ])
        .unwrap();
    let account = scope.account();
    let source = mapped.weight_source(&account);
    source.prepare_reusable_static_backing(&plan).unwrap();
    let failure = match committed(&plan, runtime.clone()).initialize_static_managed(
        prepared.family(),
        &plan,
        &source,
        &account,
        policy(),
    ) {
        Ok(_) => panic!("one physical byte short cannot allocate the reusable slot"),
        Err(failure) => failure,
    };
    assert!(!failure.is_indeterminate());
    assert_eq!(parent.usage().unwrap().allocated_bytes(), 0);
    let spec = &prepared.family().stored_weight_schema().unwrap().components[0];
    assert!(source.component(spec).is_err());
    assert!(source.prepare_reusable_static_backing(&plan).is_err());
    drop(failure.recover().unwrap());
    drop(source);
    scope.close().unwrap();
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
    // A separate lease with the real grant can retry; the failed one never
    // silently changes its policy or publishes a partially prepared payload.
    let source = mapped.weight_source(&parent);
    source.prepare_reusable_static_backing(&plan).unwrap();
    let initialized = committed(&plan, runtime)
        .initialize_static_managed(prepared.family(), &plan, &source, &parent, policy())
        .unwrap();
    close(initialized.into_plan_runtime().unwrap());
    drop(source);
    assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn reusable_static_source_fresh_core_hook_seals_the_original_policy() {
    struct SealProbe<'a, 'source> {
        source: &'a crate::vnext::ManagedProductionWeightSource<'source>,
        plan: &'a ExecutionPlan,
        called: std::sync::atomic::AtomicBool,
    }
    impl ferrum_runtime::WeightComponentSource for SealProbe<'_, '_> {
        fn validate_static_initialization(
            &self,
            context: &ferrum_runtime::StaticWeightSourceContext<'_>,
        ) -> Result<(), ferrum_runtime::VNextError> {
            self.source.validate_static_initialization(context)?;
            // This is the precise old race window: the core's fresh hook has
            // returned, but no zero command or component request has occurred.
            assert!(self
                .source
                .prepare_reusable_static_backing(self.plan)
                .is_err());
            self.called.store(true, std::sync::atomic::Ordering::SeqCst);
            Ok(())
        }
        fn component<'s>(
            &'s self,
            spec: &WeightComponentSpec,
        ) -> Result<ferrum_runtime::WeightComponentPayload<'s>, ferrum_runtime::VNextError>
        {
            self.source.component(spec)
        }
    }
    let fixture = Fixture::new();
    let prepared = prepare(fixture.open().unwrap());
    let mapped = prepared.managed_safetensors().unwrap();
    let (runtime, plan) = compile(&prepared);
    let account = host(1 << 18);
    let source = mapped.weight_source(&account);
    let probe = SealProbe {
        source: &source,
        plan: &plan,
        called: std::sync::atomic::AtomicBool::new(false),
    };
    let initialized = committed(&plan, runtime.clone())
        .initialize_static_managed(prepared.family(), &plan, &probe, &account, policy())
        .unwrap();
    assert!(probe.called.load(std::sync::atomic::Ordering::SeqCst));
    assert!(source.prepare_reusable_static_backing(&plan).is_err());
    assert_eq!(
        account.usage().unwrap().allocated_bytes(),
        0,
        "the default fresh route retains no conversion slot"
    );
    close(initialized.into_plan_runtime().unwrap());
    let spec = component(&prepared, "model.language_model.embed_tokens.weight");
    for _ in 0..2 {
        let payload = source.component(spec).unwrap();
        assert_eq!(
            payload.bytes(),
            prepared.weights().component(spec).unwrap().bytes()
        );
        drop(payload);
        assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    }
    let initialized = committed(&plan, runtime)
        .initialize_static_managed(prepared.family(), &plan, &probe, &account, policy())
        .unwrap();
    close(initialized.into_plan_runtime().unwrap());
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn reusable_static_source_rejects_another_actual_precision_plan() {
    let fixture = Fixture::new();
    let prepared = prepare(fixture.open().unwrap());
    let mapped = prepared.managed_safetensors().unwrap();
    let sources = Arc::clone(prepared.sources());
    let foreign = crate::vnext::resolve_registered_model_from_sources(&sources)
        .unwrap()
        .into_required()
        .unwrap()
        .define_from_sources(sources)
        .unwrap()
        .prepare(&F32_MASTER_NUMERICAL_PROFILE_ID.parse().unwrap())
        .unwrap();
    let (runtime, foreign_plan) = compile(&foreign);
    let account = host(1 << 18);
    let source = mapped.weight_source(&account);
    assert!(source
        .prepare_reusable_static_backing(&foreign_plan)
        .is_err());
    assert!(source
        .component(&prepared.family().stored_weight_schema().unwrap().components[0])
        .is_err());
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    drop(runtime);
}

#[test]
fn reusable_static_source_racing_arm_and_core_hook_never_switches_a_live_policy() {
    let fixture = Fixture::new();
    let prepared = prepare(fixture.open().unwrap());
    let mapped = prepared.managed_safetensors().unwrap();
    let (runtime, plan) = compile(&prepared);
    let account = host(1 << 18);
    let source = mapped.weight_source(&account);
    let transaction = committed(&plan, runtime);
    let barrier = std::sync::Barrier::new(2);
    std::thread::scope(|threads| {
        let arming = threads.spawn(|| {
            barrier.wait();
            source.prepare_reusable_static_backing(&plan)
        });
        barrier.wait();
        let initialized = transaction.initialize_static_managed(
            prepared.family(),
            &plan,
            &source,
            &account,
            policy(),
        );
        let armed = arming.join().unwrap();
        match initialized {
            Ok(initialized) => {
                assert!(source.prepare_reusable_static_backing(&plan).is_err());
                let used = account.usage().unwrap().allocated_bytes();
                assert_eq!(
                    used != 0,
                    armed.is_ok(),
                    "fresh and reusable policies cannot mix"
                );
                close(initialized.into_plan_runtime().unwrap());
            }
            Err(failure) => {
                // A hook which observed metadata preparation still in flight
                // refuses before writes. The armer may finish in the meantime;
                // this stale observer must not overwrite its newer state.
                assert!(!failure.is_indeterminate());
                assert_eq!(account.usage().unwrap().allocated_bytes(), 0);
                assert!(source.prepare_reusable_static_backing(&plan).is_err());
                assert!(source
                    .component(&prepared.family().stored_weight_schema().unwrap().components[0])
                    .is_err());
                drop(failure.recover().unwrap());
            }
        }
    });
    drop(source);
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn reusable_static_source_rejects_existing_process_mapped_policy_without_allocating() {
    let fixture = Fixture::new();
    let prepared = prepare(fixture.open().unwrap());
    let mut mapped = prepared.managed_safetensors().unwrap();
    let (runtime, plan) = compile(&prepared);
    let directory = tempfile::tempdir().unwrap();
    mapped
        .configure_process_mapped_weights(
            directory.path().to_owned(),
            ferrum_quantization::ProcessMappedDenseLimits {
                maximum_file_bytes: 1 << 20,
                maximum_metadata_bytes: 1 << 18,
            },
        )
        .unwrap();
    let account = host(1 << 18);
    let source = mapped.weight_source(&account);
    assert!(source.prepare_reusable_static_backing(&plan).is_err());
    assert_eq!(account.usage().unwrap().committed_bytes(), 0);
    drop(runtime);
}
