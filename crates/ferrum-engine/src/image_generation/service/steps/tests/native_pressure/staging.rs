//! Exact managed uploads through the existing real Stored transport fixture.
//! The source and physical destination remain separate from charged staging.
use super::*;
use crate::image_generation::host_budget::stored_staging_bytes;
use ferrum_execution::component::TensorComponentPolicy;
use ferrum_execution::component::{
    CompiledTensorComponent, ComponentInitializationConfig, ComponentInitializationError,
    ComponentWeightSource,
};
use ferrum_kernels::backend::reference::ReferenceVNextComposition;
use std::sync::atomic::{AtomicU64, AtomicUsize};

struct Source {
    bytes: [u8; 8],
    calls: AtomicUsize,
}
impl WeightComponentSource for Source {
    fn component<'a>(
        &'a self,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        WeightComponentPayload::new(
            component,
            "table",
            "typed-static-table",
            vec![4, 1],
            ElementType::Bf16,
            self.bytes.as_slice(),
        )
    }
}
fn family(rows: u64) -> PreparedModelFamily {
    TypedFamilyRegistration::new(Family {
        id: id("family.fixture.service.native_pressure"),
    })
    .prepare_with_profile(
        &serde_json::to_value(Config { rows }).unwrap(),
        &id(PROFILE),
    )
    .unwrap()
}
fn host() -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            64 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 0,
                output_bytes: 0,
                transfer_staging_bytes: 64 << 20,
                weight_backing_bytes: 0,
            },
        )
        .unwrap(),
        64,
    )
    .unwrap()
}
fn compiled(
    composition: &ReferenceVNextComposition,
    rows: u64,
) -> CompiledTensorComponent<Runtime> {
    let registry = ferrum_kernels::backend::reference::reference_vnext_operation_registry(
        composition.runtime(),
    )
    .unwrap();
    CompiledTensorComponent::compile(
        family(rows),
        &registry,
        composition.catalog(),
        &TensorComponentPolicy::new(
            1 << 20,
            composition
                .runtime()
                .descriptor()
                .dynamic_storage_profiles
                .iter()
                .copied()
                .collect::<Vec<_>>(),
        )
        .unwrap(),
        &WeightMaterializerRegistry::identity_only().unwrap(),
        &ProgramPlanCompileOptions::new(BTreeMap::from([
            (id(X), spec(1)),
            (id(G), spec(1)),
            (id(Y), spec(2)),
        ]))
        .unwrap(),
    )
    .unwrap()
}

#[test]
fn staged_upload_query_tiny_plan_exact_grant_initializes_and_one_byte_short_refuses() {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    for short in [false, true] {
        let composition = ReferenceVNextComposition::create_with_capacity(
            DeviceId::new(format!(
                "device.staging.query.{}",
                NEXT.fetch_add(1, Ordering::Relaxed)
            ))
            .unwrap(),
            1 << 20,
        )
        .unwrap();
        assert!(composition.runtime().requires_managed_static_staging());
        let component = compiled(&composition, 4);
        let host = host();
        let policy = StaticInitializationPolicy::new(64 << 20, 64).unwrap();
        let exact = stored_staging_bytes(
            &host,
            component
                .executable()
                .execution_plan()
                .payload()
                .execution_weights(),
            policy,
        )
        .unwrap();
        assert_eq!(
            exact,
            host.backing_bytes(8).unwrap(),
            "one actual separately padded chunk, not a full unused 64MiB allowance"
        );
        let scope = host
            .reserve_scope(&[(HostMemoryClass::TransferStaging, exact - usize::from(short))])
            .unwrap();
        let source = Source {
            bytes: [0, 0, 0x80, 0x3f, 0, 0xc0, 0x80, 0x3e],
            calls: AtomicUsize::new(0),
        };
        let initialized = component.initialize(
            composition.runtime().clone(),
            scope.account(),
            ComponentWeightSource::Stored(&source),
            ComponentInitializationConfig {
                request_id: id("request.staging.query"),
                run_id: id("run.staging.query"),
                transaction_id: id("transaction.staging.query"),
                static_policy: policy,
                executor: Default::default(),
            },
        );
        if short {
            let failure = initialized
                .err()
                .expect("one physical byte below the real first chunk must refuse");
            let ComponentInitializationError::Static(mut failure) = failure else {
                panic!("expected the original static initialization failure, got {failure}");
            };
            assert!(!failure.is_indeterminate());
            assert_eq!(
                source.calls.load(Ordering::SeqCst),
                0,
                "admit before source materialization and submission"
            );
            failure.retire_in_place().unwrap();
        } else {
            let session = initialized.unwrap_or_else(|e| panic!("exact managed upload grant: {e}"));
            let ferrum_execution::component::ComponentInitializationReceipt::Initialized {
                receipt,
            } = session.initialization_receipt()
            else {
                panic!("real Stored upload receipt required");
            };
            assert_eq!(receipt.uploaded_bytes(), 8);
            assert_eq!(source.calls.load(Ordering::SeqCst), 1);
            assert_eq!(
                host.usage()
                    .unwrap()
                    .class_allocated_bytes(HostMemoryClass::TransferStaging),
                0,
                "fenced transfer owners retire before successful return"
            );
            match session.close() {
                Ok(PlanRuntimeCloseOutcome::Closed(_)) => (),
                Ok(_) => panic!("original initialized root remains referenced"),
                Err(failure) => panic!("original close failed: {:?}", failure.failure()),
            }
        }
        drop(scope);
        assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        assert_eq!(composition.runtime().snapshot().live_allocations, 0);
    }
}

#[test]
fn staged_upload_query_keeps_original_cap_for_multiple_padded_chunks() {
    let composition = ReferenceVNextComposition::create_with_capacity(
        DeviceId::new("device.staging.query.multichunk").unwrap(),
        1 << 20,
    )
    .unwrap();
    let component = compiled(&composition, 64);
    let host = host();
    let policy = StaticInitializationPolicy::new(8, 2).unwrap();
    let old_cap = 8 + 2 * (host.allocation_alignment() - 1) * policy.maximum_commands_per_batch();
    let actual = stored_staging_bytes(
        &host,
        component
            .executable()
            .execution_plan()
            .payload()
            .execution_weights(),
        policy,
    )
    .unwrap();
    assert!(16 * host.backing_bytes(8).unwrap() > old_cap);
    assert_eq!(
        actual, old_cap,
        "a plan larger than its bounded batch retains the existing policy ceiling"
    );
    assert!(actual >= host.backing_bytes(8).unwrap());
    assert!(
        stored_staging_bytes(
            &host,
            component
                .executable()
                .execution_plan()
                .payload()
                .execution_weights(),
            StaticInitializationPolicy::new(1, 2).unwrap()
        )
        .is_err(),
        "cannot budget zero for a BF16 element that the policy cannot upload"
    );
    assert_eq!(composition.runtime().snapshot().live_allocations, 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
