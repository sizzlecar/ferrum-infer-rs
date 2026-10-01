//! Exercises the production CUDA numeric cache without a CUDA context/device.
use super::*;
use crate::backend::cpu::vnext_ops::CpuVNextComposition;
use ferrum_interfaces::vnext::{
    DeviceCostGraphCatalogLimits, DeviceId, DeviceRuntime, ExecutionLane,
    ReusableExecutionBucketSpec, ReusableExecutionCapacity, ReusableExecutionClassId,
};

struct Fixture {
    cache: CudaExecutableCache,
    fingerprint: String,
    lane: ExecutionLaneId,
    budget: Arc<DeviceObservationTemplateBudget>,
}
impl Fixture {
    fn new() -> Self {
        let composition = CpuVNextComposition::create(
            DeviceId::new("device.cpu.prepared-cuda-directory").unwrap(),
            1024 * 1024,
        )
        .unwrap();
        let lane = ExecutionLane::create(Arc::clone(composition.runtime())).unwrap();
        let mut fixture = Self {
            cache: CudaExecutableCache::new(),
            fingerprint: composition
                .runtime()
                .descriptor()
                .runtime_implementation_fingerprint
                .clone(),
            lane: lane.id(),
            budget: DeviceObservationTemplateBudget::new(1024 * 1024).unwrap(),
        };
        fixture.bind();
        fixture
    }
    fn bind(&mut self) {
        self.cache
            .bind_cost_catalog_lane(1, 1, &self.fingerprint, self.lane)
            .unwrap();
    }
    fn configure(&mut self) {
        self.cache
            .configure(DeviceReusableExecutionPlan::new(4).unwrap())
            .unwrap();
    }
    fn capture(&self, slot: u64) -> DeviceReusableExecutionCapture {
        let bucket = ReusableExecutionBucketSpec::new(
            ReusableExecutionClassId::new("prepared-cuda-directory").unwrap(),
            ReusableExecutionCapacity::new(1, 1, 1).unwrap(),
        )
        .unwrap();
        DeviceReusableExecutionCapture::new(
            DeviceReusableExecutionProgramId::new(
                serde_json::from_value(serde_json::json!("a".repeat(64))).unwrap(),
                self.fingerprint.clone(),
                self.lane,
                bucket.bucket_id().clone(),
                "c".repeat(64),
                "d".repeat(64),
                slot,
                1,
                1,
                1,
            )
            .unwrap(),
            3,
            vec![],
            vec![],
        )
        .unwrap()
    }
    fn register(&mut self, capture: &DeviceReusableExecutionCapture) {
        self.cache
            .register_program(
                capture,
                &[],
                &[],
                &[],
                &[],
                &CudaExecutablePreparation::default(),
            )
            .unwrap();
    }
    fn publish(&mut self) {
        self.cache.publish_prepared_cost_catalog(
            1,
            1,
            &self.fingerprint,
            &self.budget,
            &mut || Ok(()),
        );
    }
    fn available(&self) -> DevicePreparedCostGraphCatalogAvailability {
        self.cache
            .cost_prepared_reusable_graph_catalog(1, 1, &self.fingerprint, &mut || Ok(()))
            .unwrap()
    }
    fn root(&self) -> Arc<DevicePreparedCostGraphCatalog> {
        match self.available() {
            DevicePreparedCostGraphCatalogAvailability::Ready(root) => root,
            state => panic!("expected prepared root, got {state:?}"),
        }
    }
    fn current(&self, root: &DevicePreparedCostGraphCatalog) -> bool {
        root.is_current(
            self.cache.prepared_catalog.source.as_ref().unwrap(),
            self.cache.cost_graph_stream_state().unwrap(),
        )
    }
}

#[test]
fn cuda_prepared_catalog_raw_stream_is_unprepared_and_bound_empty_is_complete() {
    let mut fixture = Fixture::new();
    let mut raw = CudaExecutableCache::new();
    raw.publish_prepared_cost_catalog(1, 1, &fixture.fingerprint, &fixture.budget, &mut || Ok(()));
    assert!(matches!(
        raw.cost_prepared_reusable_graph_catalog(1, 1, &fixture.fingerprint, &mut || Ok(())),
        Ok(DevicePreparedCostGraphCatalogAvailability::Unprepared)
    ));
    assert_eq!(fixture.budget.retained_payload_bytes(), 0);

    fixture.publish();
    let empty = fixture.root();
    assert!(empty.catalog().programs().is_empty());
    assert!(empty
        .lookup(
            fixture.capture(1).program_id(),
            DeviceCostGraphCatalogLimits::new(1, 3, 3).unwrap(),
            &mut || Ok(()),
        )
        .unwrap()
        .is_none());
    fixture.bind();
    fixture.publish();
    assert!(Arc::ptr_eq(&empty, &fixture.root()));
    assert!(matches!(
        fixture
            .cache
            .cost_prepared_reusable_graph_catalog(1, 2, &fixture.fingerprint, &mut || Ok(())),
        Ok(DevicePreparedCostGraphCatalogAvailability::Unprepared)
    ));

    let other = Fixture::new();
    assert!(fixture
        .cache
        .bind_cost_catalog_lane(1, 1, &fixture.fingerprint, other.lane)
        .is_err());
    assert!(!fixture.current(&empty));
    assert!(matches!(
        fixture.available(),
        DevicePreparedCostGraphCatalogAvailability::Unprepared
    ));
}

#[test]
fn cuda_prepared_catalog_all_programs_retained_but_query_bounds_only_selected_program() {
    let mut fixture = Fixture::new();
    fixture.configure();
    let a = fixture.capture(1);
    let b = fixture.capture(2);
    fixture.register(&a);
    fixture.register(&b);
    // Reproduce the old aggregate coupling through the unchanged real producer.
    assert!(fixture
        .cache
        .cost_reusable_graph_catalog(
            DeviceCostGraphCatalogLimits::new(2, 5, 6).unwrap(),
            &mut || Ok(())
        )
        .is_err());
    assert!(matches!(
        fixture.available(),
        DevicePreparedCostGraphCatalogAvailability::Unprepared
    ));
    fixture.publish();
    let root = fixture.root();
    assert_eq!(root.catalog().programs().len(), 2);
    for capture in [&a, &b] {
        let selected = root
            .lookup(
                capture.program_id(),
                DeviceCostGraphCatalogLimits::new(1, 3, 3).unwrap(),
                &mut || Ok(()),
            )
            .unwrap()
            .unwrap();
        assert_eq!(selected.program().program_id(), capture.program_id());
    }
    assert!(root
        .lookup(
            a.program_id(),
            DeviceCostGraphCatalogLimits::new(1, 2, 3).unwrap(),
            &mut || Ok(())
        )
        .is_err());
    fixture.register(&a);
    fixture.cache.tick();
    fixture.publish();
    assert!(
        Arc::ptr_eq(&root, &fixture.root()),
        "identical registration and LRU clock are not semantic mutations"
    );
}

#[test]
fn cuda_prepared_catalog_failed_registration_revokes_root_without_changing_native_registry() {
    let mut fixture = Fixture::new();
    fixture.configure();
    let capture = fixture.capture(1);
    fixture.register(&capture);
    fixture.publish();
    let old = fixture.root();
    let bad = DeviceReusableExecutionCapture::new(capture.program_id().clone(), 4, vec![], vec![])
        .unwrap();
    assert!(fixture
        .cache
        .register_program(
            &bad,
            &[],
            &[],
            &[],
            &[],
            &CudaExecutablePreparation::default(),
        )
        .is_err());
    assert!(!fixture.current(&old));
    assert!(matches!(
        fixture.available(),
        DevicePreparedCostGraphCatalogAvailability::Unprepared
    ));
    assert_eq!(
        fixture.cache.programs[capture.program_id()]
            .descriptor
            .node_count(),
        3
    );
    assert_eq!(old.catalog().programs()[0].program().node_count(), 3);
    fixture.publish();
    assert!(!Arc::ptr_eq(&old, &fixture.root()));
    assert!(!fixture.current(&old));
}

#[test]
fn cuda_prepared_catalog_configuration_seal_rejection_and_trim_advance_proof() {
    let mut fixture = Fixture::new();
    fixture.publish();
    let unconfigured = fixture.root();
    fixture.configure();
    assert!(!fixture.current(&unconfigured));
    fixture.publish();
    let preparing = fixture.root();
    fixture.cache.seal().unwrap();
    assert!(!fixture.current(&preparing));
    fixture.publish();
    let sealed = fixture.root();
    // A rejected configuration is a failed mutator, never a retained proof.
    assert!(fixture
        .cache
        .configure(DeviceReusableExecutionPlan::new(4).unwrap())
        .is_err());
    assert!(!fixture.current(&sealed));
    fixture.publish();
    let after_rejection = fixture.root();
    fixture.cache.trim_quiescent();
    assert!(
        !fixture.current(&after_rejection),
        "empty trim still retires its numeric revision"
    );
    fixture.publish();
    assert!(fixture.root().catalog().programs().is_empty());
}

#[test]
fn cuda_prepared_catalog_build_failure_is_unprepared_and_never_lazily_retried_by_query() {
    let mut fixture = Fixture::new();
    fixture.configure();
    let capture = fixture.capture(1);
    fixture.register(&capture);
    fixture.budget = DeviceObservationTemplateBudget::new(1).unwrap();
    fixture.publish();
    assert!(matches!(
        fixture.cache.prepared_catalog.failure.as_ref(),
        Some(PreparationFailure::BuildRejected {
            error: DevicePreparedCostGraphCatalogBuildError::Capacity { .. },
            ..
        })
    ));
    assert_eq!(
        fixture.cache.programs.len(),
        1,
        "optional metadata failure preserves native registration"
    );
    for _ in 0..3 {
        assert!(matches!(
            fixture.available(),
            DevicePreparedCostGraphCatalogAvailability::Unprepared
        ));
    }
    // Full passive equality preserves the unpublished revision and its quota
    // receipt. Numeric Eq alone is insufficient (see library-template test).
    fixture.register(&capture);
    assert!(matches!(
        fixture.cache.prepared_catalog.retry,
        PreparationRetry::Capacity { .. }
    ));
    fixture.cache.publish_prepared_cost_catalog(
        1,
        1,
        &fixture.fingerprint,
        &fixture.budget,
        &mut || {
            panic!("same failed revision must not be rebuilt without released metadata capacity")
        },
    );
    assert_eq!(fixture.budget.retained_payload_bytes(), 0);
}

#[test]
fn cuda_prepared_catalog_cancelled_full_build_keeps_error_and_never_publishes_partial_absence() {
    let mut fixture = Fixture::new();
    fixture.configure();
    let a = fixture.capture(1);
    let b = fixture.capture(2);
    fixture.register(&a);
    fixture.register(&b);
    let mut remaining = 3;
    fixture.cache.publish_prepared_cost_catalog(
        1,
        1,
        &fixture.fingerprint,
        &fixture.budget,
        &mut || {
            if remaining == 0 {
                Err(VNextError::InvalidExecutionPlan {
                    reason: "test preparation cancelled".into(),
                })
            } else {
                remaining -= 1;
                Ok(())
            }
        },
    );
    match fixture.cache.prepared_catalog.failure.as_ref() {
        Some(PreparationFailure::BuildRejected {
            error:
                DevicePreparedCostGraphCatalogBuildError::Rejected(VNextError::InvalidExecutionPlan {
                    reason,
                }),
            ..
        }) => assert_eq!(reason, "test preparation cancelled"),
        failure => panic!("original typed error lost: {failure:?}"),
    }
    assert!(matches!(
        fixture.available(),
        DevicePreparedCostGraphCatalogAvailability::Unprepared
    ));
    assert_eq!(fixture.budget.retained_payload_bytes(), 0);
    assert!(matches!(
        fixture.cache.prepared_catalog.retry,
        PreparationRetry::NextBoundary
    ));
    // Cancellation keeps its original typed reason and may retry at the next
    // explicit preparation boundary; it is never a capacity receipt.
    fixture.publish();
    assert_eq!(fixture.root().catalog().programs().len(), 2);
}

#[test]
fn cuda_prepared_catalog_old_snapshot_holds_lease_after_native_trim() {
    let mut fixture = Fixture::new();
    fixture.configure();
    let capture = fixture.capture(1);
    fixture.register(&capture);
    fixture.publish();
    let old = fixture.root();
    let retained = fixture.budget.retained_payload_bytes();
    assert!(retained > 0);
    fixture.cache.trim_quiescent();
    assert_eq!(fixture.budget.retained_payload_bytes(), retained);
    assert!(!fixture.current(&old));
    fixture.publish();
    let empty = fixture.root();
    assert!(empty.catalog().programs().is_empty());
    assert_eq!(old.catalog().programs().len(), 1);
    assert!(fixture.budget.retained_payload_bytes() > retained);
    let budget = Arc::clone(&fixture.budget);
    drop(fixture);
    drop(empty);
    assert_eq!(budget.retained_payload_bytes(), retained);
    drop(old);
    assert_eq!(budget.retained_payload_bytes(), 0);
}

#[test]
fn cuda_prepared_catalog_query_cancellation_does_not_return_ready_or_absence() {
    let mut fixture = Fixture::new();
    fixture.publish();
    let mut first = true;
    assert!(fixture
        .cache
        .cost_prepared_reusable_graph_catalog(1, 1, &fixture.fingerprint, &mut || {
            if std::mem::replace(&mut first, false) {
                Ok(())
            } else {
                Err(VNextError::InvalidExecutionPlan {
                    reason: "query deadline".into(),
                })
            }
        },)
        .is_err());
    assert!(fixture.current(&fixture.root()));
}

#[test]
fn cuda_prepared_catalog_retries_at_preparation_after_other_snapshot_capacity_is_released() {
    let mut fixture = Fixture::new();
    fixture.configure();
    let capture = fixture.capture(1);
    fixture.register(&capture);
    let probe = fixture.budget.reserve(1).unwrap();
    let overhead = fixture.budget.retained_payload_bytes() - 1;
    drop(probe);
    let previous_owner = fixture
        .budget
        .reserve(fixture.budget.maximum_bytes() - overhead - 1)
        .unwrap();
    fixture.publish();
    assert!(matches!(
        fixture.available(),
        DevicePreparedCostGraphCatalogAvailability::Unprepared
    ));
    drop(previous_owner);
    fixture.publish();
    assert_eq!(fixture.root().catalog().programs().len(), 1);
}

fn program_with_library_template(
    capture: &DeviceReusableExecutionCapture,
    library_policy: u8,
) -> CudaExecutableProgram {
    use ferrum_interfaces::execution_cost::{
        LibraryApiNumericWorkV1, LibraryReplayParametersV1, SelectedAlgorithmClassV1,
        SelectedCommandCostBuilderV1,
    };
    use ferrum_interfaces::vnext::{DeviceBatchingForm, DeviceNativeOperationId};
    let segment = DeviceReusableExecutionSegment::new(0, 0, 3, 3).unwrap();
    let descriptor =
        DeviceReusableExecutionProgram::new(capture, vec![segment.clone()], vec![], vec![])
            .unwrap();
    let mut evidence = SelectedCommandCostBuilderV1::new_with_algorithm_work(1);
    evidence
        .library_call_with_replay_parameters(
            SelectedAlgorithmClassV1::library_api(
                "test.vendor.GemmEx",
                1,
                [library_policy; 32],
                [2; 32],
            )
            .unwrap(),
            LibraryApiNumericWorkV1 {
                output_elements: 64,
                reduction_units_per_output: 32,
            },
            LibraryReplayParametersV1 {
                fixed_parameters: &[1, 64, 32],
            },
        )
        .unwrap();
    let template =
        SelectedReplayAlgorithmTemplateV1::from_selected(&evidence.finish().unwrap(), 1, 1, 0)
            .unwrap();
    let rows: Arc<[DeviceReplayedLogicalCommandAttribution]> = (0..3)
        .map(|n| {
            DeviceReplayedLogicalCommandAttribution::new(
                n,
                n,
                DeviceNativeOperationId::new("test.prepared-catalog.kernel").unwrap(),
                DeviceBatchingForm::Scalar,
                1,
                1,
                1,
                0,
                1,
            )
            .unwrap()
            .with_captured_replay_template(Some(template))
        })
        .collect::<Vec<_>>()
        .into();
    CudaExecutableProgram {
        descriptor,
        segments: vec![CudaExecutableProgramSegment {
            descriptor: segment.clone(),
            key: CudaExecutableSegmentKey([1; 32]),
            reusable_executable_fingerprint: Arc::from("e".repeat(64)),
            logical_commands: Some(Arc::new(
                DeviceReplayedCommandCatalogue::new(segment, rows).unwrap(),
            )),
        }],
    }
}

#[test]
fn cuda_prepared_catalog_passive_library_change_retires_equal_execution_metadata() {
    let mut fixture = Fixture::new();
    fixture.configure();
    let capture = fixture.capture(1);
    let before = program_with_library_template(&capture, 1);
    let after = program_with_library_template(&capture, 2);
    assert!(
        before == after,
        "execution equality intentionally ignores passive templates"
    );
    assert!(!before.same_cost_catalog_metadata(&after));
    // The production registration commit receives already validated numeric
    // descriptors. No fabricated graph handle is needed to test its identity
    // rule: absent uploaded resources remain absent from the published root.
    fixture
        .cache
        .commit_registered_program(capture.program_id(), before.clone())
        .unwrap();
    fixture.publish();
    let old = fixture.root();
    assert!(old.catalog().programs()[0].uploaded_segments().is_empty());
    fixture
        .cache
        .commit_registered_program(capture.program_id(), before)
        .unwrap();
    fixture.publish();
    assert!(Arc::ptr_eq(&old, &fixture.root()));
    fixture
        .cache
        .commit_registered_program(capture.program_id(), after)
        .unwrap();
    assert!(!fixture.current(&old));
    assert!(matches!(
        fixture.available(),
        DevicePreparedCostGraphCatalogAvailability::Unprepared
    ));
    fixture.publish();
    assert!(!old.same_generation(&fixture.root()));
    assert_eq!(
        old.catalog().stream_state(),
        fixture.root().catalog().stream_state()
    );
}

#[test]
fn cuda_prepared_catalog_capacity_receipt_survives_release_before_failure_is_recorded() {
    use ferrum_interfaces::vnext::DeviceObservationTemplateReservation;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Mutex,
    };

    // A deterministic interleaving at the original failure-report boundary.
    // The other owner releases after quota rejection/rollback, before the cache
    // records its retry state. Recording retained-after-failure would miss this
    // recovery because both that reading and the next reading would be zero.
    struct ReleaseOnFailure {
        owner: Mutex<Option<DeviceObservationTemplateReservation>>,
        failures: Arc<AtomicUsize>,
    }
    impl tracing::Subscriber for ReleaseOnFailure {
        fn enabled(&self, metadata: &tracing::Metadata<'_>) -> bool {
            metadata.target() == "ferrum::cost_catalog_diagnostics"
        }
        fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
            tracing::span::Id::from_u64(1)
        }
        fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
        fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
        fn enter(&self, _: &tracing::span::Id) {}
        fn exit(&self, _: &tracing::span::Id) {}
        fn event(&self, event: &tracing::Event<'_>) {
            if self.enabled(event.metadata()) {
                self.failures.fetch_add(1, Ordering::SeqCst);
                drop(self.owner.lock().unwrap().take());
            }
        }
        fn max_level_hint(&self) -> Option<tracing::level_filters::LevelFilter> {
            Some(tracing::level_filters::LevelFilter::DEBUG)
        }
    }

    let mut fixture = Fixture::new();
    fixture.configure();
    let capture = fixture.capture(1);
    fixture.register(&capture);
    let probe = fixture.budget.reserve(1).unwrap();
    let overhead = fixture.budget.retained_payload_bytes() - 1;
    drop(probe);
    let held = fixture
        .budget
        .reserve(fixture.budget.maximum_bytes() - overhead - 1)
        .unwrap();
    let failures = Arc::new(AtomicUsize::new(0));
    tracing::subscriber::with_default(
        ReleaseOnFailure {
            owner: Mutex::new(Some(held)),
            failures: Arc::clone(&failures),
        },
        || {
            fixture.publish();
            assert_eq!(failures.load(Ordering::SeqCst), 1);
            assert_eq!(fixture.budget.retained_payload_bytes(), 0);
            let required = match fixture.cache.prepared_catalog.retry {
                PreparationRetry::Capacity {
                    required_exclusive_peak_bytes,
                } => required_exclusive_peak_bytes,
                ref other => panic!("quota rejection lost its exclusive requirement: {other:?}"),
            };
            assert!(required > 1 && required <= fixture.budget.maximum_bytes());
            // A competitor can consume the recovered bytes before this next
            // writer boundary. There is still no query-time rebuild.
            let competitor = fixture
                .budget
                .reserve(fixture.budget.maximum_bytes() - overhead - 1)
                .unwrap();
            fixture.cache.publish_prepared_cost_catalog(
                1,
                1,
                &fixture.fingerprint,
                &fixture.budget,
                &mut || panic!("available metadata is below the captured exclusive requirement"),
            );
            assert_eq!(failures.load(Ordering::SeqCst), 1);
            drop(competitor);
            fixture.publish();
            assert_eq!(fixture.root().catalog().programs().len(), 1);
            assert_eq!(failures.load(Ordering::SeqCst), 1);
        },
    );
}
