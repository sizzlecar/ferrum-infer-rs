//! Actual CPU embedding/head installation, attachment and token lifetimes.
use super::*;
use ferrum_execution::{
    driver::{ProgramSharedWeightsReceipt, SharedProgramWeightsCloseOutcome},
    PreparedExecutionInput, ProgramExecutorConfig, RuntimeComposition,
};
use ferrum_kernels::backend::cpu::{
    vnext_ops::CpuVNextComposition, vnext_runtime::CpuDeviceRuntime,
};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct OriginalSource<'a> {
    original: &'a dyn WeightComponentSource,
    reads: AtomicUsize,
    reject_reads: AtomicBool,
}
impl WeightComponentSource for OriginalSource<'_> {
    fn component<'a>(
        &'a self,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        if self.reject_reads.load(Ordering::SeqCst) {
            return Err(VNextError::InvalidExecutionPlan {
                reason: "shared attachment must not reopen the original payload".into(),
            });
        }
        self.original.component(component)
    }
}

fn composition(name: &str) -> (Arc<CpuDeviceRuntime>, RuntimeComposition<CpuDeviceRuntime>) {
    let (runtime, registry, materializers, _, catalog) =
        CpuVNextComposition::create(id(name), 32 << 20)
            .unwrap()
            .into_parts()
            .unwrap();
    let composition = RuntimeComposition::new(runtime.clone(), registry, materializers, catalog);
    (runtime, composition)
}

#[tokio::test]
async fn shared_token_installation_uses_one_static_claim_and_survives_peer_cancellation() {
    let (runtime, composition) = composition("device.fixture.shared-token-life");
    let (directory, (first, second, weights, usable)) = Fixture::<CpuDeviceRuntime>::with_prepared(
        SamplingReadbackMode::Uniform,
        Device::CPU,
        FixtureOptions::default(),
        |_| {},
        |prepared, info, engine, registry, definition| {
            let original = prepared.execution_input().unwrap();
            let source = OriginalSource {
                original: original.weights(),
                reads: AtomicUsize::new(0),
                reject_reads: AtomicBool::new(false),
            };
            let input = PreparedExecutionInput::new(
                original.family(),
                &source,
                *original.facts(),
                original.original_sources(),
                original.resolved_sources(),
            )
            .unwrap();
            let config =
                ProgramExecutorConfig::from_engine_config(engine, &info, runtime.as_ref())?;
            let compile = || {
                composition.compile_model(
                    input,
                    info.clone(),
                    engine,
                    config.clone(),
                    WeightMaterializerSelection::exact(id(IDENTITY_WEIGHT_MATERIALIZER_ID)),
                )
            };
            let resolve = |policy: &ResolvedRuntimePolicy,
                           catalog: &CapabilityCatalog,
                           compilation: &ProgramPlanCompilation| {
                resolve_plan(
                    prepared,
                    policy,
                    catalog,
                    compilation,
                    registry,
                    definition,
                    ferrum_types::KvStorageFormat::F16,
                    StopPolicy {
                        maximum_output_tokens: 8,
                        token_ids: BTreeSet::from([2]),
                        strings: vec![],
                        collision_policy: StopTokenCollisionPolicy::require_distinct(),
                    },
                )
            };
            let (first, weights) = compile()?.initialize_shared_weights(&resolve)?;
            let cold_reads = source.reads.load(Ordering::SeqCst);
            assert!(
                cold_reads > 0,
                "the published owner must have read actual weights"
            );
            assert_eq!(
                runtime.resident_bytes(),
                weights.evidence().physical_bytes()
            );
            source.reject_reads.store(true, Ordering::SeqCst);
            // Different token work capacity produces a distinct plan, while
            // retaining the same source, weight layout and device ceilings.
            let mut second_engine = engine.clone();
            second_engine.batching.max_num_batched_tokens = 8;
            let second_config =
                ProgramExecutorConfig::from_engine_config(&second_engine, &info, runtime.as_ref())?;
            let second = composition
                .compile_model(
                    input,
                    info.clone(),
                    &second_engine,
                    second_config,
                    WeightMaterializerSelection::exact(id(IDENTITY_WEIGHT_MATERIALIZER_ID)),
                )?
                .initialize_with_shared_weights(&weights, &resolve)?;
            assert_ne!(
                first.resolved_plan().execution_plan().plan_hash(),
                second.resolved_plan().execution_plan().plan_hash()
            );
            assert_eq!(source.reads.load(Ordering::SeqCst), cold_reads);
            assert_eq!(
                runtime.resident_bytes(),
                weights.evidence().physical_bytes()
            );
            assert!(matches!(
                first.shared_weights_receipt(),
                Some(ProgramSharedWeightsReceipt::Published { .. })
            ));
            assert!(matches!(
                second.shared_weights_receipt(),
                Some(ProgramSharedWeightsReceipt::Attached { .. })
            ));
            let attached =
                ferrum_execution::ModelExecutor::execution_attribution_snapshot(&second).unwrap();
            assert_eq!(attached["kind"], "attached");
            assert!(
                attached.get("completed_components").is_none(),
                "attachment is not another initialization receipt"
            );
            let usable = first
                .resolved_plan()
                .execution_plan()
                .payload()
                .memory()
                .usable_capacity_bytes();
            let capacity = DeviceCapacityOwner::open(runtime.descriptor(), usable).unwrap();
            let static_bytes = weights.evidence().physical_bytes();
            drop(
                capacity
                    .reserve(DeviceOpaqueMemoryKind::Bootstrap, usable - static_bytes)
                    .unwrap(),
            );
            assert!(capacity
                .reserve(DeviceOpaqueMemoryKind::Bootstrap, usable - static_bytes + 1)
                .is_err());
            Ok((first, second, weights, usable))
        },
    )
    .unwrap();
    ferrum_execution::test_support::shared_weights::assert_shared_token_lifecycle(
        first, second, weights,
    )
    .await;
    assert_eq!(runtime.resident_bytes(), 0);
    let capacity = DeviceCapacityOwner::open(runtime.descriptor(), usable).unwrap();
    drop(
        capacity
            .reserve(DeviceOpaqueMemoryKind::Bootstrap, usable)
            .unwrap(),
    );
    drop(directory);
}

#[test]
fn shared_token_installation_rejects_foreign_layout_and_runtime_before_allocation() {
    let (runtime, composition) = composition("device.fixture.shared-token-reject");
    let (directory, ()) = Fixture::<CpuDeviceRuntime>::with_prepared(
        SamplingReadbackMode::Uniform,
        Device::CPU,
        FixtureOptions::default(),
        |_| {},
        |prepared, info, engine, registry, definition| {
            let original = prepared.execution_input().unwrap();
            let source = OriginalSource {
                original: original.weights(),
                reads: AtomicUsize::new(0),
                reject_reads: AtomicBool::new(false),
            };
            let input = PreparedExecutionInput::new(
                original.family(),
                &source,
                *original.facts(),
                original.original_sources(),
                original.resolved_sources(),
            )
            .unwrap();
            let config =
                ProgramExecutorConfig::from_engine_config(engine, &info, runtime.as_ref())?;
            let resolve = |policy: &ResolvedRuntimePolicy,
                           catalog: &CapabilityCatalog,
                           compilation: &ProgramPlanCompilation| {
                resolve_plan(
                    prepared,
                    policy,
                    catalog,
                    compilation,
                    registry,
                    definition,
                    ferrum_types::KvStorageFormat::F16,
                    StopPolicy {
                        maximum_output_tokens: 8,
                        token_ids: BTreeSet::from([2]),
                        strings: vec![],
                        collision_policy: StopTokenCollisionPolicy::require_distinct(),
                    },
                )
            };
            let (first, weights) = composition
                .compile_model(
                    input,
                    info.clone(),
                    engine,
                    config.clone(),
                    WeightMaterializerSelection::exact(id(IDENTITY_WEIGHT_MATERIALIZER_ID)),
                )?
                .initialize_shared_weights(&resolve)?;
            source.reject_reads.store(true, Ordering::SeqCst);
            let reads = source.reads.load(Ordering::SeqCst);
            let before = runtime.resident_bytes();
            let mut repacked = config.clone();
            repacked.weight_arena_packing = WeightArenaPacking::Bounded {
                target_bytes: std::num::NonZeroU64::new(32).unwrap(),
            };
            let bad = composition.compile_model(
                input,
                info.clone(),
                engine,
                repacked,
                WeightMaterializerSelection::exact(id(IDENTITY_WEIGHT_MATERIALIZER_ID)),
            )?;
            assert!(bad
                .initialize_with_shared_weights(&weights, &resolve)
                .is_err());
            assert_eq!(runtime.resident_bytes(), before);
            assert_eq!(source.reads.load(Ordering::SeqCst), reads);

            // A second actual archive has identical tensor names/schema but
            // different bytes. Its valid product evidence must not be relabeled
            // as the cold installation merely because the layouts agree.
            let (other_directory, ()) = Fixture::<CpuDeviceRuntime>::with_prepared(
                SamplingReadbackMode::Uniform,
                Device::CPU,
                FixtureOptions {
                    nonfinite_token: Some(0),
                    ..FixtureOptions::default()
                },
                |_| {},
                |other_prepared, other_info, other_engine, other_registry, other_definition| {
                    let bad = composition.compile_model(
                        other_prepared.execution_input().unwrap(),
                        other_info,
                        other_engine,
                        config.clone(),
                        WeightMaterializerSelection::exact(id(IDENTITY_WEIGHT_MATERIALIZER_ID)),
                    )?;
                    assert_eq!(
                        bad.compilation()
                            .executable()
                            .execution_plan()
                            .payload()
                            .execution_weights(),
                        first
                            .resolved_plan()
                            .execution_plan()
                            .payload()
                            .execution_weights()
                    );
                    let result = bad.initialize_with_shared_weights(
                        &weights,
                        |policy, catalog, compilation| {
                            resolve_plan(
                                other_prepared,
                                policy,
                                catalog,
                                compilation,
                                other_registry,
                                other_definition,
                                ferrum_types::KvStorageFormat::F16,
                                StopPolicy {
                                    maximum_output_tokens: 8,
                                    token_ids: BTreeSet::from([2]),
                                    strings: vec![],
                                    collision_policy: StopTokenCollisionPolicy::require_distinct(),
                                },
                            )
                        },
                    );
                    assert!(result.is_err());
                    assert_eq!(runtime.resident_bytes(), before);
                    Ok(())
                },
            )?;
            drop(other_directory);

            // Equal device metadata is still not the original runtime instance.
            let (foreign, other) = self::composition("device.fixture.shared-token-reject");
            let bad = other.compile_model(
                input,
                info,
                engine,
                config,
                WeightMaterializerSelection::exact(id(IDENTITY_WEIGHT_MATERIALIZER_ID)),
            )?;
            assert!(bad
                .initialize_with_shared_weights(&weights, &resolve)
                .is_err());
            assert_eq!(foreign.resident_bytes(), 0);
            assert_eq!(runtime.resident_bytes(), before);
            assert_eq!(source.reads.load(Ordering::SeqCst), reads);
            ferrum_execution::test_support::assert_executor_closed(first);
            assert!(matches!(
                weights
                    .close()
                    .unwrap_or_else(|e| panic!("close: {:?}", e.failure())),
                SharedProgramWeightsCloseOutcome::Closed(_)
            ));
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(runtime.resident_bytes(), 0);
    drop(directory);
}
