//! Actual CPU embedding/head installation, attachment and token lifetimes.
use super::*;
use ferrum_execution::{
    driver::{ProgramSharedWeightsReceipt, SharedProgramWeights, SharedProgramWeightsCloseOutcome},
    PreparedExecutionInput, ProgramExecutorConfig, RuntimeComposition,
};
use ferrum_kernels::backend::cpu::{
    vnext_ops::CpuVNextComposition, vnext_runtime::CpuDeviceRuntime,
};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use ferrum_execution::model_executor::{LogitsReturnPolicy, PrefillChunk};
use ferrum_execution::token_execution::*;
use ferrum_execution::{
    ExecutorOwner, ExecutorShutdownPhase, ManagedCausalIoProfile, ModelExecutor,
};
use ferrum_types::{RequestId, TokenId};

struct ManagedSource<'a> {
    original: &'a dyn WeightComponentSource,
    host: HostMemoryAccount,
    reads: AtomicUsize,
    reject: AtomicBool,
}
impl WeightComponentSource for ManagedSource<'_> {
    fn component<'a>(
        &'a self,
        component: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        assert!(
            !self.reject.load(Ordering::SeqCst),
            "attachment reopened source"
        );
        self.reads.fetch_add(1, Ordering::SeqCst);
        let original = self.original.component(component)?;
        let mut bytes = self
            .host
            .allocate_zeroed(HostMemoryClass::WeightBacking, original.bytes().len())
            .map_err(|e| VNextError::InvalidExecutionPlan {
                reason: e.to_string(),
            })?;
        bytes.as_mut().copy_from_slice(original.bytes());
        WeightComponentPayload::from_managed_host(
            component,
            original.external_names().to_vec(),
            original.source_files().to_vec(),
            original.dimensions().to_vec(),
            original.element_type(),
            bytes.freeze(),
        )
    }
}
struct PhaseDrop(Arc<AtomicBool>);
impl Drop for PhaseDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::SeqCst);
    }
}
const PHASE_CLASS_BYTES: usize = 4096;
const PHASE_BYTES: usize = PHASE_CLASS_BYTES * 4;
fn managed_host() -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            PHASE_BYTES * 3,
            HostMemoryClassLimits {
                input_preprocess_bytes: PHASE_CLASS_BYTES * 3,
                transfer_staging_bytes: PHASE_CLASS_BYTES * 3,
                output_bytes: PHASE_CLASS_BYTES * 3,
                weight_backing_bytes: PHASE_CLASS_BYTES * 3,
            },
        )
        .unwrap(),
    )
}
fn managed_phase(host: &HostMemoryAccount, backing: usize) -> (HostMemoryScope, Arc<AtomicBool>) {
    (
        host.reserve_scope(&[
            (HostMemoryClass::InputPreprocess, PHASE_CLASS_BYTES),
            (HostMemoryClass::TransferStaging, PHASE_CLASS_BYTES),
            (HostMemoryClass::Output, PHASE_CLASS_BYTES),
            (HostMemoryClass::WeightBacking, backing),
        ])
        .unwrap(),
        Arc::new(AtomicBool::new(false)),
    )
}

// Uses the same authenticated tiny archive/CPU token family as the legacy
// shared tests. No new registry, fake resolved plan, or fabricated Host grant.
fn managed_pair(
    check_failures: bool,
) -> (
    tempfile::TempDir,
    Arc<CpuDeviceRuntime>,
    HostMemoryAccount,
    ExecutorOwner,
    ExecutorOwner,
    SharedProgramWeights<CpuDeviceRuntime>,
    [Arc<AtomicBool>; 2],
) {
    let (runtime, composition) = composition("device.fixture.managed-shared");
    let host = managed_host();
    let (directory, (first, second, weights, phases)) = Fixture::<CpuDeviceRuntime>::with_prepared(
        SamplingReadbackMode::Uniform,
        Device::CPU,
        FixtureOptions::default(),
        |_| {},
        |prepared, info, engine, registry, definition| {
            let original = prepared.execution_input().unwrap();
            let mut config =
                ProgramExecutorConfig::from_engine_config(engine, &info, runtime.as_ref())?;
            config.static_initialization = StaticInitializationPolicy::new(16, 1).unwrap();
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
            if check_failures {
                // Actual managed source allocation exceeds this original
                // phase by one byte, after static native acquisition began.
                let demand = host.backing_bytes(32).unwrap();
                let (phase, dropped) = managed_phase(&host, demand - 1);
                let scoped = phase.account();
                let source = ManagedSource {
                    original: original.weights(),
                    host: scoped.clone(),
                    reads: AtomicUsize::new(0),
                    reject: AtomicBool::new(false),
                };
                let input = PreparedExecutionInput::new(
                    original.family(),
                    &source,
                    *original.facts(),
                    original.original_sources(),
                    original.resolved_sources(),
                )?;
                let result = composition
                    .compile_model(
                        input,
                        info.clone(),
                        engine,
                        config.clone(),
                        WeightMaterializerSelection::exact(id(IDENTITY_WEIGHT_MATERIALIZER_ID)),
                    )?
                    .initialize_shared_weights_with_managed_io_retained(
                        &scoped,
                        ManagedCausalIoProfile::ManagedGreedyF32V1,
                        (phase, PhaseDrop(dropped.clone())),
                        &resolve,
                    );
                let error = result
                    .err()
                    .expect("underfunded managed source must fail cold publication");
                assert!(
                    error.cleanup().is_none(),
                    "CPU quiescent failure retires original transaction synchronously"
                );
                assert!(source.reads.load(Ordering::SeqCst) > 0);
                assert!(dropped.load(Ordering::SeqCst));
                assert_eq!(host.usage().unwrap().committed_bytes(), 0);
                assert_eq!(runtime.resident_bytes(), 0);
            }
            let (phase, first_dropped) = managed_phase(&host, PHASE_CLASS_BYTES);
            let scoped = phase.account();
            let source = ManagedSource {
                original: original.weights(),
                host: scoped.clone(),
                reads: AtomicUsize::new(0),
                reject: AtomicBool::new(false),
            };
            let input = PreparedExecutionInput::new(
                original.family(),
                &source,
                *original.facts(),
                original.original_sources(),
                original.resolved_sources(),
            )?;
            let compile = || {
                composition.compile_model(
                    input,
                    info.clone(),
                    engine,
                    config.clone(),
                    WeightMaterializerSelection::exact(id(IDENTITY_WEIGHT_MATERIALIZER_ID)),
                )
            };
            let (first, weights) = compile()?.initialize_shared_weights_with_managed_io_retained(
                &scoped,
                ManagedCausalIoProfile::ManagedGreedyF32V1,
                (phase, PhaseDrop(first_dropped.clone())),
                &resolve,
            )?;
            assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
            assert_eq!(host.usage().unwrap().committed_bytes(), PHASE_BYTES);
            assert_eq!(
                runtime.resident_bytes(),
                weights.evidence().physical_bytes()
            );
            let reads = source.reads.load(Ordering::SeqCst);
            assert!(reads > 0);
            source.reject.store(true, Ordering::SeqCst);
            if check_failures {
                let (phase, dropped) = managed_phase(&host, PHASE_CLASS_BYTES);
                let scoped = phase.account();
                let result = compile()?.initialize_with_shared_weights_and_managed_io_retained(
                    &weights,
                    &scoped,
                    ManagedCausalIoProfile::ManagedGreedyF16V1,
                    (phase, PhaseDrop(dropped.clone())),
                    &resolve,
                );
                assert!(
                    result.is_err(),
                    "F16 managed profile cannot attach to actual F32 logits"
                );
                assert!(dropped.load(Ordering::SeqCst));
                assert!(!first_dropped.load(Ordering::SeqCst));
                assert_eq!(host.usage().unwrap().committed_bytes(), PHASE_BYTES);
                assert_eq!(
                    runtime.resident_bytes(),
                    weights.evidence().physical_bytes()
                );
            }
            let (phase, second_dropped) = managed_phase(&host, PHASE_CLASS_BYTES);
            let scoped = phase.account();
            let second = compile()?.initialize_with_shared_weights_and_managed_io_retained(
                &weights,
                &scoped,
                ManagedCausalIoProfile::ManagedGreedyF32V1,
                (phase, PhaseDrop(second_dropped.clone())),
                &resolve,
            )?;
            assert_eq!(source.reads.load(Ordering::SeqCst), reads);
            assert_eq!(
                runtime.resident_bytes(),
                weights.evidence().physical_bytes(),
                "attachment has no second static allocation"
            );
            assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
            assert_eq!(host.usage().unwrap().committed_bytes(), 2 * PHASE_BYTES);
            let b = second
                .scope()
                .unwrap()
                .execution_attribution_snapshot()
                .unwrap();
            assert_eq!(b["kind"], "attached");
            assert_eq!(
                b["evidence"],
                serde_json::to_value(weights.evidence()).unwrap()
            );
            Ok((first, second, weights, [first_dropped, second_dropped]))
        },
    )
    .unwrap();
    (directory, runtime, host, first, second, weights, phases)
}
async fn managed_admit(executor: &dyn ModelExecutor, token: u32) -> ExecutionFrontier {
    let request = RequestId::new();
    let tokens = [TokenId::new(token)];
    loop {
        match executor
            .admit_tokens(ExecutionAdmission {
                context: ferrum_execution::ExecutorPrefillAdmission::for_diagnostic(
                    &request, &tokens, 16,
                ),
                adapter: None,
            })
            .await
            .unwrap()
        {
            TokenAdmissionOutcome::Admitted(frontier) => return frontier,
            TokenAdmissionOutcome::MaintenanceDeferred(_) => {
                assert!(matches!(
                    executor.maintain_prefill_backing(&request).unwrap(),
                    ferrum_execution::ExecutorPrefillMaintenanceOutcome::Maintained { .. }
                        | ferrum_execution::ExecutorPrefillMaintenanceOutcome::RetryAdmission { .. }
                ));
            }
            other => panic!("unexpected managed admission {other:?}"),
        }
    }
}
async fn managed_step(
    executor: &dyn ModelExecutor,
    frontier: &ExecutionFrontier,
    work: TokenWork,
    expected: u32,
) -> ExecutionCompletion {
    let policy = LogitsReturnPolicy::GreedyArgmax {
        token_mask: None,
        repetition_penalty: None,
    };
    let work = ExecutionWork::new(frontier, work, &policy).unwrap();
    let ExecutionBatchOutcome::Completed(batch) = executor.execute_tokens(&[work]).await.unwrap()
    else {
        panic!("managed work did not complete")
    };
    let mut completed = batch.into_completions();
    assert_eq!(completed.len(), 1);
    let completed = completed.pop().unwrap();
    let ExecutionProduct::Sampling(sampling) = completed.product() else {
        panic!("managed final work did not return its sampling product");
    };
    assert_eq!(
        sampling,
        &ferrum_execution::ExecutorSamplingOutput::GreedyToken(TokenId::new(expected))
    );
    completed
}
async fn managed_close(owner: &ExecutorOwner) {
    let report = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        owner.begin_shutdown().wait(),
    )
    .await
    .unwrap();
    assert_eq!(report.phase(), ExecutorShutdownPhase::Retired);
    report.result().unwrap();
}

#[tokio::test]
async fn managed_retained_shared_installation_keeps_one_physical_owner_and_isolates_cancellation() {
    let (directory, runtime, host, first, second, weights, phases) = managed_pair(false);
    let left = first.scope().unwrap();
    let right = second.scope().unwrap();
    let a = managed_admit(left.as_ref(), 0).await;
    let b = managed_admit(right.as_ref(), 3).await;
    let output = managed_step(
        right.as_ref(),
        &b,
        TokenWork::Prefill(PrefillChunk::new(0, 1, 1).unwrap()),
        1,
    )
    .await;
    assert!(matches!(
        left.cancel_tokens(a.sequence()).unwrap().disposition,
        ExecutionCloseDisposition::Ended(_)
    ));
    drop((a, left));
    managed_close(&first).await;
    assert!(phases[0].load(Ordering::SeqCst));
    assert!(!phases[1].load(Ordering::SeqCst));
    assert_eq!(host.usage().unwrap().committed_bytes(), PHASE_BYTES);
    let decoded = managed_step(
        right.as_ref(),
        output.next_frontier(),
        TokenWork::Decode {
            token: TokenId::new(1),
            position: 1,
        },
        0,
    )
    .await;
    assert!(matches!(
        right
            .finish_tokens(
                b.sequence(),
                ExecutionEndReason::Completed {
                    input_tokens: 1,
                    output_tokens: 1
                }
            )
            .await
            .unwrap()
            .disposition,
        ExecutionCloseDisposition::Ended(_)
    ));
    drop((decoded, output, b, right));
    managed_close(&second).await;
    assert!(phases[1].load(Ordering::SeqCst));
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(
        runtime.resident_bytes(),
        weights.evidence().physical_bytes()
    );
    assert!(matches!(
        weights
            .close()
            .unwrap_or_else(|e| panic!("close {:?}", e.failure())),
        SharedProgramWeightsCloseOutcome::Closed(_)
    ));
    assert_eq!(runtime.resident_bytes(), 0);
    drop((first, second, directory));
}

#[tokio::test]
async fn managed_retained_shared_build_failures_refund_failed_phase_without_poisoning_installation()
{
    let (directory, runtime, host, first, second, weights, phases) = managed_pair(true);
    let scope = second.scope().unwrap();
    let frontier = managed_admit(scope.as_ref(), 1).await;
    let output = managed_step(
        scope.as_ref(),
        &frontier,
        TokenWork::Prefill(PrefillChunk::new(0, 1, 1).unwrap()),
        0,
    )
    .await;
    assert!(matches!(
        scope
            .cancel_tokens(frontier.sequence())
            .unwrap()
            .disposition,
        ExecutionCloseDisposition::Ended(_)
    ));
    drop((output, frontier, scope));
    managed_close(&second).await;
    managed_close(&first).await;
    assert!(phases.iter().all(|p| p.load(Ordering::SeqCst)));
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert!(matches!(
        weights
            .close()
            .unwrap_or_else(|e| panic!("close {:?}", e.failure())),
        SharedProgramWeightsCloseOutcome::Closed(_)
    ));
    assert_eq!(runtime.resident_bytes(), 0);
    drop((first, second, directory));
}

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
