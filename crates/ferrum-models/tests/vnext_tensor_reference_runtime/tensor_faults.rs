//! Faults change observations around real Reference computation. They do not
//! establish GPU-asynchronous behavior or model precision coverage.

use super::*;
use ferrum_kernels::backend::reference::{
    reference_vnext_dense_linear_provider, ReferenceDeviceBuffer, ReferenceDeviceCommand,
};
use std::sync::atomic::Ordering;

#[path = "fault_runtime.rs"]
mod runtime;
use runtime::{FaultRuntime, FenceMode, ReadbackMode};

struct ForwardProvider(Box<dyn OperationProvider<ReferenceDeviceRuntime>>);

impl OperationResourceEstimator for ForwardProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        self.0.descriptor()
    }
    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        self.0.estimate_resources(request)
    }
}

impl OperationProvider<FaultRuntime> for ForwardProvider {
    fn reusable_execution_topology(
        &self,
        request: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        self.0.reusable_execution_topology(request)
    }
    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, ReferenceDeviceBuffer>,
    ) -> Result<EncodedDeviceOperation<ReferenceDeviceCommand>, OperationFailure> {
        self.0.encode_selected(invocation)
    }
    fn encode_tensor_selected(
        &self,
        invocation: TensorOperationInvocation<'_, ReferenceDeviceBuffer>,
    ) -> Result<Option<EncodedDeviceOperation<ReferenceDeviceCommand>>, OperationFailure> {
        self.0.encode_tensor_selected(invocation)
    }
}

struct FaultFixture {
    runtime: Arc<FaultRuntime>,
    resources: Arc<PlanRuntimeResources<FaultRuntime>>,
    executor: PreparedTensorExecutor<FaultRuntime>,
    host: HostMemoryAccount,
    _source_directory: tempfile::TempDir,
}

impl FaultFixture {
    fn new() -> Self {
        let (source_directory, source, family) = source_and_family();
        let composition =
            ReferenceVNextComposition::create(id("device.reference.tensor-fault")).unwrap();
        let runtime = Arc::new(FaultRuntime::new(composition.runtime().clone()));
        let registry = OperationRuntimeRegistry::new(
            vec![Box::new(dense_linear_contract().unwrap())],
            vec![Box::new(ForwardProvider(
                reference_vnext_dense_linear_provider(&runtime.inner).unwrap(),
            )) as Box<dyn OperationProvider<FaultRuntime>>],
        )
        .unwrap();
        let engines = vec![EngineProviderDescriptor::new(
            id("provider.engine.reference.fault-observer"),
            ContractVersion::new(1, 0),
            runtime
                .descriptor()
                .runtime_implementation_fingerprint
                .clone(),
            runtime.descriptor().id.clone(),
            runtime.descriptor().capabilities.clone(),
        )
        .unwrap()];
        let catalog = registry
            .capability_catalog(runtime.descriptor().clone(), engines)
            .unwrap();
        let catalog = composition
            .weight_materializers()
            .augment_catalog(catalog)
            .unwrap();
        let policy = TensorPolicy {
            capacity_bytes: runtime.descriptor().total_memory_bytes,
            storage_profiles: runtime
                .descriptor()
                .dynamic_storage_profiles
                .iter()
                .copied()
                .collect(),
        };
        let mut options = ProgramPlanCompileOptions::new(BTreeMap::from([(
            id(INPUT_ID),
            ProgramTensorSpec {
                dimensions: vec![2, 3],
                element_type: ElementType::F16,
                layout: ResolvedTensorLayout::Contiguous,
            },
        )]))
        .unwrap();
        options.require_weight_materializer(composition.weight_materializer_id().clone());
        options.retain_completion_value(id(OUTPUT_ID));
        let compilation = ProgramPlanCompiler::compile_with_weight_materializers(
            &family,
            &catalog,
            &policy,
            &registry.planning(),
            composition.weight_materializers(),
            &options,
        )
        .unwrap();
        let executable = compilation.executable().clone();
        let resources = initialize_resources(
            runtime.clone(),
            &family,
            executable.execution_plan(),
            &source,
        );
        let providers = registry.bind_plan(&executable).unwrap();
        let host = host_account(20, 8);
        let executor = PreparedTensorExecutor::new(
            executable,
            providers,
            resources.clone(),
            host.clone(),
            TensorExecutorConfig::default(),
        )
        .unwrap();
        Self {
            runtime,
            resources,
            executor,
            host,
            _source_directory: source_directory,
        }
    }

    fn execute(
        &mut self,
    ) -> Result<Vec<ferrum_execution::TensorExecutionOutput>, TensorExecutionError<FaultRuntime>>
    {
        self.executor.execute(
            id("run.tensor.fault"),
            id("request.tensor.fault"),
            vec![typed_input(managed_input(&self.host))],
        )
    }

    fn assert_success(&mut self) {
        let outputs = self.execute().unwrap();
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].bytes().bytes(), f16_bytes(&EXPECTED_VALUES));
        assert_eq!(self.host.usage().unwrap().committed_bytes(), 8);
        drop(outputs);
        assert_eq!(self.host.usage().unwrap().committed_bytes(), 0);
        assert_eq!(self.executor.completion_reaper().retained_count(), 0);
    }

    fn close(self) {
        let Self {
            runtime,
            resources,
            executor,
            host,
            _source_directory,
        } = self;
        drop(executor);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        match PlanRuntimeResources::close(resources) {
            Ok(PlanRuntimeCloseOutcome::Closed(receipt)) => {
                assert_eq!(receipt.released_static_resources(), 1)
            }
            Ok(PlanRuntimeCloseOutcome::Referenced { strong_count, .. }) => {
                panic!("fault case retains {strong_count} plan owners")
            }
            Err(error) => panic!("fault case cannot close: {:?}", error.failure()),
        }
        assert_eq!(runtime.inner.snapshot().live_allocations, 0);
    }
}

fn assert_only_input_retained(fixture: &FaultFixture) {
    let usage = fixture.host.usage().unwrap();
    assert_eq!(usage.committed_bytes(), 12);
    assert_eq!(
        usage.class_committed_bytes(HostMemoryClass::InputPreprocess),
        12
    );
    assert_eq!(usage.class_committed_bytes(HostMemoryClass::Output), 0);
    assert_eq!(fixture.runtime.readback_calls.load(Ordering::SeqCst), 0);
    assert_eq!(fixture.runtime.inner.snapshot().dense_linear_launches, 1);
}

fn assert_idle_trim_busy_without_releasing(fixture: &FaultFixture) {
    let before = fixture.resources.dynamic_pool_status().unwrap();
    let native = fixture.runtime.inner.snapshot();
    assert!(matches!(
        fixture
            .resources
            .try_release_idle_dynamic_backing()
            .unwrap(),
        IdleDynamicReleaseOutcome::Busy(_)
    ));
    assert_eq!(fixture.resources.dynamic_pool_status().unwrap(), before);
    assert_eq!(
        fixture.runtime.inner.snapshot().live_allocations,
        native.live_allocations
    );
}

#[test]
fn fault_observer_proxy_preserves_the_real_reference_dense_oracle() {
    let mut fixture = FaultFixture::new();
    fixture.assert_success();
    assert_eq!(fixture.runtime.inner.snapshot().dense_linear_launches, 1);
    assert_eq!(fixture.runtime.readback_calls.load(Ordering::SeqCst), 1);
    fixture.close();
}

#[test]
fn indeterminate_and_pending_observations_retain_authority_until_terminal_recovery() {
    let mut fixture = FaultFixture::new();
    fixture.runtime.set_fence(FenceMode::Indeterminate);
    let mut error = fixture.execute().unwrap_err();
    assert_only_input_retained(&fixture);
    assert_idle_trim_busy_without_releasing(&fixture);
    let TensorExecutionError::NonTerminal {
        observation,
        completion,
        retirement,
        reaper,
    } = &mut error
    else {
        panic!("expected nonterminal ownership, received {error:?}")
    };
    assert!(matches!(
        observation,
        CompletionObservation::Indeterminate(_)
    ));
    assert_eq!(reaper.retained_count(), 1);
    assert!(retirement.try_abort().is_err());

    // execute uses blocking wait, which cannot express Pending. Poll the exact
    // retained completion handle to exercise the nonblocking Pending contract.
    fixture.runtime.set_fence(FenceMode::Pending);
    assert!(matches!(
        completion.poll().unwrap(),
        CompletionObservation::Pending
    ));
    assert!(retirement.try_abort().is_err());
    assert_eq!(reaper.retained_count(), 1);
    assert_only_input_retained(&fixture);

    assert_idle_trim_busy_without_releasing(&fixture);

    fixture.runtime.set_fence(FenceMode::Real);
    assert!(matches!(
        completion.wait().unwrap(),
        CompletionObservation::Terminal(_)
    ));
    retirement.try_abort().unwrap();
    retirement.try_abort().unwrap();
    assert_eq!(reaper.retained_count(), 0);
    assert_eq!(fixture.host.usage().unwrap().committed_bytes(), 0);
    assert!(matches!(
        fixture
            .resources
            .try_release_idle_dynamic_backing()
            .unwrap(),
        IdleDynamicReleaseOutcome::Released(_)
    ));
    // Keep the diagnostic error alive: retirement must release the single
    // execution slot and input owner, not merely promise release on Drop.
    fixture.assert_success();
    fixture.close();
    drop(error);
}

#[test]
fn failed_drain_quarantines_owners_until_a_later_successful_drain() {
    let mut fixture = FaultFixture::new();
    fixture.runtime.set_fence(FenceMode::Indeterminate);
    let mut error = fixture.execute().unwrap_err();
    let TensorExecutionError::NonTerminal {
        completion,
        retirement,
        reaper,
        ..
    } = &mut error
    else {
        panic!("expected nonterminal ownership, received {error:?}")
    };
    fixture.runtime.set_synchronize_fails(true);
    assert!(matches!(
        reaper
            .recover_slot_by_draining_lane(completion.slot_id())
            .unwrap(),
        CompletionRecoveryOutcome::Quarantined(_)
    ));
    assert_eq!(reaper.retained_count(), 1);
    assert_eq!(reaper.quarantined_count(), 1);
    assert!(retirement.try_abort().is_err());
    assert_only_input_retained(&fixture);

    assert_idle_trim_busy_without_releasing(&fixture);

    fixture.runtime.set_synchronize_fails(false);
    assert!(matches!(
        reaper
            .recover_slot_by_draining_lane(completion.slot_id())
            .unwrap(),
        CompletionRecoveryOutcome::Drained(_)
    ));
    retirement.try_abort().unwrap();
    assert_eq!(reaper.retained_count(), 0);
    assert_eq!(reaper.quarantined_count(), 0);
    assert_eq!(fixture.host.usage().unwrap().committed_bytes(), 0);
    fixture.runtime.set_fence(FenceMode::Real);
    fixture.assert_success();
    fixture.close();
    drop(error);
}

#[test]
fn failed_but_quiescent_compute_does_not_publish_output_or_hold_retired_capacity() {
    let mut fixture = FaultFixture::new();
    fixture.runtime.set_fence(FenceMode::FailedButQuiescent);
    let mut error = fixture.execute().unwrap_err();
    assert_only_input_retained(&fixture);
    let TensorExecutionError::ExecutionFailed {
        receipt,
        retirement,
    } = &mut error
    else {
        panic!("expected terminal failure, received {error:?}")
    };
    assert!(matches!(
        receipt.disposition(),
        OperationCompletionDisposition::FailedButQuiescent(_)
    ));
    retirement.try_abort().unwrap();
    assert_eq!(fixture.host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(fixture.executor.completion_reaper().retained_count(), 0);
    fixture.runtime.set_fence(FenceMode::Real);
    fixture.assert_success();
    fixture.close();
    drop(error);
}

#[test]
fn managed_readback_error_or_panic_after_partial_write_never_publishes_the_output() {
    for mode in [ReadbackMode::ErrorAfterWrite, ReadbackMode::PanicAfterWrite] {
        let mut fixture = FaultFixture::new();
        fixture.runtime.set_readback(mode);
        let mut error = fixture.execute().unwrap_err();
        assert_eq!(fixture.runtime.inner.snapshot().dense_linear_launches, 1);
        assert_eq!(fixture.runtime.readback_calls.load(Ordering::SeqCst), 1);
        assert_eq!(
            fixture
                .host
                .usage()
                .unwrap()
                .class_committed_bytes(HostMemoryClass::Output),
            0
        );
        assert_eq!(
            fixture
                .host
                .usage()
                .unwrap()
                .class_committed_bytes(HostMemoryClass::InputPreprocess),
            12
        );
        let TensorExecutionError::Readback {
            receipt,
            retirement,
            ..
        } = &mut error
        else {
            panic!("expected quiescent readback failure, received {error:?}")
        };
        assert!(matches!(
            receipt.disposition(),
            OperationCompletionDisposition::Succeeded
        ));
        retirement.try_abort().unwrap();
        assert_eq!(fixture.host.usage().unwrap().committed_bytes(), 0);
        assert_eq!(fixture.executor.completion_reaper().retained_count(), 0);
        fixture.runtime.set_readback(ReadbackMode::Real);
        fixture.assert_success();
        fixture.close();
        drop(error);
    }
}
