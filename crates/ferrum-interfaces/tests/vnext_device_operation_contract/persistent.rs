//! Real Reference-runtime provisioning and invocation views for padded Plan storage.
use super::*;

#[derive(Debug, Clone, Copy)]
pub(crate) enum PersistentDescriptorFault {
    Short,
    Type,
    Identity,
}

#[derive(Clone, Copy)]
pub(crate) enum DependencyMode {
    None,
    Valid,
    Conflict,
    Stale,
}

struct PersistentProvider {
    base: TestProvider,
    bytes: u64,
    dependency_mode: DependencyMode,
    previous: Mutex<Option<RetainedPlanDependencyAuthority>>,
}

impl OperationResourceEstimator for PersistentProvider {
    fn descriptor(&self) -> &OperationProviderDescriptor {
        &self.base.descriptor
    }

    fn estimate_resources(
        &self,
        request: OperationResourceEstimateRequest<'_>,
    ) -> Result<OperationResourceEstimate, VNextError> {
        let d = self.descriptor();
        let estimate = OperationResourceEstimate::new(
            d.resource_estimator_id(),
            d.resource_estimator_version(),
            d.resource_estimator_implementation_fingerprint(),
            request.input_fingerprint(),
            16,
            None,
            Some(ProviderWorkspaceRequirement::new(
                self.bytes,
                16,
                ProviderWorkspaceScope::Plan,
                ProviderWorkspaceReusePolicy::Preserve,
                DynamicStorageRequirement::contiguous(),
            )?),
        );
        if matches!(self.dependency_mode, DependencyMode::None) {
            Ok(estimate)
        } else {
            Ok(
                estimate.with_binding(ProviderWorkspaceRequirement::from_formula(
                    ProviderWorkspaceSizeFormula::actual_sequences(16)?,
                    16,
                    ProviderWorkspaceScope::Invocation,
                    ProviderWorkspaceReusePolicy::OverwriteBeforeRead,
                    DynamicStorageRequirement::contiguous(),
                )?),
            )
        }
    }
}

impl OperationProvider<TestRuntime> for PersistentProvider {
    fn reusable_execution_topology(
        &self,
        request: ReusableExecutionTopologyRequest<'_>,
    ) -> Result<ReusableExecutionTopology, VNextError> {
        self.base.reusable_execution_topology(request)
    }

    fn encode_selected(
        &self,
        invocation: BatchedOperationInvocation<'_, TestBuffer>,
    ) -> Result<EncodedDeviceOperation<TestCommand>, OperationFailure> {
        for participant in invocation.participants() {
            let view = participant
                .persistent_view()
                .expect("admitted Plan workspace");
            let descriptor = view.descriptor();
            assert_eq!(descriptor.element_type, ElementType::U8);
            assert_eq!(descriptor.usage, BufferUsage::Persistent);
            assert_eq!(view.allocation_lifetime(), AllocationLifetime::Plan);
            assert!(descriptor.size_bytes >= self.bytes);
            // The provider consumes exactly the payload, never the alignment
            // tail retained by its immutable ResourceAllocation.
            let translated = view.translate(0, self.bytes).unwrap();
            let regions: Vec<_> = translated.iter().collect();
            assert_eq!(regions.len(), 1);
            let (buffer, range, retention) = regions[0].buffer_and_physical_range();
            assert_eq!(range, 0..self.bytes);
            assert_eq!(buffer.descriptor, *descriptor);
            assert_eq!(
                retention.reusable_address_scope(),
                Some(DeviceReusableAddressScope::Plan)
            );
            for (offset, bytes) in [
                (0, descriptor.size_bytes + 1),
                (descriptor.size_bytes, 1),
                (u64::MAX, 1),
                (0, 0),
            ] {
                assert!(view.translate(offset, bytes).is_err());
            }
            for bank in 0..self.bytes / 4 {
                let translated = view.translate(bank * 4, 4).unwrap();
                let part = translated.iter().next().unwrap();
                assert_eq!(part.buffer_and_physical_range().1, bank * 4..bank * 4 + 4);
            }
        }
        let mut dependencies = Vec::new();
        if !matches!(self.dependency_mode, DependencyMode::None) {
            let component: WeightId = id("weight.component.left");
            let spec = || RetainedPlanDependencySpec {
                input_ordinal: 1,
                component_id: &component,
                source_offset_bytes: 0,
                source_length_bytes: 8,
                persistent_offset_bytes: 0,
                persistent_length_bytes: 4,
                alignment_bytes: 4,
                validation_identity: "fixture.exact-weight-validation.v1",
            };
            for change in 0..8 {
                let mut invalid = spec();
                match change {
                    0 => invalid.input_ordinal = 0,
                    1 => invalid.source_offset_bytes = 4,
                    2 => invalid.source_length_bytes = 0,
                    3 => invalid.persistent_offset_bytes = u64::MAX,
                    4 => invalid.persistent_offset_bytes = 1,
                    5 => invalid.persistent_length_bytes = 3,
                    6 => invalid.alignment_bytes = 3,
                    7 => invalid.validation_identity = "",
                    _ => unreachable!(),
                }
                assert!(invocation.retained_plan_dependency(invalid).is_err());
            }
            dependencies.push(
                invocation
                    .retained_plan_dependency(spec())
                    .unwrap()
                    .encode(TestCommand::DynamicBinding),
            );
            // Only one ordered wave/lane may deduplicate this exact bank.
            dependencies.push(
                invocation
                    .retained_plan_dependency(spec())
                    .unwrap()
                    .encode(TestCommand::DynamicBinding),
            );
            if matches!(self.dependency_mode, DependencyMode::Conflict) {
                let mut conflict = spec();
                conflict.validation_identity = "fixture.different-arithmetic.v2";
                dependencies.push(
                    invocation
                        .retained_plan_dependency(conflict)
                        .unwrap()
                        .encode(TestCommand::DynamicBinding),
                );
            }
            if matches!(self.dependency_mode, DependencyMode::Stale) {
                let mut previous = self.previous.lock().unwrap();
                if let Some(stale) = previous.take() {
                    dependencies.push(stale.encode(TestCommand::DynamicBinding));
                } else {
                    *previous = Some(invocation.retained_plan_dependency(spec()).unwrap());
                }
            }
        }
        self.base.encode_selected(invocation).map(|operation| {
            dependencies
                .into_iter()
                .fold(operation, |operation, dependency| {
                    operation.with_retained_plan_dependency(dependency)
                })
        })
    }
}

pub(crate) fn fixture_with_padded_persistent(bytes: u64) -> Fixture {
    fixture_with_retained_dependencies(bytes, DependencyMode::None)
}

pub(crate) fn fixture_with_retained_dependencies(
    bytes: u64,
    dependency_mode: DependencyMode,
) -> Fixture {
    fixture_with_retained_dependencies_and_bucket(bytes, dependency_mode, None)
}

pub(crate) fn fixture_with_retained_dependencies_and_bucket(
    bytes: u64,
    dependency_mode: DependencyMode,
    bucket: Option<ReusableExecutionBucketSpec>,
) -> Fixture {
    let original = catalog();
    let mut operation = operation();
    operation.resources.persistent = ResourcePresenceRequirement::Required;
    let provider = OperationProviderDescriptor::new(
        id("provider.operation.device-operation"),
        operation.id.clone(),
        operation.fingerprint().unwrap(),
        sha('c'),
        ProviderExecutionSemantics::bitwise_eager_and_replay(),
        ContractVersion::new(1, 0),
        original.device().id.clone(),
        original.device().capabilities.clone(),
        BTreeSet::from([id("weight-format.device-operation-composite")]),
        BTreeSet::new(),
        contiguous_storage_bindings(&operation),
        "resource-estimator.persistent-padding",
        ContractVersion::new(1, 0),
        sha('b'),
    )
    .unwrap();
    let catalog = CapabilityCatalog::new(
        original.device().clone(),
        vec![operation.clone()],
        BTreeMap::from([(operation.id.clone(), vec![provider.clone()])]),
        original.engine_providers().values().cloned().collect(),
    )
    .unwrap();
    let behavior = Arc::new(Mutex::new(
        if matches!(dependency_mode, DependencyMode::None) {
            ProviderBehavior::Success
        } else {
            ProviderBehavior::ProgramBinding
        },
    ));
    let trace = Arc::new(Mutex::new(ProviderTrace::default()));
    let registry = || {
        OperationRuntimeRegistry::new(
            vec![Box::new(TestOperationContract {
                descriptor: operation.clone(),
            }) as Box<dyn OperationContract>],
            vec![Box::new(PersistentProvider {
                base: TestProvider {
                    descriptor: provider.clone(),
                    behavior: Arc::clone(&behavior),
                    trace: Arc::clone(&trace),
                },
                bytes,
                dependency_mode,
                previous: Mutex::new(None),
            }) as Box<dyn OperationProvider<TestRuntime>>],
        )
        .unwrap()
    };
    let registry = registry();
    let (runtime_policy, reusable_execution_bucket) =
        if matches!(dependency_mode, DependencyMode::None) {
            (policy(), None)
        } else {
            let (original_policy, original_bucket) = reusable_policy();
            match bucket {
                Some(bucket) => (
                    policy_with_reusable_execution(Some(
                        ReusableExecutionPolicy::new(1, vec![bucket.clone()]).unwrap(),
                    )),
                    Some(bucket),
                ),
                None => (original_policy, Some(original_bucket)),
            }
        };
    let (resolved, plan) = planning::resolved_model_plan_with_zero_state_and_policy(
        &registry,
        TestStateProfile::none(),
        &runtime_policy,
        &catalog,
        false,
    );
    let impostor_registry =
        operation_registry(&original, Arc::clone(&behavior), Arc::clone(&trace));
    let impostor_plan_hash = plan_for_registry(&impostor_registry).plan_hash().clone();
    let (runtime, runtime_trace) = runtime(&catalog);
    let plan_resources = plan_runtime_resources(&plan, Arc::clone(&runtime));
    Fixture {
        registry,
        impostor_registry,
        resolved,
        plan,
        impostor_plan_hash,
        runtime,
        runtime_trace,
        provider_behavior: behavior,
        provider_trace: trace,
        plan_resources,
        reusable_execution_bucket,
    }
}
