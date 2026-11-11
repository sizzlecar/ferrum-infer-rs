//! The original static resource transaction shared by language and causal token
//! executors. Cleanup retains the transaction's existing native/Host owners.
use super::*;

pub(in crate::driver) struct InitializedProgramResources<R: DeviceRuntime> {
    pub(in crate::driver) plan_resources: Arc<PlanRuntimeResources<R>>,
    pub(in crate::driver) static_provider_attribution: Option<StaticProviderAttributionWitness>,
    pub(in crate::driver) shared_weights_receipt: Option<ProgramSharedWeightsReceipt>,
    pub(in crate::driver) published_weights: Option<SharedProgramWeights<R>>,
}

/// Call before creating a lane, worker or native buffer. Attachment repeats its
/// compatibility check under Core's installation gate during resource binding.
pub(in crate::driver) fn validate_static_installation<R: DeviceRuntime>(
    composition: &RuntimeComposition<R>,
    prepared: PreparedExecutionInput<'_>,
    resolved_plan: &ResolvedModelPlan,
    installation: &StaticInstallation<'_, R>,
) -> Result<()> {
    match installation {
        StaticInstallation::Private => {}
        StaticInstallation::PublishShared => {
            SharedComponentWeights::<R>::validate_plan(resolved_plan.execution_plan())
                .map_err(|e| FerrumError::model(format!("shared static publication: {e}")))?;
        }
        StaticInstallation::AttachShared(weights) => {
            weights.validate_for(
                resolved_plan.execution_plan(),
                &composition.runtime,
                &prepared.resolved_sources().weights,
            )?;
        }
    }
    Ok(())
}

/// Start native ownership only after the caller finishes fallible cold assembly.
/// Retained constructors pass the marker from their original PhaseCustody.
#[allow(clippy::too_many_arguments)]
pub(in crate::driver) fn initialize_program_resources<R: DeviceRuntime>(
    composition: &RuntimeComposition<R>,
    prepared: PreparedExecutionInput<'_>,
    resolved_plan: &ResolvedModelPlan,
    static_installation: StaticInstallation<'_, R>,
    static_host: Option<&HostMemoryAccount>,
    static_policy: StaticInitializationPolicy,
    run_id: &RunId,
    phase_native_started: Option<&std::cell::Cell<bool>>,
) -> crate::ExecutorBuildResult<InitializedProgramResources<R>> {
    let runtime = Arc::clone(&composition.runtime);
    let provision_request = RequestIdentity::new(format!("request.vnext.provision.{run_id}"))
        .map_err(|error| FerrumError::internal(error.to_string()))?;
    let provision_phase = StartupPhaseTimer::start("static_provision");
    // The owned-phase constructor armed its original resource custody
    // before entering here. From this point a raw unwind may invoke the
    // runtime's native-abandonment quarantine, never a proven close.
    if let Some(marker) = phase_native_started {
        marker.set(true);
    }
    let mut published_weights = None;
    let mut shared_weights_receipt = None;
    let (plan_resources, static_provider_attribution) = if let StaticInstallation::AttachShared(
        weights,
    ) = &static_installation
    {
        let resources = weights
            .weights
            .bind_plan(
                resolved_plan.execution_plan(),
                Arc::clone(&runtime),
                provision_request,
            )
            .map_err(|e| FerrumError::device(format!("shared static attachment: {e}")))?;
        provision_phase.finish();
        shared_weights_receipt = Some(ProgramSharedWeightsReceipt::Attached {
            evidence: weights.evidence().clone(),
            source: weights.source.clone(),
        });
        (resources, None)
    } else {
        let provisioned = resolved_plan
            .execution_plan()
            .provision_static(Arc::clone(&runtime), provision_request)
            .map_err(|error| FerrumError::device(format!("vNext static provision: {error}")))?;
        provision_phase.finish();
        match provisioned.into_provisioning() {
            StaticProvisioning::NoStatic(no_static) => (no_static.into_plan_runtime(), None),
            StaticProvisioning::Required(permit) => {
                let identity = ResourceTransactionIdentity::for_admission(
                    permit.binding(),
                    run_id.clone(),
                    TransactionId::new(format!("transaction.vnext.provision.{run_id}"))
                        .map_err(|error| FerrumError::internal(error.to_string()))?,
                );
                let driver = RuntimeResourceDriver::new(Arc::clone(&runtime))
                    .map_err(|error| FerrumError::device(error.to_string()))?;
                let transaction_begin_phase =
                    StartupPhaseTimer::start("resource_transaction_begin");
                let transaction = ResourceTransaction::<VNextDriver<R>, TransactionNew>::begin(
                    driver, identity, permit,
                )
                .map_err(|error| FerrumError::device(error.to_string()))?;
                transaction_begin_phase.finish();
                let reserve_phase = StartupPhaseTimer::start("resource_reserve");
                let reserved = match transaction.reserve() {
                    Ok(reserved) => reserved,
                    Err(error) => {
                        let cause = FerrumError::device(format!(
                            "vNext static reserve failed: {:?}",
                            error.failure()
                        ));
                        let original =
                            ferrum_runtime::ResourcePreparationRetirement::from_reserve(error);
                        return Err(shutdown::preparation_error(original, cause));
                    }
                };
                reserve_phase.finish();
                let commit_phase = StartupPhaseTimer::start("resource_commit");
                let committed = match reserved.commit() {
                    Ok(committed) => committed,
                    Err(error) => {
                        let cause = match &error {
                            ResourceCommitTransitionError::Recoverable(error) => {
                                FerrumError::device(format!(
                                    "vNext static commit failed: {:?}",
                                    error.failure()
                                ))
                            }
                            ResourceCommitTransitionError::Poisoned(error) => {
                                FerrumError::device(format!(
                                    "vNext static commit was indeterminate: {:?}",
                                    error.failure()
                                ))
                            }
                        };
                        let original =
                            ferrum_runtime::ResourcePreparationRetirement::from_commit(error);
                        return Err(shutdown::preparation_error(original, cause));
                    }
                };
                commit_phase.finish();
                let initialized = match super::initialize_static_with_host(
                    committed,
                    prepared.family(),
                    resolved_plan.execution_plan(),
                    prepared.weights(),
                    static_host,
                    static_policy,
                ) {
                    Ok(initialized) => initialized,
                    Err(failure) => {
                        let cause = FerrumError::device(format!(
                            "vNext static initialization failed: {}",
                            failure.failure().message()
                        ));
                        return Err(shutdown::initialization_error(failure, cause));
                    }
                };
                log_static_initialization_receipt(initialized.receipt());
                let provider_attribution =
                    match StaticProviderAttributionWitness::from_completed_static_initialization(
                        resolved_plan,
                        initialized.receipt(),
                    ) {
                        Ok(attribution) => attribution,
                        Err(error) => {
                            let cause = FerrumError::model(format!(
                                "vNext static provider attribution failed: {error}"
                            ));
                            let original = initialized.abort_activation(error);
                            return Err(shutdown::initialization_error(original, cause));
                        }
                    };
                let handoff_phase = StartupPhaseTimer::start("runtime_handoff");
                let handoff = if matches!(static_installation, StaticInstallation::PublishShared) {
                    initialized
                        .into_shared_plan_runtime(resolved_plan.execution_plan())
                        .map(|(resources, weights)| {
                            shared_weights_receipt = Some(ProgramSharedWeightsReceipt::Published {
                                evidence: weights.evidence().clone(),
                                source: prepared.resolved_sources().weights.clone(),
                            });
                            published_weights = Some(SharedProgramWeights {
                                weights,
                                source: prepared.resolved_sources().weights.clone(),
                            });
                            resources
                        })
                } else {
                    initialized.into_plan_runtime()
                };
                match handoff {
                    Ok(resources) => {
                        handoff_phase.finish();
                        (resources, provider_attribution)
                    }
                    Err(original) => {
                        let cause = FerrumError::device(format!(
                            "vNext runtime handoff failed: {}",
                            original.failure().message()
                        ));
                        return Err(shutdown::initialization_error(original, cause));
                    }
                }
            }
        }
    };
    Ok(InitializedProgramResources {
        plan_resources,
        static_provider_attribution,
        shared_weights_receipt,
        published_weights,
    })
}
