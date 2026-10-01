//! Shared run/serve startup checks. Loading a compatible calibration permits
//! planning attempts; it never certifies coverage or an SLO for a new request.
use super::*;
use ferrum_interfaces::model_executor::ExecutorSloCapability;
use ferrum_types::{SloMode, SloOutputTransport, SloTimeAdmissionPolicy};

#[cfg(test)]
mod tests;

pub(super) fn validate_legacy_entry(config: &EngineConfig) -> Result<()> {
    if config.scheduler.slo.mode == SloMode::Enforce
        && !config
            .scheduler
            .slo
            .experiment_stage
            .is_some_and(|stage| stage.output_transport() == SloOutputTransport::Legacy)
    {
        return Err(FerrumError::unsupported(
            "SLO Enforce requires infer_credited_stream with an explicit bounded output contract; this inference entrypoint does not provide output credits",
        ));
    }
    Ok(())
}

pub(super) fn validate_execution(
    config: &EngineConfig,
    executor: &dyn ModelExecutor,
    speculative: bool,
) -> Result<()> {
    let slo = &config.scheduler.slo;
    slo.validate_experiment_prefix(config.scheduler.prefix_rendezvous_max_wait_ms)
        .map_err(FerrumError::config)?;
    // Configuration resolution precedes executor construction. Only this
    // shared run/serve boundary can check the installed plan and backend;
    // a requested cache is not evidence of guarded transfer support.
    if config.runtime.prefix_state_cache_enabled
        && (slo.mode == SloMode::Enforce || automatic_reference_enabled(config))
        && (speculative
            || executor.execution_resource_authority() != ExecutionResourceAuthority::PlanRuntime
            || !executor.supports_guarded_prefix_maintenance()
            || !matches!(
                executor.slo_execution_capability(),
                ExecutorSloCapability::GuardedEagerWaves
                    | ExecutorSloCapability::GuardedOnDemandWaves
            ))
    {
        return Err(FerrumError::unsupported(
            "SLO Enforce or automatic calibration with prefix state caching requires an installed PlanRuntime executor with guarded checkpoint capture/restore and guarded wave execution; disable runtime.prefix_cache or select a capable executor",
        ));
    }
    if slo.mode != SloMode::Enforce {
        return Ok(());
    }
    if slo.output.transport != SloOutputTransport::Credited
        && !slo
            .experiment_stage
            .is_some_and(|stage| stage.output_transport() == SloOutputTransport::Legacy)
    {
        return Err(FerrumError::unsupported(
            "SLO Enforce requires credited output for every request",
        ));
    }
    if speculative
        || executor.execution_resource_authority() != ExecutionResourceAuthority::PlanRuntime
        || !matches!(
            executor.slo_execution_capability(),
            ExecutorSloCapability::GuardedEagerWaves | ExecutorSloCapability::GuardedOnDemandWaves
        )
    {
        return Err(FerrumError::unsupported(
            "SLO Enforce requires guarded single-wave PlanRuntime execution and declared eager/on-demand projection without speculative execution",
        ));
    }
    if slo.execution_policy().cost_observation
        && !automatic_reference_enabled(config)
        && (slo.cost_profile.is_none() || slo.prefill_reference.is_none())
    {
        return Err(FerrumError::config(
            "SLO Enforce requires an explicit cost_profile and prefill_reference artifact",
        ));
    }
    Ok(())
}

pub(super) fn validate_loaded(
    config: &EngineConfig,
    cost: Option<&inner::cost_observation::EngineCostRuntime>,
    reference: Option<&inner::prefill_reference_runtime::EnginePrefillReferenceRuntime>,
) -> Result<()> {
    if config.scheduler.slo.mode != SloMode::Enforce
        || !config.scheduler.slo.execution_policy().cost_observation
    {
        return Ok(());
    }
    let automatic = automatic_reference_enabled(config);
    let reference_required = !automatic || config.scheduler.slo.prefill_reference.is_some();
    let imported_cost_required = !automatic || config.scheduler.slo.cost_profile.is_some();
    if (reference_required && reference.is_none())
        || (imported_cost_required
            && (cost
                .and_then(|runtime| runtime.profile_receipt())
                .is_none_or(|receipt| receipt.recorded_samples == 0 || receipt.bucket_count == 0)
                || cost.and_then(|runtime| runtime.snapshot()).is_none()))
    {
        return Err(FerrumError::config(
            "SLO Enforce requires a compatible loaded reference and a nonempty fresh imported cost model; artifact paths alone are insufficient",
        ));
    }
    Ok(())
}

/// Automatic CompleteRequests starts honestly without coverage. Explicit
/// artifact loading still happens before this check and remains strict.
pub(in crate::continuous_engine) fn automatic_reference_enabled(config: &EngineConfig) -> bool {
    config.scheduler.slo.mode != SloMode::Off
        && config.scheduler.slo.admission.time_policy == SloTimeAdmissionPolicy::CompleteRequests
        && matches!(
            config
                .scheduler
                .slo
                .cost_observation
                .live_structured_calibration,
            ferrum_types::SloLiveStructuredCalibration::AutomaticV1 { .. }
        )
}
