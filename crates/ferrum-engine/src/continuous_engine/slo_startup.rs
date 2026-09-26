//! Shared run/serve startup checks. Loading a compatible calibration permits
//! planning attempts; it never certifies coverage or an SLO for a new request.
use super::*;
use ferrum_interfaces::model_executor::ExecutorSloCapability;
use ferrum_types::{SloMode, SloOutputTransport, SloTimeAdmissionPolicy};

#[cfg(test)]
mod tests;

pub(super) fn validate_legacy_entry(config: &EngineConfig) -> Result<()> {
    if config.scheduler.slo.mode == SloMode::Enforce {
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
    if slo.mode != SloMode::Enforce {
        return Ok(());
    }
    if slo.admission.time_policy != SloTimeAdmissionPolicy::CompleteRequests {
        return Err(FerrumError::unsupported(
            "SLO Enforce currently requires CompleteRequests; strict time admission is not connected to transport acceptance",
        ));
    }
    if slo.output.transport != SloOutputTransport::Credited {
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
    if slo.cost_profile.is_none() || slo.prefill_reference.is_none() {
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
    if config.scheduler.slo.mode != SloMode::Enforce {
        return Ok(());
    }
    if reference.is_none()
        || cost
            .and_then(|runtime| runtime.profile_receipt())
            .is_none_or(|receipt| receipt.recorded_samples == 0 || receipt.bucket_count == 0)
        || cost.and_then(|runtime| runtime.snapshot()).is_none()
    {
        return Err(FerrumError::config(
            "SLO Enforce requires a compatible loaded reference and a nonempty fresh imported cost model; artifact paths alone are insufficient",
        ));
    }
    Ok(())
}
