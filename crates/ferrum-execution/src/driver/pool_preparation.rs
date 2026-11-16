//! Physical capacity preparation shares the compiled demand formulas and the
//! ordinary pool maintenance authority; it owns no allocator or request lease.

use super::*;

pub(super) fn prepare<R: DeviceRuntime>(
    resources: &Arc<PlanRuntimeResources<R>>,
    memory: &MemoryPlan,
    mode: ferrum_types::RuntimePoolPreparation,
    context_tokens: usize,
    step_tokens: u64,
) -> Result<()> {
    if mode == ferrum_types::RuntimePoolPreparation::OnDemand {
        return Ok(());
    }
    let context = u64::try_from(context_tokens)
        .map_err(|_| FerrumError::config("pool preparation context exceeds u64"))?;
    let targets = memory
        .startup_workload_pool_bytes(
            context,
            context,
            memory.maximum_active_sequences(),
            step_tokens,
        )
        .map_err(|error| FerrumError::config(format!("compiled pool preparation: {error}")))?;
    let requested_total = targets
        .values()
        .try_fold(memory.static_bytes(), |total, bytes| {
            total
                .checked_add(*bytes)
                .ok_or_else(|| FerrumError::config("pool preparation bytes overflow u64"))
        })?;
    // The configured admission limits are not a promise that every live
    // sequence can hold its maximum context simultaneously. An oversized
    // preparation target must leave the original on-demand service runnable.
    if requested_total > memory.usable_capacity_bytes() {
        tracing::info!(target: "ferrum.runtime.pool_preparation",
            outcome = "on_demand", requested_total_bytes = requested_total,
            budget_bytes = memory.usable_capacity_bytes(),
            "compiled pool preparation exceeds budget; retaining on-demand growth");
        return Ok(());
    }
    let phase = StartupPhaseTimer::start("dynamic_pool_preparation");
    match resources.ensure_pool_residency(targets) {
        Ok(receipt) => {
            tracing::info!(target: "ferrum.runtime.pool_preparation",
                outcome = "prepared", requested_total_bytes = requested_total,
                growths = ?receipt.growths(),
                "compiled physical pool capacity prepared before readiness");
        }
        Err(
            error @ (VNextError::DeviceCapacityUnavailable(_)
            | VNextError::DynamicPoolResidentUnavailable(_)),
        ) => {
            // These typed denials precede native allocation. Contention is not
            // permission to evict owners or reject otherwise valid startup.
            tracing::info!(target: "ferrum.runtime.pool_preparation",
                outcome = "on_demand", requested_total_bytes = requested_total,
                reason = %error,
                "optional pool preparation denied; retaining on-demand growth");
        }
        Err(error) => {
            return Err(FerrumError::model(format!(
                "runtime pool preparation failed: {error}"
            )))
        }
    }
    phase.finish();
    Ok(())
}
