//! Explicit initial residency through the original provisioning authority.

use super::*;

#[derive(Debug, Serialize)]
pub(super) struct StartupPersistentPreparation {
    mode: ferrum_types::PersistentPoolPreparation,
    plan: Option<PersistentPreallocationPlan>,
    actual_growth: Option<DynamicPoolGrowthBatchReceipt>,
}

impl StartupPersistentPreparation {
    pub(super) fn new(
        mode: ferrum_types::PersistentPoolPreparation,
        plan: Option<PersistentPreallocationPlan>,
    ) -> Self {
        Self {
            mode,
            plan,
            actual_growth: None,
        }
    }

    pub(super) fn apply<R: DeviceRuntime>(
        &mut self,
        controller: &DynamicPoolMaintenanceController<R>,
    ) -> Result<()> {
        use ferrum_types::PersistentPoolPreparation as Mode;
        let plan = match (self.mode, self.plan.as_ref()) {
            (Mode::DemandDriven, None) => return Ok(()),
            (Mode::ConfiguredEnvelope, Some(plan)) if self.actual_growth.is_none() => plan,
            _ => {
                return Err(FerrumError::internal(
                    "invalid persistent startup preparation state",
                ))
            }
        };
        let phase = StartupPhaseTimer::start("persistent_pool_preparation");
        let status = controller
            .status()
            .map_err(|error| FerrumError::device(error.to_string()))?;
        if status
            .pools()
            .iter()
            .any(|pool| pool.resident_bytes() != 0 || pool.pending_growth_bytes() != 0)
        {
            return Err(FerrumError::internal(
                "persistent startup preparation requires fresh empty pools",
            ));
        }
        let requests = plan
            .initial_growth_requests()
            .map_err(|error| FerrumError::config(error.to_string()))?;
        let receipt = controller.grow_pools(requests).map_err(|error| {
            FerrumError::device(format!("persistent startup preparation: {error}"))
        })?;
        tracing::info!(
            checked_peak_bytes = plan.checked_peak_bytes(),
            persistent_bytes = plan.persistent_bytes(),
            pools = receipt.growths().len(),
            "configured persistent envelope allocated through original pool maintenance"
        );
        self.actual_growth = Some(receipt);
        phase.finish();
        Ok(())
    }
}
