//! Bounded foreground maintenance for an authentic teacher admission deferral.
use super::*;

/// One opportunity per owner, in addition to the existing backing-maintenance
/// bound. A receipt permits a fresh probe, never admission by itself.
#[derive(Default)]
pub(super) struct TeacherAdmissionPressureMaintenance {
    attempted: bool,
}

impl TeacherAdmissionPressureMaintenance {
    pub(super) fn try_maintain<R: DeviceRuntime>(
        &mut self,
        resources: &Arc<PlanRuntimeResources<R>>,
        deferred: &AdmissionDeferred,
    ) -> Result<bool> {
        if self.attempted {
            return Ok(false);
        }
        self.attempted = true;
        // Keep the real sealed coordinator/demand evidence. This API rechecks
        // BOTH physical free space and logical availability under the actual
        // allocator's maintenance authority. It cannot create slots, alter
        // the pool/device budget or claim the full-input fit requirement.
        let Some(receipt) = resources
            .try_maintain_for_capacity_pressure(deferred)
            .map_err(|error| FerrumError::backend(error.to_string()))?
        else {
            return Ok(false);
        };
        let mut availability = Vec::new();
        let current = resources
            .write_dynamic_capacity_availability(&mut availability)
            .map_err(|error| FerrumError::backend(error.to_string()))?;
        let observed = deferred.epochs();
        if receipt.coordinator_id() != observed.coordinator_id()
            || current.coordinator_id() != observed.coordinator_id()
            || receipt.capacity_epoch() > current.capacity_epoch()
        {
            return Err(FerrumError::internal(
                "teacher admission maintenance returned incoherent capacity authority",
            ));
        }
        let progressed = deferred
            .wait_condition()
            .changed_since(&availability)
            .map_err(|error| FerrumError::backend(error.to_string()))?;
        // An empty receipt can be valid after a release raced the original
        // observation. Only one of this demand's exact waited sources may
        // justify retry: a global epoch from an unrelated pool is insufficient.
        if progressed {
            tracing::debug!(
                ?observed,
                ?current,
                ?receipt,
                "teacher foreground admission maintenance permits one fresh probe"
            );
        }
        Ok(progressed)
    }
}

#[cfg(test)]
mod tests;
