//! Reserve the shared auxiliary peak before suspending any runnable DiT owner.
use super::*;

impl<R: DeviceRuntime> PH3ImagePipeline<R> {
    fn reserve_cohort_host(&self, prepared: &mut PreparedCohort<R>) -> InitResult<bool> {
        if prepared.phase.is_some() {
            return Ok(true);
        }
        let mut peak = prepared.requirements.claims().to_vec();
        let outputs = prepared.members.iter().try_fold(0usize, |sum, member| {
            sum.checked_add(member.output_credit).ok_or_else(|| {
                ImageBatchError::Rejected("H3 cohort output credit overflows".into())
            })
        })?;
        peak.push((HostMemoryClass::Output, outputs));
        let ingress = prepared.members.iter().try_fold(0usize, |sum, member| {
            sum.checked_add(member.ingress_backing_bytes)
                .ok_or_else(|| {
                    ImageBatchError::Rejected("H3 cohort ingress backing overflows".into())
                })
        })?;
        peak.push((HostMemoryClass::InputPreprocess, ingress));
        self.execution
            .check_scope_capacity_with_retained_diagnostics(&peak)
            .map_err(|e| preparation_error(e.into()))?;
        match self
            .execution
            .host
            .reserve_scope(&prepared.requirements.claims())
        {
            Ok(phase) => {
                prepared.phase = Some(phase);
                Ok(true)
            }
            Err(error) if error.is_temporary_capacity() => Ok(false),
            Err(error) => Err(preparation_error(error.into())),
        }
    }

    pub(super) fn admit_expansion_cohort(
        &mut self,
        prepared: &mut PreparedCohort<R>,
        batch: Option<&mut ContinuousBatch<R>>,
        requests: &mut [&mut ContinuousRequest<R>],
    ) -> InitResult<bool> {
        if !Arc::ptr_eq(
            &prepared.pipeline,
            &self.execution.service_preparation_identity,
        ) || prepared.prepared.is_none()
            || self.is_stopped()
        {
            return Err(ImageBatchError::Rejected(
                "H3 expansion cohort is unavailable or foreign".into(),
            ));
        }
        if prepared.phase.is_some() {
            return Ok(true);
        }
        let ready = self
            .execution
            .admit_auxiliary_peak(
                prepared.requirements.startup_device_bytes,
                &mut prepared.receipts,
            )
            .map_err(|failure| self.h3_step_error(failure))?;
        if ready {
            return self.reserve_cohort_host(prepared);
        }
        let waiting = || ImageBatchError::InitializationAwaitDenoisingProgress {
            diagnostic: "prompt cohort awaits original image owners' device capacity".into(),
        };
        let Some(batch) = batch else {
            return Err(waiting());
        };
        if requests.is_empty() || batch.suspended {
            return Err(waiting());
        }
        if !self.reserve_cohort_host(prepared)? {
            return Ok(false);
        }
        if !self
            .suspend_continuous_batch(batch, requests)
            .map_err(|failure| self.suspension_error(failure))?
        {
            drop(prepared.phase.take());
            return Err(waiting());
        }
        let ready = self
            .execution
            .admit_auxiliary_peak(
                prepared.requirements.startup_device_bytes,
                &mut prepared.receipts,
            )
            .map_err(|failure| self.suspension_error(failure))?;
        if !ready {
            drop(prepared.phase.take());
            self.restore_continuous_batch(batch, requests)
                .map_err(|failure| self.suspension_error(failure))?;
            return Err(waiting());
        }
        Ok(true)
    }
}
