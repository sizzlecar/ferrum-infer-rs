//! A source may bind only the private checkpoints selected by its frozen input.
//! Capacity misses remain a pre-source choice; identity failures are fatal.
use super::*;
use crate::continuous_engine::inner::calibration::startup::AcquiredProbePrefix;
use ferrum_scheduler::implementations::continuous::cost_profile::{
    StructuredNativePrefixAcquisitionCohortV1, StructuredNativePrefixAcquisitionPlanV1,
};

impl PreparedProbeSource<'_> {
    pub fn acquisitions(&self) -> &[layout::work::PreparedProbeAcquisition] {
        &self.acquisitions
    }

    pub fn acquisition_request_for(&self, key: usize) -> Result<ProbeRequest> {
        self.acquisitions
            .get(key)
            .ok_or_else(|| error("acquisition key outside frozen source"))?
            .request(&self.series.plan.execution.templates)
    }

    pub(super) fn external_base_bytes(&self) -> Result<usize> {
        self.external_retained_bytes
            .checked_sub(self.bound_acquisition_payload_bytes)
            .and_then(|n| n.checked_sub(self.lease_vec_bytes))
            .ok_or_else(|| error("source native retained ledger differs"))
    }

    /// The caller supplies the complete lease Vec capacity, including its
    /// header. Successful binding retains that charge through collection.
    pub fn acquisition_host_allowance(
        &self,
        retained_lease_vec_bytes: usize,
    ) -> Result<Option<usize>> {
        let used = self
            .external_base_bytes()?
            .checked_add(self.bound_acquisition_payload_bytes)
            .and_then(|n| n.checked_add(retained_lease_vec_bytes))
            .and_then(|n| n.checked_add(self.declaration.retained_payload_bytes()?))
            .ok_or_else(|| error("native source host accounting overflow"))?;
        Ok(self
            .declaration
            .population
            .maximum_retained_numeric_bytes
            .checked_sub(used)
            .filter(|remaining| *remaining > 0))
    }

    /// False means only that the original host allowance cannot retain this
    /// checkpoint and its scopes. No source samples have been collected yet.
    pub fn bind_verified_scope(
        &mut self,
        key: usize,
        acquired: &AcquiredProbePrefix,
        retained_lease_vec_bytes: usize,
    ) -> Result<bool> {
        let expected = *self
            .acquisitions
            .get(key)
            .ok_or_else(|| error("native binding key outside source"))?;
        if self.bound_acquisitions.get(key).copied() != Some(false)
            || !acquired.ready()
            || acquired.plan() != expected.plan()
            || acquired.input_tokens_sha256() != expected.input_tokens_sha256()
        {
            return Err(error(
                "native binding is stale, duplicate or differs from frozen input",
            ));
        }
        let Some(allowance) = self.acquisition_host_allowance(retained_lease_vec_bytes)? else {
            return Ok(false);
        };
        let acquired_bytes = acquired.retained_payload_bytes()?;
        let headers = if self.declaration.native_prefix_acquisition.is_none() {
            self.cohorts
                .len()
                .checked_mul(std::mem::size_of::<
                    Option<StructuredNativePrefixAcquisitionCohortV1>,
                >())
                .ok_or_else(|| error("native source headers overflow"))?
        } else {
            0
        };
        // Native identity strings are cloned by declaration(). Charge its
        // temporary copy before allocating it, in addition to the live lease.
        let first_peak = acquired_bytes
            .checked_mul(2)
            .and_then(|n| n.checked_add(headers))
            .ok_or_else(|| error("native declaration clone peak overflow"))?;
        if first_peak > allowance {
            return Ok(false);
        }
        let declaration = acquired.declaration();
        if !expected.matches_declaration(&declaration) {
            return Err(error(
                "native input digest or boundary differs from frozen source",
            ));
        }
        let copies = self
            .cohorts
            .iter()
            .filter(|cohort| cohort.acquisition_key == Some(key))
            .count();
        if copies == 0 {
            return Err(error("native acquisition has no frozen members"));
        }
        let scope_bytes = declaration
            .native_scope
            .retained_heap_bytes()
            .ok_or_else(|| error("native scope payload overflow"))?;
        let peak = copies
            .checked_add(1)
            .and_then(|n| n.checked_mul(scope_bytes))
            .and_then(|n| n.checked_add(headers))
            .and_then(|n| n.checked_add(acquired_bytes))
            .ok_or_else(|| error("native source scope peak overflow"))?;
        if peak > allowance {
            return Ok(false);
        }
        if self.declaration.native_prefix_acquisition.is_none() {
            self.declaration.native_prefix_acquisition =
                Some(StructuredNativePrefixAcquisitionPlanV1 {
                    phases: std::array::from_fn(|pass| {
                        vec![None; self.declaration.cohort_plan.phases[pass].len()]
                    }),
                });
        }
        let native = self.declaration.native_prefix_acquisition.as_mut().unwrap();
        for cohort in &self.cohorts {
            if cohort.acquisition_key == Some(key) {
                native.phases[cohort.pass][cohort.ordinal] = Some(declaration.clone());
            }
        }
        let base = self.external_base_bytes()?;
        self.bound_acquisition_payload_bytes = self
            .bound_acquisition_payload_bytes
            .checked_add(acquired_bytes)
            .ok_or_else(|| error("native handle payload overflow"))?;
        self.lease_vec_bytes = retained_lease_vec_bytes;
        self.external_retained_bytes = base
            .checked_add(self.bound_acquisition_payload_bytes)
            .and_then(|n| n.checked_add(self.lease_vec_bytes))
            .ok_or_else(|| error("native retained source payload overflow"))?;
        self.bound_acquisitions[key] = true;
        self.collection_ready = self.bound_acquisitions.iter().all(|ready| *ready);
        Ok(true)
    }

    pub fn ensure_acquisitions_bound(&self) -> Result<()> {
        if !self.collection_ready || self.bound_acquisitions.iter().any(|ready| !ready) {
            return Err(error(
                "native source cannot start without every original acquisition ACK",
            ));
        }
        for cohort in &self.cohorts {
            let actual = self
                .declaration
                .native_prefix_acquisition
                .as_ref()
                .and_then(|native| native.phases.get(cohort.pass))
                .and_then(|phase| phase.get(cohort.ordinal))
                .and_then(Option::as_ref);
            match cohort.acquisition_key {
                Some(key)
                    if self.acquisitions.get(key).is_some_and(|expected| {
                        cohort.native_acquisition == Some(*expected)
                            && actual.is_some_and(|scope| expected.matches_declaration(scope))
                    }) => {}
                None if actual.is_none() && cohort.native_acquisition.is_none() => {}
                _ => return Err(error("native source acquisition membership differs")),
            }
        }
        self.declaration
            .validate()
            .map_err(|e| error(format!("bound native source declaration: {e}")))
    }
}
