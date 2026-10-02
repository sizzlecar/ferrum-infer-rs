use super::*;
use layout::source_inputs::{ColdSourceRebuild as RecomputedColdSource, ColdSourceSkip};

#[derive(Debug)]
pub(in crate::continuous_engine::inner::calibration) enum ColdSourceRebuild {
    Ready {
        additional_collection_actions: usize,
        additional_offer_rows: usize,
    },
    Skip(ColdSourceSkip),
}

impl PreparedProbeSource<'_> {
    /// Called only after inventory drain/close and dropping all acquired leases.
    /// Original setup reservations stay spent. Recompute the same members and
    /// cycles with cold preparation before admitting any numerical collection.
    pub fn try_rebuild_cold(&mut self) -> Result<ColdSourceRebuild> {
        if self.acquisitions.is_empty() {
            return Err(error(
                "cold rebuild requires the original native source choice",
            ));
        }
        self.collection_ready = false;
        let base = self.external_base_bytes()?;
        self.declaration.native_prefix_acquisition = None;
        self.acquisitions.clear();
        self.bound_acquisitions.clear();
        self.bound_acquisition_payload_bytes = 0;
        self.lease_vec_bytes = std::mem::size_of::<
            Vec<crate::continuous_engine::inner::calibration::startup::AcquiredProbePrefix>,
        >();
        self.external_retained_bytes = base
            .checked_add(self.lease_vec_bytes)
            .ok_or_else(|| error("cold source retained bytes overflow"))?;
        for cohort in &mut self.cohorts {
            cohort.native_acquisition = None;
            cohort.acquisition_key = None;
        }
        let maximum = self.declaration.population.maximum_retained_numeric_bytes;
        let available = maximum
            .checked_sub(self.external_retained_bytes)
            .and_then(|n| n.checked_sub(self.declaration.retained_payload_bytes()?));
        let Some(available) = available.filter(|n| *n > 0) else {
            return Ok(ColdSourceRebuild::Skip(ColdSourceSkip::RetainedCapacity));
        };
        let inputs = self
            .series
            .plan
            .execution
            .source_inputs
            .get(self.index)
            .ok_or_else(|| error("cold source has no original input opportunities"))?;
        let cold = match inputs.cold_plan(
            &self.declaration.population,
            self.series.plan.execution.prefill_chunk,
            self.series.plan.execution.prefill_row_ceiling,
            available,
        )? {
            RecomputedColdSource::Ready(cold) => cold,
            RecomputedColdSource::Skip(reason) => return Ok(ColdSourceRebuild::Skip(reason)),
        };
        if cold.cycles != self.work.planned_cycles {
            return Err(error("cold rebuild changed the original source cycles"));
        }
        let additional_collection_actions = cold
            .execution_actions
            .saturating_sub(cold.original_collection_actions);
        let additional_offer_rows = cold
            .declared_offer_row_bound
            .saturating_sub(cold.original_declared_offer_row_bound);
        self.declaration.population.schedule = cold.schedule;
        self.declaration.population.settings.max_phase_samples = *self
            .declaration
            .population
            .schedule
            .maximum_phase_members
            .iter()
            .max()
            .unwrap();
        self.work = manifest::SourceWork {
            planned_cycles: cold.cycles,
            maximum_anchor_span: Some(cold.maximum_anchor_span),
            requests: cold.requests,
            execution_actions: cold.execution_actions,
            declared_offer_row_bound: cold.declared_offer_row_bound,
            serial_token_work: Some(cold.serial_token_work),
        };
        // The old manifest remains live until its replacement is complete.
        // Authorize that simultaneous peak rather than merely the final size.
        let remaining = maximum
            .checked_sub(self.external_retained_bytes)
            .and_then(|n| n.checked_sub(self.declaration.retained_payload_bytes()?))
            .unwrap_or(0);
        let payload = manifest::freeze_source(
            &self.series.plan.declaration.cohort_manifest_payload,
            self.series.parent_sha256,
            self.index,
            self.series.len(),
            self.series.ranges[self.index].clone(),
            &self.cohorts,
            manifest::SourcePreparationChoice::ColdFallback,
            self.work,
            &self.declaration.population,
            Some(&cold.input_opportunities),
            remaining,
        )?;
        let Some(payload) = payload else {
            return Ok(ColdSourceRebuild::Skip(ColdSourceSkip::RetainedCapacity));
        };
        self.declaration.cohort_manifest_payload = payload;
        self.collection_ready = true;
        self.ensure_acquisitions_bound()?;
        Ok(ColdSourceRebuild::Ready {
            additional_collection_actions,
            additional_offer_rows,
        })
    }
}
