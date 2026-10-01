//! Every input-selected population is an independently complete source. The
//! parent input manifest and all source ranges are frozen before measured work;
//! outcomes never select, trim or reorder a source's members.
use super::*;
use std::ops::Range;

pub(in crate::continuous_engine::inner::calibration) struct PreparedProbeSeries {
    plan: PreparedProbePlan,
    ranges: Vec<Range<usize>>,
    parent_sha256: [u8; 32],
    retained_bytes: usize,
}

pub(in crate::continuous_engine::inner::calibration) struct PreparedProbeSource {
    pub declaration: StructuredPreparedOwnerBlockDeclarationV8,
    pub limits: CostProfileLoadLimits,
    pub cohorts: Vec<PreparedProbeCohort>,
    /// Parent remains alive while this source is collected. The declaration
    /// is charged separately by the collector that takes ownership of it.
    pub external_retained_bytes: usize,
}

impl PreparedProbePlan {
    pub(in crate::continuous_engine::inner::calibration) fn into_series(
        self,
    ) -> Result<PreparedProbeSeries> {
        use sha2::{Digest, Sha256};
        let sources = self
            .execution
            .audit
            .checked_selection
            .as_ref()
            .map_or(1, |s| s.batches.iter().filter(|b| b.scheduled).count());
        let range_bytes = sources
            .checked_mul(std::mem::size_of::<Range<usize>>())
            .ok_or_else(|| error("startup source range count overflow"))?;
        self.declaration
            .retained_payload_bytes()
            .and_then(|n| n.checked_add(self.execution.retained_payload_bytes()?))
            .and_then(|n| n.checked_add(self.external_retained_bytes))
            .and_then(|n| n.checked_add(range_bytes))
            .filter(|n| *n < self.declaration.population.maximum_retained_numeric_bytes)
            .ok_or_else(|| error("startup source ranges exceed retained capacity"))?;
        let mut ranges = Vec::with_capacity(sources);
        let mut end = 0usize;
        if let Some(selection) = &self.execution.audit.checked_selection {
            for batch in selection.batches.iter().filter(|batch| batch.scheduled) {
                let count = batch
                    .representative_case_indices
                    .len()
                    .checked_mul(batch.planned_cycles)
                    .ok_or_else(|| error("startup source cohort count overflow"))?;
                let start = end;
                end = end
                    .checked_add(count)
                    .ok_or_else(|| error("startup source range overflow"))?;
                if count < 3 || end > self.execution.cohorts.len() {
                    return Err(error("startup source is not a complete declared batch"));
                }
                ranges.push(start..end);
            }
        } else {
            end = self.execution.cohorts.len();
            ranges.push(0..end);
        }
        if ranges.is_empty() || end != self.execution.cohorts.len() {
            return Err(error(
                "startup sources do not partition the original input plan",
            ));
        }
        let retained_bytes = self
            .declaration
            .retained_payload_bytes()
            .and_then(|n| n.checked_add(self.execution.retained_payload_bytes()?))
            .and_then(|n| {
                n.checked_add(
                    ranges
                        .capacity()
                        .checked_mul(std::mem::size_of::<Range<usize>>())?,
                )
            })
            .and_then(|n| n.checked_add(std::mem::size_of::<PreparedProbeSeries>()))
            .and_then(|n| n.checked_add(self.external_retained_bytes))
            .filter(|n| *n < self.declaration.population.maximum_retained_numeric_bytes)
            .ok_or_else(|| error("startup source series retained capacity exhausted"))?;
        let parent_sha256 =
            Sha256::digest(self.declaration.cohort_manifest_payload.get().as_bytes()).into();
        Ok(PreparedProbeSeries {
            plan: self,
            ranges,
            parent_sha256,
            retained_bytes,
        })
    }
}

impl PreparedProbeSeries {
    pub fn len(&self) -> usize {
        self.ranges.len()
    }

    pub fn planned_cohorts(&self) -> usize {
        self.plan.audit().planned_cohorts
    }

    pub fn requests_for(
        &self,
        source: usize,
        ordinal: usize,
    ) -> Result<(Vec<ProbeRequest>, ProbeCohortSettings)> {
        let range = self
            .ranges
            .get(source)
            .filter(|range| ordinal < range.len())
            .ok_or_else(|| error("probe request outside frozen startup source"))?;
        // Keep the original execution-plan pointer identity check. Child
        // protocol ordinals never authorize a fabricated request/template.
        self.plan
            .execution
            .requests_for(&self.plan.execution.cohorts[range.start + ordinal])
    }

    /// Encode only the current source. This bounds simultaneous storage while
    /// retaining the frozen parent manifest, including all unscheduled gaps.
    pub fn source(&self, index: usize) -> Result<PreparedProbeSource> {
        let range = self
            .ranges
            .get(index)
            .ok_or_else(|| error("startup source index outside frozen series"))?
            .clone();
        let original = &self.plan.declaration;
        let count = range.len();
        let maximum = original.population.maximum_retained_numeric_bytes;
        let batch = self
            .plan
            .execution
            .audit
            .checked_selection
            .as_ref()
            .map(|selection| {
                selection
                    .batches
                    .iter()
                    .filter(|batch| batch.scheduled)
                    .nth(index)
                    .ok_or_else(|| error("startup source schedule outside frozen selection"))
            })
            .transpose()?;
        if let Some(batch) = batch {
            if !batch.schedule_within_capacity {
                return Err(error("startup source schedule exceeds native capacity"));
            }
            if batch.algorithm_universe.is_some()
                && original.population.nonnegative_envelope.is_none()
            {
                return Err(error(
                    "startup combination source requires a nonnegative envelope contract",
                ));
            }
        }
        // The parent selection and this child both retain the immutable
        // universe. Follow the existing conservative shared-payload charge,
        // including this child's contract before allocating its manifest.
        let source_universe_bytes = batch
            .and_then(|batch| batch.algorithm_universe.as_ref())
            .map_or(Some(0), |universe| universe.retained_payload_bytes())
            .ok_or_else(|| error("startup source algorithm universe retained size overflow"))?;
        let external_retained_bytes = self
            .retained_bytes
            .checked_add(
                count
                    .checked_mul(std::mem::size_of::<PreparedProbeCohort>())
                    .ok_or_else(|| error("startup source retained count overflow"))?,
            )
            .ok_or_else(|| error("startup source retained count overflow"))?;
        // Exact capacities below prevent Vec growth. The full original owns
        // at least the population, every selected request and prefix slot.
        let payload_limit = maximum
            .checked_sub(external_retained_bytes)
            .and_then(|n| n.checked_sub(original.retained_payload_bytes()?))
            .and_then(|n| n.checked_sub(source_universe_bytes))
            .filter(|n| *n > 0)
            .ok_or_else(|| error("startup source manifest retained capacity exhausted"))?;
        // Three independent phase labels are protocol identities. Numeric
        // phases still advance only at the original block/qualification gates.
        let mut cohorts = Vec::with_capacity(count);
        let mut counts = [0; 3];
        for position in 0..count {
            counts[(position * 3 / count).min(2)] += 1;
        }
        let mut phases: [Vec<CohortV2>; 3] =
            std::array::from_fn(|pass| Vec::with_capacity(counts[pass]));
        let mut prefixes: [Vec<Option<StructuredPrefixCohortV5>>; 3] =
            std::array::from_fn(|pass| Vec::with_capacity(counts[pass]));
        for (position, old) in self.plan.execution.cohorts[range.clone()]
            .iter()
            .enumerate()
        {
            let pass = (position * 3 / count).min(2);
            let ordinal = phases[pass].len();
            let mut declared = original.cohort_plan.phases[old.pass][old.ordinal].clone();
            declared.manifest_case =
                u32::try_from(ordinal).map_err(|_| error("startup source ordinal overflow"))?;
            phases[pass].push(declared);
            prefixes[pass].push(original.prefix_plan.phases[old.pass][old.ordinal].clone());
            let mut cohort = old.clone();
            cohort.pass = pass;
            cohort.ordinal = ordinal;
            cohorts.push(cohort);
        }
        let payload = manifest::freeze_source(
            &original.cohort_manifest_payload,
            self.parent_sha256,
            index,
            self.len(),
            range,
            &cohorts,
            payload_limit,
        )?;
        let mut population = original.population.clone();
        if let Some(batch) = batch {
            if let Some(contract) = population.nonnegative_envelope.as_mut() {
                // This declaration is frozen before any source samples exist.
                // Raw sources retain None; a combination requires its own
                // original Fit, Residual and Qualification populations.
                contract.algorithm_universe = batch.algorithm_universe.clone();
            }
            population.schedule = batch.schedule.clone();
            population.settings.max_phase_samples = *population
                .schedule
                .maximum_phase_members
                .iter()
                .max()
                .unwrap();
            population
                .schedule
                .validate(&population.settings)
                .map_err(|reason| error(format!("startup source schedule: {reason:?}")))?;
        }
        let declaration = StructuredPreparedOwnerBlockDeclarationV8 {
            population,
            cohort_plan: CohortPlanV2 { phases },
            native_prefix_acquisition: None,
            prefix_plan: StructuredPrefixPlanV5 { phases: prefixes },
            cohort_manifest_payload: payload,
            maximum_offered_waves: original.maximum_offered_waves,
        };
        declaration
            .validate()
            .map_err(|e| error(format!("startup source declaration: {e}")))?;
        declaration
            .retained_payload_bytes()
            .and_then(|n| n.checked_add(external_retained_bytes))
            .filter(|n| *n <= maximum)
            .ok_or_else(|| error("startup source exceeds shared retained capacity"))?;
        Ok(PreparedProbeSource {
            declaration,
            limits: self.plan.limits.clone(),
            cohorts,
            external_retained_bytes,
        })
    }
}
