use super::*;

/// Owns the final source members while borrowing the immutable parent request
/// provenance. The startup driver holds native leases separately until drain.
pub(in crate::continuous_engine::inner::calibration) struct PreparedProbeSourceExecution<'a> {
    series: &'a PreparedProbeSeries,
    source: usize,
    cohorts: Vec<PreparedProbeCohort>,
    acquisitions: Vec<layout::work::PreparedProbeAcquisition>,
    external_retained_bytes: usize,
}

impl<'a> PreparedProbeSource<'a> {
    pub fn into_parts(
        self,
    ) -> Result<(
        StructuredPreparedOwnerBlockDeclarationV8,
        CostProfileLoadLimits,
        PreparedProbeSourceExecution<'a>,
    )> {
        self.ensure_acquisitions_bound()?;
        let execution = PreparedProbeSourceExecution {
            series: self.series,
            source: self.index,
            cohorts: self.cohorts,
            acquisitions: self.acquisitions,
            external_retained_bytes: self.external_retained_bytes,
        };
        Ok((self.declaration, self.limits, execution))
    }
}

impl PreparedProbeSourceExecution<'_> {
    pub fn cohorts(&self) -> &[PreparedProbeCohort] {
        &self.cohorts
    }

    fn ordinal(&self, cohort: &PreparedProbeCohort) -> Result<usize> {
        self.cohorts
            .iter()
            .position(|original| std::ptr::eq(original, cohort))
            .ok_or_else(|| error("probe cohort is not in the final source execution plan"))
    }

    pub fn acquisition_key(&self, cohort: &PreparedProbeCohort) -> Result<Option<usize>> {
        self.ordinal(cohort)?;
        match cohort.acquisition_key {
            Some(key) if self.acquisitions.get(key).copied() == cohort.native_acquisition => {
                Ok(Some(key))
            }
            None if cohort.native_acquisition.is_none() => Ok(None),
            _ => Err(error("final source acquisition binding differs")),
        }
    }

    pub fn requests_for(
        &self,
        cohort: &PreparedProbeCohort,
    ) -> Result<(Vec<ProbeRequest>, ProbeCohortSettings)> {
        let ordinal = self.ordinal(cohort)?;
        self.acquisition_key(cohort)?;
        // Both identities are checked: this final Vec authorizes the choice,
        // and the parent Vec authorizes the original template, seed and order.
        self.series.requests_for(self.source, ordinal)
    }

    pub fn external_retained_bytes(&self) -> usize {
        self.external_retained_bytes
    }
}
