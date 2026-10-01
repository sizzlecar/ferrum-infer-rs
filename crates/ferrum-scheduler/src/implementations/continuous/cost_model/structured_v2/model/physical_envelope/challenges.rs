use super::*;
use ferrum_types::FinishReason;

/// Actual phase facts. Position is retained in the raw receipts; these shared
/// work coefficients require branch challenges, not simultaneous EOS at every
/// row position. Cause masks are audit facts and never select eligible walls.
pub(super) struct ChallengeCoverage {
    prospective_completion_work: bool,
    positive: Vec<bool>,
    zero: Vec<bool>,
    pending: [bool; 2],
    length: [bool; 2],
    repetition: [bool; 2],
    early: bool,
    continuation: bool,
    early_cause_mask: u8,
    observed_early_rows: usize,
    observed_continuation_rows: usize,
}
impl ChallengeCoverage {
    /// Cold publication comparison of the same structural authorization used
    /// by `authorize`. Observed timing and coefficient values play no role.
    pub(super) fn contains_coverage(&self, previous: &Self) -> bool {
        let contains = |new: &[bool], old: &[bool]| {
            new.len() == old.len() && new.iter().zip(old).all(|(&n, &o)| !o || n)
        };
        self.prospective_completion_work == previous.prospective_completion_work
            && contains(&self.positive, &previous.positive)
            && contains(&self.pending, &previous.pending)
            && contains(&self.length, &previous.length)
            && contains(&self.repetition, &previous.repetition)
            && (self.prospective_completion_work
                || !(previous.early && previous.continuation)
                || (self.early && self.continuation))
    }

    pub fn observe(samples: &[StructuredNumericObservationV2]) -> Result<Self> {
        Self::observe_for(
            samples,
            NonNegativePlanningEstimatorV1::CoefficientEnvelopeV1,
        )
    }
    pub fn observe_for(
        samples: &[StructuredNumericObservationV2],
        estimator: NonNegativePlanningEstimatorV1,
    ) -> Result<Self> {
        let dims = samples
            .first()
            .ok_or(StructuredUnknown::InsufficientSamples)?
            .input
            .basis
            .len();
        let mut value = Self {
            prospective_completion_work: estimator.prospective_completion_work(),
            positive: vec![false; dims],
            zero: vec![false; dims],
            pending: [false; 2],
            length: [false; 2],
            repetition: [false; 2],
            early: false,
            continuation: false,
            early_cause_mask: 0,
            observed_early_rows: 0,
            observed_continuation_rows: 0,
        };
        for sample in samples {
            let input = &sample.input;
            input.validate_actual_completion()?;
            let axes = envelope::model_axes(input, estimator)?;
            if axes.len() != dims {
                return Err(StructuredUnknown::WrongDomain);
            }
            for (i, &x) in axes.iter().enumerate() {
                value.positive[i] |= x != 0;
                value.zero[i] |= x == 0;
            }
            let branches = envelope::input_branches(input);
            value.pending[usize::from(branches[0].unwrap())] = true;
            value.length[usize::from(branches[1].unwrap())] = true;
            if let Some(present) = branches[3] {
                value.repetition[usize::from(present)] = true;
            }
            let Some(completion) = &input.completion else {
                continue;
            };
            let causes = input
                .settled_terminal_causes()
                .ok_or(StructuredUnknown::MissingEvidence)?;
            if causes.len() != completion.positions.len()
                || causes
                    .iter()
                    .map(|(p, _)| *p)
                    .ne(completion.positions.iter().copied())
            {
                return Err(StructuredUnknown::MissingEvidence);
            }
            for row in &input.physical_host_rows {
                if !envelope::early_capable(row) {
                    continue;
                }
                match causes.binary_search_by_key(&row.physical_position, |(p, _)| *p) {
                    Ok(i) => match &causes[i].1 {
                        FinishReason::EOS => {
                            value.early = true;
                            value.early_cause_mask |= 1;
                            value.observed_early_rows += 1;
                        }
                        FinishReason::Stop => {
                            value.early = true;
                            value.early_cause_mask |= 2;
                            value.observed_early_rows += 1;
                        }
                        // Length is only a forced-at-capacity branch. It cannot
                        // certify the optional early-work branch.
                        _ => return Err(StructuredUnknown::InvalidInput),
                    },
                    Err(_) => {
                        value.continuation = true;
                        value.observed_continuation_rows += 1;
                    }
                }
            }
        }
        Ok(value)
    }
    /// Legacy outcome regressors require both outcomes in every phase. The
    /// global empirical model uses pre-execution opportunities instead; its
    /// independent whole-wall qualification is not unseen-EOS-tail evidence.
    pub fn validate_branches(&self, samples: &[StructuredNumericObservationV2]) -> Result<()> {
        use ferrum_interfaces::execution_cost::HostContentDomainV1;
        let may_terminate_early = samples.iter().flat_map(|sample| &sample.input.physical_host_rows)
            // Token production is the validated pre-execution work projection.
            // A body-only Fit freezes its produced-token basis axis at zero;
            // any later emitting input is still an unseen work direction.
            .any(|row| row.terminal_expectation != ferrum_interfaces::execution_cost::HostTerminalExpectationV1::NoTokenProduced
                && matches!(row.installed_policy.empirical_content_domain,
                Some(HostContentDomainV1::PlainTextInstalledV2(policy)) if policy.model_eos || policy.user_stop));
        if !self.prospective_completion_work
            && may_terminate_early
            && !(self.early && self.continuation)
        {
            return Err(StructuredUnknown::QualificationCoverage);
        }
        Ok(())
    }

    pub(super) fn audit_phase(&self, phase: StructuredPhaseV2, samples: usize) {
        if self.prospective_completion_work {
            tracing::debug!(
                target: "ferrum::slo_calibration",
                ?phase, samples,
                observed_early_rows = self.observed_early_rows,
                observed_continuation_rows = self.observed_continuation_rows,
                early_cause_mask = self.early_cause_mask,
                prospective_completion_work = true,
                "Empirical phase completion outcomes; unobserved terminal tails remain unvalidated"
            );
        }
    }

    pub fn validate_positive_axes(&self, maxima: &[u64]) -> Result<()> {
        self.validate_positive_axes_detailed(maxima)
            .map_err(StructuredQueryFailureV2::reason)
    }
    fn validate_positive_axes_detailed(&self, maxima: &[u64]) -> QueryResult<()> {
        if self.positive.len() != maxima.len() {
            return Err(StructuredUnknown::QualificationCoverage.into());
        }
        if self
            .positive
            .iter()
            .zip(maxima)
            .any(|(&seen, &max)| max != 0 && !seen)
        {
            return Err(StructuredQueryFailureV2::OutsideSupport(
                StructuredUnknown::QualificationCoverage,
            ));
        }
        Ok(())
    }
    pub fn authorize(&self, query: &StructuredQueryV2, upper: &[u64]) -> Result<()> {
        self.authorize_detailed(query, upper)
            .map_err(StructuredQueryFailureV2::reason)
    }
    pub fn authorize_detailed(&self, query: &StructuredQueryV2, upper: &[u64]) -> QueryResult<()> {
        self.authorize_query_completion(
            query,
            upper,
            query.input.completion.as_ref().is_some_and(|c| !c.settled),
        )
    }
    pub fn authorize_prospective_query(
        &self,
        query: &StructuredQueryV2,
        upper: &[u64],
    ) -> QueryResult<()> {
        self.authorize_query_completion(query, upper, true)
    }
    fn authorize_query_completion(
        &self,
        query: &StructuredQueryV2,
        upper: &[u64],
        prospective_completion: bool,
    ) -> QueryResult<()> {
        self.validate_positive_axes_detailed(upper)?;
        let input = &query.input;
        let (min, max) = query.pending_count_range()?;
        self.authorize_branches(input, upper, min, max, prospective_completion)
    }
    /// Reconstruct prospective eligibility from the original input even when
    /// its receipt already carries a settled completion. Thus EOS and Continue
    /// twins have the same membership; only their validity checks may differ.
    pub fn authorize_original_input(&self, input: &StructuredInputV2, upper: &[u64]) -> Result<()> {
        self.validate_positive_axes(upper)?;
        let count = input.pending_positions.len();
        self.authorize_branches(input, upper, count, count, true)
            .map_err(StructuredQueryFailureV2::reason)
    }
    fn authorize_branches(
        &self,
        input: &StructuredInputV2,
        upper: &[u64],
        min: usize,
        max: usize,
        prospective_completion: bool,
    ) -> QueryResult<()> {
        if (min == 0 && !self.pending[0])
            || (max != 0 && !self.pending[1])
            || !self.length[usize::from(!input.length_positions.is_empty())]
        {
            return Err(StructuredQueryFailureV2::OutsideSupport(
                StructuredUnknown::QualificationCoverage,
            ));
        }
        if let Some((basis, _)) = input.repetition_offsets {
            if (input.basis[basis] == 0. && !self.repetition[0])
                || (upper[basis] != 0 && !self.repetition[1])
            {
                return Err(StructuredQueryFailureV2::OutsideSupport(
                    StructuredUnknown::QualificationCoverage,
                ));
            }
        }
        if !self.prospective_completion_work
            && prospective_completion
            && input.physical_host_rows.iter().any(envelope::early_capable)
            && !(self.early && self.continuation)
        {
            return Err(StructuredQueryFailureV2::OutsideSupport(
                StructuredUnknown::QualificationCoverage,
            ));
        }
        Ok(())
    }
    /// Cold inspection using the same observe/branch/positive checks. Only the
    /// first failed check is reported; these facts never select a population.
    pub(super) fn diagnose_phase(
        samples: &[StructuredNumericObservationV2],
        maxima: &[u64],
        estimator: NonNegativePlanningEstimatorV1,
    ) -> Option<serde_json::Value> {
        use serde_json::json;
        let coverage = match Self::observe_for(samples, estimator) {
            Ok(value) => value,
            Err(error) => {
                return Some(json!({"gate":"phase_observe","reason":format!("{error:?}")}))
            }
        };
        if let Err(error) = coverage.validate_branches(samples) {
            return Some(
                json!({"gate":"phase_branches","reason":format!("{error:?}"),
                "early":coverage.early,"continuation":coverage.continuation,
                "early_cause_mask":coverage.early_cause_mask,
                "samples":samples.len()}),
            );
        }
        if let Err(error) = coverage.validate_positive_axes(maxima) {
            let axis = coverage
                .positive
                .iter()
                .zip(maxima)
                .position(|(&seen, &max)| max != 0 && !seen);
            return Some(
                json!({"gate":"phase_positive_axes","reason":format!("{error:?}"),
                "first_missing_axis":axis.map(|i|json!({"basis_axis":i,
                    "axis_label":samples.first().map(|s|super::diagnostic::axis_label(&s.input,i)),
                    "fit_maximum":maxima[i],"phase_maximum":0})),
                "fit_basis_axes":maxima.len(),"phase_basis_axes":coverage.positive.len(),
                "samples":samples.len()}),
            );
        }
        None
    }
    pub fn bind(&self, digest: &mut Sha256) {
        digest.update((self.positive.len() as u64).to_le_bytes());
        for (&positive, &zero) in self.positive.iter().zip(&self.zero) {
            digest.update([u8::from(positive), u8::from(zero)]);
        }
        for pair in [self.pending, self.length, self.repetition] {
            digest.update(pair.map(u8::from));
        }
        digest.update([
            u8::from(self.early),
            u8::from(self.continuation),
            self.early_cause_mask,
        ]);
    }
    pub fn retained_heap_bytes(&self) -> Option<usize> {
        self.positive.capacity().checked_add(self.zero.capacity())
    }
}

#[cfg(test)]
impl ChallengeCoverage {
    pub(super) fn diagnose_query_coverage(
        &self,
        query: &StructuredQueryV2,
        upper: &[u64],
    ) -> serde_json::Value {
        use serde_json::json;
        let input = &query.input;
        let (min, max) = query.pending_count_range().unwrap();
        let missing:Vec<_>=upper.iter().enumerate().filter(|(i,v)|**v!=0&&!self.positive.get(*i).copied().unwrap_or(false))
            .map(|(i,v)|json!({"axis":i,"query_upper":v,"description":super::diagnostic::diagnostic_axis(input,i)})).collect();
        json!({
            "authorization":format!("{:?}",self.authorize_detailed(query,upper)),
            "missing_positive_axes":missing,
            "phase_branches":{"pending_absent_present":self.pending,"length_absent_present":self.length,
                "repetition_absent_present":self.repetition,"early":self.early,"continuation":self.continuation,
                "prospective_completion_work":self.prospective_completion_work,
                "observed_early_rows":self.observed_early_rows,
                "observed_continuation_rows":self.observed_continuation_rows,
                "early_cause_mask":self.early_cause_mask},
            "query_branches":{"pending_count_range":[min,max],"length_present":!input.length_positions.is_empty(),
                "repetition_required":input.repetition_offsets.map(|(i,_)|[input.basis[i]!=0.,upper[i]!=0]),
                "prospective_early_required":input.completion.as_ref().is_some_and(|c|!c.settled)
                    &&input.physical_host_rows.iter().any(envelope::early_capable)}
        })
    }
}
