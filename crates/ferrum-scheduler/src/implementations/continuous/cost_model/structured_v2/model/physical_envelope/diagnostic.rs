//! Explicit cold inspection of original inputs and the frozen fit. Never used
//! for membership, numerical acceptance, signatures, or execution authority.
use super::*;
use serde_json::{json, Value};
#[cfg(test)]
mod previous_completion_binding;

#[cfg(test)]
impl FittedStructuredModelV2 {
    /// Original Fit only: the experiment supplies independently collected R/Q
    /// cells. No existing F/R/Q support or qualification is claimed by this API.
    pub(crate) fn diagnose_empirical_fitted_input(
        &self,
        query: &StructuredQueryV2,
        now_ns: u64,
    ) -> Option<(u64, Vec<u64>, Vec<u64>)> {
        check_time(now_ns, self.frozen_at_ns, self.state.expires_at_ns).ok()?;
        let physical = self.numerical.physical()?;
        let projected = match physical.contract.algorithm_universe.as_ref() {
            Some(universe) => query.clone().with_algorithm_universe(universe).ok()?,
            None => query.clone(),
        };
        self.same_population_domain(projected.input()).ok()?;
        let axes = physical.authorized_query_upper(&projected, 0, false).ok()?;
        let point = NonNegativePlanningEstimatorV1::FittedResidualV1
            .prediction_detailed(&physical.fit, &axes)
            .ok()?;
        Some((
            point,
            projected.input().joint_support_coordinates().to_vec(),
            axes,
        ))
    }

    pub(crate) fn diagnose_empirical_fit_limits(&self) -> (u64, u64) {
        (
            self.numerical.fit_error_floor_ns(),
            self.state.expires_at_ns,
        )
    }
}

#[cfg(test)]
impl QualifiedStructuredModelV2 {
    /// Test-only empirical successor input. Keeps original domain/branch gates,
    /// but selects the existing fitted-residual point estimator explicitly.
    pub(crate) fn diagnose_empirical_cell_input(
        &self,
        query: &StructuredQueryV2,
    ) -> Option<(u64, Vec<u64>, Vec<u64>)> {
        let physical = self.calibrated.fitted.numerical.physical()?;
        let projected = match physical.contract.algorithm_universe.as_ref() {
            Some(universe) => query.clone().with_algorithm_universe(universe).ok()?,
            None => query.clone(),
        };
        let axes = physical.authorized_query_upper(&projected, 3, false).ok()?;
        let point = NonNegativePlanningEstimatorV1::FittedResidualV1
            .prediction_detailed(&physical.fit, &axes)
            .ok()?;
        let coordinates = projected.input().joint_support_coordinates().to_vec();
        Some((point, coordinates, axes))
    }

    /// Offline original-source audit. It cannot alter a model or authorize a query.
    pub(crate) fn diagnose_archived_bound(&self, query: &StructuredQueryV2) -> Option<Value> {
        let physical = self.calibrated.fitted.numerical.physical()?;
        let projected = match physical.contract.algorithm_universe.as_ref() {
            Some(universe) => query.clone().with_algorithm_universe(universe).ok()?,
            None => query.clone(),
        };
        let axes = envelope::model_query_upper(
            &projected,
            &physical.contract.workload_domain,
            physical.contract.planning_estimator,
            false,
        )
        .ok()?;
        let fit = &physical.fit;
        let mut identified_comparison = fit.diagnose_identified_bound(&axes);
        if let Some(axis) = identified_comparison["maximum_normalized_support"]["axis"].as_u64() {
            identified_comparison["maximum_normalized_support"]["axis_label"] =
                json!(axis_label(projected.input(), axis as usize));
        }
        if let Some(details) =
            identified_comparison["signed"]["largest_positive_remainders"].as_array_mut()
        {
            for detail in details {
                if let Some(axis) = detail["axis"].as_u64() {
                    detail["axis_label"] = json!(axis_label(projected.input(), axis as usize));
                }
            }
        }
        let positive = fit.diagnose_bound(&axes).map(|bound| {
            json!({
                "upper_ns": bound.upper_ns,
                "certificate_index": bound.certificate_index,
                "limiting_axis": bound.limiting_axis,
                "axis_label": axis_label(projected.input(), bound.limiting_axis),
                "input_value": bound.input_value,
                "fit_maximum": bound.fit_maximum,
                // Existing archive examples copy this object wholesale. Keep
                // every original field and add only a bounded cold explanation.
                "identified_comparison": &identified_comparison,
            })
        });
        Some(json!({
            "fitted_point_ns": fit.predict_fitted(&axes).ok(),
            "positive_certificate": positive,
            "identified_envelope_ns": fit.predict_identified_envelope_detailed(&axes).ok().map(|v| v.upper_ns),
            "identified_bound_diagnostic": identified_comparison,
            "fit_certificate": fit.certificate(),
            "query_upper_axes": axes,
        }))
    }
}

impl CalibratedStructuredModelV2 {
    /// Cold audit access to the original fit, without cloning or refitting it.
    pub(crate) fn diagnostic_fit_certificate(&self) -> Option<&FitCertificate> {
        self.fitted.nonnegative_fit_certificate()
    }
}

impl FittedStructuredModelV2 {
    /// First failing numerical check in Residual order. This does not validate
    /// phase clocks/populations and cannot replace the original transition.
    pub fn diagnose_nonnegative_residual(
        &self,
        samples: &[StructuredNumericObservationV2],
    ) -> Option<Value> {
        let physical = self.numerical.physical()?;
        for sample in samples {
            let input = &sample.input;
            let checks = [
                ("same_domain", self.same_population_domain(input)),
                ("input_validation", input.validate(&self.settings)),
            ];
            for (gate, result) in checks {
                if let Err(error) = result {
                    return Some(sample_failure(physical, sample, gate, error));
                }
            }
            // Preserve the actual NumericalModel::actual_upper ordering. The
            // membership gate expands completion before inspecting Fit zeros.
            if let Err(error) = physical.membership(input) {
                return Some(sample_failure(physical, sample, "membership_axes", error));
            }
            if let Err(error) = physical.actual_upper(input) {
                return Some(sample_failure(physical, sample, "actual_upper", error));
            }
        }
        ChallengeCoverage::diagnose_phase(
            samples,
            &physical.fit.certificate().column_maxima,
            physical.contract.planning_estimator,
        )
    }
}

fn sample_failure(
    physical: &PhysicalEnvelope,
    sample: &StructuredNumericObservationV2,
    gate: &str,
    error: StructuredUnknown,
) -> Value {
    let input = &sample.input;
    let actual = envelope::model_axes(input, physical.contract.planning_estimator).ok();
    let membership = envelope::membership_axes(input).ok();
    let selected = if gate == "membership_axes" {
        &membership
    } else {
        &actual
    };
    let maxima = &physical.fit.certificate().column_maxima;
    let first_unseen = selected.as_ref().and_then(|axes| {
        axes.iter()
            .zip(maxima)
            .enumerate()
            .find(|(_, (x, max))| **x != 0 && **max == 0)
            .map(|(axis, (x, max))| {
                json!({
                    "basis_axis":axis,"axis_label":axis_label(input,axis),
                    "checked_value":x,"fit_maximum":max,
                    "actual_value":actual.as_ref().and_then(|v|v.get(axis)),
                    "membership_value":membership.as_ref().and_then(|v|v.get(axis))
                })
            })
    });
    json!({
        "gate":gate,"reason":format!("{error:?}"),"first_unseen_axis":first_unseen,
        "fit_basis_axes":maxima.len(),"input_basis_axes":input.basis.len(),
        "original_sample":{
            "ticket":sample.membership.offered_ordinal,"fifo":sample.ordinal,
            "member_ordinal":sample.membership.member_ordinal,"call_id":sample.call_id,
            "observed_at_ns":sample.observed_at_ns,"owner":input.owner(),
            "pending_positions":input.pending_positions,"length_positions":input.length_positions,
            "terminal_causes":input.settled_terminal_causes(),
            "completion_settled":input.completion.as_ref().map(|v|v.settled),
            "completion_basis_offset":input.completion.as_ref().map(|v|v.basis_offset),
            "repetition_basis_offset":input.repetition_offsets.map(|v|v.0),
            "host_rows":input.physical_host_rows
        }
    })
}

pub(super) fn axis_label(input: &StructuredInputV2, axis: usize) -> String {
    let moments = ["count", "position_sum", "position_square_sum"];
    if axis == 0 {
        return "intercept".into();
    }
    if (input.pending_basis_offset..input.pending_basis_offset + 3).contains(&axis) {
        return format!("pending.{}", moments[axis - input.pending_basis_offset]);
    }
    let length = input.pending_basis_offset - 3;
    if (length..length + 3).contains(&axis) {
        return format!("length.{}", moments[axis - length]);
    }
    if let Some(c) = &input.completion {
        if (c.basis_offset..c.basis_offset + 3).contains(&axis) {
            return format!("completion.{}", moments[axis - c.basis_offset]);
        }
    }
    if input.repetition_offsets.is_some_and(|(i, _)| i == axis) {
        return "repetition_tokens_sum".into();
    }
    // Layout is fixed by input::project_numeric. Device/replay offsets remain
    // explicitly unlabelled here rather than guessing an algorithm identity.
    let tail = length
        - 3 * usize::from(input.completion.is_some())
        - usize::from(input.repetition_offsets.is_some());
    let host = [
        "prefill_tokens",
        "attention_pairs",
        "sampling_history_sum",
        "decoded_text_bytes_sum",
        "decode_scratch_bytes_sum",
        "decode_rows",
        "prefill_rows",
        "no_generated_history_rows",
        "initial_prefill_rows",
        "final_prefill_rows",
        "mask_upload_required_rows",
        "token_producing_rows",
    ];
    if tail >= host.len() && (tail - host.len()..tail).contains(&axis) {
        return format!("host.{}", host[axis - (tail - host.len())]);
    }
    "device_or_replay_work".into()
}

#[cfg(test)]
impl QualifiedStructuredModelV2 {
    /// Original identified Fit only, using exact settled population axes as
    /// original Residual calibration does. No refit, support/clock mutation, or
    /// source enrollment; the caller first replays every original phase record.
    pub(crate) fn diagnose_original_actual_point(&self, input: &StructuredInputV2) -> Option<u64> {
        let fitted = &self.calibrated.fitted;
        fitted.same_population_domain(input).ok()?;
        input.validate(&fitted.settings).ok()?;
        let physical = fitted.numerical.physical()?;
        physical.contract.validate_input(input).ok()?;
        input.validate_actual_completion().ok()?;
        physical
            .fit
            .predict_fitted(&envelope::axes(input).ok()?)
            .ok()
    }
}

#[cfg(test)]
pub(super) fn diagnostic_axis(input: &StructuredInputV2, axis: usize) -> Value {
    let mut offset = 1usize;
    // with_algorithm_universe keeps local algorithm_axes while projecting basis.
    let layout = match &input.algorithm_universe {
        Some(universe) => serde_json::to_value(universe).unwrap()["algorithms"].clone(),
        None => serde_json::to_value(&input.algorithm_axes).unwrap(),
    };
    for value in layout.as_array().unwrap() {
        let kind = value["kind"].as_u64().unwrap();
        let labels: &[&str] = match kind {
            0 => &[
                "commands",
                "inner_work_units",
                "padded_units",
                "grid_blocks",
            ],
            1..=4 => &["commands", "bytes"],
            5 => &["commands", "logical_units", "inner_work_units"],
            _ => unreachable!(),
        };
        if (offset..offset + labels.len()).contains(&axis) {
            return json!({"kind":"declared_algorithm_work","algorithm":value,
                "coordinate":labels[axis-offset]});
        }
        offset += labels.len();
    }
    json!({"kind":axis_label(input,axis),"basis_axis":axis})
}

#[cfg(test)]
impl QualifiedStructuredModelV2 {
    pub(crate) fn diagnose_original_query_phase_coverage(
        &self,
        query: &StructuredQueryV2,
    ) -> Option<Value> {
        let fitted = &self.calibrated.fitted;
        let physical = fitted.numerical.physical()?;
        let projected = fitted.numerical_query(query).ok()?;
        fitted.same_query_population(&projected.input).ok()?;
        let upper = envelope::query_upper(&projected, &physical.contract.workload_domain).ok()?;
        Some(
            json!(physical.coverage.iter().enumerate().map(|(phase,c)|json!({
            "phase":phase,"coverage":c.as_ref().map(|c|c.diagnose_query_coverage(&projected,&upper))
        })).collect::<Vec<_>>()),
        )
    }
}

#[cfg(test)]
impl FittedStructuredModelV2 {
    pub(crate) fn diagnose_declared_fit_input(
        &self,
        input: &StructuredInputV2,
        now: u64,
    ) -> Option<StructuredInputV2> {
        check_time(now, self.frozen_at_ns, self.state.expires_at_ns).ok()?;
        let physical = self.numerical.physical()?;
        let input = physical.contract.project_input(input.clone()).ok()?;
        self.same_population_domain(&input).ok()?;
        (self.service_input_membership(&input).ok()?
            == StructuredServiceInputMembershipV1::Eligible)
            .then_some(input)
    }
    pub(crate) fn diagnose_frozen_residual_input_gate(
        &self,
        samples: &[StructuredNumericObservationV2],
    ) -> Option<Box<dyn Fn(&StructuredInputV2) -> bool>> {
        let estimator = self.numerical.physical()?.contract.planning_estimator;
        let coverage = ChallengeCoverage::observe_for(samples, estimator).ok()?;
        coverage.validate_branches(samples).ok()?;
        Some(Box::new(move |input| {
            envelope::membership_axes(input)
                .ok()
                .is_some_and(|axes| coverage.authorize_original_input(input, &axes).is_ok())
        }))
    }
}

#[cfg(test)]
impl FittedStructuredModelV2 {
    pub(crate) fn diagnose_original_actual_point(&self, input: &StructuredInputV2) -> Option<u64> {
        self.same_population_domain(input).ok()?;
        input.validate(&self.settings).ok()?;
        let physical = self.numerical.physical()?;
        physical.contract.validate_input(input).ok()?;
        input.validate_actual_completion().ok()?;
        physical
            .fit
            .predict_fitted(
                &envelope::model_axes(input, physical.contract.planning_estimator).ok()?,
            )
            .ok()
    }
    pub(crate) fn diagnose_fit_covered_point(
        &self,
        query: &StructuredQueryV2,
        now: u64,
    ) -> Option<u64> {
        check_time(now, self.frozen_at_ns, self.state.expires_at_ns).ok()?;
        let projected = self.numerical_query(query).ok()?;
        self.same_query_population(&projected.input).ok()?;
        projected.validate_for_prediction(&self.settings).ok()?;
        let physical = self.numerical.physical()?;
        let work = physical.authorized_query_upper(&projected, 1, false).ok()?;
        physical.fit.predict_fitted(&work).ok()
    }
    pub(crate) fn diagnose_frozen_phase_query_gate(
        &self,
        samples: &[StructuredNumericObservationV2],
    ) -> Option<Box<dyn Fn(&StructuredQueryV2) -> bool>> {
        let contract = self.numerical.physical()?.contract.clone();
        let coverage = ChallengeCoverage::observe_for(samples, contract.planning_estimator).ok()?;
        coverage.validate_branches(samples).ok()?;
        Some(Box::new(move |query| {
            let projected = match &contract.algorithm_universe {
                Some(u) => match query.clone().with_algorithm_universe(u) {
                    Ok(q) => q,
                    Err(_) => return false,
                },
                None => query.clone(),
            };
            let upper = if contract.planning_estimator.prospective_completion_work() {
                envelope::prospective_query_upper(&projected, &contract.workload_domain)
            } else {
                envelope::query_upper(&projected, &contract.workload_domain)
            };
            upper
                .ok()
                .is_some_and(|upper| coverage.authorize_detailed(&projected, &upper).is_ok())
        }))
    }
}
