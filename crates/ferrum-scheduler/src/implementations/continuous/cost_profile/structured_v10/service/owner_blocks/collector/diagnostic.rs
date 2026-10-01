//! Explicit cold inspection only. Original push/close remains the authority.
use super::*;
use serde_json::{json, Value};

// Fixed cold scratch, independent of sample/axis/owner counts. Every failed
// freeze is logged; only its additional input context is capped. Nothing here
// is retained by the collector or used by the numerical transition.
const MAX_FAILURE_CONTEXTS: usize = 8;

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct FitAxisContext {
    fit_axes: Option<usize>,
    fit_geometry_rank: Option<usize>,
    fit_epsilon_ns: Option<u64>,
    first_actual_axis_absent_from_fit: Option<(usize, f64)>,
    first_fit_positive_axis_absent_from_phase: Option<(usize, u64)>,
    missing_positive_axes: usize,
}
impl FitAxisContext {
    pub(super) fn capture(state: &State, samples: &[StructuredNumericObservationV2]) -> Self {
        let certificate = match state {
            State::Fitted(model) => model.nonnegative_fit_certificate(),
            State::Calibrated(model) => model.diagnostic_fit_certificate(),
            _ => None,
        };
        let Some(certificate) = certificate else {
            return Self::default();
        };
        let maxima = &certificate.column_maxima;
        let mut out = Self {
            fit_axes: Some(maxima.len()),
            fit_geometry_rank: Some(certificate.geometry_rank),
            fit_epsilon_ns: Some(certificate.epsilon_ns),
            ..Self::default()
        };
        for (axis, &maximum) in maxima.iter().enumerate() {
            let positive = samples
                .iter()
                .filter_map(|s| s.input.regression_axes().get(axis))
                .copied()
                .find(|&v| v > 0.);
            if maximum == 0 {
                if let Some(value) = positive {
                    out.first_actual_axis_absent_from_fit
                        .get_or_insert((axis, value));
                }
            } else if positive.is_none() {
                out.missing_positive_axes += 1;
                out.first_fit_positive_axis_absent_from_phase
                    .get_or_insert((axis, maximum));
            }
        }
        out
    }
}

/// Observed input facts, not a replacement for membership, completion-upper,
/// branch validation or a certificate of the reason for numerical rejection.
#[derive(Clone, Copy, Debug, Default)]
struct PhaseInputContext {
    samples: usize,
    rows_min: u32,
    rows_max: u32,
    axes_min: usize,
    axes_max: usize,
    wall_min_ns: u64,
    wall_max_ns: u64,
    pending_rows: usize,
    length_rows: usize,
    natural_termination_policy_rows: usize,
    early_opportunity_rows: usize,
    observed_early_rows: usize,
    observed_continuation_rows: usize,
    settled_samples: usize,
    fit: FitAxisContext,
}
impl PhaseInputContext {
    fn capture(samples: &[StructuredNumericObservationV2], fit: FitAxisContext) -> Self {
        use ferrum_interfaces::execution_cost::{HostContentDomainV1, HostTerminalExpectationV1};
        use ferrum_types::FinishReason;
        let Some(first) = samples.first() else {
            return Self {
                fit,
                ..Self::default()
            };
        };
        let mut out = Self {
            samples: samples.len(),
            rows_min: first.input.owner().rows,
            rows_max: first.input.owner().rows,
            axes_min: first.input.regression_axes().len(),
            axes_max: first.input.regression_axes().len(),
            wall_min_ns: first.wall_ns,
            wall_max_ns: first.wall_ns,
            fit,
            ..Self::default()
        };
        for sample in samples {
            let input = &sample.input;
            out.rows_min = out.rows_min.min(input.owner().rows);
            out.rows_max = out.rows_max.max(input.owner().rows);
            out.axes_min = out.axes_min.min(input.regression_axes().len());
            out.axes_max = out.axes_max.max(input.regression_axes().len());
            out.wall_min_ns = out.wall_min_ns.min(sample.wall_ns);
            out.wall_max_ns = out.wall_max_ns.max(sample.wall_ns);
            let causes = input.settled_terminal_causes();
            out.settled_samples += usize::from(causes.is_some());
            for row in input.physical_host_rows() {
                out.pending_rows += usize::from(row.pending_decoded_utf8);
                out.length_rows += usize::from(
                    row.terminal_expectation == HostTerminalExpectationV1::LengthBoundary,
                );
                let natural = matches!(row.installed_policy.empirical_content_domain,
                    Some(HostContentDomainV1::PlainTextInstalledV2(policy)) if policy.model_eos || policy.user_stop);
                out.natural_termination_policy_rows += usize::from(
                    natural
                        && row.terminal_expectation != HostTerminalExpectationV1::NoTokenProduced,
                );
                if natural
                    && row.terminal_expectation == HostTerminalExpectationV1::TokenMayTerminate
                {
                    out.early_opportunity_rows += 1;
                    if let Some(causes) = causes {
                        match causes.binary_search_by_key(&row.physical_position, |(p, _)| *p) {
                            Ok(i)
                                if matches!(
                                    causes[i].1,
                                    FinishReason::EOS | FinishReason::Stop
                                ) =>
                            {
                                out.observed_early_rows += 1
                            }
                            Err(_) => out.observed_continuation_rows += 1,
                            _ => {}
                        }
                    }
                }
            }
        }
        out
    }
}

#[derive(Default)]
pub(super) struct BlockCloseDiagnostics {
    contexts: [Option<(u64, PhaseInputContext)>; MAX_FAILURE_CONTEXTS],
    used: usize,
}
impl BlockCloseDiagnostics {
    pub(super) fn has_capacity(&self) -> bool {
        self.used < self.contexts.len()
    }
    pub(super) fn capture(
        &mut self,
        owner: u64,
        samples: &[StructuredNumericObservationV2],
        axes: Option<FitAxisContext>,
    ) {
        if self.has_capacity() {
            self.contexts[self.used] = Some((
                owner,
                PhaseInputContext::capture(samples, axes.unwrap_or_default()),
            ));
            self.used += 1;
        }
    }
    pub(super) fn emit(
        &self,
        collector: &StructuredServiceCollectorV7,
        record: &StructuredServiceRecordV7,
    ) {
        let StructuredServiceRecordV7::BlockClose {
            block,
            offered,
            freezes,
            ..
        } = record
        else {
            return;
        };
        let source_schema = match collector.header.source_kind {
            PopulationSource::OwnerBlocksV7 => 7u32,
            PopulationSource::PreparedOwnerBlocksV8 => 8,
        };
        let mut failures = 0usize;
        for freeze in freezes.iter().filter(|freeze| freeze.failure.is_some()) {
            failures += 1;
            let owner = collector
                .owners
                .iter()
                .find(|owner| owner.contract.owner_attempt_id == freeze.owner_attempt_id);
            let context = self
                .contexts
                .iter()
                .flatten()
                .find(|(id, _)| *id == freeze.owner_attempt_id)
                .map(|(_, context)| context);
            // Use the immutable original close, never the now-cleared state.
            tracing::warn!(target: "ferrum_scheduler::structured_owner_diagnostics",
                event = "structured_owner_phase_failure_v1", source_schema, block,
                capture_identity = ?collector.header.capture_identity,
                owner_attempt_id = freeze.owner_attempt_id, phase = ?freeze.close.phase,
                owner_rows = ?owner.map(|owner| owner.scope.owner.rows),
                owner_role = ?owner.map(|owner| owner.scope.owner.role),
                owner_product = ?owner.map(|owner| owner.scope.owner.product),
                numerical_family = owner.is_some_and(|owner| owner.scope.numerical_family.is_some()),
                member_count = freeze.close.member_count,
                first_offered = freeze.close.boundary.first_offered,
                last_offered = freeze.close.boundary.last_offered,
                opening_fifo_cutoff = freeze.close.boundary.opening_fifo_cutoff,
                closing_fifo_cutoff = freeze.close.boundary.closing_fifo_cutoff,
                owner_offered = freeze.domain.owner_offered, eligible = freeze.domain.eligible,
                outside_fit_support = freeze.domain.outside_fit_support,
                outside_residual_support = freeze.domain.outside_residual_support,
                unclassified_failed_owner = freeze.domain.unclassified_failed_owner,
                reason = freeze.failure.as_deref().unwrap_or(""),
                input_context = ?context, input_context_truncated = context.is_none(),
                "Original owner phase failed at accepted BlockClose");
        }
        let mut waiting = [0usize; 3];
        let mut pending_members = [0usize; 3];
        let mut failed = 0usize;
        for owner in &collector.owners {
            if let Some(phase) = owner.state.phase() {
                let index = match phase {
                    StructuredPhaseV2::Fit => 0,
                    StructuredPhaseV2::Residual => 1,
                    StructuredPhaseV2::Qualification => 2,
                };
                waiting[index] += 1;
                pending_members[index] = pending_members[index].saturating_add(owner.samples.len());
            }
            failed += usize::from(matches!(owner.state, State::Failed(_)));
        }
        let qualified = collector.qualified_children();
        if failed != 0 || qualified == 0 {
            tracing::warn!(target: "ferrum_scheduler::structured_owner_diagnostics",
                event = "structured_owner_block_summary_v1", source_schema, block, offered,
                capture_identity = ?collector.header.capture_identity,
                owners = collector.owners.len(), qualified, failed,
                all_owners_failed = !collector.owners.is_empty() && failed == collector.owners.len(),
                newly_failed = failures, waiting_fit = waiting[0], waiting_residual = waiting[1],
                waiting_qualification = waiting[2], pending_fit_members = pending_members[0],
                pending_residual_members = pending_members[1], pending_qualification_members = pending_members[2],
                phase_min_offered = ?collector.header.declaration.schedule.phase_min_offered,
                phase_min_members = ?collector.header.declaration.schedule.min_members,
                omitted_input_contexts = failures.saturating_sub(self.used),
                "Original owner block qualification summary");
        }
    }
}

impl StructuredServiceCollectorV7 {
    /// Inspect the original pending Residual samples before BlockClose consumes
    /// them. Callers must validate that original close before reporting these
    /// provisional numerical facts. No mutation, refit, or synthetic samples.
    pub fn diagnose_pending_residuals(&self) -> Vec<Value> {
        if self.poisoned
            || self.closed
            || self.opened.is_none()
            || self.block_count != self.header.declaration.schedule.block_offered
        {
            return Vec::new();
        }
        self.owners
            .iter()
            .filter_map(|owner| {
                let State::Fitted(model) = &owner.state else {
                    return None;
                };
                let boundary = owner.boundary?;
                let offers = self
                    .offered
                    .checked_sub(boundary.first_offered)?
                    .checked_add(1)?;
                if owner
                    .contract
                    .schedule
                    .is_ready(StructuredPhaseV2::Residual, offers, owner.samples.len())
                    .ok()
                    != Some(true)
                {
                    return None;
                }
                model
                    .diagnose_nonnegative_residual(&owner.samples)
                    .map(|detail| {
                        json!({
                            "owner_attempt_id":owner.contract.owner_attempt_id,
                            "owner":owner.scope.owner,"phase":"residual",
                            "global_offers":offers,"members":owner.samples.len(),"detail":detail
                        })
                    })
            })
            .collect()
    }
}
#[cfg(test)]
impl StructuredServiceCollectorV7 {
    pub(crate) fn previous_completion_fit_bindings_for_test(&self) -> Vec<(u64, [u8; 32])> {
        self.owners
            .iter()
            .filter_map(|owner| match &owner.state {
                State::Fitted(model) => Some((
                    owner.contract.owner_attempt_id,
                    model.previous_completion_binding_for_test(),
                )),
                _ => None,
            })
            .collect()
    }
}
