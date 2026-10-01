//! Original live receipt -> V2 numerical observation. JSON and public numerical
//! fields cannot create the private settled provenance used by this adapter.
use super::super::PreparedStructuredFactsV2;
use super::*;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::{
    StructuredInputV2, StructuredMemberBindingV2, StructuredNumericObservationV2,
    StructuredUnknownV2,
};
use ferrum_types::FinishReason;

pub(in crate::continuous_engine::inner::cost_observation) struct ValidatedStructuredWaveV2 {
    pub stages: Arc<HostStageEvidenceV1>,
    pub actual: Arc<host_content::statistical::CompleteSelectedObservation>,
    input: StructuredInputV2,
    sessions: Box<[Arc<StructuredCaptureSessionBinding>]>,
    ordinal: u64,
}
impl ValidatedStructuredWaveV2 {
    pub fn into_member(
        self,
        membership: StructuredMemberBindingV2,
    ) -> Result<StructuredNumericObservationV2, StructuredUnknownV2> {
        let [session] = self.sessions.as_ref() else {
            return Err(StructuredUnknownV2::WrongSource);
        };
        self.member_for(session, membership)
    }
    pub fn member_for(
        &self,
        session: &Arc<StructuredCaptureSessionBinding>,
        membership: StructuredMemberBindingV2,
    ) -> Result<StructuredNumericObservationV2, StructuredUnknownV2> {
        if !self
            .sessions
            .iter()
            .any(|bound| Arc::ptr_eq(bound, session))
        {
            return Err(StructuredUnknownV2::WrongSource);
        }
        let input = self.input.clone();
        Ok(StructuredNumericObservationV2 {
            source: session.identity(),
            protocol: session.protocol(),
            ordinal: self.ordinal,
            membership,
            call_id: self.actual.call_id,
            fingerprint: self.actual.fingerprint.clone(),
            input,
            boundary: self.actual.boundary,
            outcome: self.actual.outcome,
            observed_at_ns: self.actual.observed_at_ns,
            wall_ns: self.actual.wall_ns,
        })
    }
    pub fn ordinal(&self) -> u64 {
        self.ordinal
    }
}

pub(in crate::continuous_engine::inner::cost_observation) fn validate_capture_v2(
    capture: &CostCalibrationCapture,
    reconciled: bool,
    prepared: &PreparedStructuredFactsV2,
) -> Result<ValidatedStructuredWaveV2, StructuredUnknownV2> {
    use StructuredUnknownV2 as U;
    if !reconciled {
        return Err(U::InvalidSample);
    }
    prepared.validate()?;
    let sessions = capture.structured_sessions();
    if sessions.is_empty() {
        return Err(U::WrongSource);
    }
    let ordinal = match capture.host_stage_queue() {
        Some(HostStageQueueReceipt {
            disposition: HostStageQueueDisposition::Published,
            accepted_ordinal: Some(ordinal),
        }) if ordinal > 0 => ordinal,
        _ => return Err(U::WrongSource),
    };
    let CostCalibrationStatus::Complete(result) = capture.status() else {
        return Err(U::InvalidSample);
    };
    let stages = capture.host_stages().ok_or(U::MissingEvidence)?;
    let entry = match result.as_ref() {
        CostCalibrationResult::Observed {
            sample,
            accepted_ordinal,
            disposition: CostCallDisposition::Published,
            ..
        } if *accepted_ordinal == Some(ordinal) => CostEvidenceEntry::Training {
            sample: (**sample).clone(),
            stages: Some(Arc::clone(&stages)),
        },
        CostCalibrationResult::Rejected(reason) => CostEvidenceEntry::StagesOnly {
            stages: Arc::clone(&stages),
            legacy_rejection: *reason,
        },
        _ => return Err(U::InvalidSample),
    };
    if sessions.iter().any(|session| {
        stages
            .prepare_started_at_ns
            .is_none_or(|at| at < session.opened_at_ns())
    }) {
        return Err(U::Clock);
    }
    let qualified = stages
        .structured_evidence
        .as_ref()
        .ok_or(U::MissingEvidence)?
        .as_ref()
        .map_err(|_| U::MissingEvidence)?;
    let shared = capture
        .structured_projection(&stages)
        .transpose()
        .map_err(|_| U::InvalidSample)?;
    let actual = if let Some(shared) = &shared {
        shared.actual.clone()
    } else {
        // Imported/copied/mutated diagnostics never receive a cached proof.
        qualified
            .validate_host_stages(&stages)
            .map_err(|_| U::InvalidSample)?;
        Arc::new(
            host_content::statistical::complete_structured_observation_v2(&entry)
                .map_err(|_| U::InvalidSample)?,
        )
    };
    if sessions
        .iter()
        .any(|session| &actual.fingerprint != session.fingerprint())
    {
        return Err(U::WrongFingerprint);
    }
    if actual.wall_ns != qualified.full_wall_ns()
        || actual.exact != prepared.exact
        || qualified.recipe() != prepared.recipe.as_ref()
        || stages.rows.len() != prepared.rows.len()
    {
        return Err(U::InvalidSample);
    }
    let recipe = Arc::clone(
        actual
            .selected
            .structured_capture()
            .ok_or(U::MissingEvidence)?
            .map_err(|_| U::MissingEvidence)?,
    );
    // The qualified producer holds the original attached Arc. Equal numbers
    // or a separately rebuilt recipe cannot replace that private receipt.
    if !std::ptr::eq(qualified.recipe(), recipe.as_ref()) {
        return Err(U::InvalidSample);
    }
    for ((row, declared), before) in stages
        .rows
        .iter()
        .zip(recipe.physical_host_rows())
        .zip(&prepared.rows)
    {
        if row.request_id != before.request_id
            || row.owner_incarnation != before.owner_incarnation
            || row.work_generation != before.work_generation
        {
            return Err(U::InvalidSample);
        }
        match (declared.terminal_expectation, row.terminal.as_ref()) {
            (HostTerminalExpectationV1::NoTokenProduced, None)
            | (HostTerminalExpectationV1::TokenMayTerminate, None) => {}
            (HostTerminalExpectationV1::LengthBoundary, Some(terminal))
                if terminal.finish_reason == FinishReason::Length => {}
            _ => return Err(U::UnsupportedScope),
        }
    }
    let input = match shared {
        Some(shared) => shared.base_input.clone(),
        None => StructuredInputV2::from_actual(&actual.exact, &actual.selected, &recipe)?,
    };
    Ok(ValidatedStructuredWaveV2 {
        stages,
        actual,
        input,
        sessions: sessions.to_vec().into_boxed_slice(),
        ordinal,
    })
}

/// Independent read-only discovery. A numerical input owns no source/FIFO,
/// member/phase coordinate or private receipt and cannot train the live model.
pub(in crate::continuous_engine) fn structured_discovery_input_v2(
    stages: &Arc<HostStageEvidenceV1>,
) -> Result<StructuredInputV2, StructuredUnknownV2> {
    structured_serving_observation_v2(stages).map(|(input, _)| input)
}

pub(in crate::continuous_engine) fn structured_capture_input_v2(
    capture: &CostCalibrationCapture,
    stages: &Arc<HostStageEvidenceV1>,
) -> Result<StructuredInputV2, StructuredUnknownV2> {
    match capture.structured_projection(stages) {
        Some(Ok(shared)) => {
            shared.feedback_scope?;
            Ok(shared.query.input().clone())
        }
        Some(Err(_)) => Err(StructuredUnknownV2::InvalidSample),
        None => structured_discovery_input_v2(stages),
    }
}

/// Common live settlement validation for discovery and runtime feedback. This
/// returns no calibration membership and cannot train or qualify a new model.
pub(in crate::continuous_engine::inner::cost_observation) fn structured_serving_observation_v2(
    stages: &Arc<HostStageEvidenceV1>,
) -> Result<
    (
        StructuredInputV2,
        host_content::statistical::CompleteSelectedObservation,
    ),
    StructuredUnknownV2,
> {
    use StructuredUnknownV2 as U;
    let qualified = stages
        .structured_evidence
        .as_ref()
        .ok_or(U::MissingEvidence)?
        .as_ref()
        .map_err(|_| U::MissingEvidence)?;
    qualified
        .validate_host_stages(stages)
        .map_err(|_| U::InvalidSample)?;
    let entry = CostEvidenceEntry::StagesOnly {
        stages: Arc::clone(stages),
        legacy_rejection: CostCallRejection::Composite,
    };
    let actual = host_content::statistical::complete_structured_observation_v2(&entry)
        .map_err(|_| U::InvalidSample)?;
    feedback_scope_for_actual(stages, &actual)?;
    let recipe = actual
        .selected
        .structured_capture()
        .ok_or(U::MissingEvidence)?
        .map_err(|_| U::MissingEvidence)?;
    let terminal_positions = stages
        .rows
        .iter()
        .enumerate()
        .filter_map(|(p, row)| row.terminal.as_ref().map(|_| p as u32))
        .collect::<Vec<_>>();
    let input = StructuredInputV2::from_actual(&actual.exact, &actual.selected, recipe)?
        .with_settled_completion(&terminal_positions)?;
    if input.regression_axes().len() > 4096 || input.joint_support_coordinates().len() > 4096 {
        return Err(U::Capacity);
    }
    Ok((input, actual))
}

/// Additional consumer scope, using the already validated original settlement.
/// It does not re-project or re-hash the numerical actual wave.
pub(in crate::continuous_engine::inner::cost_observation) fn feedback_scope_for_actual(
    stages: &HostStageEvidenceV1,
    actual: &host_content::statistical::CompleteSelectedObservation,
) -> Result<(), StructuredUnknownV2> {
    use StructuredUnknownV2 as U;
    let qualified = stages
        .structured_evidence
        .as_ref()
        .ok_or(U::MissingEvidence)?
        .as_ref()
        .map_err(|_| U::MissingEvidence)?;
    let recipe = actual
        .selected
        .structured_capture()
        .ok_or(U::MissingEvidence)?
        .map_err(|_| U::MissingEvidence)?;
    if actual.wall_ns != qualified.full_wall_ns()
        || !std::ptr::eq(qualified.recipe(), recipe.as_ref())
        || recipe.physical_host_rows().len() != stages.rows.len()
    {
        return Err(U::InvalidSample);
    }
    for (declared, row) in recipe.physical_host_rows().iter().zip(&stages.rows) {
        // Only this separately bound installed domain admits natural completion.
        // The original private receipt above has already proved generated-count,
        // host ordering and complete no-additional-work settlement.
        let natural = |reason| {
            matches!(
                declared.installed_policy.empirical_content_domain,
                Some(ferrum_interfaces::execution_cost::HostContentDomainV1::PlainTextInstalledV2(policy))
                    if match reason {
                        FinishReason::EOS => policy.model_eos,
                        FinishReason::Stop => policy.user_stop,
                        _ => false,
                    }
            )
        };
        match (declared.terminal_expectation, row.terminal.as_ref()) {
            (HostTerminalExpectationV1::NoTokenProduced, None)
            | (HostTerminalExpectationV1::TokenMayTerminate, None) => {}
            (HostTerminalExpectationV1::LengthBoundary, Some(terminal))
                if terminal.finish_reason == FinishReason::Length
                    || natural(terminal.finish_reason) => {}
            (HostTerminalExpectationV1::TokenMayTerminate, Some(terminal))
                if natural(terminal.finish_reason) => {}
            _ => return Err(U::UnsupportedScope),
        }
    }
    Ok(())
}
