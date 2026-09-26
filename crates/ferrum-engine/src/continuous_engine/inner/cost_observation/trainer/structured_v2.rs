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
    pub actual: host_content::statistical::CompleteSelectedObservation,
    source: [u8; 32],
    protocol: [u8; 32],
    ordinal: u64,
    recipe: Arc<UnsettledStructuredWaveEvidenceV1>,
}
impl ValidatedStructuredWaveV2 {
    pub fn into_member(
        self,
        membership: StructuredMemberBindingV2,
    ) -> Result<StructuredNumericObservationV2, StructuredUnknownV2> {
        let input = StructuredInputV2::from_actual(
            &self.actual.exact,
            &self.actual.selected,
            &self.recipe,
        )?;
        Ok(StructuredNumericObservationV2 {
            source: self.source,
            protocol: self.protocol,
            ordinal: self.ordinal,
            membership,
            call_id: self.actual.call_id,
            fingerprint: self.actual.fingerprint,
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
    let session = capture.structured_session().ok_or(U::WrongSource)?;
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
    if stages
        .prepare_started_at_ns
        .is_none_or(|at| at < session.opened_at_ns())
    {
        return Err(U::Clock);
    }
    let qualified = stages
        .structured_evidence
        .as_ref()
        .ok_or(U::MissingEvidence)?
        .as_ref()
        .map_err(|_| U::MissingEvidence)?;
    qualified
        .validate_host_stages(&stages)
        .map_err(|_| U::InvalidSample)?;
    let actual = host_content::statistical::complete_structured_observation_v2(&entry)
        .map_err(|_| U::InvalidSample)?;
    if &actual.fingerprint != session.fingerprint() {
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
    Ok(ValidatedStructuredWaveV2 {
        stages,
        actual,
        source: session.identity(),
        protocol: session.protocol(),
        ordinal,
        recipe,
    })
}

/// Independent read-only discovery. A numerical input owns no source/FIFO,
/// member/phase coordinate or private receipt and cannot train the live model.
pub(in crate::continuous_engine) fn structured_discovery_input_v2(
    stages: &Arc<HostStageEvidenceV1>,
) -> Result<StructuredInputV2, StructuredUnknownV2> {
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
        match (declared.terminal_expectation, row.terminal.as_ref()) {
            (HostTerminalExpectationV1::NoTokenProduced, None)
            | (HostTerminalExpectationV1::TokenMayTerminate, None) => {}
            (HostTerminalExpectationV1::LengthBoundary, Some(terminal))
                if terminal.finish_reason == FinishReason::Length => {}
            _ => return Err(U::UnsupportedScope),
        }
    }
    let input = StructuredInputV2::from_actual(&actual.exact, &actual.selected, recipe)?;
    if input.regression_axes().len() > 4096 || input.joint_support_coordinates().len() > 4096 {
        return Err(U::Capacity);
    }
    Ok(input)
}
