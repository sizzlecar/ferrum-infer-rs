//! Live-only bridge from the original numbered FIFO/call capture. No JSON
//! deserializer can produce the private qualified receipt consumed here.
//! This does not install a model or change the normal training worker.
use super::super::calibration_capture::StructuredCaptureSessionBinding;
use super::*;
use ferrum_scheduler::implementations::continuous::cost_model::structured::{
    StructuredInputV1, StructuredNumericObservationV1, StructuredUnknown,
};
use ferrum_types::FinishReason;

/// Only the capture adapter below supplies the binding fixed before execution.
/// A future collector must additionally seal source bytes and original phase
/// freeze times; this helper cannot relabel raw JSON as live observations.
fn whole_wave_numeric_observation(
    entry: &CostEvidenceEntry,
    accepted_ordinal: u64,
    session: &StructuredCaptureSessionBinding,
) -> Result<StructuredNumericObservationV1, StructuredUnknown> {
    if accepted_ordinal == 0 {
        return Err(StructuredUnknown::WrongSource);
    }
    let stages = match entry {
        CostEvidenceEntry::Training { stages, .. } => stages.as_deref(),
        CostEvidenceEntry::StagesOnly { stages, .. } => Some(stages.as_ref()),
    }
    .ok_or(StructuredUnknown::MissingEvidence)?;
    if stages
        .prepare_started_at_ns
        .is_none_or(|at| at < session.opened_at_ns())
    {
        return Err(StructuredUnknown::Clock);
    }
    let qualified = stages
        .structured_evidence
        .as_ref()
        .ok_or(StructuredUnknown::MissingEvidence)?
        .as_ref()
        .map_err(|_| StructuredUnknown::MissingEvidence)?;
    qualified
        .validate_host_stages(stages)
        .map_err(|_| StructuredUnknown::InvalidSample)?;
    // This checks sample/stages identity, real clocks, physical rows, selected
    // exact binding and the host-settled boundary. A public sample field saying
    // Completed is not accepted as proof of any of those facts.
    let actual = host_content::statistical::complete_observation(entry)
        .map_err(|_| StructuredUnknown::InvalidSample)?;
    if &actual.fingerprint != session.fingerprint() {
        return Err(StructuredUnknown::WrongFingerprint);
    }
    if actual.wall_ns != qualified.full_wall_ns() {
        return Err(StructuredUnknown::InvalidSample);
    }
    let recipe = qualified.recipe();
    if recipe.physical_host_rows().len() != stages.rows.len() {
        return Err(StructuredUnknown::InvalidSample);
    }
    for (declared, observed) in recipe.physical_host_rows().iter().zip(&stages.rows) {
        match (declared.terminal_expectation, observed.terminal.as_ref()) {
            (HostTerminalExpectationV1::TokenMayTerminate, None) => {}
            (HostTerminalExpectationV1::LengthBoundary, Some(terminal))
                if terminal.finish_reason == FinishReason::Length => {}
            _ => return Err(StructuredUnknown::UnsupportedScope),
        }
    }
    let input = StructuredInputV1::from_future(&actual.exact, &actual.selected, recipe)?;
    Ok(StructuredNumericObservationV1 {
        source: session.identity(),
        protocol: session.protocol(),
        ordinal: accepted_ordinal,
        membership: None,
        call_id: actual.call_id,
        fingerprint: actual.fingerprint,
        input,
        boundary: actual.boundary,
        outcome: actual.outcome,
        observed_at_ns: actual.observed_at_ns,
        wall_ns: actual.wall_ns,
    })
}

/// The acceptance ordinal comes from the original host-stage queue receipt and
/// must agree with the original physical call result. It is never a re-counted
/// raw-file position, caller's arbitrary ordinal, or an inferred call ID.
pub(in crate::continuous_engine::inner::cost_observation) fn capture_numeric_observation(
    capture: &CostCalibrationCapture,
    reconciled: bool,
) -> Result<StructuredNumericObservationV1, StructuredUnknown> {
    if !reconciled {
        return Err(StructuredUnknown::InvalidSample);
    }
    let session = capture
        .structured_session()
        .ok_or(StructuredUnknown::WrongSource)?;
    let ordinal = match capture.host_stage_queue() {
        Some(HostStageQueueReceipt {
            disposition: HostStageQueueDisposition::Published,
            accepted_ordinal: Some(n),
        }) if n > 0 => n,
        _ => return Err(StructuredUnknown::WrongSource),
    };
    let CostCalibrationStatus::Complete(result) = capture.status() else {
        return Err(StructuredUnknown::InvalidSample);
    };
    let stages = capture.host_stages();
    let entry = match result.as_ref() {
        CostCalibrationResult::Observed {
            sample,
            accepted_ordinal,
            disposition: CostCallDisposition::Published,
            ..
        } if *accepted_ordinal == Some(ordinal) => CostEvidenceEntry::Training {
            sample: (**sample).clone(),
            stages,
        },
        CostCalibrationResult::Rejected(reason) => CostEvidenceEntry::StagesOnly {
            stages: stages.ok_or(StructuredUnknown::MissingEvidence)?,
            legacy_rejection: *reason,
        },
        _ => return Err(StructuredUnknown::InvalidSample),
    };
    whole_wave_numeric_observation(&entry, ordinal, session)
}

// Tests may inspect conversion failures after deliberately mutating diagnostic
// fields. Production callers only enter through the original bound capture.
#[cfg(test)]
pub(in crate::continuous_engine::inner::cost_observation) fn inspect_numeric_observation_for_test(
    entry: &CostEvidenceEntry,
    accepted_ordinal: u64,
    session: &StructuredCaptureSessionBinding,
) -> Result<StructuredNumericObservationV1, StructuredUnknown> {
    whole_wave_numeric_observation(entry, accepted_ordinal, session)
}
