use super::*;
use ferrum_engine::continuous_engine::{CalibrationSession, StructuredCapturePhase};
use ferrum_scheduler::implementations::continuous::cost_profile::structured_v9::export_structured_profile_v9;

pub(crate) async fn collect(
    session: &mut CalibrationSession,
    manifest: &manifest::Manifest,
    inputs: &inputs::PreparedInputs,
    artifacts: &mut super::super::report::Artifacts,
    summary: &mut super::super::report::Summary,
) -> Result<()> {
    use super::super::report::Phase;
    let capture = manifest.validation_model.structured().ok_or_else(|| {
        FerrumError::internal("structured driver needs the explicit structured manifest")
    })?;
    summary.structured_calibration = Some(StructuredReport::new());
    session
        .begin_structured_cost_calibration(capture.options(manifest)?)
        .await?;
    update_progress(session, summary);
    artifacts.record(&serde_json::json!({
        "schema_version": 1, "event": "structured_calibration_started",
        "progress": session.structured_cost_progress(),
    }))?;
    for (cohorts, phase, expected) in [
        (
            &manifest.training[..],
            Phase::Training,
            StructuredCapturePhase::Fit,
        ),
        (
            manifest.validation_model.residual(),
            Phase::Residual,
            StructuredCapturePhase::Residual,
        ),
        (
            &manifest.validation[..],
            Phase::Qualification,
            StructuredCapturePhase::Qualification,
        ),
    ] {
        for (index, case) in cohorts.iter().enumerate() {
            for repetition in 0..case.repetitions.get() {
                // This shared driver drains every original request and credited
                // terminal before returning. Never freeze/cancel at a count.
                super::super::driver::cohort(
                    session, manifest, inputs, case, phase, index, repetition, None, None,
                    artifacts, summary,
                )
                .await?;
                update_progress(session, summary);
            }
        }
        update_progress(session, summary);
        let receipt = session.freeze_structured_cost_phase().await?;
        if receipt.phase != expected {
            return Err(FerrumError::internal(
                "structured live freeze returned the wrong phase",
            ));
        }
        artifacts.record(&serde_json::json!({
            "schema_version": 1, "event": "structured_phase_frozen", "receipt": receipt,
        }))?;
        summary
            .structured_calibration
            .as_mut()
            .expect("initialized above")
            .freezes
            .push(receipt);
        update_progress(session, summary);
    }
    Ok(())
}

fn update_progress(session: &CalibrationSession, summary: &mut super::super::report::Summary) {
    if let Some(report) = &mut summary.structured_calibration {
        if let Some(progress) = session.structured_cost_progress() {
            report.progress = Some(progress);
        }
    }
}

/// Called outside the collection timeout, before unconditional session shutdown.
/// Even an incomplete/failed phase is retained when the real session can close
/// its FIFO. A pending or indeterminate operation may prevent finalization; that
/// failure is reported explicitly and never replaced with a synthetic footer.
pub(crate) async fn finish(
    session: &mut CalibrationSession,
    manifest: &manifest::Manifest,
    completed: bool,
    artifacts: &mut super::super::report::Artifacts,
    summary: &mut super::super::report::Summary,
) -> Result<()> {
    let Some(capture) = manifest.validation_model.structured() else {
        return Ok(());
    };
    update_progress(session, summary);
    if session.structured_cost_progress().is_none() {
        return if completed {
            Err(FerrumError::internal(
                "structured calibration completed without a live collector",
            ))
        } else {
            Ok(())
        };
    }
    let source = session.finish_structured_cost_calibration().await?;
    let source_receipt = report::SourceReceipt::from(&source);
    let eligibility = report::require_exportable(&source_receipt, completed);
    let report = summary
        .structured_calibration
        .as_mut()
        .ok_or_else(|| FerrumError::internal("structured collector has no report owner"))?;
    report.source = Some(source_receipt);
    artifacts.record(&serde_json::json!({
        "schema_version": 1, "event": "structured_source_finished", "receipt": report.source,
    }))?;
    // Preserve the initial collection failure rather than add a redundant
    // qualification error; export remains forbidden on every failed path.
    if !completed {
        return Ok(());
    }
    eligibility?;
    let limits = config::export_limits(
        &session
            .configuration()
            .scheduler
            .slo
            .cost_observation
            .profile_import,
    );
    let receipt = export_structured_profile_v9(
        &source.source_path,
        source.source_sha256,
        &capture.profile,
        capture.declared_source_clock_error_ns,
        &limits,
    )
    .map_err(|error| FerrumError::config(format!("export structured profile9: {error}")))?;
    report.exported_profile = Some(receipt);
    artifacts.record(&serde_json::json!({
        "schema_version": 1, "event": "structured_profile_exported", "receipt": report.exported_profile,
    }))?;
    Ok(())
}
