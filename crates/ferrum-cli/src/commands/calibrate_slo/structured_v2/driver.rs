use super::*;
use ferrum_engine::continuous_engine::{CalibrationSession, StructuredCapturePhase};
use ferrum_scheduler::implementations::continuous::cost_profile::export_structured_profile_v10;

pub(crate) async fn collect(
    session: &mut CalibrationSession,
    manifest: &manifest::Manifest,
    inputs: &inputs::PreparedInputs,
    artifacts: &mut super::super::report::Artifacts,
    summary: &mut super::super::report::Summary,
) -> Result<()> {
    use super::super::report::Phase;
    let capture = manifest
        .validation_model
        .structured_v2()
        .ok_or_else(|| FerrumError::internal("V2 driver needs its explicit manifest"))?;
    let options = capture.options(manifest, inputs, session)?;
    summary.structured_calibration_v2 = Some(StructuredReportV2::new(capture.scope.clone()));
    session
        .begin_structured_cost_calibration_v2(options)
        .await?;
    update_progress(session, summary);
    artifacts.record(&serde_json::json!({"schema_version":1,"event":"structured_v2_calibration_started","progress":session.structured_cost_progress_v2()}))?;
    for (cases, phase, expected) in [
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
        let mut expanded = 0;
        for (index, case) in cases.iter().enumerate() {
            for repetition in 0..case.repetitions.get() {
                session.begin_structured_cost_cohort_v2(expanded)?;
                super::super::driver::cohort(
                    session, manifest, inputs, case, phase, index, repetition, None, None,
                    artifacts, summary,
                )
                .await?;
                // The shared driver consumed terminal frames AND successful
                // output completion. Engine separately checks original owners.
                session.end_structured_cost_cohort_v2()?;
                expanded += 1;
                update_progress(session, summary);
            }
        }
        let coverage = session.structured_cost_coverage_v2()?;
        artifacts.record(&serde_json::json!({"schema_version":1,"event":"structured_v2_phase_coverage","phase":expected,"coverage":coverage}))?;
        summary
            .structured_calibration_v2
            .as_mut()
            .unwrap()
            .phase_coverage
            .push(report::PhaseCoverageV2 {
                phase: expected,
                coverage,
            });
        let receipt = session.freeze_structured_cost_phase_v2().await?;
        if receipt.phase != expected {
            return Err(FerrumError::internal(
                "V2 live freeze returned another phase",
            ));
        }
        artifacts.record(&serde_json::json!({"schema_version":1,"event":"structured_v2_phase_frozen","receipt":receipt}))?;
        summary
            .structured_calibration_v2
            .as_mut()
            .unwrap()
            .freezes
            .push(receipt);
        update_progress(session, summary);
    }
    Ok(())
}
fn update_progress(session: &CalibrationSession, summary: &mut super::super::report::Summary) {
    if let Some(report) = &mut summary.structured_calibration_v2 {
        report.progress = session.structured_cost_progress_v2();
    }
}
/// Outside the collection timeout. A failure cannot manufacture source closure,
/// a live receipt, or qualification; the schema10 exporter replays actual source.
pub(crate) async fn finish(
    session: &mut CalibrationSession,
    manifest: &manifest::Manifest,
    completed: bool,
    artifacts: &mut super::super::report::Artifacts,
    summary: &mut super::super::report::Summary,
) -> Result<()> {
    let Some(capture) = manifest.validation_model.structured_v2() else {
        return Ok(());
    };
    update_progress(session, summary);
    if session.structured_cost_progress_v2().is_none() {
        return if completed {
            Err(FerrumError::internal("V2 completed without live collector"))
        } else {
            Ok(())
        };
    }
    let source = session.finish_structured_cost_calibration_v2().await?;
    let source_receipt = report::SourceReceiptV2::from(&source);
    let eligible = report::require_exportable(&source_receipt, completed);
    let report = summary
        .structured_calibration_v2
        .as_mut()
        .ok_or_else(|| FerrumError::internal("V2 collector has no report owner"))?;
    report.source = Some(source_receipt);
    artifacts.record(&serde_json::json!({"schema_version":1,"event":"structured_v2_source_finished","receipt":report.source}))?;
    if !completed {
        return Ok(());
    }
    eligible?;
    let limits = structured::export_limits(
        &session
            .configuration()
            .scheduler
            .slo
            .cost_observation
            .profile_import,
    );
    report.exported_profile = Some(
        export_structured_profile_v10(
            &source.source_path,
            source.source_sha256,
            &capture.profile,
            capture.declared_source_clock_error_ns,
            &limits,
        )
        .map_err(|e| FerrumError::config(format!("export structured profile10: {e}")))?,
    );
    artifacts.record(&serde_json::json!({"schema_version":1,"event":"structured_v2_profile_exported","receipt":report.exported_profile}))?;
    Ok(())
}
