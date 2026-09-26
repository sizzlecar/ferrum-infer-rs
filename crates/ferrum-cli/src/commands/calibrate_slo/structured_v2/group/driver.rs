use super::super::super::report::{Artifacts, Phase, Summary};
use super::*;

pub(in crate::commands::calibrate_slo) async fn collect(
    session: &mut CalibrationSession,
    manifest: &manifest::Manifest,
    inputs: &inputs::PreparedInputs,
    artifacts: &mut Artifacts,
    summary: &mut Summary,
) -> Result<()> {
    let capture = manifest
        .validation_model
        .structured_group_v2()
        .ok_or_else(|| FerrumError::internal("group driver needs its explicit manifest"))?;
    // Freeze all child populations and the shared full request plan before any
    // warmup outcome. The core checks aggregate allocation/file bounds again.
    let options = capture.options(manifest, inputs, session)?;
    discovery::complete_cases(
        session,
        manifest,
        inputs,
        &capture.warmup,
        Phase::Warmup,
        artifacts,
        summary,
    )
    .await?;
    summary.structured_calibration_group_v2 = Some(GroupReportV2::new(capture));
    session.begin_structured_cost_group_v2(options).await?;
    update_progress(session, summary)?;
    artifacts.record(&serde_json::json!({"schema_version":1,"event":"structured_v2_group_started","progress":session.structured_cost_group_progress_v2()}))?;
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
                session.begin_structured_cost_group_cohort_v2(expanded)?;
                super::super::super::driver::cohort(
                    session, manifest, inputs, case, phase, index, repetition, None, None,
                    artifacts, summary,
                )
                .await?;
                // Complete credited terminal/output drain precedes the shared
                // boundary; no child changes phase when its own count fills.
                session.end_structured_cost_group_cohort_v2()?;
                expanded += 1;
                update_progress(session, summary)?;
            }
        }
        let coverage = session.structured_cost_group_coverage_v2()?;
        if coverage.len() != capture.children.len() {
            return Err(FerrumError::internal("group coverage cardinality changed"));
        }
        artifacts.record(&serde_json::json!({"schema_version":1,"event":"structured_v2_group_phase_coverage","phase":expected,"coverage":coverage}))?;
        let report = summary.structured_calibration_group_v2.as_mut().unwrap();
        for (child, coverage) in report.children.iter_mut().zip(coverage) {
            child.phase_coverage.push(report::PhaseCoverageV2 {
                phase: expected,
                coverage,
            });
        }
        let freezes = session.freeze_structured_cost_group_phase_v2().await?;
        if freezes.len() != capture.children.len()
            || freezes.iter().any(|r| r.phase != expected)
            || freezes.windows(2).any(|p| {
                p[0].frozen_at_ns != p[1].frozen_at_ns
                    || p[0].accepted_fifo_cutoff != p[1].accepted_fifo_cutoff
            })
        {
            return Err(FerrumError::internal(
                "group did not freeze a common phase/clock/FIFO boundary",
            ));
        }
        artifacts.record(&serde_json::json!({"schema_version":1,"event":"structured_v2_group_phase_frozen","receipts":freezes}))?;
        for (child, receipt) in summary
            .structured_calibration_group_v2
            .as_mut()
            .unwrap()
            .children
            .iter_mut()
            .zip(freezes)
        {
            child.freezes.push(receipt);
        }
        update_progress(session, summary)?;
    }
    Ok(())
}
fn update_progress(session: &CalibrationSession, summary: &mut Summary) -> Result<()> {
    if let (Some(report), Some(progress)) = (
        &mut summary.structured_calibration_group_v2,
        session.structured_cost_group_progress_v2(),
    ) {
        if report.children.len() != progress.len() {
            return Err(FerrumError::internal("group progress cardinality changed"));
        }
        for (child, progress) in report.children.iter_mut().zip(progress) {
            child.progress = Some(progress);
        }
    }
    Ok(())
}
pub(in crate::commands::calibrate_slo) async fn finish(
    session: &mut CalibrationSession,
    manifest: &manifest::Manifest,
    completed: bool,
    artifacts: &mut Artifacts,
    summary: &mut Summary,
) -> Result<()> {
    let Some(capture) = manifest.validation_model.structured_group_v2() else {
        return Ok(());
    };
    update_progress(session, summary)?;
    if session.structured_cost_group_progress_v2().is_none() {
        return if completed {
            Err(FerrumError::internal(
                "group completed without live collector",
            ))
        } else {
            Ok(())
        };
    }
    let source = session.finish_structured_cost_group_v2().await?;
    let report = summary
        .structured_calibration_group_v2
        .as_mut()
        .ok_or_else(|| FerrumError::internal("group has no report owner"))?;
    report.group_failure = source.failure.clone();
    if source.children.len() != report.children.len() {
        return Err(FerrumError::internal("group source cardinality changed"));
    }
    for (child, source) in report.children.iter_mut().zip(&source.children) {
        child.source = Some(report::SourceReceiptV2::from(source));
    }
    artifacts.record(&serde_json::json!({"schema_version":1,"event":"structured_v2_group_sources_finished","failure":report.group_failure,"sources":report.children.iter().map(|c| &c.source).collect::<Vec<_>>()}))?;
    if !completed {
        return Ok(());
    }
    // Validate ALL children before exporting the first one. Export and import
    // retain their own original source replay and expiration checks.
    export::require_group_exportable(report)?;
    export::export_and_inspect(session, capture, source, report, artifacts)?;
    Ok(())
}
