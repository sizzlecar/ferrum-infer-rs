//! Independent discovery uses complete real cohorts and never opens a model source.
use super::super::report::{Artifacts, DiscoverySummaryV2, Phase, Summary};
use super::*;
use ferrum_engine::continuous_engine::CalibrationSession;

pub(crate) async fn collect(
    session: &mut CalibrationSession,
    manifest: &manifest::Manifest,
    inputs: &inputs::PreparedInputs,
    artifacts: &mut Artifacts,
    summary: &mut Summary,
) -> Result<()> {
    if !manifest.validation_model.is_discovery_v2() {
        return Err(FerrumError::internal(
            "discovery driver requires its explicit manifest",
        ));
    }
    if session
        .configuration()
        .scheduler
        .slo
        .cost_observation
        .structured_capture
        != ferrum_types::SloStructuredCostCapture::HostSettledV1
    {
        return Err(FerrumError::config(
            "discovery needs capture enabled on the actual session",
        ));
    }
    summary.structured_discovery_v2 = Some(DiscoverySummaryV2::new());
    complete_cases(
        session,
        manifest,
        inputs,
        manifest.validation_model.warmup_v2(),
        Phase::Warmup,
        artifacts,
        summary,
    )
    .await?;
    complete_cases(
        session,
        manifest,
        inputs,
        &manifest.training,
        Phase::Discovery,
        artifacts,
        summary,
    )
    .await?;
    summary
        .structured_discovery_v2
        .as_mut()
        .expect("created before requests")
        .collection_completed = true;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn complete_cases(
    session: &mut CalibrationSession,
    manifest: &manifest::Manifest,
    inputs: &inputs::PreparedInputs,
    cases: &[manifest::Cohort],
    phase: Phase,
    artifacts: &mut Artifacts,
    summary: &mut Summary,
) -> Result<()> {
    for (index, case) in cases.iter().enumerate() {
        for repetition in 0..case.repetitions.get() {
            // This shared driver consumes terminal wire AND real completion,
            // without shortening any request once a target is observed.
            super::super::driver::cohort(
                session, manifest, inputs, case, phase, index, repetition, None, None, artifacts,
                summary,
            )
            .await?;
        }
    }
    Ok(())
}
