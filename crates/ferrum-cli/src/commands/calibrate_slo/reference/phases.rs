use super::super::{driver, report};
use super::*;

/// Shared full-request driver for every phase. Observers can only collect
/// evidence; they cannot reduce max_tokens, pick a subset wave, or end a cohort.
pub(in crate::commands::calibrate_slo) async fn collect(
    session: &mut CalibrationSession,
    manifest: &manifest::Manifest,
    inputs: &inputs::PreparedInputs,
    artifacts: &mut report::Artifacts,
    summary: &mut report::Summary,
) -> Result<Option<FrozenReference>> {
    let Some(config) = &manifest.reference else {
        return Ok(None);
    };
    config.validate(manifest, inputs)?;
    for (index, case) in config.warmup.iter().enumerate() {
        for repetition in 0..case.repetitions.get() {
            driver::cohort(
                session,
                manifest,
                inputs,
                case,
                report::Phase::Warmup,
                index,
                repetition,
                None,
                None,
                artifacts,
                summary,
            )
            .await?;
        }
    }
    let mut discovered = DiscoverySet::new(config.clone(), manifest, inputs, session)?;
    for target in targets(config) {
        let case = config.singleton_case(target)?;
        let mut observer = discovered.observer(target)?;
        driver::cohort(
            session,
            manifest,
            inputs,
            &case,
            report::Phase::Discovery,
            case_index(config, target),
            0,
            None,
            Some(&mut observer),
            artifacts,
            summary,
        )
        .await?;
        discovered.accept(observer)?;
    }
    let mut frozen = discovered.freeze(session).await?;
    let receipt = serde_json::to_value(frozen.frozen_plan())
        .map_err(|error| invalid(format!("encode frozen plan receipt: {error}")))?;
    summary.reference_frozen_plan = Some(receipt.clone());
    artifacts.record(
        &serde_json::json!({"schema_version":1,"event":"reference_plan_frozen","receipt":receipt}),
    )?;
    for target in targets(config) {
        let case = config.singleton_case(target)?;
        for repetition in 0..config.repetitions.get() {
            let key = match target {
                DiscoveryTarget::Prefill { curve } => {
                    CalibrationReferenceTrial::Prefill { curve, repetition }
                }
                DiscoveryTarget::Decode => CalibrationReferenceTrial::Decode { repetition },
            };
            let mut observer = frozen.trial(key);
            driver::cohort(
                session,
                manifest,
                inputs,
                &case,
                report::Phase::Reference,
                case_index(config, target),
                repetition,
                None,
                Some(&mut observer),
                artifacts,
                summary,
            )
            .await?;
            observer.finish()?;
        }
    }
    Ok(Some(frozen))
}

fn targets(config: &ReferenceConfig) -> impl Iterator<Item = DiscoveryTarget> {
    (0..config.curve_prompt_indices.len())
        .map(|curve| DiscoveryTarget::Prefill { curve })
        .chain(std::iter::once(DiscoveryTarget::Decode))
}
fn case_index(config: &ReferenceConfig, target: DiscoveryTarget) -> usize {
    match target {
        DiscoveryTarget::Prefill { curve } => curve,
        DiscoveryTarget::Decode => config.curve_prompt_indices.len(),
    }
}
