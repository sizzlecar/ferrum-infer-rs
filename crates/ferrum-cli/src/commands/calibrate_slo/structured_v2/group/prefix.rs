//! Static source5 shape validation; actual output budgets and tokenizer bytes
//! are independently bound after real input preparation and before collection.
use super::*;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::prefixes::StructuredPrefixPlanV5;

pub(in crate::commands::calibrate_slo) fn validate_prefix_v5(
    manifest: &manifest::Manifest,
    prefixes: &StructuredPrefixPlanV5,
) -> Result<()> {
    let bad = || {
        FerrumError::config("source5 requires one shared source, exact phase/cohort/slot declarations and fixed nonrolling preparation cohorts")
    };
    if manifest
        .validation_model
        .structured_group_v2()
        .is_none_or(|c| c.shared_source.is_none())
    {
        return Err(bad());
    }
    for (phase, cases) in [
        &manifest.training[..],
        manifest.validation_model.residual(),
        &manifest.validation[..],
    ]
    .into_iter()
    .enumerate()
    {
        let mut ordinal = 0;
        for case in cases {
            for _ in 0..case.repetitions.get() {
                let declared = prefixes.phases[phase].get(ordinal).ok_or_else(bad)?;
                ordinal += 1;
                if let Some(cohort) = declared {
                    if cohort.release_generated == 0
                        || cohort.slots.len() != case.prompts.len()
                        || case.rolling_window.is_some()
                        || case.wave_plan.is_some()
                    {
                        return Err(bad());
                    }
                    for (slot, &prompt) in cohort.slots.iter().zip(&case.prompts) {
                        if slot.token_ids.len() as u64 != cohort.release_generated {
                            return Err(bad());
                        }
                        slot.expected_pending().map_err(|_| bad())?;
                        if manifest.sharegpt.is_none()
                            && manifest.prompts.get(prompt).is_none_or(|p| {
                                cohort.release_generated >= p.sampling.max_tokens as u64
                            })
                        {
                            return Err(bad());
                        }
                    }
                }
            }
        }
        if ordinal != prefixes.phases[phase].len() {
            return Err(bad());
        }
    }
    Ok(())
}
