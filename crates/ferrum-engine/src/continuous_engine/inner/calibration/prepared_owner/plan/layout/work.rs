//! Offered inference and execution work are separate clocks. Native setup and
//! restore consume the execution budget without creating numerical members.
use super::*;
use crate::continuous_engine::inner::calibration::{
    cohort_driver::ProbePrefillPlan, startup::ProbePrefixAcquisitionPlan,
};
use ferrum_interfaces::vnext::CheckpointTokenSpanConstraint;

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub(in crate::continuous_engine::inner::calibration::prepared_owner::plan) struct PrefixBlueprint {
    pub prompt_tokens: usize,
    pub boundary: usize,
    pub span: CheckpointTokenSpanConstraint,
    pub input_tokens_sha256: [u8; 32],
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub(in crate::continuous_engine::inner::calibration) struct PreparedProbeAcquisition {
    template: usize,
    maximum_output: NonZeroUsize,
    preset: SloAutomaticCostProbeSamplingPresetV1,
    plan: ProbePrefixAcquisitionPlan,
    input_tokens_sha256: [u8; 32],
}

// A proper prefix has not sampled output. Future sampling and target admission
// remain each case's original policy, not part of the reusable checkpoint key.
impl PartialEq for PreparedProbeAcquisition {
    fn eq(&self, other: &Self) -> bool {
        self.template == other.template
            && self.plan == other.plan
            && self.input_tokens_sha256 == other.input_tokens_sha256
    }
}
impl Eq for PreparedProbeAcquisition {}

impl PreparedProbeAcquisition {
    pub fn plan(&self) -> ProbePrefixAcquisitionPlan {
        self.plan
    }
    pub fn template(&self) -> usize {
        self.template
    }
    pub fn prefill_chunk(&self) -> NonZeroU32 {
        self.plan.prefill_chunk()
    }

    fn validate_case_binding(&self, case: &Case) -> Result<()> {
        if matches!(case.prefix, PrefixKind::Ordinary)
            || self.template != case.template
            || self.maximum_output != case.maximum_output
            || self.preset != case.preset
        {
            return Err(error("native probe work differs from its frozen case"));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct CaseWork {
    pub declared_offers_upper: usize,
    pub declared_offers_minimum: usize,
    pub requests: usize,
    pub serial_declared_offer_rows: usize,
    pub execution_actions: usize,
    pub serial_token_work: usize,
}

pub(super) fn case_work(
    case: &Case,
    prompt: usize,
    whole: usize,
    row_ceiling: Option<NonZeroU32>,
) -> Result<CaseWork> {
    let whole = u32::try_from(whole)
        .ok()
        .and_then(NonZeroU32::new)
        .ok_or_else(|| error("invalid probe whole-wave capacity"))?;
    let chunk = case.prefill_chunk(whole, row_ceiling, case.width)?;
    let (remaining, restores) = match case.acquisition {
        Some(key) => {
            key.validate_case_binding(case)?;
            if key.plan.prompt_tokens() != prompt || key.plan.prefill_chunk() != chunk {
                return Err(error("native probe work differs from its frozen case"));
            }
            (
                prompt
                    .checked_sub(key.plan.boundary())
                    .filter(|n| *n > 0)
                    .ok_or_else(|| error("invalid native suffix"))?,
                case.width,
            )
        }
        None => (prompt, 0),
    };
    let chunks = remaining.div_ceil(chunk.get() as usize);
    let prefill = if matches!(case.prefix, PrefixKind::Ordinary) {
        ProbePrefillPlan::Joint
    } else {
        ProbePrefillPlan::PreparedSequentialV1
    };
    let declared_prefills = prefill
        .waves(chunks, case.width)
        .ok_or_else(|| error("probe offered prefill overflow"))?;
    let decodes = case.maximum_output.get() - 1;
    let declared_offers_upper = checked_add(declared_prefills, decodes)?;
    let declared_offers_minimum =
        if case.preset == SloAutomaticCostProbeSamplingPresetV1::GreedyLength {
            declared_offers_upper
        } else {
            checked_add(declared_prefills, case.release_generated)?
        };
    let serial_declared_offer_rows = checked_mul(checked_add(chunks, decodes)?, case.width)?;
    Ok(CaseWork {
        declared_offers_upper,
        declared_offers_minimum,
        requests: case.width,
        serial_declared_offer_rows,
        execution_actions: checked_add(serial_declared_offer_rows, restores)?,
        serial_token_work: checked_mul(checked_add(remaining, decodes)?, case.width)?,
    })
}

pub(super) fn declared_plan(
    case: &Case,
    blueprint: PrefixBlueprint,
    whole: NonZeroU32,
    row_ceiling: Option<NonZeroU32>,
) -> Result<Option<PreparedProbeAcquisition>> {
    if matches!(case.prefix, PrefixKind::Ordinary) {
        return Ok(None);
    }
    let chunk = case.prefill_chunk(whole, row_ceiling, case.width)?;
    let plan = ProbePrefixAcquisitionPlan::new(blueprint.prompt_tokens, blueprint.boundary, chunk)?;
    let last = (plan.boundary() - 1) % chunk.get() as usize + 1;
    if !blueprint.span.permits(last as u64) {
        return Ok(None);
    }
    Ok(Some(PreparedProbeAcquisition {
        template: case.template,
        maximum_output: case.maximum_output,
        preset: case.preset,
        plan,
        input_tokens_sha256: blueprint.input_tokens_sha256,
    }))
}

/// Each call describes one source. Selection and freezing use the same
/// first-occurrence rule without allocating another key collection.
pub(super) fn setup_for_cases(cases: &[Case]) -> Result<CaseWork> {
    let mut result = CaseWork::default();
    for (index, case) in cases.iter().enumerate() {
        let Some(key) = case.acquisition else {
            continue;
        };
        key.validate_case_binding(case)?;
        if cases[..index]
            .iter()
            .any(|old| old.acquisition == Some(key))
        {
            continue;
        }
        add_setup(&mut result, key)?;
    }
    Ok(result)
}

pub(super) fn setup_for_indices(cases: &[Case], indices: &[usize]) -> Result<CaseWork> {
    let mut result = CaseWork::default();
    for (position, &index) in indices.iter().enumerate() {
        let case = cases
            .get(index)
            .ok_or_else(|| error("setup case outside frozen source"))?;
        let Some(key) = case.acquisition else {
            continue;
        };
        key.validate_case_binding(case)?;
        if indices[..position]
            .iter()
            .any(|&old| cases.get(old).is_some_and(|v| v.acquisition == Some(key)))
        {
            continue;
        }
        add_setup(&mut result, key)?;
    }
    Ok(result)
}

pub(super) fn add_setup(total: &mut CaseWork, key: PreparedProbeAcquisition) -> Result<()> {
    let setup = key.plan.setup_work();
    let requests = checked_add(total.requests, setup.requests)?;
    let execution_actions = checked_add(total.execution_actions, setup.actions()?)?;
    let serial_token_work = checked_add(total.serial_token_work, key.plan.boundary())?;
    total.requests = requests;
    total.execution_actions = execution_actions;
    total.serial_token_work = serial_token_work;
    Ok(())
}

fn checked_add(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b)
        .ok_or_else(|| error("probe work accounting overflow"))
}

fn checked_mul(a: usize, b: usize) -> Result<usize> {
    a.checked_mul(b)
        .ok_or_else(|| error("probe work accounting overflow"))
}

#[cfg(test)]
mod tests;
