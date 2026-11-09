//! One sequential visual/causal phase, with per-member immutable inputs kept
//! through later vision producers. Original ingress and output credits are
//! outside this grant. Materializer leases and conversion peaks stay distinct.
use super::*;
use crate::image_generation::p_h3::expansion::vision::ExpansionVisionHostRequirements;

pub(super) struct VisualRequirements {
    pub report: ExpansionVisionHostRequirements,
    pub decoded_rgba: usize,
    pub persistent_input: usize,
}
pub(super) struct CohortRequirements {
    pub budget: ExecutionHostBudget,
    pub startup_device_bytes: u64,
    _causal: ExpansionHostRequirements,
    _visual: Vec<ExpansionVisionHostRequirements>,
}
impl CohortRequirements {
    pub fn new(
        causal: ExpansionHostRequirements,
        visual: Vec<Option<VisualRequirements>>,
    ) -> Result<Self> {
        let budget = sequential_budget(
            causal.budget,
            causal.materializer_additional_bytes(),
            visual.iter().flatten().map(|member| {
                (
                    member.report.budget,
                    member.report.materializer_additional_bytes(),
                    member.decoded_rgba,
                    member.persistent_input,
                )
            }),
        )?;
        let startup_device_bytes = visual
            .iter()
            .flatten()
            .fold(causal.startup_device_bytes, |peak, member| {
                peak.max(member.report.startup_device_bytes)
            });
        let leases = visual
            .into_iter()
            .flatten()
            .map(|member| member.report)
            .collect();
        Ok(Self {
            budget,
            startup_device_bytes,
            _causal: causal,
            _visual: leases,
        })
    }
    pub fn claims(&self) -> [(HostMemoryClass, usize); 4] {
        self.budget.claims()
    }
}

fn sequential_budget(
    causal: ExecutionHostBudget,
    causal_materializer: usize,
    visual: impl Iterator<Item = (ExecutionHostBudget, usize, usize, usize)>,
) -> Result<ExecutionHostBudget> {
    let mut budget = causal;
    let mut retained_input = 0usize;
    let mut persistent_weights = causal_materializer;
    let mut transient_weights = causal
        .weights
        .checked_sub(causal_materializer)
        .context("causal weight budget excludes its materializer")?;
    for (report, persistent, decoded_rgba, persistent_input) in visual {
        budget.input = budget.input.max(
            retained_input
                .checked_add(decoded_rgba)
                .and_then(|n| n.checked_add(report.input))
                .context("cohort retained inputs and visual preprocessing overflow")?,
        );
        retained_input = retained_input
            .checked_add(persistent_input)
            .context("cohort retained visual inputs overflow")?;
        budget.input = budget.input.max(retained_input);
        budget.output = budget.output.max(report.output);
        budget.staging = budget.staging.max(report.staging);
        persistent_weights = persistent_weights
            .checked_add(persistent)
            .context("cohort materializer backing overflows")?;
        transient_weights = transient_weights.max(
            report
                .weights
                .checked_sub(persistent)
                .context("visual weight budget excludes its materializer")?,
        );
    }
    budget.weights = persistent_weights
        .checked_add(transient_weights)
        .context("cohort WeightBacking capacity overflows")?;
    Ok(budget)
}
#[cfg(test)]
mod tests;
