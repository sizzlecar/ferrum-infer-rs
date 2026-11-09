//! One shared phase floor proves that a cohort can produce its first decoded
//! image without waiting on credit held by its own request lanes.
use super::*;
use std::collections::BTreeMap;

impl<R: DeviceRuntime> PH3ImagePipeline<R> {
    pub(super) fn continuous_phase_budget(
        &self,
        steady: &CompiledSteady<R>,
        requests: &[&mut ContinuousRequest<R>],
        previous: Option<&ContinuousBatch<R>>,
    ) -> Result<(
        ExecutionHostBudget,
        BTreeMap<String, HostBackingPreparation>,
    )> {
        let step = self.steady_host_requirements(steady)?;
        let mut phase = step;
        let mut backing = BTreeMap::new();
        for request in requests {
            let decoder = &request.decoder;
            let requirements = decoder.recipe.host_requirements(&self.execution.host)?;
            phase.input = phase.input.max(requirements.input_peak_bytes);
            // A returned canvas can coexist with the next real velocity
            // readback. Count that canvas once, plus the larger work phase.
            phase.output = phase.output.max(requirements.output_peak_bytes).max(
                requirements
                    .canvas_output_bytes
                    .checked_add(step.output)
                    .context("H3 decoded canvas plus survivor velocity backing overflows")?,
            );
            for stage in &decoder.stages {
                let (weights, retention) = self.execution.stage_host_backing(stage)?;
                phase.weights = phase.weights.max(weights);
                if self.execution.runtime.requires_managed_static_staging() {
                    phase.staging = phase.staging.max(host_budget::staging_bytes(
                        &self.execution.host,
                        stage,
                        self.execution.config.static_initialization,
                    )?);
                }
                let key = stage
                    .executable()
                    .execution_plan()
                    .payload()
                    .execution_weights()
                    .stored()?
                    .fingerprint()?;
                backing.entry(key).or_insert(retention);
            }
        }
        if let Some(previous) = previous {
            // Outgoing completed lanes have not decoded yet. Preserve the old
            // larger phase floor until this stream's work owner retires; this
            // is one shared phase allowance, never one canvas per request.
            phase.input = phase.input.max(previous.work_budget.input);
            phase.output = phase.output.max(previous.work_budget.output);
            phase.staging = phase.staging.max(previous.work_budget.staging);
            phase.weights = phase.weights.max(previous.work_budget.weights);
        }
        Ok((phase, backing))
    }
}
