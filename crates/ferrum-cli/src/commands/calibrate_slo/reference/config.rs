use super::*;

impl ReferenceConfig {
    pub(in crate::commands::calibrate_slo) fn resolve_paths(&mut self, base: &Path) {
        for path in [&mut self.artifact_path, &mut self.frozen_plan_path] {
            if path.is_relative() {
                *path = base.join(&*path);
            }
        }
    }

    /// Called after input recovery, before any index reaches request(). Service
    /// output policy remains intact; independent reference requests use their declaration.
    pub(in crate::commands::calibrate_slo) fn validate(
        &self,
        manifest: &manifest::Manifest,
        inputs: &inputs::PreparedInputs,
    ) -> Result<()> {
        let (count, output) = input_lengths(manifest, inputs);
        self.validate_shape(manifest, count)?;
        self.request_policy.validate()?;
        let decode_output = self
            .request_policy
            .output_tokens(output(self.decode_unit.prompt_index));
        if decode_output <= self.decode_unit.generated_before.get() as usize {
            return Err(invalid("decode target must precede the declared reference request output limit; service limits are unchanged"));
        }
        if self
            .curve_prompt_indices
            .iter()
            .any(|&index| self.request_policy.output_tokens(output(index)) != decode_output)
        {
            return Err(invalid("V1 needs one reference host output policy; OriginalInput cannot combine distinct original max_tokens, while FixedReferenceOutput declares independent requests"));
        }
        Ok(())
    }

    /// Declaration-only bounds; schema 2 recovery supplies the exact input
    /// count and output policy to validate() before constructing the engine.
    pub(in crate::commands::calibrate_slo) fn validate_shape(
        &self,
        manifest: &manifest::Manifest,
        count: usize,
    ) -> Result<()> {
        self.limits.validate().map_err(invalid)?;
        if let Some(spec) = &self.piecewise {
            spec.validate().map_err(|e| invalid(e.to_string()))?;
            if spec.body_endpoints.len() + 2 > self.limits.max_points_per_curve.get() {
                return Err(invalid("piecewise partition exceeds point limit"));
            }
        }
        self.request_policy.validate()?;
        if !matches!(
            manifest.validation_model,
            manifest::ValidationSource::ExportedProfile { .. }
        ) {
            return Err(invalid(
                "reference publication requires exported_profile validation",
            ));
        }
        if self.artifact_path.as_os_str().is_empty()
            || self.frozen_plan_path.as_os_str().is_empty()
            || self.artifact_path == self.frozen_plan_path
            || self.curve_prompt_indices.is_empty()
            || self.curve_prompt_indices.len() > self.limits.max_curves.get()
            || self.repetitions.get() > 64
            || self.granule_tokens.get() > 1_048_576
            || self.warmup.len() > 256
        {
            return Err(invalid(
                "invalid paths, curve, granule, repetition or warmup bounds",
            ));
        }
        let mut unique = std::collections::BTreeSet::new();
        if self.decode_unit.prompt_index >= count
            || self
                .curve_prompt_indices
                .iter()
                .any(|index| *index >= count || !unique.insert(*index))
        {
            return Err(invalid(
                "reference input is absent or a curve index is duplicated",
            ));
        }
        let mut owners = 0usize;
        for case in manifest
            .training
            .iter()
            .chain(&manifest.validation)
            .chain(&self.warmup)
        {
            if case.prompts.is_empty()
                || case.prompts.len() > manifest.protocol.maximum_requests.get()
                || case.repetitions.get() > 64
                || case.prefill_chunk_tokens.get() > 1_048_576
                || case.prompts.iter().any(|&index| index >= count)
            {
                return Err(invalid("invalid warmup/cohort input or resource bounds"));
            }
            owners = owners
                .checked_add(
                    case.prompts
                        .len()
                        .checked_mul(case.repetitions.get())
                        .ok_or_else(|| invalid("owner count overflow"))?,
                )
                .ok_or_else(|| invalid("owner count overflow"))?;
        }
        owners = owners
            .checked_add(
                (self.curve_prompt_indices.len() + 1)
                    .checked_mul(self.repetitions.get() + 1)
                    .ok_or_else(|| invalid("owner count overflow"))?,
            )
            .ok_or_else(|| invalid("owner count overflow"))?;
        if owners > 65_536 {
            return Err(invalid(
                "all phases together exceed the calibration owner bound",
            ));
        }
        Ok(())
    }

    pub(in crate::commands::calibrate_slo) fn singleton_case(
        &self,
        target: DiscoveryTarget,
    ) -> Result<manifest::Cohort> {
        let index = match target {
            DiscoveryTarget::Prefill { curve } => *self
                .curve_prompt_indices
                .get(curve)
                .ok_or_else(|| invalid("unknown reference curve"))?,
            DiscoveryTarget::Decode => self.decode_unit.prompt_index,
        };
        Ok(manifest::Cohort {
            prompts: vec![index],
            repetitions: NonZeroUsize::MIN,
            prefill_chunk_tokens: self.granule_tokens,
            execution: manifest::Execution::Split,
            decode_route: ferrum_engine::continuous_engine::CalibrationDecodeRoute::Actual,
            token_policy_residency: manifest::TokenPolicyResidencyPolicy::Preserve,
        })
    }
}

fn input_lengths<'a>(
    manifest: &'a manifest::Manifest,
    inputs: &'a inputs::PreparedInputs,
) -> (usize, Box<dyn Fn(usize) -> usize + 'a>) {
    match inputs {
        inputs::PreparedInputs::Rendered => (
            manifest.prompts.len(),
            Box::new(move |index| manifest.prompts[index].sampling.max_tokens),
        ),
        inputs::PreparedInputs::ShareGpt { recovered, .. } => (
            recovered.prompts.len(),
            Box::new(move |index| recovered.prompts[index].sample.requested_output_tokens as usize),
        ),
    }
}

/// Only explicit singleton reference discovery/trials use this partition.
/// Normal training/heldout and warmup retain their declared chunk policy.
pub(in crate::commands::calibrate_slo) fn prefill_count(
    config: Option<&ReferenceConfig>,
    phase: super::super::report::Phase,
    total: u32,
    offset: u32,
    fallback: NonZeroU32,
) -> Result<NonZeroU32> {
    if matches!(
        phase,
        super::super::report::Phase::Discovery | super::super::report::Phase::Reference
    ) {
        if let Some(spec) = config.and_then(|value| value.piecewise.as_ref()) {
            return spec
                .next_count(total, offset)
                .map_err(|e| invalid(e.to_string()));
        }
    }
    NonZeroU32::new(
        total
            .checked_sub(offset)
            .ok_or_else(|| invalid("reference offset exceeds input"))?
            .min(fallback.get()),
    )
    .ok_or_else(|| invalid("no remaining prefill work"))
}
