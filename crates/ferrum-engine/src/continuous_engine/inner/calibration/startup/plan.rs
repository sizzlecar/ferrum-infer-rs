use super::*;
use ferrum_scheduler::implementations::continuous::prefill_reference::{
    PiecewiseReferenceSpec, ReferenceEstimator, ReferenceGraphRoutes, ReferenceProtocolV1,
};
use ferrum_types::SloPrefillReferenceLimits;
use sha2::{Digest, Sha256};

pub(super) const OUTPUT_TOKENS: usize = 3;

#[derive(serde::Serialize)]
pub(super) struct ProbePrompt {
    pub text: String,
    pub tokens: NonZeroU32,
}

pub(super) struct ProbePlan {
    pub prompts: Vec<ProbePrompt>,
    pub partition: PiecewiseReferenceSpec,
    limits: SloPrefillReferenceLimits,
    chunk: NonZeroU32,
}

pub(super) use super::budget::ProbeBudget;

impl ProbePlan {
    pub fn new(
        session: &CalibrationSession,
        settings: &SloAutomaticReferenceProbeSettingsV1,
    ) -> Result<Self> {
        let inner = &session.engine.inner;
        let capacity = crate::continuous_engine::effective_request_context_capacity(
            &inner.config,
            &inner.runtime_config,
            inner.model_executor.kv_capacity(),
        )
        .unwrap_or(session.context_capacity())
        .min(session.context_capacity());
        let maximum = capacity
            .checked_sub(OUTPUT_TOKENS)
            .and_then(|n| u32::try_from(n).ok())
            .and_then(NonZeroU32::new)
            .ok_or_else(|| invalid("automatic reference context cannot fit a complete probe"))?;
        let maximum_bytes = usize::try_from(settings.maximum_source_bytes.get())
            .unwrap_or(usize::MAX)
            .min(ferrum_types::PREFILL_REFERENCE_MAX_FILE_BYTES);
        let mut limits = SloPrefillReferenceLimits::default();
        limits.max_file_bytes = NonZeroUsize::new(maximum_bytes)
            .ok_or_else(|| invalid("automatic reference byte budget is empty"))?;
        let prompts = prepare_prompts(inner.tokenizer.as_ref(), maximum, maximum_bytes)?;
        let minimum = prompts
            .first()
            .ok_or_else(|| invalid("automatic reference has no valid ordinary prompt"))?
            .tokens;
        let maximum = prompts.last().expect("nonempty prompts").tokens;
        let probe_passes = settings
            .fresh_trials_per_anchor
            .get()
            .checked_add(2)
            .ok_or_else(|| invalid("automatic reference request count overflow"))?;
        let requests = prompts
            .len()
            .checked_add(1)
            .and_then(|n| n.checked_mul(probe_passes))
            .ok_or_else(|| invalid("automatic reference request count overflow"))?;
        if 2usize
            .checked_mul(probe_passes)
            .is_none_or(|minimum| minimum > settings.maximum_probe_requests.get())
        {
            return Err(invalid(
                "minimum complete automatic reference protocol exceeds probe request budget",
            ));
        }
        // The frozen scoring protocol measures a separate final one-token
        // wave and may end a body segment at N-1. Require that logical quantum
        // from the executor; config alone cannot declare smaller native work.
        if inner.model_executor.guarded_prefill_granularity() != Some(NonZeroUsize::MIN) {
            return Err(invalid(
                "automatic reference requires declared one-token guarded prefill granularity",
            ));
        }
        let chunk = inner
            .config
            .batching
            .max_num_batched_tokens
            .min(
                inner
                    .config
                    .scheduler
                    .prefill_step_chunk
                    .unwrap_or(usize::MAX),
            )
            .min(
                inner
                    .runtime_config
                    .chunked_prefill_size
                    .unwrap_or(usize::MAX),
            );
        let chunk = NonZeroU32::new(u32::try_from(chunk).unwrap_or(u32::MAX).max(1))
            .expect("positive prefill chunk");
        let partition = chunk_partition(minimum, maximum, chunk, limits.max_points_per_curve)?;
        let segments = prompts.iter().try_fold(0usize, |sum, prompt| {
            let count = partition
                .segment_count(prompt.tokens.get())
                .map_err(|error| invalid(error.to_string()))?;
            sum.checked_add(count)
                .ok_or_else(|| invalid("reference sample count overflow"))
        })?;
        let decode_preparation = partition
            .segment_count(minimum.get())
            .map_err(|error| invalid(error.to_string()))?;
        let samples = segments
            .checked_add(decode_preparation)
            .and_then(|n| n.checked_add(1))
            .and_then(|n| n.checked_mul(settings.fresh_trials_per_anchor.get()))
            .and_then(|n| n.checked_add(segments + 1));
        if samples.is_none_or(|n| n > limits.max_samples.get())
            || segments
                .checked_add(prompts.len())
                .is_none_or(|n| n > limits.max_points.get())
            || prompts.len() > limits.max_curves.get()
        {
            return Err(invalid(
                "complete automatic reference evidence exceeds retention budget",
            ));
        }
        // Each anchor, including the separate decode anchor, has a complete
        // warmup and discovery request followed by the declared fresh trials.
        // These are planned successful waves, excluding maintenance and retries.
        let planned_prefill_tokens = prompts
            .iter()
            .try_fold(u64::from(minimum.get()), |sum, prompt| {
                sum.checked_add(u64::from(prompt.tokens.get()))
            })
            .and_then(|n| n.checked_mul(u64::try_from(probe_passes).ok()?))
            .ok_or_else(|| invalid("automatic reference planned token count overflow"))?;
        let planned_prefill_waves = segments
            .checked_add(decode_preparation)
            .and_then(|n| n.checked_mul(probe_passes))
            .ok_or_else(|| invalid("automatic reference planned wave count overflow"))?;
        let planned_decode_waves = requests
            .checked_mul(OUTPUT_TOKENS - 1)
            .ok_or_else(|| invalid("automatic reference planned wave count overflow"))?;
        let planned_successful_model_waves = planned_prefill_waves
            .checked_add(planned_decode_waves)
            .ok_or_else(|| invalid("automatic reference planned wave count overflow"))?;
        tracing::info!(
            prompt_anchors = prompts.len(),
            minimum_prompt_tokens = minimum.get(),
            maximum_prompt_tokens = maximum.get(),
            prefill_chunk_tokens = chunk.get(),
            fresh_trials_per_anchor = settings.fresh_trials_per_anchor.get(),
            probe_requests = requests,
            maximum_probe_requests = settings.maximum_probe_requests.get(),
            planned_prefill_tokens,
            planned_prefill_waves,
            planned_decode_waves,
            planned_successful_model_waves,
            "automatic reference candidate probe domain prepared"
        );
        Ok(Self {
            prompts,
            partition,
            limits,
            chunk,
        })
    }

    pub fn retain_prefix(&mut self, selected: usize) -> Result<()> {
        if selected == 0 || selected > self.prompts.len() {
            return Err(invalid(
                "automatic reference selected domain is not a nonempty prefix",
            ));
        }
        self.prompts.truncate(selected);
        self.partition = chunk_partition(
            self.prompts[0].tokens,
            self.prompts[selected - 1].tokens,
            self.chunk,
            self.limits.max_points_per_curve,
        )?;
        Ok(())
    }

    pub fn freeze(
        &self,
        session: &CalibrationSession,
        settings: &SloAutomaticReferenceProbeSettingsV1,
        curves: Vec<CalibrationReferenceCurve>,
        observations: &[CalibrationReferenceDiscoverySample],
        decode: &CalibrationReferenceDiscoverySample,
    ) -> Result<CalibrationReferencePlan> {
        let host = observations
            .first()
            .ok_or_else(|| invalid("missing reference discovery"))?
            .host_features();
        if observations
            .iter()
            .any(|sample| sample.host_features() != host)
            || decode.host_features().state.generated_tokens_before != 1
        {
            return Err(invalid(
                "automatic reference discovered incompatible singleton host policies",
            ));
        }
        let conditions = serde_json::to_vec(&(
            "automatic-reference-v1:plain-text-greedy-ignore-eos-three-output:graph-route-ready-discovery-v1",
            session.configuration(),
            settings,
            &self.prompts,
            &self.partition,
        ))
        .map_err(|error| invalid(error.to_string()))?;
        if conditions.len() > self.limits.max_file_bytes.get() {
            return Err(invalid(
                "automatic reference conditions exceed source bound",
            ));
        }
        Ok(CalibrationReferencePlan {
            reference_revision: NonZeroU64::MIN,
            protocol: ReferenceProtocolV1 {
                graph_routes: ReferenceGraphRoutes::ExactObserved,
                granule_tokens: NonZeroU32::MIN,
                repetitions: settings.fresh_trials_per_anchor,
                estimator: ReferenceEstimator::UpperMedianWallV1,
                input_preprocessing_sha256: Sha256::digest(
                    b"ferrum.automatic-reference.v1:ordinary-tokenizer-encode-with-special-tokens",
                )
                .into(),
                measurement_conditions_sha256: Sha256::digest(&conditions).into(),
                prefill_host: host,
                decode_host: decode.host_features(),
                decode_shape: decode.shape(),
            },
            piecewise: Some(self.partition.clone()),
            decode_input_tokens: self.prompts[0].tokens,
            decode_input_tokens_sha256: decode.request_evidence().original_input_tokens_sha256,
            curves,
            limits: self.limits.clone(),
        })
    }
}

/// Use the executor's effective chunk boundary for body work. Every shorter
/// anchor truncates this same partition at N-1 and retains a final one-token
/// wave; the partition is included in both protocol and conditions hashes.
fn chunk_partition(
    minimum: NonZeroU32,
    maximum: NonZeroU32,
    chunk: NonZeroU32,
    maximum_points: NonZeroUsize,
) -> Result<PiecewiseReferenceSpec> {
    let body_tokens = maximum.get() - 1;
    let endpoint_count = usize::try_from(body_tokens.div_ceil(chunk.get()))
        .map_err(|_| invalid("automatic reference partition size overflow"))?;
    if endpoint_count
        .checked_add(2)
        .is_none_or(|n| n > maximum_points.get())
    {
        return Err(invalid(
            "complete automatic reference partition exceeds point budget",
        ));
    }
    let mut body_endpoints = Vec::with_capacity(endpoint_count);
    let mut offset = 0u32;
    while offset < body_tokens {
        offset += chunk.get().min(body_tokens - offset);
        body_endpoints.push(NonZeroU32::new(offset).expect("positive endpoint"));
    }
    let partition = PiecewiseReferenceSpec {
        minimum_prompt_tokens: minimum,
        maximum_prompt_tokens: maximum,
        body_endpoints,
    };
    partition
        .validate()
        .map_err(|error| invalid(error.to_string()))?;
    Ok(partition)
}

/// Build bounded ordinary text and measure its actual tokenization. Only those
/// measured lengths enter the domain; byte count never stands in for tokens.
fn prepare_prompts(
    tokenizer: &(dyn Tokenizer + Send + Sync),
    maximum: NonZeroU32,
    maximum_bytes: usize,
) -> Result<Vec<ProbePrompt>> {
    let mut prompts = Vec::new();
    let mut retained = 0usize;
    let mut count = 1usize;
    let mut last_fitting = 0usize;
    let mut first_too_long = None;
    // This text is a protocol input, independent of model/client names.
    let make = |count: usize| " a".repeat(count);
    loop {
        let bytes = count
            .checked_mul(2)
            .ok_or_else(|| invalid("probe input bytes overflow"))?;
        if retained
            .checked_add(bytes)
            .is_none_or(|n| n > maximum_bytes)
        {
            return Err(invalid(
                "automatic reference input discovery exceeds source byte budget",
            ));
        }
        let text = make(count);
        let tokens = tokenizer.encode(&text, true)?.len();
        if tokens > maximum.get() as usize {
            first_too_long = Some(count);
            break;
        }
        let tokens = u32::try_from(tokens)
            .ok()
            .and_then(NonZeroU32::new)
            .ok_or_else(|| invalid("automatic reference tokenizer produced empty input"))?;
        last_fitting = count;
        if prompts
            .last()
            .is_none_or(|prompt: &ProbePrompt| prompt.tokens < tokens)
        {
            retained += text.len();
            prompts.push(ProbePrompt { text, tokens });
        }
        if tokens == maximum {
            break;
        }
        count = count
            .checked_mul(2)
            .ok_or_else(|| invalid("probe input count overflow"))?;
    }
    if let Some(mut high) = first_too_long {
        let mut low = last_fitting;
        let mut best = None;
        while high > low + 1 {
            let middle = low + (high - low) / 2;
            let text = make(middle);
            let tokens = tokenizer.encode(&text, true)?.len();
            if tokens > maximum.get() as usize {
                high = middle;
            } else {
                low = middle;
                if let Some(tokens) = u32::try_from(tokens).ok().and_then(NonZeroU32::new) {
                    best = Some(ProbePrompt { text, tokens });
                }
            }
        }
        if let Some(prompt) = best {
            if prompts
                .last()
                .is_none_or(|last| last.tokens < prompt.tokens)
            {
                if retained
                    .checked_add(prompt.text.len())
                    .is_none_or(|n| n > maximum_bytes)
                {
                    return Err(invalid(
                        "automatic reference input retention exceeds source bound",
                    ));
                }
                prompts.push(prompt);
            }
        }
    }
    Ok(prompts)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn budget_selected_domain_preserves_every_previously_discovered_partition() {
        let limits = SloPrefillReferenceLimits::default();
        let chunk = NonZeroU32::new(8).unwrap();
        let tokens = [1, 2, 8, 16, 31];
        let mut plan = ProbePlan {
            prompts: tokens
                .iter()
                .map(|n| ProbePrompt {
                    text: " a".repeat(*n as usize),
                    tokens: NonZeroU32::new(*n).unwrap(),
                })
                .collect(),
            partition: chunk_partition(
                NonZeroU32::MIN,
                NonZeroU32::new(31).unwrap(),
                chunk,
                limits.max_points_per_curve,
            )
            .unwrap(),
            limits,
            chunk,
        };
        let original = plan.partition.clone();
        plan.retain_prefix(4).unwrap();
        assert_eq!(plan.partition.maximum_prompt_tokens.get(), 16);
        assert!(plan.partition.next_count(17, 0).is_err());
        for prompt in &plan.prompts {
            let total = prompt.tokens.get();
            let mut offset = 0;
            while offset < total {
                let before = original.next_count(total, offset).unwrap();
                let after = plan.partition.next_count(total, offset).unwrap();
                assert_eq!(after, before);
                offset += after.get();
            }
        }
    }

    #[test]
    fn chunk_partition_covers_each_legal_prompt_with_a_separate_final_token() {
        let limits = SloPrefillReferenceLimits::default();
        // Enumerate capacities and smaller/equal/larger chunks, including
        // single-token prompts, exact boundaries and truncated final chunks.
        for capacity in OUTPUT_TOKENS as u32 + 1..=OUTPUT_TOKENS as u32 + 64 {
            let maximum = NonZeroU32::new(capacity - OUTPUT_TOKENS as u32).unwrap();
            for chunk in 1..=capacity + 1 {
                let chunk = NonZeroU32::new(chunk).unwrap();
                for minimum in [1, maximum.get().div_ceil(2), maximum.get()] {
                    let minimum = NonZeroU32::new(minimum).unwrap();
                    let partition =
                        chunk_partition(minimum, maximum, chunk, limits.max_points_per_curve)
                            .unwrap();
                    partition.validate().unwrap();
                    let mut previous = 0;
                    for endpoint in &partition.body_endpoints {
                        assert_eq!(
                            endpoint.get() - previous,
                            chunk.get().min(maximum.get() - 1 - previous)
                        );
                        previous = endpoint.get();
                    }
                    assert_eq!(previous, maximum.get() - 1);

                    for total in minimum.get()..=maximum.get() {
                        let mut offset = 0;
                        let mut segments = 0;
                        while offset < total {
                            let count = partition.next_count(total, offset).unwrap().get();
                            assert!(count <= chunk.get());
                            if offset == total - 1 {
                                assert_eq!(count, 1);
                            } else {
                                assert_eq!(count, chunk.get().min(total - 1 - offset));
                                assert!(offset + count < total);
                            }
                            offset += count;
                            segments += 1;
                        }
                        assert_eq!(offset, total);
                        assert_eq!(segments, partition.segment_count(total).unwrap());
                    }
                }
            }
        }
    }
}
