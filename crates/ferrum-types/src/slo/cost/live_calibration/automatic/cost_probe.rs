//! Probe resource allowances, independent of numerical qualification rules.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SloAutomaticCostProbeSamplingPresetV1 {
    /// Resolved product defaults, preserving penalties, stop and model policy.
    Configured,
    /// A separate unpenalized greedy, ignore-EOS, complete-Length workload.
    /// Its model cannot stand in for a differently configured product owner.
    GreedyLength,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SloAutomaticPrefixTokenDiscoveryBudgetV1 {
    pub maximum_token_ids: NonZeroUsize,
    pub maximum_token_bytes: NonZeroUsize,
    pub maximum_total_token_bytes: NonZeroUsize,
    pub maximum_prefix_tokens: NonZeroUsize,
    pub maximum_utf8_transitions: NonZeroUsize,
    pub maximum_search_states: NonZeroUsize,
}
impl Default for SloAutomaticPrefixTokenDiscoveryBudgetV1 {
    fn default() -> Self {
        Self {
            maximum_token_ids: NonZeroUsize::new(262_144).unwrap(),
            maximum_token_bytes: NonZeroUsize::new(65_536).unwrap(),
            maximum_total_token_bytes: NonZeroUsize::new(16 * 1024 * 1024).unwrap(),
            maximum_prefix_tokens: NonZeroUsize::new(4).unwrap(),
            maximum_utf8_transitions: NonZeroUsize::new(1_048_576).unwrap(),
            maximum_search_states: NonZeroUsize::new(16_384).unwrap(),
        }
    }
}
impl SloAutomaticPrefixTokenDiscoveryBudgetV1 {
    pub fn validate(&self) -> Result<(), String> {
        if self.maximum_token_ids.get() > 1_048_576
            || self.maximum_token_bytes.get() > 1_048_576
            || self.maximum_total_token_bytes.get() > 128 * 1024 * 1024
            || self.maximum_token_bytes > self.maximum_total_token_bytes
            || self.maximum_prefix_tokens.get() > 64
            || self.maximum_utf8_transitions.get() > 16_777_216
            || self.maximum_search_states.get() > 65_536
        {
            return Err(
                "automatic prefix discovery exceeds bounded scan/byte/search capacities".into(),
            );
        }
        Ok(())
    }
}

/// Hitting any allowance leaves the uncompleted coverage Unknown. These
/// fields cannot lower sample minima, shorten a phase or reuse a fitting row
/// as independent residual/qualification evidence. Time bounds exclude the
/// necessary cancellation/retirement of already submitted work.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SloAutomaticCostProbeSettingsV1 {
    pub maximum_duration_ms: NonZeroU64,
    /// Includes preparation, outside-route and proven unsubmitted attempts.
    pub maximum_offered_waves: NonZeroUsize,
    /// Execution-cohort request reservations, including readiness, failed and
    /// proven unsubmitted attempts. Reservations are never refunded.
    pub maximum_probe_requests: NonZeroUsize,
    /// Cumulative owner reservations for input-only projection; no wave may be
    /// submitted from this allowance. Independent of execution requests, but
    /// sharing the deadline, retained bytes and active-owner limits. Recapture
    /// consumes new reservations. Both defaults are 2048, so total owner
    /// creation can exceed the execution-request allowance.
    pub maximum_input_projection_requests: NonZeroUsize,
    pub maximum_concurrent_requests: NonZeroUsize,
    pub maximum_output_tokens: NonZeroUsize,
    pub token_discovery: SloAutomaticPrefixTokenDiscoveryBudgetV1,
    pub sampling_presets: Vec<SloAutomaticCostProbeSamplingPresetV1>,
}
impl Default for SloAutomaticCostProbeSettingsV1 {
    fn default() -> Self {
        Self {
            maximum_duration_ms: NonZeroU64::new(120_000).unwrap(),
            maximum_offered_waves: NonZeroUsize::new(16_384).unwrap(),
            maximum_probe_requests: NonZeroUsize::new(2_048).unwrap(),
            maximum_input_projection_requests: NonZeroUsize::new(2_048).unwrap(),
            maximum_concurrent_requests: NonZeroUsize::new(8).unwrap(),
            maximum_output_tokens: NonZeroUsize::new(32).unwrap(),
            token_discovery: SloAutomaticPrefixTokenDiscoveryBudgetV1::default(),
            sampling_presets: vec![
                SloAutomaticCostProbeSamplingPresetV1::Configured,
                SloAutomaticCostProbeSamplingPresetV1::GreedyLength,
            ],
        }
    }
}
impl SloAutomaticCostProbeSettingsV1 {
    pub fn validate(&self) -> Result<(), String> {
        self.token_discovery.validate()?;
        if self.maximum_duration_ms.get() > MAX_PROBE_DURATION_MS
            || self.maximum_offered_waves.get() > 65_536
            || self.maximum_probe_requests.get() > 65_536
            || self.maximum_input_projection_requests.get() > 65_536
            || self.maximum_concurrent_requests.get() > 4_096
            || self.maximum_concurrent_requests > self.maximum_probe_requests
            || self.maximum_output_tokens.get() > 65_536
            || self.sampling_presets.is_empty()
            || self.sampling_presets.len() > 2
            || self
                .sampling_presets
                .iter()
                .enumerate()
                .any(|(i, p)| self.sampling_presets[..i].contains(p))
        {
            return Err("automatic cost probe requires bounded requests/work/time and unique sampling presets".into());
        }
        Ok(())
    }
}

#[cfg(test)]
#[path = "cost_probe/tests.rs"]
mod tests;
