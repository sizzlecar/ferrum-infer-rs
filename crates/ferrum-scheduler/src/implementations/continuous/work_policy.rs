//! Product work limits shared by logical planning and completion selection.
//! These values restrict proposals; they contain no resource authority or time estimate.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlanningWorkPolicy {
    pub maximum_wave_tokens: u64,
    pub prefill_step_chunk: Option<u64>,
    pub active_decode_prefill_chunk: Option<u64>,
    pub active_decode_prefill_token_budget: Option<u64>,
    pub default_active_decode_prefill_chunk: Option<u64>,
    pub active_decode_pressure_threshold: usize,
    pub capacity_deferred_decode: bool,
    pub allow_mixed: bool,
}

impl Default for PlanningWorkPolicy {
    fn default() -> Self {
        Self {
            maximum_wave_tokens: u64::MAX,
            prefill_step_chunk: None,
            active_decode_prefill_chunk: None,
            active_decode_prefill_token_budget: None,
            default_active_decode_prefill_chunk: None,
            active_decode_pressure_threshold: usize::MAX,
            capacity_deferred_decode: false,
            allow_mixed: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WaveWorkEnvelope {
    pub maximum_wave_tokens: u64,
    pub maximum_prefill_chunk: Option<u64>,
    pub maximum_prefill_tokens: Option<u64>,
    pub allow_mixed: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WaveWorkUsage {
    pub decode_rows: usize,
    pub prefill_rows: usize,
    pub tokens: u64,
    pub prefill_tokens: u64,
}

impl PlanningWorkPolicy {
    /// Additional finite proposal boundaries introduced by declared limits.
    /// Alignment, endpoint and physical legality are still checked separately.
    pub fn declared_prefill_chunks(self) -> impl Iterator<Item = u64> {
        [
            self.prefill_step_chunk,
            self.active_decode_prefill_chunk,
            self.active_decode_prefill_token_budget,
            self.default_active_decode_prefill_chunk,
        ]
        .into_iter()
        .flatten()
        .filter(|n| *n != 0)
    }

    /// Recompute after every logical edge, including final-prefill -> decode.
    /// A P-only wave in Split still sees the runnable decoders in its parent.
    pub fn for_ready_decoders(self, ready_decoders: usize) -> WaveWorkEnvelope {
        let active = if ready_decoders == 0 {
            None
        } else {
            self.active_decode_prefill_chunk.or_else(|| {
                if ready_decoders < self.active_decode_pressure_threshold
                    && !self.capacity_deferred_decode
                {
                    self.active_decode_prefill_token_budget.filter(|n| *n != 0)
                } else {
                    self.default_active_decode_prefill_chunk
                }
            })
        };
        let maximum_prefill_chunk = match (self.prefill_step_chunk, active) {
            (Some(step), Some(active)) => Some(step.min(active)),
            (step, active) => step.or(active),
        };
        WaveWorkEnvelope {
            maximum_wave_tokens: self.maximum_wave_tokens,
            maximum_prefill_chunk,
            maximum_prefill_tokens: (ready_decoders > 0)
                .then_some(self.active_decode_prefill_token_budget)
                .flatten()
                .filter(|n| *n != 0),
            allow_mixed: self.allow_mixed,
        }
    }
}

impl WaveWorkEnvelope {
    pub fn prefill_tokens_available(self, used: WaveWorkUsage) -> u64 {
        if !self.allow_mixed && used.decode_rows != 0 {
            return 0;
        }
        self.maximum_wave_tokens
            .saturating_sub(used.tokens)
            .min(self.maximum_prefill_chunk.unwrap_or(u64::MAX))
            .min(
                self.maximum_prefill_tokens
                    .map_or(u64::MAX, |total| total.saturating_sub(used.prefill_tokens)),
            )
    }

    /// No mutation on rejection. Independent of actual physical row ordering.
    pub fn include(self, used: &mut WaveWorkUsage, prefill_tokens: Option<NonZeroU64>) -> bool {
        let mut next = *used;
        let count = prefill_tokens.map_or(1, NonZeroU64::get);
        if let Some(prefill) = prefill_tokens {
            if prefill.get() > self.prefill_tokens_available(*used) {
                return false;
            }
            let Some(rows) = next.prefill_rows.checked_add(1) else {
                return false;
            };
            let Some(tokens) = next.prefill_tokens.checked_add(count) else {
                return false;
            };
            next.prefill_rows = rows;
            next.prefill_tokens = tokens;
        } else {
            if !self.allow_mixed && next.prefill_rows != 0 {
                return false;
            }
            let Some(rows) = next.decode_rows.checked_add(1) else {
                return false;
            };
            next.decode_rows = rows;
        }
        let Some(tokens) = next.tokens.checked_add(count) else {
            return false;
        };
        if tokens > self.maximum_wave_tokens {
            return false;
        }
        next.tokens = tokens;
        *used = next;
        true
    }
}

impl ContinuousBatchScheduler {
    /// Reuse the scheduler's resolved defaults, not a second engine policy.
    /// The deferred flag comes from the same sealed queue as the proposal.
    pub fn planning_work_policy(
        &self,
        hint: &BatchHint,
        allow_mixed: bool,
        capacity_deferred_decode: bool,
    ) -> PlanningWorkPolicy {
        PlanningWorkPolicy {
            maximum_wave_tokens: hint.max_tokens as u64,
            prefill_step_chunk: self.runtime_config.prefill_step_chunk.map(|n| n as u64),
            active_decode_prefill_chunk: self
                .runtime_config
                .active_decode_prefill_chunk
                .map(|n| n as u64),
            active_decode_prefill_token_budget: self
                .runtime_config
                .active_decode_prefill_token_budget
                .map(|n| n as u64),
            default_active_decode_prefill_chunk: Some(
                self.default_active_decode_prefill_chunk() as u64
            ),
            active_decode_pressure_threshold: self.decode_pressure_prefill_cap_threshold(hint),
            capacity_deferred_decode,
            allow_mixed,
        }
    }
}

#[cfg(test)]
mod tests;
