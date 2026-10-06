//! Small online controller for the shared prefill budget. These estimates guide
//! scheduling; they do not certify client-observed latency or SLO feasibility.

use ferrum_types::SchedulerSloConfig;
use serde::Serialize;
use std::{collections::BTreeMap, time::Duration};

/// Read on demand by health reporting; no serialization occurs during a step.
#[derive(Debug, Clone, Serialize)]
pub struct SchedulerSloSnapshot {
    pub enabled: bool,
    pub observed_steps: u64,
    /// Last mixed-iteration budget after static and live token limits. None
    /// means cold start, no runnable decoders, or overload fallback to static limits.
    pub prefill_budget_tokens: Option<usize>,
    pub budget_updates: u64,
    /// Successful prefill steps whose plan used the dynamic aggregate budget.
    pub adapted_prefill_steps: u64,
    /// Pure decode waves already longer than min(TPOT, ITL). Reducing prefill
    /// cannot fix these observations.
    pub decode_overload_steps: u64,
    /// Current decode estimate alone cannot meet the tighter TPOT/ITL target.
    pub decode_target_infeasible: bool,
    /// Selected wave target before 10% headroom, including cumulative TPOT
    /// allowance or overload fallback. This can differ from user SLO thresholds.
    /// None means no dynamic budget, including fallback to static limits.
    pub budget_target_ms: Option<f64>,
    pub prefill_ms_per_token: Option<f64>,
    pub decode_step_ms: Option<f64>,
}

#[derive(Default)]
pub(super) struct SloController {
    prefill_ms_per_token: Option<f64>,
    decode_ms_by_width: BTreeMap<usize, f64>,
    observed_steps: u64,
    prefill_budget_tokens: Option<usize>,
    budget_updates: u64,
    adapted_prefill_steps: u64,
    decode_overload_steps: u64,
    decode_target_infeasible: bool,
    budget_target_ms: Option<f64>,
    decode_step_ms: Option<f64>,
    pending_adapted_prefill: bool,
}

impl SloController {
    pub(super) fn begin_iteration(&mut self) {
        // A failed or empty execution must never lend its adaptation marker to
        // a later successful wave. The engine schedules and executes serially.
        self.pending_adapted_prefill = false;
    }

    pub(super) fn prefill_ms_per_token(&self) -> Option<f64> {
        self.prefill_ms_per_token
    }

    fn decode_ms(&self, width: usize) -> Option<f64> {
        if width == 0 {
            return Some(0.0);
        }
        // Use a measured larger width conservatively. Above the observed range
        // extrapolate linearly, without inventing a calibrated backend model.
        self.decode_ms_by_width
            .range(width..)
            .next()
            .map(|(_, ms)| *ms)
            .or_else(|| {
                self.decode_ms_by_width
                    .last_key_value()
                    .map(|(measured_width, ms)| ms * width as f64 / *measured_width as f64)
            })
    }

    pub(super) fn budget(
        &mut self,
        targets: SchedulerSloConfig,
        decode_sequences: usize,
        static_limit: usize,
        tpot_allowance_ms: Option<f64>,
    ) -> Option<usize> {
        let previous = self.prefill_budget_tokens.take();
        self.decode_target_infeasible = false;
        self.budget_target_ms = None;
        self.decode_step_ms = self.decode_ms(decode_sequences);
        if decode_sequences == 0 {
            return None;
        }
        let decode_ms = self.decode_step_ms?;
        let strict_target_ms = targets.tpot_ms.min(targets.itl_ms);
        self.decode_target_infeasible = decode_ms >= strict_target_ms;
        let target_ms = if self.decode_target_infeasible {
            targets.tpot_ms.max(targets.itl_ms)
        } else {
            // TPOT is an average over generated tokens, so earlier fast waves
            // can fund a longer mixed wave. ITL still limits each visible gap.
            targets
                .itl_ms
                .min(tpot_allowance_ms.unwrap_or(targets.tpot_ms))
        };
        // Reserve 10% for host/transport work that execution timing does not
        // measure. This is headroom, not a promise about client-visible P99.
        let step_ms = target_ms * 0.9;
        // Pure decode already violates the tighter target. Protect the wider
        // target when it still has room; otherwise use the static budget rather
        // than starving prefill one token at a time for an infeasible target.
        // This fallback does not change or certify the user's original SLOs.
        if self.decode_target_infeasible && decode_ms >= step_ms {
            return None;
        }
        let prefill_ms = self.prefill_ms_per_token?;
        self.budget_target_ms = Some(target_ms);
        let budget = (((step_ms - decode_ms).max(0.0) / prefill_ms).floor() as usize)
            .max(1)
            .min(static_limit);
        self.prefill_budget_tokens = Some(budget);
        if previous != Some(budget) {
            self.budget_updates = self.budget_updates.saturating_add(1);
        }
        Some(budget)
    }

    pub(super) fn mark_adapted_prefill(&mut self, adapted: bool) {
        self.pending_adapted_prefill = adapted;
    }

    pub(super) fn record(
        &mut self,
        targets: SchedulerSloConfig,
        prefill_tokens: usize,
        decode_sequences: usize,
        elapsed: Duration,
    ) {
        let adapted = std::mem::take(&mut self.pending_adapted_prefill);
        if (prefill_tokens == 0 && decode_sequences == 0) || elapsed.is_zero() {
            return;
        }
        let elapsed_ms = elapsed.as_secs_f64() * 1000.0;
        self.observed_steps = self.observed_steps.saturating_add(1);
        if prefill_tokens > 0 {
            if adapted {
                self.adapted_prefill_steps = self.adapted_prefill_steps.saturating_add(1);
            }
            // Mixed waves include dispatch and decode overhead that cannot be
            // reliably attributed per prefill token. Learning from tiny mixed
            // chunks makes costs inflate as chunks shrink; use pure prefills.
            if decode_sequences == 0 {
                update_cost(
                    &mut self.prefill_ms_per_token,
                    elapsed_ms / prefill_tokens as f64,
                );
            }
        } else {
            if elapsed_ms > targets.tpot_ms.min(targets.itl_ms) {
                self.decode_overload_steps = self.decode_overload_steps.saturating_add(1);
            }
            let mut cost = self.decode_ms_by_width.get(&decode_sequences).copied();
            update_cost(&mut cost, elapsed_ms);
            self.decode_ms_by_width
                .insert(decode_sequences, cost.unwrap());
        }
    }

    pub(super) fn snapshot(&self) -> SchedulerSloSnapshot {
        SchedulerSloSnapshot {
            enabled: true,
            observed_steps: self.observed_steps,
            prefill_budget_tokens: self.prefill_budget_tokens,
            budget_updates: self.budget_updates,
            adapted_prefill_steps: self.adapted_prefill_steps,
            decode_overload_steps: self.decode_overload_steps,
            decode_target_infeasible: self.decode_target_infeasible,
            budget_target_ms: self.budget_target_ms,
            prefill_ms_per_token: self.prefill_ms_per_token,
            decode_step_ms: self.decode_step_ms,
        }
    }
}

fn update_cost(estimate: &mut Option<f64>, sample: f64) {
    // React immediately to slow waves; a cheap wave only gradually increases
    // the next budget, avoiding an aggressive jump after one easy request.
    *estimate = Some(match *estimate {
        Some(previous) if sample < previous => previous * 0.8 + sample * 0.2,
        _ => sample,
    });
}
