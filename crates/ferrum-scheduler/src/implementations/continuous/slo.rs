//! Small online controller for the shared prefill budget. These estimates guide
//! scheduling; they do not certify client-observed latency or SLO feasibility.

use ferrum_types::SchedulerSloConfig;
use serde::Serialize;
use std::time::Duration;

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
    /// The dynamic budget cannot meet even an optimistic active-prefill TTFT.
    pub ttft_budget_infeasible: bool,
    /// Scheduling iterations that retained static limits for that reason.
    pub ttft_fallback_steps: u64,
    /// Selected wave target before 10% headroom, including cumulative TPOT
    /// allowance. This can differ from user SLO thresholds.
    /// None means no dynamic budget, including fallback to static limits.
    pub budget_target_ms: Option<f64>,
    pub prefill_ms_per_token: Option<f64>,
    pub decode_step_ms: Option<f64>,
    pub fixed_step_ms: Option<f64>,
    pub decode_ms_per_sequence: Option<f64>,
}

#[derive(Default)]
pub(super) struct SloController {
    cost: OnlineAffineCost,
    observed_steps: u64,
    prefill_budget_tokens: Option<usize>,
    budget_updates: u64,
    adapted_prefill_steps: u64,
    decode_overload_steps: u64,
    decode_target_infeasible: bool,
    ttft_budget_infeasible: bool,
    ttft_fallback_steps: u64,
    budget_target_ms: Option<f64>,
    decode_step_ms: Option<f64>,
    pending_adapted_prefill: bool,
}

pub(super) struct PrefillTtftWork {
    pub remaining_tokens: usize,
    pub chunk_cap: usize,
    pub age_ms: f64,
}

impl SloController {
    pub(super) fn begin_iteration(&mut self) {
        // A failed or empty execution must never lend its adaptation marker to
        // a later successful wave. The engine schedules and executes serially.
        self.pending_adapted_prefill = false;
    }

    pub(super) fn prefill_ms_per_token(&self) -> Option<f64> {
        self.cost.fit().map(|cost| cost.prefill_ms_per_token)
    }

    pub(super) fn budget(
        &mut self,
        targets: SchedulerSloConfig,
        decode_sequences: usize,
        static_limit: usize,
        tpot_allowance_ms: Option<f64>,
        active_prefills: impl IntoIterator<Item = PrefillTtftWork>,
    ) -> Option<usize> {
        let previous = self.prefill_budget_tokens.take();
        self.decode_target_infeasible = false;
        self.ttft_budget_infeasible = false;
        self.budget_target_ms = None;
        self.decode_step_ms = None;
        if decode_sequences == 0 {
            return None;
        }
        // Three non-collinear naturally observed (decode, prefill) shapes are
        // needed to distinguish fixed overhead from both incremental costs.
        // Cold or unidentifiable models immediately retain static scheduling.
        let cost = self.cost.fit()?;
        let decode_ms = cost.fixed_ms + cost.decode_ms_per_sequence * decode_sequences as f64;
        self.decode_step_ms = Some(decode_ms);
        let strict_target_ms = targets.tpot_ms.min(targets.itl_ms);
        self.decode_target_infeasible = decode_ms >= strict_target_ms;
        // Prefill throttling cannot repair an already infeasible pure decode.
        // Keep the original strict targets and return to static limits.
        if self.decode_target_infeasible {
            return None;
        }
        // TPOT is an average over generated tokens, so earlier fast waves
        // can fund a longer mixed wave. ITL still limits each visible gap.
        let target_ms = targets
            .itl_ms
            .min(tpot_allowance_ms.unwrap_or(targets.tpot_ms));
        // Reserve 10% for host/transport work that execution timing does not
        // measure. This is headroom, not a promise about client-visible P99.
        let step_ms = target_ms * 0.9;
        let budget = (((step_ms - decode_ms).max(0.0) / cost.prefill_ms_per_token).floor()
            as usize)
            .max(1)
            .min(static_limit);
        // Optimistically give each active prefill the entire budget on every
        // wave. If even that cannot finish before TTFT, shrinking its progress
        // is not a feasible three-target choice. This is a necessary condition,
        // not a guarantee for shared budgets, future arrivals or transport.
        if active_prefills.into_iter().any(|request| {
            if request.remaining_tokens == 0 {
                return false;
            }
            let chunk = budget.min(request.chunk_cap);
            let completion_ms = if chunk == 0 {
                f64::INFINITY
            } else {
                cost.prefill_ms_per_token * request.remaining_tokens as f64
                    + decode_ms * request.remaining_tokens.div_ceil(chunk) as f64
            };
            completion_ms > (targets.ttft_ms - request.age_ms).max(0.0) * 0.9
        }) {
            self.ttft_budget_infeasible = true;
            self.ttft_fallback_steps = self.ttft_fallback_steps.saturating_add(1);
            return None;
        }
        self.budget_target_ms = Some(target_ms);
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
        if prefill_tokens > 0 && adapted {
            self.adapted_prefill_steps = self.adapted_prefill_steps.saturating_add(1);
        }
        if prefill_tokens == 0 && elapsed_ms > targets.tpot_ms.min(targets.itl_ms) {
            self.decode_overload_steps = self.decode_overload_steps.saturating_add(1);
        }
        // Fit all successful waves together; the intercept absorbs fixed work
        // instead of charging it once per token in small mixed chunks.
        self.cost
            .record(decode_sequences as f64, prefill_tokens as f64, elapsed_ms);
    }

    pub(super) fn snapshot(&self) -> SchedulerSloSnapshot {
        let cost = self.cost.fit();
        SchedulerSloSnapshot {
            enabled: true,
            observed_steps: self.observed_steps,
            prefill_budget_tokens: self.prefill_budget_tokens,
            budget_updates: self.budget_updates,
            adapted_prefill_steps: self.adapted_prefill_steps,
            decode_overload_steps: self.decode_overload_steps,
            decode_target_infeasible: self.decode_target_infeasible,
            ttft_budget_infeasible: self.ttft_budget_infeasible,
            ttft_fallback_steps: self.ttft_fallback_steps,
            budget_target_ms: self.budget_target_ms,
            prefill_ms_per_token: cost.map(|cost| cost.prefill_ms_per_token),
            decode_step_ms: self.decode_step_ms,
            fixed_step_ms: cost.map(|cost| cost.fixed_ms),
            decode_ms_per_sequence: cost.map(|cost| cost.decode_ms_per_sequence),
        }
    }
}

#[derive(Clone, Copy)]
struct AffineCost {
    fixed_ms: f64,
    decode_ms_per_sequence: f64,
    prefill_ms_per_token: f64,
}

/// Exponentially weighted, centered sufficient statistics for t = a + b*d + c*p.
/// No samples, per-width tables, startup calibration, or candidate search.
#[derive(Default)]
struct OnlineAffineCost {
    weight: f64,
    mean: [f64; 3],       // decode rows, prefill tokens, elapsed milliseconds
    covariance: [f64; 5], // dd, dp, pp, dt, pt
}

impl OnlineAffineCost {
    fn record(&mut self, decode: f64, prefill: f64, elapsed_ms: f64) {
        const DECAY: f64 = 0.8;
        let sample = [decode, prefill, elapsed_ms];
        let delta = std::array::from_fn::<_, 3, _>(|i| sample[i] - self.mean[i]);
        self.weight = DECAY * self.weight + 1.0;
        for (mean, change) in self.mean.iter_mut().zip(delta) {
            *mean += change / self.weight;
        }
        let correction = 1.0 - self.weight.recip();
        let products = [
            delta[0] * delta[0],
            delta[0] * delta[1],
            delta[1] * delta[1],
            delta[0] * delta[2],
            delta[1] * delta[2],
        ];
        for (covariance, product) in self.covariance.iter_mut().zip(products) {
            *covariance = DECAY * *covariance + correction * product;
        }
    }

    fn fit(&self) -> Option<AffineCost> {
        let [dd, dp, pp, dt, pt] = self.covariance;
        let determinant = dd * pp - dp * dp;
        // This relative rank check handles different row/token units. Two
        // shapes, or collinear shapes, cannot identify all three coefficients.
        if dd <= 0.0 || pp <= 0.0 || determinant <= 1e-9 * dd * pp {
            return None;
        }
        let decode_ms_per_sequence = (dt * pp - pt * dp) / determinant;
        let prefill_ms_per_token = (pt * dd - dt * dp) / determinant;
        let fixed_ms = self.mean[2]
            - decode_ms_per_sequence * self.mean[0]
            - prefill_ms_per_token * self.mean[1];
        if !fixed_ms.is_finite()
            || !decode_ms_per_sequence.is_finite()
            || !prefill_ms_per_token.is_finite()
            || fixed_ms < 0.0
            || decode_ms_per_sequence < 0.0
            || prefill_ms_per_token <= 0.0
        {
            return None;
        }
        Some(AffineCost {
            fixed_ms,
            decode_ms_per_sequence,
            prefill_ms_per_token,
        })
    }
}
