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
    /// means cold start, no runnable decoders, or unchanged static limits.
    pub prefill_budget_tokens: Option<usize>,
    pub budget_updates: u64,
    /// Successful prefill steps whose work was reduced by the dynamic budget.
    pub adapted_prefill_steps: u64,
    /// Pure decode waves already longer than ITL. Reducing prefill
    /// cannot fix these observations.
    pub decode_overload_steps: u64,
    /// Current decode estimate alone cannot meet the ITL target.
    pub decode_target_infeasible: bool,
    /// The dynamic budget cannot meet even an optimistic active-prefill TTFT.
    pub ttft_budget_infeasible: bool,
    /// Scheduling iterations that retained static limits for that reason.
    pub ttft_fallback_steps: u64,
    /// ITL target used for the dynamic wave budget. None means static limits.
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
        self.decode_target_infeasible = decode_ms >= targets.itl_ms;
        // Prefill throttling cannot repair an already infeasible pure decode.
        // Return to static limits; TPOT is not a per-wave constraint.
        if self.decode_target_infeasible {
            return None;
        }
        // Do not shrink work that already fits ITL, including its boundary.
        // Execution estimates do not certify client-visible P99 compliance.
        if decode_ms + cost.prefill_ms_per_token * static_limit as f64 <= targets.itl_ms {
            return None;
        }
        let budget = (((targets.itl_ms - decode_ms) / cost.prefill_ms_per_token).floor() as usize)
            .max(1)
            .min(static_limit);
        // Optimistically give each active prefill the entire budget on every
        // wave. If even that cannot finish before TTFT, shrinking its progress
        // cannot preserve its TTFT. This is a necessary condition,
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
        self.budget_target_ms = Some(targets.itl_ms);
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
        if prefill_tokens == 0 && elapsed_ms > targets.itl_ms {
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
        // Prefill observations are sparse between long decode runs. Retain
        // their information longer instead of halving its weight every 3 waves.
        const DECAY: f64 = 0.99;
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
        if !decode_ms_per_sequence.is_finite() {
            return None;
        }
        // For the constraint b >= 0, a negative unconstrained optimum lies
        // beyond the boundary b = 0. Refit a and c on that boundary instead
        // of clamping b while retaining the unconstrained coefficients.
        let (decode_ms_per_sequence, prefill_ms_per_token) = if decode_ms_per_sequence < 0.0 {
            (0.0, pt / pp)
        } else {
            (decode_ms_per_sequence, (pt * dd - dt * dp) / determinant)
        };
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

#[test]
fn slo_affine_negative_decode_coefficient_refits_the_boundary() {
    let cost = OnlineAffineCost {
        weight: 74.76393369106533,
        mean: [1.303653103805007, 4.605877310372727, 12.562799866932473],
        covariance: [
            29.75091583102048,
            -129.93724156068672,
            36278.12667787943,
            -72.52633283997464,
            8584.304492705858,
        ],
    }
    .fit()
    .expect("full-rank live observations have a nonnegative boundary fit");
    assert_eq!(cost.decode_ms_per_sequence, 0.0);
    assert!((cost.fixed_ms - 11.472935066598309).abs() < 1e-10);
    assert!((cost.prefill_ms_per_token - 0.23662480063889665).abs() < 1e-12);
    let predict = |decode, prefill| {
        cost.fixed_ms + cost.decode_ms_per_sequence * decode + cost.prefill_ms_per_token * prefill
    };
    let predictions = [predict(1.0, 0.0), predict(1.0, 128.0), predict(8.0, 128.0)];
    assert!(predictions.iter().all(|value| value.is_finite()));
    assert!(predictions.windows(2).all(|pair| pair[0] <= pair[1]));
}
