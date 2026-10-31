//! Online execution efficiency, separate from the authoritative capacity search.
//! Samples contain no requests, images, compiled plans, or device owners.
use super::*;
use crate::image_generation::{ImageResidencyPolicy, ImageStage};

#[derive(Debug, Clone)]
pub struct ImageBatchStageFeedback {
    pub stage: ImageStage,
    /// Actual installation of new physical weights, never session attachment.
    pub initialization_us: u64,
    /// Whether the completed batch leaves this required installation resident.
    pub weights_retained: bool,
}

#[derive(Debug, Clone, Default)]
pub struct ImageBatchExecutionFeedback {
    pub stages: Vec<ImageBatchStageFeedback>,
}

pub(super) struct ExecutionContext {
    pub(super) context: ImageCapacitySearchContext,
    pub(super) adapter: Option<Arc<ImageAdapterBatchIdentity>>,
    /// Capacity-proof upper bound independent of a short queue or tail.
    pub(super) capacity_upper: usize,
    pub(super) batch_size: usize,
    pub(super) exploration: bool,
}

#[derive(Default)]
pub(super) struct BatchEfficiency(Option<Domain>);

struct Domain {
    shape: [u8; 32],
    adapter: Option<Arc<ImageAdapterBatchIdentity>>,
    capacity_upper: usize,
    batches: Vec<BatchSample>,
    stages: Vec<StageSample>,
    explored_from: Option<usize>,
    savings_us: u64,
    /// Real execution savings against the first warm observed batch. Unknown
    /// spilling probes spend a whole execution; subsequent reloads are charged
    /// by their actual execution too. Bootstrap debt must be repaid first.
    exploration_credit_us: i128,
}

struct StageSample {
    stage: ImageStage,
    // The first installation may include disk reads/quantization. Only a later
    // physical installation is an observed recurring reload cost.
    reload_us: Option<u64>,
    /// Conservative probe reserve only, never a recurring cost estimate.
    installation_ceiling_us: u64,
}

struct BatchSample {
    count: usize,
    base_us: u64,
    last_execution_us: u64,
    missing: Vec<ImageStage>,
    eviction_observations: u64,
}

impl BatchSample {
    fn cost(&self, stages: &[StageSample]) -> Option<u64> {
        self.missing.iter().try_fold(self.base_us, |cost, stage| {
            cost.checked_add(stages.iter().find(|s| s.stage == *stage)?.reload_us?)
        })
    }
}

impl Domain {
    /// Price completion of the owned FIFO cohort, including its short tail.
    /// Per-row ranking alone can split a fitting batch into a cheaper-looking
    /// prefix and an expensive singleton. Unknown tail shapes conservatively
    /// use a whole observed execution as their estimate, never as admission.
    fn cohort_first(&self, queued: usize, maximum: usize) -> Option<&BatchSample> {
        let costs: Vec<_> = self
            .batches
            .iter()
            .filter(|b| b.count <= queued.min(maximum))
            .filter_map(|b| b.cost(&self.stages).map(|cost| (b, u128::from(cost))))
            .collect();
        if costs.is_empty() {
            return None;
        }
        // (finish time, summed image completion times, first batch). Finish
        // time preserves throughput; ties minimize waiting across the cohort.
        let mut plans = vec![(0_u128, 0_u128, 0_usize); queued + 1];
        for remaining in 1..=queued {
            plans[remaining] = costs
                .iter()
                .map(|(batch, cost)| {
                    let tail = plans[remaining.saturating_sub(batch.count)];
                    (
                        cost.saturating_add(tail.0),
                        cost.saturating_mul(remaining as u128)
                            .saturating_add(tail.1),
                        batch.count,
                    )
                })
                .min()
                .expect("observed costs retained");
        }
        self.batches.iter().find(|b| b.count == plans[queued].2)
    }

    fn matches(
        &self,
        context: ImageCapacitySearchContext,
        adapter: &Option<Arc<ImageAdapterBatchIdentity>>,
    ) -> bool {
        self.shape == context.shape && self.adapter == *adapter
    }
}

impl BatchEfficiency {
    pub(super) fn allow_spilling_candidate(
        &self,
        context: Option<ImageCapacitySearchContext>,
        adapter: &Option<Arc<ImageAdapterBatchIdentity>>,
        candidate: usize,
        resident_upper: usize,
    ) -> bool {
        if resident_upper == 0 {
            return true;
        }
        let Some(context) = context else {
            return true;
        };
        let Some(domain) = self.0.as_ref().filter(|d| d.matches(context, adapter)) else {
            return false;
        };
        let best = domain
            .batches
            .iter()
            .filter(|b| b.count <= candidate)
            .filter_map(|b| b.cost(&domain.stages).map(|cost| (b, cost)))
            .min_by(|(a, ac), (b, bc)| {
                (u128::from(*ac) * b.count as u128)
                    .cmp(&(u128::from(*bc) * a.count as u128))
                    .then_with(|| a.count.cmp(&b.count))
            });
        let Some((best, _)) = best else {
            return false;
        };
        if let Some(sample) = domain.batches.iter().find(|b| b.count == candidate) {
            if sample.cost(&domain.stages).is_none() {
                return false;
            }
            if best.count == candidate {
                return true;
            }
        }
        // When only a singleton fits without swapping, permit one adjacent
        // real-work calibration. Its full cost becomes debt below. There is
        // no machine-specific batch cap and no repeated free calibration.
        if candidate == 2 && resident_upper == 1 && domain.batches.iter().all(|b| b.count == 1) {
            return true;
        }
        let executions = candidate.div_ceil(best.count) as u128;
        let reserve = u128::from(best.last_execution_us)
            .saturating_mul(executions)
            .saturating_add(
                domain
                    .stages
                    .iter()
                    .map(|s| u128::from(s.installation_ceiling_us))
                    .sum::<u128>(),
            );
        domain.exploration_credit_us >= 0 && domain.exploration_credit_us as u128 >= reserve
    }

    /// Only shrink the already available FIFO prefix. Unknown costs and sizes
    /// use the original capacity search; this never waits for more arrivals.
    pub(super) fn select(
        &mut self,
        context: Option<ImageCapacitySearchContext>,
        adapter: &Option<Arc<ImageAdapterBatchIdentity>>,
        queued_upper: usize,
        capacity_upper: usize,
    ) -> usize {
        let Some(context) = context else {
            return queued_upper;
        };
        let Some(domain) = self.0.as_mut().filter(|d| d.matches(context, adapter)) else {
            return queued_upper;
        };
        // Only crossing an actual capacity proof triggers early reassessment.
        // Physical free-byte jitter and short queues do not reset learning.
        if capacity_upper > domain.capacity_upper {
            return queued_upper;
        }
        let Some(upper) = domain.batches.iter().find(|b| b.count == queued_upper) else {
            return queued_upper;
        };
        let Some(upper_cost) = upper.cost(&domain.stages) else {
            return queued_upper;
        };
        let largest = domain.batches.iter().map(|b| b.count).max().unwrap_or(1);
        if queued_upper == largest && domain.savings_us >= upper.last_execution_us {
            tracing::debug!(target: "ferrum_engine::image_generation::timing",
                chosen_batch = queued_upper, capacity_upper, saved_us = domain.savings_us,
                reason = "earned_larger_batch_reassessment",
                "image batch efficiency selection");
            return queued_upper;
        }
        // One adjacent real-work probe per observed larger size, only after
        // repeated eviction and a known hot reload. Never walk a ladder of
        // successively smaller probes, or compile a synthetic probe batch.
        if queued_upper == largest
            && queued_upper > 1
            && upper.eviction_observations >= 2
            && !upper.missing.is_empty()
            && domain.explored_from != Some(largest)
            && !domain.batches.iter().any(|b| b.count == largest - 1)
        {
            tracing::debug!(target: "ferrum_engine::image_generation::timing",
                chosen_batch = largest - 1, observed_max_batch = largest, capacity_upper,
                upper_base_us = upper.base_us, upper_reload_us = upper_cost - upper.base_us,
                reason = "observe_adjacent_batch_after_recurring_reload",
                "image batch efficiency selection");
            return largest - 1;
        }
        let Some(best) = domain.cohort_first(queued_upper, capacity_upper) else {
            return queued_upper;
        };
        let cost = best.cost(&domain.stages).expect("priced cohort batch");
        if best.count < queued_upper {
            tracing::debug!(target: "ferrum_engine::image_generation::timing",
                chosen_batch = best.count, observed_max_batch = largest, capacity_upper,
                predicted_execution_us = cost, predicted_per_image_us = cost / best.count as u64,
                predicted_reload_us = cost - best.base_us,
                upper_predicted_per_image_us = upper_cost / queued_upper as u64,
                reason = "observed_sustained_cost",
                "image batch efficiency selection");
        }
        best.count
    }

    /// Prime an idle execution lane with an observed efficient shape. The
    /// preparation lane still performs the normal search of the remaining
    /// backlog, so this neither caps capacity nor suppresses larger probes.
    pub(super) fn warm_start(
        &self,
        context: Option<ImageCapacitySearchContext>,
        adapter: &Option<Arc<ImageAdapterBatchIdentity>>,
        queued: usize,
        capacity_upper: usize,
    ) -> Option<usize> {
        let context = context?;
        let domain = self.0.as_ref().filter(|d| d.matches(context, adapter))?;
        domain.cohort_first(queued, capacity_upper).map(|b| b.count)
    }

    /// Recheck a prepared batch after the preceding execution supplies feedback.
    /// Price a just-observed eviction using known work once, then apply the
    /// normal cost model before committing a plan prepared from older samples.
    pub(super) fn reconsider_ready(
        &mut self,
        context: ImageCapacitySearchContext,
        adapter: &Option<Arc<ImageAdapterBatchIdentity>>,
        ready: usize,
        capacity_upper: usize,
    ) -> Option<usize> {
        let domain = self.0.as_ref().filter(|d| d.matches(context, adapter))?;
        let sample = domain.batches.iter().find(|b| b.count == ready)?;
        if sample.cost(&domain.stages).is_some() {
            let chosen = self.select(Some(context), adapter, ready, capacity_upper);
            return (chosen < ready).then_some(chosen);
        }
        self.warm_start(Some(context), adapter, ready, ready)
            .filter(|chosen| *chosen < ready)
    }

    pub(super) fn observe(
        &mut self,
        execution: ExecutionContext,
        elapsed: Duration,
        feedback: ImageBatchExecutionFeedback,
        maximum_batch: usize,
    ) {
        if execution.batch_size == 0
            || execution.batch_size > maximum_batch
            || feedback.stages.is_empty()
        {
            return;
        }
        if self
            .0
            .as_ref()
            .is_none_or(|d| !d.matches(execution.context, &execution.adapter))
        {
            self.0 = Some(Domain {
                shape: execution.context.shape,
                adapter: execution.adapter,
                capacity_upper: execution.capacity_upper,
                batches: Vec::new(),
                stages: Vec::new(),
                explored_from: None,
                savings_us: 0,
                exploration_credit_us: 0,
            });
        }
        let domain = self.0.as_mut().unwrap();
        domain.capacity_upper = execution.capacity_upper;
        let actual_us = u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX);
        let mut initialization_us = 0_u64;
        let mut cold = false;
        let mut missing = Vec::new();
        for stage in feedback.stages {
            initialization_us = initialization_us.saturating_add(stage.initialization_us);
            match domain.stages.iter_mut().find(|s| s.stage == stage.stage) {
                Some(previous) if stage.initialization_us > 0 => {
                    previous.reload_us = Some(stage.initialization_us);
                    previous.installation_ceiling_us = previous
                        .installation_ceiling_us
                        .max(stage.initialization_us);
                }
                Some(_) => (),
                None => {
                    cold |= stage.initialization_us > 0;
                    domain.stages.push(StageSample {
                        stage: stage.stage,
                        reload_us: None,
                        installation_ceiling_us: stage.initialization_us,
                    });
                }
            }
            if !stage.weights_retained && !missing.contains(&stage.stage) {
                missing.push(stage.stage);
            }
        }
        // Cold initialization can also warm native plans/driver code beyond
        // the explicit install timer. Do not use that batch to rank sizes.
        if cold || actual_us <= initialization_us {
            return;
        }
        let base_us = actual_us - initialization_us;
        if let Some(reference) = domain
            .batches
            .first()
            .and_then(|b| b.cost(&domain.stages).map(|cost| (b.count, cost)))
        {
            let baseline =
                i128::from(reference.1) * execution.batch_size as i128 / reference.0 as i128;
            domain.exploration_credit_us = domain
                .exploration_credit_us
                .saturating_add(baseline.saturating_sub(i128::from(actual_us)));
        }
        if execution.exploration {
            domain.exploration_credit_us = domain
                .exploration_credit_us
                .saturating_sub(i128::from(actual_us));
        }
        let previous_largest = domain.batches.iter().map(|b| b.count).max().unwrap_or(0);
        if execution.batch_size + 1 == previous_largest {
            domain.explored_from = Some(previous_largest);
        }
        let index = match domain
            .batches
            .iter()
            .position(|b| b.count == execution.batch_size)
        {
            Some(index) => index,
            None => {
                domain.batches.push(BatchSample {
                    count: execution.batch_size,
                    base_us,
                    last_execution_us: actual_us,
                    missing: Vec::new(),
                    eviction_observations: 0,
                });
                domain.batches.len() - 1
            }
        };
        let sample = &mut domain.batches[index];
        // A bounded moving estimate adapts to changed execution conditions.
        // No old cold/sample-count minimum can permanently pin a chosen size.
        sample.base_us = ((u128::from(sample.base_us) + u128::from(base_us)) / 2) as u64;
        sample.last_execution_us = actual_us;
        sample.eviction_observations = if missing.is_empty() {
            0
        } else {
            sample.eviction_observations.saturating_add(1)
        };
        sample.missing = missing;
        let cost = sample.cost(&domain.stages);
        let largest = domain.batches.iter().max_by_key(|b| b.count).unwrap();
        if execution.batch_size >= largest.count {
            domain.savings_us = 0;
        } else if let (Some(cost), Some(upper_cost)) = (cost, largest.cost(&domain.stages)) {
            let counterfactual =
                u128::from(upper_cost) * execution.batch_size as u128 / largest.count as u128;
            domain.savings_us = domain.savings_us.saturating_add(
                u64::try_from(counterfactual.saturating_sub(u128::from(cost))).unwrap_or(u64::MAX),
            );
        }
        tracing::debug!(target: "ferrum_engine::image_generation::timing",
            batch_size = execution.batch_size, actual_execution_us = actual_us,
            actual_weight_installation_us = initialization_us, predicted_sustained_us = ?cost,
            missing_weights = ?domain.batches[index].missing,
            saved_us = domain.savings_us, "image batch efficiency observation");
    }
}

impl<R: DeviceRuntime> QwenImage21Pipeline<R> {
    pub(super) fn service_batch_execution_feedback(
        &self,
        outputs: &[ImageGenerationOutput],
    ) -> Option<ImageBatchExecutionFeedback> {
        if self.config.residency != ImageResidencyPolicy::Adaptive {
            return None;
        }
        let receipt = &outputs.first()?.receipt;
        let mut feedback = ImageBatchExecutionFeedback::default();
        for stage in &receipt.stages {
            let execution = &stage.execution;
            if execution.plan.static_bytes == 0 || execution.completed_executions == 0 {
                continue;
            }
            let initialization_us = if execution.weights_reused {
                0
            } else {
                execution.initialization_wall_duration_us
            };
            if let Some(previous) = feedback
                .stages
                .iter_mut()
                .find(|s| s.stage == execution.plan.stage)
            {
                previous.initialization_us =
                    previous.initialization_us.saturating_add(initialization_us);
            } else {
                feedback.stages.push(ImageBatchStageFeedback {
                    stage: execution.plan.stage,
                    initialization_us,
                    weights_retained: self.resident_stages.has_weights(execution.plan.stage),
                });
            }
        }
        Some(feedback)
    }
}

#[cfg(test)]
mod scheduler_tests;
#[cfg(test)]
mod tests;
