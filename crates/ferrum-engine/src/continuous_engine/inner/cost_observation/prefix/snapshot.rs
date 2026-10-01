use super::*;
use ferrum_scheduler::implementations::continuous::slo_planner::{
    PlanningCost, PlanningCostClockAnchor, PlanningPrefixCostModel, PrefixMaintenanceStage,
    PrefixRendezvousOffer,
};

pub(in crate::continuous_engine) struct PrefixCostSnapshot {
    model: Arc<model::CostModelSnapshot>,
    epoch: Arc<Epoch>,
    revision: u64,
    _memory: Arc<ObservationBytePermit>,
}
impl PrefixCostSnapshot {
    pub(super) fn new(
        model: Arc<model::CostModelSnapshot>,
        epoch: Arc<Epoch>,
        revision: u64,
        memory: Arc<ObservationBytePermit>,
    ) -> Self {
        Self {
            model,
            epoch,
            revision,
            _memory: memory,
        }
    }
    pub(in crate::continuous_engine) fn current(&self) -> bool {
        self.epoch.current(self.revision)
    }
    pub(in crate::continuous_engine) fn model_version(&self) -> u64 {
        self.revision
    }
    #[cfg(test)]
    pub(in crate::continuous_engine) fn diagnostic_prediction(
        &self,
        fingerprint: &model::ExecutionFingerprint,
        shape: &model::WaveExecutionShape,
        now_ns: u64,
    ) -> model::CostPrediction {
        self.model.predict(
            fingerprint,
            shape,
            model::CostBoundary::PreparationToCommit,
            now_ns,
        )
    }
    /// Missing empirical evidence may admit cold sampling. Protocol and clock
    /// errors never authorize untimed native work.
    pub(in crate::continuous_engine) fn cost_missing(
        &self,
        shape: &model::WaveExecutionShape,
        now_ns: u64,
    ) -> bool {
        if !self.current()
            || !matches!(
                shape.kind,
                model::WaveKind::Maintenance | model::WaveKind::Restore
            )
            || !shape.decode_kv_tokens.is_empty()
            || !shape.prefill_chunks.is_empty()
            || shape.numeric_features.is_some()
            || shape.host_content_features.is_some()
            || shape.row_multiset_features.is_some()
        {
            return false;
        }
        let missing = matches!(
            self.model.predict(
                self.model.fingerprint(),
                shape,
                model::CostBoundary::PreparationToCommit,
                now_ns
            ),
            model::CostPrediction::Unknown(
                model::CostUnknownReason::UnobservedBucket
                    | model::CostUnknownReason::InsufficientSamples
                    | model::CostUnknownReason::OutsideObservedCoverage
                    | model::CostUnknownReason::StaleSamples
            )
        );
        missing && self.current()
    }
    pub(in crate::continuous_engine) fn anchored(
        &self,
        anchor: PlanningCostClockAnchor,
        offer: &PrefixRendezvousOffer,
    ) -> AnchoredPrefixCostModel<'_> {
        AnchoredPrefixCostModel {
            snapshot: self,
            anchor,
            offer: offer.clone(),
        }
    }
}
pub(in crate::continuous_engine) struct AnchoredPrefixCostModel<'a> {
    snapshot: &'a PrefixCostSnapshot,
    anchor: PlanningCostClockAnchor,
    offer: PrefixRendezvousOffer,
}

pub(in crate::continuous_engine) struct AnchoredCapturePrefixCostModel<'a> {
    snapshot: &'a PrefixCostSnapshot,
    anchor: PlanningCostClockAnchor,
    offer: ferrum_scheduler::implementations::continuous::slo_planner::PrefixCacheCaptureOffer,
}
impl PrefixCostSnapshot {
    pub(in crate::continuous_engine) fn anchored_capture(
        &self,
        anchor: PlanningCostClockAnchor,
        offer: &ferrum_scheduler::implementations::continuous::slo_planner::PrefixCacheCaptureOffer,
    ) -> AnchoredCapturePrefixCostModel<'_> {
        AnchoredCapturePrefixCostModel {
            snapshot: self,
            anchor,
            offer: offer.clone(),
        }
    }
}
impl PlanningPrefixCostModel for AnchoredCapturePrefixCostModel<'_> {
    fn model_version(&self) -> u64 {
        self.snapshot.model_version()
    }
    fn predict(
        &self,
        _: &model::ExecutionFingerprint,
        _: &PrefixRendezvousOffer,
        _: PrefixMaintenanceStage,
        _: &model::WaveExecutionShape,
        _: u64,
    ) -> Option<PlanningCost> {
        None
    }
    fn predict_cache_capture(
        &self,
        fingerprint: &model::ExecutionFingerprint,
        offer: &ferrum_scheduler::implementations::continuous::slo_planner::PrefixCacheCaptureOffer,
        shape: &model::WaveExecutionShape,
        now_ns: u64,
    ) -> Option<PlanningCost> {
        if !self.snapshot.current()
            || offer != &self.offer
            || now_ns > offer.expires_at_ns
            || shape.kind != model::WaveKind::Maintenance
            || !shape.decode_kv_tokens.is_empty()
            || !shape.prefill_chunks.is_empty()
            || shape.numeric_features.is_some()
            || shape.host_content_features.is_some()
            || shape.row_multiset_features.is_some()
        {
            return None;
        }
        let now = self.anchor.cost_time_ns(now_ns).ok()?;
        let model::CostPrediction::Known(value) = self.snapshot.model.predict(
            fingerprint,
            shape,
            model::CostBoundary::PreparationToCommit,
            now,
        ) else {
            return None;
        };
        if !self.snapshot.current() {
            return None;
        }
        Some(PlanningCost {
            typical_ns: value.typical_ns,
            planning_ns: value.planning_ns,
            model_version: self.snapshot.model_version(),
            valid_for_ns: value
                .valid_for_ns
                .min(offer.expires_at_ns.checked_sub(now_ns)?),
        })
    }
}
impl PlanningPrefixCostModel for AnchoredPrefixCostModel<'_> {
    fn model_version(&self) -> u64 {
        self.snapshot.model_version()
    }
    fn predict(
        &self,
        fingerprint: &model::ExecutionFingerprint,
        offer: &PrefixRendezvousOffer,
        stage: PrefixMaintenanceStage,
        shape: &model::WaveExecutionShape,
        now_ns: u64,
    ) -> Option<PlanningCost> {
        let current = self.snapshot.current();
        if !current
            || offer != &self.offer
            || now_ns > offer.expires_at_ns
            || !shape.decode_kv_tokens.is_empty()
            || !shape.prefill_chunks.is_empty()
            || shape.numeric_features.is_some()
            || shape.host_content_features.is_some()
            || shape.row_multiset_features.is_some()
            || shape.kind
                != match stage {
                    PrefixMaintenanceStage::Capture => model::WaveKind::Maintenance,
                    PrefixMaintenanceStage::Restore => model::WaveKind::Restore,
                }
        {
            #[cfg(test)]
            prefix_cost_diagnostic(|| {
                eprintln!("prefix maintenance lookup rejected before model: stage={stage:?} current={current} offer_matches={} now={now_ns} expires={} epoch={} shape={shape:?}",
                offer == &self.offer, offer.expires_at_ns, self.snapshot.model_version())
            });
            return None;
        }
        let now = match self.anchor.cost_time_ns(now_ns) {
            Ok(now) => now,
            Err(_reason) => {
                #[cfg(test)]
                prefix_cost_diagnostic(|| {
                    eprintln!("prefix maintenance clock rejected: stage={stage:?} reason={_reason:?} planning_now={now_ns} epoch={}", self.snapshot.model_version())
                });
                return None;
            }
        };
        let prediction = self.snapshot.model.predict(
            fingerprint,
            shape,
            model::CostBoundary::PreparationToCommit,
            now,
        );
        let model::CostPrediction::Known(value) = prediction else {
            #[cfg(test)]
            prefix_cost_diagnostic(|| {
                eprintln!("prefix maintenance original prediction: stage={stage:?} prediction={prediction:?} cost_now={now} planning_now={now_ns} expires={} epoch={} shape={shape:?}",
                offer.expires_at_ns, self.snapshot.model_version())
            });
            return None;
        };
        // Publication/revocation may race this immutable lookup.
        if !self.snapshot.current() {
            #[cfg(test)]
            prefix_cost_diagnostic(|| {
                eprintln!("prefix maintenance epoch changed after original lookup: stage={stage:?} epoch={} shape={shape:?}", self.snapshot.model_version())
            });
            return None;
        }
        Some(PlanningCost {
            typical_ns: value.typical_ns,
            planning_ns: value.planning_ns,
            model_version: self.snapshot.model_version(),
            valid_for_ns: value
                .valid_for_ns
                .min(offer.expires_at_ns.checked_sub(now_ns)?),
        })
    }
}

pub(in crate::continuous_engine) struct AnchoredReadyPrefixCostModel<'a> {
    snapshot: &'a PrefixCostSnapshot,
    anchor: PlanningCostClockAnchor,
    offer: ferrum_scheduler::implementations::continuous::slo_planner::ReadyPrefixRestoreOffer,
}
impl PrefixCostSnapshot {
    pub(in crate::continuous_engine) fn anchored_ready(
        &self,
        anchor: PlanningCostClockAnchor,
        offer: &ferrum_scheduler::implementations::continuous::slo_planner::ReadyPrefixRestoreOffer,
    ) -> AnchoredReadyPrefixCostModel<'_> {
        AnchoredReadyPrefixCostModel {
            snapshot: self,
            anchor,
            offer: offer.clone(),
        }
    }
}
impl PlanningPrefixCostModel for AnchoredReadyPrefixCostModel<'_> {
    fn model_version(&self) -> u64 {
        self.snapshot.model_version()
    }
    fn predict(
        &self,
        _: &model::ExecutionFingerprint,
        _: &PrefixRendezvousOffer,
        _: PrefixMaintenanceStage,
        _: &model::WaveExecutionShape,
        _: u64,
    ) -> Option<PlanningCost> {
        None
    }
    fn predict_ready_restore(
        &self,
        fingerprint: &model::ExecutionFingerprint,
        offer: &ferrum_scheduler::implementations::continuous::slo_planner::ReadyPrefixRestoreOffer,
        shape: &model::WaveExecutionShape,
        now_ns: u64,
    ) -> Option<PlanningCost> {
        if !self.snapshot.current()
            || offer != &self.offer
            || now_ns > offer.expires_at_ns
            || shape.kind != model::WaveKind::Restore
            || !shape.decode_kv_tokens.is_empty()
            || !shape.prefill_chunks.is_empty()
            || shape.numeric_features.is_some()
            || shape.host_content_features.is_some()
            || shape.row_multiset_features.is_some()
        {
            return None;
        }
        let now = self.anchor.cost_time_ns(now_ns).ok()?;
        let model::CostPrediction::Known(value) = self.snapshot.model.predict(
            fingerprint,
            shape,
            model::CostBoundary::PreparationToCommit,
            now,
        ) else {
            return None;
        };
        if !self.snapshot.current() {
            return None;
        }
        Some(PlanningCost {
            typical_ns: value.typical_ns,
            planning_ns: value.planning_ns,
            model_version: self.snapshot.model_version(),
            valid_for_ns: value
                .valid_for_ns
                .min(offer.expires_at_ns.checked_sub(now_ns)?),
        })
    }
}

// Keep failure-only CPU diagnostics bounded without any production state or work.
#[cfg(test)]
fn prefix_cost_diagnostic(emit: impl FnOnce()) {
    std::thread_local! {
        static EMITTED: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    }
    EMITTED.with(|emitted| {
        let count = emitted.get();
        emitted.set(count.saturating_add(1));
        if count < 128 {
            emit();
        } else if count == 128 {
            eprintln!("additional prefix maintenance rejection diagnostics suppressed for this test thread");
        }
    });
}
