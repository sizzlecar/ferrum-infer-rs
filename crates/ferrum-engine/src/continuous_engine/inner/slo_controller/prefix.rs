//! Optional admitted prefix sharing uses the same complete queue transaction.
//! Every later turn replays the remaining trajectory; a comparison is no permit.
use super::*;
use ferrum_interfaces::model_executor::{
    PrefixCaptureBoundary, PrefixCaptureLease, PrefixCapturePurpose, PrefixCaptureRequest,
};
use ferrum_scheduler::implementations::continuous::{PrefixRendezvousHold, PrefixRequestKey};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::continuous_engine::profile::prefix as prefix_observation;

mod execute;
mod modeled_hold;
pub(super) mod producer;
mod ready;
pub(super) use ready::ReadyPrefixCohort;
mod sampling;
pub(super) use sampling::PrefixSample;
pub(in crate::continuous_engine) use sampling::PreparedPrefixSampling;

static NEXT_COHORT: AtomicU64 = AtomicU64::new(1);

pub(super) struct SloPrefixCohort {
    hold: PrefixRendezvousHold,
    target: PrefixRequestKey,
    source_incarnation: u64,
    target_incarnation: u64,
    identity: [u8; 32],
    expires_at: Instant,
    source_tokens: Vec<TokenId>,
    maximum_sequence_tokens: usize,
    capture: Arc<dyn PrefixCaptureLease>,
    capture_span_start: Option<u32>,
    restored: bool,
}

pub(in crate::continuous_engine) enum PreparedPrefixMaintenance {
    Rendezvous(PreparedRendezvousMaintenance),
    Ready(ready::PreparedReadyMaintenance),
    CacheCapture(producer::PreparedCacheCapture),
}

pub(in crate::continuous_engine) struct PreparedRendezvousMaintenance {
    pub(super) cohort: SloPrefixCohort,
    pub(super) evidence: PrefixMaintenanceEvidence,
    pub(super) maintenance: Arc<super::super::cost_observation::PrefixCostSnapshot>,
    pub(super) valid_until: Instant,
    pub(super) model_version: u64,
    pub(super) predicted_wall_ns: u64,
    pub(super) proof: ControllerSafetyProof,
}

pub(super) enum PrefixPlan {
    None,
    ReadyMaintenance(ready::PreparedReadyPlan),
    CacheCapture(producer::PreparedCacheCapturePlan),
    CachePreparation {
        selected: SelectedWave,
        maintenance: Arc<super::super::cost_observation::PrefixCostSnapshot>,
    },
    CostUnavailable,
    IncompleteCostLookup,
    Changed,
    Inconclusive,
    Wave {
        selected: SelectedWave,
        checkpoint: Arc<dyn PrefixCaptureLease>,
        maintenance: Arc<super::super::cost_observation::PrefixCostSnapshot>,
    },
    Maintenance {
        cohort: SloPrefixCohort,
        evidence: PrefixMaintenanceEvidence,
        maintenance: Arc<super::super::cost_observation::PrefixCostSnapshot>,
        valid_until: Instant,
        predicted_wall_ns: u64,
    },
}

impl EngineInner {
    fn record_prefix_decision(
        &self,
        route: prefix_observation::Route,
        owner: &RequestWorkKey,
        peer: Option<&RequestWorkKey>,
        boundary_tokens: u32,
        captured: &ControllerSnapshot,
        maintenance_epoch: u64,
        decision: prefix_observation::Decision,
    ) {
        let Some(recorder) = &self.prefix_resource_recorder else {
            return;
        };
        recorder.record(prefix_observation::Event::Decision {
            route,
            owner: owner.into(),
            peer: peer.map(Into::into),
            boundary_tokens,
            snapshot_generation: captured.snapshot.generation,
            inference_epoch: captured.model.model_version(),
            maintenance_epoch,
            decision,
        });
    }

    pub(in crate::continuous_engine::inner) fn release_slo_prefix_for_capacity_pressure(&self) {
        let retired = {
            let mut state = self.slo_controller.lock();
            (
                state.prefix.take(),
                state.prefix_sample.take(),
                state.ready_prefix.take(),
                state.prefix_cache_preparing.take(),
            )
        };
        drop(retired);
    }

    pub(in crate::continuous_engine::inner) fn slo_prefix_deadline(&self) -> Option<Instant> {
        let state = self.slo_controller.lock();
        state
            .prefix
            .as_ref()
            .map(|cohort| cohort.expires_at)
            .into_iter()
            .chain(state.ready_prefix.as_ref().map(|cohort| cohort.expires_at))
            .chain(
                state
                    .prefix_cache_preparing
                    .as_ref()
                    .map(|cohort| cohort.expires_at),
            )
            .min()
    }

    pub(super) fn drop_slo_prefix_trajectory(&self) -> bool {
        let retired = {
            let mut state = self.slo_controller.lock();
            (
                state.prefix.take(),
                state.ready_prefix.take(),
                state.prefix_cache_preparing.take(),
            )
        };
        retired.0.is_some() || retired.1.is_some() || retired.2.is_some()
    }

    pub(super) fn plan_slo_prefix(&self, captured: &ControllerSnapshot) -> Result<PrefixPlan> {
        if self.config.scheduler.slo.mode != ferrum_types::SloMode::Enforce
            || !self.config.runtime.prefix_state_cache_enabled
            || !self.model_executor.supports_guarded_prefix_maintenance()
        {
            return Ok(PrefixPlan::None);
        }
        let maintenance = self
            .cost_runtime
            .as_ref()
            .and_then(|r| r.try_prefix_cost_snapshot());
        let maintenance = match maintenance {
            Some(Some(maintenance)) => maintenance,
            state => {
                let (old, ready) = {
                    let mut state = self.slo_controller.lock();
                    state.prefix_cache_preparing = None;
                    (state.prefix.take(), state.ready_prefix.take())
                };
                return Ok(if old.is_some() || ready.is_some() {
                    PrefixPlan::Changed
                } else if state.is_some() {
                    PrefixPlan::CostUnavailable
                } else {
                    PrefixPlan::Inconclusive
                });
            }
        };
        let existing = self.slo_controller.lock().prefix.take();
        if let Some(cohort) = existing {
            return Ok(self.continue_slo_prefix(captured, cohort, maintenance));
        }
        let ready = self.plan_ready_prefix(captured, Arc::clone(&maintenance))?;
        if !matches!(ready, PrefixPlan::None) {
            return Ok(ready);
        }
        if self
            .config
            .scheduler
            .prefix_rendezvous_max_wait_ms
            .is_none()
        {
            return self.plan_prefix_cache_capture(captured, maintenance);
        }
        self.compare_slo_prefix(captured, maintenance)
    }

    fn continue_slo_prefix(
        &self,
        captured: &ControllerSnapshot,
        mut cohort: SloPrefixCohort,
        maintenance: Arc<super::super::cost_observation::PrefixCostSnapshot>,
    ) -> PrefixPlan {
        if slo_clock_now() >= cohort.expires_at || !captured.poll_planning() {
            #[cfg(test)]
            eprintln!("prefix continuation rejected: stage=expiry_or_budget source={} target={} boundary={}", cohort.hold.source().request_id(), cohort.target.request_id(), cohort.hold.boundary());
            return PrefixPlan::Changed;
        }
        if (!cohort.restored && !cohort.hold.is_pending())
            || self
                .scheduler
                .prefix_request_progress(cohort.hold.source())
                .is_none()
            || self
                .scheduler
                .prefix_request_progress(&cohort.target)
                .is_none()
        {
            #[cfg(test)]
            eprintln!("prefix continuation rejected: stage=hold_or_scheduler_owner_missing source={} target={} boundary={}", cohort.hold.source().request_id(), cohort.target.request_id(), cohort.hold.boundary());
            return PrefixPlan::Changed;
        }
        let rows = &captured.snapshot.requests;
        let source = rows
            .iter()
            .find(|r| matches_key(&r.key, cohort.hold.source(), cohort.source_incarnation));
        let target = rows
            .iter()
            .find(|r| matches_key(&r.key, &cohort.target, cohort.target_incarnation));
        let (Some(source), Some(target)) = (source, target) else {
            #[cfg(test)]
            eprintln!("prefix continuation rejected: stage=captured_owner_missing source={} target={} boundary={}", cohort.hold.source().request_id(), cohort.target.request_id(), cohort.hold.boundary());
            return PrefixPlan::Changed;
        };
        let Ok(expires_at_ns) = captured.origin.at_ns(cohort.expires_at) else {
            #[cfg(test)]
            eprintln!(
                "prefix continuation rejected: stage=cohort_clock source={} target={} boundary={}",
                cohort.hold.source().request_id(),
                cohort.target.request_id(),
                cohort.hold.boundary()
            );
            return PrefixPlan::Changed;
        };
        let Some(boundary) = u32::try_from(cohort.hold.boundary())
            .ok()
            .and_then(NonZeroU32::new)
        else {
            #[cfg(test)]
            eprintln!("prefix continuation rejected: stage=boundary_range source={} target={} boundary={}", cohort.hold.source().request_id(), cohort.target.request_id(), cohort.hold.boundary());
            return PrefixPlan::Changed;
        };
        let offer = PrefixRendezvousOffer {
            identity: cohort.identity,
            based_on_generation: captured.snapshot.generation,
            producer: source.key.clone(),
            target: target.key.clone(),
            boundary_tokens: boundary,
            expires_at_ns,
        };
        let phase = if cohort.restored {
            PrefixContinuationPhase::Restored
        } else if let Some(capture_span_start) = cohort.capture_span_start {
            PrefixContinuationPhase::CheckpointReady { capture_span_start }
        } else {
            let RequestPhaseView::Prefill(progress) = &source.phase else {
                #[cfg(test)]
                eprintln!("prefix continuation rejected: stage=source_not_prefill source={} target={} boundary={}", cohort.hold.source().request_id(), cohort.target.request_id(), cohort.hold.boundary());
                return PrefixPlan::Changed;
            };
            if progress.offset < boundary.get() {
                PrefixContinuationPhase::HeldAwaitingProducer
            } else if progress.offset == boundary.get() {
                let index = rows.iter().position(|r| r.key == source.key).unwrap();
                let Some(proof) =
                    captured.resources.participants()[index].completed_checkpoint_boundary()
                else {
                    #[cfg(test)]
                    eprintln!("prefix continuation rejected: stage=completed_boundary_missing source={} target={} boundary={}", cohort.hold.source().request_id(), cohort.target.request_id(), cohort.hold.boundary());
                    return PrefixPlan::Changed;
                };
                let Ok(capture_span_start) = u32::try_from(proof.span_start()) else {
                    #[cfg(test)]
                    eprintln!("prefix continuation rejected: stage=capture_span_range source={} target={} boundary={}", cohort.hold.source().request_id(), cohort.target.request_id(), cohort.hold.boundary());
                    return PrefixPlan::Changed;
                };
                PrefixContinuationPhase::AtCaptureBoundary { capture_span_start }
            } else {
                #[cfg(test)]
                eprintln!("prefix continuation rejected: stage=source_past_boundary source={} target={} boundary={}", cohort.hold.source().request_id(), cohort.target.request_id(), cohort.hold.boundary());
                return PrefixPlan::Changed;
            }
        };
        #[cfg(test)]
        eprintln!("prefix continuation input: phase={phase:?} source={} target={} source_context={} target_context={} source_ready={:?} target_ready={:?} boundary={} inference_epoch={} maintenance_epoch={}",
            source.key.request_id, target.key.request_id, source.context_tokens,
            target.context_tokens, source.readiness, target.readiness, boundary,
            captured.model.model_version(), maintenance.model_version());
        let planner = self.prefix_planner();
        let cost = AnchoredPlanningCostModel::new(captured.model.as_ref(), captured.anchor);
        let prefix_cost = maintenance.anchored(captured.anchor, &offer);
        let shape = shape::PrefixExecutorShape {
            source: shape::ExecutorShape {
                engine: self,
                captured,
            },
            offer: &offer,
            lease: Some(cohort.capture.as_ref()),
        };
        let Ok(window) = captured.planning_window() else {
            #[cfg(test)]
            eprintln!("prefix continuation rejected: stage=planning_window source={} target={} boundary={}", cohort.hold.source().request_id(), cohort.target.request_id(), cohort.hold.boundary());
            return PrefixPlan::Changed;
        };
        let decision = captured
            .origin
            .continue_prefix_rendezvous_scoped_with_execution_budget_window(
                &planner,
                &captured.snapshot,
                &offer,
                phase,
                &cost,
                &prefix_cost,
                &shape,
                Some(Arc::clone(&captured.protection)),
                window,
                slo_clock_now,
            );
        if let Ok(result) = &decision {
            let (disposition, reason) = captured.budget.record_prefix_continuation(result);
            self.record_controller(ControllerObservation {
                obligations: captured.snapshot.requests.len(),
                disposition,
                reason,
            });
        }
        self.record_prefix_decision(
            prefix_observation::Route::RendezvousContinuation,
            &offer.producer,
            Some(&offer.target),
            offer.boundary_tokens.get(),
            captured,
            maintenance.model_version(),
            match &decision {
                Ok(PrefixContinuationDecision::Ready { .. }) => prefix_observation::Decision::Known,
                Ok(PrefixContinuationDecision::Unknown { reason, .. }) => {
                    prefix_observation::Decision::Unknown(*reason)
                }
                Err(reason) => prefix_observation::Decision::ClockError(*reason),
            },
        );
        let Ok(PrefixContinuationDecision::Ready { continuation, .. }) = decision else {
            #[cfg(test)]
            eprintln!("prefix continuation rejected: stage=planner_rejected source={} target={} boundary={} phase={phase:?} decision={decision:?}", cohort.hold.source().request_id(), cohort.target.request_id(), cohort.hold.boundary());
            return PrefixPlan::Changed;
        };
        if !maintenance.current()
            || !captured.poll_planning()
            || !self.controller_frontiers_match(captured)
            || continuation.protection().as_ref() != captured.protection.as_ref()
        {
            #[cfg(test)]
            eprintln!("prefix continuation rejected: stage=post_replay_current_or_fence source={} target={} boundary={}", cohort.hold.source().request_id(), cohort.target.request_id(), cohort.hold.boundary());
            return PrefixPlan::Changed;
        }
        let Ok(valid_until) = captured.origin.instant_at_ns(continuation.valid_until_ns()) else {
            #[cfg(test)]
            eprintln!(
                "prefix continuation rejected: stage=witness_clock source={} target={} boundary={}",
                cohort.hold.source().request_id(),
                cohort.target.request_id(),
                cohort.hold.boundary()
            );
            return PrefixPlan::Changed;
        };
        match continuation.action().clone() {
            PrefixContinuationAction::Wave(wave) => {
                let checkpoint = Arc::clone(&cohort.capture);
                if !cohort.restored {
                    self.slo_controller.lock().prefix = Some(cohort);
                }
                PrefixPlan::Wave {
                    selected: wave,
                    checkpoint,
                    maintenance,
                }
            }
            PrefixContinuationAction::Maintenance(evidence) => PrefixPlan::Maintenance {
                cohort,
                evidence,
                maintenance,
                valid_until,
                predicted_wall_ns: continuation.first_action_cost_ns(),
            },
        }
    }

    fn compare_slo_prefix(
        &self,
        captured: &ControllerSnapshot,
        maintenance: Arc<super::super::cost_observation::PrefixCostSnapshot>,
    ) -> Result<PrefixPlan> {
        let mut poll = || captured.poll_planning();
        let Some(candidates) = self.scheduler.try_prefix_rendezvous_candidates(
            captured.snapshot.requests.len() + captured.waiting_fences.len(),
            &mut poll,
        ) else {
            return Ok(PrefixPlan::None);
        };
        let Some(sequences) = self.sequences.try_read() else {
            return Ok(PrefixPlan::None);
        };
        // Select one compatible pair within this transaction. Any comparison
        // consumes the original window; it never starts a second search.
        for producer in candidates.iter().filter(|c| !c.waiting) {
            if !poll() {
                break;
            }
            let Some(source) = sequences.get(producer.key.request_id()) else {
                continue;
            };
            if source.prefill_complete
                || !source.generated_tokens.is_empty()
                || source.preemption_count != 0
            {
                continue;
            }
            let source_tokens = source.prefill_context_tokens();
            for follower in candidates
                .iter()
                .filter(|c| !c.waiting && c.processed_tokens == 0)
            {
                if !poll() {
                    break;
                }
                if follower.key.request_id() == producer.key.request_id()
                    || follower.priority > producer.priority
                {
                    continue;
                }
                let Some(target) = sequences.get(follower.key.request_id()) else {
                    continue;
                };
                if target.prefill_complete
                    || !target.generated_tokens.is_empty()
                    || target.preemption_count != 0
                {
                    continue;
                }
                let tokens = target.prefill_context_tokens();
                let common = source_tokens
                    .iter()
                    .zip(&tokens)
                    .take_while(|(a, b)| a == b)
                    .count();
                let Some(plan) =
                    self.model_executor
                        .plan_prefix_capture_boundary(PrefixCaptureBoundary {
                            processed_tokens: producer.processed_tokens,
                            source_prompt_tokens: source_tokens.len(),
                            common_prefix_tokens: common,
                            follower_prompt_tokens: &[tokens.len()],
                        })
                else {
                    continue;
                };
                let Some(boundary) = u32::try_from(plan.boundary).ok().and_then(NonZeroU32::new)
                else {
                    continue;
                };
                let (Some(source_frontier), Some(target_frontier)) =
                    (source.cost_frontier, target.cost_frontier)
                else {
                    continue;
                };
                let source_incarnation = source_frontier.owner_incarnation.get();
                let target_incarnation = target_frontier.owner_incarnation.get();
                let source_row = captured
                    .snapshot
                    .requests
                    .iter()
                    .find(|r| matches_current_key(&r.key, &producer.key, source_incarnation));
                let target_row = captured
                    .snapshot
                    .requests
                    .iter()
                    .find(|r| matches_current_key(&r.key, &follower.key, target_incarnation));
                let (Some(source_row), Some(target_row)) = (source_row, target_row) else {
                    continue;
                };
                if source_row.readiness != RequestReadiness::Ready
                    || target_row.readiness != RequestReadiness::Ready
                {
                    continue;
                }
                let Some(configured_end) = slo_clock_now().checked_add(Duration::from_millis(
                    self.config
                        .scheduler
                        .prefix_rendezvous_max_wait_ms
                        .unwrap()
                        .get(),
                )) else {
                    continue;
                };
                let Ok(expires_at_ns) = captured.origin.at_ns(configured_end) else {
                    continue;
                };
                let Some(identity) = cohort_identity(&producer.key, &follower.key, plan.boundary)
                else {
                    continue;
                };
                let offer = PrefixRendezvousOffer {
                    identity,
                    based_on_generation: captured.snapshot.generation,
                    producer: source_row.key.clone(),
                    target: target_row.key.clone(),
                    boundary_tokens: boundary,
                    expires_at_ns,
                };
                let maximum_sequence_tokens = source.model_maximum_sequence_tokens();
                let source_key = producer.key.clone();
                let target_key = follower.key.clone();
                drop(sequences);
                let planner = self.prefix_planner();
                let cost = AnchoredPlanningCostModel::new(captured.model.as_ref(), captured.anchor);
                let prefix_cost = maintenance.anchored(captured.anchor, &offer);
                let context = shape::PrefixExecutorShape {
                    source: shape::ExecutorShape {
                        engine: self,
                        captured,
                    },
                    offer: &offer,
                    lease: None,
                };
                let Ok(window) = captured.planning_window() else {
                    return Ok(PrefixPlan::Inconclusive);
                };
                let compared = captured
                    .origin
                    .compare_prefix_rendezvous_scoped_with_execution_budget_window(
                        &planner,
                        &captured.snapshot,
                        &offer,
                        &cost,
                        &prefix_cost,
                        &context,
                        Some(Arc::clone(&captured.protection)),
                        window,
                        slo_clock_now,
                    );
                if let Ok(result) = &compared {
                    let (disposition, reason) = captured.budget.record_prefix_comparison(result);
                    self.record_controller(ControllerObservation {
                        obligations: captured.snapshot.requests.len(),
                        disposition,
                        reason,
                    });
                }
                self.record_prefix_decision(
                    prefix_observation::Route::RendezvousComparison,
                    &offer.producer,
                    Some(&offer.target),
                    offer.boundary_tokens.get(),
                    captured,
                    maintenance.model_version(),
                    match &compared {
                        Ok(PrefixRendezvousDecision::Compared { comparison, .. }) => {
                            if comparison.should_hold() {
                                prefix_observation::Decision::HoldRecommended
                            } else {
                                prefix_observation::Decision::PreferDirect
                            }
                        }
                        Ok(PrefixRendezvousDecision::Unknown { reason, .. }) => {
                            prefix_observation::Decision::Unknown(*reason)
                        }
                        Err(reason) => prefix_observation::Decision::ClockError(*reason),
                    },
                );
                let comparison = match compared {
                    Ok(PrefixRendezvousDecision::Compared { comparison, .. }) => comparison,
                    Ok(PrefixRendezvousDecision::Unknown {
                        reason: PlanningUnknownReason::CostUnavailable,
                        ..
                    }) if maintenance.current() && captured.poll_planning() => {
                        return Ok(PrefixPlan::CostUnavailable);
                    }
                    Ok(PrefixRendezvousDecision::Unknown {
                        reason: PlanningUnknownReason::SearchIncomplete,
                        search,
                    }) if search.cost_unknown_candidates > 0
                        && maintenance.current()
                        && captured.poll_planning() =>
                    {
                        // The counter is not a missing-cost proof. A cold
                        // selector must independently project the exact current
                        // Capture and prove its domain is missing; the native
                        // final guard checks actual cost again.
                        return Ok(PrefixPlan::IncompleteCostLookup);
                    }
                    _ => return Ok(PrefixPlan::Inconclusive),
                };
                if !comparison.should_hold()
                    || !maintenance.current()
                    || !captured.poll_planning()
                    || !self.controller_frontiers_match(captured)
                {
                    return Ok(PrefixPlan::Inconclusive);
                }
                let Ok(comparison_valid_until) =
                    captured.origin.instant_at_ns(comparison.valid_until_ns())
                else {
                    return Ok(PrefixPlan::Inconclusive);
                };
                // The comparison's deadline bounds adoption of its start-time
                // proof. The lease and all later independently replayed turns
                // retain the original configuration deadline; replacing it with
                // start slack would subtract the remaining trajectory twice.
                let Some(capture) =
                    self.model_executor
                        .retain_prefix_capture_interest(PrefixCaptureRequest {
                            purpose: PrefixCapturePurpose::SharedCache,
                            source_request_id: source_key.request_id(),
                            source_tokens: &source_tokens,
                            maximum_sequence_tokens,
                            boundary: plan.boundary,
                            expires_at: configured_end,
                        })?
                else {
                    return Ok(PrefixPlan::Inconclusive);
                };
                if !self.prefix_comparison_current(captured, &maintenance, comparison_valid_until) {
                    return Ok(PrefixPlan::Inconclusive);
                }
                let Some(hold) =
                    self.scheduler
                        .hold_admitted_prefix_follower(&source_key, &target_key, plan)
                else {
                    return Ok(PrefixPlan::Inconclusive);
                };
                if !self.prefix_comparison_current(captured, &maintenance, comparison_valid_until) {
                    drop(hold);
                    return Ok(PrefixPlan::Changed);
                }
                let mut controller = self.slo_controller.lock();
                controller.completion_next = None;
                controller.prefix = Some(SloPrefixCohort {
                    hold,
                    target: target_key,
                    identity,
                    expires_at: configured_end,
                    source_tokens,
                    source_incarnation,
                    target_incarnation,
                    maximum_sequence_tokens,
                    capture,
                    capture_span_start: None,
                    restored: false,
                });
                return Ok(PrefixPlan::Changed);
            }
        }
        Ok(PrefixPlan::None)
    }

    fn prefix_planner(&self) -> BoundedSloPlanner {
        BoundedSloPlanner {
            settings: BoundedPlannerSettings {
                search: self.config.scheduler.slo.planner.clone(),
                ..Default::default()
            },
        }
    }

    fn prefix_comparison_current(
        &self,
        captured: &ControllerSnapshot,
        maintenance: &super::super::cost_observation::PrefixCostSnapshot,
        expires_at: Instant,
    ) -> bool {
        slo_clock_now() < expires_at
            && maintenance.current()
            && captured.poll_planning()
            && self.cost_runtime.as_ref().and_then(|runtime| {
                runtime.try_model_version_current(captured.model.model_version())
            }) == Some(true)
            && self.controller_frontiers_match(captured)
            && captured.poll_planning()
            && slo_clock_now() < expires_at
            && maintenance.current()
    }
}

fn matches_key(key: &RequestWorkKey, prefix: &PrefixRequestKey, engine_incarnation: u64) -> bool {
    key.request_id == *prefix.request_id() && key.incarnation == engine_incarnation
}
fn matches_current_key(
    key: &RequestWorkKey,
    prefix: &PrefixRequestKey,
    engine_incarnation: u64,
) -> bool {
    matches_key(key, prefix, engine_incarnation) && key.work_generation == prefix.work_generation()
}
fn cohort_identity(
    source: &PrefixRequestKey,
    target: &PrefixRequestKey,
    boundary: usize,
) -> Option<[u8; 32]> {
    let serial = NEXT_COHORT
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_add(1))
        .ok()?;
    let mut result = [0; 32];
    for (chunk, value) in result.chunks_exact_mut(8).zip([
        source.ordinal(),
        target.ordinal(),
        u64::try_from(boundary).ok()?,
        serial,
    ]) {
        chunk.copy_from_slice(&value.to_le_bytes());
    }
    Some(result)
}

impl EngineInner {
    pub(super) fn controller_projection_limits(&self) -> ResourcePlanningLimits {
        let maintenance = if self
            .config
            .scheduler
            .prefix_rendezvous_max_wait_ms
            .is_some()
        {
            2
        } else if self.config.runtime.prefix_state_cache_enabled
            && self.model_executor.supports_guarded_prefix_maintenance()
        {
            1
        } else {
            0
        };
        ResourcePlanningLimits {
            maximum_participants: 256,
            // H still bounds inference waves. The numeric ledger also records
            // at most Capture+Restore, or a sole ready-cache Restore edge.
            maximum_projected_waves: self.config.scheduler.slo.planner.lookahead_waves.get()
                + maintenance,
            ..Default::default()
        }
    }
}
