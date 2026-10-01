//! An already published checkpoint needs no live producer and no waiting hold.
use super::*;
use ferrum_interfaces::model_executor::{PrefixCaptureStatus, PrefixReadyRestoreRequest};
use ferrum_interfaces::vnext::ExecutionCostRouteAvailability;

pub(in crate::continuous_engine::inner::slo_controller) struct ReadyPrefixCohort {
    pub(super) target: RequestId,
    pub(super) owner: Arc<()>,
    pub(super) incarnation: u64,
    pub(super) identity: [u8; 32],
    pub(super) boundary: NonZeroU32,
    pub(super) expires_at: Instant,
    pub(super) lease: Arc<dyn PrefixCaptureLease>,
    pub(super) restored: bool,
}

pub(in crate::continuous_engine::inner::slo_controller) struct PreparedReadyPlan {
    cohort: ReadyPrefixCohort,
    evidence: ReadyPrefixRestoreEvidence,
    maintenance: Arc<super::super::super::cost_observation::PrefixCostSnapshot>,
    valid_until: Instant,
}

pub(in crate::continuous_engine) struct PreparedReadyMaintenance {
    pub(super) cohort: ReadyPrefixCohort,
    pub(super) evidence: ReadyPrefixRestoreEvidence,
    pub(super) maintenance: Arc<super::super::super::cost_observation::PrefixCostSnapshot>,
    pub(super) valid_until: Instant,
    pub(super) model_version: u64,
    pub(super) proof: ControllerSafetyProof,
}

impl PreparedReadyPlan {
    pub(in crate::continuous_engine::inner::slo_controller) fn prepare(
        self,
        proof: ControllerSafetyProof,
        model_version: u64,
    ) -> PreparedReadyMaintenance {
        PreparedReadyMaintenance {
            cohort: self.cohort,
            evidence: self.evidence,
            maintenance: self.maintenance,
            valid_until: self.valid_until,
            model_version,
            proof,
        }
    }
}

impl EngineInner {
    pub(super) fn plan_ready_prefix(
        &self,
        captured: &ControllerSnapshot,
        maintenance: Arc<super::super::super::cost_observation::PrefixCostSnapshot>,
    ) -> Result<PrefixPlan> {
        let existing = self.slo_controller.lock().ready_prefix.take();
        let cohort = if let Some(cohort) = existing {
            cohort
        } else {
            let mut selected = None;
            for row in &captured.snapshot.requests {
                if !captured.poll_planning() {
                    return Ok(PrefixPlan::Inconclusive);
                }
                if row.context_tokens != 0
                    || row.timing.committed_tokens != 0
                    || row.readiness != RequestReadiness::Ready
                    || !matches!(&row.phase, RequestPhaseView::Prefill(p) if p.offset == 0)
                {
                    continue;
                }
                let Some((tokens, maximum, owner)) =
                    self.sequences.try_read().and_then(|sequences| {
                        let sequence = sequences.get(&row.key.request_id)?;
                        let frontier = sequence.cost_frontier?;
                        (frontier.owner_incarnation.get() == row.key.incarnation
                            && sequence.prefill_tokens_processed == 0
                            && !sequence.prefill_complete
                            && sequence.generated_tokens.is_empty())
                        .then(|| {
                            (
                                sequence.prefill_context_tokens(),
                                sequence.model_maximum_sequence_tokens(),
                                Arc::clone(&sequence.stream_projection_identity),
                            )
                        })
                    })
                else {
                    continue;
                };
                let result = self.model_executor.try_retain_ready_prefix(
                    PrefixReadyRestoreRequest {
                        request_id: &row.key.request_id,
                        input_tokens: &tokens,
                        maximum_sequence_tokens: maximum,
                    },
                    &mut captured
                        .budget
                        .observed_resource_budget(&mut || captured.poll_planning()),
                );
                let lease = match result {
                    ExecutionCostRouteAvailability::Known(Some(lease)) => lease,
                    ExecutionCostRouteAvailability::Known(None) => continue,
                    ExecutionCostRouteAvailability::Unknown(
                        ferrum_interfaces::vnext::ExecutionCostRouteUnknown::Unsupported,
                    ) => {
                        // An optional facade may be absent on a provider that
                        // still supports the original rendezvous protocol.
                        // This makes no claim that its cache is empty.
                        return Ok(PrefixPlan::None);
                    }
                    ExecutionCostRouteAvailability::Unknown(_) => {
                        return Ok(PrefixPlan::Inconclusive)
                    }
                };
                let Some(boundary) = u32::try_from(lease.boundary())
                    .ok()
                    .and_then(NonZeroU32::new)
                    .filter(|b| (b.get() as usize) < tokens.len())
                else {
                    return Ok(PrefixPlan::Inconclusive);
                };
                let Some(serial) = NEXT_COHORT
                    .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_add(1))
                    .ok()
                else {
                    return Ok(PrefixPlan::Inconclusive);
                };
                let mut identity = [0; 32];
                for (part, value) in identity.chunks_exact_mut(8).zip([
                    row.key.incarnation,
                    u64::from(boundary.get()),
                    captured.snapshot.generation,
                    serial,
                ]) {
                    part.copy_from_slice(&value.to_le_bytes());
                }
                let Ok(expires_at) = captured
                    .origin
                    .instant_at_ns(captured.snapshot.scope.horizon_end_ns)
                else {
                    return Ok(PrefixPlan::Inconclusive);
                };
                selected = Some(ReadyPrefixCohort {
                    target: row.key.request_id.clone(),
                    owner,
                    incarnation: row.key.incarnation,
                    identity,
                    boundary,
                    expires_at,
                    lease,
                    restored: false,
                });
                break;
            }
            let Some(cohort) = selected else {
                return Ok(PrefixPlan::None);
            };
            cohort
        };
        if !self.prefix_comparison_current(captured, &maintenance, cohort.expires_at)
            || cohort.lease.status() != PrefixCaptureStatus::Ready
        {
            return Ok(PrefixPlan::Changed);
        }
        let Some(target) = captured.snapshot.requests.iter().find(|row| {
            row.key.request_id == cohort.target && row.key.incarnation == cohort.incarnation
        }) else {
            return Ok(PrefixPlan::Changed);
        };
        if !self.sequences.try_read().is_some_and(|sequences| {
            sequences.get(&cohort.target).is_some_and(|sequence| {
                Arc::ptr_eq(&sequence.stream_projection_identity, &cohort.owner)
            })
        }) {
            return Ok(PrefixPlan::Changed);
        }
        let Ok(expires_at_ns) = captured.origin.at_ns(cohort.expires_at) else {
            return Ok(PrefixPlan::Changed);
        };
        let offer = ReadyPrefixRestoreOffer {
            identity: cohort.identity,
            based_on_generation: captured.snapshot.generation,
            target: target.key.clone(),
            boundary_tokens: cohort.boundary,
            expires_at_ns,
        };
        let phase = if cohort.restored {
            ReadyPrefixPhase::Restored
        } else {
            ReadyPrefixPhase::Ready
        };
        let context = shape::ReadyPrefixExecutorShape {
            source: shape::ExecutorShape {
                engine: self,
                captured,
            },
            offer: &offer,
            lease: cohort.lease.as_ref(),
        };
        let cost = AnchoredPlanningCostModel::new(captured.model.as_ref(), captured.anchor);
        let prefix_cost = maintenance.anchored_ready(captured.anchor, &offer);
        let Ok(window) = captured.planning_window() else {
            return Ok(PrefixPlan::Changed);
        };
        let decision = captured
            .origin
            .plan_ready_prefix_restore_scoped_with_execution_budget_window(
                &self.prefix_planner(),
                &captured.snapshot,
                &offer,
                phase,
                &cost,
                &prefix_cost,
                &context,
                Some(Arc::clone(&captured.protection)),
                window,
                slo_clock_now,
            );
        if let Ok(result) = &decision {
            let (disposition, reason) = captured.budget.record_ready_prefix(result);
            self.record_controller(ControllerObservation {
                obligations: captured.snapshot.requests.len(),
                disposition,
                reason,
            });
        }
        self.record_prefix_decision(
            prefix_observation::Route::ReadyRestore,
            &offer.target,
            None,
            offer.boundary_tokens.get(),
            captured,
            maintenance.model_version(),
            match &decision {
                Ok(ReadyPrefixDecision::Ready { .. }) => prefix_observation::Decision::Known,
                Ok(ReadyPrefixDecision::PreferDirect { .. }) => {
                    prefix_observation::Decision::PreferDirect
                }
                Ok(ReadyPrefixDecision::Unknown { reason, .. }) => {
                    prefix_observation::Decision::Unknown(*reason)
                }
                Err(reason) => prefix_observation::Decision::ClockError(*reason),
            },
        );
        let evidence = match decision {
            Ok(ReadyPrefixDecision::Ready { evidence, .. }) => evidence,
            Ok(ReadyPrefixDecision::PreferDirect { .. }) => return Ok(PrefixPlan::Inconclusive),
            _ => return Ok(PrefixPlan::Inconclusive),
        };
        if !self.prefix_comparison_current(captured, &maintenance, cohort.expires_at)
            || evidence.protection().as_ref() != captured.protection.as_ref()
        {
            return Ok(PrefixPlan::Changed);
        }
        let Ok(valid_until) = captured.origin.instant_at_ns(evidence.valid_until_ns()) else {
            return Ok(PrefixPlan::Changed);
        };
        match evidence.action().clone() {
            ReadyPrefixAction::Wave(selected) => {
                let checkpoint = Arc::clone(&cohort.lease);
                // Peer waves retain this continuation until a real target suffix wave.
                // That wave still carries both model epochs.
                // Its ordinary submit guard owns the fresh independent replay.
                let target_suffix = selected
                    .candidate
                    .work
                    .iter()
                    .any(|work| work.key == offer.target);
                if !cohort.restored || !target_suffix {
                    self.slo_controller.lock().ready_prefix = Some(cohort);
                }
                Ok(PrefixPlan::Wave {
                    checkpoint,
                    selected,
                    maintenance,
                })
            }
            ReadyPrefixAction::Restore(evidence) => {
                Ok(PrefixPlan::ReadyMaintenance(PreparedReadyPlan {
                    cohort,
                    evidence,
                    maintenance,
                    valid_until,
                }))
            }
        }
    }
}
