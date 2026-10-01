//! A business checkpoint is optional work with an ordinary full-queue proof.
//! It does not hold a follower or manufacture a future cache-hit benefit.
use super::*;

#[derive(Clone)]
pub(in crate::continuous_engine::inner::slo_controller) struct CacheCaptureContinuation {
    request_id: RequestId,
    incarnation: u64,
    owner: Arc<()>,
    identity: [u8; 32],
    boundary: NonZeroU32,
    pub(super) expires_at: Instant,
}

pub(in crate::continuous_engine::inner::slo_controller) struct PreparedCacheCapturePlan {
    pub(super) request_id: RequestId,
    pub(super) tokens: Vec<TokenId>,
    pub(super) maximum_sequence_tokens: usize,
    pub(super) offer: PrefixCacheCaptureOffer,
    pub(super) evidence: PrefixMaintenanceEvidence,
    pub(super) maintenance: Arc<super::super::super::cost_observation::PrefixCostSnapshot>,
    // Latest native submission time and the original retention horizon are
    // separate: spending the proven operation time must not expire the lease.
    pub(super) valid_until: Instant,
    pub(super) expires_at: Instant,
}

pub(in crate::continuous_engine) struct PreparedCacheCapture {
    pub(super) plan: PreparedCacheCapturePlan,
    pub(super) proof: ControllerSafetyProof,
    pub(super) model_version: u64,
}

impl PreparedCacheCapturePlan {
    pub(in crate::continuous_engine::inner::slo_controller) fn prepare(
        self,
        proof: ControllerSafetyProof,
        model_version: u64,
    ) -> PreparedCacheCapture {
        PreparedCacheCapture {
            plan: self,
            proof,
            model_version,
        }
    }
}

impl EngineInner {
    pub(super) fn plan_prefix_cache_capture(
        &self,
        captured: &ControllerSnapshot,
        maintenance: Arc<super::super::super::cost_observation::PrefixCostSnapshot>,
    ) -> Result<PrefixPlan> {
        let pending = {
            let mut state = self.slo_controller.lock();
            state.prefix_cache_captures.retain(|(id, incarnation, _)| {
                captured
                    .snapshot
                    .requests
                    .iter()
                    .any(|row| &row.key.request_id == id && row.key.incarnation == *incarnation)
            });
            state.prefix_cache_preparing.take()
        };
        if pending.as_ref().is_some_and(|old| {
            slo_clock_now() >= old.expires_at
                || !captured.snapshot.requests.iter().any(|r| {
                    r.key.request_id == old.request_id && r.key.incarnation == old.incarnation
                })
        }) {
            return Ok(PrefixPlan::None);
        }
        for (index, row) in captured.snapshot.requests.iter().enumerate() {
            if !captured.poll_planning() {
                return Ok(PrefixPlan::None);
            }
            if pending.as_ref().is_some_and(|old| {
                old.request_id != row.key.request_id || old.incarnation != row.key.incarnation
            }) {
                continue;
            }
            let RequestPhaseView::Prefill(progress) = &row.phase else {
                continue;
            };
            if row.readiness != RequestReadiness::Ready
                || row.timing.committed_tokens != 0
                || progress.offset >= progress.total_prompt_tokens.get()
            {
                continue;
            }
            let Some((tokens, maximum_sequence_tokens, owner)) =
                self.sequences.try_read().and_then(|sequences| {
                    let sequence = sequences.get(&row.key.request_id)?;
                    let frontier = sequence.cost_frontier?;
                    (frontier.owner_incarnation.get() == row.key.incarnation
                        && sequence.prefill_tokens_processed == progress.offset as usize
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
                return Ok(PrefixPlan::None);
            };
            if pending
                .as_ref()
                .is_some_and(|old| !Arc::ptr_eq(&owner, &old.owner))
            {
                return Ok(PrefixPlan::None);
            }
            // Query the actual model declaration over the current legal work
            // envelope. This does not grant that span as an executable candidate.
            let ready_decoders = captured
                .snapshot
                .requests
                .iter()
                .filter(|r| {
                    r.readiness == RequestReadiness::Ready
                        && matches!(r.phase, RequestPhaseView::Decode)
                        && !r.timing.completed()
                })
                .count();
            let envelope = captured
                .snapshot
                .capabilities
                .work_policy
                .for_ready_decoders(ready_decoders);
            let count = u64::from(progress.total_prompt_tokens.get() - progress.offset)
                .min(
                    captured
                        .snapshot
                        .capabilities
                        .max_prefill_tokens_per_wave
                        .get(),
                )
                .min(envelope.maximum_wave_tokens)
                .min(envelope.maximum_prefill_chunk.unwrap_or(u64::MAX))
                .min(envelope.maximum_prefill_tokens.unwrap_or(u64::MAX));
            let completed = captured
                .resources
                .participants()
                .get(index)
                .and_then(|participant| participant.completed_checkpoint_boundary());
            let at_boundary = completed
                .filter(|proof| {
                    proof.completed_tokens() == u64::from(progress.offset)
                        && proof.prompt_tokens() == u64::from(progress.total_prompt_tokens.get())
                        && proof.span_start() < proof.completed_tokens()
                })
                .and_then(|proof| {
                    let start = u32::try_from(proof.span_start()).ok()?;
                    let chunk = ferrum_interfaces::model_executor::PrefillChunk::new(
                        start as usize,
                        (progress.offset - start) as usize,
                        progress.total_prompt_tokens.get() as usize,
                    )
                    .ok()?;
                    let declared = self
                        .model_executor
                        .plan_prompt_tail_capture_boundary(chunk)?;
                    (declared.boundary == progress.offset as usize
                        && declared.span.permits(u64::from(progress.offset - start)))
                    .then_some((progress.offset, start, PrefixCacheCapturePhase::AtBoundary))
                });
            let preparing = || {
                let count = u32::try_from(count).ok()?;
                if count == 0 {
                    return None;
                }
                let chunk = ferrum_interfaces::model_executor::PrefillChunk::new(
                    progress.offset as usize,
                    count as usize,
                    progress.total_prompt_tokens.get() as usize,
                )
                .ok()?;
                let declared = self
                    .model_executor
                    .plan_prompt_tail_capture_boundary(chunk)?;
                let boundary = u32::try_from(declared.boundary).ok()?;
                (boundary > progress.offset
                    && boundary < progress.total_prompt_tokens.get()
                    && boundary <= chunk.end() as u32
                    && declared.span.permits(u64::from(boundary - progress.offset)))
                .then_some((
                    boundary,
                    progress.offset,
                    PrefixCacheCapturePhase::Preparing,
                ))
            };
            let Some((boundary, span_start, phase)) = at_boundary.or_else(preparing) else {
                continue;
            };
            if pending
                .as_ref()
                .is_some_and(|old| old.boundary.get() != boundary)
            {
                return Ok(PrefixPlan::None);
            }
            if self.slo_controller.lock().prefix_cache_captures.iter().any(
                |(id, incarnation, attempted)| {
                    id == &row.key.request_id
                        && *incarnation == row.key.incarnation
                        && *attempted == boundary
                },
            ) {
                continue;
            }
            let mut cohort = if let Some(old) = &pending {
                old.clone()
            } else {
                let Some(serial) = NEXT_COHORT
                    .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_add(1))
                    .ok()
                else {
                    return Ok(PrefixPlan::None);
                };
                let mut identity = [0; 32];
                for (part, value) in identity.chunks_exact_mut(8).zip([
                    row.key.incarnation,
                    u64::from(boundary),
                    captured.snapshot.generation,
                    serial,
                ]) {
                    part.copy_from_slice(&value.to_le_bytes());
                }
                let Ok(expires_at) = captured
                    .origin
                    .instant_at_ns(captured.snapshot.scope.horizon_end_ns)
                else {
                    return Ok(PrefixPlan::None);
                };
                CacheCaptureContinuation {
                    request_id: row.key.request_id.clone(),
                    incarnation: row.key.incarnation,
                    owner,
                    identity,
                    boundary: NonZeroU32::new(boundary).unwrap(),
                    expires_at,
                }
            };
            let Ok(current_horizon) = captured
                .origin
                .instant_at_ns(captured.snapshot.scope.horizon_end_ns)
            else {
                return Ok(PrefixPlan::None);
            };
            cohort.expires_at = cohort.expires_at.min(current_horizon);
            let Ok(expires_at_ns) = captured.origin.at_ns(cohort.expires_at) else {
                return Ok(PrefixPlan::None);
            };
            let offer = PrefixCacheCaptureOffer {
                identity: cohort.identity,
                based_on_generation: captured.snapshot.generation,
                source: row.key.clone(),
                capture_span_start: span_start,
                boundary_tokens: cohort.boundary,
                expires_at_ns: expires_at_ns.min(captured.snapshot.scope.horizon_end_ns),
            };
            let context = shape::CacheCaptureExecutorShape {
                source: shape::ExecutorShape {
                    engine: self,
                    captured,
                },
                offer: &offer,
                phase,
            };
            let cost = AnchoredPlanningCostModel::new(captured.model.as_ref(), captured.anchor);
            let prefix_cost = maintenance.anchored_capture(captured.anchor, &offer);
            let Ok(window) = captured.planning_window() else {
                return Ok(PrefixPlan::None);
            };
            let decision = captured
                .origin
                .plan_prefix_cache_capture_in_phase_scoped_with_execution_budget_window(
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
                let (disposition, reason) = captured.budget.record_prefix_cache_capture(result);
                self.record_controller(ControllerObservation {
                    obligations: captured.snapshot.requests.len(),
                    disposition,
                    reason,
                });
            }
            self.record_prefix_decision(
                prefix_observation::Route::CacheCapture,
                &offer.source,
                None,
                offer.boundary_tokens.get(),
                captured,
                maintenance.model_version(),
                match &decision {
                    Ok(PrefixCacheCaptureDecision::Known { .. }) => {
                        prefix_observation::Decision::Known
                    }
                    Ok(PrefixCacheCaptureDecision::Unknown { reason, .. }) => {
                        prefix_observation::Decision::Unknown(*reason)
                    }
                    Err(reason) => prefix_observation::Decision::ClockError(*reason),
                },
            );
            let Ok(PrefixCacheCaptureDecision::Known { evidence, .. }) = decision else {
                return Ok(PrefixPlan::None);
            };
            let Ok(valid_until) = captured.origin.instant_at_ns(evidence.valid_until_ns()) else {
                return Ok(PrefixPlan::None);
            };
            if evidence.protection().as_ref() != captured.protection.as_ref()
                || !self.prefix_comparison_current(
                    captured,
                    &maintenance,
                    valid_until.min(cohort.expires_at),
                )
            {
                return Ok(PrefixPlan::None);
            }
            match evidence.action().clone() {
                PrefixCacheCaptureAction::Wave(selected) => {
                    // No hold and no future physical allocation is installed.
                    // The next turn must prove this same source/current remainder
                    // again, within the original offer's absolute horizon.
                    self.slo_controller.lock().prefix_cache_preparing = Some(cohort);
                    return Ok(PrefixPlan::CachePreparation {
                        selected,
                        maintenance,
                    });
                }
                PrefixCacheCaptureAction::Capture(capture) => {
                    let mut state = self.slo_controller.lock();
                    state.prefix_cache_captures.retain(|(id, incarnation, _)| {
                        id != &row.key.request_id || *incarnation != row.key.incarnation
                    });
                    state.prefix_cache_captures.push((
                        row.key.request_id.clone(),
                        row.key.incarnation,
                        boundary,
                    ));
                    drop(state);
                    return Ok(PrefixPlan::CacheCapture(PreparedCacheCapturePlan {
                        request_id: row.key.request_id.clone(),
                        tokens,
                        maximum_sequence_tokens,
                        offer,
                        evidence: capture,
                        maintenance,
                        valid_until,
                        expires_at: cohort.expires_at,
                    }));
                }
            }
        }
        Ok(PrefixPlan::None)
    }
}
