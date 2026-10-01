use super::super::prefix_rendezvous::PathOffer;
use super::super::{
    PlanningPrefixCacheCaptureInput, PlanningPrefixCostModel, PlanningPrefixTransitionInput,
    PlanningReadyPrefixRestoreInput, PrefixMaintenanceEvidence, PrefixMaintenanceStage,
    PrefixPathStep, PrefixRendezvousOffer, PrefixRestoredFrontier, ReadyPrefixRestoreEvidence,
};
use super::*;
use crate::implementations::continuous::cost_model::WaveKind;

/// Restricts only scheduling eligibility. It creates no resource successor and
/// cannot unblock the target; only a validated restore transition can do that.
pub(in super::super) fn restrict_prefix_wait(
    state: &mut PlanningState<'_>,
    offer: &PrefixRendezvousOffer,
) -> Result<(), PlanningUnknownReason> {
    let producer = state
        .logical
        .requests
        .iter_mut()
        .find(|r| r.key == offer.producer)
        .ok_or(PlanningUnknownReason::InvalidSnapshot)?;
    let RequestPhaseView::Prefill(progress) = &mut producer.phase else {
        return Err(PlanningUnknownReason::InvalidSnapshot);
    };
    progress.executable_until = progress.executable_until.min(offer.boundary_tokens.get());
    state
        .logical
        .requests
        .iter_mut()
        .find(|r| r.key == offer.target)
        .ok_or(PlanningUnknownReason::InvalidSnapshot)?
        .readiness = RequestReadiness::StateBlocked;
    Ok(())
}

/// Eligibility only; the provider separately proves the declared boundary.
pub(in super::super) fn restrict_cache_capture(
    state: &mut PlanningState<'_>,
    offer: &super::super::PrefixCacheCaptureOffer,
) -> Result<(), PlanningUnknownReason> {
    let source = state
        .logical
        .requests
        .iter_mut()
        .find(|r| r.key == offer.source)
        .ok_or(PlanningUnknownReason::InvalidSnapshot)?;
    let RequestPhaseView::Prefill(progress) = &mut source.phase else {
        return Err(PlanningUnknownReason::InvalidSnapshot);
    };
    progress.executable_until = progress.executable_until.min(offer.boundary_tokens.get());
    Ok(())
}

pub(in super::super) fn advance_prefix<'epoch>(
    snapshot: &SchedulerSnapshot,
    parent: &PlanningState<'epoch>,
    offer: PathOffer<'_>,
    stage: PrefixMaintenanceStage,
    capture_span_start: Option<u32>,
    model: &dyn PlanningPrefixCostModel,
    expected_model_version: u64,
    poll: &mut dyn FnMut() -> Result<(), PlanningUnknownReason>,
    milestones_enabled: bool,
    protection: &PlanningObligationSet,
) -> Result<(PrefixPathStep, PlanningState<'epoch>), SimulationFailure> {
    let rendezvous = match offer {
        PathOffer::Rendezvous(offer) => Some(offer),
        PathOffer::Ready(_) | PathOffer::CacheCapture(_) => None,
    };
    if let Some(offer) = rendezvous {
        let producer = parent
            .requests
            .iter()
            .find(|r| r.key == offer.producer)
            .ok_or(PlanningUnknownReason::InvalidSnapshot)?;
        if (stage == PrefixMaintenanceStage::Capture
            && !matches!(&producer.phase, RequestPhaseView::Prefill(source)
                if source.offset == offer.boundary_tokens.get()))
            || capture_span_start.is_none_or(|start| start >= offer.boundary_tokens.get())
        {
            return Err(SimulationFailure::SequenceViolation);
        }
    } else if let PathOffer::CacheCapture(offer) = offer {
        let source = parent
            .requests
            .iter()
            .find(|r| r.key == offer.source)
            .ok_or(PlanningUnknownReason::InvalidSnapshot)?;
        if stage != PrefixMaintenanceStage::Capture
            || capture_span_start.is_none_or(|start| start >= offer.boundary_tokens.get())
            || source.readiness != RequestReadiness::Ready
            || !matches!(&source.phase, RequestPhaseView::Prefill(p)
                if p.offset == offer.boundary_tokens.get()
                    && capture_span_start.is_some_and(|start| start < p.offset))
        {
            return Err(PlanningUnknownReason::InvalidShapeEvidence.into());
        }
    } else if stage != PrefixMaintenanceStage::Restore || capture_span_start.is_some() {
        return Err(PlanningUnknownReason::InvalidShapeEvidence.into());
    }
    if protection.required_service(&parent.requests).is_some() {
        return Err(SimulationFailure::SequenceViolation);
    }
    let destination_offset = if let Some(target_key) = offer.target() {
        let target = parent
            .requests
            .iter()
            .find(|r| r.key == *target_key)
            .ok_or(PlanningUnknownReason::InvalidSnapshot)?;
        let RequestPhaseView::Prefill(destination) = &target.phase else {
            return Err(PlanningUnknownReason::InvalidShapeEvidence.into());
        };
        if target.readiness
            != if rendezvous.is_some() {
                RequestReadiness::StateBlocked
            } else {
                RequestReadiness::Ready
            }
            || target.timing.committed_tokens != 0
            || destination.offset != 0
        {
            return Err(SimulationFailure::SequenceViolation);
        }
        Some(destination.offset)
    } else {
        None
    };
    let projected = execution::checked(poll, |poll| match offer {
        PathOffer::Rendezvous(offer) => parent.execution.project_prefix_transition(
            &PlanningPrefixTransitionInput {
                snapshot,
                offer,
                stage,
                capture_span_start: capture_span_start.unwrap(),
                requests: &parent.requests,
            },
            poll,
        ),
        PathOffer::CacheCapture(offer) => parent.execution.project_prefix_cache_capture(
            &PlanningPrefixCacheCaptureInput {
                snapshot,
                offer,
                capture_span_start: capture_span_start.unwrap(),
                requests: &parent.requests,
            },
            poll,
        ),
        PathOffer::Ready(offer) => parent.execution.project_ready_prefix_restore(
            &PlanningReadyPrefixRestoreInput {
                snapshot,
                offer,
                requests: &parent.requests,
            },
            poll,
        ),
    })?
    .ok_or(PlanningUnknownReason::UnknownResourceEvidence)?;
    let expected_frontier = match stage {
        PrefixMaintenanceStage::Capture => None,
        PrefixMaintenanceStage::Restore => Some(PrefixRestoredFrontier {
            target: offer
                .target()
                .ok_or(PlanningUnknownReason::InvalidShapeEvidence)?
                .clone(),
            previous_offset: destination_offset
                .ok_or(PlanningUnknownReason::InvalidShapeEvidence)?,
            restored_tokens: offer.boundary(),
        }),
    };
    if projected.restored_frontier != expected_frontier {
        return Err(PlanningUnknownReason::InvalidShapeEvidence.into());
    }
    let mut logical = parent.logical.clone();
    if parent.depth > 0 {
        logical.now_ns = logical
            .now_ns
            .checked_add(parent.future_controller_ns)
            .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
    }
    let domain = &projected.cost_domain;
    if domain.shapes().is_empty() || domain.shapes().len() > 256 {
        return Err(PlanningUnknownReason::ShapeCapacity.into());
    }
    let mut duration_ns = 0;
    let mut valid_for_ns = u64::MAX;
    for shape in domain.shapes() {
        poll()?;
        let expected_kind = match stage {
            PrefixMaintenanceStage::Capture => WaveKind::Maintenance,
            PrefixMaintenanceStage::Restore => WaveKind::Restore,
        };
        if shape.kind != expected_kind
            || !shape.decode_kv_tokens.is_empty()
            || !shape.prefill_chunks.is_empty()
        {
            return Err(PlanningUnknownReason::InvalidShapeEvidence.into());
        }
        if model.model_version() != expected_model_version {
            return Err(PlanningUnknownReason::ModelVersionMismatch.into());
        }
        let cost = execution::checked(poll, |_| {
            match offer {
                PathOffer::Rendezvous(offer) => {
                    model.predict(&snapshot.fingerprint, offer, stage, shape, logical.now_ns)
                }
                PathOffer::Ready(offer) => {
                    model.predict_ready_restore(&snapshot.fingerprint, offer, shape, logical.now_ns)
                }
                PathOffer::CacheCapture(offer) => {
                    model.predict_cache_capture(&snapshot.fingerprint, offer, shape, logical.now_ns)
                }
            }
            .ok_or_else(|| {
                #[cfg(test)]
                prefix_cost_diagnostic(|| eprintln!("prefix maintenance edge lacks original prediction: stage={stage:?} now={} expires={} expected_epoch={expected_model_version} shape={shape:?}",
                    logical.now_ns, offer.expires_at_ns()));
                PlanningUnknownReason::CostUnavailable
            })
        })?;
        if cost.model_version != expected_model_version {
            return Err(PlanningUnknownReason::ModelVersionMismatch.into());
        }
        if cost.typical_ns == 0 || cost.planning_ns < cost.typical_ns {
            #[cfg(test)]
            prefix_cost_diagnostic(|| {
                eprintln!("prefix maintenance edge invalid original numeric cost: stage={stage:?} typical={} planning={} valid_for={} shape={shape:?}",
                cost.typical_ns, cost.planning_ns, cost.valid_for_ns)
            });
            return Err(PlanningUnknownReason::CostUnavailable.into());
        }
        duration_ns = duration_ns.max(cost.planning_ns);
        valid_for_ns = valid_for_ns.min(cost.valid_for_ns);
    }
    // The entire operation, including initialization/settlement, must remain
    // inside the original model and checkpoint retention lifetime.
    let freshness = valid_for_ns
        .checked_sub(duration_ns)
        .ok_or_else(|| {
            #[cfg(test)]
            prefix_cost_diagnostic(|| eprintln!("prefix maintenance edge original TTL cannot cover duration: stage={stage:?} valid_for={valid_for_ns} duration={duration_ns} now={} expires={} epoch={expected_model_version}",
                logical.now_ns, offer.expires_at_ns()));
            PlanningUnknownReason::CostUnavailable
        })?;
    let end_ns = logical
        .now_ns
        .checked_add(duration_ns)
        .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
    if end_ns >= offer.expires_at_ns() {
        return Err(SimulationFailure::SequenceViolation);
    }
    logical.minimum_cost_freshness_slack_ns =
        logical.minimum_cost_freshness_slack_ns.min(freshness);
    logical.minimum_start_slack_ns = logical
        .minimum_start_slack_ns
        .min(offer.expires_at_ns() - end_ns - 1);
    for (index, request) in logical.requests.iter().enumerate() {
        poll()?;
        if !request.timing.completed() && protection.protects(index) {
            let deadline = request
                .timing
                .next_deadline_ns()
                .ok_or(PlanningUnknownReason::ArithmeticOverflow)?;
            // Maintenance emits no token, including for its producer/target.
            if end_ns >= deadline {
                return Err(SimulationFailure::SequenceViolation);
            }
            logical.minimum_start_slack_ns =
                logical.minimum_start_slack_ns.min(deadline - end_ns - 1);
        }
        if milestones_enabled {
            check_milestones(snapshot, index, request, end_ns, false, Some(protection))?;
        }
    }
    if stage == PrefixMaintenanceStage::Restore {
        let target_key = offer
            .target()
            .ok_or(PlanningUnknownReason::InvalidShapeEvidence)?;
        let target = logical
            .requests
            .iter_mut()
            .find(|r| r.key == *target_key)
            .unwrap();
        let RequestPhaseView::Prefill(progress) = &mut target.phase else {
            unreachable!()
        };
        let restored = offer.boundary();
        if restored >= progress.total_prompt_tokens.get()
            || restored > progress.executable_until
            || restored > snapshot.capacity.maximum_context_tokens.get()
            || progress.reference.work_at(restored).is_none()
        {
            return Err(PlanningUnknownReason::InvalidShapeEvidence.into());
        }
        progress.offset = restored;
        progress.logical_high_water = progress.logical_high_water.max(restored);
        target.context_tokens = restored;
        target.readiness = RequestReadiness::Ready;
        // Do not reset the original admission/milestone anchors or count a
        // restored prefix as compute executed by this path.
        if let Some(offer) = rendezvous {
            let source = logical
                .requests
                .iter_mut()
                .find(|r| r.key == offer.producer)
                .unwrap();
            let initial = snapshot
                .requests
                .iter()
                .find(|r| r.key == offer.producer)
                .unwrap();
            if let (RequestPhaseView::Prefill(current), RequestPhaseView::Prefill(initial)) =
                (&mut source.phase, &initial.phase)
            {
                current.executable_until = initial.executable_until;
            }
        }
    }
    if let PathOffer::CacheCapture(offer) = offer {
        // The declared preparation cap ends at the successful capture edge.
        // It never changes the snapshot or any other queue owner's readiness.
        let source = logical
            .requests
            .iter_mut()
            .find(|r| r.key == offer.source)
            .ok_or(PlanningUnknownReason::InvalidSnapshot)?;
        let initial = snapshot
            .requests
            .iter()
            .find(|r| r.key == offer.source)
            .ok_or(PlanningUnknownReason::InvalidSnapshot)?;
        if let (RequestPhaseView::Prefill(current), RequestPhaseView::Prefill(initial)) =
            (&mut source.phase, &initial.phase)
        {
            current.executable_until = initial.executable_until;
        } else {
            return Err(PlanningUnknownReason::InvalidShapeEvidence.into());
        }
    }
    for (index, request) in logical.requests.iter().enumerate() {
        poll()?;
        if milestones_enabled {
            check_milestones(snapshot, index, request, end_ns, true, Some(protection))?;
            if let Some(slack) = newly_completed_milestone_slack(
                snapshot,
                index,
                logical_work(&snapshot.requests[index], &parent.requests[index])?,
                logical_work(&snapshot.requests[index], request)?,
                end_ns,
                Some(protection),
            ) {
                logical.minimum_start_slack_ns = logical.minimum_start_slack_ns.min(slack);
            }
        }
    }
    logical.now_ns = end_ns;
    Ok((
        match offer {
            PathOffer::Rendezvous(_) | PathOffer::CacheCapture(_) => {
                PrefixPathStep::Maintenance(PrefixMaintenanceEvidence {
                    stage,
                    capture_span_start: capture_span_start.unwrap(),
                    cost_domain: projected.cost_domain,
                    restored_frontier: projected.restored_frontier,
                    model_version: expected_model_version,
                })
            }
            PathOffer::Ready(_) => PrefixPathStep::ReadyRestore(ReadyPrefixRestoreEvidence {
                cost_domain: projected.cost_domain,
                restored_frontier: projected
                    .restored_frontier
                    .ok_or(PlanningUnknownReason::InvalidShapeEvidence)?,
                model_version: expected_model_version,
            }),
        },
        PlanningState {
            logical,
            execution: projected.successor,
            depth: parent.depth + 1,
            observed_replay: parent.observed_replay,
            future_controller_ns: parent.future_controller_ns,
            first_wave_candidate: parent.first_wave_candidate.clone(),
            first_wave_statistics: parent.first_wave_statistics.clone(),
            first_wave_canonical: parent.first_wave_canonical.clone(),
        },
    ))
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
            eprintln!(
                "additional prefix edge rejection diagnostics suppressed for this test thread"
            );
        }
    });
}
