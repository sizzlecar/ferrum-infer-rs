use super::super::prefill_budget::{PrefillPacking, PrefillPackingMode, PrefillWorkBudget};
use crate::{cost_prediction::*, joint_policy::*};
use ferrum_execution::{
    execution_attempt::{AttemptPrefixRestore, AttemptWorkKind},
    model_executor::{PrefillChunk, PrefillShapePolicy},
    token_execution::ExecutionRange,
};
use std::num::NonZeroUsize;

#[cfg(test)]
mod tests;

/// Dispatch edits admitted work; admission may preview a pure policy cap for
/// exact queued owners. Phases end at the original prompt completion fence.
#[derive(Debug, Clone, Copy)]
pub(super) enum FirstAction {
    Static,
    PrefillLimit(NonZeroUsize),
    PrefillPhaseLimit(NonZeroUsize),
    ReservedPrefillPhaseLimit(NonZeroUsize),
    /// Preview a durable cap before admission, including natural first-wave
    /// repacking. This does not itself grant an execution slot or reservation.
    PrefillCapCommit(NonZeroUsize),
    ReservedPrefillCapCommit(NonZeroUsize),
    /// The captured waiters may first need original resident slot releases.
    /// This positive cap never bypasses that admission plan.
    DeferredPrefillCapCommit(NonZeroUsize),
    DeferredReservedPrefillCapCommit(NonZeroUsize),
    /// Omit prefill from a mixed first wave while all its decoders progress.
    /// This is not an empty wave or permission to change later continuation.
    DecodeOnly,
}

impl FirstAction {
    pub(super) fn packing_mode(self) -> PrefillPackingMode {
        match self {
            Self::ReservedPrefillPhaseLimit(_)
            | Self::ReservedPrefillCapCommit(_)
            | Self::DeferredReservedPrefillCapCommit(_) => PrefillPackingMode::ReservePolicyChunk,
            _ => PrefillPackingMode::Refill,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct RequestShape {
    pub restored_tokens: usize,
    pub execution_ceiling: Option<usize>,
    /// An already selected phase is common to static and all new candidates.
    pub phase_cap: Option<NonZeroUsize>,
    pub phase_packing: PrefillPacking,
    pub decode_only_allowed: bool,
}

pub(super) struct CohortTrajectory {
    pub first_wave: Vec<GroupMember>,
    pub cap_commit_targets: Vec<PrefillTarget>,
    pub shapes: Vec<RequestShape>,
    pub admission: AdmissionSlotPlan,
    pub execution: CostExecution,
    pub shape_policy: PrefillShapePolicy,
    pub max_batch: usize,
    pub max_tokens: usize,
    pub prefill_step_chunk: Option<usize>,
    pub active_decode_prefill_chunk: Option<usize>,
    pub active_decode_prefill_budget: Option<usize>,
    pub decode_pressure_threshold: usize,
    pub default_active_chunk: usize,
    pub chunking: bool,
}

impl CohortTrajectory {
    /// A finite positive search scale from the declared prefill policy and
    /// frozen work, even when the current natural wave is decode-only. It is
    /// not a prediction that a waiting owner already has an execution slot.
    pub(super) fn deferred_prefill_scale(&self, requests: &[SimulatedRequest]) -> usize {
        self.cap_commit_targets
            .iter()
            .filter_map(|target| {
                if !matches!(
                    self.admission.initial_admission(target.request),
                    Some(InitialAdmission::Waiting { .. })
                ) {
                    return None;
                }
                let request = requests.get(target.request)?;
                let shape = self.shapes.get(target.request)?;
                let left = request.prompt_tokens.checked_sub(request.prefill_tokens)?;
                let mut scale = left
                    .min(self.prefill_step_chunk.unwrap_or(self.max_tokens))
                    .min(self.max_tokens);
                if let Some(ceiling) = shape.execution_ceiling {
                    scale = scale.min(ceiling);
                }
                if let Some(cap) = shape.phase_cap {
                    scale = scale.min(cap.get());
                }
                Some(scale)
            })
            .max()
            .unwrap_or(0)
    }

    pub(super) fn allows_cap_commit(
        &self,
        requests: &[SimulatedRequest],
        limit: NonZeroUsize,
    ) -> bool {
        self.first_wave.is_empty()
            && self.cap_commit_targets.iter().any(|target| {
                self.shapes
                    .get(target.request)
                    .zip(requests.get(target.request))
                    .is_some_and(|(shape, request)| {
                        shape.phase_cap.is_none_or(|old| old > limit)
                            && request.prompt_tokens == target.prompt_tokens
                            && request.prompt_tokens.saturating_sub(request.prefill_tokens)
                                > limit.get()
                    })
            })
    }

    pub(super) fn allows_prefill_phase(
        &self,
        requests: &[SimulatedRequest],
        limit: NonZeroUsize,
    ) -> bool {
        // Empty first_wave is the admission projection, not an actual selected
        // resident wave. It cannot promise a commitment for a waiting request.
        self.first_wave.iter().any(|member| {
            member.work.kind == AttemptWorkKind::Prefill
                && self.shapes[member.request]
                    .phase_cap
                    .is_none_or(|existing| existing > limit)
                && requests[member.request]
                    .prompt_tokens
                    .saturating_sub(requests[member.request].prefill_tokens)
                    > limit.get()
        })
    }

    pub(super) fn allows_decode_only(&self, members: &[GroupMember]) -> bool {
        let mut has_prefill = false;
        let mut has_decode = false;
        for member in members {
            let Some(shape) = self.shapes.get(member.request) else {
                return false;
            };
            match member.work.kind {
                AttemptWorkKind::Prefill => {
                    if !shape.decode_only_allowed {
                        return false;
                    }
                    has_prefill = true;
                }
                AttemptWorkKind::Decode => has_decode = true,
            }
        }
        has_prefill && has_decode
    }

    fn work(
        &self,
        index: usize,
        request: &SimulatedRequest,
        tokens: usize,
    ) -> Result<CostWork, Unsupported> {
        let shape = self.shapes.get(index).ok_or(Unsupported::InvalidState)?;
        let (kind, requested, context) = if request.output_tokens == 0 {
            let chunk = PrefillChunk::new(request.prefill_tokens, tokens, request.prompt_tokens)
                .map_err(|_| Unsupported::InvalidPlan)?;
            let projected = self
                .shape_policy
                .project(chunk, true)
                .map_err(|_| Unsupported::InvalidPlan)?;
            if !projected.is_exact() {
                return Err(Unsupported::Cost(CostUnknownReason::InexactWork));
            }
            let chunk = projected.chunk();
            (
                AttemptWorkKind::Prefill,
                ExecutionRange::new(chunk.tokens_processed(), chunk.tokens_to_process())
                    .map_err(|_| Unsupported::InvalidPlan)?,
                request.prompt_tokens,
            )
        } else {
            let context = request
                .prompt_tokens
                .checked_add(request.output_tokens)
                .ok_or(Unsupported::Arithmetic)?;
            (
                AttemptWorkKind::Decode,
                ExecutionRange::new(context.checked_sub(1).ok_or(Unsupported::InvalidState)?, 1)
                    .map_err(|_| Unsupported::InvalidPlan)?,
                context,
            )
        };
        Ok(CostWork {
            kind,
            requested,
            context_tokens: context,
            prefix_restore: if shape.restored_tokens == 0 {
                AttemptPrefixRestore::NoneObserved
            } else {
                AttemptPrefixRestore::Restored {
                    tokens: shape.restored_tokens,
                }
            },
            internally_reused_tokens: None,
        })
    }

    fn push(&self, members: &[GroupMember], plan: &mut PlanBuilder) -> Result<(), Unsupported> {
        if self.execution == CostExecution::SingleInvocation {
            return plan.push_group(members);
        }
        for kind in [AttemptWorkKind::Prefill, AttemptWorkKind::Decode] {
            plan.push_group_kind(members, kind)?;
        }
        Ok(())
    }
}

impl CohortTrajectory {
    pub(super) fn natural_members(
        &self,
        requests: &[SimulatedRequest],
    ) -> Result<Vec<GroupMember>, Unsupported> {
        self.natural_members_with_phase(requests, None)
    }

    fn natural_members_with_phase(
        &self,
        requests: &[SimulatedRequest],
        phase: Option<(NonZeroUsize, &[PrefillTarget])>,
    ) -> Result<Vec<GroupMember>, Unsupported> {
        if requests.len() != self.shapes.len() {
            return Err(Unsupported::ParticipantBudget);
        }
        let active = self.admission.active_mask(requests)?;
        self.natural_members_from_active(
            requests,
            phase.map(|(limit, targets)| (limit, targets, PrefillPackingMode::Refill)),
            &active,
        )
    }

    fn natural_members_from_active(
        &self,
        requests: &[SimulatedRequest],
        phase: Option<(NonZeroUsize, &[PrefillTarget], PrefillPackingMode)>,
        active: &[bool],
    ) -> Result<Vec<GroupMember>, Unsupported> {
        let mut members = Vec::new();
        self.natural_members_into(requests, phase, active, &mut members)?;
        Ok(members)
    }

    fn natural_members_into(
        &self,
        requests: &[SimulatedRequest],
        phase: Option<(NonZeroUsize, &[PrefillTarget], PrefillPackingMode)>,
        active: &[bool],
        members: &mut Vec<GroupMember>,
    ) -> Result<(), Unsupported> {
        if requests.len() != self.shapes.len() || active.len() != requests.len() {
            return Err(Unsupported::ParticipantBudget);
        }
        members.clear();
        members.reserve_exact(requests.len());
        let mut remaining = self.max_tokens;
        // The supported cohort fits the full decode width, so future cursor
        // rotation cannot change which resident decoders run in a wave.
        for (index, request) in requests.iter().enumerate() {
            if !active[index] || request.finished() || request.output_tokens == 0 {
                continue;
            }
            if remaining == 0 {
                return Err(Unsupported::ParticipantBudget);
            }
            members.push(GroupMember {
                request: index,
                work: self.work(index, request, 1)?,
            });
            remaining -= 1;
        }
        let decode_count = members.len();
        let active_chunk = if decode_count == 0 {
            None
        } else {
            self.active_decode_prefill_chunk.or_else(|| {
                if decode_count >= self.decode_pressure_threshold {
                    Some(self.default_active_chunk)
                } else {
                    self.active_decode_prefill_budget
                }
            })
        };
        let effective_chunk = active_chunk.map(|chunk| {
            self.prefill_step_chunk
                .map_or(chunk, |step| step.min(chunk))
                .max(1)
        });
        let mut prefill_budget = effective_chunk
            .map(|chunk| {
                let backlog = requests
                    .iter()
                    .enumerate()
                    .filter(|(index, request)| {
                        active[*index] && !request.finished() && request.output_tokens == 0
                    })
                    .count();
                let quota = chunk
                    .saturating_mul(self.max_batch.saturating_sub(decode_count).min(backlog))
                    .min(remaining);
                self.active_decode_prefill_budget
                    .map_or(quota, |budget| quota.min(budget))
            })
            .unwrap_or(remaining);
        for (index, request) in requests.iter().enumerate() {
            if !active[index] || request.finished() || request.output_tokens > 0 {
                continue;
            }
            let budget = remaining.min(prefill_budget);
            if budget == 0 || members.len() >= self.max_batch {
                break;
            }
            let left = request
                .prompt_tokens
                .checked_sub(request.prefill_tokens)
                .ok_or(Unsupported::InvalidState)?;
            if left == 0 {
                return Err(Unsupported::InvalidState);
            }
            let chunk = effective_chunk.or(self.prefill_step_chunk);
            let tokens = if let Some(chunk) = chunk {
                left.min(chunk).min(budget)
            } else if self.chunking {
                left.min(budget)
            } else if left <= budget {
                left
            } else {
                continue;
            };
            let tokens = self.shapes[index]
                .execution_ceiling
                .map_or(tokens, |ceiling| ceiling.min(tokens));
            let shape = &self.shapes[index];
            let (cap, packing) = match phase {
                Some((limit, targets, mode))
                    if targets.iter().any(|target| target.request == index) =>
                {
                    (
                        Some(shape.phase_cap.map_or(limit, |prior| prior.min(limit))),
                        shape.phase_packing.after_selection(shape.phase_cap, mode),
                    )
                }
                _ => (shape.phase_cap, shape.phase_packing),
            };
            let quantum = chunk.map(|chunk| {
                let quantum = chunk.min(budget);
                shape
                    .execution_ceiling
                    .map_or(quantum, |ceiling| quantum.min(ceiling))
            });
            let charge = PrefillWorkBudget::for_cap(tokens, quantum, cap, packing);
            let work = self.work(index, request, charge.execution_tokens)?;
            members.push(GroupMember {
                request: index,
                work,
            });
            // Static scheduling spends the nominal chunk before the executor's
            // prompt-tail projection. A shorter execution is not extra budget
            // for a later participant in this same wave.
            remaining -= charge.nominal_charge;
            prefill_budget -= charge.nominal_charge;
        }
        if members.is_empty() {
            return Err(Unsupported::NoProgress);
        }
        Ok(())
    }
}

impl ReplayableTrajectory<FirstAction> for CohortTrajectory {}

impl TrajectoryAdapter<FirstAction> for CohortTrajectory {
    fn admission_slots(&self) -> Option<&AdmissionSlotPlan> {
        Some(&self.admission)
    }

    fn continuation_phase(
        &self,
        action: &FirstAction,
        _: &[SimulatedRequest],
        phase: &mut PhaseBuilder<'_>,
    ) -> Result<(), Unsupported> {
        if matches!(
            action,
            FirstAction::PrefillPhaseLimit(_) | FirstAction::ReservedPrefillPhaseLimit(_)
        ) {
            if self.first_wave.is_empty() {
                return Err(Unsupported::InvalidPlan);
            }
            for member in &self.first_wave {
                if member.work.kind == AttemptWorkKind::Prefill {
                    phase.prefill_target(member.request)?;
                }
            }
        }
        if matches!(
            action,
            FirstAction::PrefillCapCommit(_) | FirstAction::ReservedPrefillCapCommit(_)
        ) {
            if !self.first_wave.is_empty() || self.cap_commit_targets.is_empty() {
                return Err(Unsupported::InvalidPlan);
            }
            for target in &self.cap_commit_targets {
                phase.prefill_target(target.request)?;
            }
        }
        if matches!(
            action,
            FirstAction::DeferredPrefillCapCommit(_)
                | FirstAction::DeferredReservedPrefillCapCommit(_)
        ) {
            if !self.first_wave.is_empty() || self.cap_commit_targets.is_empty() {
                return Err(Unsupported::InvalidPlan);
            }
            for target in &self.cap_commit_targets {
                phase.deferred_prefill_target(target.request)?;
            }
        }
        Ok(())
    }

    fn plan_phase(
        &self,
        action: &FirstAction,
        phase: &ContinuationPhase,
        requests: &[SimulatedRequest],
        plan: &mut PlanBuilder,
    ) -> Result<(), Unsupported> {
        let (FirstAction::PrefillPhaseLimit(limit)
        | FirstAction::ReservedPrefillPhaseLimit(limit)
        | FirstAction::PrefillCapCommit(limit)
        | FirstAction::ReservedPrefillCapCommit(limit)
        | FirstAction::DeferredPrefillCapCommit(limit)
        | FirstAction::DeferredReservedPrefillCapCommit(limit)) = action
        else {
            return Err(Unsupported::InvalidPlan);
        };
        plan.with_members(|members, plan| {
            self.natural_members_into(
                requests,
                Some((*limit, phase.prefill_targets(), action.packing_mode())),
                plan.active_mask().ok_or(Unsupported::InvalidState)?,
                members,
            )?;
            self.push(members, plan)
        })
    }

    fn plan(
        &self,
        first_action: Option<&FirstAction>,
        requests: &[SimulatedRequest],
        plan: &mut PlanBuilder,
    ) -> Result<(), Unsupported> {
        plan.with_members(|members, plan| self.plan_into(first_action, requests, members, plan))
    }
}

impl CohortTrajectory {
    fn plan_into(
        &self,
        first_action: Option<&FirstAction>,
        requests: &[SimulatedRequest],
        members: &mut Vec<GroupMember>,
        plan: &mut PlanBuilder,
    ) -> Result<(), Unsupported> {
        if let Some(
            action @ (FirstAction::PrefillCapCommit(limit)
            | FirstAction::ReservedPrefillCapCommit(limit)
            | FirstAction::DeferredPrefillCapCommit(limit)
            | FirstAction::DeferredReservedPrefillCapCommit(limit)),
        ) = first_action
        {
            if !self.first_wave.is_empty() || self.cap_commit_targets.is_empty() {
                return Err(Unsupported::InvalidPlan);
            }
            self.natural_members_into(
                requests,
                Some((*limit, &self.cap_commit_targets, action.packing_mode())),
                plan.active_mask().ok_or(Unsupported::InvalidState)?,
                members,
            )?;
            return self.push(members, plan);
        }
        if first_action.is_some() && !self.first_wave.is_empty() {
            members.reserve_exact(self.first_wave.len());
            members.extend_from_slice(&self.first_wave);
        } else {
            self.natural_members_into(
                requests,
                None,
                plan.active_mask().ok_or(Unsupported::InvalidState)?,
                members,
            )?;
        }
        match first_action {
            Some(FirstAction::DecodeOnly) => {
                if !self.allows_decode_only(members) {
                    return Err(Unsupported::InvalidPlan);
                }
                members.retain(|member| member.work.kind == AttemptWorkKind::Decode);
            }
            Some(
                FirstAction::PrefillLimit(limit)
                | FirstAction::PrefillPhaseLimit(limit)
                | FirstAction::ReservedPrefillPhaseLimit(limit),
            ) => {
                for member in members.iter_mut() {
                    if member.work.kind == AttemptWorkKind::Prefill {
                        let request = requests
                            .get(member.request)
                            .ok_or(Unsupported::InvalidPlan)?;
                        member.work = self.work(
                            member.request,
                            request,
                            limit.get().min(member.work.requested.as_range().len()),
                        )?;
                    }
                }
            }
            Some(FirstAction::Static) | None => {}
            Some(
                FirstAction::PrefillCapCommit(_)
                | FirstAction::ReservedPrefillCapCommit(_)
                | FirstAction::DeferredPrefillCapCommit(_)
                | FirstAction::DeferredReservedPrefillCapCommit(_),
            ) => {
                unreachable!("handled before fixed-wave planning")
            }
        }
        if members.is_empty() {
            return Err(Unsupported::InvalidPlan);
        }
        self.push(members, plan)
    }
}
