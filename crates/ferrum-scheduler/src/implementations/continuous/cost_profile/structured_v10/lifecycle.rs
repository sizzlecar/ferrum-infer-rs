//! Every request starts at zero and completes its original immutable budget.
//! Outside-window settlements are still necessary lifecycle/FIFO evidence.
use super::*;
use ferrum_interfaces::execution_cost::{HostContentDomainV1, HostCostPolicyV2};
use std::collections::VecDeque;

/// Source3/4/5 keep their original Length-only rule. Source8 explicitly binds
/// the installed ordinary policy and accepts only its real terminal causes.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum LifecycleMode {
    LegacyLengthOnly,
    OriginalInstalledPlainText,
}
impl LifecycleMode {
    pub(super) fn accepts_policy(self, policy: HostCostPolicyV2) -> bool {
        match (self, policy.empirical_content_domain) {
            (_, Some(HostContentDomainV1::PlainTextGreedyV1)) => true,
            (
                Self::OriginalInstalledPlainText,
                Some(HostContentDomainV1::PlainTextInstalledV2(_)),
            ) => true,
            _ => false,
        }
    }
    fn terminal(
        self,
        policy: Option<HostCostPolicyV2>,
        value: &Terminal,
        generated: u64,
        maximum: u64,
    ) -> Result<(), CostProfileError> {
        if value.generated_tokens != generated || generated == 0 || generated > maximum {
            return Err(invalid(
                "original terminal frontier differs from request budget",
            ));
        }
        if self == Self::LegacyLengthOnly {
            if generated != maximum {
                return Err(invalid("legacy source requires complete Length budget"));
            }
            return observation::terminal(value);
        }
        let policy =
            policy.ok_or_else(|| invalid("source8 terminal lacks original installed policy"))?;
        if !self.accepts_policy(policy) {
            return Err(invalid("source8 unsupported original host policy"));
        }
        let allowed = match (&value.finish_reason, policy.empirical_content_domain) {
            (ferrum_types::FinishReason::Length, _) => generated == maximum,
            (
                ferrum_types::FinishReason::EOS,
                Some(HostContentDomainV1::PlainTextInstalledV2(cap)),
            ) => cap.model_eos,
            (
                ferrum_types::FinishReason::Stop,
                Some(HostContentDomainV1::PlainTextInstalledV2(cap)),
            ) => cap.user_stop,
            _ => false,
        };
        if !allowed {
            return Err(invalid(
                "source8 terminal cause differs from original installed policy",
            ));
        }
        observation::terminal_service(value)
    }
}
struct Slot {
    id: Option<String>,
    owner: Option<u64>,
    generated: u64,
    maximum: u64,
    completed: bool,
    policy: Option<HostCostPolicyV2>,
}
/// Original non-numerical prepared facts from a fully validated outside-route
/// settlement. This view carries no cost recipe or numerical sample authority.
pub(super) struct OriginalCohortRow<'a> {
    pub request_id: &'a str,
    pub owner_incarnation: u64,
    pub work_generation: u64,
    pub frontier: PreparedRowFactsV2,
    pub policy: HostCostPolicyV2,
    pub pending_decoded_utf8: bool,
}
struct Cohort {
    phase: usize,
    ordinal: usize,
    slots: Vec<Slot>,
    admitted: usize,
}
pub(super) struct Lifecycle {
    mode: LifecycleMode,
    plan: CohortPlanV2,
    ended: [usize; 3],
    active: Option<Cohort>,
    requests: Vec<String>,
    expected: VecDeque<CompletedRequest>,
}
impl Lifecycle {
    pub(super) fn not_submitted(
        &self,
        phase: usize,
        cohort: usize,
        participants: &[ferrum_interfaces::execution_cost::CallNoSubmissionParticipantV1],
    ) -> Result<(), CostProfileError> {
        self.active(phase, cohort)?;
        let c = self
            .active
            .as_ref()
            .ok_or_else(|| invalid("source8 no-submission outside cohort"))?;
        if participants.is_empty() {
            return Err(invalid("source8 no-submission missing participants"));
        }
        for p in participants {
            let id = p.request_id.to_string();
            if c.slots
                .iter()
                .find(|s| s.id.as_deref() == Some(id.as_str()))
                .is_none_or(|s| {
                    s.completed || s.owner.is_some_and(|owner| owner != p.owner_incarnation)
                })
            {
                return Err(invalid(
                    "source8 no-submission participant differs from active admitted cohort",
                ));
            }
        }
        Ok(())
    }
    pub(super) fn outside_completed(
        &mut self,
        phase: StructuredProfilePhaseV10,
        ordinal: usize,
        rows: &[OriginalCohortRow<'_>],
        s: &Stages,
        fifo: u64,
    ) -> Result<(), CostProfileError> {
        self.active(phase.index(), ordinal)?;
        let fail = || invalid("source8 outside settlement differs from original cohort");
        let c = self.active.as_mut().ok_or_else(fail)?;
        if rows.len() != s.rows.len() {
            return Err(fail());
        }
        for (before, row) in rows.iter().zip(&s.rows) {
            let index = c
                .slots
                .iter()
                .position(|v| v.id.as_deref() == Some(before.request_id))
                .ok_or_else(fail)?;
            let slot = &mut c.slots[index];
            if slot.completed
                || slot.generated != before.frontier.generated_before
                || slot.maximum != before.frontier.maximum_output
                || slot.owner.is_some_and(|o| o != before.owner_incarnation)
                || slot.policy.is_some_and(|p| p != before.policy)
                || !self.mode.accepts_policy(before.policy)
                || row.request_id != before.request_id
                || row.owner_incarnation != before.owner_incarnation
                || row.work_generation != before.work_generation
            {
                return Err(fail());
            }
            let next = slot
                .generated
                .checked_add(u64::from(
                    before.frontier.work.emits_token().map_err(numeric_error)?,
                ))
                .ok_or_else(fail)?;
            match &row.terminal {
                Some(t) => self
                    .mode
                    .terminal(Some(before.policy), t, next, slot.maximum)?,
                None if next < slot.maximum => {}
                _ => return Err(fail()),
            }
            slot.owner = Some(before.owner_incarnation);
            slot.policy = Some(before.policy);
            slot.generated = next;
            if let Some(terminal) = &row.terminal {
                slot.completed = true;
                self.expected.push_back(CompletedRequest {
                    phase,
                    cohort: ordinal,
                    slot: index,
                    request_id: row.request_id.clone(),
                    owner_incarnation: row.owner_incarnation,
                    call_id: s.call_id,
                    fifo,
                    generated_tokens: next,
                    terminal: terminal.clone(),
                });
            }
        }
        Ok(())
    }
    pub fn new(plan: CohortPlanV2) -> Self {
        Self::with_mode(plan, LifecycleMode::LegacyLengthOnly)
    }
    pub(super) fn with_mode(plan: CohortPlanV2, mode: LifecycleMode) -> Self {
        Self {
            mode,
            plan,
            ended: [0; 3],
            active: None,
            requests: Vec::new(),
            expected: VecDeque::new(),
        }
    }
    pub(super) fn retained_heap_bytes(&self) -> Option<usize> {
        let mut n = self
            .requests
            .capacity()
            .checked_mul(std::mem::size_of::<String>())?;
        for s in &self.requests {
            n = n.checked_add(s.capacity())?;
        }
        for phase in &self.plan.phases {
            n = n.checked_add(phase.capacity().checked_mul(std::mem::size_of::<
                crate::implementations::continuous::cost_model::structured_v2::windows::CohortV2,
            >())?)?;
            for c in phase {
                n = n.checked_add(c.requests.capacity().checked_mul(std::mem::size_of::<crate::implementations::continuous::cost_model::structured_v2::windows::CohortRequestV2>())?)?;
            }
        }
        if let Some(c) = &self.active {
            n = n.checked_add(
                c.slots
                    .capacity()
                    .checked_mul(std::mem::size_of::<Slot>())?,
            )?;
            for s in &c.slots {
                n = n.checked_add(s.id.as_ref().map_or(0, String::capacity))?;
            }
        }
        n = n.checked_add(
            self.expected
                .capacity()
                .checked_mul(std::mem::size_of::<CompletedRequest>())?,
        )?;
        for r in &self.expected {
            n = n.checked_add(r.request_id.capacity())?;
        }
        Some(n)
    }
    pub fn expects_completion(&self) -> bool {
        !self.expected.is_empty()
    }
    pub fn active(&self, phase: usize, ordinal: usize) -> Result<(), CostProfileError> {
        if self
            .active
            .as_ref()
            .is_none_or(|c| c.phase != phase || c.ordinal != ordinal)
            || !self.expected.is_empty()
        {
            return Err(invalid("V2 event outside its complete declared cohort"));
        }
        Ok(())
    }
    pub fn begin(
        &mut self,
        phase: usize,
        ordinal: usize,
        case: u32,
        repetition: u32,
    ) -> Result<(), CostProfileError> {
        let fail = || invalid("V2 cohort order differs from original plan");
        if phase >= 3
            || self.active.is_some()
            || !self.expected.is_empty()
            || self.ended[phase] != ordinal
            || (0..phase).any(|p| self.ended[p] != self.plan.phases[p].len())
        {
            return Err(fail());
        }
        let declared = self.plan.phases[phase].get(ordinal).ok_or_else(fail)?;
        if declared.manifest_case != case || declared.repetition != repetition {
            return Err(fail());
        }
        self.active = Some(Cohort {
            phase,
            ordinal,
            admitted: 0,
            slots: declared
                .requests
                .iter()
                .map(|r| Slot {
                    id: None,
                    owner: None,
                    generated: 0,
                    maximum: r.maximum_output,
                    completed: false,
                    policy: None,
                })
                .collect(),
        });
        Ok(())
    }
    pub fn admit(
        &mut self,
        phase: usize,
        cohort: usize,
        slot: usize,
        id: String,
        maximum: u64,
        limit: usize,
    ) -> Result<(), CostProfileError> {
        self.active(phase, cohort)?;
        let fail = || invalid("V2 request admission differs from immutable cohort slot");
        let c = self.active.as_mut().ok_or_else(fail)?;
        if slot != c.admitted
            || id.is_empty()
            || id.len() > limit
            || id.chars().any(char::is_control)
            || self.requests.binary_search(&id).is_ok()
        {
            return Err(fail());
        }
        let position = self.requests.binary_search(&id).unwrap_err();
        self.requests.insert(position, id.clone());
        let r = c.slots.get_mut(slot).ok_or_else(fail)?;
        if r.maximum != maximum {
            return Err(fail());
        }
        r.id = Some(id);
        c.admitted += 1;
        Ok(())
    }
    pub fn prepared(
        &mut self,
        phase: usize,
        ordinal: usize,
        p: &Prepared,
    ) -> Result<(), CostProfileError> {
        self.active(phase, ordinal)?;
        let fail = || invalid("V2 Prepared frontier differs from admitted request lifecycle");
        let c = self.active.as_mut().ok_or_else(fail)?;
        for r in &p.rows {
            let slot = c
                .slots
                .iter_mut()
                .find(|s| s.id.as_ref() == Some(&r.request_id))
                .ok_or_else(fail)?;
            if slot.completed
                || slot.generated != r.frontier.generated_before
                || slot.maximum != r.frontier.maximum_output
                || slot.owner.is_some_and(|v| v != r.owner_incarnation)
            {
                return Err(fail());
            }
            slot.owner = Some(r.owner_incarnation);
            if self.mode == LifecycleMode::OriginalInstalledPlainText {
                let policy = p
                    .recipe
                    .physical_host_rows
                    .get(r.frontier.physical_position as usize)
                    .ok_or_else(fail)?
                    .installed_policy;
                if !self.mode.accepts_policy(policy) || slot.policy.is_some_and(|old| old != policy)
                {
                    return Err(invalid("source8 admitted host policy changed"));
                }
                slot.policy = Some(policy);
            }
        }
        Ok(())
    }
    pub fn completed(
        &mut self,
        phase: StructuredProfilePhaseV10,
        ordinal: usize,
        p: &Prepared,
        s: &Stages,
        fifo: u64,
    ) -> Result<(), CostProfileError> {
        self.active(phase.index(), ordinal)?;
        let fail = || invalid("V2 original settlement does not advance the request frontier");
        let c = self.active.as_mut().ok_or_else(fail)?;
        for (before, row) in p.rows.iter().zip(&s.rows) {
            let index = c
                .slots
                .iter()
                .position(|slot| slot.id.as_ref() == Some(&before.request_id))
                .ok_or_else(fail)?;
            let slot = &mut c.slots[index];
            let next = slot
                .generated
                .checked_add(u64::from(
                    before.frontier.work.emits_token().map_err(numeric_error)?,
                ))
                .ok_or_else(fail)?;
            if slot.completed
                || slot.generated != before.frontier.generated_before
                || slot.owner != Some(row.owner_incarnation)
                || row.request_id != before.request_id
                || next > slot.maximum
            {
                return Err(fail());
            }
            match &row.terminal {
                Some(t) => self.mode.terminal(slot.policy, t, next, slot.maximum)?,
                None if next < slot.maximum => {}
                _ => return Err(fail()),
            }
            slot.generated = next;
            if let Some(terminal) = &row.terminal {
                slot.completed = true;
                self.expected.push_back(CompletedRequest {
                    phase,
                    cohort: ordinal,
                    slot: index,
                    request_id: row.request_id.clone(),
                    owner_incarnation: row.owner_incarnation,
                    call_id: s.call_id,
                    fifo,
                    generated_tokens: next,
                    terminal: terminal.clone(),
                });
            }
        }
        Ok(())
    }
    /// Advance only already validated real preparation settlements. This does
    /// not manufacture Prepared numeric evidence or a terminal receipt.
    pub fn preparation_row(
        &mut self,
        phase: usize,
        cohort: usize,
        id: &str,
        owner: u64,
        before: u64,
        after: u64,
    ) -> Result<(), CostProfileError> {
        self.active(phase, cohort)?;
        let slot = self
            .active
            .as_mut()
            .and_then(|c| c.slots.iter_mut().find(|s| s.id.as_deref() == Some(id)))
            .ok_or_else(|| invalid("source5 preparation owner was not admitted"))?;
        if slot.completed
            || slot.generated != before
            || slot.owner.is_some_and(|v| v != owner)
            || owner == 0
            || after < before
            || after > before.saturating_add(1)
            || after >= slot.maximum
        {
            return Err(invalid(
                "source5 preparation cannot shorten original Length lifecycle",
            ));
        }
        slot.generated = after;
        slot.owner = Some(owner);
        Ok(())
    }
    pub fn request_completed(&mut self, request: CompletedRequest) -> Result<(), CostProfileError> {
        if self.expected.pop_front().as_ref() != Some(&request) {
            return Err(invalid(
                "V2 request_completed lacks matching original terminal settlement",
            ));
        }
        Ok(())
    }
    pub fn end(
        &mut self,
        phase: usize,
        cohort: usize,
        admitted: usize,
        completed: usize,
    ) -> Result<(), CostProfileError> {
        self.active(phase, cohort)?;
        let c = self.active.as_ref().unwrap();
        if admitted != c.slots.len()
            || completed != c.slots.len()
            || c.admitted != c.slots.len()
            || c.slots.iter().any(|s| {
                !s.completed
                    || s.generated > s.maximum
                    || (self.mode == LifecycleMode::LegacyLengthOnly && s.generated != s.maximum)
            })
        {
            return Err(invalid("V2 cohort omitted incomplete requests"));
        }
        self.active = None;
        self.ended[phase] += 1;
        Ok(())
    }
    pub fn freeze(&self, phase: usize) -> Result<(), CostProfileError> {
        if phase >= 3
            || self.active.is_some()
            || !self.expected.is_empty()
            || self.ended[phase] != self.plan.phases[phase].len()
        {
            return Err(invalid("V2 phase lacks all complete declared cohorts"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod source8_tests;
