//! Every request starts at zero and completes its original immutable budget.
//! Outside-window settlements are still necessary lifecycle/FIFO evidence.
use super::*;
use std::collections::VecDeque;
struct Slot {
    id: Option<String>,
    owner: Option<u64>,
    generated: u64,
    maximum: u64,
    completed: bool,
}
struct Cohort {
    phase: usize,
    ordinal: usize,
    slots: Vec<Slot>,
    admitted: usize,
}
pub(super) struct Lifecycle {
    plan: CohortPlanV2,
    ended: [usize; 3],
    active: Option<Cohort>,
    requests: HashSet<String>,
    expected: VecDeque<CompletedRequest>,
}
impl Lifecycle {
    pub fn new(plan: CohortPlanV2) -> Self {
        Self {
            plan,
            ended: [0; 3],
            active: None,
            requests: HashSet::new(),
            expected: VecDeque::new(),
        }
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
            || !self.requests.insert(id.clone())
        {
            return Err(fail());
        }
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
                Some(t) if t.generated_tokens == next && next == slot.maximum => {
                    observation::terminal(t)?;
                }
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
            || c.slots
                .iter()
                .any(|s| !s.completed || s.generated != s.maximum)
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
