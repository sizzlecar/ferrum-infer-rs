//! Full request lifecycle, independent from which Prepared waves hit windows.
//! Only the live receipt adapter feeds successful settlements into this ledger.
use super::*;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::windows::CohortPlanV2;

struct RequestSlot {
    request_id: Option<RequestId>,
    owner: Option<u64>,
    generated: u64,
    maximum: u64,
    completed: bool,
}
struct ActiveCohort {
    phase: usize,
    ordinal: usize,
    slots: Vec<RequestSlot>,
    admitted: usize,
}
pub(super) struct CohortLedgerV2 {
    plan: CohortPlanV2,
    ended: [usize; 3],
    seen_requests: std::collections::HashSet<RequestId>,
    active: Option<ActiveCohort>,
}
#[derive(Serialize)]
pub(super) struct CompletedRequestV2 {
    pub phase: StructuredCapturePhase,
    pub cohort: usize,
    pub slot: usize,
    pub request_id: RequestId,
    pub owner_incarnation: u64,
    pub call_id: u64,
    pub fifo: u64,
    pub generated_tokens: u64,
    pub terminal: HostTerminalStageV1,
}
fn phase_index(phase: StructuredCapturePhase) -> Result<usize, ExportError> {
    match phase {
        StructuredCapturePhase::Fit => Ok(0),
        StructuredCapturePhase::Residual => Ok(1),
        StructuredCapturePhase::Qualification => Ok(2),
        _ => Err(ExportError::Source("cohort outside a collecting phase")),
    }
}
impl CohortLedgerV2 {
    pub fn new(plan: CohortPlanV2) -> Result<Self, ExportError> {
        plan.validate().map_err(numeric_error)?;
        Ok(Self {
            plan,
            ended: [0; 3],
            seen_requests: Default::default(),
            active: None,
        })
    }
    pub fn begin(
        &mut self,
        phase: StructuredCapturePhase,
        ordinal: usize,
    ) -> Result<(), ExportError> {
        let index = phase_index(phase)?;
        if self.active.is_some()
            || ordinal != self.ended[index]
            || (0..index).any(|p| self.ended[p] != self.plan.phases[p].len())
        {
            return Err(ExportError::Source(
                "cohort order or previous phase is incomplete",
            ));
        }
        let declared = self.plan.phases[index]
            .get(ordinal)
            .ok_or(ExportError::Source("undeclared cohort"))?;
        self.active = Some(ActiveCohort {
            phase: index,
            ordinal,
            admitted: 0,
            slots: declared
                .requests
                .iter()
                .map(|r| RequestSlot {
                    request_id: None,
                    owner: None,
                    generated: 0,
                    maximum: r.maximum_output,
                    completed: false,
                })
                .collect(),
        });
        Ok(())
    }
    pub fn active_ordinal(&self) -> Result<usize, ExportError> {
        self.active
            .as_ref()
            .map(|c| c.ordinal)
            .ok_or(ExportError::Source("no declared cohort active"))
    }
    pub fn admit(&mut self, id: RequestId, maximum: u64) -> Result<usize, ExportError> {
        let active = self
            .active
            .as_mut()
            .ok_or(ExportError::Source("request outside declared cohort"))?;
        let ordinal = active.admitted;
        let slot = active
            .slots
            .get_mut(ordinal)
            .ok_or(ExportError::Source("too many requests in cohort"))?;
        if maximum != slot.maximum || self.seen_requests.contains(&id) {
            return Err(ExportError::Source(
                "request identity or budget differs from immutable slot",
            ));
        }
        self.seen_requests.insert(id.clone());
        slot.request_id = Some(id);
        active.admitted += 1;
        Ok(ordinal)
    }
    pub fn prepared(&mut self, facts: &PreparedStructuredFactsV2) -> Result<(), ExportError> {
        let active = self
            .active
            .as_mut()
            .ok_or(ExportError::Source("Prepared outside declared cohort"))?;
        // Validate every row before attaching its initial owner to a slot.
        for row in &facts.rows {
            let slot = active
                .slots
                .iter()
                .find(|s| s.request_id.as_ref() == Some(&row.request_id))
                .ok_or(ExportError::Source(
                    "Prepared request was never admitted in this cohort",
                ))?;
            if slot.completed
                || slot.generated != row.frontier.generated_before
                || slot.maximum != row.frontier.maximum_output
                || slot
                    .owner
                    .is_some_and(|owner| owner != row.owner_incarnation)
            {
                return Err(ExportError::Source(
                    "Prepared request frontier differs from original lifecycle",
                ));
            }
        }
        for row in &facts.rows {
            let slot = active
                .slots
                .iter_mut()
                .find(|s| s.request_id.as_ref() == Some(&row.request_id))
                .unwrap();
            slot.owner = Some(row.owner_incarnation);
        }
        Ok(())
    }
    pub fn completed(
        &mut self,
        phase: StructuredCapturePhase,
        prepared: &PreparedStructuredFactsV2,
        actual: &super::super::super::trainer::structured_v2::ValidatedStructuredWaveV2,
    ) -> Result<Vec<CompletedRequestV2>, ExportError> {
        let active = self
            .active
            .as_mut()
            .ok_or(ExportError::Source("settlement outside declared cohort"))?;
        if phase_index(phase)? != active.phase || actual.stages.rows.len() != prepared.rows.len() {
            return Err(ExportError::Source(
                "settlement cohort or row count changed",
            ));
        }
        let mut updates = Vec::with_capacity(prepared.rows.len());
        for (before, row) in prepared.rows.iter().zip(&actual.stages.rows) {
            let index = active
                .slots
                .iter()
                .position(|s| s.request_id.as_ref() == Some(&before.request_id))
                .ok_or(ExportError::Source("settlement request was never admitted"))?;
            let slot = &active.slots[index];
            let next = slot
                .generated
                .checked_add(u64::from(
                    before.frontier.work.emits_token().map_err(numeric_error)?,
                ))
                .ok_or(ExportError::Source("request generated frontier overflow"))?;
            if slot.completed
                || slot.generated != before.frontier.generated_before
                || slot.owner != Some(row.owner_incarnation)
                || row.request_id != before.request_id
                || next > slot.maximum
            {
                return Err(ExportError::Source(
                    "settlement does not advance the original request frontier",
                ));
            }
            match &row.terminal {
                Some(terminal)
                    if terminal.finish_reason == ferrum_types::FinishReason::Length
                        && terminal.generated_tokens == next
                        && next == slot.maximum => {}
                None if next < slot.maximum => {}
                _ => {
                    return Err(ExportError::Source(
                        "request did not complete its declared Length policy",
                    ))
                }
            }
            updates.push((index, next, row));
        }
        let mut completed = Vec::new();
        for (index, next, row) in updates {
            let slot = &mut active.slots[index];
            slot.generated = next;
            if let Some(terminal) = &row.terminal {
                slot.completed = true;
                completed.push(CompletedRequestV2 {
                    phase,
                    cohort: active.ordinal,
                    slot: index,
                    request_id: row.request_id.clone(),
                    owner_incarnation: row.owner_incarnation,
                    call_id: actual.actual.call_id,
                    fifo: actual.ordinal(),
                    generated_tokens: next,
                    terminal: terminal.clone(),
                });
            }
        }
        Ok(completed)
    }
    pub fn end(&mut self, phase: StructuredCapturePhase) -> Result<(usize, usize), ExportError> {
        let index = phase_index(phase)?;
        let active = self
            .active
            .as_ref()
            .ok_or(ExportError::Source("no cohort to finish"))?;
        if active.phase != index
            || active.admitted != active.slots.len()
            || active
                .slots
                .iter()
                .any(|slot| !slot.completed || slot.generated != slot.maximum)
        {
            return Err(ExportError::Source(
                "cohort still has unadmitted or unfinished requests",
            ));
        }
        let result = (active.ordinal, active.slots.len());
        self.ended[index] += 1;
        self.active = None;
        Ok(result)
    }
    pub fn freeze(&self, phase: StructuredCapturePhase) -> Result<(), ExportError> {
        let index = phase_index(phase)?;
        if self.active.is_some() || self.ended[index] != self.plan.phases[index].len() {
            return Err(ExportError::Source(
                "phase does not contain all declared complete cohorts",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::windows::{
        CohortRequestV2, CohortV2,
    };
    fn plan() -> CohortPlanV2 {
        CohortPlanV2 {
            phases: std::array::from_fn(|_| {
                vec![CohortV2 {
                    manifest_case: 0,
                    repetition: 0,
                    requests: vec![
                        CohortRequestV2 {
                            manifest_prompt: 0,
                            maximum_output: 3,
                        },
                        CohortRequestV2 {
                            manifest_prompt: 0,
                            maximum_output: 3,
                        },
                    ],
                }]
            }),
        }
    }
    #[test]
    fn v2_full_cohort_requires_real_completion_even_after_every_slot_is_admitted() {
        let mut ledger = CohortLedgerV2::new(plan()).unwrap();
        assert!(ledger.begin(StructuredCapturePhase::Residual, 0).is_err());
        ledger.begin(StructuredCapturePhase::Fit, 0).unwrap();
        let first = RequestId::new();
        assert!(ledger.admit(first.clone(), 2).is_err());
        assert_eq!(ledger.admit(first.clone(), 3).unwrap(), 0);
        assert!(ledger.admit(first, 3).is_err());
        assert_eq!(ledger.admit(RequestId::new(), 3).unwrap(), 1);
        assert!(ledger.admit(RequestId::new(), 3).is_err());
        // Admission and counters alone cannot manufacture terminal receipts.
        assert!(ledger.end(StructuredCapturePhase::Fit).is_err());
        assert!(ledger.freeze(StructuredCapturePhase::Fit).is_err());
        assert!(ledger.begin(StructuredCapturePhase::Residual, 0).is_err());
    }
}
