//! A reservation is decided only from the original Prepared facts. A failed
//! reservation remains in the population; FIFO order includes every outside call.
use super::*;

pub(super) struct OfferedWaveV2 {
    pub offered: u64,
    pub phase: StructuredCapturePhase,
    pub cohort: usize,
}
pub(super) struct ReservedWaveV2 {
    pub attempt: OfferedWaveV2,
    pub member: Option<u64>,
    pub window: Option<u32>,
    pub prepared: Arc<PreparedStructuredFactsV2>,
    pub capture: Arc<CostCalibrationCapture>,
}
pub(super) struct PopulationLedgerV2 {
    pub offered: u64,
    pub members: u64,
    pub reserved: [usize; 3],
    pub completed: [usize; 3],
    pub failed: [usize; 3],
    pub last_fifo: u64,
    pub incomplete_fifo: bool,
    pub attempt: Option<OfferedWaveV2>,
    pub pending: Option<ReservedWaveV2>,
}
pub(super) fn phase_index(phase: StructuredCapturePhase) -> Result<usize, ExportError> {
    match phase {
        StructuredCapturePhase::Fit => Ok(0),
        StructuredCapturePhase::Residual => Ok(1),
        StructuredCapturePhase::Qualification => Ok(2),
        _ => Err(ExportError::Source(
            "V2 population outside collecting phase",
        )),
    }
}
pub(super) fn numeric_phase(
    phase: StructuredCapturePhase,
) -> Result<StructuredPhaseV2, ExportError> {
    match phase_index(phase)? {
        0 => Ok(StructuredPhaseV2::Fit),
        1 => Ok(StructuredPhaseV2::Residual),
        _ => Ok(StructuredPhaseV2::Qualification),
    }
}
impl PopulationLedgerV2 {
    pub fn new(cutoff: u64) -> Self {
        Self {
            offered: 0,
            members: 0,
            reserved: [0; 3],
            completed: [0; 3],
            failed: [0; 3],
            last_fifo: cutoff,
            incomplete_fifo: false,
            attempt: None,
            pending: None,
        }
    }
    pub fn offer(
        &mut self,
        phase: StructuredCapturePhase,
        cohort: usize,
        limit: usize,
    ) -> Result<&OfferedWaveV2, ExportError> {
        phase_index(phase)?;
        if self.attempt.is_some() || self.pending.is_some() || self.offered >= limit as u64 {
            return Err(ExportError::Source(
                "V2 offered bound or outstanding attempt",
            ));
        }
        self.offered += 1;
        self.attempt = Some(OfferedWaveV2 {
            offered: self.offered,
            phase,
            cohort,
        });
        Ok(self.attempt.as_ref().unwrap())
    }
    pub fn reserve(
        &mut self,
        prepared: PreparedStructuredFactsV2,
        rule: &MembershipRuleV2,
        counts: [usize; 3],
        binding: Arc<StructuredCaptureSessionBinding>,
    ) -> Result<&ReservedWaveV2, ExportError> {
        let owner = prepared.validate().map_err(numeric_error)?;
        self.reserve_shared(
            Arc::new(prepared),
            &owner,
            rule,
            counts,
            Arc::new(CostCalibrationCapture::for_structured_session(binding)),
        )
    }
    pub fn reserve_shared(
        &mut self,
        prepared: Arc<PreparedStructuredFactsV2>,
        owner: &StructuredOwnerKeyV2,
        rule: &MembershipRuleV2,
        counts: [usize; 3],
        capture: Arc<CostCalibrationCapture>,
    ) -> Result<&ReservedWaveV2, ExportError> {
        let rows = prepared
            .rows
            .iter()
            .map(|r| r.frontier.clone())
            .collect::<Vec<_>>();
        let window = rule.classify(owner, &rows).map_err(numeric_error)?;
        let attempt = self
            .attempt
            .as_ref()
            .ok_or(ExportError::Source("V2 reservation has no original offer"))?;
        let phase = phase_index(attempt.phase)?;
        if window.is_some() && self.reserved[phase] >= counts[phase] {
            return Err(ExportError::Source(
                "V2 declared member population exceeded; source cannot stop at a sample count",
            ));
        }
        let member = if window.is_some() {
            self.members += 1;
            self.reserved[phase] += 1;
            Some(self.members)
        } else {
            None
        };
        self.pending = Some(ReservedWaveV2 {
            attempt: self.attempt.take().unwrap(),
            member,
            window,
            prepared,
            capture,
        });
        Ok(self.pending.as_ref().unwrap())
    }
    pub fn take_pending(
        &mut self,
        capture: Option<&Arc<CostCalibrationCapture>>,
    ) -> Result<ReservedWaveV2, ExportError> {
        let pending = self
            .pending
            .as_ref()
            .ok_or(ExportError::Source("no original V2 reservation to settle"))?;
        if capture.is_some_and(|c| !Arc::ptr_eq(c, &pending.capture)) {
            return Err(ExportError::Source(
                "another call cannot settle this V2 reservation",
            ));
        }
        Ok(self.pending.take().unwrap())
    }
    pub fn accept_fifo(&mut self, queue: Option<HostStageQueueReceipt>) -> Result<(), ExportError> {
        let ordinal = match queue {
            Some(HostStageQueueReceipt {
                disposition: HostStageQueueDisposition::Published,
                accepted_ordinal: Some(n),
            }) => n,
            _ => {
                self.incomplete_fifo = true;
                return Err(ExportError::Source(
                    "V2 original host FIFO receipt unavailable",
                ));
            }
        };
        if self.last_fifo.checked_add(1) != Some(ordinal) {
            self.incomplete_fifo = true;
            return Err(ExportError::Source(
                "V2 original FIFO has a gap or reordered entry",
            ));
        }
        self.last_fifo = ordinal;
        Ok(())
    }
    pub fn settle_member(&mut self, reserved: &ReservedWaveV2, success: bool) {
        if reserved.member.is_some() {
            let phase = phase_index(reserved.attempt.phase).expect("reserved collecting phase");
            if success {
                self.completed[phase] += 1;
            } else {
                self.failed[phase] += 1;
            }
        }
    }
    pub fn freeze(
        &self,
        phase: StructuredCapturePhase,
        cutoff: u64,
        samples: usize,
        required: [usize; 3],
    ) -> Result<(), ExportError> {
        let index = phase_index(phase)?;
        if !self.audit_complete(cutoff)
            || self.reserved[index] != required[index]
            || self.completed[index] != required[index]
            || self.failed[index] != 0
            || samples != required[index]
        {
            return Err(ExportError::Source("V2 phase population is incomplete"));
        }
        Ok(())
    }
    pub fn audit_complete(&self, cutoff: u64) -> bool {
        !self.incomplete_fifo
            && self.last_fifo == cutoff
            && self.attempt.is_none()
            && self.pending.is_none()
    }
    pub fn abandon(&mut self) {
        if let Some(pending) = self.pending.take() {
            self.settle_member(&pending, false);
        }
        self.attempt.take();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fifo(n: u64) -> Option<HostStageQueueReceipt> {
        Some(HostStageQueueReceipt {
            disposition: HostStageQueueDisposition::Published,
            accepted_ordinal: Some(n),
        })
    }
    #[test]
    fn v2_fifo_requires_every_outside_call_and_never_repairs_a_gap() {
        let mut ledger = PopulationLedgerV2::new(40);
        ledger.accept_fifo(fifo(41)).unwrap();
        assert!(ledger.accept_fifo(fifo(43)).is_err());
        // Later receiving the omitted entry cannot undo the original violation.
        ledger.accept_fifo(fifo(42)).unwrap();
        assert!(!ledger.audit_complete(42));
        assert!(ledger
            .freeze(StructuredCapturePhase::Fit, 42, 0, [0; 3])
            .is_err());
        let mut dense = PopulationLedgerV2::new(40);
        dense.accept_fifo(fifo(41)).unwrap();
        dense.accept_fifo(fifo(42)).unwrap();
        assert!(dense.audit_complete(42));
        assert!(!dense.audit_complete(43));
    }
    #[test]
    fn v2_unprepared_attempt_consumes_offer_but_no_member_and_retry_is_distinct() {
        let mut ledger = PopulationLedgerV2::new(0);
        assert_eq!(
            ledger
                .offer(StructuredCapturePhase::Fit, 0, 3)
                .unwrap()
                .offered,
            1
        );
        assert!(ledger.offer(StructuredCapturePhase::Fit, 0, 3).is_err());
        assert!(ledger
            .freeze(StructuredCapturePhase::Fit, 0, 0, [0; 3])
            .is_err());
        let old = ledger.attempt.take().unwrap();
        assert_eq!(old.offered, 1);
        assert_eq!(ledger.members, 0);
        assert_eq!(
            ledger
                .offer(StructuredCapturePhase::Fit, 0, 3)
                .unwrap()
                .offered,
            2
        );
        ledger.attempt.take();
        assert_eq!(ledger.members, 0);
        assert!(ledger.audit_complete(0));
        ledger.offer(StructuredCapturePhase::Fit, 0, 3).unwrap();
        ledger.attempt.take();
        assert!(ledger.offer(StructuredCapturePhase::Fit, 0, 3).is_err());
    }
    #[test]
    fn v2_population_cannot_freeze_failed_or_extra_members() {
        let mut ledger = PopulationLedgerV2::new(7);
        ledger.reserved[0] = 8;
        ledger.completed[0] = 7;
        ledger.failed[0] = 1;
        assert!(ledger
            .freeze(StructuredCapturePhase::Fit, 7, 7, [8; 3])
            .is_err());
        ledger.completed[0] = 8;
        assert!(ledger
            .freeze(StructuredCapturePhase::Fit, 7, 8, [8; 3])
            .is_err());
        ledger.failed[0] = 0;
        ledger
            .freeze(StructuredCapturePhase::Fit, 7, 8, [8; 3])
            .unwrap();
        ledger.reserved[0] = 9;
        assert!(ledger
            .freeze(StructuredCapturePhase::Fit, 7, 8, [8; 3])
            .is_err());
    }
}
