//! One immutable scope and one outstanding physical call. A failed offered
//! prepared member consumes its slot permanently; receipt conversion never selects it.
use super::*;

pub(super) struct OfferedWave {
    pub offered: u64,
    pub member_candidate: bool,
    pub phase: StructuredCapturePhase,
}
pub(super) struct ReservedWave {
    pub offered: u64,
    pub member: Option<u64>,
    pub phase: StructuredCapturePhase,
    capture: Arc<CostCalibrationCapture>,
}

pub(super) struct PopulationLedger {
    rows: usize,
    required: [usize; 3],
    maximum_offered: usize,
    offered: u64,
    members: u64,
    reserved_by_phase: [usize; 3],
    completed_by_phase: [usize; 3],
    failed_by_phase: [usize; 3],
    last_fifo: u64,
    incomplete_fifo: bool,
    pending: Option<ReservedWave>,
    pending_attempt: Option<OfferedWave>,
}

fn phase_index(phase: StructuredCapturePhase) -> Result<usize, StructuredUnknown> {
    match phase {
        StructuredCapturePhase::Fit => Ok(0),
        StructuredCapturePhase::Residual => Ok(1),
        StructuredCapturePhase::Qualification => Ok(2),
        _ => Err(StructuredUnknown::PhaseLeakage),
    }
}

impl PopulationLedger {
    pub fn new(options: &StructuredCalibrationOptions, initial_fifo_cutoff: u64) -> Self {
        Self {
            rows: options.scope.rows.get(),
            required: options.counts(),
            maximum_offered: options.maximum_offered_waves.get(),
            offered: 0,
            members: 0,
            reserved_by_phase: [0; 3],
            completed_by_phase: [0; 3],
            failed_by_phase: [0; 3],
            last_fifo: initial_fifo_cutoff,
            incomplete_fifo: false,
            pending: None,
            pending_attempt: None,
        }
    }

    pub fn offer(
        &mut self,
        work: &[CalibrationWork],
        phase: StructuredCapturePhase,
    ) -> Result<&OfferedWave, ExportError> {
        let ordinary = work.len() == self.rows
            && work.iter().all(|row| {
                matches!(row.work, ActualRowWork::Decode { .. })
                    && row.frontier.generated_tokens() > 0
            });
        self.offer_classified(ordinary, phase)
    }
    fn offer_classified(
        &mut self,
        member_candidate: bool,
        phase: StructuredCapturePhase,
    ) -> Result<&OfferedWave, ExportError> {
        if self.pending.is_some() || self.pending_attempt.is_some() {
            return Err(ExportError::Source(
                "structured attempt or reservation already pending",
            ));
        }
        phase_index(phase).map_err(numeric_error)?;
        if self.offered >= self.maximum_offered as u64 {
            return Err(ExportError::Source(
                "structured offered attempt bound reached",
            ));
        }
        self.offered += 1;
        self.pending_attempt = Some(OfferedWave {
            offered: self.offered,
            member_candidate,
            phase,
        });
        Ok(self.pending_attempt.as_ref().unwrap())
    }
    pub fn reserve_prepared(
        &mut self,
        binding: Arc<StructuredCaptureSessionBinding>,
    ) -> Result<&ReservedWave, ExportError> {
        let attempt = self.pending_attempt.as_ref().ok_or(ExportError::Source(
            "structured prepared wave has no original attempt",
        ))?;
        let index = phase_index(attempt.phase).map_err(numeric_error)?;
        if attempt.member_candidate && self.reserved_by_phase[index] >= self.required[index] {
            return Err(ExportError::Source(
                "structured declared member population bound reached",
            ));
        }
        let attempt = self.pending_attempt.take().unwrap();
        let member = if attempt.member_candidate {
            self.members += 1;
            self.reserved_by_phase[index] += 1;
            Some(self.members)
        } else {
            None
        };
        self.pending = Some(ReservedWave {
            offered: attempt.offered,
            member,
            phase: attempt.phase,
            capture: Arc::new(CostCalibrationCapture::for_structured_session(binding)),
        });
        Ok(self.pending.as_ref().unwrap())
    }
    pub fn take_unprepared_attempt(&mut self) -> Option<OfferedWave> {
        self.pending_attempt.take()
    }
    #[cfg(test)]
    fn reserve_classified(
        &mut self,
        member: bool,
        phase: StructuredCapturePhase,
        binding: Arc<StructuredCaptureSessionBinding>,
    ) -> Result<&ReservedWave, ExportError> {
        self.offer_classified(member, phase)?;
        self.reserve_prepared(binding)
    }

    pub fn pending_capture(&self) -> Result<Arc<CostCalibrationCapture>, ExportError> {
        self.pending
            .as_ref()
            .map(|p| Arc::clone(&p.capture))
            .ok_or(ExportError::Source("no structured reservation is pending"))
    }

    pub fn take_pending(
        &mut self,
        capture: Option<&Arc<CostCalibrationCapture>>,
    ) -> Result<ReservedWave, ExportError> {
        let pending = self.pending.as_ref().ok_or(ExportError::Source(
            "structured reservation already settled or absent",
        ))?;
        if capture.is_some_and(|capture| !Arc::ptr_eq(capture, &pending.capture)) {
            return Err(ExportError::Source(
                "another call cannot settle this structured reservation",
            ));
        }
        Ok(self.pending.take().unwrap())
    }

    pub fn accept_fifo(
        &mut self,
        queue: Option<HostStageQueueReceipt>,
    ) -> Result<(), StructuredUnknown> {
        let ordinal = match queue {
            Some(HostStageQueueReceipt {
                disposition: HostStageQueueDisposition::Published,
                accepted_ordinal: Some(n),
            }) => n,
            _ => {
                self.incomplete_fifo = true;
                return Err(StructuredUnknown::WrongSource);
            }
        };
        let Some(expected) = self.last_fifo.checked_add(1) else {
            self.incomplete_fifo = true;
            return Err(StructuredUnknown::Capacity);
        };
        if ordinal != expected {
            self.incomplete_fifo = true;
            return Err(StructuredUnknown::IncompletePhasePopulation);
        }
        self.last_fifo = ordinal;
        Ok(())
    }

    pub fn complete_member(&mut self, reserved: &ReservedWave) {
        if reserved.member.is_some() {
            self.completed_by_phase
                [phase_index(reserved.phase).expect("reserved collecting phase")] += 1;
        }
    }
    pub fn fail_member(&mut self, reserved: &ReservedWave) {
        if reserved.member.is_some() {
            self.failed_by_phase
                [phase_index(reserved.phase).expect("reserved collecting phase")] += 1;
        }
    }
    pub fn freeze(
        &self,
        phase: StructuredCapturePhase,
        cutoff: u64,
        samples: usize,
    ) -> Result<(), StructuredUnknown> {
        let index = phase_index(phase)?;
        if self.pending.is_some()
            || self.pending_attempt.is_some()
            || self.incomplete_fifo
            || cutoff != self.last_fifo
            || self.reserved_by_phase[index] != self.required[index]
            || self.completed_by_phase[index] != self.required[index]
            || self.failed_by_phase[index] != 0
            || samples != self.required[index]
        {
            return Err(StructuredUnknown::IncompletePhasePopulation);
        }
        Ok(())
    }
    pub fn abandon_pending(&mut self) {
        if let Some(reserved) = self.pending.take() {
            self.fail_member(&reserved);
        }
        self.pending_attempt.take();
    }
    pub fn audit_complete(&self, cutoff: u64) -> bool {
        self.pending.is_none()
            && self.pending_attempt.is_none()
            && !self.incomplete_fifo
            && self.last_fifo == cutoff
    }
    pub fn phase_counts(&self) -> ([usize; 3], [usize; 3], [usize; 3]) {
        (
            self.reserved_by_phase,
            self.completed_by_phase,
            self.failed_by_phase,
        )
    }
    pub fn offered(&self) -> u64 {
        self.offered
    }
    pub fn members(&self) -> u64 {
        self.members
    }
    pub fn failures(&self) -> u64 {
        self.failed_by_phase.iter().sum::<usize>() as u64
    }
    pub fn last_fifo(&self) -> u64 {
        self.last_fifo
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    struct ZeroClock;
    impl CostObservationClock for ZeroClock {
        fn now_ns(&self) -> Option<u64> {
            Some(0)
        }
    }

    fn binding() -> Arc<StructuredCaptureSessionBinding> {
        Arc::new(
            StructuredCaptureSessionBinding::new(
                [8; 32],
                model::ExecutionFingerprint {
                    model_weights: [1; 32],
                    numerical_policy: [2; 32],
                    device_runtime: [3; 32],
                    execution_config: [4; 32],
                },
                &ZeroClock,
            )
            .unwrap(),
        )
    }
    fn ledger() -> PopulationLedger {
        PopulationLedger {
            rows: 8,
            required: [2, 2, 2],
            maximum_offered: 12,
            offered: 0,
            members: 0,
            reserved_by_phase: [0; 3],
            completed_by_phase: [0; 3],
            failed_by_phase: [0; 3],
            last_fifo: 10,
            incomplete_fifo: false,
            pending: None,
            pending_attempt: None,
        }
    }
    fn publish(ledger: &mut PopulationLedger, ordinal: u64) -> ReservedWave {
        let capture = ledger.pending_capture().unwrap();
        let reserved = ledger.take_pending(Some(&capture)).unwrap();
        ledger
            .accept_fifo(Some(HostStageQueueReceipt {
                disposition: HostStageQueueDisposition::Published,
                accepted_ordinal: Some(ordinal),
            }))
            .unwrap();
        reserved
    }
    #[test]
    fn unprepared_resource_turnaround_is_audited_without_consuming_member_slots() {
        let mut ledger = ledger();
        ledger
            .offer_classified(true, StructuredCapturePhase::Fit)
            .unwrap();
        let attempt = ledger.take_unprepared_attempt().unwrap();
        assert!(attempt.member_candidate);
        assert_eq!(ledger.members(), 0);
        assert_eq!(ledger.failures(), 0);
        ledger
            .reserve_classified(true, StructuredCapturePhase::Fit, binding())
            .unwrap();
        let member = publish(&mut ledger, 11);
        assert_eq!(member.offered, 2);
        assert_eq!(member.member, Some(1));
        ledger.fail_member(&member);
        assert_eq!(ledger.failures(), 1);
    }
    #[test]
    fn reserved_members_keep_original_interleaved_fifo_and_failed_slots() {
        let mut ledger = ledger();
        let binding = binding();
        ledger
            .reserve_classified(false, StructuredCapturePhase::Fit, Arc::clone(&binding))
            .unwrap();
        let outside = publish(&mut ledger, 11);
        assert_eq!(outside.member, None);
        ledger
            .reserve_classified(true, StructuredCapturePhase::Fit, Arc::clone(&binding))
            .unwrap();
        let first = publish(&mut ledger, 12);
        assert_eq!(first.member, Some(1));
        ledger.fail_member(&first);
        ledger
            .reserve_classified(false, StructuredCapturePhase::Fit, Arc::clone(&binding))
            .unwrap();
        publish(&mut ledger, 13);
        ledger
            .reserve_classified(true, StructuredCapturePhase::Fit, binding)
            .unwrap();
        let second = publish(&mut ledger, 14);
        assert_eq!(second.member, Some(2));
        ledger.complete_member(&second);
        assert_eq!(ledger.offered(), 4);
        assert_eq!(ledger.last_fifo(), 14);
        assert_eq!(ledger.failures(), 1);
        assert_eq!(
            ledger.freeze(StructuredCapturePhase::Fit, 14, 1),
            Err(StructuredUnknown::IncompletePhasePopulation)
        );
    }
    #[test]
    fn reservation_is_single_use_and_cannot_be_settled_by_another_call() {
        let mut ledger = ledger();
        let binding = binding();
        ledger
            .reserve_classified(true, StructuredCapturePhase::Fit, Arc::clone(&binding))
            .unwrap();
        assert!(ledger
            .reserve_classified(true, StructuredCapturePhase::Fit, Arc::clone(&binding))
            .is_err());
        let foreign = Arc::new(CostCalibrationCapture::for_structured_session(binding));
        assert!(ledger.take_pending(Some(&foreign)).is_err());
        let original = ledger.pending_capture().unwrap();
        ledger.take_pending(Some(&original)).unwrap();
        assert!(ledger.take_pending(Some(&original)).is_err());
    }
    #[test]
    fn global_fifo_gap_or_drop_prevents_scope_qualification() {
        for queue in [
            None,
            Some(HostStageQueueReceipt {
                disposition: HostStageQueueDisposition::Published,
                accepted_ordinal: Some(12),
            }),
        ] {
            let mut ledger = ledger();
            assert!(ledger.accept_fifo(queue).is_err());
            ledger.reserved_by_phase[0] = 2;
            ledger.completed_by_phase[0] = 2;
            assert!(ledger.freeze(StructuredCapturePhase::Fit, 10, 2).is_err());
        }
    }
}
