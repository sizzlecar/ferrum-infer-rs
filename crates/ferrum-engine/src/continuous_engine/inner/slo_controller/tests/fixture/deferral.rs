//! Controlled backend pre-submit deferrals. The test executor never fabricates
//! a native cleanup receipt or claims these counters measure hardware work.
use super::*;

#[derive(Default)]
pub(in crate::continuous_engine) struct ControlledDeferrals {
    pending: Mutex<Option<GuardedDispatchOutcome<()>>>,
    origin: Arc<()>,
    pub capacity_epoch: AtomicU64,
    pub planning_captures: AtomicUsize,
    pub maintenance_calls: AtomicUsize,
}

struct Maintenance {
    origin: Arc<()>,
    change_capacity: bool,
}

impl ControlledDeferrals {
    pub fn capacity(&self, ids: &[RequestId], maintenance: Option<bool>) {
        let epoch = self.capacity_epoch.load(Ordering::Acquire);
        let observed = ExecutorAdmissionEpochs::new(NonZeroU64::new(47).unwrap(), 0, epoch);
        let condition = vnext::CapacityWaitCondition::from_observation(
            47,
            vec![vnext::CapacityAvailabilityEpoch::new(
                vnext::CapacityAvailabilitySource::ActiveSequenceSlots,
                epoch + 1,
            )
            .unwrap()],
        )
        .unwrap();
        let pressure = vnext::DeviceCapacityPressure::new(
            vnext::DeviceCapacityPressureScope::PlanBudget,
            "device.controller-deferral".to_owned(),
            1,
            1,
            1,
            1,
            1,
        )
        .unwrap();
        let stage = ExecutorExecutionCapacityStage::StepAdmission;
        let deferral = ExecutorExecutionCapacityDeferral::from_backing_pressure(
            observed,
            condition,
            pressure.into(),
            stage,
        )
        .unwrap()
        .into();
        let outcome = match maintenance {
            Some(change_capacity) => GuardedDispatchOutcome::MaintenanceDeferred {
                deferral,
                ticket: ExecutorExecutionMaintenanceTicket::from_backend(
                    stage,
                    ids.to_vec(),
                    Maintenance {
                        origin: Arc::clone(&self.origin),
                        change_capacity,
                    },
                )
                .unwrap(),
            },
            None => GuardedDispatchOutcome::Deferred(deferral),
        };
        assert!(self.pending.lock().replace(outcome).is_none());
    }

    pub fn take<T>(
        &self,
        guard: &dyn NonblockingHostSubmissionGuard,
        entries: &AtomicUsize,
    ) -> Option<GuardedDispatchOutcome<T>> {
        let value = self.pending.lock().take()?;
        entries.fetch_add(1, Ordering::AcqRel);
        if guard.check().is_err() {
            return Some(GuardedDispatchOutcome::ReplanBeforeEncode);
        }
        Some(match value {
            GuardedDispatchOutcome::Deferred(deferral) => {
                GuardedDispatchOutcome::Deferred(deferral)
            }
            GuardedDispatchOutcome::MaintenanceDeferred { deferral, ticket } => {
                GuardedDispatchOutcome::MaintenanceDeferred { deferral, ticket }
            }
            _ => unreachable!("only typed zero-submit deferrals enter this fixture slot"),
        })
    }

    pub fn maintain(
        &self,
        ticket: ExecutorExecutionMaintenanceTicket,
        guard: &dyn NonblockingHostSubmissionGuard,
    ) -> Result<ExecutorExecutionMaintenanceOutcome> {
        let Some(context) = ticket.into_backend::<Maintenance>() else {
            return Ok(ExecutorExecutionMaintenanceOutcome::Unsupported);
        };
        if !Arc::ptr_eq(&context.origin, &self.origin) {
            return Ok(ExecutorExecutionMaintenanceOutcome::Unsupported);
        }
        if let Err(reason) = guard.check() {
            return Ok(ExecutorExecutionMaintenanceOutcome::Rejected(reason));
        }
        self.maintenance_calls.fetch_add(1, Ordering::AcqRel);
        if context.change_capacity {
            self.capacity_epoch.fetch_add(1, Ordering::AcqRel);
        }
        Ok(ExecutorExecutionMaintenanceOutcome::Recapture {
            observed: ExecutorAdmissionEpochs::new(
                NonZeroU64::new(47).unwrap(),
                0,
                self.capacity_epoch.load(Ordering::Acquire),
            ),
            progress: None,
        })
    }
}
