//! Exact manual cohort selection. It shares all publication/output guards with
//! completion, but never chooses another row or narrows a requested chunk.
use super::super::calibration::{CalibrationBlockReason, CalibrationWork};
use super::*;

pub(in crate::continuous_engine::inner) enum CalibrationPreparation {
    Selected(owner::PreparedControllerWave),
    Blocked(CalibrationBlockReason),
}

impl EngineInner {
    pub(in crate::continuous_engine::inner) fn prepare_calibration_wave(
        &self,
        rows: &[CalibrationWork],
        maximum_requests: NonZeroUsize,
    ) -> Result<CalibrationPreparation> {
        if !self.manual_calibration_driver {
            return Err(FerrumError::invalid_request(
                "calibration driver is not active",
            ));
        }
        if self.slo_controller.lock().pending_maintenance.is_some() {
            return Ok(CalibrationPreparation::Blocked(
                CalibrationBlockReason::MaintenanceUnavailable,
            ));
        }
        if !self.retry_slo_publication_release()? {
            return Ok(CalibrationPreparation::Blocked(
                CalibrationBlockReason::PublicationUnavailable,
            ));
        }
        let budget = ControllerBudget::new(
            slo_clock_now(),
            self.config.scheduler.slo.planner.planning_budget(),
        )?;
        let mut hint = ferrum_interfaces::BatchHint::simple(self.config.batching.max_batch_size);
        hint.max_tokens = self.config.batching.max_num_batched_tokens;
        // A successful capture followed by a publication race returns Idle
        // without recording a new selection failure. Never attribute that
        // current turn to an earlier rejected frontier or other old reason.
        {
            let mut state = self.slo_controller.lock();
            state.last_observation = None;
            state.last_resource_unavailable = None;
        }
        let prepared =
            self.prepare_exact_calibration_selection(&hint, &budget, rows, maximum_requests)?;
        let live = budget.finish_planning();
        match prepared {
            SloIterationPlan::Selected(prepared) if live => {
                Ok(CalibrationPreparation::Selected(prepared))
            }
            SloIterationPlan::Selected(prepared) => {
                drop(prepared);
                // The abandoned publication is reaped by the session before
                // another action; no device call has entered.
                Ok(CalibrationPreparation::Blocked(
                    CalibrationBlockReason::SelectionUnavailable("compute_budget_exhausted"),
                ))
            }
            _ => {
                self.finish_controller_audit(&budget, "calibration_blocked");
                let state = self.slo_controller.lock();
                let reason = state
                    .last_observation
                    .filter(|observation| observation.disposition == "calibration_blocked")
                    .map_or(
                        CalibrationBlockReason::PublicationUnavailable,
                        |observation| {
                            state.last_resource_unavailable.map_or(
                                CalibrationBlockReason::SelectionUnavailable(observation.reason),
                                CalibrationBlockReason::ResourceUnavailable,
                            )
                        },
                    );
                Ok(CalibrationPreparation::Blocked(reason))
            }
        }
    }
}
