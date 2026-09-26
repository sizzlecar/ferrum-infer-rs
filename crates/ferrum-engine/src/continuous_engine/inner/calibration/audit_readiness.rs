//! Read-only trigger evidence. This is neither an execution reservation nor a
//! cost witness: the subsequent audit still captures and checks all resources.
use super::*;
use crate::continuous_engine::output_flow_runtime::OutputPlanningCreditView;
use ferrum_interfaces::scheduler::Scheduler;
use ferrum_scheduler::implementations::continuous::planning_state::PlanningQueueKind;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct CalibrationAuditRowReadinessV2 {
    pub frontier_index: usize,
    pub physically_admitted: bool,
    pub scheduler_ready: bool,
    pub output_ready: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct CalibrationAuditReadinessV2 {
    pub rows: Vec<CalibrationAuditRowReadinessV2>,
}
impl CalibrationAuditReadinessV2 {
    pub fn all_ready(&self) -> bool {
        !self.rows.is_empty()
            && self
                .rows
                .iter()
                .all(|r| r.physically_admitted && r.scheduler_ready && r.output_ready)
    }
}

impl CalibrationSession {
    /// Observe the complete, original frontier before prepare/reserve. Admission
    /// comes from the scheduler's committed physical-admission state, not from
    /// logical request creation. Output credit is observed without acquiring it.
    /// No seed/model, cost query, submission, admission or clock advancement occurs.
    pub fn audit_frontier_readiness_v2(
        &self,
        frontiers: &[CalibrationFrontier],
    ) -> Result<CalibrationAuditReadinessV2> {
        if frontiers.is_empty()
            || frontiers.len() > 256
            || frontiers.len() > self.limits.maximum_requests().get()
            || self.pending.is_some()
            || self.indeterminate
            || frontiers
                .iter()
                .any(|f| !Arc::ptr_eq(&f.session, &self.identity))
        {
            return Err(FerrumError::invalid_request(
                "audit readiness requires this session's complete settled frontier",
            ));
        }
        let inner = &self.engine.inner;
        let _iteration = inner
            .iteration_lock
            .try_lock()
            .map_err(|_| FerrumError::resource_exhausted("calibration iteration is busy"))?;
        if inner.slo_controller.lock().has_pending_calibration_work() {
            return Err(FerrumError::invalid_request(
                "audit readiness cannot bypass pending physical work",
            ));
        }
        let mut availability =
            inner
                .dynamic_admission_availability
                .try_lock()
                .ok_or_else(|| {
                    FerrumError::resource_exhausted("audit readiness capacity snapshot is busy")
                })?;
        let epochs = inner
            .model_executor
            .write_execution_capacity_snapshot(&mut availability)?
            .ok_or_else(|| {
                FerrumError::invalid_request("audit readiness capacity epochs unavailable")
            })?;
        let queue = inner
            .scheduler
            .planning_state(
                self.limits.maximum_requests(),
                AdmissionWakeSnapshot::new(
                    AdmissionWakeEpochs::new(
                        epochs.coordinator_id,
                        epochs.release_epoch,
                        epochs.capacity_epoch,
                        0,
                    ),
                    &availability,
                ),
            )
            .map_err(|reason| {
                FerrumError::resource_exhausted(format!(
                    "audit readiness scheduler snapshot: {reason:?}"
                ))
            })?;
        drop(availability);
        let sequences = inner.sequences.try_read().ok_or_else(|| {
            FerrumError::resource_exhausted("audit readiness sequence snapshot is busy")
        })?;
        if sequences.len() != frontiers.len() || queue.requests().len() != frontiers.len() {
            return Err(FerrumError::invalid_request(
                "audit readiness omits live owners",
            ));
        }
        let mut rows = Vec::new();
        rows.try_reserve_exact(frontiers.len())
            .map_err(|_| FerrumError::resource_exhausted("audit readiness allocation failed"))?;
        for (index, frontier) in frontiers.iter().enumerate() {
            let sequence = sequences.get(&frontier.request_id).ok_or_else(|| {
                FerrumError::invalid_request("audit readiness owner is no longer live")
            })?;
            if frontiers[..index]
                .iter()
                .any(|f| f.request_id == frontier.request_id)
                || sequence.cost_frontier.is_none_or(|f| {
                    f.owner_incarnation != frontier.owner
                        || f.work_generation != frontier.generation
                })
                || sequence.generated_tokens.len() != frontier.generated
                || (!sequence.prefill_complete).then_some((
                    sequence.prefill_tokens_processed,
                    sequence.prefill_context_len(),
                )) != frontier.prefill
                || sequence
                    .model_kv
                    .as_ref()
                    .map_or(0, |kv| kv.handle().num_tokens())
                    != frontier.kv_tokens
            {
                return Err(FerrumError::invalid_request(
                    "audit readiness frontier changed",
                ));
            }
            let row = queue
                .requests()
                .iter()
                .find(|r| r.key.request_id == frontier.request_id)
                .ok_or_else(|| {
                    FerrumError::invalid_request("audit readiness scheduler owner missing")
                })?;
            // Waiting owners need not yet have a scheduler prefill boundary.
            // Once admitted, require the same real stage as ordinary selection.
            if row.readiness.admitted
                && ((row.queue == PlanningQueueKind::Decode) != sequence.prefill_complete
                    || !matches!(
                        row.queue,
                        PlanningQueueKind::Prefill | PlanningQueueKind::Decode
                    )
                    || row.committed_output_tokens != frontier.generated
                    || (!sequence.prefill_complete
                        && row.prefill_offset != sequence.prefill_tokens_processed))
            {
                return Err(FerrumError::invalid_request(
                    "audit readiness scheduler frontier changed",
                ));
            }
            let output_ready = sequence.credited_output.as_ref().is_some_and(|output| {
                let view = output.port.planning_snapshot();
                output.failure.is_none()
                    && !output.port.consumer_closed()
                    && output.grant.is_none()
                    && !matches!(view.readiness, OutputPlanningCreditView::OutputBlocked(_))
                    && view.future_capacity.is_some()
            });
            rows.push(CalibrationAuditRowReadinessV2 {
                frontier_index: index,
                physically_admitted: row.readiness.admitted,
                scheduler_ready: row.readiness.ready(),
                output_ready,
            });
        }
        Ok(CalibrationAuditReadinessV2 { rows })
    }
}
