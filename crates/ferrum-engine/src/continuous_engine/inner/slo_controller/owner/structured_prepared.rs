//! Numerical membership facts captured before the one real dispatch. These
//! facts cannot grant submission permission or change the prepared cohort.
use super::*;
use crate::continuous_engine::inner::cost_observation::{
    participant_host_features, PreparedRowBindingV2, PreparedStructuredFactsV2,
};
use ferrum_interfaces::execution_cost::{
    host_history_cost_signature, ActualRowWork, ExpectedWaveInput,
};
use ferrum_interfaces::vnext::{
    ExecutionCostRouteAvailability, FutureCostOutput, FutureWaveCostQuery, FutureWaveCostRow,
};
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::{
    windows::{PreparedRowFactsV2, PreparedWorkV2},
    StructuredOwnerFactsV2,
};

impl PreparedControllerWave {
    /// Only the explicit V2 calibration collector calls this. A failed
    /// projection invalidates its source; it must never classify the attempt as
    /// outside the population, narrow the cohort, or suppress the real request.
    pub(in crate::continuous_engine::inner) fn structured_prepared_facts(
        &self,
        engine: &EngineInner,
        configured_budget: Option<
            super::super::super::calibration::StructuredPreparedProjectionBudgetV2,
        >,
    ) -> Result<PreparedStructuredFactsV2> {
        use super::super::super::calibration::StructuredPreparedProjectionReportV2;
        let (result, report) = StructuredPreparedProjectionReportV2::capture(
            configured_budget,
            engine.config.scheduler.slo.planner.planning_budget(),
            slo_clock_now,
            |started, allowance| {
                self.structured_prepared_facts_with_budget(engine, started, allowance)
            },
        );
        if let Some(receipt) = &self.flight.calibration {
            receipt.record_structured_prepared_projection(report)?;
        }
        result
    }

    fn structured_prepared_facts_with_budget(
        &self,
        engine: &EngineInner,
        started: Instant,
        allowance: std::time::Duration,
    ) -> Result<PreparedStructuredFactsV2> {
        if !engine.manual_calibration_driver
            || !self.armed
            || self.flight.phase.load(Ordering::Acquire) != READY
            || self.flight.calibration.is_none()
            || !matches!(
                self.flight.work.timing,
                ControllerTimingCommitment::CompleteRequests
            )
        {
            return Err(FerrumError::invalid_request(
                "structured Prepared facts require an undispatched manual calibration wave",
            ));
        }
        let runtime = engine.cost_runtime.as_ref().ok_or_else(|| {
            FerrumError::unsupported("structured Prepared facts require cost observation")
        })?;
        if !runtime.structured_capture {
            return Err(FerrumError::config(
                "structured Prepared facts require configured structured capture",
            ));
        }
        // An independent bounded diagnostic read: this never extends the
        // original publication/witness deadline or supplies an execution token.
        let budget = ControllerBudget::new(started, allowance)?;
        let poll = || -> Result<()> {
            if budget.poll() {
                Ok(())
            } else {
                Err(FerrumError::unsupported(
                    "structured Prepared projection budget exhausted",
                ))
            }
        };
        let work = &self.flight.work;
        let guard = HostGuard { engine, work };
        guard.check().map_err(|reason| {
            FerrumError::invalid_request(format!("structured Prepared guard: {reason:?}"))
        })?;
        let expected = work.expected.work();
        if expected.participants().is_empty()
            || expected.participants().len() != work.proof.fences.len()
            || expected.participants().len() > 256
        {
            return Err(FerrumError::internal(
                "structured Prepared cohort correspondence changed",
            ));
        }
        let sequences = engine
            .sequences
            .try_read()
            .ok_or_else(|| FerrumError::unsupported("structured Prepared sequence view is busy"))?;
        let requests: Vec<_> = work
            .proof
            .fences
            .iter()
            .map(|fence| ExecutorResourcePlanningRequest {
                request_id: &fence.key.request_id,
                cache_id: fence.resource_cache_id(),
            })
            .collect();
        let view = match engine.model_executor.execution_cost_route_view(
            &requests,
            ResourcePlanningLimits {
                maximum_participants: 256,
                maximum_projected_waves: 1,
                ..Default::default()
            },
            &mut || budget.poll(),
        ) {
            ExecutionCostRouteAvailability::Known(view) => view.with_structured_capture(true),
            ExecutionCostRouteAvailability::Unknown(reason) => {
                return Err(FerrumError::unsupported(format!(
                    "structured Prepared route: {reason:?}"
                )));
            }
        };
        let resources = view.resource_view();
        if resources.plan_hash() != expected.plan_hash()
            || resources.coordinator_id() != expected.coordinator_id()
            || resources.lane_id() != Some(expected.lane_id())
        {
            return Err(FerrumError::invalid_request(
                "structured Prepared resource epoch changed",
            ));
        }
        let mut projected_rows = Vec::with_capacity(expected.participants().len());
        let mut rows = Vec::with_capacity(expected.participants().len());
        // Already sorted by the real resource authority. The participant index
        // refers to the captured registry order, not this physical position.
        for (position, participant) in expected.participants().iter().enumerate() {
            poll()?;
            let selected = participant.selection();
            let fence = work
                .proof
                .fences
                .get(selected.participant_index)
                .ok_or_else(|| {
                    FerrumError::internal("structured Prepared participant index changed")
                })?;
            let sequence = sequences
                .get(&selected.request_id)
                .ok_or_else(|| FerrumError::cancelled("structured Prepared request disappeared"))?;
            if fence.key.request_id != selected.request_id
                || fence.incarnation != selected.owner_incarnation.get()
                || fence.generation != selected.work_generation.get()
                || !fence.matches_sequence(sequence)
                || resources.participants().get(selected.participant_index)
                    != Some(participant.resource())
            {
                return Err(FerrumError::invalid_request(
                    "structured Prepared frontier changed",
                ));
            }
            let host = participant_host_features(sequence)
                .filter(|host| host.supports_empirical_plain_text_content())
                .ok_or_else(|| {
                    FerrumError::unsupported("structured Prepared host domain is unavailable")
                })?;
            let output = match &selected.input {
                ExpectedWaveInput::Decode { .. } => FutureCostOutput::Decode {
                    policy: selected.decode_policy.as_ref().ok_or_else(|| {
                        FerrumError::internal("structured Prepared decode policy is absent")
                    })?,
                },
                ExpectedWaveInput::Prefill { chunk } => FutureCostOutput::Prefill {
                    final_logits: chunk.is_final(),
                },
            };
            let policy = sequence.cost_policy_signature.ok_or_else(|| {
                FerrumError::unsupported("structured Prepared host policy identity is absent")
            })?;
            projected_rows.push(FutureWaveCostRow {
                participant_index: selected.participant_index,
                work: selected.work,
                host_policy_signature: host_history_cost_signature(
                    policy,
                    host.state.generated_tokens_before,
                ),
                host_features: Some(host),
                output,
            });
            let numerical_work = match selected.work {
                ActualRowWork::Decode { kv_tokens } => PreparedWorkV2::Decode { kv_tokens },
                ActualRowWork::Prefill {
                    offset,
                    count,
                    total_prompt_tokens,
                } => PreparedWorkV2::Prefill {
                    offset,
                    count,
                    total_prompt_tokens,
                },
                _ => {
                    return Err(FerrumError::unsupported(
                        "structured Prepared work is not prefill/decode",
                    ))
                }
            };
            rows.push(PreparedRowBindingV2 {
                request_id: selected.request_id.clone(),
                owner_incarnation: selected.owner_incarnation.get(),
                work_generation: selected.work_generation.get(),
                frontier: PreparedRowFactsV2 {
                    physical_position: position as u32,
                    work: numerical_work,
                    generated_before: host.state.generated_tokens_before,
                    maximum_output: host.state.maximum_output_tokens,
                    context_before: u64::try_from(fence.context).map_err(|_| {
                        FerrumError::internal("structured Prepared context overflow")
                    })?,
                },
            });
        }
        drop(requests);
        drop(sequences);
        let projected = match engine.model_executor.project_execution_cost_wave(
            &view,
            &view.initial_state(),
            &FutureWaveCostQuery {
                kind: expected.kind(),
                rows: &projected_rows,
            },
            &mut || budget.poll(),
        ) {
            ExecutionCostRouteAvailability::Known(projected) => projected,
            ExecutionCostRouteAvailability::Unknown(reason) => {
                return Err(FerrumError::unsupported(format!(
                    "structured Prepared projection: {reason:?}"
                )));
            }
        };
        poll()?;
        let selected = projected.statistical_evidence.ok_or_else(|| {
            FerrumError::unsupported("structured Prepared selected evidence is absent")
        })?;
        let recipe = Arc::clone(
            selected
                .structured_capture()
                .ok_or_else(|| {
                    FerrumError::unsupported("structured Prepared recipe was not captured")
                })?
                .map_err(|reason| {
                    FerrumError::unsupported(format!("structured Prepared recipe: {reason:?}"))
                })?,
        );
        let owner = StructuredOwnerFactsV2::from_prepared(&projected.shape, &selected, &recipe)
            .map_err(|reason| {
                FerrumError::unsupported(format!("structured Prepared owner: {reason:?}"))
            })?;
        guard.check().map_err(|reason| {
            FerrumError::invalid_request(format!("structured Prepared guard changed: {reason:?}"))
        })?;
        if self.flight.phase.load(Ordering::Acquire) != READY {
            return Err(FerrumError::invalid_request(
                "structured Prepared wave already entered dispatch",
            ));
        }
        poll()?;
        Ok(PreparedStructuredFactsV2 {
            exact: projected.shape,
            selected,
            recipe,
            owner,
            rows,
        })
    }
}
