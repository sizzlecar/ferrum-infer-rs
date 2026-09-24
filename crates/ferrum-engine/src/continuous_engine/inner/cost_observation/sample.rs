use super::*;
use ferrum_scheduler::implementations::continuous::cost_model as model;

impl EngineCostCall {
    pub(super) fn make_sample(&mut self) -> Result<model::WaveCostObservation, CostCallRejection> {
        if let Some(reason) = self.rejection {
            return Err(reason);
        }
        if self.prepare_started_at_ns.is_none() {
            return Err(CostCallRejection::Clock);
        }
        if self.dispatch.waves == 0 {
            return Err(CostCallRejection::NoPhysicalWave);
        }
        if self.dispatch.waves != 1
            || self.recorder.observations().len() != 1
            || self.boundary != WaveObservationBoundary::IsolatedPreparationToCommit
        {
            return Err(CostCallRejection::Composite);
        }
        match self.dispatch.outcome {
            Some(ObservedCallOutcome::Completed) => {}
            Some(ObservedCallOutcome::Failed) => return Err(CostCallRejection::ExecutorFailed),
            _ => return Err(CostCallRejection::ExecutorIncomplete),
        }
        if self.dispatch.unknown.is_some() {
            return Err(CostCallRejection::ActualEvidenceUnknown);
        }
        let wave = &self.recorder.observations()[0];
        if wave.boundary != WaveObservationBoundary::IsolatedPreparationToCommit {
            return Err(CostCallRejection::Composite);
        }
        if wave.outcome != Some(ActualWaveOutcome::Completed) {
            return Err(CostCallRejection::ExecutorFailed);
        }
        let shape = wave
            .shape
            .as_ref()
            .ok_or(CostCallRejection::ActualEvidenceUnknown)?;
        let terminal = wave
            .terminal_at_ns
            .ok_or(CostCallRejection::ExecutorIncomplete)?;
        let mut committed_at = terminal;
        for row in &shape.rows {
            if self
                .participants
                .iter()
                .find(|participant| participant.request_id == row.request_id)
                .is_none_or(|participant| participant.output_policy_signature.is_none())
            {
                return Err(CostCallRejection::OutputPolicyUnknown);
            }
            let evidence = self
                .host
                .iter()
                .flatten()
                .find(|evidence| evidence.request_id == row.request_id)
                .ok_or(CostCallRejection::HostMissing)?;
            if evidence.owner_incarnation != row.owner_incarnation
                || evidence.work_generation != row.work_generation
                || evidence.input_index != row.input_index
            {
                return Err(CostCallRejection::FrontierMismatch);
            }
            match evidence.outcome {
                HostCommitOutcome::Committed(work) => validate_work(row.work, work)?,
                HostCommitOutcome::Cancelled => return Err(CostCallRejection::HostCancelled),
                HostCommitOutcome::Failed => return Err(CostCallRejection::HostFailed),
            }
            let at = evidence
                .committed_at_ns
                .filter(|at| *at >= terminal)
                .ok_or(CostCallRejection::Clock)?;
            committed_at = committed_at.max(at);
        }
        if self.host.iter().flatten().any(|evidence| {
            !shape
                .rows
                .iter()
                .any(|row| row.request_id == evidence.request_id)
        }) {
            return Err(CostCallRejection::HostUnexpected);
        }
        let now = self
            .clock
            .now_ns()
            .filter(|now| *now >= committed_at)
            .ok_or(CostCallRejection::Clock)?;
        let actual_shape = scheduler_shape(shape)?;
        let identity = match &self.identity {
            ExecutorCostIdentityAvailability::Known(identity) => identity,
            ExecutorCostIdentityAvailability::Unknown { .. } => {
                return Err(CostCallRejection::IdentityUnknown)
            }
        };
        if identity.schema_version != EXECUTOR_COST_IDENTITY_SCHEMA {
            return Err(CostCallRejection::IdentitySchema);
        }
        let fingerprint = model::ExecutionFingerprint {
            model_weights: identity.model_weights,
            numerical_policy: identity.numerical_policy,
            device_runtime: identity.device_runtime,
            execution_config: identity.execution_config,
        };
        let device_elapsed_ns = wave.device_elapsed_ns.map(NonZeroU64::get);
        let handle = self
            .dispatch
            .handle
            .as_ref()
            .ok_or(CostCallRejection::ExecutorIncomplete)?;
        self.recorder
            .host_committed(handle, committed_at)
            .map_err(|_| CostCallRejection::InvalidWall)?;
        let wall_total_ns = self
            .recorder
            .trainable_wall_ns(handle)
            .map_err(|_| CostCallRejection::InvalidWall)?
            .get();
        Ok(model::WaveCostObservation {
            fingerprint,
            actual_shape,
            boundary: model::CostBoundary::PreparationToCommit,
            outcome: model::WaveObservationOutcome::Completed,
            timing: model::WaveTiming {
                wall_total_ns,
                device_elapsed_ns,
                stages: model::WaveStageTimings::default(),
            },
            observed_at_ns: now,
        })
    }
}

pub(super) fn validate_work(
    actual: ActualRowWork,
    committed: HostCommittedWork,
) -> Result<(), CostCallRejection> {
    let valid = match (actual, committed) {
        (
            ActualRowWork::Decode { kv_tokens },
            HostCommittedWork::Decode {
                kv_tokens_before,
                kv_tokens_after,
                generated_tokens_before,
                generated_tokens_after,
            },
        ) => {
            kv_tokens == kv_tokens_before
                && kv_tokens_before.checked_add(1) == Some(kv_tokens_after)
                && generated_tokens_before.checked_add(1) == Some(generated_tokens_after)
        }
        (
            ActualRowWork::Prefill {
                offset,
                count,
                total_prompt_tokens,
            },
            HostCommittedWork::Prefill {
                start,
                end,
                total_prompt_tokens: committed_total,
                generated_tokens_before,
                generated_tokens_after,
            },
        ) => {
            offset == start
                && offset.checked_add(count) == Some(end)
                && total_prompt_tokens == committed_total
                && if end == total_prompt_tokens {
                    // Original final prefill commits the first token; recomputation
                    // may resume an existing generation and still commits exactly one.
                    generated_tokens_before.checked_add(1) == Some(generated_tokens_after)
                } else {
                    generated_tokens_before == generated_tokens_after
                }
        }
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(CostCallRejection::WorkMismatch)
    }
}

/// Lossless representation adapter. The executor canonical builder already
/// includes mixed row order; this boundary must not rehash the cost key.
pub(super) fn scheduler_shape(
    shape: &ActualWaveShape,
) -> Result<model::WaveExecutionShape, CostCallRejection> {
    use ferrum_scheduler::implementations::continuous::slo_planner::{
        actual_cost_shape, PlanningUnknownReason,
    };
    actual_cost_shape(shape).map_err(|reason| match reason {
        PlanningUnknownReason::ShapeCapacity => CostCallRejection::RecorderCapacity,
        _ => CostCallRejection::ActualEvidenceUnknown,
    })
}
