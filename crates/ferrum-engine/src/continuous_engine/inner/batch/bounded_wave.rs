//! One explicit execution boundary. These observations carry no resource permits.

use super::*;
use ferrum_interfaces::model_executor::{PlanRuntimeMixedBatchOutcome, PrefillChunk};
use std::collections::HashSet;

/// A planner selects exactly one wave from an already published scheduler batch.
/// A split plan must choose its next phase again after the preceding receipt.
#[derive(Debug, Clone)]
pub(in crate::continuous_engine) enum PlanRuntimeWaveSelection {
    Prefill {
        request_ids: Vec<RequestId>,
    },
    Decode {
        request_ids: Vec<RequestId>,
    },
    Mixed {
        prefill_ids: Vec<RequestId>,
        decode_ids: Vec<RequestId>,
    },
}

impl PlanRuntimeWaveSelection {
    fn participants(&self) -> (&[RequestId], &[RequestId]) {
        match self {
            Self::Prefill { request_ids } => (request_ids, &[]),
            Self::Decode { request_ids } => (&[], request_ids),
            Self::Mixed {
                prefill_ids,
                decode_ids,
            } => (prefill_ids, decode_ids),
        }
    }
}

/// Actual model work, including a capacity-narrowed chunk. This is observation,
/// not a substitute for the completion's opaque KV handle or execution fence.
#[derive(Debug, Clone)]
pub(in crate::continuous_engine) struct PlanRuntimePrefillReceipt {
    pub request_id: RequestId,
    pub planned_chunk: PrefillChunk,
    pub completed_chunk: PrefillChunk,
    pub capacity_probe_count: u32,
}

/// Covers executor entry through host publication/commit only. Preparation,
/// prefix restore and maintenance in the outer iteration are excluded. Commit
/// currently includes any streaming send wait; it is not a device-only metric.
#[derive(Debug)]
pub(in crate::continuous_engine) struct PlanRuntimeWaveReceipt {
    pub prefills: Vec<PlanRuntimePrefillReceipt>,
    pub decode_request_ids: Vec<RequestId>,
    pub executor_started_at: Instant,
    pub executor_completed_at: Instant,
    pub commit_completed_at: Instant,
}

pub(in crate::continuous_engine) enum PlanRuntimeWaveOutcome {
    Completed(PlanRuntimeWaveReceipt),
    /// The existing executor contract proves no encoding/submission occurred.
    /// The caller must replan; no same-call split, retry or maintenance is run.
    NotSubmitted(ExecutorExecutionDeferral),
    /// Also a proven unchanged frontier, requiring a new planner decision.
    Unsupported,
    NoWork,
}

impl EngineInner {
    /// Compatibility entrypoint until the time planner supplies an explicit
    /// selection. Startup still rejects Enforce until the complete controller
    /// is wired. Off/Observe retain their existing adaptive execution paths.
    pub(super) async fn process_one_plan_runtime_wave(
        &self,
        batch: &ferrum_interfaces::BatchPlan,
    ) -> Result<()> {
        let (prefill_ids, decode_ids) = self.classify_published_batch_sequences(batch)?;
        let selection = if self.config.batching.prefill_decode_execution
            == ferrum_types::PrefillDecodeExecution::Mixed
            && !prefill_ids.is_empty()
            && !decode_ids.is_empty()
        {
            PlanRuntimeWaveSelection::Mixed {
                prefill_ids,
                decode_ids,
            }
        } else if !decode_ids.is_empty() {
            PlanRuntimeWaveSelection::Decode {
                request_ids: decode_ids,
            }
        } else {
            PlanRuntimeWaveSelection::Prefill {
                request_ids: prefill_ids,
            }
        };
        let outcome = self.execute_plan_runtime_wave(batch, &selection).await?;
        let observation = match outcome {
            PlanRuntimeWaveOutcome::Completed(receipt) => serde_json::json!({
                "disposition": "completed",
                "boundary": "executor_entry_through_host_commit",
                "prefills": receipt.prefills.iter().map(|progress| serde_json::json!({
                    "request_id": progress.request_id,
                    "planned_chunk": progress.planned_chunk,
                    "completed_chunk": progress.completed_chunk,
                    "capacity_probe_count": progress.capacity_probe_count,
                })).collect::<Vec<_>>(),
                "decode_request_ids": receipt.decode_request_ids,
                "executor_duration_ms": receipt.executor_completed_at.duration_since(receipt.executor_started_at).as_secs_f64() * 1000.0,
                "executor_through_commit_ms": receipt.commit_completed_at.duration_since(receipt.executor_started_at).as_secs_f64() * 1000.0,
            }),
            PlanRuntimeWaveOutcome::NotSubmitted(ExecutorExecutionDeferral::Capacity(deferral)) => {
                serde_json::json!({
                    "disposition": "not_submitted",
                    "reason": "capacity",
                    "stage": deferral.stage(),
                    "wait_condition": deferral.wait_condition(),
                    "shortfalls": deferral.shortfalls(),
                })
            }
            PlanRuntimeWaveOutcome::NotSubmitted(ExecutorExecutionDeferral::RequestState(
                deferral,
            )) => serde_json::json!({
                "disposition": "not_submitted",
                "reason": "request_state",
                "request_ids": deferral.request_ids(),
            }),
            PlanRuntimeWaveOutcome::Unsupported => {
                serde_json::json!({"disposition": "unsupported"})
            }
            PlanRuntimeWaveOutcome::NoWork => serde_json::json!({"disposition": "no_work"}),
        };
        self.write_scheduler_trace_event(serde_json::json!({
            "event": "engine_bounded_plan_runtime_wave",
            "observation": observation,
        }));
        Ok(())
    }

    /// Invoke at most one physical model wave, validate its complete receipt,
    /// publish completed frontiers, then return control. An ordinary executor
    /// error can mean submission occurred, so every selected request is made
    /// terminal even if a peer's cleanup fails; it can never be blindly retried.
    pub(in crate::continuous_engine) async fn execute_plan_runtime_wave(
        &self,
        batch: &ferrum_interfaces::BatchPlan,
        selection: &PlanRuntimeWaveSelection,
    ) -> Result<PlanRuntimeWaveOutcome> {
        if self.model_executor.execution_resource_authority()
            != ferrum_interfaces::model_executor::ExecutionResourceAuthority::PlanRuntime
        {
            return Err(FerrumError::unsupported(
                "bounded waves require PlanRuntime authority",
            ));
        }
        let (prefill_ids, decode_ids) = selection.participants();
        if prefill_ids.is_empty() && decode_ids.is_empty() {
            return Ok(PlanRuntimeWaveOutcome::NoWork);
        }
        if matches!(selection, PlanRuntimeWaveSelection::Mixed { .. })
            && (prefill_ids.is_empty() || decode_ids.is_empty())
        {
            return Err(FerrumError::scheduler(
                "mixed wave selection requires both phases",
            ));
        }
        // Anchor before the preparation locks/allocations. The legacy receipt
        // below still starts at executor entry; it is a distinct observation.
        let mut preparation = self.prepare_cost_observation();
        self.validate_plan_runtime_wave_selection(batch, prefill_ids, decode_ids)?;
        let mut prefills = Vec::with_capacity(prefill_ids.len());
        for rid in prefill_ids {
            let scheduled = batch
                .requests
                .iter()
                .find(|item| item.request.id == *rid)
                .expect("validated selected batch member");
            let Some(input) = self.prepare_plan_runtime_prefill(scheduled, &mut preparation)?
            else {
                return Ok(PlanRuntimeWaveOutcome::NoWork);
            };
            prefills.push(input);
        }
        let decodes = self.prepare_plan_runtime_decodes(decode_ids, &mut preparation);
        if decodes.len() != decode_ids.len() {
            // Cancellation/readiness changed after the planning snapshot. Do
            // not silently submit a narrower shape than the planner selected.
            return Ok(PlanRuntimeWaveOutcome::NoWork);
        }
        let mut cost = preparation.and_then(EngineCostPreparation::begin);
        let result = self
            .execute_prepared_plan_runtime_wave(&prefills, &decodes, &mut cost)
            .await
            .and_then(|outcome| {
                if let PlanRuntimeWaveOutcome::NotSubmitted(deferral) = &outcome {
                    let participants = prefill_ids
                        .iter()
                        .chain(decode_ids)
                        .cloned()
                        .collect::<Vec<_>>();
                    match deferral {
                        ExecutorExecutionDeferral::RequestState(deferral) => {
                            if deferral
                                .request_ids()
                                .iter()
                                .any(|rid| !participants.contains(rid))
                            {
                                return Err(FerrumError::internal(
                                    "bounded wave deferral names another frontier",
                                ));
                            }
                        }
                        ExecutorExecutionDeferral::Capacity(deferral) => {
                            deferral.validated_maintenance_retry_scope(&participants)?;
                        }
                    }
                }
                Ok(outcome)
            });
        if let Err(error) = result {
            let message = error.to_string();
            let exhausted = is_resource_exhausted_error(&error);
            let mut cleanup_errors = Vec::new();
            for rid in prefill_ids.iter().chain(decode_ids) {
                let terminal_error = if exhausted {
                    FerrumError::resource_exhausted(message.clone())
                } else {
                    FerrumError::backend(message.clone())
                };
                if let Err(cleanup) = self.complete_request_with_error(rid, terminal_error).await {
                    cleanup_errors.push(format!("{rid}: {cleanup}"));
                }
            }
            if !cleanup_errors.is_empty() {
                return Err(FerrumError::internal(format!(
                    "{message}; bounded-wave participant cleanup failures: {}",
                    cleanup_errors.join("; ")
                )));
            }
            return Err(error);
        }
        result
    }

    fn validate_plan_runtime_wave_selection(
        &self,
        batch: &ferrum_interfaces::BatchPlan,
        prefill_ids: &[RequestId],
        decode_ids: &[RequestId],
    ) -> Result<()> {
        let sequences = self.sequences.read();
        let mut unique = HashSet::new();
        for (rid, is_decode) in prefill_ids
            .iter()
            .map(|rid| (rid, false))
            .chain(decode_ids.iter().map(|rid| (rid, true)))
        {
            if !unique.insert(rid) {
                return Err(FerrumError::scheduler(
                    "wave selection contains a duplicate request",
                ));
            }
            let scheduled = batch
                .requests
                .iter()
                .find(|item| item.request.id == *rid)
                .ok_or_else(|| {
                    FerrumError::scheduler("wave selection is outside its scheduler batch")
                })?;
            let sequence = sequences.get(rid).ok_or_else(|| {
                FerrumError::scheduler("wave selection has no published sequence")
            })?;
            if sequence.prefill_complete != is_decode {
                return Err(FerrumError::scheduler(
                    "wave selection has a stale execution phase",
                ));
            }
            if !is_decode && scheduled.tokens_processed != sequence.prefill_tokens_processed {
                return Err(FerrumError::scheduler(
                    "wave selection has a stale prefill offset",
                ));
            }
        }
        Ok(())
    }

    async fn execute_prepared_plan_runtime_wave(
        &self,
        prefills: &[PlanRuntimePrefillInput],
        decodes: &[PlanRuntimeDecodeInput],
        cost: &mut Option<ObservedCostCall>,
    ) -> Result<PlanRuntimeWaveOutcome> {
        let decode_ids = decodes
            .iter()
            .map(|input| input.request_id.clone())
            .collect::<Vec<_>>();
        let executor_started_at = Instant::now();
        let (prefill_outputs, decode_outputs) = if !prefills.is_empty() && !decodes.is_empty() {
            match self.cost_mixed(prefills, decodes, cost).await? {
                PlanRuntimeMixedBatchOutcome::Completed { prefills, decodes } => {
                    (prefills, decodes)
                }
                PlanRuntimeMixedBatchOutcome::NotSubmitted(deferral) => {
                    return Ok(PlanRuntimeWaveOutcome::NotSubmitted(deferral))
                }
                PlanRuntimeMixedBatchOutcome::Unsupported => {
                    return Ok(PlanRuntimeWaveOutcome::Unsupported)
                }
            }
        } else if prefills.len() == 1 {
            match self.cost_prefill(&prefills[0], cost).await? {
                PlanRuntimePrefillOutcome::Completed(completion) => (vec![completion], Vec::new()),
                PlanRuntimePrefillOutcome::Deferred(deferral) => {
                    return Ok(PlanRuntimeWaveOutcome::NotSubmitted(deferral))
                }
            }
        } else if !prefills.is_empty() {
            match self.cost_batch_prefill(prefills, cost).await? {
                PlanRuntimeBatchPrefillOutcome::Completed(completions) => (completions, Vec::new()),
                PlanRuntimeBatchPrefillOutcome::NotSubmitted(deferral) => {
                    return Ok(PlanRuntimeWaveOutcome::NotSubmitted(deferral))
                }
                PlanRuntimeBatchPrefillOutcome::Unsupported => {
                    return Ok(PlanRuntimeWaveOutcome::Unsupported)
                }
            }
        } else {
            match self.cost_batch_decode(decodes, cost).await? {
                PlanRuntimeBatchDecodeOutcome::Completed(outputs) => (Vec::new(), outputs),
                PlanRuntimeBatchDecodeOutcome::Deferred(deferral) => {
                    return Ok(PlanRuntimeWaveOutcome::NotSubmitted(deferral))
                }
            }
        };
        let executor_completed_at = Instant::now();
        let validation = (|| {
            if prefill_outputs.len() != prefills.len() {
                return Err(FerrumError::internal(
                    "bounded wave returned a different prefill participant count",
                ));
            }
            for (input, output) in prefills.iter().zip(&prefill_outputs) {
                output.validate_for(
                    &input.request_id,
                    input.chunk,
                    self.model_executor.info().vocab_size,
                )?;
            }
            self.validate_plan_runtime_decode_outputs(decodes, &decode_outputs)
        })();
        if let Err(error) = validation {
            self.discard_plan_runtime_prefill_completions(prefill_outputs)?;
            return Err(error);
        }
        let prefill_receipts = prefills
            .iter()
            .zip(&prefill_outputs)
            .map(|(input, output)| PlanRuntimePrefillReceipt {
                request_id: input.request_id.clone(),
                planned_chunk: output.planned_chunk(),
                completed_chunk: output.completed_chunk(),
                capacity_probe_count: output.capacity_probe_count(),
            })
            .collect();
        if let Err(error) = self
            .commit_plan_runtime_decode_outputs(
                &decode_ids,
                decode_outputs,
                Some(executor_completed_at),
                cost,
            )
            .await
        {
            self.discard_plan_runtime_prefill_completions(prefill_outputs)?;
            return Err(error);
        }
        let mut outputs = prefill_outputs.into_iter();
        for input in prefills {
            let output = outputs
                .next()
                .expect("validated bounded prefill cardinality");
            if let Err(error) = self
                .commit_plan_runtime_prefill_completion(
                    &input.request_id,
                    input.input_tokens.len(),
                    input.chunk,
                    output,
                    cost,
                )
                .await
            {
                self.discard_plan_runtime_prefill_completions(outputs)?;
                return Err(error);
            }
        }
        if !decode_ids.is_empty() {
            self.scheduler
                .record_decode_execution_capacity_success(decode_ids.len());
        }
        Ok(PlanRuntimeWaveOutcome::Completed(PlanRuntimeWaveReceipt {
            prefills: prefill_receipts,
            decode_request_ids: decode_ids,
            executor_started_at,
            executor_completed_at,
            commit_completed_at: Instant::now(),
        }))
    }
}
