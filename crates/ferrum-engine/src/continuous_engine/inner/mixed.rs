use super::*;
use ferrum_interfaces::model_executor::PlanRuntimeMixedBatchOutcome;

pub(in crate::continuous_engine) enum MixedBatchDisposition {
    Completed,
    Split {
        prefill_ids: Vec<RequestId>,
        decode_ids: Vec<RequestId>,
    },
}

impl EngineInner {
    /// Submit the scheduler's prefill chunks and ready decode frontiers in one
    /// physical wave. Only a proven zero-submission result permits fallback.
    pub(in crate::continuous_engine) async fn run_plan_runtime_mixed_batch(
        &self,
        batch: &ferrum_interfaces::BatchPlan,
        prefill_ids: &[RequestId],
        decode_ids: &[RequestId],
    ) -> Result<MixedBatchDisposition> {
        let mut prefills = Vec::with_capacity(prefill_ids.len());
        for rid in prefill_ids {
            let scheduled = batch
                .requests
                .iter()
                .find(|item| item.request.id == *rid)
                .ok_or_else(|| {
                    FerrumError::internal(format!(
                        "PlanRuntime mixed batch {:?} lost scheduled request {rid}",
                        batch.batch_id
                    ))
                })?;
            if let Some(input) = self.prepare_plan_runtime_prefill(scheduled)? {
                prefills.push(input);
            }
        }
        let decodes = self.prepare_plan_runtime_decodes(decode_ids, false);
        let prefill_ids = prefills
            .iter()
            .map(|input| input.request_id.clone())
            .collect::<Vec<_>>();
        let decode_ids = decodes
            .iter()
            .map(|input| input.request_id.clone())
            .collect::<Vec<_>>();
        if prefills.is_empty() || decodes.is_empty() {
            return Ok(MixedBatchDisposition::Split {
                prefill_ids,
                decode_ids,
            });
        }

        // A zero-submission fallback still belongs to the decode scheduling
        // interval. Close that interval only once mixed execution completes.
        let started_at = self
            .config
            .runtime
            .profile_detail
            .captures_engine_token_timing()
            .then(Instant::now);
        let (prefill_outputs, decode_outputs) = match self
            .model_executor
            .plan_runtime_mixed_batch_with_capacity(&prefills, &decodes)
            .await?
        {
            PlanRuntimeMixedBatchOutcome::Completed { prefills, decodes } => (prefills, decodes),
            PlanRuntimeMixedBatchOutcome::Unsupported => {
                return Ok(MixedBatchDisposition::Split {
                    prefill_ids,
                    decode_ids,
                });
            }
            PlanRuntimeMixedBatchOutcome::NotSubmitted(deferral) => {
                let request_ids = prefill_ids
                    .iter()
                    .chain(&decode_ids)
                    .cloned()
                    .collect::<Vec<_>>();
                // Maintenance tickets exclude their exact frontiers from this
                // iteration. Other zero-submit capacity results can narrow or
                // split through the existing stage-specific controllers.
                let affected = match deferral {
                    ExecutorExecutionDeferral::RequestState(deferral) => {
                        let affected = deferral.request_ids().to_vec();
                        if affected.iter().any(|rid| !request_ids.contains(rid)) {
                            return Err(FerrumError::internal(
                                "mixed batch deferral names another frontier",
                            ));
                        }
                        self.defer_for_request_state_readiness(deferral).await?;
                        self.write_scheduler_trace_event(serde_json::json!({
                            "event": "engine_plan_runtime_mixed_batch_not_submitted",
                            "reason": "request_state",
                            "affected_request_ids": affected,
                        }));
                        affected
                    }
                    ExecutorExecutionDeferral::Capacity(deferral) => {
                        self.write_scheduler_trace_event(serde_json::json!({
                            "event": "engine_plan_runtime_mixed_batch_not_submitted",
                            "reason": "capacity",
                            "stage": deferral.stage(),
                            "request_ids": request_ids,
                            "shortfalls": deferral.shortfalls(),
                        }));
                        if let Some(retry) =
                            deferral.validated_maintenance_retry_scope(&request_ids)?
                        {
                            let affected = retry.affected_request_ids().to_vec();
                            let receipt = self
                                .scheduler
                                .defer_retry_after_execution_maintenance(retry)?;
                            if receipt.deferred_count() != affected.len() {
                                return Err(FerrumError::scheduler(
                                    "mixed batch maintenance retry lost scheduler entries",
                                ));
                            }
                            affected
                        } else {
                            Vec::new()
                        }
                    }
                };
                return Ok(MixedBatchDisposition::Split {
                    prefill_ids: unaffected_maintenance_retry_frontiers(&prefill_ids, &affected),
                    decode_ids: unaffected_maintenance_retry_frontiers(&decode_ids, &affected),
                });
            }
        };
        let completed_at = started_at.map(|_| Instant::now());
        if let (Some(started_at), Some(completed_at)) = (started_at, completed_at) {
            self.record_plan_runtime_decode_execution(&decode_ids, started_at, completed_at);
        }

        // Validate both phases before publishing any token or cache frontier.
        let validation = (|| {
            if prefill_outputs.len() != prefills.len() {
                return Err(FerrumError::internal(format!(
                    "PlanRuntime mixed batch returned {} prefill outputs for {} inputs",
                    prefill_outputs.len(),
                    prefills.len()
                )));
            }
            for (input, output) in prefills.iter().zip(&prefill_outputs) {
                output.validate_for(
                    &input.request_id,
                    input.chunk,
                    self.model_executor.info().vocab_size,
                )?;
            }
            self.validate_plan_runtime_decode_outputs(&decodes, &decode_outputs)
        })();
        if let Err(error) = validation {
            let cleanup = self.discard_plan_runtime_prefill_completions(prefill_outputs);
            for input in &prefills {
                self.model_executor
                    .cancel_prefill_admission(&input.request_id);
            }
            if let Err(cleanup) = cleanup {
                return Err(FerrumError::internal(format!(
                    "{error}; mixed prefill cleanup failed: {cleanup}"
                )));
            }
            return Err(error);
        }

        // Stream the already completed decode tokens before prefill sampling.
        if let Err(error) = self
            .commit_plan_runtime_decode_outputs(&decode_ids, decode_outputs, completed_at)
            .await
        {
            if let Err(cleanup) = self.discard_plan_runtime_prefill_completions(prefill_outputs) {
                return Err(FerrumError::internal(format!(
                    "{error}; mixed prefill cleanup failed: {cleanup}"
                )));
            }
            return Err(error);
        }
        let mut prefill_outputs = prefill_outputs.into_iter();
        for input in prefills {
            let output = prefill_outputs
                .next()
                .expect("validated mixed prefill cardinality");
            if let Err(error) = self
                .commit_plan_runtime_prefill_completion(
                    &input.request_id,
                    input.input_tokens.len(),
                    input.chunk,
                    output,
                )
                .await
            {
                if let Err(cleanup) = self.discard_plan_runtime_prefill_completions(prefill_outputs)
                {
                    return Err(FerrumError::internal(format!(
                        "{error}; remaining mixed prefill cleanup failed: {cleanup}"
                    )));
                }
                return Err(error);
            }
        }
        self.scheduler
            .record_decode_execution_capacity_success(decode_ids.len());
        self.write_scheduler_trace_event(serde_json::json!({
            "event": "engine_plan_runtime_mixed_batch_completed",
            "prefill_request_ids": prefill_ids,
            "decode_request_ids": decode_ids,
        }));
        Ok(MixedBatchDisposition::Completed)
    }
}
