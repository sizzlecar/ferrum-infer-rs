//! Scenario hooks for the existing scheduler fixtures. No Tensor or cache owner
//! is manufactured: all work carries the common protocol's original frontier.
use super::*;

fn plain_rows(inputs: &[FixtureInput<'_, '_>], token: usize, vocabulary: usize) -> FixtureOutcome {
    FixtureOutcome::Rows(
        inputs
            .iter()
            .map(|input| input.exact(token, vocabulary))
            .collect(),
    )
}

impl ControlledHooks for PlanRuntimeChunkedPrefillTestExecutor {
    fn work(&self, inputs: &[FixtureInput<'_, '_>]) -> Result<FixtureOutcome> {
        let prefills = inputs
            .iter()
            .filter(|input| matches!(input.work.work(), TokenWork::Prefill(_)))
            .count();
        if prefills == 0 {
            return Ok(plain_rows(inputs, 42, self.info().vocab_size));
        }
        if prefills != inputs.len() || (prefills > 1 && self.prompt_tail_reserve.is_none()) {
            return Ok(FixtureOutcome::Unsupported);
        }
        if prefills > 1 {
            self.tail_batch_prefill_calls
                .fetch_add(1, Ordering::Relaxed);
        }
        let mut rows = Vec::new();
        for input in inputs {
            let TokenWork::Prefill(chunk) = input.work.work() else {
                unreachable!()
            };
            if !self.retained.lock().unwrap().contains(input.request_id) {
                return Err(FerrumError::internal(
                    "chunked prefill submit has no retained admission",
                ));
            }
            self.attempted_chunks.lock().unwrap().push(chunk);
            if let Some(state) = &self.prefix_retry {
                state.computed(input.request_id, chunk);
            }
            if self.defer_next_prefill.swap(false, Ordering::AcqRel) {
                assert!(
                    rows.is_empty(),
                    "controlled deferral must precede every product"
                );
                return Ok(FixtureOutcome::Deferred(
                    test_execution_capacity_deferral(
                        self.epochs(),
                        self.wait_condition(),
                        ExecutorExecutionCapacityStage::StepAdmission,
                    )
                    .into(),
                ));
            }
            let completed = if self.narrow_next_prefill.swap(false, Ordering::AcqRel) {
                PrefillChunk::new(
                    chunk.tokens_processed(),
                    chunk.tokens_to_process().div_ceil(2),
                    chunk.total_prompt_tokens(),
                )?
            } else {
                chunk
            };
            if self.prompt_tail_reserve.is_some() {
                assert_eq!(
                    completed, chunk,
                    "planned tail cuts are not capacity failures"
                );
            }
            if let Some(state) = &self.rendezvous {
                state.retired(input.request_id, completed);
            }
            self.completed_chunks.fetch_add(1, Ordering::Relaxed);
            if completed.is_final() {
                assert!(self.retained.lock().unwrap().remove(input.request_id));
            }
            rows.push(input.row(
                ExecutedTokenWork::Prefill {
                    chunk: completed,
                    capacity_probe_count: u32::from(completed != chunk),
                },
                42,
                self.info().vocab_size,
            ));
        }
        Ok(FixtureOutcome::Rows(rows))
    }
}

impl ControlledHooks for PlanRuntimeBatchDecodeTestExecutor {
    fn work(&self, inputs: &[FixtureInput<'_, '_>]) -> Result<FixtureOutcome> {
        let prefills = inputs
            .iter()
            .filter(|input| matches!(input.work.work(), TokenWork::Prefill(_)))
            .count();
        if prefills != 0 && prefills != inputs.len() {
            return self.mock_mixed_tokens(inputs);
        }
        if prefills == inputs.len() {
            return Ok(plain_rows(inputs, 6, self.info().vocab_size));
        }
        let call = self.batch_decode_calls.fetch_add(1, Ordering::Relaxed);
        if matches!(self.behavior, PlanRuntimeBatchDecodeBehavior::Exact)
            || (matches!(
                self.behavior,
                PlanRuntimeBatchDecodeBehavior::DeferUntilPeerCacheRelease
            ) && self.released_cache_count.load(Ordering::Acquire) > 0)
        {
            return Ok(plain_rows(inputs, 6, self.info().vocab_size));
        }
        let mut availability = Vec::new();
        let observed = self
            .write_execution_capacity_snapshot(&mut availability)?
            .unwrap();
        let wait = ferrum_runtime::CapacityWaitCondition::from_observation(
            observed.coordinator_id.get(),
            availability,
        )
        .unwrap();
        Ok(FixtureOutcome::Deferred(
            test_execution_capacity_deferral(observed, wait, self.capacity_stage_for_call(call))
                .into(),
        ))
    }

    fn closed(&self, _request: &RequestId, _reason: ExecutionEndReason) -> Result<()> {
        self.released_cache_count.fetch_add(1, Ordering::Release);
        Ok(())
    }
}

impl ControlledHooks for PlanRuntimeAdmissionTestExecutor {
    fn work(&self, inputs: &[FixtureInput<'_, '_>]) -> Result<FixtureOutcome> {
        let prefills = inputs
            .iter()
            .filter(|input| matches!(input.work.work(), TokenWork::Prefill(_)))
            .count();
        if prefills == 0 {
            return Ok(plain_rows(inputs, 42, self.info().vocab_size));
        }
        if prefills != inputs.len() {
            return Ok(FixtureOutcome::Unsupported);
        }
        if prefills > 1 {
            self.batch_prefill_calls.fetch_add(1, Ordering::Relaxed);
            if self.defer_batch_prefill_once.swap(false, Ordering::AcqRel) {
                return Ok(FixtureOutcome::Deferred(
                    test_execution_capacity_deferral(
                        self.capacity_epochs(),
                        self.capacity_wait_condition(),
                        ExecutorExecutionCapacityStage::SubmissionWave,
                    )
                    .into(),
                ));
            }
        }
        for input in inputs {
            if !self.retained.lock().unwrap().remove(input.request_id) {
                return Err(FerrumError::internal(
                    "test prefill reached submit without retained admission",
                ));
            }
            if prefills == 1 {
                self.prefill_calls.fetch_add(1, Ordering::Relaxed);
                if self
                    .fail_next_prefill_after_admission
                    .swap(false, Ordering::AcqRel)
                {
                    return Err(FerrumError::resource_exhausted(
                        "typed diagnostic resource failure after prefill submission",
                    ));
                }
            }
        }
        Ok(plain_rows(inputs, 42, self.info().vocab_size))
    }

    fn closed(&self, request: &RequestId, reason: ExecutionEndReason) -> Result<()> {
        if let ExecutionEndReason::Completed {
            input_tokens,
            output_tokens,
        } = reason
        {
            self.completions
                .lock()
                .unwrap()
                .push(ExecutorSequenceCompletion::new(
                    request.clone(),
                    request.to_string(),
                    input_tokens,
                    output_tokens,
                )?);
        }
        Ok(())
    }
}

impl ControlledHooks for PlanRuntimeMaintenanceTestExecutor {
    fn work(&self, inputs: &[FixtureInput<'_, '_>]) -> Result<FixtureOutcome> {
        let prefills = inputs
            .iter()
            .filter(|input| matches!(input.work.work(), TokenWork::Prefill(_)))
            .count();
        if prefills == 0 {
            return Ok(plain_rows(inputs, 42, self.info().vocab_size));
        }
        if inputs.len() != 1 {
            return Ok(FixtureOutcome::Unsupported);
        }
        if !self.retained.lock().unwrap().remove(inputs[0].request_id) {
            return Err(FerrumError::internal(
                "test prefill reached submit without retained admission",
            ));
        }
        self.prefill_calls.fetch_add(1, Ordering::Relaxed);
        self.call_order.lock().unwrap().push("prefill");
        Ok(plain_rows(inputs, 42, self.info().vocab_size))
    }
}
