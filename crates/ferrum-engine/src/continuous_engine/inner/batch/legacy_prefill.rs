//! Explicit incremental prefill for legacy executors with observed KV progress.
use super::*;
use ferrum_interfaces::model_executor::{ExecutorPrefillOutcome, PrefillChunk, PrefillInput};

pub(in crate::continuous_engine) enum LegacyPrefillWaveOutcome {
    Completed {
        request_id: RequestId,
        planned_chunk: PrefillChunk,
        completed_chunk: PrefillChunk,
        capacity_probe_count: u32,
    },
    NotSubmitted(ExecutorExecutionDeferral),
}

impl EngineInner {
    pub(super) async fn process_one_legacy_prefill_wave(
        &self,
        batch: &ferrum_interfaces::BatchPlan,
    ) -> Result<()> {
        let (prefill_ids, decode_ids) = self.classify_published_batch_sequences(batch)?;
        if !decode_ids.is_empty() {
            return Err(FerrumError::unsupported(
                "legacy SLO execution does not yet declare a bounded decode wave",
            ));
        }
        let Some(rid) = prefill_ids.first() else {
            return Ok(());
        };
        let scheduled = batch
            .requests
            .iter()
            .find(|item| item.request.id == *rid)
            .expect("classified scheduler batch member");
        let observation = match self.execute_legacy_prefill_wave(scheduled).await? {
            LegacyPrefillWaveOutcome::Completed {
                request_id,
                planned_chunk,
                completed_chunk,
                capacity_probe_count,
            } => serde_json::json!({
                "disposition": "completed", "request_id": request_id,
                "planned_chunk": planned_chunk, "completed_chunk": completed_chunk,
                "capacity_probe_count": capacity_probe_count,
            }),
            LegacyPrefillWaveOutcome::NotSubmitted(deferral) => serde_json::json!({
                "disposition": "not_submitted", "reason": format!("{deferral:?}"),
            }),
        };
        self.write_scheduler_trace_event(serde_json::json!({
            "event": "engine_bounded_legacy_prefill", "observation": observation,
        }));
        Ok(())
    }

    /// One authorized chunk. Resource allocation happens once, intermediate
    /// chunks retain it, and only the final chunk samples the first token.
    /// Unsupported recurrent/prefix combinations are rejected before mutation.
    pub(in crate::continuous_engine) async fn execute_legacy_prefill_wave(
        &self,
        scheduled: &ferrum_interfaces::scheduler::ScheduledRequest,
    ) -> Result<LegacyPrefillWaveOutcome> {
        if self.model_executor.execution_resource_authority()
            == ExecutionResourceAuthority::PlanRuntime
            || !self.model_executor.supports_bounded_incremental_prefill()
        {
            return Err(FerrumError::unsupported(
                "executor does not declare bounded legacy prefill",
            ));
        }
        if self.runtime_config.prefix_cache_enabled {
            return Err(FerrumError::unsupported(
                "bounded legacy prefill requires prefix import to be disabled",
            ));
        }
        let rid = &scheduled.request.id;
        let (context, maximum, resources, metadata) = {
            let sequences = self.sequences.read();
            let sequence = sequences.get(rid).ok_or_else(|| {
                FerrumError::scheduler("bounded legacy prefill has no published sequence")
            })?;
            if sequence.prefill_complete
                || sequence.prefill_tokens_processed != scheduled.tokens_processed
            {
                return Err(FerrumError::scheduler(
                    "bounded legacy prefill has a stale phase or offset",
                ));
            }
            (
                sequence.prefill_context_tokens(),
                sequence.model_maximum_sequence_tokens(),
                sequence.prefill_resources(),
                sequence.model_decode_metadata(),
            )
        };
        if resources.recurrent_state.is_some()
            || self
                .model_executor
                .recurrent_state_spec(rid, &context)?
                .is_some()
        {
            return Err(FerrumError::unsupported(
                "bounded legacy prefill does not support recurrent continuation",
            ));
        }
        let chunk = PrefillChunk::new(
            scheduled.tokens_processed,
            scheduled.tokens_to_process.ok_or_else(|| {
                FerrumError::scheduler("bounded legacy prefill requires a scheduled budget")
            })?,
            context.len(),
        )?;
        let tensor =
            self.tokens_to_tensor(&context.iter().map(|token| token.get()).collect::<Vec<_>>())?;
        let mut fresh_lease = None;
        let (cache, allocation) = match (resources.kv_cache, resources.legacy_kv_allocation) {
            (Some(cache), Some(allocation)) => (cache, allocation),
            (None, None) if chunk.tokens_processed() == 0 => {
                let model = self.model_executor.info();
                let request = AllocationRequest {
                    request_id: rid.clone(),
                    initial_tokens: context.len(),
                    max_sequence_length: model.max_sequence_length,
                    num_layers: model.num_layers,
                    num_heads: model.num_kv_heads,
                    head_dim: model.hidden_size / model.num_heads.max(1),
                    device: self.config.backend.device.clone(),
                    dtype: model.dtype,
                    priority: Priority::Normal,
                };
                let lease = self
                    .allocate_kv_lease(rid, rid.clone(), &request, context.len())
                    .await?;
                let cache = lease.handle();
                let allocation =
                    SequenceKvAllocation::new(lease.allocation_request_id.clone(), lease.blocks());
                fresh_lease = Some(lease);
                (cache, allocation)
            }
            _ => {
                return Err(FerrumError::internal(
                    "bounded legacy prefill lost its paired model cache and KV allocation",
                ))
            }
        };
        let input = PrefillInput::new(tensor)
            .with_request_context(rid.clone(), maximum)
            .with_chunk(chunk)
            .with_kv_cache(cache.clone())
            .with_metadata(metadata);
        let outcome = self
            .model_executor
            .bounded_incremental_prefill(&input)
            .await;
        let completion = match outcome {
            Ok(ExecutorPrefillOutcome::Deferred(deferral)) => {
                if let Some(lease) = fresh_lease {
                    lease.release(self).await;
                }
                return Ok(LegacyPrefillWaveOutcome::NotSubmitted(deferral));
            }
            Ok(ExecutorPrefillOutcome::Completed(completion)) => completion,
            Err(error) => {
                if let Some(lease) = fresh_lease {
                    lease.release(self).await;
                }
                self.complete_request_with_error(rid, FerrumError::backend(error.to_string()))
                    .await?;
                return Err(error);
            }
        };
        let (output, planned_chunk, completed_chunk, capacity_probe_count) =
            completion.into_parts();
        let fresh_model_cache = fresh_lease.is_some();
        let mut published = false;
        let result = async {
            if planned_chunk != chunk
                || output.kv_cache.block_table().sequence_length != completed_chunk.end()
                || output.recurrent_state.is_some()
                || (!fresh_model_cache && output.kv_cache.cache_id() != cache.cache_id())
            {
                return Err(FerrumError::backend(
                    "bounded legacy prefill returned an inconsistent chunk/cache receipt",
                ));
            }
            let logits = if completed_chunk.is_final() {
                let logits = output.last_token_logits()?.to_vec_f32()?;
                if logits.len() != self.model_executor.info().vocab_size {
                    return Err(FerrumError::backend(
                        "bounded final prefill requires full vocabulary logits",
                    ));
                }
                Some(logits)
            } else {
                None
            };
            let (token, update) = {
                let mut sequences = self.sequences.write();
                let sequence = sequences.get_mut(rid).ok_or_else(|| {
                    FerrumError::scheduler("bounded legacy prefill completed after request removal")
                })?;
                let publication = if logits.is_some() {
                    Some(self.scheduler.prepare_prefill_output_publication(
                        rid,
                        context.len(),
                        sequence.generated_tokens.len(),
                    )?)
                } else {
                    None
                };
                let token = if let Some(mut logits) = logits {
                    sequence.reset_guided_processors()?;
                    Some(sequence.sample_and_commit_with_processors_and_tokenizer(
                        &mut logits,
                        Some(self.tokenizer.as_ref()),
                    )?)
                } else {
                    None
                };
                let update = sequence.commit_prefill_chunk_physical_resources(
                    output.kv_cache.clone(),
                    allocation,
                    None,
                    completed_chunk.end(),
                    completed_chunk.is_final(),
                );
                if token.is_some() {
                    sequence.record_generated_token_commit();
                }
                if let Some(publication) = publication {
                    self.finish_sampled_prefill(
                        sequence,
                        &publication,
                        context.len(),
                        Some((
                            chunk.tokens_to_process(),
                            completed_chunk.tokens_to_process(),
                        )),
                    )?;
                }
                (token, update)
            };
            published = true;
            if let Some(lease) = fresh_lease.take() {
                let _ = lease.into_committed_parts();
            }
            self.apply_model_cache_ref_update(rid, update);
            let promoted = if token.is_some() {
                true
            } else {
                self.scheduler
                    .mark_prefill_chunk_processed_with_capacity_feedback(
                        rid,
                        context.len(),
                        chunk.tokens_to_process(),
                        completed_chunk.tokens_to_process(),
                    )?
            };
            if promoted != completed_chunk.is_final() {
                return Err(FerrumError::scheduler(
                    "bounded legacy prefill scheduler phase disagrees with completed chunk",
                ));
            }
            self.total_prefill_tokens.fetch_add(
                completed_chunk.tokens_to_process() as u64,
                Ordering::Relaxed,
            );
            counter!("ferrum.engine.prefill_tokens_total")
                .increment(completed_chunk.tokens_to_process() as u64);
            if let Some(token) = token {
                counter!("ferrum.engine.prefills_total").increment(1);
                let stop_reason = self.stop_reason_for_request(rid);
                if self.should_stream_generated_token(rid, token, stop_reason) {
                    self.send_stream_update(rid, token).await;
                }
                if let Some(reason) = stop_reason {
                    self.complete_request(rid, reason).await?;
                }
            }
            Ok(LegacyPrefillWaveOutcome::Completed {
                request_id: rid.clone(),
                planned_chunk,
                completed_chunk,
                capacity_probe_count,
            })
        }
        .await;
        if let Err(error) = result {
            if fresh_model_cache && !published {
                self.model_executor
                    .release_cache(&output.kv_cache.cache_id());
            }
            if let Some(lease) = fresh_lease {
                lease.release(self).await;
            }
            self.complete_request_with_error(rid, FerrumError::backend(error.to_string()))
                .await?;
            return Err(error);
        }
        result
    }
}
