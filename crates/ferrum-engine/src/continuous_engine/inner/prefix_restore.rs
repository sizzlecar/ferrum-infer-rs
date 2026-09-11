use super::*;
use ferrum_interfaces::model_executor::PlanRuntimePrefixRestoreInput;

impl EngineInner {
    /// Called under the existing iteration lock, after admission and before
    /// publishing a batch. Submit/replacement uses the same lock, so one
    /// scheduler preparation cannot be applied to another outer sequence.
    pub(super) async fn restore_admitted_prefixes(&self) -> Result<()> {
        let candidates = self
            .sequences
            .read()
            .iter()
            .filter(|(_, sequence)| {
                !sequence.prefill_complete && sequence.prefill_tokens_processed == 0
            })
            .map(|(id, sequence)| (id.clone(), sequence.prefill_context_len()))
            .collect::<Vec<_>>();

        for (request_id, prompt_tokens) in candidates {
            let Some(prepared) =
                self.scheduler
                    .prepare_prefix_restore(&request_id, 0, prompt_tokens)?
            else {
                continue;
            };
            let Some((tokens, maximum_sequence_tokens)) =
                self.sequences.read().get(&request_id).map(|sequence| {
                    (
                        sequence.prefill_context_tokens(),
                        sequence.model_maximum_sequence_tokens(),
                    )
                })
            else {
                continue;
            };
            let started = Instant::now();
            let output = match self
                .model_executor
                .try_restore_plan_runtime_prefix(PlanRuntimePrefixRestoreInput {
                    request_id: &request_id,
                    input_tokens: &tokens,
                    maximum_sequence_tokens,
                })
                .await
            {
                Ok(Some(output)) => output,
                Ok(None) => continue,
                Err(error) => {
                    drop(prepared);
                    self.complete_request_with_error(&request_id, error).await?;
                    continue;
                }
            };
            if let Err(error) = output.validate_for(&request_id, prompt_tokens) {
                drop(output);
                drop(prepared);
                self.complete_request_with_error(&request_id, error).await?;
                continue;
            }
            let restored_tokens = output.restored_tokens();
            let committed = (|| {
                let mut sequences = self.sequences.write();
                let sequence = sequences.get_mut(&request_id).ok_or_else(|| {
                    FerrumError::internal("prefix restore lost its outer sequence")
                })?;
                if sequence.prefill_tokens_processed != 0
                    || sequence.prefill_complete
                    || sequence.prefill_context_tokens() != tokens
                {
                    return Err(FerrumError::internal(
                        "prefix restore outer sequence changed before publication",
                    ));
                }
                self.scheduler
                    .commit_prefix_restored(prepared, restored_tokens)?;
                let update = sequence.commit_plan_runtime_prefill_chunk_resources(
                    Arc::clone(output.kv_cache()),
                    restored_tokens,
                    false,
                );
                // Both outer owners now agree. Only the native publication
                // may open the target execution gate. The closure also drops
                // that owner before error cleanup awaits request release.
                Ok((update, output.acknowledge()))
            })();
            let (update, acknowledged) = match committed {
                Ok(value) => value,
                Err(error) => {
                    self.complete_request_with_error(&request_id, error).await?;
                    continue;
                }
            };
            self.apply_model_cache_ref_update(&request_id, update);
            if let Err(error) = acknowledged {
                self.complete_request_with_error(&request_id, error).await?;
                continue;
            }
            self.prefix_cache_hits.fetch_add(1, Ordering::Relaxed);
            counter!("ferrum.engine.prefix_cache_hits").increment(1);
            counter!("ferrum.engine.prefix_cache_tokens_total").increment(restored_tokens as u64);
            self.write_executor_scheduler_profile_event(
                &request_id,
                "vnext.prefix_restore",
                ProfileEventKind::TimedSpan,
                ProfileStatus::Ok,
                Some(duration_to_us(started.elapsed())),
                BTreeMap::from([
                    (
                        "restored_tokens".to_string(),
                        serde_json::json!(restored_tokens),
                    ),
                    (
                        "prompt_tokens".to_string(),
                        serde_json::json!(prompt_tokens),
                    ),
                ]),
                BTreeMap::new(),
                None,
            );
        }
        Ok(())
    }
}
