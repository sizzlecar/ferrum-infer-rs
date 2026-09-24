use super::*;
use ferrum_interfaces::model_executor::{ExecutorPrefillCompletion, ExecutorPrefillOutcome};
use ferrum_interfaces::KvCacheHandle;

#[cfg(test)]
mod tests;

impl LlmExecutor {
    pub(super) fn execute_bounded_incremental_prefill(
        &self,
        input: &PrefillInput,
    ) -> Result<ExecutorPrefillOutcome> {
        let request_id = input.request_id.as_ref().ok_or_else(|| {
            FerrumError::request_validation("bounded prefill requires a request identity")
        })?;
        let chunk = input.chunk.ok_or_else(|| {
            FerrumError::request_validation("bounded prefill requires an exact chunk")
        })?;
        if input.batch_size() != 1 || input.attention_mask.is_some() || input.position_ids.is_some()
        {
            return Err(FerrumError::unsupported(
                "bounded legacy prefill requires one sequence with model-derived mask and positions",
            ));
        }
        if input.recurrent_state.is_some() {
            return Err(FerrumError::unsupported(
                "bounded legacy prefill does not declare recurrent-state continuation",
            ));
        }
        let tokens = common::tensor_to_tokens(&input.input_ids)?;
        if tokens.len() != chunk.total_prompt_tokens() {
            return Err(FerrumError::request_validation(
                "bounded prefill tensor must contain the exact full prompt",
            ));
        }
        let maximum = input.maximum_sequence_tokens.ok_or_else(|| {
            FerrumError::request_validation("bounded prefill requires a sequence extent")
        })?;
        if maximum < tokens.len() {
            return Err(FerrumError::request_validation(
                "bounded prefill prompt exceeds sequence extent",
            ));
        }
        let prior_handle = input
            .kv_cache
            .as_ref()
            .and_then(|handle| handle.as_any().downcast_ref::<GenericKvCacheHandle>());
        if chunk.tokens_processed() > 0 && prior_handle.is_none() {
            return Err(FerrumError::request_validation(
                "bounded prefill continuation requires the preceding model cache handle",
            ));
        }
        if prior_handle
            .is_some_and(|handle| handle.block_table().sequence_length != chunk.tokens_processed())
        {
            return Err(FerrumError::request_validation(
                "bounded prefill chunk starts outside its cache handle frontier",
            ));
        }
        let cache_id = prior_handle
            .map(|handle| handle.request_cache_id().to_owned())
            .unwrap_or_else(|| self.gen_cache_id());
        let mut model = self.lock_model();
        if !model.supports_bounded_incremental_prefill() {
            return Err(FerrumError::unsupported(
                "model configuration does not declare bounded KV-only prefill without implicit prefix import",
            ));
        }
        let context = tokens.iter().copied().map(TokenId::new).collect::<Vec<_>>();
        if model.recurrent_state_spec(request_id, &context)?.is_some() {
            return Err(FerrumError::unsupported(
                "bounded legacy prefill cannot ignore model recurrent-state requirements",
            ));
        }
        if chunk.end() > model.kv_capacity() {
            return Err(FerrumError::resource_exhausted(
                "bounded prefill chunk exceeds model KV capacity",
            ));
        }
        if model.incremental_prefill_cache_len(&cache_id)? != chunk.tokens_processed() {
            return Err(FerrumError::request_validation(
                "bounded prefill chunk starts outside actual model KV frontier",
            ));
        }
        let execution = (|| {
            model.set_lora_adapter_for_cache(
                &cache_id,
                active_lora_from_metadata(&input.metadata)?,
            )?;
            if chunk.tokens_processed() == 0 {
                if let Some(hint) = metadata_kv_capacity_hint(&input.metadata) {
                    model.prepare_kv_capacity(&cache_id, hint);
                }
            }
            model.reserve_kv_slots(&[KvSlotRequest {
                cache_id: cache_id.clone(),
                target_len: chunk.end(),
                admission_target_len: Some(maximum),
            }])?;
            // This path deliberately invokes exactly the requested token
            // range. Model capability covers continuation positions and
            // terminal host readback; no unified serial fallback is involved.
            let logits = model.prefill(&cache_id, &tokens[chunk.range()]);
            let observed = model.incremental_prefill_cache_len(&cache_id)?;
            if observed != chunk.end() {
                return Err(FerrumError::backend(format!(
                    "bounded prefill completed actual KV offset {observed}, expected {}",
                    chunk.end()
                )));
            }
            if logits.len() != model.config().vocab_size {
                return Err(FerrumError::backend(
                    "bounded prefill did not return full vocabulary logits",
                ));
            }
            let tensor = candle_core::Tensor::new(&logits[..], &candle_core::Device::Cpu)
                .and_then(|tensor| tensor.reshape((1, 1, logits.len())))
                .map_err(|error| FerrumError::model(format!("bounded prefill logits: {error}")))?;
            let config = model.config();
            let cache = Arc::new(GenericKvCacheHandle::new(
                config.num_layers,
                config.num_kv_heads,
                config.head_dim,
                candle_core::Device::Cpu,
                observed,
                cache_id.clone(),
            ));
            Ok(ExecutorPrefillOutcome::Completed(
                ExecutorPrefillCompletion::exact(
                    PrefillOutput::new(common::wrap_tensor(tensor), cache),
                    chunk,
                ),
            ))
        })();
        if execution.is_err() && prior_handle.is_none() {
            // Fresh model state has not yet escaped in an output handle. Its
            // owner must clean it even though the engine only holds a KV lease.
            model.release(&cache_id);
        }
        execution
    }
}
