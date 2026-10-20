//! Prompt-bound prefix references belong to the original model cache.
//!
//! Admission pins storage without consuming tokens. Successful token work
//! advances the physical cursor and publishes hashes of the immutable prompt.

use std::sync::Arc;

use ferrum_execution::model_executor::{
    KvSlotAllocation, KvSlotCapacitySnapshot, KvSlotReservation,
};
use ferrum_execution::token_execution::TokenAdmissionCapacityDeferral;
use ferrum_kernels::backend::{KvLayer, MoeLlmBackend};
use ferrum_types::{FerrumError, Result, TokenId};

use crate::common::llm::{ModelKvPreparation, ModelKvReservation, ModelTokenWork};
use crate::common::paged_pool::{block_hash_chain, BlockHash};

use super::{DeviceFrameResult, LlamaFamilyModel};

pub(super) struct PreparedPrefix {
    prompt: Arc<[u32]>,
    hashes: Vec<BlockHash>,
    reusable_tokens: usize,
    published_blocks: usize,
    recorded_hit: bool,
}

impl<B: MoeLlmBackend, K: KvLayer<B>> LlamaFamilyModel<B, K> {
    /// The retained numerical entrypoints do not carry a token admission.
    /// Once a cache *does* have one, they must preserve its immutable prompt
    /// and may never encode into a pinned shared span. Unprepared numerical
    /// fixtures and pipeline stage caches retain their original contract.
    pub(in crate::models) fn validate_prepared_legacy_write(
        &self,
        cache_id: &str,
        position: usize,
        count: usize,
        tokens: Option<&[u32]>,
    ) -> Result<()> {
        let Some(prefix) = self.prepared_prefixes.get(cache_id) else {
            return Ok(());
        };
        self.ensure_preparation_available()?;
        let end = position.checked_add(count).ok_or_else(|| {
            FerrumError::request_validation("prepared numerical write range overflow")
        })?;
        if count == 0
            || end > self.kv_capacity_limit()
            || self
                .kv_caches
                .get(cache_id)
                .is_none_or(|layers| layers.iter().any(|cache| K::len(cache) != position))
        {
            return Err(FerrumError::request_validation(
                "numerical write does not match the prepared per-layer KV frontier",
            ));
        }
        if position < prefix.reusable_tokens {
            return Err(FerrumError::request_validation(
                "prepared shared prefix must be consumed through token execution",
            ));
        }
        if position < prefix.prompt.len()
            && !tokens.is_some_and(|actual| {
                actual.len() == count
                    && prefix
                        .prompt
                        .get(position..end)
                        .is_some_and(|expected| expected == actual)
            })
        {
            return Err(FerrumError::request_validation(
                "numerical write cannot prove the immutable prepared prompt",
            ));
        }
        Ok(())
    }

    pub(in crate::models) fn validate_prepared_legacy_decode(
        &self,
        cache_id: &str,
        token: u32,
        position: u32,
    ) -> Result<()> {
        if self.prepared_prefixes.contains_key(cache_id) {
            self.validate_managed_token_work(cache_id, ModelTokenWork::Decode { token, position })?;
        }
        Ok(())
    }

    pub(super) fn validate_prepared_truncate(&self, cache_id: &str, new_len: usize) -> Result<()> {
        if self.prepared_prefixes.contains_key(cache_id)
            && self
                .kv_caches
                .get(cache_id)
                .is_some_and(|layers| layers.iter().any(|cache| new_len < K::len(cache)))
        {
            return Err(FerrumError::request_validation(
                "a prepared token incarnation cannot rewind its committed prefix",
            ));
        }
        Ok(())
    }

    pub(super) fn prepare_token_kv_in_frame(
        &mut self,
        request: ModelKvPreparation<'_>,
    ) -> Result<ModelKvReservation> {
        self.prepare_prompt_storage(request, true)
    }

    /// Layer-split stages process their supplied hidden states in full. They
    /// must not independently reuse or publish a token-prefix mapping.
    pub(in crate::models) fn prepare_pipeline_token_kv(
        &mut self,
        request: ModelKvPreparation<'_>,
    ) -> Result<ModelKvReservation> {
        self.prepare_prompt_storage(request, false)
    }

    fn prepare_prompt_storage(
        &mut self,
        request: ModelKvPreparation<'_>,
        allow_prefix: bool,
    ) -> Result<ModelKvReservation> {
        self.ensure_preparation_available()?;
        if request.prompt.is_empty()
            || request.prompt.len() > request.capacity_hint
            || request.capacity_hint > self.kv_capacity_limit()
            || request.capacity_hint > u32::MAX as usize
            || request
                .prompt
                .iter()
                .any(|&token| token as usize >= self.cfg.vocab_size)
        {
            return Err(FerrumError::request_validation(
                "invalid immutable prompt or KV capacity",
            ));
        }
        if self
            .kv_caches
            .get(request.cache_id)
            .is_some_and(|layers| layers.iter().any(|cache| K::len(cache) != 0))
        {
            return Err(FerrumError::request_validation(
                "preparation cannot replace a live KV frontier",
            ));
        }
        if let Some(prepared) = self.prepared_prefixes.get(request.cache_id) {
            if prepared.prompt.as_ref() != request.prompt.as_ref()
                || self
                    .lora_cache_adapters
                    .get(request.cache_id)
                    .map(String::as_str)
                    != request.adapter.map(|adapter| adapter.name.as_str())
            {
                return Err(FerrumError::request_validation(
                    "preparation cannot replace a bound prompt or adapter",
                ));
            }
            return self.reserve_paged_token_kv_slots(&[super::KvSlotRequest {
                cache_id: request.cache_id.to_owned(),
                target_len: request.prompt.len(),
                admission_target_len: Some(request.prompt.len()),
            }]);
        }

        // A base prefix can never be acquired before the adapter's semantics
        // have been validated. Prefix-enabled LoRA remains explicitly rejected.
        if let Some(adapter) = request.adapter {
            self.ensure_lora_adapter_loaded(adapter.clone())?;
        }
        let use_prefix = allow_prefix && self.runtime_env.prefix_cache;
        let reservation = if self.kv_caches.contains_key(request.cache_id) && !use_prefix {
            self.reserve_paged_token_kv_slots(&[super::KvSlotRequest {
                cache_id: request.cache_id.to_owned(),
                target_len: request.prompt.len(),
                admission_target_len: Some(request.prompt.len()),
            }])
        } else {
            match self.run_preparation_frame(|model, frame| {
                model.ensure_kv_in_context(
                    &mut frame.context,
                    request.cache_id,
                    Some(request.capacity_hint),
                );
                if use_prefix && model.paged_pools.is_some() {
                    model.prepare_prefix_mapping(&mut frame.context, request)
                } else {
                    model.reserve_paged_token_kv_slots_in_context(
                        Some(&mut frame.context),
                        &[super::KvSlotRequest {
                            cache_id: request.cache_id.to_owned(),
                            target_len: request.prompt.len(),
                            admission_target_len: Some(request.prompt.len()),
                        }],
                    )
                }
            }) {
                DeviceFrameResult::Terminal(result) => result?,
                DeviceFrameResult::Rejected(error) | DeviceFrameResult::Held(error) => Err(error),
            }
        };
        // Storage preparation does not execute adapter projections. Publish
        // its binding only after reservation succeeds; a deferred or failed
        // attempt must neither add a binding nor erase a prior one. Unknown
        // device work still retains its original cache/frame for checked close.
        if matches!(&reservation, Ok(ModelKvReservation::Reserved(_))) {
            if let Some(adapter) = request.adapter {
                self.lora_cache_adapters
                    .insert(request.cache_id.to_owned(), adapter.name.clone());
            } else {
                self.lora_cache_adapters.remove(request.cache_id);
            }
        }
        reservation
    }

    fn prepare_prefix_mapping(
        &mut self,
        ctx: &mut B::Context,
        request: ModelKvPreparation<'_>,
    ) -> Result<ModelKvReservation> {
        let caches = self
            .kv_caches
            .get(request.cache_id)
            .ok_or_else(|| FerrumError::model("prefix cache is missing"))?;
        let first = caches
            .first()
            .ok_or_else(|| FerrumError::model("prefix cache has no layers"))?;
        let block_size = K::block_size(first);
        if block_size == 0
            || caches.iter().any(|cache| {
                K::block_size(cache) != block_size
                    || K::len(cache) != 0
                    || !K::paged_block_indices(cache).is_empty()
            })
        {
            return Err(FerrumError::request_validation(
                "prefix preparation requires an empty, consistent paged mapping",
            ));
        }
        let table_capacity = K::capacity(first) / block_size;
        let total_required = request.prompt.len().div_ceil(block_size);
        if request.capacity_hint.div_ceil(block_size) > table_capacity {
            return Err(FerrumError::request_validation(
                "prompt capacity exceeds the per-sequence block table",
            ));
        }
        let token_ids: Vec<_> = request.prompt.iter().copied().map(TokenId::new).collect();
        let hashes = block_hash_chain(&token_ids, block_size);
        let reusable_limit = request.prompt.len().saturating_sub(1) / block_size;
        let allocator = self
            .paged_block_alloc
            .as_ref()
            .ok_or_else(|| FerrumError::model("prefix allocator is missing"))?;
        let mut allocator = allocator
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let total_blocks = allocator.capacity() as usize;
        if total_required > total_blocks {
            return Err(FerrumError::resource_exhausted(
                "prompt exceeds the physical paged KV pool",
            ));
        }
        let free_before = allocator.free_count();
        let mut mapping = Vec::with_capacity(total_required);
        for &hash in hashes.iter().take(reusable_limit) {
            match allocator.try_acquire_by_hash(hash) {
                Some(block) => mapping.push(block),
                None => break,
            }
        }
        let matched = mapping.len();
        let free_pinned = allocator.free_count();
        let new_blocks = total_required - matched;
        if new_blocks > free_pinned {
            // Returning a deferral must restore every speculative shared ref.
            // Revived soft-free blocks become free again without losing hashes.
            allocator.free(&mapping);
            let revived = free_before - free_pinned;
            return Ok(ModelKvReservation::CapacityDeferred(
                TokenAdmissionCapacityDeferral::new(
                    allocator.capacity_domain().clone(),
                    new_blocks + revived,
                    KvSlotCapacitySnapshot {
                        block_size,
                        total_blocks,
                        free_blocks: allocator.free_count(),
                    },
                )?,
            ));
        }
        let private = match allocator.allocate_n(new_blocks) {
            Ok(blocks) => blocks,
            Err(error) => {
                allocator.free(&mapping);
                return Err(error);
            }
        };
        mapping.extend(private);
        let free_after = allocator.free_count();
        drop(allocator);

        // Install every actual reference in the original owner before any
        // device table write can fail or become indeterminate.
        let caches = self
            .kv_caches
            .get_mut(request.cache_id)
            .expect("validated prefix cache");
        for cache in caches.iter_mut() {
            *K::paged_block_indices_mut(cache) = mapping.clone();
        }
        self.prepared_prefixes.insert(
            request.cache_id.to_owned(),
            PreparedPrefix {
                prompt: Arc::clone(request.prompt),
                hashes,
                reusable_tokens: matched * block_size,
                published_blocks: matched,
                recorded_hit: false,
            },
        );
        let mut padded = mapping;
        padded.resize(table_capacity, 0);
        for cache in caches.iter_mut() {
            if let Some(table) = K::block_table_mut(cache) {
                B::write_typed::<u32>(ctx, table, &padded);
            }
        }
        if matched == 0 {
            self.prefix_cache_misses += 1;
        }
        Ok(ModelKvReservation::Reserved(Some(KvSlotReservation {
            block_size,
            total_blocks,
            free_blocks_before: free_before,
            free_blocks_after: free_after,
            allocations: vec![KvSlotAllocation {
                cache_id: request.cache_id.to_owned(),
                blocks_before: matched,
                blocks_after: total_required,
                new_blocks,
            }],
        })))
    }

    pub(in crate::models) fn validate_prepared_prompt(
        &self,
        cache_id: &str,
        work: ModelTokenWork<'_>,
    ) -> Result<()> {
        let Some(prefix) = self.prepared_prefixes.get(cache_id) else {
            return Ok(());
        };
        let valid = match work {
            ModelTokenWork::Prefill { tokens, position } => position
                .checked_add(tokens.len())
                .and_then(|end| prefix.prompt.get(position..end))
                .is_some_and(|expected| expected == tokens),
            ModelTokenWork::Decode { position, .. } => position as usize >= prefix.prompt.len(),
        };
        if !valid {
            return Err(FerrumError::request_validation(
                "token work differs from the immutable prepared prompt",
            ));
        }
        Ok(())
    }

    pub(in crate::models) fn reusable_prefix_for_work(
        &self,
        cache_id: &str,
        work: ModelTokenWork<'_>,
    ) -> usize {
        let ModelTokenWork::Prefill { position, tokens } = work else {
            return 0;
        };
        self.prepared_prefixes.get(cache_id).map_or(0, |prefix| {
            prefix
                .reusable_tokens
                .saturating_sub(position)
                .min(tokens.len())
        })
    }

    pub(super) fn advance_cached_prefix(
        &mut self,
        ctx: &mut B::Context,
        cache_id: &str,
        new_len: usize,
    ) {
        let caches = self
            .kv_caches
            .get_mut(cache_id)
            .expect("validated prefix cache");
        for cache in caches {
            K::set_len(cache, new_len);
            if let Some(lengths) = K::context_lens_mut(cache) {
                B::write_typed::<u32>(ctx, lengths, &[new_len as u32]);
            }
        }
    }

    pub(in crate::models) fn publish_prepared_prefix(&mut self, cache_id: &str, reused: usize) {
        let Some(prefix) = self.prepared_prefixes.get_mut(cache_id) else {
            return;
        };
        let Some(first) = self
            .kv_caches
            .get(cache_id)
            .and_then(|layers| layers.first())
        else {
            return;
        };
        let Some(allocator) = self.paged_block_alloc.as_ref() else {
            return;
        };
        let block_size = K::block_size(first);
        let completed = (K::len(first) / block_size).min(prefix.hashes.len());
        let mut allocator = allocator
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        for index in prefix.published_blocks..completed {
            allocator
                .register_block_hash(K::paged_block_indices(first)[index], prefix.hashes[index]);
        }
        prefix.published_blocks = prefix.published_blocks.max(completed);
        if reused != 0 {
            if !prefix.recorded_hit {
                prefix.recorded_hit = true;
                self.prefix_cache_hits += 1;
            }
            self.prefix_cache_saved_prefill_tokens += reused as u64;
        }
    }
}
