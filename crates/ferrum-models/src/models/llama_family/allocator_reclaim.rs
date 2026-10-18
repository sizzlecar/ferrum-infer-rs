//! Cold observation of the original paged allocator and cache mappings.
//! No sequence ownership or capacity ledger is duplicated here.

use super::*;
use ferrum_execution::token_execution::{AllocatorCapacityDomain, TokenAllocatorReclaim};
use ferrum_types::FerrumError;
use std::collections::HashSet;

impl<B: MoeLlmBackend, K: KvLayer<B>> LlamaFamilyModel<B, K> {
    pub(super) fn observe_paged_allocator_reclaim(
        &self,
        blocked_cache_id: &str,
        domain: &AllocatorCapacityDomain,
        candidates: &[&str],
    ) -> Result<Option<TokenAllocatorReclaim>> {
        let Some(allocator) = self.paged_block_alloc.as_ref() else {
            return Ok(None);
        };
        let allocator = allocator
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if allocator.capacity_domain() != domain {
            return Ok(None);
        }
        let blocked = self.kv_caches.get(blocked_cache_id).ok_or_else(|| {
            FerrumError::request_validation("allocator reclaim observation lost the blocked cache")
        })?;
        let first = blocked.first().ok_or_else(|| {
            FerrumError::model("allocator reclaim observation found no blocked cache layers")
        })?;
        let block_size = K::block_size(first);
        if block_size == 0 {
            return Err(FerrumError::model(
                "allocator reclaim observation requires actual paged cache storage",
            ));
        }
        let _blocked_blocks = self.reclaim_cache_blocks(blocked_cache_id, block_size)?;

        let mut distinct = HashSet::with_capacity(candidates.len());
        let mut mappings = Vec::with_capacity(candidates.len());
        for &candidate in candidates {
            if candidate == blocked_cache_id || !distinct.insert(candidate) {
                return Err(FerrumError::request_validation(
                    "allocator reclaim candidates must be distinct and exclude the blocked cache",
                ));
            }
            // A concurrently completed close can already have removed its
            // cache. Its blocks are accounted for by the current free count;
            // an empty slice keeps subsequent candidate indices unchanged.
            mappings.push(
                self.reclaim_cache_blocks(candidate, block_size)?
                    .unwrap_or(&[]),
            );
        }
        let observation = allocator.observe_reclaimable_blocks(mappings)?;
        Ok(Some(TokenAllocatorReclaim {
            snapshot: KvSlotCapacitySnapshot {
                block_size,
                total_blocks: allocator.capacity() as usize,
                free_blocks: allocator.free_count(),
            },
            reclaimable_blocks: observation.reclaimable_blocks,
            contributing_candidates: observation.contributing_candidates,
        }))
    }

    fn reclaim_cache_blocks(&self, cache_id: &str, block_size: usize) -> Result<Option<&[u32]>> {
        let Some(layers) = self.kv_caches.get(cache_id) else {
            return Ok(None);
        };
        let first = layers.first().ok_or_else(|| {
            FerrumError::model("allocator reclaim observation found an empty layer mapping")
        })?;
        let blocks = K::paged_block_indices(first);
        if layers.len() != self.layers.len()
            || layers.iter().any(|layer| {
                K::block_size(layer) != block_size || K::paged_block_indices(layer) != blocks
            })
        {
            return Err(FerrumError::model(
                "allocator reclaim observation requires one consistent block mapping across layers",
            ));
        }
        Ok(Some(blocks))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::llm::{ModelKvPreparation, ModelKvReservation};
    use crate::test_support::SyntheticLlamaLoader;
    use ferrum_kernels::backend::cpu::CpuBackend;

    // Real CPU pool allocation and table preparation; no CPU paged attention
    // kernel or numerical execution is claimed by this ownership observation.
    fn model() -> LlamaFamilyModel<CpuBackend, KvFp16> {
        let config = LlamaFamilyConfig {
            hidden_size: 128,
            intermediate_size: 256,
            num_heads: 2,
            num_kv_heads: 2,
            head_dim: 64,
            num_layers: 2,
            vocab_size: 128,
            max_seq_len: 32,
            ..Default::default()
        };
        let loader = SyntheticLlamaLoader::new(config.clone());
        let mut model = LlamaFamilyModel::new(config, &loader).unwrap();
        model.runtime_env.metal_paged_kv = Some(true);
        model.runtime_env.prefix_cache = false;
        model.runtime_env.cuda_graph = false;
        model.runtime_env.kv_capacity = Some(32);
        model.runtime_env.kv_max_blocks = Some(4);
        model.runtime_env.paged_max_seqs = 2;
        model
    }

    fn prepare(model: &mut LlamaFamilyModel<CpuBackend, KvFp16>, cache_id: &str, tokens: usize) {
        assert!(matches!(
            model
                .prepare_token_kv(ModelKvPreparation {
                    cache_id,
                    capacity_hint: 32,
                    prompt: &std::sync::Arc::from(vec![1; tokens]),
                    adapter: None,
                })
                .unwrap(),
            ModelKvReservation::Reserved(_)
        ));
    }

    fn domain(model: &LlamaFamilyModel<CpuBackend, KvFp16>) -> AllocatorCapacityDomain {
        model
            .paged_block_alloc
            .as_ref()
            .unwrap()
            .lock()
            .unwrap()
            .capacity_domain()
            .clone()
    }

    #[test]
    fn model_reclaim_observation_uses_actual_pool_and_preserves_missing_candidate_indices() {
        let mut model = model();
        prepare(&mut model, "blocked", 1);
        prepare(&mut model, "victim", 17);
        let domain = domain(&model);
        let blocked_mapping = model.kv_caches["blocked"][0].paged_block_indices.clone();
        let observed = model
            .observe_token_allocator_reclaim("blocked", &domain, &["already-closed", "victim"])
            .unwrap()
            .unwrap();
        assert_eq!(observed.snapshot.block_size, 16);
        assert_eq!(observed.snapshot.total_blocks, 4);
        assert_eq!(observed.snapshot.free_blocks, 1);
        assert_eq!(observed.reclaimable_blocks, 2);
        assert_eq!(observed.contributing_candidates, vec![1]);
        assert_eq!(model.kv_caches["victim"][0].paged_block_indices.len(), 2);
        assert_eq!(
            model
                .paged_block_alloc
                .as_ref()
                .unwrap()
                .lock()
                .unwrap()
                .free_count(),
            1
        );

        model.close_token_sequence("victim").unwrap();
        let after = model
            .observe_token_allocator_reclaim("blocked", &domain, &["already-closed", "victim"])
            .unwrap()
            .unwrap();
        assert_eq!(
            after.snapshot.free_blocks - observed.snapshot.free_blocks,
            observed.reclaimable_blocks
        );
        assert_eq!(after.reclaimable_blocks, 0);
        assert!(after.contributing_candidates.is_empty());
        assert!(model.kv_caches["blocked"]
            .iter()
            .all(|layer| { layer.len == 0 && layer.paged_block_indices == blocked_mapping }));
        model.close_token_sequence("blocked").unwrap();
        assert!(model.kv_caches.is_empty());
        assert_eq!(
            model
                .paged_block_alloc
                .as_ref()
                .unwrap()
                .lock()
                .unwrap()
                .free_count(),
            4
        );
    }

    #[test]
    fn model_reclaim_observation_rejects_wrong_owner_sets_and_inconsistent_layers() {
        let mut model = model();
        prepare(&mut model, "blocked", 1);
        prepare(&mut model, "victim", 1);
        let domain = domain(&model);
        assert!(model
            .observe_token_allocator_reclaim(
                "blocked",
                &AllocatorCapacityDomain::new(),
                &["victim"],
            )
            .unwrap()
            .is_none());
        for candidates in [
            &["blocked"][..],
            &["victim", "victim"][..],
            &["absent", "absent"][..],
        ] {
            assert!(model
                .observe_token_allocator_reclaim("blocked", &domain, candidates)
                .is_err());
        }
        assert!(model
            .observe_token_allocator_reclaim("absent", &domain, &["victim"])
            .is_err());
        let mapping =
            std::mem::take(&mut model.kv_caches.get_mut("victim").unwrap()[1].paged_block_indices);
        assert!(model
            .observe_token_allocator_reclaim("blocked", &domain, &["victim"])
            .is_err());
        model.kv_caches.get_mut("victim").unwrap()[1].paged_block_indices = mapping;
        assert_eq!(
            model
                .paged_block_alloc
                .as_ref()
                .unwrap()
                .lock()
                .unwrap()
                .free_count(),
            2
        );
        model.close_token_sequence("victim").unwrap();
        model.close_token_sequence("blocked").unwrap();
        assert_eq!(
            model
                .paged_block_alloc
                .as_ref()
                .unwrap()
                .lock()
                .unwrap()
                .free_count(),
            4
        );
    }
}
