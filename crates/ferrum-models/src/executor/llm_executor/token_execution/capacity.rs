//! Cold observations over the original registry and physical model allocator.

use super::*;

impl LlmExecutor {
    pub(in crate::executor::llm_executor) fn observe_allocator_reclaim(
        &self,
        blocked: &ExecutionFrontier,
        domain: &AllocatorCapacityDomain,
        candidates: &[ExecutorSequence],
    ) -> Result<Option<TokenAllocatorReclaim>> {
        let (blocked_cache, candidate_caches) = {
            let sequences = self.token_sequences.lock();
            let entry = sequences
                .live
                .get(blocked.sequence())
                .filter(|entry| {
                    !entry.closing
                        && entry.frontier == *blocked
                        && blocked.sequence().observed_end().is_none()
                })
                .ok_or_else(|| {
                    FerrumError::request_validation(
                        "allocator observation requires the exact live blocked frontier",
                    )
                })?;
            let mut caches = Vec::with_capacity(candidates.len());
            let mut unique = std::collections::HashSet::with_capacity(candidates.len());
            for (index, candidate) in candidates.iter().enumerate() {
                // Missing retired owners are valid observations; identities
                // from a different executor are not candidates for this pool.
                self.token_issuer.frontier(candidate, 0)?;
                if candidate == blocked.sequence() || !unique.insert(candidate) {
                    return Err(FerrumError::request_validation(
                        "allocator candidates must be distinct external incarnations",
                    ));
                }
                if let Some(candidate) = sequences.live.get(candidate) {
                    caches.push((index, candidate.cache_id.clone()));
                }
            }
            (entry.cache_id.clone(), caches)
        };
        // Never hold the metadata lock while waiting for numerical work. Cache
        // identities cannot be rebound; any concurrent physical close is
        // reflected by the model's current cache/refcount observation below.
        let cache_ids = candidate_caches
            .iter()
            .map(|(_, id)| id.as_str())
            .collect::<Vec<_>>();
        let mut observed = match self.lock_model().observe_token_allocator_reclaim(
            &blocked_cache,
            domain,
            &cache_ids,
        )? {
            Some(observed) => observed,
            None => return Ok(None),
        };
        for index in &mut observed.contributing_candidates {
            *index = candidate_caches
                .get(*index)
                .ok_or_else(|| {
                    FerrumError::internal("model allocator observation named an unknown candidate")
                })?
                .0;
        }
        Ok(Some(observed))
    }
}
