/// Identity preparation observed through dispatch return, including failures.
/// Later diagnostic/serialization consumers may still materialize owned parts.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct InvocationPreparationStats {
    pub projected_identities: u64,
    pub parts_materialized: u64,
    pub pool_version_proofs: u64,
    pub pool_version_hits: u64,
    pub pool_version_fallbacks: u64,
}

pub trait InvocationPreparationSink {
    fn record_preparation(&self, stats: InvocationPreparationStats);
}

pub(super) struct PreparationObservation<'a, S: InvocationPreparationSink> {
    pub identity: &'a super::BatchOperationIdentity,
    pub sink: &'a S,
    pub pool_version: &'a PoolVersionCounts,
}
impl<S: InvocationPreparationSink> Drop for PreparationObservation<'_, S> {
    fn drop(&mut self) {
        let mut stats = self.identity.preparation_snapshot();
        stats.pool_version_proofs = self.pool_version.proofs.get();
        stats.pool_version_hits = self.pool_version.hits.get();
        stats.pool_version_fallbacks = self.pool_version.fallbacks.get();
        self.sink.record_preparation(stats);
    }
}

/// One dispatch-local aggregate. No per-participant atomic or journal work.
#[derive(Default)]
pub(super) struct PoolVersionCounts {
    pub(super) proofs: std::cell::Cell<u64>,
    pub(super) hits: std::cell::Cell<u64>,
    pub(super) fallbacks: std::cell::Cell<u64>,
}
impl PoolVersionCounts {
    pub(super) fn record_proof(&self, present: bool) {
        if present {
            self.proofs.set(self.proofs.get().saturating_add(1));
        }
    }
    pub(super) fn record_check(&self, hit: bool) {
        let counter = if hit { &self.hits } else { &self.fallbacks };
        counter.set(counter.get().saturating_add(1));
    }
}
