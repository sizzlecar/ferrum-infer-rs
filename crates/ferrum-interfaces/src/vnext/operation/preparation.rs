/// Identity preparation observed through dispatch return, including failures.
/// Later diagnostic/serialization consumers may still materialize owned parts.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct InvocationPreparationStats {
    pub projected_identities: u64,
    pub parts_materialized: u64,
}

/// Resident node preparation through dispatch return, including later
/// failures. Prepared means a fresh checked patch was successfully encoded,
/// not that the GPU completed or that a cold recipe was published.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SteadyRecipePreparationStats {
    pub considered_nodes: u64,
    pub prepared_nodes: u64,
    pub fallback_nodes: u64,
    pub invalid_nodes: u64,
}

pub trait InvocationPreparationSink {
    fn record_preparation(&self, stats: InvocationPreparationStats);
    fn record_steady_recipe(&self, _stats: SteadyRecipePreparationStats) {}
}

pub(super) struct SteadyRecipeObservation<'a> {
    pub stats: SteadyRecipePreparationStats,
    sink: Option<&'a dyn InvocationPreparationSink>,
}
impl<'a> SteadyRecipeObservation<'a> {
    pub fn new(sink: Option<&'a dyn InvocationPreparationSink>) -> Self {
        Self {
            stats: SteadyRecipePreparationStats::default(),
            sink,
        }
    }
}
impl Drop for SteadyRecipeObservation<'_> {
    fn drop(&mut self) {
        if let Some(sink) = self.sink {
            sink.record_steady_recipe(self.stats);
        }
    }
}

pub(super) struct PreparationObservation<'a, S: InvocationPreparationSink> {
    pub identity: &'a super::BatchOperationIdentity,
    pub sink: &'a S,
}
impl<S: InvocationPreparationSink> Drop for PreparationObservation<'_, S> {
    fn drop(&mut self) {
        self.sink
            .record_preparation(self.identity.preparation_snapshot());
    }
}
