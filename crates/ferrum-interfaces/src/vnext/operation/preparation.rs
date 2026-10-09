/// Identity preparation observed through dispatch return, including failures.
/// Later diagnostic/serialization consumers may still materialize owned parts.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct InvocationPreparationStats {
    pub projected_identities: u64,
    pub parts_materialized: u64,
    /// Borrowed invocations constructed for resident nodes, before provider encode.
    pub borrowed_view_nodes: u64,
}

pub trait InvocationPreparationSink {
    fn record_preparation(&self, stats: InvocationPreparationStats);
}

pub(super) struct PreparationObservation<'a, S: InvocationPreparationSink> {
    pub identity: &'a super::BatchOperationIdentity,
    pub sink: &'a S,
    pub borrowed_view_nodes: &'a std::cell::Cell<u64>,
}
impl<S: InvocationPreparationSink> Drop for PreparationObservation<'_, S> {
    fn drop(&mut self) {
        let mut stats = self.identity.preparation_snapshot();
        stats.borrowed_view_nodes = self.borrowed_view_nodes.get();
        self.sink.record_preparation(stats);
    }
}
