/// Preparation observed through dispatch return, including failures.
/// Later diagnostic/serialization consumers may still materialize owned parts.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct InvocationPreparationStats {
    pub projected_identities: u64,
    pub parts_materialized: u64,
    /// Participant proofs published only after the whole node is constructed.
    pub agreement_builds: u64,
    /// Current consumers using immutable agreement; later live checks can fail.
    pub agreement_reuses: u64,
    /// Existing proofs whose changed current operands require the full checks.
    pub agreement_fallbacks: u64,
}

pub trait InvocationPreparationSink {
    fn record_preparation(&self, stats: InvocationPreparationStats);
}

pub(super) struct PreparationObservation<'a, S: InvocationPreparationSink> {
    pub identity: &'a super::BatchOperationIdentity,
    pub sink: &'a S,
    pub agreements: &'a std::cell::Cell<InvocationPreparationStats>,
}
impl<S: InvocationPreparationSink> Drop for PreparationObservation<'_, S> {
    fn drop(&mut self) {
        let mut stats = self.identity.preparation_snapshot();
        let agreements = self.agreements.get();
        stats.agreement_builds = agreements.agreement_builds;
        stats.agreement_reuses = agreements.agreement_reuses;
        stats.agreement_fallbacks = agreements.agreement_fallbacks;
        self.sink.record_preparation(stats);
    }
}
