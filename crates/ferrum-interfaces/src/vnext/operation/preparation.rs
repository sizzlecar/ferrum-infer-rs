/// Identity preparation observed through dispatch return, including failures.
/// Later diagnostic/serialization consumers may still materialize owned parts.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct InvocationPreparationStats {
    pub projected_identities: u64,
    pub parts_materialized: u64,
    pub segment_hits: u64,
    pub segment_misses: u64,
    pub segment_encoded_nodes: u64,
    pub segment_dynamic_resource_requests: u64,
    pub segment_unique_physical_buffers: u64,
    pub segment_no_resident_program: u64,
    pub segment_no_cached_recipe: u64,
    pub segment_immutable_capability_unavailable: u64,
    pub segment_unsupported_encoder: u64,
    pub segment_incomplete_declarations: u64,
}

#[derive(Clone, Copy)]
pub(super) enum SegmentPreparationMissReason {
    NoResidentProgram,
    NoCachedRecipe,
    ImmutableCapabilityUnavailable,
    UnsupportedEncoder,
    IncompleteDeclarations,
}

pub trait InvocationPreparationSink {
    fn record_preparation(&self, stats: InvocationPreparationStats);
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
