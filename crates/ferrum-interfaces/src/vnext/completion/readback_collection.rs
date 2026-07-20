use super::{
    invalid_completion, BatchOperationIdentity, CompletionReadbackBatchObservation,
    CompletionReadbackBatchReceipt, CompletionReadbackBatchRequest, CompletionReadbackRequest,
    VNextError,
};

const MAX_COMPLETION_READBACK_GROUPS: usize = 64;

/// Canonical terminal readbacks for multiple node/resource groups. Every
/// group remains a complete participant batch; this type does not weaken the
/// single-node invariant of [`CompletionReadbackBatchRequest`].
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use = "a completion readback collection must be consumed by one terminal wait"]
pub struct CompletionReadbackCollectionRequest {
    batches: Vec<CompletionReadbackBatchRequest>,
}

impl CompletionReadbackCollectionRequest {
    pub fn new(mut batches: Vec<CompletionReadbackBatchRequest>) -> Result<Self, VNextError> {
        if batches.is_empty() || batches.len() > MAX_COMPLETION_READBACK_GROUPS {
            return Err(invalid_completion(
                "completion readback collection is empty or exceeds its group limit",
            ));
        }
        let participant_count = batches[0].len();
        if batches.iter().any(|batch| batch.len() != participant_count) {
            return Err(invalid_completion(
                "completion readback collection groups must cover the same participant count",
            ));
        }
        batches.sort_by(|left, right| {
            let left = &left.requests()[0];
            let right = &right.requests()[0];
            left.node_id()
                .cmp(right.node_id())
                .then_with(|| left.resource_id().cmp(right.resource_id()))
        });
        if batches.windows(2).any(|pair| {
            let left = &pair[0].requests()[0];
            let right = &pair[1].requests()[0];
            left.node_id() == right.node_id() && left.resource_id() == right.resource_id()
        }) {
            return Err(invalid_completion(
                "completion readback collection contains a duplicate node/resource group",
            ));
        }
        Ok(Self { batches })
    }

    pub fn batches(&self) -> &[CompletionReadbackBatchRequest] {
        &self.batches
    }

    pub fn len(&self) -> usize {
        self.batches.len()
    }

    pub fn is_empty(&self) -> bool {
        self.batches.is_empty()
    }

    pub fn request_count(&self) -> usize {
        self.batches
            .iter()
            .map(CompletionReadbackBatchRequest::len)
            .sum()
    }

    pub(super) fn validate_for(
        &self,
        batch_identity: &BatchOperationIdentity,
    ) -> Result<(), VNextError> {
        for batch in &self.batches {
            batch.validate_for(batch_identity)?;
        }
        Ok(())
    }

    pub(super) fn into_requests(self) -> Vec<CompletionReadbackRequest> {
        self.batches
            .into_iter()
            .flat_map(CompletionReadbackBatchRequest::into_requests)
            .collect()
    }
}

/// Collection receipts use the same ordered, fingerprinted disposition
/// evidence as a single readback batch.
pub type CompletionReadbackCollectionReceipt = CompletionReadbackBatchReceipt;
pub type CompletionReadbackCollectionObservation = CompletionReadbackBatchObservation;
