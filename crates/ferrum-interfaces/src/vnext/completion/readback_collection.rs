use std::cmp::Ordering;

use super::{
    invalid_completion, BatchOperationIdentity, CompletionReadbackBatchObservation,
    CompletionReadbackBatchReceipt, CompletionReadbackBatchRequest, CompletionReadbackRequest,
    VNextError,
};

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
        if batches.is_empty() || u32::try_from(batches.len()).is_err() {
            return Err(invalid_completion(
                "completion readback collection is empty or its group count exceeds u32",
            ));
        }
        let participant_count = batches[0].len();
        if batches.iter().any(|batch| batch.len() != participant_count) {
            return Err(invalid_completion(
                "completion readback collection groups must cover the same participant count",
            ));
        }
        batches.sort_by(compare_readback_batches);
        if batches
            .windows(2)
            .any(|pair| compare_readback_batches(&pair[0], &pair[1]) == Ordering::Equal)
        {
            return Err(invalid_completion(
                "completion readback collection contains a duplicate typed physical range",
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

fn compare_readback_batches(
    left: &CompletionReadbackBatchRequest,
    right: &CompletionReadbackBatchRequest,
) -> Ordering {
    let left_first = &left.requests()[0];
    let right_first = &right.requests()[0];
    let group_order = left_first
        .node_id()
        .cmp(right_first.node_id())
        .then_with(|| left_first.resource_id().cmp(right_first.resource_id()))
        .then_with(|| {
            left_first
                .expected_usage()
                .cmp(&right_first.expected_usage())
        })
        .then_with(|| {
            left_first
                .logical_offset_bytes()
                .cmp(&right_first.logical_offset_bytes())
        })
        .then_with(|| left.len().cmp(&right.len()));
    if group_order != Ordering::Equal {
        return group_order;
    }
    left.requests()
        .iter()
        .zip(right.requests())
        .find_map(|(left, right)| {
            let layout_order = left
                .output_layout()
                .element_type()
                .cmp(&right.output_layout().element_type())
                .then_with(|| {
                    left.output_layout()
                        .element_count()
                        .cmp(&right.output_layout().element_count())
                });
            (layout_order != Ordering::Equal).then_some(layout_order)
        })
        .unwrap_or(Ordering::Equal)
}

/// Collection receipts use the same ordered, fingerprinted disposition
/// evidence as a single readback batch.
pub type CompletionReadbackCollectionReceipt = CompletionReadbackBatchReceipt;
pub type CompletionReadbackCollectionObservation = CompletionReadbackBatchObservation;

#[cfg(test)]
mod tests {
    use super::{CompletionReadbackBatchRequest, CompletionReadbackCollectionRequest};
    use crate::vnext::{
        BufferUsage, CompletionReadbackRequest, ElementType, HostTransferLayout, NodeId, ResourceId,
    };

    fn request(
        participant_index: u32,
        logical_offset_bytes: u64,
        element_count: u64,
    ) -> CompletionReadbackRequest {
        CompletionReadbackRequest::new_typed(
            NodeId::new("node/readback").unwrap(),
            participant_index,
            ResourceId::new("resource/readback").unwrap(),
            BufferUsage::State,
            logical_offset_bytes,
            HostTransferLayout::new(ElementType::U8, element_count).unwrap(),
        )
        .unwrap()
    }

    fn batch(logical_offset_bytes: u64, element_count: u64) -> CompletionReadbackBatchRequest {
        CompletionReadbackBatchRequest::new(vec![request(0, logical_offset_bytes, element_count)])
            .unwrap()
    }

    #[test]
    fn participant_layouts_may_follow_distinct_active_token_extents() {
        let batch =
            CompletionReadbackBatchRequest::new(vec![request(0, 0, 4), request(1, 0, 7)]).unwrap();
        assert_eq!(batch.requests()[0].output_layout().element_count(), 4);
        assert_eq!(batch.requests()[1].output_layout().element_count(), 7);
        let wrong_element = CompletionReadbackRequest::new_typed(
            NodeId::new("node/readback").unwrap(),
            1,
            ResourceId::new("resource/readback").unwrap(),
            BufferUsage::State,
            0,
            HostTransferLayout::new(ElementType::F16, 2).unwrap(),
        )
        .unwrap();
        assert!(
            CompletionReadbackBatchRequest::new(vec![request(0, 0, 4), wrong_element]).is_err()
        );
    }

    #[test]
    fn collection_keys_the_complete_typed_physical_range() {
        let first = batch(0, 4);
        assert!(
            CompletionReadbackCollectionRequest::new(vec![first.clone(), first.clone(),]).is_err()
        );
        let collection =
            CompletionReadbackCollectionRequest::new(vec![first, batch(4, 4), batch(0, 8)])
                .unwrap();
        assert_eq!(collection.len(), 3);
    }

    #[test]
    fn collection_is_not_limited_to_the_legacy_sixty_four_groups() {
        let batches = (0..65).map(|offset| batch(offset, 1)).collect::<Vec<_>>();
        assert_eq!(
            CompletionReadbackCollectionRequest::new(batches)
                .unwrap()
                .len(),
            65
        );
    }
}
