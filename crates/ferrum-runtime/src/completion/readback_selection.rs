use super::{
    invalid_completion, BatchOperationIdentity, CompletionReadbackBatchObservation,
    CompletionReadbackBatchReceipt, CompletionReadbackBatchRequest, CompletionReadbackRequest,
    VNextError,
};
use crate::ParticipantNodeKey;

/// Complete typed output alternatives, with exactly one choice per canonical
/// participant ordinal. Alternatives retain the complete-node contract of
/// [`CompletionReadbackBatchRequest`]; only the selected ranges are transferred.
#[derive(Debug, Clone, PartialEq, Eq)]
#[must_use = "a completion readback selection must be consumed by one terminal wait"]
pub struct CompletionReadbackSelectionRequest {
    alternatives: Vec<CompletionReadbackBatchRequest>,
    selected_alternatives: Vec<u32>,
}

impl CompletionReadbackSelectionRequest {
    pub fn new(
        alternatives: Vec<CompletionReadbackBatchRequest>,
        selected_alternatives: Vec<u32>,
    ) -> Result<Self, VNextError> {
        let Some(first) = alternatives.first() else {
            return Err(invalid_completion(
                "completion readback selection requires a complete typed alternative",
            ));
        };
        if u32::try_from(alternatives.len()).is_err()
            || alternatives.iter().any(|batch| batch.len() != first.len())
        {
            return Err(invalid_completion(
                "completion readback selection alternatives must cover the same participant count and their count must fit u32",
            ));
        }
        if alternatives
            .iter()
            .enumerate()
            .any(|(index, batch)| alternatives[..index].iter().any(|prior| prior == batch))
        {
            return Err(invalid_completion(
                "completion readback selection contains a duplicate typed alternative",
            ));
        }
        if selected_alternatives.len() != first.len()
            || selected_alternatives.iter().any(|choice| {
                usize::try_from(*choice).map_or(true, |choice| choice >= alternatives.len())
            })
        {
            return Err(invalid_completion(
                "completion readback selection must choose one valid alternative for every canonical participant exactly once",
            ));
        }
        Ok(Self {
            alternatives,
            selected_alternatives,
        })
    }

    /// Alternative indices keep the caller's declared order.
    pub fn alternatives(&self) -> &[CompletionReadbackBatchRequest] {
        &self.alternatives
    }

    /// The entry at each canonical participant ordinal selects its alternative.
    pub fn selected_alternatives(&self) -> &[u32] {
        &self.selected_alternatives
    }

    pub fn len(&self) -> usize {
        self.selected_alternatives.len()
    }

    pub fn is_empty(&self) -> bool {
        self.selected_alternatives.is_empty()
    }

    pub fn selected_requests(
        &self,
    ) -> impl ExactSizeIterator<Item = &CompletionReadbackRequest> + '_ {
        self.selected_alternatives
            .iter()
            .enumerate()
            .map(|(ordinal, choice)| &self.alternatives[*choice as usize].requests()[ordinal])
    }

    pub(super) fn validate_for(
        &self,
        batch_identity: &BatchOperationIdentity,
    ) -> Result<(), VNextError> {
        // Validate even unused alternatives against the actual bound submission.
        for batch in &self.alternatives {
            batch.validate_for(batch_identity)?;
        }
        let anchor_index = batch_identity
            .node_index(self.alternatives[0].requests()[0].node_id())
            .ok_or_else(|| invalid_completion("completion readback selection node is absent"))?;
        let anchor = batch_identity.materialize_node(anchor_index)?;
        if anchor.participants().len() != self.len() {
            return Err(invalid_completion(
                "completion readback selection differs from its materialized participant count",
            ));
        }
        for batch in self.alternatives.iter().skip(1) {
            let node_index = batch_identity
                .node_index(batch.requests()[0].node_id())
                .ok_or_else(|| {
                    invalid_completion("completion readback selection node is absent")
                })?;
            let alternative = batch_identity.materialize_node(node_index)?;
            if alternative.participants().len() != anchor.participants().len()
                || alternative
                    .participants()
                    .iter()
                    .zip(anchor.participants())
                    .any(|(candidate, anchor)| {
                        !same_participant(candidate.node_key(), anchor.node_key())
                    })
            {
                return Err(invalid_completion(
                    "completion readback selection alternatives differ in canonical participant sequence, request, or frame authority",
                ));
            }
        }
        Ok(())
    }

    pub(super) fn into_requests(self) -> Vec<CompletionReadbackRequest> {
        self.selected_requests().cloned().collect()
    }
}

fn same_participant(left: &ParticipantNodeKey, right: &ParticipantNodeKey) -> bool {
    left.sequence_authority() == right.sequence_authority()
        && left.request_authority() == right.request_authority()
        && left.frame_id() == right.frame_id()
}

/// Selected receipts preserve the existing ordered, fingerprinted dispositions.
pub type CompletionReadbackSelectionReceipt = CompletionReadbackBatchReceipt;
pub type CompletionReadbackSelectionObservation = CompletionReadbackBatchObservation;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BufferUsage, ElementType, ExecutionFrameId, HostTransferLayout, NodeId, ResourceId,
    };

    fn batch(node: &str, count: u32, layout: HostTransferLayout) -> CompletionReadbackBatchRequest {
        CompletionReadbackBatchRequest::new(
            (0..count)
                .map(|ordinal| {
                    CompletionReadbackRequest::new_typed(
                        NodeId::new(node).unwrap(),
                        ordinal,
                        ResourceId::new("resource/readback").unwrap(),
                        BufferUsage::Activations,
                        u64::from(ordinal) * layout.byte_len().unwrap(),
                        layout,
                    )
                    .unwrap()
                })
                .collect(),
        )
        .unwrap()
    }

    fn alternatives() -> Vec<CompletionReadbackBatchRequest> {
        vec![
            batch(
                "node/logits",
                3,
                HostTransferLayout::new(ElementType::F32, 7).unwrap(),
            ),
            batch(
                "node/token",
                3,
                HostTransferLayout::new(ElementType::U32, 1).unwrap(),
            ),
        ]
    }

    #[test]
    fn selection_requires_complete_unique_alternatives_and_one_valid_choice_per_participant() {
        assert!(CompletionReadbackSelectionRequest::new(Vec::new(), vec![0]).is_err());
        for choices in [vec![], vec![0, 1], vec![0, 1, 0, 1], vec![0, 2, 0]] {
            assert!(CompletionReadbackSelectionRequest::new(alternatives(), choices).is_err());
        }
        let first = alternatives().remove(0);
        assert!(
            CompletionReadbackSelectionRequest::new(vec![first.clone(), first], vec![0, 1, 0])
                .is_err()
        );
        assert!(CompletionReadbackSelectionRequest::new(
            vec![
                alternatives().remove(0),
                batch(
                    "node/token",
                    2,
                    HostTransferLayout::new(ElementType::U32, 1).unwrap()
                )
            ],
            vec![0, 1, 0]
        )
        .is_err());
    }

    #[test]
    fn selection_preserves_alternative_and_participant_order_with_exact_typed_ranges() {
        let selection =
            CompletionReadbackSelectionRequest::new(alternatives(), vec![1, 0, 1]).unwrap();
        assert_eq!(selection.selected_alternatives(), &[1, 0, 1]);
        let requests = selection.selected_requests().collect::<Vec<_>>();
        assert_eq!(
            requests
                .iter()
                .map(|request| request.participant_index())
                .collect::<Vec<_>>(),
            [0, 1, 2]
        );
        assert_eq!(
            requests
                .iter()
                .map(|request| request.node_id().as_str())
                .collect::<Vec<_>>(),
            ["node/token", "node/logits", "node/token"]
        );
        assert_eq!(
            requests
                .iter()
                .map(|request| request.output_layout().byte_len().unwrap())
                .collect::<Vec<_>>(),
            [4, 28, 4]
        );
        assert_eq!(
            requests
                .iter()
                .map(|request| request.logical_offset_bytes())
                .collect::<Vec<_>>(),
            [0, 28, 8]
        );
        assert_eq!(selection.into_requests().len(), 3);
    }

    #[test]
    fn cross_node_correspondence_requires_the_same_sequence_request_generation_and_frame() {
        use crate::{BatchParticipantAuthority, RequestAuthorityId, SequenceAuthorityId};
        let key = |sequence, request, frame, node: &str| {
            ParticipantNodeKey::new(
                BatchParticipantAuthority::new(sequence, request),
                ExecutionFrameId::try_from(frame).unwrap(),
                NodeId::new(node).unwrap(),
            )
        };
        let sequence = SequenceAuthorityId::test_only(1, 1);
        let request = RequestAuthorityId::test_only(1, 1);
        let anchor = key(sequence, request, 1_u64, "node/logits");
        assert!(same_participant(
            &anchor,
            &key(sequence, request, 1, "node/token")
        ));
        assert!(!same_participant(
            &anchor,
            &key(
                SequenceAuthorityId::test_only(1, 2),
                request,
                1,
                "node/token"
            )
        ));
        assert!(!same_participant(
            &anchor,
            &key(
                sequence,
                RequestAuthorityId::test_only(1, 2),
                1,
                "node/token"
            )
        ));
        assert!(!same_participant(
            &anchor,
            &key(sequence, request, 2, "node/token")
        ));
    }
}
