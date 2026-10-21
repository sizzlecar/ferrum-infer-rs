use super::{
    invalid_resource, AdmissionFitPolicy, AdmissionPressureAction, Digest, DynamicResourceShape,
    ExecutionFrameId, NodeId, RequestAuthorityId, ResourceWorkShape, SequenceAuthorityId,
    Serialize, Sha256, TokenSpanWork, VNextError,
};
use crate::ReusableExecutionBucketId;
use std::{ops::Range, sync::Arc};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepResourceAdmissionRequest {
    pub(super) work_shape: Arc<BatchWorkShape>,
    pub(super) fit_policy: AdmissionFitPolicy,
    pub(super) pressure_action: AdmissionPressureAction,
    pub(super) reusable_execution_bucket_id: Option<ReusableExecutionBucketId>,
}

impl StepResourceAdmissionRequest {
    pub fn new(
        work_shape: impl Into<Arc<BatchWorkShape>>,
        fit_policy: AdmissionFitPolicy,
        pressure_action: AdmissionPressureAction,
    ) -> Result<Self, VNextError> {
        Ok(Self {
            work_shape: work_shape.into(),
            fit_policy,
            pressure_action,
            reusable_execution_bucket_id: None,
        })
    }

    pub fn with_reusable_execution_bucket(mut self, bucket_id: ReusableExecutionBucketId) -> Self {
        self.reusable_execution_bucket_id = Some(bucket_id);
        self
    }

    pub fn work_shape(&self) -> &BatchWorkShape {
        &self.work_shape
    }

    pub(crate) fn immediate_shape(&self) -> DynamicResourceShape {
        self.work_shape.immediate_shape()
    }

    pub const fn fit_policy(&self) -> AdmissionFitPolicy {
        self.fit_policy
    }

    pub const fn pressure_action(&self) -> AdmissionPressureAction {
        self.pressure_action
    }

    pub fn reusable_execution_bucket_id(&self) -> Option<&ReusableExecutionBucketId> {
        self.reusable_execution_bucket_id.as_ref()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct BatchParticipantAuthority {
    sequence_authority: SequenceAuthorityId,
    request_authority: RequestAuthorityId,
}

impl BatchParticipantAuthority {
    pub const fn new(
        sequence_authority: SequenceAuthorityId,
        request_authority: RequestAuthorityId,
    ) -> Self {
        Self {
            sequence_authority,
            request_authority,
        }
    }

    pub const fn sequence_authority(self) -> SequenceAuthorityId {
        self.sequence_authority
    }

    pub const fn request_authority(self) -> RequestAuthorityId {
        self.request_authority
    }

    pub(super) const fn canonical_key(self) -> (u32, u64, u32, u64) {
        (
            self.sequence_authority.sparse_id(),
            self.sequence_authority.generation(),
            self.request_authority.sparse_id(),
            self.request_authority.generation(),
        )
    }
}

/// One participant-local node topology key in the physical batch ledger.
/// Attempt ids are deliberately absent so a fresh id cannot bypass overlap
/// detection for the same sequence/frame/node work.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct ParticipantNodeKey {
    sequence_authority: SequenceAuthorityId,
    request_authority: RequestAuthorityId,
    frame_id: ExecutionFrameId,
    node_id: NodeId,
}

impl ParticipantNodeKey {
    pub(crate) fn new(
        participant: BatchParticipantAuthority,
        frame_id: ExecutionFrameId,
        node_id: NodeId,
    ) -> Self {
        Self {
            sequence_authority: participant.sequence_authority(),
            request_authority: participant.request_authority(),
            frame_id,
            node_id,
        }
    }

    pub const fn sequence_authority(&self) -> SequenceAuthorityId {
        self.sequence_authority
    }

    pub const fn request_authority(&self) -> RequestAuthorityId {
        self.request_authority
    }

    pub const fn frame_id(&self) -> ExecutionFrameId {
        self.frame_id
    }

    pub fn node_id(&self) -> &NodeId {
        &self.node_id
    }
}

/// Opaque association between one exact admitted participant and token work
/// derived from that participant's actual token ids.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BatchParticipantTokenSpan {
    participant: BatchParticipantAuthority,
    token_span: TokenSpanWork,
}

/// Exact packed-token projection for one participant in a scheduler step.
/// The range addresses the shared batch transient arena; it is derived from
/// the canonical participant work and cannot be supplied independently.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BatchParticipantTokenRange {
    participant: BatchParticipantAuthority,
    immediate_start_token: u64,
    immediate_end_token: u64,
    source_start_token: u64,
    source_end_token: u64,
    full_input_tokens: u64,
}

impl BatchParticipantTokenRange {
    fn new(
        participant: BatchParticipantAuthority,
        immediate_start_token: u64,
        immediate_end_token: u64,
        source_start_token: u64,
        source_end_token: u64,
        full_input_tokens: u64,
    ) -> Result<Self, VNextError> {
        if immediate_start_token >= immediate_end_token
            || source_start_token >= source_end_token
            || source_end_token > full_input_tokens
            || immediate_end_token - immediate_start_token != source_end_token - source_start_token
        {
            return Err(invalid_resource(
                "batch participant packed-token range is empty or exceeds its full input",
            ));
        }
        Ok(Self {
            participant,
            immediate_start_token,
            immediate_end_token,
            source_start_token,
            source_end_token,
            full_input_tokens,
        })
    }

    pub const fn participant(&self) -> BatchParticipantAuthority {
        self.participant
    }

    pub fn immediate_token_range(&self) -> Range<u64> {
        self.immediate_start_token..self.immediate_end_token
    }

    pub const fn immediate_tokens(&self) -> u64 {
        self.immediate_end_token - self.immediate_start_token
    }

    pub fn source_token_range(&self) -> Range<u64> {
        self.source_start_token..self.source_end_token
    }

    pub const fn full_input_tokens(&self) -> u64 {
        self.full_input_tokens
    }
}

impl BatchParticipantTokenSpan {
    pub(super) fn new(participant: BatchParticipantAuthority, token_span: TokenSpanWork) -> Self {
        Self {
            participant,
            token_span,
        }
    }

    pub const fn participant(&self) -> BatchParticipantAuthority {
        self.participant
    }

    pub fn token_span(&self) -> &TokenSpanWork {
        &self.token_span
    }
}

/// Immutable work authority for one exact non-empty participant set. The
/// dimensions remain private so downstream claims and dispatch can only use
/// the shape that core bound to this participant topology and fingerprint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct BatchWorkShape {
    participants: Vec<BatchParticipantAuthority>,
    token_work: Option<TokenBatchEvidence>,
    resource_work: ResourceWorkShape,
    fingerprint: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
struct TokenBatchEvidence {
    participant_work: Vec<BatchParticipantTokenSpan>,
    participant_token_ranges: Vec<BatchParticipantTokenRange>,
}

/// A checked token-only projection. Tensor work cannot construct this view.
#[derive(Debug, Clone, Copy)]
pub struct TokenBatchWorkView<'a> {
    batch: &'a BatchWorkShape,
    token: &'a crate::TokenResourceWorkShape,
    evidence: &'a TokenBatchEvidence,
}
impl std::ops::Deref for TokenBatchWorkView<'_> {
    type Target = BatchWorkShape;
    fn deref(&self) -> &Self::Target {
        self.batch
    }
}
impl TokenBatchWorkView<'_> {
    pub fn participant_work(&self) -> &[BatchParticipantTokenSpan] {
        &self.evidence.participant_work
    }
    pub fn participant_token_ranges(&self) -> &[BatchParticipantTokenRange] {
        &self.evidence.participant_token_ranges
    }
    pub const fn immediate_tokens(&self) -> u64 {
        self.token.immediate_tokens()
    }
    pub const fn immediate_pages(&self) -> u64 {
        self.token.immediate_pages()
    }
    pub const fn fit_tokens(&self) -> u64 {
        self.token.fit_tokens()
    }
    pub const fn fit_pages(&self) -> u64 {
        self.token.fit_pages()
    }
}

impl BatchWorkShape {
    pub(super) fn new(
        participant_work: Vec<BatchParticipantTokenSpan>,
    ) -> Result<Self, VNextError> {
        if participant_work.is_empty()
            || participant_work.windows(2).any(|pair| {
                pair[0].participant().canonical_key() >= pair[1].participant().canonical_key()
            })
        {
            return Err(invalid_resource(
                "batch work shape requires canonical non-empty unique participant work",
            ));
        }
        let participants = participant_work
            .iter()
            .map(BatchParticipantTokenSpan::participant)
            .collect::<Vec<_>>();
        let mut next_token = 0_u64;
        let participant_token_ranges = participant_work
            .iter()
            .map(|work| {
                let start = next_token;
                next_token = next_token
                    .checked_add(work.token_span().immediate_tokens())
                    .ok_or_else(|| invalid_resource("packed batch token range overflows u64"))?;
                BatchParticipantTokenRange::new(
                    work.participant(),
                    start,
                    next_token,
                    work.token_span().immediate_token_range().start,
                    work.token_span().immediate_token_range().end,
                    work.token_span().full_input_tokens(),
                )
            })
            .collect::<Result<Vec<_>, VNextError>>()?;
        let resource_work = ResourceWorkShape::from_token_spans(
            participant_work
                .iter()
                .map(|work| work.token_span().clone())
                .collect(),
        )?;
        if resource_work.immediate_sequences()
            != u32::try_from(participants.len())
                .map_err(|_| invalid_resource("batch work participant count exceeds u32"))?
            || next_token != resource_work.immediate_tokens()?
        {
            return Err(invalid_resource(
                "batch work shape aggregate differs from participant evidence",
            ));
        }
        #[derive(Serialize)]
        struct FingerprintInput<'a> {
            domain: &'static str,
            participant_work: &'a [BatchParticipantTokenSpan],
            participant_token_ranges: &'a [BatchParticipantTokenRange],
            resource_work_fingerprint: &'a str,
        }
        let input = FingerprintInput {
            domain: "ferrum.runtime-vnext.batch-work-shape.v3",
            participant_work: &participant_work,
            participant_token_ranges: &participant_token_ranges,
            resource_work_fingerprint: resource_work.fingerprint(),
        };
        let bytes = serde_json::to_vec(&input).map_err(|error| {
            invalid_resource(format!("batch work shape encode failed: {error}"))
        })?;
        Ok(Self {
            participants,
            token_work: Some(TokenBatchEvidence {
                participant_work,
                participant_token_ranges,
            }),
            resource_work,
            fingerprint: format!("{:x}", Sha256::digest(bytes)),
        })
    }

    #[cfg(test)]
    pub(crate) fn test_only(token_spans: Vec<TokenSpanWork>) -> Result<Self, VNextError> {
        let participant_work = token_spans
            .into_iter()
            .enumerate()
            .map(|(index, token_span)| {
                let sparse_id =
                    u32::try_from(index + 1).expect("bounded test participant index fits u32");
                BatchParticipantTokenSpan::new(
                    BatchParticipantAuthority::new(
                        SequenceAuthorityId::test_only(sparse_id, 1),
                        RequestAuthorityId::test_only(sparse_id, 1),
                    ),
                    token_span,
                )
            })
            .collect();
        Self::new(participant_work)
    }

    pub fn participants(&self) -> &[BatchParticipantAuthority] {
        &self.participants
    }

    pub fn token_view(&self) -> Result<TokenBatchWorkView<'_>, VNextError> {
        let evidence = self
            .token_work
            .as_ref()
            .ok_or_else(|| invalid_resource("tensor batch has no token work view"))?;
        Ok(TokenBatchWorkView {
            batch: self,
            token: self.resource_work.token_view()?,
            evidence,
        })
    }

    pub fn participant_work(&self) -> Result<&[BatchParticipantTokenSpan], VNextError> {
        Ok(&self
            .token_work
            .as_ref()
            .ok_or_else(|| invalid_resource("tensor batch has no token spans"))?
            .participant_work)
    }
    pub fn participant_token_ranges(&self) -> Result<&[BatchParticipantTokenRange], VNextError> {
        Ok(&self
            .token_work
            .as_ref()
            .ok_or_else(|| invalid_resource("tensor batch has no token ranges"))?
            .participant_token_ranges)
    }

    pub(super) fn from_tensor(
        participant: BatchParticipantAuthority,
        resource_work: ResourceWorkShape,
    ) -> Result<Self, VNextError> {
        if !resource_work.is_tensor() {
            return Err(invalid_resource(
                "tensor batch requires root-bound tensor evidence",
            ));
        }
        let participants = vec![participant];
        let bytes = serde_json::to_vec(&(
            "ferrum.tensor-batch.v1",
            &participants,
            resource_work.fingerprint(),
        ))
        .map_err(|error| invalid_resource(format!("tensor batch fingerprint: {error}")))?;
        Ok(Self {
            participants,
            token_work: None,
            resource_work,
            fingerprint: format!("{:x}", Sha256::digest(bytes)),
        })
    }

    pub fn resource_work(&self) -> &ResourceWorkShape {
        &self.resource_work
    }

    pub const fn immediate_sequences(&self) -> u32 {
        self.resource_work.immediate_sequences()
    }

    pub fn immediate_tokens(&self) -> Result<u64, VNextError> {
        self.resource_work.immediate_tokens()
    }

    pub fn immediate_pages(&self) -> Result<u64, VNextError> {
        self.resource_work.immediate_pages()
    }

    pub const fn fit_sequences(&self) -> u32 {
        self.resource_work.fit_sequences()
    }

    pub fn fit_tokens(&self) -> Result<u64, VNextError> {
        self.resource_work.fit_tokens()
    }

    pub fn fit_pages(&self) -> Result<u64, VNextError> {
        self.resource_work.fit_pages()
    }

    pub fn fingerprint(&self) -> &str {
        &self.fingerprint
    }

    /// Stable logical work identity for comparisons across fresh admissions.
    /// Unlike [`Self::fingerprint`], this excludes request and sequence
    /// authorities while retaining the exact token and committed-page shape.
    pub fn logical_work_fingerprint(&self) -> &str {
        self.resource_work.fingerprint()
    }

    pub(crate) const fn immediate_shape(&self) -> DynamicResourceShape {
        self.resource_work.immediate_shape()
    }

    pub(crate) const fn fit_shape(&self) -> DynamicResourceShape {
        self.resource_work.fit_shape()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct StepParticipantFrameAssignment {
    participant: BatchParticipantAuthority,
    frame_id: ExecutionFrameId,
}

impl StepParticipantFrameAssignment {
    pub(super) const fn new(
        sequence_authority: SequenceAuthorityId,
        request_authority: RequestAuthorityId,
        frame_id: ExecutionFrameId,
    ) -> Self {
        Self {
            participant: BatchParticipantAuthority::new(sequence_authority, request_authority),
            frame_id,
        }
    }

    pub const fn participant(self) -> BatchParticipantAuthority {
        self.participant
    }

    pub const fn sequence_authority(self) -> SequenceAuthorityId {
        self.participant.sequence_authority()
    }

    pub const fn request_authority(self) -> RequestAuthorityId {
        self.participant.request_authority()
    }

    pub const fn frame_id(self) -> ExecutionFrameId {
        self.frame_id
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InvocationResourceAdmissionRequest {
    pub(super) node_id: NodeId,
    pub(super) work_shape: Arc<BatchWorkShape>,
    pub(super) fit_policy: AdmissionFitPolicy,
    pub(super) pressure_action: AdmissionPressureAction,
}

impl InvocationResourceAdmissionRequest {
    pub fn new(
        node_id: NodeId,
        work_shape: impl Into<Arc<BatchWorkShape>>,
        fit_policy: AdmissionFitPolicy,
        pressure_action: AdmissionPressureAction,
    ) -> Result<Self, VNextError> {
        Ok(Self {
            node_id,
            work_shape: work_shape.into(),
            fit_policy,
            pressure_action,
        })
    }

    pub fn for_all_step_participants(
        node_id: NodeId,
        work_shape: impl Into<Arc<BatchWorkShape>>,
        fit_policy: AdmissionFitPolicy,
        pressure_action: AdmissionPressureAction,
    ) -> Result<Self, VNextError> {
        Self::new(node_id, work_shape, fit_policy, pressure_action)
    }

    pub fn node_id(&self) -> &NodeId {
        &self.node_id
    }

    pub fn work_shape(&self) -> &BatchWorkShape {
        self.work_shape.as_ref()
    }

    pub const fn fit_policy(&self) -> AdmissionFitPolicy {
        self.fit_policy
    }

    pub const fn pressure_action(&self) -> AdmissionPressureAction {
        self.pressure_action
    }
}
