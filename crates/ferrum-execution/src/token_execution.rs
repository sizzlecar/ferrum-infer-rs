//! Tensor-free product execution identities, work and participant outcomes.
//!
//! These types validate a trusted executor implementation's reports. They do
//! not grant runtime admission, submission, completion or recovery authority.
//! Before mutation, the implementation must check the handle, live incarnation,
//! request binding, immutable admission and current frontier under its existing
//! sequence guard. No physical state or second sequence registry lives here.

use std::hash::{Hash, Hasher};
use std::num::NonZeroU64;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use ferrum_types::{FerrumError, Result, TokenId};

use crate::model_executor::{
    ExecutorExecutionDeferral, ExecutorSamplingOutput, LogitsReturnPolicy, PrefillChunk,
};

mod lifecycle;
pub use lifecycle::*;

#[derive(Debug)]
struct IssuerIdentity {
    next_identity: AtomicU64,
}

/// An executor's identity issuer, not a resource owner or native permission.
///
/// External executor implementations may create their own issuer. There is no
/// constructor accepting another issuer's integer/address or serialized data.
#[derive(Debug, Clone)]
pub struct ExecutorSequenceIssuer {
    identity: Arc<IssuerIdentity>,
}

impl Default for ExecutorSequenceIssuer {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutorSequenceIssuer {
    pub fn new() -> Self {
        Self {
            identity: Arc::new(IssuerIdentity {
                next_identity: AtomicU64::new(1),
            }),
        }
    }

    fn next_identity(&self) -> Result<NonZeroU64> {
        let value = self
            .identity
            .next_identity
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| invalid("executor identity space exhausted"))?;
        NonZeroU64::new(value).ok_or_else(|| invalid("zero executor identity"))
    }

    /// Issue a fresh incarnation after genuine admission has succeeded.
    ///
    /// Retain the existing `ExecutorPrefillAdmission` and its immutable prompt,
    /// extent, model/source and adapter binding in the actual executor state.
    /// Cloning this handle does not duplicate that state or grant another release.
    pub fn issue_sequence(&self) -> Result<ExecutorSequence> {
        Ok(ExecutorSequence {
            identity: Arc::new(SequenceIdentity {
                issuer: Arc::clone(&self.identity),
                incarnation: self.next_identity()?,
                ended: std::sync::atomic::AtomicBool::new(false),
            }),
        })
    }

    fn require_sequence(&self, sequence: &ExecutorSequence) -> Result<()> {
        if !Arc::ptr_eq(&self.identity, &sequence.identity.issuer) {
            return Err(invalid("sequence belongs to a different executor"));
        }
        Ok(())
    }

    /// Report a frontier established by admission, commit or prefix publication.
    /// This does not update state or establish that the incarnation is still live.
    pub fn frontier(
        &self,
        sequence: &ExecutorSequence,
        consumed_tokens: usize,
    ) -> Result<ExecutionFrontier> {
        self.require_sequence(sequence)?;
        Ok(ExecutionFrontier {
            sequence: sequence.clone(),
            consumed_tokens,
        })
    }

    /// Name recovery already retained by this executor/reaper for this sequence.
    /// The key is identity only: it cannot drain, release or recreate that owner.
    pub fn recovery_key(&self, sequence: &ExecutorSequence) -> Result<ExecutionRecoveryKey> {
        self.require_sequence(sequence)?;
        Ok(ExecutionRecoveryKey {
            sequence: sequence.clone(),
            recovery: self.next_identity()?,
        })
    }

    /// Validate immutable work shape and issue one result-construction scope.
    ///
    /// The live registry/frontier and admitted maximum extent must also be
    /// checked by the implementation before any allocation or model mutation.
    /// This borrowed scope allocates no participant table or success ledger.
    pub fn batch<'a>(
        &'a self,
        inputs: &'a [ExecutionWork<'a>],
        vocabulary_size: usize,
    ) -> Result<ExecutionBatch<'a>> {
        if inputs.is_empty() || vocabulary_size == 0 {
            return Err(invalid("execution batch and vocabulary must be non-empty"));
        }
        for (index, input) in inputs.iter().enumerate() {
            self.require_sequence(input.sequence())?;
            input.validate()?;
            if let TokenWork::Decode { token, .. } = input.work {
                if usize::try_from(token.get())
                    .ok()
                    .is_none_or(|token| token >= vocabulary_size)
                {
                    return Err(invalid("decode token is outside the model vocabulary"));
                }
            }
            if inputs[..index]
                .iter()
                .any(|previous| previous.sequence() == input.sequence())
            {
                return Err(invalid("execution batch contains a duplicate incarnation"));
            }
        }
        Ok(ExecutionBatch {
            inputs,
            vocabulary_size,
            identity: self.next_identity()?,
        })
    }
}

/// Opaque process-local executor instance plus never-reused incarnation.
#[derive(Debug, Clone)]
pub struct ExecutorSequence {
    identity: Arc<SequenceIdentity>,
}

#[derive(Debug)]
struct SequenceIdentity {
    issuer: Arc<IssuerIdentity>,
    incarnation: NonZeroU64,
    // A terminal fact published by the original issuer, not another resource
    // registry. Its lifetime follows the caller's opaque identity, so removing
    // a closed native entry cannot erase proof needed by a later close observer.
    ended: std::sync::atomic::AtomicBool,
}

impl PartialEq for ExecutorSequence {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.identity, &other.identity)
    }
}

impl Eq for ExecutorSequence {}

impl Hash for ExecutorSequence {
    fn hash<H: Hasher>(&self, state: &mut H) {
        Arc::as_ptr(&self.identity.issuer).hash(state);
        self.identity.incarnation.hash(state);
    }
}

/// An implementation-attested position bound to one exact incarnation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionFrontier {
    sequence: ExecutorSequence,
    consumed_tokens: usize,
}

impl ExecutionFrontier {
    pub fn sequence(&self) -> &ExecutorSequence {
        &self.sequence
    }

    pub const fn consumed_tokens(&self) -> usize {
        self.consumed_tokens
    }
}

/// A non-empty, non-overflowing causal token range; not execution evidence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionRange {
    start: usize,
    end: usize,
}

impl ExecutionRange {
    pub fn new(start: usize, token_count: usize) -> Result<Self> {
        let end = start
            .checked_add(token_count)
            .ok_or_else(|| invalid("execution range overflows"))?;
        if token_count == 0 {
            return Err(invalid("execution range must be non-empty"));
        }
        Ok(Self { start, end })
    }

    pub const fn start(self) -> usize {
        self.start
    }

    pub const fn end(self) -> usize {
        self.end
    }

    pub fn as_range(self) -> Range<usize> {
        self.start..self.end
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TokenWork {
    /// A range of the prompt retained at admission, including one-token tails.
    Prefill(PrefillChunk),
    Decode {
        token: TokenId,
        position: usize,
    },
}

impl TokenWork {
    pub fn range(self) -> Result<ExecutionRange> {
        match self {
            Self::Prefill(chunk) => {
                ExecutionRange::new(chunk.tokens_processed(), chunk.tokens_to_process())
            }
            Self::Decode { position, .. } => ExecutionRange::new(position, 1),
        }
    }
}

#[derive(Debug)]
pub struct ExecutionWork<'a> {
    expected_frontier: &'a ExecutionFrontier,
    work: TokenWork,
    logits_policy: &'a LogitsReturnPolicy,
}

impl<'a> ExecutionWork<'a> {
    pub fn new(
        expected_frontier: &'a ExecutionFrontier,
        work: TokenWork,
        logits_policy: &'a LogitsReturnPolicy,
    ) -> Result<Self> {
        let input = Self {
            expected_frontier,
            work,
            logits_policy,
        };
        input.validate()?;
        Ok(input)
    }

    fn validate(&self) -> Result<()> {
        if self.work.range()?.start() != self.expected_frontier.consumed_tokens {
            return Err(invalid("work does not start at its expected frontier"));
        }
        Ok(())
    }

    pub fn sequence(&self) -> &ExecutorSequence {
        self.expected_frontier.sequence()
    }

    pub fn expected_frontier(&self) -> &ExecutionFrontier {
        self.expected_frontier
    }

    pub const fn work(&self) -> TokenWork {
        self.work
    }

    pub fn logits_policy(&self) -> &LogitsReturnPolicy {
        self.logits_policy
    }
}

/// Actual completed work; prefill retains the existing capacity-probe accounting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutedTokenWork {
    Prefill {
        chunk: PrefillChunk,
        capacity_probe_count: u32,
    },
    Decode {
        position: usize,
    },
}

impl ExecutedTokenWork {
    pub fn range(self) -> Result<ExecutionRange> {
        match self {
            Self::Prefill { chunk, .. } => {
                ExecutionRange::new(chunk.tokens_processed(), chunk.tokens_to_process())
            }
            Self::Decode { position } => ExecutionRange::new(position, 1),
        }
    }

    fn validate_for(self, planned: TokenWork) -> Result<bool> {
        match (self, planned) {
            (
                Self::Prefill {
                    chunk,
                    capacity_probe_count,
                },
                TokenWork::Prefill(planned),
            ) => {
                if chunk.tokens_processed() != planned.tokens_processed()
                    || chunk.total_prompt_tokens() != planned.total_prompt_tokens()
                    || chunk.tokens_to_process() > planned.tokens_to_process()
                    || (chunk != planned && capacity_probe_count == 0)
                {
                    return Err(invalid(
                        "invalid completed prefill prefix or capacity probe",
                    ));
                }
                Ok(chunk.is_final())
            }
            (
                Self::Decode { position },
                TokenWork::Decode {
                    position: planned, ..
                },
            ) if position == planned => {
                self.range()?;
                Ok(true)
            }
            _ => Err(invalid(
                "completed work does not match its planned role/range",
            )),
        }
    }
}

#[derive(Debug)]
pub enum ExecutionProduct {
    Intermediate,
    Sampling(ExecutorSamplingOutput),
}

impl ExecutionProduct {
    fn validate_for(
        &self,
        executed: ExecutedTokenWork,
        input: &ExecutionWork<'_>,
        vocabulary_size: usize,
    ) -> Result<()> {
        let needs_sampling = executed.validate_for(input.work)?;
        match (self, needs_sampling) {
            (Self::Intermediate, false) => Ok(()),
            (Self::Sampling(output), true) => {
                output.validate_for_policy(input.logits_policy, vocabulary_size)
            }
            _ => Err(invalid(
                "completion product does not match actual work finality",
            )),
        }
    }
}

/// A successfully retired model operation and committed executor frontier.
/// Product sampling/publication may still fail after the Engine receives it.
#[derive(Debug)]
pub struct ExecutionCompletion {
    batch_identity: NonZeroU64,
    input_index: usize,
    next_frontier: ExecutionFrontier,
    executed: ExecutedTokenWork,
    product: ExecutionProduct,
}

impl ExecutionCompletion {
    pub const fn input_index(&self) -> usize {
        self.input_index
    }

    pub fn next_frontier(&self) -> &ExecutionFrontier {
        &self.next_frontier
    }

    pub const fn executed(&self) -> ExecutedTokenWork {
        self.executed
    }

    pub fn product(&self) -> &ExecutionProduct {
        &self.product
    }

    /// Check an individual result against the caller's retained input before
    /// publishing it. In particular, ContractViolation may retain a completion
    /// from another batch. The caller must separately reject duplicate indices
    /// and conflicts with interrupted/rejected participants across the batch.
    pub fn validate_for(
        &self,
        input_index: usize,
        input: &ExecutionWork<'_>,
        vocabulary_size: usize,
    ) -> Result<()> {
        input.validate()?;
        if self.input_index != input_index
            || self.next_frontier.sequence() != input.sequence()
            || self.executed.range()?.start() != input.expected_frontier.consumed_tokens()
            || self.executed.range()?.end() != self.next_frontier.consumed_tokens()
        {
            return Err(invalid(
                "completion does not match the caller's current incarnation/frontier",
            ));
        }
        self.product
            .validate_for(self.executed, input, vocabulary_size)
    }

    pub fn into_parts(
        self,
    ) -> (
        usize,
        ExecutionFrontier,
        ExecutedTokenWork,
        ExecutionProduct,
    ) {
        (
            self.input_index,
            self.next_frontier,
            self.executed,
            self.product,
        )
    }
}

#[derive(Debug)]
pub struct CompletedExecutionBatch {
    completions: Vec<ExecutionCompletion>,
}

impl CompletedExecutionBatch {
    pub fn completions(&self) -> &[ExecutionCompletion] {
        &self.completions
    }

    pub fn into_completions(self) -> Vec<ExecutionCompletion> {
        self.completions
    }
}

#[derive(Debug)]
pub enum NoSubmissionReason {
    /// The implementation must validate the existing evidence's wake/retry scope.
    Deferred(ExecutorExecutionDeferral),
    UnsupportedShape,
    Rejected(FerrumError),
}

/// Whole-batch zero-submit outcome with every original frontier still usable.
/// Allocator growth may remain. Only `UnsupportedShape` permits shape fallback;
/// `Rejected` is not scheduling permission merely because no model work ran.
#[derive(Debug)]
pub struct BatchNotSubmitted {
    reason: NoSubmissionReason,
}

impl BatchNotSubmitted {
    pub fn reason(&self) -> &NoSubmissionReason {
        &self.reason
    }

    pub fn into_reason(self) -> NoSubmissionReason {
        self.reason
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoSubmissionStage {
    /// No submission occurred; the implementation has no finer phase evidence.
    BeforeSubmit,
    Validation,
    Preparation,
    Encoding,
    DefinitelyNotEnqueued,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminalModelDisposition {
    Successful,
    Failed,
    /// Quiescence was proved; successful model execution was not established.
    UnknownAfterQuiescence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LastKnownExecutionStage {
    EnteredSubmit,
    Submitted,
    /// For example, model work ended but an auxiliary checkpoint copy is unknown.
    ModelTerminal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionProgress {
    NotSubmitted {
        stage: NoSubmissionStage,
    },
    Terminal {
        disposition: TerminalModelDisposition,
        executed_range: Option<ExecutionRange>,
    },
    Unknown {
        last_known: LastKnownExecutionStage,
        known_executed_range: Option<ExecutionRange>,
    },
}

/// Cloning the key duplicates identity, never the retained native recovery owner.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ExecutionRecoveryKey {
    sequence: ExecutorSequence,
    recovery: NonZeroU64,
}

impl ExecutionRecoveryKey {
    pub fn sequence(&self) -> &ExecutorSequence {
        &self.sequence
    }
}

#[derive(Debug)]
pub enum ContinuationDisposition {
    /// The original work can be attempted again at exactly the original frontier.
    Retryable(ExecutionFrontier),
    Ended(ExecutionEnded),
    Held(ExecutionHeld),
}

#[derive(Debug)]
pub struct InterruptedParticipant {
    batch_identity: NonZeroU64,
    input_index: usize,
    sequence: ExecutorSequence,
    progress: ExecutionProgress,
    continuation: ContinuationDisposition,
    cause_index: usize,
}

impl InterruptedParticipant {
    pub const fn input_index(&self) -> usize {
        self.input_index
    }

    pub fn sequence(&self) -> &ExecutorSequence {
        &self.sequence
    }

    pub const fn progress(&self) -> ExecutionProgress {
        self.progress
    }

    pub fn continuation(&self) -> &ContinuationDisposition {
        &self.continuation
    }

    pub const fn cause_index(&self) -> usize {
        self.cause_index
    }

    pub fn into_parts(
        self,
    ) -> (
        usize,
        ExecutorSequence,
        ExecutionProgress,
        ContinuationDisposition,
        usize,
    ) {
        (
            self.input_index,
            self.sequence,
            self.progress,
            self.continuation,
            self.cause_index,
        )
    }
}

/// Cold-path ownership of accumulated products and a complete failure ledger.
#[derive(Debug)]
pub struct BatchInterruption {
    completed: Vec<ExecutionCompletion>,
    remaining: Vec<InterruptedParticipant>,
    causes: Vec<FerrumError>,
}

impl BatchInterruption {
    pub fn completed(&self) -> &[ExecutionCompletion] {
        &self.completed
    }

    pub fn remaining(&self) -> &[InterruptedParticipant] {
        &self.remaining
    }

    pub fn causes(&self) -> &[FerrumError] {
        &self.causes
    }

    pub fn into_parts(self) -> InterruptionParts {
        (self.completed, self.remaining, self.causes)
    }
}

pub type InterruptionParts = (
    Vec<ExecutionCompletion>,
    Vec<InterruptedParticipant>,
    Vec<FerrumError>,
);

#[derive(Debug)]
pub enum ExecutionBatchOutcome {
    Completed(CompletedExecutionBatch),
    NotSubmitted(BatchNotSubmitted),
    Interrupted(BatchInterruption),
    /// The implementation produced inconsistent result metadata after work may
    /// have run. Preserve every observation; this never permits batch replay.
    ContractViolation(ExecutionContractViolation),
}

pub type CompletionParts = (usize, ExecutedTokenWork, ExecutionProduct);
pub type ParticipantParts = (usize, ExecutionProgress, ContinuationDisposition, usize);

/// Cold-path diagnostic facts, including rejected records that cannot be
/// published as valid products. Their retained close observers remain owned.
#[derive(Debug)]
pub struct ExecutionContractViolation {
    pub error: FerrumError,
    pub completed: Vec<ExecutionCompletion>,
    pub remaining: Vec<InterruptedParticipant>,
    pub causes: Vec<FerrumError>,
    pub rejected_completions: Vec<CompletionParts>,
    pub rejected_participants: Vec<ParticipantParts>,
    /// Extra or unassignable rows from a malformed model result. They have no
    /// fabricated participant index and cannot be published as completions.
    pub unassigned_products: Vec<ExecutionProduct>,
}

impl ExecutionContractViolation {
    pub fn new(error: FerrumError, parts: InterruptionParts) -> Self {
        Self {
            error,
            completed: parts.0,
            remaining: parts.1,
            causes: parts.2,
            rejected_completions: Vec::new(),
            rejected_participants: Vec::new(),
            unassigned_products: Vec::new(),
        }
    }
}

/// Validation failure retains collected products/ledger for honest error handling.
/// It is not a no-submit certificate or permission to replay the old batch.
#[derive(Debug)]
pub struct RejectedBatch<T> {
    error: FerrumError,
    parts: T,
}

impl<T> RejectedBatch<T> {
    pub fn error(&self) -> &FerrumError {
        &self.error
    }

    pub fn into_parts(self) -> (FerrumError, T) {
        (self.error, self.parts)
    }
}

impl RejectedBatch<Vec<ExecutionCompletion>> {
    pub fn into_outcome(self) -> ExecutionBatchOutcome {
        ExecutionBatchOutcome::ContractViolation(ExecutionContractViolation::new(
            self.error,
            (self.parts, Vec::new(), Vec::new()),
        ))
    }
}

impl RejectedBatch<InterruptionParts> {
    pub fn into_outcome(self) -> ExecutionBatchOutcome {
        ExecutionBatchOutcome::ContractViolation(ExecutionContractViolation::new(
            self.error, self.parts,
        ))
    }
}

impl RejectedBatch<CompletionParts> {
    pub fn into_outcome(self, collected: InterruptionParts) -> ExecutionBatchOutcome {
        let mut violation = ExecutionContractViolation::new(self.error, collected);
        violation.rejected_completions.push(self.parts);
        ExecutionBatchOutcome::ContractViolation(violation)
    }
}

impl RejectedBatch<ParticipantParts> {
    pub fn into_outcome(self, collected: InterruptionParts) -> ExecutionBatchOutcome {
        let mut violation = ExecutionContractViolation::new(self.error, collected);
        violation.rejected_participants.push(self.parts);
        ExecutionBatchOutcome::ContractViolation(violation)
    }
}

/// Borrowed checked input scope. Successful execution retains only its one Vec.
#[derive(Debug)]
pub struct ExecutionBatch<'a> {
    inputs: &'a [ExecutionWork<'a>],
    vocabulary_size: usize,
    identity: NonZeroU64,
}

impl ExecutionBatch<'_> {
    fn input(&self, index: usize) -> Result<&ExecutionWork<'_>> {
        self.inputs
            .get(index)
            .ok_or_else(|| invalid("execution result input index is out of bounds"))
    }

    /// Attest actual terminal success, retirement and committed executor frontier.
    /// A validation error here is post-execution and must preserve that progress.
    pub fn completion(
        &self,
        input_index: usize,
        executed: ExecutedTokenWork,
        product: ExecutionProduct,
    ) -> std::result::Result<
        ExecutionCompletion,
        RejectedBatch<(usize, ExecutedTokenWork, ExecutionProduct)>,
    > {
        let checked = (|| {
            let input = self.input(input_index)?;
            product.validate_for(executed, input, self.vocabulary_size)?;
            Ok(ExecutionFrontier {
                sequence: input.sequence().clone(),
                consumed_tokens: executed.range()?.end(),
            })
        })();
        let next_frontier = match checked {
            Ok(frontier) => frontier,
            Err(error) => {
                return Err(RejectedBatch {
                    error,
                    parts: (input_index, executed, product),
                })
            }
        };
        Ok(ExecutionCompletion {
            batch_identity: self.identity,
            input_index,
            next_frontier,
            executed,
            product,
        })
    }

    /// Record actual progress separately from continuation cleanup/usability.
    pub fn participant(
        &self,
        input_index: usize,
        progress: ExecutionProgress,
        continuation: ContinuationDisposition,
        cause_index: usize,
    ) -> std::result::Result<
        InterruptedParticipant,
        RejectedBatch<(usize, ExecutionProgress, ContinuationDisposition, usize)>,
    > {
        let checked = (|| {
            let input = self.input(input_index)?;
            let known_range = match progress {
                ExecutionProgress::NotSubmitted { .. } => None,
                ExecutionProgress::Terminal {
                    disposition,
                    executed_range,
                } => {
                    if disposition == TerminalModelDisposition::Successful
                        && executed_range.is_none()
                    {
                        return Err(invalid(
                            "successful terminal work requires its actual range",
                        ));
                    }
                    executed_range
                }
                ExecutionProgress::Unknown {
                    last_known,
                    known_executed_range,
                } => {
                    if known_executed_range.is_some()
                        && last_known != LastKnownExecutionStage::ModelTerminal
                    {
                        return Err(invalid(
                            "known executed range requires known model completion",
                        ));
                    }
                    known_executed_range
                }
            };
            if let Some(range) = known_range {
                let planned = input.work.range()?;
                if range.start() != planned.start() || range.end() > planned.end() {
                    return Err(invalid(
                        "known execution is outside the participant's planned range",
                    ));
                }
            }
            match &continuation {
                ContinuationDisposition::Retryable(frontier) => {
                    if !matches!(progress, ExecutionProgress::NotSubmitted { .. })
                        || frontier != input.expected_frontier
                    {
                        return Err(invalid(
                            "retry requires zero submission and the original frontier",
                        ));
                    }
                }
                ContinuationDisposition::Ended(ended) => {
                    if ended.sequence() != input.sequence() {
                        return Err(invalid(
                            "terminal result belongs to a different incarnation",
                        ));
                    }
                    if matches!(progress, ExecutionProgress::Unknown { .. }) {
                        return Err(invalid("unknown execution must retain its recovery owner"));
                    }
                }
                ContinuationDisposition::Held(held) => {
                    if held.key().sequence() != input.sequence() {
                        return Err(invalid("recovery key belongs to a different incarnation"));
                    }
                }
            }
            Ok(input.sequence().clone())
        })();
        let sequence = match checked {
            Ok(sequence) => sequence,
            Err(error) => {
                return Err(RejectedBatch {
                    error,
                    parts: (input_index, progress, continuation, cause_index),
                })
            }
        };
        Ok(InterruptedParticipant {
            batch_identity: self.identity,
            input_index,
            sequence,
            progress,
            continuation,
            cause_index,
        })
    }

    fn validate_completion_scope(&self, completion: &ExecutionCompletion) -> Result<()> {
        if completion.batch_identity != self.identity
            || completion.next_frontier.sequence() != self.input(completion.input_index)?.sequence()
        {
            return Err(invalid(
                "completion belongs to a different batch/incarnation",
            ));
        }
        Ok(())
    }

    pub fn completed(
        &self,
        completions: Vec<ExecutionCompletion>,
    ) -> std::result::Result<ExecutionBatchOutcome, RejectedBatch<Vec<ExecutionCompletion>>> {
        let check = (|| {
            if completions.len() != self.inputs.len() {
                return Err(invalid("completed batch does not cover every input"));
            }
            for (index, completion) in completions.iter().enumerate() {
                self.validate_completion_scope(completion)?;
                if completion.input_index != index {
                    return Err(invalid("completed batch must preserve exact input order"));
                }
            }
            Ok(())
        })();
        match check {
            Ok(()) => Ok(ExecutionBatchOutcome::Completed(CompletedExecutionBatch {
                completions,
            })),
            Err(error) => Err(RejectedBatch {
                error,
                parts: completions,
            }),
        }
    }

    /// Attest no model command enqueued and every original frontier still usable.
    /// This method cannot establish those facts from error variants or counters.
    pub fn not_submitted(&self, reason: NoSubmissionReason) -> ExecutionBatchOutcome {
        ExecutionBatchOutcome::NotSubmitted(BatchNotSubmitted { reason })
    }

    pub fn interrupted(
        &self,
        completed: Vec<ExecutionCompletion>,
        remaining: Vec<InterruptedParticipant>,
        causes: Vec<FerrumError>,
    ) -> std::result::Result<ExecutionBatchOutcome, RejectedBatch<InterruptionParts>> {
        let check = (|| {
            if remaining.is_empty()
                || completed.len().checked_add(remaining.len()) != Some(self.inputs.len())
            {
                return Err(invalid(
                    "interruption must cover every input and include a failure",
                ));
            }
            for completion in &completed {
                self.validate_completion_scope(completion)?;
            }
            for participant in &remaining {
                if participant.batch_identity != self.identity
                    || &participant.sequence != self.input(participant.input_index)?.sequence()
                    || participant.cause_index >= causes.len()
                {
                    return Err(invalid(
                        "invalid interrupted participant scope or cause index",
                    ));
                }
            }
            let mut successes = completed.iter().peekable();
            let mut failures = remaining.iter().peekable();
            for index in 0..self.inputs.len() {
                let success = successes
                    .peek()
                    .is_some_and(|item| item.input_index == index);
                let failure = failures
                    .peek()
                    .is_some_and(|item| item.input_index == index);
                match (success, failure) {
                    (true, false) => {
                        successes.next();
                    }
                    (false, true) => {
                        failures.next();
                    }
                    _ => {
                        return Err(invalid(
                            "interruption coverage is missing, duplicated or unordered",
                        ))
                    }
                }
            }
            if successes.next().is_some() || failures.next().is_some() {
                return Err(invalid("interruption has extra participants"));
            }
            Ok(())
        })();
        match check {
            Ok(()) => Ok(ExecutionBatchOutcome::Interrupted(BatchInterruption {
                completed,
                remaining,
                causes,
            })),
            Err(error) => Err(RejectedBatch {
                error,
                parts: (completed, remaining, causes),
            }),
        }
    }
}

fn invalid(message: &'static str) -> FerrumError {
    FerrumError::request_validation(message)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn held(
        issuer: &ExecutorSequenceIssuer,
        sequence: &ExecutorSequence,
    ) -> ContinuationDisposition {
        ContinuationDisposition::Held(
            issuer
                .held(
                    issuer.recovery_key(sequence).unwrap(),
                    std::future::pending(),
                )
                .unwrap(),
        )
    }

    fn ended(
        issuer: &ExecutorSequenceIssuer,
        sequence: &ExecutorSequence,
    ) -> ContinuationDisposition {
        ContinuationDisposition::Ended(issuer.ended(sequence).unwrap())
    }

    fn decode<'a>(
        frontier: &'a ExecutionFrontier,
        policy: &'a LogitsReturnPolicy,
    ) -> ExecutionWork<'a> {
        ExecutionWork::new(
            frontier,
            TokenWork::Decode {
                token: TokenId(1),
                position: frontier.consumed_tokens(),
            },
            policy,
        )
        .unwrap()
    }

    fn logits() -> ExecutionProduct {
        ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(vec![0.25, -0.5]))
    }

    fn not_submitted() -> ExecutionProgress {
        ExecutionProgress::NotSubmitted {
            stage: NoSubmissionStage::Preparation,
        }
    }

    #[test]
    fn issuer_rejects_foreign_sequences_and_never_reuses_incarnations() {
        let first = ExecutorSequenceIssuer::new();
        let second = ExecutorSequenceIssuer::new();
        let sequence = first.issue_sequence().unwrap();
        let other = second.issue_sequence().unwrap();
        assert_ne!(sequence, other);
        assert_ne!(sequence, first.issue_sequence().unwrap());
        assert!(second.frontier(&sequence, 0).is_err());
        assert!(second.recovery_key(&sequence).is_err());
        let frontier = first.frontier(&sequence, 0).unwrap();
        let policy = LogitsReturnPolicy::FullLogits;
        let inputs = [decode(&frontier, &policy)];
        assert!(second.batch(&inputs, 2).is_err());
        first
            .identity
            .next_identity
            .store(u64::MAX, Ordering::Relaxed);
        assert!(first.issue_sequence().is_err());
        assert!(first.issue_sequence().is_err());
    }

    #[test]
    fn work_rejects_overflow_wrong_frontier_and_duplicate_incarnation() {
        assert!(ExecutionRange::new(0, 0).is_err());
        assert!(ExecutionRange::new(usize::MAX, 1).is_err());
        let issuer = ExecutorSequenceIssuer::new();
        let sequence = issuer.issue_sequence().unwrap();
        let frontier = issuer.frontier(&sequence, 7).unwrap();
        let policy = LogitsReturnPolicy::FullLogits;
        assert!(ExecutionWork::new(
            &frontier,
            TokenWork::Decode {
                token: TokenId(1),
                position: 6
            },
            &policy,
        )
        .is_err());
        let inputs = [decode(&frontier, &policy), decode(&frontier, &policy)];
        assert!(issuer.batch(&inputs, 2).is_err());
        let invalid_token = [ExecutionWork::new(
            &frontier,
            TokenWork::Decode {
                token: TokenId(2),
                position: 7,
            },
            &policy,
        )
        .unwrap()];
        assert!(issuer.batch(&invalid_token, 2).is_err());
        let overflow = issuer.frontier(&sequence, usize::MAX).unwrap();
        assert!(ExecutionWork::new(
            &overflow,
            TokenWork::Decode {
                token: TokenId(1),
                position: usize::MAX
            },
            &policy,
        )
        .is_err());
    }

    #[test]
    fn retained_completion_requires_the_callers_current_input_before_publication() {
        let issuer = ExecutorSequenceIssuer::new();
        let sequence = issuer.issue_sequence().unwrap();
        let frontier = issuer.frontier(&sequence, 3).unwrap();
        let policy = LogitsReturnPolicy::FullLogits;
        let inputs = [decode(&frontier, &policy)];
        let scope = issuer.batch(&inputs, 2).unwrap();
        let completion = scope
            .completion(0, ExecutedTokenWork::Decode { position: 3 }, logits())
            .unwrap();

        completion.validate_for(0, &inputs[0], 2).unwrap();
        assert!(completion.validate_for(1, &inputs[0], 2).is_err());
        assert!(completion.validate_for(0, &inputs[0], 3).is_err());

        let replacement = issuer
            .frontier(&issuer.issue_sequence().unwrap(), 3)
            .unwrap();
        assert!(completion
            .validate_for(0, &decode(&replacement, &policy), 2)
            .is_err());
        assert!(completion
            .validate_for(0, &decode(completion.next_frontier(), &policy), 2)
            .is_err());
        let prefill = ExecutionWork::new(
            &frontier,
            TokenWork::Prefill(PrefillChunk::new(3, 1, 4).unwrap()),
            &policy,
        )
        .unwrap();
        assert!(completion.validate_for(0, &prefill, 2).is_err());
        // Rejecting stale or foreign publication must retain the original
        // successfully executed product for the cold-path consumer.
        assert!(matches!(
            completion.product(),
            ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(values))
                if values == &[0.25, -0.5]
        ));
    }

    #[test]
    fn completion_preserves_one_token_prefill_role_and_sampling_policy() {
        let issuer = ExecutorSequenceIssuer::new();
        let sequence = issuer.issue_sequence().unwrap();
        let frontier = issuer.frontier(&sequence, 3).unwrap();
        let policy = LogitsReturnPolicy::FullLogits;
        let chunk = PrefillChunk::new(3, 1, 4).unwrap();
        let inputs = [ExecutionWork::new(&frontier, TokenWork::Prefill(chunk), &policy).unwrap()];
        let batch = issuer.batch(&inputs, 2).unwrap();
        let executed = ExecutedTokenWork::Prefill {
            chunk,
            capacity_probe_count: 0,
        };
        assert!(batch
            .completion(0, ExecutedTokenWork::Decode { position: 3 }, logits())
            .is_err());
        assert!(batch
            .completion(0, executed, ExecutionProduct::Intermediate)
            .is_err());
        assert!(batch
            .completion(
                0,
                executed,
                ExecutionProduct::Sampling(ExecutorSamplingOutput::greedy_token(TokenId(1)),)
            )
            .is_err());
        assert!(batch
            .completion(
                0,
                executed,
                ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(vec![0.0]),)
            )
            .is_err());
        let completion = batch.completion(0, executed, logits()).unwrap();
        assert_eq!(completion.executed().range().unwrap().as_range(), 3..4);
        assert_eq!(completion.next_frontier().consumed_tokens(), 4);
        let ExecutionBatchOutcome::Completed(result) = batch.completed(vec![completion]).unwrap()
        else {
            panic!("expected completed batch");
        };
        assert_eq!(result.completions().len(), 1);
    }

    #[test]
    fn narrowed_prefill_requires_probe_and_has_no_sampling_product() {
        let issuer = ExecutorSequenceIssuer::new();
        let sequence = issuer.issue_sequence().unwrap();
        let frontier = issuer.frontier(&sequence, 0).unwrap();
        let policy = LogitsReturnPolicy::FullLogits;
        let planned = PrefillChunk::new(0, 4, 4).unwrap();
        let actual = PrefillChunk::new(0, 2, 4).unwrap();
        let inputs = [ExecutionWork::new(&frontier, TokenWork::Prefill(planned), &policy).unwrap()];
        let batch = issuer.batch(&inputs, 2).unwrap();
        assert!(batch
            .completion(
                0,
                ExecutedTokenWork::Prefill {
                    chunk: actual,
                    capacity_probe_count: 0,
                },
                ExecutionProduct::Intermediate
            )
            .is_err());
        let executed = ExecutedTokenWork::Prefill {
            chunk: actual,
            capacity_probe_count: 1,
        };
        assert!(batch.completion(0, executed, logits()).is_err());
        let completion = batch
            .completion(0, executed, ExecutionProduct::Intermediate)
            .unwrap();
        assert_eq!(completion.next_frontier().consumed_tokens(), 2);
        assert!(batch.completed(vec![completion]).is_ok());
    }

    #[test]
    fn completed_coverage_is_exact_ordered_and_bound_to_one_batch() {
        let issuer = ExecutorSequenceIssuer::new();
        let first = issuer
            .frontier(&issuer.issue_sequence().unwrap(), 2)
            .unwrap();
        let second = issuer
            .frontier(&issuer.issue_sequence().unwrap(), 5)
            .unwrap();
        let policy = LogitsReturnPolicy::FullLogits;
        let inputs = [decode(&first, &policy), decode(&second, &policy)];
        let batch = issuer.batch(&inputs, 2).unwrap();
        let one = batch
            .completion(0, ExecutedTokenWork::Decode { position: 2 }, logits())
            .unwrap();
        let two = batch
            .completion(1, ExecutedTokenWork::Decode { position: 5 }, logits())
            .unwrap();
        let error = batch.completed(vec![two, one]).unwrap_err();
        let (_, mut retained) = error.into_parts();
        assert_eq!(retained.len(), 2);
        assert_eq!(retained[1].input_index(), 0);
        retained.reverse();
        assert_eq!(retained[0].input_index(), 0);
        let next_batch = issuer.batch(&inputs, 2).unwrap();
        assert!(next_batch.completed(retained).is_err());
        let incomplete = issuer.batch(&inputs, 2).unwrap();
        let one = incomplete
            .completion(0, ExecutedTokenWork::Decode { position: 2 }, logits())
            .unwrap();
        let (error, retained) = incomplete.completed(vec![one]).unwrap_err().into_parts();
        assert_eq!(retained.len(), 1);
        let missing = incomplete
            .participant(
                1,
                ExecutionProgress::Terminal {
                    disposition: TerminalModelDisposition::Successful,
                    executed_range: Some(ExecutionRange::new(5, 1).unwrap()),
                },
                ended(&issuer, second.sequence()),
                0,
            )
            .unwrap();
        let ExecutionBatchOutcome::Interrupted(recovered) = incomplete
            .interrupted(retained, vec![missing], vec![error])
            .unwrap()
        else {
            panic!("expected preserved interruption");
        };
        assert_eq!(
            recovered.completed()[0].next_frontier().consumed_tokens(),
            3
        );
    }

    #[test]
    fn interruption_preserves_success_and_distinguishes_failed_and_untouched_peers() {
        let issuer = ExecutorSequenceIssuer::new();
        let first = issuer
            .frontier(&issuer.issue_sequence().unwrap(), 2)
            .unwrap();
        let second = issuer
            .frontier(&issuer.issue_sequence().unwrap(), 5)
            .unwrap();
        let third = issuer
            .frontier(&issuer.issue_sequence().unwrap(), 8)
            .unwrap();
        let policy = LogitsReturnPolicy::FullLogits;
        let inputs = [
            decode(&first, &policy),
            decode(&second, &policy),
            decode(&third, &policy),
        ];
        let batch = issuer.batch(&inputs, 2).unwrap();
        let completed = batch
            .completion(0, ExecutedTokenWork::Decode { position: 2 }, logits())
            .unwrap();
        let failed = batch
            .participant(
                1,
                ExecutionProgress::Terminal {
                    disposition: TerminalModelDisposition::Successful,
                    executed_range: Some(ExecutionRange::new(5, 1).unwrap()),
                },
                ended(&issuer, second.sequence()),
                0,
            )
            .unwrap();
        let untouched = batch
            .participant(
                2,
                not_submitted(),
                ContinuationDisposition::Retryable(third.clone()),
                0,
            )
            .unwrap();
        let ExecutionBatchOutcome::Interrupted(result) = batch
            .interrupted(
                vec![completed],
                vec![failed, untouched],
                vec![FerrumError::backend("readback failed")],
            )
            .unwrap()
        else {
            panic!("expected interruption");
        };
        assert_eq!(result.completed()[0].next_frontier().consumed_tokens(), 3);
        assert!(matches!(
            result.remaining()[0].continuation(),
            ContinuationDisposition::Ended(_)
        ));
        assert!(matches!(
            result.remaining()[1].continuation(),
            ContinuationDisposition::Retryable(_)
        ));
        assert_eq!(result.causes().len(), 1);
    }

    #[test]
    fn unknown_requires_same_sequence_recovery_and_never_grants_retry() {
        let issuer = ExecutorSequenceIssuer::new();
        let sequence = issuer.issue_sequence().unwrap();
        let frontier = issuer.frontier(&sequence, 4).unwrap();
        let other = issuer.issue_sequence().unwrap();
        let policy = LogitsReturnPolicy::FullLogits;
        let inputs = [decode(&frontier, &policy)];
        let batch = issuer.batch(&inputs, 2).unwrap();
        let progress = ExecutionProgress::Unknown {
            last_known: LastKnownExecutionStage::Submitted,
            known_executed_range: None,
        };
        assert!(batch
            .participant(0, progress, ended(&issuer, &sequence), 0)
            .is_err());
        assert!(batch
            .participant(
                0,
                progress,
                ContinuationDisposition::Retryable(frontier.clone()),
                0
            )
            .is_err());
        assert!(batch
            .participant(0, progress, held(&issuer, &other), 0)
            .is_err());
        let participant = batch
            .participant(0, progress, held(&issuer, &sequence), 0)
            .unwrap();
        assert!(batch
            .interrupted(
                vec![],
                vec![participant],
                vec![FerrumError::backend("pending")]
            )
            .is_ok());
    }

    #[test]
    fn retry_requires_exact_original_frontier_even_after_zero_submission() {
        let issuer = ExecutorSequenceIssuer::new();
        let sequence = issuer.issue_sequence().unwrap();
        let frontier = issuer.frontier(&sequence, 4).unwrap();
        let policy = LogitsReturnPolicy::FullLogits;
        let inputs = [decode(&frontier, &policy)];
        let batch = issuer.batch(&inputs, 2).unwrap();
        let advanced = issuer.frontier(&sequence, 5).unwrap();
        assert!(batch
            .participant(
                0,
                not_submitted(),
                ContinuationDisposition::Retryable(advanced),
                0
            )
            .is_err());
        let terminal = ExecutionProgress::Terminal {
            disposition: TerminalModelDisposition::Successful,
            executed_range: Some(ExecutionRange::new(4, 1).unwrap()),
        };
        assert!(batch
            .participant(
                0,
                terminal,
                ContinuationDisposition::Retryable(frontier.clone()),
                0
            )
            .is_err());
        assert!(batch
            .participant(0, not_submitted(), ended(&issuer, &sequence), 0)
            .is_ok());
    }

    #[test]
    fn known_model_work_survives_unknown_auxiliary_completion() {
        let issuer = ExecutorSequenceIssuer::new();
        let sequence = issuer.issue_sequence().unwrap();
        let frontier = issuer.frontier(&sequence, 4).unwrap();
        let policy = LogitsReturnPolicy::FullLogits;
        let inputs = [decode(&frontier, &policy)];
        let batch = issuer.batch(&inputs, 2).unwrap();
        let range = ExecutionRange::new(4, 1).unwrap();
        let progress = ExecutionProgress::Unknown {
            last_known: LastKnownExecutionStage::ModelTerminal,
            known_executed_range: Some(range),
        };
        let participant = batch
            .participant(0, progress, held(&issuer, &sequence), 0)
            .unwrap();
        assert_eq!(participant.progress(), progress);
        assert!(batch
            .participant(
                0,
                ExecutionProgress::Terminal {
                    disposition: TerminalModelDisposition::Successful,
                    executed_range: None,
                },
                ended(&issuer, &sequence),
                0
            )
            .is_err());
        assert!(batch
            .participant(
                0,
                ExecutionProgress::Terminal {
                    disposition: TerminalModelDisposition::Successful,
                    executed_range: Some(ExecutionRange::new(4, 2).unwrap()),
                },
                ended(&issuer, &sequence),
                0
            )
            .is_err());
    }

    #[test]
    fn interruption_rejects_conflicting_coverage_and_bad_cause_without_losing_parts() {
        let issuer = ExecutorSequenceIssuer::new();
        let first = issuer
            .frontier(&issuer.issue_sequence().unwrap(), 2)
            .unwrap();
        let second = issuer
            .frontier(&issuer.issue_sequence().unwrap(), 5)
            .unwrap();
        let policy = LogitsReturnPolicy::FullLogits;
        let inputs = [decode(&first, &policy), decode(&second, &policy)];
        let batch = issuer.batch(&inputs, 2).unwrap();
        let completed = batch
            .completion(0, ExecutedTokenWork::Decode { position: 2 }, logits())
            .unwrap();
        let duplicate = batch
            .participant(0, not_submitted(), ended(&issuer, first.sequence()), 0)
            .unwrap();
        let error = batch
            .interrupted(
                vec![completed],
                vec![duplicate],
                vec![FerrumError::backend("failed")],
            )
            .unwrap_err();
        let (_, (completed, remaining, causes)) = error.into_parts();
        assert_eq!(completed.len(), 1);
        assert_eq!(remaining.len(), 1);
        assert_eq!(causes.len(), 1);
        let actual_failure = batch
            .participant(1, not_submitted(), ended(&issuer, second.sequence()), 0)
            .unwrap();
        assert!(batch
            .interrupted(completed, vec![actual_failure], causes)
            .is_ok());
        let single = [decode(&first, &policy)];
        let batch = issuer.batch(&single, 2).unwrap();
        let bad_cause = batch
            .participant(0, not_submitted(), ended(&issuer, first.sequence()), 1)
            .unwrap();
        assert!(batch
            .interrupted(
                vec![],
                vec![bad_cause],
                vec![FerrumError::backend("only cause")]
            )
            .is_err());
    }

    #[test]
    fn malformed_completion_keeps_original_product_as_post_execution_failure() {
        let issuer = ExecutorSequenceIssuer::new();
        let sequence = issuer.issue_sequence().unwrap();
        let frontier = issuer.frontier(&sequence, 2).unwrap();
        let policy = LogitsReturnPolicy::FullLogits;
        let inputs = [decode(&frontier, &policy)];
        let batch = issuer.batch(&inputs, 2).unwrap();
        let error = batch
            .completion(
                0,
                ExecutedTokenWork::Decode { position: 2 },
                ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(vec![0.5])),
            )
            .unwrap_err();
        let ExecutionBatchOutcome::ContractViolation(violation) =
            error.into_outcome((Vec::new(), Vec::new(), Vec::new()))
        else {
            panic!("invalid output cannot become a no-submit outcome")
        };
        let (_, actual, product) = violation.rejected_completions.into_iter().next().unwrap();
        assert_eq!(actual, ExecutedTokenWork::Decode { position: 2 });
        let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(logits)) = product else {
            panic!("original output must survive validation failure")
        };
        assert_eq!(logits, [0.5]);
    }

    #[test]
    fn malformed_participant_keeps_original_held_observer() {
        let issuer = ExecutorSequenceIssuer::new();
        let sequence = issuer.issue_sequence().unwrap();
        let frontier = issuer.frontier(&sequence, 2).unwrap();
        let policy = LogitsReturnPolicy::FullLogits;
        let inputs = [decode(&frontier, &policy)];
        let batch = issuer.batch(&inputs, 2).unwrap();
        let (sender, receiver) = futures::channel::oneshot::channel();
        let held = issuer
            .held(issuer.recovery_key(&sequence).unwrap(), async move {
                receiver
                    .await
                    .map_err(|_| invalid("terminal publisher disappeared"))
            })
            .unwrap();
        let error = batch
            .participant(
                1,
                ExecutionProgress::Unknown {
                    last_known: LastKnownExecutionStage::Submitted,
                    known_executed_range: None,
                },
                ContinuationDisposition::Held(held),
                0,
            )
            .unwrap_err();
        let ExecutionBatchOutcome::ContractViolation(violation) = error.into_outcome((
            Vec::new(),
            Vec::new(),
            vec![FerrumError::backend("original failure")],
        )) else {
            panic!("invalid metadata cannot discard actual recovery")
        };
        sender
            .send(ExecutionTerminalOutcome::ended(
                issuer.ended(&sequence).unwrap(),
            ))
            .unwrap();
        let (_, _, continuation, _) = violation.rejected_participants.into_iter().next().unwrap();
        let ContinuationDisposition::Held(held) = continuation else {
            panic!("missing original observer")
        };
        assert_eq!(
            futures::executor::block_on(held.wait())
                .unwrap()
                .ended
                .sequence(),
            &sequence
        );
    }
}
