//! Independent engine-evidence envelope within the move-only projection lease.
use super::OutputFlowError;
use ferrum_types::{
    EngineDecodeStageInterval, InferenceEvidenceRequest, InferenceExecutionEvidence, TokenId,
    TokenUsage,
};

/// Derived only from a resolved request, never supplied as a client multiplier.
/// The engine calls derivation after applying the model's context limit. This
/// plan adds storage to the existing per-request AND global projection limits.
/// No evidence enters the wire queue and no second unbounded log is created.
#[derive(Debug, Clone, Copy, Default)]
pub struct EngineEvidenceRetentionPlan {
    prompt_tokens: usize,
    output_tokens: usize,
    timing: bool,
    enabled: bool,
    stage_intervals: usize,
    bytes: usize,
}
impl EngineEvidenceRetentionPlan {
    pub(super) fn derive(
        request: &InferenceEvidenceRequest,
        prompt_tokens: usize,
        output_tokens: usize,
    ) -> Result<Self, OutputFlowError> {
        let enabled = request.capture_prompt_token_ids || request.capture_engine_token_timing;
        if !enabled {
            return Ok(Self::default());
        }
        let prompt_tokens = if request.capture_prompt_token_ids {
            prompt_tokens
        } else {
            0
        };
        let timing = request.capture_engine_token_timing;
        // Three stage kinds per possible committed token. Retries may generate
        // arbitrarily many stages without progress, so retain only this prefix
        // and explicitly count omissions. This is NOT a model execution limit.
        let stage_intervals = if timing {
            output_tokens
                .checked_mul(3)
                .ok_or(OutputFlowError::Overflow)?
        } else {
            0
        };
        let token_bytes = prompt_tokens
            .checked_add(output_tokens)
            .and_then(|n| n.checked_mul(size_of::<TokenId>()))
            .ok_or(OutputFlowError::Overflow)?;
        let timing_bytes = if timing {
            output_tokens
                .checked_mul(size_of::<u64>())
                .and_then(|n| {
                    stage_intervals
                        .checked_mul(size_of::<EngineDecodeStageInterval>())
                        .and_then(|s| n.checked_add(s))
                })
                .and_then(|n| n.checked_add("rust_std_instant".len()))
                .ok_or(OutputFlowError::Overflow)?
        } else {
            0
        };
        Ok(Self {
            prompt_tokens,
            output_tokens,
            timing,
            enabled,
            stage_intervals,
            bytes: token_bytes
                .checked_add(timing_bytes)
                .ok_or(OutputFlowError::Overflow)?,
        })
    }
    pub fn enabled(self) -> bool {
        self.enabled
    }
    pub fn captures_timing(self) -> bool {
        self.timing
    }
    pub fn maximum_output_tokens(self) -> usize {
        self.output_tokens
    }
    pub fn maximum_stage_intervals(self) -> usize {
        self.stage_intervals
    }
    pub fn retained_bytes(self) -> usize {
        self.bytes
    }

    /// Validate owned capacity, not merely payload length, before handing the
    /// terminal lease to an external consumer. No omitted commit is accepted.
    pub fn validate(
        self,
        evidence: Option<&InferenceExecutionEvidence>,
        usage: &TokenUsage,
    ) -> Result<(), OutputFlowError> {
        let fail = || OutputFlowError::BoundExceeded;
        if !self.enabled {
            return if evidence.is_none() {
                Ok(())
            } else {
                Err(fail())
            };
        }
        let evidence = evidence.ok_or_else(fail)?;
        if usage.completion_tokens > self.output_tokens
            || evidence.output_token_ids.len() != usage.completion_tokens
            || evidence.output_token_ids.capacity() > self.output_tokens
            || evidence.prompt_token_ids.len() != self.prompt_tokens
            || evidence.prompt_token_ids.capacity() > self.prompt_tokens
            || (self.prompt_tokens != 0 && usage.prompt_tokens != self.prompt_tokens)
        {
            return Err(fail());
        }
        match (self.timing, evidence.engine_token_timing.as_ref()) {
            (false, None) => Ok(()),
            (true, Some(timing))
                if timing.clock_source.capacity() <= "rust_std_instant".len()
                    && timing.token_commit_nanos_since_request_start.capacity()
                        <= self.output_tokens
                    && timing.decode_stage_intervals.capacity() <= self.stage_intervals =>
            {
                timing.validate(usage.completion_tokens).map_err(|_| fail())
            }
            _ => Err(fail()),
        }
    }
}
