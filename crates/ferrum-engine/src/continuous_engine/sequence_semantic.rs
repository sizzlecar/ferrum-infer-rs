//! Credit follows the original request state and moves with its final payload.
use super::*;

impl SequenceState {
    /// Completion has already moved tokens/evidence and captured its trace.
    /// Drop the remaining generated-history owners before reusing their credit
    /// for the immutable completion; clearing a collection alone is not handoff.
    pub(in crate::continuous_engine) fn retire_generated_output_state(&mut self) {
        drop(self.structured_output_processor.take());
        drop(std::mem::take(&mut self.sampling_history));
        drop(self.token_timing.take());
        drop(std::mem::take(&mut self.pending_decoded_utf8_bytes));
    }

    #[cfg(test)]
    pub(in crate::continuous_engine) fn bind_output_session(
        &mut self,
        session: OutputSession,
    ) -> Result<()> {
        // Publication calls this before any generated work. Direct fixtures may
        // bind an already populated history and pay its actual retained payload.
        let mut bytes = self
            .generated_tokens
            .len()
            .checked_mul(4)
            .ok_or_else(|| FerrumError::resource_exhausted("generated token size overflow"))?;
        bytes = bytes
            .checked_add(
                self.sampling_history.token_frequencies.len()
                    * SequenceSamplingHistory::ENTRY_BYTES,
            )
            .ok_or_else(|| FerrumError::resource_exhausted("sampling history size overflow"))?;
        if let Some(processor) = &self.structured_output_processor {
            // Preserve replay capacity, including tokens not yet consumed by
            // mask/stop/classification. This scan is once, before publication.
            for &token in &self.generated_tokens {
                bytes = bytes
                    .checked_add(processor.retained_token_growth_bound(token)?)
                    .ok_or_else(|| {
                        FerrumError::resource_exhausted("composed history size overflow")
                    })?;
            }
        }
        if let Some(timing) = &self.token_timing {
            bytes = bytes
                .checked_add(semantic_cost(
                    &timing.token_commit_nanos_since_request_start,
                )?)
                .and_then(|size| {
                    size.checked_add(semantic_cost(&timing.decode_stage_intervals).ok()?)
                })
                .ok_or_else(|| FerrumError::resource_exhausted("timing evidence size overflow"))?;
        }
        self.output_semantic = session.try_reserve_semantic(bytes)?;
        Ok(())
    }

    pub(in crate::continuous_engine) fn reserve_output_bytes(
        &mut self,
        bytes: usize,
    ) -> Result<()> {
        self.output_semantic.try_grow(bytes)
    }

    pub(in crate::continuous_engine) fn output_accounting_error(&self) -> Option<&FerrumError> {
        self.output_accounting_error.as_ref()
    }

    pub(super) fn reserve_timing_bytes(&mut self, bytes: usize) -> bool {
        if self.output_accounting_error.is_some() {
            return false;
        }
        if let Err(error) = self.output_semantic.try_grow(bytes) {
            self.output_accounting_error = Some(error);
            return false;
        }
        true
    }
}
