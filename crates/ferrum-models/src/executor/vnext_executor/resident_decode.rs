//! Explicit decode resource reservations. Per-step completion and retirement
//! continue through the ordinary executor; no previous Step is retained here.

use super::*;

impl<R: DeviceRuntime> VNextModelExecutor<R> {
    pub(super) fn release_decode_session(&self) -> bool {
        if self.invocation_preparation_strategy
            != InvocationPreparationStrategy::DecodeSessionProgram
        {
            return false;
        }
        self.decode_session.lock().take().is_some()
    }

    pub(super) fn try_prepare_resident_decode(
        &self,
        batch: &ExecutionBatchParticipants<R>,
        spans: &[TokenSpanWork],
        kind: VNextExecutionWaveKind,
    ) -> Result<Option<PreparedVNextPrefill<R>>> {
        if self.invocation_preparation_strategy
            != InvocationPreparationStrategy::DecodeSessionProgram
        {
            return Ok(None);
        }
        if kind != VNextExecutionWaveKind::Decode
            || spans.iter().any(|span| span.immediate_tokens() != 1)
        {
            self.release_decode_session();
            return Ok(None);
        }
        let request = self.step_request_for_spans(batch, spans, kind)?;
        let mut cached = self.decode_session.lock();
        if cached
            .as_ref()
            .is_some_and(|session| !session.matches_participants(batch))
        {
            // A pending device use owns the reservation independently. Evicting
            // the cache cannot release its backing before terminal completion.
            cached.take();
        }
        if cached.is_none() {
            *cached = batch
                .try_reserve_decode_session(&request, &self.lane)
                .map_err(|error| FerrumError::backend(error.to_string()))?;
        }
        let Some(session) = cached.as_mut() else {
            return Ok(None);
        };
        match session.try_prepare_step(&request) {
            Ok(DecodeSessionPrepareDecision::Prepared { step, wave }) => {
                self.metrics.prepared_wave_topology.record(&wave);
                Ok(Some(PreparedVNextPrefill { step, wave }))
            }
            Ok(DecodeSessionPrepareDecision::Fallback)
            | Ok(DecodeSessionPrepareDecision::RequestStateDeferred(_)) => {
                // Release idle claims before the ordinary path performs its
                // current admission/hazard checks and returns a typed deferral.
                cached.take();
                Ok(None)
            }
            Ok(DecodeSessionPrepareDecision::RequestStateSplitRequired(split)) => {
                cached.take();
                Err(FerrumError::internal(format!(
                    "vNext decode session requires a Request-state split for {:?}: {:?}",
                    split.request(),
                    split.resource_ids()
                )))
            }
            Ok(DecodeSessionPrepareDecision::RequestStatePoisoned(poison)) => {
                cached.take();
                Err(FerrumError::backend(format!(
                    "vNext decode-session Request-state resource is poisoned: {poison:?}"
                )))
            }
            Err(error) => {
                cached.take();
                Err(FerrumError::backend(error.to_string()))
            }
        }
    }
}
