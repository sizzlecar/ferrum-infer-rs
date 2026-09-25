use super::*;
use crate::continuous_engine::{
    DelimitedPayloadCompletionState, ResponseCompletionState, ResponseEnvelopePhase,
    SequenceSamplingHistoryScope,
};
use crate::continuous_engine::{EngineInner, SequenceState};
use sha2::{Digest, Sha256};

/// Full-history projection and host sampling can differ at the same total KV
/// length. Preserve the actual generated history separately from that KV key.
/// Only a cached digest and one checked scalar are hashed here; no history or
/// JSON is copied. Calibration may later expose this as a typed numeric axis.
fn participant_output_policy(sequence: &SequenceState) -> Option<[u8; 32]> {
    let policy = sequence.cost_policy_signature?;
    let generated = u64::try_from(sequence.generated_tokens.len()).ok()?;
    Some(host_history_cost_signature(policy, generated))
}

/// Read under the same sequence view as the actual input. This helper also
/// serves controller snapshots; it does not predict future completion state.
pub(in crate::continuous_engine) fn participant_host_features(
    sequence: &SequenceState,
) -> Option<HostCostFeaturesV1> {
    let sampling_history_tokens = u64::try_from(
        sequence
            .sampling_history
            .previous_tokens(&sequence.generated_tokens)
            .ok()?
            .len(),
    )
    .ok()?;
    let sampling_history_scope = match sequence.sampling_history.scope() {
        SequenceSamplingHistoryScope::FullGeneration => CostSamplingHistoryScope::FullGeneration,
        SequenceSamplingHistoryScope::HiddenStructuredOutput => {
            CostSamplingHistoryScope::HiddenStructuredOutput
        }
        SequenceSamplingHistoryScope::VisibleStructuredOutput { .. } => {
            CostSamplingHistoryScope::VisibleStructuredOutput
        }
    };
    Some(HostCostFeaturesV1 {
        policy: sequence.cost_numeric_policy?,
        state: HostCostStateV1 {
            generated_tokens_before: u64::try_from(sequence.generated_tokens.len()).ok()?,
            maximum_output_tokens: u64::try_from(sequence.sampling_params.max_tokens).ok()?,
            sampling_history_tokens,
            sampling_history_scope,
            pending_decoded_utf8: sequence.pending_decoded_utf8_fragment,
            completion_state_signature: completion_state_signature(
                &sequence.response_completion_state,
            )?,
        },
    })
}

fn completion_state_signature(state: &ResponseCompletionState) -> Option<[u8; 32]> {
    let ResponseCompletionState::Pending {
        delimited_payload,
        alternate_envelope,
    } = state
    else {
        return Some(satisfied_completion_cost_signature());
    };
    let mut digest = Sha256::new();
    digest.update(b"ferrum.host-completion-state.v1\0pending");
    match delimited_payload {
        DelimitedPayloadCompletionState::AwaitingDelimiter(matcher) => {
            digest.update([0]);
            digest.update(u64::try_from(matcher.matched).ok()?.to_le_bytes());
        }
        DelimitedPayloadCompletionState::AwaitingPayload => digest.update([1]),
    }
    match alternate_envelope {
        None => digest.update([0]),
        Some(envelope) => {
            digest.update([1]);
            digest.update([match envelope.phase {
                ResponseEnvelopePhase::AwaitingOpen => 0,
                ResponseEnvelopePhase::AwaitingClose => 1,
            }]);
            for value in [
                envelope.open.matched,
                envelope.close.matched,
                envelope.completed_envelopes,
                envelope.max_envelopes,
            ] {
                digest.update(u64::try_from(value).ok()?.to_le_bytes());
            }
        }
    }
    Some(digest.finalize().into())
}

pub(in crate::continuous_engine) struct EngineCostPreparation {
    runtime: Arc<EngineCostRuntime>,
    started_at: Option<u64>,
    participants: Vec<CostObservationParticipant>,
    rejection: Option<CostCallRejection>,
    finished: bool,
}
impl EngineCostPreparation {
    /// Must run in the same read-locked sequence view used to build this input.
    pub fn capture(&mut self, sequence: &SequenceState) {
        if self.participants.len() >= self.runtime.recorder_limits.max_rows_per_wave {
            self.rejection = Some(CostCallRejection::RecorderCapacity);
            return;
        }
        let Some(frontier) = sequence.cost_frontier else {
            self.rejection = Some(CostCallRejection::FrontierMismatch);
            return;
        };
        self.participants.push(CostObservationParticipant {
            request_id: sequence.request_id.clone(),
            owner_incarnation: frontier.owner_incarnation.get(),
            work_generation: frontier.work_generation.get(),
            input_index: self.participants.len() as u32,
            output_policy_signature: participant_output_policy(sequence),
            host_features: participant_host_features(sequence),
        });
    }
    pub fn begin(mut self) -> Option<ObservedCostCall> {
        self.finished = true;
        if let Some(reason) = self.rejection {
            self.runtime.sink.reject_preparation(reason);
            return None;
        }
        if self.participants.is_empty() {
            self.runtime
                .sink
                .reject_preparation(CostCallRejection::NoPhysicalWave);
            return None;
        }
        EngineCostCall::begin(
            &self.runtime.ids,
            self.runtime.clock.clone(),
            Arc::clone(&self.runtime.sink),
            EngineCostCallSpec {
                identity: self.runtime.identity.clone(),
                participants: std::mem::take(&mut self.participants),
                prepare_started_at_ns: self.started_at,
                boundary: WaveObservationBoundary::IsolatedPreparationToCommit,
                recorder_limits: self.runtime.recorder_limits,
            },
        )
        .ok()
        .map(|call| call.with_structured_capture(self.runtime.structured_capture))
        .map(ObservedCostCall::new)
    }
}

impl Drop for EngineCostPreparation {
    fn drop(&mut self) {
        if !self.finished {
            self.runtime.sink.preparation_abandoned();
        }
    }
}

#[cfg(test)]
mod audit_tests {
    use super::*;

    #[test]
    fn cancelled_preparation_and_empty_begin_do_not_invent_executor_calls() {
        let runtime = Arc::new(
            EngineCostRuntime::with_clock(Default::default(), Arc::new(EngineCostClock::default()))
                .unwrap(),
        );
        let preparation = || {
            runtime.sink.preparation_started();
            EngineCostPreparation {
                runtime: runtime.clone(),
                started_at: Some(0),
                participants: Vec::new(),
                rejection: None,
                finished: false,
            }
        };
        drop(preparation());
        assert!(preparation().begin().is_none());
        let stats = runtime.sink.stats();
        assert_eq!(stats.preparations_started, 2);
        assert_eq!(stats.preparations_abandoned, 1);
        assert_eq!(
            stats.preparation_rejected[CostCallRejection::NoPhysicalWave.index()],
            1
        );
        assert_eq!(stats.calls_started, 0);
        assert_eq!(stats.calls_finished, 0);
        assert_eq!(stats.offered, 0);
    }
}

impl EngineInner {
    pub(in crate::continuous_engine) fn initialize_sequence_cost(
        &self,
        sequence: &mut SequenceState,
    ) {
        let Some(runtime) = &self.cost_runtime else {
            return;
        };
        match runtime.ids.new_frontier() {
            Ok(frontier) => sequence.cost_frontier = Some(frontier),
            Err(reason) => {
                runtime.sink.reject_initialization(reason);
                sequence.cost_frontier = None;
            }
        }
        sequence.cost_policy_signature =
            policy::host_policy_signature(sequence, self.tokenizer.as_ref());
        sequence.cost_numeric_policy =
            policy::host_numeric_policy(sequence, self.tokenizer.as_ref());
    }
    pub(in crate::continuous_engine) fn prepare_cost_observation(
        &self,
    ) -> Option<EngineCostPreparation> {
        let runtime = self.cost_runtime.as_ref()?;
        // Read the anchor before allocation and before preparing any real input.
        let started_at = runtime.clock.now_ns();
        runtime.sink.preparation_started();
        Some(EngineCostPreparation {
            runtime: Arc::clone(runtime),
            started_at,
            participants: Vec::new(),
            rejection: None,
            finished: false,
        })
    }
    pub(in crate::continuous_engine) fn wake_cost_trainer(&self) {
        if let Some(runtime) = &self.cost_runtime {
            runtime.wake_trainer();
        }
    }
}

/// Host counters and incarnation are read at the real commit, not copied from
/// the planned participant. Nothing in this object can allocate/use KV.
pub(in crate::continuous_engine) struct CostHostCommitStart {
    request_id: RequestId,
    frontier: Option<CostFrontier>,
    input_index: Option<u32>,
    generated_before: Option<u64>,
    kv_before: Option<u32>,
}
impl CostHostCommitStart {
    pub fn capture(sequence: &SequenceState, call: &EngineCostCall) -> Self {
        Self {
            request_id: sequence.request_id.clone(),
            frontier: sequence.cost_frontier,
            input_index: call
                .participants
                .iter()
                .find(|row| row.request_id == sequence.request_id)
                .map(|row| row.input_index),
            generated_before: u64::try_from(sequence.generated_tokens.len()).ok(),
            kv_before: sequence
                .model_kv
                .as_ref()
                .and_then(|state| u32::try_from(state.handle().num_tokens()).ok()),
        }
    }
    pub fn prefill(
        self,
        call: &mut EngineCostCall,
        sequence: &SequenceState,
        start: usize,
        total: usize,
    ) -> Option<HostCommitEvidence> {
        let work = HostCommittedWork::Prefill {
            start: u32::try_from(start).ok()?,
            end: u32::try_from(sequence.prefill_tokens_processed).ok()?,
            total_prompt_tokens: u32::try_from(total).ok()?,
            generated_tokens_before: self.generated_before?,
            generated_tokens_after: u64::try_from(sequence.generated_tokens.len()).ok()?,
        };
        self.finish(call, work)
    }
    pub fn decode(
        self,
        call: &mut EngineCostCall,
        sequence: &SequenceState,
    ) -> Option<HostCommitEvidence> {
        let work = HostCommittedWork::Decode {
            kv_tokens_before: self.kv_before?,
            kv_tokens_after: sequence
                .model_kv
                .as_ref()
                .and_then(|state| u32::try_from(state.handle().num_tokens()).ok())?,
            generated_tokens_before: self.generated_before?,
            generated_tokens_after: u64::try_from(sequence.generated_tokens.len()).ok()?,
        };
        self.finish(call, work)
    }
    fn finish(
        self,
        call: &mut EngineCostCall,
        work: HostCommittedWork,
    ) -> Option<HostCommitEvidence> {
        let Some(frontier) = self.frontier else {
            call.reject(CostCallRejection::FrontierMismatch);
            return None;
        };
        Some(HostCommitEvidence {
            request_id: self.request_id,
            owner_incarnation: frontier.owner_incarnation.get(),
            work_generation: frontier.work_generation.get(),
            input_index: self.input_index?,
            outcome: HostCommitOutcome::Committed(work),
            committed_at_ns: None,
        })
    }
}
