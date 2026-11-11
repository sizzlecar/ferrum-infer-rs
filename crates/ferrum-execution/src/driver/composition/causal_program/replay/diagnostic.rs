//! Test-only observations of the original dispatch; no replay policy changes.
use super::*;
use std::sync::OnceLock;

pub(in super::super) struct Observation {
    metadata: serde_json::Value,
    program_id: Option<DeviceReusableExecutionProgramId>,
    preparation: OnceLock<(bool, bool, bool, DeviceReusableExecutionObservation)>,
    gaps: OnceLock<(usize, [Option<DeviceReusableExecutionProgramGap>; 16])>,
}

impl<R: DeviceRuntime> CausalProgramSession<R> {
    /// Enable bounded per-wave receipts for the real CUDA correctness test.
    /// This entrypoint and its storage do not exist in product builds.
    pub fn enable_reusable_execution_test_diagnostics(&self) {
        if let Some(replay) = &self.replay {
            replay.diagnostic_enabled.store(true, Ordering::Relaxed);
        }
    }

    /// The exact direct program whose original wave completed successfully.
    /// This observation is absent unless test diagnostics were enabled.
    pub fn last_successful_reusable_execution_test_program(
        &self,
    ) -> Option<DeviceReusableExecutionProgramId> {
        self.replay
            .as_ref()
            .and_then(|replay| replay.last_successful_program.lock().clone())
    }
}

impl Observation {
    pub(in super::super) fn begin<R: DeviceRuntime>(
        session: &CausalProgramSession<R>,
        wave: &PreparedStepSubmissionWave<R>,
        lane: &Arc<ExecutionLane<R>>,
        adaptive: bool,
        direct: bool,
    ) -> Option<Self> {
        if !session
            .replay
            .as_ref()?
            .diagnostic_enabled
            .load(Ordering::Relaxed)
        {
            return None;
        }
        let backing = wave.claimed_backing();
        let identity = OperationDispatch::reusable_execution_program_id_for_wave(
            session.providers.providers(),
            &session.plan,
            wave,
            lane,
        );
        let program_id = identity.as_ref().ok().cloned().flatten();
        let work = backing.work_shape();
        Some(Self {
            metadata: serde_json::json!({
                "diagnostic": "mtp_replay_wave",
                "plan": session.plan.execution_plan().plan_hash(),
                "candidate_state": session.plan.execution_plan().requires_selected_prefix_commit(),
                "sequences": work.immediate_sequences(),
                "tokens": work.immediate_tokens().ok(),
                "pages": work.immediate_pages().ok(),
                "step_bucket": wave.step_resources().reusable_execution_bucket(),
                "backing_bucket": backing.reusable_execution_bucket_id(),
                "binding_layout_present": backing.program_binding_layout().is_some(),
                "binding_slot_present": backing.program_binding_lane_slot_identity().is_some(),
                "invocation_slices": backing.backing_slices().len(),
                "program_id_present": program_id.is_some(),
                "program_id": program_id.as_ref().map(|id| id.fingerprint()),
                "lane_slot": program_id.as_ref().map(|id| id.lane_slot_id()),
                "stable_layout": program_id.as_ref().map(|id| id.lane_stable_layout_fingerprint()),
                "identity_error": identity.err().map(|error| error.to_string()),
                "adaptive": adaptive,
                "direct": direct,
            }),
            program_id,
            preparation: OnceLock::new(),
            gaps: OnceLock::new(),
        })
    }

    pub(in super::super) fn finish(mut self, submitted: bool) {
        self.metadata["submitted"] = submitted.into();
        self.metadata["backend_preparation"] = self
            .preparation
            .into_inner()
            .map(|(present, matches, allowed, segments)| {
                serde_json::json!({"program_id_present": present, "identity_matches": matches,
                "capture_allowed": allowed, "segments": segments})
            })
            .into();
        self.metadata["program_gaps"] = self.gaps.into_inner().map(|(count, first)| {
            serde_json::json!({"total": count, "first_16": first.into_iter().flatten().collect::<Vec<_>>()})
        }).into();
        eprintln!("{}", self.metadata);
    }
}

impl DeviceSubmissionTimingSink for Observation {
    const ENABLED: bool = true;
    fn record_device_submission(&self, _: DeviceSubmissionStage, _: Duration) {}
    fn record_reusable_preparation(
        &self,
        id: Option<&DeviceReusableExecutionProgramId>,
        allowed: bool,
        segments: DeviceReusableExecutionObservation,
    ) {
        let _ = self.preparation.set((
            id.is_some(),
            id == self.program_id.as_ref(),
            allowed,
            segments,
        ));
    }
    fn record_reusable_program_gaps(
        &self,
        _: &DeviceReusableExecutionProgramId,
        gaps: &[DeviceReusableExecutionProgramGap],
    ) {
        let _ = self
            .gaps
            .set((gaps.len(), std::array::from_fn(|i| gaps.get(i).copied())));
    }
}

impl SubmissionWaveDispatchTimingSink for Observation {
    fn record(&self, _: SubmissionWaveDispatchStage, _: Duration) {}
}
