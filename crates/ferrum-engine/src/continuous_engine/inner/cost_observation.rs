//! Passive per-call evidence. This module owns no scheduler, KV, output credit,
//! or retry authority. Errors disable training, never authorize another dispatch.

use std::{
    num::NonZeroU64,
    ops::{Deref, DerefMut},
    sync::Arc,
};

use ferrum_interfaces::execution_cost::*;
use ferrum_types::RequestId;

mod audit;
mod calibration_capture;
pub(in crate::continuous_engine) use calibration_capture::*;
mod checkpoint;
pub(in crate::continuous_engine) use checkpoint::FrozenCostCheckpoint;
mod clock;
mod dispatch;
mod engine;
mod host_stages;
pub use host_stages::{
    HostRowStageV1, HostStageCompleteness, HostStageEvidenceV1, HostStageQueueDisposition,
    HostStageQueueReceipt, HostStageWork, HostTerminalStageV1,
};
pub(in crate::continuous_engine) use host_stages::{HostSettledReceipt, PendingHostRow};
mod policy;
mod profile;
mod profile_export;
pub(in crate::continuous_engine::inner) use profile_export::selected::SelectedCalibrationCapture;
pub use profile_export::selected::SelectedFitFreezeReceipt;
pub(in crate::continuous_engine::inner) use profile_export::structured::StructuredCalibrationCollector;
pub use profile_export::structured::{
    StructuredCalibrationArtifact, StructuredCalibrationOptions, StructuredCalibrationProgress,
    StructuredCalibrationScopeV1, StructuredCapturePhase, StructuredPhaseFreezeReceipt,
};
pub(in crate::continuous_engine) use profile_export::{CostProfileCutPaths, CostProfileCutReceipt};
mod publication;
mod query_metrics;
mod runtime;
mod trainer;
mod worker;
pub(in crate::continuous_engine) use engine::*;
pub(in crate::continuous_engine) use runtime::*;
mod presubmit;
mod sample;
mod selected_feedback;
mod sink;
pub(in crate::continuous_engine) use clock::*;
pub(in crate::continuous_engine) use sink::*;

#[cfg(test)]
mod leaf_tests;
#[cfg(test)]
pub(in crate::continuous_engine::inner) mod tests;

/// Values read at the real host commit, not the original scheduler plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::continuous_engine) enum HostCommittedWork {
    Prefill {
        start: u32,
        end: u32,
        total_prompt_tokens: u32,
        generated_tokens_before: u64,
        generated_tokens_after: u64,
    },
    Decode {
        kv_tokens_before: u32,
        kv_tokens_after: u32,
        generated_tokens_before: u64,
        generated_tokens_after: u64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::continuous_engine) enum HostCommitOutcome {
    /// Both sequence state and scheduler progress are published, and required
    /// synchronous output handling has completed. Later transport delivery is
    /// not implied. A terminal cleanup mixed into this interval needs its own
    /// shape or must make the call non-isolated.
    Committed(HostCommittedWork),
    Cancelled,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(in crate::continuous_engine) struct HostCommitEvidence {
    pub request_id: RequestId,
    /// The incarnation/generation inspected under the commit's sequence lock,
    /// before advancing that generation, not a blindly copied input receipt.
    pub owner_incarnation: u64,
    pub work_generation: u64,
    pub input_index: u32,
    pub outcome: HostCommitOutcome,
    pub committed_at_ns: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(in crate::continuous_engine) enum CostCallRejection {
    Clock,
    IdExhausted,
    InvalidParticipants,
    RecorderCapacity,
    IdentityUnknown,
    IdentitySchema,
    OutputPolicyUnknown,
    Unavailable,
    NoPhysicalWave,
    Composite,
    ExecutorIncomplete,
    ExecutorFailed,
    ActualEvidenceUnknown,
    HostMissing,
    HostDuplicate,
    HostUnexpected,
    FrontierMismatch,
    WorkMismatch,
    HostCancelled,
    HostFailed,
    InvalidWall,
    Abandoned,
}

impl CostCallRejection {
    pub(super) const COUNT: usize = Self::Abandoned as usize + 1;
    pub(super) const fn index(self) -> usize {
        self as usize
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(in crate::continuous_engine) enum CostCallDisposition {
    Published,
    Rejected(CostCallRejection),
    Dropped(CostSampleDrop),
}

pub(in crate::continuous_engine) struct EngineCostCallSpec {
    pub identity: ExecutorCostIdentityAvailability,
    pub participants: Vec<CostObservationParticipant>,
    /// Capture before preparing inputs or taking preparation locks.
    pub prepare_started_at_ns: Option<u64>,
    pub boundary: WaveObservationBoundary,
    pub recorder_limits: CostRecorderLimits,
}

#[derive(Default)]
struct DispatchSummary {
    handle: Option<WaveObservationHandle>,
    waves: usize,
    outcome: Option<ObservedCallOutcome>,
    unknown: Option<ActualWaveEvidenceUnknown>,
    returned_at_ns: Option<u64>,
}

pub(in crate::continuous_engine) struct EngineCostCall {
    call_id: NonZeroU64,
    clock: Arc<dyn CostObservationClock>,
    sink: Arc<BoundedCostSampleSink>,
    identity: ExecutorCostIdentityAvailability,
    participants: Vec<CostObservationParticipant>,
    prepare_started_at_ns: Option<u64>,
    boundary: WaveObservationBoundary,
    recorder: BoundedWaveRecorder,
    dispatch: DispatchSummary,
    context_created: bool,
    structured_capture: bool,
    host: Vec<Option<HostCommitEvidence>>,
    host_fence: Arc<()>,
    host_stages: Vec<host_stages::HostRowProgress>,
    host_processing_ordinal: u32,
    rejection: Option<CostCallRejection>,
    stage_rejection: Option<CostCallRejection>,
    finished: bool,
    calibration_capture: Option<Arc<CostCalibrationCapture>>,
    presubmit_prediction: Option<presubmit::PendingPrediction>,
}

/// Dropping this guard copies only already observed facts. In particular,
/// cancellation never manufactures a terminal fence or host commit.
pub(in crate::continuous_engine) struct EngineCostContext<'a> {
    inner: PlanRuntimeCostObservationContext<'a>,
    summary: &'a mut DispatchSummary,
}

impl<'a> Deref for EngineCostContext<'a> {
    type Target = PlanRuntimeCostObservationContext<'a>;
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
impl DerefMut for EngineCostContext<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
impl Drop for EngineCostContext<'_> {
    fn drop(&mut self) {
        self.summary.handle = self.inner.wave_handle();
        self.summary.waves = self.inner.physical_wave_count();
        self.summary.outcome = self.inner.call_outcome();
        self.summary.unknown = self.inner.unknown_reason();
        if self.summary.outcome.is_some() {
            self.summary.returned_at_ns = self.inner.now_ns();
        }
    }
}

impl EngineCostCall {
    pub fn begin(
        ids: &EngineCostIds,
        clock: Arc<dyn CostObservationClock>,
        sink: Arc<BoundedCostSampleSink>,
        spec: EngineCostCallSpec,
    ) -> Result<Self, CostCallRejection> {
        sink.call_started();
        let result = Self::begin_inner(ids, clock, Arc::clone(&sink), spec);
        if let Err(reason) = &result {
            sink.reject(*reason);
        }
        result
    }

    fn begin_inner(
        ids: &EngineCostIds,
        clock: Arc<dyn CostObservationClock>,
        sink: Arc<BoundedCostSampleSink>,
        spec: EngineCostCallSpec,
    ) -> Result<Self, CostCallRejection> {
        let call_id = ids.next_call()?;
        spec.recorder_limits
            .validate()
            .map_err(|_| CostCallRejection::RecorderCapacity)?;
        if spec.participants.is_empty()
            || spec.participants.len() > spec.recorder_limits.max_rows_per_wave
            || spec.participants.capacity() > spec.recorder_limits.max_rows_per_wave
            || spec.participants.iter().enumerate().any(|(index, row)| {
                row.owner_incarnation == 0
                    || row.work_generation == 0
                    || spec.participants[..index].iter().any(|prior| {
                        prior.request_id == row.request_id || prior.input_index == row.input_index
                    })
            })
        {
            return Err(CostCallRejection::InvalidParticipants);
        }
        let recorder = BoundedWaveRecorder::new(call_id, spec.recorder_limits)
            .map_err(|_| CostCallRejection::RecorderCapacity)?;
        let mut host = Vec::new();
        host.try_reserve_exact(spec.participants.len())
            .map_err(|_| CostCallRejection::RecorderCapacity)?;
        host.resize_with(spec.participants.len(), || None);
        let mut host_stages = Vec::new();
        host_stages
            .try_reserve_exact(spec.participants.len())
            .map_err(|_| CostCallRejection::RecorderCapacity)?;
        host_stages.resize_with(
            spec.participants.len(),
            host_stages::HostRowProgress::default,
        );
        Ok(Self {
            call_id,
            clock,
            sink,
            identity: spec.identity,
            participants: spec.participants,
            prepare_started_at_ns: spec.prepare_started_at_ns,
            boundary: spec.boundary,
            recorder,
            dispatch: DispatchSummary::default(),
            context_created: false,
            structured_capture: false,
            host,
            host_fence: Arc::new(()),
            host_stages,
            host_processing_ordinal: 0,
            rejection: None,
            stage_rejection: None,
            finished: false,
            calibration_capture: None,
            presubmit_prediction: None,
        })
    }

    pub fn with_structured_capture(mut self, enabled: bool) -> Self {
        self.structured_capture = enabled;
        self
    }

    pub fn now_ns(&self) -> Option<u64> {
        self.clock.now_ns()
    }

    pub fn context(&mut self) -> Result<EngineCostContext<'_>, CostCallRejection> {
        if self.context_created {
            self.reject(CostCallRejection::Composite);
            return Err(CostCallRejection::Composite);
        }
        self.context_created = true;
        Ok(EngineCostContext {
            inner: PlanRuntimeCostObservationContext::new(
                &mut self.recorder,
                self.clock.as_ref(),
                &self.participants,
                self.prepare_started_at_ns,
                self.boundary,
            )
            .with_structured_capture(self.structured_capture),
            summary: &mut self.dispatch,
        })
    }

    pub fn reject(&mut self, reason: CostCallRejection) {
        if reason != CostCallRejection::Composite {
            self.stage_rejection.get_or_insert(reason);
        }
        self.rejection.get_or_insert(reason);
    }

    /// Negative evidence names the prepared row, without claiming that its
    /// incarnation was still present or any state was successfully published.
    pub fn host_cancelled(&mut self, request_id: &RequestId) {
        self.record_host_failure(request_id, HostCommitOutcome::Cancelled);
    }
    pub fn host_failed(&mut self, request_id: &RequestId) {
        self.record_host_failure(request_id, HostCommitOutcome::Failed);
    }
    fn record_host_failure(&mut self, request_id: &RequestId, outcome: HostCommitOutcome) {
        self.note_host_failure(request_id);
        self.reject(match outcome {
            HostCommitOutcome::Cancelled => CostCallRejection::HostCancelled,
            _ => CostCallRejection::HostFailed,
        });
        if let Some(row) = self
            .participants
            .iter()
            .find(|row| &row.request_id == request_id)
        {
            self.record_host_result(HostCommitEvidence {
                request_id: row.request_id.clone(),
                owner_incarnation: row.owner_incarnation,
                work_generation: row.work_generation,
                input_index: row.input_index,
                outcome,
                committed_at_ns: None,
            });
        }
    }

    pub fn record_host_result(&mut self, evidence: HostCommitEvidence) {
        let Some(index) = self
            .participants
            .iter()
            .position(|row| row.request_id == evidence.request_id)
        else {
            self.reject(CostCallRejection::HostUnexpected);
            return;
        };
        let expected = &self.participants[index];
        if expected.owner_incarnation != evidence.owner_incarnation
            || expected.work_generation != evidence.work_generation
            || expected.input_index != evidence.input_index
        {
            self.reject(CostCallRejection::FrontierMismatch);
            return;
        }
        if self.host[index].is_some() {
            self.reject(CostCallRejection::HostDuplicate);
            return;
        }
        self.host[index] = Some(evidence);
    }

    #[cfg(test)]
    pub fn observations(&self) -> &[ActualWaveObservation] {
        self.recorder.observations()
    }

    pub fn finish(mut self) -> CostCallDisposition {
        self.finished = true;
        let stages = self.make_host_stages();
        if let (Some(capture), Some(stages)) = (&self.calibration_capture, &stages) {
            capture.complete_host_stages(Arc::clone(stages));
        }
        match self.make_sample() {
            Ok(sample) => {
                // The normal observer remains the sole author of this sample.
                // Only an explicitly attached calibration consumer copies the
                // bounded facts; no before/after state is guessed here.
                let captured = self.calibration_capture.as_ref().map(|_| {
                    let rows = self.recorder.observations()[0]
                        .shape
                        .as_ref()
                        .expect("validated sample shape")
                        .rows
                        .clone();
                    let commits = rows
                        .iter()
                        .map(|row| {
                            self.host
                                .iter()
                                .flatten()
                                .find(|commit| {
                                    commit.request_id == row.request_id
                                        && commit.owner_incarnation == row.owner_incarnation
                                        && commit.work_generation == row.work_generation
                                        && commit.input_index == row.input_index
                                })
                                .expect("validated physical/host identity")
                                .clone()
                        })
                        .collect();
                    let host_features = rows
                        .iter()
                        .map(|row| {
                            self.participants
                                .iter()
                                .find(|participant| {
                                    participant.request_id == row.request_id
                                        && participant.owner_incarnation == row.owner_incarnation
                                        && participant.work_generation == row.work_generation
                                        && participant.input_index == row.input_index
                                })
                                .expect("validated physical/participant identity")
                                .host_features
                        })
                        .collect();
                    (sample.clone(), rows, commits, host_features)
                });
                self.sink.call_finished();
                let offered = self
                    .sink
                    .offer_evidence_numbered(CostEvidenceEntry::Training { sample, stages });
                self.record_host_stage_queue(&offered);
                let (disposition, accepted_ordinal) = match offered {
                    Ok(ordinal) => (CostCallDisposition::Published, Some(ordinal)),
                    Err(reason) => (CostCallDisposition::Dropped(reason), None),
                };
                if let (Some(capture), Some((sample, actual_rows, commits, host_features))) =
                    (&self.calibration_capture, captured)
                {
                    capture.complete(CostCalibrationResult::Observed {
                        sample: Box::new(sample),
                        actual_rows,
                        commits,
                        host_features,
                        accepted_ordinal,
                        disposition,
                    });
                }
                disposition
            }
            Err(reason) => {
                self.sink.reject(reason);
                if let Some(stages) = stages {
                    let offered =
                        self.sink
                            .offer_evidence_numbered(CostEvidenceEntry::StagesOnly {
                                stages,
                                legacy_rejection: reason,
                            });
                    self.record_host_stage_queue(&offered);
                }
                if let Some(capture) = &self.calibration_capture {
                    capture.complete(CostCalibrationResult::Rejected(reason));
                }
                CostCallDisposition::Rejected(reason)
            }
        }
    }
}

impl Drop for EngineCostCall {
    fn drop(&mut self) {
        if !self.finished {
            self.sink.reject(CostCallRejection::Abandoned);
            if let Some(stages) = self.make_host_stages() {
                if let Some(capture) = &self.calibration_capture {
                    capture.complete_host_stages(Arc::clone(&stages));
                }
                let offered = self
                    .sink
                    .offer_evidence_numbered(CostEvidenceEntry::StagesOnly {
                        stages,
                        legacy_rejection: CostCallRejection::Abandoned,
                    });
                self.record_host_stage_queue(&offered);
            }
            if let Some(capture) = &self.calibration_capture {
                capture.complete(CostCalibrationResult::Rejected(
                    CostCallRejection::Abandoned,
                ));
            }
        }
    }
}
