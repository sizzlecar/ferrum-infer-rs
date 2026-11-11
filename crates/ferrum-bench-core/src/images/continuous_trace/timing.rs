//! Same-session host-clock intervals. These are neither GPU idle intervals nor
//! a subtraction between client and server clocks. Missing endpoints stay None.
use super::*;
use std::collections::{BTreeMap, BTreeSet};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageTraceTimingUnavailable {
    SchemaOne,
    IncompleteTrace,
    MissingEndpoints,
    NoTimingEvents,
}
#[derive(Clone, Debug, Serialize)]
pub struct ImageTraceInterval {
    pub start_ns: u64,
    pub end_ns: u64,
    pub duration_ns: u64,
}
#[derive(Clone, Debug, Serialize)]
pub struct ImageTraceRequestTiming {
    pub request_id: String,
    pub client_attempt_id: Option<String>,
    pub request: Option<String>,
    /// A metadata link can precede initialization or survive an early refusal.
    /// Only an actual RequestInitialized observation proves this model ID ran
    /// through the initialization consumer; it does not prove a forward.
    pub model_request_initialized: bool,
    pub never_queued: bool,
    pub accepted_ns: Option<u64>,
    pub queue_intervals: Vec<ImageTraceInterval>,
    pub queue_total_ns: Option<u64>,
    pub initialization_started_ns: Option<u64>,
    /// Includes preparation and admission, not merely time in the FIFO.
    pub accepted_to_initialization_ns: Option<u64>,
    pub service_finished_ns: Option<u64>,
    pub outcome: Option<String>,
}
#[derive(Clone, Debug, Serialize)]
pub struct ImageTracePhaseTiming {
    pub start_sequence: u64,
    pub finish_sequence: u64,
    pub request_id: Option<String>,
    pub request: Option<String>,
    pub phase: String,
    pub outcome: String,
    pub interval: ImageTraceInterval,
    /// Present for decoder work only when its original completed denoising
    /// exit fence was observed. Other phases do not invent an eligibility time.
    pub ready_ns: Option<u64>,
    pub wait_ns: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub main_vision_binding: Option<ImageTraceMainVisionBinding>,
}

/// The source can have been cancelled after its visual phase completed. A
/// survivor's immutable readback does not require producer image completion.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ImageTraceMainVisionBinding {
    pub cohort_first_job: u64,
    pub producer_job: u64,
    pub producer_request: String,
    pub reused: bool,
    pub retained_output_bytes: usize,
}
#[derive(Clone, Debug, Serialize)]
pub struct ImageTraceStepMemberTiming {
    pub request: String,
    pub step_index: u64,
    /// None for a first step or when its prior commit endpoint is absent.
    pub previous_commit_ns: Option<u64>,
    /// Host wall gap, including other phases/admission/CPU scheduling.
    pub wait_ns: Option<u64>,
}
#[derive(Clone, Debug, Serialize)]
pub struct ImageTraceStepTiming {
    pub start_sequence: u64,
    pub commit_sequence: u64,
    pub cohort_job: u64,
    pub interval: ImageTraceInterval,
    pub members: Vec<ImageTraceStepMemberTiming>,
}
#[derive(Clone, Debug, Serialize)]
pub struct ImageTraceTimingAnalysis {
    pub status: ImageStepEvidenceStatus,
    pub unavailable_reasons: Vec<ImageTraceTimingUnavailable>,
    pub requests: Vec<ImageTraceRequestTiming>,
    pub phases: Vec<ImageTracePhaseTiming>,
    pub steps: Vec<ImageTraceStepTiming>,
}
impl Default for ImageTraceTimingAnalysis {
    fn default() -> Self {
        Self {
            status: ImageStepEvidenceStatus::NotEvaluated,
            unavailable_reasons: Vec::new(),
            requests: Vec::new(),
            phases: Vec::new(),
            steps: Vec::new(),
        }
    }
}

struct RequestState {
    report: ImageTraceRequestTiming,
    queued_at: Option<u64>,
    last_boundary: u64,
}
struct PhaseStart {
    at: u64,
    request_id: Option<String>,
    request: Option<String>,
    phase: String,
    ready_ns: Option<u64>,
}
struct StepStart {
    at: u64,
    job: u64,
    plan: String,
    program: String,
    members: Vec<ImageTraceStepStartMember>,
    previous: Vec<Option<(u64, u64)>>,
}
pub(super) struct TimingAnalyzer {
    limits: ImageContinuousTraceReadLimits,
    requests: BTreeMap<String, RequestState>,
    links: BTreeMap<String, String>,
    model_initialized: BTreeMap<String, u64>,
    model_exits: BTreeMap<String, u64>,
    phase_starts: BTreeMap<u64, PhaseStart>,
    step_starts: BTreeMap<u64, StepStart>,
    last_commits: BTreeMap<String, (u64, u64)>,
    completed_exits: BTreeMap<String, u64>,
    phases: Vec<ImageTracePhaseTiming>,
    steps: Vec<ImageTraceStepTiming>,
    missing: bool,
    observed: bool,
}
impl TimingAnalyzer {
    pub(super) fn new(limits: ImageContinuousTraceReadLimits) -> Self {
        Self {
            limits,
            requests: BTreeMap::new(),
            links: BTreeMap::new(),
            model_initialized: BTreeMap::new(),
            model_exits: BTreeMap::new(),
            phase_starts: BTreeMap::new(),
            step_starts: BTreeMap::new(),
            last_commits: BTreeMap::new(),
            completed_exits: BTreeMap::new(),
            phases: Vec::new(),
            steps: Vec::new(),
            missing: false,
            observed: false,
        }
    }
    pub(super) fn event(
        &mut self,
        event: &Event,
        sequence: u64,
        at: Option<u64>,
        line: u64,
    ) -> Result<(), ImageContinuousTraceError> {
        let Some(at) = at else {
            return Ok(());
        };
        match event {
            Event::RequestQueued { correlation } => {
                self.insert(*correlation, at, false, None, line)?
            }
            Event::RequestSubmissionRejected {
                correlation,
                outcome,
            } => {
                outcome_valid(outcome, line)?;
                if outcome == "completed" {
                    return Err(invalid(
                        line,
                        "pre-queue rejection cannot complete an image",
                    ));
                }
                self.insert(*correlation, at, true, Some(outcome.clone()), line)?;
            }
            Event::RequestDequeued { request_id } => {
                let request = self.request(request_id, at, line)?;
                close_queue(request, at, line)?;
            }
            Event::RequestRequeued { request_id } => {
                let request = self.request(request_id, at, line)?;
                if request.queued_at.is_some() || request.report.initialization_started_ns.is_some()
                {
                    return Err(invalid(
                        line,
                        "request requeued without leaving its queue or after initialization",
                    ));
                }
                request.queued_at = Some(at);
                request.report.queue_total_ns = None;
            }
            Event::InitializationStarted { request_id } => {
                let request = self.request(request_id, at, line)?;
                if request.queued_at.is_some() || request.report.initialization_started_ns.is_some()
                {
                    return Err(invalid(
                        line,
                        "initialization repeats or begins before dequeue",
                    ));
                }
                request.report.initialization_started_ns = Some(at);
                request.report.accepted_to_initialization_ns =
                    Some(duration(request.report.accepted_ns.unwrap(), at, line)?);
            }
            Event::RequestLinked {
                request_id,
                request,
            } => {
                if request.is_empty() || self.links.contains_key(request) {
                    return Err(invalid(
                        line,
                        "model request correlation is empty or repeats a binding",
                    ));
                }
                let state = self.request(request_id, at, line)?;
                if state.report.request.is_some() {
                    return Err(invalid(
                        line,
                        "service request correlation repeats a binding",
                    ));
                }
                state.report.request = Some(request.clone());
                self.links.insert(request.clone(), request_id.clone());
            }
            Event::RequestServiceFinished {
                request_id,
                outcome,
            } => {
                outcome_valid(outcome, line)?;
                if let Some(model) = self
                    .requests
                    .get(request_id)
                    .and_then(|state| state.report.request.as_ref())
                {
                    if let Some(initialized) = self.model_initialized.get(model) {
                        duration(*initialized, at, line)?;
                        if let Some((_, committed)) = self.last_commits.get(model) {
                            duration(*committed, at, line)?;
                        }
                        if let Some(exited) = self.model_exits.get(model) {
                            duration(*exited, at, line)?;
                        }
                    }
                }
                if outcome == "completed" {
                    let model = self
                        .requests
                        .get(request_id)
                        .and_then(|state| state.report.request.as_ref());
                    if !model.is_some_and(|model| {
                        self.model_initialized.contains_key(model)
                            && self.completed_exits.contains_key(model)
                            && self.last_commits.contains_key(model)
                    }) {
                        return Err(invalid(line,
                            "completed service outcome lacks its bound actual model initialization and completed exit"));
                    }
                    let exited = self.completed_exits[model.unwrap()];
                    duration(exited, at, line)?;
                }
                let request = self.request(request_id, at, line)?;
                if outcome == "completed" && request.queued_at.is_some() {
                    return Err(invalid(line, "completed image was never dequeued"));
                }
                if request.queued_at.is_some() {
                    close_queue(request, at, line)?;
                }
                request.report.service_finished_ns = Some(at);
                request.report.outcome = Some(outcome.clone());
            }
            Event::PhaseStarted {
                request_id,
                request,
                phase,
            } => {
                if request_id.is_none() && request.is_none() {
                    return Err(invalid(line, "phase start lacks request correlation"));
                }
                if let Some(id) = request_id {
                    let state = self.request(id, at, line)?;
                    if request
                        .as_ref()
                        .is_some_and(|model| state.report.request.as_ref() != Some(model))
                    {
                        return Err(invalid(
                            line,
                            "phase model identity differs from its original service link",
                        ));
                    }
                }
                if request.as_ref().is_some_and(|r| r.is_empty()) || !valid_phase(phase) {
                    return Err(invalid(line, "phase start has an invalid phase/request"));
                }
                let ready_ns = if phase == "decoder" {
                    request
                        .as_ref()
                        .and_then(|request| self.completed_exits.get(request))
                        .copied()
                } else {
                    None
                };
                self.phase_starts.insert(
                    sequence,
                    PhaseStart {
                        at,
                        request_id: request_id.clone(),
                        request: request.clone(),
                        phase: phase.clone(),
                        ready_ns,
                    },
                );
                self.observed = true;
            }
            Event::PhaseFinished {
                start_sequence,
                outcome,
                main_vision_binding,
            } => {
                if !["completed", "failed", "cancelled"].contains(&outcome.as_str()) {
                    return Err(invalid(line, "phase outcome is outside the typed schema"));
                }
                let start = self
                    .phase_starts
                    .remove(start_sequence)
                    .ok_or_else(|| invalid(line, "phase finish has no unique original start"))?;
                if let Some(binding) = main_vision_binding {
                    if start.phase != "conditioning_vision"
                        || outcome != "completed"
                        || binding.producer_request.trim().is_empty()
                        || binding.retained_output_bytes == 0
                        || (!binding.reused
                            && start.request.as_ref() != Some(&binding.producer_request))
                    {
                        return Err(invalid(line, "invalid main vision binding for this phase"));
                    }
                }
                if let Some(id) = &start.request_id {
                    let state = self.request(id, at, line)?;
                    if start
                        .request
                        .as_ref()
                        .is_some_and(|model| state.report.request.as_ref() != Some(model))
                    {
                        return Err(invalid(
                            line,
                            "phase model identity differs from its original service link",
                        ));
                    }
                }
                self.phases.push(ImageTracePhaseTiming {
                    start_sequence: *start_sequence,
                    finish_sequence: sequence,
                    request_id: start.request_id,
                    request: start.request,
                    phase: start.phase,
                    outcome: outcome.clone(),
                    interval: interval(start.at, at, line)?,
                    ready_ns: start.ready_ns,
                    wait_ns: start
                        .ready_ns
                        .map(|ready| duration(ready, start.at, line))
                        .transpose()?,
                    main_vision_binding: main_vision_binding.clone(),
                });
            }
            Event::PackedStepStarted {
                cohort_job,
                plan_hash,
                program_fingerprint,
                members,
            } => {
                if members.is_empty()
                    || members.len() > self.limits.maximum_members
                    || plan_hash.is_empty()
                    || program_fingerprint.is_empty()
                {
                    return Err(invalid(
                        line,
                        "packed start lacks bounded membership or plan identity",
                    ));
                }
                let mut unique = BTreeSet::new();
                for member in members {
                    if member.request.is_empty() || !unique.insert(&member.request) {
                        return Err(invalid(line, "packed start repeats or lacks a request"));
                    }
                    if let Some(initialized) = self.model_initialized.get(&member.request) {
                        duration(*initialized, at, line)?;
                    }
                }
                let previous = members
                    .iter()
                    .map(|member| self.last_commits.get(&member.request).copied())
                    .collect();
                self.step_starts.insert(
                    sequence,
                    StepStart {
                        at,
                        job: *cohort_job,
                        plan: plan_hash.clone(),
                        program: program_fingerprint.clone(),
                        members: members.clone(),
                        previous,
                    },
                );
                self.observed = true;
            }
            Event::PackedStepCommitted {
                start_sequence,
                cohort_job,
                plan_hash,
                program_fingerprint,
                members,
                ..
            } => {
                if let Some(start_sequence) = start_sequence {
                    let start = self.step_starts.remove(start_sequence).ok_or_else(|| {
                        invalid(line, "packed commit has no unique original start")
                    })?;
                    if start.job != *cohort_job
                        || start.plan != *plan_hash
                        || start.program != *program_fingerprint
                        || start.members.len() != members.len()
                        || start
                            .members
                            .iter()
                            .zip(members)
                            .any(|(a, b)| a.request != b.request || a.step_index != b.step_index)
                    {
                        return Err(invalid(
                            line,
                            "packed start and commit differ in membership/progress/plan",
                        ));
                    }
                    let mut samples = Vec::with_capacity(members.len());
                    for (member, prior) in members.iter().zip(start.previous) {
                        let previous_commit_ns = match prior {
                            Some((step, time))
                                if step.checked_add(1) == Some(member.step_index) =>
                            {
                                Some(time)
                            }
                            _ => None,
                        };
                        if member.step_index > 0 && previous_commit_ns.is_none() {
                            self.missing = true;
                        }
                        samples.push(ImageTraceStepMemberTiming {
                            request: member.request.clone(),
                            step_index: member.step_index,
                            previous_commit_ns,
                            wait_ns: previous_commit_ns
                                .map(|before| duration(before, start.at, line))
                                .transpose()?,
                        });
                    }
                    self.steps.push(ImageTraceStepTiming {
                        start_sequence: *start_sequence,
                        commit_sequence: sequence,
                        cohort_job: *cohort_job,
                        interval: interval(start.at, at, line)?,
                        members: samples,
                    });
                } else {
                    self.missing = true;
                }
                for member in members {
                    if self.last_commits.len() >= self.limits.maximum_requests
                        && !self.last_commits.contains_key(&member.request)
                    {
                        return Err(capacity(line));
                    }
                    self.last_commits
                        .insert(member.request.clone(), (member.step_index, at));
                }
            }
            Event::RequestExitedAtFence {
                request, reason, ..
            } => {
                if let Some(initialized) = self.model_initialized.get(request) {
                    duration(*initialized, at, line)?;
                }
                if let Some((_, committed)) = self.last_commits.get(request) {
                    duration(*committed, at, line)?;
                }
                if self.model_exits.len() >= self.limits.maximum_requests
                    && !self.model_exits.contains_key(request)
                {
                    return Err(capacity(line));
                }
                // A cancelled/failed service return is also later than any
                // actual model fence it claims. An absent fence remains a
                // missing endpoint, never a made-up retirement or zero wait.
                if let Some(finished) = self
                    .links
                    .get(request)
                    .and_then(|id| self.requests.get(id))
                    .and_then(|state| state.report.service_finished_ns)
                {
                    duration(at, finished, line)?;
                }
                self.model_exits.insert(request.clone(), at);
                if matches!(reason, ExitReason::Completed) {
                    if self.completed_exits.len() >= self.limits.maximum_requests
                        && !self.completed_exits.contains_key(request)
                    {
                        return Err(capacity(line));
                    }
                    self.completed_exits.insert(request.clone(), at);
                }
            }
            Event::RequestInitialized { request, .. } => {
                if self.model_initialized.len() >= self.limits.maximum_requests {
                    return Err(capacity(line));
                }
                self.model_initialized.insert(request.clone(), at);
            }
            _ => (),
        }
        Ok(())
    }
    fn insert(
        &mut self,
        correlation: ferrum_types::ImageRequestCorrelation,
        at: u64,
        never_queued: bool,
        outcome: Option<String>,
        line: u64,
    ) -> Result<(), ImageContinuousTraceError> {
        let id = correlation.request_id.to_string();
        if self.requests.contains_key(&id) {
            return Err(invalid(
                line,
                "service request correlation repeats an identity",
            ));
        }
        if self.requests.len() >= self.limits.maximum_requests {
            return Err(capacity(line));
        }
        self.requests.insert(
            id.clone(),
            RequestState {
                report: ImageTraceRequestTiming {
                    request_id: id,
                    client_attempt_id: correlation.client_attempt_id.map(|id| id.to_string()),
                    request: None,
                    model_request_initialized: false,
                    never_queued,
                    accepted_ns: (!never_queued).then_some(at),
                    queue_intervals: Vec::new(),
                    queue_total_ns: None,
                    initialization_started_ns: None,
                    accepted_to_initialization_ns: None,
                    service_finished_ns: never_queued.then_some(at),
                    outcome,
                },
                queued_at: (!never_queued).then_some(at),
                last_boundary: at,
            },
        );
        self.observed = true;
        Ok(())
    }
    fn request(
        &mut self,
        id: &str,
        at: u64,
        line: u64,
    ) -> Result<&mut RequestState, ImageContinuousTraceError> {
        valid_uuid(id, line)?;
        let request = self
            .requests
            .get_mut(id)
            .ok_or_else(|| invalid(line, "service boundary has no original queued request"))?;
        if request.report.service_finished_ns.is_some() || at < request.last_boundary {
            return Err(invalid(
                line,
                "service boundary repeats after terminal or moves backwards",
            ));
        }
        request.last_boundary = at;
        Ok(request)
    }
    pub(super) fn finish(mut self, schema: u32, complete: bool) -> ImageTraceTimingAnalysis {
        for state in self.requests.values_mut() {
            state.report.model_request_initialized = state
                .report
                .request
                .as_ref()
                .is_some_and(|request| self.model_initialized.contains_key(request));
        }
        let mut reasons = Vec::new();
        if schema == 1 {
            reasons.push(ImageTraceTimingUnavailable::SchemaOne);
        }
        if !complete {
            reasons.push(ImageTraceTimingUnavailable::IncompleteTrace);
        }
        if !self.phase_starts.is_empty()
            || !self.step_starts.is_empty()
            || self
                .requests
                .values()
                .any(|r| r.queued_at.is_some() || r.report.service_finished_ns.is_none())
        {
            self.missing = true;
        }
        if self.missing {
            reasons.push(ImageTraceTimingUnavailable::MissingEndpoints);
        }
        if !self.observed {
            reasons.push(ImageTraceTimingUnavailable::NoTimingEvents);
        }
        ImageTraceTimingAnalysis {
            status: if reasons.is_empty() {
                ImageStepEvidenceStatus::Observed
            } else {
                ImageStepEvidenceStatus::NotEvaluated
            },
            unavailable_reasons: reasons,
            requests: self.requests.into_values().map(|r| r.report).collect(),
            phases: self.phases,
            steps: self.steps,
        }
    }
}
fn close_queue(
    request: &mut RequestState,
    at: u64,
    line: u64,
) -> Result<(), ImageContinuousTraceError> {
    let start = request
        .queued_at
        .take()
        .ok_or_else(|| invalid(line, "dequeue has no original queue entry"))?;
    request
        .report
        .queue_intervals
        .push(interval(start, at, line)?);
    request.report.queue_total_ns = Some(
        request
            .report
            .queue_intervals
            .iter()
            .try_fold(0u64, |sum, sample| sum.checked_add(sample.duration_ns))
            .ok_or_else(|| invalid(line, "queue interval sum overflows"))?,
    );
    Ok(())
}
fn outcome_valid(value: &str, line: u64) -> Result<(), ImageContinuousTraceError> {
    if ["completed", "cancelled", "rejected", "failed", "shutdown"].contains(&value) {
        Ok(())
    } else {
        Err(invalid(line, "service outcome is outside the typed schema"))
    }
}
fn valid_phase(value: &str) -> bool {
    [
        "metadata_preparation",
        "initialization",
        "initialization_advance",
        "prompt_expansion",
        "prompt_expansion_vision",
        "conditioning",
        "conditioning_vision",
        "reference_encoder",
        "refiner",
        "decoder",
        "noise",
        "vision",
        "projection",
        "text_encoder",
    ]
    .contains(&value)
}
fn valid_uuid(id: &str, line: u64) -> Result<(), ImageContinuousTraceError> {
    ferrum_types::parse_image_attempt_id(Some(id))
        .map(|_| ())
        .map_err(|_| invalid(line, "request correlation is not a bounded UUID"))
}
fn duration(start: u64, end: u64, line: u64) -> Result<u64, ImageContinuousTraceError> {
    end.checked_sub(start)
        .ok_or_else(|| invalid(line, "interval boundary moves backwards"))
}
fn interval(
    start_ns: u64,
    end_ns: u64,
    line: u64,
) -> Result<ImageTraceInterval, ImageContinuousTraceError> {
    Ok(ImageTraceInterval {
        start_ns,
        end_ns,
        duration_ns: duration(start_ns, end_ns, line)?,
    })
}
fn invalid(line: u64, reason: &str) -> ImageContinuousTraceError {
    error(line, ImageContinuousTraceErrorKind::Invalid, reason)
}
fn capacity(line: u64) -> ImageContinuousTraceError {
    error(
        line,
        ImageContinuousTraceErrorKind::Capacity,
        "timing request state exceeds reader capacity",
    )
}
