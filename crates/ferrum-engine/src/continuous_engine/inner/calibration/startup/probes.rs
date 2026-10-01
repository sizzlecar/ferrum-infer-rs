use super::*;
use ferrum_interfaces::output_flow::OutputCompletion;
use ferrum_scheduler::implementations::continuous::prefill_reference::PiecewiseReferenceSpec;
use futures::StreamExt;
use progress::{Stage, StartupProgress};

#[derive(Clone, Copy, Debug)]
pub(super) enum Discovery {
    Prefill,
    Decode,
}

enum ProbeOutcome {
    Ready(Vec<CalibrationReferenceDiscoverySample>),
    RouteNotReady { cold_waves: usize },
}

pub(super) enum Capture<'a> {
    Warmup,
    PrefillDiscovery,
    DecodeDiscovery,
    Trial {
        collector: &'a mut CalibrationReferenceCollector,
        key: CalibrationReferenceTrial,
    },
}

impl CalibrationSession {
    #[cfg(test)]
    pub(super) async fn startup_probe(
        &mut self,
        prompt: &plan::ProbePrompt,
        partition: &PiecewiseReferenceSpec,
        capture: Capture<'_>,
        budget: &mut plan::ProbeBudget,
    ) -> Result<Vec<CalibrationReferenceDiscoverySample>> {
        self.startup_probe_tracked(
            prompt,
            partition,
            capture,
            budget,
            &mut StartupProgress::default(),
        )
        .await
    }

    pub(super) async fn startup_probe_tracked(
        &mut self,
        prompt: &plan::ProbePrompt,
        partition: &PiecewiseReferenceSpec,
        capture: Capture<'_>,
        budget: &mut plan::ProbeBudget,
        progress: &mut StartupProgress,
    ) -> Result<Vec<CalibrationReferenceDiscoverySample>> {
        match self
            .startup_probe_attempt(prompt, partition, capture, budget, false, 0, progress)
            .await?
        {
            ProbeOutcome::Ready(samples) => Ok(samples),
            ProbeOutcome::RouteNotReady { .. } => Err(invalid(
                "non-discovery probe cannot replace a cold observation",
            )),
        }
    }

    #[cfg(test)]
    pub(super) async fn startup_discovery(
        &mut self,
        prompt: &plan::ProbePrompt,
        partition: &PiecewiseReferenceSpec,
        discovery: Discovery,
        budget: &mut plan::ProbeBudget,
    ) -> Result<Vec<CalibrationReferenceDiscoverySample>> {
        self.startup_discovery_tracked(
            prompt,
            partition,
            discovery,
            budget,
            &mut StartupProgress::default(),
        )
        .await
    }

    pub(super) async fn startup_discovery_tracked(
        &mut self,
        prompt: &plan::ProbePrompt,
        partition: &PiecewiseReferenceSpec,
        discovery: Discovery,
        budget: &mut plan::ProbeBudget,
        progress: &mut StartupProgress,
    ) -> Result<Vec<CalibrationReferenceDiscoverySample>> {
        let mut attempt = 0;
        loop {
            attempt += 1;
            let capture = match discovery {
                Discovery::Prefill => Capture::PrefillDiscovery,
                Discovery::Decode => Capture::DecodeDiscovery,
            };
            match self
                .startup_probe_attempt(prompt, partition, capture, budget, true, attempt, progress)
                .await?
            {
                ProbeOutcome::Ready(samples) => return Ok(samples),
                ProbeOutcome::RouteNotReady { cold_waves } => {
                    // The complete request was consumed and retired. None of
                    // this owner's observations may enter a different trial.
                    tracing::info!(
                        ?discovery,
                        prompt_tokens = prompt.tokens.get(),
                        cold_waves,
                        probe_requests_used = budget.used(),
                        maximum_probe_requests = budget.maximum(),
                        "Automatic reference discovery completed before a fresh route-readiness retry"
                    );
                }
            }
        }
    }

    async fn startup_probe_attempt(
        &mut self,
        prompt: &plan::ProbePrompt,
        partition: &PiecewiseReferenceSpec,
        mut capture: Capture<'_>,
        budget: &mut plan::ProbeBudget,
        retry_cold_discovery: bool,
        discovery_attempt: usize,
        progress: &mut StartupProgress,
    ) -> Result<ProbeOutcome> {
        // Claim before allocating/adding a new owner. The caller's one startup
        // deadline also spans all attempts; neither budget restarts on retry.
        budget.claim()?;
        progress.begin(
            budget.used(),
            prompt.tokens.get(),
            &capture,
            discovery_attempt,
            self.engine.inner.controller_timing_snapshot(),
        );
        let result = self
            .startup_probe_request(
                prompt,
                partition,
                &mut capture,
                retry_cold_discovery,
                progress,
            )
            .await;
        let outcome = match &result {
            Ok(ProbeOutcome::Ready(_)) => "complete",
            Ok(ProbeOutcome::RouteNotReady { .. }) => "complete_cold_route",
            Err(_) => "failed",
        };
        progress.finish(self.engine.inner.controller_timing_snapshot(), outcome);
        result
    }

    async fn startup_probe_request(
        &mut self,
        prompt: &plan::ProbePrompt,
        partition: &PiecewiseReferenceSpec,
        capture: &mut Capture<'_>,
        retry_cold_discovery: bool,
        progress: &mut StartupProgress,
    ) -> Result<ProbeOutcome> {
        let mut request = ferrum_types::InferenceRequest::new(
            prompt.text.clone(),
            self.configuration().model.model_id.clone(),
        );
        request.stream = true;
        request.sampling_params = ferrum_types::SamplingParams::greedy();
        request.sampling_params.max_tokens = plan::OUTPUT_TOKENS;
        request
            .metadata
            .insert("ferrum_ignore_eos".into(), true.into());
        let id = request.id.clone();
        let output = self
            .add_request(
                request,
                InferenceRequestContext::from_ingress(slo_clock_now()),
                Arc::new(OutputProjectionContract::cli_text()),
            )
            .await?;
        progress.stage(Stage::Frontier);
        let initial = self
            .frontiers()?
            .into_iter()
            .find(|frontier| frontier.request_id() == &id)
            .ok_or_else(|| invalid("automatic probe original frontier missing"))?;
        if initial.request_evidence().original_input_tokens != prompt.tokens.get() as usize {
            return Err(invalid(
                "automatic probe tokenization changed after declaration",
            ));
        }
        if let Capture::Trial { collector, key } = &mut *capture {
            self.begin_reference_trial(collector, *key, &initial)?;
        }
        let driver = async {
            let mut observations = Vec::new();
            let mut complete = false;
            let mut cold_waves = 0usize;
            loop {
                progress.stage(Stage::Frontier);
                let Some(before) = self
                    .frontiers()?
                    .into_iter()
                    .find(|frontier| frontier.request_id() == &id)
                else {
                    break;
                };
                // A zero-submit wave may have retained a one-use backing
                // maintenance ticket. As in the manual calibration driver,
                // retire it in its own turn before recapturing exact work.
                progress.stage(Stage::Maintenance);
                progress.counts().maintenance_turns += 1;
                match self.step(CalibrationAction::Maintenance).await? {
                    CalibrationTurn::MaintenanceReconciled => {
                        progress.counts().maintenance_reconciled += 1;
                        continue;
                    }
                    CalibrationTurn::Blocked(CalibrationBlockReason::MaintenanceUnavailable) => {}
                    CalibrationTurn::Blocked(CalibrationBlockReason::PublicationUnavailable) => {
                        progress.counts().maintenance_blocked += 1;
                        tokio::task::yield_now().await;
                        continue;
                    }
                    other => {
                        return Err(invalid(format!(
                            "automatic probe maintenance unavailable: {other:?}"
                        )))
                    }
                }
                progress.stage(Stage::OutputReadiness);
                self.startup_output_ready(&id).await?;
                if self.engine.inner.scheduler.waiting_count() != 0 {
                    progress.stage(Stage::Admission);
                    progress.counts().admission_turns += 1;
                    match self.step(CalibrationAction::AdmitOne).await? {
                        CalibrationTurn::AdmittedOrMaintained => {
                            progress.counts().admission_progressed += 1;
                            tokio::task::yield_now().await;
                            continue;
                        }
                        CalibrationTurn::Blocked(CalibrationBlockReason::AdmissionUnavailable) => {
                            progress.counts().admission_blocked += 1;
                            tokio::task::yield_now().await;
                            continue;
                        }
                        other => {
                            return Err(invalid(format!(
                                "automatic probe admission failed: {other:?}"
                            )))
                        }
                    }
                }
                progress.stage(Stage::Frontier);
                let work = if let Some((offset, total)) = before.prefill_progress() {
                    before.prefill_work(
                        partition
                            .next_count(
                                u32::try_from(total)
                                    .map_err(|_| invalid("probe input overflow"))?,
                                u32::try_from(offset)
                                    .map_err(|_| invalid("probe prefix overflow"))?,
                            )
                            .map_err(|error| invalid(error.to_string()))?,
                    )?
                } else {
                    before.decode_work()?
                };
                progress.begin_wave(&before, self.engine.inner.controller_timing_snapshot());
                let turn = self.step(CalibrationAction::Wave(vec![work])).await;
                progress.end_wave(
                    match &turn {
                        Ok(CalibrationTurn::Wave(report) | CalibrationTurn::Reaped(report)) => {
                            Some(report)
                        }
                        _ => None,
                    },
                    self.engine.inner.controller_timing_snapshot(),
                    false,
                );
                progress.stage(Stage::ReferenceCapture);
                let report = match turn? {
                    CalibrationTurn::Wave(report) | CalibrationTurn::Reaped(report) => report,
                    CalibrationTurn::Blocked(CalibrationBlockReason::ResourceUnavailable(
                        ferrum_interfaces::vnext::ResourcePlanningUnknown::ReadUnavailable(_),
                    ))
                    | CalibrationTurn::Blocked(CalibrationBlockReason::PublicationUnavailable)
                    | CalibrationTurn::Blocked(CalibrationBlockReason::MaintenanceUnavailable)
                    | CalibrationTurn::Blocked(CalibrationBlockReason::SelectionUnavailable(
                        "output_or_resource_blocked",
                    )) => {
                        progress.counts().wave_blocked += 1;
                        tokio::task::yield_now().await;
                        continue;
                    }
                    other => {
                        return Err(invalid(format!(
                            "automatic probe wave unavailable: {other:?}"
                        )))
                    }
                };
                if report.submission == CalibrationSubmissionState::NotSubmitted
                    && report.error.is_none()
                {
                    progress.counts().zero_submit_retries += 1;
                    // This conclusive receipt proves no model work executed.
                    // Re-read the original owner frontier without recording a
                    // discovery/trial observation or advancing its partition.
                    // The outer startup duration budget still bounds retries.
                    tokio::task::yield_now().await;
                    continue;
                }
                if report.submission != CalibrationSubmissionState::HostReconciled
                    || report.error.is_some()
                {
                    return Err(invalid(format!(
                        "automatic probe wave failed: submission={:?}; diagnostic={}",
                        report.submission,
                        reference::unavailable_summary(&report)
                    )));
                }
                let discovery_required = match &*capture {
                    Capture::PrefillDiscovery => before.prefill_progress().is_some(),
                    // Frozen decode trials retain their complete preparation
                    // chain, so discovery must establish readiness there too.
                    Capture::DecodeDiscovery => {
                        before.prefill_progress().is_some() || before.generated_tokens() == 1
                    }
                    _ => false,
                };
                if discovery_required {
                    if retry_cold_discovery && graph_route_not_ready(&report) {
                        cold_waves += 1;
                        observations.clear();
                    } else {
                        let sample = self.capture_reference_discovery(&before, &report)?;
                        let selected = matches!(&*capture, Capture::PrefillDiscovery)
                            || before.prefill_progress().is_none();
                        if selected && cold_waves == 0 {
                            observations.push(sample);
                        }
                    }
                    continue;
                }
                match &mut *capture {
                    Capture::Warmup => {}
                    Capture::Trial { collector, key } if !complete => {
                        complete = collector.observe(*key, &report)?;
                    }
                    _ => {}
                }
            }
            if matches!(&*capture, Capture::Trial { .. }) && !complete {
                return Err(invalid(
                    "automatic reference trial ended before all declared observations",
                ));
            }
            progress.stage(Stage::OutputCompletion);
            Ok(if cold_waves == 0 {
                ProbeOutcome::Ready(observations)
            } else {
                ProbeOutcome::RouteNotReady { cold_waves }
            })
        };
        let (outcome, ()) = tokio::try_join!(driver, consume(output))?;
        Ok(outcome)
    }

    pub(in crate::continuous_engine::inner::calibration) async fn startup_output_ready(
        &self,
        id: &RequestId,
    ) -> Result<()> {
        use crate::continuous_engine::output_flow_runtime::OutputReadinessState;
        let mut changed = self
            .engine
            .inner
            .sequences
            .read()
            .get(id)
            .and_then(|sequence| sequence.credited_output.as_ref())
            .ok_or_else(|| invalid("automatic probe output owner disappeared"))?
            .port
            .subscribe();
        loop {
            let readiness = self
                .engine
                .inner
                .sequences
                .read()
                .get(id)
                .and_then(|sequence| sequence.credited_output.as_ref())
                .ok_or_else(|| invalid("automatic probe output owner disappeared"))?
                .port
                .readiness();
            match readiness {
                OutputReadinessState::Ready => return Ok(()),
                OutputReadinessState::Closing(_) => {
                    return Err(invalid("automatic probe output closed before execution"))
                }
                _ => changed
                    .changed()
                    .await
                    .map_err(|_| invalid("automatic probe output readiness was lost"))?,
            }
        }
    }
}

/// A retry decision only, never a replacement for Witness::capture. Require
/// the original recorder's complete singleton GraphPath failure; queue loss,
/// other unknown reasons and contradictory observations remain fatal.
fn graph_route_not_ready(report: &CalibrationWaveReport) -> bool {
    use ferrum_interfaces::execution_cost::ActualWaveEvidenceUnknown;
    let Some(actual) = &report.actual_evidence_diagnostic else {
        return false;
    };
    report.submission == CalibrationSubmissionState::HostReconciled
        && report.error.is_none()
        && matches!(report.observation, CalibrationObservation::Rejected { .. })
        && actual.physical_waves == 1
        && actual.retained_waves == 1
        && actual.lost_observations == 0
        && actual.retained_wave_details_complete
        && actual.dispatch_unknown == Some(ActualWaveEvidenceUnknown::GraphPath)
        && actual.waves.len() == 1
        && actual.waves[0].reason == ActualWaveEvidenceUnknown::GraphPath
}

async fn consume(output: CreditedOutputSession) -> Result<()> {
    let CreditedOutputSession {
        mut frames,
        completion,
    } = output;
    let mut terminal = false;
    while let Some(frame) = frames.next().await {
        let metadata = frame.metadata();
        if frame.wire().credit().events == 0
            || frame.wire().payload().capacity() > frame.wire().credit().bytes
        {
            return Err(invalid("automatic probe output exceeded its real credit"));
        }
        drop(frame);
        if metadata.terminal {
            terminal = true;
            break;
        }
    }
    if !terminal {
        return Err(invalid("automatic probe output closed before terminal"));
    }
    let completion = completion
        .await
        .map_err(|_| invalid("automatic probe completion owner was lost"))?;
    match completion.payload() {
        OutputCompletion::Succeeded {
            reason: ferrum_types::FinishReason::Length,
            usage,
            ..
        } if usage.completion_tokens == plan::OUTPUT_TOKENS => Ok(()),
        OutputCompletion::Succeeded { .. } => Err(invalid(
            "automatic probe did not complete its declared output",
        )),
        OutputCompletion::Failed(error) => Err(invalid(format!(
            "automatic probe output failed: {}",
            error.message()
        ))),
    }
}
