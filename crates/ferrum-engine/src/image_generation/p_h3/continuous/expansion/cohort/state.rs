//! One original phase and at most one native owner. Visual outputs are bound
//! and closed one at a time; the survivor map never replaces sealed job IDs.
use super::super::vision::{RetainedCohortVisual, VisualInputBinding};
use super::*;

#[cfg(test)]
#[path = "state/vision_reuse_tests.rs"]
mod vision_reuse_tests;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum CohortPhase {
    Visuals,
    Install,
    Native,
    Done,
    Failed,
}

pub(super) struct Cohort<R: DeviceRuntime> {
    pub pipeline: Arc<()>,
    pub assets: Arc<NativeTextExpansion>,
    pub members: Vec<Member>,
    pub receipts: Vec<ImageStageReceipt>,
    pub stage: CohortPhase,
    phase: Option<HostMemoryScope>,
    preparations: Vec<Option<PreparedTextExpansion>>,
    visuals: Vec<Option<PreparedVisual<R>>>,
    next_visual: usize,
    retained_visual: Option<RetainedCohortVisual<R>>,
    native: Option<ProductExpansionCohort<R>>,
    active_indices: Vec<usize>,
    outcomes: Vec<Option<ExpansionParticipantResult>>,
    unvalidated_outcomes: Vec<ExpansionParticipantResult>,
    retired_at: Option<Instant>,
    // Keep the original allocation owners independently of the core's move
    // into initialization. Its error may own native cleanup before admissions
    // retain these per-request inputs.
    inputs: Vec<Arc<ferrum_execution::CausalRequestInputs>>,
    _requirements: Option<CohortRequirements>,
}
impl<R: DeviceRuntime> Cohort<R> {
    pub fn new(mut prepared: PreparedCohort<R>) -> Self {
        let stage = if prepared.visuals.iter().any(Option::is_some) {
            CohortPhase::Visuals
        } else {
            CohortPhase::Install
        };
        let count = prepared.members.len();
        Self {
            pipeline: prepared.pipeline,
            assets: prepared.assets,
            members: prepared.members,
            receipts: prepared.receipts,
            stage,
            phase: prepared.phase,
            preparations: prepared
                .prepared
                .take()
                .expect("validated original cohort plan")
                .into_requests()
                .into_iter()
                .map(Some)
                .collect(),
            visuals: prepared.visuals,
            next_visual: 0,
            retained_visual: None,
            native: None,
            active_indices: vec![],
            outcomes: (0..count).map(|_| None).collect(),
            unvalidated_outcomes: vec![],
            retired_at: None,
            inputs: vec![],
            _requirements: Some(prepared.requirements),
        }
    }
    pub fn failed_marker(pipeline: Arc<()>, assets: Arc<NativeTextExpansion>) -> Self {
        Self {
            pipeline,
            assets,
            members: vec![],
            receipts: vec![],
            stage: CohortPhase::Failed,
            phase: None,
            preparations: vec![],
            visuals: vec![],
            next_visual: 0,
            retained_visual: None,
            native: None,
            active_indices: vec![],
            outcomes: vec![],
            unvalidated_outcomes: vec![],
            retired_at: None,
            inputs: vec![],
            _requirements: None,
        }
    }
}

/// Native recovery, prior bound inputs and all source assumptions move into
/// the actual returned error. Correct custody does not rely on an opaque
/// worker preparation surviving a Fatal response.
pub(super) struct CohortFailure<R: DeviceRuntime> {
    source: ImageBatchError,
    _cohort: Cohort<R>,
}
impl<R: DeviceRuntime> CohortFailure<R> {
    #[cfg(test)]
    pub(super) fn cleanup(&self) -> Option<ferrum_execution::ExecutorShutdownObserver> {
        self._cohort
            .native
            .as_ref()
            .map(ProductExpansionCohort::cleanup)
    }

    pub fn new(source: ImageBatchError, cohort: Cohort<R>) -> Self {
        Self {
            source,
            _cohort: cohort,
        }
    }
}
impl<R: DeviceRuntime> std::fmt::Display for CohortFailure<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "H3 initialization cohort: {}", self.source)
    }
}
impl<R: DeviceRuntime> std::fmt::Debug for CohortFailure<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(self, f)
    }
}
impl<R: DeviceRuntime> std::error::Error for CohortFailure<R> {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.source {
            ImageBatchError::Fatal(source) => Some(source.as_ref()),
            _ => None,
        }
    }
}

impl<R: DeviceRuntime> PH3ImagePipeline<R> {
    pub(super) fn cancel_pre_native_members(
        &mut self,
        cohort: &mut Cohort<R>,
        requests: &[ImageInitializationCohortRequest<'_>],
    ) {
        if !matches!(cohort.stage, CohortPhase::Visuals | CohortPhase::Install) {
            return;
        }
        for (index, input) in requests.iter().enumerate() {
            if input.cancelled {
                // Every previous producer already returned an actual Closed
                // acknowledgement. These immutable inputs can now be dropped.
                drop(cohort.preparations[index].take());
                drop(cohort.visuals[index].take());
                if cohort.outcomes[index].is_none() {
                    cohort.outcomes[index] = Some(ExpansionParticipantResult::Cancelled)
                }
            }
        }
    }

    pub(super) fn install_cohort(&mut self, cohort: &mut Cohort<R>) -> InitResult<()> {
        // This Output grant is only for sequential visuals, never causal
        // installation. Bound per-request Inputs already own their copies.
        drop(cohort.retained_visual.take());
        cohort.active_indices.clear();
        let mut surviving = Vec::new();
        for (index, prepared) in cohort.preparations.iter_mut().enumerate() {
            if let Some(prepared) = prepared.take() {
                if let Some(inputs) = prepared.retained_causal_inputs() {
                    cohort.inputs.push(inputs)
                }
                cohort.active_indices.push(index);
                surviving.push(prepared);
            } else if cohort.outcomes[index].is_none() {
                return Err(ImageBatchError::Rejected(
                    "cohort member lost both its preparation and terminal outcome".into(),
                ));
            }
        }
        if surviving.is_empty() {
            drop(cohort.phase.take());
            cohort.stage = CohortPhase::Done;
            cohort.retired_at = Some(Instant::now());
            return Ok(());
        }
        for request in &mut surviving {
            cohort
                .assets
                .configure_request_timing(
                    request,
                    self.execution.config.executor.device_timing_mode,
                )
                .map_err(|error| preparation_error(error.into()))?;
        }
        let prepared = cohort
            .assets
            .prepare_cohort(surviving)
            .map_err(|e| preparation_error(e.into()))?;
        let phase = cohort.phase.take().ok_or_else(|| {
            ImageBatchError::Rejected("cohort lost its original phase before installation".into())
        })?;
        let diagnostics=cohort.active_indices.iter().map(|&index|{
            let member=&cohort.members[index];
            match self.prompt_expansion_diagnostics{
                crate::image_generation::p_h3::PH3ImagePromptExpansionDiagnosticPolicy::Disabled=>None,
                policy=>self.execution.continuous_trace.as_ref().map(|trace|crate::image_generation::p_h3::expansion::DiagnosticContext{policy,trace:trace.handle(),job:member.job,request_id:member.correlation.map(|c|c.request_id)}),
            }
        }).collect();
        match ProductExpansionCohort::initialize_with_diagnostics(
            cohort.assets.clone(),
            self.expansion_composition(),
            prepared,
            phase,
            diagnostics,
        ) {
            Ok(native) => {
                cohort.native = Some(native);
                cohort.stage = CohortPhase::Native;
                Ok(())
            }
            Err(error) if error.cleanup().is_some() => Err(ImageBatchError::Fatal(Box::new(error))),
            Err(error) => {
                // No cleanup obligation is the constructor's explicit safe
                // rejection proof. Preserve earlier member outcomes.
                let diagnostic = error.to_string();
                for &index in &cohort.active_indices {
                    cohort.outcomes[index] = Some(ExpansionParticipantResult::Failed(
                        ferrum_types::FerrumError::request_validation(diagnostic.clone()),
                    ))
                }
                cohort.stage = CohortPhase::Done;
                cohort.retired_at = Some(Instant::now());
                Ok(())
            }
        }
    }

    pub(super) fn advance_cohort(
        &mut self,
        cohort: &mut Cohort<R>,
        requests: &[ImageInitializationCohortRequest<'_>],
        wake: &Waker,
    ) -> InitResult<ImageInitializationCohortProgress> {
        self.cancel_pre_native_members(cohort, requests);
        match cohort.stage {
            CohortPhase::Visuals => {
                while cohort.next_visual < cohort.members.len() {
                    let index = cohort.next_visual;
                    cohort.next_visual += 1;
                    if cohort.preparations[index].is_none() {
                        continue;
                    }
                    let visual = cohort.visuals[index].as_ref().ok_or_else(|| {
                        ImageBatchError::Rejected(
                            "visual cohort member has no original producer".into(),
                        )
                    })?;
                    let phase = cohort.phase.take().ok_or_else(|| {
                        ImageBatchError::Rejected("visual cohort lost its original phase".into())
                    })?;
                    let reuse_enabled = cohort.assets.vision_reuse()
                        == crate::image_generation::p_h3::PH3ImagePromptExpansionVisionReuse::ExactImageWithinCohort;
                    let reuse = reuse_enabled
                        && cohort
                            .retained_visual
                            .as_ref()
                            .is_some_and(|saved| saved.matches(&cohort.assets, visual));
                    if !reuse {
                        // Keep the admitted maximum of one visual Output.
                        drop(cohort.retained_visual.take());
                    }
                    let result = self.bind_visual_input_with_reuse(
                        &cohort.assets,
                        cohort.preparations[index].as_mut().unwrap(),
                        visual,
                        phase,
                        cohort.members[index].job,
                        cohort.members[index].correlation,
                        cohort.retained_visual.as_ref(),
                    )?;
                    match result {
                        VisualInputBinding::Bound { phase, output } => {
                            cohort.phase = Some(phase);
                            if reuse_enabled && !reuse {
                                cohort.retained_visual = Some(RetainedCohortVisual::new(
                                    cohort.assets.clone(),
                                    cohort.visuals[index].take().unwrap(),
                                    output,
                                    cohort.members[index].job,
                                    cohort.members[index].correlation,
                                ));
                            }
                        }
                        VisualInputBinding::Rejected { phase, diagnostic } => {
                            cohort.phase = Some(phase);
                            drop(cohort.preparations[index].take());
                            cohort.outcomes[index] = Some(ExpansionParticipantResult::Failed(
                                ferrum_types::FerrumError::request_validation(diagnostic),
                            ));
                        }
                    }
                    drop(cohort.visuals[index].take());
                    if cohort.next_visual == cohort.members.len() {
                        cohort.stage = CohortPhase::Install
                    }
                    // One member binds per worker boundary, even on a hit.
                    // A fresh cancellation snapshot precedes installation.
                    return Ok(pending(true));
                }
                cohort.stage = CohortPhase::Install;
                return Ok(pending(true));
            }
            CohortPhase::Install => {
                self.install_cohort(cohort)?;
                if cohort.stage != CohortPhase::Done {
                    return Ok(pending(true));
                }
            }
            CohortPhase::Native => {
                let cancelled = cohort
                    .active_indices
                    .iter()
                    .map(|&index| requests[index].cancelled)
                    .collect::<Vec<_>>();
                let native = cohort.native.as_mut().ok_or_else(|| {
                    ImageBatchError::Rejected("native cohort owner disappeared".into())
                })?;
                let outcomes = match native.advance(&cancelled, wake) {
                    Ok(ExpansionCohortProgress::Pending { made_progress }) => {
                        return Ok(pending(made_progress))
                    }
                    Ok(ExpansionCohortProgress::Retired(outcomes)) => outcomes,
                    Err(error) => {
                        let cleanup = native.cleanup();
                        if cleanup.snapshot().phase() != ExecutorShutdownPhase::Retired {
                            return Err(ImageBatchError::Fatal(Box::new(
                                ExpansionCleanupFailure { error, cleanup },
                            )));
                        }
                        cohort
                            .active_indices
                            .iter()
                            .map(|_| {
                                ExpansionParticipantResult::Failed(
                                    ferrum_types::FerrumError::request_validation(
                                        error.to_string(),
                                    ),
                                )
                            })
                            .collect()
                    }
                };
                if outcomes.len() != cohort.active_indices.len() {
                    cohort.unvalidated_outcomes = outcomes;
                    return Err(ImageBatchError::Rejected(
                        "native cohort retired with a different survivor membership".into(),
                    ));
                }
                for (index, outcome) in cohort.active_indices.iter().copied().zip(outcomes) {
                    if cohort.outcomes[index].is_some() {
                        cohort.unvalidated_outcomes.push(outcome);
                        return Err(ImageBatchError::Rejected(
                            "native cohort overwrote an original terminal member".into(),
                        ));
                    }
                    cohort.outcomes[index] = Some(outcome);
                }
                cohort.retired_at = Some(Instant::now());
                drop(cohort.native.take());
                cohort.inputs.clear();
                cohort.stage = CohortPhase::Done;
            }
            CohortPhase::Done => (),
            CohortPhase::Failed => {
                return Err(ImageBatchError::Rejected(
                    "cohort retains an earlier fatal owner".into(),
                ))
            }
        }
        self.complete_cohort(cohort, requests)
    }

    fn complete_cohort(
        &mut self,
        cohort: &mut Cohort<R>,
        requests: &[ImageInitializationCohortRequest<'_>],
    ) -> InitResult<ImageInitializationCohortProgress> {
        if cohort.outcomes.len() != cohort.members.len()
            || cohort.outcomes.iter().any(Option::is_none)
        {
            return Err(ImageBatchError::Rejected(
                "retired cohort lacks an original member outcome".into(),
            ));
        }
        let outcomes = std::mem::take(&mut cohort.outcomes)
            .into_iter()
            .map(Option::unwrap)
            .collect::<Vec<_>>();
        let retired_at = cohort.retired_at.ok_or_else(|| {
            ImageBatchError::Rejected("cohort has no actual retirement boundary".into())
        })?;
        finish_retired_prompt_spans(
            &mut cohort.members,
            requests,
            &outcomes,
            retired_at,
            |start, outcome, at| self.trace_phase_finished_at(start, outcome, at),
        );
        // The native owner and all visual sessions are quiescent. Release the
        // final initialization-only aliases before preparing downstream work.
        cohort.inputs.clear();
        let mut completed = Vec::with_capacity(outcomes.len());
        for ((member, input), outcome) in std::mem::take(&mut cohort.members)
            .into_iter()
            .zip(requests)
            .zip(outcomes)
        {
            let outcome = if input.cancelled {
                ImageInitializationCohortOutcome::Cancelled
            } else {
                match outcome {
                    ExpansionParticipantResult::Expanded(text) => {
                        let effective = EffectiveRequest::from_expanded(input.request, text);
                        match self.prepare_continuous_image_request_with_job(
                            effective.as_ref(),
                            member.job,
                        ) {
                            Ok(prepared) => {
                                if let Some(correlation) = member.correlation {
                                    prepared.link_correlation(self, correlation)
                                }
                                ImageInitializationCohortOutcome::Pending(
                                    ImageRequestInitialization::new(Initialization::<R> {
                                        correlation: member.correlation,
                                        prompt_trace: None,
                                        pipeline: cohort.pipeline.clone(),
                                        identity: member.identity,
                                        job: member.job,
                                        output_credit: member.output_credit,
                                        receipts: std::mem::take(&mut cohort.receipts),
                                        state: InitializationState::Downstream {
                                            effective,
                                            prepared,
                                        },
                                    }),
                                )
                            }
                            Err(error) => ImageInitializationCohortOutcome::Rejected {
                                diagnostic: format!(
                                    "expanded prompt cannot prepare image: {error:#}"
                                ),
                            },
                        }
                    }
                    ExpansionParticipantResult::Cancelled => {
                        ImageInitializationCohortOutcome::Cancelled
                    }
                    ExpansionParticipantResult::Failed(error) => {
                        ImageInitializationCohortOutcome::Rejected {
                            diagnostic: error.to_string(),
                        }
                    }
                }
            };
            completed.push(ImageInitializationCohortCompletion {
                request_id: member.request_id,
                outcome,
            });
        }
        if self.execution.sources.assets.main_vision_reuse
            == crate::image_generation::p_h3::PH3ImageMainVisionReuse::ExactImageWithinCohort
        {
            let mut capacity = 0;
            let mut count = 0;
            let mut first_job = u64::MAX;
            for member in &mut completed {
                if let ImageInitializationCohortOutcome::Pending(initialization) =
                    &mut member.outcome
                {
                    let initialization = initialization.get_mut::<Initialization<R>>()?;
                    if let InitializationState::Downstream { prepared, .. } = &initialization.state
                    {
                        let bytes = prepared
                            .main_vision_output_capacity(&self.execution.host)
                            .map_err(|e| preparation_error(e.into()))?;
                        if bytes > 0 {
                            capacity = capacity.max(bytes);
                            count += 1;
                            first_job = first_job.min(initialization.job);
                        }
                    }
                }
            }
            // Single-image and generation cohorts keep the original route.
            if count > 1 {
                let shared =
                    crate::image_generation::p_h3::vision::reuse::CohortMainVisionReuse::new(
                        &self.execution.host,
                        self.execution.service_preparation_identity.clone(),
                        capacity,
                        first_job,
                    )
                    .map_err(|e| preparation_error(e.into()))?;
                for member in &mut completed {
                    if let ImageInitializationCohortOutcome::Pending(initialization) =
                        &mut member.outcome
                    {
                        if let InitializationState::Downstream { prepared, .. } =
                            &mut initialization.get_mut::<Initialization<R>>()?.state
                        {
                            prepared.bind_main_vision_reuse(shared.clone());
                        }
                    }
                }
            }
        }
        Ok(ImageInitializationCohortProgress {
            made_progress: true,
            finished: true,
            completed,
        })
    }
}
fn pending(made_progress: bool) -> ImageInitializationCohortProgress {
    ImageInitializationCohortProgress {
        made_progress,
        finished: false,
        completed: vec![],
    }
}
