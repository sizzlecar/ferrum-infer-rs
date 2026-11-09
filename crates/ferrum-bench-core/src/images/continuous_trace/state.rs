use super::*;
use std::collections::{BTreeMap, BTreeSet};

struct Progress {
    step: u64,
    next_sigma: f32,
    wave: u64,
}
struct Request {
    progress: Option<Progress>,
    layout: Option<Vec<(u32, u64)>>,
    exited: bool,
    refill_departure: bool,
}
struct Publication {
    job: u64,
    members: Vec<String>,
    plan: String,
    program: String,
}
pub(super) struct Analyzer {
    limits: ImageContinuousTraceReadLimits,
    identity: Option<ImageTraceIdentity>,
    requests: BTreeMap<String, Request>,
    cohort_jobs: BTreeSet<u64>,
    publication: Option<Publication>,
    last_wave: Option<(u64, Vec<String>)>,
    facts: ImageContinuousTraceFacts,
    expansion_observations: Vec<ImageTracePromptExpansionObservation>,
    expansion_jobs: BTreeSet<u64>,
    expansion_requests: BTreeSet<uuid::Uuid>,
    token_waves: producer_extensions::TokenWaves,
    reference_latents: BTreeMap<String, ImageTraceReferenceLatentDigest>,
}
impl Analyzer {
    pub(super) fn new(limits: ImageContinuousTraceReadLimits) -> Self {
        Self {
            limits,
            identity: None,
            requests: BTreeMap::new(),
            cohort_jobs: BTreeSet::new(),
            publication: None,
            last_wave: None,
            facts: Default::default(),
            expansion_observations: Vec::new(),
            expansion_jobs: BTreeSet::new(),
            expansion_requests: BTreeSet::new(),
            token_waves: Default::default(),
            reference_latents: BTreeMap::new(),
        }
    }
    pub(super) fn event(
        &mut self,
        event: Event,
        raw: &[u8],
        sequence: u64,
        line: u64,
    ) -> Result<(), ImageContinuousTraceError> {
        if let Event::PromptExpansionTokenWave(wave) = event {
            if self.expansion_jobs.contains(&wave.job) {
                return Err(invalid(
                    line,
                    "expansion token wave follows its terminal observation",
                ));
            }
            return self.token_waves.event(wave, &self.limits, line);
        }
        if let Event::PromptExpansionObserved(observation) = event {
            observation.validate(line)?;
            self.token_waves
                .observation(&observation, &self.limits, line)?;
            if self.expansion_observations.len() >= self.limits.maximum_requests {
                return Err(capacity(
                    line,
                    "expansion diagnostic count exceeds reader capacity",
                ));
            }
            if !self.expansion_jobs.insert(observation.job)
                || observation
                    .request_id
                    .is_some_and(|id| !self.expansion_requests.insert(id))
            {
                return Err(invalid(
                    line,
                    "expansion diagnostic repeats its job/request identity",
                ));
            }
            self.expansion_observations.push(observation);
            return Ok(());
        }
        if let Event::Header(header) = event {
            if self.identity.is_some() || sequence != 1 {
                return Err(invalid(
                    line,
                    "header must appear exactly once as the first event",
                ));
            }
            if [&header.initial_noise_policy, &header.edit_noise_policy]
                .into_iter()
                .flatten()
                .any(|policy| policy.trim().is_empty())
            {
                return Err(invalid(line, "declared noise policy must be nonempty"));
            }
            let nonempty_object = |v: &Value| v.as_object().is_some_and(|m| !m.is_empty());
            if !nonempty_object(&header.device)
                || !nonempty_object(&header.assets)
                || !nonempty_object(&header.host)
                || !header.context.is_object()
                || header.pipeline["maximum_batch_size"]
                    .as_u64()
                    .is_none_or(|n| n == 0)
                || header.pipeline["device_capacity_bytes"]
                    .as_u64()
                    .is_none_or(|n| n == 0)
                || header.noise_profile.is_null()
                || !(1..=4).contains(&header.vae_decoder_tile_batch)
            {
                return Err(invalid(
                    line,
                    "header lacks actual runtime/source/pipeline identity",
                ));
            }
            let serve_context_present = header.context["entrypoint"] == "serve"
                && nonempty_object(&header.context["composition"])
                && nonempty_object(&header.context["service_config"]);
            self.identity = Some(ImageTraceIdentity {
                header_event_sha256: crate::images::sha256(raw),
                device: header.device,
                device_opaque_allowance_bytes: header.device_opaque_allowance_bytes,
                assets: header.assets,
                prompt_expansion: (!header.prompt_expansion.is_null())
                    .then_some(header.prompt_expansion),
                noise_profile: header.noise_profile,
                initial_noise_policy: header.initial_noise_policy,
                edit_noise_policy: header.edit_noise_policy,
                vae_decoder_tile_batch: header.vae_decoder_tile_batch,
                pipeline: header.pipeline,
                host: header.host,
                context: header.context,
                serve_context_present,
                metadata_digests_authenticate_weights: false,
            });
            return Ok(());
        }
        // A component-only diagnostic can contain genuine commits without
        // the full pipeline header/publisher. Keep it NotEvaluated and do not
        // elevate those records into full-engine cohort evidence.
        if self.identity.is_none() {
            return Ok(());
        }
        match event {
            Event::Header(_) => unreachable!(),
            Event::RequestInitialized {
                request,
                job,
                seed,
                width,
                height,
                editing,
                reference_latent,
            } => {
                let _ = (job, seed);
                if request.is_empty()
                    || width == 0
                    || height == 0
                    || self.requests.contains_key(&request)
                {
                    return Err(invalid(
                        line,
                        "request initialization repeats an ID or has empty geometry/identity",
                    ));
                }
                if self.requests.len() >= self.limits.maximum_requests {
                    return Err(capacity(
                        line,
                        "initialized request count exceeds reader capacity",
                    ));
                }
                if let Some(digest) = reference_latent {
                    digest.validate(editing, line)?;
                    self.reference_latents.insert(request.clone(), digest);
                }
                self.requests.insert(
                    request,
                    Request {
                        progress: None,
                        layout: None,
                        exited: false,
                        refill_departure: false,
                    },
                );
                self.facts.initialized_requests += 1;
            }
            Event::CohortPublished {
                cohort_job,
                ordered_members,
                plan_hash,
                program_fingerprint,
            } => {
                if ordered_members.is_empty()
                    || plan_hash.is_empty()
                    || program_fingerprint.is_empty()
                    || self.cohort_jobs.contains(&cohort_job)
                {
                    return Err(invalid(
                        line,
                        "cohort publication repeats an ID or lacks members/plan identity",
                    ));
                }
                self.check_members(&ordered_members, line)?;
                self.cohort_jobs.insert(cohort_job);
                self.publication = Some(Publication {
                    job: cohort_job,
                    members: ordered_members,
                    plan: plan_hash,
                    program: program_fingerprint,
                });
                self.facts.published_cohorts += 1;
            }
            Event::PackedStepCommitted {
                start_sequence: _,
                cohort_job,
                plan_hash,
                program_fingerprint,
                packed_request_count,
                time_basis,
                members,
            } => {
                self.commit(
                    cohort_job,
                    plan_hash,
                    program_fingerprint,
                    packed_request_count,
                    time_basis,
                    members,
                    sequence,
                    line,
                )?;
            }
            Event::RequestExitedAtFence {
                request,
                completed_steps,
                reason,
            } => {
                let state = self
                    .requests
                    .get_mut(&request)
                    .ok_or_else(|| invalid(line, "uninitialized request exited"))?;
                if state.exited
                    || state.progress.as_ref().map_or(0, |p| p.step + 1) != completed_steps
                {
                    return Err(invalid(
                        line,
                        "request exit repeats or disagrees with actual committed steps",
                    ));
                }
                if matches!(reason, ExitReason::Completed)
                    && !state.progress.as_ref().is_some_and(|p| p.next_sigma == 0.)
                {
                    return Err(invalid(
                        line,
                        "completion exit lacks a terminal committed sigma",
                    ));
                }
                state.exited = true;
                let last = state.progress.as_ref().map(|p| p.wave);
                if let Some((wave, members)) = &self.last_wave {
                    if last == Some(*wave) {
                        // Only a real prior committed peer relationship marks
                        // a survivor; publication alone cannot prove refill.
                        for id in members {
                            if let Some(peer) = self.requests.get_mut(id).filter(|r| !r.exited) {
                                peer.refill_departure = true;
                            }
                        }
                    }
                }
                match reason {
                    ExitReason::Completed => self.facts.completed_exits += 1,
                    ExitReason::Cancelled => self.facts.cancelled_exits += 1,
                    ExitReason::Failed => self.facts.failed_exits += 1,
                }
            }
            // Service/phase timing is checked by the independent bounded
            // metadata analyzer; none of these observations proves a commit.
            _ => (),
        }
        Ok(())
    }
    fn check_members(
        &self,
        members: &[String],
        line: u64,
    ) -> Result<(), ImageContinuousTraceError> {
        if members.len() > self.limits.maximum_members {
            return Err(capacity(line, "cohort membership exceeds reader capacity"));
        }
        let declared = self.identity.as_ref().unwrap().pipeline["maximum_batch_size"]
            .as_u64()
            .unwrap();
        if members.len() as u64 > declared {
            return Err(invalid(
                line,
                "actual cohort exceeds the pipeline's declared batch capacity",
            ));
        }
        let mut unique = BTreeSet::new();
        for id in members {
            if !unique.insert(id) || self.requests.get(id).is_none_or(|r| r.exited) {
                return Err(invalid(
                    line,
                    "cohort repeats, resurrects or contains an uninitialized request",
                ));
            }
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn commit(
        &mut self,
        job: u64,
        plan: String,
        program: String,
        packed_count: usize,
        basis: Vec<f32>,
        members: Vec<ImageTraceMember>,
        sequence: u64,
        line: u64,
    ) -> Result<(), ImageContinuousTraceError> {
        let publication = self
            .publication
            .as_ref()
            .ok_or_else(|| invalid(line, "step committed without a published cohort"))?;
        if publication.job != job
            || publication.plan != plan
            || publication.program != program
            || packed_count != members.len()
            || packed_count == 0
            || publication.members.len() != members.len()
            || publication
                .members
                .iter()
                .zip(&members)
                .any(|(id, member)| id != &member.request)
        {
            return Err(invalid(
                line,
                "commit differs from published membership/order/plan identity",
            ));
        }
        self.check_members(&publication.members, line)?;
        if basis.is_empty() || basis.iter().any(|v| !v.is_finite()) {
            return Err(invalid(line, "committed time basis is empty or nonfinite"));
        }
        let mut cursor = 0u64;
        let mut fresh = false;
        let mut continuing_after_exit = false;
        for member in &members {
            if member.row_runs.len() > self.limits.maximum_row_runs_per_member {
                return Err(capacity(line, "row-run count exceeds reader capacity"));
            }
            if member.packed_row_start != cursor
                || member.packed_row_end <= cursor
                || member.row_runs.is_empty()
                || !member.sigma.is_finite()
                || !member.next_sigma.is_finite()
                || !member.scheduler_timestep.is_finite()
                || member.sigma > 1.
                || member.next_sigma < 0.
                || member.sigma <= member.next_sigma
            {
                return Err(invalid(
                    line,
                    "member rows or scheduler scalars are inconsistent",
                ));
            }
            let mut layout: Vec<(u32, u64)> = Vec::new();
            let mut previous = None;
            for run in &member.row_runs {
                if run.length == 0
                    || run.modality > 2
                    || run.time_index >= basis.len() as u64
                    || previous == Some((run.modality, run.time_index))
                {
                    return Err(invalid(
                        line,
                        "row run is empty, noncanonical or outside the actual time basis",
                    ));
                }
                previous = Some((run.modality, run.time_index));
                cursor = cursor
                    .checked_add(run.length)
                    .ok_or_else(|| invalid(line, "packed row extent overflows"))?;
                if let Some(last) = layout.last_mut().filter(|v| v.0 == run.modality) {
                    last.1 = last
                        .1
                        .checked_add(run.length)
                        .ok_or_else(|| invalid(line, "modality extent overflows"))?;
                } else {
                    layout.push((run.modality, run.length));
                }
            }
            if cursor != member.packed_row_end {
                return Err(invalid(
                    line,
                    "row runs do not cover the original packed interval",
                ));
            }
            let request = self.requests.get_mut(&member.request).unwrap();
            if let Some(previous) = &request.progress {
                if previous.step.checked_add(1) != Some(member.step_index)
                    || previous.next_sigma != member.sigma
                {
                    return Err(invalid(
                        line,
                        "request step/sigma does not continue its prior committed work",
                    ));
                }
                continuing_after_exit |= request.refill_departure;
            } else {
                if member.step_index != 0 {
                    return Err(invalid(line, "first committed request step is not zero"));
                }
                fresh = true;
            }
            if request.layout.as_ref().is_some_and(|old| old != &layout) {
                return Err(invalid(
                    line,
                    "the same request changed its original row/modality layout",
                ));
            }
            request.layout = Some(layout);
        }
        let joint = members.len() >= 2;
        let distinct_step = joint
            && members
                .iter()
                .any(|m| m.step_index != members[0].step_index);
        let distinct_time = joint
            && members
                .iter()
                .any(|m| m.scheduler_timestep != members[0].scheduler_timestep);
        let refill = joint && fresh && continuing_after_exit;
        self.facts.committed_waves += 1;
        self.facts.multi_request_committed_waves += u64::from(joint);
        self.facts.distinct_step_index_waves += u64::from(distinct_step);
        self.facts.distinct_scheduler_timestep_waves += u64::from(distinct_time);
        self.facts.nonuniform_time_basis_waves += u64::from(basis.iter().any(|&v| v != basis[0]));
        self.facts.refill_continuation_waves += u64::from(refill);
        let witness = || ImageTraceCommitWitness {
            sequence,
            cohort_job: job,
            time_basis: basis.clone(),
            members: members.clone(),
        };
        if joint && self.facts.first_joint_commit.is_none() {
            self.facts.first_joint_commit = Some(witness());
        }
        if (distinct_step || distinct_time) && self.facts.first_mixed_progress_commit.is_none() {
            self.facts.first_mixed_progress_commit = Some(witness());
        }
        if refill && self.facts.first_refill_commit.is_none() {
            self.facts.first_refill_commit = Some(witness());
        }
        for member in &members {
            let request = self.requests.get_mut(&member.request).unwrap();
            request.progress = Some(Progress {
                step: member.step_index,
                next_sigma: member.next_sigma,
                wave: sequence,
            });
            if refill {
                request.refill_departure = false;
            }
        }
        self.last_wave = Some((sequence, members.into_iter().map(|m| m.request).collect()));
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    pub(super) fn finish(
        mut self,
        session: Option<String>,
        records: u64,
        event_bytes: u64,
        input_bytes: u64,
        footer: Option<ImageTraceFooter>,
        mut incomplete: Vec<ImageTraceIncompleteReason>,
        line: u64,
    ) -> Result<ImageContinuousTraceAnalysis, ImageContinuousTraceError> {
        if self.identity.is_none() {
            incomplete.push(ImageTraceIncompleteReason::MissingHeader);
        }
        if let Some(footer) = &footer {
            if footer.dropped > 0 && !footer.truncated {
                return Err(invalid(
                    line,
                    "footer lost records without its sticky truncation flag",
                ));
            }
            if !footer.sealed {
                incomplete.push(ImageTraceIncompleteReason::Unsealed);
            }
            if footer.truncated {
                incomplete.push(ImageTraceIncompleteReason::ProducerTruncated);
            }
            if footer.dropped > 0 {
                incomplete.push(ImageTraceIncompleteReason::ProducerDroppedRecords);
            }
        } else {
            incomplete.push(ImageTraceIncompleteReason::MissingFooter);
        }
        self.facts.requests_without_exit = self.requests.values().filter(|r| !r.exited).count();
        if self.facts.requests_without_exit > 0 {
            incomplete.push(ImageTraceIncompleteReason::RequestsWithoutExit);
        }
        Ok(ImageContinuousTraceAnalysis {
            status: if incomplete.is_empty() {
                ImageStepEvidenceStatus::Observed
            } else {
                ImageStepEvidenceStatus::NotEvaluated
            },
            schema_version: 1,
            session_uuid: session,
            identity: self.identity,
            records,
            event_bytes,
            input_bytes,
            footer,
            facts: self.facts,
            incomplete_reasons: incomplete,
            timing: ImageTraceTimingAnalysis::default(),
            prompt_expansion_observations: self.expansion_observations,
            prompt_expansion_token_waves: self.token_waves.finish(),
            reference_latents: self.reference_latents,
        })
    }
}
fn invalid(line: u64, reason: &str) -> ImageContinuousTraceError {
    error(line, ImageContinuousTraceErrorKind::Invalid, reason)
}
fn capacity(line: u64, reason: &str) -> ImageContinuousTraceError {
    error(line, ImageContinuousTraceErrorKind::Capacity, reason)
}
