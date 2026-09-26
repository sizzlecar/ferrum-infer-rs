use super::super::super::cost_observation::{HostStageCompleteness, HostStageQueueDisposition};
use super::*;
use ferrum_interfaces::execution_cost::HostContentDomainV1;
use sha2::{Digest, Sha256};
use std::num::NonZeroUsize;

impl CalibrationSession {
    /// Preparation-only API. No ordinary request metadata can install this
    /// capability; source3/source4 collectors cannot consume this session.
    pub async fn add_request_with_prefix_preparation(
        &mut self,
        request: ferrum_types::InferenceRequest,
        context: InferenceRequestContext,
        contract: Arc<OutputProjectionContract>,
        declaration: CalibrationPrefixTokensV1,
    ) -> Result<CreditedOutputSession> {
        let config = &self.configuration().scheduler.slo.cost_observation;
        if self.pending.is_some()
            || self.indeterminate
            || !self.engine.inner.manual_calibration_driver
            || self.engine.inner.bg_loop_spawned.load(Ordering::Acquire)
            || self.engine.inner.is_running.load(Ordering::Acquire)
            || self.selected_capture_identity.is_some()
            || self.structured_capture.is_some()
            || self.structured_capture_v2.is_some()
            || self.structured_group_v2.is_some()
            || config.predictor != ferrum_types::SloCostPredictor::StructuredWholeWaveV2
            || !config.selected_feedback.is_disabled()
            || !config.structured_feedback.is_disabled()
            || config.profile_export.is_some()
            || matches!(
                config.model.feature_model,
                ferrum_types::SloCostFeatureModel::EmpiricalHostContentV1 { .. }
                    | ferrum_types::SloCostFeatureModel::EmpiricalRowMultisetV2 { .. }
                    | ferrum_types::SloCostFeatureModel::EmpiricalPromptRangeV3 { .. }
            )
        {
            return Err(invalid("prefix requires isolated manual Structured V2 session without old collectors, exporters or feedback"));
        }
        if declaration.token_ids.len() > self.context_capacity() {
            return Err(invalid("prefix exceeds actual engine context capacity"));
        }
        let maximum = NonZeroUsize::new(request.sampling_params.max_tokens)
            .ok_or_else(|| invalid("prefix request has no output budget"))?;
        let plan =
            declaration.validate_for_tokenizer(self.engine.inner.tokenizer.as_ref(), maximum)?;
        if let Some(run) = &self.prefix_preparation {
            if run.failure.is_some()
                || run.pending_wave.is_some()
                || run.pending_offer.is_some()
                || run.records.len() >= self.limits.maximum_requests().get()
                || run.records.contains_key(&request.id)
                || run.records.values().any(|r| r.last_call != 0)
            {
                return Err(invalid(
                    "all prefix requests must be declared before the first physical wave",
                ));
            }
        } else {
            if !self.frontiers()?.is_empty()
                || self.engine.inner.scheduler.active_count() != 0
                || self.engine.inner.scheduler.waiting_count() != 0
            {
                return Err(invalid(
                    "prefix declaration cannot attach to existing owners",
                ));
            }
            let cut = self.freeze_cost_model().await?.accepted_ordinal();
            self.prefix_preparation = Some(PrefixPreparationRun {
                last_fifo: cut,
                ..Default::default()
            });
        }
        let id = request.id.clone();
        // No scheduler task exists in this private session: ensure_bg_loop
        // returns at manual_calibration_driver. After this await there is no
        // yield before binding the actual newly initialized owner below.
        let output = self.add_request_inner(request, context, contract).await?;
        let bind = (|| {
            let mut sequences = self.engine.inner.sequences.write();
            let sequence = sequences
                .get_mut(&id)
                .ok_or_else(|| invalid("prefix admitted owner missing"))?;
            let frontier = sequence
                .cost_frontier
                .ok_or_else(|| invalid("prefix real owner unavailable"))?;
            let normal_policy = sequence
                .cost_policy_signature
                .ok_or_else(|| invalid("prefix ordinary host policy unavailable"))?;
            let normal_numeric = sequence
                .cost_numeric_policy
                .ok_or_else(|| invalid("prefix ordinary numeric host policy unavailable"))?;
            if sequence.calibration_prefix.is_some()
                || !sequence.generated_tokens.is_empty()
                || sequence.model_kv.is_some()
                || sequence.prefill_tokens_processed != 0
                || normal_numeric.empirical_content_domain
                    != Some(HostContentDomainV1::PlainTextGreedyV1)
                || sequence.sampling_params.max_tokens != maximum.get()
                || sequence.credited_output.is_none()
            {
                return Err(invalid(
                    "prefix requires fresh ordinary plain-text Length owner",
                ));
            }
            sequence.calibration_prefix = Some(InstalledPrefix {
                plan,
                owner: frontier.owner_incarnation.get(),
                original_policy: normal_policy,
                original_numeric: normal_numeric,
                pending_commit: None,
            });
            self.engine.inner.refresh_sequence_cost_policy(sequence);
            if sequence
                .cost_numeric_policy
                .is_none_or(|p| p.empirical_content_domain.is_some())
                || sequence.cost_policy_signature == Some(normal_policy)
            {
                return Err(invalid(
                    "prefix failed to distinguish intervention authority",
                ));
            }
            Ok(frontier.owner_incarnation.get())
        })();
        let owner = match bind {
            Ok(owner) => owner,
            Err(error) => {
                self.prefix_preparation
                    .as_mut()
                    .expect("declared prefix run")
                    .failure = Some(error.to_string());
                // Drop cancels only this real credited request; it cannot
                // issue a model wave or turn failure into valid population.
                drop(output);
                return Err(error);
            }
        };
        self.prefix_preparation
            .as_mut()
            .expect("declared prefix run")
            .records
            .insert(
                id,
                RequestRecord {
                    owner,
                    maximum_output: maximum.get(),
                    declaration,
                    released: None,
                    completed_length: false,
                    last_call: 0,
                    last_fifo: 0,
                },
            );
        Ok(output)
    }

    pub(in crate::continuous_engine::inner::calibration) fn check_prefix_wave(
        &self,
        rows: &[CalibrationWork],
    ) -> Result<()> {
        let Some(run) = &self.prefix_preparation else {
            return Ok(());
        };
        if run.failure.is_some() || run.pending_wave.is_some() || run.pending_offer.is_some() {
            return Err(invalid(
                "prefix run failed or its previous wave evidence is untaken",
            ));
        }
        let sequences = self.engine.inner.sequences.read();
        for row in rows {
            let id = row.frontier.request_id();
            let record = run
                .records
                .get(id)
                .ok_or_else(|| invalid("undeclared prefix owner in wave"))?;
            let sequence = sequences
                .get(id)
                .ok_or_else(|| invalid("prefix live owner absent"))?;
            if record.owner != row.frontier.owner_incarnation().get()
                || record.completed_length
                || (record.released.is_none()
                    && sequence.generated_tokens.len() >= record.declaration.release_generated)
                || row.decode_route != CalibrationDecodeRoute::Actual
            {
                return Err(invalid(
                    "prefix wave must use original route and release at its fixed frontier",
                ));
            }
        }
        Ok(())
    }

    pub(in crate::continuous_engine::inner::calibration) fn offer_prefix_wave(
        &mut self,
        rows: &[CalibrationWork],
    ) -> Result<()> {
        if self.prefix_preparation.is_none() {
            return Ok(());
        }
        let sequences = self.engine.inner.sequences.read();
        let before = rows
            .iter()
            .map(|row| {
                sequences
                    .get(row.frontier.request_id())
                    .ok_or_else(|| invalid("prefix owner missing at publication"))
                    .and_then(PrefixFrontierV1::capture)
            })
            .collect::<Result<Vec<_>>>()?;
        self.prefix_preparation
            .as_mut()
            .expect("prefix run")
            .pending_offer = Some(before);
        Ok(())
    }

    pub(in crate::continuous_engine::inner::calibration) fn record_prefix_wave(
        &mut self,
        report: &CalibrationWaveReport,
    ) {
        let Some(run) = &mut self.prefix_preparation else {
            return;
        };
        let Some(before) = run.pending_offer.take() else {
            run.failure
                .get_or_insert_with(|| "prefix wave has no original offer".into());
            return;
        };
        let mut sequences = self.engine.inner.sequences.write();
        let rows = before
            .into_iter()
            .map(|before| {
                let sequence = sequences.get_mut(&before.request_id);
                let (after, preparation_commit) = match sequence {
                    Some(sequence) => (
                        PrefixFrontierV1::capture(sequence).ok(),
                        sequence
                            .calibration_prefix
                            .as_mut()
                            .and_then(|p| p.pending_commit.take()),
                    ),
                    None => (None, None),
                };
                PrefixRowEvidenceV1 {
                    before,
                    after,
                    preparation_commit,
                }
            })
            .collect::<Vec<_>>();
        let joined = (|| -> Result<()> {
            if report.submission != CalibrationSubmissionState::HostReconciled
                || report.error.is_some()
            {
                return Err(invalid("prefix actual wave did not reconcile successfully"));
            }
            let stages = report
                .host_stages
                .as_ref()
                .ok_or_else(|| invalid("prefix actual host settlement absent"))?;
            let queue = report
                .host_stage_queue
                .ok_or_else(|| invalid("prefix original FIFO receipt absent"))?;
            let fifo = queue
                .accepted_ordinal
                .ok_or_else(|| invalid("prefix wave was not accepted by original FIFO"))?;
            if stages.completeness != HostStageCompleteness::CompleteSingleWave
                || stages.call_id <= run.last_call
                || Some(fifo) != run.last_fifo.checked_add(1)
                || queue.disposition != HostStageQueueDisposition::Published
                || stages.rows.len() != rows.len()
            {
                return Err(invalid(
                    "prefix actual settlement/FIFO coverage is incomplete",
                ));
            }
            for row in &rows {
                let record = run
                    .records
                    .get_mut(&row.before.request_id)
                    .ok_or_else(|| invalid("prefix unknown owner"))?;
                let stage = stages
                    .rows
                    .iter()
                    .find(|s| {
                        s.request_id == row.before.request_id
                            && s.owner_incarnation == row.before.owner_incarnation
                            && s.work_generation == row.before.work_generation
                    })
                    .ok_or_else(|| invalid("prefix settlement owner/frontier mismatch"))?;
                if stage.completeness != HostStageCompleteness::CompleteSingleWave {
                    return Err(invalid("prefix row settlement incomplete"));
                }
                let generated_after = match (&row.after, &stage.terminal) {
                    (Some(after), None) if after.owner_incarnation == record.owner => {
                        after.generated_tokens
                    }
                    (None, Some(terminal))
                        if terminal.finish_reason == ferrum_types::FinishReason::Length
                            && terminal.generated_tokens == record.maximum_output as u64
                            && terminal.terminal_handoff_succeeded
                            && terminal.owner_matched
                            && !terminal.output_failed
                            && !terminal.physical_failed
                            && !terminal.scheduler_failed
                            && record.released.is_some() =>
                    {
                        record.completed_length = true;
                        record.maximum_output
                    }
                    _ => {
                        return Err(invalid(
                            "prefix terminal/frontier differs from complete Length population",
                        ))
                    }
                };
                if let Some(commit) = &row.preparation_commit {
                    if record.released.is_some()
                        || commit.owner_incarnation != record.owner
                        || commit.work_generation != row.before.work_generation
                        || commit.generated_before != row.before.generated_tokens
                        || commit.generated_after != generated_after
                        || commit.request_id != row.before.request_id
                        || commit.pending_before != row.before.pending_utf8
                        || row
                            .after
                            .as_ref()
                            .is_none_or(|after| commit.pending_after != after.pending_utf8)
                        || generated_after != row.before.generated_tokens + 1
                        || record.declaration.token_ids.get(commit.generated_before)
                            != Some(&commit.committed_token)
                    {
                        return Err(invalid("prefix candidate/commit does not bind actual wave"));
                    }
                } else if record.released.is_none()
                    && generated_after != row.before.generated_tokens
                {
                    return Err(invalid(
                        "prefix generation advanced without actual intervention receipt",
                    ));
                }
                record.last_call = stages.call_id;
                record.last_fifo = fifo;
            }
            run.last_call = stages.call_id;
            run.last_fifo = fifo;
            Ok(())
        })();
        let chain_error = joined.err().map(|e| e.to_string());
        if let Some(error) = &chain_error {
            run.failure.get_or_insert_with(|| error.clone());
        }
        run.pending_wave = Some(PrefixWaveEvidenceV1 {
            rows,
            submission: report.submission,
            error: report.error.as_ref().map(ToString::to_string),
            host_stages: report.host_stages.clone(),
            host_stage_queue: report.host_stage_queue,
            actual_evidence_diagnostic: report.actual_evidence_diagnostic.clone(),
            chain_error,
        });
    }

    pub fn release_prefix_preparation(&mut self, id: &RequestId) -> Result<PrefixReleasedV1> {
        if self.pending.is_some() || self.indeterminate {
            return Err(invalid("prefix release requires a reconciled call"));
        }
        let run = self
            .prefix_preparation
            .as_mut()
            .ok_or_else(|| invalid("prefix run absent"))?;
        if run.failure.is_some() || run.pending_wave.is_some() || run.pending_offer.is_some() {
            return Err(invalid(
                "prefix release requires complete consumed wave evidence",
            ));
        }
        let record = run
            .records
            .get_mut(id)
            .ok_or_else(|| invalid("prefix request not declared"))?;
        if record.released.is_some() || record.last_call == 0 {
            return Err(invalid("prefix release is absent or duplicated"));
        }
        let mut sequences = self.engine.inner.sequences.write();
        let sequence = sequences
            .get_mut(id)
            .ok_or_else(|| invalid("prefix owner terminated before release"))?;
        let installed = sequence
            .calibration_prefix
            .as_ref()
            .ok_or_else(|| invalid("prefix capability absent"))?;
        let frontier = PrefixFrontierV1::capture(sequence)?;
        let output = sequence
            .credited_output
            .as_ref()
            .ok_or_else(|| invalid("prefix credited output absent"))?;
        let applied = output
            .port
            .applied_ordinal_while_ready()
            .ok_or_else(|| invalid("prefix output actor has not published its next real credit"))?;
        if installed.owner != frontier.owner_incarnation
            || frontier.generated_tokens != record.declaration.release_generated
            || sequence.generated_tokens != record.declaration.token_ids
            || sequence.pending_decoded_utf8_bytes != installed.plan.expected_pending_bytes()
            || installed.pending_commit.is_some()
            || applied != output.accepted_ordinal
            || output.failure.is_some()
            || output.grant.is_some()
        {
            return Err(invalid(
                "prefix release actual state/credit differs from fixed declaration",
            ));
        }
        let mut digest = Sha256::new();
        digest.update(b"ferrum.calibration.generated-prefix.v1\0");
        for token in &sequence.generated_tokens {
            digest.update(token.get().to_le_bytes());
        }
        let installed = sequence
            .calibration_prefix
            .take()
            .expect("checked private capability");
        self.engine.inner.refresh_sequence_cost_policy(sequence);
        if sequence.cost_policy_signature != Some(installed.original_policy)
            || sequence.cost_numeric_policy != Some(installed.original_numeric)
        {
            sequence.calibration_prefix = Some(installed);
            self.engine.inner.refresh_sequence_cost_policy(sequence);
            run.failure = Some("ordinary host policy drifted at prefix release".into());
            return Err(invalid("ordinary host policy drifted at prefix release"));
        }
        let released = PrefixReleasedV1 {
            frontier,
            original_policy_signature: installed.original_policy,
            original_numeric_policy: installed.original_numeric,
            generated_prefix_sha256: digest.finalize().into(),
            through_call_id: record.last_call,
            through_fifo_ordinal: record.last_fifo,
            actor_applied_output_ordinal: applied,
        };
        record.released = Some(released.clone());
        Ok(released)
    }

    /// Verifies Stage A's entire real request lifecycle. This is not a source5
    /// artifact and cannot be passed to an existing calibration model loader.
    pub async fn finish_prefix_preparation_run(&mut self) -> Result<()> {
        let run = self
            .prefix_preparation
            .as_ref()
            .ok_or_else(|| invalid("prefix run absent"))?;
        if self.pending.is_some()
            || self.indeterminate
            || run.failure.is_some()
            || run.pending_wave.is_some()
            || run.pending_offer.is_some()
            || run.records.is_empty()
            || run
                .records
                .values()
                .any(|r| r.released.is_none() || !r.completed_length)
            || !self.frontiers()?.is_empty()
            || self.engine.inner.scheduler.active_count() != 0
            || self.engine.inner.scheduler.waiting_count() != 0
        {
            return Err(invalid(
                "prefix run has incomplete preparation, suffix, Length or evidence",
            ));
        }
        let final_fifo = run.last_fifo;
        if self.freeze_cost_model().await?.accepted_ordinal() != final_fifo {
            return Err(invalid(
                "prefix final FIFO barrier differs from observed complete population",
            ));
        }
        Ok(())
    }
}
