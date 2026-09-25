use super::*;
use ferrum_engine::continuous_engine::{
    CalibrationAction, CalibrationSession, CalibrationSubmissionState, CalibrationTurn,
    CalibrationWork,
};
use ferrum_interfaces::{
    output_flow::{CreditedOutputSession, OutputCompletion},
    InferenceRequestContext,
};
use futures::StreamExt;
use sha2::{Digest, Sha256};
use std::sync::Arc;
use tokio::task::JoinSet;

mod admission;
mod wave_plan;
use admission::AdmissionWindow;

pub(super) async fn collect(
    session: &mut CalibrationSession,
    manifest: &manifest::Manifest,
    inputs: &inputs::PreparedInputs,
    artifacts: &mut report::Artifacts,
    summary: &mut report::Summary,
) -> Result<()> {
    artifacts.set_structured_capture(
        session
            .configuration()
            .scheduler
            .slo
            .cost_observation
            .structured_capture,
    );
    if manifest.validation_model.is_discovery_v2() {
        return structured_v2::discovery::collect(session, manifest, inputs, artifacts, summary)
            .await;
    }
    if manifest.validation_model.structured_v2().is_some() {
        return structured_v2::collect(session, manifest, inputs, artifacts, summary).await;
    }
    if manifest.validation_model.structured().is_some() {
        return structured::collect(session, manifest, inputs, artifacts, summary).await;
    }
    if manifest.reference.is_some() {
        summary.reference_input_identities =
            Some(reference::InputIdentityLedger::new(inputs.count(manifest))?);
    }
    let reference = reference::collect(session, manifest, inputs, artifacts, summary).await?;
    if let Some((_, export, _)) = manifest.validation_model.selected() {
        let protocol = serde_json::to_vec(manifest).map_err(|error| {
            FerrumError::config(format!("encode calibration protocol: {error}"))
        })?;
        session.begin_selected_cost_calibration(
            ferrum_engine::continuous_engine::SelectedCalibrationOptions {
                export: export.clone(),
                protocol_sha256: Sha256::digest(protocol).into(),
            },
        )?;
    }
    for (index, case) in manifest.training.iter().enumerate() {
        for repetition in 0..case.repetitions.get() {
            cohort(
                session,
                manifest,
                inputs,
                case,
                report::Phase::Training,
                index,
                repetition,
                None,
                None,
                artifacts,
                summary,
            )
            .await?;
        }
    }
    let frozen = if let Some((_, _, residual)) = manifest.validation_model.selected() {
        let receipt = session.freeze_selected_cost_fit().await?;
        summary.selected_fit_freeze = Some(
            serde_json::to_value(&receipt)
                .map_err(|error| FerrumError::internal(format!("encode fit freeze: {error}")))?,
        );
        artifacts.record(&serde_json::json!({"schema_version":1,"event":"selected_fit_frozen","receipt":summary.selected_fit_freeze}))?;
        for (index, case) in residual.iter().enumerate() {
            for repetition in 0..case.repetitions.get() {
                cohort(
                    session,
                    manifest,
                    inputs,
                    case,
                    report::Phase::Residual,
                    index,
                    repetition,
                    None,
                    None,
                    artifacts,
                    summary,
                )
                .await?;
            }
        }
        validation::ValidationModel::SelectedWholeWave(
            session.finish_selected_cost_residual().await?,
            manifest
                .validation_model
                .selected_kind()
                .expect("selected above"),
        )
    } else {
        validation::ValidationModel::prepare(session, &manifest.validation_model).await?
    };
    summary.validation_model = Some(frozen.receipt()?);
    artifacts.record(
        &serde_json::json!({"schema_version":1,"event":"validation_model", "receipt":summary.validation_model}),
    )?;
    if let Some(reference) = reference {
        let cut = frozen.artifact().ok_or_else(|| {
            FerrumError::internal("reference phases require the actual exported training cut")
        })?;
        summary.reference = Some(reference.finish(cut)?);
        artifacts.record(&serde_json::json!({"schema_version":1,"event":"reference_artifact", "receipt":summary.reference}))?;
    }
    for (index, case) in manifest.validation.iter().enumerate() {
        for repetition in 0..case.repetitions.get() {
            cohort(
                session,
                manifest,
                inputs,
                case,
                report::Phase::Heldout,
                index,
                repetition,
                Some(&frozen),
                None,
                artifacts,
                summary,
            )
            .await?;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub(super) async fn cohort(
    session: &mut CalibrationSession,
    manifest: &manifest::Manifest,
    inputs: &inputs::PreparedInputs,
    case: &manifest::Cohort,
    phase: report::Phase,
    index: usize,
    repetition: usize,
    frozen: Option<&validation::ValidationModel>,
    mut observer: Option<&mut dyn reference::CohortObserver>,
    artifacts: &mut report::Artifacts,
    totals: &mut report::Summary,
) -> Result<()> {
    if case.token_policy_residency == manifest::TokenPolicyResidencyPolicy::InvalidateBeforeCohort {
        // The previous call drained every output consumer and its completion.
        // The exclusive session and executor still independently verify that
        // no live owner, pending publication or physical work remains.
        let receipt = session.invalidate_token_policy_residency().await;
        totals.token_policy_residency_attempts += 1;
        if let ferrum_interfaces::model_executor::TokenPolicyResidencyInvalidation::Cleared {
            cleared_entries,
        } = receipt
        {
            totals.token_policy_residency_invalidations += 1;
            totals.token_policy_residency_cleared_entries = totals
                .token_policy_residency_cleared_entries
                .checked_add(
                    u64::try_from(cleared_entries).map_err(|_| {
                        FerrumError::internal("token-policy residency count overflow")
                    })?,
                )
                .ok_or_else(|| FerrumError::internal("token-policy residency total overflow"))?;
        }
        artifacts.record(&serde_json::json!({
            "schema_version": 1,
            "event": "token_policy_residency_invalidation",
            "phase": phase, "case": index, "repetition": repetition,
            "receipt": receipt,
        }))?;
        match receipt {
            ferrum_interfaces::model_executor::TokenPolicyResidencyInvalidation::Cleared {
                ..
            } => {}
            other => {
                return Err(FerrumError::invalid_request(format!(
                    "declared token-policy residency invalidation did not complete: {other:?}"
                )))
            }
        }
    }
    // Each invocation is one phase/case/repetition; no cursor crosses its drain.
    let mut wave_plan = wave_plan::Cursor::new(case)?;
    let mut consumers = JoinSet::<Result<(ferrum_types::RequestId, serde_json::Value)>>::new();
    let mut window =
        AdmissionWindow::<ferrum_types::RequestId>::new(&case.prompts, case.maximum_in_flight())?;
    while !window.drained() {
        // Completion means both terminal wire and the retained completion lease
        // were consumed successfully. GPU completion or an early terminal
        // frontier alone cannot open an arrival slot.
        while let Some(joined) = consumers.try_join_next() {
            let (id, output) = joined.map_err(|error| {
                FerrumError::internal(format!("calibration output task: {error}"))
            })??;
            artifacts.record(&output)?;
            let completed = window.completed(&id)?;
            if case.rolling_window.is_some() {
                artifacts.record(&serde_json::json!({
                    "schema_version": 1, "event": "rolling_completion",
                    "phase": phase, "case": index, "repetition": repetition,
                    "request_id": id, "cohort_request_ordinal": completed.ordinal,
                    "in_flight": window.active().len(),
                }))?;
            }
        }
        // Initial fill and later refill share the exact same product request
        // conversion and add_request path. No sample or output budget changes.
        while let Some((ordinal, prompt_index)) = window.next_prompt() {
            let context = InferenceRequestContext::capture();
            let (request, input_evidence) = inputs.request_for_phase(
                manifest,
                prompt_index,
                &session.configuration().model.model_id,
                session
                    .configuration()
                    .sampling
                    .default_params
                    .model_output_protocol,
                phase,
            )?;
            let maximum_output_tokens = request.sampling_params.max_tokens;
            let id = request.id.clone();
            let contract = inputs.output_contract(manifest.protocol.output, &request);
            let output = session
                .add_request(request, context, Arc::new(contract))
                .await?;
            window.admitted(ordinal, id.clone())?;
            artifacts.record(
                &serde_json::json!({"schema_version":1,"event":"request","phase":phase,
                "case":index,"repetition":repetition,"request_id":id,"prompt_index":prompt_index,
                "input":input_evidence,"maximum_output_tokens":maximum_output_tokens}),
            )?;
            if case.rolling_window.is_some() {
                artifacts.record(&serde_json::json!({
                    "schema_version": 1, "event": "rolling_admission",
                    "phase": phase, "case": index, "repetition": repetition,
                    "request_id": id, "cohort_request_ordinal": ordinal,
                    "prompt_index": prompt_index, "in_flight": window.active().len(),
                    "maximum_in_flight": case.maximum_in_flight(),
                }))?;
            }
            consumers.spawn(async move {
                let record = consume(id.clone(), output).await?;
                Ok::<_, FerrumError>((id, record))
            });
            totals.request(phase);
        }
        if window.drained() {
            break;
        }
        // A guarded wave can return a one-use backing-maintenance ticket.
        // Admission probes do not consume execution tickets: reconcile this
        // separate turn before asking the exact cohort to execute again.
        let maintenance = session.step(CalibrationAction::Maintenance).await?;
        if matches!(maintenance, CalibrationTurn::MaintenanceReconciled) {
            artifacts.record(
                &serde_json::json!({"schema_version":1,"event":"execution_maintenance",
                "phase":phase,"case":index,"repetition":repetition,
                "outcome":"reconciled_before_exact_wave_retry"}),
            )?;
            continue;
        }
        if !matches!(maintenance, CalibrationTurn::Blocked(_)) {
            return Err(FerrumError::internal(
                "calibration maintenance returned an unexpected pending wave",
            ));
        }
        let admission = session.step(CalibrationAction::AdmitOne).await?;
        if !matches!(admission, CalibrationTurn::Blocked(_)) {
            artifacts.record(&serde_json::json!({"schema_version":1,"event":"admission_or_maintenance",
                "phase":phase,"case":index,"repetition":repetition,"outcome":format!("{admission:?}")}))?;
        }
        let frontiers = session.frontiers()?;
        for frontier in &frontiers {
            let Some(owner) = window.owner_mut(frontier.request_id()) else {
                continue;
            };
            if !owner.evidenced {
                owner.evidenced = true;
                if let Some(identities) = &mut totals.reference_input_identities {
                    let source_index = owner.source_index;
                    let independent = manifest
                        .reference
                        .as_ref()
                        .is_some_and(|config| config.request_policy.independent(phase));
                    identities.observe(source_index, *frontier.request_evidence(), independent)?;
                }
                if let Some(observer) = observer.as_deref_mut() {
                    observer.initial_frontier(session, frontier)?;
                }
                artifacts.record(&serde_json::json!({"schema_version":1,"event":"request_evidence",
                    "phase":phase,"case":index,"repetition":repetition,"request_id":frontier.request_id(),
                    "owner_incarnation":frontier.owner_incarnation(),"work_generation":frontier.work_generation(),
                    "input":frontier.request_evidence()}))?;
            }
        }
        let ordered = window
            .active()
            .iter()
            .filter_map(|owner| frontiers.iter().find(|row| row.request_id() == &owner.id));
        let choice = wave_plan.as_ref().map(wave_plan::Cursor::choice);
        let prefill_chunk = choice.map_or(case.prefill_chunk_tokens, |value| {
            value.prefill_chunk_tokens
        });
        let decode_route = choice.map_or(case.decode_route, |value| value.decode_route);
        let mut prefills = Vec::new();
        let mut decodes = Vec::new();
        for frontier in ordered {
            if let Some((offset, total)) = frontier.prefill_progress() {
                let count = super::reference::prefill_count(
                    manifest.reference.as_ref(),
                    phase,
                    u32::try_from(total)
                        .map_err(|_| FerrumError::invalid_request("calibration input overflow"))?,
                    u32::try_from(offset)
                        .map_err(|_| FerrumError::invalid_request("calibration offset overflow"))?,
                    prefill_chunk,
                )?;
                prefills.push(frontier.prefill_work(count)?);
            } else {
                decodes.push(frontier.decode_work_with_route(decode_route)?);
            }
        }
        let rows = select_rows(prefills, decodes, case.execution);
        if !rows.is_empty() {
            let reference_rows = observer.as_ref().map(|_| rows.clone());
            let planned = wave_plan
                .as_ref()
                .map(|plan| plan.begin(&rows))
                .transpose()?;
            totals.wave_attempts = totals
                .wave_attempts
                .checked_add(1)
                .filter(|count| *count <= manifest.protocol.maximum_wave_attempts.get())
                .ok_or_else(|| {
                    FerrumError::resource_exhausted("calibration wave-attempt limit exceeded")
                })?;
            totals.phases.get_mut(phase).wave_attempts += 1;
            if let Some(attempt) = &planned {
                artifacts.record(
                    &serde_json::json!({"schema_version":1,"event":"wave_plan_attempt",
                    "phase":phase,"case":index,"repetition":repetition,
                    "wave_attempt":totals.wave_attempts,"plan":attempt}),
                )?;
            }
            match session.step(CalibrationAction::Wave(rows)).await? {
                CalibrationTurn::Wave(wave) | CalibrationTurn::Reaped(wave) => {
                    artifacts.wave(phase, index, repetition, &wave, frozen, totals)?;
                    if let (Some(cursor), Some(attempt)) = (wave_plan.as_mut(), planned.as_ref()) {
                        let outcome = cursor.reconcile(attempt, &wave);
                        artifacts.record(&serde_json::json!({"schema_version":1,"event":"wave_plan_result",
                            "phase":phase,"case":index,"repetition":repetition,
                            "wave_attempt":totals.wave_attempts,"selection":attempt.choice,
                            "submission":format!("{:?}",wave.submission),
                            "host_call_id":wave.host_stages.as_ref().map(|host| host.call_id),
                            "advanced":outcome.as_ref().ok().copied(),
                            "error":outcome.as_ref().err().map(ToString::to_string),
                            "next":cursor.choice()}))?;
                        outcome?;
                    }
                    if let Some(observer) = observer.as_deref_mut() {
                        let progress = observer.observe_wave(session, reference_rows.as_deref().ok_or_else(|| FerrumError::internal("reference work correlation missing"))?, &wave)?;
                        totals.reference_progress(phase, progress)?;
                        artifacts.record(&serde_json::json!({"schema_version":1,"event":"reference_wave_role","phase":phase,"case":index,"repetition":repetition,"role":progress}))?;
                    }
                    if wave.submission == CalibrationSubmissionState::InFlightUnknown {
                        return Err(FerrumError::backend("calibration submission is indeterminate; refusing replay"));
                    }
                    if let Some(error) = wave.error { return Err(error); }
                }
                other => artifacts.record(&serde_json::json!({"schema_version":1,"event":"wave_blocked",
                    "phase":phase,"case":index,"repetition":repetition,"outcome":format!("{other:?}")}))?,
            }
        }
        // Output consumers release real credits. Bounded retries never spin or
        // shorten max_tokens to force a particular calibration shape.
        tokio::time::sleep(std::time::Duration::from_millis(1)).await;
    }
    if case.rolling_window.is_some() {
        artifacts.record(&serde_json::json!({
            "schema_version": 1, "event": "rolling_cohort_drained",
            "phase": phase, "case": index, "repetition": repetition,
            "completed_requests": case.prompts.len(), "in_flight": 0,
        }))?;
    }
    Ok(())
}

fn select_rows(
    mut prefills: Vec<CalibrationWork>,
    decodes: Vec<CalibrationWork>,
    execution: manifest::Execution,
) -> Vec<CalibrationWork> {
    match execution {
        manifest::Execution::Split if !prefills.is_empty() => prefills,
        manifest::Execution::Split => decodes,
        manifest::Execution::Mixed => {
            prefills.extend(decodes);
            prefills
        }
    }
}

async fn consume(
    id: ferrum_types::RequestId,
    session: CreditedOutputSession,
) -> Result<serde_json::Value> {
    let CreditedOutputSession {
        mut frames,
        completion,
    } = session;
    let mut frames_seen = 0u64;
    let mut bytes_seen = 0u64;
    let mut hash = Sha256::new();
    let mut terminal_seen = false;
    while let Some(frame) = frames.next().await {
        let metadata = frame.metadata();
        let wire = frame.wire();
        if wire.credit().events == 0 || wire.payload().capacity() > wire.credit().bytes {
            return Err(FerrumError::internal(
                "calibration frame exceeded its output lease",
            ));
        }
        hash.update(wire.payload());
        frames_seen = frames_seen
            .checked_add(1)
            .ok_or_else(|| FerrumError::internal("frame count overflow"))?;
        bytes_seen = bytes_seen
            .checked_add(wire.payload().len() as u64)
            .ok_or_else(|| FerrumError::internal("output byte count overflow"))?;
        drop(frame);
        if metadata.terminal {
            terminal_seen = true;
            break;
        }
    }
    if !terminal_seen {
        return Err(FerrumError::backend(
            "calibration output closed without terminal frame",
        ));
    }
    let completion = completion
        .await
        .map_err(|_| FerrumError::backend("calibration output owner lost its completion"))?;
    match completion.payload() {
        OutputCompletion::Succeeded { usage, reason, .. } => {
            Ok(serde_json::json!({"schema_version":1,
            "event":"request_completed","request_id":id,"usage":usage,"finish_reason":format!("{reason:?}"),
            "wire_frames":frames_seen,"wire_bytes":bytes_seen,"wire_sha256":format!("{:x}",hash.finalize())}))
        }
        OutputCompletion::Failed(error) => Err(FerrumError::backend(format!(
            "calibration request failed: {}",
            error.message()
        ))),
    }
}
