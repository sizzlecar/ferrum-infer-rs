//! EngineInner iteration and request-processing implementation.

use super::*;

mod batch;
mod completion;
mod decode;
mod prefill;

pub(super) fn is_resource_exhausted_error(error: &FerrumError) -> bool {
    matches!(error, FerrumError::ResourceExhausted { .. })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct PagedKvAdmissionPressure {
    pub(super) admission_blocks: usize,
    pub(super) immediate_blocks: usize,
    pub(super) free_blocks: usize,
}

pub(super) fn paged_kv_admission_pressure(error: &FerrumError) -> Option<PagedKvAdmissionPressure> {
    let FerrumError::ResourceExhausted { message } = error else {
        return None;
    };
    let (_, after_need) = message.split_once("paged KV admission: need ")?;
    let (admission_blocks, rest) = parse_usize_prefix(after_need)?;
    let rest = rest.strip_prefix(" admission blocks (")?;
    let (immediate_blocks, rest) = parse_usize_prefix(rest)?;
    let rest = rest.strip_prefix(" immediate) but only ")?;
    let (free_blocks, rest) = parse_usize_prefix(rest)?;
    if rest != " free" {
        return None;
    }
    Some(PagedKvAdmissionPressure {
        admission_blocks,
        immediate_blocks,
        free_blocks,
    })
}

fn parse_usize_prefix(input: &str) -> Option<(usize, &str)> {
    let end = input
        .char_indices()
        .find_map(|(index, ch)| (!ch.is_ascii_digit()).then_some(index))
        .unwrap_or(input.len());
    if end == 0 {
        return None;
    }
    let value = input[..end].parse().ok()?;
    Some((value, &input[end..]))
}

pub(super) fn kv_slot_requests_for_unified_batch(
    batch: &ferrum_interfaces::model_executor::UnifiedBatch,
) -> Vec<KvSlotRequest> {
    batch
        .items
        .iter()
        .map(|item| {
            let target_len = item.pos_offset.saturating_add(item.q_tokens.len());
            KvSlotRequest {
                cache_id: item.seq_id.clone(),
                target_len,
                admission_target_len: metadata_kv_admission_target_len(&item.metadata)
                    .map(|len| len.max(target_len)),
            }
        })
        .collect()
}

pub(super) fn metadata_kv_admission_target_len(
    metadata: &HashMap<String, serde_json::Value>,
) -> Option<usize> {
    metadata
        .get(KV_ADMISSION_TARGET_LEN_METADATA_KEY)
        .and_then(|value| value.as_u64())
        .map(|value| value as usize)
        .filter(|&value| value > 0)
}

#[derive(Debug, Default, serde::Serialize)]
pub(super) struct SchedulerTraceDistribution {
    pub(super) count: usize,
    pub(super) min: Option<usize>,
    pub(super) p50: Option<usize>,
    pub(super) max: Option<usize>,
}

#[derive(Debug, Default, serde::Serialize)]
pub(super) struct SchedulerTracePlanStats {
    pub(super) batch_size: usize,
    pub(super) prefill_items: usize,
    pub(super) decode_items: usize,
    pub(super) waiting_items: usize,
    pub(super) preempted_items: usize,
    pub(super) unknown_items: usize,
    pub(super) scheduled_tokens_total: usize,
    pub(super) prefill_tokens: usize,
    pub(super) decode_tokens: usize,
    pub(super) tokens_to_process_missing: usize,
    pub(super) decode_generated_tokens: SchedulerTraceDistribution,
    pub(super) prefill_prompt_tokens: SchedulerTraceDistribution,
    pub(super) requests: Vec<SchedulerTraceRequestStats>,
}

#[derive(Debug, serde::Serialize)]
pub(super) struct SchedulerTraceRequestStats {
    pub(super) request_id: String,
    pub(super) phase: Option<String>,
    pub(super) scheduled_tokens: usize,
    pub(super) tokens_to_process_missing: bool,
    pub(super) prompt_tokens: Option<usize>,
    pub(super) generated_tokens: Option<usize>,
    pub(super) prefill_tokens_processed: Option<usize>,
    pub(super) prefill_tokens_remaining_before: Option<usize>,
    pub(super) is_final_prefill_chunk: Option<bool>,
}

struct ExecutorPrefillProbeResult {
    outcome: ExecutorAdmissionProbeOutcome,
    maintenance: Option<ExecutorPrefillMaintenanceDeferral>,
    trace: Option<ExecutorPrefillAdmissionTrace>,
}

enum ExecutorPrefillAdmissionTraceEvidence {
    Admitted(ExecutorPrefillAdmissionReceipt),
    Deferred(AdmissionDeferred),
    MaintenanceDeferred(ExecutorPrefillMaintenanceDeferral),
    PermanentRejected(AdmissionRejected),
    Faulted(String),
}

struct ExecutorPrefillAdmissionTrace {
    request_id: RequestId,
    evidence: ExecutorPrefillAdmissionTraceEvidence,
}

fn scheduler_trace_monotonic_nanos() -> u64 {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    ORIGIN
        .get_or_init(Instant::now)
        .elapsed()
        .as_nanos()
        .min(u64::MAX as u128) as u64
}

fn scheduler_trace_distribution(mut values: Vec<usize>) -> SchedulerTraceDistribution {
    if values.is_empty() {
        return SchedulerTraceDistribution::default();
    }
    values.sort_unstable();
    SchedulerTraceDistribution {
        count: values.len(),
        min: values.first().copied(),
        p50: values.get(values.len() / 2).copied(),
        max: values.last().copied(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paged_kv_admission_pressure_parses_qwen35_resource_error() {
        let error = FerrumError::resource_exhausted(
            "Qwen3.5 paged KV admission: need 24 admission blocks (4 immediate) but only 3 free",
        );

        assert_eq!(
            paged_kv_admission_pressure(&error),
            Some(PagedKvAdmissionPressure {
                admission_blocks: 24,
                immediate_blocks: 4,
                free_blocks: 3,
            })
        );
    }

    #[test]
    fn paged_kv_admission_pressure_parses_generic_llama_resource_error() {
        let error = FerrumError::resource_exhausted(
            "paged KV admission: need 1 admission blocks (1 immediate) but only 0 free",
        );

        assert_eq!(
            paged_kv_admission_pressure(&error),
            Some(PagedKvAdmissionPressure {
                admission_blocks: 1,
                immediate_blocks: 1,
                free_blocks: 0,
            })
        );
    }

    #[test]
    fn paged_kv_admission_pressure_ignores_unrelated_resource_errors() {
        let error = FerrumError::resource_exhausted("synthetic unified reserve failure");

        assert_eq!(paged_kv_admission_pressure(&error), None);
    }
}

impl EngineInner {
    fn probe_executor_prefill_admission(
        &self,
        request: &InferenceRequest,
        capture_trace: bool,
    ) -> ExecutorPrefillProbeResult {
        let Some((input_tokens, maximum_sequence_tokens)) =
            self.sequences.read().get(&request.id).map(|sequence| {
                (
                    sequence.prefill_context_tokens(),
                    sequence.model_maximum_sequence_tokens(),
                )
            })
        else {
            let error = FerrumError::internal(format!(
                "request {} reached typed admission before its sequence state was published",
                request.id
            ));
            return ExecutorPrefillProbeResult {
                trace: capture_trace.then(|| ExecutorPrefillAdmissionTrace {
                    request_id: request.id.clone(),
                    evidence: ExecutorPrefillAdmissionTraceEvidence::Faulted(error.to_string()),
                }),
                outcome: AdmissionProbeOutcome::Faulted(error),
                maintenance: None,
            };
        };
        match self
            .model_executor
            .try_admit_prefill(ExecutorPrefillAdmission::new(
                &request.id,
                &input_tokens,
                maximum_sequence_tokens,
            )) {
            Ok(ExecutorPrefillAdmissionDecision::Admitted(receipt)) => ExecutorPrefillProbeResult {
                trace: capture_trace.then(|| ExecutorPrefillAdmissionTrace {
                    request_id: request.id.clone(),
                    evidence: ExecutorPrefillAdmissionTraceEvidence::Admitted(receipt.clone()),
                }),
                outcome: AdmissionProbeOutcome::Admitted(receipt),
                maintenance: None,
            },
            Ok(ExecutorPrefillAdmissionDecision::Deferred(deferred)) => {
                if deferred.action() == DeferredAction::AwaitBackingGrowth {
                    self.model_executor.cancel_prefill_admission(&request.id);
                    let error = FerrumError::internal(format!(
                        "executor returned AwaitBackingGrowth for {} without retaining typed maintenance state",
                        request.id
                    ));
                    return ExecutorPrefillProbeResult {
                        trace: capture_trace.then(|| ExecutorPrefillAdmissionTrace {
                            request_id: request.id.clone(),
                            evidence: ExecutorPrefillAdmissionTraceEvidence::Faulted(
                                error.to_string(),
                            ),
                        }),
                        outcome: AdmissionProbeOutcome::Faulted(error),
                        maintenance: None,
                    };
                }
                let outcome = AdmissionProbeOutcome::Deferred(AdmissionDeferral::from_admission(
                    &deferred, 0,
                ));
                ExecutorPrefillProbeResult {
                    trace: capture_trace.then(|| ExecutorPrefillAdmissionTrace {
                        request_id: request.id.clone(),
                        evidence: ExecutorPrefillAdmissionTraceEvidence::Deferred(deferred),
                    }),
                    outcome,
                    maintenance: None,
                }
            }
            Ok(ExecutorPrefillAdmissionDecision::MaintenanceDeferred(deferred)) => {
                if deferred.request_id() != &request.id {
                    self.model_executor.cancel_prefill_admission(&request.id);
                    let error = FerrumError::internal(format!(
                        "executor maintenance deferral belongs to {}, expected {}",
                        deferred.request_id(),
                        request.id
                    ));
                    return ExecutorPrefillProbeResult {
                        trace: capture_trace.then(|| ExecutorPrefillAdmissionTrace {
                            request_id: request.id.clone(),
                            evidence: ExecutorPrefillAdmissionTraceEvidence::Faulted(
                                error.to_string(),
                            ),
                        }),
                        outcome: AdmissionProbeOutcome::Faulted(error),
                        maintenance: None,
                    };
                }
                let observed = deferred.observed();
                let outcome = AdmissionProbeOutcome::Deferred(AdmissionDeferral::new(
                    DeferredAction::AwaitBackingGrowth,
                    AdmissionWakeEpochs::new(
                        observed.coordinator_id,
                        observed.release_epoch,
                        observed.capacity_epoch,
                        0,
                    ),
                    deferred.wait_condition().clone(),
                ));
                ExecutorPrefillProbeResult {
                    trace: capture_trace.then(|| ExecutorPrefillAdmissionTrace {
                        request_id: request.id.clone(),
                        evidence: ExecutorPrefillAdmissionTraceEvidence::MaintenanceDeferred(
                            deferred.clone(),
                        ),
                    }),
                    outcome,
                    maintenance: Some(deferred),
                }
            }
            Ok(ExecutorPrefillAdmissionDecision::PermanentRejected(rejected)) => {
                ExecutorPrefillProbeResult {
                    trace: capture_trace.then(|| ExecutorPrefillAdmissionTrace {
                        request_id: request.id.clone(),
                        evidence: ExecutorPrefillAdmissionTraceEvidence::PermanentRejected(
                            rejected.clone(),
                        ),
                    }),
                    outcome: AdmissionProbeOutcome::PermanentRejected(rejected),
                    maintenance: None,
                }
            }
            Err(error) => ExecutorPrefillProbeResult {
                trace: capture_trace.then(|| ExecutorPrefillAdmissionTrace {
                    request_id: request.id.clone(),
                    evidence: ExecutorPrefillAdmissionTraceEvidence::Faulted(error.to_string()),
                }),
                outcome: AdmissionProbeOutcome::Faulted(error),
                maintenance: None,
            },
        }
    }

    fn write_executor_scheduler_profile_event(
        &self,
        request_id: &RequestId,
        phase: &str,
        event_kind: ProfileEventKind,
        status: ProfileStatus,
        duration_us: Option<u64>,
        shape: BTreeMap<String, serde_json::Value>,
        mut attributes: BTreeMap<String, serde_json::Value>,
        error: Option<ProfileError>,
    ) {
        let Some(sink) = &self.scheduler_trace_jsonl else {
            return;
        };
        let entrypoint = self.trace_entrypoint();
        attributes.extend([
            (
                "actual_model_smoke".to_string(),
                serde_json::json!(matches!(
                    entrypoint,
                    ProfileEntrypoint::Run | ProfileEntrypoint::Serve
                )),
            ),
            (
                "active_sequence_count".to_string(),
                serde_json::json!(self.scheduler.active_count()),
            ),
            (
                "monotonic_nanos".to_string(),
                serde_json::json!(scheduler_trace_monotonic_nanos()),
            ),
            (
                "scheduler_snapshot".to_string(),
                serde_json::to_value(self.scheduler.trace_snapshot())
                    .unwrap_or(serde_json::Value::Null),
            ),
        ]);
        let timestamp = chrono::Utc::now();
        let event_num = self
            .resource_trace_event_counter
            .fetch_add(1, Ordering::Relaxed);
        let event = FerrumProfileEvent {
            schema_version: OBSERVABILITY_PROFILE_SCHEMA_VERSION,
            ts_unix_nanos: timestamp
                .timestamp_nanos_opt()
                .unwrap_or_else(|| timestamp.timestamp_micros() * 1_000),
            event_id: format!("evt-engine-vnext-admission-{event_num}"),
            request_id: request_id.to_string(),
            correlation_id: Some(request_id.to_string()),
            entrypoint,
            backend: "actual".to_string(),
            runtime_preset_hash: ENGINE_RUNTIME_TRACE_PRESET_HASH.to_string(),
            phase: phase.to_string(),
            event_kind,
            timestamp,
            status,
            model: Some(self.config.model.model_id.to_string()),
            duration_us,
            memory: None,
            resource: None,
            error,
            replay: None,
            shape,
            backend_detail: Some(BTreeMap::from([
                (
                    "backend_device".to_string(),
                    serde_json::json!(format!("{:?}", self.config.backend.device)),
                ),
                (
                    "backend_type".to_string(),
                    serde_json::json!(format!("{:?}", self.config.backend.backend_type)),
                ),
            ])),
            attributes,
        };
        if let Err(error) = event.validate() {
            warn!("Skipping invalid executor admission trace event: {}", error);
            return;
        }
        let mut line = match serde_json::to_string(&event) {
            Ok(line) => line,
            Err(error) => {
                warn!("Failed to serialize executor admission trace: {}", error);
                return;
            }
        };
        line.push('\n');
        if let Err(error) = sink.lock().write_all(line.as_bytes()) {
            warn!("Failed to write executor admission trace: {}", error);
        }
    }

    fn trace_executor_prefill_admission(&self, trace: ExecutorPrefillAdmissionTrace) {
        if self.scheduler_trace_jsonl.is_none() {
            return;
        }
        let (decision, evidence, blocker_count, retained, maintenance, error) = match trace.evidence
        {
            ExecutorPrefillAdmissionTraceEvidence::Admitted(receipt) => (
                "admitted",
                serde_json::to_value(receipt).unwrap_or(serde_json::Value::Null),
                0,
                true,
                false,
                None,
            ),
            ExecutorPrefillAdmissionTraceEvidence::Deferred(deferred) => (
                "deferred",
                serde_json::to_value(&deferred).unwrap_or(serde_json::Value::Null),
                deferred.blockers().len(),
                false,
                false,
                None,
            ),
            ExecutorPrefillAdmissionTraceEvidence::MaintenanceDeferred(deferred) => (
                "maintenance_deferred",
                serde_json::to_value(&deferred).unwrap_or(serde_json::Value::Null),
                deferred.blockers().len(),
                false,
                true,
                None,
            ),
            ExecutorPrefillAdmissionTraceEvidence::PermanentRejected(rejected) => (
                "permanent_rejected",
                serde_json::to_value(&rejected).unwrap_or(serde_json::Value::Null),
                rejected.blockers().len(),
                false,
                false,
                None,
            ),
            ExecutorPrefillAdmissionTraceEvidence::Faulted(message) => {
                let error = ProfileError {
                    kind: "executor_prefill_admission_fault".to_string(),
                    message: message.clone(),
                    blocking: true,
                };
                (
                    "faulted",
                    serde_json::json!({ "message": message }),
                    0,
                    false,
                    false,
                    Some(error),
                )
            }
        };
        self.write_executor_scheduler_profile_event(
            &trace.request_id,
            "vnext.prefill_admission",
            if error.is_some() {
                ProfileEventKind::Error
            } else {
                ProfileEventKind::Instant
            },
            if error.is_some() {
                ProfileStatus::Failure
            } else {
                ProfileStatus::Ok
            },
            None,
            BTreeMap::from([
                ("decision".to_string(), serde_json::json!(decision)),
                (
                    "blocker_count".to_string(),
                    serde_json::json!(blocker_count),
                ),
                (
                    "execution_authority_retained".to_string(),
                    serde_json::json!(retained),
                ),
                (
                    "maintenance_required".to_string(),
                    serde_json::json!(maintenance),
                ),
                (
                    "prefill_submit_observed".to_string(),
                    serde_json::json!(false),
                ),
            ]),
            BTreeMap::from([("admission_evidence".to_string(), evidence)]),
            error,
        );
    }

    fn trace_executor_admission_queue_observation(
        &self,
        observation: ExecutorAdmissionQueueObservation,
    ) {
        let epochs = |value: AdmissionWakeEpochs| {
            serde_json::json!({
                "coordinator_id": value.coordinator_id().get(),
                "release_epoch": value.release_epoch(),
                "capacity_epoch": value.capacity_epoch(),
                "policy_epoch": value.policy_epoch(),
            })
        };
        match observation {
            ExecutorAdmissionQueueObservation::SkippedUnchanged {
                request_id,
                ticket,
                deferral,
                current,
            } => self.write_executor_scheduler_profile_event(
                &request_id,
                "vnext.prefill_admission_skipped_unchanged",
                ProfileEventKind::Instant,
                ProfileStatus::Ok,
                None,
                BTreeMap::from([
                    (
                        "decision".to_string(),
                        serde_json::json!("skipped_unchanged"),
                    ),
                    ("waiting_ticket".to_string(), serde_json::json!(ticket)),
                    (
                        "prefill_submit_observed".to_string(),
                        serde_json::json!(false),
                    ),
                    ("probe_performed".to_string(), serde_json::json!(false)),
                ]),
                BTreeMap::from([(
                    "deferral_evidence".to_string(),
                    serde_json::json!({
                        "action": deferral.action(),
                        "observed": epochs(deferral.observed()),
                        "current": epochs(current),
                        "wait_condition": deferral.wait_condition(),
                    }),
                )]),
                None,
            ),
            ExecutorAdmissionQueueObservation::DecodeSkippedUnchanged {
                request_id,
                deferral,
                current,
            } => self.write_executor_scheduler_profile_event(
                &request_id,
                "vnext.decode_capacity_skipped_unchanged",
                ProfileEventKind::Instant,
                ProfileStatus::Ok,
                None,
                BTreeMap::from([
                    (
                        "decision".to_string(),
                        serde_json::json!("skipped_unchanged"),
                    ),
                    (
                        "decode_submit_observed".to_string(),
                        serde_json::json!(false),
                    ),
                    ("probe_performed".to_string(), serde_json::json!(false)),
                ]),
                BTreeMap::from([(
                    "deferral_evidence".to_string(),
                    serde_json::json!({
                        "action": deferral.action(),
                        "observed": epochs(deferral.observed()),
                        "current": epochs(current),
                        "wait_condition": deferral.wait_condition(),
                    }),
                )]),
                None,
            ),
            ExecutorAdmissionQueueObservation::DecodeResumed {
                request_id,
                deferral,
                current,
                exact_source_changed,
                policy_epoch_changed,
            } => self.write_executor_scheduler_profile_event(
                &request_id,
                "vnext.decode_capacity_resumed",
                ProfileEventKind::Instant,
                ProfileStatus::Ok,
                None,
                BTreeMap::from([
                    (
                        "decision".to_string(),
                        serde_json::json!(if exact_source_changed {
                            "exact_source_changed"
                        } else {
                            "policy_epoch_changed"
                        }),
                    ),
                    (
                        "exact_source_changed".to_string(),
                        serde_json::json!(exact_source_changed),
                    ),
                    (
                        "policy_epoch_changed".to_string(),
                        serde_json::json!(policy_epoch_changed),
                    ),
                    (
                        "decode_submit_observed".to_string(),
                        serde_json::json!(false),
                    ),
                    ("probe_performed".to_string(), serde_json::json!(false)),
                ]),
                BTreeMap::from([(
                    "deferral_evidence".to_string(),
                    serde_json::json!({
                        "action": deferral.action(),
                        "observed": epochs(deferral.observed()),
                        "current": epochs(current),
                        "wait_condition": deferral.wait_condition(),
                    }),
                )]),
                None,
            ),
        }
    }

    fn trace_executor_decode_capacity_decision(
        &self,
        request_ids: &[RequestId],
        deferral: &ExecutorExecutionCapacityDeferral,
        decision: &'static str,
    ) {
        let Some(request_id) = request_ids.first() else {
            return;
        };
        let observed = deferral.observed();
        self.write_executor_scheduler_profile_event(
            request_id,
            "vnext.decode_capacity_deferred",
            ProfileEventKind::Instant,
            ProfileStatus::Ok,
            None,
            BTreeMap::from([
                ("decision".to_string(), serde_json::json!(decision)),
                (
                    "attempted_decode_width".to_string(),
                    serde_json::json!(request_ids.len()),
                ),
                (
                    "execution_stage".to_string(),
                    serde_json::json!(deferral.stage()),
                ),
                (
                    "decode_submit_observed".to_string(),
                    serde_json::json!(false),
                ),
            ]),
            BTreeMap::from([
                ("request_ids".to_string(), serde_json::json!(request_ids)),
                (
                    "capacity_evidence".to_string(),
                    serde_json::json!({
                        "observed": {
                            "coordinator_id": observed.coordinator_id,
                            "release_epoch": observed.release_epoch,
                            "capacity_epoch": observed.capacity_epoch,
                        },
                        "wait_condition": deferral.wait_condition(),
                    }),
                ),
            ]),
            None,
        );
    }

    fn validate_executor_prefill_maintenance(
        &self,
        deferral: &ExecutorPrefillMaintenanceDeferral,
        outcome: &ExecutorPrefillMaintenanceOutcome,
    ) -> Result<()> {
        let observed = deferral.observed();
        let validate_current = |current: ExecutorAdmissionEpochs| -> Result<()> {
            if current.coordinator_id != observed.coordinator_id {
                return Err(FerrumError::internal(format!(
                    "executor maintenance for {} changed admission coordinator",
                    deferral.request_id()
                )));
            }
            if current.release_epoch < observed.release_epoch
                || current.capacity_epoch < observed.capacity_epoch
            {
                return Err(FerrumError::internal(format!(
                    "executor maintenance for {} regressed admission epochs",
                    deferral.request_id()
                )));
            }
            Ok(())
        };
        match outcome {
            ExecutorPrefillMaintenanceOutcome::NoLongerPending => {
                if self.scheduler.trace_phase(deferral.request_id()) == Some(RequestPhase::Waiting)
                {
                    return Err(FerrumError::internal(format!(
                        "executor lost retained maintenance for waiting request {}",
                        deferral.request_id()
                    )));
                }
            }
            ExecutorPrefillMaintenanceOutcome::RetryAdmission { current } => {
                validate_current(*current)?;
            }
            ExecutorPrefillMaintenanceOutcome::WaitForRelease {
                current,
                wait_condition,
                pressure,
            } => {
                validate_current(*current)?;
                if wait_condition.coordinator_id().get() != current.coordinator_id.get()
                    || !matches!(
                        pressure.scope(),
                        DeviceCapacityPressureScope::PlanBudget
                            | DeviceCapacityPressureScope::ProcessWide
                    )
                    || pressure.requested_bytes() == 0
                    || pressure.available_bytes() >= pressure.requested_bytes()
                {
                    return Err(FerrumError::internal(format!(
                        "executor maintenance for {} reported invalid device pressure",
                        deferral.request_id()
                    )));
                }
            }
            ExecutorPrefillMaintenanceOutcome::Maintained {
                current,
                pools_grown,
                allocated_bytes,
            } => {
                validate_current(*current)?;
                if current.capacity_epoch <= observed.capacity_epoch
                    || *pools_grown == 0
                    || *allocated_bytes == 0
                {
                    return Err(FerrumError::internal(format!(
                        "executor maintenance for {} reported growth without installed capacity",
                        deferral.request_id()
                    )));
                }
            }
        }
        Ok(())
    }

    fn trace_executor_prefill_maintenance(
        &self,
        request_id: &RequestId,
        result: &Result<ExecutorPrefillMaintenanceOutcome>,
        elapsed: Duration,
    ) {
        if self.scheduler_trace_jsonl.is_none() {
            return;
        }
        let (outcome_name, evidence, error) = match result {
            Ok(ExecutorPrefillMaintenanceOutcome::NoLongerPending) => (
                "no_longer_pending",
                serde_json::to_value(result.as_ref().unwrap()).unwrap_or(serde_json::Value::Null),
                None,
            ),
            Ok(ExecutorPrefillMaintenanceOutcome::RetryAdmission { .. }) => (
                "retry_admission",
                serde_json::to_value(result.as_ref().unwrap()).unwrap_or(serde_json::Value::Null),
                None,
            ),
            Ok(ExecutorPrefillMaintenanceOutcome::WaitForRelease { .. }) => (
                "wait_for_release",
                serde_json::to_value(result.as_ref().unwrap()).unwrap_or(serde_json::Value::Null),
                None,
            ),
            Ok(ExecutorPrefillMaintenanceOutcome::Maintained { .. }) => (
                "maintained",
                serde_json::to_value(result.as_ref().unwrap()).unwrap_or(serde_json::Value::Null),
                None,
            ),
            Err(failure) => {
                let message = failure.to_string();
                (
                    "faulted",
                    serde_json::json!({ "message": message }),
                    Some(ProfileError {
                        kind: "executor_prefill_maintenance_fault".to_string(),
                        message,
                        blocking: true,
                    }),
                )
            }
        };
        self.write_executor_scheduler_profile_event(
            request_id,
            "vnext.prefill_backing_maintenance",
            if error.is_some() {
                ProfileEventKind::Error
            } else {
                ProfileEventKind::TimedSpan
            },
            if error.is_some() {
                ProfileStatus::Failure
            } else {
                ProfileStatus::Ok
            },
            Some(duration_to_us(elapsed)),
            BTreeMap::from([
                ("outcome".to_string(), serde_json::json!(outcome_name)),
                (
                    "duration_us".to_string(),
                    serde_json::json!(duration_to_us(elapsed)),
                ),
            ]),
            BTreeMap::from([("maintenance_evidence".to_string(), evidence)]),
            error,
        );
    }

    fn execute_executor_prefill_maintenance(
        &self,
        maintenance: Vec<ExecutorPrefillMaintenanceDeferral>,
    ) {
        for deferral in maintenance {
            let started = Instant::now();
            let result = self
                .model_executor
                .maintain_prefill_backing(deferral.request_id())
                .and_then(|outcome| {
                    self.validate_executor_prefill_maintenance(&deferral, &outcome)?;
                    let transitioned = match &outcome {
                        ExecutorPrefillMaintenanceOutcome::RetryAdmission { current } => {
                            let observed = AdmissionWakeEpochs::new(
                                current.coordinator_id,
                                current.release_epoch,
                                current.capacity_epoch,
                                0,
                            );
                            Some(
                                self.scheduler
                                    .retry_after_backing_recheck(deferral.request_id(), observed)?,
                            )
                        }
                        ExecutorPrefillMaintenanceOutcome::WaitForRelease {
                            current,
                            wait_condition,
                            ..
                        } => {
                            let observed = AdmissionWakeEpochs::new(
                                current.coordinator_id,
                                current.release_epoch,
                                current.capacity_epoch,
                                0,
                            );
                            Some(self.scheduler.wait_for_release_after_backing_pressure(
                                deferral.request_id(),
                                observed,
                                wait_condition.clone(),
                            )?)
                        }
                        _ => None,
                    };
                    if transitioned == Some(false)
                        && self.scheduler.trace_phase(deferral.request_id())
                            == Some(RequestPhase::Waiting)
                    {
                        return Err(FerrumError::internal(format!(
                            "waiting request {} lost its backing-growth deferral",
                            deferral.request_id()
                        )));
                    }
                    Ok(outcome)
                });
            self.trace_executor_prefill_maintenance(
                deferral.request_id(),
                &result,
                started.elapsed(),
            );
            if let Err(error) = result {
                warn!(
                    request_id = %deferral.request_id(),
                    error = %error,
                    "Executor prefill backing maintenance failed"
                );
                self.model_executor
                    .cancel_prefill_admission(deferral.request_id());
                self.scheduler
                    .fail_waiting_admission(deferral.request_id(), error);
            }
        }
    }

    async fn complete_typed_admission_failures(&self) -> Result<()> {
        for (request_id, error) in self.scheduler.take_admission_failures() {
            warn!(
                request_id = %request_id,
                error = %error,
                "Typed prefill admission failed before device submission"
            );
            self.model_executor.cancel_prefill_admission(&request_id);
            self.complete_request(&request_id, FinishReason::Error)
                .await?;
        }
        Ok(())
    }

    // ── tensor helper ──────────────────────────────────────────────────

    pub(super) fn tokens_to_tensor(&self, token_ids: &[u32]) -> Result<TensorRef> {
        let f32_data: Vec<f32> = token_ids.iter().map(|&v| v as f32).collect();
        let len = f32_data.len();
        self.tensor_factory
            .from_slice(&f32_data, &[1, len], DataType::FP32, Device::CPU)
    }

    /// Rebuild a KvCacheHandle with a corrected sequence_length.
    ///
    /// Only meaningful for `GenericKvCacheHandle`, which is what the LLM
    /// executor (`LlmExecutor::prefill` / `decode`) constructs and threads
    /// through speculative decoding. Resource handles minted by
    /// `KvCacheManager` impls (Paged / Default) are returned as a plain
    /// clone — those handles don't track per-iter position (the model's
    /// internal paged_pool does), and the engine no longer reads
    /// `sequence_length` from them for position purposes (see
    /// `process_batch_unified` for the SequenceState-sourced pos_offset).
    pub(super) fn make_kv_handle_with_seq(
        &self,
        h: &std::sync::Arc<dyn ferrum_interfaces::KvCacheHandle>,
        new_seq: usize,
    ) -> std::sync::Arc<dyn ferrum_interfaces::KvCacheHandle> {
        if let Some(g) = h
            .as_any()
            .downcast_ref::<ferrum_models::executor::common::GenericKvCacheHandle>()
        {
            std::sync::Arc::new(g.with_sequence_length(new_seq))
        } else {
            h.clone()
        }
    }

    /// Build the model-executor KV handle used by `LlmExecutor::decode`.
    ///
    /// The continuous engine has two KV identities:
    /// - `KvCacheManager` allocations, keyed by request id, track resource
    ///   lifetime and are deallocated on preemption/completion.
    /// - model/executor cache ids track the actual model-side KV contents.
    ///
    /// `SequenceState` model-KV state must carry the second identity because the
    /// fallback single-request decode path downcasts it to
    /// `GenericKvCacheHandle`. Unified CUDA paths don't read the handle body,
    /// but keeping this invariant prevents resource-pressure fallbacks from
    /// feeding a manager handle into `LlmExecutor::decode`.
    pub(super) fn make_model_kv_handle_with_seq(
        &self,
        cache_id: String,
        seq_len: usize,
    ) -> std::sync::Arc<dyn ferrum_interfaces::KvCacheHandle> {
        let info = self.model_executor.info();
        let head_dim = if info.num_heads == 0 {
            info.hidden_size.max(1)
        } else {
            (info.hidden_size / info.num_heads).max(1)
        };
        let num_kv_heads = if info.num_kv_heads == 0 {
            info.num_heads.max(1)
        } else {
            info.num_kv_heads
        };
        std::sync::Arc::new(ferrum_models::executor::common::GenericKvCacheHandle::new(
            info.num_layers,
            num_kv_heads,
            head_dim,
            candle_core::Device::Cpu,
            seq_len,
            cache_id,
        ))
    }

    pub(super) fn decode_ready_request_ids(&self, request_ids: &[RequestId]) -> Vec<RequestId> {
        let sequences = self.sequences.read();
        request_ids
            .iter()
            .filter(|rid| {
                sequences.get(*rid).is_some_and(|seq| {
                    seq.prefill_complete
                        && seq.kv_cache_handle().is_some()
                        && !seq.generated_tokens.is_empty()
                })
            })
            .cloned()
            .collect()
    }

    fn scheduler_trace_enabled(&self) -> bool {
        self.legacy_scheduler_trace_jsonl.is_some()
    }

    pub(super) fn scheduler_trace_plan_stats(
        &self,
        batch: &ferrum_interfaces::BatchPlan,
    ) -> SchedulerTracePlanStats {
        let mut stats = SchedulerTracePlanStats {
            batch_size: batch.size(),
            ..SchedulerTracePlanStats::default()
        };
        let mut decode_generated_tokens = Vec::new();
        let mut prefill_prompt_tokens = Vec::new();
        let sequences = self.sequences.read();

        for scheduled_req in &batch.requests {
            let request_id = &scheduled_req.request.id;
            let scheduled_tokens = scheduled_req.tokens_to_process.unwrap_or(0);
            if scheduled_req.tokens_to_process.is_none() {
                stats.tokens_to_process_missing += 1;
            }
            stats.scheduled_tokens_total += scheduled_tokens;

            let phase = self.scheduler.trace_phase(request_id);
            let seq = sequences.get(request_id);
            let prompt_tokens = seq.map(|seq| seq.prefill_context_len());
            let generated_tokens = seq.map(|seq| seq.generated_tokens.len());
            let prefill_tokens_processed = seq.map(|seq| seq.prefill_tokens_processed);
            let prefill_tokens_remaining_before = prompt_tokens
                .zip(prefill_tokens_processed)
                .map(|(prompt, processed)| prompt.saturating_sub(processed));
            let is_final_prefill_chunk = match (phase, prefill_tokens_remaining_before) {
                (Some(RequestPhase::Prefilling), Some(remaining)) => {
                    Some(scheduled_tokens >= remaining)
                }
                _ => None,
            };

            match phase {
                Some(RequestPhase::Decoding) => {
                    stats.decode_items += 1;
                    stats.decode_tokens += scheduled_tokens;
                    if let Some(generated_tokens) = generated_tokens {
                        decode_generated_tokens.push(generated_tokens);
                    }
                }
                Some(RequestPhase::Prefilling) => {
                    stats.prefill_items += 1;
                    stats.prefill_tokens += scheduled_tokens;
                    if let Some(prompt_tokens) = prompt_tokens {
                        prefill_prompt_tokens.push(prompt_tokens);
                    }
                }
                Some(RequestPhase::Waiting) => {
                    stats.waiting_items += 1;
                    stats.prefill_tokens += scheduled_tokens;
                }
                Some(RequestPhase::Preempted) => {
                    stats.preempted_items += 1;
                }
                Some(
                    RequestPhase::Completed
                    | RequestPhase::Cancelled
                    | RequestPhase::AdmissionFailed,
                )
                | None => {
                    stats.unknown_items += 1;
                }
            }

            stats.requests.push(SchedulerTraceRequestStats {
                request_id: request_id.to_string(),
                phase: phase.map(|phase| format!("{phase:?}")),
                scheduled_tokens,
                tokens_to_process_missing: scheduled_req.tokens_to_process.is_none(),
                prompt_tokens,
                generated_tokens,
                prefill_tokens_processed,
                prefill_tokens_remaining_before,
                is_final_prefill_chunk,
            });
        }

        stats.decode_generated_tokens = scheduler_trace_distribution(decode_generated_tokens);
        stats.prefill_prompt_tokens = scheduler_trace_distribution(prefill_prompt_tokens);
        stats
    }

    fn write_scheduler_trace_event(&self, event: serde_json::Value) {
        let Some(file) = &self.legacy_scheduler_trace_jsonl else {
            return;
        };
        let mut file = file.lock();
        if let Err(error) = serde_json::to_writer(&mut *file, &event) {
            warn!("Failed to write scheduler trace event: {}", error);
            return;
        }
        if let Err(error) = file.write_all(b"\n") {
            warn!("Failed to terminate scheduler trace event: {}", error);
        }
    }

    // ── iteration loop ─────────────────────────────────────────────────

    /// Run one iteration: ask the scheduler for a batch, then process it.
    pub(super) async fn run_iteration(&self) -> Result<EngineIterationOutcome> {
        self.cancel_abandoned_requests().await?;

        let iteration = self.iteration_count.fetch_add(1, Ordering::Relaxed);
        counter!("ferrum.engine.iterations_total").increment(1);
        let prof = self.runtime_config.batch_decode_prof;
        let t_iter_start = if prof { Some(Instant::now()) } else { None };
        let trace_enabled = self.scheduler_trace_enabled();
        let trace_scheduler_before = trace_enabled.then(|| self.scheduler.trace_snapshot());
        let trace_prefill_tokens_before = if trace_enabled {
            Some(self.total_prefill_tokens.load(Ordering::Relaxed))
        } else {
            None
        };
        let trace_decode_tokens_before = if trace_enabled {
            Some(self.total_decode_tokens.load(Ordering::Relaxed))
        } else {
            None
        };

        // Phase 3 token-budget hint: scheduler emits a mixed batch
        // summing to at most `max_num_batched_tokens` Q tokens. This
        // replaces the prior `max_batch_size * 2048` heuristic which
        // never bit and left scheduler-side prefill admission capped
        // at `max_prefill_batch=8`. Defaults to 4096 (autosizer can
        // override via `FERRUM_MAX_BATCHED_TOKENS`).
        let hint = ferrum_interfaces::BatchHint {
            max_batch_size: self.config.batching.max_batch_size,
            max_tokens: self.config.batching.max_num_batched_tokens,
            target_latency_ms: None,
            available_memory: None,
            resource_constraints: ferrum_interfaces::scheduler::ResourceConstraints::default(),
        };
        let hint_max_batch_size = hint.max_batch_size;
        let hint_max_tokens = hint.max_tokens;

        // FERRUM_NEXT_BATCH_PROF=1: count Some/None returns to root-cause
        // the apples HTTP-serve 17 ms inter-batch-iter gap. Prints every
        // 1024 run_iteration calls. Set None_size to 0 explicitly so the
        // bg-loop tight-spin theory can be confirmed.
        let nb_prof = self.runtime_config.next_batch_prof;
        let sched_t0 = Instant::now();
        let nb_t0 = if nb_prof { Some(Instant::now()) } else { None };
        let mut prefill_maintenance = Vec::new();
        let plan_runtime_managed = self.model_executor.execution_resource_authority()
            == ExecutionResourceAuthority::PlanRuntime;
        let nb_result = if plan_runtime_managed {
            let mut availability = self.dynamic_admission_availability.lock();
            let epochs = self
                .model_executor
                .write_execution_capacity_snapshot(&mut availability)?
                .ok_or_else(|| {
                    FerrumError::scheduler("plan runtime did not expose typed admission epochs")
                })?;
            let wake_epochs = AdmissionWakeEpochs::new(
                epochs.coordinator_id,
                epochs.release_epoch,
                epochs.capacity_epoch,
                0,
            );
            let wake = AdmissionWakeSnapshot::new(wake_epochs, &availability);
            let capture_trace = self.scheduler_trace_jsonl.is_some();
            let mut admission_traces = capture_trace.then(Vec::new);
            let mut admission_queue_observations = capture_trace.then(Vec::new);
            let mut probe = |request: &InferenceRequest| {
                let result = self.probe_executor_prefill_admission(request, capture_trace);
                if let Some(maintenance) = result.maintenance {
                    prefill_maintenance.push(maintenance);
                }
                if let (Some(traces), Some(trace)) = (&mut admission_traces, result.trace) {
                    traces.push(trace);
                }
                result.outcome
            };
            let scheduled = if let Some(observations) = &mut admission_queue_observations {
                self.scheduler.next_batch_with_dynamic_admission_observed(
                    hint,
                    wake,
                    &mut probe,
                    &mut |observation| observations.push(observation),
                )
            } else {
                self.scheduler
                    .next_batch_with_dynamic_admission(hint, wake, &mut probe)
            };
            drop(availability);
            drop(probe);
            if let Some(traces) = admission_traces {
                for trace in traces {
                    self.trace_executor_prefill_admission(trace);
                }
            }
            if let Some(observations) = admission_queue_observations {
                for observation in observations {
                    self.trace_executor_admission_queue_observation(observation);
                }
            }
            if scheduled.is_err() {
                for deferral in &prefill_maintenance {
                    self.model_executor
                        .cancel_prefill_admission(deferral.request_id());
                }
                prefill_maintenance.clear();
            }
            scheduled?
        } else {
            self.scheduler.next_batch(hint).await
        };
        self.complete_typed_admission_failures().await?;
        let sched_elapsed = sched_t0.elapsed();
        self.record_scheduling_time(sched_elapsed);
        if let Some(t0) = nb_t0 {
            use std::sync::atomic::AtomicU64;
            static SOME_N: AtomicU64 = AtomicU64::new(0);
            static NONE_N: AtomicU64 = AtomicU64::new(0);
            static SOME_US: AtomicU64 = AtomicU64::new(0);
            static NONE_US: AtomicU64 = AtomicU64::new(0);
            let us = t0.elapsed().as_micros() as u64;
            let is_some = nb_result.is_some();
            let batch_size = nb_result.as_ref().map_or(0, |b| b.size());
            if is_some {
                SOME_N.fetch_add(1, Ordering::Relaxed);
                SOME_US.fetch_add(us, Ordering::Relaxed);
            } else {
                NONE_N.fetch_add(1, Ordering::Relaxed);
                NONE_US.fetch_add(us, Ordering::Relaxed);
            }
            let total = SOME_N.load(Ordering::Relaxed) + NONE_N.load(Ordering::Relaxed);
            if total.is_multiple_of(1024) {
                let s_n = SOME_N.load(Ordering::Relaxed);
                let n_n = NONE_N.load(Ordering::Relaxed);
                let s_us = SOME_US.load(Ordering::Relaxed);
                let n_us = NONE_US.load(Ordering::Relaxed);
                eprintln!(
                    "[nb-prof] total={} some={} none={} ratio={:.3} | some_avg={}us none_avg={}us last_batch_size={} last_was_some={}",
                    total,
                    s_n,
                    n_n,
                    s_n as f64 / total as f64,
                    if s_n > 0 { s_us / s_n } else { 0 },
                    if n_n > 0 { n_us / n_n } else { 0 },
                    batch_size,
                    is_some,
                );
            }
        }

        let batch = match nb_result {
            Some(b) => b,
            None => {
                self.execute_executor_prefill_maintenance(prefill_maintenance);
                self.complete_typed_admission_failures().await?;
                if trace_enabled {
                    let none_streak = self
                        .scheduler_trace_none_streak
                        .fetch_add(1, Ordering::Relaxed)
                        + 1;
                    if none_streak <= 4 || none_streak.is_multiple_of(128) {
                        self.write_scheduler_trace_event(serde_json::json!({
                            "event": "scheduler_iteration",
                            "iteration": iteration,
                            "result": "none",
                            "none_streak": none_streak,
                            "hint": {
                                "max_batch_size": hint_max_batch_size,
                                "max_tokens": hint_max_tokens,
                            },
                            "scheduler_before": trace_scheduler_before.as_ref(),
                            "scheduler_after_schedule": self.scheduler.trace_snapshot(),
                            "timing_us": {
                                "schedule": duration_to_us(sched_elapsed),
                            },
                        }));
                    }
                }
                if plan_runtime_managed {
                    if let Some(observed) = self.scheduler.passive_capacity_wait_condition()? {
                        let registration = self
                            .model_executor
                            .register_execution_capacity_waiter(&observed)?
                            .ok_or_else(|| {
                                FerrumError::scheduler(
                                    "plan runtime did not register its capacity waiter",
                                )
                            })?;
                        return Ok(EngineIterationOutcome::CapacityBlocked(registration));
                    }
                }
                if self.scheduler.active_count() > 0 {
                    return Ok(EngineIterationOutcome::Progressed);
                }
                return if self.scheduler.waiting_count() == 0 {
                    Ok(EngineIterationOutcome::Idle)
                } else {
                    Ok(EngineIterationOutcome::Progressed)
                };
            }
        };
        let trace_none_since_last_some = if trace_enabled {
            Some(self.scheduler_trace_none_streak.swap(0, Ordering::Relaxed))
        } else {
            None
        };
        let trace_scheduler_after_schedule = trace_enabled.then(|| self.scheduler.trace_snapshot());
        let trace_plan = trace_enabled.then(|| self.scheduler_trace_plan_stats(&batch));
        let t_after_sched = if prof { Some(Instant::now()) } else { None };

        debug!(
            "Iteration {}: batch with {} requests",
            iteration,
            batch.size()
        );

        let process_t0 = Instant::now();
        let r = self.process_batch(&batch).await;
        let process_elapsed = process_t0.elapsed();
        self.record_model_execution_time(process_elapsed);
        self.execute_executor_prefill_maintenance(prefill_maintenance);
        self.complete_typed_admission_failures().await?;
        if trace_enabled {
            let prefill_tokens_after = self.total_prefill_tokens.load(Ordering::Relaxed);
            let decode_tokens_after = self.total_decode_tokens.load(Ordering::Relaxed);
            self.write_scheduler_trace_event(serde_json::json!({
                "event": "scheduler_iteration",
                "iteration": iteration,
                "result": if r.is_ok() { "some_ok" } else { "some_error" },
                "error": r.as_ref().err().map(|error| error.to_string()),
                "none_since_last_some": trace_none_since_last_some,
                "hint": {
                    "max_batch_size": hint_max_batch_size,
                    "max_tokens": hint_max_tokens,
                },
                "scheduler_before": trace_scheduler_before.as_ref(),
                "scheduler_after_schedule": trace_scheduler_after_schedule.as_ref(),
                "scheduler_after_process": self.scheduler.trace_snapshot(),
                "plan": trace_plan.as_ref(),
                "engine_counters": {
                    "prefill_tokens_before": trace_prefill_tokens_before,
                    "prefill_tokens_after": prefill_tokens_after,
                    "prefill_tokens_delta": prefill_tokens_after
                        .saturating_sub(trace_prefill_tokens_before.unwrap_or(prefill_tokens_after)),
                    "decode_tokens_before": trace_decode_tokens_before,
                    "decode_tokens_after": decode_tokens_after,
                    "decode_tokens_delta": decode_tokens_after
                        .saturating_sub(trace_decode_tokens_before.unwrap_or(decode_tokens_after)),
                },
                "timing_us": {
                    "schedule": duration_to_us(sched_elapsed),
                    "process": duration_to_us(process_elapsed),
                    "total_since_schedule_start": duration_to_us(sched_t0.elapsed()),
                },
            }));
        }
        if let (Some(t0), Some(ts)) = (t_iter_start, t_after_sched) {
            let n = self.iteration_count.load(Ordering::Relaxed);
            if n < 64 || n.is_multiple_of(32) {
                let total = t0.elapsed().as_micros();
                let sched = ts.duration_since(t0).as_micros();
                let proc = ts.elapsed().as_micros();
                eprintln!(
                    "[iter-prof] iter#{} total={}us sched={}us process={}us batch_size={}",
                    iteration,
                    total,
                    sched,
                    proc,
                    batch.size()
                );
                let profile = global_profile();
                if profile.is_enabled() {
                    let _ = profile.push_event(
                        "iter_prof",
                        profile_fields_from_json(serde_json::json!({
                            "iter": iteration,
                            "batch_size": batch.size(),
                        })),
                        profile_fields_from_json(serde_json::json!({
                            "total": total,
                            "sched": sched,
                            "process": proc,
                        })),
                        false,
                    );
                }
            }
        }
        r?;
        Ok(EngineIterationOutcome::Progressed)
    }
}
