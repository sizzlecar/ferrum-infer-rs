//! EngineInner iteration and request-processing implementation.

use super::*;
use ferrum_execution::token_execution::{
    ExecutionAdmission, ExecutionEndReason, TokenAdmissionCapacityDeferral, TokenAdmissionOutcome,
};
use ferrum_runtime::DynamicBackingPressure;

mod batch;
pub(super) mod completion;
pub(super) mod prefix_rendezvous;
pub(super) mod prefix_restore;
mod token_admission;
mod token_batch;

/// Retained admission maintenance is an owned obligation even while product
/// completion/prefix work awaits. An early error or dropped iteration must
/// cancel those exact pending admissions, not abandon their native owners.
struct PendingPrefillMaintenance<'a> {
    executor: &'a (dyn ModelExecutor + Send + Sync),
    pending: Vec<ExecutorPrefillMaintenanceDeferral>,
}

impl<'a> PendingPrefillMaintenance<'a> {
    fn new(
        executor: &'a (dyn ModelExecutor + Send + Sync),
        pending: Vec<ExecutorPrefillMaintenanceDeferral>,
    ) -> Self {
        Self { executor, pending }
    }

    fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    fn into_pending(mut self) -> Vec<ExecutorPrefillMaintenanceDeferral> {
        std::mem::take(&mut self.pending)
    }
}

impl Drop for PendingPrefillMaintenance<'_> {
    fn drop(&mut self) {
        for deferral in &self.pending {
            self.executor
                .cancel_prefill_admission(deferral.request_id());
        }
    }
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
    CapacityDeferred(TokenAdmissionCapacityDeferral),
    MaintenanceDeferred(ExecutorPrefillMaintenanceDeferral),
    PermanentRejected(AdmissionRejected),
    Faulted(String),
}

struct ExecutorPrefillAdmissionTrace {
    request_id: RequestId,
    input_token_count: Option<usize>,
    maximum_sequence_tokens: Option<usize>,
    evidence: ExecutorPrefillAdmissionTraceEvidence,
}

enum ExecutorSchedulerTraceRecord {
    PrefillAdmission(ExecutorPrefillAdmissionTrace),
    AdmissionQueueObservation(ExecutorAdmissionQueueObservation),
}

pub(super) fn scheduler_trace_monotonic_nanos() -> u64 {
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

struct SchedulerProfileFields {
    shape: BTreeMap<String, serde_json::Value>,
    attributes: BTreeMap<String, serde_json::Value>,
    error: Option<ProfileError>,
}

impl EngineInner {
    fn write_executor_scheduler_profile_event(
        &self,
        request_id: &RequestId,
        phase: &str,
        event_kind: ProfileEventKind,
        status: ProfileStatus,
        duration_us: Option<u64>,
        build_fields: impl FnOnce() -> SchedulerProfileFields,
    ) {
        let Some(sink) = &self.scheduler_trace_jsonl else {
            return;
        };
        #[cfg(test)]
        self.trace_payload_builds.fetch_add(1, Ordering::Relaxed);
        let SchedulerProfileFields {
            shape,
            mut attributes,
            error,
        } = build_fields();
        let entrypoint = self.trace_entrypoint();
        attributes.insert(
            "actual_model_smoke".to_string(),
            serde_json::json!(matches!(
                entrypoint,
                ProfileEntrypoint::Run | ProfileEntrypoint::Serve
            )),
        );
        self.extend_scheduler_timeline_attributes(&mut attributes);
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
            self.report_observation_failure(ExecutionEventSinkError::contract(format!(
                "invalid executor admission trace event: {error}"
            )));
            return;
        }
        if let Err(error) = sink.enqueue(event) {
            self.report_observation_failure(error);
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
            ExecutorPrefillAdmissionTraceEvidence::CapacityDeferred(deferred) => (
                "capacity_deferred",
                serde_json::json!({
                    "required_blocks": deferred.required_blocks(),
                    "free_blocks": deferred.snapshot().free_blocks,
                    "total_blocks": deferred.snapshot().total_blocks,
                    "block_size": deferred.snapshot().block_size,
                    "retry_after": "confirmed_physical_release",
                }),
                1,
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
            || SchedulerProfileFields {
                shape: BTreeMap::from([
                    ("decision".to_string(), serde_json::json!(decision)),
                    (
                        "blocker_count".to_string(),
                        serde_json::json!(blocker_count),
                    ),
                    (
                        "input_token_count".to_string(),
                        serde_json::json!(trace.input_token_count),
                    ),
                    (
                        "maximum_sequence_tokens".to_string(),
                        serde_json::json!(trace.maximum_sequence_tokens),
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
                attributes: BTreeMap::from([("admission_evidence".to_string(), evidence)]),
                error: error,
            },
        );
    }

    fn trace_executor_admission_queue_observation(
        &self,
        observation: ExecutorAdmissionQueueObservation,
    ) {
        if self.scheduler_trace_jsonl.is_none() {
            return;
        }
        let epochs = |value: AdmissionWakeEpochs| {
            serde_json::json!({
                "coordinator_id": value.coordinator_id().get(),
                "release_epoch": value.release_epoch(),
                "capacity_epoch": value.capacity_epoch(),
                "policy_epoch": value.policy_epoch(),
            })
        };
        match observation {
            ExecutorAdmissionQueueObservation::AllocatorCapacityWaitingForOwner {
                request_id,
                domain,
                progress_owner_id,
                observed_release_epoch,
                current_release_epoch,
            } => self.write_executor_scheduler_profile_event(
                &request_id,
                "execution.allocator_capacity_wait",
                ProfileEventKind::Instant,
                ProfileStatus::Ok,
                None,
                || SchedulerProfileFields {
                    shape: BTreeMap::from([
                        (
                            "decision".to_string(),
                            serde_json::json!("waiting_for_owner"),
                        ),
                        ("probe_performed".to_string(), serde_json::json!(false)),
                    ]),
                    attributes: BTreeMap::from([(
                        "deferral_evidence".to_string(),
                        serde_json::json!({
                            "allocator_domain": domain,
                            "progress_owner_id": progress_owner_id,
                            "observed_release_epoch": observed_release_epoch,
                            "current_release_epoch": current_release_epoch,
                        }),
                    )]),
                    error: None,
                },
            ),
            ExecutorAdmissionQueueObservation::AllocatorCapacitySkippedUnchanged {
                request_id,
                domain,
                observed_release_epoch,
                current_release_epoch,
            }
            | ExecutorAdmissionQueueObservation::AllocatorCapacityResumed {
                request_id,
                domain,
                observed_release_epoch,
                current_release_epoch,
            } => self.write_executor_scheduler_profile_event(
                &request_id,
                "execution.allocator_capacity_wait",
                ProfileEventKind::Instant,
                ProfileStatus::Ok,
                None,
                || SchedulerProfileFields {
                    shape: BTreeMap::from([
                        (
                            "decision".to_string(),
                            serde_json::json!(if current_release_epoch > observed_release_epoch {
                                "release_observed"
                            } else {
                                "skipped_unchanged"
                            }),
                        ),
                        ("probe_performed".to_string(), serde_json::json!(false)),
                    ]),
                    attributes: BTreeMap::from([(
                        "deferral_evidence".to_string(),
                        serde_json::json!({
                            "allocator_domain": domain,
                            "observed_release_epoch": observed_release_epoch,
                            "current_release_epoch": current_release_epoch,
                        }),
                    )]),
                    error: None,
                },
            ),
            ExecutorAdmissionQueueObservation::PressureHoldReleased {
                episode_id,
                transition_ordinal,
                request_id,
                progress_owner_id,
                progress_baseline,
                progress_current,
                reason,
                previous_wait_condition,
                current_wait_condition,
                ticket,
            } => self.write_executor_scheduler_profile_event(
                &request_id,
                "vnext.execution_capacity_pressure_hold_released",
                ProfileEventKind::Instant,
                ProfileStatus::Ok,
                None,
                || SchedulerProfileFields {
                    shape: BTreeMap::from([
                        ("decision".to_string(), serde_json::json!(reason.as_str())),
                        (
                            "episode_id".to_string(),
                            serde_json::json!(episode_id.get()),
                        ),
                        (
                            "transition_ordinal".to_string(),
                            serde_json::json!(transition_ordinal.get()),
                        ),
                        ("waiting_ticket".to_string(), serde_json::json!(ticket)),
                        (
                            "progress_owner_id".to_string(),
                            serde_json::json!(progress_owner_id.0),
                        ),
                        (
                            "progress_baseline".to_string(),
                            serde_json::json!(progress_baseline.get()),
                        ),
                        (
                            "progress_current".to_string(),
                            serde_json::json!(progress_current.get()),
                        ),
                        (
                            "previous_wait_condition".to_string(),
                            serde_json::json!(previous_wait_condition),
                        ),
                        (
                            "current_wait_condition".to_string(),
                            serde_json::json!(current_wait_condition),
                        ),
                        ("admission_eligible".to_string(), serde_json::json!(true)),
                        ("probe_performed".to_string(), serde_json::json!(false)),
                        (
                            "prefill_submit_observed".to_string(),
                            serde_json::json!(false),
                        ),
                    ]),
                    attributes: BTreeMap::new(),
                    error: None,
                },
            ),
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
                || SchedulerProfileFields {
                    shape: BTreeMap::from([
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
                    attributes: BTreeMap::from([(
                        "deferral_evidence".to_string(),
                        serde_json::json!({
                            "action": deferral.action(),
                            "observed": epochs(deferral.observed()),
                            "current": epochs(current),
                            "wait_condition": deferral.wait_condition(),
                        }),
                    )]),
                    error: None,
                },
            ),
            ExecutorAdmissionQueueObservation::DecodeSkippedUnchanged {
                request_id,
                deferral,
                current,
                current_wait_sources,
            } => self.write_executor_scheduler_profile_event(
                &request_id,
                "vnext.decode_capacity_skipped_unchanged",
                ProfileEventKind::Instant,
                ProfileStatus::Ok,
                None,
                || SchedulerProfileFields {
                    shape: BTreeMap::from([
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
                    attributes: BTreeMap::from([(
                        "deferral_evidence".to_string(),
                        serde_json::json!({
                            "action": deferral.action(),
                            "observed": epochs(deferral.observed()),
                            "current": epochs(current),
                            "wait_condition": deferral.wait_condition(),
                            "current_wait_sources": current_wait_sources,
                        }),
                    )]),
                    error: None,
                },
            ),
            ExecutorAdmissionQueueObservation::DecodeResumed {
                request_id,
                deferral,
                current,
                current_wait_sources,
                exact_source_changed,
                policy_epoch_changed,
            } => self.write_executor_scheduler_profile_event(
                &request_id,
                "vnext.decode_capacity_resumed",
                ProfileEventKind::Instant,
                ProfileStatus::Ok,
                None,
                || SchedulerProfileFields {
                    shape: BTreeMap::from([
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
                    attributes: BTreeMap::from([(
                        "deferral_evidence".to_string(),
                        serde_json::json!({
                            "action": deferral.action(),
                            "observed": epochs(deferral.observed()),
                            "current": epochs(current),
                            "wait_condition": deferral.wait_condition(),
                            "current_wait_sources": current_wait_sources,
                        }),
                    )]),
                    error: None,
                },
            ),
            ExecutorAdmissionQueueObservation::PrefillSkippedUnchanged {
                request_id,
                deferral,
                current,
                current_wait_sources,
            } => self.write_executor_scheduler_profile_event(
                &request_id,
                "vnext.prefill_capacity_skipped_unchanged",
                ProfileEventKind::Instant,
                ProfileStatus::Ok,
                None,
                || SchedulerProfileFields {
                    shape: BTreeMap::from([
                        (
                            "decision".to_string(),
                            serde_json::json!("skipped_unchanged"),
                        ),
                        (
                            "prefill_submit_observed".to_string(),
                            serde_json::json!(false),
                        ),
                        ("probe_performed".to_string(), serde_json::json!(false)),
                    ]),
                    attributes: BTreeMap::from([(
                        "deferral_evidence".to_string(),
                        serde_json::json!({
                            "action": deferral.action(),
                            "observed": epochs(deferral.observed()),
                            "current": epochs(current),
                            "wait_condition": deferral.wait_condition(),
                            "current_wait_sources": current_wait_sources,
                        }),
                    )]),
                    error: None,
                },
            ),
            ExecutorAdmissionQueueObservation::PrefillResumed {
                request_id,
                deferral,
                current,
                current_wait_sources,
                exact_source_changed,
                policy_epoch_changed,
            } => self.write_executor_scheduler_profile_event(
                &request_id,
                "vnext.prefill_capacity_resumed",
                ProfileEventKind::Instant,
                ProfileStatus::Ok,
                None,
                || SchedulerProfileFields {
                    shape: BTreeMap::from([
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
                            "prefill_submit_observed".to_string(),
                            serde_json::json!(false),
                        ),
                        ("probe_performed".to_string(), serde_json::json!(false)),
                    ]),
                    attributes: BTreeMap::from([(
                        "deferral_evidence".to_string(),
                        serde_json::json!({
                            "action": deferral.action(),
                            "observed": epochs(deferral.observed()),
                            "current": epochs(current),
                            "wait_condition": deferral.wait_condition(),
                            "current_wait_sources": current_wait_sources,
                        }),
                    )]),
                    error: None,
                },
            ),
        }
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
                let valid_pressure = match pressure {
                    DynamicBackingPressure::DeviceCapacity(pressure) => matches!(
                        pressure.scope(),
                        DeviceCapacityPressureScope::PlanBudget
                            | DeviceCapacityPressureScope::ProcessWide
                    ),
                    DynamicBackingPressure::PoolResident(pressure) => {
                        pressure.resident_bytes() <= pressure.maximum_resident_bytes()
                            && pressure.available_bytes() < pressure.requested_bytes()
                    }
                };
                if wait_condition.coordinator_id().get() != current.coordinator_id.get()
                    || pressure.requested_bytes() == 0
                    || pressure.available_bytes() >= pressure.requested_bytes()
                    || !valid_pressure
                {
                    return Err(FerrumError::internal(format!(
                        "executor maintenance for {} reported invalid backing pressure",
                        deferral.request_id()
                    )));
                }
            }
            ExecutorPrefillMaintenanceOutcome::Maintained {
                current,
                pools_grown,
                allocated_bytes,
                pools_reclaimed,
                chunks_reclaimed,
                reclaimed_bytes,
                rebalance,
            } => {
                validate_current(*current)?;
                let rebalance_matches = match rebalance {
                    None => {
                        *pools_reclaimed == 0 && *chunks_reclaimed == 0 && *reclaimed_bytes == 0
                    }
                    Some(rebalance) => {
                        let detailed_chunks =
                            rebalance.pools().iter().try_fold(0_usize, |total, pool| {
                                total.checked_add(pool.chunks().len())
                            });
                        let detailed_bytes =
                            rebalance.pools().iter().try_fold(0_u64, |total, pool| {
                                total.checked_add(pool.reclaimed_bytes())
                            });
                        !rebalance.pools().is_empty()
                            && rebalance
                                .pools()
                                .iter()
                                .all(|pool| !pool.chunks().is_empty() && pool.reclaimed_bytes() > 0)
                            && rebalance.pools().len() == *pools_reclaimed
                            && rebalance.reclaimed_chunks() == *chunks_reclaimed
                            && rebalance.reclaimed_bytes() == *reclaimed_bytes
                            && detailed_chunks == Some(*chunks_reclaimed)
                            && detailed_bytes == Some(*reclaimed_bytes)
                    }
                };
                if current.capacity_epoch <= observed.capacity_epoch
                    || *pools_grown == 0
                    || *allocated_bytes == 0
                    || !rebalance_matches
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
            || SchedulerProfileFields {
                shape: BTreeMap::from([
                    ("outcome".to_string(), serde_json::json!(outcome_name)),
                    (
                        "duration_us".to_string(),
                        serde_json::json!(duration_to_us(elapsed)),
                    ),
                ]),
                attributes: BTreeMap::from([("maintenance_evidence".to_string(), evidence)]),
                error: error,
            },
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
                    if matches!(
                        &outcome,
                        ExecutorPrefillMaintenanceOutcome::WaitForRelease { .. }
                    ) {
                        // A normal GrowthRequired deferral may simply be cold
                        // materialization. Only authoritative pressure abandons
                        // optional sharing; drops happen after scheduler locks.
                        self.release_prefix_rendezvous_for_capacity_pressure();
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

    /// Fill-first is allowed to delay decode until its active target, so form
    /// that target's retained admission cohort before constructing a physical
    /// batch. Each round performs one bounded maintenance attempt per deferred
    /// request and then re-probes against fresh typed capacity evidence.
    async fn prepare_token_prefill_cohort(
        &self,
        hint: &ferrum_interfaces::BatchHint,
        mut planning: Option<&mut TimePlanningRound>,
    ) -> Result<bool> {
        let Some(configured_target) = self.config.scheduler.prefill_first_until_active else {
            return Ok(false);
        };
        let target = configured_target
            .min(hint.max_batch_size)
            .min(self.config.scheduler.max_running_requests);
        if target <= 1 {
            return Ok(false);
        }

        let mut progressed = false;
        for _ in 0..=target {
            let desired = self
                .scheduler
                .fill_first_dynamic_admission_limit(hint, target);
            if desired == 0 || self.scheduler.waiting_count() == 0 {
                break;
            }
            let (admitted, maintenance) = self
                .prepare_dynamic_admission_round(desired, planning.as_deref_mut())
                .await?;
            let maintenance =
                PendingPrefillMaintenance::new(self.model_executor.as_ref().as_ref(), maintenance);
            progressed |= admitted > 0;
            self.complete_typed_admission_failures().await?;
            if maintenance.is_empty() {
                break;
            }
            progressed = true;
            self.execute_executor_prefill_maintenance(maintenance.into_pending());
            self.complete_typed_admission_failures().await?;
        }

        let active = self.scheduler.active_count();
        let waiting = self.scheduler.waiting_count();
        Ok(progressed && active > 0 && active < target && waiting == 0)
    }

    async fn complete_typed_admission_failures(&self) -> Result<()> {
        let mut first_error = None;
        // Keep not-yet-processed failures in the scheduler's original queue.
        // Explicit-error completion transfers this one's frontier/slot before
        // its first await, so cancellation cannot lose the remaining owners.
        while let Some((request_id, error)) = self.scheduler.take_next_admission_failure() {
            warn!(
                request_id = %request_id,
                error = %error,
                "Typed prefill admission failed"
            );
            if let Err(error) =
                self.complete_request_detached(&request_id, FinishReason::Error, Some(error))
            {
                warn!(request_id = %request_id, %error, "Admission failure cleanup reported an error");
                first_error.get_or_insert(error);
            }
        }
        first_error.map_or(Ok(()), Err)
    }

    async fn defer_for_request_state_readiness(
        &self,
        deferral: ExecutorRequestStateDeferral,
        handles: Vec<(
            Arc<SequenceHandle>,
            ferrum_execution::token_execution::ExecutorSequence,
        )>,
    ) -> Result<()> {
        let slot = self.execution_readiness_waiters.reserve().await?;
        let registration = deferral.register_waiter()?;
        // Slot reservation can suspend. Use the captured original cohort;
        // the short index transaction excludes replacement publication while
        // the synchronous scheduler operation still accepts RequestIds.
        let receipt =
            self.sequences
                .with_current(handles.iter().map(|(handle, _)| handle), || {
                    self.scheduler
                        .defer_for_execution_readiness(deferral.request_ids())
                })?;
        if receipt.deferred_count() != deferral.request_ids().len() {
            receipt.wake().cancel();
            return Err(FerrumError::scheduler(format!(
                "Request-state readiness retained {} of {} frontiers",
                receipt.deferred_count(),
                deferral.request_ids().len()
            )));
        }
        let ticket_id = receipt.wake().ticket_id().get();
        self.execution_readiness_waiters.track(
            receipt.into_wake(),
            async move {
                registration
                    .wait_for_change()
                    .await
                    .map(|_| ())
                    .map_err(|error| error.to_string())
            },
            handles
                .iter()
                .map(|(handle, sequence)| ExecutionReadinessWaitTarget {
                    request: Arc::downgrade(handle),
                    sequence: sequence.clone(),
                })
                .collect(),
            Arc::clone(&self.work_notify),
            slot,
        )?;
        self.write_scheduler_trace_event(|| {
            serde_json::json!({
                "event": "scheduler_execution_request_state_defer",
                "ticket_id": ticket_id,
                "request_ids": deferral.request_ids(),
                "stage": deferral.stage(),
                "observed_change_epoch": deferral.hazard().observed_change_epoch(),
                "blockers": deferral.hazard().blockers(),
                "pending_waiters": self.execution_readiness_waiters.pending_count(),
                "scheduler": self.scheduler.trace_snapshot(),
            })
        });
        Ok(())
    }

    async fn complete_execution_readiness_failures(&self) -> Result<()> {
        self.execution_readiness_waiters.reap_finished();
        for failure in self.execution_readiness_waiters.take_failures() {
            for request in failure.requests {
                let Some(handle) = request.request.upgrade() else {
                    continue;
                };
                // Cancelling a scheduler wake does not retract a later waiter
                // error. A recompute keeps its logical handle but replaces the
                // executor incarnation, which an old failure cannot close.
                self.complete_request_exact_frontier_detached(
                    &handle,
                    &request.sequence,
                    FinishReason::Error,
                    Some(FerrumError::backend(format!(
                        "Request-state readiness ticket {} failed: {}",
                        failure.ticket_id, failure.message
                    ))),
                )?;
            }
        }
        Ok(())
    }

    fn scheduler_trace_enabled(&self) -> bool {
        self.legacy_scheduler_trace_jsonl.is_some()
    }

    pub(super) fn scheduler_trace_plan_stats(
        &self,
        batch: &ferrum_interfaces::BatchPlan,
    ) -> SchedulerTracePlanStats {
        #[cfg(test)]
        self.trace_payload_builds.fetch_add(1, Ordering::Relaxed);
        let mut stats = SchedulerTracePlanStats {
            batch_size: batch.size(),
            ..SchedulerTracePlanStats::default()
        };
        let mut decode_generated_tokens = Vec::new();
        let mut prefill_prompt_tokens = Vec::new();

        for scheduled_req in &batch.requests {
            let request_id = &scheduled_req.request.id;
            let scheduled_tokens = scheduled_req.tokens_to_process.unwrap_or(0);
            if scheduled_req.tokens_to_process.is_none() {
                stats.tokens_to_process_missing += 1;
            }
            stats.scheduled_tokens_total += scheduled_tokens;

            let phase = self.scheduler.trace_phase(request_id);
            let handle = self.sequences.get(request_id);
            let state = handle.as_ref().map(|handle| handle.lock());
            let seq = state.as_ref().and_then(|state| state.as_ref());
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

    pub(in crate::continuous_engine) fn write_scheduler_trace_event(
        &self,
        build_event: impl FnOnce() -> serde_json::Value,
    ) {
        let Some(journal) = &self.legacy_scheduler_trace_jsonl else {
            return;
        };
        #[cfg(test)]
        self.trace_payload_builds.fetch_add(1, Ordering::Relaxed);
        if let Err(error) = journal.try_enqueue(build_event()) {
            let message = format!("enqueue scheduler trace: {error}");
            let error = if matches!(
                error.kind(),
                ferrum_journal::JsonlJournalErrorKind::Encoding
                    | ferrum_journal::JsonlJournalErrorKind::Panic
                    | ferrum_journal::JsonlJournalErrorKind::Config
            ) {
                ExecutionEventSinkError::contract(message)
            } else {
                ExecutionEventSinkError::transport(message)
            };
            self.report_observation_failure(error);
        }
    }

    // ── iteration loop ─────────────────────────────────────────────────

    /// Run one iteration: ask the scheduler for a batch, then process it.
    pub(super) async fn run_iteration(&self) -> Result<EngineIterationOutcome> {
        let lock_wait_start = Instant::now();
        let iteration_guard = self.iteration_lock.lock().await;
        self.record_iteration_lock_wait(lock_wait_start.elapsed());
        let close_error = self.drain_execution_closes().err();
        let output_control_error = self.output_tasks.take_control_error();
        if output_control_error.is_some() {
            counter!("ferrum.engine.completion_control_errors_total").increment(1);
        }
        let failure_completion = self.complete_failed_execution_recomputes().await;
        if let Some(error) = close_error.or(output_control_error) {
            return Err(error);
        }
        failure_completion?;
        self.cancel_abandoned_requests().await?;
        self.complete_execution_readiness_failures().await?;
        self.prepare_prefix_rendezvous()?;

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

        let sched_t0 = Instant::now();
        // A planning iteration may contain multiple asynchronous admission
        // rounds. Keep one immutable prediction version across all of them;
        // prediction queries still use their current monotonic time.
        let prefill_shape_policy = self.model_executor.prefill_shape_policy();
        let mut planning = self.config.scheduler.slo.map(|policy| {
            let mut round = TimePlanningRound::new(
                self.cost_snapshot_source
                    .as_ref()
                    .expect("feedback validated before startup")
                    .load(),
                policy,
                prefill_shape_policy,
                match self.config.batching.prefill_decode_execution {
                    ferrum_types::PrefillDecodeExecution::Split => {
                        ferrum_scheduler::cost_prediction::CostExecution::SequentialPrefillThenDecode
                    }
                    ferrum_types::PrefillDecodeExecution::Mixed => {
                        ferrum_scheduler::cost_prediction::CostExecution::SingleInvocation
                    }
                },
            );
            self.capture_planning_request_facts(&mut round, policy.planning.max_participants);
            round
        });
        if self
            .prepare_token_prefill_cohort(&hint, planning.as_mut())
            .await?
        {
            return Ok(EngineIterationOutcome::Progressed);
        }
        let selection_intent = self.scheduler.prepare_token_batch_selection(hint);
        let (_, prefill_maintenance) = self
            .prepare_dynamic_admission_round(hint_max_batch_size, planning.as_mut())
            .await?;
        let prefill_maintenance = PendingPrefillMaintenance::new(
            self.model_executor.as_ref().as_ref(),
            prefill_maintenance,
        );
        self.complete_typed_admission_failures().await?;
        self.refresh_prefix_rendezvous()?;
        if self.model_executor.supports_plan_runtime_prefix_restore() {
            self.restore_admitted_prefixes().await?;
            self.refresh_prefix_rendezvous()?;
        }
        let (scheduled, observations) = {
            let mut availability = self.dynamic_admission_availability.lock();
            let epochs = self
                .model_executor
                .write_execution_capacity_snapshot(&mut availability)?;
            let wake = epochs.map(|epochs| {
                AdmissionWakeSnapshot::new(
                    AdmissionWakeEpochs::new(
                        epochs.coordinator_id,
                        epochs.release_epoch,
                        epochs.capacity_epoch,
                        0,
                    ),
                    &availability,
                )
            });
            let mut observations = Vec::new();
            let scheduled = self.scheduler.next_batch_with_token_admissions(
                selection_intent,
                wake,
                &mut |observation| observations.push(observation),
                planning.as_mut(),
            );
            (scheduled, observations)
        };
        for observation in observations {
            self.trace_executor_admission_queue_observation(observation);
        }
        let nb_result = scheduled?;
        self.complete_typed_admission_failures().await?;
        let sched_elapsed = sched_t0.elapsed();
        self.record_scheduling_time(sched_elapsed);

        let retry_at = nb_result.retry_at;
        let time_blocked = nb_result.time_blocked;
        let batch = match nb_result.batch {
            Some(b) => b,
            None => {
                self.execute_executor_prefill_maintenance(prefill_maintenance.into_pending());
                self.complete_typed_admission_failures().await?;
                if trace_enabled {
                    let none_streak = self
                        .scheduler_trace_none_streak
                        .fetch_add(1, Ordering::Relaxed)
                        + 1;
                    if none_streak <= 4 || none_streak.is_multiple_of(128) {
                        self.write_scheduler_trace_event(|| {
                            serde_json::json!({
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
                            })
                        });
                    }
                }
                {
                    if let Some(observed) = self.scheduler.passive_capacity_wait_condition()? {
                        let registration = self
                            .model_executor
                            .register_execution_capacity_waiter(&observed)?
                            .ok_or_else(|| {
                                FerrumError::scheduler(
                                    "plan runtime did not register its capacity waiter",
                                )
                            })?;
                        return Ok(EngineIterationOutcome::Waiting {
                            capacity: Some(registration),
                            retry_at,
                        });
                    }
                    if self.scheduler.all_active_execution_readiness_blocked()
                        || self.scheduler.all_active_allocator_capacity_blocked()
                    {
                        return Ok(EngineIterationOutcome::Waiting {
                            capacity: None,
                            retry_at,
                        });
                    }
                }
                if self.scheduler.active_count() == 0
                    && self.scheduler.waiting_for_confirmed_capacity_release()
                {
                    return Ok(EngineIterationOutcome::Waiting {
                        capacity: None,
                        retry_at,
                    });
                }
                if time_blocked {
                    return Ok(EngineIterationOutcome::Waiting {
                        capacity: None,
                        retry_at,
                    });
                }
                if self.scheduler.active_count() > 0 {
                    return Ok(EngineIterationOutcome::Progressed);
                }
                return if self.scheduler.waiting_count() == 0 {
                    Ok(EngineIterationOutcome::Waiting {
                        capacity: None,
                        retry_at,
                    })
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

        // Request publication only needs to be atomic with planning. The
        // selected batch owns its request authorities, so new ingress may be
        // published while the device executes this wave.
        drop(iteration_guard);
        let process_t0 = Instant::now();
        let r = self
            .process_batch_with_shape(
                &batch,
                prefill_shape_policy,
                planning
                    .as_ref()
                    .is_some_and(|round| round.selection_is_exact(batch.size())),
            )
            .await;
        let process_elapsed = process_t0.elapsed();
        self.record_model_execution_time(process_elapsed);
        self.execute_executor_prefill_maintenance(prefill_maintenance.into_pending());
        self.complete_typed_admission_failures().await?;
        if trace_enabled {
            let prefill_tokens_after = self.total_prefill_tokens.load(Ordering::Relaxed);
            let decode_tokens_after = self.total_decode_tokens.load(Ordering::Relaxed);
            self.write_scheduler_trace_event(|| serde_json::json!({
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
                self.record_host_observation(EngineHostObservation::Iteration {
                    iteration,
                    batch_size: batch.size(),
                    started: t0,
                    scheduled: ts,
                    completed: Instant::now(),
                    error: r.as_ref().err(),
                });
            }
        }
        r?;
        Ok(EngineIterationOutcome::Progressed)
    }
}
