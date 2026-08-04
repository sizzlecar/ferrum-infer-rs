use super::*;

fn vnext_execution_event_name(kind: VNextExecutionEventKind) -> &'static str {
    match kind {
        VNextExecutionEventKind::RequestAccepted => "request_accepted",
        VNextExecutionEventKind::PlanBuilt => "plan_built",
        VNextExecutionEventKind::FrameStarted => "frame_started",
        VNextExecutionEventKind::NodeStarted => "node_started",
        VNextExecutionEventKind::OperationSubmitted => "operation_submitted",
        VNextExecutionEventKind::NodeRetired => "node_retired",
        VNextExecutionEventKind::FrameCompleted => "frame_completed",
        VNextExecutionEventKind::FailureObserved => "failure_observed",
        VNextExecutionEventKind::SequenceCompleted => "sequence_completed",
        VNextExecutionEventKind::SequenceAborted => "sequence_aborted",
        VNextExecutionEventKind::RequestCompleted => "request_completed",
        VNextExecutionEventKind::RequestFailed => "request_failed",
    }
}

fn compact_basic_node_identity_field(field: &str) -> bool {
    matches!(
        field,
        "resource_pool_identity_fingerprint"
            | "provisioning_run_id"
            | "provisioning_request_id"
            | "transaction_id"
            | "runtime_implementation_fingerprint"
            | "active_sequence_fingerprint"
            | "completed_sequence_fingerprint"
            | "aborted_sequence_fingerprint"
            | "resource_id"
            | "resource_batch_fingerprint"
    )
}

struct VNextProfileEventContext {
    entrypoint: ProfileEntrypoint,
    model: String,
    backend_device: String,
    backend_type: String,
    profile_detail: ObservabilityProfileDetail,
    capture_policy: ExecutionEventCapturePolicy,
}

#[derive(Clone)]
struct DeferredVNextProfileEvent {
    event: ExecutionEvent,
    timestamp: chrono::DateTime<chrono::Utc>,
    context: Arc<VNextProfileEventContext>,
}

enum SchedulerTraceRecord {
    Profile(FerrumProfileEvent),
    DeferredVNext(DeferredVNextProfileEvent),
}

impl serde::Serialize for SchedulerTraceRecord {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Profile(event) => serde::Serialize::serialize(event, serializer),
            Self::DeferredVNext(record) => {
                let event = record
                    .context
                    .profile_event(&record.event, record.timestamp.to_owned())
                    .map_err(serde::ser::Error::custom)?;
                serde::Serialize::serialize(&event, serializer)
            }
        }
    }
}

#[derive(Clone)]
pub(super) struct SchedulerTraceJournal {
    inner: JsonlJournal<SchedulerTraceRecord>,
}

impl SchedulerTraceJournal {
    pub(super) fn create(path: impl Into<PathBuf>) -> std::io::Result<Self> {
        JsonlJournal::create(path).map(|inner| Self { inner })
    }

    pub(super) fn enqueue(
        &self,
        event: FerrumProfileEvent,
    ) -> std::result::Result<(), JsonlJournalError> {
        self.inner.enqueue(SchedulerTraceRecord::Profile(event))
    }

    pub(super) fn enqueue_batch(
        &self,
        events: Vec<FerrumProfileEvent>,
    ) -> std::result::Result<(), JsonlJournalError> {
        self.inner.enqueue_batch(
            events
                .into_iter()
                .map(SchedulerTraceRecord::Profile)
                .collect(),
        )
    }

    fn enqueue_deferred_vnext(
        &self,
        event: DeferredVNextProfileEvent,
    ) -> std::result::Result<(), JsonlJournalError> {
        self.inner
            .enqueue(SchedulerTraceRecord::DeferredVNext(event))
    }

    fn enqueue_deferred_vnext_batch(
        &self,
        events: Vec<DeferredVNextProfileEvent>,
    ) -> std::result::Result<(), JsonlJournalError> {
        self.inner.enqueue_batch(
            events
                .into_iter()
                .map(SchedulerTraceRecord::DeferredVNext)
                .collect(),
        )
    }

    #[cfg(test)]
    pub(super) fn flush(&self) -> std::result::Result<(), JsonlJournalError> {
        self.inner.flush()
    }

    pub(super) fn close(&self) -> std::result::Result<(), JsonlJournalError> {
        self.inner.close()
    }

    pub(super) fn path(&self) -> &Path {
        self.inner.path()
    }
}

pub(super) struct VNextProfileExecutionEventSink {
    journals: Box<[SchedulerTraceJournal]>,
    context: Arc<VNextProfileEventContext>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum DeviceCommandTimingUnavailable {
    Backend(DeviceTimingUnavailableReason),
    MissingSpan,
}

impl DeviceCommandTimingUnavailable {
    fn label(self) -> String {
        match self {
            Self::Backend(reason) => format!("{reason:?}").to_ascii_lowercase(),
            Self::MissingSpan => "missing_span".to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub(super) struct DeviceCommandTimingProjection<'a> {
    pub(super) physical_span: Option<&'a DeviceSubmissionExecutionSpan>,
    pub(super) command_measurement: Option<&'a DeviceExecutionSpanMeasurement>,
    pub(super) status: &'static str,
    pub(super) unavailable: Option<DeviceCommandTimingUnavailable>,
}

pub(super) fn project_device_command_timing<'a>(
    terminal_timing: &'a DeviceTimingMeasurement<DeviceSubmissionExecutionTiming>,
    command_index: u32,
) -> DeviceCommandTimingProjection<'a> {
    match terminal_timing {
        DeviceTimingMeasurement::NotRequested => DeviceCommandTimingProjection {
            physical_span: None,
            command_measurement: None,
            status: "not_requested",
            unavailable: None,
        },
        DeviceTimingMeasurement::Unavailable(reason) => DeviceCommandTimingProjection {
            physical_span: None,
            command_measurement: None,
            status: "unavailable",
            unavailable: Some(DeviceCommandTimingUnavailable::Backend(*reason)),
        },
        DeviceTimingMeasurement::Measured(timing) => {
            let physical_span = timing.span_for_command(command_index);
            let Some(physical_span) = physical_span else {
                return DeviceCommandTimingProjection {
                    physical_span: None,
                    command_measurement: None,
                    status: "unavailable",
                    unavailable: Some(DeviceCommandTimingUnavailable::MissingSpan),
                };
            };
            let measured = physical_span.measurement().elapsed_ns().is_some();
            let replayed = physical_span.kind() == DeviceExecutionSpanKind::ReusableExecutable;
            DeviceCommandTimingProjection {
                physical_span: Some(physical_span),
                command_measurement: (!replayed).then(|| physical_span.measurement()),
                status: if measured {
                    if replayed {
                        "covered_by_physical_span"
                    } else {
                        "measured"
                    }
                } else {
                    "unavailable"
                },
                unavailable: physical_span
                    .measurement()
                    .unavailable_reason()
                    .map(DeviceCommandTimingUnavailable::Backend),
            }
        }
    }
}

impl VNextProfileExecutionEventSink {
    #[cfg(test)]
    pub(super) fn new(
        journal: SchedulerTraceJournal,
        entrypoint: ProfileEntrypoint,
        config: &EngineConfig,
    ) -> Self {
        Self::with_journals(vec![journal], entrypoint, config)
    }

    pub(super) fn with_journals(
        journals: Vec<SchedulerTraceJournal>,
        entrypoint: ProfileEntrypoint,
        config: &EngineConfig,
    ) -> Self {
        assert!(
            !journals.is_empty(),
            "vNext profile sink requires at least one JSONL journal"
        );
        Self {
            journals: journals.into_boxed_slice(),
            context: Arc::new(VNextProfileEventContext {
                entrypoint,
                model: config.model.model_id.to_string(),
                backend_device: format!("{:?}", config.backend.device),
                backend_type: format!("{:?}", config.backend.backend_type),
                profile_detail: config.runtime.profile_detail,
                capture_policy: if matches!(
                    config.runtime.profile_detail,
                    ObservabilityProfileDetail::Resource
                        | ObservabilityProfileDetail::Latency
                        | ObservabilityProfileDetail::Kernel
                        | ObservabilityProfileDetail::Replay
                        | ObservabilityProfileDetail::Verify
                        | ObservabilityProfileDetail::Full
                ) {
                    ExecutionEventCapturePolicy::AllFrames
                } else {
                    ExecutionEventCapturePolicy::FirstFramePerRequest
                },
            }),
        }
    }

    fn enqueue_profile_batch(
        &self,
        events: Vec<FerrumProfileEvent>,
    ) -> std::result::Result<(), ExecutionEventSinkError> {
        for journal in &self.journals {
            journal.enqueue_batch(events.clone()).map_err(|error| {
                ExecutionEventSinkError::new(format!(
                    "enqueue vNext profile events to {}: {error}",
                    journal.path().display()
                ))
            })?;
        }
        Ok(())
    }

    fn enqueue_deferred_event(
        &self,
        event: DeferredVNextProfileEvent,
    ) -> std::result::Result<(), ExecutionEventSinkError> {
        for journal in &self.journals {
            journal
                .enqueue_deferred_vnext(event.clone())
                .map_err(|error| {
                    ExecutionEventSinkError::new(format!(
                        "enqueue deferred vNext profile event to {}: {error}",
                        journal.path().display()
                    ))
                })?;
        }
        Ok(())
    }

    fn enqueue_deferred_batch(
        &self,
        events: Vec<DeferredVNextProfileEvent>,
    ) -> std::result::Result<(), ExecutionEventSinkError> {
        for journal in &self.journals {
            journal
                .enqueue_deferred_vnext_batch(events.clone())
                .map_err(|error| {
                    ExecutionEventSinkError::new(format!(
                        "enqueue deferred vNext profile batch to {}: {error}",
                        journal.path().display()
                    ))
                })?;
        }
        Ok(())
    }
}

impl VNextProfileEventContext {
    fn capture_policy_for_request(
        &self,
        origin: ExecutorRequestOrigin,
    ) -> ExecutionEventCapturePolicy {
        if origin == ExecutorRequestOrigin::Startup
            && matches!(
                self.profile_detail,
                ObservabilityProfileDetail::Off | ObservabilityProfileDetail::Basic
            )
        {
            ExecutionEventCapturePolicy::LifecycleOnly
        } else {
            self.capture_policy
        }
    }

    fn profile_event(
        &self,
        event: &ExecutionEvent,
        timestamp: chrono::DateTime<chrono::Utc>,
    ) -> std::result::Result<FerrumProfileEvent, ExecutionEventSinkError> {
        let identity = event.identity().parts();
        let request_origin =
            ExecutorRequestOrigin::from_namespaced_request_identity(identity.request_id.as_str());
        let capture_policy = request_origin
            .map(|origin| self.capture_policy_for_request(origin))
            .unwrap_or(self.capture_policy);
        let event_name = vnext_execution_event_name(event.kind());
        let compact_basic = self.profile_detail == ObservabilityProfileDetail::Basic;
        let compact_repeated_node = compact_basic
            && matches!(
                event.kind(),
                VNextExecutionEventKind::NodeStarted | VNextExecutionEventKind::NodeRetired
            );
        let failure = match event.detail() {
            ExecutionEventDetail::Failure(failure) => Some(ProfileError {
                kind: failure.failure().code().to_string(),
                message: failure.failure().message().to_string(),
                blocking: true,
            }),
            ExecutionEventDetail::FailureTerminal {
                first_failure_fingerprint,
            } => Some(ProfileError {
                kind: "vnext_request_failed".to_string(),
                message: format!("request terminated after failure {first_failure_fingerprint}"),
                blocking: false,
            }),
            _ => None,
        };
        let status = if failure.is_some() {
            ProfileStatus::Failure
        } else {
            ProfileStatus::Ok
        };
        let mut shape = BTreeMap::from([(
            "execution_sequence".to_string(),
            serde_json::json!(identity.sequence),
        )]);
        if let Some(frame_id) = identity.frame_id {
            shape.insert("frame_id".to_string(), serde_json::json!(frame_id.get()));
        }
        if let Some(invocation_id) = identity.node_invocation_id {
            shape.insert(
                "node_invocation_id".to_string(),
                serde_json::json!(invocation_id.get()),
            );
        }
        if let ExecutionEventDetail::Counters { input, output } = event.detail() {
            shape.insert("event_input_count".to_string(), serde_json::json!(input));
            shape.insert("event_output_count".to_string(), serde_json::json!(output));
        }
        let mut attributes = BTreeMap::from([
            (
                "actual_model_smoke".to_string(),
                serde_json::json!(matches!(
                    self.entrypoint,
                    ProfileEntrypoint::Run | ProfileEntrypoint::Serve
                )),
            ),
            (
                "diagnostic_only".to_string(),
                serde_json::json!(self.profile_detail.diagnostic_only()),
            ),
            ("l0_only".to_string(), serde_json::json!(false)),
            (
                "profile_detail".to_string(),
                serde_json::json!(self.profile_detail.as_str()),
            ),
            (
                "execution_capture_policy".to_string(),
                serde_json::json!(capture_policy.as_str()),
            ),
            (
                "execution_event_kind".to_string(),
                serde_json::json!(event_name),
            ),
            (
                "execution_phase".to_string(),
                serde_json::json!(format!("{:?}", event.phase()).to_ascii_lowercase()),
            ),
            (
                "execution_trace_source".to_string(),
                serde_json::json!("vnext"),
            ),
            (
                "monotonic_nanos_since_run_start".to_string(),
                serde_json::json!(event.timestamp().nanos_since_run_start),
            ),
            (
                "run_id".to_string(),
                serde_json::json!(identity.run_id.to_string()),
            ),
            (
                "span_id".to_string(),
                serde_json::json!(identity.span_id.to_string()),
            ),
            (
                "execution_identity_version".to_string(),
                serde_json::json!(identity.version.to_string()),
            ),
        ]);
        if !compact_basic {
            attributes.insert(
                "backend_device".to_string(),
                serde_json::json!(self.backend_device),
            );
            attributes.insert(
                "backend_type".to_string(),
                serde_json::json!(self.backend_type),
            );
        }
        if !compact_basic || failure.is_some() {
            attributes.insert(
                "execution_identity".to_string(),
                serde_json::to_value(identity).map_err(|error| {
                    ExecutionEventSinkError::new(format!(
                        "failed to serialize canonical vNext execution identity: {error}"
                    ))
                })?,
            );
            attributes.insert(
                "execution_request_id".to_string(),
                serde_json::json!(identity.request_id.to_string()),
            );
        }
        if !compact_basic || !identity.async_links.is_empty() {
            attributes.insert(
                "async_links".to_string(),
                serde_json::json!(identity
                    .async_links
                    .iter()
                    .map(ToString::to_string)
                    .collect::<Vec<_>>()),
            );
        }
        if let Some(origin) = request_origin {
            attributes.insert(
                "execution_request_origin".to_string(),
                serde_json::json!(origin.namespace()),
            );
        }
        match event.detail() {
            ExecutionEventDetail::Failure(failure) => {
                attributes.insert("first_failure_event".to_string(), serde_json::json!(true));
                attributes.insert(
                    "first_failure_fingerprint".to_string(),
                    serde_json::json!(failure.fingerprint()),
                );
                attributes.insert(
                    "failure_domain".to_string(),
                    serde_json::json!(
                        format!("{:?}", failure.failure().domain()).to_ascii_lowercase()
                    ),
                );
                attributes.insert(
                    "failure_retryable".to_string(),
                    serde_json::json!(failure.failure().retryable()),
                );
                if let Some(snapshot) = failure.failure().resource_snapshot() {
                    attributes.insert(
                        "plan_runtime_resource_snapshot".to_string(),
                        serde_json::to_value(snapshot).map_err(|error| {
                            ExecutionEventSinkError::new(format!(
                                "failed to serialize vNext failure resource snapshot: {error}"
                            ))
                        })?,
                    );
                }
            }
            ExecutionEventDetail::FailureTerminal {
                first_failure_fingerprint,
            } => {
                attributes.insert(
                    "terminal_failure_event".to_string(),
                    serde_json::json!(true),
                );
                attributes.insert(
                    "first_failure_fingerprint".to_string(),
                    serde_json::json!(first_failure_fingerprint),
                );
            }
            _ => {}
        }
        for (key, value) in [
            (
                "plan_id",
                identity.plan_id.as_ref().map(ToString::to_string),
            ),
            (
                "plan_hash",
                identity.plan_hash.as_ref().map(ToString::to_string),
            ),
            (
                "node_id",
                identity.node_id.as_ref().map(ToString::to_string),
            ),
            (
                "operation_id",
                identity.operation_id.as_ref().map(ToString::to_string),
            ),
            (
                "provider_id",
                identity.provider_id.as_ref().map(ToString::to_string),
            ),
            (
                "device_id",
                identity.device_id.as_ref().map(ToString::to_string),
            ),
            (
                "parent_span_id",
                identity.parent_span_id.as_ref().map(ToString::to_string),
            ),
            (
                "resource_pool_id",
                identity.resource_pool_id.as_ref().map(ToString::to_string),
            ),
            (
                "resource_pool_identity_fingerprint",
                identity.resource_pool_identity_fingerprint.clone(),
            ),
            (
                "provisioning_run_id",
                identity
                    .provisioning_run_id
                    .as_ref()
                    .map(ToString::to_string),
            ),
            (
                "provisioning_request_id",
                identity
                    .provisioning_request_id
                    .as_ref()
                    .map(ToString::to_string),
            ),
            (
                "transaction_id",
                identity.transaction_id.as_ref().map(ToString::to_string),
            ),
            (
                "runtime_implementation_fingerprint",
                identity.runtime_implementation_fingerprint.clone(),
            ),
            (
                "active_sequence_fingerprint",
                identity.active_sequence_fingerprint.clone(),
            ),
            (
                "completed_sequence_fingerprint",
                identity.completed_sequence_fingerprint.clone(),
            ),
            (
                "aborted_sequence_fingerprint",
                identity.aborted_sequence_fingerprint.clone(),
            ),
            (
                "resource_id",
                identity.resource_id.as_ref().map(ToString::to_string),
            ),
            (
                "resource_batch_fingerprint",
                identity.resource_batch_fingerprint.clone(),
            ),
        ] {
            if compact_repeated_node && compact_basic_node_identity_field(key) {
                continue;
            }
            if let Some(value) = value {
                attributes.insert(key.to_string(), serde_json::json!(value));
            }
        }
        for (key, value) in [
            (
                "active_sequence_slot",
                identity.active_sequence_slot.map(u64::from),
            ),
            ("admission_generation", identity.admission_generation),
            ("activation_epoch", identity.activation_epoch),
            ("resource_generation", identity.resource_generation),
        ] {
            if let Some(value) = value {
                attributes.insert(key.to_string(), serde_json::json!(value));
            }
        }
        let resource = match event.detail() {
            ExecutionEventDetail::Failure(failure) => failure
                .failure()
                .resource_snapshot()
                .map(|snapshot| {
                    let available = snapshot.available_bytes().map_err(|error| {
                        ExecutionEventSinkError::new(format!(
                            "invalid vNext failure resource availability: {error}"
                        ))
                    })?;
                    let to_i64 = |value: u64, label: &str| {
                        i64::try_from(value).map_err(|_| {
                            ExecutionEventSinkError::new(format!(
                                "vNext failure {label} exceeds profile i64 range"
                            ))
                        })
                    };
                    Ok::<_, ExecutionEventSinkError>(ResourceTraceEvent {
                        owner_kind: if identity.resource_pool_id.is_some() {
                            "resource_pool".to_string()
                        } else {
                            "request".to_string()
                        },
                        owner_id: identity
                            .resource_pool_id
                            .as_ref()
                            .map_or_else(|| identity.request_id.to_string(), ToString::to_string),
                        resource_kind: "plan_runtime_memory".to_string(),
                        action: ResourceAction::Reject,
                        amount: None,
                        before: Some(to_i64(available, "available bytes")?),
                        after: Some(to_i64(available, "available bytes")?),
                        capacity: Some(to_i64(
                            snapshot.usable_capacity_bytes(),
                            "usable capacity",
                        )?),
                        underflow_amount: None,
                        reason: Some(failure.failure().message().to_string()),
                        error_kind: Some(failure.failure().code().to_string()),
                        message: Some(failure.failure().message().to_string()),
                        resource_error_kind: Some("plan_runtime_resource_failure".to_string()),
                    })
                })
                .transpose()?,
            _ => None,
        };
        let profile = FerrumProfileEvent {
            schema_version: OBSERVABILITY_PROFILE_SCHEMA_VERSION,
            ts_unix_nanos: timestamp
                .timestamp_nanos_opt()
                .unwrap_or_else(|| timestamp.timestamp_micros() * 1_000),
            event_id: format!(
                "evt-vnext-{}-{}-{}",
                identity.run_id, identity.request_id, identity.sequence
            ),
            request_id: identity.request_id.to_string(),
            correlation_id: Some(identity.request_id.to_string()),
            entrypoint: self.entrypoint,
            backend: "actual".to_string(),
            runtime_preset_hash: ENGINE_RUNTIME_TRACE_PRESET_HASH.to_string(),
            phase: format!("vnext.{event_name}"),
            event_kind: if failure.is_some() {
                ProfileEventKind::Error
            } else {
                ProfileEventKind::Instant
            },
            timestamp,
            status,
            model: Some(self.model.clone()),
            duration_us: None,
            memory: None,
            resource,
            error: failure,
            replay: None,
            shape,
            backend_detail: Some(BTreeMap::from([
                (
                    "backend_device".to_string(),
                    serde_json::json!(self.backend_device),
                ),
                (
                    "backend_type".to_string(),
                    serde_json::json!(self.backend_type),
                ),
            ])),
            attributes,
        };
        profile.validate().map_err(|error| {
            ExecutionEventSinkError::new(format!("invalid vNext profile event: {error}"))
        })?;
        Ok(profile)
    }

    fn execution_resource_maintenance_profile_event(
        &self,
        maintenance: &BoundExecutionResourceMaintenance,
        timestamp: chrono::DateTime<chrono::Utc>,
    ) -> std::result::Result<FerrumProfileEvent, ExecutionEventSinkError> {
        let first = maintenance.participants().first().ok_or_else(|| {
            ExecutionEventSinkError::new("execution resource maintenance has no participant")
        })?;
        let receipt = maintenance.receipt();
        let allocated_bytes = receipt
            .growths()
            .iter()
            .try_fold(0_u64, |total, growth| {
                total.checked_add(growth.chunk_bytes())
            })
            .ok_or_else(|| {
                ExecutionEventSinkError::new(
                    "execution resource maintenance allocated byte count overflow",
                )
            })?;
        let (pools_reclaimed, chunks_reclaimed, reclaimed_bytes) =
            receipt.rebalance().map_or((0, 0, 0), |rebalance| {
                (
                    rebalance.pools().len(),
                    rebalance.reclaimed_chunks(),
                    rebalance.reclaimed_bytes(),
                )
            });
        let maintenance_evidence = serde_json::json!({
            "schema_version": maintenance.schema_version(),
            "outcome": "maintained",
            "stage": maintenance.stage().as_str(),
            "coordinator_id": receipt.coordinator_id(),
            "pools_grown": receipt.growths().len(),
            "allocated_bytes": allocated_bytes,
            "pools_reclaimed": pools_reclaimed,
            "chunks_reclaimed": chunks_reclaimed,
            "reclaimed_bytes": reclaimed_bytes,
            "rebalance": receipt.rebalance(),
            "receipt": receipt,
            "event_fingerprint": maintenance.fingerprint(),
            "participants": maintenance.participants(),
        });
        let timestamp_nanos = timestamp
            .timestamp_nanos_opt()
            .unwrap_or_else(|| timestamp.timestamp_micros() * 1_000);
        let profile = FerrumProfileEvent {
            schema_version: OBSERVABILITY_PROFILE_SCHEMA_VERSION,
            ts_unix_nanos: timestamp_nanos,
            event_id: format!(
                "evt-vnext-execution-resource-maintenance-{}",
                maintenance.fingerprint()
            ),
            request_id: first.request_id().to_string(),
            correlation_id: Some(maintenance.fingerprint().to_owned()),
            entrypoint: self.entrypoint,
            backend: "actual".to_string(),
            runtime_preset_hash: ENGINE_RUNTIME_TRACE_PRESET_HASH.to_string(),
            phase: "vnext.execution_backing_maintenance".to_string(),
            event_kind: ProfileEventKind::Instant,
            timestamp,
            status: ProfileStatus::Ok,
            model: Some(self.model.clone()),
            duration_us: None,
            memory: None,
            resource: None,
            error: None,
            replay: None,
            shape: BTreeMap::from([
                (
                    "allocated_bytes".to_string(),
                    serde_json::json!(allocated_bytes),
                ),
                (
                    "participant_count".to_string(),
                    serde_json::json!(maintenance.participants().len()),
                ),
                (
                    "pools_grown".to_string(),
                    serde_json::json!(receipt.growths().len()),
                ),
                (
                    "pools_reclaimed".to_string(),
                    serde_json::json!(pools_reclaimed),
                ),
                (
                    "stage".to_string(),
                    serde_json::json!(maintenance.stage().as_str()),
                ),
            ]),
            backend_detail: Some(BTreeMap::from([
                (
                    "backend_device".to_string(),
                    serde_json::json!(self.backend_device),
                ),
                (
                    "backend_type".to_string(),
                    serde_json::json!(self.backend_type),
                ),
            ])),
            attributes: BTreeMap::from([
                (
                    "execution_trace_source".to_string(),
                    serde_json::json!("vnext_resource_maintenance"),
                ),
                ("maintenance_evidence".to_string(), maintenance_evidence),
                (
                    "plan_hash".to_string(),
                    serde_json::json!(maintenance.plan().plan_hash()),
                ),
                (
                    "plan_id".to_string(),
                    serde_json::json!(maintenance.plan().plan_id()),
                ),
                ("run_id".to_string(), serde_json::json!(first.run_id())),
            ]),
        };
        profile.validate().map_err(|error| {
            ExecutionEventSinkError::new(format!(
                "invalid vNext execution resource maintenance profile event: {error}"
            ))
        })?;
        Ok(profile)
    }

    fn physical_device_submission_timing_event(
        &self,
        completion: &OperationCompletionReceipt,
        timestamp: chrono::DateTime<chrono::Utc>,
    ) -> std::result::Result<Option<FerrumProfileEvent>, ExecutionEventSinkError> {
        let submission = completion.submission();
        let batch = submission.batch_identity();
        let first = batch
            .nodes()
            .first()
            .and_then(|node| node.participants().first())
            .ok_or_else(|| {
                ExecutionEventSinkError::new(
                    "physical device submission timing has no batch participant",
                )
            })?;
        let first_identity = first.identity().parts();
        let mut participant_request_ids = batch
            .nodes()
            .iter()
            .flat_map(|node| node.participants())
            .map(|participant| participant.identity().parts().request_id.to_string())
            .collect::<Vec<_>>();
        participant_request_ids.sort();
        participant_request_ids.dedup();

        let (command_count, spans, timing_status, unavailable_reason) =
            match completion.submission_timing() {
                DeviceTimingMeasurement::NotRequested => return Ok(None),
                DeviceTimingMeasurement::Unavailable(reason) => (
                    0_u32,
                    None,
                    "unavailable",
                    Some(format!("{reason:?}").to_ascii_lowercase()),
                ),
                DeviceTimingMeasurement::Measured(timing) => (
                    timing.command_count(),
                    Some(timing.spans()),
                    "measured",
                    None,
                ),
            };
        let eager_span_count = spans.map_or(0, |spans| {
            spans
                .iter()
                .filter(|span| span.kind() == DeviceExecutionSpanKind::EagerCommand)
                .count()
        });
        let reusable_span_count = spans.map_or(0, |spans| {
            spans
                .iter()
                .filter(|span| span.kind() == DeviceExecutionSpanKind::ReusableExecutable)
                .count()
        });
        let shape = BTreeMap::from([
            (
                "command_count".to_string(),
                serde_json::json!(command_count),
            ),
            (
                "eager_span_count".to_string(),
                serde_json::json!(eager_span_count),
            ),
            (
                "participant_count".to_string(),
                serde_json::json!(participant_request_ids.len()),
            ),
            (
                "reusable_span_count".to_string(),
                serde_json::json!(reusable_span_count),
            ),
            (
                "span_count".to_string(),
                serde_json::json!(spans.map_or(0, |spans| spans.len())),
            ),
        ]);
        let mut attributes = BTreeMap::from([
            (
                "actual_model_smoke".to_string(),
                serde_json::json!(matches!(
                    self.entrypoint,
                    ProfileEntrypoint::Run | ProfileEntrypoint::Serve
                )),
            ),
            (
                "attribution_scope".to_string(),
                serde_json::json!("physical_submission"),
            ),
            (
                "backend_device".to_string(),
                serde_json::json!(self.backend_device),
            ),
            (
                "backend_type".to_string(),
                serde_json::json!(self.backend_type),
            ),
            (
                "device_id".to_string(),
                serde_json::json!(batch.device_id().to_string()),
            ),
            ("diagnostic_only".to_string(), serde_json::json!(true)),
            (
                "device_timing_semantics".to_string(),
                serde_json::json!("submission_relative_duration_only"),
            ),
            (
                "device_timing_status".to_string(),
                serde_json::json!(timing_status),
            ),
            (
                "formal_device_busy_time_eligible".to_string(),
                serde_json::json!(false),
            ),
            (
                "participant_request_ids".to_string(),
                serde_json::json!(participant_request_ids),
            ),
            (
                "physical_submission_fingerprint".to_string(),
                serde_json::json!(submission.fingerprint()),
            ),
            (
                "plan_hash".to_string(),
                serde_json::json!(batch.plan_hash().to_string()),
            ),
            (
                "plan_id".to_string(),
                serde_json::json!(batch.plan_id().to_string()),
            ),
            (
                "profile_detail".to_string(),
                serde_json::json!(self.profile_detail.as_str()),
            ),
            (
                "runtime_implementation_fingerprint".to_string(),
                serde_json::json!(batch.runtime_implementation_fingerprint()),
            ),
            (
                "production_reusable_execution_selection_preserved".to_string(),
                serde_json::json!(self.profile_detail == ObservabilityProfileDetail::Replay),
            ),
            (
                "execution_path_policy".to_string(),
                serde_json::json!(match self.profile_detail {
                    ObservabilityProfileDetail::Verify => "compiled_bindings_eager_commands",
                    ObservabilityProfileDetail::Kernel | ObservabilityProfileDetail::Full => {
                        "logical_commands_reusable_segments"
                    }
                    _ => "production_selection",
                }),
            ),
            (
                "measurement_instrumentation_present".to_string(),
                serde_json::json!(true),
            ),
        ]);
        if let Some(reason) = unavailable_reason {
            attributes.insert(
                "device_timing_unavailable_reason".to_string(),
                serde_json::json!(reason),
            );
        }
        let event = FerrumProfileEvent {
            schema_version: OBSERVABILITY_PROFILE_SCHEMA_VERSION,
            ts_unix_nanos: timestamp
                .timestamp_nanos_opt()
                .unwrap_or_else(|| timestamp.timestamp_micros() * 1_000),
            event_id: format!("evt-vnext-physical-{}", submission.fingerprint()),
            request_id: first_identity.request_id.to_string(),
            correlation_id: Some(submission.fingerprint().to_owned()),
            entrypoint: self.entrypoint,
            backend: "actual".to_string(),
            runtime_preset_hash: ENGINE_RUNTIME_TRACE_PRESET_HASH.to_string(),
            phase: "vnext.device_physical_submission".to_string(),
            event_kind: ProfileEventKind::Instant,
            timestamp,
            status: ProfileStatus::DiagnosticOnly,
            model: Some(self.model.clone()),
            duration_us: None,
            memory: None,
            resource: None,
            error: None,
            replay: None,
            shape,
            backend_detail: Some(BTreeMap::from([
                (
                    "backend_device".to_string(),
                    serde_json::json!(self.backend_device),
                ),
                (
                    "backend_type".to_string(),
                    serde_json::json!(self.backend_type),
                ),
                ("physical_spans".to_string(), serde_json::json!(spans)),
            ])),
            attributes,
        };
        event.validate().map_err(|error| {
            ExecutionEventSinkError::new(format!(
                "invalid vNext physical submission timing event: {error}"
            ))
        })?;
        Ok(Some(event))
    }

    fn device_submission_attribution_events(
        &self,
        attribution: &BoundDeviceSubmissionAttribution,
        timestamp: chrono::DateTime<chrono::Utc>,
    ) -> std::result::Result<Vec<FerrumProfileEvent>, ExecutionEventSinkError> {
        let batch = attribution.batch_identity();
        let measured_timings = match attribution.terminal_timing() {
            DeviceTimingMeasurement::Measured(timing) => Some(timing),
            DeviceTimingMeasurement::NotRequested | DeviceTimingMeasurement::Unavailable(_) => None,
        };
        let reusable_span_count = measured_timings.map_or(0, |timing| {
            timing
                .spans()
                .iter()
                .filter(|span| span.kind() == DeviceExecutionSpanKind::ReusableExecutable)
                .count()
        });
        let mut events =
            Vec::with_capacity(attribution.device().commands().len() + reusable_span_count);
        for command in attribution.device().commands() {
            let timing = project_device_command_timing(
                attribution.terminal_timing(),
                command.command_index(),
            );
            let physical_span = timing.physical_span;
            let command_measurement = timing.command_measurement;
            let device_elapsed_ns =
                command_measurement.and_then(DeviceExecutionSpanMeasurement::elapsed_ns);
            let device_intervals = command_measurement
                .and_then(DeviceExecutionSpanMeasurement::intervals)
                .map(|intervals| {
                    intervals
                        .iter()
                        .map(|interval| {
                            serde_json::json!({
                                "kind": interval.kind().as_str(),
                                "start_offset_ns": interval.start_offset_ns(),
                                "end_offset_ns": interval.end_offset_ns(),
                                "subwork_id": interval.subwork_id(),
                            })
                        })
                        .collect::<Vec<_>>()
                });
            let (node_index, node, first, participant_request_ids) = if let Some(node_index) =
                command.node_index()
            {
                let node_index_usize = usize::try_from(node_index).map_err(|_| {
                    ExecutionEventSinkError::new(
                        "device attribution node index exceeds the host index range",
                    )
                })?;
                let node = batch.nodes().get(node_index_usize).ok_or_else(|| {
                    ExecutionEventSinkError::new(
                        "device attribution node index is absent from its batch identity",
                    )
                })?;
                let participant_start =
                    usize::try_from(command.participant_start()).map_err(|_| {
                        ExecutionEventSinkError::new(
                            "device attribution participant start exceeds the host index range",
                        )
                    })?;
                let participant_end = usize::try_from(command.participant_end()).map_err(|_| {
                    ExecutionEventSinkError::new(
                        "device attribution participant end exceeds the host index range",
                    )
                })?;
                let participants = node
                    .participants()
                    .get(participant_start..participant_end)
                    .ok_or_else(|| {
                        ExecutionEventSinkError::new(
                            "device attribution participant range is absent from its batch node",
                        )
                    })?;
                let first = participants.first().ok_or_else(|| {
                    ExecutionEventSinkError::new("device attribution participant range is empty")
                })?;
                let participant_request_ids = participants
                    .iter()
                    .map(|participant| participant.identity().parts().request_id.to_string())
                    .collect::<Vec<_>>();
                (Some(node_index), Some(node), first, participant_request_ids)
            } else {
                let first_node = batch.nodes().first().ok_or_else(|| {
                    ExecutionEventSinkError::new("wave-level device attribution has no batch nodes")
                })?;
                let first = first_node.participants().first().ok_or_else(|| {
                    ExecutionEventSinkError::new(
                        "wave-level device attribution has no participants",
                    )
                })?;
                let mut participant_request_ids = batch
                    .nodes()
                    .iter()
                    .flat_map(|node| node.participants())
                    .map(|participant| participant.identity().parts().request_id.to_string())
                    .collect::<Vec<_>>();
                participant_request_ids.sort();
                participant_request_ids.dedup();
                (None, None, first, participant_request_ids)
            };
            let first_identity = first.identity().parts();
            let shape = BTreeMap::from([
                (
                    "command_index".to_string(),
                    serde_json::json!(command.command_index()),
                ),
                ("node_index".to_string(), serde_json::json!(node_index)),
                (
                    "participant_start".to_string(),
                    serde_json::json!(command.participant_start()),
                ),
                (
                    "participant_count".to_string(),
                    serde_json::json!(command.participant_count()),
                ),
                (
                    "participant_end".to_string(),
                    serde_json::json!(command.participant_end()),
                ),
                (
                    "token_count".to_string(),
                    serde_json::json!(command.token_count()),
                ),
                (
                    "physical_compute_dispatch_count".to_string(),
                    serde_json::json!(command.compute_dispatch_count()),
                ),
                (
                    "physical_transfer_command_count".to_string(),
                    serde_json::json!(command.transfer_command_count()),
                ),
                (
                    "reusable_graph_node_count".to_string(),
                    serde_json::json!(command.reusable_graph_node_count()),
                ),
                (
                    "device_interval_count".to_string(),
                    serde_json::json!(command_measurement
                        .and_then(DeviceExecutionSpanMeasurement::intervals)
                        .map_or(0, |intervals| intervals.len())),
                ),
                (
                    "device_elapsed_ns".to_string(),
                    serde_json::json!(device_elapsed_ns),
                ),
                (
                    "device_span_start_command_index".to_string(),
                    serde_json::json!(physical_span.map(|span| span.start_command_index())),
                ),
                (
                    "device_span_end_command_index".to_string(),
                    serde_json::json!(physical_span.map(|span| span.end_command_index())),
                ),
            ]);
            let mut attributes = BTreeMap::from([
                (
                    "actual_model_smoke".to_string(),
                    serde_json::json!(matches!(
                        self.entrypoint,
                        ProfileEntrypoint::Run | ProfileEntrypoint::Serve
                    )),
                ),
                (
                    "attribution_scope".to_string(),
                    serde_json::json!(if node.is_some() { "node" } else { "wave" }),
                ),
                (
                    "backend_device".to_string(),
                    serde_json::json!(self.backend_device),
                ),
                (
                    "backend_type".to_string(),
                    serde_json::json!(self.backend_type),
                ),
                (
                    "batching_form".to_string(),
                    serde_json::json!(command.batching_form().as_str()),
                ),
                (
                    "command_phase".to_string(),
                    serde_json::json!(format!("{:?}", command.command_phase()).to_ascii_lowercase()),
                ),
                (
                    "device_id".to_string(),
                    serde_json::json!(batch.device_id().to_string()),
                ),
                ("diagnostic_only".to_string(), serde_json::json!(true)),
                (
                    "execution_path".to_string(),
                    serde_json::json!(command.execution_path().as_str()),
                ),
                (
                    "native_op_id".to_string(),
                    serde_json::json!(command.native_op_id()),
                ),
                (
                    "participant_request_ids".to_string(),
                    serde_json::json!(participant_request_ids),
                ),
                (
                    "plan_hash".to_string(),
                    serde_json::json!(batch.plan_hash().to_string()),
                ),
                (
                    "plan_id".to_string(),
                    serde_json::json!(batch.plan_id().to_string()),
                ),
                (
                    "profile_detail".to_string(),
                    serde_json::json!(self.profile_detail.as_str()),
                ),
                (
                    "physical_submission_fingerprint".to_string(),
                    serde_json::json!(attribution.submission_fingerprint()),
                ),
                (
                    "runtime_implementation_fingerprint".to_string(),
                    serde_json::json!(batch.runtime_implementation_fingerprint()),
                ),
            ]);
            if let Some(node) = node {
                attributes.insert(
                    "node_id".to_string(),
                    serde_json::json!(node.node_id().to_string()),
                );
                attributes.insert(
                    "operation_id".to_string(),
                    serde_json::json!(node.operation_id().to_string()),
                );
                attributes.insert(
                    "provider_id".to_string(),
                    serde_json::json!(node.provider_id().to_string()),
                );
                attributes.insert(
                    "provider_implementation_fingerprint".to_string(),
                    serde_json::json!(node.provider_implementation_fingerprint()),
                );
                let semantics = node.provider_execution_semantics();
                attributes.insert(
                    "provider_execution_contract_version".to_string(),
                    serde_json::json!(semantics.contract_version().to_string()),
                );
                attributes.insert(
                    "provider_execution_contract_fingerprint".to_string(),
                    serde_json::json!(semantics.contract_fingerprint().to_string()),
                );
                attributes.insert(
                    "provider_execution_repeatability".to_string(),
                    serde_json::json!(semantics.repeatability().as_str()),
                );
                attributes.insert(
                    "provider_replay_equivalence".to_string(),
                    serde_json::json!(semantics.replay_equivalence().as_str()),
                );
            }
            attributes.insert(
                "device_timing_status".to_string(),
                serde_json::json!(timing.status),
            );
            attributes.insert(
                "device_timing_semantics".to_string(),
                serde_json::json!("submission_relative_duration_only"),
            );
            attributes.insert(
                "formal_device_busy_time_eligible".to_string(),
                serde_json::json!(false),
            );
            if let Some(span) = physical_span {
                attributes.insert(
                    "device_timing_span_kind".to_string(),
                    serde_json::json!(span.kind().as_str()),
                );
                if let Some(fingerprint) = span.reusable_executable_fingerprint() {
                    attributes.insert(
                        "reusable_executable_fingerprint".to_string(),
                        serde_json::json!(fingerprint),
                    );
                }
            }
            if let Some(reason) = timing.unavailable {
                attributes.insert(
                    "device_timing_unavailable_reason".to_string(),
                    serde_json::json!(reason.label()),
                );
            }
            let event = FerrumProfileEvent {
                schema_version: OBSERVABILITY_PROFILE_SCHEMA_VERSION,
                ts_unix_nanos: timestamp
                    .timestamp_nanos_opt()
                    .unwrap_or_else(|| timestamp.timestamp_micros() * 1_000),
                event_id: format!(
                    "evt-vnext-native-{}-{}-{}",
                    attribution.submission_fingerprint(),
                    node_index.map_or_else(|| "wave".to_owned(), |index| index.to_string()),
                    command.command_index()
                ),
                request_id: first_identity.request_id.to_string(),
                correlation_id: Some(attribution.submission_fingerprint().to_owned()),
                entrypoint: self.entrypoint,
                backend: "actual".to_string(),
                runtime_preset_hash: ENGINE_RUNTIME_TRACE_PRESET_HASH.to_string(),
                phase: "vnext.device_native_work".to_string(),
                event_kind: ProfileEventKind::Instant,
                timestamp,
                status: ProfileStatus::DiagnosticOnly,
                model: Some(self.model.clone()),
                duration_us: None,
                memory: None,
                resource: None,
                error: None,
                replay: None,
                shape,
                backend_detail: Some(BTreeMap::from([
                    (
                        "backend_device".to_string(),
                        serde_json::json!(self.backend_device),
                    ),
                    (
                        "backend_type".to_string(),
                        serde_json::json!(self.backend_type),
                    ),
                    (
                        "device_intervals".to_string(),
                        serde_json::json!(device_intervals),
                    ),
                ])),
                attributes,
            };
            event.validate().map_err(|error| {
                ExecutionEventSinkError::new(format!(
                    "invalid vNext device attribution profile event: {error}"
                ))
            })?;
            events.push(event);
        }
        if let Some(timing) = measured_timings {
            let first_node = batch.nodes().first().ok_or_else(|| {
                ExecutionEventSinkError::new("physical device timing span has no batch nodes")
            })?;
            let first = first_node.participants().first().ok_or_else(|| {
                ExecutionEventSinkError::new("physical device timing span has no participants")
            })?;
            let first_identity = first.identity().parts();
            let mut participant_request_ids = batch
                .nodes()
                .iter()
                .flat_map(|node| node.participants())
                .map(|participant| participant.identity().parts().request_id.to_string())
                .collect::<Vec<_>>();
            participant_request_ids.sort();
            participant_request_ids.dedup();
            for span in timing
                .spans()
                .iter()
                .filter(|span| span.kind() == DeviceExecutionSpanKind::ReusableExecutable)
            {
                let device_intervals = span.measurement().intervals().map(|intervals| {
                    intervals
                        .iter()
                        .map(|interval| {
                            serde_json::json!({
                                "kind": interval.kind().as_str(),
                                "start_offset_ns": interval.start_offset_ns(),
                                "end_offset_ns": interval.end_offset_ns(),
                                "subwork_id": interval.subwork_id(),
                            })
                        })
                        .collect::<Vec<_>>()
                });
                let shape = BTreeMap::from([
                    (
                        "start_command_index".to_string(),
                        serde_json::json!(span.start_command_index()),
                    ),
                    (
                        "end_command_index".to_string(),
                        serde_json::json!(span.end_command_index()),
                    ),
                    (
                        "command_count".to_string(),
                        serde_json::json!(span.command_count()),
                    ),
                    (
                        "participant_count".to_string(),
                        serde_json::json!(participant_request_ids.len()),
                    ),
                    (
                        "device_interval_count".to_string(),
                        serde_json::json!(span
                            .measurement()
                            .intervals()
                            .map_or(0, |intervals| intervals.len())),
                    ),
                    (
                        "device_elapsed_ns".to_string(),
                        serde_json::json!(span.measurement().elapsed_ns()),
                    ),
                ]);
                let mut attributes = BTreeMap::from([
                    (
                        "actual_model_smoke".to_string(),
                        serde_json::json!(matches!(
                            self.entrypoint,
                            ProfileEntrypoint::Run | ProfileEntrypoint::Serve
                        )),
                    ),
                    (
                        "attribution_scope".to_string(),
                        serde_json::json!("physical_span"),
                    ),
                    (
                        "backend_device".to_string(),
                        serde_json::json!(self.backend_device),
                    ),
                    (
                        "backend_type".to_string(),
                        serde_json::json!(self.backend_type),
                    ),
                    (
                        "device_id".to_string(),
                        serde_json::json!(batch.device_id().to_string()),
                    ),
                    ("diagnostic_only".to_string(), serde_json::json!(true)),
                    (
                        "device_timing_semantics".to_string(),
                        serde_json::json!("submission_relative_duration_only"),
                    ),
                    (
                        "device_timing_span_kind".to_string(),
                        serde_json::json!(span.kind().as_str()),
                    ),
                    (
                        "device_timing_status".to_string(),
                        serde_json::json!(if span.measurement().elapsed_ns().is_some() {
                            "measured"
                        } else {
                            "unavailable"
                        }),
                    ),
                    ("execution_path".to_string(), serde_json::json!("replayed")),
                    (
                        "formal_device_busy_time_eligible".to_string(),
                        serde_json::json!(false),
                    ),
                    (
                        "participant_request_ids".to_string(),
                        serde_json::json!(participant_request_ids),
                    ),
                    (
                        "physical_submission_fingerprint".to_string(),
                        serde_json::json!(attribution.submission_fingerprint()),
                    ),
                    (
                        "plan_hash".to_string(),
                        serde_json::json!(batch.plan_hash().to_string()),
                    ),
                    (
                        "plan_id".to_string(),
                        serde_json::json!(batch.plan_id().to_string()),
                    ),
                    (
                        "profile_detail".to_string(),
                        serde_json::json!(self.profile_detail.as_str()),
                    ),
                    (
                        "runtime_implementation_fingerprint".to_string(),
                        serde_json::json!(batch.runtime_implementation_fingerprint()),
                    ),
                ]);
                if let Some(reason) = span.measurement().unavailable_reason() {
                    attributes.insert(
                        "device_timing_unavailable_reason".to_string(),
                        serde_json::json!(format!("{reason:?}").to_ascii_lowercase()),
                    );
                }
                if let Some(fingerprint) = span.reusable_executable_fingerprint() {
                    attributes.insert(
                        "reusable_executable_fingerprint".to_string(),
                        serde_json::json!(fingerprint),
                    );
                }
                let event = FerrumProfileEvent {
                    schema_version: OBSERVABILITY_PROFILE_SCHEMA_VERSION,
                    ts_unix_nanos: timestamp
                        .timestamp_nanos_opt()
                        .unwrap_or_else(|| timestamp.timestamp_micros() * 1_000),
                    event_id: format!(
                        "evt-vnext-device-span-{}-{}-{}",
                        attribution.submission_fingerprint(),
                        span.start_command_index(),
                        span.end_command_index()
                    ),
                    request_id: first_identity.request_id.to_string(),
                    correlation_id: Some(attribution.submission_fingerprint().to_owned()),
                    entrypoint: self.entrypoint,
                    backend: "actual".to_string(),
                    runtime_preset_hash: ENGINE_RUNTIME_TRACE_PRESET_HASH.to_string(),
                    phase: "vnext.device_execution_span".to_string(),
                    event_kind: ProfileEventKind::Instant,
                    timestamp,
                    status: ProfileStatus::DiagnosticOnly,
                    model: Some(self.model.clone()),
                    duration_us: None,
                    memory: None,
                    resource: None,
                    error: None,
                    replay: None,
                    shape,
                    backend_detail: Some(BTreeMap::from([
                        (
                            "backend_device".to_string(),
                            serde_json::json!(self.backend_device),
                        ),
                        (
                            "backend_type".to_string(),
                            serde_json::json!(self.backend_type),
                        ),
                        (
                            "device_intervals".to_string(),
                            serde_json::json!(device_intervals),
                        ),
                    ])),
                    attributes,
                };
                event.validate().map_err(|error| {
                    ExecutionEventSinkError::new(format!(
                        "invalid vNext physical device timing span event: {error}"
                    ))
                })?;
                events.push(event);
            }
        }
        Ok(events)
    }
}

impl VNextProfileExecutionEventSink {
    #[cfg(test)]
    pub(super) fn profile_event(
        &self,
        event: &ExecutionEvent,
    ) -> std::result::Result<FerrumProfileEvent, ExecutionEventSinkError> {
        self.context.profile_event(event, chrono::Utc::now())
    }

    pub(super) fn enqueue_events(
        &self,
        events: Vec<ExecutionEvent>,
    ) -> std::result::Result<(), ExecutionEventSinkError> {
        if events.is_empty() {
            return Ok(());
        }
        let timestamp = chrono::Utc::now();
        let mut deferred = Vec::new();
        deferred.try_reserve(events.len()).map_err(|error| {
            ExecutionEventSinkError::new(format!("reserve deferred vNext profile batch: {error}"))
        })?;
        for event in events {
            deferred.push(DeferredVNextProfileEvent {
                event,
                timestamp,
                context: Arc::clone(&self.context),
            });
        }
        self.enqueue_deferred_batch(deferred)
    }
}

impl ExecutionEventSink for VNextProfileExecutionEventSink {
    fn enablement(&self) -> ferrum_interfaces::vnext::ExecutionEventSinkEnablement {
        ferrum_interfaces::vnext::ExecutionEventSinkEnablement::All
    }

    fn is_enabled(&self, _kind: VNextExecutionEventKind) -> bool {
        true
    }

    fn device_timing_mode(&self) -> ferrum_interfaces::vnext::DeviceTimingMode {
        match self.context.profile_detail {
            ObservabilityProfileDetail::Off | ObservabilityProfileDetail::Resource => {
                ferrum_interfaces::vnext::DeviceTimingMode::Off
            }
            ObservabilityProfileDetail::Basic
            | ObservabilityProfileDetail::Latency
            | ObservabilityProfileDetail::Debug => {
                ferrum_interfaces::vnext::DeviceTimingMode::Completion
            }
            ObservabilityProfileDetail::Replay => {
                ferrum_interfaces::vnext::DeviceTimingMode::Replay
            }
            ObservabilityProfileDetail::Verify => {
                ferrum_interfaces::vnext::DeviceTimingMode::Verification
            }
            ObservabilityProfileDetail::Kernel | ObservabilityProfileDetail::Full => {
                ferrum_interfaces::vnext::DeviceTimingMode::Kernel
            }
        }
    }

    fn capture_policy(&self) -> ExecutionEventCapturePolicy {
        self.context.capture_policy
    }

    fn capture_policy_for_request(
        &self,
        origin: ExecutorRequestOrigin,
    ) -> ExecutionEventCapturePolicy {
        self.context.capture_policy_for_request(origin)
    }

    fn record_device_submission_attribution(
        &self,
        attribution: &BoundDeviceSubmissionAttribution,
    ) -> std::result::Result<(), ExecutionEventSinkError> {
        let events = self
            .context
            .device_submission_attribution_events(attribution, chrono::Utc::now())?;
        if events.is_empty() {
            return Ok(());
        }
        self.enqueue_profile_batch(events)
    }

    fn record_physical_device_submission_timing(
        &self,
        completion: &OperationCompletionReceipt,
    ) -> std::result::Result<(), ExecutionEventSinkError> {
        let Some(event) = self
            .context
            .physical_device_submission_timing_event(completion, chrono::Utc::now())?
        else {
            return Ok(());
        };
        self.enqueue_profile_batch(vec![event])
    }

    fn records_execution_resource_maintenance(&self) -> bool {
        true
    }

    fn record_execution_resource_maintenance(
        &self,
        maintenance: BoundExecutionResourceMaintenance,
    ) -> std::result::Result<(), ExecutionEventSinkError> {
        let event = self
            .context
            .execution_resource_maintenance_profile_event(&maintenance, chrono::Utc::now())?;
        self.enqueue_profile_batch(vec![event])
    }

    fn record(
        &self,
        permit: EventEmissionPermit,
    ) -> std::result::Result<(), ExecutionEventSinkError> {
        let event = DeferredVNextProfileEvent {
            event: permit.into_event(),
            timestamp: chrono::Utc::now(),
            context: Arc::clone(&self.context),
        };
        self.enqueue_deferred_event(event)
    }

    fn record_batch(
        &self,
        permit: EventBatchEmissionPermit,
    ) -> std::result::Result<(), ExecutionEventSinkError> {
        self.enqueue_events(permit.into_events())
    }
}

pub(super) fn create_scheduler_trace_sink(path: Option<&Path>) -> Option<SchedulerTraceJournal> {
    let path = path?;
    match SchedulerTraceJournal::create(path.to_path_buf()) {
        Ok(journal) => Some(journal),
        Err(error) => {
            warn!(
                "Failed to open scheduler trace JSONL {}: {}",
                path.display(),
                error
            );
            None
        }
    }
}

pub(super) fn create_legacy_scheduler_trace_sink(
    path: Option<&Path>,
) -> Option<Arc<Mutex<std::fs::File>>> {
    let path = path?;
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(error) = std::fs::create_dir_all(parent) {
                warn!(
                    "Failed to create legacy scheduler trace directory {}: {}",
                    parent.display(),
                    error
                );
                return None;
            }
        }
    }
    if let Err(error) = std::fs::remove_file(path) {
        if error.kind() != std::io::ErrorKind::NotFound {
            warn!(
                "Failed to clear legacy scheduler trace JSONL {}: {}",
                path.display(),
                error
            );
            return None;
        }
    }
    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
    {
        Ok(file) => Some(Arc::new(Mutex::new(file))),
        Err(error) => {
            warn!(
                "Failed to open legacy scheduler trace JSONL {}: {}",
                path.display(),
                error
            );
            None
        }
    }
}
