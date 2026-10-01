//! Original call facts transferred to the sole observation consumer.
use super::*;

pub(super) fn canonical_retained_bytes(
    shape: &ferrum_interfaces::execution_cost::CanonicalWaveCostShape,
) -> Option<usize> {
    std::mem::size_of_val(shape)
        .checked_add(
            shape
                .rows
                .capacity()
                .checked_mul(std::mem::size_of::<ActualRowWork>())?,
        )?
        .checked_add(match &shape.numeric_features {
            Some(rows) => rows
                .rows
                .capacity()
                .checked_mul(std::mem::size_of::<CostRowNumericFeatures>())?,
            None => 0,
        })?
        .checked_add(match &shape.row_multiset_features {
            Some(rows) => rows
                .rows
                .capacity()
                .checked_mul(std::mem::size_of::<HostRowStaticCostFeaturesV2>())?,
            None => 0,
        })
}

/// This envelope contains CPU facts only. Its detached call has no sink Arc,
/// engine, executor or execution authority, so an abandoned queue cannot cycle.
pub(super) struct SealedCostCall {
    call: Option<Box<EngineCostCall>>,
    pub(super) retained_bytes: usize,
    pub(super) retained_rows: usize,
    pub(super) working_bytes: usize,
}
impl std::fmt::Debug for SealedCostCall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SealedCostCall")
            .field("call_id", &self.call.as_ref().map(|c| c.call_id))
            .field("retained_bytes", &self.retained_bytes)
            .finish()
    }
}
impl EngineCostCall {
    pub(super) fn observation_time(&self) -> Option<u64> {
        self.sealed_at_ns.unwrap_or_else(|| self.clock.now_ns())
    }

    pub(super) fn enqueue_frozen(&mut self) -> CostCallDisposition {
        self.finished = true;
        let Some(sink) = self.sink.take() else {
            return CostCallDisposition::Rejected(CostCallRejection::Abandoned);
        };
        self.sealed_at_ns = Some(self.clock.now_ns());
        sink.call_finished();
        sink.capture_memory(self.call_id.get(), self.recorder.memory_audit());
        let retained_rows = self
            .participants
            .capacity()
            .checked_add(self.host.capacity())
            .and_then(|n| n.checked_add(self.host_stages.capacity()));
        let retained_bytes = self
            .recorder
            .retained_payload_bytes_upper_bound()
            .and_then(|n| n.checked_add(std::mem::size_of::<EngineCostCall>()))
            .and_then(|n| {
                n.checked_add(
                    self.participants
                        .capacity()
                        .checked_mul(std::mem::size_of::<CostObservationParticipant>())?,
                )
            })
            .and_then(|n| {
                n.checked_add(
                    self.host
                        .capacity()
                        .checked_mul(std::mem::size_of::<Option<HostCommitEvidence>>())?,
                )
            })
            .and_then(|n| {
                n.checked_add(
                    self.host_stages
                        .capacity()
                        .checked_mul(std::mem::size_of::<host_stages::HostRowProgress>())?,
                )
            })
            .and_then(|n| {
                n.checked_add(match &self.presubmit_prediction {
                    Some(value) => value.retained_bytes()?,
                    None => 0,
                })
            })
            .and_then(|n| {
                n.checked_add(match &self.prospective_capture {
                    Some(value) => value.retained_bytes()?,
                    None => 0,
                })
            })
            .and_then(|n| {
                n.checked_add(match &self.calibration_capture {
                    Some(value) => value.retained_bytes()?,
                    None => 0,
                })
            });
        // Raw facts and expanded output are separate resources. The device
        // bound already accounts for raw/COW and command projection scratch.
        // Host copies are participant-sized, never another copy of templates.
        let working_bytes = self
            .recorder
            .maximum_working_bytes_upper_bound()
            .and_then(|n| {
                n.checked_add(
                    retained_bytes?
                        .checked_sub(self.recorder.retained_payload_bytes_upper_bound()?)?,
                )
            })
            .and_then(|n| {
                n.checked_add(self.participants.len().checked_mul(
                    std::mem::size_of::<HostRowStageV1>()
                        + 2 * std::mem::size_of::<HostCommitEvidence>()
                        + std::mem::size_of::<Option<HostCostFeaturesV1>>()
                        + 4 * std::mem::size_of::<CostRowNumericFeatures>()
                        + 4 * std::mem::size_of::<HostRowStaticCostFeaturesV2>()
                        + 4 * std::mem::size_of::<ActualWaveRow>(),
                )?)
            })
            .and_then(|n| {
                n.checked_add(if self.recorder.no_submission().is_some() {
                    live_calibration::NoSubmissionReceipt::maximum_retained_bytes(
                        self.participants.len(),
                    )?
                    .checked_add(
                        live_calibration::OriginalNoSubmissionReceipt::maximum_retained_bytes(
                            self.participants.len(),
                        )?,
                    )?
                } else {
                    0
                })
            })
            .and_then(|n| n.checked_add(std::mem::size_of::<HostStageEvidenceV1>()))
            .and_then(|n| n.checked_add(std::mem::size_of::<CostCalibrationResult>()))
            .and_then(|n| {
                n.checked_add(super::memory::maximum_resolution_overhead(
                    self.participants.len(),
                )?)
            })
            .and_then(|n| n.checked_add(std::mem::size_of::<CalibrationActualEvidenceDiagnostic>()))
            .and_then(|n| {
                n.checked_add(
                    self.recorder
                        .observations()
                        .len()
                        .checked_mul(std::mem::size_of::<CalibrationActualWaveUnknown>())?,
                )
            });
        let frozen = Self {
            call_id: self.call_id,
            clock: self.clock.clone(),
            sink: None,
            sealed_at_ns: self.sealed_at_ns,
            observation_memory: None,
            identity: self.identity.clone(),
            participants: std::mem::take(&mut self.participants),
            prepare_started_at_ns: self.prepare_started_at_ns,
            boundary: self.boundary,
            recorder: self.recorder.take_frozen(),
            dispatch: std::mem::take(&mut self.dispatch),
            context_created: self.context_created,
            structured_capture: self.structured_capture,
            numeric_observation: self.numeric_observation,
            host: std::mem::take(&mut self.host),
            host_fence: self.host_fence.clone(),
            host_stages: std::mem::take(&mut self.host_stages),
            host_processing_ordinal: self.host_processing_ordinal,
            rejection: self.rejection,
            stage_rejection: self.stage_rejection,
            finished: true,
            calibration_capture: self.calibration_capture.take(),
            presubmit_prediction: self.presubmit_prediction.take(),
            prospective_capture: self.prospective_capture.take(),
            live_ticket: self.live_ticket.take(),
            source_generation: self.source_generation,
            feedback_population: self.feedback_population,
        };
        let raw = SealedCostCall {
            call: Some(Box::new(frozen)),
            retained_bytes: retained_bytes.unwrap_or(usize::MAX),
            retained_rows: retained_rows.unwrap_or(usize::MAX),
            working_bytes: working_bytes.unwrap_or(usize::MAX),
        };
        match sink.offer_raw(raw) {
            Ok(_) => CostCallDisposition::Queued,
            Err(reason) => CostCallDisposition::Dropped(reason),
        }
    }
}

impl SealedCostCall {
    pub(super) fn call_id(&self) -> u64 {
        self.call.as_ref().map_or(0, |call| call.call_id.get())
    }
    pub(super) fn queue_wait_ns(&self) -> Option<u64> {
        let call = self.call.as_ref()?;
        call.clock.now_ns()?.checked_sub(call.sealed_at_ns??)
    }
    pub(super) fn ticket(&self) -> Option<&live_calibration::Ticket> {
        self.call.as_ref()?.live_ticket.as_ref()
    }
    pub(super) fn source_generation(&self) -> u64 {
        self.call.as_ref().map_or(0, |call| call.source_generation)
    }
    pub(super) fn accepted(&self, ordinal: u64) {
        if let Some(capture) = self
            .call
            .as_ref()
            .and_then(|call| call.prospective_capture.as_ref())
        {
            capture.queue_result(&Ok(ordinal));
        }
    }
    pub(super) fn dropped(&self, reason: CostSampleDrop) {
        if let Some(call) = &self.call {
            if let Some(capture) = &call.prospective_capture {
                capture.queue_result(&Err(reason));
            }
            if let Some(capture) = &call.calibration_capture {
                capture.complete(CostCalibrationResult::UnresolvedDropped(reason));
            }
        }
    }
    pub(super) fn resolve(
        mut self,
        sink: &BoundedCostSampleSink,
        ordinal: u64,
        memory: Arc<super::memory::ObservationBytePermit>,
    ) -> (
        Option<super::resolved::ResolvedCostEntry>,
        Option<live_calibration::Ticket>,
        u64,
        Option<Arc<live_calibration::OriginalNoSubmissionReceipt>>,
    ) {
        // Keep ownership in this guard until the original completion is set.
        // Unwinding a projector must finish its waiter and fail its ticket.
        let call = self.call.as_mut().expect("single owned resolution");
        call.observation_memory = Some(Arc::clone(&memory));
        if let Err(reason) = call.recorder.resolve_pending() {
            call.dispatch.unknown.get_or_insert(reason);
        }
        call.capture_actual_unknown_diagnostic();
        let original_no_submission =
            call.original_no_submission_receipt(ordinal, Arc::clone(&memory));
        let no_submission = original_no_submission
            .as_ref()
            .and_then(|original| call.no_submission_receipt(original, Arc::clone(&memory)));
        // Capture and feedback own the original proof independently of source
        // membership. A closed source block cannot erase execution facts.
        let original_no_submission = original_no_submission
            .filter(|_| call.calibration_capture.is_some() || call.feedback_allows_no_submission());
        if no_submission.is_none() {
            call.capture_live_route_failure_diagnostic();
        }
        let (stages, preparation) = call
            .make_host_stages_with_preparation()
            .map_or((None, None), |(stages, proof)| (Some(stages), proof));
        let mut capture_result = None;
        let entry = match call.make_sample() {
            Ok(sample) => {
                if let Some(capture) = &call.calibration_capture {
                    let rows = &call.recorder.observations()[0]
                        .shape
                        .as_ref()
                        .expect("validated sample shape")
                        .rows;
                    let commits = rows
                        .iter()
                        .map(|row| {
                            call.host
                                .iter()
                                .flatten()
                                .find(|commit| {
                                    commit.request_id == row.request_id
                                        && commit.owner_incarnation == row.owner_incarnation
                                        && commit.work_generation == row.work_generation
                                        && commit.input_index == row.input_index
                                })
                                .expect("validated original host identity")
                                .clone()
                        })
                        .collect();
                    let host_features = rows
                        .iter()
                        .map(|row| {
                            call.participants
                                .iter()
                                .find(|p| {
                                    p.request_id == row.request_id
                                        && p.owner_incarnation == row.owner_incarnation
                                        && p.work_generation == row.work_generation
                                        && p.input_index == row.input_index
                                })
                                .expect("validated original participant")
                                .host_features
                        })
                        .collect();
                    let _ = capture;
                    capture_result = Some(CostCalibrationResult::Observed {
                        observation_memory: Some(Arc::clone(&memory)),
                        sample: Box::new(sample.clone()),
                        actual_rows: rows.clone(),
                        commits,
                        host_features,
                        accepted_ordinal: Some(ordinal),
                        disposition: CostCallDisposition::Published,
                    });
                }
                Some(CostEvidenceEntry::Training { sample, stages })
            }
            Err(reason) => {
                sink.reject_resolved(reason);
                // Legacy token-commit rejection is independent of complete
                // HostSettled evidence and of an original V2 no-submit proof.
                // Only an entry-less unresolved call fails at this boundary.
                if stages.is_none() && no_submission.is_none() {
                    if let Some(ticket) = &call.live_ticket {
                        ticket.record_call_failure(&call, reason);
                    }
                }
                if let Some(capture) = &call.calibration_capture {
                    let _ = capture;
                    capture_result = Some(CostCalibrationResult::Rejected(reason));
                }
                stages.map(|stages| CostEvidenceEntry::StagesOnly {
                    stages,
                    legacy_rejection: reason,
                })
            }
        };
        let entry = entry.map(|entry| {
            super::resolved::ResolvedCostEntry::new_with_domain(entry, sink.workload_domain())
                .with_original_preparation(preparation.as_ref())
                .with_memory(Arc::clone(&memory))
        });
        // Scratch and raw/COW storage are no longer retained by the results.
        // Account each real output once before lowering the shared permit;
        // shared recipe references remain conservatively charged, no pointer cache.
        let retained = entry
            .as_ref()
            .map_or(Some(0), |v| v.retained_payload_bytes())
            .and_then(|n| {
                n.checked_add(
                    no_submission
                        .as_ref()
                        .map_or(Some(0), |v| v.retained_bytes())?,
                )
            })
            .and_then(|n| {
                n.checked_add(
                    original_no_submission
                        .as_ref()
                        .map_or(Some(0), |proof| proof.retained_bytes())?,
                )
            })
            .and_then(|n| {
                n.checked_add(
                    capture_result
                        .as_ref()
                        .map_or(Some(0), |v| v.retained_payload_bytes())?,
                )
            })
            .and_then(|n| {
                n.checked_add(
                    call.calibration_capture
                        .as_ref()
                        .map_or(Some(0), |capture| capture.retained_diagnostic_bytes())?,
                )
            });
        let Some(retained) = retained.filter(|n| *n <= memory.bytes()) else {
            call.rejection
                .get_or_insert(CostCallRejection::RecorderCapacity);
            if let Some(capture) = &call.calibration_capture {
                capture.complete(CostCalibrationResult::Rejected(
                    CostCallRejection::RecorderCapacity,
                ));
            }
            if let Some(ticket) = &call.live_ticket {
                ticket.record_call_failure(call, CostCallRejection::RecorderCapacity);
            }
            return (None, call.live_ticket.take(), call.source_generation, None);
        };
        call.record_host_stage_queue(&Ok(ordinal));
        let capture = call.calibration_capture.take();
        if let Some(receipt) = no_submission {
            call.live_ticket
                .as_mut()
                .expect("validated original ticket")
                .bind_no_submission(receipt);
        }
        let ticket = call.live_ticket.take();
        let generation = call.source_generation;
        let feedback_no_submission = call
            .feedback_allows_no_submission()
            .then(|| original_no_submission.clone())
            .flatten();
        if let Some(capture) = &call.prospective_capture {
            // Only original successfully retained resolution enters issued-cost error statistics.
            if entry.is_some() {
                capture.record_issued_settlement(ordinal);
            }
            capture.abandon_unresolved();
        }
        // Destroy original recorder/COW inputs before releasing their scratch
        // reservation. Completed output owners already hold the same permit.
        drop(self.call.take());
        let shrunk = memory.shrink_to(retained);
        debug_assert!(shrunk);
        if let (Some(capture), Some(entry)) = (&capture, &entry) {
            let stages = match entry.entry() {
                CostEvidenceEntry::Training { stages, .. } => stages.as_ref(),
                CostEvidenceEntry::StagesOnly { stages, .. } => Some(stages),
            };
            if let Some(stages) = stages {
                capture.complete_host_stages(Arc::clone(stages));
                capture.complete_host_stage_queue(HostStageQueueReceipt {
                    accepted_ordinal: Some(ordinal),
                    disposition: HostStageQueueDisposition::Published,
                });
            }
            capture.complete_actual_projection(entry.shared_actual());
            capture.complete_structured_projection(entry.shared_projection());
        }
        if let (Some(capture), Some(proof)) = (&capture, original_no_submission) {
            capture.complete_no_submission(proof);
        }
        if let (Some(capture), Some(result)) = (&capture, capture_result) {
            capture.complete(result);
        }
        (entry, ticket, generation, feedback_no_submission)
    }
}
impl Drop for SealedCostCall {
    fn drop(&mut self) {
        if let Some(call) = &self.call {
            if let Some(capture) = &call.prospective_capture {
                capture.abandon_unresolved();
            }
            if let Some(ticket) = &call.live_ticket {
                ticket.record_call_failure(call, CostCallRejection::Abandoned);
            }
            if let Some(capture) = &call.calibration_capture {
                capture.complete_if_pending(CostCalibrationResult::Rejected(
                    CostCallRejection::Abandoned,
                ));
            }
        }
    }
}
