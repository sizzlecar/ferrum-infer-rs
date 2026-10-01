//! The same replayed maintenance step is checked at the native final commit.
use super::*;
use ferrum_interfaces::execution_cost::{GuardedNotSubmittedReason, HostSubmissionRejection};
use ferrum_interfaces::model_executor::{
    PlanRuntimePrefixRestoreInput, PlanRuntimePrefixRestoreOutcome, PrefixCaptureStatus,
};
use ferrum_interfaces::vnext::{CheckpointTransferSubmissionGuard, PreparedCheckpointTransfer};

pub(super) struct MaintenanceGuard {
    engine: std::sync::Weak<EngineInner>,
    observed_owner: Option<prefix_observation::Owner>,
    proof: ControllerSafetyProof,
    cost_domain: PlanningShapeDomain<
        ferrum_scheduler::implementations::continuous::cost_model::WaveExecutionShape,
    >,
    stage: PrefixMaintenanceStage,
    maintenance_model_version: u64,
    maintenance: Arc<super::super::super::cost_observation::PrefixCostSnapshot>,
    valid_until: Instant,
    model_version: u64,
}

impl MaintenanceGuard {
    fn current(&self, engine: &EngineInner) -> std::result::Result<(), HostSubmissionRejection> {
        use HostSubmissionRejection::*;
        if engine.shutdown_started.load(Ordering::Acquire) {
            return Err(Cancelled);
        }
        if slo_clock_now() > self.valid_until {
            return Err(WitnessExpired);
        }
        if !self.maintenance.current()
            || self.maintenance.model_version() != self.maintenance_model_version
        {
            return Err(CostModelChanged);
        }
        match engine
            .cost_runtime
            .as_ref()
            .ok_or(CostModelChanged)?
            .try_model_version_current(self.model_version)
        {
            Some(true) => Ok(()),
            Some(false) => Err(CostModelChanged),
            None => Err(Busy),
        }
    }
}

impl CheckpointTransferSubmissionGuard for MaintenanceGuard {
    fn relies_on_cost_witness(&self) -> bool {
        true
    }
    fn check(
        &self,
        actual: &PreparedCheckpointTransfer<'_>,
    ) -> std::result::Result<(), GuardedNotSubmittedReason> {
        use HostSubmissionRejection::*;
        let result = (|| {
            let check = || {
                let engine = self.engine.upgrade().ok_or(Cancelled)?;
                self.current(&engine)?;
                let sequences = engine.sequences.try_read().ok_or(Busy)?;
                if sequences.len() != self.proof.fences.len() + self.proof.waiting_fences.len() {
                    return Err(FrontierChanged);
                }
                for fence in self.proof.fences.iter().chain(&self.proof.waiting_fences) {
                    let sequence = sequences
                        .get(&fence.key.request_id)
                        .ok_or(FrontierChanged)?;
                    if !fence.matches_sequence(sequence) {
                        return Err(FrontierChanged);
                    }
                    let output = sequence.credited_output.as_ref().ok_or(OutputRevoked)?;
                    if output.failure.is_some()
                        || output.port.consumer_closed()
                        || output.grant.is_some()
                        || output.port.planning_snapshot() != fence.output
                    {
                        return Err(OutputRevoked);
                    }
                }
                if self.proof.protection.as_ref().is_some_and(|scope| {
                    scope.rows().len() != self.proof.fences.len()
                        || scope.required_first_service().is_some()
                }) {
                    return Err(FrontierChanged);
                }
                for peer in &self.proof.recovery_peers {
                    let sequence = sequences.get(&peer.id).ok_or(FrontierChanged)?;
                    if !Arc::ptr_eq(&peer.owner, &sequence.stream_projection_identity)
                        || sequence
                            .time_admission
                            .as_ref()
                            .is_none_or(|state| state.recovery_service != peer.debt)
                    {
                        return Err(FrontierChanged);
                    }
                }
                self.current(&engine)
            };
            check().map_err(GuardedNotSubmittedReason::HostRejected)?;
            let shape = super::super::super::cost_observation::prefix_cost_shape(
                actual.cost_domain(),
                actual.host_work(),
            )
            .map_err(|_| GuardedNotSubmittedReason::AttributionUnavailable)?;
            if !self.cost_domain.shapes().contains(&shape)
                || (self.stage == PrefixMaintenanceStage::Restore)
                    != actual.source_capture_identity().is_some()
            {
                return Err(GuardedNotSubmittedReason::ActualRouteMismatch);
            }
            let engine = self
                .engine
                .upgrade()
                .ok_or(GuardedNotSubmittedReason::HostRejected(Cancelled))?;
            self.current(&engine)
                .map_err(GuardedNotSubmittedReason::HostRejected)
        })();
        if let Some(owner) = &self.observed_owner {
            if let Some(engine) = self.engine.upgrade() {
                if let Some(recorder) = &engine.prefix_resource_recorder {
                    recorder.record(prefix_observation::Event::NativeGuard {
                        owner: owner.clone(),
                        identity: actual.identity().into(),
                        source_capture: actual.source_capture_identity().map(Into::into),
                        inference_epoch: Some(self.model_version),
                        maintenance_epoch: Some(self.maintenance_model_version),
                        rejection: result.as_ref().err().copied(),
                    });
                }
            }
        }
        result
    }
}

impl EngineInner {
    async fn execute_prefix_cache_capture(
        self: &Arc<Self>,
        work: super::producer::PreparedCacheCapture,
    ) -> Result<EngineIterationOutcome> {
        let super::producer::PreparedCacheCapture {
            plan,
            proof,
            model_version,
        } = work;
        let budget = Arc::clone(&proof.budget);
        if plan.evidence.stage != PrefixMaintenanceStage::Capture
            || plan.evidence.capture_span_start != plan.offer.capture_span_start
        {
            self.finish_controller_audit(&budget, "prefix_cache_capture_changed");
            return Ok(EngineIterationOutcome::Progressed);
        }
        let guard = Arc::new(MaintenanceGuard {
            engine: Arc::downgrade(self),
            observed_owner: self.prefix_resource_recorder.as_ref().and_then(|_| {
                let request_id = &plan.request_id;
                proof
                    .fences
                    .iter()
                    .find(|fence| &fence.key.request_id == request_id)
                    .map(|fence| {
                        prefix_observation::Owner::new(
                            &fence.key.request_id,
                            fence.incarnation,
                            fence.key.generation.get(),
                        )
                    })
            }),
            proof,
            cost_domain: plan.evidence.cost_domain,
            stage: PrefixMaintenanceStage::Capture,
            maintenance_model_version: plan.evidence.model_version,
            maintenance: plan.maintenance,
            valid_until: plan.valid_until,
            model_version,
        });
        let result = self
            .model_executor
            .try_capture_plan_runtime_prefix_guarded(
                PrefixCaptureRequest {
                    source_request_id: &plan.request_id,
                    source_tokens: &plan.tokens,
                    maximum_sequence_tokens: plan.maximum_sequence_tokens,
                    boundary: plan.offer.boundary_tokens.get() as usize,
                    expires_at: plan.expires_at,
                },
                guard,
            )
            .await;
        // Success means the model published the actual cache and acknowledged
        // its native transfer. The installed FIFO sink owns learning; no synthetic
        // sample or future hit is recorded here.
        self.finish_controller_audit(&budget, "prefix_cache_capture");
        if let Err(error) = result {
            self.complete_request_with_error(&plan.request_id, error)
                .await?;
        }
        Ok(EngineIterationOutcome::Progressed)
    }

    pub(in crate::continuous_engine::inner) async fn execute_slo_prefix_maintenance(
        self: &Arc<Self>,
        work: PreparedPrefixMaintenance,
    ) -> Result<EngineIterationOutcome> {
        let work = match work {
            PreparedPrefixMaintenance::Rendezvous(work) => work,
            PreparedPrefixMaintenance::Ready(work) => {
                return self.execute_ready_prefix_maintenance(work).await
            }
            PreparedPrefixMaintenance::CacheCapture(work) => {
                return self.execute_prefix_cache_capture(work).await
            }
        };
        let PreparedRendezvousMaintenance {
            mut cohort,
            evidence,
            maintenance,
            valid_until,
            model_version,
            predicted_wall_ns: _,
            proof,
        } = work;
        let budget = Arc::clone(&proof.budget);
        let stage = evidence.stage;
        let capture_span_start = evidence.capture_span_start;
        let guard = Arc::new(MaintenanceGuard {
            engine: Arc::downgrade(self),
            observed_owner: self.prefix_resource_recorder.as_ref().and_then(|_| {
                let request_id = match stage {
                    PrefixMaintenanceStage::Capture => cohort.hold.source().request_id(),
                    PrefixMaintenanceStage::Restore => cohort.target.request_id(),
                };
                proof
                    .fences
                    .iter()
                    .find(|fence| &fence.key.request_id == request_id)
                    .map(|fence| {
                        prefix_observation::Owner::new(
                            &fence.key.request_id,
                            fence.incarnation,
                            fence.key.generation.get(),
                        )
                    })
            }),
            proof,
            cost_domain: evidence.cost_domain,
            stage,
            maintenance_model_version: evidence.model_version,
            maintenance,
            valid_until: valid_until.min(cohort.expires_at),
            model_version,
        });
        let error_request_id = match stage {
            PrefixMaintenanceStage::Capture => cohort.hold.source().request_id().clone(),
            PrefixMaintenanceStage::Restore => cohort.target.request_id().clone(),
        };
        let result = match stage {
            PrefixMaintenanceStage::Capture => {
                self.model_executor
                    .try_capture_plan_runtime_prefix_guarded(
                        PrefixCaptureRequest {
                            source_request_id: cohort.hold.source().request_id(),
                            source_tokens: &cohort.source_tokens,
                            maximum_sequence_tokens: cohort.maximum_sequence_tokens,
                            boundary: cohort.hold.boundary(),
                            expires_at: cohort.expires_at,
                        },
                        guard,
                    )
                    .await
                    .map(|ready| {
                        if ready && cohort.capture.status() == PrefixCaptureStatus::Ready {
                            // Only this successful full model publication may mint
                            // the next phase's exact capture-span provenance.
                            cohort.capture_span_start = Some(capture_span_start);
                            self.slo_controller.lock().prefix = Some(cohort);
                        }
                    })
            }
            PrefixMaintenanceStage::Restore => {
                // Release the logical hold while retaining its actual source
                // lease and the common iteration lock through publication.
                cohort.hold.release();
                let target = self.sequences.try_read().and_then(|sequences| {
                    sequences.get(cohort.target.request_id()).map(|sequence| {
                        (
                            sequence.prefill_context_tokens(),
                            sequence.model_maximum_sequence_tokens(),
                        )
                    })
                });
                if let Some((tokens, maximum_sequence_tokens)) = target {
                    if let Some(prepared) = self.scheduler.prepare_prefix_restore(
                        cohort.target.request_id(),
                        0,
                        tokens.len(),
                    )? {
                        match self
                            .model_executor
                            .try_restore_plan_runtime_prefix_guarded(
                                PlanRuntimePrefixRestoreInput {
                                    request_id: cohort.target.request_id(),
                                    input_tokens: &tokens,
                                    maximum_sequence_tokens,
                                    checkpoint: Some(cohort.capture.as_ref()),
                                    retry: None,
                                },
                                guard,
                            )
                            .await
                        {
                            Ok(PlanRuntimePrefixRestoreOutcome::Restored(output)) => {
                                if output.restored_tokens() != cohort.hold.boundary() {
                                    Err(FerrumError::backend(
                                        "guarded restore published a different prefix boundary",
                                    ))
                                } else {
                                    self.commit_prefix_restore_output(
                                        cohort.target.request_id(),
                                        prepared,
                                        &tokens,
                                        output,
                                    )
                                    .map(|()| {
                                        cohort.restored = true;
                                        self.slo_controller.lock().prefix = Some(cohort);
                                    })
                                }
                            }
                            Ok(
                                PlanRuntimePrefixRestoreOutcome::Unavailable
                                | PlanRuntimePrefixRestoreOutcome::Deferred(_),
                            ) => Ok(()),
                            Err(error) => Err(error),
                        }
                    } else {
                        Ok(())
                    }
                } else {
                    Ok(())
                }
            }
        };
        self.finish_controller_audit(&budget, "prefix_maintenance");
        if let Err(error) = result {
            self.complete_request_with_error(&error_request_id, error)
                .await?;
        }
        Ok(EngineIterationOutcome::Progressed)
    }
}

impl EngineInner {
    async fn execute_ready_prefix_maintenance(
        self: &Arc<Self>,
        work: super::ready::PreparedReadyMaintenance,
    ) -> Result<EngineIterationOutcome> {
        let super::ready::PreparedReadyMaintenance {
            mut cohort,
            evidence,
            maintenance,
            valid_until,
            model_version,
            proof,
        } = work;
        let budget = Arc::clone(&proof.budget);
        if evidence.restored_frontier.target.request_id != cohort.target
            || evidence.restored_frontier.target.incarnation != cohort.incarnation
            || evidence.restored_frontier.previous_offset != 0
            || evidence.restored_frontier.restored_tokens != cohort.boundary.get()
        {
            self.finish_controller_audit(&budget, "ready_prefix_identity_changed");
            return Ok(EngineIterationOutcome::Progressed);
        }
        let guard = Arc::new(MaintenanceGuard {
            engine: Arc::downgrade(self),
            observed_owner: self.prefix_resource_recorder.as_ref().and_then(|_| {
                let request_id = &cohort.target;
                proof
                    .fences
                    .iter()
                    .find(|fence| &fence.key.request_id == request_id)
                    .map(|fence| {
                        prefix_observation::Owner::new(
                            &fence.key.request_id,
                            fence.incarnation,
                            fence.key.generation.get(),
                        )
                    })
            }),
            proof,
            cost_domain: evidence.cost_domain,
            stage: PrefixMaintenanceStage::Restore,
            maintenance_model_version: evidence.model_version,
            maintenance,
            valid_until: valid_until.min(cohort.expires_at),
            model_version,
        });
        let target = self.sequences.try_read().and_then(|sequences| {
            sequences.get(&cohort.target).and_then(|sequence| {
                (Arc::ptr_eq(&sequence.stream_projection_identity, &cohort.owner)
                    && sequence.prefill_tokens_processed == 0
                    && !sequence.prefill_complete
                    && sequence.generated_tokens.is_empty())
                .then(|| {
                    (
                        sequence.prefill_context_tokens(),
                        sequence.model_maximum_sequence_tokens(),
                    )
                })
            })
        });
        let Some((tokens, maximum_sequence_tokens)) = target else {
            self.finish_controller_audit(&budget, "ready_prefix_target_changed");
            return Ok(EngineIterationOutcome::Progressed);
        };
        let Some(prepared) =
            self.scheduler
                .prepare_prefix_restore(&cohort.target, 0, tokens.len())?
        else {
            self.finish_controller_audit(&budget, "ready_prefix_target_unavailable");
            return Ok(EngineIterationOutcome::Progressed);
        };
        let request_id = cohort.target.clone();
        let result = match self
            .model_executor
            .try_restore_plan_runtime_prefix_guarded(
                PlanRuntimePrefixRestoreInput {
                    request_id: &cohort.target,
                    input_tokens: &tokens,
                    maximum_sequence_tokens,
                    checkpoint: Some(cohort.lease.as_ref()),
                    retry: None,
                },
                guard,
            )
            .await
        {
            Ok(PlanRuntimePrefixRestoreOutcome::Restored(output)) => {
                if output.restored_tokens() != cohort.boundary.get() as usize {
                    Err(FerrumError::backend(
                        "ready checkpoint restore published another boundary",
                    ))
                } else {
                    self.commit_prefix_restore_output(&cohort.target, prepared, &tokens, output)
                        .map(|()| {
                            cohort.restored = true;
                            self.slo_controller.lock().ready_prefix = Some(cohort);
                        })
                }
            }
            Ok(
                PlanRuntimePrefixRestoreOutcome::Unavailable
                | PlanRuntimePrefixRestoreOutcome::Deferred(_),
            ) => Ok(()),
            Err(error) => Err(error),
        };
        self.finish_controller_audit(&budget, "ready_prefix_restore");
        if let Err(error) = result {
            self.complete_request_with_error(&request_id, error).await?;
        }
        Ok(EngineIterationOutcome::Progressed)
    }
}
