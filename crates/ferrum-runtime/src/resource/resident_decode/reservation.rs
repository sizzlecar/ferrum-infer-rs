use super::*;
use crate::DisabledSubmissionWaveDispatchTimingSink;

impl<R: DeviceRuntime> ResidentDecodeReservation<R> {
    pub fn is_idle(&self) -> bool {
        self.state.load(Ordering::Acquire) == IDLE
    }
    pub fn revoke(&self) {
        self.state.store(REVOKED, Ordering::Release);
    }
    pub fn matches_participants(&self, batch: &ExecutionBatchParticipants<R>) -> bool {
        self.participants.sessions.len() == batch.sessions.len()
            && self
                .participants
                .sessions
                .iter()
                .zip(&batch.sessions)
                .all(|(a, b)| Arc::ptr_eq(a, b))
    }
    pub fn matches_bootstrap(
        &self,
        wave: &PreparedStepSubmissionWave<R>,
    ) -> Result<bool, VNextError> {
        Ok(wave.purpose() == SubmissionWavePurpose::FullPlan
            && wave.execution_lane_id() == self.lane.id()
            && wave
                .claimed_backing()
                .same_reserved_owner(&self.wave_backing)
            && wave
                .step_resources()
                .claimed_backing()
                .same_reserved_owner(&self.step_backing)
            && wave
                .step_resources()
                .participants
                .iter()
                .map(|p| &p.session)
                .zip(&self.participants.sessions)
                .all(|(a, b)| Arc::ptr_eq(a, b))
            && wave.step_resources().participant_count() as usize
                == self.participants.sessions.len())
    }
    fn matches_request(&self, request: &StepResourceAdmissionRequest) -> bool {
        supported_request(request)
            && request.reusable_execution_bucket_id() == Some(&self.bucket_id)
            && request.pressure_action() == self.pressure_action
            && request.immediate_shape() == self.step_backing.work_shape().immediate_shape()
            && request.work_shape().participants() == self.step_backing.work_shape().participants()
    }
    fn validate_current_shared(&self) -> Result<bool, VNextError> {
        let guards = self
            .used_pools
            .iter()
            .map(|p| {
                p.state
                    .lock()
                    .map_err(|_| invalid_resource("resident shared pool lock poisoned"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (state, proof) in guards.iter().zip(&self.pool_proofs) {
            if state.poisoned {
                return Err(invalid_resource("resident shared pool poisoned"));
            }
            let epoch = state.chunks.epoch();
            if epoch == 0 {
                return Ok(false);
            }
            for (ordinal, backing) in &proof.chunks {
                // A pool directory epoch does not certify mutable backend
                // descriptor metadata. The few reserved shared chunks retain
                // the ordinary runtime descriptor check on every use.
                if self.lane.runtime().buffer_descriptor(&backing.buffer) != backing.descriptor {
                    return Ok(false);
                }
                if epoch == proof.epoch.load(Ordering::Acquire) {
                    continue;
                }
                let Some(current) = state.chunks.get(ordinal) else {
                    return Ok(false);
                };
                if !Arc::ptr_eq(&current.backing, backing)
                    || self
                        .lane
                        .runtime()
                        .buffer_descriptor(&current.backing.buffer)
                        != backing.descriptor
                {
                    return Ok(false);
                }
            }
            proof.epoch.store(epoch, Ordering::Release);
        }
        Ok(true)
    }
    fn validate_live(&self) -> Result<bool, VNextError> {
        let plan = &self.participants.sessions[0].resources().request.plan;
        plan.logical_admission().epochs()?;
        if !self.lane.is_reusable() || !Arc::ptr_eq(plan.runtime(), self.lane.runtime_arc()) {
            return Err(invalid_resource(
                "resident lane is no longer its live runtime",
            ));
        }
        for session in &self.participants.sessions {
            session.live_witness()?.ensure_open()?;
            if session.resources().is_poisoned() {
                return Err(invalid_resource("resident participant is poisoned"));
            }
        }
        self.validate_current_shared()
    }
    /// Cold only: the ordinary oracle/capture path uses the reserved addresses.
    pub fn prepare_bootstrap(
        self: &Arc<Self>,
        request: &StepResourceAdmissionRequest,
    ) -> Result<ResidentBootstrapDecision<R>, VNextError> {
        if !self.is_idle() {
            return Err(invalid_resource(
                "resident decode reservation is busy or revoked",
            ));
        }
        if !self.matches_request(request) {
            return Ok(ResidentBootstrapDecision::Fallback);
        }
        let plan = &self.participants.sessions[0].resources().request.plan;
        let lifecycle = plan.resources.read_lifecycle("bootstrap resident decode")?;
        if !self.validate_live()? {
            return Ok(ResidentBootstrapDecision::Fallback);
        }
        let gate = ResidentUseGate::acquire(&self.state)?;
        let step = self.participants.prepare_reserved_step(
            &self.lane,
            Some(self.bucket.clone()),
            self.step_backing
                .borrow_reserved(Arc::clone(&request.work_shape))?,
            gate,
        )?;
        drop(lifecycle);
        let result = step.try_prepare_reserved_full_plan_submission_wave(
            Arc::clone(&request.work_shape),
            request.pressure_action(),
            &self.wave_backing,
        );
        let outcome = match result {
            Ok(StepSubmissionWaveAdmissionDecision::Prepared(wave)) => {
                return Ok(ResidentBootstrapDecision::Prepared { step, wave })
            }
            Ok(StepSubmissionWaveAdmissionDecision::RequestStateDeferred(x)) => {
                Ok(ResidentBootstrapDecision::RequestStateDeferred(x))
            }
            Ok(StepSubmissionWaveAdmissionDecision::RequestStateSplitRequired(x)) => {
                Ok(ResidentBootstrapDecision::RequestStateSplitRequired(x))
            }
            Ok(StepSubmissionWaveAdmissionDecision::RequestStatePoisoned(x)) => {
                Ok(ResidentBootstrapDecision::RequestStatePoisoned(x))
            }
            Ok(_) => Err(invalid_resource(
                "reserved bootstrap attempted ordinary capacity admission",
            )),
            Err(error) => Err(error),
        };
        step.try_rollback_unsubmitted()
            .map_err(|failure| failure.error)?;
        outcome
    }
    pub fn try_begin_use(
        self: &Arc<Self>,
        batch: &ExecutionBatchParticipants<R>,
        request: &StepResourceAdmissionRequest,
    ) -> Result<ResidentDecodeDecision<R>, VNextError> {
        if !self.is_idle() {
            return Err(invalid_resource(
                "resident decode reservation is busy or revoked",
            ));
        }
        if !self.matches_participants(batch) || !self.matches_request(request) {
            return Ok(ResidentDecodeDecision::Fallback);
        }
        let plan = &self.participants.sessions[0].resources().request.plan;
        let _lifecycle = plan
            .resources
            .read_lifecycle("begin a fresh resident decode")?;
        if !self.validate_live()? {
            return Ok(ResidentDecodeDecision::Fallback);
        }
        let mut gate = ResidentUseGate::acquire(&self.state)?;
        let step_id = issue_batch_step_id()?;
        let captured = match acquire_session_frames_with_backing(
            &session_frame_capture_candidates(&self.participants.sessions),
            step_id,
            &request.work_shape,
        ) {
            Ok(captured) => captured,
            Err(error) => {
                gate.mark_finalized();
                return Err(error);
            }
        };
        let participants = self
            .participants
            .sessions
            .iter()
            .cloned()
            .zip(captured)
            .map(|(session, captured)| AdmittedStepParticipant {
                session,
                frame: captured.hold,
                backing_snapshot: captured.backing_snapshot,
            })
            .collect::<Vec<_>>();
        let frame = Arc::new(ResidentFrameOwner {
            program_retention: Mutex::new(None),
            participants,
            work: Arc::clone(&request.work_shape),
            tables: std::sync::OnceLock::new(),
            step_id,
            completed_boundary: Mutex::new(StepCompletedBoundarySlot::default()),
            submitted: AtomicBool::new(false),
            finalized: false,
            reservation: Arc::clone(self),
            gate,
        });
        let outcome = self.prepare_use(Arc::clone(&frame));
        match outcome {
            Ok(ResidentDecodeDecision::Ready { use_, .. }) => Ok(ResidentDecodeDecision::Ready {
                use_,
                retirement: ResidentFrameRetirement { frame },
            }),
            other => {
                ResidentFrameRetirement { frame }
                    .try_rollback_unsubmitted()
                    .map_err(|failure| failure.error)?;
                other
            }
        }
    }
    fn prepare_use(
        self: &Arc<Self>,
        frame: Arc<ResidentFrameOwner<R>>,
    ) -> Result<ResidentDecodeDecision<R>, VNextError> {
        let plan = &self.participants.sessions[0].resources().request.plan;
        let mut tables = Vec::with_capacity(frame.participants.len());
        let mut prior = self
            .physical_snapshots
            .lock()
            .map_err(|_| invalid_resource("resident physical append history is poisoned"))?;
        let mut advanced = false;
        for p in &frame.participants {
            let Some(table) = p.backing_snapshot.physical_binding_table(
                plan.dynamic_pools(),
                plan.runtime().as_ref(),
                &DisabledSubmissionWaveDispatchTimingSink,
            )?
            else {
                return Ok(ResidentDecodeDecision::Fallback);
            };
            tables.push(table);
        }
        if !SequencePhysicalBindingTable::current_batch(
            &tables,
            plan.dynamic_pools(),
            plan.runtime().as_ref(),
        )? {
            return Ok(ResidentDecodeDecision::Fallback);
        }
        for ((p, cold), table) in frame.participants.iter().zip(prior.iter()).zip(&tables) {
            if !p.backing_snapshot.shares_physical_catalog(cold) {
                advanced = true;
                let Some(previous) = cold.physical_binding_table(
                    plan.dynamic_pools(),
                    plan.runtime().as_ref(),
                    &DisabledSubmissionWaveDispatchTimingSink,
                )?
                else {
                    return Ok(ResidentDecodeDecision::Fallback);
                };
                if !table.is_append_of(&previous) {
                    return Ok(ResidentDecodeDecision::Fallback);
                }
            }
        }
        if advanced {
            *prior = frame
                .participants
                .iter()
                .map(|p| Arc::clone(&p.backing_snapshot))
                .collect();
        }
        drop(prior);
        let requests = frame
            .participants
            .iter()
            .map(|p| {
                let request = Arc::clone(p.session.resources().request_resources());
                RequestStateHazardParticipant::new(
                    Arc::clone(&request.plan.dynamic_pools().request_state_hazards),
                    request.request_authority(),
                    request,
                )
            })
            .collect::<Vec<_>>();
        let hazards = match plan
            .dynamic_pools()
            .request_state_hazards
            .try_acquire(&requests, &self.full_nodes)?
        {
            RequestStateHazardAcquireDecision::Acquired(permit) => permit,
            RequestStateHazardAcquireDecision::Deferred(x) => {
                return Ok(ResidentDecodeDecision::RequestStateDeferred(x))
            }
            RequestStateHazardAcquireDecision::SplitRequired(x) => {
                return Ok(ResidentDecodeDecision::RequestStateSplitRequired(x))
            }
            RequestStateHazardAcquireDecision::Poisoned(x) => {
                return Ok(ResidentDecodeDecision::RequestStatePoisoned(x))
            }
        };
        let candidates = frame
            .participants
            .iter()
            .map(|p| ParticipantFlightCandidate {
                slot: Arc::clone(&p.session.slot),
                epoch: p.session.epoch(),
                fingerprint: p.session.fingerprint().clone(),
                frame: ActiveSequenceFrame {
                    frame_id: p.frame.frame_id,
                    batch_step_id: frame.step_id,
                },
                participant: BatchParticipantAuthority::new(
                    p.session.sequence_authority(),
                    p.session.request_authority(),
                ),
            })
            .collect::<Vec<_>>();
        let flights = prepare_submission_wave_participant_flights(&candidates)?;
        let attempt = issue_batch_invocation_id()?;
        let fingerprint = resident_fingerprint(&frame, attempt)?;
        // Retired topology tombstones belong to one fresh frame. Keep the
        // single full-program ledger per use, never reuse a previous frame's
        // retired ledger or construct participant × node entries.
        let registry = Arc::new(InvocationRegistry::default());
        let active_wave = registry.enter_submission_wave(
            self.full_nodes
                .len()
                .checked_mul(frame.participants.len())
                .ok_or_else(|| invalid_resource("resident topology size overflow"))?,
            attempt,
            &fingerprint,
        )?;
        let use_ = ResidentDecodeUse {
            terminal_payload: None,
            initializations: None,
            hazards,
            flights,
            active_wave,
            frame: Arc::clone(&frame),
            attempt,
            fingerprint,
            terminal_request: None,
        };
        // Tables are immutable, already validated and retained by the snapshots;
        // the use exposes them without rebuilding any node/window frame.
        use_.frame
            .tables
            .set(tables)
            .map_err(|_| invalid_resource("resident tables already installed"))?;
        Ok(ResidentDecodeDecision::Ready {
            use_,
            retirement: ResidentFrameRetirement { frame },
        })
    }
}

fn resident_fingerprint<R: DeviceRuntime>(
    frame: &ResidentFrameOwner<R>,
    attempt: BatchInvocationId,
) -> Result<String, VNextError> {
    #[derive(Serialize)]
    struct Input<'a> {
        domain: &'static str,
        step: BatchStepId,
        attempt: BatchInvocationId,
        work: &'a str,
        step_backing: &'a str,
        wave_backing: &'a str,
    }
    let bytes = serde_json::to_vec(&Input {
        domain: "ferrum.resident-decode-use.v1",
        step: frame.step_id,
        attempt,
        work: frame.work.fingerprint(),
        step_backing: frame.reservation.step_backing.fingerprint(),
        wave_backing: frame.reservation.wave_backing.fingerprint(),
    })
    .map_err(|e| invalid_resource(format!("resident use fingerprint: {e}")))?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}
