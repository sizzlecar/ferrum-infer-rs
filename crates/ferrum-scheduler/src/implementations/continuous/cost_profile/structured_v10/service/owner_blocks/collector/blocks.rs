use super::*;
fn empty_domain(previous: Option<[u8; 32]>) -> StructuredServiceDomainFreezeV1 {
    StructuredServiceDomainFreezeV1 {
        owner_offered: 0,
        eligible: 0,
        outside_fit_support: 0,
        outside_residual_support: 0,
        unclassified_failed_owner: 0,
        frozen_domain_parameters_sha256: previous,
    }
}
impl StructuredServiceCollectorV7 {
    pub fn open_block(
        &mut self,
        opened_at_ns: u64,
        fifo_cutoff: u64,
    ) -> Result<StructuredServiceRecordV7, CostProfileError> {
        let result = self.open_inner(opened_at_ns, fifo_cutoff);
        match result {
            Ok(record) => {
                self.append(&record)?;
                Ok(record)
            }
            Err(e) => {
                self.poisoned = true;
                Err(e)
            }
        }
    }
    pub(super) fn open_inner(
        &mut self,
        now: u64,
        fifo: u64,
    ) -> Result<StructuredServiceRecordV7, CostProfileError> {
        if self.poisoned
            || self.closed
            || self.prepared_tail.is_some()
            || self.opened.is_some()
            || fifo < self.last_fifo
            || now > self.epoch_deadline()?
            || now
                < self
                    .last_close
                    .map_or(self.header.opening.monotonic_ns, |v| v.monotonic_ns)
        {
            return Err(invalid("source7 block opening/cutoff differs"));
        }
        // A processed FIFO gap explicitly separates the original observed
        // populations. Unoffered runtime work may advance a retained request.
        // Without such a gap the original frontier remains continuous.
        if fifo > self.last_fifo {
            self.frontiers = Default::default();
        }
        self.block = self
            .block
            .checked_add(1)
            .ok_or(CostProfileError::Limit("source7 block overflow"))?;
        let mut assignments = Vec::new();
        for o in &mut self.owners {
            if now > o.contract.expires_at_ns && !matches!(o.state, State::Failed(_)) {
                o.state = State::Failed("original owner attempt expired".into());
                o.samples.clear();
                o.samples.shrink_to_fit();
                o.boundary = None;
                o.workspace_bytes = 0;
                o.sample_heap_bytes = 0;
                o.maximum_sample_bytes = 0;
                o.sample_axes = 0;
            }
            if let Some(phase) = o.state.phase() {
                assignments.push(StructuredOwnerAssignmentV7 {
                    owner_attempt_id: o.contract.owner_attempt_id,
                    phase,
                });
                if o.boundary.is_none() {
                    o.boundary = Some(StructuredOwnerPhaseBoundaryV1 {
                        first_block: self.block,
                        last_block: self.block,
                        first_offered: self
                            .offered
                            .checked_add(1)
                            .ok_or(CostProfileError::Limit("source7 offered overflow"))?,
                        last_offered: self.offered,
                        opening_fifo_cutoff: fifo,
                        closing_fifo_cutoff: fifo,
                        opened_at_ns: now,
                        frozen_at_ns: now,
                    });
                    o.domain = empty_domain(o.state.parameters());
                }
            }
        }
        self.opened = Some(now);
        self.block_count = 0;
        self.last_fifo = fifo;
        self.block_routes = Default::default();
        let mut discovery = discovery::DiscoveryWindow::new(
            discovery::DiscoveryPolicy {
                population_policy: self.header.declaration.population_policy(),
                offered_waves: self.header.declaration.schedule.block_offered,
                maximum_owners: self.header.declaration.maximum_owners,
                maximum_retained_bytes: self.header.declaration.maximum_discovery_bytes,
            },
            fifo,
        )
        .map_err(|_| invalid("source7 discovery declaration differs"))?;
        if self.header.declaration.schedule.input_readiness.is_some() {
            discovery = discovery.with_input_readiness(self.header.declaration.settings.max_axes);
        }
        if self.generation_universe.is_none()
            && self
                .header
                .declaration
                .schedule
                .algorithm_universe
                .is_some()
        {
            discovery = discovery
                .with_algorithm_universe(self.header.declaration.settings.max_axes)
                .map_err(|_| invalid("source7 universe discovery capacity differs"))?;
        }
        self.discovery = Some(discovery);
        self.check_retained()?;
        Ok(StructuredServiceRecordV7::BlockOpen {
            block: self.block,
            opened_at_ns: now,
            fifo_cutoff: fifo,
            assignments,
        })
    }
    pub fn close_block(
        &mut self,
        closing: StructuredServiceClockV7,
    ) -> Result<StructuredServiceRecordV7, CostProfileError> {
        let mut diagnostics = tracing::enabled!(
            target: "ferrum_scheduler::structured_owner_diagnostics",
            tracing::Level::WARN
        )
        .then(diagnostic::BlockCloseDiagnostics::default);
        let result = self.close_diagnosed_inner(closing, None, diagnostics.as_mut());
        match result {
            Ok(r) => {
                self.append(&r)?;
                if let Some(diagnostics) = diagnostics {
                    diagnostics.emit(self, &r);
                }
                Ok(r)
            }
            Err(e) => {
                self.poisoned = true;
                Err(e)
            }
        }
    }
    pub(super) fn close_inner(
        &mut self,
        closing: StructuredServiceClockV7,
        replay_freezes: Option<&[StructuredOwnerFreezeV7]>,
    ) -> Result<StructuredServiceRecordV7, CostProfileError> {
        // Replay verifies its original record separately. It must never emit
        // an accepted live-close event before that comparison has succeeded.
        self.close_diagnosed_inner(closing, replay_freezes, None)
    }
    fn close_diagnosed_inner(
        &mut self,
        closing: StructuredServiceClockV7,
        replay_freezes: Option<&[StructuredOwnerFreezeV7]>,
        mut diagnostics: Option<&mut diagnostic::BlockCloseDiagnostics>,
    ) -> Result<StructuredServiceRecordV7, CostProfileError> {
        if self.poisoned
            || self.closed
            || self.opened.is_none()
            || self.block_count != self.header.declaration.schedule.block_offered
            || closing.monotonic_ns < self.last_observed.max(self.opened.unwrap_or(0))
            || closing.wall_unix_ns == 0
            || closing.monotonic_ns > self.epoch_deadline()?
        {
            return Err(invalid("source7 incomplete block or closing clock differs"));
        }
        let (source_prefix_bytes, source_prefix_sha256) = self.source_receipt();
        let mut freezes = Vec::new();
        let serial_workspace = self
            .serial_physical_workspace()
            .then_some(self.numeric_bytes - self.persistent_bytes);
        for o in &mut self.owners {
            let Some(phase) = o.state.phase() else {
                continue;
            };
            let b = o
                .boundary
                .as_mut()
                .ok_or_else(|| invalid("source7 missing owner phase opening"))?;
            b.last_block = self.block;
            b.last_offered = self.offered;
            b.closing_fifo_cutoff = self.last_fifo;
            b.frozen_at_ns = closing.monotonic_ns;
            let actual_offered = b
                .last_offered
                .checked_sub(b.first_offered)
                .and_then(|n| n.checked_add(1))
                .ok_or_else(|| invalid("source7 offer interval differs"))?;
            if o.contract.schedule.input_readiness.is_some() {
                let scratch = o
                    .contract
                    .schedule
                    .readiness_scratch_bytes(&o.samples)
                    .map_err(numeric_error)?;
                if scratch > serial_workspace.unwrap_or(o.workspace_bytes) {
                    return Err(CostProfileError::Limit(
                        "source7 input geometry workspace capacity",
                    ));
                }
            }
            let visits_before = o.geometry_visits;
            let decision = o.contract.assess_inputs(
                phase,
                actual_offered,
                &o.samples,
                o.input_target.as_ref(),
                &self.header.declaration.settings,
                &mut o.geometry_visits,
            );
            self.readiness_scalar_visits = self
                .readiness_scalar_visits
                .checked_add(
                    o.geometry_visits
                        .checked_sub(visits_before)
                        .ok_or_else(|| invalid("source7 readiness counter decreased"))?,
                )
                .ok_or(CostProfileError::Limit("source7 cumulative readiness work"))?;
            let decision = decision.map_err(numeric_error)?;
            if decision == OwnerInputReadinessDecisionV1::Wait
                && closing.monotonic_ns <= o.contract.expires_at_ns
            {
                continue;
            }
            let readiness_failure = match decision {
                OwnerInputReadinessDecisionV1::Exhausted(gap) => {
                    Some(format!("InputReadiness{gap:?}"))
                }
                _ => None,
            };
            self.phase_transition_attempts =
                self.phase_transition_attempts
                    .checked_add(1)
                    .ok_or(CostProfileError::Limit(
                        "source7 cumulative phase transitions",
                    ))?;
            let close = StructuredOwnerPhaseCloseV1::new(
                phase,
                *b,
                source_prefix_sha256,
                o.state.parameters(),
                &o.samples,
            );
            let axes = diagnostics
                .as_ref()
                .filter(|d| d.has_capacity())
                .map(|_| diagnostic::FitAxisContext::capture(&o.state, &o.samples));
            let old = std::mem::replace(&mut o.state, State::Empty);
            let replay_child = replay_freezes.and_then(|values| {
                values
                    .iter()
                    .find(|v| v.owner_attempt_id == o.contract.owner_attempt_id)
            });
            if let Some(c) = replay_child {
                let expects_cert = self.header.declaration.domain_policy
                    == StructuredServiceDomainPolicyV1::NonNegativePhysicalEnvelopeV1
                    && phase == StructuredPhaseV2::Fit
                    && c.failure.is_none();
                if c.nonnegative_fit_certificate.is_some() != expects_cert {
                    return Err(invalid("source7 Fit certificate phase/policy differs"));
                }
            }
            let result: Result<State, String> = if closing.monotonic_ns > o.contract.expires_at_ns {
                Err("original owner attempt expired".into())
            } else if let Some(reason) = readiness_failure {
                Err(reason)
            } else {
                match (phase, old) {
                    (StructuredPhaseV2::Fit, State::Empty) => {
                        let result = match replay_child
                            .and_then(|c| c.nonnegative_fit_certificate.as_ref())
                        {
                            Some(c) => FittedStructuredModelV2::fit_owner_blocks_from_certificate(
                                self.header.fingerprint.clone().into(),
                                self.header.declaration.settings.clone(),
                                o.scope.clone(),
                                o.contract.clone(),
                                close.clone(),
                                &o.samples,
                                c.clone(),
                            ),
                            None => FittedStructuredModelV2::fit_owner_blocks(
                                self.header.fingerprint.clone().into(),
                                self.header.declaration.settings.clone(),
                                o.scope.clone(),
                                o.contract.clone(),
                                close.clone(),
                                &o.samples,
                            ),
                        };
                        result.map(State::Fitted).map_err(|e| format!("{e:?}"))
                    }
                    (StructuredPhaseV2::Residual, State::Fitted(m)) => m
                        .calibrate_owner_blocks(close.clone(), &o.samples)
                        .map(State::Calibrated)
                        .map_err(|e| format!("{e:?}")),
                    (StructuredPhaseV2::Qualification, State::Calibrated(m)) => m
                        .qualify_owner_blocks(close.clone(), &o.samples)
                        .map(|m| State::Qualified(Arc::new(m)))
                        .map_err(|e| format!("{e:?}")),
                    _ => return Err(invalid("source7 owner numerical phase differs")),
                }
            };
            o.prior_members = o
                .prior_members
                .checked_add(o.samples.len())
                .ok_or(CostProfileError::Limit("source7 member ordinal overflow"))?;
            let (state, failure) = match result {
                Ok(s) => (s, None),
                Err(reason) => (State::Failed(reason.clone()), Some(reason)),
            };
            if failure.is_some() {
                if let Some(diagnostics) = &mut diagnostics {
                    diagnostics.capture(o.contract.owner_attempt_id, &o.samples, axes);
                }
            }
            if phase == StructuredPhaseV2::Fit
                && failure.is_none()
                && o.contract.schedule.input_readiness.is_some()
            {
                o.input_target =
                    Some(OwnerInputTargetV1::from_samples(&o.samples).map_err(numeric_error)?);
            }
            let parameters_sha256 = state.parameters();
            let nonnegative_fit_certificate = match &state {
                State::Fitted(m) => m.nonnegative_fit_certificate().cloned(),
                _ => None,
            };
            if let Some(parameters_sha256) = parameters_sha256 {
                o.phases.push(StructuredPhaseProvenanceV10 {
                    phase: match phase {
                        StructuredPhaseV2::Fit => StructuredProfilePhaseV10::Fit,
                        StructuredPhaseV2::Residual => StructuredProfilePhaseV10::Residual,
                        StructuredPhaseV2::Qualification => {
                            StructuredProfilePhaseV10::Qualification
                        }
                    },
                    members: o.samples.len(),
                    member_cutoff: o.prior_members as u64,
                    accepted_fifo_cutoff: self.last_fifo,
                    frozen_at_ns: closing.monotonic_ns,
                    source_prefix_bytes,
                    source_prefix_sha256,
                    parameters_sha256,
                });
            }
            freezes.push(StructuredOwnerFreezeV7 {
                owner_attempt_id: o.contract.owner_attempt_id,
                close,
                domain: o.domain.clone(),
                parameters_sha256,
                failure,
                nonnegative_fit_certificate,
            });
            o.state = state;
            o.samples = Vec::new();
            o.workspace_bytes = 0;
            o.sample_heap_bytes = 0;
            o.maximum_sample_bytes = 0;
            o.sample_axes = 0;
            o.geometry_visits = 0;
            o.boundary = None;
        }
        let frozen = self
            .discovery
            .take()
            .ok_or_else(|| invalid("source7 discovery missing"))?
            .freeze_with_algorithm_seed(
                if self.generation_universe.is_none()
                    && matches!(self.header.declaration.schedule.algorithm_universe,
                    Some(OwnerAlgorithmUniversePolicyV1::SeededFirstOrdinaryDiscoveryBlockSubsetV1))
                {
                    self.header
                        .declaration
                        .nonnegative_envelope
                        .as_ref()
                        .and_then(|c| c.algorithm_universe.as_ref())
                } else {
                    None
                },
            )
            .map_err(|_| invalid("source7 incomplete input discovery"))?;
        if self.generation_universe.is_none()
            && self
                .header
                .declaration
                .schedule
                .algorithm_universe
                .is_some()
        {
            // Includes the generation copy, retained owner contracts and the
            // simultaneously returned BlockClose declarations before allocation.
            let copies = frozen
                .scopes()
                .len()
                .checked_mul(2)
                .and_then(|n| n.checked_add(1))
                .ok_or(CostProfileError::Limit(
                    "source7 universe retained overflow",
                ))?;
            let universe_bytes = frozen
                .algorithm_universe
                .as_ref()
                .map_or(Some(0), |u| u.retained_payload_bytes())
                .and_then(|n| n.checked_mul(copies))
                .ok_or(CostProfileError::Limit(
                    "source7 universe retained overflow",
                ))?;
            if self
                .numeric_bytes
                .checked_add(universe_bytes)
                .is_none_or(|n| n > self.header.declaration.maximum_retained_numeric_bytes)
            {
                return Err(CostProfileError::Limit(
                    "source7 universe retained capacity",
                ));
            }
            self.generation_universe = frozen.algorithm_universe.clone();
        }
        let mut discoveries = Vec::new();
        for (scope, input_target) in frozen.into_scopes_and_targets() {
            // An owner that failed numerically in this closing block was an
            // active owner at BlockOpen, and cannot reuse this block as discovery.
            if self.owners.len() >= self.header.declaration.maximum_owners {
                return Err(CostProfileError::Limit("source7 owner attempt capacity"));
            }
            let owner_attempt_id = self.owners.len() as u64 + 1;
            let mut h = Sha256::new();
            h.update(b"ferrum.owner-block-membership.v1\0");
            h.update(self.header.declaration_sha256);
            h.update(owner_attempt_id.to_le_bytes());
            h.update(serde_json::to_vec(&scope)?);
            if let Some(target) = &input_target {
                target.bind(&mut h);
            }
            let mut nonnegative_envelope = self.header.declaration.nonnegative_envelope.clone();
            if self
                .header
                .declaration
                .schedule
                .algorithm_universe
                .is_some()
            {
                if let Some(envelope) = &mut nonnegative_envelope {
                    envelope.algorithm_universe = self.generation_universe.clone();
                }
            }
            let contract = StructuredOwnerPhaseContractV1 {
                capture_identity: self.header.capture_identity,
                protocol: self.header.protocol,
                membership_rule: h.finalize().into(),
                declaration_sha256: self.header.declaration_sha256,
                owner_attempt_id,
                schedule: self.header.declaration.schedule.clone(),
                discovery_block: self.block,
                discovery_offered_cutoff: self.offered,
                discovery_fifo_cutoff: self.last_fifo,
                discovery_closed_at_ns: closing.monotonic_ns,
                expires_at_ns: self.epoch_deadline()?,
                domain_policy: self.header.declaration.domain_policy,
                nonnegative_envelope,
                input_target: input_target.clone(),
            };
            contract
                .validate(&self.header.declaration.settings)
                .map_err(numeric_error)?;
            discoveries.push(StructuredOwnerDiscoveryV7 {
                owner_attempt_id,
                scope: scope.clone(),
                contract: contract.clone(),
            });
            self.owners.push(Owner {
                scope,
                contract,
                state: State::Empty,
                boundary: None,
                samples: Vec::new(),
                workspace_bytes: 0,
                sample_heap_bytes: 0,
                maximum_sample_bytes: 0,
                sample_axes: 0,
                prior_members: 0,
                domain: empty_domain(None),
                phases: Vec::new(),
                oldest: u64::MAX,
                newest: 0,
                input_target,
                geometry_visits: 0,
            });
        }
        self.opened = None;
        self.last_close = Some(closing);
        // The pre-close reservation also covers every returned certificate
        // until close_block appends this record, before another collector
        // operation can run. This final retained value is not that peak.
        self.check_retained()?;
        Ok(StructuredServiceRecordV7::BlockClose {
            block: self.block,
            closing,
            offered: self.offered,
            accepted_fifo_cutoff: self.last_fifo,
            source_prefix_bytes,
            source_prefix_sha256,
            route_population: self.block_routes,
            discoveries,
            freezes,
        })
    }
    pub fn checkpoint(
        &mut self,
        closing: StructuredServiceClockV7,
    ) -> Result<(StructuredServiceRecordV7, StructuredServiceCheckpointV7), CostProfileError> {
        let record = self.checkpoint_record(closing)?;
        self.append(&record)?;
        let checkpoint = StructuredServiceCheckpointV7::from_collector(self, closing)?;
        Ok((record, checkpoint))
    }
    pub(super) fn checkpoint_record(
        &self,
        closing: StructuredServiceClockV7,
    ) -> Result<StructuredServiceRecordV7, CostProfileError> {
        if self.poisoned || self.closed || self.opened.is_some() || self.last_close != Some(closing)
        {
            return Err(invalid(
                "source7 checkpoint is not a complete current block",
            ));
        }
        let (source_prefix_bytes, source_prefix_sha256) = self.source_receipt();
        Ok(StructuredServiceRecordV7::Checkpoint {
            block: self.block,
            closing,
            offered: self.offered,
            accepted_fifo_cutoff: self.last_fifo,
            source_prefix_bytes,
            source_prefix_sha256,
            qualified_attempts: self
                .owners
                .iter()
                .filter(|o| matches!(o.state, State::Qualified(_)))
                .map(|o| o.contract.owner_attempt_id)
                .collect(),
            pending_attempts: self
                .owners
                .iter()
                .filter(|o| o.state.phase().is_some())
                .map(|o| o.contract.owner_attempt_id)
                .collect(),
        })
    }
}
