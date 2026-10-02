use super::*;
use crate::implementations::continuous::cost_model::structured_v2::OwnerOpeningFrontierPolicyV1;
impl StructuredServiceCollectorV7 {
    /// A source8 restore ACK consumes its already published maintenance FIFO
    /// position without becoming an inference offer or changing block cuts.
    pub(in super::super) fn preparation_maintenance(
        &mut self,
        fifo: u64,
    ) -> Result<(), CostProfileError> {
        if self.header.source_kind != PopulationSource::PreparedOwnerBlocksV8
            || self.last_fifo.checked_add(1) != Some(fifo)
        {
            return Err(invalid(
                "source8 maintenance FIFO is missing, duplicate or reordered",
            ));
        }
        self.last_fifo = fifo;
        Ok(())
    }

    pub(in super::super) fn position(
        &mut self,
        ticket: u64,
        fifo: u64,
        issued: u64,
        call: u64,
    ) -> Result<u64, CostProfileError> {
        let opened = self
            .opened
            .ok_or_else(|| invalid("source7 attempt outside block"))?;
        if self.block_count >= self.header.declaration.schedule.block_offered
            || self.offered.checked_add(1) != Some(ticket)
            || fifo <= self.last_fifo
            || call == 0
            || self.calls.binary_search(&call).is_ok()
            || self.calls.len() >= self.limits.max_samples.get()
            || issued < opened
            || issued > self.epoch_deadline()?
        {
            return Err(invalid("source7 original offer/FIFO/call differs"));
        }
        if self.block_count == 0
            && self.header.source_kind == PopulationSource::OwnerBlocksV7
            && self.header.declaration.schedule.opening_frontier
                == Some(OwnerOpeningFrontierPolicyV1::FirstOfferFifoV1)
            && self
                .last_fifo
                .checked_add(1)
                .is_some_and(|next| fifo > next)
        {
            // The original first offer proves a nonempty unoffered FIFO
            // prefix after BlockOpen. The immutable ticket/clock/phase ledger
            // is unchanged; only the opening request frontier is recaptured.
            // A later in-block gap can never excuse a frontier discontinuity.
            self.frontiers = Default::default();
        }
        Ok(opened)
    }
    pub(super) fn counted(
        &mut self,
        fifo: u64,
        call: u64,
        rows: usize,
        observed: u64,
    ) -> Result<(), CostProfileError> {
        if observed < self.opened.unwrap() || observed > self.epoch_deadline()? {
            return Err(invalid("source7 observation clock differs"));
        }
        self.total_rows = self
            .total_rows
            .checked_add(rows as u64)
            .ok_or(CostProfileError::Limit("source7 row overflow"))?;
        if self.total_rows > self.limits.max_total_shape_rows.get() as u64 {
            return Err(CostProfileError::Limit("source7 total row capacity"));
        }
        self.last_observed = self.last_observed.max(observed);
        self.offered += 1;
        self.last_fifo = fifo;
        self.block_count += 1;
        let index = self.calls.binary_search(&call).unwrap_err();
        self.reserve_original_call()?;
        self.calls.insert(index, call);
        self.block_routes.attempted += 1;
        self.check_retained()
    }
    /// A validated source8 preparation consumes its original offer exactly
    /// once, but is neither an eligible numerical sample nor an outside route.
    pub(in super::super) fn preparation_counted(
        &mut self,
        ticket: u64,
        fifo: u64,
        issued: u64,
        call: u64,
        rows: usize,
        observed: u64,
    ) -> Result<(), CostProfileError> {
        self.position(ticket, fifo, issued, call)?;
        self.discovery
            .as_mut()
            .ok_or_else(|| invalid("source8 block discovery missing"))?
            .observe_outside(self.block_count as u64 + 1, fifo)
            .map_err(|_| invalid("source8 preparation input position differs"))?;
        self.counted(fifo, call, rows, observed)
    }
    pub(super) fn wave(&mut self, w: &StructuredServiceWaveV7) -> Result<(), CostProfileError> {
        self.wave_with_membership(w, |_, _| Ok(true))
    }
    pub(in super::super) fn wave_with_membership(
        &mut self,
        w: &StructuredServiceWaveV7,
        mut membership: impl FnMut(
            &discovery::PopulationKey,
            Option<(u64, StructuredPhaseV2)>,
        ) -> Result<bool, CostProfileError>,
    ) -> Result<(), CostProfileError> {
        let call = w.host_stages.call_id;

        let opened = self.position(w.ticket, w.fifo, w.issued_at_ns, call)?;
        if !self.header.declaration.route_population.is_all_attempts() {
            route_population::validate_eligible_parts(
                w.issued_at_ns,
                w.prepared_route.as_ref(),
                &w.host_stages,
            )?;
        }
        let (mut input, wall_ns, observed_at_ns) = physical::validate_parts(
            &self.header.fingerprint,
            self.header.opening.monotonic_ns,
            self.seeded_input_contract.as_ref().or(self
                .header
                .declaration
                .nonnegative_envelope
                .as_ref()),
            opened,
            w.ticket,
            w.fifo,
            w.issued_at_ns,
            &w.host_stages,
            w.independent.as_ref(),
            &mut self.frontiers,
        )?;
        let rows = input.owner().rows as usize;
        if self.generation_universe.is_some()
            && self
                .header
                .declaration
                .schedule
                .algorithm_universe
                .is_some()
        {
            // Classify only after complete original physical/settlement/frontier
            // validation. Unknown C remains in the source and consumes the same
            // original offered quota; no execution failure is exempted.
            let eligible = match input.numerical_family_key() {
                Ok(_) => match &self.generation_universe {
                    Some(universe) => universe.contains_checked_algorithms(&input).map_err(numeric_error)?,
                    None => unreachable!("checked above"),
                },
                Err(crate::implementations::continuous::cost_model::structured_v2::StructuredUnknownV2::UnsupportedScope) => true,
                Err(error) => return Err(numeric_error(error)),
            };
            if !eligible {
                self.discovery
                    .as_mut()
                    .ok_or_else(|| invalid("source7 block discovery missing"))?
                    .observe_outside(self.block_count as u64 + 1, w.fifo)
                    .map_err(|_| invalid("source7 universe exclusion position differs"))?;
                self.block_routes.eligible_route += 1;
                return self.counted(w.fifo, call, rows, observed_at_ns);
            }
            if input.numerical_family_key().is_ok() {
                // Physical validation may have grown the original frontier.
                // Reserve projection against the current full retained state.
                self.check_retained()?;
                let universe = self.generation_universe.as_ref().unwrap();
                let (retained, transient) = if self.serial_physical_workspace() {
                    let projection = universe
                        .projection_vector_bytes(&input)
                        .map_err(numeric_error)?
                        .checked_add(input.retained_payload_bytes().ok_or(
                            CostProfileError::Limit("source7 universe projection overflow"),
                        )?)
                        .ok_or(CostProfileError::Limit(
                            "source7 universe projection overflow",
                        ))?;
                    (
                        self.persistent_bytes,
                        projection.max(self.numeric_bytes - self.persistent_bytes),
                    )
                } else {
                    (
                        self.numeric_bytes,
                        universe
                            .projected_input_retained_bytes(&input)
                            .and_then(|n| n.checked_add(input.retained_payload_bytes()?))
                            .ok_or(CostProfileError::Limit(
                                "source7 universe projection overflow",
                            ))?,
                    )
                };
                self.observe_reservation(retained.checked_add(transient).ok_or(
                    CostProfileError::Limit("source7 universe projection overflow"),
                )?);
                if retained
                    .checked_add(transient)
                    .is_none_or(|n| n > self.header.declaration.maximum_retained_numeric_bytes)
                {
                    return Err(self.capacity_error(
                        "source7 universe projection capacity",
                        retained,
                        transient,
                    ));
                }
                input = input
                    .with_algorithm_universe(universe)
                    .map_err(numeric_error)?;
            }
        }
        // Original physical/domain validation above must precede population
        // classification. Invalid physical input is never an exact fallback.
        let key = discovery::PopulationKey::from_input(
            &input,
            self.header.declaration.population_policy(),
        )
        .map_err(numeric_error)?;
        let active = self
            .owners
            .iter()
            .position(|o| key.matches_scope(&o.scope) && !matches!(o.state, State::Failed(_)));
        let include = membership(
            &key,
            active.and_then(|i| {
                self.owners[i]
                    .state
                    .phase()
                    .map(|p| (self.owners[i].contract.owner_attempt_id, p))
            }),
        )?;
        let discovery = self
            .discovery
            .as_mut()
            .ok_or_else(|| invalid("source7 block discovery missing"))?;
        if active.is_some() {
            discovery.observe_outside(self.block_count as u64 + 1, w.fifo)
        } else {
            discovery.observe(self.block_count as u64 + 1, w.fifo, &input)
        }
        .map_err(|_| invalid("source7 input discovery differs"))?;
        if !include {
            if self.header.source_kind != PopulationSource::PreparedOwnerBlocksV8 {
                return Err(invalid("source7 cannot exclude original owner phases"));
            }
            if let Some(i) = active {
                if self.owners[i].state.phase().is_some() {
                    self.owners[i].domain.owner_offered += 1;
                }
            }
            self.block_routes.eligible_route += 1;
            return self.counted(w.fifo, call, rows, observed_at_ns);
        }
        if let Some(i) = active {
            let o = &mut self.owners[i];
            if let Some(phase) = o.state.phase() {
                o.domain.owner_offered += 1;
                let input_membership = self.header.declaration.domain_policy
                    == StructuredServiceDomainPolicyV1::FrozenFitSupportV1
                    || self.header.declaration.schedule.phase_support.is_some()
                    || self
                        .header
                        .declaration
                        .nonnegative_envelope
                        .as_ref()
                        .is_some_and(|c| {
                            c.planning_estimator
                                == NonNegativePlanningEstimatorV1::IdentifiedFitJointCellsV1
                        });
                // Settlement, physical D, population and original FIFO have
                // already been validated. This gate only excludes legitimate
                // inputs outside an earlier frozen numerical phase's support.
                let membership = if !input_membership {
                    StructuredServiceInputMembershipV1::Eligible
                } else {
                    match &o.state {
                        State::Empty => StructuredServiceInputMembershipV1::Eligible,
                        State::Fitted(m) => {
                            m.service_input_membership(&input).map_err(numeric_error)?
                        }
                        State::Calibrated(m) => {
                            m.service_input_membership(&input).map_err(numeric_error)?
                        }
                        _ => unreachable!(),
                    }
                };
                match membership {
                    StructuredServiceInputMembershipV1::Eligible => {
                        let limit = self.header.declaration.schedule.maximum_phase_members
                            [phase_index(phase)];
                        if o.samples.len() >= limit
                            || o.samples.len() >= self.header.declaration.settings.max_phase_samples
                        {
                            return Err(CostProfileError::Limit("source7 phase member capacity"));
                        }
                        if self.serial_physical_workspace() {
                            self.reserve_physical_sample(i, &input)?;
                        } else {
                            let o = &mut self.owners[i];
                            let charge = input
                                .retained_numeric_bytes()
                                .and_then(|v| v.checked_mul(12))
                                .and_then(|v| {
                                    v.checked_add(
                                        4 * std::mem::size_of::<StructuredNumericObservationV2>(),
                                    )
                                })
                                .ok_or(CostProfileError::Limit("source7 workspace overflow"))?;
                            if self.numeric_bytes.checked_add(charge).is_none_or(|n| {
                                n > self.header.declaration.maximum_retained_numeric_bytes
                            }) {
                                return Err(CostProfileError::Limit(
                                    "source7 shared workspace capacity",
                                ));
                            }
                            o.workspace_bytes = o
                                .workspace_bytes
                                .checked_add(charge)
                                .ok_or(CostProfileError::Limit("source7 workspace overflow"))?;
                        }
                        let o = &mut self.owners[i];
                        o.domain.eligible += 1;
                        o.oldest = o.oldest.min(observed_at_ns);
                        o.newest = o.newest.max(observed_at_ns);
                        o.samples.push(StructuredNumericObservationV2 {
                            source: self.header.capture_identity,
                            protocol: self.header.protocol,
                            ordinal: w.fifo,
                            membership: StructuredMemberBindingV2 {
                                rule_signature: o.contract.membership_rule,
                                offered_ordinal: w.ticket,
                                member_ordinal: (o.prior_members + o.samples.len() + 1) as u64,
                                phase,
                            },
                            call_id: call,
                            fingerprint: self.header.fingerprint.clone().into(),
                            input,
                            boundary: CostBoundary::PreparationToHostSettledV1,
                            outcome: WaveObservationOutcome::Completed,
                            observed_at_ns,
                            wall_ns,
                        });
                    }
                    StructuredServiceInputMembershipV1::OutsideFitSupport => {
                        o.domain.outside_fit_support += 1
                    }
                    StructuredServiceInputMembershipV1::OutsideResidualSupport => {
                        o.domain.outside_residual_support += 1
                    }
                }
            }
        }
        self.block_routes.eligible_route += 1;
        self.counted(w.fifo, call, rows, observed_at_ns)
    }
    pub(in super::super) fn outside(
        &mut self,
        w: &StructuredServiceOutsideRouteV7,
    ) -> Result<(), CostProfileError> {
        if self.header.declaration.route_population.is_all_attempts() {
            return Err(invalid("source7 undeclared route exclusion"));
        }
        let call = w.evidence.call_id();
        let opened = self.position(w.ticket, w.fifo, w.issued_at_ns, call)?;
        let (rows, observed) = route_population::validate_settlement_parts(
            &self.header.fingerprint,
            self.header.opening.monotonic_ns,
            opened,
            w.issued_at_ns,
            &w.evidence,
            &mut self.frontiers,
        )?;
        self.discovery
            .as_mut()
            .unwrap()
            .observe_outside(self.block_count as u64 + 1, w.fifo)
            .map_err(|_| invalid("source7 input discovery differs"))?;
        self.block_routes.outside_declared_route += 1;
        self.counted(w.fifo, call, rows, observed)
    }
    pub(in super::super) fn not_submitted(
        &mut self,
        a: &StructuredServiceNoSubmissionV7,
    ) -> Result<(), CostProfileError> {
        if !self
            .header
            .declaration
            .route_population
            .allows_no_submission()
        {
            return Err(invalid("source7 undeclared no-submission exclusion"));
        }
        let opened = self.position(a.ticket, a.fifo, a.issued_at_ns, a.call_id())?;
        let observed = a.validate_settlement(&self.header.fingerprint, opened)?;
        self.discovery
            .as_mut()
            .unwrap()
            .observe_outside(self.block_count as u64 + 1, a.fifo)
            .map_err(|_| invalid("source7 input discovery differs"))?;
        self.block_routes.no_submission += 1;
        self.counted(a.fifo, a.call_id(), 0, observed)
    }
}
