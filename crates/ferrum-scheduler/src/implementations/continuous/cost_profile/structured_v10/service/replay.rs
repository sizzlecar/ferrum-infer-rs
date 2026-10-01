//! One physical stream, complete offered windows, independent owner fits.
//! The same state machine freezes live worker output and verifies file replay.
use super::*;
mod memory;
use memory::RetainedCalls;

enum ChildState {
    Empty,
    Fitted(FittedStructuredModelV2),
    Calibrated(CalibratedStructuredModelV2),
    Qualified(QualifiedStructuredModelV2),
    Failed(String),
}

/// Replayed from immutable Prepared inputs after all physical checks. This
/// classifier deliberately has no access to wave timing or prediction values.
fn select_domain_population(
    state: &ChildState,
    samples: Vec<StructuredNumericObservationV2>,
    preceding_eligible: usize,
) -> Result<
    (
        Vec<StructuredNumericObservationV2>,
        StructuredServiceDomainFreezeV1,
    ),
    CostProfileError,
> {
    let mut domain = StructuredServiceDomainFreezeV1 {
        owner_offered: samples.len(),
        eligible: 0,
        outside_fit_support: 0,
        outside_residual_support: 0,
        unclassified_failed_owner: 0,
        frozen_domain_parameters_sha256: match state {
            ChildState::Fitted(model) => Some(model.parameters_signature()),
            ChildState::Calibrated(model) => Some(model.parameters_signature()),
            _ => None,
        },
    };
    let mut eligible = Vec::with_capacity(samples.len());
    for mut sample in samples {
        let membership = match state {
            ChildState::Empty => StructuredServiceInputMembershipV1::Eligible,
            ChildState::Fitted(model) => model
                .service_input_membership(&sample.input)
                .map_err(numeric_error)?,
            ChildState::Calibrated(model) => model
                .service_input_membership(&sample.input)
                .map_err(numeric_error)?,
            ChildState::Failed(_) => {
                domain.unclassified_failed_owner += 1;
                continue;
            }
            ChildState::Qualified(_) => {
                return Err(invalid("source6 classification after qualification"))
            }
        };
        match membership {
            StructuredServiceInputMembershipV1::Eligible => {
                domain.eligible += 1;
                // Offered tickets and original FIFO/call/clock stay unchanged.
                // Only the declared eligible-member coordinate is contiguous.
                sample.membership.member_ordinal = preceding_eligible
                    .checked_add(domain.eligible)
                    .and_then(|n| u64::try_from(n).ok())
                    .ok_or(CostProfileError::Limit(
                        "source6 eligible member ordinal overflow",
                    ))?;
                eligible.push(sample);
            }
            StructuredServiceInputMembershipV1::OutsideFitSupport => {
                domain.outside_fit_support += 1
            }
            StructuredServiceInputMembershipV1::OutsideResidualSupport => {
                domain.outside_residual_support += 1
            }
        }
    }
    Ok((eligible, domain))
}
pub struct StructuredServiceCollectorV6 {
    pub(super) header: StructuredServiceHeaderV6,
    limits: CostProfileLoadLimits,
    prefix: Sha256,
    prefix_bytes: u64,
    next_phase: usize,
    opened: Option<u64>,
    last_freeze: u64,
    offered: u64,
    phase_count: usize,
    route_counts: [StructuredServiceRouteCountsV1; 3],
    last_fifo: u64,
    calls: RetainedCalls,
    frontiers: physical::Frontiers,
    states: Vec<ChildState>,
    samples: Vec<Vec<StructuredNumericObservationV2>>,
    member_counts: Vec<usize>,
    pub(super) phases: Vec<Vec<StructuredPhaseProvenanceV10>>,
    pub(super) child_ages: Vec<(u64, u64)>,
    phase_workspace_bytes: usize,
    pub(super) total_rows: u64,
    pub(super) oldest: u64,
    pub(super) newest: u64,
    pub(super) closing: Option<StructuredServiceClockV6>,
    poisoned: bool,
}
impl StructuredServiceCollectorV6 {
    pub fn new(
        header: StructuredServiceHeaderV6,
        limits: CostProfileLoadLimits,
    ) -> Result<Self, CostProfileError> {
        limits.validate()?;
        header.validate()?;
        let count = header.declaration.scopes.len();
        let mut out = Self {
            header,
            limits,
            prefix: Sha256::new(),
            prefix_bytes: 0,
            next_phase: 0,
            opened: None,
            last_freeze: 0,
            offered: 0,
            phase_count: 0,
            route_counts: [StructuredServiceRouteCountsV1::default(); 3],
            last_fifo: 0,
            calls: RetainedCalls::default(),
            frontiers: Default::default(),
            states: (0..count).map(|_| ChildState::Empty).collect(),
            samples: (0..count).map(|_| Vec::new()).collect(),
            member_counts: vec![0; count],
            phases: (0..count).map(|_| Vec::new()).collect(),
            child_ages: vec![(u64::MAX, 0); count],
            phase_workspace_bytes: 0,
            total_rows: 0,
            oldest: u64::MAX,
            newest: 0,
            closing: None,
            poisoned: false,
        };
        let header = out.header.clone();
        out.check_retained_capacity(0)?;
        out.append(&header)?;
        Ok(out)
    }
    fn append(&mut self, value: &impl Serialize) -> Result<(), CostProfileError> {
        let bytes = record_bytes(value)?;
        let n = self
            .prefix_bytes
            .checked_add(bytes.len() as u64)
            .ok_or(CostProfileError::Limit("source6 bytes overflow"))?;
        if n > self.header.maximum_file_bytes || n > self.limits.max_file_bytes.get() as u64 {
            self.poisoned = true;
            return Err(CostProfileError::Limit("source6 byte capacity"));
        }
        self.prefix.update(&bytes);
        self.prefix_bytes = n;
        Ok(())
    }
    /// Failed input permanently closes this candidate; callers cannot retry a
    /// different record and silently replace a failed or slow member.
    pub fn push(&mut self, record: &StructuredServiceRecordV6) -> Result<(), CostProfileError> {
        if self.poisoned || self.closing.is_some() {
            return Err(invalid("source6 is closed"));
        }
        let result = self.push_inner(record);
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
    fn push_inner(&mut self, record: &StructuredServiceRecordV6) -> Result<(), CostProfileError> {
        match record {
            StructuredServiceRecordV6::PhaseOpen {
                phase,
                opened_at_ns,
                fifo_cutoff,
            } => {
                if self.next_phase >= 3
                    || self.opened.is_some()
                    || phase_index(*phase) != self.next_phase
                    || *opened_at_ns < self.last_freeze.max(self.header.opening.monotonic_ns)
                    || *fifo_cutoff < self.last_fifo
                    || opened_at_ns
                        .checked_sub(self.header.opening.monotonic_ns)
                        .is_none_or(|age| age > self.header.declaration.maximum_window_ns)
                {
                    return Err(invalid("source6 phase open or explicit gap differs"));
                }
                self.opened = Some(*opened_at_ns);
                self.last_fifo = *fifo_cutoff;
                self.phase_count = 0;
                self.frontiers = Default::default();
            }
            StructuredServiceRecordV6::Completed { wave } => self.wave(wave)?,
            StructuredServiceRecordV6::OutsideDeclaredRoute { wave } => self.outside_route(wave)?,
            StructuredServiceRecordV6::NotSubmitted { attempt } => self.no_submission(attempt)?,
            StructuredServiceRecordV6::TicketFailed { .. } => {
                return Err(invalid("source6 contains a failed offered ticket"))
            }
            StructuredServiceRecordV6::PhaseFreeze {
                phase,
                frozen_at_ns,
                source_prefix_bytes,
                source_prefix_sha256,
                route_population,
                children,
            } => {
                let expected = self.freeze_inner(*frozen_at_ns, Some(children))?;
                let StructuredServiceRecordV6::PhaseFreeze {
                    phase: p,
                    source_prefix_bytes: b,
                    source_prefix_sha256: h,
                    route_population: r,
                    children: c,
                    ..
                } = expected
                else {
                    unreachable!()
                };
                if *phase != p
                    || *source_prefix_bytes != b
                    || *source_prefix_sha256 != h
                    || *route_population != r
                    || *children != c
                {
                    return Err(invalid("source6 frozen population/parameters differ"));
                }
            }
            StructuredServiceRecordV6::Footer {
                offered,
                accepted_fifo_cutoff,
                closing,
                failure,
            } => {
                if self.next_phase != 3
                    || self.opened.is_some()
                    || failure.is_some()
                    || *offered != self.offered
                    || *accepted_fifo_cutoff != self.last_fifo
                    || closing.monotonic_ns < self.last_freeze
                    || closing.wall_unix_ns < self.header.opening.wall_unix_ns
                {
                    return Err(invalid("source6 footer is not complete"));
                }
                self.closing = Some(*closing);
            }
        }
        self.check_retained_capacity(0)?;
        self.append(record)
    }
    fn wave(&mut self, w: &StructuredServiceWaveV6) -> Result<(), CostProfileError> {
        let opened = self
            .opened
            .ok_or_else(|| invalid("source6 wave outside an open phase"))?;
        if phase_index(w.phase) != self.next_phase
            || self.phase_count >= self.header.declaration.phase_offered_waves[self.next_phase]
            || w.ticket != self.offered + 1
            || w.fifo <= self.last_fifo
            || !self.calls.insert(w.host_stages.call_id)
            || self.calls.len() > self.limits.max_samples.get()
            || w.issued_at_ns
                .checked_sub(self.header.opening.monotonic_ns)
                .is_none_or(|age| age > self.header.declaration.maximum_window_ns)
        {
            return Err(invalid("source6 ticket/FIFO/call/window differs"));
        }
        if !self.header.declaration.route_population.is_all_attempts() {
            route_population::validate_eligible(w)?;
        }
        let (input, wall, observed) =
            physical::validate(&self.header, opened, w, &mut self.frontiers)?;
        self.total_rows = self
            .total_rows
            .checked_add(input.physical_host_rows().len() as u64)
            .ok_or(CostProfileError::Limit("source6 rows overflow"))?;
        if self.total_rows > self.limits.max_total_shape_rows.get() as u64 {
            return Err(CostProfileError::Limit("source6 rows capacity"));
        }
        self.oldest = self.oldest.min(observed);
        self.newest = self.newest.max(observed);
        if let Some(child) = self
            .header
            .declaration
            .scopes
            .iter()
            .position(|s| &s.owner == input.owner())
        {
            // Failed-child and nonmember waves also passed all physical checks.
            // Retain the complete numerical population until this phase closes.
            self.child_ages[child].0 = self.child_ages[child].0.min(observed);
            self.child_ages[child].1 = self.child_ages[child].1.max(observed);
            let contract = self.header.child_contract(child)?;
            let next = self.samples[child].len() + 1;
            if next > self.header.declaration.settings.max_phase_samples {
                return Err(CostProfileError::Limit("source6 child population capacity"));
            }
            let charge = input
                .retained_numeric_bytes()
                .and_then(|n| n.checked_mul(12))
                .and_then(|n| {
                    n.checked_add(std::mem::size_of::<StructuredNumericObservationV2>() * 4)
                })
                .ok_or(CostProfileError::Limit(
                    "source6 numeric allocation overflow",
                ))?;
            self.check_retained_capacity(charge)?;
            self.phase_workspace_bytes =
                self.phase_workspace_bytes
                    .checked_add(charge)
                    .ok_or(CostProfileError::Limit(
                        "source6 numeric allocation overflow",
                    ))?;
            self.samples[child].push(StructuredNumericObservationV2 {
                source: contract.capture_identity,
                protocol: contract.protocol,
                ordinal: w.fifo,
                membership: StructuredMemberBindingV2 {
                    rule_signature: contract.membership_rule,
                    offered_ordinal: w.ticket,
                    member_ordinal: (self.member_counts[child] + next) as u64,
                    phase: w.phase,
                },
                call_id: w.host_stages.call_id,
                fingerprint: self.header.fingerprint.clone().into(),
                input,
                boundary: CostBoundary::PreparationToHostSettledV1,
                outcome: WaveObservationOutcome::Completed,
                observed_at_ns: observed,
                wall_ns: wall,
            });
        }
        self.offered += 1;
        self.phase_count += 1;
        self.route_counts[self.next_phase].attempted += 1;
        self.route_counts[self.next_phase].eligible_route += 1;
        self.last_fifo = w.fifo;
        Ok(())
    }
    fn outside_route(
        &mut self,
        w: &StructuredServiceOutsideRouteV6,
    ) -> Result<(), CostProfileError> {
        let opened = self
            .opened
            .ok_or_else(|| invalid("source6 outside attempt without open phase"))?;
        if self.header.declaration.route_population.is_all_attempts()
            || phase_index(w.phase) != self.next_phase
            || self.phase_count >= self.header.declaration.phase_offered_waves[self.next_phase]
            || w.ticket != self.offered + 1
            || w.fifo <= self.last_fifo
            || !self.calls.insert(w.call_id())
            || self.calls.len() > self.limits.max_samples.get()
            || w.issued_at_ns
                .checked_sub(self.header.opening.monotonic_ns)
                .is_none_or(|age| age > self.header.declaration.maximum_window_ns)
        {
            return Err(invalid(
                "source6 outside ticket/FIFO/call/window/policy differs",
            ));
        }
        let (rows, observed) =
            route_population::validate(&self.header, opened, w, &mut self.frontiers)?;
        self.total_rows = self
            .total_rows
            .checked_add(rows as u64)
            .filter(|n| *n <= self.limits.max_total_shape_rows.get() as u64)
            .ok_or(CostProfileError::Limit("source6 outside rows capacity"))?;
        self.oldest = self.oldest.min(observed);
        self.newest = self.newest.max(observed);
        self.offered += 1;
        self.phase_count += 1;
        self.route_counts[self.next_phase].attempted += 1;
        self.route_counts[self.next_phase].outside_declared_route += 1;
        self.last_fifo = w.fifo;
        Ok(())
    }
    fn no_submission(
        &mut self,
        attempt: &StructuredServiceNoSubmissionV6,
    ) -> Result<(), CostProfileError> {
        let opened = self
            .opened
            .ok_or_else(|| invalid("source6 no-submission without open phase"))?;
        if !self
            .header
            .declaration
            .route_population
            .allows_no_submission()
            || phase_index(attempt.phase) != self.next_phase
            || self.phase_count >= self.header.declaration.phase_offered_waves[self.next_phase]
            || attempt.ticket != self.offered + 1
            || attempt.fifo <= self.last_fifo
            || !self.calls.insert(attempt.call_id())
            || self.calls.len() > self.limits.max_samples.get()
            || attempt
                .issued_at_ns
                .checked_sub(self.header.opening.monotonic_ns)
                .is_none_or(|age| age > self.header.declaration.maximum_window_ns)
        {
            return Err(invalid(
                "source6 no-submission ticket/FIFO/call/window/policy differs",
            ));
        }
        let returned = attempt.validate_settlement(&self.header.fingerprint, opened)?;
        // No numerical observation or owner frontier transition exists here.
        // Original closure still constrains source clock order and total age.
        self.oldest = self.oldest.min(returned);
        self.newest = self.newest.max(returned);
        self.offered += 1;
        self.phase_count += 1;
        self.route_counts[self.next_phase].attempted += 1;
        self.route_counts[self.next_phase].no_submission += 1;
        self.last_fifo = attempt.fifo;
        Ok(())
    }
    pub fn route_population_counts(&self) -> [StructuredServiceRouteCountsV1; 3] {
        self.route_counts
    }

    /// Optional worker diagnostic, before freeze consumes these original rows.
    /// Returns at most one fixed-size witness per declared owner. Neither the
    /// source records nor retained samples/model states are changed. Callers
    /// should invoke this only when the corresponding debug target is enabled.
    pub fn first_outside_fit_diagnostics(&self) -> Vec<(usize, StructuredFitSupportDiagnosticV1)> {
        if self.header.declaration.domain_policy
            != StructuredServiceDomainPolicyV1::FrozenFitSupportV1
        {
            return Vec::new();
        }
        let mut diagnostics = Vec::new();
        for (child, (state, samples)) in self.states.iter().zip(&self.samples).enumerate() {
            for sample in samples {
                let result = match state {
                    ChildState::Fitted(model) => model.diagnose_service_fit_support(&sample.input),
                    ChildState::Calibrated(model) => {
                        model.diagnose_service_fit_support(&sample.input)
                    }
                    // No fitted domain exists, or classification is already over.
                    // A failed owner must not be misreported as out of support.
                    _ => break,
                };
                match result {
                    Ok(Some(diagnostic)) => {
                        diagnostics.push((child, diagnostic));
                        break;
                    }
                    Ok(None) => {}
                    // The original classifier will reject this invalid sample.
                    // It is not an OutsideFitSupport witness and cannot be
                    // skipped to report a later row from an invalid population.
                    Err(_) => break,
                }
            }
        }
        diagnostics
    }

    /// CPU/IO worker only. Finishes the immutable fixed window once.
    /// AllOffered preserves the original full owner population. Explicit frozen
    /// domain selection changes membership by input only, retaining every raw
    /// offered record and checking every eligible heldout result.
    pub fn freeze(&mut self, now: u64) -> Result<StructuredServiceRecordV6, CostProfileError> {
        if self.poisoned || self.closing.is_some() {
            return Err(invalid("source6 is closed"));
        }
        let result = (|| {
            let record = self.freeze_inner(now, None)?;
            self.append(&record)?;
            Ok(record)
        })();
        if result.is_err() {
            self.poisoned = true;
        }
        result
    }
    fn freeze_inner(
        &mut self,
        now: u64,
        replay_children: Option<&[StructuredServiceChildFreezeV6]>,
    ) -> Result<StructuredServiceRecordV6, CostProfileError> {
        if self.next_phase >= 3
            || self.opened.is_none()
            || self.phase_count != self.header.declaration.phase_offered_waves[self.next_phase]
            || now < self.newest
            || now < self.opened.unwrap()
            || now
                .checked_sub(self.header.opening.monotonic_ns)
                .is_none_or(|age| age > self.header.declaration.maximum_window_ns)
        {
            return Err(invalid("source6 incomplete or expired offered window"));
        }
        let phase = phase_at(self.next_phase);
        let nonnegative = self.header.declaration.domain_policy
            == StructuredServiceDomainPolicyV1::NonNegativePhysicalEnvelopeV1;
        if let Some(children) = replay_children {
            if children.len() != self.states.len()
                || children.iter().enumerate().any(|(i, c)| {
                    c.child != i
                        || (c.nonnegative_fit_certificate.is_some()
                            && (!nonnegative || phase != StructuredPhaseV2::Fit))
                        || (nonnegative
                            && phase == StructuredPhaseV2::Fit
                            && ((c.failure.is_none()
                                && (c.parameters_sha256.is_none()
                                    || c.nonnegative_fit_certificate.is_none()))
                                || (c.failure.is_some()
                                    && (c.parameters_sha256.is_some()
                                        || c.nonnegative_fit_certificate.is_some()))))
                })
            {
                return Err(invalid(
                    "source6 Fit certificate policy/phase/child differs",
                ));
            }
        }
        let hash = self.prefix.clone().finalize().into();
        let mut children = Vec::with_capacity(self.states.len());
        for i in 0..self.states.len() {
            let samples = std::mem::take(&mut self.samples[i]);
            let state = std::mem::replace(&mut self.states[i], ChildState::Empty);
            let (samples, domain) = if self.header.declaration.domain_policy
                == StructuredServiceDomainPolicyV1::FrozenFitSupportV1
            {
                let (samples, domain) =
                    select_domain_population(&state, samples, self.member_counts[i])?;
                (samples, Some(domain))
            } else {
                (samples, None)
            };
            let close = StructuredServiceWindowCloseV2::new(phase, hash, &samples);
            let result: Result<ChildState, String> = match (phase, state) {
                (_, ChildState::Failed(reason)) => Err(reason),
                (StructuredPhaseV2::Fit, ChildState::Empty) => {
                    let frozen_certificate = replay_children
                        .and_then(|children| children[i].nonnegative_fit_certificate.as_ref());
                    let fitted = if let Some(certificate) = frozen_certificate {
                        FittedStructuredModelV2::fit_service_window_from_certificate(
                            self.header.fingerprint.clone().into(),
                            self.header.declaration.settings.clone(),
                            self.header.declaration.scopes[i].clone(),
                            self.header.child_contract(i)?,
                            close,
                            &samples,
                            now,
                            certificate.clone(),
                        )
                    } else {
                        FittedStructuredModelV2::fit_service_window(
                            self.header.fingerprint.clone().into(),
                            self.header.declaration.settings.clone(),
                            self.header.declaration.scopes[i].clone(),
                            self.header.child_contract(i)?,
                            close,
                            &samples,
                            now,
                        )
                    };
                    fitted.map(ChildState::Fitted).map_err(|e| format!("{e:?}"))
                }
                (StructuredPhaseV2::Residual, ChildState::Fitted(f)) => f
                    .calibrate_service_window(close, &samples, now)
                    .map(ChildState::Calibrated)
                    .map_err(|e| format!("{e:?}")),
                (StructuredPhaseV2::Qualification, ChildState::Calibrated(c)) => c
                    .qualify_service_window(close, &samples, now)
                    .map(ChildState::Qualified)
                    .map_err(|e| format!("{e:?}")),
                _ => return Err(invalid("source6 numerical phase state differs")),
            };
            self.member_counts[i] += samples.len();
            let (state, parameters, failure) = match result {
                Ok(s) => {
                    let signature = match &s {
                        ChildState::Fitted(f) => f.parameters_signature(),
                        ChildState::Calibrated(c) => c.parameters_signature(),
                        ChildState::Qualified(q) => q.parameters_signature(),
                        _ => unreachable!(),
                    };
                    (s, Some(signature), None)
                }
                Err(reason) => (ChildState::Failed(reason.clone()), None, Some(reason)),
            };
            let nonnegative_fit_certificate = match &state {
                ChildState::Fitted(f) if phase == StructuredPhaseV2::Fit => {
                    f.nonnegative_fit_certificate().cloned()
                }
                _ => None,
            };
            self.states[i] = state;
            if let Some(parameters_sha256) = parameters {
                self.phases[i].push(StructuredPhaseProvenanceV10 {
                    phase: match phase {
                        StructuredPhaseV2::Fit => StructuredProfilePhaseV10::Fit,
                        StructuredPhaseV2::Residual => StructuredProfilePhaseV10::Residual,
                        StructuredPhaseV2::Qualification => {
                            StructuredProfilePhaseV10::Qualification
                        }
                    },
                    members: samples.len(),
                    member_cutoff: self.member_counts[i] as u64,
                    accepted_fifo_cutoff: self.last_fifo,
                    frozen_at_ns: now,
                    source_prefix_bytes: self.prefix_bytes,
                    source_prefix_sha256: hash,
                    parameters_sha256,
                });
            }
            children.push(StructuredServiceChildFreezeV6 {
                child: i,
                members: samples.len(),
                parameters_sha256: parameters,
                failure,
                domain,
                nonnegative_fit_certificate,
            });
        }
        // All original sample vectors above have now been consumed and dropped.
        // Keep their full workspace reservation through the numerical work;
        // replace it only here with the measured surviving model payload.
        self.phase_workspace_bytes = 0;
        self.check_retained_capacity(0)?;
        self.next_phase += 1;
        self.opened = None;
        self.last_freeze = now;
        Ok(StructuredServiceRecordV6::PhaseFreeze {
            phase,
            frozen_at_ns: now,
            source_prefix_bytes: self.prefix_bytes,
            source_prefix_sha256: hash,
            route_population: (!self.header.declaration.route_population.is_all_attempts())
                .then_some(self.route_counts[self.next_phase - 1]),
            children,
        })
    }
    pub fn qualified_children(&self) -> usize {
        self.states
            .iter()
            .filter(|s| matches!(s, ChildState::Qualified(_)))
            .count()
    }
    pub fn offered(&self) -> u64 {
        self.offered
    }
    pub fn last_fifo(&self) -> u64 {
        self.last_fifo
    }
    pub(super) fn sealed_source(&self) -> Result<(u64, [u8; 32]), CostProfileError> {
        if self.poisoned || self.closing.is_none() {
            return Err(invalid("source6 missing successful physical footer"));
        }
        Ok((self.prefix_bytes, self.prefix.clone().finalize().into()))
    }
    pub(super) fn models(&self) -> impl Iterator<Item = (usize, &QualifiedStructuredModelV2)> {
        self.states.iter().enumerate().filter_map(|(i, s)| match s {
            ChildState::Qualified(m) => Some((i, m)),
            _ => None,
        })
    }
    pub(super) fn into_models(
        self,
    ) -> Result<Vec<(usize, QualifiedStructuredModelV2)>, CostProfileError> {
        if self.poisoned || self.closing.is_none() {
            return Err(invalid("source6 missing successful physical footer"));
        }
        let models = self
            .states
            .into_iter()
            .enumerate()
            .filter_map(|(i, s)| match s {
                ChildState::Qualified(m) => Some((i, m)),
                _ => None,
            })
            .collect::<Vec<_>>();
        if models.is_empty() {
            return Err(invalid("source6 has no qualified child"));
        }
        Ok(models)
    }
}
pub(super) fn record_bytes(value: &impl Serialize) -> Result<Vec<u8>, CostProfileError> {
    let mut bytes = serde_json::to_vec(value)?;
    if bytes.len() >= 8 * 1024 * 1024 {
        return Err(CostProfileError::Limit(
            "source6 record exceeds original 8MiB bound",
        ));
    }
    bytes.push(b'\n');
    Ok(bytes)
}
pub(super) fn replay_source(
    bytes: &[u8],
    limits: &CostProfileLoadLimits,
) -> Result<StructuredServiceCollectorV6, CostProfileError> {
    if bytes.is_empty() || bytes.len() > limits.max_file_bytes.get() || bytes.last() != Some(&b'\n')
    {
        return Err(invalid("source6 incomplete/oversized stream"));
    }
    let mut lines = bytes.split_inclusive(|b| *b == b'\n');
    let first = lines
        .next()
        .ok_or_else(|| invalid("source6 missing header"))?;
    if first.len() > 8 * 1024 * 1024 {
        return Err(CostProfileError::Limit("source6 header line limit"));
    }
    let header: StructuredServiceHeaderV6 = serde_json::from_slice(first)?;
    if record_bytes(&header)? != first {
        return Err(invalid("source6 requires canonical header encoding"));
    }
    let mut replay = StructuredServiceCollectorV6::new(header, limits.clone())?;
    for line in lines {
        if line.len() > 8 * 1024 * 1024 {
            return Err(CostProfileError::Limit("source6 record line limit"));
        }
        let record: StructuredServiceRecordV6 = serde_json::from_slice(line)?;
        if record_bytes(&record)? != line {
            return Err(invalid("source6 requires canonical record encoding"));
        }
        replay.push(&record)?;
    }
    if replay.closing.is_none() {
        return Err(invalid("source6 missing footer"));
    }
    Ok(replay)
}
