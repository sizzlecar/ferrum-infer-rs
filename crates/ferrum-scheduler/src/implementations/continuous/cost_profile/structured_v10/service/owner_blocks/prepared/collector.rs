use super::super::discovery::PopulationKey;
use super::*;
use crate::implementations::continuous::cost_profile::structured_v10::shared::PreparedCohortLedgerV8;

/// The common population record types keep their original shape, but this is
/// one source8 stream with its own header, protocol, declaration and hash.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StructuredPreparedOwnerBlockRecordV8 {
    Population(StructuredServiceRecordV7),
    Cohort(StructuredCohortEventV8),
    Preparation(StructuredPreparationEventV8),
    PreparationDisposition(StructuredPreparationDispositionV8),
    Tail(StructuredPreparedTailRecordV8),
}

/// A unique, original non-numerical preparation outcome. The DTO retains the
/// complete original private-proof record; it cannot manufacture that proof.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum StructuredPreparationDispositionV8 {
    PreparationNotSubmitted {
        phase: StructuredProfilePhaseV10,
        cohort: usize,
        attempt: StructuredServiceNoSubmissionV7,
    },
}
impl StructuredPreparedOwnerBlockRecordV8 {
    pub fn preparation_not_submitted(
        phase: StructuredProfilePhaseV10,
        cohort: usize,
        attempt: StructuredServiceNoSubmissionV7,
    ) -> Self {
        Self::PreparationDisposition(
            StructuredPreparationDispositionV8::PreparationNotSubmitted {
                phase,
                cohort,
                attempt,
            },
        )
    }
}

pub struct StructuredPreparedOwnerBlockCheckpointV8 {
    pub(super) header: StructuredPreparedOwnerBlockHeaderV8,
    pub(super) population: StructuredServiceCheckpointV7,
}

pub struct StructuredPreparedOwnerBlockCollectorV8 {
    pub(super) header: StructuredPreparedOwnerBlockHeaderV8,
    pub(super) population: StructuredServiceCollectorV7,
    pub(super) lifecycle: PreparedCohortLedgerV8,
    limits: CostProfileLoadLimits,
    preparation_attempts: u64,
    adapter_bytes: usize,
    phase_cohort: Option<(usize, usize)>,
    // Original owner identity, original attempt, and numerical phase. The
    // driver pass label never supplies or changes a numerical phase.
    cohort_memberships: Vec<(PopulationKey, u64, StructuredPhaseV2)>,
    excluded_cohort_phase_attempts: u64,
    phase_exclusions: Vec<StructuredCohortPhaseExclusionAuditV8>,
}
#[derive(Debug, Clone, Serialize)]
pub struct StructuredCohortPhaseExclusionAuditV8 {
    pub owner_attempt_id: u64,
    pub phase: StructuredPhaseV2,
    pub excluded_original_offers: u64,
}
#[derive(Debug, Clone, Serialize)]
pub struct StructuredPreparedOwnerBlockAuditV8 {
    pub cohort_phase_policy: StructuredPreparedCohortPhasePolicyV8,
    pub tail_policy: StructuredPreparedTailPolicyV8,
    pub sealed_partial_tail: Option<StructuredPreparedPartialTailV8>,
    pub preparation_attempts: u64,
    pub excluded_cohort_phase_attempts: u64,
    pub phase_exclusions: Vec<StructuredCohortPhaseExclusionAuditV8>,
    pub population: StructuredServiceAuditV7,
}
impl StructuredPreparedOwnerBlockCollectorV8 {
    pub fn prepared_audit(&self) -> StructuredPreparedOwnerBlockAuditV8 {
        StructuredPreparedOwnerBlockAuditV8 {
            cohort_phase_policy: self.header.cohort_phase_policy(),
            tail_policy: self.header.tail_policy(),
            sealed_partial_tail: self.population.prepared_tail_audit(),
            preparation_attempts: self.preparation_attempts,
            excluded_cohort_phase_attempts: self.excluded_cohort_phase_attempts,
            phase_exclusions: self.phase_exclusions.clone(),
            population: self.population.audit(),
        }
    }

    pub fn new(
        header: StructuredPreparedOwnerBlockHeaderV8,
        limits: CostProfileLoadLimits,
    ) -> Result<Self, CostProfileError> {
        Self::new_with_budget(header, limits, None, None)
    }
    /// Original source8 journal is hashed incrementally. Its cumulative work
    /// allowance does not allocate that many resident source bytes.
    pub fn new_streaming(
        header: StructuredPreparedOwnerBlockHeaderV8,
        limits: CostProfileLoadLimits,
        maximum_encoded_source_bytes: std::num::NonZeroU64,
    ) -> Result<Self, CostProfileError> {
        Self::new_with_budget(header, limits, Some(maximum_encoded_source_bytes), None)
    }
    /// Optional archival observes the exact canonical stream; it grants no
    /// numerical or execution authority and owns no extra collector quota.
    pub fn new_with_record_sink(
        header: StructuredPreparedOwnerBlockHeaderV8,
        limits: CostProfileLoadLimits,
        maximum_encoded_source_bytes: Option<std::num::NonZeroU64>,
        sink: Box<dyn StructuredSourceRecordSinkV1>,
    ) -> Result<Self, CostProfileError> {
        Self::new_with_budget(header, limits, maximum_encoded_source_bytes, Some(sink))
    }
    fn new_with_budget(
        header: StructuredPreparedOwnerBlockHeaderV8,
        limits: CostProfileLoadLimits,
        maximum_encoded_source_bytes: Option<std::num::NonZeroU64>,
        record_sink: Option<Box<dyn StructuredSourceRecordSinkV1>>,
    ) -> Result<Self, CostProfileError> {
        header.validate()?;
        limits.validate()?;
        let original = record_bytes_v7(&header)?;
        let common = super::super::collector::PopulationHeader {
            capture_identity: header.capture_identity,
            protocol: header.protocol,
            fingerprint: header.fingerprint.clone(),
            producer: header.producer.clone(),
            opening: header.opening,
            monotonic_domain: header.monotonic_domain.clone(),
            declaration: header.declaration.population.clone(),
            declaration_sha256: header.declaration_sha256,
            maximum_file_bytes: header.maximum_file_bytes,
            source_kind: super::super::collector::PopulationSource::PreparedOwnerBlocksV8,
        };
        let population = if let Some(sink) = record_sink {
            StructuredServiceCollectorV7::from_original_header_with_record_sink(
                common,
                limits.clone(),
                &original,
                maximum_encoded_source_bytes,
                sink,
            )?
        } else {
            match maximum_encoded_source_bytes {
                Some(budget) => StructuredServiceCollectorV7::from_original_header_streaming(
                    common,
                    limits.clone(),
                    &original,
                    budget,
                )?,
                None => StructuredServiceCollectorV7::from_original_header(
                    common,
                    limits.clone(),
                    &original,
                )?,
            }
        };
        let lifecycle = PreparedCohortLedgerV8::new(
            header.declaration.cohort_plan.clone(),
            header.declaration.prefix_plan.clone(),
            header.declaration.native_prefix_acquisition.clone(),
        );
        let mut out = Self {
            header,
            population,
            lifecycle,
            limits,
            preparation_attempts: 0,
            adapter_bytes: 0,
            phase_cohort: None,
            cohort_memberships: Vec::new(),
            excluded_cohort_phase_attempts: 0,
            phase_exclusions: Vec::new(),
        };
        out.check_retained()?;
        Ok(out)
    }
    pub fn source_receipt(&self) -> (u64, [u8; 32]) {
        self.population.source_receipt()
    }
    pub fn offered(&self) -> u64 {
        self.population.offered()
    }
    pub fn last_fifo(&self) -> u64 {
        self.population.last_fifo()
    }
    pub fn preparation_attempts(&self) -> u64 {
        self.preparation_attempts
    }
    pub fn audit(&self) -> StructuredServiceAuditV7 {
        self.population.audit()
    }
    pub fn qualified_children(&self) -> usize {
        self.population.qualified_children()
    }
    /// The driver contributes its live slots and auxiliary owned buffers to
    /// this same population ledger. It receives no independent memory quota.
    pub fn retain_external(&mut self, bytes: usize) -> Result<(), CostProfileError> {
        self.population.ensure_active()?;
        self.adapter_bytes = bytes;
        let result = self.check_retained();
        if result.is_err() {
            self.population.poison();
        }
        result
    }
    fn check_retained(&mut self) -> Result<(), CostProfileError> {
        let bytes =
            self.header
                .declaration
                .retained_payload_bytes()
                .and_then(|n| n.checked_add(self.lifecycle.retained_payload_bytes()?))
                .and_then(|n| {
                    n.checked_add(self.cohort_memberships.capacity().checked_mul(
                        std::mem::size_of::<(PopulationKey, u64, StructuredPhaseV2)>(),
                    )?)
                })
                .and_then(|n| n.checked_add(std::mem::size_of::<Self>()))
                .and_then(|n| n.checked_add(self.adapter_bytes))
                .and_then(|n| {
                    n.checked_add(
                        self.phase_exclusions
                            .capacity()
                            .checked_mul(std::mem::size_of::<
                                StructuredCohortPhaseExclusionAuditV8,
                            >())?,
                    )
                })
                .ok_or(CostProfileError::Limit("source8 retained payload overflow"))?;
        self.population.retain_external(bytes)
    }
    pub fn open_block(
        &mut self,
        now: u64,
        fifo: u64,
    ) -> Result<StructuredPreparedOwnerBlockRecordV8, CostProfileError> {
        if !self.lifecycle.idle() {
            return Err(invalid("source8 block crossed unfinished preparation"));
        }
        Ok(StructuredPreparedOwnerBlockRecordV8::Population(
            self.population.open_block(now, fifo)?,
        ))
    }
    pub fn close_block(
        &mut self,
        closing: StructuredServiceClockV7,
    ) -> Result<StructuredPreparedOwnerBlockRecordV8, CostProfileError> {
        if !self.lifecycle.idle() {
            return Err(invalid("source8 block crossed unfinished preparation"));
        }
        Ok(StructuredPreparedOwnerBlockRecordV8::Population(
            self.population.close_block(closing)?,
        ))
    }
    pub fn checkpoint(
        &mut self,
        closing: StructuredServiceClockV7,
    ) -> Result<
        (
            StructuredPreparedOwnerBlockRecordV8,
            StructuredPreparedOwnerBlockCheckpointV8,
        ),
        CostProfileError,
    > {
        self.lifecycle.complete()?;
        let (record, population) = self.population.checkpoint(closing)?;
        Ok((
            StructuredPreparedOwnerBlockRecordV8::Population(record),
            StructuredPreparedOwnerBlockCheckpointV8 {
                header: self.header.clone(),
                population,
            },
        ))
    }
    pub fn fail(
        &mut self,
        ticket: u64,
        fifo: u64,
        at_ns: u64,
        reason: impl Into<String>,
    ) -> Result<StructuredPreparedOwnerBlockRecordV8, CostProfileError> {
        Ok(StructuredPreparedOwnerBlockRecordV8::Population(
            self.population.fail(ticket, fifo, at_ns, reason)?,
        ))
    }
    pub fn stop(
        &mut self,
        closing: StructuredServiceClockV7,
    ) -> Result<StructuredPreparedOwnerBlockRecordV8, CostProfileError> {
        Ok(StructuredPreparedOwnerBlockRecordV8::Population(
            self.population.stop(closing)?,
        ))
    }
    pub fn push(
        &mut self,
        record: &StructuredPreparedOwnerBlockRecordV8,
    ) -> Result<(), CostProfileError> {
        let result = self.push_inner(record).and_then(|()| self.check_retained());
        if result.is_err() {
            self.population.poison();
        }
        result
    }
    fn push_inner(
        &mut self,
        record: &StructuredPreparedOwnerBlockRecordV8,
    ) -> Result<(), CostProfileError> {
        use StructuredPreparedOwnerBlockRecordV8 as R;
        if !matches!(
            record,
            R::Population(
                StructuredServiceRecordV7::Failed { .. } | StructuredServiceRecordV7::Footer { .. }
            )
        ) {
            self.population.ensure_active()?;
        }
        match record {
            R::Tail(tail) => {
                self.lifecycle.complete()?;
                self.population.push_prepared_tail(tail)?;
                self.population.append(record)?;
            }
            R::PreparationDisposition(
                StructuredPreparationDispositionV8::PreparationNotSubmitted {
                    phase,
                    cohort,
                    attempt,
                },
            ) => {
                if attempt.ticket > self.header.declaration.maximum_offered_waves as u64 {
                    return Err(CostProfileError::Limit("source8 original offer capacity"));
                }
                self.population.not_submitted(attempt)?;
                self.lifecycle.no_submission(
                    Some((phase.index(), *cohort)),
                    attempt.ticket,
                    attempt.call_id(),
                    attempt.finalized_at_ns(),
                    attempt.original_participants(),
                )?;
                self.preparation_attempts =
                    self.preparation_attempts
                        .checked_add(1)
                        .ok_or(CostProfileError::Limit(
                            "source8 preparation count overflow",
                        ))?;
                self.population.append(record)?;
            }
            R::Cohort(e) => {
                if !self.lifecycle.idle() {
                    return Err(invalid(
                        "source8 cohort event crossed unfinished preparation",
                    ));
                }
                self.lifecycle.event(e, &self.limits)?;
                self.population.append(record)?;
            }
            R::Preparation(e) => {
                if let Some(ticket) = e.offered() {
                    if ticket
                        != self
                            .offered()
                            .checked_add(1)
                            .ok_or(CostProfileError::Limit("source8 ticket overflow"))?
                        || ticket > self.header.declaration.maximum_offered_waves as u64
                    {
                        return Err(invalid("source8 preparation original offer differs"));
                    }
                }
                let position = e.position();
                let earliest = if let Some((ticket, fifo, issued, call, _, _)) = position {
                    self.population.position(ticket, fifo, issued, call)?
                } else {
                    self.header.opening.monotonic_ns
                };
                let settled = self.lifecycle.prepare(
                    e,
                    self.offered(),
                    self.header.declaration.maximum_offered_waves as u64,
                    self.last_fifo(),
                    &self.header.fingerprint,
                    self.header.opening.monotonic_ns,
                    earliest,
                    &self.limits,
                )?;
                if settled {
                    let (ticket, fifo, issued, call, rows, observed) = position
                        .ok_or_else(|| invalid("source8 preparation lacks original settlement"))?;
                    self.population
                        .preparation_counted(ticket, fifo, issued, call, rows, observed)?;
                    self.preparation_attempts =
                        self.preparation_attempts
                            .checked_add(1)
                            .ok_or(CostProfileError::Limit(
                                "source8 preparation count overflow",
                            ))?;
                }
                self.population.append(record)?;
            }
            R::Population(StructuredServiceRecordV7::Completed { wave }) => {
                if wave.ticket > self.header.declaration.maximum_offered_waves as u64 {
                    return Err(CostProfileError::Limit("source8 original offer capacity"));
                }
                let cohort = self.lifecycle.cohort()?;
                if self.phase_cohort != Some(cohort) {
                    self.cohort_memberships.clear();
                    self.phase_cohort = Some(cohort);
                }
                let (prepared, _) =
                    physical::original_prepared(&wave.host_stages, wave.independent.as_ref())?;
                self.lifecycle
                    .ordinary(&prepared, &wave.host_stages, wave.fifo)?;
                let bindings = &mut self.cohort_memberships;
                let exclusions = &mut self.phase_exclusions;
                let excluded_total = &mut self.excluded_cohort_phase_attempts;
                let cap = self.header.declaration.population.maximum_owners;
                self.population
                    .wave_with_membership(wave, |key, assignment| {
                        if let Some((attempt, phase)) = assignment {
                            match bindings.iter().find(|(k, _, _)| k == key) {
                                Some((_, old_attempt, old_phase))
                                    if *old_attempt != attempt || *old_phase != phase =>
                                {
                                    *excluded_total = excluded_total.checked_add(1).ok_or(
                                        CostProfileError::Limit("source8 exclusion count overflow"),
                                    )?;
                                    if let Some(v) = exclusions
                                        .iter_mut()
                                        .find(|v| v.owner_attempt_id == attempt && v.phase == phase)
                                    {
                                        v.excluded_original_offers = v
                                            .excluded_original_offers
                                            .checked_add(1)
                                            .ok_or(CostProfileError::Limit(
                                                "source8 exclusion count overflow",
                                            ))?;
                                    } else {
                                        if exclusions.len() >= cap * 3 {
                                            return Err(CostProfileError::Limit(
                                                "source8 exclusion audit capacity",
                                            ));
                                        }
                                        exclusions.push(StructuredCohortPhaseExclusionAuditV8 {
                                            owner_attempt_id: attempt,
                                            phase,
                                            excluded_original_offers: 1,
                                        });
                                    }
                                    return Ok(false);
                                }

                                Some(_) => {}
                                None => {
                                    if bindings.len() >= cap {
                                        return Err(CostProfileError::Limit(
                                            "source8 cohort owner capacity",
                                        ));
                                    }
                                    bindings.push((key.clone(), attempt, phase));
                                }
                            }
                        }
                        Ok(true)
                    })?;
                self.population.append(record)?;
            }
            R::Population(StructuredServiceRecordV7::OutsideDeclaredRoute { wave }) => {
                if wave.ticket > self.header.declaration.maximum_offered_waves as u64 {
                    return Err(CostProfileError::Limit("source8 original offer capacity"));
                }
                // Original selection/submission/host proof is checked before
                // these non-numerical views advance the cohort. No recipe is
                // synthesized and no outside wave becomes a fitted sample.
                self.population.outside(wave)?;
                let rows = wave.evidence.original_cohort_rows()?;
                self.lifecycle
                    .outside(&rows, wave.evidence.original_host_stages(), wave.fifo)?;
                self.population.append(record)?;
            }
            R::Population(r) => {
                if matches!(r, StructuredServiceRecordV7::NotSubmitted { .. }) {
                    self.lifecycle.cohort()?;
                    if !self.lifecycle.idle()
                        || self.offered() >= self.header.declaration.maximum_offered_waves as u64
                    {
                        return Err(invalid(
                            "source8 no-submission crossed preparation/capacity",
                        ));
                    }
                }
                if matches!(
                    r,
                    StructuredServiceRecordV7::BlockOpen { .. }
                        | StructuredServiceRecordV7::BlockClose { .. }
                ) && !self.lifecycle.idle()
                {
                    return Err(invalid("source8 block crossed unfinished preparation"));
                }
                if matches!(r, StructuredServiceRecordV7::Checkpoint { .. }) {
                    self.lifecycle.complete()?;
                }
                self.population.push(r)?;
                if let StructuredServiceRecordV7::NotSubmitted { attempt } = r {
                    self.lifecycle.no_submission(
                        None,
                        attempt.ticket,
                        attempt.call_id(),
                        attempt.finalized_at_ns(),
                        attempt.original_participants(),
                    )?;
                }
            }
        }
        Ok(())
    }
}

pub fn replay_structured_source_v8(
    bytes: &[u8],
    limits: &CostProfileLoadLimits,
) -> Result<StructuredPreparedOwnerBlockCheckpointV8, CostProfileError> {
    limits.validate()?;
    if bytes.is_empty() || bytes.len() > limits.max_file_bytes.get() || bytes.last() != Some(&b'\n')
    {
        return Err(invalid("source8 incomplete/oversized checkpoint prefix"));
    }
    let mut lines = bytes.split_inclusive(|b| *b == b'\n');
    let first = lines
        .next()
        .ok_or_else(|| invalid("source8 missing header"))?;
    if first.len() > 8 * 1024 * 1024 {
        return Err(CostProfileError::Limit("source8 header line limit"));
    }
    let header: StructuredPreparedOwnerBlockHeaderV8 = serde_json::from_slice(first)?;
    if record_bytes_v7(&header)? != first {
        return Err(invalid("source8 noncanonical header"));
    }
    let mut collector = StructuredPreparedOwnerBlockCollectorV8::new(header, limits.clone())?;
    let mut checkpoint = None;
    for line in lines {
        if line.len() > 8 * 1024 * 1024 {
            return Err(CostProfileError::Limit("source8 record line limit"));
        }
        let record: StructuredPreparedOwnerBlockRecordV8 = serde_json::from_slice(line)?;
        if record_bytes_v7(&record)? != line {
            return Err(invalid("source8 noncanonical record"));
        }
        collector.push(&record)?;
        checkpoint = match record {
            StructuredPreparedOwnerBlockRecordV8::Population(
                StructuredServiceRecordV7::Checkpoint { closing, .. },
            ) => Some(closing),
            _ => None,
        };
    }
    collector.lifecycle.complete()?;
    Ok(StructuredPreparedOwnerBlockCheckpointV8 {
        header: collector.header,
        population: StructuredServiceCheckpointV7::from_collector(
            &collector.population,
            checkpoint.ok_or_else(|| invalid("source8 prefix lacks final complete checkpoint"))?,
        )?,
    })
}
