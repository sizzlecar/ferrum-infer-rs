use super::super::session::paired;
use super::*;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::NonNegativePlanningEstimatorV1 as PlanningEstimator;
use ferrum_scheduler::implementations::continuous::cost_profile::StructuredServiceClockV6;
use ferrum_scheduler::implementations::continuous::{
    cost_model::structured_v2::{
        NonNegativeEnvelopeContractV1, OwnerPhaseSupportPolicyV1, StructuredCostTemplatePolicyV1,
        StructuredPopulationPolicyV1, StructuredServiceDomainPolicyV1,
        WorkAxisAndBranchChallengesV1,
    },
    cost_profile::{
        self as file, StructuredServiceCollectorV7, StructuredServiceDeclarationV7,
        StructuredServiceHeaderV7, StructuredServiceNoSubmissionV7,
        StructuredServiceOutsideRouteV7, StructuredServiceRecordV7, StructuredServiceWaveV7,
    },
};
use ferrum_types::SloAutomaticCalibrationNumericalStrategyV1 as NumericalStrategy;

mod retirement;
pub(super) use retirement::{PreparedRetirement, RetirementReport};

/// Frozen only after the original ticket population and FIFO cut are complete.
pub(super) struct PreparedClose {
    pub closing: StructuredServiceClockV6,
}

pub(in super::super) struct BlockSession {
    collector: StructuredServiceCollectorV7,
    #[cfg(test)]
    original_header: StructuredServiceHeaderV7,
    generation: u64,
    capture_identity: [u8; 32],
    protocol: [u8; 32],
    deadline_ns: u64,
    successor_ready: bool,
    import_attempts: u64,
    readiness_replay_bound_per_transition: u64,
    block: usize,
    block_offset: u64,
    block_offered: usize,
    ingested_tickets: usize,
    maximum_encoded_source_bytes: std::num::NonZeroU64,
    published_children: usize,
    published_domains: Vec<[u8; 32]>,
    diagnostic_written_ticket: u64,
    source: Option<source::Source>,
    reuse_journal: Option<
        crate::continuous_engine::inner::cost_observation::automatic_reuse::OriginalSourceJournal,
    >,
    pub(in super::super) next_pending: bool,
    pub(in super::super) declaration_sha256: [u8; 32],
}

impl BlockSession {
    pub(super) fn open(
        live: &LiveCalibration,
        controller: &Controller,
        generation: u64,
        opening: ExportClockReading,
        cutoff: u64,
        clock: &dyn CostObservationClock,
        seed: Option<&ferrum_scheduler::implementations::continuous::cost_model::structured_v2::DeclaredAlgorithmUniverseV1>,
        collector_capacity: Option<usize>,
    ) -> Result<Self, FerrumError> {
        use sha2::Digest;
        let mut owner_schedule = schedule(&controller.settings, &live.declaration.settings)?;
        if live.workload_domain.is_some() {
            owner_schedule.phase_support = Some(if controller.uses_rolling_owner_blocks() {
                OwnerPhaseSupportPolicyV1::FrozenInputIntersectionV2
            } else {
                OwnerPhaseSupportPolicyV1::FrozenInputIntersectionV1
            });
            owner_schedule.algorithm_universe = Some(
                ferrum_scheduler::implementations::continuous::cost_model::structured_v2::OwnerAlgorithmUniversePolicyV1::FirstOrdinaryDiscoveryBlockSubsetV1,
            );
        }
        let seed_bytes = seed
            .map_or(Some(0), |u| u.retained_payload_bytes())
            .ok_or_else(|| error("algorithm seed retained size overflow"))?;
        if seed.is_some() {
            owner_schedule.algorithm_universe = Some(
                ferrum_scheduler::implementations::continuous::cost_model::structured_v2::OwnerAlgorithmUniversePolicyV1::SeededFirstOrdinaryDiscoveryBlockSubsetV1);
        }
        let readiness_replay_bound_per_transition = owner_schedule
            .input_readiness
            .as_ref()
            .map_or(0, |policy| policy.maximum_geometry_visits / 2);
        let declaration = StructuredServiceDeclarationV7 {
            schedule: owner_schedule,
            route_population: live.route_population(),
            domain_policy: if live.workload_domain.is_some() {
                StructuredServiceDomainPolicyV1::NonNegativePhysicalEnvelopeV1
            } else {
                StructuredServiceDomainPolicyV1::FrozenFitSupportV1
            },
            nonnegative_envelope: live.workload_domain.as_ref().map(|domain| {
                NonNegativeEnvelopeContractV1 {
                    algorithm_universe: seed.cloned(),
                    population_policy: StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1,
                    planning_estimator: match controller.settings.numerical_strategy {
                        NumericalStrategy::IdentifiedEnvelopeV2 => {
                            PlanningEstimator::IdentifiedEnvelopeV2
                        }
                        NumericalStrategy::SameSourceJointCellsV1 => {
                            PlanningEstimator::IdentifiedFitJointCellsV1
                        }
                        NumericalStrategy::IdentifiedFitGlobalResidualV1 => {
                            PlanningEstimator::IdentifiedFitGlobalResidualV1
                        }
                    },
                    template_policy: StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1,
                    workload_domain: domain.clone(),
                    settings: Default::default(),
                    challenge: WorkAxisAndBranchChallengesV1::WorkAxesAndPlainTextBranchesV1,
                }
            }),
            settings: live.declaration.settings.clone(),
            maximum_window_ns: live.declaration.maximum_window_ns,
            maximum_owners: controller.settings.maximum_owners.get(),
            maximum_discovery_bytes: controller
                .settings
                .maximum_discovery_bytes
                .get()
                .checked_sub(seed_bytes)
                .filter(|n| *n > 0)
                .ok_or_else(|| error("algorithm seed exhausted discovery capacity"))?,
            maximum_retained_numeric_bytes: collector_capacity
                .unwrap_or(live.maximum_retained_bytes)
                .checked_sub(seed_bytes)
                .filter(|n| *n > 0)
                .ok_or_else(|| error("algorithm seed exhausted numeric capacity"))?,
        };
        let block_offered = declaration.schedule.block_offered;
        let deadline_ns = opening
            .monotonic_ns
            .checked_add(live.declaration.maximum_window_ns)
            .ok_or_else(|| error("source collection deadline overflow"))?;
        let original_numeric = declaration.maximum_retained_numeric_bytes;
        let capture = sha2::Sha256::digest(uuid::Uuid::new_v4().as_bytes()).into();
        let producer = serde_json::to_value(live.producer.current()?).map_err(error)?;
        let build_header = |numeric| {
            let mut declared = declaration.clone();
            declared.maximum_retained_numeric_bytes = numeric;
            let header = StructuredServiceHeaderV7::new(
                capture,
                generation,
                file::ProfileFingerprint::from(&live.fingerprint),
                producer.clone(),
                paired(opening),
                declared,
                controller.settings.maximum_encoded_source_bytes.get(),
            )
            .map_err(error)?;
            match clock.monotonic_domain() {
                Some(domain) => header.with_monotonic_domain(domain.clone()).map_err(error),
                None => Ok(header),
            }
        };
        // This reservation shares the original per-source/rolling slot quota.
        // The signed declaration is emitted only after cache admission resolves.
        // An unavailable sink restores the original numeric allowance.
        let cache_numeric = original_numeric.checked_sub(
            crate::continuous_engine::inner::cost_observation::automatic_reuse::OriginalSourceJournal::writer_retained_bytes()
        ).filter(|n| *n > 0);
        let cached = live.reuse.get().and_then(|reuse| {
            let header = build_header(cache_numeric?).ok()?;
            let canonical_header = file::canonical_value_v7(&header).ok()?;
            match reuse.open_source(
                crate::continuous_engine::inner::cost_observation::automatic_reuse::SourceKind::OwnerBlocksV7,
                header.capture_identity, header.protocol,
            ) {
                Ok(journal) => Some((header, canonical_header, journal)),
                Err(reason) => {
                    tracing::info!(?reason, "Automatic source7 restart journal unavailable");
                    None
                }
            }
        });
        let (header, canonical_header, reuse_journal) = match cached {
            Some((header, canonical, journal)) => (header, canonical, Some(journal)),
            None => {
                let header = build_header(original_numeric)?;
                let canonical = file::canonical_value_v7(&header).map_err(error)?;
                (header, canonical, None)
            }
        };
        let declaration_sha256 = header.declaration_sha256;
        let capture_identity = header.capture_identity;
        let protocol = header.protocol;
        let source = controller.diagnostics.as_ref().and_then(|store| {
            match source::Source::open_header(
                live,
                generation,
                store,
                header.maximum_file_bytes,
                &canonical_header,
            ) {
                Ok(source) => Some(source),
                Err(reason) => {
                    live.record_automatic_diagnostic_failure(
                        generation,
                        source::DiagnosticFailureStage::Open,
                        &reason,
                    );
                    None
                }
            }
        });
        #[cfg(test)]
        let original_header = header.clone();
        let collector = match &reuse_journal {
            Some(journal) => StructuredServiceCollectorV7::new_streaming_with_record_sink(
                header,
                profile::load_limits(&controller.import),
                controller.settings.maximum_encoded_source_bytes,
                Box::new(journal.clone()),
            ),
            None => StructuredServiceCollectorV7::new_streaming(
                header,
                profile::load_limits(&controller.import),
                controller.settings.maximum_encoded_source_bytes,
            ),
        };
        let collector = match collector {
            Ok(collector) => collector,
            Err(reason) => {
                if let Some(journal) = &reuse_journal {
                    journal.abandon();
                }
                return Err(error(reason));
            }
        };
        let mut out = Self {
            collector,
            #[cfg(test)]
            original_header,
            generation,
            capture_identity,
            protocol,
            deadline_ns,
            successor_ready: false,
            import_attempts: 0,
            readiness_replay_bound_per_transition,
            block: 0,
            block_offset: 0,
            block_offered,
            ingested_tickets: 0,
            maximum_encoded_source_bytes: controller.settings.maximum_encoded_source_bytes,
            published_children: 0,
            published_domains: if controller.uses_rolling_owner_blocks() {
                Vec::with_capacity(controller.settings.maximum_owners.get())
            } else {
                Vec::new()
            },
            diagnostic_written_ticket: 0,
            source,
            reuse_journal,
            next_pending: true,
            declaration_sha256,
        };
        if let Err(reason) = out.open_next(live, opening.monotonic_ns, cutoff) {
            if let Some(journal) = &out.reuse_journal {
                journal.abandon();
            }
            return Err(reason);
        }
        Ok(out)
    }

    pub(in super::super) fn validate_ticket(&self, ticket: &Ticket) -> Result<(), &'static str> {
        if let Some(enrollment) = ticket.window.enrollment(self.generation) {
            return if !self.next_pending && *enrollment == self.enrollment() {
                Ok(())
            } else {
                Err("automatic_original_enrollment_mismatch")
            };
        }
        if ticket.window.is_rolling() {
            return Err("automatic_source_not_enrolled");
        }
        if self.next_pending
            || ticket.window.generation != self.generation
            || ticket.phase() != self.block
        {
            return Err("automatic_original_block_mismatch");
        }
        // No discovery or numerical state mutates here. The original raw
        // acceptance can still fail its resource/frontier check afterwards.
        Ok(())
    }

    pub(super) fn enrollment(&self) -> tickets::SourceEnrollment {
        tickets::SourceEnrollment {
            generation: self.generation,
            capture_identity: self.capture_identity,
            protocol: self.protocol,
            block: self.block,
            offered_offset: self.block_offset,
            deadline_ns: self.deadline_ns,
        }
    }

    fn enrolled_in(&self, window: &tickets::Window) -> bool {
        if window.is_rolling() {
            window
                .enrollment(self.generation)
                .is_some_and(|enrollment| *enrollment == self.enrollment())
        } else {
            window.generation == self.generation && window.phase == self.block
        }
    }

    pub(super) fn work(&self) -> budget::Work {
        let audit = self.collector.audit();
        budget::Work {
            canonical_source_bytes: audit.source_bytes,
            readiness_scalar_visits: audit.readiness_scalar_visits,
            readiness_replay_upper_bound: audit
                .phase_transition_attempts
                .checked_mul(self.readiness_replay_bound_per_transition)
                .expect("bounded phase attempts and declared geometry allowance"),
            phase_transition_attempts: audit.phase_transition_attempts,
            imports: self.import_attempts,
        }
    }

    pub(super) fn take_successor_seed(&mut self) -> Option<Option<ferrum_scheduler::implementations::continuous::cost_model::structured_v2::DeclaredAlgorithmUniverseV1>>{
        if !self.successor_ready {
            return None;
        }
        self.successor_ready = false;
        Some(self.collector.frozen_algorithm_universe().cloned())
    }

    #[cfg(test)]
    pub(in crate::continuous_engine::inner::cost_observation) fn header_for_test(
        &self,
    ) -> StructuredServiceHeaderV7 {
        self.original_header.clone()
    }

    pub(in super::super) fn audit(&self) -> StructuredServiceAuditV7 {
        self.collector.audit()
    }

    pub(super) fn open_next(
        &mut self,
        live: &LiveCalibration,
        at: u64,
        cutoff: u64,
    ) -> Result<usize, FerrumError> {
        if !self.next_pending {
            return Err(error("previous original block is still open"));
        }
        let block = self
            .block
            .checked_add(1)
            .ok_or_else(|| error("block counter exhausted"))?;
        let record = self.collector.open_block(at, cutoff).map_err(error)?;
        self.block_offset = self.collector.offered();
        self.block = block;
        self.ingested_tickets = 0;
        self.next_pending = false;
        self.write_diagnostic(live, &record);
        Ok(block)
    }

    /// At most one accepted original FIFO record per worker turn. Numerical
    /// phase transitions remain exclusive to a complete BlockClose below.
    pub(super) fn ingest_one(
        &mut self,
        live: &LiveCalibration,
        cutoff: u64,
    ) -> Result<(), FerrumError> {
        if self.next_pending {
            return Ok(());
        }
        let collected = live.collected.lock();
        if collected.failure.is_some() {
            return Err(error("original block raw collection failed"));
        }
        if self.ingested_tickets == collected.waves.len() {
            return Ok(());
        }
        let next = self
            .ingested_tickets
            .checked_add(1)
            .filter(|next| *next <= self.block_offered)
            .ok_or_else(|| error("original block ticket capacity"))?;
        let Some(wave) = collected
            .waves
            .iter()
            .find(|wave| wave.ticket == next as u64)
        else {
            // A later call can settle before an earlier original ticket. Keep
            // the bounded raw block until that ticket settles or fails; never
            // skip it or close issuance merely because it is still in flight.
            return if collected.waves.len() == self.block_offered {
                Err(error("original block ticket sequence"))
            } else {
                Ok(())
            };
        };
        if wave.fifo > cutoff {
            return Err(error("original block FIFO cutoff"));
        }
        let record = completed_record(wave, self.block_offset)?;
        self.write_diagnostic(live, &record);
        // push may count the original offer before a later capacity failure.
        // Preserve diagnostic exact-once position independently of that count.
        self.diagnostic_written_ticket = self
            .block_offset
            .checked_add(wave.ticket)
            .ok_or_else(|| error("original ticket overflow"))?;
        self.collector.push(&record).map_err(error)?;
        self.ingested_tickets = next;
        Ok(())
    }

    pub(super) fn has_pending_records(&self, live: &LiveCalibration) -> bool {
        !self.next_pending
            && live
                .collected
                .lock()
                .waves
                .iter()
                .any(|wave| wave.ticket == self.ingested_tickets as u64 + 1)
    }

    pub(super) fn records_complete(&self) -> bool {
        self.ingested_tickets == self.block_offered
    }

    fn write_diagnostic(&mut self, live: &LiveCalibration, record: &impl Serialize) {
        if let Some(source) = &mut self.source {
            let written = file::canonical_value_v7(record)
                .map_err(error)
                .and_then(|value| source.write(&value));
            if let Err(reason) = written {
                source.note_failure(live, self.generation, &reason.to_string());
                live.record_automatic_diagnostic_failure(
                    self.generation,
                    source::DiagnosticFailureStage::Write,
                    &reason,
                );
                self.source = None;
            }
        }
    }

    pub(super) fn complete(
        &mut self,
        live: &LiveCalibration,
        window: &Arc<tickets::Window>,
        clock: &dyn CostObservationClock,
        cutoff: u64,
        import: &ferrum_types::SloCostProfileImportConfig,
    ) -> Result<Option<publication::Publication>, FerrumError> {
        let prepared = self.prepare_close(live, window, clock, cutoff)?;
        let close = self.compute_close(&prepared)?;
        self.finish_close(live, window, clock, import, prepared, close)
    }

    pub(super) fn prepare_close(
        &self,
        live: &LiveCalibration,
        window: &Arc<tickets::Window>,
        clock: &dyn CostObservationClock,
        cutoff: u64,
    ) -> Result<PreparedClose, FerrumError> {
        let now = clock
            .now_ns()
            .ok_or_else(|| error("block close clock unavailable"))?;
        let assigned = window.enrollment(self.generation).map_or(
            !window.is_rolling()
                && window.generation == self.generation
                && window.phase == self.block,
            |enrollment| *enrollment == self.enrollment(),
        );
        if !assigned || !window.complete(now) || self.next_pending {
            return Err(error("original block is incomplete or differs"));
        }
        let collected = live.collected.lock();
        if collected.failure.is_some()
            || collected.waves.len() != self.block_offered
            || !self.records_complete()
            || collected.waves.iter().any(|wave| wave.fifo > cutoff)
        {
            return Err(error("original block collection is incomplete"));
        }
        Ok(PreparedClose {
            closing: paired(ExportClockReading::closing(clock).map_err(error)?),
        })
    }

    /// The sole expensive operation: owns no runtime, FIFO, or feedback lock.
    pub(super) fn compute_close(
        &mut self,
        prepared: &PreparedClose,
    ) -> Result<StructuredServiceRecordV7, FerrumError> {
        self.collector.close_block(prepared.closing).map_err(error)
    }

    pub(super) fn abandon(&mut self) {
        if let Some(journal) = &self.reuse_journal {
            journal.abandon();
        }
    }

    pub(super) fn finish_close(
        &mut self,
        live: &LiveCalibration,
        window: &Arc<tickets::Window>,
        clock: &dyn CostObservationClock,
        import: &ferrum_types::SloCostProfileImportConfig,
        prepared: PreparedClose,
        close: StructuredServiceRecordV7,
    ) -> Result<Option<publication::Publication>, FerrumError> {
        let closing = prepared.closing;
        // Declared exactly at the first original Discovery close. This never
        // consults timing error, qualification, or a subsequently seen shape.
        if let StructuredServiceRecordV7::BlockClose { discoveries, .. } = &close {
            if !discoveries.is_empty() && self.collector.audit().owners.len() == discoveries.len() {
                self.successor_ready = true;
            }
        }
        self.write_diagnostic(live, &close);
        if tracing::enabled!(target: "ferrum_engine::continuous_engine::inner::cost_observation::runtime", tracing::Level::DEBUG)
        {
            tracing::debug!(
                target: "ferrum_engine::continuous_engine::inner::cost_observation::runtime",
                generation = self.generation, block = self.block,
                offered = self.collector.offered(), qualified = self.collector.qualified_children(),
                "Automatic owner block closed"
            );
        }
        self.next_pending = true;
        let count = self.collector.qualified_children();
        if count == self.published_children {
            if !window.is_rolling() {
                *live.collected.lock() = Collected::default();
            }
            return Ok(None);
        }
        let (record, checkpoint) = self.collector.checkpoint(closing).map_err(error)?;
        if let Some(journal) = &self.reuse_journal {
            if let Err(reason) = journal.checkpoint(checkpoint.source_receipt()) {
                tracing::info!(?reason, "Automatic source7 restart checkpoint unavailable");
            }
        }
        self.write_diagnostic(live, &record);
        let installed = paired(ExportClockReading::closing(clock).map_err(error)?);
        let limits = profile::load_limits(import);
        self.import_attempts = self
            .import_attempts
            .checked_add(1)
            .ok_or_else(|| error("source import attempt counter overflow"))?;
        let mut imported = match clock.monotonic_domain() {
            Some(domain) => checkpoint.activate_same_boot_memory_streaming(
                installed,
                domain,
                &limits,
                self.maximum_encoded_source_bytes,
            ),
            None => checkpoint.activate_same_process_memory_streaming(
                installed,
                &limits,
                self.maximum_encoded_source_bytes,
            ),
        }
        .map_err(error)?;
        // A later family must not rebind an unchanged earlier predictor to a
        // newer prefix. Preserve its original provenance and drift correction
        // until that family has independently qualified a fresh model.
        imported
            .children
            .retain(|child| !self.published_domains.contains(child.domain_signature()));
        let new_domains: Vec<_> = imported
            .children
            .iter()
            .map(|child| *child.domain_signature())
            .collect();
        let (children, receipt) =
            profile::EngineCostSnapshot::live_owner_block_catalog_with_domain(
                None,
                imported,
                live.workload_domain(),
            )?;
        self.published_children = count;
        self.published_domains.extend(new_domains);
        // Keep this complete original population until installation returns.
        // The next BlockOpen clears it after freezing the new assignment.
        Ok(Some(publication::Publication { children, receipt }))
    }

    pub(super) fn finish(
        &mut self,
        live: &LiveCalibration,
        clock: &dyn CostObservationClock,
    ) -> Result<Option<super::super::super::super::profile_export::PublishedFile>, FerrumError>
    {
        let record = self
            .collector
            .stop(paired(ExportClockReading::closing(clock).map_err(error)?))
            .map_err(error)?;
        self.write_diagnostic(live, &record);
        if let Some(journal) = &self.reuse_journal {
            journal.finish(self.collector.source_receipt());
        }
        let Some(mut source) = self.source.take() else {
            return Ok(None);
        };
        source.footer();
        match source.publish() {
            Ok(receipt) => {
                if (receipt.bytes, receipt.digest) != self.collector.source_receipt() {
                    live.record_automatic_diagnostic_failure(
                        self.generation,
                        source::DiagnosticFailureStage::ReceiptMismatch,
                        &"owner block journal receipt differs",
                    );
                    return Ok(None);
                }
                Ok(Some(receipt))
            }
            Err(reason) => {
                source.note_failure(live, self.generation, &reason.to_string());
                live.record_automatic_diagnostic_failure(
                    self.generation,
                    source::DiagnosticFailureStage::Publish,
                    &reason,
                );
                Ok(None)
            }
        }
    }

    pub(super) fn fail(
        &mut self,
        live: &LiveCalibration,
        window: &tickets::Window,
        clock: &dyn CostObservationClock,
        cutoff: u64,
        reason: &str,
    ) {
        let result = (|| {
            // Preserve any original successful raw records preceding failure.
            // No block close or numerical transition is attempted.
            let enrolled = self.enrolled_in(window);
            if enrolled {
                let mut collected = live.collected.lock();
                collected.waves.sort_by_key(|wave| wave.ticket);
                let mut canonical = !self.collector.audit().poisoned;
                for wave in &collected.waves {
                    let ticket = self
                        .block_offset
                        .checked_add(wave.ticket)
                        .ok_or_else(|| error("failed ticket overflow"))?;
                    if ticket > self.diagnostic_written_ticket {
                        let record = completed_record(wave, self.block_offset)?;
                        // The first malformed or non-contiguous record poisons the
                        // source; retain it diagnostically without declaring it accepted.
                        self.write_diagnostic(live, &record);
                        self.diagnostic_written_ticket = ticket;
                        if canonical
                            && ticket > self.collector.offered()
                            && self.collector.push(&record).is_err()
                        {
                            canonical = false;
                        }
                    }
                }
            }
            // A source which opened after this Window cannot backfill its
            // raw population during failed admission. Seal the genuinely
            // empty/partial source at its own offered frontier instead.
            let ticket = enrolled
                .then(|| window.failed_tickets().next())
                .flatten()
                .map(|t| self.block_offset + t.ticket)
                .unwrap_or_else(|| self.collector.offered().saturating_add(1));
            let failed = self
                .collector
                .fail(
                    ticket,
                    cutoff,
                    clock.now_ns().unwrap_or(0),
                    reason.chars().take(4096).collect::<String>(),
                )
                .map_err(error)?;
            self.write_diagnostic(live, &failed);
            let footer = self
                .collector
                .stop(paired(ExportClockReading::closing(clock).map_err(error)?))
                .map_err(error)?;
            self.write_diagnostic(live, &footer);
            if let Some(journal) = &self.reuse_journal {
                journal.finish(self.collector.source_receipt());
            }
            // After an original gap, diagnostic bytes intentionally include
            // later settled records that the canonical collector rejected.
            // Preserve the actual failed archive receipt; it cannot be used
            // as a successful checkpoint or imported training population.
            let receipt = match self.source.take() {
                Some(mut source) => {
                    source.footer();
                    match source.publish() {
                        Ok(receipt) => Some(receipt),
                        Err(write) => {
                            source.note_failure(live, self.generation, reason);
                            return Err(write);
                        }
                    }
                }
                None => None,
            };
            live.remember_failed_source(self.generation, receipt, true, false, reason);
            Ok::<(), FerrumError>(())
        })();
        // Even a canonical capacity failure must close its optional writer.
        // Use the genuine accepted prefix receipt; do not manufacture a
        // footer or turn an unqualified capture into a reusable checkpoint.
        if let Some(journal) = &self.reuse_journal {
            journal.finish(self.collector.source_receipt());
        }
        if let Err(write) = result {
            if let Some(source) = &self.source {
                source.note_failure(live, self.generation, reason);
            }
            live.record_automatic_diagnostic_failure(
                self.generation,
                source::DiagnosticFailureStage::FailedPopulation,
                &write,
            );
        }
    }
}

fn completed_record(wave: &Wave, offset: u64) -> Result<StructuredServiceRecordV7, FerrumError> {
    let ticket = offset
        .checked_add(wave.ticket)
        .ok_or_else(|| error("original ticket overflow"))?;
    let stages = match &wave.evidence {
        WaveEvidence::NoSubmission(receipt) => {
            return Ok(StructuredServiceRecordV7::NotSubmitted {
                attempt: StructuredServiceNoSubmissionV7::from_original_record(
                    &receipt.wire,
                    ticket,
                )
                .map_err(error)?,
            });
        }
        WaveEvidence::Settled(stages) => stages,
    };
    let issued = stages
        .prepare_started_at_ns
        .ok_or_else(|| error("original ticket clock missing"))?;
    if let Some(route) = stages.route_evidence.as_ref().filter(|r| r.is_outside()) {
        return Ok(StructuredServiceRecordV7::OutsideDeclaredRoute {
            wave: StructuredServiceOutsideRouteV7::from_diagnostic(
                ticket,
                issued,
                wave.fifo,
                serde_json::to_value(route.outside_diagnostic(stages)).map_err(error)?,
            )
            .map_err(error)?,
        });
    }
    let independent = stages
        .statistical_evidence
        .as_ref()
        .and_then(|v| v.independent_attention_v2())
        .map(|v| v.to_wire_v2());
    let wire = StructuredServiceWaveV7::from_diagnostic(
        ticket,
        issued,
        wave.fifo,
        serde_json::to_value(stages.structured_source_view()).map_err(error)?,
        independent,
    )
    .map_err(error)?;
    let wire = if let Some(route) = &stages.route_evidence {
        wire.with_prepared_route(serde_json::to_value(route.eligible_diagnostic()).map_err(error)?)
            .map_err(error)?
    } else {
        wire
    };
    Ok(StructuredServiceRecordV7::Completed { wave: wire })
}
