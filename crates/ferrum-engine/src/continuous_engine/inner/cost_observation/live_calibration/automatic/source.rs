//! Optional bounded raw stream. Failed records never reenter the collector.
use super::super::super::profile_export::{PublishedFile, StagedFile};
use super::session::{paired, phase};
use super::*;
use ferrum_scheduler::implementations::continuous::cost_profile::{
    StructuredServiceHeaderV6, StructuredServiceRecordV6,
};

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub(in crate::continuous_engine::inner::cost_observation) enum DiagnosticFailureStage {
    Reference,
    Open,
    Write,
    Publish,
    ReceiptMismatch,
    FailedPopulation,
}

#[derive(Debug, Clone, Serialize)]
pub(in crate::continuous_engine::inner::cost_observation) struct DiagnosticFailure {
    /// Zero denotes startup reference persistence before the first generation.
    pub generation: u64,
    pub stage: DiagnosticFailureStage,
    pub reason: String,
}

/// At most two bounded reasons, regardless of the number of generations.
/// These fields never participate in collector qualification or ticket state.
#[derive(Debug, Clone, Default, Serialize)]
pub(in crate::continuous_engine::inner::cost_observation) struct DiagnosticFailureAudit {
    pub count: u64,
    pub first: Option<DiagnosticFailure>,
    pub last: Option<DiagnosticFailure>,
}

impl LiveCalibration {
    pub(super) fn record_automatic_diagnostic_failure(
        &self,
        generation: u64,
        stage: DiagnosticFailureStage,
        reason: &dyn std::fmt::Display,
    ) {
        let Some(controller) = &self.automatic else {
            return;
        };
        let failure = DiagnosticFailure {
            generation,
            stage,
            reason: reason.to_string().chars().take(1024).collect(),
        };
        let mut audit = controller.diagnostic_failures.lock();
        audit.count = audit.count.saturating_add(1);
        audit.first.get_or_insert_with(|| failure.clone());
        audit.last = Some(failure);
    }
}

pub(super) struct Source {
    _reservation: diagnostics::Reservation,
    writer: Option<StagedFile>,
    published: Option<PublishedFile>,
    written_ticket: u64,
    failed: bool,
    footer: bool,
}
impl Source {
    pub fn open(
        live: &LiveCalibration,
        generation: u64,
        store: &diagnostics::Store,
        header: &StructuredServiceHeaderV6,
    ) -> Result<Self, FerrumError> {
        Self::open_header(live, generation, store, header.maximum_file_bytes, header)
    }

    pub(super) fn open_header(
        live: &LiveCalibration,
        generation: u64,
        store: &diagnostics::Store,
        maximum_file_bytes: u64,
        header: &impl Serialize,
    ) -> Result<Self, FerrumError> {
        // The optional file has its own quota. Never lower the canonical
        // collector's source bound merely because an archive was requested.
        let maximum_bytes = store.source_limit().min(maximum_file_bytes);
        let reservation = store.reserve(diagnostics::Kind::Service, maximum_bytes)?;
        let mut writer =
            StagedFile::create(&reservation.directory.join("source.jsonl"), maximum_bytes)
                .map_err(error)?;
        writer.preserve_unpublished();
        let mut source = Self {
            _reservation: reservation,
            writer: Some(writer),
            published: None,
            written_ticket: 0,
            failed: false,
            footer: false,
        };
        if let Err(reason) = source.write(header) {
            source.note_failure(live, generation, &reason.to_string());
            return Err(reason);
        }
        Ok(source)
    }
    pub fn write(&mut self, record: &impl Serialize) -> Result<(), FerrumError> {
        if self.failed {
            return Err(error("diagnostic source has a partial write"));
        }
        let result = self
            .writer
            .as_mut()
            .ok_or_else(|| error("diagnostic source is sealed"))?
            .json_line(record)
            .map_err(error);
        self.failed |= result.is_err();
        result
    }
    pub fn completed(&mut self, ticket: u64) {
        self.written_ticket = ticket;
    }
    pub fn next_phase(&mut self) {
        self.written_ticket = 0;
    }
    pub fn footer(&mut self) {
        self.footer = true;
    }
    pub fn publish(&mut self) -> Result<PublishedFile, FerrumError> {
        let writer = self
            .writer
            .take()
            .ok_or_else(|| error("diagnostic source unavailable"))?;
        self.published = Some(writer.unpublished_receipt());
        let published = writer.publish().map_err(error)?;
        self.published = Some(published.clone());
        Ok(published)
    }
    pub fn note_failure(&self, live: &LiveCalibration, generation: u64, reason: &str) {
        let (receipt, footer, temporary) = self.failure_receipt();
        live.remember_failed_source(generation, receipt, footer, temporary, reason);
    }
    /// Snapshot metadata only; safe to return from an owned archive task.
    pub(super) fn failure_receipt(&self) -> (Option<PublishedFile>, bool, bool) {
        let receipt = self
            .writer
            .as_ref()
            .map(StagedFile::unpublished_receipt)
            .or_else(|| self.published.clone());
        let temporary = receipt
            .as_ref()
            .is_some_and(|value| value.path.extension().is_some_and(|ext| ext == "tmp"));
        (receipt, self.footer, temporary)
    }
    pub fn fail(
        &mut self,
        live: &LiveCalibration,
        declaration: &Declaration,
        window: &tickets::Window,
        clock: &dyn CostObservationClock,
        cutoff: u64,
        reason: &str,
    ) {
        let result = (|| {
            if self.footer || self.writer.is_none() {
                return Ok(());
            }
            let offset = declaration.phase_offered_waves[..window.phase]
                .iter()
                .sum::<usize>() as u64;
            let mut collected = live.collected.lock();
            collected.waves.sort_by_key(|wave| wave.ticket);
            for wave in &collected.waves {
                if wave.ticket > self.written_ticket {
                    self.write(&publication::completed_record(wave, window.phase, offset)?)?;
                    self.written_ticket = wave.ticket;
                }
            }
            drop(collected);
            for failed in window.failed_tickets() {
                self.write(&StructuredServiceRecordV6::TicketFailed {
                    phase: phase(window.phase),
                    ticket: offset
                        .checked_add(failed.ticket)
                        .ok_or_else(|| error("failed ticket overflow"))?,
                    issued_at_ns: failed.issued_at_ns,
                    call_id: failed.call_id,
                    reason: "original private ticket failed or was abandoned".into(),
                })?;
            }
            let closing = ExportClockReading::closing(clock).map_err(error)?;
            self.write(&StructuredServiceRecordV6::Footer {
                offered: offset
                    .checked_add(window.audit().issued as u64)
                    .ok_or_else(|| error("failed offer overflow"))?,
                accepted_fifo_cutoff: cutoff,
                closing: paired(closing),
                failure: Some(reason.chars().take(4096).collect()),
            })?;
            self.footer = true;
            self.publish()?;
            Ok::<(), FerrumError>(())
        })();
        let detail = match result {
            Ok(()) => reason.to_owned(),
            Err(write) => {
                live.record_automatic_diagnostic_failure(
                    window.generation,
                    DiagnosticFailureStage::FailedPopulation,
                    &write,
                );
                format!("{reason}; incomplete failed evidence: {write}")
            }
        };
        self.note_failure(live, window.generation, &detail);
    }
}

pub(super) fn failed_discovery(
    store: &diagnostics::Store,
    live: &LiveCalibration,
    window: &tickets::Window,
    cutoff: u64,
    reason: &str,
) -> Result<(), FerrumError> {
    let reservation = store.reserve(diagnostics::Kind::DiscoveryFailure, store.source_limit())?;
    let mut source = StagedFile::create(
        &reservation.directory.join("source.jsonl"),
        store.source_limit(),
    )
    .map_err(error)?;
    source.preserve_unpublished();
    let result = (|| {
        source.json_line(&serde_json::json!({"protocol":"ferrum.automatic.discovery.failed.v1", "population":window.audit(), "accepted_fifo_cutoff":cutoff})).map_err(error)?;
        let collected = live.collected.lock();
        for wave in &collected.waves {
            match &wave.evidence {
                WaveEvidence::Settled(stages) => source.json_line(&serde_json::json!({"ticket":wave.ticket, "fifo":wave.fifo, "stages":stages.structured_diagnostic_view(), "outside_route":stages.route_evidence.as_ref().filter(|r| r.is_outside()).map(|r| r.outside_diagnostic(stages))})).map_err(error)?,
                WaveEvidence::NoSubmission(receipt) => source.json_line(&serde_json::json!({"ticket":wave.ticket, "fifo":wave.fifo, "no_submission":receipt.wire})).map_err(error)?,
            }
        }
        drop(collected);
        for ticket in window.failed_tickets() {
            source.json_line(&serde_json::json!({"failed_ticket":ticket.ticket,"issued_at_ns":ticket.issued_at_ns,"call_id":ticket.call_id})).map_err(error)?;
        }
        source
            .json_line(
                &serde_json::json!({"failure":reason.chars().take(4096).collect::<String>()}),
            )
            .map_err(error)?;
        Ok::<(), FerrumError>(())
    })();
    let unpublished = source.unpublished_receipt();
    match result {
        Ok(()) => match source.publish() {
            Ok(published) => {
                live.remember_failed_source(
                    window.generation,
                    Some(published),
                    true,
                    false,
                    reason,
                );
                Ok(())
            }
            Err(write) => {
                live.remember_failed_source(
                    window.generation,
                    Some(unpublished),
                    true,
                    true,
                    &format!("{reason}; {write}"),
                );
                Err(error(write))
            }
        },
        Err(write) => {
            live.remember_failed_source(
                window.generation,
                Some(unpublished),
                false,
                true,
                &format!("{reason}; {write}"),
            );
            Err(write)
        }
    }
}
