//! Original failed/finished source sealing with exclusive session custody.
//! No runtime, feedback consumer, or live collection lock crosses the handoff.
use super::*;
use crate::continuous_engine::inner::cost_observation::profile_export::PublishedFile;

pub(in super::super) struct PreparedRetirement {
    waves: Vec<Wave>,
    ticket: u64,
    cutoff: u64,
    closing: StructuredServiceClockV6,
    reason: Option<String>,
}

#[derive(Default)]
pub(in super::super) struct RetirementReport {
    failure: Option<String>,
    published: Option<PublishedFile>,
    diagnostics: source::DiagnosticFailureAudit,
    failed_source: Option<(Option<PublishedFile>, bool, bool, String)>,
}

fn bounded(value: impl std::fmt::Display, maximum: usize) -> String {
    let text = value.to_string();
    let mut result = String::with_capacity(maximum);
    for ch in text.chars() {
        if result.len() + ch.len_utf8() > maximum {
            break;
        }
        result.push(ch);
    }
    result
}

impl PreparedRetirement {
    /// One extra vector of Arc handles, three failure reasons and two diagnostic
    /// reasons. Raw evidence remains in the original transport memory ledger.
    pub(in super::super) fn retained_bytes(block: usize) -> Option<usize> {
        block
            .checked_mul(std::mem::size_of::<Wave>())?
            .checked_add(std::mem::size_of::<Self>())?
            .checked_add(3 * 4096 + 2 * 1024)
    }
}

impl RetirementReport {
    fn diagnostic(
        &mut self,
        generation: u64,
        stage: source::DiagnosticFailureStage,
        reason: impl std::fmt::Display,
    ) {
        let failure = source::DiagnosticFailure {
            generation,
            stage,
            reason: bounded(reason, 1024),
        };
        self.diagnostics.count = self.diagnostics.count.saturating_add(1);
        self.diagnostics
            .first
            .get_or_insert_with(|| failure.clone());
        self.diagnostics.last = Some(failure);
    }

    fn failed(
        &mut self,
        receipt: Option<PublishedFile>,
        footer: bool,
        temporary: bool,
        reason: &str,
    ) {
        if receipt.is_none()
            && self
                .failed_source
                .as_ref()
                .is_some_and(|(file, ..)| file.is_some())
        {
            return;
        }
        self.failed_source = Some((receipt, footer, temporary, bounded(reason, 4096)));
    }

    fn source_failure(&mut self, source: &source::Source, reason: &str) {
        let (receipt, footer, temporary) = source.failure_receipt();
        self.failed(receipt, footer, temporary, reason);
    }

    /// Called once by the original worker after joining the task. This only
    /// moves bounded receipts/audit data; it never encodes or flushes a source.
    pub(in super::super) fn apply(
        self,
        live: &LiveCalibration,
        generation: u64,
    ) -> (Option<String>, Option<PublishedFile>) {
        if let Some(controller) = &live.automatic {
            if self.diagnostics.count != 0 {
                let mut audit = controller.diagnostic_failures.lock();
                audit.count = audit.count.saturating_add(self.diagnostics.count);
                if audit.first.is_none() {
                    audit.first = self.diagnostics.first;
                }
                audit.last = self.diagnostics.last;
            }
        }
        if let Some((receipt, footer, temporary, reason)) = self.failed_source {
            live.remember_failed_source(generation, receipt, footer, temporary, &reason);
        }
        (self.failure, self.published)
    }
}

impl BlockSession {
    pub(in super::super) fn abandon_retirement(&mut self, live: &LiveCalibration, reason: &str) {
        self.abandon();
        if let Some(source) = &self.source {
            source.note_failure(live, self.generation, reason);
        }
        live.record_automatic_diagnostic_failure(
            self.generation,
            source::DiagnosticFailureStage::FailedPopulation,
            &reason,
        );
    }

    pub(in super::super) fn prepare_retirement(
        &self,
        live: &LiveCalibration,
        window: &tickets::Window,
        clock: &dyn CostObservationClock,
        cutoff: u64,
        reason: Option<String>,
    ) -> Result<PreparedRetirement, FerrumError> {
        let population = window.audit();
        if population.issued != population.retired {
            return Err(error(
                "retirement requires the original retired ticket frontier",
            ));
        }
        let enrolled = self.enrolled_in(window);
        let mut waves = Vec::new();
        if reason.is_some() && enrolled {
            let collected = live.collected.lock();
            if collected.waves.len() > self.block_offered
                || collected.waves.iter().any(|wave| wave.fifo > cutoff)
            {
                return Err(error(
                    "retirement original population exceeds its frozen bound",
                ));
            }
            waves
                .try_reserve_exact(collected.waves.len())
                .map_err(error)?;
            if waves.capacity() > self.block_offered {
                return Err(error(
                    "retirement reference capacity exceeds original block",
                ));
            }
            for wave in &collected.waves {
                waves.push(Wave {
                    ticket: wave.ticket,
                    fifo: wave.fifo,
                    evidence: match &wave.evidence {
                        WaveEvidence::Settled(value) => WaveEvidence::Settled(value.clone()),
                        WaveEvidence::NoSubmission(value) => {
                            WaveEvidence::NoSubmission(value.clone())
                        }
                    },
                });
            }
            waves.sort_unstable_by_key(|wave| wave.ticket);
        }
        let ticket = enrolled
            .then(|| window.failed_tickets().next())
            .flatten()
            .map(|ticket| self.block_offset.checked_add(ticket.ticket))
            .unwrap_or_else(|| self.collector.offered().checked_add(1))
            .ok_or_else(|| error("failed ticket overflow"))?;
        // Capture the genuine original boundary, after all retained raw rows
        // settled. Time spent encoding/sealing cannot extend source lifetime.
        let closing = paired(ExportClockReading::closing(clock).map_err(error)?);
        Ok(PreparedRetirement {
            waves,
            ticket,
            cutoff,
            closing,
            reason,
        })
    }

    fn write_retirement_record(&mut self, report: &mut RetirementReport, record: &impl Serialize) {
        if let Some(source) = &mut self.source {
            let written = file::canonical_value_v7(record)
                .map_err(error)
                .and_then(|value| source.write(&value));
            if let Err(reason) = written {
                report.source_failure(source, &reason.to_string());
                report.diagnostic(
                    self.generation,
                    source::DiagnosticFailureStage::Write,
                    reason,
                );
                self.source = None;
            }
        }
    }

    pub(in super::super) fn compute_retirement(
        &mut self,
        prepared: PreparedRetirement,
    ) -> RetirementReport {
        let started = std::time::Instant::now();
        let mut backfilled = 0usize;
        let mut report = RetirementReport::default();
        report.failure = prepared.reason;
        let result = (|| {
            if let Some(reason) = report.failure.clone() {
                let mut canonical = !self.collector.audit().poisoned;
                for wave in &prepared.waves {
                    let ticket = self
                        .block_offset
                        .checked_add(wave.ticket)
                        .ok_or_else(|| error("failed ticket overflow"))?;
                    if ticket > self.diagnostic_written_ticket {
                        let record = completed_record(wave, self.block_offset)?;
                        // Preserve original rejected rows diagnostically. They
                        // cannot become a successful checkpoint or qualification.
                        self.write_retirement_record(&mut report, &record);
                        self.diagnostic_written_ticket = ticket;
                        backfilled += 1;
                        if canonical
                            && ticket > self.collector.offered()
                            && self.collector.push(&record).is_err()
                        {
                            canonical = false;
                        }
                    }
                }
                let failed = self
                    .collector
                    .fail(
                        prepared.ticket,
                        prepared.cutoff,
                        prepared.closing.monotonic_ns,
                        reason,
                    )
                    .map_err(error)?;
                self.write_retirement_record(&mut report, &failed);
            }
            let footer = self.collector.stop(prepared.closing).map_err(error)?;
            self.write_retirement_record(&mut report, &footer);
            if let Some(journal) = &self.reuse_journal {
                journal.finish(self.collector.source_receipt());
            }
            if let Some(mut source) = self.source.take() {
                source.footer();
                match source.publish() {
                    Ok(receipt) => {
                        if let Some(reason) = &report.failure {
                            let reason = reason.clone();
                            report.failed(Some(receipt), true, false, &reason);
                        } else if (receipt.bytes, receipt.digest) == self.collector.source_receipt()
                        {
                            report.published = Some(receipt);
                        } else {
                            report.diagnostic(
                                self.generation,
                                source::DiagnosticFailureStage::ReceiptMismatch,
                                "owner block journal receipt differs",
                            );
                        }
                    }
                    Err(reason) => {
                        report.source_failure(&source, &reason.to_string());
                        report.diagnostic(
                            self.generation,
                            source::DiagnosticFailureStage::Publish,
                            reason,
                        );
                    }
                }
            } else if let Some(reason) = &report.failure {
                let reason = reason.clone();
                report.failed(None, true, false, &reason);
            }
            Ok::<(), FerrumError>(())
        })();
        // A failed canonical append must still close its optional writer with
        // the genuine accepted prefix. Never retry a numeric transition.
        if let Some(journal) = &self.reuse_journal {
            journal.finish(self.collector.source_receipt());
        }
        if let Err(reason) = result {
            let detail = bounded(&reason, 4096);
            if let Some(source) = &self.source {
                report.source_failure(source, &detail);
            }
            report.diagnostic(
                self.generation,
                source::DiagnosticFailureStage::FailedPopulation,
                reason,
            );
            report.failure.get_or_insert(detail);
        }
        tracing::info!(
            generation = self.generation,
            backfilled_records = backfilled,
            elapsed_ms = started.elapsed().as_secs_f64() * 1000.0,
            failed = report.failure.is_some(),
            "Automatic original source retirement completed"
        );
        report
    }
}
