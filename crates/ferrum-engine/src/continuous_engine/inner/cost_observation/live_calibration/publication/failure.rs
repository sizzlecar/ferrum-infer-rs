//! Failed populations are evidence only. Never submit these records to the
//! numerical collector, and never construct a publication from this path.
use super::*;

#[derive(Debug, Clone, serde::Serialize)]
pub(in crate::continuous_engine::inner::cost_observation) struct FailedSourceReceipt {
    pub generation: u64,
    pub path: Option<PathBuf>,
    pub bytes: u64,
    pub sha256: Option<String>,
    pub footer_complete: bool,
    pub retained_unpublished: bool,
    pub reason: String,
}

impl LiveCalibration {
    pub(in crate::continuous_engine::inner::cost_observation::live_calibration) fn remember_failed_source(
        &self,
        generation: u64,
        source: Option<PublishedFile>,
        footer_complete: bool,
        retained_unpublished: bool,
        reason: &str,
    ) {
        let mut last = self.last_failed_source.lock();
        // An early write/publish failure may already have the only on-disk
        // receipt. A subsequent error without a file must not erase it.
        if source.is_none()
            && last
                .as_ref()
                .is_some_and(|v| v.generation == generation && v.path.is_some())
        {
            return;
        }
        *last = Some(FailedSourceReceipt {
            generation,
            path: source.as_ref().map(|v| v.path.clone()),
            bytes: source.as_ref().map_or(0, |v| v.bytes),
            sha256: source.map(|v| v.sha256),
            footer_complete,
            retained_unpublished,
            reason: reason.chars().take(4096).collect(),
        });
    }

    pub(super) fn finish_failed_generation(
        &self,
        state: &mut State,
        window: &Arc<tickets::Window>,
        policy: &Policy,
        clock: &dyn CostObservationClock,
        cutoff: u64,
    ) -> Result<Option<Publication>, FerrumError> {
        window.fail();
        let audit = window.audit();
        // Failure closes the issuance CAS before this count is inspected.
        // A final lost/unsubmitted ticket still has to retire explicitly.
        if audit.issued != audit.retired {
            if state.stopping {
                self.remember_failed_source(
                    state.generation,
                    state
                        .session
                        .as_ref()
                        .and_then(|s| s.source.as_ref())
                        .map(StagedFile::unpublished_receipt),
                    false,
                    true,
                    "shutdown has unretired live tickets; source remains incomplete",
                );
            }
            return Ok(None);
        }
        let reason = state
            .failure
            .clone()
            .unwrap_or_else(|| "offered window failed".into());
        let result = if let Some(source) = state.published_source.take() {
            // A complete source may precede a profile-installation failure.
            // Preserve its immutable bytes; do not append a second footer.
            self.remember_failed_source(state.generation, Some(source), true, false, &reason);
            Ok(())
        } else {
            self.persist_failed_window(state, window, policy, clock, cutoff, &reason)
        };
        state.finished = true;
        state.session = None;
        *self.collected.lock() = Collected::default();
        let _ = self
            .failed_generations
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |n| n.checked_add(1));
        match result {
            Ok(()) => Err(error(reason)),
            Err(write) => Err(error(format!(
                "{reason}; failed evidence incomplete: {write}"
            ))),
        }
    }

    fn persist_failed_window(
        &self,
        state: &mut State,
        window: &Arc<tickets::Window>,
        policy: &Policy,
        clock: &dyn CostObservationClock,
        cutoff: u64,
        reason: &str,
    ) -> Result<(), FerrumError> {
        if state.session.is_none() {
            if state.open_attempted {
                self.remember_failed_source(state.generation, None, false, false, reason);
                return Err(error("original source open/write attempt failed"));
            }
            state.open_attempted = true;
            match Session::open(
                self,
                policy,
                state.generation,
                state.opening.unwrap_or(policy.opening),
                0,
            ) {
                Ok(session) => state.session = Some(session),
                Err(write) => {
                    self.remember_failed_source(
                        state.generation,
                        None,
                        false,
                        false,
                        &format!("{reason}; {write}"),
                    );
                    return Err(write);
                }
            }
        }
        let session = state.session.as_mut().expect("opened failure source");
        // publish() already retained the only file when it failed after taking
        // the writer. Its earlier receipt must remain authoritative.
        if session.source.is_none() {
            return Err(error("source writer unavailable after publication failure"));
        }
        let offset = self.declaration.phase_offered_waves[..window.phase]
            .iter()
            .sum::<usize>() as u64;
        let result = (|| {
            if session.write_failed {
                return Err(error(
                    "source has a partial write; preserving original bytes",
                ));
            }
            let mut collected = self.collected.lock();
            collected.waves.sort_by_key(|wave| wave.ticket);
            for wave in &collected.waves {
                if wave.ticket > session.written_ticket {
                    session.write_raw(&completed_record(wave, window.phase, offset)?)?;
                    session.written_ticket = wave.ticket;
                }
            }
            drop(collected);
            for failed in window.failed_tickets() {
                session.write_raw(&StructuredServiceRecordV6::TicketFailed {
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
            session.write_raw(&StructuredServiceRecordV6::Footer {
                offered: offset
                    .checked_add(window.audit().issued as u64)
                    .ok_or_else(|| error("failed offer count overflow"))?,
                accepted_fifo_cutoff: cutoff,
                closing: paired(closing),
                failure: Some(reason.chars().take(4096).collect()),
            })?;
            session.footer_written = true;
            Ok(())
        })();
        let staged = session
            .source
            .take()
            .expect("failure writer retained above");
        let unpublished = staged.unpublished_receipt();
        if let Err(write) = result {
            self.remember_failed_source(
                state.generation,
                Some(unpublished),
                session.footer_written,
                true,
                &format!("{reason}; {write}"),
            );
            // Opted-in Drop retains the actual partial file within its original
            // source budget, even when disk IO cannot publish another name.
            drop(staged);
            return Err(write);
        }
        match staged.publish() {
            Ok(source) => {
                self.remember_failed_source(state.generation, Some(source), true, false, reason);
                Ok(())
            }
            Err(write) => {
                self.remember_failed_source(
                    state.generation,
                    Some(unpublished),
                    session.footer_written,
                    true,
                    &format!("{reason}; {write}"),
                );
                Err(error(write))
            }
        }
    }
}
