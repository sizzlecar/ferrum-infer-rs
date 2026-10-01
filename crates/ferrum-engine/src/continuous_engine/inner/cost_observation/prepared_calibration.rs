//! Private source8 adapter. Public diagnostics never mint live evidence. The
//! scheduler collector owns the one original population/hash/retained budget;
//! this layer binds its events to actual admitted owners and pre-issued work.
use super::*;
use crate::continuous_engine::{inner::calibration::*, SequenceState};
use ferrum_scheduler::implementations::continuous::{
    cost_model::structured_v2::prefixes::StructuredPrefixSlotV5,
    cost_profile::{
        self as file, StructuredPreparedOwnerBlockCollectorV8 as Collector,
        StructuredPreparedOwnerBlockDeclarationV8 as Declaration,
        StructuredPreparedOwnerBlockHeaderV8 as Header,
        StructuredPreparedOwnerBlockRecordV8 as Record, StructuredServiceClockV7 as Clock,
    },
};
use ferrum_types::{FerrumError, Result};
use file::StructuredSourceRecordSinkV1;
use serde::Serialize;
use std::mem::size_of;
mod cohorts;
mod journal;
pub(in crate::continuous_engine::inner) use journal::*;
mod native_acquisition;
mod records;

struct Slot {
    id: Option<RequestId>,
    owner: Option<u64>,
    generated: u64,
    maximum: u64,
    released: bool,
    completed: bool,
}
struct Cohort {
    pass: usize,
    ordinal: usize,
    admitted: usize,
    slots: Vec<Slot>,
}
struct OfferedRow {
    id: RequestId,
    owner: u64,
    generation: u64,
    generated: u64,
    work: ActualRowWork,
}
struct Offer {
    ticket: u64,
    issued_not_before_ns: u64,
    preparation: bool,
    rows: Vec<OfferedRow>,
}

pub(in crate::continuous_engine::inner) struct PreparedOwnerCalibration {
    declaration: Declaration,
    collector: Collector,
    clock: Arc<dyn CostObservationClock>,
    opening: u64,
    deadline: u64,
    active: Option<Cohort>,
    pending: Option<Offer>,
    ended: [usize; 3],
    offered: u64,
    last_fifo: u64,
    last_call: u64,
    block_open: bool,
    block_count: usize,
    last_block_closing: Option<Clock>,
    failure: Option<String>,
    stopped: bool,
    external_plan_bytes: Option<usize>,
    journal: Option<PreparedSourceJournal>,
    reuse_journal: Option<super::automatic_reuse::OriginalSourceJournal>,
}
struct OriginalJournals {
    diagnostic: Option<PreparedSourceJournal>,
    reuse: Option<super::automatic_reuse::OriginalSourceJournal>,
}
impl file::StructuredSourceRecordSinkV1 for OriginalJournals {
    fn append_original(&mut self, bytes: &[u8], receipt: (u64, [u8; 32])) {
        if let Some(journal) = &mut self.diagnostic {
            journal.append_original(bytes, receipt);
        }
        if let Some(journal) = &mut self.reuse {
            journal.append_original(bytes, receipt);
        }
    }
}
fn error(reason: impl std::fmt::Display) -> FerrumError {
    FerrumError::invalid_request(format!("prepared owner calibration: {reason}"))
}
fn phase(pass: usize) -> Result<file::StructuredProfilePhaseV10> {
    match pass {
        0 => Ok(file::StructuredProfilePhaseV10::Fit),
        1 => Ok(file::StructuredProfilePhaseV10::Residual),
        2 => Ok(file::StructuredProfilePhaseV10::Qualification),
        _ => Err(error("invalid original driver pass")),
    }
}
impl PreparedOwnerCalibration {
    pub fn new(
        header: Header,
        limits: file::CostProfileLoadLimits,
        clock: Arc<dyn CostObservationClock>,
        cutoff: u64,
        maximum_encoded_source_bytes: Option<std::num::NonZeroU64>,
    ) -> Result<Self> {
        Self::new_with_journal(
            header,
            limits,
            clock,
            cutoff,
            maximum_encoded_source_bytes,
            None,
        )
    }
    pub fn new_with_journal(
        header: Header,
        limits: file::CostProfileLoadLimits,
        clock: Arc<dyn CostObservationClock>,
        cutoff: u64,
        maximum_encoded_source_bytes: Option<std::num::NonZeroU64>,
        journal: Option<PreparedSourceJournal>,
    ) -> Result<Self> {
        Self::new_with_journals(
            header,
            limits,
            clock,
            cutoff,
            maximum_encoded_source_bytes,
            journal,
            None,
        )
    }
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_journals(
        header: Header,
        limits: file::CostProfileLoadLimits,
        clock: Arc<dyn CostObservationClock>,
        cutoff: u64,
        maximum_encoded_source_bytes: Option<std::num::NonZeroU64>,
        journal: Option<PreparedSourceJournal>,
        reuse_journal: Option<super::automatic_reuse::OriginalSourceJournal>,
    ) -> Result<Self> {
        header.validate().map_err(error)?;
        let opening = header.opening.monotonic_ns;
        let deadline = opening
            .checked_add(header.declaration.population.maximum_window_ns)
            .ok_or_else(|| error("original epoch deadline overflow"))?;
        // Clone only the declaration needed by the driver, not source bytes,
        // producer diagnostics, models or the collector's original records.
        let declaration = header.declaration.clone();
        let collector = if journal.is_some() || reuse_journal.is_some() {
            Collector::new_with_record_sink(
                header,
                limits,
                maximum_encoded_source_bytes,
                Box::new(OriginalJournals {
                    diagnostic: journal.clone(),
                    reuse: reuse_journal.clone(),
                }),
            )
        } else {
            match maximum_encoded_source_bytes {
                Some(budget) => Collector::new_streaming(header, limits, budget),
                None => Collector::new(header, limits),
            }
        }
        .map_err(error)?;
        let mut out = Self {
            declaration,
            collector,
            clock,
            opening,
            deadline,
            active: None,
            pending: None,
            ended: [0; 3],
            offered: 0,
            last_fifo: cutoff,
            last_call: 0,
            block_open: false,
            block_count: 0,
            last_block_closing: None,
            failure: None,
            stopped: false,
            external_plan_bytes: None,
            journal,
            reuse_journal,
        };
        out.charge(0)?;
        Ok(out)
    }
    pub fn collecting(&self) -> bool {
        self.failure.is_none() && !self.stopped
    }
    #[cfg(test)]
    pub(in crate::continuous_engine::inner) fn journal_observer(
        &self,
    ) -> Option<PreparedSourceJournalObserver> {
        self.journal.as_ref().map(PreparedSourceJournal::observer)
    }
    /// Bind the still-live immutable driver plan once, before any request is
    /// admitted. Its payload shares the collector's original retention limit.
    pub fn retain_execution_plan(&mut self, bytes: usize) -> Result<()> {
        if self.external_plan_bytes.is_some() || self.active.is_some() || self.offered != 0 {
            return Err(error("execution plan must bind before the first cohort"));
        }
        self.now()?;
        self.external_plan_bytes = Some(bytes);
        let result = self.charge(0);
        self.checked(result)
    }
    pub fn failure(&self) -> Option<&str> {
        self.failure.as_deref()
    }
    pub fn offered(&self) -> u64 {
        self.offered
    }
    pub fn last_fifo(&self) -> u64 {
        self.last_fifo
    }
    pub fn source_receipt(&self) -> (u64, [u8; 32]) {
        self.collector.source_receipt()
    }
    pub fn audit(&self) -> file::StructuredServiceAuditV7 {
        self.collector.audit()
    }
    pub fn prepared_audit(&self) -> file::StructuredPreparedOwnerBlockAuditV8 {
        self.collector.prepared_audit()
    }
    pub fn invalidate(&mut self, reason: impl std::fmt::Display) {
        if self.failure.is_some() || self.stopped {
            return;
        }
        let mut reason = reason.to_string();
        let mut end = reason.len().min(2048);
        while !reason.is_char_boundary(end) {
            end -= 1;
        }
        reason.truncate(end);
        let at = self.clock.now_ns().unwrap_or(self.opening);
        let ticket = self.pending.as_ref().map_or(self.offered, |p| p.ticket);
        // Failure retains the originally issued ticket, never replaces it.
        match self
            .collector
            .fail(ticket, self.last_fifo, at, reason.clone())
        {
            Ok(_) => {}
            Err(_) => self.archive_collector_failure(),
        }
        self.failure = Some(reason);
        let _ = self.charge(0);
    }
    fn checked<T>(&mut self, result: Result<T>) -> Result<T> {
        if let Err(e) = &result {
            self.invalidate(e);
        }
        result
    }
    fn now(&self) -> Result<u64> {
        if !self.collecting() {
            return Err(error("source is failed or stopped"));
        }
        self.clock
            .now_ns()
            .filter(|n| *n >= self.opening && *n <= self.deadline)
            .ok_or_else(|| error("original epoch clock is unavailable or expired"))
    }
    fn closing(&self) -> Result<Clock> {
        self.now()?;
        let c = profile_export::ExportClockReading::closing(self.clock.as_ref()).map_err(error)?;
        if c.monotonic_ns > self.deadline {
            return Err(error("original epoch expired"));
        }
        Ok(Clock {
            monotonic_ns: c.monotonic_ns,
            wall_unix_ns: c.wall_unix_ns,
        })
    }
    fn retained(&self) -> Option<usize> {
        let mut n = size_of::<Self>().checked_add(self.declaration.retained_payload_bytes()?)?;
        if let Some(c) = &self.active {
            n = n.checked_add(c.slots.capacity().checked_mul(size_of::<Slot>())?)?;
        }
        if let Some(p) = &self.pending {
            n = n.checked_add(p.rows.capacity().checked_mul(size_of::<OfferedRow>())?)?;
        }
        n.checked_add(self.failure.as_ref().map_or(0, String::capacity))?
            .checked_add(self.external_plan_bytes.unwrap_or(0))
    }
    fn charge(&mut self, extra: usize) -> Result<()> {
        let bytes = self
            .retained()
            .and_then(|n| n.checked_add(extra))
            .ok_or_else(|| error("driver retained payload overflow"))?;
        self.collector.retain_external(bytes).map_err(error)
    }
    fn push(&mut self, r: &Record) -> Result<()> {
        self.now()?;
        self.charge(0)?;
        // The optional sink observes only bytes the original collector hashes,
        // including a later original Failed record after any poisoned push.
        self.collector.push(r).map_err(error)?;
        self.charge(0)
    }
    fn archive_collector_failure(&mut self) {
        if let Some(journal) = &mut self.journal {
            journal.collector_failed();
        }
    }
    fn cohort_event(&mut self, value: serde_json::Value) -> Result<()> {
        self.push(&Record::Cohort(
            file::StructuredCohortEventV8::from_diagnostic(value).map_err(error)?,
        ))
    }
    fn preparation_event(&mut self, value: serde_json::Value) -> Result<()> {
        self.push(&Record::Preparation(
            file::StructuredPreparationEventV8::from_diagnostic(value).map_err(error)?,
        ))
    }
    fn ensure_block(&mut self) -> Result<()> {
        let now = self.now()?;
        if !self.block_open {
            self.collector
                .open_block(now, self.last_fifo)
                .map_err(error)?;
            self.block_open = true;
            self.block_count = 0;
        }
        Ok(())
    }
    fn settled(&mut self, call: u64, fifo: u64) -> Result<()> {
        if call <= self.last_call || self.last_fifo.checked_add(1) != Some(fifo) {
            return Err(error(
                "original call/FIFO is missing, duplicate or reordered",
            ));
        }
        self.last_call = call;
        self.last_fifo = fifo;
        self.pending = None;
        self.charge(0)?;
        self.block_count = self
            .block_count
            .checked_add(1)
            .ok_or_else(|| error("block overflow"))?;
        if self.block_count == self.declaration.population.schedule.block_offered {
            let closing = self.closing()?;
            self.collector.close_block(closing).map_err(error)?;
            self.last_block_closing = Some(closing);
            self.block_open = false;
        }
        Ok(())
    }
    pub fn checkpoint(&mut self) -> Result<file::StructuredPreparedOwnerBlockCheckpointV8> {
        let r = (|| {
            if self.pending.is_some()
                || self.active.is_some()
                || (0..3).any(|p| self.ended[p] != self.declaration.cohort_plan.phases[p].len())
            {
                return Err(error("checkpoint lacks complete original cohorts or block"));
            }
            self.now()?;
            let closing = if self.block_open {
                let closing = self.closing()?;
                self.collector
                    .seal_complete_cohorts_with_partial_tail(closing)
                    .map_err(error)?;
                self.block_open = false;
                self.last_block_closing = Some(closing);
                closing
            } else {
                // The checkpoint binds the original completed block. A later
                // clock reading cannot replace that source receipt; fresh
                // age/expiry is checked again when the worker activates it.
                self.last_block_closing
                    .ok_or_else(|| error("checkpoint has no original closing receipt"))?
            };
            let (_record, checkpoint) = self.collector.checkpoint(closing).map_err(error)?;
            if let Some(journal) = &self.reuse_journal {
                if let Err(reason) = journal.checkpoint(checkpoint.source_receipt()) {
                    tracing::info!(?reason, "Automatic source8 restart checkpoint unavailable");
                }
            }
            if let Some(journal) = &mut self.journal {
                let (source_bytes, source_sha256) = checkpoint.source_receipt();
                journal.checkpoint(PreparedSourceCheckpointReceipt {
                    source_bytes,
                    source_sha256,
                    qualified_children: checkpoint.qualified_children(),
                });
            }
            Ok(checkpoint)
        })();
        self.checked(r)
    }
    /// Optional cache archival runs only after the caller reaches the existing
    /// retirement boundary. Failure cannot change request or source retirement.
    pub fn finish_journal(&mut self) {
        if (self.journal.is_none() && self.reuse_journal.is_none()) || self.stopped {
            return;
        }
        if self.stop().is_err() {
            self.archive_collector_failure();
        }
    }
    pub fn stop(&mut self) -> Result<()> {
        if self.stopped {
            return Err(error("source already stopped"));
        }
        if self.pending.is_some() || self.active.is_some() || self.block_open {
            self.invalidate("stopped with incomplete original population");
        }
        let c = profile_export::ExportClockReading::closing(self.clock.as_ref()).map_err(error)?;
        self.collector
            .stop(Clock {
                monotonic_ns: c.monotonic_ns,
                wall_unix_ns: c.wall_unix_ns,
            })
            .map_err(error)?;
        if let Some(journal) = &mut self.journal {
            journal.finish(self.collector.source_receipt(), self.failure.is_none());
        }
        if let Some(journal) = &self.reuse_journal {
            journal.finish(self.collector.source_receipt());
        }
        self.stopped = true;
        Ok(())
    }
}

impl Drop for PreparedOwnerCalibration {
    fn drop(&mut self) {
        if let Some(journal) = &self.journal {
            journal.abandon();
        }
    }
}
