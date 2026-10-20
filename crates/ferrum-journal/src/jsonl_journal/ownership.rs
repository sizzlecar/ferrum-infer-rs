//! Unique writer ownership and component publication scopes on the original queue.
use super::*;

/// Journal-wide prefix captured while enqueueing the barrier under the gate.
/// These are not per-scope counts; later rejected publications do not alter it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct JsonlPublicationCut {
    pub accepted: u64,
    pub unbounded_accepted: u64,
    pub dropped: u64,
}

/// A fixed FIFO observation. Retirement and successful flushing are separate facts.
#[must_use = "inspect result() as well as scope retirement and barrier facts"]
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct JsonlFlushReceipt {
    pub cut: JsonlPublicationCut,
    pub flushed: u64,
    pub failed: u64,
    pub failure: Option<JsonlJournalError>,
    pub barrier_observed: bool,
    pub scope_retired: bool,
}
impl JsonlFlushReceipt {
    pub fn result(&self) -> Result<(), JsonlJournalError> {
        if let Some(error) = &self.failure {
            return Err(error.clone());
        }
        if !self.barrier_observed {
            return Err(JsonlJournalError::new(
                "JSONL flush barrier was not observed",
            ));
        }
        Ok(())
    }
    pub(super) fn rejected(cut: JsonlPublicationCut, error: JsonlJournalError) -> Self {
        Self {
            cut,
            flushed: 0,
            failed: 0,
            failure: Some(error),
            barrier_observed: false,
            scope_retired: false,
        }
    }
}

/// Final join evidence; success does not erase earlier drops or tool delivery.
#[must_use = "inspect result() as well as the worker join fact"]
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct JsonlFinalReceipt {
    pub flush: JsonlFlushReceipt,
    pub stats: JsonlJournalDeliveryStats,
    pub worker_joined: bool,
}
impl JsonlFinalReceipt {
    pub fn result(&self) -> Result<(), JsonlJournalError> {
        self.flush.result()?;
        if !self.worker_joined {
            return Err(JsonlJournalError::new("JSONL worker was not joined"));
        }
        Ok(())
    }
}

#[derive(Default)]
pub(super) struct PublicationState {
    pub(super) active_scopes: usize,
    pub(super) final_receipt: Option<JsonlFinalReceipt>,
}
pub(super) struct ScopeState {
    pub(super) retired: AtomicBool,
}
pub(super) struct BarrierRequest {
    pub(super) cut: JsonlPublicationCut,
    pub(super) scope_retired: bool,
    pub(super) reply: SyncSender<JsonlFlushReceipt>,
}
impl BarrierRequest {
    pub(super) fn reply(self, delivery: &DeliveryState, failure: &JournalFailure) {
        let stats = delivery.snapshot(false, failure.current_ref().map(JsonlJournalError::kind));
        let _ = self.reply.send(JsonlFlushReceipt {
            cut: self.cut,
            scope_retired: self.scope_retired,
            barrier_observed: true,
            flushed: stats.flushed,
            failed: stats.failed,
            failure: failure.current(),
        });
    }
}

/// The one final-close authority for a canonical writer. Not Clone.
/// Dropping it does not retire live scopes; the last original inner drains as fallback.
pub struct JsonlJournalOwner {
    pub(super) inner: Arc<JsonlJournalInner>,
}
impl fmt::Debug for JsonlJournalOwner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JsonlJournalOwner")
            .field("path", &self.path())
            .finish_non_exhaustive()
    }
}
impl JsonlJournalOwner {
    pub fn open(
        path: impl Into<PathBuf>,
        mode: JsonlJournalOpenMode,
        config: JsonlJournalConfig,
    ) -> io::Result<Self> {
        Ok(Self {
            inner: open_shared_inner(path, mode, Some(config))?,
        })
    }
    pub fn producer_scope(&self) -> Result<JsonlProducerScope, JsonlJournalError> {
        let mut gate = self
            .inner
            .send_gate
            .write()
            .map_err(|_| self.inner.record_poison("publication gate"))?;
        if self.inner.closed.load(Ordering::Acquire) {
            return Err(JsonlJournalError::with_kind(
                JsonlJournalErrorKind::Closed,
                "JSONL owner is closed",
            ));
        }
        if let Some(error) = self.inner.failure.current() {
            return Err(error);
        }
        gate.active_scopes = gate.active_scopes.checked_add(1).ok_or_else(|| {
            JsonlJournalError::with_kind(
                JsonlJournalErrorKind::Config,
                "too many JSONL producer scopes",
            )
        })?;
        Ok(JsonlProducerScope {
            inner: self.inner.clone(),
            state: Arc::new(ScopeState {
                retired: AtomicBool::new(false),
            }),
            receipt: None,
        })
    }
    pub fn close(&mut self) -> Result<JsonlFinalReceipt, JsonlJournalError> {
        self.inner.close_and_join()
    }
    pub fn path(&self) -> &Path {
        &self.inner.path
    }
    pub fn config(&self) -> JsonlJournalConfig {
        self.inner.config
    }
    pub fn delivery_stats(&self) -> JsonlJournalDeliveryStats {
        self.inner.delivery.snapshot(
            self.inner.closed.load(Ordering::Acquire),
            self.inner
                .failure
                .current_ref()
                .map(JsonlJournalError::kind),
        )
    }
}

/// One component's publication authority. Cloned typed producers share its retirement.
pub struct JsonlProducerScope {
    pub(super) inner: Arc<JsonlJournalInner>,
    pub(super) state: Arc<ScopeState>,
    receipt: Option<JsonlFlushReceipt>,
}
impl fmt::Debug for JsonlProducerScope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("JsonlProducerScope")
            .field("path", &self.path())
            .field("retired", &self.is_retired())
            .finish_non_exhaustive()
    }
}
impl JsonlProducerScope {
    pub fn producer<T: Serialize + Send + 'static>(&self) -> JsonlJournal<T> {
        JsonlJournal {
            inner: self.inner.clone(),
            scope: self.state.clone(),
            marker: PhantomData,
        }
    }
    pub fn finish(&mut self) -> JsonlFlushReceipt {
        if let Some(receipt) = &self.receipt {
            return receipt.clone();
        }
        let mut gate = self.inner.cleanup_gate();
        self.retire(&mut gate);
        let (cut, receive) = self.inner.enqueue_barrier(true, false);
        drop(gate);
        let receipt = self.inner.receive_barrier(cut, receive, true);
        self.receipt = Some(receipt.clone());
        receipt
    }
    fn retire(&self, gate: &mut PublicationState) {
        if !self.state.retired.swap(true, Ordering::AcqRel) {
            // Exactly the unique scope, not any producer clone, decrements this.
            gate.active_scopes -= 1;
        }
    }
    pub fn is_retired(&self) -> bool {
        self.state.retired.load(Ordering::Acquire)
    }
    pub fn path(&self) -> &Path {
        &self.inner.path
    }
    pub fn config(&self) -> JsonlJournalConfig {
        self.inner.config
    }
    pub fn delivery_stats(&self) -> JsonlJournalDeliveryStats {
        self.inner.delivery.snapshot(
            self.inner.closed.load(Ordering::Acquire),
            self.inner
                .failure
                .current_ref()
                .map(JsonlJournalError::kind),
        )
    }
}
impl Drop for JsonlProducerScope {
    fn drop(&mut self) {
        if !self.is_retired() {
            // Retirement only: Drop does not invent a flush receipt.
            let mut gate = self.inner.cleanup_gate();
            self.retire(&mut gate);
        }
    }
}

/// Standalone tools own one scope and its final-close authority; never Clone.
pub struct OwnedJsonlJournal<T: Serialize + Send + 'static> {
    pub(super) producer: JsonlJournal<T>,
    pub(super) scope: JsonlProducerScope,
    pub(super) owner: JsonlJournalOwner,
}
impl<T: Serialize + Send + 'static> fmt::Debug for OwnedJsonlJournal<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OwnedJsonlJournal")
            .field("path", &self.path())
            .finish_non_exhaustive()
    }
}
impl<T: Serialize + Send + 'static> OwnedJsonlJournal<T> {
    pub fn open(
        path: impl Into<PathBuf>,
        mode: JsonlJournalOpenMode,
        config: JsonlJournalConfig,
    ) -> io::Result<Self> {
        let owner = JsonlJournalOwner::open(path, mode, config)?;
        let scope = owner.producer_scope().map_err(io::Error::other)?;
        let producer = scope.producer();
        Ok(Self {
            producer,
            scope,
            owner,
        })
    }
    pub fn create(path: impl Into<PathBuf>) -> io::Result<Self> {
        Self::open(
            path,
            JsonlJournalOpenMode::Truncate,
            JsonlJournalConfig::default(),
        )
    }
    pub fn append(path: impl Into<PathBuf>) -> io::Result<Self> {
        Self::open(
            path,
            JsonlJournalOpenMode::Append,
            JsonlJournalConfig::default(),
        )
    }
    pub fn producer(&self) -> &JsonlJournal<T> {
        &self.producer
    }
    pub fn enqueue(&self, event: T) -> Result<(), JsonlJournalError> {
        self.producer.enqueue(event)
    }
    pub fn enqueue_batch(&self, events: Vec<T>) -> Result<(), JsonlJournalError> {
        self.producer.enqueue_batch(events)
    }
    pub fn try_enqueue(&self, event: T) -> Result<(), JsonlJournalError> {
        self.producer.try_enqueue(event)
    }
    pub fn try_enqueue_batch(&self, events: Vec<T>) -> Result<(), JsonlJournalError> {
        self.producer.try_enqueue_batch(events)
    }
    pub fn flush(&self) -> JsonlFlushReceipt {
        self.producer.flush()
    }
    pub fn close(&mut self) -> Result<JsonlFinalReceipt, JsonlJournalError> {
        // Always attempt actual join even when this scope's flush reports a failure.
        let _scope_receipt = self.scope.finish();
        self.owner.close()
    }
    pub fn path(&self) -> &Path {
        self.producer.path()
    }
    pub fn config(&self) -> JsonlJournalConfig {
        self.producer.config()
    }
    pub fn delivery_stats(&self) -> JsonlJournalDeliveryStats {
        self.producer.delivery_stats()
    }
}
