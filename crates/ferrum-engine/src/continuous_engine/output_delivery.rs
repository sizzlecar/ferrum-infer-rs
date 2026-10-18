//! Output custody is independent of model custody. Backpressure retains byte
//! ranges or charged chunks, never a second token history or physical ledger.
use super::*;
use ferrum_output::{OutputChunk, OutputSession};
use ferrum_scheduler::implementations::continuous::OutputReadinessWake;
use std::ops::Range;

mod registry;
pub(super) use registry::OutputTaskRegistry;

pub(super) struct TextSpan {
    pub(super) bytes: Range<usize>,
    pub(super) token: Option<TokenId>,
    pub(super) terminal: Option<FinishReason>,
    pub(super) created_at: chrono::DateTime<chrono::Utc>,
    cost: usize,
    charged: Option<OutputChunk>,
}

struct DeliveryState {
    pending: VecDeque<TextSpan>,
    outstanding: usize,
    running: bool,
    generation: u64,
    wake: Option<OutputReadinessWake>,
    completion: Option<inner::completion::CompletionHandoff>,
    failure: Option<FerrumError>,
    enqueued_bytes: usize,
}

struct Delivery {
    sender: mpsc::Sender<Result<OutputChunk>>,
    session: OutputSession,
    inner: Weak<EngineInner>,
    request_id: RequestId,
    // Weak back-reference: the live state already owns this OutputSender.
    sequence: Mutex<Weak<SequenceHandle>>,
    state: Mutex<DeliveryState>,
    changed: Notify,
    terminal: Arc<Mutex<OutputTerminal>>,
    #[cfg(test)]
    failed_handoff: Mutex<Option<FailedHandoffHook>>,
    #[cfg(test)]
    panic_before_drain: AtomicBool,
}

// One bounded shutdown fallback, independent of channel capacity. It is only
// visible after queued payloads drain and every sender has been relinquished.
#[derive(Default)]
struct OutputTerminal {
    interrupted: bool,
    delivered: bool,
}

pub(super) struct OutputReceiver {
    receiver: mpsc::Receiver<Result<OutputChunk>>,
    terminal: Arc<Mutex<OutputTerminal>>,
}

impl OutputReceiver {
    fn terminal_item(&self) -> Option<Result<OutputChunk>> {
        let mut terminal = self.terminal.lock();
        if terminal.interrupted && !terminal.delivered {
            terminal.delivered = true;
            Some(Err(FerrumError::scheduler(
                "engine stopped before output completion",
            )))
        } else {
            None
        }
    }

    #[cfg(test)]
    pub(super) async fn recv(&mut self) -> Option<Result<OutputChunk>> {
        self.receiver.recv().await.or_else(|| self.terminal_item())
    }

    #[cfg(test)]
    pub(super) fn try_recv(
        &mut self,
    ) -> std::result::Result<Result<OutputChunk>, mpsc::error::TryRecvError> {
        match self.receiver.try_recv() {
            Err(mpsc::error::TryRecvError::Disconnected) => self
                .terminal_item()
                .ok_or(mpsc::error::TryRecvError::Disconnected),
            result => result,
        }
    }
}

impl Stream for OutputReceiver {
    type Item = Result<OutputChunk>;
    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match self.receiver.poll_recv(cx) {
            Poll::Ready(None) => Poll::Ready(self.terminal_item()),
            result => result,
        }
    }
}

#[derive(Clone)]
pub(super) struct OutputSender(Arc<Delivery>);

impl OutputSender {
    pub(super) fn new(
        inner: Weak<EngineInner>,
        request_id: RequestId,
        session: OutputSession,
        capacity: usize,
    ) -> (Self, OutputReceiver) {
        let (sender, receiver) = mpsc::channel(capacity);
        let terminal = Arc::new(Mutex::new(OutputTerminal::default()));
        (
            Self(Arc::new(Delivery {
                sender,
                session,
                inner,
                request_id,
                sequence: Mutex::new(Weak::new()),
                changed: Notify::new(),
                terminal: Arc::clone(&terminal),
                #[cfg(test)]
                failed_handoff: Mutex::new(None),
                #[cfg(test)]
                panic_before_drain: AtomicBool::new(false),
                state: Mutex::new(DeliveryState {
                    pending: VecDeque::new(),
                    outstanding: 0,
                    running: false,
                    generation: 0,
                    wake: None,
                    completion: None,
                    failure: None,
                    enqueued_bytes: 0,
                }),
            })),
            OutputReceiver { receiver, terminal },
        )
    }

    pub(super) fn bind_sequence(&self, sequence: Weak<SequenceHandle>) {
        *self.0.sequence.lock() = sequence;
    }

    pub(super) fn source_sequence(&self) -> Option<Arc<SequenceHandle>> {
        self.0.sequence.lock().upgrade()
    }

    pub(super) fn same_owner(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    pub(super) fn is_closed(&self) -> bool {
        self.0.sender.is_closed()
    }

    pub(super) fn failure(&self) -> Option<FerrumError> {
        self.0.state.lock().failure.clone()
    }

    /// Called under the original outer sequence lock, before another wave can
    /// be scheduled. The second span is the terminal stop-prefix flush.
    pub(super) fn publish_text(
        &self,
        inner: &EngineInner,
        chunk: StreamChunk,
        bytes: Range<usize>,
        terminal: Option<FinishReason>,
    ) -> Result<()> {
        let cost = OutputChunk::queued_cost(&chunk)?;
        let mut span = TextSpan {
            bytes,
            token: chunk.token,
            terminal,
            created_at: chunk.created_at,
            cost,
            charged: None,
        };
        let mut state = self.0.state.lock();
        if state.pending.is_empty() && !state.running {
            if let Some(lease) = self.0.session.try_reserve_queued(cost)? {
                let charged = OutputChunk::new(chunk, lease)?;
                match self.0.sender.try_send(Ok(charged)) {
                    Ok(()) => {
                        state.enqueued_bytes = span.bytes.end;
                        return Ok(());
                    }
                    Err(mpsc::error::TrySendError::Closed(_)) => return Ok(()),
                    Err(mpsc::error::TrySendError::Full(Ok(charged))) => {
                        span.charged = Some(charged);
                    }
                    Err(mpsc::error::TrySendError::Full(Err(error))) => return Err(error),
                }
            }
        }
        // Uncharged chunk has been dropped. Only its bounded descriptor is
        // retained; the original sequence or completion response owns the text.
        if state.outstanding >= 2 {
            return Err(FerrumError::internal(
                "output produced more than two pending text spans",
            ));
        }
        if state.wake.is_none() {
            state.wake = Some(inner.scheduler.pause_for_output(&self.0.request_id)?);
        }
        state.pending.push_back(span);
        state.outstanding += 1;
        self.start_locked(inner, &mut state)
    }

    fn start_locked(&self, inner: &EngineInner, state: &mut DeliveryState) -> Result<()> {
        if !state.running {
            let generation = state
                .generation
                .checked_add(1)
                .ok_or_else(|| FerrumError::internal("output generation space exhausted"))?;
            let sender = self.clone();
            let stop = inner.output_tasks.stop_receiver();
            drop(
                inner
                    .output_tasks
                    .spawn(async move { sender.run(stop, generation).await })?,
            );
            state.generation = generation;
            state.running = true;
        }
        Ok(())
    }

    pub(super) fn complete(
        &self,
        inner: &EngineInner,
        mut completion: inner::completion::CompletionHandoff,
    ) -> Result<()> {
        completion.stream_sender = None;
        let mut state = self.0.state.lock();
        if state.completion.is_some() {
            return Err(FerrumError::internal("output terminal already registered"));
        }
        if let Some(error) = state.failure.clone() {
            completion.retain_delivery_error(error);
        }
        state.completion = Some(completion);
        self.0.changed.notify_one();
        self.start_locked(inner, &mut state)
    }

    pub(super) async fn send(&self, item: Result<StreamChunk>) -> Result<()> {
        let terminal = item
            .as_ref()
            .map_or(true, |chunk| chunk.finish_reason.is_some());
        let item = match item {
            Ok(chunk) => {
                let cost = OutputChunk::queued_cost(&chunk)?;
                let lease = tokio::select! {
                    biased;
                    _ = self.0.sender.closed() => return Err(FerrumError::internal("output receiver closed")),
                    lease = self.0.session.reserve_queued(cost) => lease?,
                };
                Ok(OutputChunk::new(chunk, lease)?)
            }
            Err(error) => Err(error),
        };
        self.0
            .sender
            .send(item)
            .await
            .map_err(|_| FerrumError::internal("output receiver closed"))?;
        if terminal {
            self.0.terminal.lock().delivered = true;
        }
        Ok(())
    }

    pub(super) fn interrupt_delivery(&self) {
        self.0.terminal.lock().interrupted = true;
    }

    pub(super) fn record_failure(&self, error: FerrumError) {
        self.0.state.lock().failure.get_or_insert(error);
    }

    async fn run(
        self,
        mut stop: tokio::sync::watch::Receiver<bool>,
        generation: u64,
    ) -> Result<()> {
        let mut guard = DeliveryRunGuard {
            sender: self.clone(),
            generation,
            armed: true,
        };
        tokio::select! {
            biased;
            _ = wait_for_stop(&mut stop) => {},
            _ = async {
                let outcome = std::panic::AssertUnwindSafe(self.drain(generation)).catch_unwind().await
                    .unwrap_or_else(|_| Err(FerrumError::internal("output delivery panicked")));
                if let Err(error) = outcome {
                    let completion = self.end_failed_run(generation, error.clone());
                    // The old generation no longer owns the shared running
                    // state. A concurrent completion may now start a new one.
                    guard.armed = false;
                    #[cfg(test)]
                    {
                        let hook = self.0.failed_handoff.lock().take();
                        if let Some(hook) = hook {
                            let _ = hook.reached.send(());
                            let _ = hook.resume.await;
                        }
                    }
                    if let Some(mut completion) = completion {
                        completion.retain_delivery_error(error);
                        if let Some(inner) = self.0.inner.upgrade() {
                            inner.finish_output_completion(&self.0.request_id, completion, Some(&self)).await;
                        }
                    }
                }
                guard.armed = false;
            } => {},
        }
        Ok(())
    }

    fn end_failed_run(
        &self,
        generation: u64,
        error: FerrumError,
    ) -> Option<inner::completion::CompletionHandoff> {
        let completion = {
            let mut state = self.0.state.lock();
            if state.generation != generation || !state.running {
                return None;
            }
            state.failure.get_or_insert(error);
            state.pending.clear();
            state.outstanding = 0;
            // Publish readiness before idle becomes visible. Otherwise a new
            // pause could reuse this still-blocked wake and be resumed by us.
            if let Some(wake) = state.wake.take() {
                wake.resume();
            }
            state.running = false;
            state.completion.take()
        };
        self.notify_driver();
        completion
    }

    fn notify_driver(&self) {
        if let Some(inner) = self.0.inner.upgrade() {
            inner.work_notify.notify_one();
        }
    }

    async fn drain(&self, generation: u64) -> Result<()> {
        #[cfg(test)]
        if self.0.panic_before_drain.swap(false, Ordering::SeqCst) {
            panic!("injected output projection panic");
        }
        loop {
            // The empty transition and running flag are one transaction. A
            // producer arriving afterwards starts a new registered task.
            let (span, completion) = {
                let mut state = self.0.state.lock();
                if let Some(span) = state.pending.pop_front() {
                    (Some(span), None)
                } else if let Some(completion) = state.completion.take() {
                    (None, Some(completion))
                } else {
                    if let Some(wake) = state.wake.take() {
                        wake.resume();
                    }
                    state.running = false;
                    (None, None)
                }
            };
            if let Some(mut span) = span {
                let mut sent = false;
                if !self.is_closed() {
                    let chunk = if let Some(chunk) = span.charged.take() {
                        Some(chunk)
                    } else {
                        let lease = tokio::select! {
                            _ = self.0.sender.closed() => None,
                            lease = self.0.session.reserve_queued(span.cost) => Some(lease?),
                        };
                        match lease {
                            None => None,
                            Some(lease) => {
                                let chunk = loop {
                                    let changed = self.0.changed.notified();
                                    let chunk = {
                                        let inner = self.0.inner.upgrade().ok_or_else(|| {
                                            FerrumError::internal("output source has stopped")
                                        })?;
                                        inner.rebuild_output_span(
                                            self,
                                            &self.0.request_id,
                                            &span,
                                        )?
                                    };
                                    if let Some(chunk) = chunk {
                                        break Some(chunk);
                                    }
                                    // Client cancellation removes the source without a
                                    // completion response. It must also unblock this wait.
                                    tokio::select! {
                                        _ = self.0.sender.closed() => break None,
                                        _ = changed => {},
                                    }
                                };
                                chunk
                                    .map(|chunk| OutputChunk::new(chunk, lease))
                                    .transpose()?
                            }
                        }
                    };
                    if let Some(chunk) = chunk {
                        sent = self.0.sender.send(Ok(chunk)).await.is_ok();
                    }
                }
                let mut state = self.0.state.lock();
                if sent {
                    state.enqueued_bytes = span.bytes.end;
                }
                state.outstanding -= 1;
                continue;
            }
            if let Some(completion) = completion {
                if let Some(inner) = self.0.inner.upgrade() {
                    inner
                        .finish_output_completion(&self.0.request_id, completion, Some(self))
                        .await;
                }
                // No product producer remains after completion was handed off.
                {
                    let mut state = self.0.state.lock();
                    if state.generation == generation {
                        if let Some(wake) = state.wake.take() {
                            wake.resume();
                        }
                        state.running = false;
                    }
                }
                self.notify_driver();
                return Ok(());
            }
            self.notify_driver();
            return Ok(());
        }
    }

    pub(super) fn completed_text(&self, range: Range<usize>) -> Result<Option<String>> {
        let state = self.0.state.lock();
        state
            .completion
            .as_ref()
            .map(|completion| {
                completion
                    .response
                    .text
                    .get(range)
                    .map(str::to_owned)
                    .ok_or_else(|| {
                        FerrumError::internal("pending output span disagrees with completed text")
                    })
            })
            .transpose()
    }
}

pub(super) async fn wait_for_stop(stop: &mut tokio::sync::watch::Receiver<bool>) {
    while !*stop.borrow_and_update() {
        if stop.changed().await.is_err() {
            break;
        }
    }
}

impl EngineInner {
    #[cfg(test)]
    pub(in crate::continuous_engine) fn output_channel(
        &self,
        request_id: RequestId,
        capacity: usize,
    ) -> Result<(OutputSender, OutputReceiver)> {
        let session = self.output_budget.try_open_session()?;
        Ok(OutputSender::new(
            self.self_weak.clone(),
            request_id,
            session,
            capacity,
        ))
    }
}

struct DeliveryRunGuard {
    sender: OutputSender,
    generation: u64,
    armed: bool,
}
impl Drop for DeliveryRunGuard {
    fn drop(&mut self) {
        if self.armed {
            self.sender.interrupt_delivery();
            self.sender.end_failed_run(
                self.generation,
                FerrumError::scheduler("output delivery interrupted"),
            );
        }
    }
}

impl std::fmt::Debug for OutputSender {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OutputSender")
            .field("request_id", &self.0.request_id)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
struct FailedHandoffHook {
    reached: tokio::sync::oneshot::Sender<()>,
    resume: tokio::sync::oneshot::Receiver<()>,
}

#[cfg(test)]
mod tests;
