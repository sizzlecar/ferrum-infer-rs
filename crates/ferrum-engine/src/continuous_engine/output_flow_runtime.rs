//! Bounded per-request transport actor.
//!
//! Sampling, stop decisions and detokenization remain engine-owned. A ready
//! grant authorizes output storage and mailbox slots only, never model/KV work.
//! Slow-consumer deadlines cover continuous wire/event pressure. Projection
//! and device-owned in-flight grants do not start this output deadline.

use ferrum_interfaces::output_credit::{LeasedOutput, OutputCreditAmount, OutputCreditWake};
use ferrum_interfaces::output_flow::{
    BoundedOutputError, CreditedOutputFrame, CreditedOutputSession, OutputCompletion,
    OutputConsumerControl, OutputFlowError, OutputFrameAttempt, OutputFrameMetadata,
    OutputFramePermit, OutputTerminal, PrepaidOutputCapacityView, RequestOutputBudget,
};
use ferrum_types::{SloOutputConfig, TokenId};
use futures::future::BoxFuture;
use parking_lot::Mutex;
use std::{
    sync::{
        atomic::{AtomicBool, AtomicU8, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::sync::{mpsc, oneshot, watch, Notify};

mod chat;
mod planning;
use chat::PendingProjection;
use ferrum_interfaces::output_flow::BoundedChatProjection;

#[cfg(test)]
mod tests;

trait OutputFlowTimer: Send + Sync {
    fn now(&self) -> Instant;
    fn sleep_until(&self, deadline: Instant) -> BoxFuture<'static, ()>;
}

struct TokioOutputFlowTimer;
impl OutputFlowTimer for TokioOutputFlowTimer {
    fn now(&self) -> Instant {
        Instant::now()
    }
    fn sleep_until(&self, deadline: Instant) -> BoxFuture<'static, ()> {
        Box::pin(tokio::time::sleep_until(tokio::time::Instant::from_std(
            deadline,
        )))
    }
}

pub(super) struct OutputFlowRuntimeOptions {
    slow_consumer_timeout: Duration,
    max_queued_events: usize,
    timer: Arc<dyn OutputFlowTimer>,
}

impl OutputFlowRuntimeOptions {
    pub fn from_config(config: &SloOutputConfig) -> Self {
        Self {
            slow_consumer_timeout: config.slow_consumer_timeout(),
            max_queued_events: config.max_queued_events_per_request.get(),
            timer: Arc::new(TokioOutputFlowTimer),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum OutputBlockReason {
    EventCredit,
    WireQueue,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum OutputCloseReason {
    Complete,
    Cancelled,
    Disconnected,
    Abandoned,
    InvalidOutput,
    SlowConsumer,
}

impl OutputCloseReason {
    pub(super) fn message(self) -> &'static str {
        match self {
            Self::Complete => "output owner completed before inference retired",
            Self::Cancelled => "output consumer cancelled",
            Self::Disconnected => "output consumer disconnected",
            Self::Abandoned => "output grant was abandoned",
            Self::InvalidOutput => "output owner rejected invalid output",
            Self::SlowConsumer => "output consumer exceeded its blocked-time budget",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PortState {
    ProjectionBusy,
    Ready,
    InFlight,
    OutputBlocked(OutputBlockReason),
    Closing(OutputCloseReason),
}

pub(super) enum OutputReadiness {
    Ready(ReadyOutputGrant),
    ProjectionBusy,
    OutputBlocked(OutputBlockReason),
    Closing(OutputCloseReason),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum OutputReadinessState {
    Ready,
    ProjectionBusy,
    OutputBlocked(OutputBlockReason),
    Closing(OutputCloseReason),
}

/// Observational only: copies do not own or reserve any output capacity.
/// A device submission must still acquire and validate its real ready grant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum OutputPlanningCreditView {
    Ready(ReadyOutputCreditView),
    ProjectionBusy,
    OutputBlocked(OutputBlockReason),
    Closing(OutputCloseReason),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct ReadyOutputCreditView {
    /// Always one future committed token, independent of SSE event count.
    pub future_tokens: u32,
    /// Exact credit held by the published frame permit, not account headroom.
    /// Bytes are the remaining admitted plan reservoir: a conservative bound
    /// covering this token's whole projection, including multiple Chat frames.
    /// Projection storage and terminal escrow remain separately owner-held.
    pub reserved: OutputCreditAmount,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct OutputPlanningSnapshot {
    pub readiness: OutputPlanningCreditView,
    /// Present only while the corresponding immediate grant remains Ready.
    pub future_capacity: Option<PrepaidOutputCapacityView>,
}

pub(super) struct OutputDelta {
    /// Engine-decoded text after UTF-8/stop handling, covered by the projection
    /// grant. The actor may apply the admitted bounded Chat view, never decode.
    pub text: String,
    pub token: Option<TokenId>,
    pub generated_tokens: usize,
    pub created: u64,
}

pub(super) struct OutputTerminalDecision {
    /// Last accepted output-command ordinal, not executor work generation.
    pub through_output_ordinal: u64,
    pub outcome: OutputCompletion,
    /// Engine-projected stop/UTF-8 tail, emitted after the accepted command.
    pub final_text: String,
    pub created: u64,
}

struct GrantParts {
    ordinal: u64,
    frame: OutputFramePermit,
    command_slot: mpsc::OwnedPermit<OwnerCommand>,
    wire_slot: mpsc::OwnedPermit<CreditedOutputFrame>,
    future_capacity: PrepaidOutputCapacityView,
}

pub(super) struct ReadyOutputGrant {
    parts: Option<GrantParts>,
    shared: Arc<Shared>,
}

impl ReadyOutputGrant {
    #[cfg(test)]
    pub fn ordinal(&self) -> u64 {
        self.parts.as_ref().expect("live grant").ordinal
    }

    /// Nonblocking: both command and wire slots were reserved before this
    /// grant was published. The engine must have committed exactly this work.
    pub fn committed(mut self, delta: OutputDelta) -> u64 {
        let parts = self.parts.take().expect("grant consumed once");
        let ordinal = parts.ordinal;
        let _ = parts.command_slot.send(OwnerCommand::Committed {
            ordinal,
            frame: parts.frame,
            wire_slot: parts.wire_slot,
            delta,
        });
        ordinal
    }

    pub fn return_unsubmitted(mut self) {
        let parts = self.parts.take().expect("grant consumed once");
        let _ = parts.command_slot.send(OwnerCommand::Unsubmitted {
            frame: parts.frame,
            wire_slot: parts.wire_slot,
        });
    }
}

impl Drop for ReadyOutputGrant {
    fn drop(&mut self) {
        if let Some(parts) = self.parts.take() {
            let _ = parts.command_slot.send(OwnerCommand::Abandoned {
                frame: parts.frame,
                wire_slot: parts.wire_slot,
            });
            self.shared.engine_wake.notify_one();
        }
    }
}

enum OwnerCommand {
    Committed {
        ordinal: u64,
        frame: OutputFramePermit,
        wire_slot: mpsc::OwnedPermit<CreditedOutputFrame>,
        delta: OutputDelta,
    },
    Unsubmitted {
        frame: OutputFramePermit,
        wire_slot: mpsc::OwnedPermit<CreditedOutputFrame>,
    },
    Abandoned {
        frame: OutputFramePermit,
        wire_slot: mpsc::OwnedPermit<CreditedOutputFrame>,
    },
}

struct Mailbox {
    state: PortState,
    ready: Option<GrantParts>,
}
struct Shared {
    mailbox: Mutex<Mailbox>,
    change: watch::Sender<u64>,
    cancel: AtomicU8,
    consumer_closed: AtomicBool,
    owner_wake: Notify,
    engine_wake: Arc<Notify>,
}

impl Shared {
    fn signal(&self) {
        self.change
            .send_modify(|version| *version = version.wrapping_add(1));
        self.engine_wake.notify_one();
    }
    fn state(&self, state: PortState) {
        let mut mailbox = self.mailbox.lock();
        if !matches!(mailbox.state, PortState::Closing(_)) || matches!(state, PortState::Closing(_))
        {
            mailbox.state = state;
        }
        drop(mailbox);
        self.signal();
    }
    fn cancel(&self, disconnected: bool) {
        if disconnected {
            self.consumer_closed.store(true, Ordering::Release);
        }
        let code = if disconnected { 2 } else { 1 };
        let _ = self
            .cancel
            .compare_exchange(0, code, Ordering::AcqRel, Ordering::Acquire);
        self.state(PortState::Closing(
            self.cancel_reason().expect("cancel set"),
        ));
        self.owner_wake.notify_one();
    }
    fn cancel_reason(&self) -> Option<OutputCloseReason> {
        match self.cancel.load(Ordering::Acquire) {
            0 => None,
            2 => Some(OutputCloseReason::Disconnected),
            _ => Some(OutputCloseReason::Cancelled),
        }
    }
}

impl OutputConsumerControl for Shared {
    fn consumer_dropped(&self) {
        self.cancel(true);
    }
}

pub(super) struct OutputFlowPort {
    shared: Arc<Shared>,
    terminal: Option<oneshot::Sender<OutputTerminalDecision>>,
    engine_lifetime: Option<Arc<OutputProjectionLifetime>>,
}

struct OutputProjectionLifetime {
    _sender: oneshot::Sender<()>,
}

/// Keeps existing projection ownership alive for detached text work. It does
/// not duplicate a reservation or authorize an additional decoder workspace.
pub(super) struct OutputProjectionGuard {
    _lifetime: Arc<OutputProjectionLifetime>,
}

impl OutputFlowPort {
    /// Acquire while the sequence's port is still protected by the map lock.
    /// The actor waits for both the port and every detached projection to drop.
    pub fn projection_guard(&self) -> OutputProjectionGuard {
        OutputProjectionGuard {
            _lifetime: self.engine_lifetime.as_ref().expect("live port").clone(),
        }
    }
    pub fn consumer_closed(&self) -> bool {
        self.shared.consumer_closed.load(Ordering::Acquire)
    }
    pub fn readiness(&self) -> OutputReadinessState {
        self.planning_credit_view().readiness_state()
    }
    pub fn try_take(&self) -> OutputReadiness {
        if let Some(reason) = self.shared.cancel_reason() {
            return OutputReadiness::Closing(reason);
        }
        let Some(mut mailbox) = self.shared.mailbox.try_lock() else {
            return OutputReadiness::ProjectionBusy;
        };
        if let PortState::Closing(reason) = mailbox.state {
            return OutputReadiness::Closing(reason);
        }
        if let Some(parts) = mailbox.ready.take() {
            mailbox.state = PortState::InFlight;
            return OutputReadiness::Ready(ReadyOutputGrant {
                parts: Some(parts),
                shared: self.shared.clone(),
            });
        }
        match mailbox.state {
            PortState::OutputBlocked(reason) => OutputReadiness::OutputBlocked(reason),
            PortState::Closing(reason) => OutputReadiness::Closing(reason),
            _ => OutputReadiness::ProjectionBusy,
        }
    }
    #[cfg(test)]
    pub fn subscribe(&self) -> watch::Receiver<u64> {
        self.shared.change.subscribe()
    }
    pub fn cancel(&self) {
        self.shared.cancel(false);
    }
    pub fn terminal(
        &mut self,
        terminal: OutputTerminalDecision,
    ) -> Result<(), OutputTerminalDecision> {
        let Some(sender) = self.terminal.take() else {
            return Err(terminal);
        };
        self.shared
            .state(PortState::Closing(OutputCloseReason::Complete));
        sender.send(terminal)
    }
}

impl Drop for OutputFlowPort {
    fn drop(&mut self) {
        if self.terminal.is_some() {
            self.shared.cancel(false);
        }
        // Engine must destroy or transfer all charged sequence history first.
        drop(self.engine_lifetime.take());
    }
}

enum WaitFor {
    None,
    Credit(OutputCreditWake),
    Wire(
        BoxFuture<
            'static,
            Result<mpsc::OwnedPermit<CreditedOutputFrame>, mpsc::error::SendError<()>>,
        >,
    ),
}
enum FrameReservation {
    Reserved(OutputFramePermit, mpsc::OwnedPermit<CreditedOutputFrame>),
    Wait(WaitFor),
}

enum Event {
    Command(OwnerCommand),
    Terminal(OutputTerminalDecision),
    Capacity(Option<mpsc::OwnedPermit<CreditedOutputFrame>>),
    Close(OutputCloseReason),
}

struct Owner {
    shared: Arc<Shared>,
    commands: mpsc::Receiver<OwnerCommand>,
    command_sender: mpsc::Sender<OwnerCommand>,
    terminal: Option<oneshot::Receiver<OutputTerminalDecision>>,
    wire: mpsc::Sender<CreditedOutputFrame>,
    terminal_slot: Option<mpsc::OwnedPermit<CreditedOutputFrame>>,
    completion: Option<oneshot::Sender<LeasedOutput<OutputCompletion>>>,
    applied: u64,
    generated_tokens: usize,
    pending_terminal: Option<OutputTerminalDecision>,
    chat_projection: Option<BoundedChatProjection>,
    pending_projection: Option<PendingProjection>,
    acquired_wire_slot: Option<mpsc::OwnedPermit<CreditedOutputFrame>>,
    engine_lifetime: oneshot::Receiver<()>,
    options: OutputFlowRuntimeOptions,
    blocked_since: Option<Instant>,
    data_slots: usize,
    // Last field: commands/history must be destroyed before their storage lease.
    budget: RequestOutputBudget,
}

/// One task/command slot per request. Data slots are bounded by the configured
/// event allowance and admitted plan; terminal has its own reserved wire slot.
pub(super) fn spawn_output_flow_runtime(
    budget: RequestOutputBudget,
    engine_wake: Arc<Notify>,
    options: OutputFlowRuntimeOptions,
) -> (OutputFlowPort, CreditedOutputSession) {
    let (command_sender, commands) = mpsc::channel(1);
    let data_slots = options
        .max_queued_events
        .saturating_sub(budget.plan().terminal_credit().events)
        .min(budget.data_event_capacity())
        .min(budget.plan().max_data_frames())
        .min(tokio::sync::Semaphore::MAX_PERMITS - 1);
    let (wire, frames) = mpsc::channel(data_slots + 1);
    let terminal_slot = wire
        .clone()
        .try_reserve_owned()
        .expect("new wire channel has a terminal slot");
    let (terminal_sender, terminal) = oneshot::channel();
    let (completion_sender, completion) = oneshot::channel();
    let (engine_lifetime_sender, engine_lifetime) = oneshot::channel();
    let (change, _) = watch::channel(0);
    let shared = Arc::new(Shared {
        mailbox: Mutex::new(Mailbox {
            state: PortState::ProjectionBusy,
            ready: None,
        }),
        change,
        cancel: AtomicU8::new(0),
        consumer_closed: AtomicBool::new(false),
        owner_wake: Notify::new(),
        engine_wake,
    });
    let owner = Owner {
        budget,
        shared: shared.clone(),
        commands,
        command_sender,
        terminal: Some(terminal),
        wire,
        terminal_slot: Some(terminal_slot),
        completion: Some(completion_sender),
        applied: 0,
        generated_tokens: 0,
        pending_terminal: None,
        chat_projection: None,
        pending_projection: None,
        acquired_wire_slot: None,
        engine_lifetime,
        options,
        blocked_since: None,
        data_slots,
    };
    tokio::spawn(owner.run());
    (
        OutputFlowPort {
            shared: shared.clone(),
            terminal: Some(terminal_sender),
            engine_lifetime: Some(Arc::new(OutputProjectionLifetime {
                _sender: engine_lifetime_sender,
            })),
        },
        CreditedOutputSession::from_receivers(frames, completion, shared),
    )
}

impl Owner {
    fn blocked_deadline(&self) -> Result<Option<Instant>, OutputCloseReason> {
        self.blocked_since
            .map(|since| {
                since
                    .checked_add(self.options.slow_consumer_timeout)
                    .ok_or(OutputCloseReason::InvalidOutput)
            })
            .transpose()
    }

    fn blocked(&mut self, reason: OutputBlockReason) -> Result<(), OutputCloseReason> {
        let now = self.options.timer.now();
        self.blocked_since.get_or_insert(now);
        let deadline = self.blocked_deadline()?.expect("blocked timestamp set");
        if now >= deadline {
            return Err(OutputCloseReason::SlowConsumer);
        }
        self.shared.state(PortState::OutputBlocked(reason));
        Ok(())
    }
    fn revoke_ready(&mut self) -> Result<(), OutputFlowError> {
        let ready = self.shared.mailbox.lock().ready.take();
        if let Some(parts) = ready {
            self.budget.return_unsubmitted_frame(parts.frame)?;
            drop((parts.command_slot, parts.wire_slot));
        }
        Ok(())
    }

    fn reserve_data_frame(&mut self) -> Result<FrameReservation, OutputCloseReason> {
        let wire_slot = if let Some(slot) = self.acquired_wire_slot.take() {
            slot
        } else {
            match self.wire.clone().try_reserve_owned() {
                Ok(slot) => slot,
                Err(mpsc::error::TrySendError::Full(_)) => {
                    self.blocked(OutputBlockReason::WireQueue)?;
                    return Ok(FrameReservation::Wait(WaitFor::Wire(Box::pin(
                        self.wire.clone().reserve_owned(),
                    ))));
                }
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    return Err(OutputCloseReason::Disconnected)
                }
            }
        };
        let frame = match self
            .budget
            .try_begin_frame()
            .map_err(|_| OutputCloseReason::InvalidOutput)?
        {
            OutputFrameAttempt::Reserved(frame) => frame,
            OutputFrameAttempt::Full(wake) => {
                drop(wire_slot);
                self.blocked(OutputBlockReason::EventCredit)?;
                return Ok(FrameReservation::Wait(WaitFor::Credit(wake)));
            }
        };
        self.blocked_since = None;
        Ok(FrameReservation::Reserved(frame, wire_slot))
    }

    fn prepare(&mut self) -> Result<WaitFor, OutputCloseReason> {
        if self.pending_terminal.is_some()
            || self.budget.unspent_wire_bytes().is_none()
            || matches!(self.shared.mailbox.lock().state, PortState::Closing(_))
            || self.generated_tokens == self.budget.plan().effective_max_tokens()
        {
            return Ok(WaitFor::None);
        }
        self.budget
            .reserve_future_event_window(self.wire.capacity().min(self.data_slots))
            .map_err(|_| OutputCloseReason::InvalidOutput)?;
        let (frame, wire_slot) = match self.reserve_data_frame()? {
            FrameReservation::Reserved(frame, slot) => (frame, slot),
            FrameReservation::Wait(wait) => return Ok(wait),
        };
        let command_slot = self
            .command_sender
            .clone()
            .try_reserve_owned()
            .map_err(|_| OutputCloseReason::InvalidOutput)?;
        let ordinal = self
            .applied
            .checked_add(1)
            .ok_or(OutputCloseReason::InvalidOutput)?;
        let future_capacity = self
            .budget
            .future_capacity_view(&frame, self.generated_tokens, self.wire.capacity())
            .map_err(|_| OutputCloseReason::InvalidOutput)?;
        let parts = GrantParts {
            ordinal,
            frame,
            command_slot,
            wire_slot,
            future_capacity,
        };
        let rejected = {
            let mut mailbox = self.shared.mailbox.lock();
            if matches!(mailbox.state, PortState::Closing(_)) {
                Some(parts)
            } else {
                mailbox.ready = Some(parts);
                mailbox.state = PortState::Ready;
                None
            }
        };
        if let Some(parts) = rejected {
            self.budget
                .return_unsubmitted_frame(parts.frame)
                .map_err(|_| OutputCloseReason::InvalidOutput)?;
            drop((parts.command_slot, parts.wire_slot));
            return Ok(WaitFor::None);
        }
        self.blocked_since = None;
        self.shared.signal();
        Ok(WaitFor::None)
    }

    async fn event(&mut self, wait: WaitFor) -> Event {
        if let Some(reason) = self.shared.cancel_reason() {
            return Event::Close(reason);
        }
        let shared = self.shared.clone();
        let deadline = match self.blocked_deadline() {
            Ok(Some(deadline)) => self.options.timer.sleep_until(deadline),
            Ok(None) => Box::pin(futures::future::pending()) as BoxFuture<'static, ()>,
            Err(reason) => return Event::Close(reason),
        };
        let terminal = &mut self.terminal;
        tokio::select! {
            biased;
            _ = shared.owner_wake.notified() => Event::Close(shared.cancel_reason().unwrap_or(OutputCloseReason::Cancelled)),
            _ = self.wire.closed() => Event::Close(OutputCloseReason::Disconnected),
            decision = async { match terminal { Some(receiver) => receiver.await, None => futures::future::pending().await } } => {
                self.terminal = None;
                match decision { Ok(decision) => Event::Terminal(decision), Err(_) => Event::Close(OutputCloseReason::Cancelled) }
            },
            command = self.commands.recv() => match command { Some(command) => Event::Command(command), None => Event::Close(OutputCloseReason::Abandoned) },
            result = async move {
                match wait {
                    WaitFor::None => futures::future::pending().await,
                    WaitFor::Credit(mut wake) => wake.changed().await.map(|_| None).map_err(|_| OutputCloseReason::InvalidOutput),
                    WaitFor::Wire(wait) => wait.await.map(Some).map_err(|_| OutputCloseReason::Disconnected),
                }
            } => match result { Ok(slot) => Event::Capacity(slot), Err(reason) => Event::Close(reason) },
            // A timer wakes a capacity reprobe. Actual available slots win;
            // unrelated wakes cannot reset a still-blocked original deadline.
            _ = deadline => Event::Capacity(None),
        }
    }

    fn command(&mut self, command: OwnerCommand) -> Result<(), OutputCloseReason> {
        if self.pending_terminal.is_none() {
            // A terminal can be sent before this command is received. Keep
            // the port closed even while that control message is in flight.
            let mut mailbox = self.shared.mailbox.lock();
            if !matches!(mailbox.state, PortState::Closing(_)) {
                mailbox.state = PortState::ProjectionBusy;
            }
            drop(mailbox);
            self.shared.signal();
        }
        match command {
            OwnerCommand::Unsubmitted { frame, wire_slot } => {
                drop(wire_slot);
                self.budget
                    .return_unsubmitted_frame(frame)
                    .map_err(|_| OutputCloseReason::InvalidOutput)
            }
            OwnerCommand::Abandoned { frame, wire_slot } => {
                drop((frame, wire_slot));
                Err(OutputCloseReason::Abandoned)
            }
            OwnerCommand::Committed {
                ordinal,
                frame,
                wire_slot,
                delta,
            } => {
                if self.applied.checked_add(1) != Some(ordinal)
                    || delta.text.capacity() > self.budget.plan().max_decoded_bytes()
                    || delta.generated_tokens > self.budget.plan().effective_max_tokens()
                    || match delta.token {
                        Some(_) => {
                            self.generated_tokens.checked_add(1) != Some(delta.generated_tokens)
                        }
                        None => self.generated_tokens != delta.generated_tokens,
                    }
                {
                    return Err(OutputCloseReason::InvalidOutput);
                }
                if self.chat_projection.is_some() {
                    return self.chat_command(ordinal, frame, wire_slot, delta);
                }
                if delta.text.is_empty() {
                    self.budget
                        .finish_without_wire_frame(frame)
                        .map_err(|_| OutputCloseReason::InvalidOutput)?;
                    drop(wire_slot);
                } else {
                    let wire = self
                        .budget
                        .encode_data_frame(frame, &delta.text, delta.created)
                        .map_err(|_| OutputCloseReason::InvalidOutput)?;
                    let metadata = OutputFrameMetadata {
                        ordinal,
                        token: delta.token,
                        generated_tokens: delta.generated_tokens,
                        terminal: false,
                    };
                    let _ = wire_slot.send(CreditedOutputFrame::new(wire, metadata));
                }
                self.applied = ordinal;
                self.generated_tokens = delta.generated_tokens;
                Ok(())
            }
        }
    }

    fn validate_terminal(&self, decision: &OutputTerminalDecision) -> bool {
        if decision.final_text.capacity() > self.budget.plan().max_decoded_bytes()
            || matches!(decision.outcome, OutputCompletion::Failed(_))
                && !decision.final_text.is_empty()
            || decision.through_output_ordinal < self.applied
            || self
                .applied
                .checked_add(1)
                .is_none_or(|next| decision.through_output_ordinal > next)
        {
            return false;
        }
        if let OutputCompletion::Succeeded {
            history,
            usage,
            execution_evidence,
            ..
        } = &decision.outcome
        {
            if self
                .budget
                .plan()
                .evidence_plan()
                .validate(execution_evidence.as_ref(), usage)
                .is_err()
            {
                return false;
            }
            if usage.completion_tokens != self.generated_tokens
                && decision.through_output_ordinal == self.applied
            {
                return false;
            }
            if let Some(history) = history {
                if history.text.capacity() > self.budget.plan().max_decoded_bytes()
                    || history.tokens.capacity() > self.budget.plan().effective_max_tokens()
                    || history.tokens.len() != usage.completion_tokens
                {
                    return false;
                }
            }
        }
        true
    }

    async fn flush_final_text(
        &mut self,
        text: &str,
        created: u64,
    ) -> Result<(), OutputCloseReason> {
        if text.is_empty() {
            return Ok(());
        }
        if text.len() > self.budget.plan().max_decoded_bytes() {
            return Err(OutputCloseReason::InvalidOutput);
        }
        let shared = self.shared.clone();
        let wire_slot = loop {
            if let Some(reason) = self.shared.cancel_reason() {
                return Err(reason);
            }
            match self.wire.clone().try_reserve_owned() {
                Ok(slot) => break slot,
                Err(mpsc::error::TrySendError::Closed(_)) => {
                    return Err(OutputCloseReason::Disconnected)
                }
                Err(mpsc::error::TrySendError::Full(_)) => {
                    self.blocked(OutputBlockReason::WireQueue)?;
                    let deadline = self
                        .options
                        .timer
                        .sleep_until(self.blocked_deadline()?.expect("blocked"));
                    tokio::select! {
                        biased;
                        _ = shared.owner_wake.notified() => return Err(shared.cancel_reason().unwrap_or(OutputCloseReason::Cancelled)),
                        slot = self.wire.clone().reserve_owned() => break slot.map_err(|_| OutputCloseReason::Disconnected)?,
                        _ = deadline => {}
                    }
                }
            }
        };
        let frame = loop {
            if let Some(reason) = self.shared.cancel_reason() {
                return Err(reason);
            }
            match self
                .budget
                .try_begin_frame()
                .map_err(|_| OutputCloseReason::InvalidOutput)?
            {
                OutputFrameAttempt::Reserved(frame) => break frame,
                OutputFrameAttempt::Full(mut wake) => {
                    self.blocked(OutputBlockReason::EventCredit)?;
                    let deadline = self
                        .options
                        .timer
                        .sleep_until(self.blocked_deadline()?.expect("blocked"));
                    tokio::select! {
                        biased;
                        _ = shared.owner_wake.notified() => return Err(shared.cancel_reason().unwrap_or(OutputCloseReason::Cancelled)),
                        _ = self.wire.closed() => return Err(OutputCloseReason::Disconnected),
                        result = wake.changed() => { result.map_err(|_| OutputCloseReason::InvalidOutput)?; }
                        _ = deadline => {}
                    }
                }
            }
        };
        self.blocked_since = None;
        let wire = self
            .budget
            .encode_data_frame(frame, text, created)
            .map_err(|_| OutputCloseReason::InvalidOutput)?;
        let _ = wire_slot.send(CreditedOutputFrame::new(
            wire,
            OutputFrameMetadata {
                ordinal: self.applied,
                token: None,
                generated_tokens: self.generated_tokens,
                terminal: false,
            },
        ));
        Ok(())
    }

    async fn finish(mut self, mut decision: OutputTerminalDecision, mut reason: OutputCloseReason) {
        if self.revoke_ready().is_err() {
            reason = OutputCloseReason::InvalidOutput;
        }
        self.shared.state(PortState::Closing(reason));
        drop(self.acquired_wire_slot.take());
        if reason == OutputCloseReason::Complete {
            let flushed = if self.chat_projection.is_some() {
                self.flush_chat_terminal(&decision.final_text, decision.created)
                    .await
            } else {
                self.flush_final_text(&decision.final_text, decision.created)
                    .await
            };
            if let Err(failure) = flushed {
                reason = failure;
            }
        }
        if reason != OutputCloseReason::Complete {
            decision = self.failure_decision(reason);
            self.shared.state(PortState::Closing(reason));
        }
        // The flush payload must stop residing before transferring the grant.
        drop(std::mem::take(&mut decision.final_text));
        // No actor-owned projection may survive transfer of its grant to a
        // completion receiver that can immediately drop it.
        drop(self.pending_projection.take());
        drop(self.chat_projection.take());
        let terminal = match &decision.outcome {
            OutputCompletion::Succeeded { reason, usage, .. } => OutputTerminal::Success {
                reason: *reason,
                usage,
                created: decision.created,
            },
            OutputCompletion::Failed(error) => OutputTerminal::Error(error),
        };
        let mut encoded = self.budget.encode_terminal(terminal);
        if encoded.is_err() {
            decision = self.failure_decision(OutputCloseReason::InvalidOutput);
            self.shared
                .state(PortState::Closing(OutputCloseReason::InvalidOutput));
            if let OutputCompletion::Failed(error) = &decision.outcome {
                encoded = self.budget.encode_terminal(OutputTerminal::Error(error));
            }
        }
        if let Ok(wire) = encoded {
            let metadata = OutputFrameMetadata {
                ordinal: self.applied,
                token: None,
                generated_tokens: self.generated_tokens,
                terminal: true,
            };
            if let Some(slot) = self.terminal_slot.take() {
                let _ = slot.send(CreditedOutputFrame::new(wire, metadata));
            }
        }
        self.commands.close();
        // Consumer cancellation can precede the engine removing SequenceState.
        // Keep projection charged until its unique port is destroyed. A normal
        // terminal also waits for that explicit engine ownership handoff.
        let _ = (&mut self.engine_lifetime).await;
        while let Ok(command) = self.commands.try_recv() {
            drop(command);
        }
        drop(self.pending_terminal.take());
        // A concurrent engine cleanup can have handed history to the terminal
        // mailbox after failure began. Free that payload before its grant can
        // leave this actor (and potentially be dropped by the consumer).
        drop(self.terminal.take());
        if let Ok(completion) = self.budget.into_retained_projection(decision.outcome) {
            if let Some(sender) = self.completion.take() {
                let _ = sender.send(completion);
            }
        }
    }

    fn failure_decision(&self, reason: OutputCloseReason) -> OutputTerminalDecision {
        let message = match reason {
            OutputCloseReason::Disconnected => "output consumer disconnected",
            OutputCloseReason::Abandoned => "output grant was abandoned",
            OutputCloseReason::InvalidOutput => "output contract or command frontier was violated",
            OutputCloseReason::SlowConsumer => {
                "output consumer exceeded the configured blocking timeout"
            }
            _ => "output cancelled",
        };
        OutputTerminalDecision {
            through_output_ordinal: self.applied,
            outcome: OutputCompletion::Failed(BoundedOutputError::new(message)),
            final_text: String::new(),
            created: 0,
        }
    }

    async fn fail(self, reason: OutputCloseReason) {
        let decision = self.failure_decision(reason);
        self.finish(decision, reason).await;
    }

    async fn run(mut self) {
        if self.options.slow_consumer_timeout.is_zero() {
            self.fail(OutputCloseReason::InvalidOutput).await;
            return;
        }
        if let Err(reason) = self.initialize_chat_projection() {
            self.fail(reason).await;
            return;
        }
        loop {
            if let Some(reason) = self.shared.cancel_reason() {
                self.fail(reason).await;
                return;
            }
            if let Some(decision) = self.pending_terminal.as_ref() {
                if !self.validate_terminal(decision) {
                    self.fail(OutputCloseReason::InvalidOutput).await;
                    return;
                }
                if self.pending_projection.is_none() && self.budget.unspent_wire_bytes().is_some() {
                    if decision.through_output_ordinal != self.applied {
                        self.fail(OutputCloseReason::InvalidOutput).await;
                        return;
                    }
                    let decision = self.pending_terminal.take().unwrap();
                    self.finish(decision, OutputCloseReason::Complete).await;
                    return;
                }
            }
            let prepared = if self.pending_projection.is_some() {
                match self.project_next_chat_frame() {
                    Ok(None) => continue,
                    Ok(Some(wait)) => Ok(wait),
                    Err(reason) => Err(reason),
                }
            } else {
                self.prepare()
            };
            let wait = match prepared {
                Ok(wait) => wait,
                Err(reason) => {
                    self.fail(reason).await;
                    return;
                }
            };
            match self.event(wait).await {
                Event::Close(reason) => {
                    self.fail(reason).await;
                    return;
                }
                Event::Capacity(slot) => self.acquired_wire_slot = slot,
                Event::Command(command) => {
                    if let Err(reason) = self.command(command) {
                        self.fail(reason).await;
                        return;
                    }
                }
                Event::Terminal(decision) => {
                    self.shared
                        .state(PortState::Closing(OutputCloseReason::Complete));
                    if self.revoke_ready().is_err() {
                        self.fail(OutputCloseReason::InvalidOutput).await;
                        return;
                    }
                    self.pending_terminal = Some(decision);
                }
            }
        }
    }
}
