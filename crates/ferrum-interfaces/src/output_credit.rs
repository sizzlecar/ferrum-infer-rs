//! Move-only output ownership with simultaneous request/global capacity checks.
//!
//! `bytes` charges queued payload/transport storage; `projection_bytes` charges
//! retained parsing/assembly storage. Copies that coexist must both be charged.
//! Their sum is also subject to the global total-byte limit. These are payload
//! budgets, not an allocator-overhead or remote-client-acknowledgement metric.
//!
//! Opening an account escrows its entire terminal allowance globally. Data
//! cannot borrow that allowance. Closing returns only unused escrow; in-flight
//! reservations stay charged until their last (move-only) owner releases them.
//! A closed account also retains its admission slot until all ownership drains.
//! Data allowances are ceilings, not escrow: arbitrary heterogeneous pool/account
//! limits can share data capacity. `from_slo` validates the aggregate homogeneous
//! byte budget so all retained accounts using that configuration can fit together.
//! No API waits for capacity while holding the ledger lock.

use ferrum_types::{RequestId, SloOutputConfig};
use std::{
    collections::HashMap,
    num::NonZeroUsize,
    sync::{Arc, Mutex, MutexGuard},
    time::{Duration, Instant},
};
use tokio::sync::watch;

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct OutputCreditAmount {
    pub events: usize,
    pub bytes: usize,
    pub projection_bytes: usize,
}

impl OutputCreditAmount {
    pub const ZERO: Self = Self {
        events: 0,
        bytes: 0,
        projection_bytes: 0,
    };

    pub fn checked_add(self, other: Self) -> Option<Self> {
        Some(Self {
            events: self.events.checked_add(other.events)?,
            bytes: self.bytes.checked_add(other.bytes)?,
            projection_bytes: self.projection_bytes.checked_add(other.projection_bytes)?,
        })
    }

    pub fn checked_sub(self, other: Self) -> Option<Self> {
        Some(Self {
            events: self.events.checked_sub(other.events)?,
            bytes: self.bytes.checked_sub(other.bytes)?,
            projection_bytes: self.projection_bytes.checked_sub(other.projection_bytes)?,
        })
    }

    pub fn fits(self, limit: Self) -> bool {
        self.events <= limit.events
            && self.bytes <= limit.bytes
            && self.projection_bytes <= limit.projection_bytes
    }

    pub fn total_bytes(self) -> Option<usize> {
        self.bytes.checked_add(self.projection_bytes)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputCreditLane {
    Data,
    Terminal,
}

/// `maximum` includes `terminal`. Terminal event count must be derived from the
/// selected protocol, rather than assuming every endpoint has one final event.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputAccountLimits {
    pub maximum: OutputCreditAmount,
    pub terminal: OutputCreditAmount,
}

impl OutputAccountLimits {
    pub fn from_slo(
        config: &SloOutputConfig,
        terminal_events: NonZeroUsize,
    ) -> Result<Self, OutputCreditError> {
        config
            .validate()
            .map_err(OutputCreditError::InvalidConfiguration)?;
        let limits = Self {
            maximum: OutputCreditAmount {
                events: config.max_queued_events_per_request.get(),
                bytes: config.max_queued_bytes_per_request.get(),
                projection_bytes: config.max_projection_bytes_per_request.get(),
            },
            terminal: OutputCreditAmount {
                events: terminal_events.get(),
                bytes: config.terminal_reserve_bytes_per_request.get(),
                projection_bytes: 0,
            },
        };
        limits.validate()?;
        Ok(limits)
    }

    fn validate(self) -> Result<(), OutputCreditError> {
        if self.terminal.events == 0
            || self.terminal.bytes == 0
            || self.maximum.events <= self.terminal.events
            || self.maximum.bytes <= self.terminal.bytes
            || !self.terminal.fits(self.maximum)
            || self.maximum.total_bytes().is_none()
        {
            return Err(OutputCreditError::InvalidConfiguration(
                "output limits require terminal event/byte escrow and positive data event/byte capacity".to_owned(),
            ));
        }
        Ok(())
    }

    fn data(self) -> OutputCreditAmount {
        self.maximum
            .checked_sub(self.terminal)
            .expect("validated account limits")
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OutputPoolLimits {
    pub maximum: OutputCreditAmount,
    pub max_total_bytes: usize,
    /// Limits all retained account generations, including closed accounts with
    /// outstanding ownership. A slot returns only after the final lease drains.
    pub max_open_accounts: usize,
}

impl OutputPoolLimits {
    pub fn from_slo(
        config: &SloOutputConfig,
        max_open_accounts: NonZeroUsize,
    ) -> Result<Self, OutputCreditError> {
        config
            .validate()
            .map_err(OutputCreditError::InvalidConfiguration)?;
        let aggregate_bytes = config
            .max_queued_bytes_per_request
            .get()
            .checked_add(config.max_projection_bytes_per_request.get())
            .and_then(|bytes| bytes.checked_mul(max_open_accounts.get()))
            .ok_or(OutputCreditError::Overflow)?;
        if aggregate_bytes > config.max_total_buffer_bytes.get() {
            return Err(OutputCreditError::InvalidConfiguration(
                "global output bytes cannot cover every retained account's configured share"
                    .to_owned(),
            ));
        }
        Ok(Self {
            maximum: OutputCreditAmount {
                events: config
                    .max_queued_events_per_request
                    .get()
                    .checked_mul(max_open_accounts.get())
                    .ok_or(OutputCreditError::Overflow)?,
                bytes: config.max_total_buffer_bytes.get(),
                projection_bytes: config.max_total_buffer_bytes.get(),
            },
            max_total_bytes: config.max_total_buffer_bytes.get(),
            max_open_accounts: max_open_accounts.get(),
        })
    }

    fn fits(self, amount: OutputCreditAmount) -> bool {
        amount.fits(self.maximum)
            && amount
                .total_bytes()
                .is_some_and(|bytes| bytes <= self.max_total_bytes)
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum OutputCreditError {
    #[error("invalid output credit configuration: {0}")]
    InvalidConfiguration(String),
    #[error("output credit arithmetic overflow")]
    Overflow,
    #[error("output credit amount must be nonzero")]
    EmptyAmount,
    #[error("output credit request is already open")]
    AlreadyOpen,
    #[error("output credit account is closed")]
    Closed,
    #[error("output account admission has insufficient capacity")]
    AdmissionFull,
    #[error("requested output credit exceeds its lane's configured maximum")]
    ExceedsLimit,
    #[error("output reservation operation exceeds its owned credit")]
    ExceedsReservation,
    #[error("output reservations belong to different pools, requests, generations or lanes")]
    ForeignReservation,
    #[error("output credit wake source has closed")]
    WakeClosed,
}

/// Injectable monotonic observation source; timeout/cancellation policy belongs
/// to the consumer/controller, not the capacity ledger. `now` runs outside locks.
pub trait OutputCreditClock: Send + Sync {
    fn now(&self) -> Instant;
}

struct SystemOutputClock;
impl OutputCreditClock for SystemOutputClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}

#[derive(Debug, Clone)]
pub struct OutputPoolSnapshot {
    pub limits: OutputPoolLimits,
    pub data_used: OutputCreditAmount,
    /// Includes unused terminal allowance for open accounts and used allowance
    /// for closed accounts. It is not an additional charge on terminal_used.
    pub terminal_held: OutputCreditAmount,
    pub terminal_used: OutputCreditAmount,
    pub open_accounts: usize,
    pub retained_accounts: usize,
    pub revision: u64,
}

#[derive(Debug, Clone)]
pub struct OutputAccountSnapshot {
    pub request_id: RequestId,
    pub generation: u64,
    pub limits: OutputAccountLimits,
    pub closed: bool,
    pub data_used: OutputCreditAmount,
    pub terminal_used: OutputCreditAmount,
    pub terminal_held: OutputCreditAmount,
    /// First failed Data attempt since its last successful Data reservation.
    /// This is advisory observation: the controller owns the pending output
    /// obligation and decides whether/how to apply a slow-consumer timeout.
    pub blocked_since: Option<Instant>,
    pub terminal_blocked_since: Option<Instant>,
    pub last_release_at: Instant,
    pub observed_at: Instant,
}

impl OutputAccountSnapshot {
    pub fn blocked_for(&self) -> Option<Duration> {
        self.blocked_since
            .map(|since| self.observed_at.saturating_duration_since(since))
    }
}

struct AccountState {
    request_id: RequestId,
    limits: OutputAccountLimits,
    closed: bool,
    data_used: OutputCreditAmount,
    terminal_used: OutputCreditAmount,
    terminal_held: OutputCreditAmount,
    blocked_since: Option<Instant>,
    terminal_blocked_since: Option<Instant>,
    last_release_at: Instant,
}

struct Ledger {
    next_generation: u64,
    revision: u64,
    data_used: OutputCreditAmount,
    terminal_held: OutputCreditAmount,
    terminal_used: OutputCreditAmount,
    accounts: HashMap<u64, AccountState>,
    active: HashMap<RequestId, u64>,
}

struct PoolInner {
    limits: OutputPoolLimits,
    ledger: Mutex<Ledger>,
    wake: watch::Sender<u64>,
    clock: Arc<dyn OutputCreditClock>,
}

impl PoolInner {
    fn lock(&self) -> MutexGuard<'_, Ledger> {
        self.ledger
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn changed(&self, ledger: &mut Ledger) {
        ledger.revision = ledger.revision.wrapping_add(1);
        // watch's own change version, rather than equality of this diagnostic
        // revision, drives wakes, including diagnostic counter wraparound.
        self.wake.send_replace(ledger.revision);
    }

    fn close(&self, generation: u64) {
        let mut ledger = self.lock();
        let Some(account) = ledger.accounts.get_mut(&generation) else {
            return;
        };
        if account.closed {
            return;
        }
        account.closed = true;
        let unused_terminal = account
            .terminal_held
            .checked_sub(account.terminal_used)
            .expect("terminal escrow invariant");
        account.terminal_held = account.terminal_used;
        let request_id = account.request_id.clone();
        ledger.terminal_held = ledger
            .terminal_held
            .checked_sub(unused_terminal)
            .expect("global terminal escrow invariant");
        ledger.active.remove(&request_id);
        Self::remove_drained(&mut ledger, generation);
        self.changed(&mut ledger);
    }

    fn remove_drained(ledger: &mut Ledger, generation: u64) {
        if ledger.accounts.get(&generation).is_some_and(|account| {
            account.closed
                && account.data_used == OutputCreditAmount::ZERO
                && account.terminal_used == OutputCreditAmount::ZERO
        }) {
            ledger.accounts.remove(&generation);
        }
    }

    fn release(&self, generation: u64, lane: OutputCreditLane, amount: OutputCreditAmount) {
        if amount == OutputCreditAmount::ZERO {
            return;
        }
        let now = self.clock.now();
        let mut ledger = self.lock();
        let account = ledger
            .accounts
            .get_mut(&generation)
            .expect("outstanding reservation retains its account");
        account.last_release_at = account.last_release_at.max(now);
        let closed = account.closed;
        match lane {
            OutputCreditLane::Data => {
                account.data_used = account
                    .data_used
                    .checked_sub(amount)
                    .expect("account data ownership invariant");
                ledger.data_used = ledger
                    .data_used
                    .checked_sub(amount)
                    .expect("global data ownership invariant");
            }
            OutputCreditLane::Terminal => {
                account.terminal_used = account
                    .terminal_used
                    .checked_sub(amount)
                    .expect("account terminal ownership invariant");
                if closed {
                    account.terminal_held = account
                        .terminal_held
                        .checked_sub(amount)
                        .expect("closed terminal escrow invariant");
                    ledger.terminal_held = ledger
                        .terminal_held
                        .checked_sub(amount)
                        .expect("global closed terminal escrow invariant");
                }
                ledger.terminal_used = ledger
                    .terminal_used
                    .checked_sub(amount)
                    .expect("global terminal ownership invariant");
            }
        }
        Self::remove_drained(&mut ledger, generation);
        self.changed(&mut ledger);
    }
}

#[derive(Clone)]
pub struct OutputCreditPool {
    inner: Arc<PoolInner>,
}

impl OutputCreditPool {
    pub fn new(limits: OutputPoolLimits) -> Result<Self, OutputCreditError> {
        Self::with_clock(limits, Arc::new(SystemOutputClock))
    }

    pub fn with_clock(
        limits: OutputPoolLimits,
        clock: Arc<dyn OutputCreditClock>,
    ) -> Result<Self, OutputCreditError> {
        if limits.maximum.events == 0
            || limits.maximum.bytes == 0
            || limits.max_total_bytes == 0
            || limits.max_open_accounts == 0
        {
            return Err(OutputCreditError::InvalidConfiguration(
                "global output event/byte/account limits must be positive".to_owned(),
            ));
        }
        let (wake, _) = watch::channel(0);
        Ok(Self {
            inner: Arc::new(PoolInner {
                limits,
                clock,
                wake,
                ledger: Mutex::new(Ledger {
                    next_generation: 1,
                    revision: 0,
                    data_used: OutputCreditAmount::ZERO,
                    terminal_held: OutputCreditAmount::ZERO,
                    terminal_used: OutputCreditAmount::ZERO,
                    accounts: HashMap::new(),
                    active: HashMap::new(),
                }),
            }),
        })
    }

    pub fn open_request(
        &self,
        request_id: RequestId,
        limits: OutputAccountLimits,
    ) -> Result<OutputCreditAccount, OutputCreditError> {
        limits.validate()?;
        if !self.inner.limits.fits(limits.maximum) {
            return Err(OutputCreditError::ExceedsLimit);
        }
        let now = self.inner.clock.now();
        let mut ledger = self.inner.lock();
        if ledger.active.contains_key(&request_id) {
            return Err(OutputCreditError::AlreadyOpen);
        }
        let held = ledger
            .terminal_held
            .checked_add(limits.terminal)
            .ok_or(OutputCreditError::Overflow)?;
        let total = ledger
            .data_used
            .checked_add(held)
            .ok_or(OutputCreditError::Overflow)?;
        if ledger.accounts.len() >= self.inner.limits.max_open_accounts
            || !self.inner.limits.fits(total)
        {
            return Err(OutputCreditError::AdmissionFull);
        }
        let generation = ledger.next_generation;
        ledger.next_generation = generation
            .checked_add(1)
            .ok_or(OutputCreditError::Overflow)?;
        ledger.terminal_held = held;
        ledger.active.insert(request_id.clone(), generation);
        ledger.accounts.insert(
            generation,
            AccountState {
                request_id: request_id.clone(),
                limits,
                closed: false,
                data_used: OutputCreditAmount::ZERO,
                terminal_used: OutputCreditAmount::ZERO,
                terminal_held: limits.terminal,
                blocked_since: None,
                terminal_blocked_since: None,
                last_release_at: now,
            },
        );
        Ok(OutputCreditAccount {
            owner: Arc::new(AccountOwner {
                pool: self.inner.clone(),
                request_id,
                generation,
                limits,
                opened_at: now,
            }),
        })
    }

    pub fn snapshot(&self) -> OutputPoolSnapshot {
        let ledger = self.inner.lock();
        OutputPoolSnapshot {
            limits: self.inner.limits,
            data_used: ledger.data_used,
            terminal_held: ledger.terminal_held,
            terminal_used: ledger.terminal_used,
            open_accounts: ledger.active.len(),
            retained_accounts: ledger.accounts.len(),
            revision: ledger.revision,
        }
    }

    /// Subscribe before checking admission; retry after `changed()` to avoid
    /// losing a final lease release between an admission failure and waiting.
    pub fn subscribe(&self) -> OutputCreditWake {
        OutputCreditWake {
            receiver: self.inner.wake.subscribe(),
        }
    }
}

struct AccountOwner {
    pool: Arc<PoolInner>,
    request_id: RequestId,
    generation: u64,
    limits: OutputAccountLimits,
    opened_at: Instant,
}
impl Drop for AccountOwner {
    fn drop(&mut self) {
        self.pool.close(self.generation);
    }
}

/// Cloning a control handle does not clone a reservation. Dropping the final
/// control handle closes the account while payload reservations remain charged.
#[derive(Clone)]
pub struct OutputCreditAccount {
    owner: Arc<AccountOwner>,
}

impl OutputCreditAccount {
    pub fn request_id(&self) -> &RequestId {
        &self.owner.request_id
    }
    pub fn generation(&self) -> u64 {
        self.owner.generation
    }
    pub fn close(&self) {
        self.owner.pool.close(self.owner.generation);
    }

    pub fn snapshot(&self) -> OutputAccountSnapshot {
        let now = self.owner.pool.clock.now();
        let ledger = self.owner.pool.lock();
        let account = ledger.accounts.get(&self.owner.generation);
        OutputAccountSnapshot {
            request_id: self.owner.request_id.clone(),
            generation: self.owner.generation,
            limits: self.owner.limits,
            closed: account.is_none_or(|account| account.closed),
            data_used: account.map_or(OutputCreditAmount::ZERO, |account| account.data_used),
            terminal_used: account
                .map_or(OutputCreditAmount::ZERO, |account| account.terminal_used),
            terminal_held: account
                .map_or(OutputCreditAmount::ZERO, |account| account.terminal_held),
            blocked_since: account.and_then(|account| account.blocked_since),
            terminal_blocked_since: account.and_then(|account| account.terminal_blocked_since),
            last_release_at: account
                .map_or(self.owner.opened_at, |account| account.last_release_at),
            observed_at: now,
        }
    }

    pub fn try_reserve(
        &self,
        lane: OutputCreditLane,
        amount: OutputCreditAmount,
    ) -> Result<OutputCreditAttempt, OutputCreditError> {
        if amount == OutputCreditAmount::ZERO {
            return Err(OutputCreditError::EmptyAmount);
        }
        if amount.total_bytes().is_none() {
            return Err(OutputCreditError::Overflow);
        }
        let now = self.owner.pool.clock.now();
        let mut ledger = self.owner.pool.lock();
        let account = ledger
            .accounts
            .get(&self.owner.generation)
            .ok_or(OutputCreditError::Closed)?;
        if account.closed {
            return Err(OutputCreditError::Closed);
        }
        let (used, limit) = match lane {
            OutputCreditLane::Data => (account.data_used, account.limits.data()),
            OutputCreditLane::Terminal => (account.terminal_used, account.limits.terminal),
        };
        if !amount.fits(limit) {
            return Err(OutputCreditError::ExceedsLimit);
        }
        let new_used = used
            .checked_add(amount)
            .ok_or(OutputCreditError::Overflow)?;
        let new_global_data = if lane == OutputCreditLane::Data {
            ledger
                .data_used
                .checked_add(amount)
                .ok_or(OutputCreditError::Overflow)?
        } else {
            ledger.data_used
        };
        let global = new_global_data
            .checked_add(ledger.terminal_held)
            .ok_or(OutputCreditError::Overflow)?;
        if !new_used.fits(limit) || !self.owner.pool.limits.fits(global) {
            let account = ledger
                .accounts
                .get_mut(&self.owner.generation)
                .expect("account checked under lock");
            match lane {
                OutputCreditLane::Data => &mut account.blocked_since,
                OutputCreditLane::Terminal => &mut account.terminal_blocked_since,
            }
            .get_or_insert(now);
            // Subscribe under the same lock as the failed capacity check. A
            // release between this return and changed().await remains visible.
            return Ok(OutputCreditAttempt::Full(OutputCreditWake {
                receiver: self.owner.pool.wake.subscribe(),
            }));
        }
        let account = ledger
            .accounts
            .get_mut(&self.owner.generation)
            .expect("account checked under lock");
        match lane {
            OutputCreditLane::Data => account.blocked_since = None,
            OutputCreditLane::Terminal => account.terminal_blocked_since = None,
        }
        match lane {
            OutputCreditLane::Data => {
                account.data_used = new_used;
                ledger.data_used = new_global_data;
            }
            OutputCreditLane::Terminal => {
                account.terminal_used = new_used;
                ledger.terminal_used = ledger
                    .terminal_used
                    .checked_add(amount)
                    .expect("used terminal credit fits existing global escrow");
            }
        }
        Ok(OutputCreditAttempt::Reserved(OutputReservation {
            pool: self.owner.pool.clone(),
            request_id: self.owner.request_id.clone(),
            generation: self.owner.generation,
            lane,
            amount,
        }))
    }
}

pub enum OutputCreditAttempt {
    Reserved(OutputReservation),
    Full(OutputCreditWake),
}

/// Capacity notification only, never a permit. Wakes may be caused by another
/// account; the caller must recheck with try_reserve before scheduling a wave.
pub struct OutputCreditWake {
    receiver: watch::Receiver<u64>,
}
impl OutputCreditWake {
    pub async fn changed(&mut self) -> Result<u64, OutputCreditError> {
        self.receiver
            .changed()
            .await
            .map_err(|_| OutputCreditError::WakeClosed)?;
        Ok(*self.receiver.borrow_and_update())
    }
}

/// Not Clone. Move into the next bounded stage; split/merge only redistribute
/// already charged ownership and never acquire additional capacity.
pub struct OutputReservation {
    pool: Arc<PoolInner>,
    request_id: RequestId,
    generation: u64,
    lane: OutputCreditLane,
    amount: OutputCreditAmount,
}

impl OutputReservation {
    pub(crate) fn same_account(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.pool, &other.pool) && self.generation == other.generation
    }

    pub fn request_id(&self) -> &RequestId {
        &self.request_id
    }

    pub fn amount(&self) -> OutputCreditAmount {
        self.amount
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn lane(&self) -> OutputCreditLane {
        self.lane
    }

    pub fn split(&mut self, amount: OutputCreditAmount) -> Result<Self, OutputCreditError> {
        if amount == OutputCreditAmount::ZERO {
            return Err(OutputCreditError::EmptyAmount);
        }
        let remaining = self
            .amount
            .checked_sub(amount)
            .ok_or(OutputCreditError::ExceedsReservation)?;
        self.amount = remaining;
        Ok(Self {
            pool: self.pool.clone(),
            request_id: self.request_id.clone(),
            generation: self.generation,
            lane: self.lane,
            amount,
        })
    }

    pub fn shrink_to(&mut self, amount: OutputCreditAmount) -> Result<(), OutputCreditError> {
        let released = self
            .amount
            .checked_sub(amount)
            .ok_or(OutputCreditError::ExceedsReservation)?;
        self.amount = amount;
        self.pool.release(self.generation, self.lane, released);
        Ok(())
    }

    /// On error both arguments retain their complete original ownership. On
    /// success `other` is empty and may be safely dropped.
    pub fn try_merge(&mut self, other: &mut Self) -> Result<(), OutputCreditError> {
        if !Arc::ptr_eq(&self.pool, &other.pool)
            || self.generation != other.generation
            || self.lane != other.lane
        {
            return Err(OutputCreditError::ForeignReservation);
        }
        let combined = self
            .amount
            .checked_add(other.amount)
            .ok_or(OutputCreditError::Overflow)?;
        self.amount = combined;
        other.amount = OutputCreditAmount::ZERO;
        Ok(())
    }

    /// The producer must reserve a proven upper bound before materializing this
    /// payload, including simultaneous copies and any protocol expansion.
    pub fn into_output<T>(self, payload: T) -> LeasedOutput<T> {
        LeasedOutput {
            payload,
            reservation: self,
        }
    }
}
impl Drop for OutputReservation {
    fn drop(&mut self) {
        self.pool.release(self.generation, self.lane, self.amount);
    }
}

/// Ownership travels with the payload through projection/transport. No Clone,
/// into_inner, or mutable payload accessor can detach or outgrow the charge.
/// Reading and copying payload data still requires the caller to reserve the
/// additional copy; this module does not police allocations outside its API.
pub struct LeasedOutput<T> {
    payload: T,
    reservation: OutputReservation,
}
impl<T> LeasedOutput<T> {
    pub fn request_id(&self) -> &RequestId {
        self.reservation.request_id()
    }

    pub fn generation(&self) -> u64 {
        self.reservation.generation()
    }

    pub fn payload(&self) -> &T {
        &self.payload
    }
    pub fn credit(&self) -> OutputCreditAmount {
        self.reservation.amount()
    }

    /// Preserve ownership while projecting. The output and temporary working
    /// storage must fit the already reserved budget; this is not a growth grant.
    pub fn map<U>(self, project: impl FnOnce(T) -> U) -> LeasedOutput<U> {
        LeasedOutput {
            payload: project(self.payload),
            reservation: self.reservation,
        }
    }
}
