use super::*;

mod capacity;
pub use capacity::PrepaidOutputCapacityView;

#[derive(Clone, Copy)]
enum DataPayload<'a> {
    Text(&'a str),
    Chat(ChatOutputDelta<'a>),
}

/// Owns the future wire reservoir and all retained projection storage. It is
/// intentionally not Clone. Dropping closes admission but outstanding outputs
/// keep their byte/event charge and the account's retained admission slot.
pub struct RequestOutputBudget {
    plan: RequestOutputPlan,
    account: OutputCreditAccount,
    wire: Option<OutputReservation>,
    projection: Option<OutputReservation>,
    terminal: Option<OutputReservation>,
    future_events: Option<OutputReservation>,
    emitted_text_bytes: usize,
    frames: usize,
    closed: bool,
}

pub enum OutputFrameAttempt {
    Reserved(OutputFramePermit),
    Full(OutputCreditWake),
}

/// An event slot and the entire unspent wire reservoir. This first version
/// permits one projection at a time. It need not predict the next delta size:
/// the unused bytes transfer back into the reservoir without being released.
/// Dropping instead of returning a permit abandons that request's reservoir;
/// it cannot manufacture credit for a subsequent frame.
pub struct OutputFramePermit {
    credit: OutputReservation,
}

impl OutputFramePermit {
    pub fn credit(&self) -> OutputCreditAmount {
        self.credit.amount()
    }
    pub fn request_id(&self) -> &RequestId {
        self.credit.request_id()
    }
    pub fn generation(&self) -> u64 {
        self.credit.generation()
    }
}

impl RequestOutputBudget {
    /// Open terminal escrow, then atomically reserve all data/projection
    /// dimensions before publishing this owner. Failure drops the provisional
    /// account. Admission callers subscribe to pool release before trying this.
    pub fn open(
        pool: &OutputCreditPool,
        limits: OutputAccountLimits,
        plan: RequestOutputPlan,
    ) -> Result<Self, OutputFlowError> {
        let available = limits
            .maximum
            .checked_sub(limits.terminal)
            .ok_or(OutputFlowError::BoundExceeded)?;
        let required = OutputCreditAmount {
            events: 0,
            bytes: plan.wire_bytes,
            projection_bytes: plan.projection_bytes,
        };
        if !plan.terminal.fits(limits.terminal)
            || !required.fits(available)
            || available.events == 0
        {
            return Err(OutputFlowError::BoundExceeded);
        }
        let account = pool
            .open_request(plan.request_id.clone(), limits)
            .map_err(|error| {
                if error == OutputCreditError::AdmissionFull {
                    OutputFlowError::AdmissionFull
                } else {
                    error.into()
                }
            })?;
        let mut reservoir = match account.try_reserve(OutputCreditLane::Data, required)? {
            OutputCreditAttempt::Reserved(reservation) => reservation,
            OutputCreditAttempt::Full(_) => return Err(OutputFlowError::AdmissionFull),
        };
        let projection = reservoir.split(OutputCreditAmount {
            events: 0,
            bytes: 0,
            projection_bytes: plan.projection_bytes,
        })?;
        let terminal = match account.try_reserve(OutputCreditLane::Terminal, plan.terminal)? {
            OutputCreditAttempt::Reserved(reservation) => reservation,
            OutputCreditAttempt::Full(_) => return Err(OutputFlowError::AdmissionFull),
        };
        Ok(Self {
            plan,
            account,
            wire: Some(reservoir),
            projection: Some(projection),
            terminal: Some(terminal),
            future_events: None,
            emitted_text_bytes: 0,
            frames: 0,
            closed: false,
        })
    }

    pub fn plan(&self) -> &RequestOutputPlan {
        &self.plan
    }
    pub fn account_snapshot(&self) -> OutputAccountSnapshot {
        self.account.snapshot()
    }
    pub fn unspent_wire_bytes(&self) -> Option<usize> {
        self.wire.as_ref().map(|wire| wire.amount().bytes)
    }

    /// Must succeed before submitting a wave that can produce a token. The
    /// event slot also bounds one pending raw/projection command for this
    /// request; transfer it to the wire frame rather than duplicate the slot.
    pub fn try_begin_frame(&mut self) -> Result<OutputFrameAttempt, OutputFlowError> {
        if self.closed {
            return Err(OutputFlowError::Closed);
        }
        if self.wire.is_none() {
            return Err(OutputFlowError::FrameInFlight);
        }
        if self.frames >= self.plan.data_frames {
            return Err(OutputFlowError::BoundExceeded);
        }
        let mut event = if let Some(escrow) = self
            .future_events
            .as_mut()
            .filter(|events| events.amount().events > 0)
        {
            escrow.split(OutputCreditAmount {
                events: 1,
                bytes: 0,
                projection_bytes: 0,
            })?
        } else {
            match self.account.try_reserve(
                OutputCreditLane::Data,
                OutputCreditAmount {
                    events: 1,
                    bytes: 0,
                    projection_bytes: 0,
                },
            )? {
                OutputCreditAttempt::Reserved(reservation) => reservation,
                OutputCreditAttempt::Full(wake) => return Ok(OutputFrameAttempt::Full(wake)),
            }
        };
        let mut wire = self.wire.take().expect("checked reservoir");
        wire.try_merge(&mut event)?;
        Ok(OutputFrameAttempt::Reserved(OutputFramePermit {
            credit: wire,
        }))
    }

    fn validate_permit(&mut self, permit: &mut OutputFramePermit) -> Result<(), OutputFlowError> {
        if self.closed {
            return Err(OutputFlowError::Closed);
        }
        if self.wire.is_some() {
            return Err(OutputFlowError::FrameInFlight);
        }
        if permit.credit.request_id() != self.account.request_id()
            || permit.credit.generation() != self.account.generation()
            || permit.credit.lane() != OutputCreditLane::Data
            || permit.credit.amount().events != 1
            || permit.credit.amount().projection_bytes != 0
        {
            return Err(OutputCreditError::ForeignReservation.into());
        }
        // Generation numbers are local to a pool. Merge/split with our existing
        // projection reservation proves the pool fence as well, transferring
        // exactly the same amount without acquiring or releasing any credit.
        let amount = permit.credit.amount();
        let projection = self.projection.as_mut().ok_or(OutputFlowError::Closed)?;
        projection.try_merge(&mut permit.credit)?;
        permit.credit = projection.split(amount)?;
        Ok(())
    }

    /// The text must come from the projection grant and represent a disjoint
    /// part of the monotonically emitted history. Codec/count and allocation
    /// share one implementation. No complete StreamChunk can be cloned to
    /// multiply ownership through this API.
    pub fn encode_data_frame(
        &mut self,
        permit: OutputFramePermit,
        text: &str,
        created: u64,
    ) -> Result<LeasedOutput<Vec<u8>>, OutputFlowError> {
        if self.plan.text_reasoning.is_some() {
            return Err(OutputFlowError::Unsupported(
                "Chat raw text must pass through its bounded semantic projection",
            ));
        }
        self.encode_projected_frame(permit, DataPayload::Text(text), created)
    }

    pub fn encode_chat_data_frame(
        &mut self,
        permit: OutputFramePermit,
        delta: ChatOutputDelta<'_>,
        created: u64,
    ) -> Result<LeasedOutput<Vec<u8>>, OutputFlowError> {
        if self.plan.text_reasoning.is_none() {
            return Err(OutputFlowError::Unsupported(
                "Chat delta requires an admitted Chat plan",
            ));
        }
        self.encode_projected_frame(permit, DataPayload::Chat(delta), created)
    }

    fn encode_projected_frame(
        &mut self,
        mut permit: OutputFramePermit,
        payload: DataPayload<'_>,
        created: u64,
    ) -> Result<LeasedOutput<Vec<u8>>, OutputFlowError> {
        self.validate_permit(&mut permit)?;
        let text_bytes = match payload {
            DataPayload::Text(text) => text.len(),
            DataPayload::Chat(delta) => delta.text().len(),
        };
        let text_total = self
            .emitted_text_bytes
            .checked_add(text_bytes)
            .ok_or(OutputFlowError::Overflow)?;
        if text_total > self.plan.semantic_bytes {
            return Err(OutputFlowError::BoundExceeded);
        }
        let size = codec::count(|out| match payload {
            DataPayload::Text(text) => codec::data(out, &self.plan.contract, text, created),
            DataPayload::Chat(delta) => codec::chat_data(out, &self.plan.contract, delta, created),
        })?;
        if size > permit.credit.amount().bytes {
            return Err(OutputFlowError::BoundExceeded);
        }
        // Allocate only after owning the full bound. On any error the permit
        // is lost and the request must fail; it must not blindly retry a wave.
        let bytes = codec::encode(size, |out| match payload {
            DataPayload::Text(text) => codec::data(out, &self.plan.contract, text, created),
            DataPayload::Chat(delta) => codec::chat_data(out, &self.plan.contract, delta, created),
        })?;
        let lease = permit.credit.split(OutputCreditAmount {
            events: 1,
            bytes: size,
            projection_bytes: 0,
        })?;
        self.wire = Some(permit.credit);
        self.emitted_text_bytes = text_total;
        self.frames += 1;
        Ok(lease.into_output(bytes))
    }

    /// Return an unsubmitted/Deferred wave's entire unspent byte reservoir.
    /// Only its event slot is released. Spent bytes are never replenished.
    pub fn return_unsubmitted_frame(
        &mut self,
        permit: OutputFramePermit,
    ) -> Result<(), OutputFlowError> {
        self.reclaim_unused_frame(permit)
    }

    /// A token was committed but UTF-8/stop projection produced no wire frame.
    /// Return only unused output ownership. The caller must still advance its
    /// committed-token/output frontier exactly once; this is not permission to
    /// retry model work. No event or lifetime wire bytes have been consumed.
    pub fn finish_without_wire_frame(
        &mut self,
        permit: OutputFramePermit,
    ) -> Result<(), OutputFlowError> {
        self.reclaim_unused_frame(permit)
    }

    fn reclaim_unused_frame(
        &mut self,
        mut permit: OutputFramePermit,
    ) -> Result<(), OutputFlowError> {
        self.validate_permit(&mut permit)?;
        let mut event = permit.credit.split(OutputCreditAmount {
            events: 1,
            bytes: 0,
            projection_bytes: 0,
        })?;
        if let Some(escrow) = self.future_events.as_mut() {
            escrow.try_merge(&mut event)?;
        }
        drop(event);
        self.wire = Some(permit.credit);
        Ok(())
    }

    /// Encodes all fixed terminal frames into one leased buffer. The event
    /// charge still covers all logical SSE events, including DONE. Data text
    /// must have been sent earlier using its own pre-reserved data bytes.
    pub fn encode_terminal(
        &mut self,
        terminal: OutputTerminal<'_>,
    ) -> Result<LeasedOutput<Vec<u8>>, OutputFlowError> {
        if self.closed {
            return Err(OutputFlowError::Closed);
        }
        if self.wire.is_none() && matches!(terminal, OutputTerminal::Success { .. }) {
            return Err(OutputFlowError::FrameInFlight);
        }
        if let OutputTerminal::Success { usage, .. } = &terminal {
            self.plan.validate_usage(usage)?;
        }
        let size = codec::count(|out| codec::terminal(out, &self.plan.contract, &terminal))?;
        if size > self.plan.terminal.bytes {
            return Err(OutputFlowError::BoundExceeded);
        }
        let bytes = codec::encode(size, |out| {
            codec::terminal(out, &self.plan.contract, &terminal)
        })?;
        let mut reservation = self.terminal.take().ok_or(OutputFlowError::Closed)?;
        reservation.shrink_to(OutputCreditAmount {
            bytes: size,
            ..self.plan.terminal
        })?;
        self.closed = true;
        drop(self.wire.take());
        drop(self.future_events.take());
        self.account.close();
        Ok(reservation.into_output(bytes))
    }

    /// Consumer history can outlive engine completion, but must carry this
    /// grant until that history is freed or moved to a bounded session store.
    /// Payload and all live aliases must fit the documented projection layout.
    pub fn into_retained_projection<T>(
        mut self,
        payload: T,
    ) -> Result<LeasedOutput<T>, OutputFlowError> {
        if !self.closed {
            return Err(OutputFlowError::Unsupported(
                "projection ownership transfers only after terminal encoding",
            ));
        }
        let reservation = self.projection.take().ok_or(OutputFlowError::Closed)?;
        Ok(reservation.into_output(payload))
    }
}

impl Drop for RequestOutputBudget {
    fn drop(&mut self) {
        self.account.close();
    }
}
