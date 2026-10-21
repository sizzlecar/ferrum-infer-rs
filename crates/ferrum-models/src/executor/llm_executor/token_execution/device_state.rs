//! Metadata and delivery ownership for the same model-owned token registry.
//! Numerical work never holds this metadata mutex. Cache/context ownership
//! remains in the original model, including when the observer disappears.

use super::super::device_worker::{OwnedTokenBatch, OwnedTokenWork};
use super::*;
use crate::common::llm::{
    ModelExecutionRecovery, ModelTokenBatchOutcome, ModelTokenCloseOutcome, ModelTokenInput,
    ModelTokenProduct,
};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use tokio::sync::watch;

pub(super) struct DeviceSequenceState {
    claim: Option<u64>,
    prepared: bool,
    published: bool,
    adapter: Option<ActiveLoraAdapter>,
    operation_error: Option<FerrumError>,
    recovery_error: Option<FerrumError>,
    needs_recovery: bool,
    terminal: Option<watch::Sender<Option<TerminalObservation>>>,
}

#[derive(Clone)]
struct TerminalObservation {
    ended: ExecutionEnded,
    error: Option<FerrumError>,
}

pub(in crate::executor::llm_executor) struct TokenWorkerState {
    model: ModelSlot,
    sequences: Arc<Mutex<TokenSequences>>,
    issuer: ExecutorSequenceIssuer,
    adapter_support: AdapterExecutionSupport,
    vocabulary: usize,
    capacity: usize,
    next_claim: AtomicU64,
    shutdown: AtomicBool,
    shutdown_error: Mutex<Option<FerrumError>>,
    #[cfg(test)]
    pub(super) admission_before_registry_lock:
        Mutex<Option<(std::sync::mpsc::Sender<()>, std::sync::mpsc::Receiver<()>)>>,
}

impl TokenWorkerState {
    pub(super) fn new(executor: &LlmExecutor) -> Result<Self> {
        Ok(Self {
            model: Arc::clone(&executor.model),
            sequences: Arc::clone(&executor.token_sequences),
            issuer: executor.token_issuer.clone(),
            adapter_support: executor.adapter_support,
            vocabulary: executor.model_config.vocab_size,
            capacity: executor.model_capacity,
            next_claim: AtomicU64::new(1),
            shutdown: AtomicBool::new(false),
            shutdown_error: Mutex::new(None),
            #[cfg(test)]
            admission_before_registry_lock: Mutex::new(None),
        })
    }

    pub(in crate::executor::llm_executor) fn register_device_admission(
        &self,
        input: ExecutionAdmission<'_>,
        cache_id: String,
    ) -> Result<ExecutionFrontier> {
        input.validate()?;
        if self.shutdown.load(Ordering::Acquire) {
            return Err(FerrumError::backend("model worker is closing"));
        }
        let context = input.context;
        if context.maximum_sequence_tokens > self.capacity
            || context.maximum_sequence_tokens > u32::MAX as usize
            || context
                .input_tokens
                .iter()
                .any(|token| token.get() as usize >= self.vocabulary)
        {
            return Err(FerrumError::request_validation(
                "token admission exceeds model capacity or vocabulary",
            ));
        }
        let adapter = match input.adapter {
            Some(selection) => {
                self.adapter_support.require_lora()?;
                Some(ActiveLoraAdapter {
                    name: selection.name.to_owned(),
                    path: selection.path.into(),
                })
            }
            None => None,
        };
        #[cfg(test)]
        if let Some((entered, resume)) = self.admission_before_registry_lock.lock().take() {
            entered
                .send(())
                .expect("registration test must observe its gate");
            resume
                .recv()
                .expect("registration test must release its gate");
        }
        let mut sequences = self.sequences.lock();
        // Shutdown marks existing entries while holding this same lock.
        // An earlier fast rejection check cannot exclude a shutdown that
        // finished its scan before this registration acquired the lock.
        if sequences.closing || self.shutdown.load(Ordering::Acquire) {
            return Err(FerrumError::backend("model worker is closing"));
        }
        if sequences
            .live
            .values()
            .any(|entry| &entry.request_id == context.request_id)
        {
            return Err(FerrumError::request_validation(
                "request already has a live token incarnation",
            ));
        }
        let sequence = self.issuer.issue_sequence()?;
        let frontier = self.issuer.frontier(&sequence, 0)?;
        let recovery_key = self.issuer.recovery_key(&sequence)?;
        sequences.live.insert(
            sequence,
            TokenSequence {
                request_id: context.request_id.clone(),
                cache_id,
                prompt: context
                    .input_tokens
                    .iter()
                    .map(|token| token.get())
                    .collect(),
                maximum_tokens: context.maximum_sequence_tokens,
                product_prompt_tokens: context.product_prompt_tokens,
                replayed_output_tokens: context.replayed_output_tokens,
                frontier: frontier.clone(),
                recovery_key,
                closing: false,
                device: Some(DeviceSequenceState {
                    claim: None,
                    prepared: false,
                    published: false,
                    adapter,
                    operation_error: None,
                    recovery_error: None,
                    needs_recovery: false,
                    terminal: None,
                }),
            },
        );
        Ok(frontier)
    }

    pub(in crate::executor::llm_executor) fn prepare_device_admission(
        &self,
        frontier: &ExecutionFrontier,
    ) -> Result<TokenAdmissionOutcome> {
        let (cache_id, capacity, prompt, adapter) = {
            let sequences = self.sequences.lock();
            let entry = sequences
                .live
                .get(frontier.sequence())
                .filter(|entry| !entry.closing && entry.frontier == *frontier)
                .ok_or_else(|| FerrumError::request_validation("token admission was cancelled"))?;
            let device = entry
                .device
                .as_ref()
                .ok_or_else(|| FerrumError::internal("missing device admission"))?;
            (
                entry.cache_id.clone(),
                entry.maximum_tokens,
                Arc::clone(&entry.prompt),
                device.adapter.clone(),
            )
        };
        let result: Result<ModelKvReservation> = {
            let mut model = lock_live_model(&self.model)?;
            (|| {
                if let Some(adapter) = &adapter {
                    model.preload_lora_adapter(adapter)?;
                }
                model.prepare_token_kv(ModelKvPreparation {
                    cache_id: &cache_id,
                    capacity_hint: capacity,
                    prompt: &prompt,
                    adapter: adapter.as_ref(),
                })
            })()
        };
        let mut sequences = self.sequences.lock();
        let entry = sequences.live.get_mut(frontier.sequence()).ok_or_else(|| {
            FerrumError::internal("device preparation lost its original incarnation")
        })?;
        match result {
            Ok(ModelKvReservation::Reserved(_)) if !entry.closing => {
                entry.device.as_mut().unwrap().prepared = true;
                Ok(TokenAdmissionOutcome::Admitted(frontier.clone()))
            }
            Ok(ModelKvReservation::CapacityDeferred(deferred)) => {
                entry.closing = true;
                Ok(TokenAdmissionOutcome::CapacityDeferred(deferred))
            }
            Ok(ModelKvReservation::SequenceSlotsDeferred(deferred)) => {
                entry.closing = true;
                Ok(TokenAdmissionOutcome::SequenceSlotsDeferred(deferred))
            }
            Ok(ModelKvReservation::Reserved(_)) => Err(FerrumError::request_validation(
                "token admission was cancelled during preparation",
            )),
            Err(error) => {
                entry.closing = true;
                let device = entry.device.as_mut().unwrap();
                device.needs_recovery = true;
                device.operation_error.get_or_insert_with(|| error.clone());
                Err(error)
            }
        }
    }

    pub(in crate::executor::llm_executor) fn accept_admission(
        &self,
        frontier: &ExecutionFrontier,
    ) -> Result<()> {
        let mut sequences = self.sequences.lock();
        // Terminal publication/removal uses this same lock. Preparing a
        // response on the worker does not promise it is still usable when
        // the admission future is eventually polled again.
        if frontier.sequence().observed_end().is_some() {
            return Err(FerrumError::request_validation(
                "token admission ended before delivery was accepted",
            ));
        }
        let entry = sequences.live.get_mut(frontier.sequence()).ok_or_else(|| {
            FerrumError::internal("admission delivery lost its original incarnation")
        })?;
        let device = entry
            .device
            .as_mut()
            .ok_or_else(|| FerrumError::internal("admission delivery has no device owner"))?;
        if let Some(error) = &device.operation_error {
            return Err(error.clone());
        }
        if self.shutdown.load(Ordering::Acquire) || entry.closing {
            return Err(FerrumError::request_validation(
                "token admission closed before delivery was accepted",
            ));
        }
        if entry.frontier != *frontier
            || !device.prepared
            || device.published
            || device.claim.is_some()
            || device.needs_recovery
        {
            return Err(FerrumError::internal(
                "admission delivery does not match an unpublished prepared frontier",
            ));
        }
        device.published = true;
        Ok(())
    }

    pub(in crate::executor::llm_executor) fn abandon_admission(
        &self,
        frontier: &ExecutionFrontier,
    ) {
        if let Some(entry) = self.sequences.lock().live.get_mut(frontier.sequence()) {
            entry.closing = true;
        }
    }

    pub(in crate::executor::llm_executor) fn observe_admission(
        &self,
        frontier: &ExecutionFrontier,
    ) -> Result<ExecutionCloseOutcome> {
        let mut sequences = self.sequences.lock();
        let entry = sequences.live.get_mut(frontier.sequence()).ok_or_else(|| {
            FerrumError::internal("unpublished admission lost its original owner")
        })?;
        self.observe_close_locked(entry)
    }

    pub(in crate::executor::llm_executor) fn abandon_sequence(&self, sequence: &ExecutorSequence) {
        if let Some(entry) = self.sequences.lock().live.get_mut(sequence) {
            entry.closing = true;
        }
    }

    pub(in crate::executor::llm_executor) fn claim_device_batch(
        &self,
        inputs: &[ExecutionWork<'_>],
    ) -> Result<OwnedTokenBatch> {
        self.issuer.batch(inputs, self.vocabulary)?;
        let mut sequences = self.sequences.lock();
        for input in inputs {
            let entry = sequences
                .live
                .get(input.sequence())
                .ok_or_else(|| FerrumError::request_validation("token incarnation is not live"))?;
            entry.validate_work(input)?;
            if !entry
                .device
                .as_ref()
                .is_some_and(|device| device.prepared && device.published && device.claim.is_none())
            {
                return Err(FerrumError::request_validation(
                    "token incarnation has unpublished or outstanding work",
                ));
            }
        }
        let claim = self
            .next_claim
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| FerrumError::resource_exhausted("model delivery claim space exhausted"))?;
        let mut work = Vec::with_capacity(inputs.len());
        for input in inputs {
            let entry = sequences.live.get_mut(input.sequence()).unwrap();
            entry.device.as_mut().unwrap().claim = Some(claim);
            work.push(OwnedTokenWork {
                frontier: input.expected_frontier().clone(),
                work: input.work(),
                policy: input.logits_policy().clone(),
                cache_id: entry.cache_id.clone(),
                prompt: Arc::clone(&entry.prompt),
            });
        }
        Ok(OwnedTokenBatch { claim, work })
    }

    pub(in crate::executor::llm_executor) fn accept_device_batch(&self, batch: &OwnedTokenBatch) {
        let mut sequences = self.sequences.lock();
        for input in &batch.work {
            if let Some(device) = sequences
                .live
                .get_mut(input.frontier.sequence())
                .and_then(|entry| entry.device.as_mut())
            {
                if device.claim == Some(batch.claim) {
                    device.claim = None;
                }
            }
        }
    }

    pub(in crate::executor::llm_executor) fn abandon_device_batch(&self, batch: &OwnedTokenBatch) {
        let mut sequences = self.sequences.lock();
        for input in &batch.work {
            if let Some(entry) = sequences.live.get_mut(input.frontier.sequence()) {
                if entry
                    .device
                    .as_ref()
                    .is_some_and(|device| device.claim == Some(batch.claim))
                {
                    entry.closing = true;
                }
            }
        }
    }

    pub(in crate::executor::llm_executor) fn request_close(
        &self,
        sequence: &ExecutorSequence,
        reason: ExecutionEndReason,
    ) -> Result<ExecutionCloseOutcome> {
        self.issuer.frontier(sequence, 0)?;
        let mut sequences = self.sequences.lock();
        // End publication and removal use this same lock. Do not observe
        // false outside the lock and then mistake a completed removal for an
        // unknown incarnation.
        if let Some(ended) = sequence.observed_end() {
            return Ok(ExecutionCloseOutcome::ended(ended));
        }
        let entry = sequences
            .live
            .get_mut(sequence)
            .ok_or_else(|| FerrumError::request_validation("token incarnation is not live"))?;
        entry.validate_close(reason)?;
        entry.closing = true;
        self.observe_close_locked(entry)
    }

    fn observe_close_locked(&self, entry: &mut TokenSequence) -> Result<ExecutionCloseOutcome> {
        let device = entry
            .device
            .as_mut()
            .ok_or_else(|| FerrumError::internal("missing device close owner"))?;
        let mut receiver = device
            .terminal
            .get_or_insert_with(|| watch::channel(None).0)
            .subscribe();
        let held = self.issuer.held(entry.recovery_key.clone(), async move {
            loop {
                let terminal = receiver.borrow_and_update().clone();
                if let Some(terminal) = terminal {
                    return Ok(ExecutionTerminalOutcome {
                        ended: terminal.ended,
                        error: terminal.error,
                    });
                }
                receiver.changed().await.map_err(|_| {
                    FerrumError::backend("model terminal observer lost its retained owner")
                })?;
            }
        })?;
        let mut close = ExecutionCloseOutcome::held(held);
        if let Some(error) = &device.operation_error {
            close = close.with_error(error.clone());
        }
        Ok(close)
    }

    pub(in crate::executor::llm_executor) fn request_shutdown(&self) {
        self.shutdown.store(true, Ordering::Release);
        let mut sequences = self.sequences.lock();
        sequences.closing = true;
        for entry in sequences.live.values_mut() {
            entry.closing = true;
        }
    }

    pub(in crate::executor::llm_executor) fn record_device_worker_panic(&self) {
        tracing::error!(
            "model worker panicked; retaining original context and closing live incarnations"
        );
        for entry in self.sequences.lock().live.values_mut() {
            if let Some(device) = &mut entry.device {
                device
                    .operation_error
                    .get_or_insert_with(|| FerrumError::backend("model worker panicked"));
            }
        }
        self.request_shutdown();
        self.record_shutdown_error(FerrumError::backend("model worker panicked"));
    }

    pub(in crate::executor::llm_executor) fn is_shutting_down(&self) -> bool {
        self.shutdown.load(Ordering::Acquire)
    }

    pub(in crate::executor::llm_executor) fn shutdown_complete(&self) -> bool {
        self.shutdown.load(Ordering::Acquire) && self.sequences.lock().live.is_empty()
    }

    pub(in crate::executor::llm_executor) fn record_shutdown_error(&self, error: FerrumError) {
        if self.shutdown.load(Ordering::Acquire) {
            self.shutdown_error.lock().get_or_insert(error);
        }
    }

    pub(in crate::executor::llm_executor) fn shutdown_error(&self) -> Option<FerrumError> {
        self.shutdown_error.lock().clone()
    }

    pub(in crate::executor::llm_executor) fn retire_model(
        &self,
    ) -> crate::common::ModelRetirementOutcome {
        use crate::common::ModelRetirementOutcome;
        let mut slot = self.model.lock();
        let Some(model) = slot.as_mut() else {
            return ModelRetirementOutcome::Retired {
                error: self.shutdown_error(),
            };
        };
        let outcome = match catch_host_execution(|| model.retire_model()) {
            Ok(outcome) => outcome,
            Err(error) => ModelRetirementOutcome::Retained { error },
        };
        match outcome {
            ModelRetirementOutcome::Retained { error } => {
                self.record_shutdown_error(error.clone());
                ModelRetirementOutcome::Retained { error }
            }
            ModelRetirementOutcome::Retired { error } => {
                if let Some(error) = error {
                    self.record_shutdown_error(error);
                }
                // The checked retirement fact precedes removal. A retained
                // model is never replaced by a tombstone or dropped on Err.
                let retired_model = slot.take();
                drop(slot);
                if let Err(error) = catch_host_execution(|| drop(retired_model)) {
                    self.record_shutdown_error(error);
                }
                ModelRetirementOutcome::Retired {
                    error: self.shutdown_error(),
                }
            }
        }
    }

    pub(in crate::executor::llm_executor) fn has_device_recovery(&self) -> bool {
        self.shutdown.load(Ordering::Acquire)
            || self
                .sequences
                .lock()
                .live
                .values()
                .any(|entry| entry.closing)
    }

    pub(in crate::executor::llm_executor) fn recover_device_sequences(&self) -> bool {
        let closing: Vec<_> = self
            .sequences
            .lock()
            .live
            .iter()
            .filter(|(_, entry)| entry.closing)
            .map(|(sequence, entry)| (sequence.clone(), entry.cache_id.clone()))
            .collect();
        if closing.is_empty() {
            return true;
        }
        let mut model = match lock_live_model(&self.model) {
            Ok(model) => model,
            Err(error) => {
                self.record_shutdown_error(error);
                return false;
            }
        };
        let terminal_error = match model.recover_token_execution() {
            ModelExecutionRecovery::Held { error } => {
                self.record_shutdown_error(error.clone());
                tracing::debug!(%error, "model frame remains held");
                return false;
            }
            ModelExecutionRecovery::Quiescent { error } => error,
        };
        if let Some(error) = &terminal_error {
            self.record_shutdown_error(error.clone());
            let mut sequences = self.sequences.lock();
            for (sequence, _) in &closing {
                if let Some(device) = sequences
                    .live
                    .get_mut(sequence)
                    .and_then(|entry| entry.device.as_mut())
                {
                    if device.needs_recovery {
                        device.recovery_error.get_or_insert_with(|| error.clone());
                    }
                }
            }
        }
        let mut all_closed = true;
        for (sequence, cache_id) in closing {
            match model.close_token_execution(&cache_id) {
                ModelTokenCloseOutcome::Held { error } => {
                    self.record_shutdown_error(error.clone());
                    all_closed = false;
                    let mut sequences = self.sequences.lock();
                    if let Some(device) = sequences
                        .live
                        .get_mut(&sequence)
                        .and_then(|entry| entry.device.as_mut())
                    {
                        device.needs_recovery = true;
                        device.operation_error.get_or_insert(error);
                    }
                }
                ModelTokenCloseOutcome::Ended { error } => {
                    if let Some(error) = &error {
                        self.record_shutdown_error(error.clone());
                    }
                    // Publish and notify while holding the same metadata lock;
                    // no waiter can be registered between Ended and removal.
                    let mut sequences = self.sequences.lock();
                    match self.issuer.ended(&sequence) {
                        Ok(ended) => {
                            if let Some(mut entry) = sequences.live.remove(&sequence) {
                                if let Some(device) = &mut entry.device {
                                    if let (Some(close_error), Some(recovery_error)) =
                                        (&error, &device.recovery_error)
                                    {
                                        tracing::error!(%close_error, %recovery_error,
                                            "cache closure and retained frame both reported terminal errors");
                                    }
                                    let deferred_error =
                                        error.or_else(|| device.recovery_error.take());
                                    if let (Some(operation_error), Some(terminal_error)) =
                                        (&device.operation_error, &deferred_error)
                                    {
                                        tracing::error!(%operation_error, %terminal_error,
                                            "model operation and eventual terminal observation both failed");
                                    }
                                    let error = device.operation_error.take().or(deferred_error);
                                    if let Some(terminal) = device.terminal.take() {
                                        let _ = terminal.send_replace(Some(TerminalObservation {
                                            ended,
                                            error,
                                        }));
                                    }
                                }
                            }
                        }
                        Err(error) => {
                            self.record_shutdown_error(error.clone());
                            all_closed = false;
                            tracing::error!(%error, "model cache closed but terminal identity publication failed");
                        }
                    }
                }
            }
        }
        all_closed
    }
}

mod execute;
