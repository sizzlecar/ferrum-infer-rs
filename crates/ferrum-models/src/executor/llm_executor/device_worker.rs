//! One managed service for the original model and token registry.
//!
//! Work is bounded before enqueue. Closing is an idempotent flag in the
//! existing incarnation with one coalesced wake, not an unbounded control
//! queue. Destructors never wait for the model, device, capacity or thread.

use super::token_execution::TokenWorkerState;
use ferrum_execution::model_executor::LogitsReturnPolicy;
use ferrum_execution::token_execution::*;
use ferrum_types::{FerrumError, Result};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{mpsc, Arc};
use std::thread;
use std::time::Duration;
use tokio::sync::{oneshot, OwnedSemaphorePermit, Semaphore};

const MAX_WORK: usize = 2;
const INITIAL_DELAY: Duration = Duration::from_millis(1);
const MAX_DELAY: Duration = Duration::from_millis(250);

pub(super) struct OwnedTokenWork {
    pub(super) frontier: ExecutionFrontier,
    pub(super) work: TokenWork,
    pub(super) policy: LogitsReturnPolicy,
    pub(super) cache_id: String,
    pub(super) prompt: Arc<[u32]>,
}

pub(super) struct OwnedTokenBatch {
    pub(super) claim: u64,
    pub(super) work: Vec<OwnedTokenWork>,
}

enum ClaimKind {
    Admission(ExecutionFrontier),
    Execution(OwnedTokenBatch),
}

/// Only short registry metadata access is needed on Drop. Numerical execution
/// must never hold that metadata lock while owning the model/device context.
struct DeliveryClaim {
    kind: Option<ClaimKind>,
    state: Arc<TokenWorkerState>,
    wake: mpsc::SyncSender<()>,
}

impl DeliveryClaim {
    fn accept(&mut self) {
        if let Some(kind) = self.kind.take() {
            match kind {
                ClaimKind::Admission(frontier) => self.state.accept_admission(&frontier),
                ClaimKind::Execution(batch) => self.state.accept_device_batch(&batch),
            }
        }
    }
}

impl Drop for DeliveryClaim {
    fn drop(&mut self) {
        if let Some(kind) = self.kind.take() {
            match kind {
                ClaimKind::Admission(frontier) => self.state.abandon_admission(&frontier),
                ClaimKind::Execution(batch) => self.state.abandon_device_batch(&batch),
            }
            let _ = self.wake.try_send(());
        }
    }
}

struct Delivery<T> {
    value: T,
    claim: DeliveryClaim,
}
impl<T> Delivery<T> {
    fn accept(mut self) -> T {
        self.claim.accept();
        self.value
    }
}

enum Command {
    Admit {
        claim: DeliveryClaim,
        result: oneshot::Sender<Result<Delivery<TokenAdmissionOutcome>>>,
        _permit: OwnedSemaphorePermit,
    },
    Execute {
        claim: DeliveryClaim,
        result: oneshot::Sender<Result<Delivery<ExecutionBatchOutcome>>>,
        _permit: OwnedSemaphorePermit,
    },
}

pub(super) struct LlmDeviceWorker {
    commands: mpsc::Sender<Command>,
    wake: mpsc::SyncSender<()>,
    capacity: Arc<Semaphore>,
    state: Arc<TokenWorkerState>,
}

impl LlmDeviceWorker {
    pub(super) fn new(state: Arc<TokenWorkerState>) -> Result<Self> {
        let (commands, receiver) = mpsc::channel();
        let (wake, wakes) = mpsc::sync_channel(1);
        let owned_state = Arc::clone(&state);
        thread::Builder::new()
            .name("ferrum-llm-device".to_owned())
            .spawn(move || run_worker(owned_state, receiver, wakes))
            .map_err(|e| FerrumError::backend(format!("cannot start model device worker: {e}")))?;
        Ok(Self {
            commands,
            wake,
            capacity: Arc::new(Semaphore::new(MAX_WORK)),
            state,
        })
    }

    pub(super) async fn admit(
        &self,
        input: ExecutionAdmission<'_>,
        cache_id: String,
    ) -> Result<TokenAdmissionOutcome> {
        let frontier = self.state.register_device_admission(input, cache_id)?;
        let sequence = frontier.sequence().clone();
        // Guarded before waiting for capacity; an unpublished incarnation is
        // still retained and closed if the admission caller disappears.
        let claim = DeliveryClaim {
            kind: Some(ClaimKind::Admission(frontier)),
            state: Arc::clone(&self.state),
            wake: self.wake.clone(),
        };
        let Some(ClaimKind::Admission(frontier)) = claim.kind.as_ref() else {
            unreachable!()
        };
        // Subscribe before the worker can publish terminal and remove the
        // original entry. The receiver keeps the complete terminal fact even
        // when the admission caller has not been polled again yet.
        let terminal = self.state.observe_admission(frontier)?;
        let permit = Arc::clone(&self.capacity)
            .acquire_owned()
            .await
            .map_err(|_| FerrumError::backend("model work queue is closed"))?;
        let (result, receiver) = oneshot::channel();
        self.commands
            .send(Command::Admit {
                claim,
                result,
                _permit: permit,
            })
            .map_err(|_| FerrumError::backend("model preparation worker is unavailable"))?;
        let _ = self.wake.try_send(());
        let observed = receiver.await.map_err(|_| {
            FerrumError::backend("model preparation observation was lost; original owner retained")
        });
        match observed.and_then(|result| result) {
            Ok(delivery) if matches!(&delivery.value, TokenAdmissionOutcome::Admitted(_)) => {
                Ok(delivery.accept())
            }
            Ok(delivery) => {
                // A retryable admission result must not race the physical
                // cleanup of the unpublished incarnation that attempted it.
                self.finish_unpublished_admission(&sequence, terminal)
                    .await?;
                Ok(delivery.accept())
            }
            Err(error) => {
                if let Err(cleanup_error) =
                    self.finish_unpublished_admission(&sequence, terminal).await
                {
                    tracing::error!(%error, %cleanup_error,
                        "failed admission also observed a terminal cleanup error");
                }
                Err(error)
            }
        }
    }

    async fn finish_unpublished_admission(
        &self,
        sequence: &ExecutorSequence,
        close: ExecutionCloseOutcome,
    ) -> Result<()> {
        self.state.abandon_sequence(sequence);
        let _ = self.wake.try_send(());
        let terminal_error = match close.disposition {
            ExecutionCloseDisposition::Ended(_) => None,
            ExecutionCloseDisposition::Held(held) => held.wait().await?.error,
        };
        if let (Some(primary), Some(secondary)) = (&close.error, &terminal_error) {
            tracing::error!(%primary, %secondary,
                "admission cleanup retains operation and terminal errors");
        }
        match close.error.or(terminal_error) {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    pub(super) async fn execute(
        &self,
        inputs: &[ExecutionWork<'_>],
    ) -> Result<ExecutionBatchOutcome> {
        let batch = self.state.claim_device_batch(inputs)?;
        let claim = DeliveryClaim {
            kind: Some(ClaimKind::Execution(batch)),
            state: Arc::clone(&self.state),
            wake: self.wake.clone(),
        };
        let permit = Arc::clone(&self.capacity)
            .acquire_owned()
            .await
            .map_err(|_| FerrumError::backend("model work queue is closed"))?;
        let (result, receiver) = oneshot::channel();
        self.commands
            .send(Command::Execute {
                claim,
                result,
                _permit: permit,
            })
            .map_err(|_| FerrumError::backend("model execution worker is unavailable"))?;
        let _ = self.wake.try_send(());
        match receiver.await {
            Ok(result) => result.map(Delivery::accept),
            Err(_) => {
                let outcome = self.state.interrupt_lost_delivery(
                    inputs,
                    FerrumError::backend(
                        "model execution observation was lost; original owner retained",
                    ),
                );
                let _ = self.wake.try_send(());
                outcome
            }
        }
    }

    pub(super) fn close(
        &self,
        sequence: &ExecutorSequence,
        reason: ExecutionEndReason,
    ) -> Result<ExecutionCloseOutcome> {
        let result = self.state.request_close(sequence, reason);
        let _ = self.wake.try_send(());
        result
    }

    #[cfg(test)]
    pub(super) fn observe_terminal(
        &self,
        frontier: &ExecutionFrontier,
    ) -> Result<ExecutionCloseOutcome> {
        // Passive subscription to the original owner: this does not mark it
        // closing, issue cancellation, or produce terminal evidence.
        self.state.observe_admission(frontier)
    }
}

impl Drop for LlmDeviceWorker {
    fn drop(&mut self) {
        self.capacity.close();
        self.state.request_shutdown();
        let _ = self.wake.try_send(());
    }
}

fn run_worker(
    state: Arc<TokenWorkerState>,
    commands: mpsc::Receiver<Command>,
    wakes: mpsc::Receiver<()>,
) {
    let mut delay = INITIAL_DELAY;
    loop {
        // Observation of the same retained frame, never numerical replay.
        match catch_unwind(AssertUnwindSafe(|| state.recover_device_sequences())) {
            Ok(true) => delay = INITIAL_DELAY,
            Ok(false) => delay = (delay * 2).min(MAX_DELAY),
            Err(_) => {
                state.record_device_worker_panic();
                delay = MAX_DELAY;
            }
        }
        match commands.try_recv() {
            Ok(command) => {
                let handled = catch_unwind(AssertUnwindSafe(|| match command {
                    Command::Admit {
                        claim,
                        result,
                        _permit,
                    } => {
                        if result.is_closed() {
                            return;
                        }
                        let Some(ClaimKind::Admission(frontier)) = claim.kind.as_ref() else {
                            return;
                        };
                        match state.prepare_device_admission(frontier) {
                            Ok(value) => {
                                let _ = result.send(Ok(Delivery { value, claim }));
                            }
                            Err(error) => {
                                let _ = result.send(Err(error));
                            }
                        }
                    }
                    Command::Execute {
                        claim,
                        result,
                        _permit,
                    } => {
                        if result.is_closed() {
                            return;
                        }
                        let Some(ClaimKind::Execution(batch)) = claim.kind.as_ref() else {
                            return;
                        };
                        match state.execute_device_batch(batch) {
                            Ok(value) => {
                                let _ = result.send(Ok(Delivery { value, claim }));
                            }
                            Err(error) => {
                                let _ = result.send(Err(error));
                            }
                        }
                    }
                }));
                if handled.is_err() {
                    state.record_device_worker_panic();
                }
                continue;
            }
            Err(mpsc::TryRecvError::Disconnected) => state.request_shutdown(),
            Err(mpsc::TryRecvError::Empty) => {}
        }
        if state.shutdown_complete() {
            break;
        }
        if state.has_device_recovery() {
            let _ = wakes.recv_timeout(delay);
        } else if wakes.recv().is_err() {
            state.request_shutdown();
        }
    }
}
