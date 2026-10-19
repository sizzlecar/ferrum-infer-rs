//! CUDA lifecycle qualification through the unwrapped production Llama model.
//! Only the real mathematical head has a host fault/gate; it cannot grant mode,
//! completion, recovery, or final-close authority to the model.

use super::*;
use crate::executor::llm_executor::ModelSlot;
use cudarc::driver::CudaStream;
use ferrum_execution::executor_lifecycle::{
    ExecutorOwner, ExecutorShutdownObserver, ExecutorShutdownPhase, ExecutorShutdownReport,
};
use ferrum_kernels::backend::{cuda::CudaBackend, Backend, CudaBuf};
use ferrum_kernels::{Linear, LinearMetadata, LoadedLinear};
use ferrum_quantization::{QuantConfig, WeightLoader};
use std::sync::atomic::{AtomicBool, AtomicUsize};
use std::sync::{mpsc, Weak};
use std::thread::{self, ThreadId};
use tokio::sync::oneshot;

const WAIT: Duration = Duration::from_secs(10);
const HEAD_ERROR: &str = "qualification encoded CUDA head failure";

struct Gate {
    entered: oneshot::Sender<()>,
    release: mpsc::Receiver<()>,
}

struct Release(Option<mpsc::Sender<()>>);

impl Release {
    fn now(&mut self) {
        if let Some(release) = self.0.take() {
            let _ = release.send(());
        }
    }
}

impl Drop for Release {
    fn drop(&mut self) {
        self.now();
    }
}

struct Control {
    construction_thread: ThreadId,
    worker_thread: Mutex<Option<ThreadId>>,
    original_stream: Mutex<Option<Weak<CudaStream>>>,
    fail_head_once: AtomicBool,
    actual_forwards: AtomicUsize,
    actual_head_drops: AtomicUsize,
    gate: Mutex<Option<Gate>>,
}

impl Control {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            construction_thread: thread::current().id(),
            worker_thread: Mutex::new(None),
            original_stream: Mutex::new(None),
            fail_head_once: AtomicBool::new(false),
            actual_forwards: AtomicUsize::new(0),
            actual_head_drops: AtomicUsize::new(0),
            gate: Mutex::new(None),
        })
    }

    fn record_loaded_buffer(&self, buffer: &CudaBuf) {
        assert_eq!(thread::current().id(), self.construction_thread);
        let mut original = self.original_stream.lock();
        if let Some(original) = original.as_ref() {
            assert!(Arc::ptr_eq(
                &original.upgrade().unwrap(),
                allocation_stream(buffer)
            ));
        } else {
            *original = Some(Arc::downgrade(allocation_stream(buffer)));
        }
    }

    fn verify_worker_buffers(&self, input: &CudaBuf, output: &CudaBuf) {
        let current = thread::current().id();
        assert_ne!(current, self.construction_thread);
        let mut worker = self.worker_thread.lock();
        assert_eq!(*worker.get_or_insert(current), current);
        let original = self
            .original_stream
            .lock()
            .as_ref()
            .unwrap()
            .upgrade()
            .unwrap();
        assert!(Arc::ptr_eq(&original, allocation_stream(input)));
        assert!(Arc::ptr_eq(&original, allocation_stream(output)));
    }

    fn gate_next_head(&self) -> (oneshot::Receiver<()>, Release) {
        let (entered, started) = oneshot::channel();
        let (release, receiver) = mpsc::channel();
        assert!(self
            .gate
            .lock()
            .replace(Gate {
                entered,
                release: receiver
            })
            .is_none());
        (started, Release(Some(release)))
    }

    fn stream_is_live(&self) -> bool {
        self.original_stream
            .lock()
            .as_ref()
            .unwrap()
            .upgrade()
            .is_some()
    }
}

fn allocation_stream(buffer: &CudaBuf) -> &Arc<CudaStream> {
    match buffer {
        CudaBuf::F16(buffer) => buffer.stream(),
        CudaBuf::F32(buffer) => buffer.stream(),
        CudaBuf::U32(buffer) => buffer.stream(),
        CudaBuf::I32(buffer) => buffer.stream(),
        CudaBuf::I8(buffer) => buffer.stream(),
    }
}

struct ObservedHead {
    inner: Box<dyn Linear<CudaBackend>>,
    control: Arc<Control>,
}

impl Linear<CudaBackend> for ObservedHead {
    fn in_features(&self) -> usize {
        self.inner.in_features()
    }
    fn out_features(&self) -> usize {
        self.inner.out_features()
    }
    fn metadata(&self) -> LinearMetadata {
        self.inner.metadata()
    }
    fn forward(
        &self,
        context: &mut <CudaBackend as Backend>::Context,
        input: &CudaBuf,
        output: &mut CudaBuf,
        rows: usize,
    ) {
        self.control.verify_worker_buffers(input, output);
        self.inner.forward(context, input, output, rows);
        self.control.actual_forwards.fetch_add(1, Ordering::SeqCst);
        // The actual CUDA operation has encoded before this host boundary.
        // No external model mutex guard is kept across an async wait.
        let gate = self.control.gate.lock().take();
        if let Some(gate) = gate {
            gate.entered
                .send(())
                .expect("test must observe the original encoded head");
            gate.release
                .recv_timeout(WAIT)
                .expect("test must release the original head");
        }
        if self.control.fail_head_once.swap(false, Ordering::SeqCst) {
            std::panic::panic_any(FerrumError::backend(HEAD_ERROR));
        }
    }
}

impl Drop for ObservedHead {
    fn drop(&mut self) {
        self.control
            .actual_head_drops
            .fetch_add(1, Ordering::SeqCst);
    }
}

struct ObservedLoader {
    inner: SyntheticLlamaLoader,
    control: Arc<Control>,
}

impl WeightLoader<CudaBackend> for ObservedLoader {
    fn load_tensor(
        &self,
        context: &mut <CudaBackend as Backend>::Context,
        name: &str,
    ) -> Result<CudaBuf> {
        let buffer = <SyntheticLlamaLoader as WeightLoader<CudaBackend>>::load_tensor(
            &self.inner,
            context,
            name,
        )?;
        self.control.record_loaded_buffer(&buffer);
        Ok(buffer)
    }
    fn load_linear(
        &self,
        context: &mut <CudaBackend as Backend>::Context,
        name: &str,
    ) -> Result<LoadedLinear<CudaBackend>> {
        let inner = <SyntheticLlamaLoader as WeightLoader<CudaBackend>>::load_linear(
            &self.inner,
            context,
            name,
        )?;
        if name == "lm_head" {
            // Replay skips forward's host gate, failure and observation hooks.
            // This real operation decorator is deliberately eager-only.
            Ok(LoadedLinear::new(Box::new(ObservedHead {
                inner: inner.into_eager_box(),
                control: Arc::clone(&self.control),
            })))
        } else {
            Ok(inner)
        }
    }
    fn has_tensor(&self, name: &str) -> bool {
        <SyntheticLlamaLoader as WeightLoader<CudaBackend>>::has_tensor(&self.inner, name)
    }
    fn quant_config(&self) -> Option<&QuantConfig> {
        <SyntheticLlamaLoader as WeightLoader<CudaBackend>>::quant_config(&self.inner)
    }
}

fn observed_executor(paged: bool) -> (LlmExecutor, Arc<Control>) {
    let control = Control::new();
    let loader = ObservedLoader {
        inner: SyntheticLlamaLoader::new(config()),
        control: Arc::clone(&control),
    };
    let model = LlamaFamilyModel::<CudaBackend>::new_with_execution_config(
        config(),
        &loader,
        LlamaExecutionConfig {
            kv_capacity: Some(32),
            paged_max_seqs: 4,
            options: LlamaExecutionOptions {
                metal_paged_kv: Some(paged),
                kv_max_blocks: Some(4),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        model.token_execution_mode(),
        ModelTokenExecutionMode::ManagedWorker
    );
    assert!(control.stream_is_live());
    (executor_from_model(model, Device::CUDA(0)), control)
}

async fn retired(mut observer: ExecutorShutdownObserver) -> ExecutorShutdownReport {
    tokio::time::timeout(WAIT, async move {
        loop {
            let report = observer.snapshot();
            if report.phase() == ExecutorShutdownPhase::Retired {
                return report;
            }
            observer.changed().await;
        }
    })
    .await
    .expect("the original CUDA worker must join after checked retirement")
}

fn assert_released(slot: &ModelSlot, control: &Control) {
    assert!(
        slot.lock().is_none(),
        "Retired must remove the actual model Box"
    );
    assert_eq!(control.actual_head_drops.load(Ordering::SeqCst), 1);
    assert!(
        !control.stream_is_live(),
        "the original allocation stream must lose its final owner"
    );
}

async fn close_original(executor: LlmExecutor, control: &Control) {
    // This clone observes only the slot. It cannot keep the model alive after
    // the original worker actually takes/drops its contents on retirement.
    let slot = Arc::clone(&executor.model);
    let owner = ExecutorOwner::new(Box::new(executor));
    retired(owner.begin_shutdown()).await.result().unwrap();
    assert_released(&slot, control);
    assert!(owner.scope().is_err());
}

#[tokio::test]
#[ignore = "requires actual CUDA unwrapped completion and original owner retirement"]
async fn qualification_unwrapped_cuda_completion_and_original_owner_close() {
    for paged in [false, true] {
        let (executor, control) = observed_executor(paged);
        let executor = qualified_completion_with_executor(
            executor,
            paged,
            ModelTokenExecutionMode::ManagedWorker,
        )
        .await;
        assert!(control.worker_thread.lock().is_some());
        close_original(executor, &control).await;
    }
}

#[tokio::test]
#[ignore = "requires actual CUDA cold model allocation and checked retirement"]
async fn qualification_unwrapped_cuda_cold_owner_closes_without_admission() {
    let (executor, control) = observed_executor(true);
    assert!(executor.token_worker.get().is_none());
    assert!(control.worker_thread.lock().is_none());
    close_original(executor, &control).await;
    assert_eq!(control.actual_forwards.load(Ordering::SeqCst), 0);
}

#[tokio::test]
#[ignore = "requires actual CUDA head encoding, failure recovery and independent CPU logits"]
async fn qualification_unwrapped_cuda_encoded_failure_keeps_peer_and_closes_original() {
    let (executor, control) = observed_executor(true);
    let failed = admit(&executor, &RequestId::new()).await;
    let peer = admit(&executor, &RequestId::new()).await;
    control.fail_head_once.store(true, Ordering::SeqCst);
    let policy = LogitsReturnPolicy::FullLogits;
    let work = [ExecutionWork::new(
        &failed,
        TokenWork::Prefill(PrefillChunk::new(0, 3, 3).unwrap()),
        &policy,
    )
    .unwrap()];
    let ExecutionBatchOutcome::Interrupted(interrupted) =
        executor.execute_tokens(&work).await.unwrap()
    else {
        panic!("the original encoded participant must report its failure");
    };
    let (completed, mut remaining, causes) = interrupted.into_parts();
    assert!(completed.is_empty());
    assert_eq!(remaining.len(), 1);
    assert_eq!(causes.len(), 1);
    assert!(
        matches!(&causes[0], ExecutionInterruptionCause::Error(error) if error.to_string() == FerrumError::backend(HEAD_ERROR).to_string())
    );
    let participant = remaining.remove(0);
    assert_eq!(participant.sequence(), failed.sequence());
    assert!(matches!(
        participant.progress(),
        ExecutionProgress::Terminal {
            disposition: TerminalModelDisposition::Failed,
            ..
        }
    ));
    let (_, _, _, continuation, _) = participant.into_parts();
    let ended = match continuation {
        ContinuationDisposition::Ended(ended) => ended,
        ContinuationDisposition::Held(held) => {
            let terminal = tokio::time::timeout(WAIT, held.wait())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(
                terminal.error.unwrap().to_string(),
                FerrumError::backend(HEAD_ERROR).to_string()
            );
            terminal.ended
        }
        _ => panic!("encoded failure must not return replay authority"),
    };
    assert_eq!(ended.sequence(), failed.sequence());
    assert_eq!(control.actual_forwards.load(Ordering::SeqCst), 1);
    assert_eq!(
        executor
            .lock_model()
            .unwrap()
            .kv_slot_capacity_snapshot()
            .unwrap()
            .free_blocks,
        3
    );
    assert_eq!(
        executor.token_sequences.lock().live[peer.sequence()].frontier,
        peer
    );
    let mut reference = real_model::<CpuBackend>(false);
    let prefill = execute(
        &executor,
        &peer,
        TokenWork::Prefill(PrefillChunk::new(0, 3, 3).unwrap()),
    )
    .await;
    compare_logits(&prefill, &reference.prefill_internal("peer", &[1, 2, 3]));
    let decode = execute(
        &executor,
        prefill.next_frontier(),
        TokenWork::Decode {
            token: TokenId(4),
            position: 3,
        },
    )
    .await;
    compare_logits(&decode, &reference.decode_internal("peer", 4, 3));
    assert_eq!(
        control.actual_forwards.load(Ordering::SeqCst),
        3,
        "the failed head must never replay"
    );
    terminal(
        executor
            .finish_tokens(
                peer.sequence(),
                ExecutionEndReason::Completed {
                    input_tokens: 3,
                    output_tokens: 2,
                },
            )
            .await
            .unwrap(),
    )
    .await;
    assert_all_slots_free(&executor, true);
    reference.release("peer");
    reference.close().unwrap();
    close_original(executor, &control).await;
}

#[tokio::test]
#[ignore = "requires actual encoded CUDA cancellation and healthy peer continuation"]
async fn qualification_unwrapped_cuda_encoded_cancellation_preserves_peer() {
    let (executor, control) = observed_executor(true);
    let cancelled = admit(&executor, &RequestId::new()).await;
    let peer = admit(&executor, &RequestId::new()).await;
    let mut reference = real_model::<CpuBackend>(false);
    let expected = reference.prefill_internal("peer", &[1, 2, 3]);
    let mut frontiers = Vec::new();
    for frontier in [&cancelled, &peer] {
        let prefill = execute(
            &executor,
            frontier,
            TokenWork::Prefill(PrefillChunk::new(0, 3, 3).unwrap()),
        )
        .await;
        compare_logits(&prefill, &expected);
        frontiers.push(prefill.next_frontier().clone());
    }
    let policy = LogitsReturnPolicy::FullLogits;
    let work = [ExecutionWork::new(
        &frontiers[0],
        TokenWork::Decode {
            token: TokenId(4),
            position: 3,
        },
        &policy,
    )
    .unwrap()];
    let (entered, mut release) = control.gate_next_head();
    let mut pending = Box::pin(executor.execute_tokens(&work));
    assert!(matches!(
        futures::poll!(pending.as_mut()),
        std::task::Poll::Pending
    ));
    tokio::time::timeout(WAIT, entered).await.unwrap().unwrap();
    assert_eq!(control.actual_forwards.load(Ordering::SeqCst), 3);
    let passive = executor
        .token_device_worker()
        .unwrap()
        .observe_terminal(&frontiers[0])
        .unwrap();
    let closing = executor.cancel_tokens(cancelled.sequence()).unwrap();
    assert!(matches!(
        &closing.disposition,
        ExecutionCloseDisposition::Held(_)
    ));
    drop(pending);
    assert!(cancelled.sequence().observed_end().is_none());
    assert!(control.stream_is_live());
    assert_eq!(control.actual_head_drops.load(Ordering::SeqCst), 0);
    // The actual worker owns the model while held after encoding. Trying to
    // lock it here would deadlock the test rather than prove a resource fact.
    assert!(executor.model.try_lock().is_none());
    release.now();
    let ended = terminal(closing).await;
    assert_eq!(ended.sequence(), cancelled.sequence());
    assert_eq!(terminal(passive).await, ended);
    assert_eq!(
        executor.token_sequences.lock().live[peer.sequence()].frontier,
        frontiers[1]
    );
    assert_eq!(
        executor
            .lock_model()
            .unwrap()
            .kv_slot_capacity_snapshot()
            .unwrap()
            .free_blocks,
        3
    );
    let decoded = execute(
        &executor,
        &frontiers[1],
        TokenWork::Decode {
            token: TokenId(4),
            position: 3,
        },
    )
    .await;
    compare_logits(&decoded, &reference.decode_internal("peer", 4, 3));
    assert_eq!(
        control.actual_forwards.load(Ordering::SeqCst),
        4,
        "abandoned delivery must not replay the encoded head"
    );
    terminal(
        executor
            .finish_tokens(
                peer.sequence(),
                ExecutionEndReason::Completed {
                    input_tokens: 3,
                    output_tokens: 2,
                },
            )
            .await
            .unwrap(),
    )
    .await;
    assert_all_slots_free(&executor, true);
    reference.release("peer");
    reference.close().unwrap();
    close_original(executor, &control).await;
}

#[tokio::test]
#[ignore = "requires real encoded CUDA work and cancellation-safe final-owner retirement"]
async fn qualification_unwrapped_cuda_cancelled_close_waiter_joins_original_worker() {
    let (executor, control) = observed_executor(true);
    let frontier = admit(&executor, &RequestId::new()).await;
    let policy = LogitsReturnPolicy::FullLogits;
    let work = [ExecutionWork::new(
        &frontier,
        TokenWork::Prefill(PrefillChunk::new(0, 3, 3).unwrap()),
        &policy,
    )
    .unwrap()];
    let (entered, mut release) = control.gate_next_head();
    let mut pending = Box::pin(executor.execute_tokens(&work));
    assert!(matches!(
        futures::poll!(pending.as_mut()),
        std::task::Poll::Pending
    ));
    tokio::time::timeout(WAIT, entered).await.unwrap().unwrap();
    drop(pending);
    let slot = Arc::clone(&executor.model);
    let owner = ExecutorOwner::new(Box::new(executor));
    let observer = owner.begin_shutdown();
    assert_eq!(observer.snapshot().phase(), ExecutorShutdownPhase::Draining);
    let mut waiting = Box::pin(observer.wait());
    assert!(matches!(
        futures::poll!(waiting.as_mut()),
        std::task::Poll::Pending
    ));
    drop(waiting);
    assert!(frontier.sequence().observed_end().is_none());
    assert!(control.stream_is_live());
    assert_eq!(control.actual_head_drops.load(Ordering::SeqCst), 0);
    // The durable observer has no second close authority. Dropping the only
    // owner and its waiting future must leave the original service responsible.
    drop(owner);
    release.now();
    retired(observer).await.result().unwrap();
    assert!(frontier.sequence().observed_end().is_some());
    assert_eq!(control.actual_forwards.load(Ordering::SeqCst), 1);
    assert_released(&slot, &control);
}
