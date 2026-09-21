use std::fmt;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use tokio::sync::{mpsc, oneshot};

use super::vnext_timing::AtomicDurationMetrics;

const COMPLETION_WORKER_QUEUE_CAPACITY: usize = 1;

type VNextCompletionTask = Box<dyn FnOnce() + Send + 'static>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum VNextCompletionTaskKind {
    WaveReadback,
    PostSubmitDrain,
    IndeterminateRecovery,
    CheckpointTransfer,
}

impl VNextCompletionTaskKind {
    const fn as_str(self) -> &'static str {
        match self {
            Self::WaveReadback => "wave_readback",
            Self::PostSubmitDrain => "post_submit_drain",
            Self::IndeterminateRecovery => "indeterminate_recovery",
            Self::CheckpointTransfer => "checkpoint_transfer",
        }
    }
}

#[derive(Default)]
struct VNextCompletionTaskClassMetrics {
    scheduled_tasks: AtomicU64,
    completed_tasks: AtomicU64,
    reservation_wait: AtomicDurationMetrics,
    queued_wait: AtomicDurationMetrics,
    task_run: AtomicDurationMetrics,
}

impl VNextCompletionTaskClassMetrics {
    fn snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "scheduled_tasks": self.scheduled_tasks.load(Ordering::Relaxed),
            "completed_tasks": self.completed_tasks.load(Ordering::Relaxed),
            "reservation_wait": self.reservation_wait.snapshot(),
            "queued_wait": self.queued_wait.snapshot(),
            "task_run": self.task_run.snapshot(),
        })
    }

    fn reset(&self) {
        self.scheduled_tasks.store(0, Ordering::Relaxed);
        self.completed_tasks.store(0, Ordering::Relaxed);
        self.reservation_wait.reset();
        self.queued_wait.reset();
        self.task_run.reset();
    }
}

#[derive(Default)]
struct VNextCompletionWorkerCounters {
    scheduled_tasks: AtomicU64,
    completed_tasks: AtomicU64,
    panicked_tasks: AtomicU64,
    pending_tasks: AtomicUsize,
    wave_readback: VNextCompletionTaskClassMetrics,
    post_submit_drain: VNextCompletionTaskClassMetrics,
    indeterminate_recovery: VNextCompletionTaskClassMetrics,
    checkpoint_transfer: VNextCompletionTaskClassMetrics,
}

impl VNextCompletionWorkerCounters {
    fn task_class(&self, kind: VNextCompletionTaskKind) -> &VNextCompletionTaskClassMetrics {
        match kind {
            VNextCompletionTaskKind::WaveReadback => &self.wave_readback,
            VNextCompletionTaskKind::PostSubmitDrain => &self.post_submit_drain,
            VNextCompletionTaskKind::IndeterminateRecovery => &self.indeterminate_recovery,
            VNextCompletionTaskKind::CheckpointTransfer => &self.checkpoint_transfer,
        }
    }
}

struct VNextCompletionTaskGuard {
    counters: Arc<VNextCompletionWorkerCounters>,
}

impl Drop for VNextCompletionTaskGuard {
    fn drop(&mut self) {
        let previous = self.counters.pending_tasks.fetch_sub(1, Ordering::AcqRel);
        debug_assert!(
            previous > 0,
            "vNext completion worker pending count underflow"
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum VNextCompletionWorkerError {
    QueueClosed,
    ResultChannelClosed,
    TaskPanicked,
}

impl fmt::Display for VNextCompletionWorkerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::QueueClosed => "completion worker queue is closed",
            Self::ResultChannelClosed => "completion worker result channel is closed",
            Self::TaskPanicked => "completion worker task panicked",
        })
    }
}

impl std::error::Error for VNextCompletionWorkerError {}

/// Reserved queue capacity. Holding this before device submission lets the
/// caller hand cleanup to the worker synchronously after submission succeeds.
/// The permit borrows its worker, so dropping an idle worker cannot join a
/// thread whose channel is still kept open by an escaped reservation.
#[must_use = "a reservation must be submitted or dropped to release queue capacity"]
pub(super) struct VNextCompletionReservation<'a> {
    permit: mpsc::Permit<'a, VNextCompletionTask>,
    counters: Arc<VNextCompletionWorkerCounters>,
    reservation_wait: Duration,
}

/// Owns result observation, not the worker task. Dropping this ticket leaves
/// the accepted task and its device-resource retention on the worker.
#[must_use = "dropping a completion ticket abandons its result, not device cleanup"]
pub(super) struct VNextCompletionTicket<T> {
    receiver: oneshot::Receiver<Result<T, VNextCompletionWorkerError>>,
}

impl<T> VNextCompletionTicket<T> {
    pub(super) async fn wait(self) -> Result<T, VNextCompletionWorkerError> {
        self.receiver
            .await
            .map_err(|_| VNextCompletionWorkerError::ResultChannelClosed)?
    }
}

impl VNextCompletionReservation<'_> {
    pub(super) fn submit<T, F>(
        self,
        kind: VNextCompletionTaskKind,
        task: F,
    ) -> VNextCompletionTicket<T>
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        let Self {
            permit,
            counters,
            reservation_wait,
        } = self;
        let class = counters.task_class(kind);
        class.reservation_wait.record(reservation_wait);
        class.scheduled_tasks.fetch_add(1, Ordering::Relaxed);
        counters.scheduled_tasks.fetch_add(1, Ordering::Relaxed);
        counters.pending_tasks.fetch_add(1, Ordering::AcqRel);
        let guard = VNextCompletionTaskGuard {
            counters: Arc::clone(&counters),
        };
        let (result_sender, receiver) = oneshot::channel();
        let queued_at = Instant::now();
        let job = Box::new(move || {
            let class = counters.task_class(kind);
            class.queued_wait.record(queued_at.elapsed());
            let task_started = Instant::now();
            let result = catch_unwind(AssertUnwindSafe(task)).map_err(|_| {
                counters.panicked_tasks.fetch_add(1, Ordering::Relaxed);
                VNextCompletionWorkerError::TaskPanicked
            });
            class.task_run.record(task_started.elapsed());
            class.completed_tasks.fetch_add(1, Ordering::Relaxed);
            counters.completed_tasks.fetch_add(1, Ordering::Relaxed);
            drop(guard);
            let _ = result_sender.send(result);
        }) as VNextCompletionTask;
        permit.send(job);
        VNextCompletionTicket { receiver }
    }
}

/// One executor-owned blocking boundary for device fences and readbacks.
/// Requests enqueue through a bounded async channel; no request or token creates
/// an OS thread, and dropping an awaiting request does not cancel device cleanup.
pub(super) struct VNextCompletionWorker {
    sender: Option<mpsc::Sender<VNextCompletionTask>>,
    thread: Option<JoinHandle<()>>,
    counters: Arc<VNextCompletionWorkerCounters>,
    name: String,
}

impl fmt::Debug for VNextCompletionWorker {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("VNextCompletionWorker")
            .field("name", &self.name)
            .field("pending_tasks", &self.pending_tasks())
            .finish_non_exhaustive()
    }
}

impl VNextCompletionWorker {
    pub(super) fn new() -> std::io::Result<Self> {
        static NEXT_WORKER_ID: AtomicU64 = AtomicU64::new(1);
        let worker_id = NEXT_WORKER_ID.fetch_add(1, Ordering::Relaxed);
        let name = format!("ferrum-vnext-completion-{worker_id}");
        let (sender, mut receiver) =
            mpsc::channel::<VNextCompletionTask>(COMPLETION_WORKER_QUEUE_CAPACITY);
        let thread = thread::Builder::new().name(name.clone()).spawn(move || {
            while let Some(task) = receiver.blocking_recv() {
                // Each submitted task reports its own panic. This outer boundary
                // also keeps the worker alive if task bookkeeping regresses.
                let _ = catch_unwind(AssertUnwindSafe(task));
            }
        })?;
        Ok(Self {
            sender: Some(sender),
            thread: Some(thread),
            counters: Arc::new(VNextCompletionWorkerCounters::default()),
            name,
        })
    }

    pub(super) async fn reserve(
        &self,
    ) -> Result<VNextCompletionReservation<'_>, VNextCompletionWorkerError> {
        let sender = self
            .sender
            .as_ref()
            .ok_or(VNextCompletionWorkerError::QueueClosed)?;
        let reservation_started = Instant::now();
        let permit = sender
            .reserve()
            .await
            .map_err(|_| VNextCompletionWorkerError::QueueClosed)?;
        Ok(VNextCompletionReservation {
            permit,
            counters: Arc::clone(&self.counters),
            reservation_wait: reservation_started.elapsed(),
        })
    }

    pub(super) async fn execute<T, F>(
        &self,
        kind: VNextCompletionTaskKind,
        task: F,
    ) -> Result<T, VNextCompletionWorkerError>
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        self.reserve().await?.submit(kind, task).wait().await
    }

    fn pending_tasks(&self) -> usize {
        self.counters.pending_tasks.load(Ordering::Acquire)
    }

    pub(super) fn metrics_snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "worker_threads": 1,
            "queue_capacity": COMPLETION_WORKER_QUEUE_CAPACITY,
            "pending_tasks": self.pending_tasks(),
            "scheduled_tasks": self.counters.scheduled_tasks.load(Ordering::Relaxed),
            "completed_tasks": self.counters.completed_tasks.load(Ordering::Relaxed),
            "panicked_tasks": self.counters.panicked_tasks.load(Ordering::Relaxed),
            "task_classes": {
                VNextCompletionTaskKind::WaveReadback.as_str(): self.counters.wave_readback.snapshot(),
                VNextCompletionTaskKind::PostSubmitDrain.as_str(): self.counters.post_submit_drain.snapshot(),
                VNextCompletionTaskKind::IndeterminateRecovery.as_str(): self.counters.indeterminate_recovery.snapshot(),
                VNextCompletionTaskKind::CheckpointTransfer.as_str(): self.counters.checkpoint_transfer.snapshot(),
            },
        })
    }

    pub(super) fn reset_metrics_if_idle(&self) -> bool {
        if self.pending_tasks() != 0 {
            return false;
        }
        self.counters.scheduled_tasks.store(0, Ordering::Relaxed);
        self.counters.completed_tasks.store(0, Ordering::Relaxed);
        self.counters.panicked_tasks.store(0, Ordering::Relaxed);
        self.counters.wave_readback.reset();
        self.counters.post_submit_drain.reset();
        self.counters.indeterminate_recovery.reset();
        self.counters.checkpoint_transfer.reset();
        true
    }
}

impl Drop for VNextCompletionWorker {
    fn drop(&mut self) {
        drop(self.sender.take());
        let Some(thread) = self.thread.take() else {
            return;
        };
        if self.pending_tasks() == 0 {
            let _ = thread.join();
        } else {
            // An in-flight fence owns its reaper Arc inside the task. Detaching
            // avoids blocking engine Drop while retaining resource authority
            // until the backend wait returns. Each executor can detach at most
            // this one worker thread.
            drop(thread);
        }
    }
}

#[cfg(test)]
#[path = "vnext_completion_worker/retained_step_tests.rs"]
mod retained_step_tests;

#[cfg(test)]
mod tests {
    use super::{VNextCompletionTaskKind, VNextCompletionWorker, VNextCompletionWorkerError};
    use std::future::Future;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::task::Poll;
    use std::time::Duration;

    #[tokio::test]
    async fn reservation_is_bounded_and_drop_releases_capacity_without_a_task() {
        let worker = VNextCompletionWorker::new().unwrap();
        let reservation = worker.reserve().await.unwrap();
        let mut blocked = Box::pin(worker.reserve());
        let polled = std::future::poll_fn(|cx| Poll::Ready(blocked.as_mut().poll(cx))).await;
        assert!(polled.is_pending());
        assert_eq!(worker.pending_tasks(), 0);
        assert_eq!(worker.metrics_snapshot()["scheduled_tasks"], 0);
        drop(reservation);

        let reservation = blocked.await.unwrap();
        // Classification happens at synchronous submission, so an already
        // reserved wave can schedule its recovery without reserving again.
        let ticket = reservation.submit(VNextCompletionTaskKind::IndeterminateRecovery, || 17);
        assert_eq!(ticket.wait().await.unwrap(), 17);
        let metrics = worker.metrics_snapshot();
        assert_eq!(metrics["scheduled_tasks"], 1);
        assert_eq!(
            metrics["task_classes"]["wave_readback"]["scheduled_tasks"],
            0
        );
        assert_eq!(
            metrics["task_classes"]["indeterminate_recovery"]["scheduled_tasks"],
            1
        );
    }

    #[tokio::test]
    async fn dropped_ticket_and_worker_keep_task_ownership_until_terminal_event() {
        struct RetainedStep {
            terminal: Arc<AtomicBool>,
            released: Option<tokio::sync::oneshot::Sender<bool>>,
        }
        impl Drop for RetainedStep {
            fn drop(&mut self) {
                let _ = self
                    .released
                    .take()
                    .unwrap()
                    .send(self.terminal.load(Ordering::Acquire));
            }
        }

        let worker = VNextCompletionWorker::new().unwrap();
        let terminal = Arc::new(AtomicBool::new(false));
        let (released_sender, mut released_receiver) = tokio::sync::oneshot::channel();
        let retained = RetainedStep {
            terminal: Arc::clone(&terminal),
            released: Some(released_sender),
        };
        let (started_sender, started_receiver) = tokio::sync::oneshot::channel();
        let (finish_sender, finish_receiver) = std::sync::mpsc::channel();
        let ticket = worker.reserve().await.unwrap().submit(
            VNextCompletionTaskKind::WaveReadback,
            move || {
                let _ = started_sender.send(());
                finish_receiver.recv().unwrap();
                terminal.store(true, Ordering::Release);
                retained
            },
        );
        started_receiver.await.unwrap();
        drop(ticket);
        drop(worker);
        assert!(matches!(
            released_receiver.try_recv(),
            Err(tokio::sync::oneshot::error::TryRecvError::Empty)
        ));
        finish_sender.send(()).unwrap();
        assert!(released_receiver.await.unwrap());
    }

    #[tokio::test]
    async fn ticket_reports_task_panic_and_worker_accepts_the_next_task() {
        let worker = VNextCompletionWorker::new().unwrap();
        let ticket =
            worker
                .reserve()
                .await
                .unwrap()
                .submit(VNextCompletionTaskKind::WaveReadback, || {
                    panic!("injected completion task panic");
                });
        assert_eq!(
            ticket.wait().await,
            Err(VNextCompletionWorkerError::TaskPanicked)
        );
        assert_eq!(
            worker
                .execute(VNextCompletionTaskKind::PostSubmitDrain, || 23)
                .await
                .unwrap(),
            23
        );
        let metrics = worker.metrics_snapshot();
        assert_eq!(metrics["pending_tasks"], 0);
        assert_eq!(metrics["completed_tasks"], 2);
        assert_eq!(metrics["panicked_tasks"], 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn worker_is_single_threaded_bounded_and_cancellation_safe() {
        let worker = Arc::new(VNextCompletionWorker::new().unwrap());
        let (started_sender, started_receiver) = tokio::sync::oneshot::channel();
        let (release_sender, release_receiver) = std::sync::mpsc::channel();
        let cancelled_worker = Arc::clone(&worker);
        let cancelled = tokio::spawn(async move {
            cancelled_worker
                .execute(VNextCompletionTaskKind::WaveReadback, move || {
                    let _ = started_sender.send(());
                    release_receiver.recv().unwrap();
                    7_u64
                })
                .await
        });
        started_receiver.await.unwrap();
        cancelled.abort();
        assert!(cancelled.await.unwrap_err().is_cancelled());

        let queued_worker = Arc::clone(&worker);
        let queued = tokio::spawn(async move {
            queued_worker
                .execute(VNextCompletionTaskKind::PostSubmitDrain, || 8_u64)
                .await
                .unwrap()
        });
        tokio::time::timeout(Duration::from_secs(1), async {
            while worker.pending_tasks() != 2 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        let blocked_worker = Arc::clone(&worker);
        let (blocked_started_sender, blocked_started_receiver) = tokio::sync::oneshot::channel();
        let blocked = tokio::spawn(async move {
            let _ = blocked_started_sender.send(());
            blocked_worker
                .execute(VNextCompletionTaskKind::IndeterminateRecovery, || 9_u64)
                .await
        });
        blocked_started_receiver.await.unwrap();
        tokio::task::yield_now().await;
        blocked.abort();
        assert!(blocked.await.unwrap_err().is_cancelled());
        assert_eq!(worker.metrics_snapshot()["scheduled_tasks"], 2);
        assert_eq!(worker.pending_tasks(), 2);

        release_sender.send(()).unwrap();
        assert_eq!(queued.await.unwrap(), 8);
        tokio::time::timeout(Duration::from_secs(1), async {
            while worker.pending_tasks() != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        let active = Arc::new(AtomicUsize::new(0));
        let maximum_active = Arc::new(AtomicUsize::new(0));
        let mut tasks = Vec::new();
        for _ in 0..32 {
            let worker = Arc::clone(&worker);
            let active = Arc::clone(&active);
            let maximum_active = Arc::clone(&maximum_active);
            tasks.push(tokio::spawn(async move {
                worker
                    .execute(VNextCompletionTaskKind::WaveReadback, move || {
                        let now = active.fetch_add(1, Ordering::AcqRel) + 1;
                        maximum_active.fetch_max(now, Ordering::AcqRel);
                        std::thread::sleep(Duration::from_millis(2));
                        active.fetch_sub(1, Ordering::AcqRel);
                        std::thread::current().name().unwrap().to_owned()
                    })
                    .await
                    .unwrap()
            }));
        }
        let mut names = Vec::new();
        for task in tasks {
            names.push(task.await.unwrap());
        }
        assert_eq!(maximum_active.load(Ordering::Acquire), 1);
        assert!(names.iter().all(|name| name == &names[0]));

        let worker_name = worker
            .execute(VNextCompletionTaskKind::IndeterminateRecovery, || {
                std::thread::current().name().unwrap().to_owned()
            })
            .await
            .unwrap();
        assert_eq!(worker_name, names[0]);

        let metrics = worker.metrics_snapshot();
        assert_eq!(metrics["worker_threads"], 1);
        assert_eq!(metrics["queue_capacity"], 1);
        assert_eq!(metrics["pending_tasks"], 0);
        assert_eq!(metrics["scheduled_tasks"], 35);
        assert_eq!(metrics["completed_tasks"], 35);
        assert_eq!(metrics["panicked_tasks"], 0);
        assert_eq!(
            metrics["task_classes"]["wave_readback"]["scheduled_tasks"],
            33
        );
        assert_eq!(
            metrics["task_classes"]["wave_readback"]["completed_tasks"],
            33
        );
        assert_eq!(
            metrics["task_classes"]["post_submit_drain"]["scheduled_tasks"],
            1
        );
        assert_eq!(
            metrics["task_classes"]["post_submit_drain"]["completed_tasks"],
            1
        );
        assert_eq!(
            metrics["task_classes"]["indeterminate_recovery"]["scheduled_tasks"],
            1
        );
        assert_eq!(
            metrics["task_classes"]["indeterminate_recovery"]["completed_tasks"],
            1
        );
        assert_eq!(
            metrics["task_classes"]["wave_readback"]["task_run"]["samples"],
            33
        );
    }
}
