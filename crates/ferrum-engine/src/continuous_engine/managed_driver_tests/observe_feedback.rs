//! Identical resident ingress behind the actual numerical executor's gate.
//! The independent CPU oracle checks logits and KV; a bounded invocation witness
//! compares real work/group/order in Off and collecting-only Observe assembly.

use super::*;
use ferrum_execution::execution_attempt::*;
use ferrum_slo::{
    learning::{LearningConfig, LearningEstimator},
    model::{ExecutionIdentity, ModelFamily},
    stopping::{StoppingCollectorConfig, StoppingEstimatorOwner},
};

#[derive(Debug, PartialEq, Eq)]
struct Output {
    text: String,
    tokens: Vec<TokenId>,
    reason: FinishReason,
    usage: (usize, usize),
}

#[derive(Debug, PartialEq, Eq)]
struct Run {
    work: Vec<Vec<(RequestId, TokenWork)>>,
    outputs: Vec<Output>,
}

#[derive(Debug, Clone, Copy)]
enum Mode {
    Off,
    Learning(ModelFamily),
    FullQueue,
    Panic,
}

struct FullQueue {
    sender: crossbeam_channel::Sender<ExecutionAttemptSummary>,
    _receiver: crossbeam_channel::Receiver<ExecutionAttemptSummary>,
    dropped: AtomicU64,
}

impl ExecutionAttemptObserver for FullQueue {
    fn limits(&self) -> ExecutionAttemptLimits {
        ExecutionAttemptLimits {
            max_participants: 3,
            max_summary_bytes: 4096,
        }
    }
    fn try_record(&self, summary: &ExecutionAttemptSummary) -> ExecutionAttemptDelivery {
        assert!(self.limits().accepts(summary.participants.len()));
        match self.sender.try_send(summary.clone()) {
            Ok(()) => ExecutionAttemptDelivery::Recorded,
            Err(crossbeam_channel::TrySendError::Full(_)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                ExecutionAttemptDelivery::Dropped(ExecutionAttemptDrop::QueueFull)
            }
            Err(error) => panic!("test collector unexpectedly disconnected: {error}"),
        }
    }
    fn record_drop(&self, reason: ExecutionAttemptDrop) {
        panic!("bounded three-request cohort fits projection limits: {reason:?}")
    }
}

fn stream_output(chunks: Vec<Result<ferrum_output::OutputChunk>>) -> Output {
    let chunks = chunks.into_iter().collect::<Result<Vec<_>>>().unwrap();
    let terminal = chunks.last().unwrap().chunk();
    let usage = terminal.usage.as_ref().unwrap();
    Output {
        text: chunks
            .iter()
            .map(|chunk| chunk.chunk().text.as_str())
            .collect(),
        tokens: chunks
            .iter()
            .filter_map(|chunk| chunk.chunk().token)
            .collect(),
        reason: terminal.finish_reason.unwrap(),
        usage: (usage.prompt_tokens, usage.completion_tokens),
    }
}

async fn run(mode: Mode, requests: &[InferenceRequest; 3]) -> Run {
    let learner = if let Mode::Learning(family) = mode {
        let mut config = LearningConfig {
            window_samples: 16,
            training_samples: 2,
            validation_samples: 1,
            calibration_samples: 1,
            min_calibration_samples: 1,
            retrain_every: 1,
            ..Default::default()
        };
        config.training.max_trees = 2;
        config.training.ridge_sweeps = 8;
        Some(
            LearningEstimator::spawn(
                config,
                ExecutionIdentity::new("observe-equality/actual-tiny-cpu/split/no-prefix".into())
                    .unwrap(),
                family,
            )
            .unwrap(),
        )
    } else {
        None
    };
    let stopping = learner
        .as_ref()
        .map(|_| StoppingEstimatorOwner::spawn(StoppingCollectorConfig::default()).unwrap());
    let (sender, receiver) = crossbeam_channel::bounded(1);
    let full = Arc::new(FullQueue {
        sender,
        _receiver: receiver,
        dropped: AtomicU64::new(0),
    });
    let attempts: Option<Arc<dyn ExecutionAttemptObserver>> = match mode {
        Mode::Off => None,
        Mode::Learning(_) => Some(learner.as_ref().unwrap().observer()),
        Mode::FullQueue => Some(full.clone()),
        Mode::Panic => {
            let mut observer =
                crate::continuous_engine::attempt_observation_tests::AttemptRecorder::default();
            observer.panic_on_record = true;
            Some(Arc::new(observer))
        }
    };
    let mut fixture = fixture::Fixture::with_feedback(
        |config| {
            config.scheduler.max_running_requests = 3;
            config.batching.max_batch_size = 3;
            config.batching.max_num_batched_tokens = 32;
            config.runtime.profile_detail = ObservabilityProfileDetail::Off;
        },
        attempts,
        stopping.as_ref().map(|owner| owner.observer()),
    );
    {
        let inner = fixture.engine.active_inner().unwrap();
        assert!(inner.config.scheduler.slo.is_none());
        assert!(
            inner.cost_snapshot_source.is_none(),
            "Observe has no policy read path"
        );
    }
    let first = fixture
        .engine
        .infer_stream(requests[0].clone())
        .await
        .unwrap();
    tokio::time::timeout(Duration::from_secs(5), &mut fixture.reached)
        .await
        .unwrap()
        .unwrap();
    let mut second = Box::pin(fixture.engine.infer(requests[1].clone()));
    pending_once(second.as_mut()).await;
    let third = fixture
        .engine
        .infer_stream(requests[2].clone())
        .await
        .unwrap();
    assert_eq!(fixture.engine.active_inner().unwrap().sequences.len(), 3);
    let mut work = fixture.executor.take_work_calls();
    assert_eq!(work.len(), 1, "first actual prefill remains gated");
    fixture.release.send(()).unwrap();
    let (response, first, third) = tokio::time::timeout(Duration::from_secs(10), async {
        tokio::join!(second, first.collect::<Vec<_>>(), third.collect::<Vec<_>>())
    })
    .await
    .unwrap();
    let response = response.unwrap();
    let value = response.response();
    let second = Output {
        text: value.text.clone(),
        tokens: value.tokens.clone(),
        reason: value.finish_reason,
        usage: (value.usage.prompt_tokens, value.usage.completion_tokens),
    };
    drop(response);
    work.extend(fixture.executor.take_work_calls());
    assert!(
        work.iter().any(|group| group.len() > 1),
        "compare actual batched resident work"
    );
    assert!(work
        .iter()
        .flatten()
        .any(|(_, work)| matches!(work, TokenWork::Decode { .. })));
    fixture.engine.shutdown().await.unwrap();
    fixture.engine.wait_for_shutdown_completion().await.unwrap();
    if let Some(owner) = learner {
        let exit = owner.shutdown().await.unwrap();
        assert!(exit.stats.trained_samples > 0);
        assert!(
            exit.stats.model_fits > 0,
            "real {mode:?} samples must fit a base model"
        );
        assert_eq!(exit.stats.queued_records, 0);
        let predictions = owner.prediction_stats();
        assert_eq!(predictions.unprepared_completions, 0);
        assert!(
            predictions
                .domains
                .iter()
                .map(|domain| domain.prepared)
                .sum::<u64>()
                > 0
        );
        assert!(
            predictions
                .domains
                .iter()
                .map(|domain| domain.completed)
                .sum::<u64>()
                > 0
        );
        assert!(predictions
            .domains
            .iter()
            .all(|domain| domain.mismatched_or_noncausal == 0));
    }
    if let Some(owner) = stopping {
        let exit = owner.shutdown().await.unwrap();
        assert_eq!(exit.window.admitted, 3);
        assert_eq!(exit.window.unobserved_lifetimes, 0);
    }
    if matches!(mode, Mode::FullQueue) {
        assert!(full.dropped.load(Ordering::Relaxed) > 0);
    }
    Run {
        work,
        outputs: vec![stream_output(first), second, stream_output(third)],
    }
}

#[tokio::test]
async fn observe_and_off_have_identical_actual_resident_work_for_stream_and_nonstream() {
    let requests = [
        "same real numerical prompt",
        "same real prompt",
        "same real numerical prompt same real numerical prompt",
    ]
    .map(|prompt| {
        let mut request = InferenceRequest::new(prompt, "tiny-llama-test");
        request.sampling_params.max_tokens = 8;
        request.sampling_params.temperature = 0.0;
        request
            .metadata
            .insert("ferrum_ignore_eos".into(), true.into());
        request
    });
    let baseline = run(Mode::Off, &requests).await;
    for mode in [
        Mode::Learning(ModelFamily::Physical),
        Mode::Learning(ModelFamily::Gbdt),
        Mode::FullQueue,
        Mode::Panic,
    ] {
        assert_eq!(
            run(mode, &requests).await,
            baseline,
            "actual decisions/output changed in {mode:?}"
        );
    }
}
