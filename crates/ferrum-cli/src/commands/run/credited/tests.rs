use super::*;
use ferrum_interfaces::{
    output_credit::*,
    output_flow::{
        BoundedOutputError, CreditedOutputFrame, OutputConsumerControl, OutputFrameMetadata,
    },
};
use ferrum_types::{FinishReason, RequestId, TokenId, TokenUsage};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc, Mutex,
};
use tokio::sync::{mpsc, oneshot};

#[derive(Default)]
struct Consumer(AtomicUsize);
impl OutputConsumerControl for Consumer {
    fn consumer_dropped(&self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

#[derive(Clone, Default)]
struct Writer {
    bytes: Arc<Mutex<Vec<u8>>>,
    first_pointer: Arc<AtomicUsize>,
}
impl Write for Writer {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        let _ = self.first_pointer.compare_exchange(
            0,
            bytes.as_ptr() as usize,
            Ordering::Relaxed,
            Ordering::Relaxed,
        );
        self.bytes.lock().unwrap().extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn reserve(
    account: &OutputCreditAccount,
    lane: OutputCreditLane,
    amount: OutputCreditAmount,
) -> OutputReservation {
    match account.try_reserve(lane, amount).unwrap() {
        OutputCreditAttempt::Reserved(lease) => lease,
        OutputCreditAttempt::Full(_) => panic!("fixture capacity exhausted"),
    }
}

fn session_with_evidence(
    text: &[u8],
    terminal: Option<&[u8]>,
    evidence: Option<ferrum_types::InferenceExecutionEvidence>,
) -> (
    CreditedOutputSession,
    OutputCreditPool,
    Arc<Consumer>,
    usize,
) {
    let maximum = OutputCreditAmount {
        events: 4,
        bytes: 512,
        projection_bytes: 512,
    };
    let terminal_credit = OutputCreditAmount {
        events: 1,
        bytes: 128,
        projection_bytes: 0,
    };
    let pool = OutputCreditPool::new(OutputPoolLimits {
        maximum,
        max_total_bytes: 1024,
        max_open_accounts: 1,
    })
    .unwrap();
    let account = pool
        .open_request(
            RequestId::new(),
            OutputAccountLimits {
                maximum,
                terminal: terminal_credit,
            },
        )
        .unwrap();
    let (sender, receiver) = mpsc::channel(2);
    let (complete, completion) = oneshot::channel();
    let payload = text.to_vec();
    let pointer = payload.as_ptr() as usize;
    let wire = reserve(
        &account,
        OutputCreditLane::Data,
        OutputCreditAmount {
            events: 1,
            bytes: payload.capacity(),
            projection_bytes: 0,
        },
    )
    .into_output(payload);
    sender
        .try_send(CreditedOutputFrame::new(
            wire,
            OutputFrameMetadata {
                ordinal: 1,
                token: None,
                generated_tokens: 2,
                terminal: false,
            },
        ))
        .ok()
        .unwrap();
    if let Some(terminal) = terminal {
        let payload = terminal.to_vec();
        let terminal =
            reserve(&account, OutputCreditLane::Terminal, terminal_credit).into_output(payload);
        sender
            .try_send(CreditedOutputFrame::new(
                terminal,
                OutputFrameMetadata {
                    ordinal: 2,
                    token: None,
                    generated_tokens: 2,
                    terminal: true,
                },
            ))
            .ok()
            .unwrap();
    }
    let outcome = if terminal.is_some_and(|bytes| !bytes.is_empty()) {
        OutputCompletion::Failed(BoundedOutputError::new("injected failure"))
    } else {
        OutputCompletion::Succeeded {
            execution_evidence: evidence,
            history: None,
            reason: FinishReason::Length,
            usage: TokenUsage::new(1, 2),
        }
    };
    let retained = reserve(
        &account,
        OutputCreditLane::Data,
        OutputCreditAmount {
            events: 0,
            bytes: 0,
            projection_bytes: 128,
        },
    )
    .into_output(outcome);
    assert!(complete.send(retained).is_ok());
    drop(sender);
    account.close();
    let consumer = Arc::new(Consumer::default());
    (
        CreditedOutputSession::from_receivers(receiver, completion, consumer.clone()),
        pool,
        consumer,
        pointer,
    )
}

async fn bounded<T>(future: impl std::future::Future<Output = T>) -> T {
    tokio::time::timeout(std::time::Duration::from_secs(3), future)
        .await
        .expect("CLI writer made no progress")
}

#[tokio::test]
async fn credited_cli_moves_wire_without_trimming_or_duplicate_terminal_text() {
    let (session, pool, consumer, pointer) = session(" 中🙂 \n".as_bytes(), Some(b""));
    let output = Writer::default();
    let errors = Writer::default();
    let result = bounded(write_with(session, output.clone(), errors.clone()))
        .await
        .unwrap();
    assert_eq!(
        output.bytes.lock().unwrap().as_slice(),
        " 中🙂 \n".as_bytes()
    );
    assert_eq!(output.first_pointer.load(Ordering::Relaxed), pointer);
    assert!(errors.bytes.lock().unwrap().is_empty());
    assert_eq!(result.visible_frames, 1);
    assert_eq!(pool.snapshot().data_used.events, 0);
    assert_eq!(pool.snapshot().data_used.projection_bytes, 128);
    assert_eq!(consumer.0.load(Ordering::Relaxed), 0);
    drop(result);
    assert_eq!(pool.snapshot().retained_accounts, 0);
}

struct FlushGate {
    entered: Option<oneshot::Sender<()>>,
    release: std::sync::mpsc::Receiver<()>,
}
impl Write for FlushGate {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        if let Some(entered) = self.entered.take() {
            entered.send(()).unwrap();
            self.release.recv().unwrap();
        }
        Ok(())
    }
}

#[tokio::test]
async fn credited_cli_keeps_frame_charged_until_real_flush_returns() {
    let (session, pool, _, _) = session(b"held", Some(b""));
    let before = pool.snapshot().data_used;
    let (entered, entered_rx) = oneshot::channel();
    let (release, release_rx) = std::sync::mpsc::channel();
    let writer = tokio::spawn(write_with(
        session,
        FlushGate {
            entered: Some(entered),
            release: release_rx,
        },
        std::io::sink(),
    ));
    bounded(entered_rx).await.unwrap();
    assert_eq!(pool.snapshot().data_used, before);
    release.send(()).unwrap();
    drop(bounded(writer).await.unwrap().unwrap());
    assert_eq!(pool.snapshot().retained_accounts, 0);
}

struct BrokenWriter;
impl Write for BrokenWriter {
    fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
        Err(std::io::ErrorKind::BrokenPipe.into())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        unreachable!()
    }
}

#[tokio::test]
async fn credited_cli_write_failure_cancels_consumer_and_releases_queued_frames() {
    let (session, pool, consumer, _) = session(b"held", Some(b""));
    assert!(bounded(write_with(session, BrokenWriter, std::io::sink()))
        .await
        .is_err());
    assert_eq!(consumer.0.load(Ordering::Relaxed), 1);
    assert_eq!(pool.snapshot().retained_accounts, 0);
    assert_eq!(pool.snapshot().data_used, OutputCreditAmount::ZERO);
}

#[tokio::test]
async fn credited_cli_terminal_error_uses_stderr_and_preserves_failure_completion() {
    let (session, pool, consumer, _) = session(b"partial", Some(b"Error: injected failure\n"));
    let output = Writer::default();
    let errors = Writer::default();
    let result = bounded(write_with(session, output.clone(), errors.clone()))
        .await
        .unwrap();
    assert_eq!(output.bytes.lock().unwrap().as_slice(), b"partial");
    assert_eq!(
        errors.bytes.lock().unwrap().as_slice(),
        b"Error: injected failure\n"
    );
    assert!(matches!(
        result.completion.payload(),
        OutputCompletion::Failed(_)
    ));
    assert_eq!(consumer.0.load(Ordering::Relaxed), 0);
    drop(result);
    assert_eq!(pool.snapshot().retained_accounts, 0);
}

#[tokio::test]
async fn credited_cli_requires_terminal_even_if_completion_reports_success() {
    let (session, pool, consumer, _) = session(b"partial", None);
    assert!(
        bounded(write_with(session, std::io::sink(), std::io::sink()))
            .await
            .is_err()
    );
    assert_eq!(consumer.0.load(Ordering::Relaxed), 1);
    assert_eq!(pool.snapshot().retained_accounts, 0);
}

struct Engine {
    config: ferrum_types::EngineConfig,
    session: Mutex<Option<CreditedOutputSession>>,
    seen: Mutex<Option<(InferenceRequest, InferenceRequestContext)>>,
    shutdowns: AtomicUsize,
    reject: bool,
}

impl Engine {
    fn new(session: CreditedOutputSession) -> Self {
        Self {
            config: ferrum_types::EngineConfig::default(),
            session: Mutex::new(Some(session)),
            seen: Mutex::new(None),
            shutdowns: AtomicUsize::new(0),
            reject: false,
        }
    }
}

#[async_trait::async_trait]
impl ferrum_interfaces::InferenceEngine for Engine {
    async fn status(&self) -> ferrum_types::EngineStatus {
        unreachable!("transport does not inspect engine status")
    }
    async fn shutdown(&self) -> Result<()> {
        self.shutdowns.fetch_add(1, Ordering::Relaxed);
        drop(self.session.lock().unwrap().take());
        Ok(())
    }
    fn config(&self) -> &ferrum_types::EngineConfig {
        &self.config
    }
    fn metrics(&self) -> ferrum_types::EngineMetrics {
        unreachable!("transport does not inspect metrics")
    }
    async fn health_check(&self) -> ferrum_types::HealthStatus {
        unreachable!("transport does not inspect health")
    }
}

#[async_trait::async_trait]
impl LlmInferenceEngine for Engine {
    async fn infer(&self, _: InferenceRequest) -> Result<ferrum_types::InferenceResponse> {
        panic!("credited run must not fall back to legacy infer")
    }
    async fn infer_stream(
        &self,
        _: InferenceRequest,
    ) -> Result<
        std::pin::Pin<Box<dyn futures::Stream<Item = Result<ferrum_types::StreamChunk>> + Send>>,
    > {
        panic!("credited run must not fall back to a legacy stream")
    }
    async fn infer_credited_stream(
        &self,
        request: InferenceRequest,
        context: InferenceRequestContext,
        _: Arc<OutputProjectionContract>,
    ) -> Result<CreditedOutputSession> {
        *self.seen.lock().unwrap() = Some((request, context));
        if self.reject {
            return Err(FerrumError::unsupported("injected admission rejection"));
        }
        Ok(self.session.lock().unwrap().take().unwrap())
    }
}

#[tokio::test]
async fn credited_one_shot_forwards_request_and_ingress_without_legacy_collection() {
    let (session, pool, _, _) = session(b"answer\n", Some(b""));
    let engine = Engine::new(session);
    let mut request = InferenceRequest::new("prompt", "model");
    request.sampling_params.max_tokens = 19;
    let context = InferenceRequestContext::capture();
    let ingress = context.ingress();
    let id = request.id.clone();
    let output = Writer::default();
    let stats = bounded(execute_with(
        &engine,
        request,
        context,
        output.clone(),
        std::io::sink(),
    ))
    .await
    .unwrap();
    assert_eq!(stats.usage.completion_tokens, 2);
    assert_eq!(stats.visible_frames, 1);
    assert_eq!(output.bytes.lock().unwrap().as_slice(), b"answer\n");
    let seen = engine.seen.lock().unwrap();
    let (actual, context) = seen.as_ref().unwrap();
    assert_eq!(actual.id, id);
    assert_eq!(actual.sampling_params.max_tokens, 19);
    assert!(actual.stream);
    assert_eq!(context.ingress(), ingress);
    assert_eq!(engine.shutdowns.load(Ordering::Relaxed), 1);
    assert_eq!(pool.snapshot().retained_accounts, 0);
}

#[tokio::test]
async fn credited_one_shot_rejection_shuts_down_without_legacy_retry() {
    let (session, pool, _, _) = session(b"unused", Some(b""));
    let mut engine = Engine::new(session);
    engine.reject = true;
    assert!(bounded(execute_with(
        &engine,
        InferenceRequest::new("prompt", "model"),
        InferenceRequestContext::capture(),
        std::io::sink(),
        std::io::sink(),
    ))
    .await
    .is_err());
    assert_eq!(engine.shutdowns.load(Ordering::Relaxed), 1);
    assert_eq!(pool.snapshot().retained_accounts, 0);
}

#[tokio::test]
async fn credited_one_shot_write_failure_shuts_down_and_cancels() {
    let (session, pool, consumer, _) = session(b"answer", Some(b""));
    let engine = Engine::new(session);
    assert!(bounded(execute_with(
        &engine,
        InferenceRequest::new("prompt", "model"),
        InferenceRequestContext::capture(),
        BrokenWriter,
        std::io::sink(),
    ))
    .await
    .is_err());
    assert_eq!(consumer.0.load(Ordering::Relaxed), 1);
    assert_eq!(engine.shutdowns.load(Ordering::Relaxed), 1);
    assert_eq!(pool.snapshot().retained_accounts, 0);
}

#[test]
fn credited_cli_options_require_a_supported_projection_before_model_loading() {
    assert!(validate_options(true, super::super::OutputFormat::Text).is_ok());
    assert!(validate_options(false, super::super::OutputFormat::Text).is_err());
    assert!(validate_options(true, super::super::OutputFormat::Jsonl).is_err());
}

fn session(
    text: &[u8],
    terminal: Option<&[u8]>,
) -> (
    CreditedOutputSession,
    OutputCreditPool,
    Arc<Consumer>,
    usize,
) {
    session_with_evidence(text, terminal, None)
}
#[tokio::test]
async fn credited_run_latency_and_kernel_write_full_commit_evidence_without_legacy_collection() {
    for detail in [
        crate::observability_product::ProfileDetailArg::Latency,
        crate::observability_product::ProfileDetailArg::Kernel,
    ] {
        let dir = tempfile::tempdir().unwrap();
        let profile = dir.path().join("profile.jsonl");
        let config = crate::observability_product::ProductObservabilityConfig::new(
            ferrum_types::ProfileEntrypoint::Run,
            "model",
            Some(&profile),
            detail,
            None,
            None,
            None,
            1.0,
        );
        let evidence = ferrum_types::InferenceExecutionEvidence {
            prompt_token_ids: vec![],
            output_token_ids: vec![TokenId::new(1), TokenId::new(2)],
            engine_token_timing: Some(ferrum_types::EngineTokenTimingEvidence {
                clock_source: "rust_std_instant".into(),
                wall_anchor_unix_nanos: 1,
                wall_anchor_max_error_nanos: 0,
                decode_ready_nanos_since_request_start: Some(1000),
                token_commit_nanos_since_request_start: vec![1000, 2000],
                decode_stage_intervals: vec![],
                decode_stage_intervals_omitted: 7,
            }),
        };
        let (session, pool, _, _) = session_with_evidence(b"answer", Some(b""), Some(evidence));
        let engine = Engine::new(session);
        let mut request = InferenceRequest::new("prompt", "model");
        request.sampling_params.max_tokens = 2;
        let stats = bounded(execute_observed(
            &engine,
            request,
            InferenceRequestContext::capture(),
            Writer::default(),
            std::io::sink(),
            Some(&config),
        ))
        .await
        .unwrap();
        assert_eq!(stats.usage.completion_tokens, 2);
        assert!(
            engine
                .seen
                .lock()
                .unwrap()
                .as_ref()
                .unwrap()
                .0
                .evidence_request
                .capture_engine_token_timing
        );
        let event: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(profile).unwrap()).unwrap();
        assert_eq!(event["attributes"]["engine_token_commit_count"], 2);
        assert_eq!(
            event["attributes"]["engine_decode_stage_intervals_omitted"],
            7
        );
        assert_eq!(event["attributes"]["itl_nanos"], serde_json::json!([1000]));
        assert_eq!(pool.snapshot().retained_accounts, 0);
    }
}
