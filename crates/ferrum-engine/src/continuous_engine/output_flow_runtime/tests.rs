use super::*;
use ferrum_interfaces::{
    output_credit::{OutputAccountLimits, OutputCreditAmount, OutputCreditPool, OutputPoolLimits},
    output_flow::{OutputHistory, OutputProjectionContract, RequestOutputPlan},
    tokenizer::{BoundedDecodeBound, DecodedTextBound, TokenizerInfo},
    Tokenizer,
};
use ferrum_types::{FinishReason, InferenceRequest, SpecialTokens, TokenUsage};
use futures::{Future, StreamExt};
use std::{num::NonZeroUsize, time::Duration};

struct BoundedTokenizer(SpecialTokens);
impl Tokenizer for BoundedTokenizer {
    fn encode(&self, _: &str, _: bool) -> ferrum_types::Result<Vec<TokenId>> {
        unreachable!()
    }
    fn decode(&self, _: &[TokenId], _: bool) -> ferrum_types::Result<String> {
        unreachable!()
    }
    fn decode_incremental(&self, _: &[TokenId], _: TokenId) -> ferrum_types::Result<String> {
        unreachable!()
    }
    fn vocab_size(&self) -> usize {
        16
    }
    fn special_tokens(&self) -> &SpecialTokens {
        &self.0
    }
    fn token_id(&self, _: &str) -> Option<TokenId> {
        None
    }
    fn token_text(&self, _: TokenId) -> Option<&str> {
        None
    }
    fn info(&self) -> TokenizerInfo {
        unreachable!()
    }
    fn decoded_text_bound(&self) -> Option<DecodedTextBound> {
        Some(DecodedTextBound::new(NonZeroUsize::new(6).unwrap()))
    }
    fn bounded_decode_bound(&self) -> Option<BoundedDecodeBound> {
        Some(BoundedDecodeBound::new(NonZeroUsize::new(6).unwrap(), 0))
    }
    fn bounded_token_bytes_bound(&self) -> Option<NonZeroUsize> {
        NonZeroUsize::new(6)
    }
}

fn plan() -> RequestOutputPlan {
    let mut request = InferenceRequest::new("prompt", "test-model");
    request.sampling_params.max_tokens = 4;
    RequestOutputPlan::derive(
        Arc::new(OutputProjectionContract::cli_text()),
        &BoundedTokenizer(SpecialTokens::default()),
        &request,
        3,
    )
    .unwrap()
}

fn limits(plan: &RequestOutputPlan, data_events: usize) -> OutputAccountLimits {
    OutputAccountLimits {
        maximum: OutputCreditAmount {
            events: plan.terminal_credit().events + data_events,
            bytes: plan.lifetime_wire_bytes() + plan.terminal_credit().bytes,
            projection_bytes: plan.retained_projection_bytes(),
        },
        terminal: plan.terminal_credit(),
    }
}

fn pool(data_events: usize) -> OutputCreditPool {
    let maximum = limits(&plan(), data_events).maximum;
    OutputCreditPool::new(OutputPoolLimits {
        maximum: OutputCreditAmount {
            events: maximum.events * 2,
            bytes: maximum.bytes * 2,
            projection_bytes: maximum.projection_bytes * 2,
        },
        max_total_bytes: maximum.total_bytes().unwrap() * 2,
        max_open_accounts: 2,
    })
    .unwrap()
}

fn start(pool: &OutputCreditPool, data_events: usize) -> (OutputFlowPort, CreditedOutputSession) {
    let plan = plan();
    // These backpressure fixtures intentionally use one data queue slot,
    // independently of event leases already retained by a consumer.
    let config = SloOutputConfig {
        max_queued_events_per_request: NonZeroUsize::new(plan.terminal_credit().events + 1)
            .unwrap(),
        ..SloOutputConfig::default()
    };
    let budget = RequestOutputBudget::open(pool, limits(&plan, data_events), plan).unwrap();
    spawn_output_flow_runtime(
        budget,
        Arc::new(Notify::new()),
        OutputFlowRuntimeOptions::from_config(&config),
    )
}

async fn bounded<T>(future: impl Future<Output = T>) -> T {
    // A failure guard only. Synchronization below uses notifications/receivers.
    tokio::time::timeout(Duration::from_secs(3), future)
        .await
        .expect("output owner made no progress")
}

async fn state(port: &OutputFlowPort, wanted: OutputReadinessState) {
    bounded(async {
        let mut changed = port.subscribe();
        while port.readiness() != wanted {
            changed.changed().await.unwrap();
        }
    })
    .await;
}

async fn ready(port: &OutputFlowPort) -> ReadyOutputGrant {
    state(port, OutputReadinessState::Ready).await;
    match port.try_take() {
        OutputReadiness::Ready(grant) => grant,
        _ => panic!("single consumer lost ready grant"),
    }
}

fn delta(text: &str, generated_tokens: usize) -> OutputDelta {
    OutputDelta {
        text: text.to_owned(),
        token: Some(TokenId(generated_tokens as u32)),
        generated_tokens,
        created: 7,
    }
}

fn terminal(through: u64, generated_tokens: usize, final_text: &str) -> OutputTerminalDecision {
    OutputTerminalDecision {
        through_output_ordinal: through,
        outcome: OutputCompletion::Succeeded {
            execution_evidence: None,
            history: None,
            reason: FinishReason::Stop,
            usage: TokenUsage {
                prompt_tokens: 3,
                completion_tokens: generated_tokens,
                total_tokens: 3 + generated_tokens,
            },
        },
        final_text: final_text.to_owned(),
        created: 8,
    }
}

async fn drained(pool: &OutputCreditPool) {
    bounded(async {
        let mut wake = pool.subscribe();
        while pool.snapshot().retained_accounts != 0 {
            wake.changed().await.unwrap();
        }
    })
    .await;
    let snapshot = pool.snapshot();
    assert_eq!(snapshot.data_used, OutputCreditAmount::ZERO);
    assert_eq!(snapshot.terminal_held, OutputCreditAmount::ZERO);
}

#[tokio::test]
async fn output_owner_requires_prepared_grant_and_deferred_preserves_reservoir() {
    let pool = pool(1);
    let (mut port, mut session) = start(&pool, 1);
    assert!(matches!(port.try_take(), OutputReadiness::ProjectionBusy));
    assert_eq!(pool.snapshot().data_used.events, 0);
    let grant = ready(&port).await;
    assert_eq!(grant.ordinal(), 1);
    let owned = pool.snapshot().data_used;
    assert_eq!(owned.events, 1);
    assert!(matches!(port.try_take(), OutputReadiness::ProjectionBusy));
    grant.return_unsubmitted();
    let next = ready(&port).await;
    assert_eq!(next.ordinal(), 1);
    assert_eq!(pool.snapshot().data_used, owned);
    next.return_unsubmitted();
    state(&port, OutputReadinessState::Ready).await;
    assert!(port.terminal(terminal(0, 0, "")).is_ok());
    assert!(matches!(
        port.try_take(),
        OutputReadiness::Closing(OutputCloseReason::Complete)
    ));
    drop(port);
    let frame = bounded(session.frames.next()).await.unwrap();
    assert!(frame.metadata().terminal);
    drop(frame);
    drop(bounded(session.completion).await.unwrap());
    drop(session.frames);
    drained(&pool).await;
}

#[tokio::test]
async fn output_owner_committed_empty_advances_once_without_wire_then_flushes_in_order() {
    let pool = pool(1);
    let (mut port, mut session) = start(&pool, 1);
    assert_eq!(ready(&port).await.committed(delta("", 1)), 1);
    let next = ready(&port).await;
    assert_eq!(next.ordinal(), 2);
    assert!(futures::poll!(session.frames.next()).is_pending());
    assert_eq!(next.committed(delta("中", 2)), 2);
    let mut decision = terminal(2, 2, "!");
    if let OutputCompletion::Succeeded { history, .. } = &mut decision.outcome {
        *history = Some(OutputHistory {
            text: "中!".into(),
            tokens: vec![TokenId(1), TokenId(2)],
        });
    }
    assert!(port.terminal(decision).is_ok());
    drop(port);
    let first = bounded(session.frames.next()).await.unwrap();
    assert_eq!(first.wire().payload(), "中".as_bytes());
    assert_eq!(first.metadata().token, Some(TokenId(2)));
    assert_eq!(first.metadata().generated_tokens, 2);
    drop(first);
    let flush = bounded(session.frames.next()).await.unwrap();
    assert_eq!(flush.wire().payload(), b"!");
    assert_eq!(flush.metadata().ordinal, 2);
    assert_eq!(flush.metadata().token, None);
    assert!(!flush.metadata().terminal);
    drop(flush.into_wire());
    let terminal = bounded(session.frames.next()).await.unwrap();
    assert!(terminal.metadata().terminal);
    drop(terminal);
    let completion = bounded(session.completion).await.unwrap();
    match completion.payload() {
        OutputCompletion::Succeeded {
            execution_evidence: None,
            history: Some(history),
            ..
        } => {
            assert_eq!(history.text, "中!");
            assert_eq!(history.tokens, vec![TokenId(1), TokenId(2)]);
        }
        _ => panic!("expected retained completed history"),
    }
    assert_eq!(
        pool.snapshot().data_used.projection_bytes,
        plan().retained_projection_bytes()
    );
    assert!(pool.snapshot().data_used.bytes == 0);
    drop(completion);
    drop(session.frames);
    drained(&pool).await;
}

#[tokio::test]
async fn output_owner_event_pressure_does_not_block_healthy_account_and_release_wakes() {
    let pool = pool(1);
    let (mut slow_port, mut slow) = start(&pool, 1);
    ready(&slow_port).await.committed(delta("a", 1));
    let retained = bounded(slow.frames.next()).await.unwrap();
    state(
        &slow_port,
        OutputReadinessState::OutputBlocked(OutputBlockReason::EventCredit),
    )
    .await;
    assert!(matches!(
        slow_port.try_take(),
        OutputReadiness::OutputBlocked(OutputBlockReason::EventCredit)
    ));
    let (mut healthy_port, mut healthy) = start(&pool, 1);
    ready(&healthy_port).await.committed(delta("b", 1));
    assert!(healthy_port.terminal(terminal(1, 1, "")).is_ok());
    drop(healthy_port);
    assert_eq!(
        bounded(healthy.frames.next())
            .await
            .unwrap()
            .wire()
            .payload(),
        b"b"
    );
    drop(bounded(healthy.frames.next()).await.unwrap());
    drop(bounded(healthy.completion).await.unwrap());
    drop(healthy.frames);
    drop(retained);
    let resumed = ready(&slow_port).await;
    assert_eq!(resumed.ordinal(), 2);
    resumed.return_unsubmitted();
    assert!(slow_port.terminal(terminal(1, 1, "")).is_ok());
    drop(slow_port);
    drop(bounded(slow.frames.next()).await.unwrap());
    drop(bounded(slow.completion).await.unwrap());
    drop(slow.frames);
    drained(&pool).await;
}

#[tokio::test]
async fn output_owner_reserved_terminal_slot_survives_full_wire_queue() {
    let pool = pool(2);
    let (mut port, mut session) = start(&pool, 2);
    ready(&port).await.committed(delta("a", 1));
    state(
        &port,
        OutputReadinessState::OutputBlocked(OutputBlockReason::WireQueue),
    )
    .await;
    assert!(port.terminal(terminal(1, 1, "")).is_ok());
    drop(port);
    // Completion proves terminal encoding finished before consuming either
    // wire frame. Its independent reserved slot was usable with data full.
    drop(bounded(session.completion).await.unwrap());
    let first = bounded(session.frames.next()).await.unwrap();
    assert_eq!(first.wire().payload(), b"a");
    assert!(!first.metadata().terminal);
    let second = bounded(session.frames.next()).await.unwrap();
    assert!(second.metadata().terminal);
    drop((first, second, session.frames));
    drained(&pool).await;
}

#[tokio::test]
async fn output_owner_terminal_revokes_unclaimed_grant_without_new_publication() {
    let pool = pool(1);
    let (mut port, mut session) = start(&pool, 1);
    state(&port, OutputReadinessState::Ready).await;
    assert_eq!(pool.snapshot().data_used.events, 1);
    assert!(port.terminal(terminal(0, 0, "")).is_ok());
    assert_eq!(
        port.readiness(),
        OutputReadinessState::Closing(OutputCloseReason::Complete)
    );
    drop(port);
    drop(bounded(session.frames.next()).await.unwrap());
    drop(bounded(session.completion).await.unwrap());
    drop(session.frames);
    drained(&pool).await;
}

#[tokio::test]
async fn output_owner_impossible_terminal_frontier_fails_without_waiting() {
    let pool = pool(1);
    let (mut port, mut session) = start(&pool, 1);
    state(&port, OutputReadinessState::Ready).await;
    assert!(port.terminal(terminal(2, 0, "")).is_ok());
    drop(port);
    let end = bounded(session.frames.next()).await.unwrap();
    assert!(end.metadata().terminal);
    assert!(String::from_utf8_lossy(end.wire().payload()).contains("frontier"));
    drop(end);
    let completion = bounded(session.completion).await.unwrap();
    assert!(matches!(completion.payload(), OutputCompletion::Failed(_)));
    drop((completion, session.frames));
    drained(&pool).await;
}

#[tokio::test]
async fn output_owner_disconnect_keeps_projection_until_engine_destroys_port() {
    let pool = pool(1);
    let (port, session) = start(&pool, 1);
    state(&port, OutputReadinessState::Ready).await;
    drop(session.frames);
    assert!(port.consumer_closed());
    let mut completion = session.completion;
    assert!(futures::poll!(&mut completion).is_pending());
    assert_eq!(
        pool.snapshot().data_used.projection_bytes,
        plan().retained_projection_bytes()
    );
    drop(port);
    let failed = bounded(completion).await.unwrap();
    assert!(matches!(failed.payload(), OutputCompletion::Failed(_)));
    drop(failed);
    drained(&pool).await;
}

#[tokio::test]
async fn output_owner_completion_receiver_drop_does_not_cancel_wire() {
    let pool = pool(1);
    let (mut port, session) = start(&pool, 1);
    drop(session.completion);
    let mut frames = session.frames;
    ready(&port).await.committed(delta("ok", 1));
    assert!(!port.consumer_closed());
    assert!(port.terminal(terminal(1, 1, "")).is_ok());
    drop(port);
    let data = bounded(frames.next()).await.unwrap();
    assert_eq!(data.wire().payload(), b"ok");
    drop(data);
    drop(bounded(frames.next()).await.unwrap());
    drop(frames);
    drained(&pool).await;
}

#[tokio::test]
async fn output_owner_cancel_with_held_grant_retains_credit_until_last_owner() {
    let pool = pool(1);
    let (port, mut session) = start(&pool, 1);
    let grant = ready(&port).await;
    port.cancel();
    assert!(matches!(
        port.try_take(),
        OutputReadiness::Closing(OutputCloseReason::Cancelled)
    ));
    let terminal = bounded(session.frames.next()).await.unwrap();
    assert!(terminal.metadata().terminal);
    let mut completion = session.completion;
    assert!(futures::poll!(&mut completion).is_pending());
    assert_eq!(
        pool.snapshot().data_used.projection_bytes,
        plan().retained_projection_bytes()
    );
    drop(port);
    drop(bounded(completion).await.unwrap());
    drop((terminal, session.frames));
    assert_eq!(pool.snapshot().retained_accounts, 1);
    assert_eq!(pool.snapshot().data_used.events, 1);
    drop(grant);
    drained(&pool).await;
}

#[tokio::test]
async fn output_owner_abandoned_grant_fails_and_unpolled_session_releases() {
    let pool = pool(1);
    let (port, session) = start(&pool, 1);
    drop(ready(&port).await);
    state(
        &port,
        OutputReadinessState::Closing(OutputCloseReason::Abandoned),
    )
    .await;
    assert_eq!(
        pool.snapshot().data_used.projection_bytes,
        plan().retained_projection_bytes()
    );
    drop((port, session));
    drained(&pool).await;
}

#[tokio::test]
async fn output_owner_port_drop_cancels_without_wire_polling() {
    let pool = pool(1);
    let (port, session) = start(&pool, 1);
    drop(port);
    let failure = bounded(session.completion).await.unwrap();
    assert!(matches!(failure.payload(), OutputCompletion::Failed(_)));
    drop((failure, session.frames));
    drained(&pool).await;
}

#[tokio::test]
async fn output_owner_final_token_then_flush_does_not_prepare_excess_wave() {
    let pool = pool(1);
    let (mut port, mut session) = start(&pool, 1);
    for token in 1..=4 {
        ready(&port).await.committed(delta("a", token));
        drop(bounded(session.frames.next()).await.unwrap());
    }
    assert_eq!(port.readiness(), OutputReadinessState::ProjectionBusy);
    assert!(port.terminal(terminal(4, 4, "tail")).is_ok());
    drop(port);
    let tail = bounded(session.frames.next()).await.unwrap();
    assert_eq!(tail.wire().payload(), b"tail");
    drop(tail);
    drop(bounded(session.frames.next()).await.unwrap());
    drop(bounded(session.completion).await.unwrap());
    drop(session.frames);
    drained(&pool).await;
}

mod chat;
mod future_capacity;
mod planning;
mod slow_consumer;

#[tokio::test]
async fn credited_actor_rejects_unreserved_execution_evidence_and_releases_storage() {
    let pool = pool(1);
    let (mut port, mut session) = start(&pool, 1);
    ready(&port).await.return_unsubmitted();
    state(&port, OutputReadinessState::Ready).await;
    let mut decision = terminal(0, 0, "");
    if let OutputCompletion::Succeeded {
        execution_evidence, ..
    } = &mut decision.outcome
    {
        *execution_evidence = Some(ferrum_types::InferenceExecutionEvidence {
            prompt_token_ids: vec![],
            output_token_ids: vec![],
            engine_token_timing: None,
        });
    }
    assert!(port.terminal(decision).is_ok());
    drop(port);
    while let Some(frame) = bounded(session.frames.next()).await {
        drop(frame);
    }
    let completion = bounded(session.completion).await.unwrap();
    assert!(matches!(completion.payload(), OutputCompletion::Failed(_)));
    drop(completion);
    drop(session.frames);
    drained(&pool).await;
}
