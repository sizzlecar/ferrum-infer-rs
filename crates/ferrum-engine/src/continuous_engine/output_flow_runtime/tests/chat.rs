use super::*;
use ferrum_types::{
    ApiChatRequest, ApiRequest, ApiStreamOptions, PROMPT_OPENED_REASONING_METADATA_KEY,
};

pub(super) fn chat_plan(tokens: usize, opened: bool, usage: bool) -> RequestOutputPlan {
    let mut request = InferenceRequest::new("rendered prompt", "model");
    request.stream = true;
    request.sampling_params.max_tokens = tokens;
    request
        .metadata
        .insert(PROMPT_OPENED_REASONING_METADATA_KEY.into(), opened.into());
    request.api_request = Some(ApiRequest::Chat(ApiChatRequest {
        messages: Vec::new(),
        tools: Vec::new(),
        tool_choice: None,
        tool_call_protocol: Default::default(),
        legacy_functions: Vec::new(),
        legacy_function_call: None,
        response_format: None,
        stream_options: Some(ApiStreamOptions {
            include_usage: Some(usage),
        }),
    }));
    RequestOutputPlan::derive(
        Arc::new(OutputProjectionContract::chat_sse(
            "chat-id".into(),
            "alias".into(),
            usage,
        )),
        &BoundedTokenizer(SpecialTokens::default()),
        &request,
        3,
    )
    .unwrap()
}

pub(super) fn chat_pool() -> OutputCreditPool {
    let maximum = limits(&chat_plan(8, true, true), 1).maximum;
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

pub(super) fn start_chat(
    pool: &OutputCreditPool,
    plan: RequestOutputPlan,
    options: OutputFlowRuntimeOptions,
) -> (OutputFlowPort, CreditedOutputSession) {
    let budget = RequestOutputBudget::open(pool, limits(&plan, 1), plan).unwrap();
    spawn_output_flow_runtime(budget, Arc::new(Notify::new()), options)
}
fn options() -> OutputFlowRuntimeOptions {
    OutputFlowRuntimeOptions::from_config(&SloOutputConfig::default())
}
fn events(frame: &CreditedOutputFrame) -> Vec<serde_json::Value> {
    std::str::from_utf8(frame.wire().payload())
        .unwrap()
        .split("\n\n")
        .filter_map(|event| event.strip_prefix("data: "))
        .filter(|data| *data != "[DONE]")
        .map(|data| serde_json::from_str(data).unwrap())
        .collect()
}

#[tokio::test]
async fn output_owner_chat_single_commit_drains_channels_then_terminal_tail_and_usage() {
    let pool = chat_pool();
    let (mut port, mut session) = start_chat(&pool, chat_plan(4, true, true), options());
    assert_eq!(ready(&port).await.committed(delta("r</think>a", 1)), 1);
    // Control can arrive before the actor receives or drains the data command.
    assert!(port.terminal(terminal(1, 1, "b")).is_ok());
    let first = bounded(session.frames.next()).await.unwrap();
    assert_eq!(events(&first)[0]["choices"][0]["delta"]["reasoning"], "r");
    assert_eq!(first.metadata().token, Some(TokenId(1)));
    assert_eq!(first.metadata().generated_tokens, 1);
    assert!(futures::poll!(session.frames.next()).is_pending());
    drop(first);
    let second = bounded(session.frames.next()).await.unwrap();
    assert_eq!(events(&second)[0]["choices"][0]["delta"]["content"], "a");
    assert_eq!(second.metadata().ordinal, 1);
    assert_eq!(second.metadata().token, None);
    assert_eq!(second.metadata().generated_tokens, 1);
    drop(second);
    let tail = bounded(session.frames.next()).await.unwrap();
    assert_eq!(events(&tail)[0]["choices"][0]["delta"]["content"], "b");
    assert_eq!(tail.metadata().token, None);
    assert_eq!(tail.metadata().generated_tokens, 1);
    drop(tail);
    let finish = bounded(session.frames.next()).await.unwrap();
    assert!(finish.metadata().terminal);
    let wire = events(&finish);
    assert_eq!(wire[0]["choices"][0]["delta"]["content"], "");
    assert_eq!(wire[0]["choices"][0]["finish_reason"], "stop");
    assert_eq!(wire[1]["usage"]["completion_tokens"], 1);
    assert!(finish.wire().payload().ends_with(b"data: [DONE]\n\n"));
    drop(port);
    drop(bounded(session.completion).await.unwrap());
    drop((finish, session.frames));
    drained(&pool).await;
}

#[tokio::test]
async fn output_owner_chat_deferred_and_last_token_held_marker_flush_only_once() {
    let pool = chat_pool();
    let (mut port, mut session) = start_chat(&pool, chat_plan(1, false, false), options());
    let grant = ready(&port).await;
    let before = pool.snapshot().data_used;
    grant.return_unsubmitted();
    let grant = ready(&port).await;
    assert_eq!(grant.ordinal(), 1);
    assert_eq!(pool.snapshot().data_used, before);
    grant.committed(delta("<thi", 1));
    state(&port, OutputReadinessState::ProjectionBusy).await;
    assert!(futures::poll!(session.frames.next()).is_pending());
    assert!(port.terminal(terminal(1, 1, "")).is_ok());
    drop(port);
    let data = bounded(session.frames.next()).await.unwrap();
    assert_eq!(events(&data)[0]["choices"][0]["delta"]["content"], "<thi");
    assert_eq!(data.metadata().generated_tokens, 1);
    drop(data);
    let finish = bounded(session.frames.next()).await.unwrap();
    assert_eq!(events(&finish).len(), 1, "usage remains opt-in");
    drop(finish);
    drop(bounded(session.completion).await.unwrap());
    drop(session.frames);
    drained(&pool).await;
}

#[tokio::test]
async fn output_owner_chat_full_pending_projection_keeps_healthy_account_live_and_cancel_charged() {
    let pool = chat_pool();
    let slow_plan = chat_plan(4, true, false);
    let retained = slow_plan.retained_projection_bytes();
    let (slow, slow_session) = start_chat(&pool, slow_plan, options());
    let (mut healthy, mut healthy_session) =
        start_chat(&pool, chat_plan(4, false, false), options());
    ready(&slow).await.committed(delta("r</think>a", 1));
    state(
        &slow,
        OutputReadinessState::OutputBlocked(OutputBlockReason::WireQueue),
    )
    .await;
    assert!(matches!(slow.try_take(), OutputReadiness::OutputBlocked(_)));
    ready(&healthy).await.committed(delta("healthy", 1));
    let data = bounded(healthy_session.frames.next()).await.unwrap();
    assert_eq!(
        events(&data)[0]["choices"][0]["delta"]["content"],
        "healthy"
    );
    drop(data);
    assert!(healthy.terminal(terminal(1, 1, "")).is_ok());
    drop(healthy);
    drop(bounded(healthy_session.frames.next()).await.unwrap());
    drop(bounded(healthy_session.completion).await.unwrap());
    drop(healthy_session.frames);
    let CreditedOutputSession {
        frames,
        mut completion,
    } = slow_session;
    drop(frames);
    state(
        &slow,
        OutputReadinessState::Closing(OutputCloseReason::Disconnected),
    )
    .await;
    assert!(futures::poll!(&mut completion).is_pending());
    assert_eq!(pool.snapshot().data_used.projection_bytes, retained);
    drop(slow);
    drop(bounded(completion).await.unwrap());
    drained(&pool).await;
}
