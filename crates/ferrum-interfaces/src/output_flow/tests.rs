use super::*;
use crate::{
    tokenizer::{BoundedDecodeBound, DecodedTextBound, TokenizerInfo},
    Tokenizer,
};
use ferrum_types::{SpecialTokens, TokenId};
use std::num::NonZeroUsize;

mod capacity;
mod chat;
mod evidence;

struct BoundedTokenizer {
    bound: Option<DecodedTextBound>,
    workspace: Option<usize>,
    special: SpecialTokens,
    atomic_delimiter: bool,
    incremental: bool,
}
impl BoundedTokenizer {
    fn new(bytes: usize) -> Self {
        Self {
            bound: NonZeroUsize::new(bytes).map(DecodedTextBound::new),
            workspace: Some(2),
            special: SpecialTokens::default(),
            atomic_delimiter: false,
            incremental: false,
        }
    }
}
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
        2
    }
    fn special_tokens(&self) -> &SpecialTokens {
        &self.special
    }
    fn token_id(&self, text: &str) -> Option<TokenId> {
        (self.atomic_delimiter && text == "</think>").then_some(TokenId::new(1))
    }
    fn token_text(&self, _: TokenId) -> Option<&str> {
        None
    }
    fn info(&self) -> TokenizerInfo {
        unreachable!()
    }
    fn decoded_text_bound(&self) -> Option<DecodedTextBound> {
        self.bound
    }
    fn bounded_decode_bound(&self) -> Option<BoundedDecodeBound> {
        let workspace = self.workspace?;
        self.bound.map(|bound| {
            BoundedDecodeBound::new(
                NonZeroUsize::new(bound.max_decoded_bytes(1).unwrap()).unwrap(),
                workspace,
            )
        })
    }
    fn bounded_token_bytes_bound(&self) -> Option<NonZeroUsize> {
        NonZeroUsize::new(1)
    }
    fn bounded_incremental_decode_policy(&self) -> Option<BoundedIncrementalDecodePolicy> {
        self.incremental
            .then_some(BoundedIncrementalDecodePolicy::StrictDecodedPrefix)
    }
}

#[test]
fn output_flow_requires_workspace_proof_and_reserves_it_before_admission() {
    let mut request = InferenceRequest::new("prompt", "test");
    request.sampling_params.max_tokens = 8;
    let contract = Arc::new(OutputProjectionContract::cli_text());
    let mut tokenizer = BoundedTokenizer::new(7);
    tokenizer.workspace = None;
    assert!(tokenizer.decoded_text_bound().is_some());
    assert!(matches!(
        RequestOutputPlan::derive(contract.clone(), &tokenizer, &request, 3),
        Err(OutputFlowError::Unsupported(_))
    ));
    tokenizer.workspace = Some(19);
    let plan = RequestOutputPlan::derive(contract, &tokenizer, &request, 3).unwrap();
    let pool = pool(&plan);
    let mut limits = limits(&plan);
    limits.maximum.projection_bytes -= 1;
    assert!(matches!(
        RequestOutputBudget::open(&pool, limits, plan),
        Err(OutputFlowError::BoundExceeded)
    ));
    assert_eq!(pool.snapshot().retained_accounts, 0);
    assert_eq!(pool.snapshot().data_used, OutputCreditAmount::ZERO);
}

fn plan(contract: OutputProjectionContract, tokens: usize) -> RequestOutputPlan {
    let mut request = InferenceRequest::new("prompt", "test-model");
    request.sampling_params.max_tokens = tokens;
    RequestOutputPlan::derive(Arc::new(contract), &BoundedTokenizer::new(6), &request, 3).unwrap()
}

#[test]
fn output_flow_codec_policy_uses_real_json_escaping_without_identity_values() {
    let descriptor = |id: &str, model: &str, usage| {
        plan(
            OutputProjectionContract::completions_sse(id.into(), model.into(), usage),
            4,
        )
        .codec_descriptor()
        .unwrap()
    };
    assert_eq!(
        descriptor("abcd", "model", true),
        descriptor("wxyz", "other", true)
    );
    // Same UTF-8 length, different work for the actual JSON serializer.
    assert_ne!(descriptor("a", "m", true), descriptor("\n", "m", true));
    assert_ne!(descriptor("a", "m", true), descriptor("a", "\n", true));
    assert_ne!(descriptor("a", "m", false), descriptor("a", "m", true));
    let raw = plan(OutputProjectionContract::cli_text(), 4)
        .codec_descriptor()
        .unwrap();
    assert_eq!(raw.kind, OutputCodecKind::CliText);
    assert_eq!(raw.max_data_envelope_bytes, 0);
    assert_ne!(raw, descriptor("a", "m", true));
}

fn limits(plan: &RequestOutputPlan) -> OutputAccountLimits {
    OutputAccountLimits {
        maximum: OutputCreditAmount {
            events: plan.minimum_event_capacity(),
            bytes: plan.lifetime_wire_bytes() + plan.terminal_credit().bytes,
            projection_bytes: plan.retained_projection_bytes(),
        },
        terminal: plan.terminal_credit(),
    }
}

fn pool(plan: &RequestOutputPlan) -> OutputCreditPool {
    let limits = limits(plan);
    OutputCreditPool::new(OutputPoolLimits {
        maximum: OutputCreditAmount {
            events: 2 * limits.maximum.events,
            bytes: 2 * limits.maximum.bytes,
            projection_bytes: 2 * limits.maximum.projection_bytes,
        },
        max_total_bytes: 2 * limits.maximum.total_bytes().unwrap(),
        max_open_accounts: 2,
    })
    .unwrap()
}

fn begin(budget: &mut RequestOutputBudget) -> OutputFramePermit {
    match budget.try_begin_frame().unwrap() {
        OutputFrameAttempt::Reserved(permit) => permit,
        OutputFrameAttempt::Full(_) => panic!("unexpected event pressure"),
    }
}

#[test]
fn output_flow_derive_uses_real_capability_and_effective_request_limit() {
    let mut request = InferenceRequest::new("prompt", "test");
    request.sampling_params.max_tokens = 513;
    let contract = Arc::new(OutputProjectionContract::cli_text());
    let plan = RequestOutputPlan::derive(contract.clone(), &BoundedTokenizer::new(7), &request, 8)
        .unwrap();
    assert_eq!(plan.effective_max_tokens(), 513);
    assert_eq!(plan.max_decoded_bytes(), 513 * 7);
    assert_eq!(
        plan.retained_projection_bytes(),
        4 * 513 * 7 + 513 * 2 + 3 * 513 * std::mem::size_of::<TokenId>() + MAX_OUTPUT_ERROR_BYTES,
        "the actual decoder workspace is additional to retained text and histories"
    );
    assert_eq!(
        plan.lifetime_wire_bytes(),
        513 * 7,
        "CLI wire retains one payload copy"
    );
    assert_eq!(
        plan.minimum_event_capacity(),
        2,
        "event slots must not scale with output length"
    );
    assert!(matches!(
        RequestOutputPlan::derive(contract.clone(), &BoundedTokenizer::new(0), &request, 8),
        Err(OutputFlowError::Unsupported(_))
    ));
    assert!(matches!(
        RequestOutputPlan::derive(
            contract.clone(),
            &BoundedTokenizer::new(usize::MAX),
            &request,
            8
        ),
        Err(OutputFlowError::Overflow)
    ));
    request.sampling_params.model_output_protocol = ModelOutputProtocol::HarmonyGptOss;
    assert!(matches!(
        RequestOutputPlan::derive(contract.clone(), &BoundedTokenizer::new(7), &request, 8),
        Err(OutputFlowError::Unsupported(_))
    ));
    request.sampling_params.model_output_protocol = ModelOutputProtocol::Text;
    request.evidence_request.capture_engine_token_timing = true;
    assert!(matches!(
        RequestOutputPlan::derive(contract, &BoundedTokenizer::new(7), &request, 8),
        Ok(plan) if plan.evidence_plan().captures_timing()
    ));
}

#[test]
fn output_flow_completions_codec_covers_escaping_empty_frames_and_recycled_events() {
    let plan = plan(
        OutputProjectionContract::completions_sse("id\n\"中".into(), "model\\\t".into(), true),
        512,
    );
    assert_eq!(plan.minimum_event_capacity(), 4);
    let pool = pool(&plan);
    let mut budget = RequestOutputBudget::open(&pool, limits(&plan), plan.clone()).unwrap();
    let mut actual = 0;
    // Maximal text escaping in one frame, followed by every possible empty
    // token-only frame and the final flush. Only one live data event is needed.
    let text = "\0".repeat(plan.max_decoded_bytes());
    for index in 0..plan.max_data_frames() {
        let permit = begin(&mut budget);
        let frame = budget
            .encode_data_frame(permit, if index == 0 { &text } else { "" }, u64::MAX)
            .unwrap();
        assert!(frame.payload().capacity() <= frame.credit().bytes);
        let json = frame
            .payload()
            .strip_prefix(b"data: ")
            .unwrap()
            .strip_suffix(b"\n\n")
            .unwrap();
        let value: serde_json::Value = serde_json::from_slice(json).unwrap();
        assert_eq!(value["model"], "model\\\t");
        assert_eq!(
            value["choices"][0]["text"],
            if index == 0 { &text } else { "" }
        );
        actual += frame.payload().len();
        drop(frame);
    }
    assert_eq!(
        actual,
        plan.lifetime_wire_bytes(),
        "NUL and maximal timestamps attain the declared bound"
    );
    assert_eq!(budget.unspent_wire_bytes(), Some(0));
    assert!(matches!(
        budget.try_begin_frame(),
        Err(OutputFlowError::BoundExceeded)
    ));
    let usage = TokenUsage::new(3, 512);
    let terminal = budget
        .encode_terminal(OutputTerminal::Success {
            reason: FinishReason::ContentFilter,
            usage: &usage,
            created: u64::MAX,
        })
        .unwrap();
    let wire = String::from_utf8(terminal.payload().clone()).unwrap();
    assert_eq!(wire.matches("data: ").count(), 3);
    assert!(wire.contains("\"finish_reason\":\"content_filter\""));
    assert!(wire.contains("\"total_tokens\":515"));
    assert!(wire.ends_with("data: [DONE]\n\n"));
    assert!(terminal.payload().len() <= plan.terminal_credit().bytes);
    assert!(terminal.payload().capacity() <= terminal.credit().bytes);
}

#[test]
fn output_flow_global_projection_pressure_releases_provisional_terminal_escrow() {
    let plan = plan(OutputProjectionContract::cli_text(), 4);
    let account_limits = limits(&plan);
    let pool = OutputCreditPool::new(OutputPoolLimits {
        maximum: OutputCreditAmount {
            events: 2 * account_limits.maximum.events,
            bytes: 2 * account_limits.maximum.bytes,
            projection_bytes: account_limits.maximum.projection_bytes,
        },
        max_total_bytes: 2 * account_limits.maximum.total_bytes().unwrap(),
        max_open_accounts: 2,
    })
    .unwrap();
    let first = RequestOutputBudget::open(&pool, account_limits, plan.clone()).unwrap();
    let mut other = plan.clone();
    other.request_id = RequestId::new();
    assert!(matches!(
        RequestOutputBudget::open(&pool, account_limits, other),
        Err(OutputFlowError::AdmissionFull)
    ));
    assert_eq!(pool.snapshot().open_accounts, 1);
    assert_eq!(pool.snapshot().retained_accounts, 1);
    assert_eq!(pool.snapshot().terminal_held, plan.terminal_credit());
    drop(first);
    assert_eq!(pool.snapshot().terminal_held, OutputCreditAmount::ZERO);
}

#[test]
fn output_flow_usage_option_controls_wire_and_logical_terminal_event_credit() {
    for include_usage in [false, true] {
        let plan = plan(
            OutputProjectionContract::completions_sse("id".into(), "model".into(), include_usage),
            1,
        );
        assert_eq!(
            plan.terminal_credit().events,
            2 + usize::from(include_usage)
        );
        let pool = pool(&plan);
        let mut budget = RequestOutputBudget::open(&pool, limits(&plan), plan).unwrap();
        let usage = TokenUsage::new(3, 1);
        let terminal = budget
            .encode_terminal(OutputTerminal::Success {
                reason: FinishReason::Stop,
                usage: &usage,
                created: 0,
            })
            .unwrap();
        let wire = std::str::from_utf8(terminal.payload()).unwrap();
        assert_eq!(
            wire.matches("data: ").count(),
            2 + usize::from(include_usage)
        );
        assert_eq!(
            wire.contains("\"usage\":{\"prompt_tokens\":3"),
            include_usage
        );
        assert_eq!(terminal.credit().events, 2 + usize::from(include_usage));
        assert!(wire.ends_with("data: [DONE]\n\n"));
    }
}

#[test]
fn output_flow_short_output_also_covers_retained_bounded_error_storage() {
    let mut request = InferenceRequest::new("prompt", "model");
    request.sampling_params.max_tokens = 1;
    let plan = RequestOutputPlan::derive(
        Arc::new(OutputProjectionContract::cli_text()),
        &BoundedTokenizer::new(1),
        &request,
        0,
    )
    .unwrap();
    assert_eq!(plan.max_decoded_bytes(), 1);
    let pool = pool(&plan);
    let mut budget = RequestOutputBudget::open(&pool, limits(&plan), plan).unwrap();
    let error = BoundedOutputError::new(&"x".repeat(MAX_OUTPUT_ERROR_BYTES));
    let terminal = budget
        .encode_terminal(OutputTerminal::Error(&error))
        .unwrap();
    assert!(budget.account_snapshot().data_used.projection_bytes >= error.message().len());
    assert!(terminal.credit().bytes >= terminal.payload().capacity());
    assert_eq!(
        terminal.payload().len(),
        MAX_OUTPUT_ERROR_BYTES + b"Error: \n".len()
    );
    drop(error);
    drop((terminal, budget));
    assert_eq!(pool.snapshot().retained_accounts, 0);
}

#[tokio::test]
async fn output_flow_full_data_slot_keeps_terminal_and_healthy_request_runnable() {
    let plan = plan(
        OutputProjectionContract::completions_sse("id".into(), "model".into(), true),
        10,
    );
    let pool = pool(&plan);
    let mut slow = RequestOutputBudget::open(&pool, limits(&plan), plan.clone()).unwrap();
    let mut healthy_plan = plan.clone();
    healthy_plan.request_id = RequestId::new();
    let mut healthy =
        RequestOutputBudget::open(&pool, limits(&healthy_plan), healthy_plan).unwrap();
    let permit = begin(&mut slow);
    let data = slow.encode_data_frame(permit, "hello", 0).unwrap();
    let mut wake = match slow.try_begin_frame().unwrap() {
        OutputFrameAttempt::Full(wake) => wake,
        _ => panic!("one live data slot must enforce backpressure"),
    };
    let permit = begin(&mut healthy);
    let healthy_data = healthy.encode_data_frame(permit, "ok", 0).unwrap();
    assert!(!healthy_data.payload().is_empty());
    let error = BoundedOutputError::new("slow consumer timeout");
    let terminal = slow.encode_terminal(OutputTerminal::Error(&error)).unwrap();
    assert!(String::from_utf8_lossy(terminal.payload()).contains("slow consumer timeout"));
    assert!(slow.account_snapshot().closed);
    drop(data);
    assert!(futures::poll!(Box::pin(wake.changed())).is_ready());
    drop((terminal, healthy_data, slow, healthy));
    assert_eq!(pool.snapshot().retained_accounts, 0);
    assert_eq!(pool.snapshot().data_used, OutputCreditAmount::ZERO);
    assert_eq!(pool.snapshot().terminal_held, OutputCreditAmount::ZERO);
}

#[test]
fn output_flow_returned_unsubmitted_permit_preserves_future_bytes_without_replenishing_spent_bytes()
{
    let plan = plan(OutputProjectionContract::cli_text(), 4);
    let pool = pool(&plan);
    let mut budget = RequestOutputBudget::open(&pool, limits(&plan), plan.clone()).unwrap();
    let permit = begin(&mut budget);
    assert!(matches!(
        budget.try_begin_frame(),
        Err(OutputFlowError::FrameInFlight)
    ));
    budget.return_unsubmitted_frame(permit).unwrap();
    assert_eq!(budget.unspent_wire_bytes(), Some(24));
    let permit = begin(&mut budget);
    let data = budget.encode_data_frame(permit, "abc", 0).unwrap();
    assert_eq!(budget.unspent_wire_bytes(), Some(21));
    drop(data);
    assert_eq!(
        budget.unspent_wire_bytes(),
        Some(21),
        "last-owner release is not new lifetime budget"
    );
    let permit = begin(&mut budget);
    budget.return_unsubmitted_frame(permit).unwrap();
    assert_eq!(budget.unspent_wire_bytes(), Some(21));
}

#[test]
fn output_flow_dropped_wave_permit_requires_failure_and_cannot_reissue_bytes() {
    let plan = plan(OutputProjectionContract::cli_text(), 4);
    let pool = pool(&plan);
    let mut budget = RequestOutputBudget::open(&pool, limits(&plan), plan).unwrap();
    drop(begin(&mut budget));
    assert!(matches!(
        budget.try_begin_frame(),
        Err(OutputFlowError::FrameInFlight)
    ));
    let terminal = budget
        .encode_terminal(OutputTerminal::Error(&BoundedOutputError::new("cancelled")))
        .unwrap();
    assert_eq!(terminal.payload(), b"Error: cancelled\n");
    drop((terminal, budget));
    assert_eq!(pool.snapshot().retained_accounts, 0);
}

#[test]
fn output_flow_committed_without_wire_returns_unused_event_and_preserves_byte_reservoir() {
    let plan = plan(OutputProjectionContract::cli_text(), 4);
    let pool = pool(&plan);
    let mut budget = RequestOutputBudget::open(&pool, limits(&plan), plan).unwrap();
    let initial = budget.unspent_wire_bytes();
    let permit = begin(&mut budget);
    assert_eq!(budget.account_snapshot().data_used.events, 1);
    budget.finish_without_wire_frame(permit).unwrap();
    assert_eq!(budget.account_snapshot().data_used.events, 0);
    assert_eq!(budget.unspent_wire_bytes(), initial);
    let permit = begin(&mut budget);
    let visible = budget.encode_data_frame(permit, "中", 0).unwrap();
    assert_eq!(visible.payload(), "中".as_bytes());
    assert_eq!(
        budget.unspent_wire_bytes(),
        initial.map(|bytes| bytes - "中".len())
    );
}

#[test]
fn output_flow_foreign_pool_same_request_and_generation_cannot_transfer_credit() {
    let plan = plan(OutputProjectionContract::cli_text(), 4);
    let first_pool = pool(&plan);
    let second_pool = pool(&plan);
    let mut first = RequestOutputBudget::open(&first_pool, limits(&plan), plan.clone()).unwrap();
    let mut second = RequestOutputBudget::open(&second_pool, limits(&plan), plan).unwrap();
    let first_permit = begin(&mut first);
    let second_permit = begin(&mut second);
    assert_eq!(first_permit.generation(), second_permit.generation());
    let before = first_pool.snapshot().data_used;
    assert!(matches!(
        first.encode_data_frame(second_permit, "x", 0),
        Err(OutputFlowError::Credit(
            OutputCreditError::ForeignReservation
        ))
    ));
    assert_eq!(first_pool.snapshot().data_used, before);
    first.return_unsubmitted_frame(first_permit).unwrap();
    drop((first, second));
    assert_eq!(first_pool.snapshot().retained_accounts, 0);
    assert_eq!(second_pool.snapshot().retained_accounts, 0);
}

#[test]
fn output_flow_bounded_error_and_projection_history_keep_ownership_after_terminal() {
    let plan = plan(OutputProjectionContract::cli_text(), 4);
    let pool = pool(&plan);
    let mut budget = RequestOutputBudget::open(&pool, limits(&plan), plan.clone()).unwrap();
    let error = BoundedOutputError::new(&"中".repeat(400));
    assert_eq!(error.message().len(), 510);
    let terminal = budget
        .encode_terminal(OutputTerminal::Error(&error))
        .unwrap();
    let history = budget
        .into_retained_projection("history".to_owned())
        .unwrap();
    assert_eq!(
        pool.snapshot().data_used.projection_bytes,
        plan.retained_projection_bytes()
    );
    drop(terminal);
    assert_eq!(pool.snapshot().retained_accounts, 1);
    drop(history);
    assert_eq!(pool.snapshot().retained_accounts, 0);
}

#[test]
fn output_flow_capacity_failure_does_not_leave_a_provisional_account() {
    let plan = plan(OutputProjectionContract::cli_text(), 4);
    let pool = pool(&plan);
    let mut insufficient = limits(&plan);
    insufficient.maximum.bytes -= 1;
    assert!(matches!(
        RequestOutputBudget::open(&pool, insufficient, plan.clone()),
        Err(OutputFlowError::BoundExceeded)
    ));
    assert_eq!(pool.snapshot().retained_accounts, 0);
    let mut budget = RequestOutputBudget::open(&pool, limits(&plan), plan).unwrap();
    let permit = begin(&mut budget);
    assert!(matches!(
        budget.encode_data_frame(permit, &"x".repeat(25), 0),
        Err(OutputFlowError::BoundExceeded)
    ));
    let invalid = TokenUsage {
        prompt_tokens: 3,
        completion_tokens: 1,
        total_tokens: 9,
    };
    assert!(budget
        .encode_terminal(OutputTerminal::Success {
            reason: FinishReason::Stop,
            usage: &invalid,
            created: 0
        })
        .is_err());
    drop(budget);
    assert_eq!(pool.snapshot().retained_accounts, 0);
    assert!(codec::encode(1, |writer| codec::data(
        writer,
        &OutputProjectionContract::cli_text(),
        "中",
        0
    ))
    .is_err());
}
