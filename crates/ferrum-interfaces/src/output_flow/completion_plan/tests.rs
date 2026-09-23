use super::*;
use crate::tokenizer::{BoundedDecodeBound, BoundedIncrementalDecodePolicy, TokenizerInfo};
use ferrum_types::{ResponseCompletionEnvelope, SpecialTokens};
use std::num::NonZeroUsize;

struct PreparedTokenizer {
    ids: [TokenId; 3],
    special: SpecialTokens,
}
impl PreparedTokenizer {
    fn new() -> Self {
        Self {
            ids: [2, 3, 4].map(TokenId::new),
            special: SpecialTokens::default(),
        }
    }
}
impl Tokenizer for PreparedTokenizer {
    fn encode(&self, _: &str, _: bool) -> ferrum_types::Result<Vec<TokenId>> {
        panic!("admission must not encode")
    }
    fn decode(&self, _: &[TokenId], _: bool) -> ferrum_types::Result<String> {
        panic!("admission must not decode")
    }
    fn decode_incremental(&self, _: &[TokenId], _: TokenId) -> ferrum_types::Result<String> {
        panic!("admission must not decode")
    }
    fn vocab_size(&self) -> usize {
        5
    }
    fn special_tokens(&self) -> &SpecialTokens {
        &self.special
    }
    fn token_id(&self, text: &str) -> Option<TokenId> {
        (text == "close").then_some(TokenId::new(4))
    }
    fn token_text(&self, _: TokenId) -> Option<&str> {
        None
    }
    fn info(&self) -> TokenizerInfo {
        unreachable!()
    }
    fn prepared_completion_tokens(&self, text: &str) -> Option<&[TokenId]> {
        match text {
            "delimiter" => Some(&self.ids[..2]),
            "open" => Some(&self.ids),
            _ => None,
        }
    }
    fn bounded_decode_bound(&self) -> Option<BoundedDecodeBound> {
        Some(BoundedDecodeBound::new(NonZeroUsize::new(3).unwrap(), 0))
    }
    fn bounded_token_bytes_bound(&self) -> Option<NonZeroUsize> {
        NonZeroUsize::new(3)
    }
    fn bounded_incremental_decode_policy(&self) -> Option<BoundedIncrementalDecodePolicy> {
        Some(BoundedIncrementalDecodePolicy::StrictDecodedPrefix)
    }
}
fn boundary() -> ResponseCompletionBoundary {
    ResponseCompletionBoundary::AfterDelimiterAndPayload {
        delimiter: "delimiter".into(),
        alternate_envelope: Some(ResponseCompletionEnvelope {
            open_token_text: "open".into(),
            close_token_text: "close".into(),
            max_envelopes: 2,
        }),
    }
}
#[test]
fn prepared_completion_budget_counts_all_matchers_and_fences_changed_sources() {
    let mut tokenizer = PreparedTokenizer::new();
    let plan = ResponseCompletionPlan::derive(&tokenizer, &boundary()).unwrap();
    assert_eq!(
        plan.retained_storage_bytes(),
        6 * (std::mem::size_of::<u32>() + std::mem::size_of::<usize>())
            + 3 * ("delimiter".len() + "open".len() + "close".len())
    );
    assert_eq!(
        plan.resolve_marker(&tokenizer, "delimiter")
            .unwrap()
            .as_slice(),
        &[TokenId::new(2), TokenId::new(3)]
    );
    tokenizer.ids.swap(0, 1);
    assert!(matches!(
        plan.resolve_marker(&tokenizer, "delimiter"),
        Err(OutputFlowError::BoundExceeded)
    ));
    assert!(matches!(
        plan.resolve_marker(&tokenizer, "unplanned"),
        Err(OutputFlowError::BoundExceeded)
    ));
    assert!(matches!(
        marker_storage(usize::MAX, 1),
        Err(OutputFlowError::Overflow)
    ));
    assert!(matches!(
        marker_storage(1, usize::MAX),
        Err(OutputFlowError::Overflow)
    ));
}
#[test]
fn prepared_completion_raw_envelope_budget_is_reserved_before_matcher_allocation() {
    let tokenizer = PreparedTokenizer::new();
    let mut request = InferenceRequest::new("prompt", "model");
    request.sampling_params.max_tokens = 10;
    request.sampling_params.response_completion_boundary = boundary();
    let plan = RequestOutputPlan::derive(
        Arc::new(OutputProjectionContract::cli_text()),
        &tokenizer,
        &request,
        1,
    )
    .unwrap();
    let config = ferrum_types::SloOutputConfig::default();
    let pool = OutputCreditPool::new(
        OutputPoolLimits::from_slo(&config, NonZeroUsize::new(2).unwrap()).unwrap(),
    )
    .unwrap();
    let mut limits = OutputAccountLimits::from_slo(
        &config,
        NonZeroUsize::new(plan.terminal_credit().events).unwrap(),
    )
    .unwrap();
    limits.maximum.projection_bytes = plan.retained_projection_bytes() - 1;
    assert!(matches!(
        RequestOutputBudget::open(&pool, limits, plan),
        Err(OutputFlowError::BoundExceeded)
    ));
    assert_eq!(pool.snapshot().retained_accounts, 0);
}
