use super::*;
use crate::{OutputBudget, OutputChunk};
use ferrum_types::{FinishReason, OutputConfig, RequestId, StreamChunk, TokenId, TokenUsage};
use std::collections::HashMap;

fn budget(total: usize, per_session: usize) -> OutputBudget {
    OutputBudget::new(OutputConfig {
        semantic_bytes_total: total,
        semantic_bytes_per_session: per_session,
        ..Default::default()
    })
    .unwrap()
}

#[test]
fn retained_growth_is_transactional_across_session_and_shared_limits() {
    let budget = budget(10, 8);
    let first = budget.try_open_session().unwrap();
    let second = budget.try_open_session().unwrap();
    let mut left = first.try_reserve_semantic(6).unwrap();
    let right = second.try_reserve_semantic(4).unwrap();
    // Local credit is available, but the shared pool is full. The failed grant
    // must return its temporary local reservation and leave both owners intact.
    assert!(left.try_grow(1).is_err());
    assert_eq!(
        (
            left.bytes(),
            first.semantic_usage(),
            second.semantic_usage()
        ),
        (6, 6, 4)
    );
    assert_eq!(budget.usage().semantic_bytes, 10);
    drop(right);
    left.try_grow(2).unwrap();
    assert!(left.try_grow(1).is_err());
    assert_eq!((left.bytes(), first.semantic_usage()), (8, 8));
    assert_eq!(second.semantic_usage(), 0);
    assert!(left.try_grow(usize::MAX).is_err());
    assert_eq!(budget.usage().semantic_bytes, 8);
    drop(left);
    assert_eq!(budget.usage().semantic_bytes, 0);
}

#[test]
fn split_content_keeps_original_session_until_last_successor_dies() {
    let budget = budget(16, 16);
    let session = budget.try_open_session().unwrap();
    let mut original = session.try_reserve_semantic(12).unwrap();
    assert!(original.split_off(13).is_err());
    let successor = original.split_off(7).unwrap();
    drop(session);
    assert_eq!((original.bytes(), successor.bytes()), (5, 7));
    original.try_resize(2).unwrap();
    assert_eq!(budget.usage().semantic_bytes, 9);
    assert!(original.shrink_to(3).is_err());
    drop(original);
    assert_eq!(
        (budget.usage().semantic_bytes, budget.usage().sessions),
        (7, 1)
    );
    drop(successor);
    assert_eq!(
        (budget.usage().semantic_bytes, budget.usage().sessions),
        (0, 0)
    );
}

#[test]
fn whole_credit_transfer_leaves_a_reusable_empty_original_owner() {
    let budget = budget(12, 12);
    let session = budget.try_open_session().unwrap();
    let mut original = session.try_reserve_semantic(12).unwrap();
    let held = original.take_all();
    assert_eq!(
        (original.bytes(), held.bytes(), session.semantic_usage()),
        (0, 12, 12)
    );
    assert!(original.try_grow(1).is_err());
    drop(held);
    original.try_grow(3).unwrap();
    let successor = original.split_off(2).unwrap();
    assert_eq!(
        (
            original.bytes(),
            successor.bytes(),
            session.semantic_usage()
        ),
        (1, 2, 3)
    );
    drop((original, successor, session));
    assert_eq!(
        (budget.usage().semantic_bytes, budget.usage().sessions),
        (0, 0)
    );
}

fn response(text: &str) -> InferenceResponse {
    InferenceResponse {
        request_id: RequestId::new(),
        text: text.into(),
        tokens: vec![TokenId(3), TokenId(4)],
        finish_reason: FinishReason::Length,
        usage: TokenUsage::new(3, 2),
        latency_ms: 1,
        created_at: chrono::Utc::now(),
        metadata: HashMap::new(),
        api_response: None,
        execution_evidence: None,
    }
}

#[test]
fn immutable_completion_shares_data_and_credit_after_producer_is_dropped() {
    let response = response("中文 🦀");
    let bytes = semantic_cost(&response).unwrap();
    let budget = budget(bytes, bytes);
    let session = budget.try_open_session().unwrap();
    let output = InferenceOutput::new(response, session.try_reserve_semantic(0).unwrap()).unwrap();
    let held = output.clone();
    assert!(std::ptr::eq(output.response(), held.response()));
    assert_eq!(output.response().text, "中文 🦀");
    drop(session);
    drop(output);
    assert_eq!(
        (budget.usage().semantic_bytes, budget.usage().sessions),
        (bytes, 1)
    );
    drop(held);
    assert_eq!(
        (budget.usage().semantic_bytes, budget.usage().sessions),
        (0, 0)
    );
}

#[test]
fn failed_completion_freeze_returns_credit_instead_of_unbounded_output() {
    let response = response("cannot retain this completion");
    let required = semantic_cost(&response).unwrap();
    let budget = budget(required - 1, required - 1);
    let session = budget.try_open_session().unwrap();
    let grant = session.try_reserve_semantic(4).unwrap();
    assert!(InferenceOutput::new(response, grant).is_err());
    assert_eq!(budget.usage().semantic_bytes, 0);
}

#[test]
fn terminal_chunk_cannot_transfer_content_to_a_different_session() {
    let budget = budget(4096, 4096);
    let first = budget.try_open_session().unwrap();
    let second = budget.try_open_session().unwrap();
    let chunk = StreamChunk {
        request_id: RequestId::new(),
        text: "held terminal".into(),
        token: None,
        finish_reason: Some(FinishReason::Stop),
        usage: None,
        created_at: chrono::Utc::now(),
        metadata: HashMap::new(),
        api_response: None,
        execution_evidence: None,
    };
    let queued = first
        .try_reserve_queued(OutputChunk::queued_cost(&chunk).unwrap())
        .unwrap()
        .unwrap();
    let semantic = second.try_reserve_semantic(32).unwrap();
    assert!(OutputChunk::new_retained(chunk, queued, semantic).is_err());
    assert_eq!(
        (budget.usage().queued_bytes, budget.usage().semantic_bytes),
        (0, 0)
    );
}

#[test]
fn retained_terminal_chunk_keeps_both_pools_until_content_is_released() {
    let budget = budget(4096, 4096);
    let session = budget.try_open_session().unwrap();
    let chunk = StreamChunk {
        request_id: RequestId::new(),
        text: "terminal 尾".into(),
        token: None,
        finish_reason: Some(FinishReason::Stop),
        usage: Some(TokenUsage::new(1, 2)),
        created_at: chrono::Utc::now(),
        metadata: HashMap::new(),
        api_response: None,
        execution_evidence: None,
    };
    let wire_bytes = OutputChunk::queued_cost(&chunk).unwrap();
    let content_bytes = semantic_cost(&chunk).unwrap();
    let queued = session.try_reserve_queued(wire_bytes).unwrap().unwrap();
    let semantic = session.try_reserve_semantic(content_bytes).unwrap();
    let output = OutputChunk::new_retained(chunk, queued, semantic).unwrap();
    drop(session);
    assert_eq!(output.chunk().text, "terminal 尾");
    assert_eq!(
        (budget.usage().queued_bytes, budget.usage().semantic_bytes),
        (wire_bytes, content_bytes)
    );
    assert_eq!(budget.usage().sessions, 1);
    drop(output);
    assert_eq!(
        (budget.usage().queued_bytes, budget.usage().semantic_bytes),
        (0, 0)
    );
    assert_eq!(budget.usage().sessions, 0);
}

#[test]
fn logical_size_preserves_token_width_unicode_and_nonfree_empty_trees() {
    assert_eq!(semantic_cost(&TokenId(u32::MAX)).unwrap(), 4);
    assert_eq!(semantic_cost("中文🦀").unwrap(), 10);
    let many_empty = vec![Vec::<u32>::new(); 1024];
    assert!(semantic_cost(&many_empty).unwrap() >= many_empty.len());
    let escaped = "\"\n\\";
    assert_eq!(semantic_cost(escaped).unwrap(), escaped.len());
    assert!(serde_json::to_vec(escaped).unwrap().len() > semantic_cost(escaped).unwrap());
}
