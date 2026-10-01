//! The next request receives a fresh logical/native owner after completion.
//! A retained checkpoint survives that lifecycle and does not resurrect old KV.
use super::*;

#[tokio::test]
async fn native_prefix_cpu_restored_decode_and_completed_slot_recycle() {
    let (_, executor) = startup_checkpoint_components(2).await;
    executor
        .recycle_completed_bindings
        .store(true, Ordering::Release);
    let source = input();
    let source_output = partial(&executor, &source);
    let guard = Guard::default();
    let lease = capture(&executor, &source, &guard);
    let target = input();
    let restored_authority = restored(&executor, &target, lease.as_ref(), &guard)
        .acknowledge()
        .unwrap();
    let suffix = PlanRuntimePrefillInput::new(
        target.request_id.clone(),
        target.input_tokens.clone(),
        8,
        PrefillChunk::new(2, 2, 4).unwrap(),
    )
    .unwrap();
    let mut logits = executor
        .native_structured_output(None, &[suffix.clone()], &[], &AllowHost)
        .unwrap()
        .unwrap();
    let suffix_output = executor
        .prefill_output_from_logits(&suffix, logits.remove(0))
        .unwrap();
    let decode = PlanRuntimeDecodeInput::new(
        target.request_id.clone(),
        ferrum_types::TokenId::new(6),
        suffix_output.output().kv_cache().clone(),
    );
    let mut logits = executor
        .native_structured_output(None, &[], &[decode.clone()], &AllowHost)
        .unwrap()
        .unwrap();
    let decoded = executor.decode_output_from_logits(&decode, logits.remove(0));
    assert_eq!(decoded.kv_cache.num_tokens(), 5);
    let old_owner = executor
        .prefix_session(&target.request_id)
        .unwrap()
        .sequence_authority();
    let cache = decoded.kv_cache.cache_id();
    let old_view = view(&executor, &[(&target.request_id, Some(&cache))]);
    executor
        .complete_cache(
            ExecutorSequenceCompletion::new(target.request_id.clone(), cache, 4, 2).unwrap(),
        )
        .await
        .unwrap();
    drop((restored_authority, suffix_output, decode, decoded));

    // Reusing the slot follows the original completed-binding gate. The source
    // remains live, so only the completed target may be replaced.
    let next = input();
    admit(&executor, &next);
    let new_view = view(&executor, &[(&next.request_id, None)]);
    let new_owner = executor
        .prefix_session(&next.request_id)
        .unwrap()
        .sequence_authority();
    assert_ne!(old_owner, new_owner);
    assert!(!old_view
        .resource_view()
        .same_live_evidence(new_view.resource_view()));
    assert_eq!(lease.status(), PrefixCaptureStatus::Ready);
    let restored_again = restored(&executor, &next, lease.as_ref(), &guard)
        .acknowledge()
        .unwrap();
    assert_eq!(restored_again.committed_tokens(), 2);
    assert_eq!(
        copies(&executor).last().unwrap(),
        &2u32.to_le_bytes().to_vec()
    );
    assert_eq!(source_output.output().kv_cache().num_tokens(), 2);
}

#[tokio::test]
async fn native_prefix_cpu_cancelled_restore_retires_exact_cache_for_cold_residency() {
    use ferrum_interfaces::model_executor::{
        TokenPolicyResidencyInvalidation as Outcome, TokenPolicyResidencyUnavailable as Busy,
    };
    let (_, executor) = startup_checkpoint_components(2).await;
    executor
        .recycle_completed_bindings
        .store(true, Ordering::Release);
    let source = input();
    let source_output = partial(&executor, &source);
    let source_cache = source_output.output().kv_cache().cache_id();
    let guard = Guard::default();
    let lease = capture(&executor, &source, &guard);
    let target = input();
    let restored = restored(&executor, &target, lease.as_ref(), &guard)
        .acknowledge()
        .unwrap();
    let target_cache = restored.kv_cache().cache_id();
    assert_eq!(
        executor.native_structured_history.lock()[&target.request_id].len(),
        2
    );
    let before = executor.native_structured_counts();
    // A still-admitted request cannot be made cold by releasing a cache ref.
    executor.release_cache(&target_cache);
    assert!(executor
        .native_structured_history
        .lock()
        .contains_key(&target.request_id));
    assert_eq!(
        executor.calibration_invalidate_token_policy_residency(),
        Outcome::Unavailable {
            reason: Busy::ActiveRequests
        }
    );
    assert!(executor.cancel_prefill_admission(&target.request_id));
    assert!(executor.cancel_prefill_admission(&source.request_id));
    executor.release_cache(&target_cache);
    executor.release_cache(&source_cache);
    assert!(executor.native_structured_history.lock().is_empty());
    // Real external cache ownership still vetoes the cold boundary.
    assert_eq!(
        executor.calibration_invalidate_token_policy_residency(),
        Outcome::Unavailable {
            reason: Busy::ActiveRequests
        }
    );
    drop(restored);
    drop(source_output);
    assert_eq!(
        executor.calibration_invalidate_token_policy_residency(),
        Outcome::Cleared { cleared_entries: 0 }
    );
    assert_eq!(executor.native_structured_counts(), before);
    // A published immutable checkpoint can outlive its retired source without
    // representing a live token-policy owner or losing its real backing lease.
    assert_eq!(lease.status(), PrefixCaptureStatus::Ready);
}
