use super::*;
use crate::test_support::{tiny_llama_model, tiny_model_info, TinyLlamaConfig};
use ferrum_execution::model_executor::{
    ExecutorPrefillAdmission, PersistentCapacityInspectionBudget, PersistentCapacityMember,
    RequestLifecycleId,
};

async fn admit(executor: &LlmExecutor, id: &RequestId) -> ExecutionFrontier {
    let tokens = [TokenId::new(1)];
    match executor
        .admit_tokens(ExecutionAdmission {
            context: ExecutorPrefillAdmission::for_startup(id, &tokens, 32),
            adapter: None,
        })
        .await
        .unwrap()
    {
        TokenAdmissionOutcome::Admitted(frontier) => frontier,
        other => panic!("actual CPU admission required: {other:?}"),
    }
}

#[tokio::test]
async fn persistent_capacity_executor_binds_exact_live_incarnations_and_all_owners() {
    let config = TinyLlamaConfig {
        max_seq_len: 64,
        ..Default::default()
    };
    let mut model = tiny_llama_model(&config);
    model.runtime_env.metal_paged_kv = Some(true);
    model.runtime_env.prefix_cache = false;
    model.runtime_env.kv_capacity = Some(64);
    model.runtime_env.kv_max_blocks = Some(4);
    model.runtime_env.paged_max_seqs = 2;
    let executor = LlmExecutor::new(Box::new(model), tiny_model_info(&config));
    let first_id = RequestId::new();
    let first = admit(&executor, &first_id).await;
    let second = admit(&executor, &RequestId::new()).await;
    let mut members = vec![PersistentCapacityMember {
        request_id: first_id.clone(),
        lifecycle: RequestLifecycleId {
            producer: 1,
            admission: 1,
        },
        state: PersistentCapacityState::Resident {
            sequence: first.sequence().clone(),
        },
        through_tokens: 32,
    }];
    for admission in 2..=5 {
        members.push(PersistentCapacityMember {
            request_id: RequestId::new(),
            lifecycle: RequestLifecycleId {
                producer: 1,
                admission,
            },
            state: PersistentCapacityState::Waiting,
            through_tokens: 32,
        });
    }
    let observe = |members: &[PersistentCapacityMember]| {
        executor.persistent_cohort_capacity(&PersistentCapacityQuery {
            members,
            max_live_sequences: 2,
            max_wave_sequences: 2,
            inspection_budget: PersistentCapacityInspectionBudget::default(),
        })
    };
    assert_eq!(observe(&members), Err(U::MembershipChanged));
    assert!(matches!(
        executor
            .cancel_tokens(second.sequence())
            .unwrap()
            .disposition,
        ExecutionCloseDisposition::Ended(_)
    ));
    assert_eq!(observe(&members).unwrap().members(), members.as_slice());
    let held = executor.model.lock();
    assert_eq!(observe(&members), Err(U::Contended));
    drop(held);
    executor
        .token_sequences
        .lock()
        .live
        .get_mut(first.sequence())
        .unwrap()
        .closing = true;
    assert_eq!(observe(&members), Err(U::PendingOwner));
    executor
        .token_sequences
        .lock()
        .live
        .get_mut(first.sequence())
        .unwrap()
        .closing = false;
    members[0].through_tokens = 31;
    assert_eq!(observe(&members), Err(U::MembershipChanged));
    members[0].through_tokens = 32;
    assert!(matches!(
        executor
            .cancel_tokens(first.sequence())
            .unwrap()
            .disposition,
        ExecutionCloseDisposition::Ended(_)
    ));
    let replacement = admit(&executor, &first_id).await;
    assert_eq!(
        observe(&members),
        Err(U::MembershipChanged),
        "same request ID does not reuse an incarnation proof"
    );
    members[0].state = PersistentCapacityState::Resident {
        sequence: replacement.sequence().clone(),
    };
    assert!(observe(&members).is_ok());
    let closed = executor.cancel_tokens(replacement.sequence()).unwrap();
    assert!(closed.error.is_none());
    assert!(matches!(
        closed.disposition,
        ExecutionCloseDisposition::Ended(_)
    ));
}
