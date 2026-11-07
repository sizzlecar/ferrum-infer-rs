//! Real token/session isolation across two executors sharing only static data.
//! The embedding/head fixture is attention-free; this is not KV arithmetic or
//! speculative draft-input qualification.
use super::*;

pub async fn assert_shared_token_lifecycle<R: DeviceRuntime>(
    first: ProgramExecutor<R>,
    second: ProgramExecutor<R>,
    weights: SharedProgramWeights<R>,
) {
    assert!(!Arc::ptr_eq(&first.plan_resources, &second.plan_resources));
    assert!(!Arc::ptr_eq(&first.lane, &second.lane));
    assert!(!Arc::ptr_eq(&first.sequences, &second.sequences));
    assert_eq!(
        first.shared_weights_receipt().unwrap().evidence(),
        weights.evidence()
    );
    assert_eq!(
        second.shared_weights_receipt().unwrap().evidence(),
        weights.evidence()
    );

    let left = token_execution::admit_tokens(
        &first,
        &RequestId::new(),
        &[TokenId::new(0), TokenId::new(1)],
    )
    .await;
    let right = token_execution::admit_tokens(&second, &RequestId::new(), &[TokenId::new(3)]).await;
    assert_ne!(left.sequence(), right.sequence());
    let policy = LogitsReturnPolicy::FullLogits;
    let partial = TokenWork::Prefill(PrefillChunk::new(0, 1, 2).unwrap());
    assert!(
        second
            .execute_tokens(&[ExecutionWork::new(&left, partial, &policy).unwrap()])
            .await
            .is_err(),
        "a common installation does not transfer sequence authority"
    );
    let first_row = token_execution::execute_one(&first, &left, partial).await;
    assert!(matches!(
        first_row.product(),
        ExecutionProduct::Intermediate
    ));
    assert_eq!(first_row.next_frontier().consumed_tokens(), 1);
    let second_row = token_execution::execute_one(
        &second,
        &right,
        TokenWork::Prefill(PrefillChunk::new(0, 1, 1).unwrap()),
    )
    .await;
    assert_eq!(
        sampling::sampling(&second_row),
        &ExecutorSamplingOutput::FullLogits(vec![1., 1.0009765625, -1., -2.])
    );
    assert_eq!(first.sequences.lock().total_len(), 1);
    assert_eq!(second.sequences.lock().total_len(), 1);

    // Closing the installation gates new consumers, but cannot invalidate any
    // original executor, sequence, flight or selected static binding.
    let weights = match weights
        .close()
        .unwrap_or_else(|e| panic!("close: {:?}", e.failure()))
    {
        SharedProgramWeightsCloseOutcome::Referenced { weights } => weights,
        _ => panic!("two admitted executors must retain the original installation"),
    };
    assert!(weights
        .validate_for(
            second.resolved_plan.execution_plan(),
            &second.runtime,
            weights.source()
        )
        .is_err());
    assert!(matches!(
        first.cancel_tokens(left.sequence()).unwrap().disposition,
        ExecutionCloseDisposition::Ended(_)
    ));
    assert_eq!(first.sequences.lock().total_len(), 0);
    assert_eq!(second.sequences.lock().total_len(), 1);
    drop((first_row, left));
    assert_executor_closed(first);

    let decoded = token_execution::execute_one(
        &second,
        second_row.next_frontier(),
        TokenWork::Decode {
            token: TokenId::new(1),
            position: 1,
        },
    )
    .await;
    assert_eq!(
        sampling::sampling(&decoded),
        &ExecutorSamplingOutput::FullLogits(vec![3., 2., 1., -4.])
    );
    assert_eq!(decoded.next_frontier().consumed_tokens(), 2);
    assert!(matches!(
        second
            .finish_tokens(
                right.sequence(),
                ExecutionEndReason::Completed {
                    input_tokens: 1,
                    output_tokens: 1
                }
            )
            .await
            .unwrap()
            .disposition,
        ExecutionCloseDisposition::Ended(_)
    ));
    assert_eq!(second.sequences.lock().total_len(), 0);
    drop((decoded, second_row, right));
    assert_executor_closed(second);
    let SharedProgramWeightsCloseOutcome::Closed(receipt) = weights
        .close()
        .unwrap_or_else(|e| panic!("last close: {:?}", e.failure()))
    else {
        panic!("last shared owner remained referenced")
    };
    assert!(receipt.released_static_resources() > 0);
}
