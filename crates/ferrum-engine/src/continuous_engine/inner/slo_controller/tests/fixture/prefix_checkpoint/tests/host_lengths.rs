use super::*;

struct ExactHostShapeGuard {
    expected: model::WaveExecutionShape,
    calls: AtomicUsize,
}
impl vnext::CheckpointTransferSubmissionGuard for ExactHostShapeGuard {
    fn check(
        &self,
        actual: &vnext::PreparedCheckpointTransfer<'_>,
    ) -> std::result::Result<(), GuardedNotSubmittedReason> {
        self.calls.fetch_add(1, Ordering::AcqRel);
        let shape = prefix_cost_shape(actual.cost_domain(), actual.host_work())
            .map_err(|_| GuardedNotSubmittedReason::AttributionUnavailable)?;
        if shape != self.expected {
            return Err(GuardedNotSubmittedReason::ActualRouteMismatch);
        }
        Ok(())
    }
}

#[tokio::test]
async fn native_prefix_cpu_exact_host_lengths_separate_longer_restore_and_guard() {
    let config = ferrum_types::SloCostObservationConfig::default();
    let minimum = config.model.min_samples.get();
    let (_, executor) = startup_checkpoint_components(minimum + 2).await;
    let runtime = EngineCostRuntime::build(
        executor.execution_cost_identity(),
        Arc::new(Clock(AtomicU64::new(100))),
        &config,
        false,
    )
    .unwrap();
    let tap = Tap::new(Some(runtime.prefix_cost_sink().unwrap()));
    install(&executor, &tap);
    let source = input();
    let source_output = partial(&executor, &source);
    let guard = Guard::default();
    let lease = capture(&executor, &source, &guard);
    let mut retained = Vec::new();
    for _ in 0..minimum {
        let target = input();
        retained.push(
            restored(&executor, &target, lease.as_ref(), &guard)
                .acknowledge()
                .unwrap(),
        );
        runtime.drain_calibration_fixture();
    }
    let short = tap.records.lock().last().unwrap().clone();
    let long = PlanRuntimePrefillInput::new(
        RequestId::new(),
        Arc::<[ferrum_types::TokenId]>::from(vec![ferrum_types::TokenId::new(5); 5]),
        8,
        PrefillChunk::new(0, 2, 5).unwrap(),
    )
    .unwrap();
    let offer = offer(&source.request_id, &long.request_id);
    assert!(predict(&runtime, &executor, &short, &offer));
    admit(&executor, &long);
    let cache = source_output.output().kv_cache().cache_id();
    let fresh = view(
        &executor,
        &[(&source.request_id, Some(&cache)), (&long.request_id, None)],
    );
    let bound = known(executor.bind_execution_retained_checkpoint(
        &fresh,
        &fresh.initial_state(),
        lease.as_ref(),
        0,
        &mut || true,
    ));
    let projected = known(executor.project_execution_checkpoint(
        &fresh,
        &bound.state,
        vnext::FutureCheckpointCostQuery::Restore {
            checkpoint: &bound.checkpoint,
            target: 1,
            prompt_tokens: 5,
        },
        &mut || true,
    ));
    assert_eq!(projected.cost_domain, short.domain);
    assert_eq!(projected.host_work.prefix_tokens(), 2);
    assert_eq!(projected.host_work.full_input_tokens(), 5);
    let short_shape = prefix_cost_shape(&short.domain, short.host_work.as_ref()).unwrap();
    let long_shape = prefix_cost_shape(&projected.cost_domain, Some(&projected.host_work)).unwrap();
    assert_ne!(short_shape, long_shape);
    let snapshot = runtime.prefix_cost_snapshot().unwrap();
    assert!(matches!(
        snapshot.diagnostic_prediction(&fingerprint(&executor), &long_shape, 100),
        model::CostPrediction::Unknown(model::CostUnknownReason::UnobservedBucket)
    ));

    // The same physical copy cannot consume a witness for a shorter host input.
    // Rejection reaches the native final callback and submits no extra copy.
    let strict = ExactHostShapeGuard {
        expected: short_shape,
        calls: AtomicUsize::new(0),
    };
    let before = executor.native_structured_counts();
    let before_copies = copies(&executor).len();
    assert!(matches!(
        executor
            .prefix_restore(
                PlanRuntimePrefixRestoreInput {
                    request_id: &long.request_id,
                    input_tokens: &long.input_tokens,
                    maximum_sequence_tokens: long.maximum_sequence_tokens,
                    checkpoint: Some(lease.as_ref()),
                    retry: None,
                },
                &strict,
            )
            .unwrap(),
        PlanRuntimePrefixRestoreOutcome::Unavailable
    ));
    assert_eq!(strict.calls.load(Ordering::Acquire), 1);
    assert_eq!(executor.native_structured_counts(), before);
    assert_eq!(copies(&executor).len(), before_copies);
    let authority = restored(&executor, &long, lease.as_ref(), &guard)
        .acknowledge()
        .unwrap();
    let actual = tap.records.lock().last().unwrap().clone();
    assert_eq!(actual.domain, projected.cost_domain);
    assert_eq!(actual.host_work, Some(projected.host_work));
    assert_eq!(actual.domain.geometry(), short.domain.geometry());
    runtime.drain_calibration_fixture();
    if minimum > 1 {
        assert!(!predict(&runtime, &executor, &actual, &offer));
    }
    drop((retained, authority, source_output, lease));
}
