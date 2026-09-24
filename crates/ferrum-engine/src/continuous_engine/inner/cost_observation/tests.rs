use super::*;
mod calibration_capture;
mod host_stages;
pub(in crate::continuous_engine::inner) mod statistical_model;
use ferrum_scheduler::implementations::continuous::cost_model as model;
use std::{
    num::NonZeroUsize,
    sync::atomic::{AtomicU64, Ordering},
};

pub(in crate::continuous_engine::inner) struct VirtualClock(AtomicU64);
impl CostObservationClock for VirtualClock {
    fn now_ns(&self) -> Option<u64> {
        Some(self.0.load(Ordering::Relaxed))
    }
}
impl VirtualClock {
    fn set(&self, at: u64) {
        self.0.store(at, Ordering::Relaxed);
    }
}
fn identity() -> ExecutorCostIdentityAvailability {
    ExecutorCostIdentityAvailability::Known(Arc::new(ExecutorCostIdentity {
        schema_version: EXECUTOR_COST_IDENTITY_SCHEMA,
        model_weights: [1; 32],
        numerical_policy: [2; 32],
        device_runtime: [3; 32],
        execution_config: [4; 32],
    }))
}
fn shape(works: &[ActualRowWork]) -> ActualWaveShape {
    let decode = works
        .iter()
        .any(|work| matches!(work, ActualRowWork::Decode { .. }));
    let prefill = works
        .iter()
        .any(|work| matches!(work, ActualRowWork::Prefill { .. }));
    ActualWaveShape {
        statistical_evidence: None,
        kind: match (decode, prefill) {
            (true, true) => ActualWaveKind::Mixed,
            (true, false) => ActualWaveKind::Decode,
            _ => ActualWaveKind::Prefill,
        },
        path: ActualWavePath::PlanRuntime,
        graph: ActualWaveGraphState::Disabled,
        row_order: ActualWaveRowOrder::Ordered,
        provider_signature: [5; 32],
        output_policy_signature: [6; 32],
        numeric_features: None,
        host_content_features: None,
        row_multiset_features: None,
        rows: works
            .iter()
            .enumerate()
            .map(|(index, work)| ActualWaveRow {
                request_id: RequestId::new(),
                owner_incarnation: index as u64 + 1,
                work_generation: 1,
                input_index: index as u32,
                work: *work,
            })
            .collect(),
        recurrent_state_bytes: 64,
        restore_bytes: 0,
        maintenance_bytes: 0,
        maintenance_units: 0,
    }
}
fn sink(max_samples: usize, max_shape_rows: usize) -> Arc<BoundedCostSampleSink> {
    Arc::new(
        BoundedCostSampleSink::new(CostSampleSinkLimits {
            max_samples,
            max_shape_rows,
        })
        .unwrap(),
    )
}
fn begin(
    shape: &ActualWaveShape,
    sink: &Arc<BoundedCostSampleSink>,
) -> (EngineCostCall, Arc<VirtualClock>) {
    let clock = Arc::new(VirtualClock(AtomicU64::new(2)));
    let participants = shape
        .rows
        .iter()
        .map(|row| CostObservationParticipant {
            request_id: row.request_id.clone(),
            owner_incarnation: row.owner_incarnation,
            work_generation: row.work_generation,
            input_index: row.input_index,
            output_policy_signature: Some([6; 32]),
            host_features: None,
        })
        .collect();
    let call = EngineCostCall::begin(
        &EngineCostIds::default(),
        clock.clone(),
        Arc::clone(sink),
        EngineCostCallSpec {
            identity: identity(),
            participants,
            prepare_started_at_ns: Some(1),
            boundary: WaveObservationBoundary::IsolatedPreparationToCommit,
            recorder_limits: CostRecorderLimits {
                max_waves: 4,
                max_rows_per_wave: 8,
                max_retained_rows: 32,
            },
        },
    )
    .unwrap();
    (call, clock)
}
fn execute(call: &mut EngineCostCall, clock: &VirtualClock, shape: ActualWaveShape) {
    let mut context = call.context().unwrap();
    context.physical_wave(Ok(shape), Some(3));
    clock.set(6);
    context.terminal(ActualWaveOutcome::Completed, None);
    context.finish_call(ObservedCallOutcome::Completed);
}
fn committed(row: &ActualWaveRow, at: u64) -> HostCommitEvidence {
    let work = match row.work {
        ActualRowWork::Decode { kv_tokens } => HostCommittedWork::Decode {
            kv_tokens_before: kv_tokens,
            kv_tokens_after: kv_tokens + 1,
            generated_tokens_before: 2,
            generated_tokens_after: 3,
        },
        ActualRowWork::Prefill {
            offset,
            count,
            total_prompt_tokens,
        } => HostCommittedWork::Prefill {
            start: offset,
            end: offset + count,
            total_prompt_tokens,
            generated_tokens_before: 0,
            generated_tokens_after: u64::from(offset + count == total_prompt_tokens),
        },
        _ => unreachable!(),
    };
    HostCommitEvidence {
        request_id: row.request_id.clone(),
        owner_incarnation: row.owner_incarnation,
        work_generation: row.work_generation,
        input_index: row.input_index,
        outcome: HostCommitOutcome::Committed(work),
        committed_at_ns: Some(at),
    }
}
fn completed(shape: &ActualWaveShape, sink: &Arc<BoundedCostSampleSink>) -> EngineCostCall {
    let (mut call, clock) = begin(shape, sink);
    execute(&mut call, &clock, shape.clone());
    for row in &shape.rows {
        call.record_host_result(committed(row, 9));
    }
    clock.set(10);
    call
}

#[test]
fn exact_narrowed_prefill_wall_trains_a_real_scheduler_prediction() {
    // The requested chunk could have been eight tokens. Only these actual four
    // and the unchanged generated-token count describe the observed work.
    let shape = shape(&[ActualRowWork::Prefill {
        offset: 0,
        count: 4,
        total_prompt_tokens: 8,
    }]);
    let sink = sink(4, 16);
    assert_eq!(
        completed(&shape, &sink).finish(),
        CostCallDisposition::Published
    );
    let observation = sink.pop().unwrap();
    assert_eq!(observation.timing.wall_total_ns, 8);
    assert_eq!(observation.observed_at_ns, 10);
    assert_eq!(observation.actual_shape.prefill_chunks[0].count.get(), 4);
    assert_eq!(
        observation.timing.stages,
        model::WaveStageTimings::default()
    );
    let mut trainer = model::CostModelTrainer::new(
        observation.fingerprint.clone(),
        model::CostModelSettings {
            min_samples: NonZeroUsize::MIN,
            drift_margin_ns: 0,
            ..Default::default()
        },
    )
    .unwrap();
    let query = observation.clone();
    assert_eq!(
        trainer.observe(observation).unwrap(),
        model::ObservationDisposition::Recorded
    );
    let snapshot = trainer.publish(11).unwrap();
    assert!(matches!(
        snapshot.predict(&query.fingerprint, &query.actual_shape, query.boundary, 11),
        model::CostPrediction::Known(_)
    ));
}

#[test]
fn all_actual_rows_must_commit_once_despite_engine_ok_result() {
    let shape = shape(&[
        ActualRowWork::Decode { kv_tokens: 12 },
        ActualRowWork::Decode { kv_tokens: 24 },
    ]);
    for reason in [
        CostCallRejection::HostMissing,
        CostCallRejection::HostCancelled,
        CostCallRejection::HostFailed,
        CostCallRejection::HostDuplicate,
        CostCallRejection::FrontierMismatch,
        CostCallRejection::HostUnexpected,
    ] {
        let sink = sink(4, 16);
        let (mut call, clock) = begin(&shape, &sink);
        execute(&mut call, &clock, shape.clone());
        call.record_host_result(committed(&shape.rows[0], 9));
        let mut second = committed(&shape.rows[1], 9);
        match reason {
            CostCallRejection::HostMissing => {}
            CostCallRejection::HostCancelled => {
                second.outcome = HostCommitOutcome::Cancelled;
                call.record_host_result(second);
            }
            CostCallRejection::HostFailed => {
                second.outcome = HostCommitOutcome::Failed;
                call.record_host_result(second);
            }
            CostCallRejection::HostDuplicate => {
                call.record_host_result(second.clone());
                call.record_host_result(second);
            }
            CostCallRejection::FrontierMismatch => {
                second.work_generation += 1;
                call.record_host_result(second);
            }
            CostCallRejection::HostUnexpected => {
                second.request_id = RequestId::new();
                call.record_host_result(second);
            }
            _ => unreachable!(),
        }
        clock.set(10);
        assert_eq!(call.finish(), CostCallDisposition::Rejected(reason));
        assert!(sink.pop().is_none());
        assert_eq!(sink.stats().rejected(reason), 1);
    }
}

#[test]
fn planned_instead_of_actual_prefill_or_wrong_token_progress_cannot_train() {
    let works = [
        (
            ActualRowWork::Prefill {
                offset: 0,
                count: 4,
                total_prompt_tokens: 8,
            },
            HostCommittedWork::Prefill {
                start: 0,
                end: 8,
                total_prompt_tokens: 8,
                generated_tokens_before: 0,
                generated_tokens_after: 1,
            },
        ),
        (
            ActualRowWork::Prefill {
                offset: 4,
                count: 4,
                total_prompt_tokens: 8,
            },
            HostCommittedWork::Prefill {
                start: 4,
                end: 8,
                total_prompt_tokens: 8,
                generated_tokens_before: 0,
                generated_tokens_after: 0,
            },
        ),
        (
            ActualRowWork::Decode { kv_tokens: 12 },
            HostCommittedWork::Decode {
                kv_tokens_before: 12,
                kv_tokens_after: 13,
                generated_tokens_before: 2,
                generated_tokens_after: 4,
            },
        ),
        (
            ActualRowWork::Decode { kv_tokens: 12 },
            HostCommittedWork::Decode {
                kv_tokens_before: 12,
                kv_tokens_after: 14,
                generated_tokens_before: 2,
                generated_tokens_after: 3,
            },
        ),
    ];
    for (actual, wrong) in works {
        let shape = shape(&[actual]);
        let sink = sink(4, 16);
        let (mut call, clock) = begin(&shape, &sink);
        execute(&mut call, &clock, shape.clone());
        let mut evidence = committed(&shape.rows[0], 9);
        evidence.outcome = HostCommitOutcome::Committed(wrong);
        call.record_host_result(evidence);
        clock.set(10);
        assert_eq!(
            call.finish(),
            CostCallDisposition::Rejected(CostCallRejection::WorkMismatch)
        );
    }
}

#[test]
fn cancellation_retains_no_fabricated_terminal_and_is_counted() {
    let shape = shape(&[ActualRowWork::Decode { kv_tokens: 12 }]);
    let sink = sink(4, 16);
    let (mut call, _) = begin(&shape, &sink);
    {
        let mut context = call.context().unwrap();
        context.physical_wave(Ok(shape), Some(3));
    }
    assert!(call.observations()[0].outcome.is_none());
    assert!(call.observations()[0].host_committed_at_ns.is_none());
    drop(call);
    assert_eq!(sink.stats().rejected(CostCallRejection::Abandoned), 1);
    assert!(sink.pop().is_none());
}

#[test]
fn composite_executor_only_and_unknown_shape_never_train() {
    let shape = shape(&[ActualRowWork::Decode { kv_tokens: 12 }]);
    for composite in [false, true] {
        let sink = sink(4, 16);
        let (mut call, clock) = begin(&shape, &sink);
        {
            let mut context = call.context().unwrap();
            context.physical_wave(
                if composite {
                    Ok(shape.clone())
                } else {
                    Err(ActualWaveEvidenceUnknown::GraphPath)
                },
                Some(3),
            );
            clock.set(6);
            context.terminal(ActualWaveOutcome::Completed, None);
            if composite {
                context.physical_wave(Ok(shape.clone()), Some(7));
                clock.set(8);
                context.terminal(ActualWaveOutcome::Completed, None);
            }
            context.finish_call(ObservedCallOutcome::Completed);
        }
        call.record_host_result(committed(&shape.rows[0], 9));
        clock.set(10);
        assert_eq!(
            call.finish(),
            CostCallDisposition::Rejected(if composite {
                CostCallRejection::Composite
            } else {
                CostCallRejection::ActualEvidenceUnknown
            })
        );
    }
    let sink = sink(4, 16);
    let mut call = completed(&shape, &sink);
    call.boundary = WaveObservationBoundary::ExecutorOnly;
    assert_eq!(
        call.finish(),
        CostCallDisposition::Rejected(CostCallRejection::Composite)
    );
}

#[test]
fn host_clock_reversal_and_missing_identity_are_explicit() {
    let shape = shape(&[ActualRowWork::Decode { kv_tokens: 12 }]);
    let sink = sink(4, 16);
    let (mut call, clock) = begin(&shape, &sink);
    execute(&mut call, &clock, shape.clone());
    call.record_host_result(committed(&shape.rows[0], 5));
    clock.set(10);
    assert_eq!(
        call.finish(),
        CostCallDisposition::Rejected(CostCallRejection::Clock)
    );
    let mut call = completed(&shape, &sink);
    call.identity = ExecutorCostIdentityAvailability::default();
    assert_eq!(
        call.finish(),
        CostCallDisposition::Rejected(CostCallRejection::IdentityUnknown)
    );
}

#[test]
fn bounded_sink_drops_without_refreshing_or_overwriting_retained_sample() {
    let shape = shape(&[ActualRowWork::Decode { kv_tokens: 12 }]);
    let sink = sink(1, 1);
    assert_eq!(
        completed(&shape, &sink).finish(),
        CostCallDisposition::Published
    );
    assert_eq!(
        completed(&shape, &sink).finish(),
        CostCallDisposition::Dropped(CostSampleDrop::Capacity)
    );
    assert!(sink.stats().has_lost_samples());
    assert_eq!(sink.pop().unwrap().observed_at_ns, 10);
    assert!(sink.pop().is_none());
    assert_eq!(
        completed(&shape, &sink).finish(),
        CostCallDisposition::Published
    );
}

#[test]
fn canonical_shape_key_is_preserved_by_scheduler_adapter() {
    fn canonicalize(shape: &mut ActualWaveShape) {
        let mut canonical = CanonicalWaveCostBuilder::new(0, CostProductOutput::GreedyToken);
        canonical
            .physical_command(CostPhysicalCommand {
                statistical_evidence: None,
                native_op_id: "fixture.compute",
                node_index: None,
                command_phase: ferrum_interfaces::vnext::DeviceCommandPhase::Compute,
                compute_dispatch_count: 1,
                transfer_command_count: 0,
                reusable_graph_node_count: None,
                command_index: 0,
                provider: None,
                path: CostCommandPath::Eager,
                participant_start: 0,
                participant_count: shape.rows.len() as u32,
                token_count: 5,
                batching_form: "mixed",
            })
            .unwrap();
        for row in &shape.rows {
            let output = match row.work {
                ActualRowWork::Decode { .. } => CostRowOutput::Decode {
                    requires_full_logits: false,
                    repetition_tokens: 0,
                    repetition_penalty_bits: 1.0f32.to_bits(),
                },
                ActualRowWork::Prefill {
                    offset,
                    count,
                    total_prompt_tokens,
                } => CostRowOutput::Prefill {
                    final_logits: offset + count == total_prompt_tokens,
                },
                _ => unreachable!(),
            };
            canonical
                .row(CanonicalCostRow {
                    work: row.work,
                    host_policy_signature: [9; 32],
                    host_features: None,
                    mask_upload_required: false,
                    output,
                })
                .unwrap();
        }
        let canonical = canonical
            .finish(
                shape.kind,
                shape.path,
                shape.graph,
                shape.row_order,
                shape.recurrent_state_bytes,
            )
            .unwrap();
        shape.provider_signature = canonical.provider_signature;
        shape.output_policy_signature = canonical.output_policy_signature;
    }
    let mut first = shape(&[
        ActualRowWork::Decode { kv_tokens: 12 },
        ActualRowWork::Prefill {
            offset: 0,
            count: 4,
            total_prompt_tokens: 8,
        },
    ]);
    let mut second = first.clone();
    second.rows.swap(0, 1);
    canonicalize(&mut first);
    canonicalize(&mut second);
    let sink = sink(4, 16);
    assert_eq!(
        completed(&first, &sink).finish(),
        CostCallDisposition::Published
    );
    assert_eq!(
        completed(&second, &sink).finish(),
        CostCallDisposition::Published
    );
    let first_sample = sink.pop().unwrap().actual_shape;
    let second_sample = sink.pop().unwrap().actual_shape;
    assert_eq!(
        first_sample.decode_kv_tokens,
        second_sample.decode_kv_tokens
    );
    assert_eq!(first_sample.prefill_chunks, second_sample.prefill_chunks);
    assert_eq!(first_sample.provider_signature, first.provider_signature);
    assert_eq!(
        first_sample.output_policy_signature,
        first.output_policy_signature
    );
    assert_eq!(second_sample.provider_signature, second.provider_signature);
    assert_ne!(
        first_sample.provider_signature,
        second_sample.provider_signature
    );
}

#[test]
fn failed_or_unsubmitted_dispatch_cannot_be_relabelled_by_host_success() {
    let shape = shape(&[ActualRowWork::Decode { kv_tokens: 12 }]);
    for submitted in [false, true] {
        let sink = sink(4, 16);
        let (mut call, clock) = begin(&shape, &sink);
        {
            let mut context = call.context().unwrap();
            if submitted {
                context.physical_wave(Ok(shape.clone()), Some(3));
                clock.set(6);
                context.terminal(ActualWaveOutcome::FailedAfterSubmit, None);
                context.finish_call(ObservedCallOutcome::Failed);
            } else {
                context.finish_call(ObservedCallOutcome::NotSubmitted);
            }
        }
        call.record_host_result(committed(&shape.rows[0], 9));
        clock.set(10);
        assert_eq!(
            call.finish(),
            CostCallDisposition::Rejected(if submitted {
                CostCallRejection::ExecutorFailed
            } else {
                CostCallRejection::NoPhysicalWave
            })
        );
        assert!(sink.pop().is_none());
    }
}

#[test]
fn final_prefill_commits_exactly_one_token_and_foreign_incarnation_is_rejected() {
    let shape = shape(&[ActualRowWork::Prefill {
        offset: 4,
        count: 4,
        total_prompt_tokens: 8,
    }]);
    let sink = sink(4, 16);
    assert_eq!(
        completed(&shape, &sink).finish(),
        CostCallDisposition::Published
    );
    let (mut call, clock) = begin(&shape, &sink);
    execute(&mut call, &clock, shape.clone());
    let mut evidence = committed(&shape.rows[0], 9);
    evidence.owner_incarnation += 1;
    call.record_host_result(evidence);
    clock.set(10);
    assert_eq!(
        call.finish(),
        CostCallDisposition::Rejected(CostCallRejection::FrontierMismatch)
    );
}

#[test]
fn shape_memory_and_contended_consumer_are_bounded_explicit_loss() {
    let shape = shape(&[ActualRowWork::Decode { kv_tokens: 12 }]);
    let sink = sink(4, 1);
    assert_eq!(
        completed(&shape, &sink).finish(),
        CostCallDisposition::Published
    );
    assert_eq!(
        completed(&shape, &sink).finish(),
        CostCallDisposition::Dropped(CostSampleDrop::Capacity)
    );
    sink.pop().unwrap();
    let call = completed(&shape, &sink);
    let disposition = sink.with_locked_queue(|| call.finish());
    assert_eq!(
        disposition,
        CostCallDisposition::Dropped(CostSampleDrop::Contended)
    );
    let stats = sink.stats();
    assert_eq!(stats.dropped_capacity, 1);
    assert_eq!(stats.dropped_contention, 1);
    assert!(stats.has_lost_samples());
    assert!(sink.pop().is_none());
}

#[test]
fn absent_host_output_policy_and_reused_context_do_not_train() {
    let shape = shape(&[ActualRowWork::Decode { kv_tokens: 12 }]);
    let sink = sink(4, 16);
    let mut call = completed(&shape, &sink);
    call.participants[0].output_policy_signature = None;
    assert_eq!(
        call.finish(),
        CostCallDisposition::Rejected(CostCallRejection::OutputPolicyUnknown)
    );
    let mut call = completed(&shape, &sink);
    assert!(matches!(call.context(), Err(CostCallRejection::Composite)));
    assert_eq!(
        call.finish(),
        CostCallDisposition::Rejected(CostCallRejection::Composite)
    );
}

#[test]
fn runtime_consumes_real_samples_and_publishes_a_usable_snapshot() {
    let clock = Arc::new(VirtualClock(AtomicU64::new(11)));
    let runtime = EngineCostRuntime::with_clock(identity(), clock).unwrap();
    let shape = shape(&[ActualRowWork::Decode { kv_tokens: 12 }]);
    let mut query = None;
    for _ in 0..model::CostModelSettings::default().min_samples.get() {
        let mut call = completed(&shape, &runtime.sink);
        if query.is_none() {
            query = Some(call.make_sample().unwrap());
            // make_sample closes the recorder, so enqueue that sample directly.
            call.finished = true;
            runtime.sink.offer(query.as_ref().unwrap().clone()).unwrap();
        } else {
            assert_eq!(call.finish(), CostCallDisposition::Published);
        }
    }
    runtime.consume_samples();
    let query = query.unwrap();
    assert_eq!(
        runtime.trained_samples(),
        model::CostModelSettings::default().min_samples.get() as u64
    );
    assert!(runtime.sink.pop().is_none());
    let snapshot = runtime.snapshot().unwrap();
    assert!(matches!(
        snapshot.predict(&query.fingerprint, &query.actual_shape, query.boundary, 11),
        model::CostPrediction::Known(_)
    ));
}

fn queued_training_sample(sink: &Arc<BoundedCostSampleSink>) -> model::WaveCostObservation {
    let shape = shape(&[ActualRowWork::Decode { kv_tokens: 12 }]);
    let mut call = completed(&shape, sink);
    let sample = call.make_sample().unwrap();
    call.finished = true;
    // This helper bypasses finish/offer to exercise delayed worker receipts.
    call.sink.call_finished();
    sample
}

#[test]
fn observation_call_funnel_separates_completed_rejected_and_abandoned_calls() {
    let sink = sink(1, 8);
    let shape = shape(&[ActualRowWork::Decode { kv_tokens: 12 }]);
    assert_eq!(
        completed(&shape, &sink).finish(),
        CostCallDisposition::Published
    );
    assert_eq!(
        completed(&shape, &sink).finish(),
        CostCallDisposition::Dropped(CostSampleDrop::Capacity)
    );
    let (mut failed, _) = begin(&shape, &sink);
    failed.reject(CostCallRejection::HostFailed);
    assert_eq!(
        failed.finish(),
        CostCallDisposition::Rejected(CostCallRejection::HostFailed)
    );
    drop(begin(&shape, &sink).0);
    sink.reject_initialization(CostCallRejection::IdExhausted);
    sink.reject_preparation(CostCallRejection::NoPhysicalWave);
    let stats = sink.stats();
    assert_eq!((stats.calls_started, stats.calls_finished), (4, 4));
    assert_eq!(
        (stats.offered, stats.published, stats.dropped_capacity),
        (2, 1, 1)
    );
    assert_eq!(stats.rejected(CostCallRejection::HostFailed), 1);
    assert_eq!(stats.rejected(CostCallRejection::Abandoned), 1);
    assert_eq!(
        stats.calls_finished,
        stats.offered + stats.rejected.iter().sum::<u64>()
    );
    assert_eq!(
        stats.initialization_rejected[CostCallRejection::IdExhausted.index()],
        1
    );
    assert_eq!(
        stats.preparation_rejected[CostCallRejection::NoPhysicalWave.index()],
        1
    );
}

#[test]
fn bounded_training_preserves_receipt_order_and_does_not_refresh_old_evidence() {
    let clock = Arc::new(VirtualClock(AtomicU64::new(1000)));
    let config = ferrum_types::SloCostObservationConfig {
        max_samples_per_update: NonZeroUsize::new(2).unwrap(),
        ..Default::default()
    };
    let runtime = EngineCostRuntime::build(identity(), clock.clone(), &config, false).unwrap();
    let sample = queued_training_sample(&runtime.sink);
    let n = model::CostModelSettings::default().min_samples.get();
    for index in 0..n {
        let mut sample = sample.clone();
        sample.observed_at_ns = 11 + index as u64;
        runtime.sink.offer(sample).unwrap();
    }
    runtime.consume_samples();
    assert_eq!(runtime.trained_samples(), 2);
    assert_eq!(runtime.sink.stats().drained, 2);
    for _ in 2..n {
        runtime.consume_samples();
    }
    assert_eq!(runtime.trained_samples(), n as u64);
    let snapshot = runtime.snapshot().unwrap();
    let model::CostPrediction::Known(prediction) = snapshot.predict(
        &sample.fingerprint,
        &sample.actual_shape,
        sample.boundary,
        clock.now_ns().unwrap(),
    ) else {
        panic!("queued samples preceding the consumer's wall clock must train");
    };
    assert_eq!(prediction.oldest_sample_at_ns, 11);
    assert_eq!(prediction.newest_sample_at_ns, 10 + n as u64);
    clock.set(model::CostModelSettings::default().max_sample_age_ns.get() + 11 + n as u64);
    assert!(matches!(
        snapshot.predict(
            &sample.fingerprint,
            &sample.actual_shape,
            sample.boundary,
            clock.now_ns().unwrap(),
        ),
        model::CostPrediction::Unknown(model::CostUnknownReason::StaleSamples)
    ));
}

#[tokio::test]
async fn worker_shutdown_drains_multiple_batches_without_another_inference_wave() {
    let config = ferrum_types::SloCostObservationConfig {
        max_samples_per_update: NonZeroUsize::new(2).unwrap(),
        ..Default::default()
    };
    let runtime = EngineCostRuntime::new(identity(), &config, None).unwrap();
    let sample = queued_training_sample(&runtime.sink);
    let n = model::CostModelSettings::default().min_samples.get();
    // The sink deliberately permits contention loss. Preload without a racing
    // pop so this test isolates multi-batch shutdown rather than OS scheduling.
    runtime.with_training_paused(|| {
        for _ in 0..n {
            runtime.sink.offer(sample.clone()).unwrap();
        }
    });
    runtime.shutdown().await.unwrap();
    assert_eq!(runtime.trained_samples(), n as u64);
    assert_eq!(runtime.sink.stats().drained, n as u64);
    assert!(runtime.snapshot().is_some());
    // Repeated product shutdown must not create a worker or replay samples.
    runtime.shutdown().await.unwrap();
    assert_eq!(runtime.trained_samples(), n as u64);
}

#[test]
fn dropping_runtime_joins_worker_and_releases_its_retained_state() {
    let runtime = EngineCostRuntime::new(identity(), &Default::default(), None).unwrap();
    let sink = runtime.sink.clone();
    sink.offer(queued_training_sample(&sink)).unwrap();
    drop(runtime);
    assert_eq!(sink.stats().drained, 1);
    assert!(sink.pop().is_none());
    assert_eq!(
        Arc::strong_count(&sink),
        1,
        "worker must not retain state after owner drop"
    );
}

#[tokio::test]
async fn publishing_sample_wakes_training_without_a_batch_completion_notification() {
    let runtime = EngineCostRuntime::new(identity(), &Default::default(), None).unwrap();
    runtime
        .sink
        .offer(queued_training_sample(&runtime.sink))
        .unwrap();
    // There is intentionally no process_batch/wake_trainer call. An enclosing
    // future can disappear after a successful leaf publishes its receipt.
    tokio::time::timeout(std::time::Duration::from_secs(3), async {
        while runtime.trained_samples() == 0 {
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("sample publication failed to wake the parked trainer");
    assert_eq!(runtime.sink.stats().drained, 1);
    runtime.shutdown().await.unwrap();
}

#[tokio::test]
async fn cancelling_shutdown_waiter_cannot_let_a_retry_skip_the_running_worker() {
    let runtime = Arc::new(EngineCostRuntime::new(identity(), &Default::default(), None).unwrap());
    let (held, waiting) = std::sync::mpsc::channel();
    let (release, resume) = std::sync::mpsc::channel();
    let paused = runtime.clone();
    let holder = std::thread::spawn(move || {
        paused.with_training_paused(|| {
            held.send(()).unwrap();
            let _ = resume.recv();
        });
    });
    waiting
        .recv_timeout(std::time::Duration::from_secs(3))
        .unwrap();
    runtime
        .sink
        .offer(queued_training_sample(&runtime.sink))
        .unwrap();
    let first_runtime = runtime.clone();
    let first = tokio::spawn(async move { first_runtime.shutdown().await });
    tokio::time::timeout(std::time::Duration::from_secs(3), async {
        while !runtime.shutdown_started() {
            tokio::task::yield_now().await;
        }
    })
    .await
    .unwrap();
    first.abort();
    assert!(first.await.unwrap_err().is_cancelled());
    let retry = runtime.shutdown();
    tokio::pin!(retry);
    assert!(
        futures::poll!(retry.as_mut()).is_pending(),
        "retry returned while the trainer was held"
    );
    release.send(()).unwrap();
    retry.await.unwrap();
    holder.join().unwrap();
    assert_eq!(runtime.trained_samples(), 1);
    assert_eq!(runtime.sink.stats().drained, 1);
}
