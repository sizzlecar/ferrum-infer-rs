//! Native CPU mixed work -> Engine facts -> one ordered whole-attempt cost.
//! The tiny program has stateless KV: this does not qualify prefix restoration,
//! physical attention geometry, client SLOs, or performance.

use async_trait::async_trait;
use ferrum_engine::EngineBuilder;
use ferrum_execution::{
    execution_attempt::*,
    executor_lifecycle::{
        ExecutorOwner, ExecutorScope, ExecutorShutdownObserver, ExecutorShutdownPermit,
    },
    model_executor::*,
    token_execution::*,
    ProductExecutionEventSink,
};
use ferrum_interfaces::Tokenizer;
use ferrum_kernels::backend::cpu::vnext_ops::CpuVNextComposition;
use ferrum_models::test_support::native_program::{Fixture, FixtureOptions, VOCAB};
use ferrum_runtime::{
    CapacityAvailabilityEpoch, CapacityAvailabilitySource, CapacityWaitCondition, DeviceId,
    ResolvedModelPlan,
};
use ferrum_scheduler::cost_prediction::*;
use ferrum_slo::{ReferenceEstimator, ReferenceEstimatorConfig};
use ferrum_testkit::MockTokenizer;
use ferrum_types::{
    Device, EngineConfig, ExecutorAdmissionLimits, FerrumError, FinishReason, InferenceRequest,
    ModelInfo, ObservabilityProfileDetail, PrefillDecodeExecution, RequestId, Result,
    SamplingReadbackMode, StartupMemoryPlan, TokenId,
};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::sync::oneshot;

const WATCHDOG: Duration = Duration::from_secs(10);

struct Gate {
    reached: oneshot::Sender<()>,
    release: oneshot::Receiver<()>,
}

/// Assertion failure/cancellation must also release the original invocation.
struct Release(Option<oneshot::Sender<()>>);
impl Drop for Release {
    fn drop(&mut self) {
        if let Some(sender) = self.0.take() {
            let _ = sender.send(());
        }
    }
}

#[derive(Default)]
struct Observation {
    gate: Mutex<Option<Gate>>,
    requests: Mutex<HashMap<ExecutorSequence, RequestId>>,
    submitted: Mutex<Vec<Vec<(RequestId, TokenWork)>>>,
}

impl Observation {
    fn arm(&self) -> (oneshot::Receiver<()>, Release) {
        let (reached, observed) = oneshot::channel();
        let (release, released) = oneshot::channel();
        assert!(self
            .gate
            .lock()
            .unwrap()
            .replace(Gate {
                reached,
                release: released
            })
            .is_none());
        (observed, Release(Some(release)))
    }
}

/// Delegates the original scope and its actual capacity/lifecycle contracts.
/// Only a completed singleton prefill is gated; learned mixed work is ungated.
struct GatedExecutor {
    inner: ExecutorScope,
    owner: ExecutorOwner,
    observation: Arc<Observation>,
}

#[async_trait]
impl ModelExecutor for GatedExecutor {
    fn info(&self) -> &ModelInfo {
        self.inner.info()
    }
    fn capabilities(&self) -> ExecutorCapabilities {
        self.inner.capabilities()
    }
    fn status(&self) -> ExecutorStatus {
        self.inner.status()
    }
    fn prefill_shape_policy(&self) -> PrefillShapePolicy {
        self.inner.prefill_shape_policy()
    }
    fn execution_resource_authority(&self) -> ExecutionResourceAuthority {
        self.inner.execution_resource_authority()
    }
    fn admission_limits(&self) -> Result<Option<ExecutorAdmissionLimits>> {
        self.inner.admission_limits()
    }
    fn resolved_model_plan(&self) -> Option<&ResolvedModelPlan> {
        self.inner.resolved_model_plan()
    }
    fn plan_runtime_resource_snapshot(&self) -> Result<Option<PlanRuntimeResourceSnapshot>> {
        self.inner.plan_runtime_resource_snapshot()
    }
    fn kv_capacity(&self) -> Option<usize> {
        self.inner.kv_capacity()
    }
    fn startup_memory_plan(&self) -> Option<&StartupMemoryPlan> {
        self.inner.startup_memory_plan()
    }
    async fn prepare_startup(&self) -> Result<()> {
        self.inner.prepare_startup().await
    }
    fn attach_execution_event_sink(&self, sink: Arc<dyn ProductExecutionEventSink>) {
        self.inner.attach_execution_event_sink(sink);
    }
    fn execution_capacity_epochs(&self) -> Result<Option<ExecutorAdmissionEpochs>> {
        self.inner.execution_capacity_epochs()
    }
    fn write_execution_capacity_snapshot(
        &self,
        availability: &mut Vec<CapacityAvailabilityEpoch>,
    ) -> Result<Option<ExecutorAdmissionEpochs>> {
        self.inner.write_execution_capacity_snapshot(availability)
    }
    fn register_execution_capacity_waiter(
        &self,
        observed: &CapacityWaitCondition,
    ) -> Result<Option<ExecutorCapacityWaitRegistration>> {
        self.inner.register_execution_capacity_waiter(observed)
    }
    fn try_admit_prefill(
        &self,
        input: ExecutorPrefillAdmission<'_>,
    ) -> Result<ExecutorPrefillAdmissionDecision> {
        self.inner.try_admit_prefill(input)
    }
    fn cancel_prefill_admission(&self, request: &RequestId) -> bool {
        self.inner.cancel_prefill_admission(request)
    }
    fn maintain_prefill_backing(
        &self,
        request: &RequestId,
    ) -> Result<ExecutorPrefillMaintenanceOutcome> {
        self.inner.maintain_prefill_backing(request)
    }
    fn write_execution_capacity_release_sources(
        &self,
        preemption: &ExecutorExecutionCapacityPreemption,
        sources: &mut Vec<CapacityAvailabilitySource>,
    ) -> Result<bool> {
        self.inner
            .write_execution_capacity_release_sources(preemption, sources)
    }
    async fn preempt_execution_capacity(
        &self,
        preemption: ExecutorExecutionCapacityPreemption,
    ) -> Result<ExecutorExecutionCapacityPreemptionReceipt> {
        self.inner.preempt_execution_capacity(preemption).await
    }
    fn write_token_capacity_release_sources(
        &self,
        sequence: &ExecutorSequence,
        sources: &mut Vec<CapacityAvailabilitySource>,
    ) -> Result<bool> {
        self.inner
            .write_token_capacity_release_sources(sequence, sources)
    }
    fn plan_prefix_capture_boundary(
        &self,
        input: PrefixCaptureBoundary<'_>,
    ) -> Option<PrefixCapturePlan> {
        self.inner.plan_prefix_capture_boundary(input)
    }
    fn retain_prefix_capture_interest(
        &self,
        input: PrefixCaptureRequest<'_>,
    ) -> Result<Option<Arc<dyn PrefixCaptureLease>>> {
        self.inner.retain_prefix_capture_interest(input)
    }
    fn supports_plan_runtime_prefix_restore(&self) -> bool {
        self.inner.supports_plan_runtime_prefix_restore()
    }
    async fn try_restore_plan_runtime_prefix(
        &self,
        input: PlanRuntimePrefixRestoreInput<'_>,
    ) -> Result<PlanRuntimePrefixRestoreOutcome> {
        self.inner.try_restore_plan_runtime_prefix(input).await
    }

    async fn admit_tokens(&self, input: ExecutionAdmission<'_>) -> Result<TokenAdmissionOutcome> {
        let request = input.context.request_id.clone();
        let admitted = self.inner.admit_tokens(input).await?;
        if let TokenAdmissionOutcome::Admitted(frontier) = &admitted {
            assert!(self
                .observation
                .requests
                .lock()
                .unwrap()
                .insert(frontier.sequence().clone(), request)
                .is_none());
        }
        Ok(admitted)
    }

    async fn execute_tokens(&self, work: &[ExecutionWork<'_>]) -> Result<ExecutionBatchOutcome> {
        let result = self.inner.execute_tokens(work).await?;
        let ExecutionBatchOutcome::Completed(batch) = &result else {
            panic!("real CPU fixture must finish its actual submitted work");
        };
        assert_eq!(batch.completions().len(), work.len());
        for (index, completion) in batch.completions().iter().enumerate() {
            completion.validate_for(index, &work[index], VOCAB).unwrap();
            let expected = match work[index].work() {
                TokenWork::Prefill(chunk) => {
                    assert_eq!(
                        (
                            chunk.tokens_processed(),
                            chunk.tokens_to_process(),
                            chunk.total_prompt_tokens()
                        ),
                        (0, 2, 2)
                    );
                    [3.0, 2.0, 1.0, -4.0]
                }
                TokenWork::Decode { token, position } => {
                    // Raw token 0 wins the model row but is the tokenizer's
                    // UNK. Product sampling masks UNK/BOS, so token 1 is the
                    // highest eligible result and supplies this decode row.
                    assert_eq!(token, TokenId::new(1));
                    assert_eq!(position, 2);
                    [3.0, 2.0, 1.0, -4.0]
                }
            };
            let ExecutionProduct::Sampling(ExecutorSamplingOutput::FullLogits(actual)) =
                completion.product()
            else {
                panic!("top-k=1 retains the real full-logits product path");
            };
            assert!(actual.iter().all(|value| value.is_finite()));
            assert_eq!(
                actual.as_slice(),
                expected,
                "embedding followed by identity head"
            );
            assert_eq!(
                completion.next_frontier().consumed_tokens(),
                work[index].work().range().unwrap().end()
            );
        }
        {
            let requests = self.observation.requests.lock().unwrap();
            let mut submitted = self.observation.submitted.lock().unwrap();
            assert!(submitted.len() < 16);
            submitted.push(
                work.iter()
                    .map(|input| (requests[input.sequence()].clone(), input.work()))
                    .collect(),
            );
        }
        let gate = if work.len() == 1 && matches!(work[0].work(), TokenWork::Prefill(_)) {
            self.observation.gate.lock().unwrap().take()
        } else {
            None
        };
        if let Some(gate) = gate {
            let _ = gate.reached.send(());
            let _ = gate.release.await;
        }
        Ok(result)
    }

    async fn finish_tokens(
        &self,
        sequence: &ExecutorSequence,
        reason: ExecutionEndReason,
    ) -> Result<ExecutionCloseOutcome> {
        self.inner.finish_tokens(sequence, reason).await
    }
    fn cancel_tokens(&self, sequence: &ExecutorSequence) -> Result<ExecutionCloseOutcome> {
        self.inner.cancel_tokens(sequence)
    }
    fn begin_shutdown(
        self: Arc<Self>,
        permit: &ExecutorShutdownPermit,
    ) -> ExecutorShutdownObserver {
        if let Err(error) = permit.check(self.as_ref()) {
            return ExecutorShutdownObserver::retained(Arc::new(self), error);
        }
        match Arc::try_unwrap(self) {
            Ok(Self {
                inner,
                owner,
                observation,
            }) => {
                drop(inner);
                drop(observation);
                owner.begin_shutdown()
            }
            Err(original) => ExecutorShutdownObserver::retained(
                Arc::new(original),
                FerrumError::internal("mixed fixture still has an execution borrower"),
            ),
        }
    }
}

struct Witness {
    worker: Arc<dyn ExecutionAttemptObserver>,
    summaries: Mutex<Vec<ExecutionAttemptSummary>>,
}

fn is_mixed(summary: &ExecutionAttemptSummary) -> bool {
    summary.participants.len() == 2
        && summary
            .participants
            .iter()
            .any(|member| member.kind == AttemptWorkKind::Prefill)
        && summary
            .participants
            .iter()
            .any(|member| member.kind == AttemptWorkKind::Decode)
}

impl ExecutionAttemptObserver for Witness {
    fn limits(&self) -> ExecutionAttemptLimits {
        self.worker.limits()
    }
    fn try_record(&self, summary: &ExecutionAttemptSummary) -> ExecutionAttemptDelivery {
        let mut summaries = self
            .summaries
            .try_lock()
            .expect("read witness only after inference");
        assert!(summaries.len() < 16);
        summaries.push(summary.clone());
        // Select genuine ungated mixed observations, leaving singleton domains
        // untrained so accidental per-member fitting is observable below.
        if is_mixed(summary) {
            let delivery = self.worker.try_record(summary);
            assert_eq!(delivery, ExecutionAttemptDelivery::Recorded);
            delivery
        } else {
            ExecutionAttemptDelivery::Recorded
        }
    }
    fn record_drop(&self, reason: ExecutionAttemptDrop) {
        panic!("unexpected attempt loss: {reason:?}");
    }
}

fn cost_work(summary: &ExecutionAttemptSummary) -> Vec<CostWork> {
    summary
        .participants
        .iter()
        .map(|member| CostWork {
            kind: member.kind,
            requested: member.requested,
            context_tokens: member.context_tokens,
            prefix_restore: member.prefix_restore,
            internally_reused_tokens: member.internally_reused_tokens,
        })
        .collect()
}

fn request(prompt: &str) -> InferenceRequest {
    let mut request = InferenceRequest::new(prompt, "native-mixed-fixture");
    request.sampling_params.max_tokens = 2;
    request.sampling_params.temperature = 0.0;
    request.sampling_params.top_k = Some(1);
    request
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reference_native_cpu_mixed_attempt_preserves_order_and_fits_whole_wall_once() {
    let (runtime, registry, materializers, _, catalog) = CpuVNextComposition::create(
        DeviceId::new("device.test.reference-mixed").unwrap(),
        256 << 20,
    )
    .unwrap()
    .into_parts()
    .unwrap();
    let fixture = Fixture::new(
        SamplingReadbackMode::Uniform,
        Device::CPU,
        runtime.clone(),
        registry,
        materializers,
        catalog,
        FixtureOptions::default(),
    );
    assert!(!fixture.executor.supports_plan_runtime_prefix_restore());
    assert_eq!(
        fixture.executor.prefill_shape_policy(),
        PrefillShapePolicy::Identity
    );
    let native = ExecutorOwner::new(Box::new(fixture.executor));
    let observation = Arc::new(Observation::default());
    let executor = ExecutorOwner::new(Box::new(GatedExecutor {
        inner: native.scope().unwrap(),
        owner: native,
        observation: observation.clone(),
    }));
    let estimator = ReferenceEstimator::spawn(ReferenceEstimatorConfig {
        min_samples: 2,
        safety_margin: 0.0,
        ..Default::default()
    })
    .unwrap();
    let witness = Arc::new(Witness {
        worker: estimator.observer(),
        summaries: Mutex::new(Vec::new()),
    });
    let mut config = EngineConfig::default();
    config.runtime.profile_detail = ObservabilityProfileDetail::Off;
    config.batching.prefill_decode_execution = PrefillDecodeExecution::Mixed;
    config.batching.max_batch_size = 3;
    config.batching.max_num_batched_tokens = 16;
    config.scheduler.max_running_requests = 3;
    config.scheduler.max_waiting_requests = 3;
    config.runtime.max_model_len = Some(16);
    config.runtime.sampling_readback_mode = SamplingReadbackMode::Uniform;
    config.scheduler.slo = None;
    let tokenizer = Arc::new(MockTokenizer::new(VOCAB));
    assert_eq!(tokenizer.special_tokens().unk_token, Some(TokenId::new(0)));
    assert_eq!(tokenizer.special_tokens().bos_token, Some(TokenId::new(2)));
    assert_eq!(tokenizer.special_tokens().eos_token, Some(TokenId::new(3)));
    let engine = EngineBuilder::new(config)
        .with_custom_tokenizer(tokenizer)
        .with_custom_sampler(Arc::new(ferrum_engine::registry::GreedySampler))
        .with_owned_executor(executor)
        .unwrap()
        .with_execution_attempt_observer(witness.clone())
        .build()
        .await
        .unwrap();

    let mut request_ids = Vec::new();
    for _ in 0..2 {
        let first = request("first");
        let second = request("second");
        request_ids.extend([first.id.clone(), second.id.clone()]);
        let (reached, release) = observation.arm();
        let mut first_run = Box::pin(engine.infer(first));
        tokio::time::timeout(WATCHDOG, async {
            tokio::select! {
                result = &mut first_run => panic!("request finished before its gate: {result:?}"),
                reached = reached => reached.unwrap(),
            }
        })
        .await
        .unwrap();
        // Poll the second request while the original driver is held after its
        // first execution. The actual mixed submission checked below proves
        // the overlap; Pending alone is not a publication receipt.
        let mut second_run = Box::pin(engine.infer(second));
        assert!(futures::poll!(second_run.as_mut()).is_pending());
        drop(release);
        let (first, second) =
            tokio::time::timeout(WATCHDOG, async { tokio::join!(first_run, second_run) })
                .await
                .unwrap();
        for output in [first.unwrap(), second.unwrap()] {
            assert_eq!(
                output.response().tokens,
                vec![TokenId::new(1), TokenId::new(1)]
            );
            assert_eq!(output.response().finish_reason, FinishReason::Length);
            assert_eq!(output.response().text, "w1 w1");
        }
    }

    // Await the original driver and owner before reading the pure witnesses;
    // a delivered output alone need not mean its final observer callback ran.
    tokio::time::timeout(WATCHDOG, async {
        engine.shutdown().await.unwrap();
        engine.wait_for_shutdown_completion().await.unwrap();
    })
    .await
    .unwrap();
    drop(engine);
    assert_eq!(
        runtime.resident_bytes(),
        0,
        "the original native owner really retired"
    );
    let summaries = witness.summaries.lock().unwrap().clone();
    let submitted = observation.submitted.lock().unwrap().clone();
    assert_eq!(summaries.len(), submitted.len());
    for (summary, inputs) in summaries.iter().zip(&submitted) {
        assert_eq!(summary.eligibility, ExecutionAttemptEligibility::Completed);
        assert_eq!(summary.participants.len(), inputs.len());
        for (member, (request, input)) in summary.participants.iter().zip(inputs) {
            assert_eq!(
                &member.request_id, request,
                "original submitted participant order"
            );
            assert_eq!(member.requested, input.range().unwrap());
            assert_eq!(member.context_tokens, member.requested.end());
            assert_eq!(member.prefix_restore, AttemptPrefixRestore::NoneObserved);
            assert_eq!(member.internally_reused_tokens, None);
            assert!(!member.product_failed);
            let AttemptParticipantOutcome::Completed {
                work,
                frontier_committed,
            } = member.outcome
            else {
                panic!("missing actual completion");
            };
            assert!(frontier_committed);
            assert_eq!(work.range().unwrap(), member.requested);
        }
    }
    for request in request_ids {
        let members: Vec<_> = summaries
            .iter()
            .flat_map(|summary| &summary.participants)
            .filter(|member| member.request_id == request)
            .collect();
        assert_eq!(members.len(), 2);
        assert_eq!(
            (
                members[0].kind,
                members[0].requested.start(),
                members[0].requested.end()
            ),
            (AttemptWorkKind::Prefill, 0, 2)
        );
        assert_eq!(
            (
                members[1].kind,
                members[1].requested.start(),
                members[1].requested.end()
            ),
            (AttemptWorkKind::Decode, 2, 3)
        );
    }
    let mixed: Vec<_> = summaries
        .iter()
        .filter(|summary| is_mixed(summary))
        .collect();
    assert_eq!(
        mixed.len(),
        2,
        "two real mixed invocations, not two participant samples"
    );
    let domain = cost_work(mixed[0]);
    assert_eq!(domain, cost_work(mixed[1]));
    let expected_wall = mixed
        .iter()
        .map(|summary| summary.executor_wall().unwrap())
        .max()
        .unwrap();
    assert!(!expected_wall.is_zero());
    let source = estimator.source();
    let mut revisions = source.subscribe();
    let snapshot = tokio::time::timeout(WATCHDOG, async {
        loop {
            let snapshot = source.load();
            if matches!(
                snapshot.predict(
                    &CostQuery {
                        work: &domain,
                        work_is_exact: true
                    },
                    Instant::now()
                ),
                CostPrediction::Known { samples: 2, .. }
            ) {
                break snapshot;
            }
            revisions.changed().await.unwrap();
        }
    })
    .await
    .unwrap();
    let CostPrediction::Known {
        upper_service_time,
        samples,
        ..
    } = snapshot.predict(
        &CostQuery {
            work: &domain,
            work_is_exact: true,
        },
        Instant::now(),
    )
    else {
        panic!("actual mixed domain was not learned");
    };
    assert_eq!(samples, 2);
    assert_eq!(
        upper_service_time, expected_wall,
        "fit the whole invocation's wall once"
    );
    let reversed = [domain[1], domain[0]];
    for work in [&reversed[..], &domain[..1], &domain[1..]] {
        assert_eq!(
            snapshot.predict(
                &CostQuery {
                    work,
                    work_is_exact: true
                },
                Instant::now()
            ),
            CostPrediction::Unknown(CostUnknownReason::NoCoverage)
        );
    }
    let exited = tokio::time::timeout(WATCHDOG, estimator.shutdown())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(exited.stats.accepted, 2);
    assert_eq!(exited.stats.trained_samples, 2);
    assert_eq!(exited.stats.queued_records, 0);
    assert_eq!(exited.stats.queued_bytes, 0);
}
