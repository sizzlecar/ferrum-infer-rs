use super::*;
use crate::vnext::qwen35::define_from_model_dir;

type Runtime = MetalDeviceRuntime;
pub(super) struct Fixture {
    pub executor: VNextModelExecutor<Runtime>,
    _directory: tempfile::TempDir,
}
impl Fixture {
    pub async fn new(maximum_batch_tokens: usize, prefix: bool) -> Self {
        let directory = tempfile::tempdir().unwrap();
        weights::write_config(directory.path());
        weights::write_weights(directory.path());
        std::fs::write(directory.path().join("tokenizer.json"), br#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<unk>":0,"hello":1,"<eos>":2},"unk_token":"<unk>"}}"#).unwrap();
        std::fs::write(directory.path().join("tokenizer_config.json"), br#"{"chat_template":"{% for message in messages %}{{ message['content'] }}{% endfor %}","eos_token_id":2,"unk_token":"<unk>"}"#).unwrap();
        let defined = define_from_model_dir(directory.path()).unwrap();
        let mut engine = EngineConfig::default();
        engine.backend.device = Device::Metal;
        engine.backend.enable_reusable_execution = false;
        engine.runtime.prefix_state_cache_enabled = prefix;
        // Capacity preparation owns a separate cohort alongside the at-most
        // three product participants. It must not borrow their sequence frames.
        engine.scheduler.max_running_requests = 8;
        engine.scheduler.prefill_step_chunk = Some(maximum_batch_tokens);
        engine.batching.max_num_batched_tokens = maximum_batch_tokens;
        engine.memory.usable_capacity_bytes = Some(64 << 20);
        let profiles = defined
            .definition()
            .numerical_profiles()
            .candidates(
                &engine.numerical_execution,
                ferrum_types::KvStorageFormat::F16,
            )
            .unwrap();
        let prepared = defined.prepare(&profiles[0].id).unwrap();
        let (runtime, operations, materializers, materializer, catalog) =
            MetalVNextComposition::create(
                DeviceId::new(format!("device.guarded-product.{}", uuid::Uuid::new_v4())).unwrap(),
            )
            .unwrap()
            .into_parts();
        let info = prepared.model_info(ModelId::new("guarded-product"), Device::Metal);
        let config =
            VNextExecutorConfig::from_engine_config(&engine, &info, runtime.as_ref()).unwrap();
        let composition = VNextRuntimeComposition::new(runtime, operations, materializers, catalog);
        let compiled = composition
            .compile_model(
                &prepared,
                info,
                &engine,
                config,
                WeightMaterializerSelection::exact(materializer),
            )
            .unwrap();
        let executor = compiled
            .initialize(|prepared, runtime, catalog, compilation| {
                let numerical = NumericalProfileResolution::from_static_plan(
                    engine.numerical_execution.clone(),
                    ferrum_types::KvStorageFormat::F16,
                    defined.definition(),
                    prepared.family(),
                    catalog,
                    runtime,
                    compilation.executable().execution_plan(),
                    Vec::new(),
                )
                .unwrap();
                resolver::resolve(prepared, runtime, catalog, compilation, numerical)
            })
            .unwrap();
        executor.prepare_startup().await.unwrap();
        Self {
            executor,
            _directory: directory,
        }
    }
    pub fn submissions(&self) -> u64 {
        self.executor
            .metrics
            .submitted_waves
            .load(Ordering::Relaxed)
    }
    pub fn admit(&self, input: &PlanRuntimePrefillInput) {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            assert!(Instant::now() < deadline, "tiny admission made no progress");
            let admission = ExecutorPrefillAdmission::for_product_request(
                &input.request_id,
                &input.input_tokens,
                input.maximum_sequence_tokens,
                input.input_tokens.len(),
                0,
            )
            .unwrap();
            match self.executor.try_admit_prefill(admission).unwrap() {
                ExecutorPrefillAdmissionDecision::Admitted(_) => break,
                ExecutorPrefillAdmissionDecision::MaintenanceDeferred(_) => {
                    assert!(matches!(
                        self.executor
                            .maintain_prefill_backing(&input.request_id)
                            .unwrap(),
                        ExecutorPrefillMaintenanceOutcome::Maintained { .. }
                            | ExecutorPrefillMaintenanceOutcome::RetryAdmission { .. }
                    ));
                }
                other => panic!("fixture admission: {other:?}"),
            }
        }
    }
    pub async fn prefill(&self, input: &PlanRuntimePrefillInput) -> PlanRuntimePrefillCompletion {
        match self
            .executor
            .plan_runtime_prefill_with_capacity(input)
            .await
            .unwrap()
        {
            PlanRuntimePrefillOutcome::Completed(output) => output,
            PlanRuntimePrefillOutcome::Deferred(reason) => panic!("ordinary prefill: {reason:?}"),
        }
    }
    pub async fn seed_decode(&self, tokens: &[u32]) -> PlanRuntimeDecodeInput {
        let input = prompt(tokens, tokens.len());
        self.admit(&input);
        let output = self.prefill(&input).await;
        PlanRuntimeDecodeInput::new(
            input.request_id,
            TokenId::new(1),
            Arc::clone(output.output().kv_cache()),
        )
    }
    pub fn assert_ready(&self, input: &PlanRuntimePrefillInput, processed: usize) {
        let registry = self.executor.sequences.lock();
        let slot = registry
            .prefills
            .get(&input.request_id)
            .expect("prefill lost retained slot");
        let state = slot.state.lock();
        let VNextPrefillSlotState::Ready(sequence) = &*state else {
            panic!("prefill did not restore Ready")
        };
        assert!(sequence.active.load(Ordering::Acquire));
        assert_eq!(
            sequence.prefill_tokens_processed.load(Ordering::Acquire),
            processed
        );
    }
    pub(super) fn rows(
        &self,
        prefills: &[PlanRuntimePrefillInput],
        decodes: &[PlanRuntimeDecodeInput],
    ) -> Vec<(Arc<VNextSequence<Runtime>>, Vec<u32>, Range<usize>)> {
        let registry = self.executor.sequences.lock();
        let mut rows = prefills
            .iter()
            .map(|input| {
                let slot = registry.prefills.get(&input.request_id).unwrap();
                let state = slot.state.lock();
                let VNextPrefillSlotState::Ready(sequence) = &*state else {
                    panic!("fixture is not ready")
                };
                (
                    Arc::clone(sequence),
                    input.input_tokens.iter().map(|token| token.get()).collect(),
                    input.chunk.range(),
                )
            })
            .collect::<Vec<_>>();
        for input in decodes {
            let sequence = Arc::clone(registry.active.get(&input.kv_cache.cache_id()).unwrap());
            let mut tokens = sequence.tokens.lock().clone();
            let offset = tokens.len();
            tokens.push(input.input_token.get());
            rows.push((sequence, tokens, offset..offset + 1));
        }
        rows.sort_by_key(|(sequence, _, _)| sequence.session.sequence_authority());
        rows
    }
    /// Explicit fixture-only capacity maintenance on an independent cohort.
    /// Prepared-wave Drop retains a ledger tombstone, so these auxiliary owners
    /// are aborted, never reopened with a fabricated NotSubmitted receipt.
    /// The actual target owners do not acquire a frame or initialization hold.
    pub fn warm(&self, prefills: &[PlanRuntimePrefillInput], decodes: &[PlanRuntimeDecodeInput]) {
        let targets = self.rows(prefills, decodes);
        let before = self.target_resource_evidence(prefills, decodes);
        let kind = kind(prefills, decodes);
        let auxiliary: Vec<_> = targets
            .iter()
            .map(|(sequence, tokens, range)| {
                PlanRuntimePrefillInput::new(
                    RequestId::new(),
                    tokens.iter().copied().map(TokenId::new).collect::<Vec<_>>(),
                    sequence.maximum_tokens,
                    PrefillChunk::new(range.start, range.len(), tokens.len()).unwrap(),
                )
                .unwrap()
            })
            .collect();
        for input in &auxiliary {
            self.admit(input);
        }
        // These are genuine admitted sequence authorities, used only for
        // resource preparation. The selected phase and every span remain the
        // target wave's; no auxiliary model token is executed or committed.
        let rows = self.rows(&auxiliary, &[]);
        let sequences = rows
            .iter()
            .map(|(sequence, _, _)| Arc::clone(sequence))
            .collect::<Vec<_>>();
        let batch = ExecutionBatchParticipants::new(
            sequences
                .iter()
                .map(|sequence| Arc::clone(&sequence.session))
                .collect(),
        )
        .unwrap();
        let spans = rows
            .iter()
            .map(|(_, tokens, range)| TokenSpanWork::from_token_ids(tokens, range.clone()).unwrap())
            .collect::<Vec<_>>();
        for (sequence, tokens, range) in &rows {
            let work = ResourceWorkShape::single(
                TokenSpanWork::from_token_ids(&tokens[..range.end], 0..range.end).unwrap(),
            )
            .unwrap();
            assert!(matches!(
                self.executor
                    .extend_sequence_with_capacity(sequence, work)
                    .unwrap(),
                VNextExecutionCapacityDecision::Ready(())
            ));
        }
        let step = match self
            .executor
            .begin_step_for_spans_with_capacity(&batch, &sequences, &spans, kind)
            .unwrap()
        {
            VNextExecutionCapacityDecision::Ready(step) => step,
            _ => panic!("warm Step deferred"),
        };
        let wave = match self
            .executor
            .prepare_wave_for_spans_with_capacity(&step, &sequences, &spans, kind)
            .unwrap()
        {
            VNextExecutionCapacityDecision::Ready(wave) => wave,
            _ => panic!("warm wave deferred"),
        };
        drop(wave);
        step.try_abort().unwrap();
        drop(batch);
        drop(sequences);
        drop(rows);
        for input in &auxiliary {
            assert!(self.executor.cancel_prefill_admission(&input.request_id));
        }
        assert_eq!(
            self.target_resource_evidence(prefills, decodes),
            before,
            "capacity preparation must preserve target backing/frame/initialization evidence"
        );
        for (sequence, tokens, range) in targets {
            assert!(sequence.active.load(Ordering::Acquire));
            if decodes
                .iter()
                .any(|input| input.request_id == *sequence.request_id())
            {
                assert_eq!(*sequence.tokens.lock(), tokens[..range.start]);
            } else {
                assert_eq!(
                    sequence.prefill_tokens_processed.load(Ordering::Acquire),
                    range.start
                );
            }
        }
    }

    fn target_resource_evidence(
        &self,
        prefills: &[PlanRuntimePrefillInput],
        decodes: &[PlanRuntimeDecodeInput],
    ) -> Vec<ResourcePlanningParticipant> {
        let rows = self.rows(prefills, decodes);
        let requests: Vec<_> = rows
            .iter()
            .map(|(sequence, _, _)| ExecutorResourcePlanningRequest {
                request_id: sequence.request_id(),
                cache_id: decodes
                    .iter()
                    .any(|input| input.request_id == *sequence.request_id())
                    .then_some(sequence.cache_id.as_str()),
            })
            .collect();
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            match self.executor.execution_resource_planning_view(
                &requests,
                ResourcePlanningLimits::default(),
                &mut || true,
            ) {
                ResourcePlanningAvailability::Known(view) => return view.participants().to_vec(),
                ResourcePlanningAvailability::Unknown(
                    ResourcePlanningUnknown::ReadUnavailable(_),
                ) if Instant::now() < deadline => std::thread::yield_now(),
                other => panic!("target resource evidence: {other:?}"),
            }
        }
    }
    pub fn expected(
        &self,
        prefills: &[PlanRuntimePrefillInput],
        decodes: &[PlanRuntimeDecodeInput],
    ) -> ExpectedExecutionCostWave {
        self.try_expected(prefills, decodes)
            .expect("actual product future route must be covered")
    }

    pub fn try_expected(
        &self,
        prefills: &[PlanRuntimePrefillInput],
        decodes: &[PlanRuntimeDecodeInput],
    ) -> std::result::Result<ExpectedExecutionCostWave, ExecutionCostRouteUnknown> {
        let rows = self.rows(prefills, decodes);
        let requests = rows
            .iter()
            .map(|(sequence, _, _)| ExecutorResourcePlanningRequest {
                request_id: sequence.request_id(),
                cache_id: decodes
                    .iter()
                    .any(|input| &input.request_id == sequence.request_id())
                    .then_some(sequence.cache_id.as_str()),
            })
            .collect::<Vec<_>>();
        let deadline = Instant::now() + Duration::from_secs(5);
        let view = loop {
            match self.executor.execution_cost_route_view(
                &requests,
                ResourcePlanningLimits::default(),
                &mut || true,
            ) {
                ExecutionCostRouteAvailability::Known(view) => break view,
                ExecutionCostRouteAvailability::Unknown(ExecutionCostRouteUnknown::Resource(
                    ResourcePlanningUnknown::ReadUnavailable(_),
                )) if Instant::now() < deadline => std::thread::yield_now(),
                ExecutionCostRouteAvailability::Unknown(reason) => return Err(reason),
            }
        };
        let mut participants = Vec::new();
        let query_rows = rows
            .iter()
            .enumerate()
            .map(|(index, (sequence, _, range))| {
                let host = CostObservationParticipant {
                    request_id: sequence.request_id().clone(),
                    owner_incarnation: 1,
                    work_generation: 1,
                    input_index: index as u32,
                    output_policy_signature: Some([3; 32]),
                    host_features: None,
                };
                let (input, work, output) = if let Some(input) = prefills
                    .iter()
                    .find(|input| &input.request_id == sequence.request_id())
                {
                    (
                        ExpectedWaveInput::Prefill { chunk: input.chunk },
                        ActualRowWork::Prefill {
                            offset: range.start as u32,
                            count: range.len() as u32,
                            total_prompt_tokens: input.input_tokens.len() as u32,
                        },
                        FutureCostOutput::Prefill {
                            final_logits: input.chunk.is_final(),
                        },
                    )
                } else {
                    let input = decodes
                        .iter()
                        .find(|input| &input.request_id == sequence.request_id())
                        .unwrap();
                    (
                        ExpectedWaveInput::Decode {
                            cache_id: input.kv_cache.cache_id(),
                        },
                        ActualRowWork::Decode {
                            kv_tokens: range.start as u32,
                        },
                        FutureCostOutput::Decode {
                            policy: &input.logits_policy,
                        },
                    )
                };
                participants.push(ExpectedWaveParticipant {
                    participant_index: index,
                    request_id: sequence.request_id().clone(),
                    input,
                    host,
                });
                FutureWaveCostRow {
                    participant_index: index,
                    work,
                    host_policy_signature: [3; 32],
                    host_features: None,
                    output,
                }
            })
            .collect::<Vec<_>>();
        let query = FutureWaveCostQuery {
            kind: match kind(prefills, decodes) {
                VNextExecutionWaveKind::Prefill => ActualWaveKind::Prefill,
                VNextExecutionWaveKind::Mixed => ActualWaveKind::Mixed,
                VNextExecutionWaveKind::Decode => ActualWaveKind::Decode,
            },
            rows: &query_rows,
        };
        let projection = match self.executor.project_execution_cost_wave(
            &view,
            &view.initial_state(),
            &query,
            &mut || true,
        ) {
            ExecutionCostRouteAvailability::Known(projection) => projection,
            ExecutionCostRouteAvailability::Unknown(reason) => return Err(reason),
        };
        ExpectedExecutionCostWave::new(view, projection.shape, participants)
            .map_err(|_| ExecutionCostRouteUnknown::InvalidInput)
    }
}

fn kind(
    prefills: &[PlanRuntimePrefillInput],
    decodes: &[PlanRuntimeDecodeInput],
) -> VNextExecutionWaveKind {
    if decodes.is_empty() {
        VNextExecutionWaveKind::Prefill
    } else if prefills.is_empty() {
        VNextExecutionWaveKind::Decode
    } else {
        VNextExecutionWaveKind::Mixed
    }
}
pub(super) fn prompt(tokens: &[u32], chunk: usize) -> PlanRuntimePrefillInput {
    PlanRuntimePrefillInput::new(
        RequestId::new(),
        tokens.iter().copied().map(TokenId::new).collect::<Vec<_>>(),
        16,
        PrefillChunk::new(0, chunk, tokens.len()).unwrap(),
    )
    .unwrap()
}
pub(super) fn assert_logits_same(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    assert!(
        actual
            .iter()
            .zip(expected)
            .all(|(a, b)| a.is_finite() && b.is_finite() && (a - b).abs() <= 1e-5),
        "actual {actual:?}, expected {expected:?}"
    );
}
pub(super) fn logits(output: &ExecutorSamplingOutput) -> &[f32] {
    match output {
        ExecutorSamplingOutput::FullLogits(logits) => logits,
        _ => panic!("expected actual full logits"),
    }
}
pub(super) fn assert_prefill_same(
    actual: &PlanRuntimePrefillCompletion,
    expected: &PlanRuntimePrefillCompletion,
) {
    assert_eq!(actual.completed_chunk(), expected.completed_chunk());
    assert_eq!(
        actual.output().committed_tokens(),
        expected.output().committed_tokens()
    );
    match (actual.output().product(), expected.output().product()) {
        (PlanRuntimePrefillProduct::Intermediate, PlanRuntimePrefillProduct::Intermediate) => {}
        (PlanRuntimePrefillProduct::FinalLogits(a), PlanRuntimePrefillProduct::FinalLogits(b)) => {
            assert_logits_same(a, b)
        }
        _ => panic!("prefill product differs"),
    }
}
