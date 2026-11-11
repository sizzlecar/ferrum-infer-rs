//! Real stored CPU Qwen3.5 visual-profile execution through generic causal
//! inputs. These tests qualify transport and position semantics, not CUDA or
//! the complete H3 image pipeline.

use super::*;
use ferrum_execution::{
    executor_lifecycle::{ExecutorOwner, ExecutorScope},
    model_executor::{
        ExecutorPrefillAdmission, ExecutorSamplingOutput, LogitsReturnPolicy, PrefillChunk,
    },
    token_execution::{
        ExecutionAdmission, ExecutionBatchOutcome, ExecutionCloseDisposition, ExecutionCompletion,
        ExecutionFrontier, ExecutionProduct, ExecutionWork, TokenAdmissionOutcome, TokenWork,
    },
    CausalInputSchema, CausalInputSegment, CausalRequestInputs, CausalTokenInput,
    CausalTokenInputSpec, CausalTokenInputStorage, ManagedCausalIoProfile,
};
use ferrum_kernels::backend::cpu::{
    vnext_ops::CpuVNextComposition, vnext_runtime::CpuDeviceRuntime,
};
use ferrum_models::vnext::qwen35::{
    F16_VISUAL_PROMPT_NUMERICAL_PROFILE_ID, ROPE_COS_INPUT_ID, ROPE_SIN_INPUT_ID,
    VISUAL_REPLACE_MASK_INPUT_ID, VISUAL_ROWS_INPUT_ID,
};
use ferrum_runtime::{
    ElementType, HostMemoryAccount, HostMemoryClass, HostMemoryClassLimits, HostMemoryLimits,
    ManagedHostBytes, ProgramValueId,
};
use ferrum_types::{RequestId, TokenId};

const MAXIMUM_TOKENS: usize = 16;

fn fixture() -> (
    std::path::PathBuf,
    DefinedProductionModel,
    PreparedProductionModel,
    EngineConfig,
) {
    prepared_execution_fixture_with(
        F16_VISUAL_PROMPT_NUMERICAL_PROFILE_ID.parse().unwrap(),
        |directory| {
            // Reuse the actual mixed GDN/full-attention stored fixture. Its GDN
            // state and both MLPs still execute, but zero output projections
            // isolate the attention transport test from those other equations.
            // These changes precede source freezing and mmap preparation.
            let replacements: [(&str, &[f32]); 6] = [
                ("model.layers.0.linear_attn.out_proj.weight", &[0.0, 0.0]),
                ("model.layers.0.mlp.down_proj.weight", &[0.0; 4]),
                ("model.layers.1.mlp.down_proj.weight", &[0.0; 4]),
                (
                    "model.layers.1.self_attn.v_proj.weight",
                    &[4.0, -1.0, -4.0, 1.0],
                ),
                (
                    "model.layers.1.self_attn.o_proj.weight",
                    &[16.0, 0.0, 0.0, 16.0],
                ),
                ("model.lm_head.weight", &[1.0, -1.0, -1.0, 1.0, -1.0, -1.0]),
            ];
            let path = directory.join("model.safetensors");
            let bytes = std::fs::read(&path).unwrap();
            let archive = safetensors::SafeTensors::deserialize(&bytes).unwrap();
            let replacement_bytes = replacements
                .iter()
                .map(|(_, values)| {
                    values
                        .iter()
                        .flat_map(|v| v.to_le_bytes())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let mut tensors = archive.tensors();
            for ((name, _), bytes) in replacements.iter().zip(&replacement_bytes) {
                let (_, tensor) = tensors.iter_mut().find(|(key, _)| key == name).unwrap();
                *tensor = safetensors::tensor::TensorView::new(
                    safetensors::Dtype::F32,
                    tensor.shape().to_vec(),
                    bytes,
                )
                .unwrap();
            }
            safetensors::tensor::serialize_to_file(tensors, &None, &path).unwrap();
        },
    )
}

fn id(value: &str) -> ProgramValueId {
    ProgramValueId::new(value).unwrap()
}

fn schema() -> CausalInputSchema {
    CausalInputSchema::new(vec![
        CausalTokenInputSpec {
            value_id: id(VISUAL_ROWS_INPUT_ID),
            element_type: ElementType::F16,
            row_dimensions: vec![2],
            storage: CausalTokenInputStorage::ZeroOutsideSegments,
        },
        CausalTokenInputSpec {
            value_id: id(VISUAL_REPLACE_MASK_INPUT_ID),
            element_type: ElementType::U8,
            row_dimensions: vec![],
            storage: CausalTokenInputStorage::ZeroOutsideSegments,
        },
        CausalTokenInputSpec {
            value_id: id(ROPE_COS_INPUT_ID),
            element_type: ElementType::F16,
            row_dimensions: vec![1],
            storage: CausalTokenInputStorage::Exact,
        },
        CausalTokenInputSpec {
            value_id: id(ROPE_SIN_INPUT_ID),
            element_type: ElementType::F16,
            row_dimensions: vec![1],
            storage: CausalTokenInputStorage::Exact,
        },
    ])
    .unwrap()
}

struct ActualExecutor {
    parent: HostMemoryAccount,
    host: HostMemoryAccount,
    runtime: Arc<CpuDeviceRuntime>,
    owner: ExecutorOwner,
    executor: ExecutorScope,
}

impl ActualExecutor {
    fn new(
        defined: &DefinedProductionModel,
        prepared: &PreparedProductionModel,
        engine: &EngineConfig,
    ) -> Self {
        let parent = HostMemoryAccount::with_allocation_alignment(
            HostMemoryLimits::new(
                1 << 20,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 1 << 18,
                    transfer_staging_bytes: 1 << 18,
                    output_bytes: 1 << 18,
                    weight_backing_bytes: 1 << 18,
                },
            )
            .unwrap(),
            16,
        )
        .unwrap();
        let phase = parent
            .reserve_scope(&[
                (HostMemoryClass::InputPreprocess, 1 << 18),
                (HostMemoryClass::TransferStaging, 1 << 18),
                (HostMemoryClass::Output, 1 << 18),
                (HostMemoryClass::WeightBacking, 1 << 18),
            ])
            .unwrap();
        let host = phase.account();
        let (runtime, operations, materializers, materializer, catalog) =
            CpuVNextComposition::create(
                ferrum_runtime::DeviceId::new("device.cpu.visual-prompt-consumer").unwrap(),
                16 << 20,
            )
            .unwrap()
            .into_parts()
            .unwrap();
        let composition =
            RuntimeComposition::new(Arc::clone(&runtime), operations, materializers, catalog);
        let info = prepared.model_info(engine.model.model_id.clone(), ferrum_types::Device::CPU);
        let config =
            ProgramExecutorConfig::from_engine_config(engine, &info, runtime.as_ref()).unwrap();
        let compiled = composition
            .compile_model_with_causal_inputs(
                prepared.execution_input().unwrap(),
                info,
                engine,
                config,
                WeightMaterializerSelection::exact(materializer),
                &schema(),
            )
            .unwrap();
        let owner = compiled
            .initialize_with_managed_io_retained(
                &host,
                ManagedCausalIoProfile::ManagedGreedyF16V1,
                phase,
                |policy, catalog, compilation| {
                    let numerical = NumericalProfileResolution::from_static_plan(
                        engine.numerical_execution.clone(),
                        ferrum_types::KvStorageFormat::F16,
                        defined.definition(),
                        prepared.family(),
                        catalog,
                        policy,
                        compilation.executable().execution_plan(),
                        vec![],
                    )
                    .map_err(|error| FerrumError::model(error.to_string()))?;
                    resolve_model_plan(engine, prepared, catalog, policy, compilation, &numerical)
                },
            )
            .unwrap();
        let executor = owner.scope().unwrap();
        Self {
            parent,
            host,
            runtime,
            owner,
            executor,
        }
    }

    async fn close(self) {
        let Self {
            parent,
            host,
            runtime,
            owner,
            executor,
        } = self;
        drop(executor);
        owner.begin_shutdown().wait().await.result().unwrap();
        assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
        assert_eq!(parent.usage().unwrap().committed_bytes(), 0);
        assert_eq!(runtime.resident_bytes(), 0);
        assert!(
            runtime.peak_resident_bytes() > 0,
            "the actual CPU model must execute"
        );
    }
}

fn half_bytes(host: &HostMemoryAccount, values: &[f32]) -> ManagedHostBytes {
    let mut bytes = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, values.len() * 2)
        .unwrap();
    for (destination, &value) in bytes.as_mut().chunks_exact_mut(2).zip(values) {
        destination.copy_from_slice(&half::f16::from_f32(value).to_le_bytes());
    }
    bytes.freeze()
}

fn request_inputs(
    host: &HostMemoryAccount,
    replacements: &[(usize, [f32; 2], bool)],
    rotations: &[(usize, f32, f32)],
) -> Arc<CausalRequestInputs> {
    let mut visual = Vec::new();
    let mut mask = Vec::new();
    for &(position, row, replace) in replacements {
        visual.push(CausalInputSegment::new(position, 1, half_bytes(host, &row)).unwrap());
        let mut flag = host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, 1)
            .unwrap();
        flag.as_mut()[0] = u8::from(replace);
        mask.push(CausalInputSegment::new(position, 1, flag.freeze()).unwrap());
    }
    let mut cos = [1.0; MAXIMUM_TOKENS];
    let mut sin = [0.0; MAXIMUM_TOKENS];
    for &(position, cosine, sine) in rotations {
        cos[position] = cosine;
        sin[position] = sine;
    }
    Arc::new(
        CausalRequestInputs::new(
            host,
            &schema(),
            MAXIMUM_TOKENS,
            vec![
                CausalTokenInput::new(id(VISUAL_ROWS_INPUT_ID), visual),
                CausalTokenInput::new(id(VISUAL_REPLACE_MASK_INPUT_ID), mask),
                CausalTokenInput::new(
                    id(ROPE_COS_INPUT_ID),
                    vec![
                        CausalInputSegment::new(0, MAXIMUM_TOKENS, half_bytes(host, &cos)).unwrap(),
                    ],
                ),
                CausalTokenInput::new(
                    id(ROPE_SIN_INPUT_ID),
                    vec![
                        CausalInputSegment::new(0, MAXIMUM_TOKENS, half_bytes(host, &sin)).unwrap(),
                    ],
                ),
            ],
        )
        .unwrap(),
    )
}

fn greedy() -> LogitsReturnPolicy {
    LogitsReturnPolicy::GreedyArgmax {
        token_mask: None,
        repetition_penalty: None,
    }
}

async fn admit(
    executor: &ExecutorScope,
    tokens: &[TokenId],
    inputs: Arc<CausalRequestInputs>,
) -> ExecutionFrontier {
    let request = RequestId::new();
    let admission = ExecutionAdmission {
        context: ExecutorPrefillAdmission::for_diagnostic(&request, tokens, MAXIMUM_TOKENS),
        adapter: None,
    };
    loop {
        match executor
            .admit_tokens_with_inputs(admission, Arc::clone(&inputs))
            .await
            .unwrap()
        {
            TokenAdmissionOutcome::Admitted(frontier) => return frontier,
            TokenAdmissionOutcome::MaintenanceDeferred(_) => {
                executor.maintain_prefill_backing(&request).unwrap();
            }
            other => panic!("fresh visual-profile admission failed: {other:?}"),
        }
    }
}

async fn execute(
    executor: &ExecutorScope,
    work: &[(&ExecutionFrontier, TokenWork)],
) -> Vec<ExecutionCompletion> {
    let policy = greedy();
    let inputs = work
        .iter()
        .map(|(frontier, work)| ExecutionWork::new(frontier, *work, &policy).unwrap())
        .collect::<Vec<_>>();
    let ExecutionBatchOutcome::Completed(batch) = executor.execute_tokens(&inputs).await.unwrap()
    else {
        panic!("admitted actual CPU visual wave must complete");
    };
    assert_eq!(batch.completions().len(), inputs.len());
    for (index, completion) in batch.completions().iter().enumerate() {
        completion
            .validate_for(index, &inputs[index], executor.info().vocab_size)
            .unwrap();
    }
    batch.into_completions()
}

fn selected(completion: &ExecutionCompletion) -> TokenId {
    match completion.product() {
        ExecutionProduct::Sampling(ExecutorSamplingOutput::GreedyToken(token)) => *token,
        other => panic!("expected actual F16 managed greedy output, got {other:?}"),
    }
}

fn stored_values(prepared: &PreparedProductionModel, name: &str) -> Vec<f32> {
    let component = prepared
        .family()
        .stored_weight_schema()
        .unwrap()
        .components
        .iter()
        .find(|c| c.external_names == [name])
        .unwrap();
    let payload = prepared.weights().component(component).unwrap();
    assert_eq!(
        payload.bytes().len(),
        component.dimensions.iter().product::<u64>() as usize * 2
    );
    payload
        .bytes()
        .chunks_exact(2)
        .map(|x| half::f16::from_le_bytes(x.try_into().unwrap()).to_f32())
        .collect()
}

#[derive(Debug)]
struct ScalarTrace {
    normalized: [f32; 2],
    query: [f32; 2],
    scores: Vec<f64>,
    context: [f32; 2],
    hidden: [f32; 2],
    logits: [f32; 3],
    token: TokenId,
}

// An independent scalar evaluation of this deliberately isolated fixture.
// It reads the actual prepared payloads (including DenseAffine norm +1), uses
// f64 softmax instead of the backend's online f32 recurrence, and rounds only
// the declared F16 projection/activation boundaries. Zero residual projections
// are checked rather than assuming an unexecuted GDN/MLP equation is identity.
fn scalar_trace(
    prepared: &PreparedProductionModel,
    rows: &[[f32; 2]],
    rotations: &[(usize, f32, f32)],
) -> ScalarTrace {
    for name in [
        "model.layers.0.linear_attn.out_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.1.mlp.down_proj.weight",
    ] {
        assert!(stored_values(prepared, name).iter().all(|&v| v == 0.0));
    }
    let norm = stored_values(prepared, "model.layers.1.input_layernorm.weight");
    let q = stored_values(prepared, "model.layers.1.self_attn.q_proj.weight");
    let k = stored_values(prepared, "model.layers.1.self_attn.k_proj.weight");
    let v = stored_values(prepared, "model.layers.1.self_attn.v_proj.weight");
    let o = stored_values(prepared, "model.layers.1.self_attn.o_proj.weight");
    let qn = stored_values(prepared, "model.layers.1.self_attn.q_norm.weight");
    let kn = stored_values(prepared, "model.layers.1.self_attn.k_norm.weight");
    let final_norm = stored_values(prepared, "model.norm.weight");
    let head = stored_values(prepared, "model.lm_head.weight");
    let rounded = |x: f32| half::f16::from_f32(x).to_f32();
    let rms = |x: [f32; 2], gamma: &[f32]| {
        let inv = ((x[0] * x[0] + x[1] * x[1]) * 0.5 + 1e-6).sqrt().recip();
        [
            rounded(x[0] * inv * gamma[0]),
            rounded(x[1] * inv * gamma[1]),
        ]
    };
    let linear = |w: &[f32], x: [f32; 2]| {
        [
            rounded(w[0] * x[0] + w[1] * x[1]),
            rounded(w[2] * x[0] + w[3] * x[1]),
        ]
    };
    let rotary = |x: [f32; 2], gamma: &[f32], index: usize| {
        let inv = ((x[0] * x[0] + x[1] * x[1]) * 0.5 + 1e-6).sqrt().recip();
        let x = [x[0] * inv * gamma[0], x[1] * inv * gamma[1]];
        let (_, c, s) = rotations
            .iter()
            .copied()
            .find(|r| r.0 == index)
            .unwrap_or((index, 1.0, 0.0));
        [rounded(x[0] * c - x[1] * s), rounded(x[0] * s + x[1] * c)]
    };
    let normalized = rows
        .iter()
        .copied()
        .map(|x| rms(x, &norm))
        .collect::<Vec<_>>();
    let query = rotary(linear(&q, *normalized.last().unwrap()), &qn, rows.len() - 1);
    let keys = normalized
        .iter()
        .copied()
        .enumerate()
        .map(|(i, x)| rotary(linear(&k, x), &kn, i))
        .collect::<Vec<_>>();
    let values = normalized
        .iter()
        .copied()
        .map(|x| linear(&v, x))
        .collect::<Vec<_>>();
    let scores = keys
        .iter()
        .map(|k| (query[0] as f64 * k[0] as f64 + query[1] as f64 * k[1] as f64) / 2f64.sqrt())
        .collect::<Vec<_>>();
    let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let probabilities = scores
        .iter()
        .map(|s| (s - maximum).exp())
        .collect::<Vec<_>>();
    let denominator = probabilities.iter().sum::<f64>();
    let context = std::array::from_fn(|j| {
        rounded(
            (probabilities
                .iter()
                .zip(&values)
                .map(|(p, v)| p * v[j] as f64)
                .sum::<f64>()
                / denominator) as f32,
        )
    });
    let projected = linear(&o, context);
    let last = rows.last().unwrap();
    let hidden = [
        rounded(last[0] + projected[0]),
        rounded(last[1] + projected[1]),
    ];
    let normalized_hidden = rms(hidden, &final_norm);
    let logits = std::array::from_fn(|j| {
        rounded(head[j * 2] * normalized_hidden[0] + head[j * 2 + 1] * normalized_hidden[1])
    });
    let winner = (0..3)
        .max_by(|&a, &b| logits[a].total_cmp(&logits[b]).then_with(|| b.cmp(&a)))
        .unwrap();
    ScalarTrace {
        normalized: *normalized.last().unwrap(),
        query,
        scores,
        context,
        hidden,
        logits,
        token: TokenId::new(winner as u32),
    }
}

#[tokio::test]
async fn visual_prompt_consumer_scalar_projection_trace_matches_single_and_multiple_tokens() {
    let (directory, defined, prepared, engine) = fixture();
    let actual = ActualExecutor::new(&defined, &prepared, &engine);
    let mut observed = Vec::new();
    let mut expected = Vec::new();
    let mut traces = Vec::new();
    for (tokens, rows, chunks, replacement) in [
        (vec![TokenId::new(1)], vec![[0.0, 1.0]], vec![1], vec![]),
        (
            vec![TokenId::new(0)],
            vec![[0.0, 1.0]],
            vec![1],
            vec![(0, [0.0, 1.0], true)],
        ),
        (
            vec![TokenId::new(0), TokenId::new(1)],
            vec![[1.0, 0.0], [0.0, 1.0]],
            vec![2],
            vec![],
        ),
        (
            vec![TokenId::new(0), TokenId::new(1)],
            vec![[1.0, 0.0], [0.0, 1.0]],
            vec![1, 1],
            vec![],
        ),
        (
            vec![TokenId::new(1), TokenId::new(0)],
            vec![[0.0, 1.0], [1.0, 0.0]],
            vec![2],
            vec![],
        ),
    ] {
        let trace = scalar_trace(&prepared, &rows, &[]);
        let result = run_request(
            &actual.executor,
            &tokens,
            &chunks,
            &[],
            request_inputs(&actual.host, &replacement, &[]),
        )
        .await;
        observed.push(result);
        expected.push(vec![trace.token]);
        traces.push(trace);
    }
    assert_eq!(observed, expected,
        "actual prepared scalar evidence for singleton, singleton replacement, whole01, chunked01, whole10: {traces:?}");
    actual.close().await;
    std::fs::remove_dir_all(directory).unwrap();
}

fn cancel(executor: &ExecutorScope, frontier: &ExecutionFrontier) {
    assert!(matches!(
        executor
            .cancel_tokens(frontier.sequence())
            .unwrap()
            .disposition,
        ExecutionCloseDisposition::Ended(_)
    ));
}

async fn run_request(
    executor: &ExecutorScope,
    prompt: &[TokenId],
    chunks: &[usize],
    decode_tokens: &[TokenId],
    inputs: Arc<CausalRequestInputs>,
) -> Vec<TokenId> {
    assert_eq!(chunks.iter().sum::<usize>(), prompt.len());
    let mut frontier = admit(executor, prompt, inputs).await;
    let mut outputs = Vec::new();
    let mut start = 0;
    for &count in chunks {
        let work = TokenWork::Prefill(PrefillChunk::new(start, count, prompt.len()).unwrap());
        let completion = execute(executor, &[(&frontier, work)]).await.pop().unwrap();
        start += count;
        if start == prompt.len() {
            outputs.push(selected(&completion));
        } else {
            assert!(matches!(
                completion.product(),
                ExecutionProduct::Intermediate
            ));
        }
        frontier = completion.next_frontier().clone();
    }
    for &token in decode_tokens {
        let work = TokenWork::Decode {
            token,
            position: frontier.consumed_tokens(),
        };
        let completion = execute(executor, &[(&frontier, work)]).await.pop().unwrap();
        outputs.push(selected(&completion));
        frontier = completion.next_frontier().clone();
    }
    cancel(executor, &frontier);
    outputs
}

#[tokio::test]
async fn visual_prompt_consumer_replaces_only_selected_rows_across_prefill_chunks() {
    let (directory, defined, prepared, engine) = fixture();
    let actual = ActualExecutor::new(&defined, &prepared, &engine);
    let raw = [TokenId::new(0), TokenId::new(0)];
    let plain = run_request(
        &actual.executor,
        &raw,
        &[2],
        &[],
        request_inputs(&actual.host, &[], &[]),
    )
    .await;
    let unselected = run_request(
        &actual.executor,
        &raw,
        &[2],
        &[],
        request_inputs(&actual.host, &[(1, [0.0, 1.0], false)], &[]),
    )
    .await;
    let replaced = run_request(
        &actual.executor,
        &raw,
        &[2],
        &[],
        request_inputs(&actual.host, &[(1, [0.0, 1.0], true)], &[]),
    )
    .await;
    let chunked = run_request(
        &actual.executor,
        &raw,
        &[1, 1],
        &[],
        request_inputs(&actual.host, &[(1, [0.0, 1.0], true)], &[]),
    )
    .await;
    let token_embedding = run_request(
        &actual.executor,
        &[TokenId::new(0), TokenId::new(1)],
        &[2],
        &[],
        request_inputs(&actual.host, &[], &[]),
    )
    .await;
    assert_eq!(plain, [TokenId::new(0)]);
    assert_eq!(
        unselected, plain,
        "mask zero preserves the original token embedding"
    );
    assert_eq!(replaced, [TokenId::new(1)]);
    assert_eq!(
        chunked, replaced,
        "replacement at source token 1 survives a narrowed upload"
    );
    assert_eq!(
        replaced, token_embedding,
        "the selected visual row equals the actual alternate token embedding"
    );
    actual.close().await;
    std::fs::remove_dir_all(directory).unwrap();
}

#[tokio::test]
async fn visual_prompt_consumer_uses_explicit_rope_at_prefill_and_decode_source_positions() {
    let (directory, defined, prepared, engine) = fixture();
    let actual = ActualExecutor::new(&defined, &prepared, &engine);
    let prompt = [TokenId::new(0), TokenId::new(1)];
    let decode = [TokenId::new(1)];
    let identity = run_request(
        &actual.executor,
        &prompt,
        &[2],
        &decode,
        request_inputs(&actual.host, &[], &[]),
    )
    .await;
    let chunked = run_request(
        &actual.executor,
        &prompt,
        &[1, 1],
        &decode,
        request_inputs(&actual.host, &[], &[]),
    )
    .await;
    // Physical KV slot zero is unchanged. Its supplied table rotates key zero
    // onto the second token's query, changing the actual attention result.
    let changed_key = run_request(
        &actual.executor,
        &prompt,
        &[1, 1],
        &[],
        request_inputs(&actual.host, &[], &[(0, 0.0, 1.0)]),
    )
    .await;
    // Only source token 2 changes. Both prefills must therefore agree, while
    // decode consumes its own table row even after two separate prompt chunks.
    let changed_decode = run_request(
        &actual.executor,
        &prompt,
        &[2],
        &decode,
        request_inputs(&actual.host, &[], &[(2, 0.0, -1.0)]),
    )
    .await;
    let changed_decode_chunked = run_request(
        &actual.executor,
        &prompt,
        &[1, 1],
        &decode,
        request_inputs(&actual.host, &[], &[(2, 0.0, -1.0)]),
    )
    .await;
    assert_eq!(identity, [TokenId::new(1), TokenId::new(1)]);
    assert_eq!(chunked, identity);
    assert_eq!(changed_key, [TokenId::new(0)]);
    assert_eq!(changed_decode, [TokenId::new(1), TokenId::new(0)]);
    assert_eq!(changed_decode_chunked, changed_decode);
    actual.close().await;
    std::fs::remove_dir_all(directory).unwrap();
}

#[tokio::test]
async fn visual_prompt_consumer_interleaves_distinct_request_owners_and_cancels_peer() {
    let (directory, defined, prepared, engine) = fixture();
    let actual = ActualExecutor::new(&defined, &prepared, &engine);
    let prompt = [TokenId::new(0), TokenId::new(0)];
    let decode_token = TokenId::new(1);
    let independent_a = run_request(
        &actual.executor,
        &prompt,
        &[2],
        &[decode_token],
        request_inputs(&actual.host, &[], &[]),
    )
    .await;
    let independent_b = run_request(
        &actual.executor,
        &prompt,
        &[2],
        &[decode_token],
        request_inputs(&actual.host, &[(1, [0.0, 1.0], true)], &[]),
    )
    .await;
    assert_ne!(
        independent_a, independent_b,
        "the requests must distinguish crossed input/output ownership"
    );
    let mut a = admit(
        &actual.executor,
        &prompt,
        request_inputs(&actual.host, &[], &[]),
    )
    .await;
    let first = TokenWork::Prefill(PrefillChunk::new(0, 1, prompt.len()).unwrap());
    let completion = execute(&actual.executor, &[(&a, first)])
        .await
        .pop()
        .unwrap();
    assert!(matches!(
        completion.product(),
        ExecutionProduct::Intermediate
    ));
    a = completion.next_frontier().clone();
    drop(completion);
    let mut b = admit(
        &actual.executor,
        &prompt,
        request_inputs(&actual.host, &[(1, [0.0, 1.0], true)], &[]),
    )
    .await;
    let final_prompt = TokenWork::Prefill(PrefillChunk::new(1, 1, prompt.len()).unwrap());
    // Wave order differs from admission order, with different source spans.
    let completions = execute(&actual.executor, &[(&b, first), (&a, final_prompt)]).await;
    assert!(matches!(
        completions[0].product(),
        ExecutionProduct::Intermediate
    ));
    assert_eq!(selected(&completions[1]), independent_a[0]);
    b = completions[0].next_frontier().clone();
    a = completions[1].next_frontier().clone();
    drop(completions);
    let completions = execute(
        &actual.executor,
        &[
            (
                &a,
                TokenWork::Decode {
                    token: decode_token,
                    position: 2,
                },
            ),
            (&b, final_prompt),
        ],
    )
    .await;
    assert_eq!(selected(&completions[0]), independent_a[1]);
    assert_eq!(selected(&completions[1]), independent_b[0]);
    a = completions[0].next_frontier().clone();
    b = completions[1].next_frontier().clone();
    drop(completions);
    cancel(&actual.executor, &a);
    let completion = execute(
        &actual.executor,
        &[(
            &b,
            TokenWork::Decode {
                token: decode_token,
                position: 2,
            },
        )],
    )
    .await
    .pop()
    .unwrap();
    assert_eq!(selected(&completion), independent_b[1], "cancelling the other sequence preserves this request's exact table and sparse visual owners");
    b = completion.next_frontier().clone();
    drop(completion);
    cancel(&actual.executor, &b);
    actual.close().await;
    std::fs::remove_dir_all(directory).unwrap();
}
