//! Small original Qwen GDN + full-attention source. Teacher-forced tokens
//! isolate state progression from greedy feedback; no completion is injected.
use super::*;
use ferrum_execution::{
    model_executor::{
        ExecutorPrefillAdmission, ExecutorSamplingOutput, LogitsReturnPolicy, PrefillChunk,
    },
    token_execution::*,
};
use ferrum_models::vnext::qwen35::Qwen35DensePrecision;
use ferrum_types::RequestId;
use safetensors::tensor::{Dtype, TensorView};

pub(super) struct NonzeroFixture {
    pub assets: NativeTextExpansion,
    _files: Fixture,
}

impl NonzeroFixture {
    pub fn new() -> Self {
        Self::with_precision(Qwen35DensePrecision::F16)
    }

    pub fn with_precision(precision: Qwen35DensePrecision) -> Self {
        let mut files = Fixture::new();
        drop(files.take_expansion());
        let original_dir = crate::registry::tests::write_qwen35_fixture_model_dir();
        let bytes = std::fs::read(original_dir.join("model.safetensors")).unwrap();
        let original = safetensors::SafeTensors::deserialize(&bytes).unwrap();
        let directions = [
            [1., 0.],
            [0., 1.],
            [-1., 0.],
            [0., -1.],
            [0.707, 0.707],
            [-0.707, 0.707],
            [-0.707, -0.707],
            [0.707, -0.707],
        ];
        let owned = original
            .tensors()
            .into_iter()
            .map(|(name, tensor)| {
                assert_eq!(tensor.dtype(), Dtype::F32);
                let projection =
                    name == "model.embed_tokens.weight" || name == "model.lm_head.weight";
                let shape = if projection {
                    vec![8, 2]
                } else {
                    tensor.shape().to_vec()
                };
                let values = if projection {
                    directions.into_iter().flatten().collect::<Vec<f32>>()
                } else {
                    tensor
                        .data()
                        .chunks_exact(4)
                        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                        .collect()
                };
                let bytes = values
                    .into_iter()
                    .flat_map(|v| half::bf16::from_f32(v).to_le_bytes())
                    .collect::<Vec<_>>();
                (name, shape, bytes)
            })
            .collect::<Vec<_>>();
        let views = owned.iter().map(|(name, shape, bytes)| {
            (
                name.as_str(),
                TensorView::new(Dtype::BF16, shape.clone(), bytes).unwrap(),
            )
        });
        safetensors::tensor::serialize_to_file(
            views,
            &None,
            &files.directory.join("model.safetensors"),
        )
        .unwrap();
        std::fs::remove_dir_all(original_dir).unwrap();
        let source = Arc::new(
            ProductionModelSourceBundle::open_colocated_safetensors(&files.directory).unwrap(),
        );
        let mut assets = NativeTextExpansion::from_sources_with_precision(
            source,
            super::super::tests::fixture::protocol_limits(),
            precision,
        )
        .unwrap();
        assets
            .configure_mtp(PH3ImagePromptExpansionMtp::Off)
            .unwrap();
        Self {
            assets,
            _files: files,
        }
    }
}

pub(super) async fn run<R: DeviceRuntime>(
    fixture: &NonzeroFixture,
    composition: &RuntimeComposition<R>,
    device: ferrum_types::Device,
    replay: bool,
    prompt: &[u32],
) -> (Vec<u32>, serde_json::Value) {
    run_with_replay_limit(fixture, composition, device, replay, prompt, None).await
}

pub(super) async fn run_with_replay_limit<R: DeviceRuntime>(
    fixture: &NonzeroFixture,
    composition: &RuntimeComposition<R>,
    device: ferrum_types::Device,
    replay: bool,
    prompt: &[u32],
    maximum_executables_per_lane: Option<u64>,
) -> (Vec<u32>, serde_json::Value) {
    assert!(replay || maximum_executables_per_lane.is_none());
    let mut engine = EngineConfig::default();
    engine.backend.device = device;
    if replay {
        engine.backend.reusable_execution_capture.scope =
            ferrum_types::ReusableExecutionCaptureScope::DecodeOnly;
        engine.backend.reusable_execution_capture.preparation =
            ferrum_types::ReusableExecutionPreparationMode::OnDemand;
        engine
            .backend
            .reusable_execution_capture
            .maximum_executables_per_lane = maximum_executables_per_lane;
    }
    let mut prepared = fixture
        .assets
        .prepare_text("a cat", 32, 32, 12, 3, engine)
        .unwrap();
    prepared.tokens = prompt.iter().copied().map(TokenId::new).collect();
    prepared.engine.runtime.max_model_len = Some(32);
    let host = super::super::tests::fixture::host();
    let requirements = fixture
        .assets
        .host_requirements(composition, &prepared, &host)
        .unwrap();
    let phase = host.reserve_scope(&requirements.claims()).unwrap();
    let owner = fixture
        .assets
        .initialize_owned(composition, &prepared, phase)
        .unwrap();
    let scope = owner.scope().unwrap();
    assert!(scope.status().is_ready);
    let request = RequestId::new();
    let mut frontier = tokio::time::timeout(std::time::Duration::from_secs(10), async {
        loop {
            let context = ExecutorPrefillAdmission::for_product_request(
                &request, &prepared.tokens, prompt.len() + 12, prompt.len(), 0,
            ).unwrap();
            match scope.admit_tokens(ExecutionAdmission { context, adapter: None }).await.unwrap() {
                TokenAdmissionOutcome::Admitted(frontier) => break frontier,
                TokenAdmissionOutcome::MaintenanceDeferred(_) => {
                    assert!(matches!(scope.maintain_prefill_backing(&request).unwrap(),
                        ferrum_execution::model_executor::ExecutorPrefillMaintenanceOutcome::Maintained { .. }
                        | ferrum_execution::model_executor::ExecutorPrefillMaintenanceOutcome::RetryAdmission { .. }));
                }
                other => panic!("tiny replay admission failed: {other:?}"),
            }
        }
    }).await.expect("tiny original admission must complete");
    let policy = LogitsReturnPolicy::GreedyArgmax {
        token_mask: None,
        repetition_penalty: None,
    };
    let mut observed = Vec::new();
    // Changing current inputs and advancing positions require new bindings on
    // every replay. The original GDN and KV state are deliberately nonzero.
    let decode_tokens = [0, 1, 2, 3, 4, 5, 6, 7, 0];
    let work =
        std::iter::once(TokenWork::Prefill(
            PrefillChunk::new(0, prompt.len(), prompt.len()).unwrap(),
        ))
        .chain(decode_tokens.into_iter().enumerate().map(|(index, token)| {
            TokenWork::Decode {
                token: TokenId::new(token),
                position: prompt.len() + index,
            }
        }));
    for work in work {
        let input = ExecutionWork::new(&frontier, work, &policy).unwrap();
        let batch = match scope.execute_tokens(&[input]).await.unwrap() {
            ExecutionBatchOutcome::Completed(batch) => batch,
            outcome => {
                let metrics = scope.cache_metrics_snapshot().unwrap();
                panic!(
                    "tiny replay wave must complete: replay={replay}, prompt={prompt:?}, work={work:?}, outcome={outcome:?}, counters={}, last_failure={}",
                    metrics["counters"], metrics["last_failure"]
                );
            }
        };
        let mut products = batch.into_completions();
        assert_eq!(products.len(), 1);
        let completed = products.pop().unwrap();
        let ExecutionProduct::Sampling(ExecutorSamplingOutput::GreedyToken(token)) =
            completed.product()
        else {
            panic!("managed greedy output required")
        };
        observed.push(token.get());
        frontier = completed.next_frontier().clone();
    }
    assert_eq!(
        frontier.consumed_tokens(),
        prompt.len() + decode_tokens.len()
    );
    let metrics = scope.cache_metrics_snapshot().unwrap();
    assert_eq!(metrics["counters"]["completed_waves"].as_u64(), Some(10));
    assert_eq!(metrics["counters"]["failed_waves"].as_u64(), Some(0));
    assert!(matches!(
        scope
            .cancel_tokens(frontier.sequence())
            .unwrap()
            .disposition,
        ExecutionCloseDisposition::Ended(_)
    ));
    drop(frontier);
    drop(scope);
    owner.begin_shutdown().wait().await.result().unwrap();
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    (observed, metrics)
}

#[tokio::test]
async fn decode_replay_nonzero_fixture_depends_on_current_input_and_prior_state() {
    let fixture = NonzeroFixture::new();
    let (composition, runtime) = super::super::tests::fixture::composition();
    let a = run(
        &fixture,
        &composition,
        ferrum_types::Device::CPU,
        false,
        &[0, 0, 0],
    )
    .await
    .0;
    let b = run(
        &fixture,
        &composition,
        ferrum_types::Device::CPU,
        false,
        &[1, 1, 1],
    )
    .await
    .0;
    assert_eq!(a.len(), 10);
    assert_eq!(b.len(), 10);
    assert_ne!(
        &a[1..],
        &b[1..],
        "same decode inputs must observe different retained prefix state"
    );
    assert!(
        a.windows(2).any(|v| v[0] != v[1]),
        "current inputs must affect actual output"
    );
    assert_eq!(runtime.resident_bytes(), 0);
}
