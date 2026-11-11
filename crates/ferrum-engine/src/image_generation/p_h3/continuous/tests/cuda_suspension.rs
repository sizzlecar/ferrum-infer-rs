//! Explicit production suspension on the genuine tiny CUDA union. The tiny
//! H2 expansion need not conflict with the larger main TE/VAE native minimum;
//! this therefore tests real retirement/reinstallation, not a fabricated low
//! memory hint. Official low-VRAM automatic admission is separate evidence.
use super::super::*;
use crate::image_generation::{
    p_h3::{
        expansion::MergedExpansionFixture,
        tests::{cuda_pipeline, fixture::CompleteFixture},
    },
    service::{
        ImageBatchError, ImageInitializationOutcome, ImageInitializationProgress, ImageRequestRef,
    },
};
use ferrum_kernels::backend::cuda::{vnext_runtime::CudaDeviceRuntime, CudaDenseF16CausalProfile};
use ferrum_types::{Device, EngineConfig};
use std::sync::{Mutex, MutexGuard};

type Pipeline = PH3ImagePipeline<CudaDeviceRuntime>;
type Request = ContinuousRequest<CudaDeviceRuntime>;
type Batch = ContinuousBatch<CudaDeviceRuntime>;
static GPU: Mutex<()> = Mutex::new(());
pub(super) fn guard() -> MutexGuard<'static, ()> {
    GPU.lock().unwrap_or_else(|p| p.into_inner())
}
pub(super) fn input(seed: u64) -> crate::image_generation::ImageGenerationRequest {
    crate::image_generation::ImageGenerationRequest {
        prompt: "cat".into(),
        width: 32,
        height: 32,
        seed,
        n: 1,
    }
}
fn samples(request: &Request) -> [Vec<u8>; 2] {
    [
        request.image.bytes().bytes().to_vec(),
        request.audio.as_ref().unwrap().bytes().bytes().to_vec(),
    ]
}
fn initialized(pipeline: &mut Pipeline, seed: u64) -> Request {
    let request = input(seed);
    let mut prepared = pipeline
        .prepare_initialization(ImageRequestRef::Generate(&request))
        .unwrap();
    assert!(pipeline.admit_initialization(&mut prepared, 0).unwrap());
    pipeline
        .start_initialization_blocking(prepared, ImageRequestRef::Generate(&request))
        .unwrap()
        .into_inner::<Request>()
        .unwrap()
}
fn batch(pipeline: &mut Pipeline, request: &mut Request) -> Batch {
    pipeline
        .try_prepare_continuous_batch(&mut [request])
        .unwrap_or_else(|e| panic!("real first cohort: {e}"))
        .expect("actual first native admission")
}
fn oracle(pipeline: &mut Pipeline, seed: u64) -> Vec<[Vec<u8>; 2]> {
    let mut request = initialized(pipeline, seed);
    let mut batch = batch(pipeline, &mut request);
    let mut result = vec![samples(&request)];
    while !request.is_complete() {
        batch
            .advance(pipeline, &mut [&mut request])
            .unwrap_or_else(|e| panic!("independent real B1: {e}"));
        result.push(samples(&request));
    }
    batch
        .pause(pipeline, &mut [&mut request])
        .unwrap_or_else(|e| panic!("independent B1 fence: {e}"));
    drop((batch, request));
    pipeline
        .execution
        .release_weight_owner(ImageWeightOwnerKey::Denoiser, &mut vec![])
        .unwrap_or_else(|e| panic!("independent B1 original close: {e}"));
    result
}
pub(super) fn pipeline(fixture: &CompleteFixture, expansion: &MergedExpansionFixture) -> Pipeline {
    let mut engine = EngineConfig::default();
    engine.backend.device = Device::CUDA(0);
    engine.runtime.max_model_len = Some(64);
    let assets = fixture
        .assets_with_cuda_prompt_expansion(
            expansion,
            MergedExpansionFixture::native_options(engine),
        )
        .unwrap();
    let mut configured = cuda_pipeline::config().unwrap();
    configured.pipeline.maximum_batch_size = 2;
    configured.prompt_expansion_causal = Some(
        CudaDenseF16CausalProfile::new(64 << 20, "unqualified-tiny-suspended-visual-causal-v1")
            .unwrap()
            .with_visual_prompt_v1(),
    );
    let (pipeline, receipt) = configured.create_pipeline_with_receipt(assets).unwrap();
    assert_eq!(receipt.stream_plan.resolved_streams(), 3);
    pipeline
}
pub(super) fn finish(mut pipeline: Pipeline) {
    let host = pipeline.host_account().clone();
    let runtime = Arc::downgrade(pipeline.runtime());
    pipeline
        .shutdown()
        .unwrap_or_else(|e| panic!("real final shutdown: {e}"));
    drop(pipeline);
    assert!(
        runtime.upgrade().is_none(),
        "original CUDA owners still retained"
    );
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires actual CUDA tensor+causal union, original native set and cuDNN; explicit real suspension, not official VRAM qualification"]
async fn cuda_h3_suspended_auxiliary_reinstallation_refines_new_snapshot_and_matches_mixed_b1() {
    let _guard = guard();
    let fixture = CompleteFixture::new().unwrap();
    let expansion = MergedExpansionFixture::new();
    let mut pipeline = pipeline(&fixture, &expansion);
    let independent_a = oracle(&mut pipeline, 61);
    let independent_b = oracle(&mut pipeline, 73);
    let mut a = initialized(&mut pipeline, 61);
    let mut original = batch(&mut pipeline, &mut a);
    original
        .advance(&mut pipeline, &mut [&mut a])
        .unwrap_or_else(|e| panic!("first actual A forward: {e}"));
    original
        .pause(&mut pipeline, &mut [&mut a])
        .unwrap_or_else(|e| panic!("A completed fence: {e}"));
    let before = samples(&a);
    let previous = a.weights().unwrap().weights().evidence().clone();
    assert_eq!(a.step_index(), 1);
    assert_eq!(before, independent_a[1]);
    let request_b = input(73);
    let mut prepared = pipeline
        .prepare_initialization(ImageRequestRef::Generate(&request_b))
        .unwrap();
    assert!(
        pipeline.admit_initialization(&mut prepared, 0).unwrap(),
        "real auxiliary Host phase must admit before retirement"
    );
    assert!(pipeline
        .suspend_continuous_batch(&mut original, &mut [&mut a])
        .unwrap_or_else(|e| panic!("actual DiT retirement: {e}")));
    assert!(original.suspended && original.active.is_none());
    assert!(a.weights.is_none() && a.cache.is_none());
    assert!(a.restoration.is_some());
    assert!(original.stages.iter().any(|receipt| matches!(
        receipt.retirement,
        crate::image_generation::ImageStageRetirement::SharedWeightsClosed { .. }
    )));
    // This runs actual CUDA causal prefill/decode and original EOS extraction,
    // waits for its real Retired boundary, then performs original TE/noise and
    // actual RefineOnly for B. No expanded text or completion is injected.
    let mut b = pipeline
        .start_initialization_blocking(prepared, ImageRequestRef::Generate(&request_b))
        .unwrap()
        .into_inner::<Request>()
        .unwrap();
    assert_eq!(b.step_index(), 0);
    assert_eq!(a.step_index(), 1);
    assert_eq!(samples(&a), before);
    assert_ne!(
        previous.snapshot_fingerprint(),
        b.weights()
            .unwrap()
            .weights()
            .evidence()
            .snapshot_fingerprint()
    );
    pipeline
        .restore_continuous_batch(&mut original, &mut [&mut a])
        .unwrap_or_else(|e| panic!("actual restore+RefineOnly: {e}"));
    let current = a.weights().unwrap().weights().evidence();
    assert_ne!(
        previous.snapshot_fingerprint(),
        current.snapshot_fingerprint()
    );
    assert_eq!(
        previous.weight_plan_fingerprint(),
        current.weight_plan_fingerprint()
    );
    assert!(a.cache.as_ref().unwrap().key().matches_snapshot(current));
    assert_eq!(current, b.weights().unwrap().weights().evidence());
    assert_eq!(samples(&a), before);
    assert_eq!(a.step_index(), 1);
    original
        .pause(&mut pipeline, &mut [&mut a])
        .unwrap_or_else(|e| panic!("restored real fence: {e}"));
    let mut mixed = pipeline
        .try_replace_continuous_batch(&mut original, &mut [&mut a, &mut b])
        .unwrap_or_else(|e| panic!("new actual mixed cohort: {e}"))
        .unwrap();
    mixed
        .advance(&mut pipeline, &mut [&mut a, &mut b])
        .unwrap_or_else(|e| panic!("real mixed forward: {e}"));
    assert_eq!([a.step_index(), b.step_index()], [2, 1]);
    assert_eq!(samples(&a), independent_a[2]);
    assert_eq!(samples(&b), independent_b[1]);
    mixed
        .pause(&mut pipeline, &mut [&mut a, &mut b])
        .unwrap_or_else(|e| panic!("real mixed fence: {e}"));
    let mut last = pipeline
        .try_replace_continuous_batch(&mut mixed, &mut [&mut b])
        .unwrap_or_else(|e| panic!("survivor actual compact: {e}"))
        .unwrap();
    last.advance(&mut pipeline, &mut [&mut b])
        .unwrap_or_else(|e| panic!("last actual B1: {e}"));
    assert_eq!(samples(&b), independent_b[2]);
    last.pause(&mut pipeline, &mut [&mut b])
        .unwrap_or_else(|e| panic!("last actual fence: {e}"));
    drop((a, b, original, mixed, last));
    finish(pipeline);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "requires actual CUDA union/native set/cuDNN; cancellation must retire causal before real survivor restore"]
async fn cuda_h3_suspended_auxiliary_cancel_retires_causal_before_real_survivor_restore() {
    let _guard = guard();
    let fixture = CompleteFixture::new().unwrap();
    let expansion = MergedExpansionFixture::new();
    let mut pipeline = pipeline(&fixture, &expansion);
    let independent = oracle(&mut pipeline, 61);
    let mut a = initialized(&mut pipeline, 61);
    let mut original = batch(&mut pipeline, &mut a);
    original
        .advance(&mut pipeline, &mut [&mut a])
        .unwrap_or_else(|e| panic!("A first actual forward: {e}"));
    original
        .pause(&mut pipeline, &mut [&mut a])
        .unwrap_or_else(|e| panic!("A actual fence: {e}"));
    let previous = a.weights().unwrap().weights().evidence().clone();
    let request_b = input(73);
    let mut prepared = pipeline
        .prepare_initialization(ImageRequestRef::Generate(&request_b))
        .unwrap();
    assert!(pipeline.admit_initialization(&mut prepared, 0).unwrap());
    assert!(pipeline
        .suspend_continuous_batch(&mut original, &mut [&mut a])
        .unwrap_or_else(|e| panic!("actual suspend: {e}")));
    let mut initialization = match pipeline
        .begin_initialization(prepared, ImageRequestRef::Generate(&request_b))
        .unwrap()
    {
        ImageInitializationOutcome::Pending(initialization) => initialization,
        ImageInitializationOutcome::Ready(_) => {
            panic!("Enabled causal must use its original asynchronous phase")
        }
    };
    let wake = futures::task::noop_waker();
    tokio::time::timeout(std::time::Duration::from_secs(60), async {
        loop {
            match pipeline
                .advance_initialization(
                    &mut initialization,
                    ImageRequestRef::Generate(&request_b),
                    true,
                    &wake,
                )
                .unwrap()
            {
                ImageInitializationProgress::Pending { .. } => tokio::task::yield_now().await,
                ImageInitializationProgress::Cancelled => break,
                ImageInitializationProgress::Ready(_) => {
                    panic!("cancelled expansion published a request")
                }
                ImageInitializationProgress::Rejected { diagnostic } => {
                    panic!("cancellation was rejected: {diagnostic}")
                }
            }
        }
    })
    .await
    .expect("original causal cancellation reaches Retired");
    assert!(!pipeline.initialization_has_auxiliary_owner(&initialization));
    drop(initialization);
    pipeline
        .restore_continuous_batch(&mut original, &mut [&mut a])
        .unwrap_or_else(|e| panic!("actual survivor restore: {e}"));
    assert_ne!(
        previous.snapshot_fingerprint(),
        a.weights()
            .unwrap()
            .weights()
            .evidence()
            .snapshot_fingerprint()
    );
    assert_eq!(samples(&a), independent[1]);
    original
        .advance(&mut pipeline, &mut [&mut a])
        .unwrap_or_else(|e| panic!("survivor actual forward: {e}"));
    assert_eq!(samples(&a), independent[2]);
    original
        .pause(&mut pipeline, &mut [&mut a])
        .unwrap_or_else(|e| panic!("survivor original fence: {e}"));
    drop((a, original));
    finish(pipeline);
}
