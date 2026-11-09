//! One original tiny mixed checkpoint, CUDA context, Host account and image
//! serial actor. This is a DiT consumer fixture, without invented TE/VAE assets.
use super::super::*;
use crate::image_generation::{
    continuous_trace::{
        ImageContinuousTraceHandle, ImageContinuousTraceLimits, ImageContinuousTraceRecorder,
    },
    ImageCudaAttentionPolicy, ImageDecoderBatchPolicy, ImageDynamicResidencyPolicy,
    ImageEditPrefixPolicy, ImagePipelineConfig, ImagePipelineExecution, ImagePipelineSourceAdapter,
    ImagePlanCachePolicy, ImageResidencyPolicy, ImageStage, ImageStageAttachment,
    ImageStageDeclaration, ImageStagePlanRole, ImageWeightOwnerKey,
};
use ferrum_kernels::backend::cuda::{
    vnext_ops::{
        cuda_tensor_runtime_config, cuda_vnext_affine_linear_f32_provider,
        cuda_vnext_cnn_f32_providers_with_segmented_sdpa, cuda_vnext_tensor_providers,
        CudaTensorOperation as Op,
    },
    vnext_runtime::{CudaDeviceRuntime, CudaTensorRuntimeConfig},
    CudnnSdpaPlanPolicy,
};
use ferrum_models::{
    p_h3_image::{
        PH3ImageDenoiserAdapter, PH3ImageDenoiserAttentionPolicy, PH3ImageDenoiserPhase,
        PH3ImageModality, PH3ImageReferenceImage, PH3ImageSamplingLimits, PH3ImageSamplingSchedule,
        PreparedPH3ImageDenoiser,
    },
    test_support::TinyPH3ImageDiTFixture,
};
use ferrum_quantization::comfy_convrot::ComfyCanonicalWeightSource;
use serde_json::{json, Value};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Mutex, MutexGuard, Weak,
};

pub(super) type Runtime = CudaDeviceRuntime;
pub(super) type Execution = ImagePipelineExecution<Runtime, Sources>;
static GPU_FIXTURE: Mutex<()> = Mutex::new(());
pub(super) fn gpu_guard() -> MutexGuard<'static, ()> {
    // Each test owns a separate real runtime and finite device grant. Keep
    // these fixtures serial, including after an earlier test panics.
    GPU_FIXTURE
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}
pub(super) fn id<T: TryFrom<String>>(s: impl Into<String>) -> T
where
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
pub(super) fn tensor(
    host: &HostMemoryAccount,
    name: &str,
    spec: ResolvedTensorSpec,
    values: impl IntoIterator<Item = f32>,
) -> ManagedHostTensor {
    let mut bytes = host
        .allocate_zeroed(
            HostMemoryClass::InputPreprocess,
            spec.minimum_storage_bytes().unwrap() as usize,
        )
        .unwrap();
    let mut values = values.into_iter();
    for out in bytes.as_mut().chunks_exact_mut(4) {
        out.copy_from_slice(
            &values
                .next()
                .expect("complete original F32 fixture input")
                .to_le_bytes(),
        );
    }
    assert!(values.next().is_none());
    ManagedHostTensor::new(id(name), spec, bytes.freeze()).unwrap()
}
fn spec(dims: Vec<u64>) -> ResolvedTensorSpec {
    ResolvedTensorSpec::new(dims, ElementType::F32, ResolvedTensorLayout::Contiguous).unwrap()
}
#[derive(Clone)]
pub(super) struct Sources {
    source: Arc<ComfyCanonicalWeightSource>,
    reads: Arc<AtomicUsize>,
}
pub(super) struct Source<'a>(&'a ComfyCanonicalWeightSource);
impl WeightComponentSource for Source<'_> {
    fn component<'a>(
        &'a self,
        c: &WeightComponentSpec,
    ) -> Result<WeightComponentPayload<'a>, VNextError> {
        self.0.component(c)
    }
}
impl ImagePipelineSourceAdapter for Sources {
    type Source<'a> = Source<'a>;
    fn supports_stage(&self, stage: ImageStage) -> bool {
        matches!(stage, ImageStage::Denoiser | ImageStage::TextRefiner)
    }
    fn weight_owner(&self, stage: ImageStage) -> Option<ImageWeightOwnerKey> {
        self.supports_stage(stage)
            .then_some(ImageWeightOwnerKey::Denoiser)
    }
    fn stage_source<'a>(
        &'a self,
        d: &ImageStageDeclaration,
        _: &NumericalProfileId,
        _: &'a HostMemoryAccount,
    ) -> Result<Option<Self::Source<'a>>> {
        ensure!(
            self.supports_stage(d.stage) && d.weight_owner == Some(ImageWeightOwnerKey::Denoiser),
            "only the real DiT source is available"
        );
        self.reads.fetch_add(1, Ordering::SeqCst);
        Ok(Some(Source(&self.source)))
    }
    fn prepare_host_backing(
        &self,
        d: &ImageStageDeclaration,
        _: &NumericalProfileId,
        family: &PreparedModelFamily,
        plan: &ExecutionPlan,
        _: &HostMemoryAccount,
    ) -> Result<(usize, HostBackingPreparation)> {
        ensure!(
            self.supports_stage(d.stage)
                && plan.payload().execution_weights().stored()?
                    == &ExecutionWeightPlan::identity(family)?,
            "tiny DiT uses its complete original identity source"
        );
        let backing = plan.prepare_host_backing(family, None)?;
        Ok((backing.additional_bytes(), backing))
    }
}
pub(super) struct Fixture {
    pub execution: Execution,
    pub adapter: PH3ImageDenoiserAdapter,
    pub assets: TinyPH3ImageDiTFixture,
    pub receipts: Vec<ImageStageReceipt>,
    pub runtime: Weak<Runtime>,
    pub trace: ImageContinuousTraceHandle,
    pub attention_policy: ImageCudaAttentionPolicy,
    work: Option<HostMemoryScope>,
}
impl Fixture {
    pub fn new() -> Self {
        Self::with_attention_policy(ImageCudaAttentionPolicy::Sequential)
    }
    pub fn with_attention_policy(attention_policy: ImageCudaAttentionPolicy) -> Self {
        let assets = TinyPH3ImageDiTFixture::new();
        let mut adapter = PH3ImageDenoiserAdapter::new(
            assets.transformer().clone(),
            assets.source(),
            assets.limits(),
        )
        .unwrap();
        adapter
            .set_attention_policy(match attention_policy {
                ImageCudaAttentionPolicy::Sequential => PH3ImageDenoiserAttentionPolicy::Sequential,
                ImageCudaAttentionPolicy::CudnnSdpaExperimentalV1 => {
                    PH3ImageDenoiserAttentionPolicy::CudnnSdpaExperimentalV1
                }
            })
            .unwrap();
        // Reuse the product fixture's explicit native library/dependency
        // declaration. Sequential keeps its existing no-cuDNN path unchanged.
        let cudnn =
            (attention_policy == ImageCudaAttentionPolicy::CudnnSdpaExperimentalV1).then(|| {
                crate::image_generation::p_h3::tests::cuda_pipeline::config()
                    .unwrap()
                    .cudnn
            });
        let bootstrap_memory_bytes = (256u64 << 20)
            .checked_add(
                cudnn
                    .as_ref()
                    .map_or(0, |config| config.opaque_memory_bytes),
            )
            .and_then(|bytes| bytes.checked_add(attention_policy.additional_opaque_bytes()))
            .unwrap();
        let capacity = 1 << 30;
        let runtime = Arc::new(
            CudaDeviceRuntime::new_tensor_eager(CudaTensorRuntimeConfig {
                device: cuda_tensor_runtime_config(0, id("device.cuda.0")).unwrap(),
                usable_memory_bytes: capacity,
                bootstrap_memory_bytes,
                stream_memory_bytes: 16 << 20,
                // Original serial actor plus temporary static installation.
                maximum_streams: 2,
                maximum_in_flight_per_stream: 1,
                maximum_commands_per_submission: 4096,
                opaque_memory_profile: "tiny-ph3-production-step-unqualified-capacity".into(),
            })
            .unwrap(),
        );
        // Same native selection as the existing full Stored DiT fixture. All
        // selected functions share one retained bundle and the one lane BLAS.
        let selected = [
            Op::AffineLinear,
            Op::CastF32ToBf16,
            Op::ReshapeCopy,
            Op::RowGather,
            Op::BatchedRowConcat,
            Op::RowRmsNormSingleRound,
            Op::ModulationScale,
            Op::RowSelectedModulationBf16,
            Op::RowSelectedAddcmulBf16,
            Op::PlainResidualAdd,
            Op::FusedHeadRmsPartialRope,
            Op::Addcmul,
            Op::SegmentedAttention,
            Op::RowSelectedModulationBf16F32,
            Op::ComfyConvRotW4A8,
            Op::ComfyConvRotSwiGluF32W4A8,
            Op::ComfyConvRotW8A8,
            Op::ComfyConvRotSwiGluF32W8A8,
        ];
        let mut providers = match &cudnn {
            None => cuda_vnext_tensor_providers(&runtime, &selected),
            Some(config) => cuda_vnext_cnn_f32_providers_with_segmented_sdpa(
                &runtime,
                config,
                CudnnSdpaPlanPolicy {
                    maximum_plans: 64,
                    maximum_live_bindings: 2048,
                    opaque_memory_bytes: attention_policy.additional_opaque_bytes(),
                    workspace_limit_bytes: config.workspace_limit_bytes,
                },
                &selected,
            ),
        }
        .unwrap();
        let attention = providers
            .iter()
            .filter(|provider| {
                provider.descriptor().operation_id().as_str()
                    == SEGMENTED_ATTENTION_BF16_OPERATION_ID
            })
            .collect::<Vec<_>>();
        assert_eq!(
            attention.len(),
            1,
            "one selected segmented implementation, without fallback"
        );
        if cudnn.is_some() {
            assert_eq!(
                attention[0].descriptor().provider_id().as_str(),
                "provider.cuda.segmented_attention.bf16.cudnn920"
            );
        }
        drop(attention);
        providers.push(cuda_vnext_affine_linear_f32_provider(&runtime).unwrap());
        let mut contracts: Vec<Box<dyn OperationContract>> = vec![
            Box::new(affine_linear_bf16_contract().unwrap()),
            Box::new(affine_linear_f32_contract().unwrap()),
            Box::new(cast_f32_to_bf16_contract().unwrap()),
            Box::new(reshape_copy_bf16_contract().unwrap()),
            Box::new(row_gather_bf16_contract().unwrap()),
            Box::new(batched_row_concat_bf16_contract().unwrap()),
            Box::new(row_rms_norm_single_round_bf16_contract().unwrap()),
            Box::new(modulation_scale_bf16_contract().unwrap()),
            Box::new(row_selected_modulation_bf16_contract().unwrap()),
            Box::new(row_selected_addcmul_bf16_contract().unwrap()),
            Box::new(plain_residual_add_bf16_contract().unwrap()),
            Box::new(fused_head_rms_partial_rope_bf16_contract().unwrap()),
            Box::new(addcmul_bf16_contract().unwrap()),
            Box::new(segmented_attention_bf16_contract().unwrap()),
            Box::new(row_selected_modulation_bf16_f32_contract().unwrap()),
            Box::new(comfy_convrot_w4a8_bf16_contract().unwrap()),
            Box::new(comfy_convrot_swiglu_f32_w4a8_bf16_contract().unwrap()),
            Box::new(comfy_convrot_w8a8_bf16_contract().unwrap()),
            Box::new(comfy_convrot_swiglu_f32_w8a8_bf16_contract().unwrap()),
        ];
        if cudnn.is_some() {
            // The same product factory retains its one convolution/native owner;
            // this DiT-only graph does not execute or claim to qualify that op.
            contracts.push(Box::new(conv2d_f32_contract().unwrap()));
        }
        let registry = OperationRuntimeRegistry::new(contracts, providers).unwrap();
        let materializers = WeightMaterializerRegistry::identity_only().unwrap();
        let engine = EngineProviderDescriptor::new(
            id("provider.engine.tiny.ph3.step.cuda"),
            ContractVersion::new(1, 0),
            runtime
                .descriptor()
                .runtime_implementation_fingerprint
                .clone(),
            runtime.descriptor().id.clone(),
            runtime.descriptor().capabilities.clone(),
        )
        .unwrap();
        let catalog = materializers
            .augment_catalog(
                registry
                    .capability_catalog(runtime.descriptor().clone(), vec![engine])
                    .unwrap(),
            )
            .unwrap();
        let host = HostMemoryAccount::with_allocation_alignment(
            HostMemoryLimits::new(
                256 << 20,
                HostMemoryClassLimits {
                    input_preprocess_bytes: 128 << 20,
                    output_bytes: 64 << 20,
                    transfer_staging_bytes: 32 << 20,
                    weight_backing_bytes: 32 << 20,
                },
            )
            .unwrap(),
            64,
        )
        .unwrap();
        let config = ImagePipelineConfig {
            device_capacity_bytes: capacity,
            maximum_user_tokens: 1024,
            maximum_width: 16,
            maximum_height: 16,
            maximum_batch_size: 4,
            weight_arena_target_bytes: None,
            decoder_batch_policy: ImageDecoderBatchPolicy::SingleImage,
            edit_prefix_policy: ImageEditPrefixPolicy::Recompute,
            plan_cache: ImagePlanCachePolicy::Disabled,
            residency: ImageResidencyPolicy::DenoiserAndDecoder,
            dynamic_residency: ImageDynamicResidencyPolicy::Release,
            static_initialization: StaticInitializationPolicy::new(1 << 20, 64).unwrap(),
            executor: Default::default(),
        };
        let weak = Arc::downgrade(&runtime);
        let work = host
            .reserve_scope(&[
                (HostMemoryClass::InputPreprocess, 32 << 20),
                (HostMemoryClass::Output, 16 << 20),
                (HostMemoryClass::TransferStaging, 8 << 20),
                (HostMemoryClass::WeightBacking, 0),
            ])
            .unwrap();
        // This is a DiT-only collector, not a complete asset/header recorder.
        // Its one real Output allocation is separate from every lane's exact
        // Input successor grant, including the late-refusal test below.
        let (recorder, trace) = ImageContinuousTraceRecorder::new(
            &host,
            ImageContinuousTraceLimits {
                maximum_bytes: 1 << 20,
            },
        )
        .unwrap();
        let mut execution = ImagePipelineExecution::new(
            runtime,
            host,
            registry,
            catalog,
            materializers,
            config,
            Sources {
                source: assets.source(),
                reads: Arc::new(AtomicUsize::new(0)),
            },
        )
        .unwrap();
        execution.execution_host = Some(work.account());
        execution.continuous_trace = Some(recorder);
        Self {
            execution,
            adapter,
            assets,
            receipts: vec![],
            runtime: weak,
            trace,
            attention_policy,
            work: Some(work),
        }
    }
    pub fn compile(&self, recipe: PreparedPH3ImageDenoiser) -> CompiledSteady<Runtime> {
        if self.attention_policy == ImageCudaAttentionPolicy::CudnnSdpaExperimentalV1 {
            let expected = match recipe.phase() {
                PH3ImageDenoiserPhase::RefineFreshText => {
                    Some("p_h3_image.dit_forward.cuda.refine_only.comfy_mixed.cudnn_sdpa.v1")
                }
                PH3ImageDenoiserPhase::DenoiseRefinedText => {
                    Some("p_h3_image.dit_forward.cuda.refined_text.comfy_mixed.cudnn_sdpa.v1")
                }
                PH3ImageDenoiserPhase::InstallFullWeights => None,
            };
            if let Some(expected) = expected {
                assert_eq!(
                    recipe.family().numerical_profile().id.as_str(),
                    expected,
                    "the actual attached phase must select its explicit vendor numerical profile"
                );
            }
        }
        let (stage, attachment) = match recipe.phase() {
            PH3ImageDenoiserPhase::InstallFullWeights => {
                (ImageStage::Denoiser, ImageStageAttachment::InitializeShared)
            }
            PH3ImageDenoiserPhase::RefineFreshText => {
                (ImageStage::TextRefiner, ImageStageAttachment::RequireShared)
            }
            PH3ImageDenoiserPhase::DenoiseRefinedText => {
                (ImageStage::Denoiser, ImageStageAttachment::RequireShared)
            }
        };
        let prepared = self
            .execution
            .compile_stage(
                ImageStageDeclaration {
                    stage,
                    role: ImageStagePlanRole::WholeStage,
                    batch_size: recipe.request_ids().len(),
                    weight_owner: Some(ImageWeightOwnerKey::Denoiser),
                    attachment,
                },
                recipe.family().clone(),
                recipe.compile_options(),
            )
            .unwrap();
        assert_eq!(
            prepared.plan.profile,
            recipe.family().numerical_profile().id,
            "use the prepared mixed-checkpoint numerical profile"
        );
        assert_eq!(prepared.plan.batch_size, recipe.request_ids().len());
        assert_eq!(
            prepared.plan.family_fingerprint,
            recipe.family().fingerprint().unwrap()
        );
        CompiledSteady {
            recipe,
            stage: prepared,
        }
    }
    pub fn install(&mut self, geometry: &RequestGeometry) {
        let batch = self.batch(&[geometry]);
        let compiled = self.compile(
            batch
                .prepare(&self.adapter, PH3ImageDenoiserPhase::InstallFullWeights)
                .unwrap(),
        );
        let active = self
            .execution
            .start_stage(compiled.stage, 0, &mut self.receipts)
            .unwrap_or_else(|e| panic!("install: {e}"));
        assert_eq!(
            active.receipt.completed_executions, 0,
            "install cannot advance a request"
        );
        let pin = self
            .execution
            .pin_weight_owner(ImageWeightOwnerKey::Denoiser)
            .unwrap();
        pin.weights()
            .validate_for(
                active.session.executable().execution_plan(),
                &self.execution.runtime,
            )
            .unwrap();
        drop(pin);
        self.execution
            .finish_stage(active, &mut self.receipts)
            .unwrap_or_else(|e| panic!("install retire: {e}"));
        assert_eq!(self.execution.sources.reads.load(Ordering::SeqCst), 1);
    }
    pub fn batch<'a>(&self, members: &[&'a RequestGeometry]) -> BatchGeometry<'a> {
        BatchGeometry::new(
            self.assets.transformer(),
            members.to_vec(),
            4,
            self.assets.limits().max_metadata_bytes,
        )
        .unwrap()
    }
    pub fn geometry(&self, name: &str, reference: bool) -> RequestGeometry {
        let mut tags = vec![PH3ImageModality::Text; self.assets.text_rows()];
        if reference {
            tags[1] = PH3ImageModality::Image;
        }
        RequestGeometry {
            identity: id(format!("request.ph3.step.{name}")),
            target: self.assets.target_latent(),
            reference: reference.then(|| PH3ImageReferenceImage::new(2, 4).unwrap()),
            tags,
            image_sample: spec(vec![1, 2, 1, 4, 4]),
            audio_sample: spec(vec![1, 2, 2, 2]),
        }
    }
    pub fn close(mut self) -> (Weak<Runtime>, Vec<Value>) {
        assert_eq!(
            self.execution.sources.reads.load(Ordering::SeqCst),
            1,
            "joins/shape replacement must never cold-reload the installed checkpoint"
        );
        let host = self.execution.host.clone();
        self.receipts.extend(
            self.execution.shutdown().unwrap_or_else(|e| {
                panic!("all session/snapshot retirements must acknowledge: {e}")
            }),
        );
        assert!(
            self.receipts.iter().any(|r| r.close.is_some()),
            "actual sessions must acknowledge close"
        );
        assert!(
            self.receipts.iter().any(|r| matches!(
                r.retirement,
                crate::image_generation::ImageStageRetirement::SharedWeightsClosed { .. }
            )),
            "original shared installation must acknowledge retirement"
        );
        self.execution.execution_host = None;
        drop(self.work.take());
        // CUDA currently exposes no live managed-byte snapshot. This proves
        // acknowledged fences and final owner release, not global free VRAM.
        let weak = self.runtime.clone();
        let trace = self.trace.clone();
        drop(self);
        assert!(
            weak.upgrade().is_none(),
            "all original Rust Runtime Arc owners must retire after checked closes"
        );
        let events = trace_events(&trace, true);
        assert!(
            events.iter().all(|event| event["kind"] != "header"),
            "a DiT-only fixture cannot supply complete asset/composition evidence"
        );
        assert_eq!(
            host.usage().unwrap().allocated_bytes(),
            trace.status().backing_bytes
        );
        assert_eq!(
            host.usage().unwrap().committed_bytes(),
            trace.status().backing_bytes
        );
        drop(trace);
        assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
        (weak, events)
    }
}

pub(super) fn trace_events(trace: &ImageContinuousTraceHandle, sealed: bool) -> Vec<Value> {
    let mut bytes = Vec::new();
    let status = trace.write_jsonl(&mut bytes).unwrap();
    assert_eq!(status.sealed, sealed);
    assert!(!status.truncated);
    assert_eq!(status.dropped, 0);
    let mut records = bytes
        .split(|b| *b == b'\n')
        .filter(|line| !line.is_empty())
        .map(|line| serde_json::from_slice::<Value>(line).unwrap())
        .collect::<Vec<_>>();
    let footer = records.pop().expect("original collector footer");
    assert_eq!(footer["session_uuid"], json!(status.session_uuid));
    assert_eq!(footer["footer"]["sealed"], sealed);
    assert_eq!(footer["footer"]["truncated"], false);
    assert_eq!(footer["footer"]["recorded"], status.recorded);
    assert_eq!(records.len() as u64, status.recorded);
    records
        .into_iter()
        .enumerate()
        .map(|(index, record)| {
            assert_eq!(record["sequence"], index + 1);
            assert_eq!(record["session_uuid"], json!(status.session_uuid));
            record["event"].clone()
        })
        .collect()
}
pub(super) fn schedule(sigmas: &[f32]) -> PH3ImageSamplingSchedule {
    let index = serde_json::json!({"_class_name":"PH3ImageModularPipeline","_blocks_class_name":"PH3ImageBlocks",
        "scheduler":["diffusers","MiniMaxH3Scheduler",{}],"audio_scheduler":["diffusers","MiniMaxH3Scheduler",{}],"sample_sigmas":sigmas});
    let scheduler =
        br#"{"_class_name":"MiniMaxH3Scheduler","shift":1,"shift_provided_sigmas":false}"#;
    PH3ImageSamplingSchedule::parse(
        &serde_json::to_vec(&index).unwrap(),
        scheduler,
        scheduler,
        PH3ImageSamplingLimits {
            max_metadata_bytes: 4096,
            max_evaluations: 16,
        },
    )
    .unwrap()
}
