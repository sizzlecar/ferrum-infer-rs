//! Original FastH3 stages on one Core runtime and serial actor. The existing
//! iterative service owns membership, cancellation, completion and shutdown.
use super::{
    assets::FastH3Assets,
    cuda::{FastH3CudaComposition, FastH3CudaConfig},
    worker::*,
};
use crate::{
    iterative_service::{BatchError, BatchWorker, ServiceError},
    video_generation::{
        VideoGenerationParameters, VideoGenerationRequest, VideoOutputPlan, VideoService,
        VideoServiceConfig, VideoServiceRun,
    },
};
use anyhow::{ensure, Context};
use ferrum_execution::{
    component::ComponentInitializationConfig, TensorExecutorConfig, TensorSerialActor,
};
use ferrum_kernels::backend::cuda::vnext_runtime::CudaDeviceRuntime;
use ferrum_models::fast_h3::{vsa::*, *};
use ferrum_models::spatial_tiles::SpatialTileBatch;
use ferrum_output::mp4::VideoAudioSpec;
use ferrum_runtime::*;
use std::{
    any::Any,
    fmt,
    sync::{atomic::Ordering, Arc, Mutex},
};

mod conditioning;
mod decode;
mod denoising;
mod limits;
use super::conditioning::{ConditioningKey, FastH3ConditioningReuse};
use conditioning::{ConditionedRequest, ConditioningPlan, PreparedConditioner};
use decode::{DecodingInvocation, PreparedDecoding};
use denoising::PreparedDenoisingDomain;
pub use limits::FastH3PipelineLimits;

type Device = CudaDeviceRuntime;
type Worker = dyn BatchWorker<
    Request = VideoGenerationRequest,
    Output = crate::video_generation::RawVideoOutput,
>;

/// Pure output geometry survives transfer of the pipeline into its service.
/// It owns no runtime, weight installation, native plan or request payload.
#[derive(Clone)]
pub struct FastH3OutputSpecResolver {
    width: u32,
    height: u32,
    frames: u32,
    audio_sample_rate: u32,
    audio_hop: u64,
}
impl FastH3OutputSpecResolver {
    pub fn resolve(
        &self,
        parameters: &VideoGenerationParameters,
    ) -> anyhow::Result<VideoAudioSpec> {
        let geometry = self.geometry(parameters)?;
        let spec = VideoAudioSpec {
            width: parameters.width,
            height: parameters.height,
            frames: geometry.aligned_frames() as u64,
            fps_numerator: 24,
            fps_denominator: 1,
            audio_sample_rate: self.audio_sample_rate,
            audio_sample_frames: (geometry.audio_latents_per_channel() as u64)
                .checked_mul(self.audio_hop)
                .context("audio output extent overflow")?,
        };
        spec.rgb_bytes()
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        spec.pcm_bytes()
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        Ok(spec)
    }
    fn geometry(&self, p: &VideoGenerationParameters) -> anyhow::Result<FastH3VideoGeometry> {
        ensure!(
            p.width <= self.width && p.height <= self.height && p.num_frames <= self.frames,
            "video exceeds the declared pipeline geometry limits"
        );
        ensure!(
            p.frame_rate.denominator > 0
                && u64::from(p.frame_rate.numerator) == 24 * u64::from(p.frame_rate.denominator),
            "original FastH3 pipeline requires 24 fps"
        );
        let geometry =
            FastH3VideoGeometry::new(p.width as usize, p.height as usize, p.num_frames as usize)?;
        ensure!(geometry.aligned_frames() == p.num_frames as usize,
            "FastH3 frame count must already follow 5 + 17*n; resolve the requested duration explicitly");
        Ok(geometry)
    }
}

/// Construction may already have submitted cold Time work. Retained variants
/// preserve the original failure together with its actor and runtime owner.
#[must_use]
pub struct FastH3PipelineError {
    diagnostic: String,
    _owner: Option<Box<dyn Any + Send>>,
}
impl FastH3PipelineError {
    fn preparation(error: impl fmt::Display) -> Self {
        Self {
            diagnostic: error.to_string(),
            _owner: None,
        }
    }
    fn retained<E: fmt::Display + Send + 'static, P: Send + 'static>(error: E, phase: P) -> Self {
        let owner = Box::new((error, phase));
        let diagnostic =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| owner.0.to_string()))
                .unwrap_or_else(|_| {
                    "FastH3 construction failed with original owners retained".into()
                });
        Self {
            diagnostic,
            _owner: Some(owner),
        }
    }
}
impl fmt::Display for FastH3PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.diagnostic)
    }
}
impl fmt::Debug for FastH3PipelineError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
impl std::error::Error for FastH3PipelineError {}

pub struct FastH3Pipeline {
    worker: Box<Worker>,
    host: HostMemoryAccount,
    resolver: FastH3OutputSpecResolver,
}
impl FastH3Pipeline {
    pub fn new(
        assets: Arc<FastH3Assets>,
        config: FastH3CudaConfig,
        limits: FastH3PipelineLimits,
        video_input: ferrum_models::video_input::VideoInputCodec,
    ) -> Result<Self, FastH3PipelineError> {
        let resolver = FastH3OutputSpecResolver {
            width: limits.maximum_width,
            height: limits.maximum_height,
            frames: limits.maximum_frames,
            audio_sample_rate: assets.audio_vae.sampling_rate(),
            audio_hop: assets
                .audio_vae
                .hop_length()
                .map_err(FastH3PipelineError::preparation)?,
        };
        let cuda = Arc::new(
            FastH3CudaComposition::new(&config).map_err(FastH3PipelineError::preparation)?,
        );
        let mut actor = TensorSerialActor::new(cuda.runtime.clone())
            .map_err(|e| FastH3PipelineError::retained(e, cuda.clone()))?;
        let schedule = FastH3ConditioningSchedule::all_modalities(&assets.pdd)
            .map_err(FastH3PipelineError::preparation)?;
        let registry = cuda
            .registry(None)
            .map_err(FastH3PipelineError::preparation)?;
        let catalog = cuda
            .catalog(&registry)
            .map_err(FastH3PipelineError::preparation)?;
        let mut identities = Identities {
            next: 0,
            staging: config.staging_bytes,
        };
        let cold_request = RequestIdentity::new("request.fast_h3.cold")
            .map_err(FastH3PipelineError::preparation)?;
        let compilation = super::conditioning::ConditioningCompilation {
            registry: &registry,
            catalog: &catalog,
            policy: &cuda.policy,
            materializers: &cuda.materializers,
        };
        let tables = match super::conditioning::precompute_tables(
            assets.transformer.clone(),
            &schedule,
            &compilation,
            cuda.runtime.clone(),
            &cuda.host,
            &mut actor,
            limits.time(),
            limits.output_conditioning(),
            |name| identities.initialization(&cold_request, name),
        ) {
            Ok(tables) => tables,
            Err(e) => return Err(FastH3PipelineError::retained(e, (actor, cuda))),
        };
        let conditioner = PreparedConditioner::new(
            assets.clone(),
            cuda.clone(),
            limits.conditioning,
            video_input,
        )
        .map_err(FastH3PipelineError::preparation)?;
        if let Some(bytes) = config.initial_allocation_cache_bytes {
            match cuda.runtime.prepare_tensor_allocation_cache(bytes) {
                Ok(outcome) => tracing::info!(
                    requested_bytes = bytes.get(),
                    ?outcome,
                    "FastH3 startup allocation cache preparation"
                ),
                Err(error) => {
                    return Err(FastH3PipelineError::retained(error, (actor, cuda)));
                }
            }
        }
        let state = Arc::new(Mutex::new(State {
            assets,
            cuda: cuda.clone(),
            limits,
            schedule,
            tables,
            identities,
            conditioner,
            last_domain: None,
        }));
        let stages = stages(state, resolver.clone());
        Ok(Self {
            worker: Box::new(FastH3Worker::new(actor, stages)),
            host: cuda.host.clone(),
            resolver,
        })
    }
    pub fn host_account(&self) -> &HostMemoryAccount {
        &self.host
    }
    pub fn output_spec(
        &self,
        parameters: &VideoGenerationParameters,
    ) -> anyhow::Result<VideoAudioSpec> {
        self.resolver.resolve(parameters)
    }
    pub fn output_spec_resolver(&self) -> FastH3OutputSpecResolver {
        self.resolver.clone()
    }
    pub fn into_service(self, config: VideoServiceConfig) -> Result<VideoService, ServiceError> {
        VideoService::start_worker(self.worker, config, self.host)
    }
    pub async fn run(
        self,
        config: VideoServiceConfig,
        request: VideoGenerationRequest,
        output: VideoOutputPlan,
    ) -> VideoServiceRun {
        VideoService::run_with_worker(self.worker, config, self.host, request, output).await
    }
}

struct Identities {
    next: u64,
    staging: usize,
}
impl Identities {
    fn request(&mut self) -> Result<RequestIdentity, VNextError> {
        let n = self.take()?;
        RequestIdentity::new(format!("request.fast_h3.{n}"))
    }
    fn take(&mut self) -> Result<u64, VNextError> {
        let n = self.next;
        self.next = self
            .next
            .checked_add(1)
            .ok_or_else(|| invalid("execution identity exhausted"))?;
        Ok(n)
    }
    fn initialization(
        &mut self,
        request: &RequestIdentity,
        name: &str,
    ) -> Result<ComponentInitializationConfig, VNextError> {
        let n = self.take()?;
        Ok(ComponentInitializationConfig {
            request_id: request.clone(),
            run_id: RunId::new(format!("run.fast_h3.{n}.{name}"))?,
            transaction_id: TransactionId::new(format!("transaction.fast_h3.{n}.{name}"))?,
            static_policy: StaticInitializationPolicy::new(self.staging as u64, 128)?,
            executor: TensorExecutorConfig::default(),
        })
    }
}
struct PreparedRequest {
    request: RequestIdentity,
    geometry: FastH3VideoGeometry,
    claims: Vec<(HostMemoryClass, usize)>,
    scope: Option<HostMemoryScope>,
    conditioning_key: ConditioningKey,
    cached_conditioning: Option<ConditionedRequest>,
    conditioning_plan: ConditioningPlan,
    decoder_tile_batch: SpatialTileBatch,
    decoder_postprocessing: super::decode::FastH3VideoPostprocessing,
    conditioning_retention: FastH3ConditioningReuse,
}
struct RequestContext {
    domain: Arc<Domain>,
    // Original execution credit stays with this independent request, never a
    // permanent batch slot. Payload owners retain spent credit after close.
    _scope: HostMemoryScope,
}
struct Domain {
    geometry: FastH3VideoGeometry,
    decoder_tile_batch: SpatialTileBatch,
    decoder_postprocessing: super::decode::FastH3VideoPostprocessing,
    references: Vec<FastH3Reference>,
    text_rows: usize,
    tags: Vec<FastH3TextTagRun>,
    layout: FastH3VsaLayout,
    denoising: PreparedDenoisingDomain,
    decoding: PreparedDecoding,
}
struct State {
    assets: Arc<FastH3Assets>,
    cuda: Arc<FastH3CudaComposition>,
    limits: FastH3PipelineLimits,
    schedule: FastH3ConditioningSchedule,
    tables: super::conditioning::ConditioningTables,
    identities: Identities,
    conditioner: PreparedConditioner,
    // One bounded cold metadata domain; active requests retain their original
    // Arc if the next prompt replaces it. No device weights or mutable session.
    last_domain: Option<Arc<Domain>>,
}

fn stages(
    state: Arc<Mutex<State>>,
    resolver: FastH3OutputSpecResolver,
) -> FastH3WorkerStages<Device, PreparedRequest, RequestContext> {
    let prepare_state = state.clone();
    let admit_state = state.clone();
    let condition_state = state.clone();
    let restore_state = state.clone();
    let batch_state = state.clone();
    let advance_state = state.clone();
    let preflight_state = state.clone();
    let residency_state = state.clone();
    FastH3WorkerStages {
        prepare: Box::new(move |request| {
            let mut state = lock(&prepare_state)?;
            if request
                .references
                .iter()
                .any(|v| !state.cuda.host.owns(v.bytes()))
            {
                return Err(rejected("FastH3 requires original Host-owned references"));
            }
            let geometry = resolver
                .geometry(&request.parameters)
                .map_err(rejected_error)?;
            let conditioning_plan = state
                .conditioner
                .plan(
                    request,
                    &state.cuda.host,
                    state.limits.request_input_bytes,
                    state.limits.request_output_floor_bytes,
                    state.limits.request_weight_bytes,
                )
                .map_err(rejected_error)?;
            // Original decoder buffers are BF16. Bound their actual spatial
            // and temporal ownership phases, with a separate conditioning floor.
            // This is a physical-backing admission bound, not a measured peak.
            let (tile_batch, reuse, claims) = super::decode::select_host_tile_group(
                decode::spatial_groups(&state.assets, &geometry, state.limits.decoding())
                    .map_err(rejected_error)?,
                state.limits.conditioning.reuse,
                &state.cuda.host,
                |tile_batch, reuse| {
                    let mut decoding = state.limits.decoding();
                    decoding.tile_batch = tile_batch;
                    let decoder_output = decode::output_host_bytes(
                        &state.assets,
                        &geometry,
                        &state.cuda.host,
                        decoding,
                    )?;
                    let output =
                        reuse.output_claim(decoder_output, conditioning_plan.output_bytes)?;
                    Ok(vec![
                        (
                            HostMemoryClass::InputPreprocess,
                            conditioning_plan.input_bytes,
                        ),
                        (
                            HostMemoryClass::WeightBacking,
                            conditioning_plan.weight_bytes,
                        ),
                        (
                            HostMemoryClass::TransferStaging,
                            state.limits.request_transfer_bytes,
                        ),
                        (HostMemoryClass::Output, output),
                    ])
                },
            )
            .map_err(rejected_error)?;
            Ok(PreparedRequest {
                request: state.identities.request().map_err(rejected_error)?,
                geometry,
                claims,
                scope: None,
                conditioning_key: ConditioningKey::new(request),
                cached_conditioning: None,
                conditioning_plan,
                decoder_tile_batch: tile_batch,
                decoder_postprocessing: state.limits.video_postprocessing,
                conditioning_retention: reuse,
            })
        }),
        admit: Box::new(move |prepared, _resources| {
            if prepared.scope.is_some() {
                return Ok(true);
            }
            let mut state = lock(&admit_state)?;
            let mut admission = state.cuda.host.reserve_scope(&prepared.claims);
            if admission.as_ref().is_err_and(|e| e.is_temporary_capacity())
                && state.conditioner.evict()
            {
                admission = state.cuda.host.reserve_scope(&prepared.claims);
            }
            match admission {
                Ok(scope) => {
                    prepared.scope = Some(scope);
                    prepared.cached_conditioning =
                        state.conditioner.reuse(prepared.conditioning_key);
                    Ok(true)
                }
                Err(e) if e.is_temporary_capacity() => Ok(false),
                Err(e) => Err(rejected_error(e)),
            }
        }),
        needs_auxiliary_model: Box::new(|prepared| prepared.cached_conditioning.is_none()),
        condition: Box::new(move |prepared, request, cancelled, actor| {
            let mut state = lock(&condition_state)?;
            let scope = prepared
                .scope
                .ok_or_else(|| rejected("conditioning has no admitted request scope"))?;
            let host = scope.account();
            if cancelled.load(Ordering::Acquire) {
                return Ok(ConditioningOutcome::Cancelled);
            }
            let State {
                conditioner,
                identities,
                ..
            } = &mut *state;
            tracing::debug!(conditioning_cache_hit = prepared.cached_conditioning.is_some(),
                request = %prepared.request, "FastH3 conditioning selection");
            let conditioned = match prepared.cached_conditioning {
                Some(value) => Some(value),
                None => conditioner
                    .execute(
                        request,
                        prepared.conditioning_plan,
                        &host,
                        actor,
                        cancelled,
                        |name| identities.initialization(&prepared.request, name),
                    )
                    .map_err(fatal)?,
            };
            let Some(conditioned) = conditioned else {
                return Ok(ConditioningOutcome::Cancelled);
            };
            conditioner.retain(
                prepared.conditioning_key,
                &conditioned,
                prepared.conditioning_retention,
            );
            let text_rows = usize::try_from(conditioned.text.tensor().dimensions()[0])
                .map_err(rejected_error)?;
            let references = conditioned
                .references
                .iter()
                .map(|reference| reference.geometry().map_err(rejected_error))
                .collect::<Result<Vec<_>, BatchError>>()?;
            let matches = state.last_domain.as_ref().is_some_and(|d| {
                d.geometry == prepared.geometry
                    && d.decoder_tile_batch == prepared.decoder_tile_batch
                    && d.decoder_postprocessing == prepared.decoder_postprocessing
                    && d.references == references
                    && d.text_rows == text_rows
                    && d.tags.len() == conditioned.tags.len()
                    && d.tags
                        .iter()
                        .zip(&conditioned.tags)
                        .all(|(a, b)| a.rows == b.rows && a.tag == b.tag)
            });
            let domain = if matches {
                state.last_domain.as_ref().unwrap().clone()
            } else {
                let layout = FastH3VsaLayout::ref2va(
                    &state.assets.checkpoint,
                    text_rows,
                    &references,
                    &prepared.geometry,
                    FastH3VsaLayoutLimits {
                        maximum_packed_rows: state.limits.maximum_rows,
                        maximum_tiles: state.limits.maximum_tiles,
                        maximum_metadata_bytes: state.limits.maximum_metadata_bytes,
                    },
                )
                .map_err(rejected_error)?;
                let denoising = PreparedDenoisingDomain::prepare(
                    state.assets.clone(),
                    state.cuda.clone(),
                    &layout,
                    &state.schedule,
                    &conditioned.tags,
                    state.limits.denoising(&state.assets.transformer),
                    state.limits.final_output(),
                    state.limits.prediction_handoff,
                    state.limits.feed_forward_chunk_rows,
                    state.limits.feed_forward_partition,
                )
                .map_err(rejected_error)?;
                let registry = state.cuda.registry(None).map_err(rejected_error)?;
                let catalog = state.cuda.catalog(&registry).map_err(rejected_error)?;
                let mut decoding_limits = state.limits.decoding();
                decoding_limits.tile_batch = prepared.decoder_tile_batch;
                decoding_limits.postprocessing = prepared.decoder_postprocessing;
                let decoding = PreparedDecoding::prepare(
                    state.assets.clone(),
                    state.cuda.clone(),
                    &prepared.geometry,
                    decoding_limits,
                    &state.cuda.host,
                    &registry,
                    &catalog,
                )
                .map_err(rejected_error)?;
                let domain = Arc::new(Domain {
                    geometry: prepared.geometry,
                    decoder_tile_batch: prepared.decoder_tile_batch,
                    decoder_postprocessing: prepared.decoder_postprocessing,
                    references,
                    text_rows,
                    tags: conditioned.tags,
                    layout,
                    denoising,
                    decoding,
                });
                state.last_domain = Some(domain.clone());
                domain
            };
            let reference_latents = conditioned
                .references
                .iter()
                .map(|reference| reference.latents())
                .collect::<Vec<_>>();
            let [video, audio] = prepare_fast_h3_reference_media(
                request.parameters.seed,
                &domain.geometry,
                domain.layout.packed(),
                &reference_latents,
                [
                    ProgramValueId::new(FastH3DenoisingFamily::VIDEO_INPUT_ID)
                        .map_err(rejected_error)?,
                    ProgramValueId::new(FastH3DenoisingFamily::AUDIO_INPUT_ID)
                        .map_err(rejected_error)?,
                ],
                &host,
            )
            .map_err(rejected_error)?;
            let positions = FastH3PositionPlan::ref2va(
                text_rows,
                &domain.references,
                &domain.geometry,
                FastH3PositionLimits {
                    max_sequence_rows: state.limits.maximum_rows,
                    max_regions: 32,
                },
            )
            .map_err(rejected_error)?;
            let (cosine, sine) = positions
                .managed_bf16_frequencies(
                    state.assets.transformer.config(),
                    &host,
                    ProgramValueId::new(FastH3DenoisingFamily::COSINE_ID)
                        .map_err(rejected_error)?,
                    ProgramValueId::new(FastH3DenoisingFamily::SINE_ID).map_err(rejected_error)?,
                )
                .map_err(rejected_error)?;
            let rotary = positions
                .single_request_bf16_rotary_inputs(
                    state.assets.transformer.config(),
                    [cosine, sine],
                )
                .map_err(rejected_error)?;
            let continuation = FastH3DenoisingRequest::initialize_from_plan(
                domain.denoising.request_plan(),
                prepared.request,
                state.assets.pdd.clone(),
                domain.layout.packed(),
                host.clone(),
                &video,
                &audio,
                &conditioned.text,
                rotary,
                state.tables.layers.clone(),
                state.tables.output.clone(),
            )
            .map_err(rejected_error)?;
            let later_input_bytes = domain
                .denoising
                .joined_input_bytes(1)
                .map_err(rejected_error)?
                .max(
                    domain
                        .decoding
                        .additional_input_host_bytes()
                        .map_err(rejected_error)?,
                );
            // The continuation copied its independent solver state and text.
            // Drop original conditioning temporaries before capping their free
            // credit, otherwise their later refunds would restore the old peak.
            drop(reference_latents);
            drop((video, audio, conditioned.references, conditioned.text));
            super::decode::release_conditioning_input_credit(
                &state.cuda.host,
                &scope,
                &prepared.claims,
                later_input_bytes,
            )
            .map_err(rejected_error)?;
            tracing::debug!(later_input_bytes, usage=?state.cuda.host.usage(),
                "FastH3 retired conditioning input credit at the completed phase");
            Ok(ConditioningOutcome::Ready(ReadyFastH3Request {
                continuation,
                parameters: request.parameters.clone(),
                host,
                context: RequestContext {
                    domain,
                    _scope: scope,
                },
            }))
        }),
        batch_capacity: Box::new(|ready, count, weights| {
            ready
                .context
                .domain
                .denoising
                .check_batch_capacity(count, weights)
        }),
        extend_stage: Box::new({
            let state = preflight_state.clone();
            move |stage, ready, count| {
                if stage.prepared_batch_counts().any(|n| n == count) {
                    return Ok(());
                }
                let base = lock(&state)?
                    .identities
                    .initialization(ready.continuation.request(), "attach_batch")
                    .map_err(rejected_error)?;
                ready
                    .context
                    .domain
                    .denoising
                    .extend_stage(stage, count, &base)
                    .map_err(fatal)
            }
        }),
        create_stage: Box::new(|ready| {
            ready
                .context
                .domain
                .denoising
                .new_stage()
                .map_err(rejected_error)
        }),
        restore: Box::new(move |stage, ready| {
            let base = lock(&restore_state)?
                .identities
                .initialization(ready.continuation.request(), "restore")
                .map_err(rejected_error)?;
            ready
                .context
                .domain
                .denoising
                .restore(stage, &base)
                .map_err(fatal)
        }),
        batch_scope: Box::new(move |members, scope| {
            let state = lock(&batch_state)?;
            let first = members
                .first()
                .ok_or_else(|| rejected("empty FastH3 batch"))?;
            let domain = &first.context.domain.denoising;
            let claims =
                batch_host_claims(domain, members.len(), state.limits.request_transfer_bytes)?;
            tracing::debug!(batch=members.len(),?claims,usage=?state.cuda.host.usage(),"FastH3 batch Host admission");
            if let Some(old) = scope.take() {
                match state.cuda.host.exchange_scope(old, &claims) {
                    Ok(next) => {
                        *scope = Some(next);
                        Ok(true)
                    }
                    Err(failure) => {
                        let (error, original) = failure.into_parts();
                        *scope = Some(original);
                        if error.is_temporary_capacity() {
                            Err(BatchError::AwaitHostCapacity {
                                account: state.cuda.host.clone(),
                                pressure: error,
                            })
                        } else {
                            Err(rejected_error(error))
                        }
                    }
                }
            } else {
                match state.cuda.host.reserve_scope(&claims) {
                    Ok(next) => {
                        *scope = Some(next);
                        Ok(true)
                    }
                    Err(pressure) if pressure.is_temporary_capacity() => {
                        Err(BatchError::AwaitHostCapacity {
                            account: state.cuda.host.clone(),
                            pressure,
                        })
                    }
                    Err(e) => Err(rejected_error(e)),
                }
            }
        }),
        preflight: Box::new(move |stage, actor, scope, members| {
            let first = members
                .first()
                .ok_or_else(|| rejected("empty candidate batch"))?;
            let domain = first.context.domain.clone();
            let request = first.continuation.request().clone();
            let base = lock(&preflight_state)?
                .identities
                .initialization(&request, "dynamic_prepare")
                .map_err(rejected_error)?;
            let prepared = stage
                .try_prepare_batch_on_actor(
                    actor,
                    &scope.account(),
                    base.run_id,
                    base.request_id,
                    members.iter_mut().map(|m| &mut m.continuation).collect(),
                )
                .map_err(fatal)?;
            if prepared {
                // Original rollback credit is retained until actual device
                // admission succeeded. Publication below is now infallible;
                // give up only unused credit beyond this admitted batch.
                let state = lock(&preflight_state)?;
                let claims = batch_host_claims(
                    &domain.denoising,
                    members.len(),
                    state.limits.request_transfer_bytes,
                )?;
                state
                    .cuda
                    .host
                    .shrink_scope_unused_to_at_most(scope, &claims)
                    .map_err(fatal)?;
            }
            Ok(prepared)
        }),
        advance: Box::new(move |stage, actor, host, members| {
            let first = members
                .first()
                .ok_or_else(|| rejected("empty FastH3 advance"))?;
            let domain = first.context.domain.clone();
            let base = lock(&advance_state)?
                .identities
                .initialization(first.continuation.request(), "denoise")
                .map_err(rejected_error)?;
            domain
                .denoising
                .advance(
                    stage,
                    actor,
                    host,
                    base,
                    members.iter_mut().map(|m| &mut m.continuation).collect(),
                )
                .map_err(fatal)
        }),
        can_retain_for_decode: Box::new(move |ready| {
            let state = lock(&residency_state)?;
            if state.limits.denoiser_residency
                == super::denoising::DenoisingResidencyPolicy::ReleaseBeforeDecode
            {
                return Ok(false);
            }
            drop(state);
            ready
                .context
                .domain
                .decoding
                .fits_with_current_residency()
                .map_err(rejected_error)
        }),
        decode: Box::new(move |ready, cancelled, actor| {
            let mut state = lock(&state)?;
            let video = state
                .identities
                .initialization(ready.continuation.request(), "video_decode")
                .map_err(rejected_error)?;
            let audio = state
                .identities
                .initialization(ready.continuation.request(), "audio_decode")
                .map_err(rejected_error)?;
            let run = RunId::new(format!("{}.decode", video.run_id)).map_err(rejected_error)?;
            drop(state);
            ready
                .context
                .domain
                .decoding
                .execute(
                    &ready.continuation,
                    ready.context.domain.layout.packed(),
                    &ready.host,
                    actor,
                    cancelled,
                    DecodingInvocation {
                        video_initialization: video,
                        audio_initialization: audio,
                        run,
                    },
                )
                .map_err(fatal)
        }),
    }
}
fn batch_host_claims(
    domain: &PreparedDenoisingDomain,
    count: usize,
    transfer: usize,
) -> Result<[(HostMemoryClass, usize); 3], BatchError> {
    Ok([
        (
            HostMemoryClass::InputPreprocess,
            domain.joined_input_bytes(count).map_err(rejected_error)?,
        ),
        (
            HostMemoryClass::Output,
            domain.output_bytes(count).map_err(rejected_error)?,
        ),
        (HostMemoryClass::TransferStaging, transfer),
    ])
}
fn lock<T>(value: &Mutex<T>) -> Result<std::sync::MutexGuard<'_, T>, BatchError> {
    value.lock().map_err(|_| {
        fatal(std::io::Error::other(
            "FastH3 composition authority poisoned",
        ))
    })
}
fn rejected(message: &str) -> BatchError {
    BatchError::Rejected(message.into())
}
fn rejected_error(e: impl fmt::Display) -> BatchError {
    BatchError::Rejected(format!("{e:#}"))
}
fn fatal(e: impl std::error::Error + Send + 'static) -> BatchError {
    BatchError::Fatal(Box::new(e))
}
fn invalid(message: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: message.into(),
    }
}
