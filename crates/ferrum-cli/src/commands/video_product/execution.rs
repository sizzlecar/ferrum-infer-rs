use super::{configuration::Configuration, VideoProductError, VideoReferencePath, VideoRunRequest};
use anyhow::{ensure, Context, Result};
use ferrum_engine::video_generation::{
    fast_h3::{
        assets::FastH3Assets,
        composition::{FastH3Pipeline, FastH3PipelineLimits},
    },
    VideoGenerationRequest, VideoOutputPlan, VideoReference, VideoReferenceKind, VideoServiceRun,
};
use ferrum_models::qwen_image21::image_input::ImageInputLimits;
use ferrum_runtime::{HostMemoryAccount, HostMemoryClass};
use std::{
    fs::{File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::Arc,
};

fn pipeline(config: &Configuration) -> Result<FastH3Pipeline, VideoProductError> {
    let mut limits = FastH3PipelineLimits::rtx5090();
    limits.conditioning.reuse = config.conditioning_reuse;
    limits.video_decoder_tile_batch = config.video_decoder_tile_batch;
    limits.video_postprocessing = config.video_postprocessing;
    limits.denoiser_residency = config.denoiser_residency;
    limits.prediction_handoff = config.prediction_handoff;
    limits.feed_forward_chunk_rows = config.feed_forward_chunk_rows;
    limits.feed_forward_partition = config.feed_forward_partition;
    limits.maximum_plan_nodes = config.maximum_plan_nodes.get();
    let assets = Arc::new(FastH3Assets::load(
        &config.source.paths(),
        limits.asset_limits(),
    )?);
    FastH3Pipeline::new(
        assets,
        config.cuda.clone(),
        limits,
        config.input.prepare(&config.output)?,
    )
    .map_err(|error| VideoProductError::retain_error("FastH3 startup", error))
}

fn reference_files(paths: &[VideoReferencePath]) -> Result<Vec<(VideoReferenceKind, File, usize)>> {
    ferrum_engine::video_generation::fast_h3::REFERENCE_LIMITS
        .validate_kinds(paths.iter().map(|p| p.kind))?;
    let mut files = Vec::with_capacity(paths.len());
    let mut total = 0usize;
    for path in paths {
        let file = File::open(&path.path).context("open video media reference")?;
        let metadata = file.metadata()?;
        let length =
            usize::try_from(metadata.len()).context("reference length exceeds address space")?;
        ensure!(
            metadata.is_file()
                && (1..=ImageInputLimits::DEFAULT_MAXIMUM_ENCODED_BYTES).contains(&length),
            "video reference must be a nonempty regular file within the ingress byte limit"
        );
        total = total
            .checked_add(length)
            .context("reference byte total overflow")?;
        ensure!(
            total <= ImageInputLimits::DEFAULT_MAXIMUM_ENCODED_BYTES,
            "ordered video references exceed the total ingress byte limit"
        );
        files.push((path.kind, file, length));
    }
    Ok(files)
}

fn read_reference(
    host: &HostMemoryAccount,
    kind: VideoReferenceKind,
    mut file: File,
    length: usize,
) -> Result<VideoReference> {
    let mut bytes = host.allocate_zeroed(HostMemoryClass::InputPreprocess, length)?;
    file.read_exact(bytes.as_mut())?;
    let mut extra = [0u8];
    ensure!(
        file.read(&mut extra)? == 0,
        "video reference changed length during read"
    );
    let bytes = bytes.freeze();
    match kind {
        VideoReferenceKind::Image => {
            ferrum_models::qwen_image21::image_input::ManagedRgba8Image::decode_png(
                host,
                &bytes,
                ImageInputLimits::default(),
            )?;
        }
        VideoReferenceKind::Audio => {
            ferrum_models::audio_input::WaveAudioPlan::parse(
                bytes.clone(),
                ferrum_models::audio_input::AudioInputLimits::default(),
            )?;
        }
        // The shared conditioner probes video metadata once while planning;
        // decoded frames are allocated only after request Host admission.
        VideoReferenceKind::Video => {}
    }
    Ok(VideoReference::from_encoded(kind, bytes))
}

pub(super) async fn run(request: VideoRunRequest) -> Result<(), VideoProductError> {
    let config = Configuration::load(&request.config)?;
    let service_config = config.serving.service_config()?;
    let output = config.output.prepare()?;
    let files = reference_files(&request.references)?;
    ensure_destination(&request.destination)?;
    let pipeline = pipeline(&config)?;
    let spec = pipeline.output_spec(&request.parameters)?;
    let plan = VideoOutputPlan::new(spec, output.encoder, output.spool, &output.budget)
        .map_err(anyhow::Error::from)?;
    let references = files
        .into_iter()
        .map(|(kind, file, length)| read_reference(pipeline.host_account(), kind, file, length))
        .collect::<Result<Vec<_>>>()?;
    let started = std::time::Instant::now();
    let service = pipeline
        .into_service(service_config)
        .map_err(anyhow::Error::from)?;
    let job = service.try_submit(
        VideoGenerationRequest {
            parameters: request.parameters,
            references,
        },
        plan,
    );
    let result = match job {
        Ok(job) => {
            let cancellation = job.cancellation_handle();
            let result = job.wait();
            tokio::pin!(result);
            tokio::select! {
                result = &mut result => result,
                _ = crate::commands::serve::serve_shutdown_signal() => {
                    cancellation.cancel();
                    // Keep observing the original job through native fences
                    // and codec kill/reap, even after a terminal signal.
                    result.await
                }
            }
        }
        Err(error) => Err(ferrum_engine::video_generation::VideoJobError::Generation(
            error,
        )),
    };
    let shutdown = match service.shutdown().await {
        Ok(shutdown) => shutdown,
        Err(error) => {
            return Err(VideoProductError::retain(
                "video worker shutdown failed",
                (error, result, service),
            ))
        }
    };
    let run = VideoServiceRun { result, shutdown };
    if run.shutdown.has_failure() {
        return Err(VideoProductError::retain(
            "video worker retirement failed",
            run,
        ));
    }
    let result = match run.result {
        Ok(result) => result,
        Err(error) => return Err(VideoProductError::retain_error("video generation", error)),
    };
    let receipt = result.artifact.receipt();
    let mut reader = result
        .artifact
        .open()
        .map_err(|e| VideoProductError::retain_error("open completed video", e))?;
    publish(&mut reader, &request.destination)?;
    println!(
        "{}",
        serde_json::json!({
            "model": config.model,
            "request_id": result.correlation.request_id,
            "seed": result.seed,
            "output": request.destination,
            "width": spec.width, "height": spec.height, "frames": spec.frames,
            "fps": {"numerator": spec.fps_numerator, "denominator": spec.fps_denominator},
            "audio_sample_rate": spec.audio_sample_rate,
            "audio_sample_frames": spec.audio_sample_frames,
            "container_profile": receipt.container_profile,
            "request_seconds": started.elapsed().as_secs_f64(),
        })
    );
    Ok(())
}

fn ensure_destination(path: &Path) -> Result<()> {
    ensure!(!path.try_exists()?, "video destination already exists");
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    ensure!(
        parent.is_dir(),
        "video destination parent is not a directory"
    );
    Ok(())
}

/// Same-directory publication keeps partial bytes out of the destination and
/// never replaces a file created by another process after the preflight check.
fn publish(reader: &mut impl Read, destination: &Path) -> Result<()> {
    let parent = destination
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let temporary = parent.join(format!(".ferrum-video-{}.partial", uuid::Uuid::new_v4()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    struct Cleanup(PathBuf);
    impl Drop for Cleanup {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let _cleanup = Cleanup(temporary.clone());
    std::io::copy(reader, &mut file)?;
    file.flush()?;
    drop(file);
    std::fs::hard_link(&temporary, destination)
        .context("publish video without replacing an existing destination")?;
    Ok(())
}

pub(super) async fn serve(
    path: PathBuf,
    host: String,
    port: u16,
    aliases: Vec<String>,
    api_key: Option<ferrum_server::ApiKey>,
) -> Result<(), VideoProductError> {
    use ferrum_server::{
        axum_server::AppState, AxumServer, BoundHttpListener, ServerConfig, VideoHttpConfig,
        VideoHttpService, VideoOutputPlanResolver, VideoPlanError,
    };
    if let Some(key) = &api_key {
        key.validate().map_err(anyhow::Error::from)?;
    }
    let config = Configuration::load(&path)?;
    ensure_aliases(&aliases)?;
    let model = aliases
        .into_iter()
        .next()
        .unwrap_or_else(|| config.model.clone());
    ferrum_server::ServedModelRegistry::try_new(
        model.clone(),
        ferrum_server::ServedModelKind::VideoGeneration,
        vec![model.clone()],
        vec![],
    )
    .map_err(anyhow::Error::from)?;
    let service_config = config.serving.service_config()?;
    let output = config.output.prepare()?;
    let maximum_body_bytes = ImageInputLimits::DEFAULT_MAXIMUM_ENCODED_BYTES
        .checked_mul(4)
        .and_then(|n| n.checked_div(3))
        .and_then(|n| n.checked_add(config.serving.maximum_prompt_bytes))
        .and_then(|n| n.checked_add(64 << 10))
        .context("video body limit overflow")?;
    let listener = BoundHttpListener::bind(&host, port)
        .await
        .map_err(anyhow::Error::from)?;
    let pipeline = pipeline(&config)?;
    let geometry = pipeline.output_spec_resolver();
    let resolve: VideoOutputPlanResolver = Arc::new(move |parameters| {
        let spec = geometry
            .resolve(parameters)
            .map_err(|e| VideoPlanError::Invalid(e.to_string()))?;
        Ok(VideoOutputPlan::new(
            spec,
            output.encoder.clone(),
            output.spool.clone(),
            &output.budget,
        )?)
    });
    let service = Arc::new(
        pipeline
            .into_service(service_config)
            .map_err(anyhow::Error::from)?,
    );
    let videos = match VideoHttpService::new(
        model,
        service.clone(),
        VideoHttpConfig {
            maximum_jobs: config.output.maximum_jobs,
            maximum_body_bytes,
            maximum_prompt_bytes: config.serving.maximum_prompt_bytes,
            references: ferrum_engine::video_generation::fast_h3::REFERENCE_LIMITS,
            image_limits: ImageInputLimits::default(),
            audio_limits: ferrum_models::audio_input::AudioInputLimits::default(),
            maximum_readers: 4,
            read_buffer_bytes: 64 << 10,
        },
        resolve,
    ) {
        Ok(videos) => videos,
        Err(error) => {
            let shutdown = service.shutdown().await;
            return Err(VideoProductError::retain(
                "video HTTP initialization failed",
                (error, service, shutdown),
            ));
        }
    };
    let server = Arc::new(AxumServer::from_state(
        AppState::default().with_videos(videos.clone()),
    ));
    let server_config = ServerConfig {
        api_key,
        host,
        port,
        ..Default::default()
    };
    tracing::info!(scheduler=?service_config.service(), model=videos.model_name(),
        "FastH3 video service uses the shared iterative worker");
    let mut task = {
        let server = server.clone();
        tokio::spawn(async move {
            server
                .start_with_listener(&server_config, listener, |_| {})
                .await
        })
    };
    let joined = tokio::select! {
        joined = &mut task => Some(joined),
        _ = crate::commands::serve::serve_shutdown_signal() => None,
    };
    let stop = server.finish_stop().await;
    let joined = match joined {
        Some(joined) => joined,
        None => task.await,
    };
    let shutdown = videos.take_shutdown().await;
    let start = joined
        .map_err(anyhow::Error::from)
        .and_then(|r| r.map_err(anyhow::Error::from));
    if start.is_err() || stop.is_err() || shutdown.as_ref().is_some_and(|s| s.has_failure()) {
        return Err(VideoProductError::retain(
            "video HTTP or worker shutdown failed",
            (start, stop, videos, shutdown),
        ));
    }
    Ok(())
}

fn ensure_aliases(aliases: &[String]) -> Result<()> {
    ensure!(
        aliases.len() <= 1 && aliases.first().is_none_or(|name| !name.trim().is_empty()),
        "video serve accepts at most one nonempty --served-model-name"
    );
    Ok(())
}
