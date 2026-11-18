use anyhow::{ensure, Context, Result};
use ferrum_engine::{
    iterative_service::ServiceConfig,
    video_generation::{
        fast_h3::{
            assets::FastH3AssetPaths,
            conditioning::FastH3ConditioningReuse,
            cuda::FastH3CudaConfig,
            denoising::{
                DenoisingResidencyPolicy, FeedForwardPartitionPolicy, PredictionHandoffPolicy,
            },
        },
        VideoRequestLimits, VideoServiceConfig,
    },
};
use ferrum_models::qwen_image21::image_input::ImageInputLimits;
use ferrum_models::spatial_tiles::SpatialTileBatch;
use ferrum_output::{
    mp4::{H264AacFragmentedV1, H264SpeedPreset, Mp4EncoderConfig, Mp4Spool, Mp4SpoolLimits},
    OutputBudget,
};
use ferrum_types::{IterativeBatchPolicy, OutputConfig};
use serde::Deserialize;
use std::{
    fs::File,
    io::Read,
    path::{Path, PathBuf},
    time::Duration,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Configuration {
    pub model: String,
    pub source: Source,
    pub cuda: FastH3CudaConfig,
    #[serde(default)]
    pub conditioning_reuse: FastH3ConditioningReuse,
    #[serde(default)]
    pub video_decoder_tile_batch: SpatialTileBatch,
    #[serde(default)]
    pub video_postprocessing:
        ferrum_engine::video_generation::fast_h3::decode::FastH3VideoPostprocessing,
    #[serde(default)]
    pub denoiser_residency: DenoisingResidencyPolicy,
    #[serde(default)]
    pub prediction_handoff: PredictionHandoffPolicy,
    #[serde(default = "default_feed_forward_chunk_rows")]
    pub feed_forward_chunk_rows: std::num::NonZeroU64,
    #[serde(default)]
    pub feed_forward_partition: FeedForwardPartitionPolicy,
    #[serde(default = "default_maximum_plan_nodes")]
    pub maximum_plan_nodes: std::num::NonZeroUsize,
    #[serde(default)]
    pub serving: Serving,
    pub output: Output,
    #[serde(default)]
    pub input: Input,
}

/// Input decoding inherits the output FFmpeg unless a separate build is selected.
#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct Input {
    ffmpeg: Option<PathBuf>,
    ffprobe: Option<PathBuf>,
    directory: Option<PathBuf>,
    threads: u32,
    timeout_seconds: u64,
}
impl Default for Input {
    fn default() -> Self {
        Self {
            ffmpeg: None,
            ffprobe: None,
            directory: None,
            threads: 2,
            timeout_seconds: 120,
        }
    }
}
impl Input {
    pub(super) fn prepare(
        &self,
        output: &Output,
        output_ffmpeg: &Path,
    ) -> Result<ferrum_models::video_input::VideoInputCodec> {
        let (ffmpeg, ffprobe) = super::media::reference_tools(
            output_ffmpeg,
            self.ffmpeg.as_deref(),
            self.ffprobe.as_deref(),
            std::env::var_os("PATH").as_deref(),
        )?;
        ferrum_models::video_input::VideoInputCodec::new(
            ferrum_models::video_input::VideoInputCodecConfig {
                ffmpeg,
                ffprobe,
                staging_directory: self
                    .directory
                    .clone()
                    .unwrap_or_else(|| output.directory.join("reference-input")),
                threads: self.threads,
                timeout: Duration::from_secs(self.timeout_seconds),
                maximum_duration: Duration::from_secs(16),
                maximum_encoded_bytes: ImageInputLimits::DEFAULT_MAXIMUM_ENCODED_BYTES,
                maximum_metadata_bytes: 64 << 10,
                maximum_stderr_bytes: 16 << 10,
                maximum_source_frames: 8192,
                image: ImageInputLimits::default(),
            },
        )
    }
}
fn default_feed_forward_chunk_rows() -> std::num::NonZeroU64 {
    std::num::NonZeroU64::new(4096).unwrap()
}
fn default_maximum_plan_nodes() -> std::num::NonZeroUsize {
    std::num::NonZeroUsize::new(16384).unwrap()
}
impl Configuration {
    pub fn load(path: &Path) -> Result<Self> {
        let mut data = Vec::new();
        File::open(path)?
            .take(1024 * 1024 + 1)
            .read_to_end(&mut data)?;
        ensure!(
            data.len() <= 1024 * 1024,
            "video configuration exceeds 1 MiB"
        );
        let mut value: Self = serde_json::from_slice(&data).context("video product JSON")?;
        ensure!(
            !value.model.trim().is_empty(),
            "video model name must be nonempty"
        );
        value
            .cuda
            .resolve_cudnn(|| crate::commands::cudnn_library::resolve(None))?;
        value.cuda.validate()?;
        value.conditioning_reuse.validate()?;
        value.serving.service_config()?;
        ensure!(
            value.output.directory.is_absolute(),
            "video output directory must be an absolute path"
        );
        ensure!(
            value.output.maximum_jobs > 0 && value.output.maximum_jobs <= u32::MAX as usize / 3,
            "video maximum_jobs exceeds the output file count bound"
        );
        ensure!(
            value.output.maximum_retained_jobs > 0
                && value.output.maximum_retained_jobs <= u32::MAX as usize
                && value.output.maximum_retained_bytes >= value.output.maximum_encoded_bytes
                && value.output.retention_seconds > 0
                && value.output.retention_seconds <= i64::MAX as u64,
            "video retention requires positive job/TTL limits and enough bytes for one maximum encoded output"
        );
        Ok(value)
    }
}

#[derive(Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum Source {
    ModelDirectory { path: PathBuf },
    Files { assets: FastH3AssetPaths },
}
impl Source {
    pub fn paths(&self) -> FastH3AssetPaths {
        match self {
            Self::ModelDirectory { path } => FastH3AssetPaths::from_model_directory(path),
            Self::Files { assets } => assets.clone(),
        }
    }
}

#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
pub(super) struct Serving {
    pub batching: IterativeBatchPolicy,
    pub maximum_pending: usize,
    pub maximum_pending_bytes: usize,
    pub max_num_seqs: usize,
    pub request_batch_max_wait_ms: u64,
    pub maximum_prompt_bytes: usize,
    pub upload_idle_timeout_seconds: u64,
    pub upload_timeout_seconds: u64,
}
impl Default for Serving {
    fn default() -> Self {
        Self {
            batching: IterativeBatchPolicy::Auto,
            maximum_pending: 8,
            maximum_pending_bytes: 256 << 20,
            max_num_seqs: 2,
            request_batch_max_wait_ms: 0,
            maximum_prompt_bytes: 64 << 10,
            upload_idle_timeout_seconds: 30,
            upload_timeout_seconds: 300,
        }
    }
}
impl Serving {
    pub fn service_config(&self) -> Result<VideoServiceConfig> {
        ensure!(
            self.upload_idle_timeout_seconds > 0 && self.upload_timeout_seconds > 0,
            "video upload deadlines must be positive"
        );
        ensure!(
            (1..=2).contains(&self.max_num_seqs),
            "this FastH3 preset supports one or two active denoising requests"
        );
        ensure!(
            self.maximum_prompt_bytes <= 64 << 10,
            "FastH3 prompt capacity exceeds the declared conditioning preset"
        );
        Ok(VideoServiceConfig::new(
            ServiceConfig {
                batching: self.batching,
                maximum_pending: self.maximum_pending,
                maximum_pending_bytes: self.maximum_pending_bytes,
                max_num_seqs: self.max_num_seqs,
                request_batch_max_wait_ms: self.request_batch_max_wait_ms,
            },
            VideoRequestLimits {
                maximum_prompt_bytes: self.maximum_prompt_bytes,
                references: ferrum_engine::video_generation::fast_h3::REFERENCE_LIMITS,
                maximum_encoded_reference_bytes: ImageInputLimits::DEFAULT_MAXIMUM_ENCODED_BYTES,
            },
        )?)
    }
}

#[derive(Clone, Copy, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum Speed {
    Ultrafast,
    #[default]
    Veryfast,
    Medium,
    Veryslow,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Output {
    ffmpeg: Option<PathBuf>,
    /// Parent for temporary encoder spools and the durable jobs-v1 directory.
    pub directory: PathBuf,
    #[serde(default = "default_jobs")]
    pub maximum_jobs: usize,
    #[serde(default = "default_retained_jobs")]
    pub maximum_retained_jobs: usize,
    #[serde(default = "default_retained_bytes")]
    pub maximum_retained_bytes: u64,
    #[serde(default = "default_retention_seconds")]
    pub retention_seconds: u64,
    #[serde(default = "default_spool_bytes")]
    maximum_spool_bytes: u64,
    #[serde(default = "default_encoded_bytes")]
    maximum_encoded_bytes: u64,
    #[serde(default = "default_threads")]
    threads: u32,
    #[serde(default = "default_timeout")]
    timeout_seconds: u64,
    #[serde(default = "default_crf")]
    crf: u8,
    #[serde(default)]
    speed: Speed,
}
fn default_jobs() -> usize {
    16
}
fn default_retained_jobs() -> usize {
    ferrum_server::VideoJobRetention::default().maximum_jobs
}
fn default_retained_bytes() -> u64 {
    ferrum_server::VideoJobRetention::default().maximum_bytes
}
fn default_retention_seconds() -> u64 {
    ferrum_server::VideoJobRetention::default().ttl.as_secs()
}
fn default_spool_bytes() -> u64 {
    2 << 30
}
fn default_encoded_bytes() -> u64 {
    128 << 20
}
fn default_threads() -> u32 {
    4
}
fn default_timeout() -> u64 {
    600
}
fn default_crf() -> u8 {
    18
}

pub(super) struct PreparedOutput {
    pub budget: OutputBudget,
    pub spool: Mp4Spool,
    pub encoder: Mp4EncoderConfig,
}
impl Output {
    pub fn prepare(&self) -> Result<PreparedOutput> {
        let ffmpeg = super::media::output_ffmpeg(
            self.ffmpeg.as_deref(),
            std::env::var_os("PATH").as_deref(),
        )?;
        let budget = OutputBudget::new(OutputConfig {
            max_sessions: self.maximum_jobs,
            engine_queue_items: self.maximum_jobs,
            ..OutputConfig::default()
        })?;
        std::fs::create_dir_all(&self.directory)?;
        let spool = Mp4Spool::create(
            self.directory.join(uuid::Uuid::new_v4().to_string()),
            Mp4SpoolLimits {
                maximum_bytes: self.maximum_spool_bytes,
                maximum_files: self
                    .maximum_jobs
                    .checked_mul(3)
                    .context("video spool file count overflow")?,
            },
        )
        .map_err(|e| anyhow::anyhow!("create video spool: {e}"))?;
        let encoder = Mp4EncoderConfig {
            ffmpeg,
            threads: self.threads,
            timeout: Duration::from_secs(self.timeout_seconds),
            maximum_encoded_bytes: self.maximum_encoded_bytes,
            io_buffer_bytes: 64 << 10,
            maximum_stderr_bytes: 64 << 10,
            preset: H264AacFragmentedV1 {
                h264_speed: match self.speed {
                    Speed::Ultrafast => H264SpeedPreset::Ultrafast,
                    Speed::Veryfast => H264SpeedPreset::Veryfast,
                    Speed::Medium => H264SpeedPreset::Medium,
                    Speed::Veryslow => H264SpeedPreset::Veryslow,
                },
                crf: self.crf,
                audio_bitrate: 192_000,
            },
        };
        // Validate codec bounds before opening model/runtime resources.
        // These constructors have not submitted work or started a codec child.
        ferrum_output::mp4::Mp4Encoder::new(encoder.clone(), spool.clone())
            .map_err(|e| anyhow::anyhow!("video codec configuration: {e}"))?;
        Ok(PreparedOutput {
            budget,
            spool,
            encoder,
        })
    }
}
