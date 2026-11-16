//! Bounded encoded-video input. Codec subprocesses are serialized per owner,
//! staged files live only until reaping, and decoded frames use the caller's
//! Host account. External codec RSS and OS pipe buffers are separate resources.
use crate::qwen_image21::image_input::{ImageInputLimits, ManagedRgba8Image};
use anyhow::{ensure, Context, Result};
use ferrum_runtime::*;
use serde::Deserialize;
use std::{
    ffi::OsString,
    io::Read,
    path::PathBuf,
    sync::{atomic::AtomicBool, Arc},
    time::Duration,
};

mod audio;
mod process;

#[derive(Debug, thiserror::Error)]
pub enum VideoInputInterruption {
    #[error("video input cancelled")]
    Cancelled,
    #[error("video input deadline exceeded")]
    Deadline,
}

#[derive(Clone, Debug)]
pub struct VideoInputCodecConfig {
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
    pub staging_directory: PathBuf,
    pub threads: u32,
    pub timeout: Duration,
    pub maximum_duration: Duration,
    pub maximum_encoded_bytes: usize,
    pub maximum_metadata_bytes: usize,
    pub maximum_stderr_bytes: usize,
    pub maximum_source_frames: usize,
    pub image: ImageInputLimits,
}
struct Inner {
    config: VideoInputCodecConfig,
    gate: parking_lot::Mutex<()>,
}
#[derive(Clone)]
pub struct VideoInputCodec(Arc<Inner>);
#[derive(Clone, Copy, Debug)]
pub struct VideoInputGeometry {
    pub width: u32,
    pub height: u32,
    pub frames: usize,
    pub frame_rate_numerator: u32,
    pub frame_rate_denominator: u32,
}
#[derive(Clone, Copy, Debug)]
pub struct VideoInputAudio {
    pub sample_rate: u32,
    pub channels: u32,
}
#[derive(Clone)]
pub struct VideoContainerPlan {
    codec: VideoInputCodec,
    encoded: ManagedHostBytes,
    geometry: VideoInputGeometry,
    audio: Option<VideoInputAudio>,
}
impl VideoInputCodec {
    pub fn new(config: VideoInputCodecConfig) -> Result<Self> {
        ensure!(
            config.ffmpeg.is_absolute()
                && config.ffprobe.is_absolute()
                && config.staging_directory.is_absolute()
                && config.threads > 0
                && config.threads <= 256
                && !config.timeout.is_zero()
                && !config.maximum_duration.is_zero()
                && config.maximum_encoded_bytes > 0
                && config.maximum_encoded_bytes <= isize::MAX as usize
                && config.maximum_metadata_bytes > 0
                && config.maximum_metadata_bytes <= 1 << 20
                && config.maximum_stderr_bytes > 0
                && config.maximum_stderr_bytes <= 1 << 20
                && config.maximum_source_frames > 0
                && config.maximum_source_frames <= 1 << 24,
            "invalid explicit video codec paths or capacities"
        );
        std::fs::create_dir_all(&config.staging_directory)?;
        Ok(Self(Arc::new(Inner {
            config,
            gate: parking_lot::Mutex::new(()),
        })))
    }
    pub fn config(&self) -> &VideoInputCodecConfig {
        &self.0.config
    }
    pub fn inspect(
        &self,
        encoded: ManagedHostBytes,
        cancelled: &AtomicBool,
    ) -> Result<VideoContainerPlan> {
        let config = self.config();
        let args: Vec<OsString> = ["-v", "error", "-count_frames", "-read_intervals"].into_iter().map(Into::into)
            .chain([format!("%+{}", config.maximum_duration.as_secs_f64()).into()])
            .chain(["-show_entries", "stream=codec_type,width,height,avg_frame_rate,r_frame_rate,nb_read_frames,sample_rate,channels:stream_side_data=rotation", "-of", "json"]
                .into_iter().map(Into::into)).collect();
        let raw = process::run(
            self,
            &config.ffprobe,
            &encoded,
            args,
            false,
            cancelled,
            |stdout| process::read_capped(stdout, config.maximum_metadata_bytes),
        )?;
        let metadata: Metadata = serde_json::from_slice(&raw).context("video codec metadata")?;
        let video = metadata
            .streams
            .iter()
            .find(|s| s.codec_type == "video")
            .context("reference contains no video stream")?;
        let (n, d) = [
            video.avg_frame_rate.as_deref(),
            video.r_frame_rate.as_deref(),
        ]
        .into_iter()
        .flatten()
        .find_map(rate)
        .context("video lacks a positive frame rate")?;
        let frames = video
            .nb_read_frames
            .as_deref()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&n| n > 0)
            .context("video contains no decoded frames")?
            .min(config.maximum_source_frames);
        let (mut width, mut height) = (
            video.width.context("missing video width")?,
            video.height.context("missing video height")?,
        );
        if let Some(rotation) = video.side_data_list.iter().find_map(|s| s.rotation) {
            ensure!(rotation.is_finite(), "nonfinite video display rotation");
            if ((rotation / 90.).round_ties_even() as i64).rem_euclid(2) == 1 {
                std::mem::swap(&mut width, &mut height);
            }
        }
        ensure!(
            width > 0
                && height > 0
                && width <= config.image.maximum_width()
                && height <= config.image.maximum_height()
                && u64::from(width) * u64::from(height) <= config.image.maximum_pixels() as u64,
            "decoded video frame exceeds declared pixel capacity"
        );
        let audio = metadata
            .streams
            .iter()
            .find(|s| s.codec_type == "audio")
            .map(|s| -> Result<_> {
                Ok(VideoInputAudio {
                    sample_rate: s
                        .sample_rate
                        .as_deref()
                        .and_then(|v| v.parse().ok())
                        .filter(|&n| n > 0)
                        .context("invalid soundtrack sample rate")?,
                    channels: s
                        .channels
                        .filter(|&n| n > 0)
                        .context("invalid soundtrack channels")?,
                })
            })
            .transpose()?;
        Ok(VideoContainerPlan {
            codec: self.clone(),
            encoded,
            geometry: VideoInputGeometry {
                width,
                height,
                frames,
                frame_rate_numerator: n,
                frame_rate_denominator: d,
            },
            audio,
        })
    }
}
impl VideoContainerPlan {
    pub fn geometry(&self) -> VideoInputGeometry {
        self.geometry
    }
    pub fn audio(&self) -> Option<VideoInputAudio> {
        self.audio
    }
    pub fn encoded(&self) -> &ManagedHostBytes {
        &self.encoded
    }
    /// The consumer selects original frames monotonically; skipped frames are
    /// drained through a fixed stack buffer, never retained as a full raw clip.
    pub fn with_frames<T>(
        &self,
        frames: usize,
        host: &HostMemoryAccount,
        cancelled: &AtomicBool,
        consume: impl FnOnce(&mut VideoFrameReader<'_>) -> Result<T>,
    ) -> Result<T> {
        ensure!(
            frames > 0 && frames <= self.geometry.frames && host.owns(&self.encoded),
            "video decode requires admitted source and bounded frame extent"
        );
        let args = vec![
            "-map".into(),
            "0:v:0".into(),
            "-an".into(),
            "-sn".into(),
            "-dn".into(),
            "-frames:v".into(),
            frames.to_string().into(),
            "-fps_mode".into(),
            "passthrough".into(),
            "-pix_fmt".into(),
            "rgba".into(),
            "-threads".into(),
            self.codec.config().threads.to_string().into(),
            "-f".into(),
            "rawvideo".into(),
            "pipe:1".into(),
        ];
        process::run(
            &self.codec,
            &self.codec.config().ffmpeg,
            &self.encoded,
            args,
            true,
            cancelled,
            |stdout| {
                let mut reader = VideoFrameReader {
                    stdout,
                    host,
                    geometry: self.geometry,
                    image: self.codec.config().image,
                    frames,
                    next: 0,
                };
                let result = consume(&mut reader)?;
                reader.finish()?;
                Ok(result)
            },
        )
    }
}
pub struct VideoFrameReader<'a> {
    stdout: &'a mut dyn Read,
    host: &'a HostMemoryAccount,
    geometry: VideoInputGeometry,
    image: ImageInputLimits,
    frames: usize,
    next: usize,
}
impl VideoFrameReader<'_> {
    fn frame_bytes(&self) -> Result<usize> {
        usize::try_from(
            u64::from(self.geometry.width)
                .checked_mul(u64::from(self.geometry.height))
                .and_then(|n| n.checked_mul(4))
                .context("decoded frame bytes overflow")?,
        )
        .map_err(Into::into)
    }
    fn skip(&mut self) -> Result<()> {
        let mut left = self.frame_bytes()?;
        let mut scratch = [0u8; 16 << 10];
        while left > 0 {
            let n = left.min(scratch.len());
            self.stdout.read_exact(&mut scratch[..n])?;
            left -= n;
        }
        self.next += 1;
        Ok(())
    }
    pub fn read(&mut self, index: usize) -> Result<ManagedRgba8Image> {
        ensure!(
            index >= self.next && index < self.frames,
            "video decoder frame order or extent differs from plan"
        );
        while self.next < index {
            self.skip()?;
        }
        let mut pixels = self
            .host
            .allocate_zeroed(HostMemoryClass::InputPreprocess, self.frame_bytes()?)?;
        self.stdout
            .read_exact(pixels.as_mut())
            .context("truncated decoded video frame")?;
        self.next += 1;
        Ok(ManagedRgba8Image::from_rgba8(
            self.host,
            pixels.freeze(),
            self.geometry.width,
            self.geometry.height,
            self.image,
        )?)
    }
    fn finish(&mut self) -> Result<()> {
        while self.next < self.frames {
            self.skip()?;
        }
        let mut byte = [0];
        ensure!(
            self.stdout.read(&mut byte)? == 0,
            "codec emitted pixels beyond declared frame geometry"
        );
        Ok(())
    }
}
#[derive(Deserialize)]
struct Metadata {
    streams: Vec<Stream>,
}
#[derive(Deserialize)]
struct Stream {
    codec_type: String,
    width: Option<u32>,
    height: Option<u32>,
    avg_frame_rate: Option<String>,
    r_frame_rate: Option<String>,
    nb_read_frames: Option<String>,
    sample_rate: Option<String>,
    channels: Option<u32>,
    #[serde(default)]
    side_data_list: Vec<SideData>,
}
#[derive(Deserialize)]
struct SideData {
    rotation: Option<f64>,
}
fn rate(value: &str) -> Option<(u32, u32)> {
    let (n, d) = value.split_once('/')?;
    let (n, d) = (n.parse::<u32>().ok()?, d.parse::<u32>().ok()?);
    (n > 0 && d > 0).then_some((n, d))
}

#[cfg(test)]
mod tests;
