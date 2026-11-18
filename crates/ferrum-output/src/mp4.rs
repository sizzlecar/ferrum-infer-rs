//! H.264/AAC fragmented MP4 output with retained source and artifact owners.
//!
//! Callers retain their original accounting for RGB24 and stereo F32LE inputs.
//! [`Mp4Encoder::try_admit`] reserves session I/O buffers and separate physical
//! disk file extents before producer work. Spool byte limits cover written file
//! lengths, not filesystem metadata or block-allocation rounding. Neither
//! account measures or limits the
//! external ffmpeg process's RSS, its allocator overhead or OS pipe buffers.
//! The explicit thread limit is not an RSS limit.
//!
//! A standard Rust worker owns the child until kill/wait and pipe-reader joins
//! finish. Dropping an async waiter, or its Tokio runtime, requests cancellation
//! without dropping that worker's source buffers or resource leases. Published
//! files remain valid until their last artifact/reader owner is released.
//! Atomic publication is not a promise of crash durability (no fsync policy).

use crate::{OutputBytesLease, OutputSession};
use std::{
    fmt, io,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicU32, Ordering},
        Arc, Condvar, Mutex,
    },
    time::Duration,
};
use tokio::sync::oneshot;

mod spool;
mod worker;
pub use spool::{Mp4Artifact, Mp4ArtifactReader, Mp4Spool, Mp4SpoolLimits, Mp4SpoolUsage};

/// Exact input extents. Audio is interleaved two-channel IEEE F32 little-endian.
/// Audio and video durations are independently preserved; no `-shortest` trim
/// or implicit sample-rate conversion is applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VideoAudioSpec {
    pub width: u32,
    pub height: u32,
    pub frames: u64,
    pub fps_numerator: u32,
    pub fps_denominator: u32,
    pub audio_sample_rate: u32,
    pub audio_sample_frames: u64,
}
impl VideoAudioSpec {
    pub fn rgb_bytes(self) -> Result<usize, Mp4Error> {
        self.validate()?;
        let bytes = u64::from(self.width)
            .checked_mul(u64::from(self.height))
            .and_then(|n| n.checked_mul(self.frames))
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| Mp4Error::invalid("RGB extent overflow"))?;
        extent(bytes)
    }
    pub fn pcm_bytes(self) -> Result<usize, Mp4Error> {
        self.validate()?;
        extent(
            self.audio_sample_frames
                .checked_mul(8)
                .ok_or_else(|| Mp4Error::invalid("stereo F32LE extent overflow"))?,
        )
    }
    fn validate(self) -> Result<(), Mp4Error> {
        if self.width == 0
            || self.height == 0
            || self.width % 2 != 0
            || self.height % 2 != 0
            || self.frames == 0
            || self.fps_numerator == 0
            || self.fps_denominator == 0
            || self.audio_sample_rate != 32000
            || self.audio_sample_frames == 0
        {
            return Err(Mp4Error::invalid("H264 yuv420p requires positive even dimensions and frame rate/lengths; this AAC profile requires 32000-Hz stereo"));
        }
        Ok(())
    }
}
fn extent(bytes: u64) -> Result<usize, Mp4Error> {
    usize::try_from(bytes)
        .ok()
        .filter(|n| *n <= isize::MAX as usize)
        .ok_or_else(|| Mp4Error::invalid("media extent exceeds address space"))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum H264SpeedPreset {
    Ultrafast,
    Veryfast,
    Medium,
    Veryslow,
}
impl H264SpeedPreset {
    fn argument(self) -> &'static str {
        match self {
            Self::Ultrafast => "ultrafast",
            Self::Veryfast => "veryfast",
            Self::Medium => "medium",
            Self::Veryslow => "veryslow",
        }
    }
}
/// Versioned non-seekable output profile: libx264/yuv420p + native AAC, MP4
/// `delay_moov+frag_keyframe+default_base_moof` preserves the original audio/video
/// clock origins through codec delay. This is not `faststart` MP4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct H264AacFragmentedV1 {
    pub h264_speed: H264SpeedPreset,
    pub crf: u8,
    pub audio_bitrate: u32,
}
#[derive(Debug, Clone)]
pub struct Mp4EncoderConfig {
    pub ffmpeg: PathBuf,
    pub threads: u32,
    /// Entire worker deadline, including raw-file staging and process teardown
    /// initiation. A slow OS kill/reap is retained beyond this deadline.
    pub timeout: Duration,
    pub maximum_encoded_bytes: u64,
    pub io_buffer_bytes: usize,
    pub maximum_stderr_bytes: usize,
    pub preset: H264AacFragmentedV1,
}
impl Mp4EncoderConfig {
    fn buffer_bytes(&self) -> Result<usize, Mp4Error> {
        self.io_buffer_bytes
            .checked_mul(2)
            .and_then(|n| n.checked_add(self.maximum_stderr_bytes))
            .ok_or_else(|| Mp4Error::invalid("codec buffer extent overflow"))
    }
    fn validate(&self) -> Result<(), Mp4Error> {
        if !self.ffmpeg.is_absolute()
            || self.threads == 0
            || self.timeout.is_zero()
            || self.maximum_encoded_bytes == 0
            || self.io_buffer_bytes == 0
            || self.maximum_stderr_bytes == 0
            || self.preset.crf > 51
            || self.preset.audio_bitrate == 0
        {
            return Err(Mp4Error::invalid(
                "invalid ffmpeg path, thread, time, capacity or H264/AAC settings",
            ));
        }
        let buffers = self.buffer_bytes()?;
        // OutputSession's byte-grant API uses u32 semaphore acquisitions.
        if buffers > u32::MAX as usize {
            return Err(Mp4Error::invalid(
                "codec buffers exceed OutputSession grant width",
            ));
        }
        extent(buffers as u64)?;
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct Mp4Receipt {
    pub spec: VideoAudioSpec,
    pub container_profile: &'static str,
    pub preset: H264AacFragmentedV1,
    pub ffmpeg: PathBuf,
    /// Exact arguments, including input formats, thread limits and mux flags.
    pub arguments: Vec<std::ffi::OsString>,
    pub encoded_bytes: u64,
    pub elapsed: Duration,
    pub process: CodecProcessReceipt,
}
#[derive(Debug, Clone)]
pub struct CodecProcessReceipt {
    pub pid: u32,
    pub exit_code: Option<i32>,
    pub success: bool,
    pub kill_requested: bool,
    pub reaped: bool,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mp4ErrorKind {
    InvalidInput,
    Capacity,
    Io,
    Spawn,
    Codec,
    EncodedLimit,
    Cancelled,
    Deadline,
    Reap,
}

/// Failed encodes retain their bounded stderr allocation and its session lease
/// until the error is consumed/dropped. Cleanup failures stay charged to spool.
#[derive(Debug)]
pub struct Mp4Error {
    pub kind: Mp4ErrorKind,
    pub message: String,
    pub process: Option<CodecProcessReceipt>,
    stderr: Vec<u8>,
    pub stderr_truncated: bool,
    _buffers: Option<OutputBytesLease>,
    _retained: Option<RetainedEncoder>,
}
impl Mp4Error {
    fn new(kind: Mp4ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
            process: None,
            stderr: Vec::new(),
            stderr_truncated: false,
            _buffers: None,
            _retained: None,
        }
    }
    fn invalid(message: &'static str) -> Self {
        Self::new(Mp4ErrorKind::InvalidInput, message)
    }
    pub fn stderr(&self) -> &[u8] {
        &self.stderr
    }
    /// A failed OS reap retains the original child, source owners and grants;
    /// dropping that error quarantines them rather than declaring them free.
    pub fn has_unreaped_process(&self) -> bool {
        self._retained.is_some()
    }
}
struct RetainedEncoder(Option<Box<dyn std::any::Any + Send>>);
impl fmt::Debug for RetainedEncoder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("RetainedEncoder(unreaped child and original owners)")
    }
}
impl Drop for RetainedEncoder {
    fn drop(&mut self) {
        if let Some(owner) = self.0.take() {
            std::mem::forget(owner);
        }
    }
}
impl fmt::Display for Mp4Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.kind, self.message)
    }
}
impl std::error::Error for Mp4Error {}
impl From<io::Error> for Mp4Error {
    fn from(error: io::Error) -> Self {
        Self::new(Mp4ErrorKind::Io, error.to_string())
    }
}

#[derive(Clone)]
pub struct Mp4Encoder {
    config: Arc<Mp4EncoderConfig>,
    spool: Mp4Spool,
}
impl Mp4Encoder {
    pub fn new(config: Mp4EncoderConfig, spool: Mp4Spool) -> Result<Self, Mp4Error> {
        config.validate()?;
        Ok(Self {
            config: Arc::new(config),
            spool,
        })
    }

    /// Encode one black frame and a short stereo silence through the configured
    /// H.264/AAC muxer. The worker owns and retires its process and temporary
    /// files exactly as for a full encode. No input decoder is involved.
    pub async fn check_output_profile(&self, session: OutputSession) -> Result<(), Mp4Error> {
        use std::io::Read;

        let mut config = (*self.config).clone();
        config.timeout = config.timeout.min(Duration::from_secs(10));
        config.maximum_encoded_bytes = config.maximum_encoded_bytes.min(64 << 10);
        let encoder = Self::new(config, self.spool.clone())?;
        let spec = VideoAudioSpec {
            width: 16,
            height: 16,
            frames: 1,
            fps_numerator: 25,
            fps_denominator: 1,
            audio_sample_rate: 32000,
            audio_sample_frames: 1280,
        };
        let permit = encoder.try_admit(session, spec)?.ok_or_else(|| {
            Mp4Error::new(
                Mp4ErrorKind::Capacity,
                "no output capacity for the MP4 encoder check",
            )
        })?;
        let artifact = permit
            .start(vec![0; spec.rgb_bytes()?], vec![0; spec.pcm_bytes()?])?
            .wait()
            .await?;
        let mut header = [0; 8];
        artifact.open()?.read_exact(&mut header)?;
        if &header[4..] != b"ftyp" {
            return Err(Mp4Error::new(
                Mp4ErrorKind::Codec,
                "encoder returned an invalid MP4 header",
            ));
        }
        Ok(())
    }
    /// Metadata-only admission reserves worst-case raw staging + encoded file
    /// bytes and three files. `None` means a current output owner holds capacity.
    pub fn try_admit(
        &self,
        session: OutputSession,
        spec: VideoAudioSpec,
    ) -> Result<Option<Mp4EncodingPermit>, Mp4Error> {
        let rgb_bytes = spec.rgb_bytes()?;
        let pcm_bytes = spec.pcm_bytes()?;
        let disk_bytes = (rgb_bytes as u64)
            .checked_add(pcm_bytes as u64)
            .and_then(|n| n.checked_add(self.config.maximum_encoded_bytes))
            .ok_or_else(|| Mp4Error::invalid("raw and encoded spool extent overflow"))?;
        if session.is_closed() {
            return Err(Mp4Error::new(
                Mp4ErrorKind::Capacity,
                "output session closed",
            ));
        }
        // A saturated spool must not acquire/refund I/O credit on every
        // scheduler probe: its refund notification would wake that same
        // waiter forever without any external resource change. This is only
        // a probe; try_reserve below remains the atomic grant authority.
        if !self.spool.capacity_available(disk_bytes, 3)? {
            return Ok(None);
        }
        let Some(buffers) = session
            .try_reserve_encoded(self.config.buffer_bytes()?)
            .map_err(|e| Mp4Error::new(Mp4ErrorKind::Capacity, e.to_string()))?
        else {
            return Ok(None);
        };
        let Some(entry) = self.spool.try_reserve(disk_bytes, 3)? else {
            return Ok(None);
        };
        Ok(Some(Mp4EncodingPermit {
            config: self.config.clone(),
            spec,
            rgb_bytes,
            pcm_bytes,
            session,
            buffers,
            entry,
        }))
    }
}

pub struct Mp4EncodingPermit {
    config: Arc<Mp4EncoderConfig>,
    spec: VideoAudioSpec,
    rgb_bytes: usize,
    pcm_bytes: usize,
    session: OutputSession,
    buffers: OutputBytesLease,
    entry: spool::SpoolEntry,
}
impl Mp4EncodingPermit {
    /// Move the original admitted owners (e.g. ManagedHostBytes), without a
    /// full input copy. The worker retains both through process reaping.
    pub fn start<R, P>(self, rgb: R, stereo_f32le: P) -> Result<Mp4EncodingJob, Mp4Error>
    where
        R: AsRef<[u8]> + Send + 'static,
        P: AsRef<[u8]> + Send + 'static,
    {
        if rgb.as_ref().len() != self.rgb_bytes || stereo_f32le.as_ref().len() != self.pcm_bytes {
            return Err(Mp4Error::invalid(
                "source byte lengths differ from exact RGB24/stereo F32LE spec",
            ));
        }
        let control = Arc::new(Control::default());
        let thread_control = control.clone();
        let (sender, receiver) = oneshot::channel();
        std::thread::Builder::new()
            .name("ferrum-mp4-encoder".into())
            .spawn(move || {
                let result = worker::encode(self, rgb, stereo_f32le, thread_control);
                // A disappeared waiter drops the completed artifact here, after
                // the worker has reaped its child and joined its pipe readers.
                let _ = sender.send(result);
            })
            .map_err(|e| Mp4Error::new(Mp4ErrorKind::Spawn, e.to_string()))?;
        Ok(Mp4EncodingJob {
            receiver: Some(receiver),
            control,
        })
    }
}

#[derive(Default)]
struct Control {
    cancelled: AtomicBool,
    pid: AtomicU32,
    reaped: AtomicBool,
    pipe_failed: AtomicBool,
    mutex: Mutex<()>,
    changed: Condvar,
}
impl Control {
    fn notify(&self) {
        let _guard = self.mutex.lock().unwrap_or_else(|e| e.into_inner());
        self.changed.notify_all();
    }
    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
        self.notify();
    }
    fn wait(&self, duration: Duration) {
        let guard = self.mutex.lock().unwrap_or_else(|e| e.into_inner());
        if !self.cancelled.load(Ordering::Acquire) {
            let _ = self.changed.wait_timeout(guard, duration);
        }
    }
}
#[derive(Clone)]
pub struct Mp4CancellationHandle(Arc<Control>);
impl Mp4CancellationHandle {
    pub fn cancel(&self) {
        self.0.cancel();
    }
    pub fn process_id(&self) -> Option<u32> {
        match self.0.pid.load(Ordering::Acquire) {
            0 => None,
            id => Some(id),
        }
    }
    pub fn process_reaped(&self) -> bool {
        self.0.reaped.load(Ordering::Acquire)
    }
}
pub struct Mp4EncodingJob {
    receiver: Option<oneshot::Receiver<Result<Mp4Artifact, Mp4Error>>>,
    control: Arc<Control>,
}
impl Mp4EncodingJob {
    pub fn cancellation_handle(&self) -> Mp4CancellationHandle {
        Mp4CancellationHandle(self.control.clone())
    }
    pub fn cancel(&self) {
        self.control.cancel();
    }
    pub async fn wait(mut self) -> Result<Mp4Artifact, Mp4Error> {
        self.receiver
            .take()
            .expect("job result has one consumer")
            .await
            .unwrap_or_else(|_| {
                Err(Mp4Error::new(
                    Mp4ErrorKind::Codec,
                    "encoder worker response channel closed",
                ))
            })
    }
}
impl Drop for Mp4EncodingJob {
    fn drop(&mut self) {
        self.control.cancel();
    }
}

#[cfg(test)]
mod tests;
