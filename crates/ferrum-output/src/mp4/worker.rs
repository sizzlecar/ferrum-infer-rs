use super::*;
use std::{
    ffi::OsString,
    fs::{File, OpenOptions},
    io::{Read, Write},
    process::{Child, Command, ExitStatus, Stdio},
    thread::JoinHandle,
    time::Instant,
};

struct Stderr {
    bytes: Vec<u8>,
    truncated: bool,
    error: Option<io::Error>,
}
struct Work<R, P> {
    permit: Option<Mp4EncodingPermit>,
    rgb: R,
    pcm: P,
    control: Arc<Control>,
    started: Instant,
    child: Option<Child>,
    stdout: Option<JoinHandle<Result<u64, Mp4Error>>>,
    stderr: Option<JoinHandle<Stderr>>,
    process: Option<CodecProcessReceipt>,
    arguments: Vec<OsString>,
}

pub(super) fn encode<R, P>(
    permit: Mp4EncodingPermit,
    rgb: R,
    pcm: P,
    control: Arc<Control>,
) -> Result<Mp4Artifact, Mp4Error>
where
    R: AsRef<[u8]> + Send + 'static,
    P: AsRef<[u8]> + Send + 'static,
{
    let mut work = Work {
        permit: Some(permit),
        rgb,
        pcm,
        control,
        started: Instant::now(),
        child: None,
        stdout: None,
        stderr: None,
        process: None,
        arguments: Vec::new(),
    };
    let mut outcome = work.run();
    if work.child.is_some() {
        if let Err(error) = work.kill_and_reap() {
            // No successful wait receipt: retain the original process, readers,
            // raw-file grant and input buffers, even if the error is dropped.
            let mut error = Mp4Error::new(Mp4ErrorKind::Reap, error.to_string());
            error.process = Some(CodecProcessReceipt {
                pid: work.control.pid.load(Ordering::Acquire),
                exit_code: None,
                success: false,
                kill_requested: true,
                reaped: false,
            });
            error._retained = Some(RetainedEncoder(Some(Box::new(work))));
            return Err(error);
        }
    }
    let encoded_bytes = match work.stdout.take().map(JoinHandle::join) {
        Some(Ok(Ok(bytes))) => bytes,
        Some(Ok(Err(error))) => {
            if outcome.is_ok()
                || outcome
                    .as_ref()
                    .is_err_and(|e| e.kind == Mp4ErrorKind::Codec)
            {
                outcome = Err(error);
            }
            0
        }
        Some(Err(_)) => {
            if outcome.is_ok() {
                outcome = Err(Mp4Error::new(Mp4ErrorKind::Codec, "stdout reader panicked"));
            }
            0
        }
        None => 0,
    };
    let stderr = match work.stderr.take().map(JoinHandle::join) {
        Some(Ok(stderr)) => stderr,
        Some(Err(_)) => {
            if outcome.is_ok() {
                outcome = Err(Mp4Error::new(Mp4ErrorKind::Codec, "stderr reader panicked"));
            }
            Stderr {
                bytes: Vec::new(),
                truncated: false,
                error: None,
            }
        }
        None => Stderr {
            bytes: Vec::new(),
            truncated: false,
            error: None,
        },
    };
    if outcome.is_ok() {
        if let Some(error) = stderr.error {
            outcome = Err(error.into());
        }
    }
    if outcome.is_ok() {
        outcome = work.check();
    }
    let permit = work.permit.take().expect("worker owns its admitted permit");
    if let Err(mut error) = outcome {
        error.process = work.process.take();
        error.stderr = stderr.bytes;
        error.stderr_truncated = stderr.truncated;
        error._buffers = Some(permit.buffers);
        // Remaining permit fields and both input owners can now drop: the
        // child is reaped and no pipe reader is accessing its staged files.
        return Err(error);
    }
    let process = work
        .process
        .take()
        .ok_or_else(|| Mp4Error::new(Mp4ErrorKind::Codec, "missing process exit receipt"))?;
    if encoded_bytes == 0 {
        return Err(Mp4Error::new(
            Mp4ErrorKind::Codec,
            "encoder produced no MP4 bytes",
        ));
    }
    let receipt = Mp4Receipt {
        spec: permit.spec,
        container_profile: "h264-aac-fragmented-v1",
        preset: permit.config.preset,
        ffmpeg: permit.config.ffmpeg.clone(),
        arguments: std::mem::take(&mut work.arguments),
        encoded_bytes,
        elapsed: work.started.elapsed(),
        process,
    };
    // stderr/read buffers die before their byte lease. The resulting artifact
    // retains only its session and actual finalized disk extent.
    drop(stderr.bytes);
    drop(permit.buffers);
    let process = receipt.process.clone();
    permit
        .entry
        .publish(receipt, permit.session)
        .map_err(|mut error| {
            error.process = Some(process);
            error
        })
}

impl<R: AsRef<[u8]>, P: AsRef<[u8]>> Work<R, P> {
    fn check(&self) -> Result<(), Mp4Error> {
        if self.control.cancelled.load(Ordering::Acquire) {
            return Err(Mp4Error::new(Mp4ErrorKind::Cancelled, "encoding cancelled"));
        }
        if self.started.elapsed() >= self.permit.as_ref().unwrap().config.timeout {
            return Err(Mp4Error::new(
                Mp4ErrorKind::Deadline,
                "encoding deadline exceeded",
            ));
        }
        Ok(())
    }
    fn stage(&self, name: &str, bytes: &[u8], pcm: bool) -> Result<(), Mp4Error> {
        let permit = self.permit.as_ref().unwrap();
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(permit.entry.path(name))?;
        if pcm {
            for values in bytes.chunks(permit.config.io_buffer_bytes.max(4) / 4 * 4) {
                self.check()?;
                if values
                    .chunks_exact(4)
                    .any(|v| !f32::from_le_bytes(v.try_into().unwrap()).is_finite())
                {
                    return Err(Mp4Error::invalid(
                        "stereo F32LE PCM contains non-finite samples",
                    ));
                }
                file.write_all(values)?;
            }
        } else {
            for values in bytes.chunks(permit.config.io_buffer_bytes) {
                self.check()?;
                file.write_all(values)?;
            }
        }
        file.flush()?;
        Ok(())
    }
    fn run(&mut self) -> Result<(), Mp4Error> {
        self.check()?;
        self.stage("rgb.raw", self.rgb.as_ref(), false)?;
        self.stage("audio.f32le", self.pcm.as_ref(), true)?;
        self.check()?;
        let permit = self.permit.as_ref().unwrap();
        let config = &permit.config;
        let spec = permit.spec;
        let strings = [
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
        ];
        let mut args: Vec<OsString> = strings.into_iter().map(Into::into).collect();
        args.extend([
            format!("{}x{}", spec.width, spec.height).into(),
            "-framerate".into(),
            format!("{}/{}", spec.fps_numerator, spec.fps_denominator).into(),
            "-threads".into(),
            config.threads.to_string().into(),
            "-i".into(),
            permit.entry.path("rgb.raw").into_os_string(),
            "-f".into(),
            "f32le".into(),
            "-ar".into(),
            spec.audio_sample_rate.to_string().into(),
            "-ac".into(),
            "2".into(),
            "-threads".into(),
            config.threads.to_string().into(),
            "-i".into(),
            permit.entry.path("audio.f32le").into_os_string(),
            "-map".into(),
            "0:v:0".into(),
            "-map".into(),
            "1:a:0".into(),
            "-c:v".into(),
            "libx264".into(),
            "-preset".into(),
            config.preset.h264_speed.argument().into(),
            "-crf".into(),
            config.preset.crf.to_string().into(),
            "-pix_fmt".into(),
            "yuv420p".into(),
            "-threads:v".into(),
            config.threads.to_string().into(),
            "-filter_threads".into(),
            config.threads.to_string().into(),
            "-filter_complex_threads".into(),
            config.threads.to_string().into(),
            "-c:a".into(),
            "aac".into(),
            "-b:a".into(),
            config.preset.audio_bitrate.to_string().into(),
            "-threads:a".into(),
            config.threads.to_string().into(),
            "-movflags".into(),
            // Wait for the first fragment's actual codec timestamps so AAC
            // priming and H.264 reordering can be represented by edit lists.
            // An immediate empty moov can stretch frame zero or shift video.
            "delay_moov+frag_keyframe+default_base_moof".into(),
            "-f".into(),
            "mp4".into(),
            "pipe:1".into(),
        ]);
        let output = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(permit.entry.path("video.part"))?;
        let child = Command::new(&config.ffmpeg)
            .args(&args)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| Mp4Error::new(Mp4ErrorKind::Spawn, e.to_string()))?;
        self.control.pid.store(child.id(), Ordering::Release);
        self.arguments = args;
        self.child = Some(child);
        let stdout = self
            .child
            .as_mut()
            .unwrap()
            .stdout
            .take()
            .ok_or_else(|| Mp4Error::new(Mp4ErrorKind::Codec, "missing stdout pipe"))?;
        let stderr = self
            .child
            .as_mut()
            .unwrap()
            .stderr
            .take()
            .ok_or_else(|| Mp4Error::new(Mp4ErrorKind::Codec, "missing stderr pipe"))?;
        let io_bytes = config.io_buffer_bytes;
        let limit = config.maximum_encoded_bytes;
        let stderr_bytes = config.maximum_stderr_bytes;
        let control = self.control.clone();
        self.stdout = Some(
            std::thread::Builder::new()
                .name("ferrum-mp4-stdout".into())
                .spawn(move || {
                    let result = copy_output(stdout, output, io_bytes, limit);
                    if result.is_err() {
                        control.pipe_failed.store(true, Ordering::Release);
                        control.notify();
                    }
                    result
                })?,
        );
        let control = self.control.clone();
        self.stderr = Some(
            std::thread::Builder::new()
                .name("ferrum-mp4-stderr".into())
                .spawn(move || {
                    let result = drain_stderr(stderr, io_bytes, stderr_bytes);
                    if result.error.is_some() {
                        control.pipe_failed.store(true, Ordering::Release);
                        control.notify();
                    }
                    result
                })?,
        );
        loop {
            self.check()?;
            if self.control.pipe_failed.load(Ordering::Acquire) {
                return Err(Mp4Error::new(
                    Mp4ErrorKind::Codec,
                    "codec pipe reader failed",
                ));
            }
            if let Some(status) = self.child.as_mut().unwrap().try_wait()? {
                self.record_exit(status, false);
                return if status.success() {
                    Ok(())
                } else {
                    Err(Mp4Error::new(
                        Mp4ErrorKind::Codec,
                        format!("ffmpeg exited {status}"),
                    ))
                };
            }
            self.control.wait(Duration::from_millis(5));
        }
    }
    fn record_exit(&mut self, status: ExitStatus, kill_requested: bool) {
        let child = self
            .child
            .take()
            .expect("only an owned child can be reaped");
        self.process = Some(CodecProcessReceipt {
            pid: child.id(),
            exit_code: status.code(),
            success: status.success(),
            kill_requested,
            reaped: true,
        });
        self.control.reaped.store(true, Ordering::Release);
    }
    fn kill_and_reap(&mut self) -> io::Result<()> {
        let child = self.child.as_mut().unwrap();
        // A process that already exited may reject kill; only wait establishes
        // the retirement receipt and allows any source/file owners to release.
        let _ = child.kill();
        let status = child.wait()?;
        self.record_exit(status, true);
        Ok(())
    }
}

fn copy_output(
    mut source: impl Read,
    mut target: File,
    buffer_bytes: usize,
    limit: u64,
) -> Result<u64, Mp4Error> {
    let mut buffer = vec![0u8; buffer_bytes];
    let mut total = 0u64;
    loop {
        let count = source.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        if count as u64 > limit - total {
            return Err(Mp4Error::new(
                Mp4ErrorKind::EncodedLimit,
                "MP4 exceeds admitted encoded-byte limit",
            ));
        }
        target.write_all(&buffer[..count])?;
        total += count as u64;
    }
    target.flush()?;
    Ok(total)
}
fn drain_stderr(mut source: impl Read, buffer_bytes: usize, limit: usize) -> Stderr {
    let mut buffer = vec![0u8; buffer_bytes];
    let mut result = Stderr {
        bytes: Vec::with_capacity(limit),
        truncated: false,
        error: None,
    };
    loop {
        match source.read(&mut buffer) {
            Ok(0) => break,
            Ok(count) => {
                let retained = count.min(limit - result.bytes.len());
                result.bytes.extend_from_slice(&buffer[..retained]);
                result.truncated |= retained < count;
            }
            Err(error) => {
                result.error = Some(error);
                break;
            }
        }
    }
    result
}
