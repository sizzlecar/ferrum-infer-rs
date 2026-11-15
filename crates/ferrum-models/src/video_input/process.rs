//! A scoped codec child owns its input file until exit. Cancellation, deadlines,
//! failed readers and unwinding all kill and reap before releasing that file.
use super::*;
use std::{
    io::Write,
    path::Path,
    process::{Child, Command, ExitStatus, Stdio},
    sync::atomic::Ordering,
    thread,
    time::Instant,
};

struct ChildOwner(Option<Child>);
impl ChildOwner {
    fn kill_and_reap(&mut self) -> ExitStatus {
        let child = self.0.as_mut().expect("owned codec child");
        let _ = child.kill();
        loop {
            match child.wait() {
                Ok(status) => {
                    self.0 = None;
                    return status;
                }
                // Without a wait receipt the scoped worker retains every
                // original owner. It cannot publish success or refund capacity.
                Err(_) => thread::sleep(Duration::from_millis(10)),
            }
        }
    }
}
impl Drop for ChildOwner {
    fn drop(&mut self) {
        if self.0.is_some() {
            self.kill_and_reap();
        }
    }
}
struct Abort<'a>(&'a AtomicBool);
impl Drop for Abort<'_> {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

pub(super) fn run<T>(
    codec: &VideoInputCodec,
    program: &Path,
    encoded: &ManagedHostBytes,
    arguments: Vec<OsString>,
    decode: bool,
    cancelled: &AtomicBool,
    consume: impl FnOnce(&mut dyn Read) -> Result<T>,
) -> Result<T> {
    let config = codec.config();
    ensure!(
        !encoded.is_empty() && encoded.len() <= config.maximum_encoded_bytes,
        "encoded video exceeds declared input capacity"
    );
    let started = Instant::now();
    let check = || -> Result<()> {
        if cancelled.load(Ordering::Acquire) {
            return Err(VideoInputInterruption::Cancelled.into());
        }
        if started.elapsed() >= config.timeout {
            return Err(VideoInputInterruption::Deadline.into());
        }
        Ok(())
    };
    // Shared codec staging is bounded to one source file per owner. Pressure
    // waits for its original owner rather than rejecting another valid input.
    let _serial = loop {
        check()?;
        if let Some(guard) = codec.0.gate.try_lock_for(Duration::from_millis(10)) {
            break guard;
        }
    };
    let mut staged = tempfile::NamedTempFile::new_in(&config.staging_directory)?;
    for chunk in encoded.bytes().chunks(64 << 10) {
        check()?;
        staged.write_all(chunk)?;
    }
    staged.flush()?;
    let mut command = Command::new(program);
    // Uploaded bytes describe one self-contained media container. Apply the
    // same input boundary to probing, frame decoding and soundtrack decoding;
    // playlists must not turn an upload into additional file/network reads.
    command.args([
        "-protocol_whitelist",
        "file",
        "-format_whitelist",
        "mov,matroska,webm",
    ]);
    if decode {
        command
            .args(["-nostdin", "-hide_banner", "-loglevel", "error", "-threads"])
            .arg(config.threads.to_string())
            .arg("-i")
            .arg(staged.path())
            .args(arguments);
    } else {
        command.args(arguments).arg(staged.path());
    }
    let mut child = ChildOwner(Some(
        command
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("start configured video codec")?,
    ));
    let mut stdout = child
        .0
        .as_mut()
        .unwrap()
        .stdout
        .take()
        .context("missing codec stdout")?;
    let mut stderr = child
        .0
        .as_mut()
        .unwrap()
        .stderr
        .take()
        .context("missing codec stderr")?;
    let abort = AtomicBool::new(false);
    thread::scope(|scope| -> Result<T> {
        // This guard also runs if the consumer panics or a reader cannot start.
        let _abort_on_drop = Abort(&abort);
        let abort_signal = &abort;
        let monitor = thread::Builder::new()
            .name("ferrum-video-input-process".into())
            .spawn_scoped(scope, move || -> Result<ExitStatus> {
                loop {
                    if let Some(status) = child.0.as_mut().unwrap().try_wait()? {
                        child.0 = None;
                        return Ok(status);
                    }
                    if abort_signal.load(Ordering::Acquire)
                        || cancelled.load(Ordering::Acquire)
                        || started.elapsed() >= config.timeout
                    {
                        return Ok(child.kill_and_reap());
                    }
                    thread::sleep(Duration::from_millis(5));
                }
            })?;
        let diagnostics = thread::Builder::new()
            .name("ferrum-video-input-stderr".into())
            .spawn_scoped(scope, move || -> Result<Vec<u8>> {
                let mut retained = Vec::new();
                let mut scratch = [0u8; 4096];
                loop {
                    let n = stderr.read(&mut scratch)?;
                    if n == 0 {
                        break;
                    }
                    let keep = n.min(config.maximum_stderr_bytes - retained.len());
                    retained.extend_from_slice(&scratch[..keep]);
                }
                Ok(retained)
            })?;
        let result = consume(&mut stdout);
        drop(stdout);
        if result.is_err() {
            abort.store(true, Ordering::Release);
        }
        let exit = monitor
            .join()
            .map_err(|_| anyhow::anyhow!("video codec monitor panicked"))??;
        let stderr = diagnostics
            .join()
            .map_err(|_| anyhow::anyhow!("video codec diagnostics panicked"))??;
        check()?;
        let result = result.with_context(|| {
            format!(
                "video codec exit {exit}: {}",
                String::from_utf8_lossy(&stderr)
            )
        })?;
        ensure!(
            exit.success(),
            "video codec exited {exit}: {}",
            String::from_utf8_lossy(&stderr)
        );
        Ok(result)
    })
}
pub(super) fn read_capped(source: &mut dyn Read, maximum: usize) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    source.take(maximum as u64 + 1).read_to_end(&mut bytes)?;
    ensure!(
        bytes.len() <= maximum,
        "video codec metadata exceeds declared capacity"
    );
    Ok(bytes)
}
