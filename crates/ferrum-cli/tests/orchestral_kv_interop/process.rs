use super::*;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::process::{Child, ExitStatus, Stdio};

pub(super) struct OwnedChild(Child);

impl OwnedChild {
    pub(super) fn spawn(mut command: Command, directory: &Path, label: &str) -> Result<Self> {
        write_json(
            directory.join(format!("{label}.command.json")),
            &json!({
                "program": command.get_program().to_string_lossy(),
                "args": command.get_args().map(|arg| arg.to_string_lossy()).collect::<Vec<_>>(),
                "cwd": command.get_current_dir(),
            }),
        )?;
        command
            .stdin(Stdio::null())
            .stdout(File::create(directory.join(format!("{label}.stdout.log")))?)
            .stderr(File::create(directory.join(format!("{label}.stderr.log")))?);
        Ok(Self(
            command.spawn().with_context(|| format!("spawn {label}"))?,
        ))
    }

    pub(super) fn try_wait(&mut self) -> Result<Option<ExitStatus>> {
        Ok(self.0.try_wait()?)
    }
}

impl Drop for OwnedChild {
    fn drop(&mut self) {
        // Only this process handle is touched; never stop services by port/name.
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

pub(super) struct Outcome {
    pub(super) success: bool,
    pub(super) seconds: f64,
}

pub(super) async fn run(
    command: Command,
    directory: &Path,
    label: &str,
    timeout: Duration,
) -> Result<Outcome> {
    let mut process = OwnedChild::spawn(command, directory, label)?;
    let start = Instant::now();
    loop {
        if let Some(status) = process.try_wait()? {
            let seconds = start.elapsed().as_secs_f64();
            write_json(
                directory.join(format!("{label}.exit.json")),
                &json!({
                    "success": status.success(), "exit_code": status.code(), "seconds": seconds, "timed_out": false
                }),
            )?;
            return Ok(Outcome {
                success: status.success(),
                seconds,
            });
        }
        if start.elapsed() >= timeout {
            write_json(
                directory.join(format!("{label}.exit.json")),
                &json!({
                    "success": false, "seconds": start.elapsed().as_secs_f64(), "timed_out": true
                }),
            )?;
            anyhow::bail!("{label} exceeded {} seconds", timeout.as_secs());
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
}

fn identity(path: &Path) -> Result<Value> {
    let mut file = File::open(path)?;
    let bytes = file.metadata()?.len();
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(json!({"path": path, "bytes": bytes, "sha256": format!("{:x}", digest.finalize())}))
}

pub(super) fn identities(ferrum: &Path, orchestral: &Path) -> Result<Value> {
    Ok(json!({"ferrum": identity(ferrum)?, "orchestral": identity(orchestral)?}))
}
