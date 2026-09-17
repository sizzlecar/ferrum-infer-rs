//! Child ownership, bounded waiting and durable raw process evidence.
use super::write_json;
use anyhow::{ensure, Context, Result};
use serde_json::json;
use std::fs::{self, File};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

struct OwnedChild(Child);
impl Drop for OwnedChild {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

pub(super) async fn run(
    program: &Path,
    args: &[String],
    directory: &Path,
    name: &str,
    timeout: Duration,
) -> Result<String> {
    let stdout = directory.join(format!("{name}.stdout.jsonl"));
    let stderr = directory.join(format!("{name}.stderr.txt"));
    let mut command = Command::new(program);
    let mut removed = Vec::new();
    for (key, _) in std::env::vars_os() {
        if key.to_string_lossy().starts_with("FERRUM_") {
            command.env_remove(&key);
            removed.push(key.to_string_lossy().into_owned());
        }
    }
    removed.sort();
    write_json(
        directory.join(format!("{name}.command.json")),
        &json!({
            "program":program,"args":args,"cwd":directory,"timeout_secs":timeout.as_secs_f64(),
            "environment_overrides":{"NO_COLOR":"1"},"removed_environment_keys":removed,
        }),
    )?;
    eprintln!("kv teacher regression: {name}");
    let mut child = OwnedChild(
        command
            .args(args)
            .current_dir(directory)
            .env("NO_COLOR", "1")
            .stdin(Stdio::null())
            .stdout(File::create(&stdout)?)
            .stderr(File::create(&stderr)?)
            .spawn()
            .with_context(|| format!("spawn {}", program.display()))?,
    );
    let started = Instant::now();
    let status = loop {
        if let Some(status) = child.0.try_wait()? {
            break Some(status);
        }
        if started.elapsed() >= timeout {
            break None;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    };
    // Reap on every path, including timeout and serialization errors.
    drop(child);
    write_json(
        directory.join(format!("{name}.exit.json")),
        &json!({
            "exit_code":status.and_then(|status| status.code()),"timed_out":status.is_none(),
            "success":status.is_some_and(|status| status.success()),"elapsed_ms":started.elapsed().as_millis(),
        }),
    )?;
    ensure!(
        status.is_some_and(|status| status.success()),
        "{name} failed or exceeded {timeout:?}; see {}",
        stderr.display()
    );
    fs::read_to_string(stdout).context("read child UTF-8 stdout")
}
