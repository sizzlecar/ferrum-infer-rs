use anyhow::{Context, Result};
use serde::Serialize;
use std::{
    fs::{self, File},
    path::Path,
    process::Stdio,
    time::{Duration, Instant},
};
use tokio::process::{Child, Command};

pub(crate) fn isolated(command: &mut Command, agent_dir: &Path) {
    command.env_clear();
    // Keep ordinary local toolchain discovery; do not inherit provider credentials,
    // Node injection flags, proxies, or a user's pi plugins/configuration.
    for key in [
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "TMPDIR",
        "TMP",
        "TEMP",
        "LANG",
        "LC_ALL",
        "SystemRoot",
        "COMSPEC",
        "PATHEXT",
        "CARGO_HOME",
        "RUSTUP_HOME",
    ] {
        if let Some(value) = std::env::var_os(key) {
            command.env(key, value);
        }
    }
    command
        .env("PI_CODING_AGENT_DIR", agent_dir)
        .env("PI_OFFLINE", "1")
        .env("PI_TELEMETRY", "0")
        .env("CARGO_NET_OFFLINE", "true")
        .env("NO_COLOR", "1");
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.as_std_mut().process_group(0);
    }
}

pub(crate) struct ManagedChild {
    pub child: Child,
    pub pid: u32,
}
impl ManagedChild {
    pub fn spawn(command: &mut Command) -> Result<Self> {
        let child = command.spawn().context("spawn child")?;
        let pid = child.id().context("child pid")?;
        Ok(Self { child, pid })
    }
    pub async fn stop(&mut self) {
        #[cfg(unix)]
        unsafe {
            libc::kill(-(self.pid as i32), libc::SIGTERM);
        }
        #[cfg(not(unix))]
        {
            let _ = self.child.start_kill();
        }
        if tokio::time::timeout(Duration::from_secs(2), self.child.wait())
            .await
            .is_err()
        {
            #[cfg(unix)]
            unsafe {
                libc::kill(-(self.pid as i32), libc::SIGKILL);
            }
            let _ = self.child.kill().await;
        }
        let _ = self.child.wait().await;
    }
}
impl Drop for ManagedChild {
    fn drop(&mut self) {
        // pi handles SIGTERM and cleans up its detached bash children. Normal
        // completion/timeout uses stop(), which allows that cleanup to finish.
        if matches!(self.child.try_wait(), Ok(Some(_))) {
            return;
        }
        #[cfg(unix)]
        unsafe {
            libc::kill(-(self.pid as i32), libc::SIGTERM);
        }
        #[cfg(not(unix))]
        {
            let _ = self.child.start_kill();
        }
    }
}

#[derive(Clone, Debug, Serialize)]
pub(crate) struct Outcome {
    pub exit_code: Option<i32>,
    pub timed_out: bool,
    pub elapsed_ms: u128,
}

pub(crate) async fn logged(
    program: &Path,
    args: &[String],
    cwd: &Path,
    agent_dir: &Path,
    output: &Path,
    timeout_secs: u64,
) -> Result<Outcome> {
    fs::create_dir_all(output)?;
    super::write_json(
        output.join("command.json"),
        &serde_json::json!({"program":program,"args":args,"cwd":cwd}),
    )?;
    let mut command = Command::new(program);
    isolated(&mut command, agent_dir);
    command
        .args(args)
        .current_dir(cwd)
        .stdin(Stdio::null())
        .stdout(File::create(output.join("stdout.txt"))?)
        .stderr(File::create(output.join("stderr.txt"))?);
    let start = Instant::now();
    let mut process = ManagedChild::spawn(&mut command)?;
    let (exit_code, timed_out) =
        match tokio::time::timeout(Duration::from_secs(timeout_secs), process.child.wait()).await {
            Ok(status) => (status?.code(), false),
            Err(_) => {
                process.stop().await;
                (None, true)
            }
        };
    let result = Outcome {
        exit_code,
        timed_out,
        elapsed_ms: start.elapsed().as_millis(),
    };
    super::write_json(output.join("result.json"), &result)?;
    Ok(result)
}
