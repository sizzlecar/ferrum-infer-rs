use anyhow::{ensure, Context, Result};
use serde::Serialize;
use std::{
    fs::{self, File},
    path::Path,
    process::Stdio,
    time::{Duration, Instant},
};
use tokio::process::{Child, Command};

pub(crate) fn isolated_local(command: &mut Command) {
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
        .env("CARGO_NET_OFFLINE", "true")
        .env("NO_COLOR", "1");
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.as_std_mut().process_group(0);
    }
}

pub(crate) fn isolated(command: &mut Command, agent_dir: &Path) {
    isolated_local(command);
    command
        .env("PI_CODING_AGENT_DIR", agent_dir)
        .env("PI_OFFLINE", "1")
        .env("PI_TELEMETRY", "0");
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
        self.stop_with_grace(Duration::from_secs(2)).await;
    }

    /// Orchestral handles Ctrl-C by cancelling its run and shutting down owned tools.
    /// The normal TERM/KILL cleanup remains the bounded fallback, with the same owner.
    pub(crate) async fn interrupt_then_stop(&mut self) {
        #[cfg(unix)]
        {
            unsafe {
                libc::kill(-(self.pid as i32), libc::SIGINT);
            }
            if matches!(
                tokio::time::timeout(Duration::from_secs(1), self.child.wait()).await,
                Ok(Ok(_))
            ) {
                unsafe {
                    libc::kill(-(self.pid as i32), libc::SIGKILL);
                }
                return;
            }
        }
        self.stop().await;
    }

    async fn stop_with_grace(&mut self, grace: Duration) {
        #[cfg(unix)]
        unsafe {
            libc::kill(-(self.pid as i32), libc::SIGTERM);
        }
        #[cfg(not(unix))]
        {
            let _ = self.child.start_kill();
        }
        let waited = tokio::time::timeout(grace, self.child.wait()).await;
        // A group leader exiting on TERM does not prove its descendants exited.
        // Finish terminating this owned group, then reap the direct child.
        #[cfg(unix)]
        unsafe {
            libc::kill(-(self.pid as i32), libc::SIGKILL);
        }
        if !matches!(waited, Ok(Ok(_))) {
            let _ = self.child.kill().await;
        }
        let _ = self.child.wait().await;
    }
}

/// Reused across compile/list/execute, at the layer retaining the active child.
/// Non-Unix keeps the existing direct-child timeout capability boundary.
pub(crate) struct Cancellation {
    source: CancellationSource,
    requested: bool,
}

enum CancellationSource {
    #[cfg(unix)]
    Terminate(tokio::signal::unix::Signal),
    #[cfg(not(unix))]
    Never,
    #[cfg(test)]
    Test(tokio::sync::oneshot::Receiver<()>),
}

impl Cancellation {
    pub(crate) fn controller_termination() -> Result<Self> {
        #[cfg(unix)]
        let source = CancellationSource::Terminate(tokio::signal::unix::signal(
            tokio::signal::unix::SignalKind::terminate(),
        )?);
        #[cfg(not(unix))]
        let source = CancellationSource::Never;
        Ok(Self {
            source,
            requested: false,
        })
    }

    async fn wait(&mut self) {
        if self.requested {
            return;
        }
        match &mut self.source {
            #[cfg(unix)]
            CancellationSource::Terminate(signal) => {
                signal.recv().await;
            }
            #[cfg(not(unix))]
            CancellationSource::Never => std::future::pending::<()>().await,
            #[cfg(test)]
            CancellationSource::Test(receiver) => {
                let _ = receiver.await;
            }
        }
        self.requested = true;
    }
}

async fn wait_owned(
    process: &mut ManagedChild,
    start: Instant,
    timeout: Duration,
    cancellation: Option<&mut Cancellation>,
) -> Result<(Outcome, bool)> {
    let controlled = cancellation.is_some();
    let cancel = async {
        match cancellation {
            Some(cancellation) => cancellation.wait().await,
            None => std::future::pending::<()>().await,
        }
    };
    let (exit_code, timed_out, cancelled) = tokio::select! {
        biased;
        _ = cancel => {
            // The controller allows 2s before killing this validator. Escalate
            // its child well inside that grace, retaining ownership until reaped.
            process.stop_with_grace(Duration::from_millis(250)).await;
            (None, false, true)
        },
        status = process.child.wait() => (status?.code(), false, false),
        _ = tokio::time::sleep(timeout) => {
            if controlled { process.stop_with_grace(Duration::from_millis(250)).await; }
            else { process.stop().await; }
            (None, true, false)
        },
    };
    Ok((
        Outcome {
            exit_code,
            timed_out,
            elapsed_ms: start.elapsed().as_millis(),
        },
        cancelled,
    ))
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
    logged_for(
        program,
        args,
        cwd,
        agent_dir,
        output,
        Duration::from_secs(timeout_secs),
    )
    .await
}

pub(crate) async fn logged_for(
    program: &Path,
    args: &[String],
    cwd: &Path,
    agent_dir: &Path,
    output: &Path,
    timeout: Duration,
) -> Result<Outcome> {
    logged_inner(program, args, cwd, agent_dir, output, timeout, None).await
}

pub(crate) async fn logged_cancellable(
    program: &Path,
    args: &[String],
    cwd: &Path,
    agent_dir: &Path,
    output: &Path,
    timeout_secs: u64,
    cancellation: &mut Cancellation,
) -> Result<Outcome> {
    ensure!(
        !cancellation.requested,
        "independent validation was already cancelled"
    );
    logged_inner(
        program,
        args,
        cwd,
        agent_dir,
        output,
        Duration::from_secs(timeout_secs),
        Some(cancellation),
    )
    .await
}

async fn logged_inner(
    program: &Path,
    args: &[String],
    cwd: &Path,
    agent_dir: &Path,
    output: &Path,
    timeout: Duration,
    cancellation: Option<&mut Cancellation>,
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
    let (result, cancelled) = wait_owned(&mut process, start, timeout, cancellation).await?;
    super::write_json(output.join("result.json"), &result)?;
    if cancelled {
        let reaped = process.child.try_wait()?.is_some();
        super::write_json(
            output.join("cancellation.json"),
            &serde_json::json!({
                "cancelled":true, "direct_child_reaped":reaped,
                "unix_group_kill_requested":cfg!(unix)
            }),
        )?;
        ensure!(
            reaped,
            "cancelled validator child could not be confirmed reaped"
        );
        anyhow::bail!(
            "independent validation cancelled after stopping and reaping owned child {}",
            process.pid
        );
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "subprocess fixture, invoked explicitly by the cancellation lifecycle test"]
    fn cancellation_child() {
        let Some(ready) = std::env::var_os("FERRUM_TEST_CANCELLATION_READY") else {
            return;
        };
        #[cfg(unix)]
        unsafe {
            libc::signal(libc::SIGTERM, libc::SIG_IGN);
            if std::env::var_os("FERRUM_TEST_IGNORE_INTERRUPT").is_some() {
                libc::signal(libc::SIGINT, libc::SIG_IGN);
            }
        }
        fs::write(ready, std::process::id().to_string()).unwrap();
        std::thread::sleep(Duration::from_secs(60));
    }

    #[cfg(unix)]
    #[tokio::test]
    async fn interrupt_cleanup_reaps_both_cooperative_and_signal_ignoring_children() {
        for ignores_interrupt in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let ready = dir.path().join("ready");
            let mut command = Command::new(std::env::current_exe().unwrap());
            isolated_local(&mut command);
            command
                .args([
                    "--exact",
                    "process::tests::cancellation_child",
                    "--ignored",
                    "--nocapture",
                ])
                .env("FERRUM_TEST_CANCELLATION_READY", &ready)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null());
            if ignores_interrupt {
                command.env("FERRUM_TEST_IGNORE_INTERRUPT", "1");
            }
            let mut child = ManagedChild::spawn(&mut command).unwrap();
            let ready_result = tokio::time::timeout(Duration::from_secs(5), async {
                while !ready.exists() {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await;
            if ready_result.is_err() {
                child.stop().await;
                panic!("fixture did not become ready");
            }
            assert_eq!(fs::read_to_string(&ready).unwrap(), child.pid.to_string());
            let result =
                tokio::time::timeout(Duration::from_secs(6), child.interrupt_then_stop()).await;
            if result.is_err() {
                child.stop().await;
            }
            assert!(result.is_ok());
            assert!(!child
                .child
                .try_wait()
                .unwrap()
                .expect("child must be reaped before return")
                .success());
        }
    }

    #[tokio::test]
    async fn controller_cancellation_reaps_a_running_child_before_returning() {
        let dir = tempfile::tempdir().unwrap();
        let ready = dir.path().join("ready");
        let mut command = Command::new(std::env::current_exe().unwrap());
        isolated(&mut command, dir.path());
        command
            .args([
                "--exact",
                "process::tests::cancellation_child",
                "--ignored",
                "--nocapture",
            ])
            .env("FERRUM_TEST_CANCELLATION_READY", &ready)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        let start = Instant::now();
        let mut child = ManagedChild::spawn(&mut command).unwrap();
        let ready_result = tokio::time::timeout(Duration::from_secs(5), async {
            while !ready.exists() {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await;
        if ready_result.is_err() {
            child.stop().await;
            panic!("child did not reach its live cancellation fixture");
        }
        assert!(child.child.try_wait().unwrap().is_none());
        let (sender, receiver) = tokio::sync::oneshot::channel();
        let mut cancellation = Cancellation {
            source: CancellationSource::Test(receiver),
            requested: false,
        };
        sender.send(()).unwrap();
        let (outcome, cancelled) = wait_owned(
            &mut child,
            start,
            Duration::from_secs(30),
            Some(&mut cancellation),
        )
        .await
        .unwrap();
        assert!(cancelled && cancellation.requested);
        assert!(!outcome.timed_out);
        let status = child
            .child
            .try_wait()
            .unwrap()
            .expect("cancelled child must already be reaped");
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            assert_eq!(status.signal(), Some(libc::SIGKILL));
        }
        #[cfg(not(unix))]
        assert!(!status.success());
    }
}
