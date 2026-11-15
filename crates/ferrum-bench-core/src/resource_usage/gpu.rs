//! A probe owns its child until it has exited. A timed-out driver query must
//! neither hold up benchmark completion nor accumulate replacement processes.
use super::{nvidia_memory_csv, GpuIdentity};
use std::{process::Stdio, time::Duration};
use tokio::{
    io::AsyncReadExt,
    process::{Child, Command},
    sync::Mutex,
};

#[derive(Default)]
pub struct GpuMemorySampler {
    child: Mutex<Option<Child>>,
}

impl GpuMemorySampler {
    pub async fn query(
        &self,
        gpu_indices: &[u32],
        timeout: Duration,
    ) -> Result<(Vec<GpuIdentity>, u64), String> {
        if gpu_indices.is_empty() || timeout.is_zero() {
            return Err("GPU probe needs explicit indices and a positive deadline".into());
        }
        let indices = gpu_indices
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let mut command = Command::new("nvidia-smi");
        command.args([
            "--query-gpu=index,uuid,memory.used",
            "--format=csv,noheader,nounits",
            "--id",
            &indices,
        ]);
        self.query_command(gpu_indices, timeout, command).await
    }

    async fn query_command(
        &self,
        gpu_indices: &[u32],
        timeout: Duration,
        mut command: Command,
    ) -> Result<(Vec<GpuIdentity>, u64), String> {
        let mut active = self
            .child
            .try_lock()
            .map_err(|_| "another GPU probe is in progress".to_string())?;
        if let Some(child) = active.as_mut() {
            match child.try_wait() {
                Ok(Some(_)) => *active = None,
                Ok(None) => {
                    child
                        .start_kill()
                        .map_err(|error| format!("nvidia-smi pending cancellation: {error}"))?;
                    return Err("previous nvidia-smi query is still retiring; probe skipped".into());
                }
                Err(error) => return Err(format!("nvidia-smi pending retirement: {error}")),
            }
        }
        *active = Some(
            command
                .stdin(Stdio::null())
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .kill_on_drop(true)
                .spawn()
                .map_err(|error| format!("nvidia-smi launch: {error}"))?,
        );
        let child = active.as_mut().expect("spawned probe remains owned");
        let operation = async {
            let stdout = child
                .stdout
                .take()
                .ok_or_else(|| "nvidia-smi stdout unavailable".to_string())?;
            let mut bytes = Vec::new();
            stdout
                .take(64 * 1024 + 1)
                .read_to_end(&mut bytes)
                .await
                .map_err(|error| error.to_string())?;
            if bytes.len() > 64 * 1024 {
                return Err("nvidia-smi output exceeds 64 KiB".to_string());
            }
            let status = child.wait().await.map_err(|error| error.to_string())?;
            if !status.success() {
                return Err(format!("nvidia-smi exited with {status}"));
            }
            let text = std::str::from_utf8(&bytes)
                .map_err(|_| "nvidia-smi output is not UTF-8".to_string())?;
            nvidia_memory_csv(text, gpu_indices).map_err(|error| error.to_string())
        };
        let result = tokio::time::timeout(timeout, operation)
            .await
            .unwrap_or_else(|_| Err("nvidia-smi query timed out".into()));
        match result {
            Ok(value) => {
                *active = None;
                Ok(value)
            }
            Err(error) => {
                // kill().await also waits for process exit. A stuck kernel
                // query may not exit promptly even after SIGKILL. Keep this
                // exact owner and poll retirement before another probe starts.
                match child.start_kill() {
                    Ok(()) => Err(error),
                    Err(kill) => Err(format!("{error}; cancellation request: {kill}")),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHILD_MODE: &str = "FERRUM_TEST_PENDING_GPU_QUERY";

    #[test]
    fn pending_query_child() {
        if std::env::var_os(CHILD_MODE).is_some() {
            loop {
                std::thread::park();
            }
        }
    }

    fn pending_command() -> Command {
        let mut command = Command::new(std::env::current_exe().unwrap());
        let (_, module) = module_path!().split_once("::").unwrap();
        command
            .arg("--exact")
            .arg(format!("{module}::pending_query_child"))
            .arg("--nocapture")
            .env(CHILD_MODE, "1");
        command
    }

    async fn reap(sampler: &GpuMemorySampler) {
        let mut active = sampler.child.lock().await;
        if let Some(child) = active.as_mut() {
            child.start_kill().unwrap();
            tokio::time::timeout(Duration::from_secs(5), child.wait())
                .await
                .expect("test process must retire")
                .unwrap();
        }
        *active = None;
    }

    #[tokio::test]
    async fn timeout_returns_with_probe_owned_until_reaped() {
        let sampler = GpuMemorySampler::default();
        let result = tokio::time::timeout(
            Duration::from_secs(5),
            sampler.query_command(&[0], Duration::from_millis(100), pending_command()),
        )
        .await
        .expect("query timeout must bound the observation");
        assert!(result.unwrap_err().contains("query timed out"));
        assert!(sampler.child.lock().await.is_some());
        reap(&sampler).await;
    }

    #[tokio::test]
    async fn unfinished_probe_prevents_a_replacement_process() {
        let child = pending_command()
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()
            .unwrap();
        let sampler = GpuMemorySampler {
            child: Mutex::new(Some(child)),
        };
        let result = sampler
            .query_command(
                &[0],
                Duration::from_millis(100),
                Command::new("a-replacement-probe-must-not-be-launched"),
            )
            .await;
        assert!(result
            .unwrap_err()
            .contains("still retiring; probe skipped"));
        assert!(sampler.child.lock().await.is_some());
        reap(&sampler).await;
    }
}
