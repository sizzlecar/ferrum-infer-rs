//! Optional local observation. Sampling never signals the server or changes GPU state.
use super::*;
use ferrum_bench_core::images::resource_usage::{
    nvidia_memory_csv, GpuIdentity, ProcessIdentity, SampledMemory,
};
use std::{
    process::Stdio,
    sync::{Arc, Mutex},
    time::Instant,
};
use tokio::{io::AsyncReadExt, sync::oneshot, task::JoinHandle};
mod procfs;
mod timing;
pub use timing::ImageResourceQueryTiming;
use timing::{milliseconds, PhaseCadence};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum ImageProcessScope {
    Process,
    ProcessTree,
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageResourceSamplingConfig {
    /// Empty disables RSS; otherwise exactly one local root PID per endpoint.
    pub server_pids: Vec<u32>,
    pub process_scope: ImageProcessScope,
    /// Physical NVIDIA indices; memory.used includes other processes and drivers.
    pub gpu_indices: Vec<u32>,
    pub interval_ms: u64,
    pub maximum_processes: usize,
    pub maximum_tree_nodes: usize,
    pub maximum_proc_input_bytes: usize,
    pub gpu_query_timeout_ms: u64,
}
impl Default for ImageResourceSamplingConfig {
    fn default() -> Self {
        Self {
            server_pids: Vec::new(),
            process_scope: ImageProcessScope::ProcessTree,
            gpu_indices: Vec::new(),
            interval_ms: 200,
            maximum_processes: 4096,
            maximum_tree_nodes: 65_536,
            maximum_proc_input_bytes: 16 * 1024 * 1024,
            gpu_query_timeout_ms: 2000,
        }
    }
}
impl ImageResourceSamplingConfig {
    pub(super) fn validate(&self, endpoints: &[ImageBenchmarkEndpoint]) -> anyhow::Result<()> {
        ensure!(
            !self.server_pids.is_empty() || !self.gpu_indices.is_empty(),
            "resource sampling needs a server PID or GPU index"
        );
        ensure!(
            self.server_pids.is_empty() || self.server_pids.len() == endpoints.len(),
            "provide one resource-server-pid per endpoint"
        );
        ensure!(
            self.server_pids
                .iter()
                .all(|pid| *pid > 1 && *pid <= i32::MAX as u32),
            "resource PID must be within 2..=i32::MAX"
        );
        ensure!(
            self.server_pids.is_empty() || cfg!(target_os = "linux"),
            "process RSS sampling requires local Linux procfs"
        );
        ensure!(
            (10..=60_000).contains(&self.interval_ms),
            "resource sample interval must be 10..=60000 ms"
        );
        let unique = self
            .gpu_indices
            .iter()
            .collect::<std::collections::BTreeSet<_>>();
        ensure!(
            unique.len() == self.gpu_indices.len() && unique.len() <= 16,
            "GPU indices must be distinct, at most 16"
        );
        ensure!(
            (1..=16_384).contains(&self.maximum_processes)
                && (1..=262_144).contains(&self.maximum_tree_nodes),
            "invalid proc traversal limits"
        );
        ensure!(
            (1024..=64 * 1024 * 1024).contains(&self.maximum_proc_input_bytes),
            "invalid proc input byte bound"
        );
        ensure!(
            (100..=10_000).contains(&self.gpu_query_timeout_ms),
            "invalid GPU query timeout"
        );
        for endpoint in endpoints {
            for url in std::iter::once(endpoint.url.as_str()).chain(endpoint.edit_url.as_deref()) {
                let url = reqwest::Url::parse(url)?;
                ensure!(
                    url.host_str().is_some_and(|host| host == "localhost"
                        || host
                            .trim_matches(['[', ']'])
                            .parse::<std::net::IpAddr>()
                            .is_ok_and(|ip| ip.is_loopback())),
                    "resource sampling requires explicit loopback endpoints on this machine"
                );
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ImageResourceWindow {
    pub requested_server_pid: Option<u32>,
    pub requested_gpu_indices: Vec<u32>,
    pub root_identity: Option<ProcessIdentity>,
    pub gpu_identities: Vec<GpuIdentity>,
    pub process_scope: ImageProcessScope,
    pub interval_ms: u64,
    pub started_unix_ms: Option<u128>,
    pub duration_ms: f64,
    /// Relative to the resource window, after baseline and before final probe.
    pub phase_started_offset_ms: Option<f64>,
    pub phase_finished_offset_ms: Option<f64>,
    pub attempted_samples: u64,
    pub skipped_intervals: u64,
    /// Periodic observations taking longer than the declared interval cannot
    /// establish that cadence, including the last query crossing phase end.
    pub overlong_periodic_observations: u64,
    pub maximum_sample_gap_ms: f64,
    pub first_sample_offset_ms: Option<f64>,
    pub last_sample_offset_ms: Option<f64>,
    pub host: Option<SampledMemory>,
    pub device: Option<SampledMemory>,
    pub host_query_timing: Option<ImageResourceQueryTiming>,
    pub device_query_timing: Option<ImageResourceQueryTiming>,
    /// Complete means requested observations succeeded without missed intervals;
    /// it never means an instantaneous peak was captured.
    pub complete: bool,
    pub errors: Vec<String>,
    pub omitted_errors: u64,
    pub scope: &'static str,
}
impl ImageResourceWindow {
    fn error(&mut self, message: String) {
        if self.errors.len() < 8 {
            self.errors.push(message.chars().take(512).collect());
        } else {
            self.omitted_errors += 1;
        }
    }
    fn finish(&mut self, elapsed: Duration) {
        self.duration_ms = elapsed.as_secs_f64() * 1000.0;
        self.complete = self.errors.is_empty()
            && self.omitted_errors == 0
            && self.skipped_intervals == 0
            && self.overlong_periodic_observations == 0
            && self.host.as_ref().is_none_or(SampledMemory::complete)
            && self.device.as_ref().is_none_or(SampledMemory::complete);
    }
}

pub(super) struct EndpointSampler {
    config: ImageResourceSamplingConfig,
    server_pid: Option<u32>,
    process: Option<Mutex<procfs::ProcessSampler>>,
    gpu_identity: Mutex<Option<Vec<GpuIdentity>>>,
    #[cfg(test)]
    process_query_delay_ms: std::sync::atomic::AtomicU64,
    #[cfg(test)]
    process_query_started: tokio::sync::Notify,
}
impl EndpointSampler {
    pub(super) fn new(config: &ImageResourceSamplingConfig, endpoint: usize) -> Arc<Self> {
        Arc::new(Self {
            config: config.clone(),
            server_pid: config.server_pids.get(endpoint).copied(),
            process: config
                .server_pids
                .get(endpoint)
                .map(|&pid| Mutex::new(procfs::ProcessSampler::new(PathBuf::from("/proc"), pid))),
            gpu_identity: Mutex::new(None),
            #[cfg(test)]
            process_query_delay_ms: std::sync::atomic::AtomicU64::new(0),
            #[cfg(test)]
            process_query_started: tokio::sync::Notify::new(),
        })
    }
    async fn observe(
        self: &Arc<Self>,
        report: &mut ImageResourceWindow,
        start: Instant,
        periodic: bool,
    ) {
        let offset = start.elapsed().as_secs_f64() * 1000.0;
        report.attempted_samples += 1;
        if let Some(last) = report.last_sample_offset_ms {
            report.maximum_sample_gap_ms = report.maximum_sample_gap_ms.max(offset - last);
        }
        report.first_sample_offset_ms.get_or_insert(offset);
        report.last_sample_offset_ms = Some(offset);
        let sampler = Arc::clone(self);
        let process = async move {
            if sampler.process.is_none() {
                return None;
            }
            let query_started = Instant::now();
            #[cfg(test)]
            {
                let delay = sampler
                    .process_query_delay_ms
                    .load(std::sync::atomic::Ordering::SeqCst);
                if delay > 0 {
                    sampler.process_query_started.notify_one();
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                }
            }
            let result = tokio::task::spawn_blocking(move || {
                let mut process = sampler
                    .process
                    .as_ref()
                    .unwrap()
                    .lock()
                    .map_err(|_| "process sampler is poisoned".to_string())?;
                let identity = process.identity();
                Ok::<_, String>((identity, process.sample(&sampler.config)))
            })
            .await
            .map_err(|error| format!("process sampler task: {error}"))
            .and_then(|value| value);
            Some((result, query_started, Instant::now()))
        };
        let gpu = async {
            if self.config.gpu_indices.is_empty() {
                None
            } else {
                let query_started = Instant::now();
                let result = query_gpu(&self.config).await;
                Some((result, query_started, Instant::now()))
            }
        };
        let (process, gpu) = tokio::join!(process, gpu);
        if let Some((process, query_started, query_finished)) = process {
            let value = match process {
                Ok((identity, sample)) => {
                    report.root_identity = identity;
                    sample
                }
                Err(error) => Err(error),
            };
            record_query(
                report.host.as_mut().unwrap(),
                report.host_query_timing.as_mut().unwrap(),
                value.as_ref().ok().copied(),
                start,
                query_started,
                query_finished,
                periodic.then(|| Duration::from_millis(self.config.interval_ms)),
            );
            if let Err(error) = value {
                report.error(format!("process: {error}"));
            }
        }
        if let Some((gpu, query_started, query_finished)) = gpu {
            let value = gpu.and_then(|(identities, bytes)| {
                let mut expected = self
                    .gpu_identity
                    .lock()
                    .map_err(|_| "GPU identity is poisoned".to_string())?;
                if expected
                    .as_ref()
                    .is_some_and(|expected| *expected != identities)
                {
                    return Err("physical GPU index/UUID mapping changed".into());
                }
                *expected = Some(identities.clone());
                report.gpu_identities = identities;
                Ok(bytes)
            });
            record_query(
                report.device.as_mut().unwrap(),
                report.device_query_timing.as_mut().unwrap(),
                value.as_ref().ok().copied(),
                start,
                query_started,
                query_finished,
                periodic.then(|| Duration::from_millis(self.config.interval_ms)),
            );
            if let Err(error) = value {
                report.error(format!("GPU: {error}"));
            }
        }
    }
}

fn record_query(
    memory: &mut SampledMemory,
    timing: &mut ImageResourceQueryTiming,
    value: Option<u64>,
    epoch: Instant,
    started: Instant,
    finished: Instant,
    periodic_interval: Option<Duration>,
) {
    let new_peak =
        value.is_some_and(|bytes| memory.observed_peak_bytes.is_none_or(|prior| bytes > prior));
    memory.record(
        value,
        milliseconds(started.saturating_duration_since(epoch)),
    );
    timing.record(epoch, started, finished, new_peak, periodic_interval);
}

async fn query_gpu(
    config: &ImageResourceSamplingConfig,
) -> Result<(Vec<GpuIdentity>, u64), String> {
    let indices = config
        .gpu_indices
        .iter()
        .map(u32::to_string)
        .collect::<Vec<_>>()
        .join(",");
    let mut child = tokio::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=index,uuid,memory.used",
            "--format=csv,noheader,nounits",
            "--id",
            &indices,
        ])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .kill_on_drop(true)
        .spawn()
        .map_err(|error| format!("nvidia-smi launch: {error}"))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "nvidia-smi stdout unavailable".to_string())?;
    let operation = async {
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
        nvidia_memory_csv(text, &config.gpu_indices).map_err(|error| error.to_string())
    };
    match tokio::time::timeout(
        Duration::from_millis(config.gpu_query_timeout_ms),
        operation,
    )
    .await
    {
        Ok(Ok(value)) => Ok(value),
        other => {
            let _ = child.kill().await;
            match other {
                Ok(Err(error)) => Err(error),
                _ => Err("nvidia-smi query timed out".into()),
            }
        }
    }
}

pub(super) struct RunningResourceWindow {
    stop: Option<oneshot::Sender<Instant>>,
    task: Option<JoinHandle<ImageResourceWindow>>,
    fallback: ImageResourceWindow,
    start: Instant,
}
impl RunningResourceWindow {
    /// Initial sample completes before the first request; final sample follows
    /// the final validation/capture. Only those two boundary probes are outside
    /// request timers; periodic probes compete for resources during requests.
    pub(super) async fn begin(sampler: Arc<EndpointSampler>) -> Self {
        let start = Instant::now();
        let mut report = ImageResourceWindow {
            requested_server_pid: sampler.server_pid,
            requested_gpu_indices: sampler.config.gpu_indices.clone(),
            root_identity: None, gpu_identities: Vec::new(), process_scope: sampler.config.process_scope,
            interval_ms: sampler.config.interval_ms, started_unix_ms: super::started_unix_ms().ok(),
            duration_ms: 0.0, phase_started_offset_ms: None, phase_finished_offset_ms: None,
            attempted_samples: 0, skipped_intervals: 0, overlong_periodic_observations: 0,
            maximum_sample_gap_ms: 0.0,
            first_sample_offset_ms: None, last_sample_offset_ms: None,
            host: sampler.process.as_ref().map(|_| SampledMemory::default()),
            device: (!sampler.config.gpu_indices.is_empty()).then(SampledMemory::default), complete: false,
            host_query_timing: sampler.process.as_ref().map(|_| ImageResourceQueryTiming::default()),
            device_query_timing: (!sampler.config.gpu_indices.is_empty()).then(ImageResourceQueryTiming::default),
            errors: Vec::new(), omitted_errors: 0,
            scope: "window brackets one phase/repetition: baseline before first dispatch, periodic observations through final response validation/capture, then final observation; cadence counts only deadlines within the phase after baseline and before finish was requested, including genuine holes caused by queries that finish later; boundary probes are outside request timers, periodic procfs/SMI probes compete for CPU/driver resources and can affect request latency and throughput; keep sampling settings identical across comparisons; all peaks are sampled lower bounds even when complete, peak offsets are query starts and per-source timing brackets the observation rather than claiming an exact measurement instant; query elapsed time includes task scheduling, process launch and driver/OS response, not isolated native driver time; RSS sums process VmRSS and may count shared pages multiple times; GPU sums whole-device memory.used for explicitly selected physical index/UUIDs including other processes and driver baseline; not allocator-only or per-server GPU usage; at most 8 errors of 512 characters, no raw sample series retained",
        };
        if report.started_unix_ms.is_none() {
            report.error("UNIX start timestamp unavailable".into());
        }
        sampler.observe(&mut report, start, false).await;
        let phase_started = Instant::now();
        report.phase_started_offset_ms = Some(milliseconds(phase_started.duration_since(start)));
        let fallback = report.clone();
        let (stop, mut stopped) = oneshot::channel();
        let task = tokio::spawn(async move {
            let interval = Duration::from_millis(sampler.config.interval_ms);
            let mut cadence = PhaseCadence::new(phase_started, interval);
            let phase_finished = loop {
                let observed_at = tokio::select! {
                    biased;
                    stopped_at = &mut stopped => {
                        break stopped_at.unwrap_or_else(|_| {
                            report.error("resource window stop signal disappeared".into());
                            Instant::now()
                        });
                    }
                    _ = tokio::time::sleep_until(cadence.due().into()) => Instant::now(),
                };
                sampler.observe(&mut report, start, true).await;
                let observation_finished = Instant::now();
                // Finish may have arrived during this query. Its original
                // timestamp limits cadence, even though we await and preserve
                // the real query result rather than cancelling it.
                let stopped_at = match stopped.try_recv() {
                    Ok(finished) => Some(finished),
                    Err(oneshot::error::TryRecvError::Empty) => None,
                    Err(oneshot::error::TryRecvError::Closed) => {
                        report.error("resource window stop signal disappeared".into());
                        Some(observation_finished)
                    }
                };
                if stopped_at.is_none_or(|finished| observed_at < finished) {
                    cadence.observed(observed_at);
                    if observation_finished.duration_since(observed_at) > interval {
                        report.overlong_periodic_observations += 1;
                    }
                }
                if let Some(finished) = stopped_at {
                    break finished;
                }
            };
            report.phase_finished_offset_ms =
                Some(milliseconds(phase_finished.duration_since(start)));
            report.skipped_intervals = cadence.finish(phase_finished);
            sampler.observe(&mut report, start, false).await;
            report.finish(start.elapsed());
            report
        });
        Self {
            stop: Some(stop),
            task: Some(task),
            fallback,
            start,
        }
    }
    pub(super) async fn finish(mut self) -> ImageResourceWindow {
        let finished = Instant::now();
        self.fallback.phase_finished_offset_ms =
            Some(milliseconds(finished.duration_since(self.start)));
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(finished);
        }
        match self.task.take().unwrap().await {
            Ok(report) => report,
            Err(error) => {
                let mut report = self.fallback.clone();
                report.error(format!("resource sampler task failed: {error}"));
                report.finish(self.start.elapsed());
                report
            }
        }
    }
}
impl Drop for RunningResourceWindow {
    fn drop(&mut self) {
        if let Some(task) = &self.task {
            task.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fixture_sampler(path: &std::path::Path) -> Arc<EndpointSampler> {
        Arc::new(EndpointSampler {
            config: ImageResourceSamplingConfig {
                server_pids: vec![42],
                process_scope: ImageProcessScope::Process,
                interval_ms: 60_000,
                ..Default::default()
            },
            server_pid: Some(42),
            process: Some(Mutex::new(procfs::ProcessSampler::new(path.into(), 42))),
            gpu_identity: Mutex::new(None),
            process_query_delay_ms: std::sync::atomic::AtomicU64::new(0),
            process_query_started: tokio::sync::Notify::new(),
        })
    }
    fn process(path: &std::path::Path, rss: u64) {
        std::fs::create_dir_all(path.join("42")).unwrap();
        let mut fields: Vec<String> = vec!["S".into(), "1".into()];
        fields.extend((5..22).map(|_| "0".into()));
        fields.push("100".into());
        std::fs::write(
            path.join("42/stat"),
            format!("42 (worker) {}", fields.join(" ")),
        )
        .unwrap();
        std::fs::write(path.join("42/status"), format!("VmRSS: {rss} kB\n")).unwrap();
    }
    #[tokio::test]
    async fn independent_windows_keep_identity_and_observe_both_boundaries() {
        let temp = tempfile::tempdir().unwrap();
        process(temp.path(), 100);
        let sampler = fixture_sampler(temp.path());
        let warm = RunningResourceWindow::begin(Arc::clone(&sampler))
            .await
            .finish()
            .await;
        assert!(warm.complete);
        assert_eq!(warm.host.unwrap().observed_peak_bytes, Some(100 * 1024));
        process(temp.path(), 2);
        let measured = RunningResourceWindow::begin(Arc::clone(&sampler)).await;
        process(temp.path(), 7);
        let measured = measured.finish().await;
        assert!(measured.complete);
        assert_eq!(measured.attempted_samples, 2);
        assert_eq!(measured.host.unwrap().observed_peak_bytes, Some(7 * 1024));
        assert_eq!(warm.root_identity, measured.root_identity);
        assert!(measured.device.is_none());
    }
    #[tokio::test]
    async fn missing_process_is_explicit_and_errors_remain_bounded() {
        let temp = tempfile::tempdir().unwrap();
        let mut window = RunningResourceWindow::begin(fixture_sampler(temp.path()))
            .await
            .finish()
            .await;
        assert!(!window.complete);
        assert_eq!(window.requested_server_pid, Some(42));
        assert_eq!(window.root_identity, None);
        assert_eq!(window.host.as_ref().unwrap().observed_peak_bytes, None);
        assert_eq!(window.host.as_ref().unwrap().failed_samples, 2);
        for _ in 0..20 {
            window.error("x".repeat(1024));
        }
        assert_eq!(window.errors.len(), 8);
        assert!(window
            .errors
            .iter()
            .all(|error| error.chars().count() <= 512));
        assert!(window.omitted_errors > 0);
    }
    fn delayed_sampler(path: &std::path::Path) -> Arc<EndpointSampler> {
        let mut sampler = fixture_sampler(path);
        Arc::get_mut(&mut sampler).unwrap().config.interval_ms = 20;
        sampler
    }

    #[tokio::test]
    async fn slow_boundary_queries_do_not_create_pre_or_post_phase_missed_ticks() {
        let temp = tempfile::tempdir().unwrap();
        process(temp.path(), 7);
        let sampler = delayed_sampler(temp.path());
        sampler
            .process_query_delay_ms
            .store(60, std::sync::atomic::Ordering::SeqCst);
        let window = RunningResourceWindow::begin(sampler).await.finish().await;
        assert!(window.complete, "{window:?}");
        assert_eq!(window.skipped_intervals, 0);
        assert_eq!(window.overlong_periodic_observations, 0);
        assert_eq!(window.attempted_samples, 2);
        let started = window.phase_started_offset_ms.unwrap();
        let finished = window.phase_finished_offset_ms.unwrap();
        assert!(started >= 60.0);
        assert!(finished >= started);
        assert!(window.duration_ms >= finished + 60.0);
        let timing = window.host_query_timing.unwrap();
        assert_eq!(timing.queries, 2);
        assert_eq!(timing.periodic_queries, 0);
        assert_eq!(timing.overlong_periodic_queries, 0);
        assert!(timing.maximum_query_ms >= 60.0);
        assert_eq!(window.host.unwrap().observed_peak_bytes, Some(7 * 1024));
    }

    #[tokio::test]
    async fn successful_slow_periodic_query_still_fails_coverage_at_original_stop_time() {
        let temp = tempfile::tempdir().unwrap();
        process(temp.path(), 7);
        let sampler = delayed_sampler(temp.path());
        let window = RunningResourceWindow::begin(Arc::clone(&sampler)).await;
        sampler
            .process_query_delay_ms
            .store(160, std::sync::atomic::Ordering::SeqCst);
        tokio::time::timeout(
            Duration::from_secs(5),
            sampler.process_query_started.notified(),
        )
        .await
        .unwrap();
        // Only the already-started periodic query is delayed. The final probe
        // still performs a real procfs read, without an artificial delay.
        sampler
            .process_query_delay_ms
            .store(0, std::sync::atomic::Ordering::SeqCst);
        tokio::time::sleep(Duration::from_millis(45)).await;
        process(temp.path(), 11);
        let report = window.finish().await;
        assert!(!report.complete, "{report:?}");
        assert!(report.skipped_intervals >= 2);
        assert!(report.overlong_periodic_observations > 0);
        assert!(report.errors.is_empty());
        let host = report.host.unwrap();
        assert_eq!(host.failed_samples, 0);
        assert_eq!(host.observed_peak_bytes, Some(11 * 1024));
        let timing = report.host_query_timing.unwrap();
        assert!(timing.maximum_query_ms >= 160.0);
        assert!(timing.maximum_periodic_query_ms >= 160.0);
        assert!(timing.overlong_periodic_queries > 0);
        assert!(
            timing.peak_query_started_offset_ms.unwrap()
                < timing.peak_query_finished_offset_ms.unwrap()
        );
        assert!(report.phase_finished_offset_ms.unwrap() < report.duration_ms);
    }

    #[tokio::test]
    async fn slow_last_periodic_query_cannot_claim_complete_after_an_early_stop() {
        let temp = tempfile::tempdir().unwrap();
        process(temp.path(), 7);
        let sampler = delayed_sampler(temp.path());
        let window = RunningResourceWindow::begin(Arc::clone(&sampler)).await;
        sampler
            .process_query_delay_ms
            .store(80, std::sync::atomic::Ordering::SeqCst);
        tokio::time::timeout(
            Duration::from_secs(5),
            sampler.process_query_started.notified(),
        )
        .await
        .unwrap();
        sampler
            .process_query_delay_ms
            .store(0, std::sync::atomic::Ordering::SeqCst);
        let report = window.finish().await;
        assert!(!report.complete);
        assert!(report.overlong_periodic_observations > 0);
        assert_eq!(report.host.as_ref().unwrap().failed_samples, 0);
        // Even if no further periodic deadline fell before this early stop,
        // the long observation itself cannot establish the declared cadence.
        let mut no_later_deadlines = report;
        no_later_deadlines.skipped_intervals = 0;
        let elapsed = Duration::from_secs_f64(no_later_deadlines.duration_ms / 1000.0);
        no_later_deadlines.finish(elapsed);
        assert!(!no_later_deadlines.complete);
    }

    #[test]
    fn failed_source_query_retains_timing_without_creating_a_zero_peak() {
        let epoch = Instant::now();
        let mut memory = SampledMemory::default();
        let mut timing = ImageResourceQueryTiming::default();
        record_query(
            &mut memory,
            &mut timing,
            None,
            epoch,
            epoch,
            epoch + Duration::from_secs(10),
            Some(Duration::from_secs(2)),
        );
        assert!(!memory.complete());
        assert_eq!(memory.failed_samples, 1);
        assert_eq!(memory.observed_peak_bytes, None);
        assert_eq!(timing.queries, 1);
        assert_eq!(timing.maximum_query_ms, 10_000.0);
        assert_eq!(timing.peak_query_started_offset_ms, None);
        assert_eq!(timing.peak_query_finished_offset_ms, None);
    }
    #[test]
    fn optional_configuration_requires_explicit_local_mapping() {
        let endpoint = ImageBenchmarkEndpoint {
            url: "http://127.0.0.1:1/v1/images/generations".into(),
            edit_url: None,
            model: "test".into(),
            server_config_declaration: "test".into(),
        };
        let mut config = ImageResourceSamplingConfig::default();
        assert!(config.validate(std::slice::from_ref(&endpoint)).is_err());
        config.gpu_indices = vec![0];
        config.validate(std::slice::from_ref(&endpoint)).unwrap();
        config.gpu_indices = vec![0, 0];
        assert!(config.validate(std::slice::from_ref(&endpoint)).is_err());
        config.gpu_indices = vec![0];
        let mut remote = endpoint;
        remote.url = "http://example.com/v1/images/generations".into();
        assert!(config.validate(&[remote]).is_err());
    }
}
