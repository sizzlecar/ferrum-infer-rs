//! Local, bounded observations. No GPU state or server state is changed.
use super::Result;
use clap::Args;
use ferrum_bench_core::resource_usage::{
    timing::{milliseconds, PhaseCadence, ResourceQueryTiming},
    GpuIdentity, GpuMemorySampler, ProcessIdentity, ProcessSampler, ProcessSamplingConfig,
    SampledMemory,
};
use serde::Serialize;
use std::{
    path::PathBuf,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tokio::{sync::oneshot, task::JoinHandle};

#[derive(Debug, Clone, Args, Serialize)]
pub struct ResourceArguments {
    /// Local Linux server PID; include all descendants. Run client on the server host.
    #[arg(long)]
    pub resource_pid: Option<u32>,
    /// Physical NVIDIA index; whole-device memory, including driver and other processes.
    #[arg(long)]
    pub resource_gpu_index: Vec<u32>,
    #[arg(long, default_value_t = 200)]
    pub sample_interval_ms: u64,
    #[arg(long, default_value_t = 2000)]
    pub resource_query_timeout_ms: u64,
    #[arg(long, default_value_t = 4096)]
    pub resource_maximum_processes: usize,
    #[arg(long, default_value_t = 65_536)]
    pub resource_maximum_tree_nodes: usize,
    #[arg(long, default_value_t = 16 * 1024 * 1024)]
    pub resource_maximum_proc_input_bytes: usize,
}

impl ResourceArguments {
    fn process_config(&self) -> ProcessSamplingConfig {
        ProcessSamplingConfig {
            maximum_processes: self.resource_maximum_processes,
            maximum_tree_nodes: self.resource_maximum_tree_nodes,
            maximum_proc_input_bytes: self.resource_maximum_proc_input_bytes,
            ..Default::default()
        }
    }

    pub fn validate(&self, base_url: &str) -> Result<()> {
        if !(10..=60_000).contains(&self.sample_interval_ms)
            || !(100..=10_000).contains(&self.resource_query_timeout_ms)
            || !(1..=16_384).contains(&self.resource_maximum_processes)
            || !(1..=262_144).contains(&self.resource_maximum_tree_nodes)
            || !(1024..=64 * 1024 * 1024).contains(&self.resource_maximum_proc_input_bytes)
        {
            return Err("invalid resource sampling interval, deadline or traversal bounds".into());
        }
        let unique: std::collections::BTreeSet<_> = self.resource_gpu_index.iter().collect();
        if unique.len() != self.resource_gpu_index.len() || unique.len() > 16 {
            return Err("GPU indices must be distinct, at most 16".into());
        }
        if let Some(pid) = self.resource_pid {
            if pid <= 1 || pid > i32::MAX as u32 || !cfg!(target_os = "linux") {
                return Err("resource-pid requires a local Linux PID in 2..=i32::MAX".into());
            }
        }
        if self.enabled() {
            let url = reqwest::Url::parse(base_url)?;
            if !url.host_str().is_some_and(|host| {
                host == "localhost"
                    || host
                        .trim_matches(['[', ']'])
                        .parse::<std::net::IpAddr>()
                        .is_ok_and(|ip| ip.is_loopback())
            }) {
                return Err("resource sampling requires explicit loopback URL on the server machine, not a remote or tunneled service".into());
            }
        }
        Ok(())
    }
    fn enabled(&self) -> bool {
        self.resource_pid.is_some() || !self.resource_gpu_index.is_empty()
    }
}

pub struct ServerSampler {
    config: ResourceArguments,
    process: Option<Mutex<ProcessSampler>>,
    gpu_identity: Mutex<Option<Vec<GpuIdentity>>>,
    gpu: GpuMemorySampler,
    #[cfg(test)]
    process_delay_ms: std::sync::atomic::AtomicU64,
    #[cfg(test)]
    process_started: tokio::sync::Notify,
}

impl ServerSampler {
    pub fn new(config: &ResourceArguments) -> Option<Arc<Self>> {
        config.enabled().then(|| {
            Arc::new(Self {
                config: config.clone(),
                process: config
                    .resource_pid
                    .map(|pid| Mutex::new(ProcessSampler::new(PathBuf::from("/proc"), pid))),
                gpu_identity: Mutex::new(None),
                gpu: GpuMemorySampler::default(),
                #[cfg(test)]
                process_delay_ms: std::sync::atomic::AtomicU64::new(0),
                #[cfg(test)]
                process_started: tokio::sync::Notify::new(),
            })
        })
    }

    async fn observe(
        self: &Arc<Self>,
        report: &mut ResourceReport,
        start: Instant,
        periodic: bool,
    ) {
        report.attempted_samples += 1;
        let sampler = self.clone();
        let process = async move {
            if sampler.process.is_none() {
                return None;
            }
            let began = Instant::now();
            #[cfg(test)]
            {
                let delay = sampler
                    .process_delay_ms
                    .load(std::sync::atomic::Ordering::SeqCst);
                if delay > 0 {
                    sampler.process_started.notify_one();
                    tokio::time::sleep(Duration::from_millis(delay)).await;
                }
            }
            let value = tokio::task::spawn_blocking(move || {
                let mut process = sampler
                    .process
                    .as_ref()
                    .unwrap()
                    .lock()
                    .map_err(|_| "process sampler poisoned".to_owned())?;
                let identity = process.identity();
                Ok::<_, String>((identity, process.sample(&sampler.config.process_config())))
            })
            .await
            .map_err(|error| error.to_string())
            .and_then(|value| value);
            Some((value, began, Instant::now()))
        };
        let gpu = async {
            if self.config.resource_gpu_index.is_empty() {
                return None;
            }
            let began = Instant::now();
            let result = self
                .gpu
                .query(
                    &self.config.resource_gpu_index,
                    Duration::from_millis(self.config.resource_query_timeout_ms),
                )
                .await;
            Some((result, began, Instant::now()))
        };
        let (process, gpu) = tokio::join!(process, gpu);
        let interval = periodic.then(|| Duration::from_millis(self.config.sample_interval_ms));
        if let Some((result, began, finished)) = process {
            let result = result.and_then(|(identity, value)| {
                report.root_identity = identity;
                value
            });
            record(
                report.host.as_mut().unwrap(),
                report.host_query_timing.as_mut().unwrap(),
                &result,
                start,
                began,
                finished,
                interval,
            );
            if let Err(error) = result {
                report.error(format!("process: {error}"));
            }
        }
        if let Some((result, began, finished)) = gpu {
            let result = result.and_then(|(identities, bytes)| {
                let mut pinned = self
                    .gpu_identity
                    .lock()
                    .map_err(|_| "GPU identity poisoned".to_owned())?;
                if pinned.as_ref().is_some_and(|old| old != &identities) {
                    return Err("physical GPU index/UUID mapping changed".into());
                }
                *pinned = Some(identities.clone());
                report.gpu_identities = identities;
                Ok(bytes)
            });
            record(
                report.device.as_mut().unwrap(),
                report.device_query_timing.as_mut().unwrap(),
                &result,
                start,
                began,
                finished,
                interval,
            );
            if let Err(error) = result {
                report.error(format!("GPU: {error}"));
            }
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct ResourceReport {
    pub config: ResourceArguments,
    pub root_identity: Option<ProcessIdentity>,
    pub gpu_identities: Vec<GpuIdentity>,
    pub host: Option<SampledMemory>,
    pub device: Option<SampledMemory>,
    pub host_query_timing: Option<ResourceQueryTiming>,
    pub device_query_timing: Option<ResourceQueryTiming>,
    pub attempted_samples: u64,
    pub skipped_intervals: u64,
    pub overlong_periodic_observations: u64,
    pub workload_start_offset_ms: f64,
    pub workload_finish_offset_ms: Option<f64>,
    pub duration_ms: f64,
    pub complete: bool,
    pub errors: Vec<String>,
    pub omitted_errors: u64,
    pub scope: &'static str,
}

impl ResourceReport {
    fn error(&mut self, error: String) {
        if self.errors.len() < 8 {
            self.errors.push(error.chars().take(512).collect());
        } else {
            self.omitted_errors += 1;
        }
    }
    fn finish(&mut self, elapsed: Duration) {
        self.duration_ms = milliseconds(elapsed);
        self.complete = self.errors.is_empty()
            && self.omitted_errors == 0
            && self.skipped_intervals == 0
            && self.overlong_periodic_observations == 0
            && self.host.as_ref().is_none_or(SampledMemory::complete)
            && self.device.as_ref().is_none_or(SampledMemory::complete);
    }
}

#[allow(clippy::too_many_arguments)]
fn record(
    memory: &mut SampledMemory,
    timing: &mut ResourceQueryTiming,
    value: &std::result::Result<u64, String>,
    epoch: Instant,
    began: Instant,
    finished: Instant,
    interval: Option<Duration>,
) {
    let bytes = value.as_ref().ok().copied();
    let peak =
        bytes.is_some_and(|bytes| memory.observed_peak_bytes.is_none_or(|prior| bytes > prior));
    memory.record(bytes, milliseconds(began.saturating_duration_since(epoch)));
    timing.record(epoch, began, finished, peak, interval);
}

pub struct RunningWindow {
    stop: Option<oneshot::Sender<Instant>>,
    task: Option<JoinHandle<ResourceReport>>,
    fallback: ResourceReport,
    start: Instant,
}

impl RunningWindow {
    pub async fn begin(sampler: Arc<ServerSampler>) -> Self {
        let start = Instant::now();
        let mut report = ResourceReport {
            config: sampler.config.clone(), root_identity: None, gpu_identities: Vec::new(),
            host: sampler.process.as_ref().map(|_| SampledMemory::default()),
            device: (!sampler.config.resource_gpu_index.is_empty()).then(SampledMemory::default),
            host_query_timing: sampler.process.as_ref().map(|_| ResourceQueryTiming::default()),
            device_query_timing: (!sampler.config.resource_gpu_index.is_empty()).then(ResourceQueryTiming::default),
            attempted_samples: 0, skipped_intervals: 0, overlong_periodic_observations: 0,
            workload_start_offset_ms: 0.0, workload_finish_offset_ms: None, duration_ms: 0.0,
            complete: false, errors: Vec::new(), omitted_errors: 0,
            scope: "local server only; initial/final samples bracket dispatch through artifact receipt and exclude media validation; periodic queries compete with generation and can affect timing; all peaks are sampled lower bounds even when complete; process-tree VmRSS sums may double-count shared pages; GPU memory.used sums selected whole physical devices including drivers and other processes, not server allocator bytes; query brackets do not establish atomic measurement instants; at most 8 errors of 512 characters, no raw sample series retained",
        };
        sampler.observe(&mut report, start, false).await;
        let began = Instant::now();
        report.workload_start_offset_ms = milliseconds(began.duration_since(start));
        let fallback = report.clone();
        let (stop, mut receiver) = oneshot::channel();
        let task = tokio::spawn(async move {
            let interval = Duration::from_millis(sampler.config.sample_interval_ms);
            let mut cadence = PhaseCadence::new(began, interval);
            let finished = loop {
                let observed_at = tokio::select! {
                    biased;
                    stopped = &mut receiver => {
                        break stopped.unwrap_or_else(|_| { report.error("resource stop sender disappeared".into()); Instant::now() });
                    }
                    _ = tokio::time::sleep_until(cadence.due().into()) => Instant::now(),
                };
                sampler.observe(&mut report, start, true).await;
                let observation_finished = Instant::now();
                let stopped = match receiver.try_recv() {
                    Ok(instant) => Some(instant),
                    Err(oneshot::error::TryRecvError::Empty) => None,
                    Err(oneshot::error::TryRecvError::Closed) => {
                        report.error("resource stop sender disappeared".into());
                        Some(observation_finished)
                    }
                };
                if stopped.is_none_or(|end| observed_at < end) {
                    cadence.observed(observed_at);
                    if observation_finished.duration_since(observed_at) > interval {
                        report.overlong_periodic_observations += 1;
                    }
                }
                if let Some(end) = stopped {
                    break end;
                }
            };
            report.workload_finish_offset_ms = Some(milliseconds(finished.duration_since(start)));
            report.skipped_intervals = cadence.finish(finished);
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

    pub async fn finish(mut self) -> ResourceReport {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(Instant::now());
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

impl Drop for RunningWindow {
    fn drop(&mut self) {
        if let Some(task) = &self.task {
            task.abort();
        }
    }
}

#[cfg(test)]
#[path = "resource_tests.rs"]
mod tests;
