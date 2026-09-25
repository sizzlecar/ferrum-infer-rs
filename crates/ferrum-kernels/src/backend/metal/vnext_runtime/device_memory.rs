//! Opt-in, bounded host sampling of the actual runtime's Metal device.
//! No GPU work, synchronization, model inference, or resource-size summation.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::{mpsc, Arc, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use ferrum_interfaces::vnext::DeviceMemoryTelemetrySnapshot;
use ferrum_types::{DeviceMemorySamplingConfig, DEVICE_MEMORY_SAMPLE_INTERVAL_MS};

use super::MetalDeviceRuntimeError;

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

struct SampleState {
    snapshot: DeviceMemoryTelemetrySnapshot,
    last_sample_ns: Option<u64>,
}

impl SampleState {
    fn observe(&mut self, elapsed_ns: u64, result: Result<u64, String>) {
        if let Some(previous) = self.last_sample_ns {
            if elapsed_ns < previous {
                self.error("device-memory monotonic clock moved backwards".to_owned());
                return;
            }
            self.snapshot.max_sample_gap_ns =
                self.snapshot.max_sample_gap_ns.max(elapsed_ns - previous);
        }
        self.snapshot.elapsed_ns = elapsed_ns;
        self.last_sample_ns = Some(elapsed_ns);
        match result {
            Ok(bytes) => {
                self.snapshot.current_allocated_bytes = bytes;
                self.snapshot.peak_allocated_bytes = self.snapshot.peak_allocated_bytes.max(bytes);
                self.snapshot.sample_count = self.snapshot.sample_count.saturating_add(1);
            }
            Err(message) => self.error(message),
        }
    }

    fn error(&mut self, message: String) {
        self.snapshot.error_count = self.snapshot.error_count.saturating_add(1);
        self.snapshot.last_error = Some(message);
        self.snapshot.complete = false;
    }
}

struct Worker {
    stop: mpsc::Sender<&'static str>,
    join: JoinHandle<()>,
}

pub(super) struct DeviceMemorySampler {
    state: Arc<Mutex<SampleState>>,
    worker: Mutex<Option<Worker>>,
}

impl DeviceMemorySampler {
    pub(super) fn start(
        device: metal::Device,
        config: &DeviceMemorySamplingConfig,
    ) -> Result<Self, MetalDeviceRuntimeError> {
        config
            .validate()
            .map_err(MetalDeviceRuntimeError::contract)?;
        let identity = (device.registry_id().to_string(), device.name().to_owned());
        Self::start_with_source(
            &config.jsonl_path,
            identity,
            Duration::from_millis(DEVICE_MEMORY_SAMPLE_INTERVAL_MS),
            move || Ok(device.current_allocated_size() as u64),
        )
        .map_err(MetalDeviceRuntimeError::contract)
    }

    fn start_with_source(
        path: &Path,
        identity: (String, String),
        interval: Duration,
        mut source: impl FnMut() -> Result<u64, String> + Send + 'static,
    ) -> Result<Self, String> {
        if interval.is_zero() {
            return Err("device-memory sample interval must be positive".to_owned());
        }
        // A file belongs to one runtime/device capture. Never merge another
        // model/runtime or a previous run into an apparent single peak.
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|error| format!("create device-memory JSONL {}: {error}", path.display()))?;
        let started_unix_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| format!("device-memory wall clock: {error}"))?
            .as_nanos()
            .to_string();
        let started = Instant::now();
        let state = Arc::new(Mutex::new(SampleState {
            snapshot: DeviceMemoryTelemetrySnapshot {
                schema_version: 1,
                record_type: "sample".to_owned(),
                source: "MTLDevice.currentAllocatedSize".to_owned(),
                scope: "process_metal_device".to_owned(),
                phase: "pre_weight_load_to_shutdown".to_owned(),
                pid: std::process::id(),
                device_registry_id: identity.0,
                device_name: identity.1,
                started_unix_ns,
                elapsed_ns: 0,
                current_allocated_bytes: 0,
                peak_allocated_bytes: 0,
                sample_count: 0,
                interval_ms: u64::try_from(interval.as_millis()).unwrap_or(u64::MAX),
                max_sample_gap_ns: 0,
                error_count: 0,
                last_error: None,
                complete: false,
                end_reason: None,
            },
            last_sample_ns: None,
        }));
        // Synchronous first sample, before the caller constructs providers or
        // imports weights; a spawned thread alone cannot guarantee this order.
        sample_and_write(&state, &mut file, started, &mut source)?;
        let worker_state = Arc::clone(&state);
        let (stop, stopped) = mpsc::channel();
        let join = thread::Builder::new()
            .name("metal-memory-sample".to_owned())
            .spawn(move || loop {
                let finishing = match stopped.recv_timeout(interval) {
                    Ok(reason) => Some(reason),
                    Err(mpsc::RecvTimeoutError::Disconnected) => Some("owner_disconnected"),
                    Err(mpsc::RecvTimeoutError::Timeout) => None,
                };
                if let Err(error) = sample_and_write(&worker_state, &mut file, started, &mut source)
                {
                    eprintln!("device-memory sampling failed: {error}");
                    break;
                }
                if let Some(reason) = finishing {
                    let summary = {
                        let mut state = lock(&worker_state);
                        state.snapshot.record_type = "summary".to_owned();
                        state.snapshot.complete =
                            reason == "shutdown" && state.snapshot.error_count == 0;
                        state.snapshot.end_reason = Some(reason.to_owned());
                        state.snapshot.clone()
                    };
                    if let Err(error) = write_record(&mut file, &summary) {
                        lock(&worker_state).error(error.clone());
                        eprintln!("device-memory summary failed: {error}");
                    }
                    break;
                }
            })
            .map_err(|error| format!("start device-memory sampler: {error}"))?;
        Ok(Self {
            state,
            worker: Mutex::new(Some(Worker { stop, join })),
        })
    }

    pub(super) fn snapshot(&self) -> DeviceMemoryTelemetrySnapshot {
        let mut snapshot = lock(&self.state).snapshot.clone();
        snapshot.record_type = "snapshot".to_owned();
        snapshot
    }

    pub(super) fn finish(&self) -> Result<(), MetalDeviceRuntimeError> {
        self.finish_with_reason("shutdown")
    }

    fn finish_with_reason(&self, reason: &'static str) -> Result<(), MetalDeviceRuntimeError> {
        if let Some(worker) = lock(&self.worker).take() {
            let _ = worker.stop.send(reason);
            if worker.join.join().is_err() {
                lock(&self.state).error("device-memory sampler thread panicked".to_owned());
            }
        }
        let state = lock(&self.state);
        if state.snapshot.error_count > 0 {
            Err(MetalDeviceRuntimeError::contract(
                state
                    .snapshot
                    .last_error
                    .clone()
                    .unwrap_or_else(|| "device-memory sampling failed".to_owned()),
            ))
        } else {
            Ok(())
        }
    }
}

impl Drop for DeviceMemorySampler {
    fn drop(&mut self) {
        if let Err(error) = self.finish_with_reason("runtime_drop") {
            eprintln!("device-memory sampler finalization failed: {error}");
        }
    }
}

fn write_record(
    file: &mut impl Write,
    record: &DeviceMemoryTelemetrySnapshot,
) -> Result<(), String> {
    serde_json::to_writer(&mut *file, record)
        .map_err(|error| format!("serialize device-memory record: {error}"))?;
    file.write_all(b"\n")
        .and_then(|()| file.flush())
        .map_err(|error| format!("write device-memory record: {error}"))
}

fn sample_and_write(
    state: &Mutex<SampleState>,
    file: &mut impl Write,
    started: Instant,
    source: &mut impl FnMut() -> Result<u64, String>,
) -> Result<(), String> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| source()))
        .unwrap_or_else(|_| Err("device-memory sample source panicked".to_owned()));
    let elapsed_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let record = {
        let mut state = lock(state);
        state.observe(elapsed_ns, result);
        state.snapshot.clone()
    };
    write_record(file, &record).inspect_err(|error| lock(state).error(error.clone()))
}

#[cfg(test)]
mod tests;
