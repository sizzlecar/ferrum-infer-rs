//! Opt-in host sampler. The primary count is runtime backing-allocation requests;
//! actual pool, whole-device and optional process queries have separate scopes.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;
use std::sync::{mpsc, Arc, Mutex, MutexGuard};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use cudarc::driver::CudaContext;
use ferrum_interfaces::vnext::DeviceMemoryTelemetrySnapshot;
use ferrum_types::device_memory::CudaMemoryDomains;
use ferrum_types::{DeviceMemorySamplingConfig, DEVICE_MEMORY_SAMPLE_INTERVAL_MS};
use serde::Serialize;

use super::{nvml, CudaDeviceRuntimeError};

pub(super) mod allocation;
mod query;

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[derive(Clone, Serialize)]
struct Record {
    #[serde(flatten)]
    snapshot: DeviceMemoryTelemetrySnapshot,
    cuda_memory: CudaMemoryDomains,
}

struct SampleState {
    record: Record,
    last_sample_ns: Option<u64>,
}

impl SampleState {
    fn observe(&mut self, elapsed_ns: u64, result: Result<CudaMemoryDomains, String>) {
        if let Some(previous) = self.last_sample_ns {
            if elapsed_ns < previous {
                self.error("CUDA memory clock moved backwards".to_owned());
                return;
            }
            self.record.snapshot.max_sample_gap_ns = self
                .record
                .snapshot
                .max_sample_gap_ns
                .max(elapsed_ns - previous);
        }
        self.last_sample_ns = Some(elapsed_ns);
        self.record.snapshot.elapsed_ns = elapsed_ns;
        match result {
            Ok(domains) => {
                let bytes = domains.runtime_requested_allocation_bytes;
                self.record.snapshot.current_allocated_bytes = bytes;
                self.record.snapshot.peak_allocated_bytes =
                    self.record.snapshot.peak_allocated_bytes.max(bytes);
                self.record.snapshot.sample_count =
                    self.record.snapshot.sample_count.saturating_add(1);
                self.record.cuda_memory = domains;
            }
            Err(error) => self.error(error),
        }
    }

    fn error(&mut self, error: String) {
        self.record.snapshot.error_count = self.record.snapshot.error_count.saturating_add(1);
        self.record.snapshot.last_error = Some(error);
        self.record.snapshot.complete = false;
    }
}

struct Worker {
    stop: mpsc::Sender<&'static str>,
    join: JoinHandle<()>,
}

pub(super) struct DeviceMemorySampler {
    state: Arc<Mutex<SampleState>>,
    worker: Mutex<Option<Worker>>,
    tracker: Arc<allocation::AllocationTracker>,
}

impl DeviceMemorySampler {
    pub(super) fn start(
        context: Arc<CudaContext>,
        config: &DeviceMemorySamplingConfig,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        config
            .validate()
            .map_err(CudaDeviceRuntimeError::contract)?;
        let uuid = context
            .uuid()
            .map_err(|error| CudaDeviceRuntimeError::driver("memory device UUID", error))?
            .bytes
            .map(|byte| byte as u8);
        let name = context
            .name()
            .map_err(|error| CudaDeviceRuntimeError::driver("memory device name", error))?;
        let tracker = Arc::new(allocation::AllocationTracker::default());
        let queries = query::MemoryQueries {
            context,
            tracker: Arc::clone(&tracker),
            uuid,
            nvml: nvml::Nvml::load(),
        };
        Self::start_with_source(
            &config.jsonl_path,
            (format!("GPU-{}", nvml::uuid_text(uuid)), name),
            Duration::from_millis(DEVICE_MEMORY_SAMPLE_INTERVAL_MS),
            tracker,
            move || queries.sample(),
        )
        .map_err(CudaDeviceRuntimeError::contract)
    }

    fn start_with_source(
        path: &Path,
        identity: (String, String),
        interval: Duration,
        tracker: Arc<allocation::AllocationTracker>,
        mut source: impl FnMut() -> Result<CudaMemoryDomains, String> + Send + 'static,
    ) -> Result<Self, String> {
        if interval.is_zero() {
            return Err("CUDA memory interval must be positive".to_owned());
        }
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .map_err(|error| format!("create CUDA memory JSONL {}: {error}", path.display()))?;
        let started_unix_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| error.to_string())?
            .as_nanos()
            .to_string();
        let started = Instant::now();
        // Obtain an actual synchronous sample before publishing the sampler.
        // No placeholder zeros become measurements if the primary query fails.
        let first = source()?;
        let bytes = first.runtime_requested_allocation_bytes;
        let state = Arc::new(Mutex::new(SampleState {
            record: Record {
                snapshot: DeviceMemoryTelemetrySnapshot {
                    schema_version: 2,
                    record_type: "sample".to_owned(),
                    source: "CudaDeviceRuntime.liveBackingAllocationRequests".to_owned(),
                    scope: "runtime_cuda_requested_allocations".to_owned(),
                    phase: "pre_weight_load_to_shutdown".to_owned(),
                    pid: std::process::id(),
                    device_registry_id: identity.0,
                    device_name: identity.1,
                    started_unix_ns,
                    elapsed_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
                    current_allocated_bytes: bytes,
                    peak_allocated_bytes: bytes,
                    sample_count: 1,
                    interval_ms: u64::try_from(interval.as_millis()).unwrap_or(u64::MAX),
                    max_sample_gap_ns: 0,
                    error_count: 0,
                    last_error: None,
                    complete: false,
                    end_reason: None,
                },
                cuda_memory: first,
            },
            last_sample_ns: None,
        }));
        {
            let mut state = lock(&state);
            state.last_sample_ns = Some(state.record.snapshot.elapsed_ns);
            write_record(&mut file, &state.record)?;
        }
        let worker_state = Arc::clone(&state);
        let (stop, stopped) = mpsc::channel();
        let join = thread::Builder::new()
            .name("cuda-memory-sample".to_owned())
            .spawn(move || loop {
                let finishing = match stopped.recv_timeout(interval) {
                    Ok(reason) => Some(reason),
                    Err(mpsc::RecvTimeoutError::Disconnected) => Some("owner_disconnected"),
                    Err(mpsc::RecvTimeoutError::Timeout) => None,
                };
                if sample_and_write(&worker_state, &mut file, started, &mut source).is_err() {
                    break;
                }
                if let Some(reason) = finishing {
                    let summary = {
                        let mut state = lock(&worker_state);
                        state.record.snapshot.record_type = "summary".to_owned();
                        state.record.snapshot.complete =
                            reason == "shutdown" && state.record.snapshot.error_count == 0;
                        state.record.snapshot.end_reason = Some(reason.to_owned());
                        state.record.clone()
                    };
                    if let Err(error) = write_record(&mut file, &summary) {
                        lock(&worker_state).error(error);
                    }
                    break;
                }
            })
            .map_err(|error| format!("start CUDA memory sampler: {error}"))?;
        Ok(Self {
            state,
            worker: Mutex::new(Some(Worker { stop, join })),
            tracker,
        })
    }

    pub(super) fn tracker(&self) -> &Arc<allocation::AllocationTracker> {
        &self.tracker
    }

    pub(super) fn snapshot(&self) -> DeviceMemoryTelemetrySnapshot {
        let mut snapshot = lock(&self.state).record.snapshot.clone();
        snapshot.record_type = "snapshot".to_owned();
        snapshot
    }

    pub(super) fn finish(&self) -> Result<(), CudaDeviceRuntimeError> {
        self.finish_with_reason("shutdown")
    }

    fn finish_with_reason(&self, reason: &'static str) -> Result<(), CudaDeviceRuntimeError> {
        if let Some(worker) = lock(&self.worker).take() {
            let _ = worker.stop.send(reason);
            if worker.join.join().is_err() {
                lock(&self.state).error("CUDA memory sampler panicked".to_owned());
            }
        }
        let state = lock(&self.state);
        match &state.record.snapshot.last_error {
            Some(error) => Err(CudaDeviceRuntimeError::contract(error.clone())),
            None => Ok(()),
        }
    }
}

impl Drop for DeviceMemorySampler {
    fn drop(&mut self) {
        let _ = self.finish_with_reason("runtime_drop");
    }
}

fn write_record(file: &mut impl Write, record: &Record) -> Result<(), String> {
    serde_json::to_writer(&mut *file, record)
        .map_err(|error| format!("serialize CUDA memory record: {error}"))?;
    file.write_all(b"\n")
        .and_then(|()| file.flush())
        .map_err(|error| format!("write CUDA memory record: {error}"))
}

fn sample_and_write(
    state: &Mutex<SampleState>,
    file: &mut impl Write,
    started: Instant,
    source: &mut impl FnMut() -> Result<CudaMemoryDomains, String>,
) -> Result<(), String> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(source))
        .unwrap_or_else(|_| Err("CUDA memory source panicked".to_owned()));
    let elapsed_ns = u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX);
    let record = {
        let mut state = lock(state);
        state.observe(elapsed_ns, result);
        state.record.clone()
    };
    write_record(file, &record).inspect_err(|error| lock(state).error(error.clone()))
}

#[cfg(test)]
mod tests;
