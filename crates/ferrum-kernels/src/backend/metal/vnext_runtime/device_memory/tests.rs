use std::io;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};

use super::*;

static NEXT_CAPTURE: AtomicU64 = AtomicU64::new(1);

struct CapturePath(PathBuf);

impl CapturePath {
    fn new() -> Self {
        Self(std::env::temp_dir().join(format!(
            "ferrum-device-memory-{}-{}.jsonl",
            std::process::id(),
            NEXT_CAPTURE.fetch_add(1, Ordering::Relaxed)
        )))
    }

    fn records(&self) -> Vec<DeviceMemoryTelemetrySnapshot> {
        std::fs::read_to_string(&self.0)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }
}

impl Drop for CapturePath {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn start_fake(
    path: &CapturePath,
    source: impl FnMut() -> Result<u64, String> + Send + 'static,
) -> DeviceMemorySampler {
    DeviceMemorySampler::start_with_source(
        &path.0,
        (u64::MAX.to_string(), "fake actual device".to_owned()),
        // Keep periodic wakeups out of lifecycle tests: stop must wake the
        // worker immediately and take the final sample, not wait an interval.
        Duration::from_secs(3600),
        source,
    )
    .unwrap()
}

#[test]
fn device_memory_runtime_samples_real_allocations_and_refuses_a_second_capture() {
    use ferrum_interfaces::vnext::DeviceRuntime;

    let runtime = super::super::tests::runtime();
    let path = CapturePath::new();
    let second_path = CapturePath::new();
    runtime
        .enable_device_memory_sampling(&DeviceMemorySamplingConfig {
            jsonl_path: path.0.clone(),
        })
        .unwrap();
    let initial = runtime.device_memory_snapshot().unwrap();
    assert!(
        initial.sample_count > 0,
        "first query precedes start returning"
    );
    assert_eq!(
        initial.device_registry_id,
        runtime.device().registry_id().to_string()
    );
    assert!(runtime
        .enable_device_memory_sampling(&DeviceMemorySamplingConfig {
            jsonl_path: second_path.0.clone(),
        })
        .is_err());
    assert!(!second_path.0.exists());

    let allocation = runtime.device().new_buffer(
        8 * 1024 * 1024,
        metal::MTLResourceOptions::StorageModeShared,
    );
    runtime.finish_device_memory_sampling().unwrap();
    let final_sample = runtime.device_memory_snapshot().unwrap();
    assert!(final_sample.complete);
    assert_eq!(final_sample.source, "MTLDevice.currentAllocatedSize");
    assert!(final_sample.current_allocated_bytes >= u64::from(allocation.allocated_size()));
    assert!(final_sample.peak_allocated_bytes >= initial.current_allocated_bytes);
    assert_eq!(final_sample.error_count, 0);
    assert_eq!(final_sample.end_reason.as_deref(), Some("shutdown"));
}

#[test]
fn device_memory_final_sample_preserves_identity_and_peak_and_finish_is_idempotent() {
    let path = CapturePath::new();
    let mut samples = [Ok(8), Ok(29)].into_iter();
    let sampler = start_fake(&path, move || samples.next().expect("unexpected sample"));
    let initial = path.records();
    assert_eq!(initial[0].current_allocated_bytes, 8);
    assert!(!initial[0].complete);
    assert_eq!(initial[0].device_registry_id, u64::MAX.to_string());
    assert_eq!(initial[0].pid, std::process::id());
    sampler.finish().unwrap();
    let finished = path.records();
    let summary = finished.last().unwrap();
    assert_eq!(summary.record_type, "summary");
    assert!(summary.complete);
    assert_eq!(summary.end_reason.as_deref(), Some("shutdown"));
    assert_eq!(summary.current_allocated_bytes, 29);
    assert_eq!(summary.peak_allocated_bytes, 29);
    assert_eq!(summary.sample_count, 2);
    assert!(summary.elapsed_ns >= initial[0].elapsed_ns);
    assert_eq!(
        summary.max_sample_gap_ns,
        summary.elapsed_ns - initial[0].elapsed_ns
    );
    sampler.finish().unwrap();
    assert_eq!(path.records(), finished);
    assert_eq!(sampler.snapshot().peak_allocated_bytes, 29);
}

#[test]
fn device_memory_state_tracks_decreases_errors_and_real_observation_gaps() {
    let path = CapturePath::new();
    let sampler = start_fake(&path, || Ok(0));
    sampler.finish().unwrap();
    let mut snapshot = sampler.snapshot();
    snapshot.sample_count = 0;
    snapshot.elapsed_ns = 0;
    snapshot.max_sample_gap_ns = 0;
    snapshot.complete = false;
    let mut state = SampleState {
        snapshot,
        last_sample_ns: None,
    };
    state.observe(10, Ok(100));
    state.observe(25, Ok(500));
    state.observe(80, Ok(200));
    assert_eq!(state.snapshot.current_allocated_bytes, 200);
    assert_eq!(state.snapshot.peak_allocated_bytes, 500);
    assert_eq!(state.snapshot.max_sample_gap_ns, 55);
    state.observe(90, Err("query unavailable".to_owned()));
    assert_eq!(state.snapshot.sample_count, 3);
    assert_eq!(state.snapshot.error_count, 1);
    assert_eq!(state.snapshot.current_allocated_bytes, 200);
    state.observe(85, Ok(9999));
    assert_eq!(state.snapshot.sample_count, 3);
    assert_eq!(state.snapshot.peak_allocated_bytes, 500);
    assert_eq!(state.snapshot.elapsed_ns, 90);
    assert_eq!(state.snapshot.error_count, 2);
}

#[test]
fn device_memory_failed_source_is_not_reported_as_complete_or_zero_usage() {
    let path = CapturePath::new();
    let mut samples = [Ok(71), Err("device query failed".to_owned())].into_iter();
    let sampler = start_fake(&path, move || samples.next().unwrap());
    assert!(sampler.finish().is_err());
    let records = path.records();
    let summary = records.last().unwrap();
    assert!(!summary.complete);
    assert_eq!(summary.error_count, 1);
    assert_eq!(summary.sample_count, 1);
    assert_eq!(summary.peak_allocated_bytes, 71);
    assert_eq!(summary.last_error.as_deref(), Some("device query failed"));
}

#[test]
fn device_memory_drop_finalizes_and_does_not_overwrite_an_existing_capture() {
    let path = CapturePath::new();
    let sampler = start_fake(&path, || Ok(64));
    let rejected = DeviceMemorySampler::start_with_source(
        &path.0,
        ("other-device".to_owned(), "other".to_owned()),
        Duration::from_secs(1),
        || Ok(999),
    );
    assert!(rejected.is_err());
    drop(sampler);
    let records = path.records();
    let summary = records.last().unwrap();
    assert!(!summary.complete);
    assert_eq!(summary.end_reason.as_deref(), Some("runtime_drop"));
    assert_eq!(summary.peak_allocated_bytes, 64);
}

struct FailedWriter;

impl Write for FailedWriter {
    fn write(&mut self, _bytes: &[u8]) -> io::Result<usize> {
        Err(io::Error::other("disk unavailable"))
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn device_memory_write_failure_remains_visible_in_health_state() {
    let path = CapturePath::new();
    let sampler = start_fake(&path, || Ok(17));
    sampler.finish().unwrap();
    let state = Mutex::new(SampleState {
        snapshot: sampler.snapshot(),
        last_sample_ns: None,
    });
    let result = sample_and_write(&state, &mut FailedWriter, Instant::now(), &mut || Ok(33));
    assert!(result.is_err());
    let snapshot = &lock(&state).snapshot;
    assert!(!snapshot.complete);
    assert_eq!(snapshot.error_count, 1);
    assert_eq!(snapshot.peak_allocated_bytes, 33);
    assert!(snapshot
        .last_error
        .as_ref()
        .unwrap()
        .contains("disk unavailable"));
}
