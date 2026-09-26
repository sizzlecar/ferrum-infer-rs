use super::*;
use ferrum_types::device_memory::{CudaMemoryReading, CudaMemoryUnavailable};
use std::sync::atomic::{AtomicU64, Ordering};

fn domains(bytes: u64) -> CudaMemoryDomains {
    let unavailable =
        CudaMemoryReading::unavailable(CudaMemoryUnavailable::ProcessAccountingUnavailable);
    CudaMemoryDomains {
        runtime_requested_allocation_bytes: bytes,
        default_pool_used_bytes: CudaMemoryReading::known(1000),
        default_pool_reserved_bytes: CudaMemoryReading::known(2000),
        cuda_driver_reported_free_bytes: CudaMemoryReading::known(3000),
        cuda_driver_reported_total_bytes: CudaMemoryReading::known(4000),
        nvml_device_free_bytes: CudaMemoryReading::known(1000),
        nvml_device_used_bytes: CudaMemoryReading::known(2500),
        nvml_device_reserved_bytes: CudaMemoryReading::known(500),
        nvml_device_total_bytes: CudaMemoryReading::known(4000),
        nvml_process_used_bytes: unavailable,
    }
}

fn path(label: &str) -> std::path::PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    std::env::temp_dir().join(format!(
        "ferrum-cuda-memory-{label}-{}-{}-{}.jsonl",
        std::process::id(),
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}

fn start(
    path: &Path,
    source: impl FnMut() -> Result<CudaMemoryDomains, String> + Send + 'static,
) -> DeviceMemorySampler {
    DeviceMemorySampler::start_with_source(
        path,
        ("injected-uuid".to_owned(), "injected-device".to_owned()),
        Duration::from_secs(3600),
        Arc::new(allocation::AllocationTracker::default()),
        source,
    )
    .unwrap()
}

#[test]
fn cuda_memory_sampler_first_sample_is_synchronous_and_domains_are_separate() {
    let path = path("domains");
    let sampler = start(&path, || Ok(domains(20)));
    let first = sampler.snapshot();
    assert_eq!(first.sample_count, 1);
    assert_eq!(first.current_allocated_bytes, 20);
    assert_eq!(
        first.source,
        "CudaDeviceRuntime.liveBackingAllocationRequests"
    );
    assert_eq!(first.scope, "runtime_cuda_requested_allocations");
    assert!(!first.complete);
    sampler.finish().unwrap();
    let done = sampler.snapshot();
    assert!(done.complete);
    assert_eq!(done.sample_count, 2);
    let records: Vec<serde_json::Value> = std::fs::read_to_string(&path)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let summary = records.last().unwrap();
    assert_eq!(summary["record_type"], "summary");
    assert_eq!(
        summary["cuda_memory"]["nvml_process_used_bytes"]["status"],
        "unavailable"
    );
    assert_eq!(
        summary["cuda_memory"]["cuda_driver_reported_total_bytes"]["bytes"],
        4000
    );
    assert_eq!(summary["peak_allocated_bytes"], 20);
    assert_eq!(
        summary["cuda_memory"]["cuda_driver_reported_free_bytes"]["bytes"],
        3000
    );
    assert_eq!(
        summary["cuda_memory"]["nvml_device_free_bytes"]["bytes"],
        1000
    );
    drop(sampler);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn cuda_memory_sampler_refuses_existing_file_and_marks_drop_incomplete() {
    let path = path("drop");
    let sampler = start(&path, || Ok(domains(10)));
    assert!(DeviceMemorySampler::start_with_source(
        &path,
        ("a".to_owned(), "b".to_owned()),
        Duration::from_secs(1),
        Arc::new(allocation::AllocationTracker::default()),
        || Ok(domains(0))
    )
    .is_err());
    drop(sampler);
    let text = std::fs::read_to_string(&path).unwrap();
    let summary: serde_json::Value = serde_json::from_str(text.lines().last().unwrap()).unwrap();
    assert_eq!(summary["complete"], false);
    assert_eq!(summary["end_reason"], "runtime_drop");
    std::fs::remove_file(path).unwrap();
}

#[test]
fn cuda_memory_sampler_cannot_finish_complete_after_accounting_failure() {
    let path = path("failure");
    let mut calls = 0;
    let sampler = start(&path, move || {
        calls += 1;
        if calls == 1 {
            Ok(domains(128))
        } else {
            Err("injected allocation accounting failure".to_owned())
        }
    });
    assert!(sampler.finish().is_err());
    let snapshot = sampler.snapshot();
    assert!(!snapshot.complete);
    assert_eq!(snapshot.error_count, 1);
    assert_eq!(snapshot.sample_count, 1);
    drop(sampler);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn cuda_memory_source_failure_cannot_publish_a_placeholder_zero_sample() {
    let path = path("first-failure");
    assert!(DeviceMemorySampler::start_with_source(
        &path,
        ("a".to_owned(), "b".to_owned()),
        Duration::from_secs(1),
        Arc::new(allocation::AllocationTracker::default()),
        || Err("unavailable".to_owned())
    )
    .is_err());
    assert!(std::fs::read_to_string(&path).unwrap().is_empty());
    std::fs::remove_file(path).unwrap();
}
