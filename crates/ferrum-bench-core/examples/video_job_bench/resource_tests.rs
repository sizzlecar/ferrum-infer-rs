use super::*;
use clap::Parser;

fn arguments() -> super::super::Arguments {
    super::super::Arguments::try_parse_from([
        "test",
        "--base-url",
        "http://localhost:1",
        "--request-json",
        "unused",
        "--provenance",
        "unused",
        "--server-config",
        "unused",
        "--output",
        "unused",
    ])
    .unwrap()
}

fn process(root: &std::path::Path, start: u64, rss: u64) {
    let directory = root.join("42");
    std::fs::create_dir_all(directory.join("task/42")).unwrap();
    let mut fields: Vec<String> = vec!["S".into(), "1".into()];
    fields.extend((5..22).map(|_| "0".into()));
    fields.push(start.to_string());
    std::fs::write(
        directory.join("stat"),
        format!("42 (test worker) {}", fields.join(" ")),
    )
    .unwrap();
    std::fs::write(directory.join("status"), format!("VmRSS: {rss} kB\n")).unwrap();
    std::fs::write(directory.join("task/42/children"), "").unwrap();
}

fn sampler(root: &std::path::Path, interval_ms: u64) -> Arc<ServerSampler> {
    let mut config = arguments().resources;
    config.resource_pid = Some(42);
    config.sample_interval_ms = interval_ms;
    Arc::new(ServerSampler {
        config,
        process: Some(Mutex::new(ProcessSampler::new(root.into(), 42))),
        gpu_identity: Mutex::new(None),
        gpu: GpuMemorySampler::default(),
        process_delay_ms: std::sync::atomic::AtomicU64::new(0),
        process_started: tokio::sync::Notify::new(),
    })
}

#[tokio::test]
async fn window_samples_boundaries_and_pins_root_across_cells() {
    let temp = tempfile::tempdir().unwrap();
    process(temp.path(), 100, 3);
    let sampler = sampler(temp.path(), 60_000);
    let first = RunningWindow::begin(sampler.clone()).await;
    process(temp.path(), 100, 5);
    let first = first.finish().await;
    assert!(first.complete);
    assert_eq!(first.attempted_samples, 2);
    assert_eq!(
        first.host.as_ref().unwrap().observed_peak_bytes,
        Some(5 * 1024)
    );
    assert!(first.device.is_none());
    process(temp.path(), 101, 999);
    let second = RunningWindow::begin(sampler).await.finish().await;
    assert!(!second.complete);
    assert_eq!(second.host.as_ref().unwrap().observed_peak_bytes, None);
    assert_eq!(second.host.as_ref().unwrap().failed_samples, 2);
}

#[tokio::test]
async fn slow_periodic_query_cannot_claim_cadence_after_early_stop() {
    let temp = tempfile::tempdir().unwrap();
    process(temp.path(), 100, 3);
    let sampler = sampler(temp.path(), 10);
    let window = RunningWindow::begin(sampler.clone()).await;
    sampler
        .process_delay_ms
        .store(80, std::sync::atomic::Ordering::SeqCst);
    tokio::time::timeout(Duration::from_secs(5), sampler.process_started.notified())
        .await
        .unwrap();
    sampler
        .process_delay_ms
        .store(0, std::sync::atomic::Ordering::SeqCst);
    let report = window.finish().await;
    assert!(!report.complete);
    assert_eq!(report.host.as_ref().unwrap().failed_samples, 0);
    assert!(report.overlong_periodic_observations > 0);
    assert!(
        report
            .host_query_timing
            .as_ref()
            .unwrap()
            .overlong_periodic_queries
            > 0
    );
}

#[test]
fn explicit_local_scope_and_distinct_gpu_identity_are_required() {
    let mut config = arguments().resources;
    config.resource_gpu_index = vec![0];
    assert!(config.validate("http://127.0.0.1:1").is_ok());
    assert!(config.validate("http://192.0.2.1:1").is_err());
    config.resource_gpu_index = vec![0, 0];
    assert!(config.validate("http://127.0.0.1:1").is_err());
}

#[test]
fn failed_observation_has_no_fabricated_peak_and_keeps_query_interval() {
    let epoch = Instant::now();
    let mut memory = SampledMemory::default();
    let mut timing = ResourceQueryTiming::default();
    record(
        &mut memory,
        &mut timing,
        &Err("unavailable".into()),
        epoch,
        epoch,
        epoch + Duration::from_secs(1),
        Some(Duration::from_millis(200)),
    );
    assert_eq!(memory.observed_peak_bytes, None);
    assert_eq!(memory.failed_samples, 1);
    assert_eq!(timing.maximum_query_ms, 1000.0);
    assert_eq!(timing.peak_query_finished_offset_ms, None);
}
