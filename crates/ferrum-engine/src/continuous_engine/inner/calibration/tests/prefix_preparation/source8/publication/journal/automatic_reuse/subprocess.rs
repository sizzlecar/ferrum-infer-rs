//! Two actual OS processes use the normal PlatformDefault cache locator.
//! The test-only job carries only declared clocks and expected receipts; no
//! collector, qualified model or feedback authority crosses the process edge.
use super::*;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Job {
    restore: bool,
    clock_ns: u64,
}
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct OriginalChild {
    domain: [u8; 32],
    source: [u8; 32],
    parameters: [u8; 32],
    source_anchor_ns: u64,
    model_anchor_ns: u64,
    maximum_age_ns: u64,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Report {
    next_clock_ns: u64,
    algorithm_seed: [u8; 32],
    children: Vec<OriginalChild>,
}

fn report_path(job: &Path) -> PathBuf {
    job.with_extension("report.json")
}

#[test]
fn automatic_reuse_default_platform_cache_restores_in_independent_process() {
    let dir = Directory::new();
    let cache_home = dir.0.join("home");
    std::fs::create_dir(&cache_home).unwrap();
    let train_job = dir.0.join("train.json");
    std::fs::write(
        &train_job,
        serde_json::to_vec(&Job {
            restore: false,
            clock_ns: 100,
        })
        .unwrap(),
    )
    .unwrap();
    let spawn = |job: &Path| {
        let mut command = std::process::Command::new(std::env::current_exe().unwrap());
        command
            .arg("automatic_reuse_default_platform_process_child")
            .arg("--nocapture")
            .env("FERRUM_TEST_AUTOMATIC_REUSE_RUNTIME_JOB", job)
            // Standard OS locators are isolated only in the child, so parallel
            // parent tests and the user's real cache are never modified.
            .env("HOME", &cache_home)
            .env("XDG_CACHE_HOME", &cache_home)
            .env("LOCALAPPDATA", &cache_home);
        assert!(command.status().unwrap().success());
    };
    spawn(&train_job);
    let first: Report =
        serde_json::from_slice(&std::fs::read(report_path(&train_job)).unwrap()).unwrap();
    let restore_job = dir.0.join("restore.json");
    std::fs::write(
        &restore_job,
        serde_json::to_vec(&Job {
            restore: true,
            clock_ns: first.next_clock_ns.checked_add(1).unwrap(),
        })
        .unwrap(),
    )
    .unwrap();
    std::fs::write(
        restore_job.with_extension("expected.json"),
        serde_json::to_vec(&first).unwrap(),
    )
    .unwrap();
    spawn(&restore_job);
    let second: Report =
        serde_json::from_slice(&std::fs::read(report_path(&restore_job)).unwrap()).unwrap();
    assert_eq!(
        first.children, second.children,
        "replay cannot renew original clocks or parameters"
    );
    assert_eq!(
        first.algorithm_seed, second.algorithm_seed,
        "checked but non-numerical declaration is loaded only from the cache"
    );
    assert!(second.next_clock_ns > first.next_clock_ns);
}

#[tokio::test]
async fn automatic_reuse_default_platform_process_child() {
    let Some(path) = std::env::var_os("FERRUM_TEST_AUTOMATIC_REUSE_RUNTIME_JOB") else {
        return;
    };
    let path = PathBuf::from(path);
    let job: Job = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let clock = Arc::new(Clock {
        now: AtomicU64::new(job.clock_ns),
        domain: CostMonotonicDomainV1::new_macos_continuous([22; 16]).unwrap(),
    });
    // No typed Directory override and no profile import. This deterministic
    // clock supplies a same-boot identity; the real OS clock has separate tests.
    let policy = ferrum_types::SloAutomaticCalibrationReuseV1::default();
    let (mut session, executor) =
        automatic_session_with_storage(2, clock.clone(), Default::default(), policy).await;
    native(&executor);
    let runtime = session.engine.inner.cost_runtime.as_ref().unwrap().clone();
    assert_eq!(runtime.reused_cost_is_fresh(), job.restore);
    assert_eq!(
        executor.physical.load(Ordering::Acquire),
        0,
        "constructor replay performs no inference"
    );
    if !job.restore {
        install_cold_seed(&mut session, &executor).await;
        let deadline = tokio::time::Instant::now() + Duration::from_secs(45);
        session.begin_startup_owner_series(1, deadline).unwrap();
        session
            .begin_prepared_owner_source(
                full_population(&session),
                CostProfileLoadLimits::default(),
            )
            .await
            .unwrap();
        assert!(session
            .prepared_owner_capture
            .as_ref()
            .unwrap()
            .journal_observer()
            .is_none());
        collect_source(&mut session, [16, 8, 8], deadline).await;
        session.activate_prepared_owner_source().await.unwrap();
        session.retire_startup_owner_source().await.unwrap();
        session.finish_startup_owner_series().unwrap();
    }
    let models = runtime.startup_series_children_for_test().unwrap();
    let mut children: Vec<_> = models
        .iter()
        .map(|child| {
            let original = child.provenance();
            OriginalChild {
                domain: *child.domain_signature(),
                source: original.source_sha256,
                parameters: original.parameters_sha256,
                source_anchor_ns: original.clock.source_monotonic_anchor_ns,
                model_anchor_ns: original.clock.model_anchor_ns,
                maximum_age_ns: child.runtime_limits().1,
            }
        })
        .collect();
    children.sort_by_key(|c| c.domain);
    assert!(!children.is_empty());
    if job.restore {
        let original: Report =
            serde_json::from_slice(&std::fs::read(path.with_extension("expected.json")).unwrap())
                .unwrap();
        assert_eq!(children, original.children);
    }
    let algorithm_seed = *runtime.startup_algorithm_seed().unwrap().signature();
    // This submits fresh native CPU work through the actual CostWitness,
    // once-only backend receipt and host reconciliation, then full shutdown.
    future::verify(session, executor).await;
    drop(runtime);
    std::fs::write(
        report_path(&path),
        serde_json::to_vec(&Report {
            next_clock_ns: clock.now.load(Ordering::Acquire),
            algorithm_seed,
            children,
        })
        .unwrap(),
    )
    .unwrap();
}
