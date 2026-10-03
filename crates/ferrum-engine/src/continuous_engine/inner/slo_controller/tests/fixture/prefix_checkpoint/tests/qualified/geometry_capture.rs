//! Exercise the real startup stop and retirement chain. Only the product hooks
//! can advance capture phases; these tests never call those hooks themselves.
use super::*;
use crate::geometry_capture::{capture, CaptureOptions, CaptureReport};
use ferrum_scheduler::implementations::continuous::cost_profile::StructuredPreparedOwnerBlockHeaderV8;
use std::{fs, io::BufRead, path::PathBuf};

struct Directory {
    path: PathBuf,
    profile: Option<PathBuf>,
}
impl Directory {
    fn new() -> Self {
        let path = std::env::temp_dir().join(format!(
            "ferrum-startup-geometry-capture-{}",
            RequestId::new()
        ));
        fs::create_dir(&path).unwrap();
        Self {
            path,
            profile: None,
        }
    }

    fn policy(&self) -> ferrum_types::SloAutomaticCalibrationDiagnosticsV1 {
        ferrum_types::SloAutomaticCalibrationDiagnosticsV1::Directory {
            directory: self.path.join("sources"),
            maximum_source_bytes: NonZeroU64::new(64 * 1024 * 1024).unwrap(),
            maximum_total_bytes: NonZeroU64::new(512 * 1024 * 1024).unwrap(),
            maximum_retained_generations: NonZeroUsize::new(16).unwrap(),
        }
    }

    fn assert_no_source8(&self) {
        let root = self.path.join("sources/ferrum-automatic-v1");
        let entries = match fs::read_dir(&root) {
            Ok(entries) => entries,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => return,
            Err(error) => panic!("cannot read startup diagnostics {root:?}: {error}"),
        };
        for entry in entries {
            let path = entry.unwrap().path().join("source.jsonl");
            if !path.is_file() {
                continue;
            }
            let first = std::io::BufReader::new(fs::File::open(&path).unwrap())
                .lines()
                .next()
                .expect("allocated source journal is empty")
                .unwrap();
            assert!(
                serde_json::from_str::<StructuredPreparedOwnerBlockHeaderV8>(&first).is_err(),
                "inventory capture opened a Source8 collector: {path:?}"
            );
        }
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        if std::thread::panicking() {
            eprintln!(
                "startup geometry failure evidence retained at {:?}; profile={:?}",
                self.path, self.profile
            );
        } else {
            fs::remove_dir_all(&self.path).unwrap();
            if let Some(profile) = &self.profile {
                if profile.exists() {
                    fs::remove_file(profile).unwrap();
                }
            }
        }
    }
}

async fn stopped_startup(maximum_bytes: NonZeroU64) -> (Directory, CaptureReport) {
    let mut directory = Directory::new();
    let (engine, executor, templates) = fresh_startup_with_probes(
        NonZeroU64::new(30_000),
        false,
        false,
        &[PROMPT],
        directory.policy(),
        |_, _| {},
    )
    .await;
    // Retain independent audit handles, not EngineInner: original startup must
    // retain exclusive ownership, including Arc::get_mut at its normal cuts.
    let runtime = Arc::clone(engine.inner.cost_runtime.as_ref().unwrap());
    let scheduler = Arc::clone(&engine.inner.scheduler);
    directory.profile = engine.inner.config.runtime.profile_jsonl.clone();
    let options = CaptureOptions {
        output: directory.path.join("capture.jsonl"),
        maximum_bytes,
        // This CPU fixture has no external input package. The capture itself
        // serializes the original typed configuration and template bytes.
        input_sha256: [0; 32],
    };
    let (startup, report) = tokio::time::timeout(
        Duration::from_secs(180),
        capture(
            options,
            engine.finish_automatic_startup_with_probes(templates),
        ),
    )
    .await
    .expect("original automatic startup exceeded its existing fixture bound")
    .unwrap();
    if let Ok(engine) = startup {
        engine.shutdown().await.unwrap();
        panic!("capture scope returned a live engine");
    }
    // Error text is not proof of the intended cut. Check the independent
    // executor/scheduler state and actual inventory counts in both paths.
    assert!(executor.native_structured_counts().1 > 0);
    assert!(executor.completion_calls.load(Ordering::Acquire) > 0);
    assert_eq!(scheduler.active_count(), 0);
    assert_eq!(scheduler.waiting_count(), 0);
    assert_eq!(scheduler.prefix_held_waiting_count(), 0);
    let lifecycle = executor.token_policy_lifecycle.lock();
    assert!(lifecycle.admitted.is_empty());
    assert_eq!(lifecycle.executing, 0);
    drop(lifecycle);
    assert!(executor.native_structured_history.lock().is_empty());
    assert!(executor
        .produced_caches
        .lock()
        .iter()
        .all(|cache| cache.upgrade().is_none()));
    assert_eq!(executor.native_prefix_live_lease_counts(), (0, 0));
    assert!(runtime.snapshot().is_none());
    assert!(runtime
        .startup_series_children_for_test()
        .unwrap()
        .is_empty());
    directory.assert_no_source8();
    let wire = serde_json::to_value(&report).unwrap();
    assert!(wire["expected_matrices"].as_u64().unwrap() > 0);
    assert_eq!(wire["observed_matrices"], wire["expected_matrices"]);
    (directory, report)
}

#[tokio::test]
async fn geometry_capture_normal_startup_stops_before_source8_and_retires_owners() {
    let (directory, report) = stopped_startup(NonZeroU64::new(64 * 1024 * 1024).unwrap()).await;
    // Complete can only be issued after original inventory/end, series finish,
    // model freeze, tracked drain and shutdown have all returned successfully.
    assert!(report.is_complete(), "{report:?}");
    let records: Vec<serde_json::Value> = fs::read_to_string(directory.path.join("capture.jsonl"))
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert!(records
        .iter()
        .any(|row| row["kind"] == "actual_startup_inputs"));
    let inventories: Vec<_> = records
        .iter()
        .filter(|row| row["kind"] == "original_cases")
        .collect();
    assert_eq!(inventories.len(), 1);
    let cases = inventories[0]["cases"].as_array().unwrap();
    let prompts = inventories[0]["prompts"].as_array().unwrap();
    assert!(!cases.is_empty());
    for case in cases {
        assert!(case["width"].as_u64().unwrap() > 0);
        let template = usize::try_from(case["template"].as_u64().unwrap()).unwrap();
        assert!(template < prompts.len());
        assert!(prompts[template].as_u64().unwrap() > 0);
    }
    for matrix in records
        .iter()
        .filter(|row| row["kind"] == "original_matrix")
    {
        let indices = matrix["cases"].as_array().unwrap();
        assert_eq!(indices.len(), matrix["axis_bits"].as_array().unwrap().len());
        for index in indices {
            let index = usize::try_from(index.as_u64().unwrap()).unwrap();
            assert!(index < cases.len());
        }
    }
    let matrices = records
        .iter()
        .filter(|row| row["kind"] == "original_matrix")
        .count();
    assert!(matrices > 0);
    assert_eq!(
        matrices,
        records
            .iter()
            .filter(|row| row["kind"] == "original_result")
            .count()
    );
    let retired: Vec<_> = records
        .iter()
        .filter(|row| row["kind"] == "inventory_retired")
        .collect();
    assert_eq!(retired.len(), 1);
    assert_eq!(retired[0]["complete"], true);
    assert_eq!(records.last().unwrap()["kind"], "completed_after_shutdown");
}

#[tokio::test]
async fn geometry_capture_byte_exhaustion_still_retires_normal_startup() {
    let (directory, report) = stopped_startup(NonZeroU64::MIN).await;
    assert!(!report.is_complete());
    assert!(!directory.path.join("capture.jsonl").exists());
    let wire = serde_json::to_value(&report).unwrap();
    assert!(wire["failure"].is_string());
    let partial = PathBuf::from(wire["path"].as_str().unwrap());
    assert!(partial.is_file());
    assert!(fs::metadata(partial).unwrap().len() <= 1);
}
