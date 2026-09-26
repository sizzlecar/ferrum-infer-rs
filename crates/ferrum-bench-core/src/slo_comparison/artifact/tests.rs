use super::*;
use crate::slo_comparison::tests::{arm, contract};
use memory::MetalRecord;
use serde::{de::DeserializeOwned, Serialize};
use std::path::PathBuf;
use tempfile::TempDir;

mod native_time;

fn save_bytes(dir: &Path, name: &str, bytes: &[u8]) -> ArtifactFileRef {
    std::fs::write(dir.join(name), bytes).unwrap();
    ArtifactFileRef {
        path: name.into(),
        sha256: digest(bytes),
        bytes: bytes.len() as u64,
    }
}
fn save<T: Serialize>(dir: &Path, name: &str, value: &T) -> ArtifactFileRef {
    save_bytes(dir, name, &serde_json::to_vec(value).unwrap())
}
fn read<T: DeserializeOwned>(dir: &Path, file: &ArtifactFileRef) -> T {
    serde_json::from_slice(&std::fs::read(dir.join(&file.path)).unwrap()).unwrap()
}
fn save_rows<T: Serialize>(dir: &Path, name: &str, rows: &[T]) -> ArtifactFileRef {
    let mut bytes = Vec::new();
    for row in rows {
        serde_json::to_writer(&mut bytes, row).unwrap();
        bytes.push(b'\n');
    }
    save_bytes(dir, name, &bytes)
}

fn metal_rows(execution: &ExecutionArtifact) -> Vec<MetalRecord> {
    let anchor = execution.measurement_started_unix_ns - 250_000_000;
    let mut rows = Vec::new();
    for index in 0..=14_u64 {
        rows.push(MetalRecord {
            schema_version: 1,
            record_type: "sample".into(),
            source: "MTLDevice.currentAllocatedSize".into(),
            scope: "process_metal_device".into(),
            phase: "pre_weight_load_to_shutdown".into(),
            pid: execution.identity.server_pid,
            device_registry_id: execution.device_registry_id.clone().unwrap(),
            device_name: "fixture GPU".into(),
            started_unix_ns: anchor.to_string(),
            elapsed_ns: index * 250_000_000,
            current_allocated_bytes: 1_000_000_000 + index,
            peak_allocated_bytes: 1_000_000_000 + index,
            sample_count: index + 1,
            interval_ms: 250,
            max_sample_gap_ns: if index == 0 { 0 } else { 250_000_000 },
            error_count: 0,
            last_error: None,
            complete: false,
            end_reason: None,
        });
    }
    let mut summary = rows.last().unwrap().clone();
    summary.record_type = "summary".into();
    summary.complete = true;
    summary.end_reason = Some("shutdown".into());
    rows.push(summary);
    rows
}

fn write_arm(dir: &Path, contract: &FrozenComparisonContract, candidate: bool) -> ArmArtifactRefs {
    let owned = arm(contract, 0, candidate, if candidate { 0.5 } else { 1.0 });
    let label = if candidate { "candidate" } else { "baseline" };
    let identity = RunArtifactIdentity {
        benchmark_run_id: owned.report.benchmark_run_id.clone().unwrap(),
        cell_id: owned.report.cell_id.clone().unwrap(),
        repeat_index: 0,
        server_pid: if candidate { 102 } else { 101 },
    };
    let execution = ExecutionArtifact {
        schema_version: 1,
        identity: identity.clone(),
        shared: owned.execution.shared,
        server: owned.execution.server,
        capacity: owned.execution.capacity,
        measurement_started_unix_ns: owned.execution.measurement_started_unix_ns,
        measurement_ended_unix_ns: owned.execution.measurement_ended_unix_ns,
        device_registry_id: Some("fixture-device".into()),
    };
    let sidecar = SloSidecarCell {
        schema_version: 1,
        config_sha256: contract.shared.client_slo_config_sha256.clone(),
        legacy_benchmark: owned.report,
        repeats: vec![SloSidecarRepeat {
            repeat_index: 0,
            evaluation: owned.evaluation,
            arrivals: vec![SidecarArrival::default(); 3],
            request_start_window_s: None,
            observed_request_start_rate_rps: None,
            max_client_dispatch_backlog: None,
        }],
    };
    let anchor = execution.measurement_started_unix_ns - 250_000_000;
    let footprint = SampledMemoryArtifact {
        schema_version: 1,
        identity: identity.clone(),
        measurement: MemoryMeasurement::SampledOsPhysicalFootprint,
        collector: "typed test observations".into(),
        window: contract.memory.os_footprint.window().into(),
        started_unix_ns: anchor,
        interval_ms: 100,
        observations: (0..=35)
            .map(|index| MemoryObservation {
                elapsed_ns: index * 100_000_000,
                bytes: Some(2_000_000_000 + index),
                error: None,
            })
            .collect(),
        finished_elapsed_ns: Some(3_500_000_000),
    };
    let rss = MaximumRssArtifact {
        schema_version: 1,
        identity,
        collector: "typed test process observation".into(),
        window: contract.memory.maximum_rss_window.clone(),
        process_started_unix_ns: anchor,
        process_ended_unix_ns: Some(execution.measurement_ended_unix_ns + 250_000_000),
        maximum_rss_bytes: Some(3_000_000_000),
        error: None,
    };
    ArmArtifactRefs {
        sidecar: SidecarRef {
            file: save_rows(dir, &format!("{label}-sidecar.jsonl"), &[sidecar]),
            record_index: 0,
        },
        repeat_index: 0,
        selection_sha256: contract.pairs[0].selection.selection_sha256.clone(),
        execution: save(dir, &format!("{label}-execution.json"), &execution),
        memory: MemoryArtifactRefs {
            device: Some(DeviceMemoryRef::FerrumMetalV1 {
                file: save_rows(
                    dir,
                    &format!("{label}-metal.jsonl"),
                    &metal_rows(&execution),
                ),
            }),
            footprint: Some(save(dir, &format!("{label}-footprint.json"), &footprint)),
            rss: Some(save(dir, &format!("{label}-rss.json"), &rss)),
            process_lifetime: None,
        },
    }
}

struct Fixture {
    dir: TempDir,
    manifest: ComparisonArtifactManifest,
}
impl Fixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let mut contract = contract(1);
        contract.memory.device_allocation = SampledMemoryPolicy {
            window: "pre_weight_load_to_shutdown".into(),
            interval_ms: 250,
            max_sample_gap_ns: 500_000_000,
        };
        let baseline = write_arm(dir.path(), &contract, false);
        let candidate = write_arm(dir.path(), &contract, true);
        let manifest = ComparisonArtifactManifest {
            schema_version: 1,
            eligibility: None,
            contract: save(dir.path(), "contract.json", &contract),
            cells: vec![CellArtifactRefs {
                concurrency: 2,
                pairs: vec![PairArtifactRefs {
                    pair_id: contract.pairs[0].pair_id.clone(),
                    baseline,
                    candidate,
                }],
            }],
        };
        Self { dir, manifest }
    }
    fn write_manifest(&self) -> PathBuf {
        save(self.dir.path(), "manifest.json", &self.manifest);
        self.dir.path().join("manifest.json")
    }
    fn compare(&self) -> Result<ArtifactComparisonReport, ArtifactError> {
        compare_manifest(&self.write_manifest(), &ArtifactLoadLimits::default())
    }
    fn candidate(&mut self) -> &mut ArmArtifactRefs {
        &mut self.manifest.cells[0].pairs[0].candidate
    }
    fn replace_metal(&mut self, edit: impl FnOnce(&mut Vec<MetalRecord>)) {
        let candidate = &self.manifest.cells[0].pairs[0].candidate;
        let execution: ExecutionArtifact = read(self.dir.path(), &candidate.execution);
        let mut rows = metal_rows(&execution);
        edit(&mut rows);
        let file = save_rows(self.dir.path(), "candidate-metal.jsonl", &rows);
        self.candidate().memory.device = Some(DeviceMemoryRef::FerrumMetalV1 { file });
    }
}

#[test]
fn actual_sidecar_and_native_metal_samples_produce_descriptive_tables() {
    let fixture = Fixture::new();
    let report = fixture.compare().unwrap();
    assert_eq!(
        report.comparison.status,
        ComparisonStatus::Inconclusive,
        "{:#?}",
        report.comparison
    );
    let memory = report.comparison.cells[0].pairs[0]
        .candidate_memory
        .as_ref()
        .unwrap();
    assert_eq!(memory.device_allocation.peak_bytes, Some(1_000_000_014));
    assert_eq!(memory.device_allocation.sample_count, Some(15));
    assert_eq!(
        memory.device_allocation.max_sample_gap_ns,
        Some(250_000_000)
    );
    assert_eq!(report.verified_files.len(), 12); // manifest, contract, five artifacts per arm
    assert!(report
        .to_markdown(MarkdownLanguage::English)
        .contains("Arm absolute SLO"));
    assert!(report
        .to_markdown(MarkdownLanguage::Chinese)
        .contains("不证明硬件真实执行"));
}

#[test]
fn corrupted_bytes_cannot_be_replaced_by_a_declared_digest() {
    let fixture = Fixture::new();
    let path = fixture
        .dir
        .path()
        .join(&fixture.manifest.cells[0].pairs[0].candidate.execution.path);
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[0] = b'[';
    std::fs::write(path, bytes).unwrap();
    assert!(fixture
        .compare()
        .unwrap_err()
        .to_string()
        .contains("SHA-256 mismatch"));
}

#[test]
fn contract_and_run_selection_identity_are_bound_to_original_files() {
    let mut fixture = Fixture::new();
    let mut execution: ExecutionArtifact = read(
        fixture.dir.path(),
        &fixture.manifest.cells[0].pairs[0].candidate.execution,
    );
    execution.identity.benchmark_run_id = "another-run".into();
    let file = save(fixture.dir.path(), "candidate-execution.json", &execution);
    fixture.candidate().execution = file;
    assert!(fixture
        .compare()
        .unwrap_err()
        .to_string()
        .contains("identity mismatch"));
    let mut fixture = Fixture::new();
    fixture.candidate().selection_sha256 = digest(b"another selection");
    assert!(fixture
        .compare()
        .unwrap_err()
        .to_string()
        .contains("ordered samples"));
}

#[test]
fn oversized_files_rows_samples_and_total_bytes_are_rejected() {
    let fixture = Fixture::new();
    let path = fixture.write_manifest();
    for limits in [
        ArtifactLoadLimits {
            max_file_bytes: 1,
            ..Default::default()
        },
        ArtifactLoadLimits {
            max_total_bytes: 1,
            ..Default::default()
        },
        ArtifactLoadLimits {
            max_files: 1,
            ..Default::default()
        },
        ArtifactLoadLimits {
            max_memory_samples: 2,
            ..Default::default()
        },
        ArtifactLoadLimits {
            max_requests_per_repeat: 2,
            ..Default::default()
        },
        ArtifactLoadLimits {
            max_total_visible_gaps: 1,
            ..Default::default()
        },
    ] {
        assert!(compare_manifest(&path, &limits).is_err(), "{limits:?}");
    }
}

#[test]
fn relative_references_cannot_escape_the_manifest_directory() {
    let mut fixture = Fixture::new();
    fixture.candidate().execution.path = "../execution.json".into();
    assert!(fixture
        .compare()
        .unwrap_err()
        .to_string()
        .contains("relative"));
}

#[cfg(unix)]
#[test]
fn symlink_escape_is_rejected_before_reading_external_bytes() {
    let mut fixture = Fixture::new();
    let outside = tempfile::tempdir().unwrap();
    let source = fixture
        .dir
        .path()
        .join(&fixture.manifest.cells[0].pairs[0].candidate.execution.path);
    std::fs::copy(source, outside.path().join("execution.json")).unwrap();
    std::os::unix::fs::symlink(
        outside.path().join("execution.json"),
        fixture.dir.path().join("escape.json"),
    )
    .unwrap();
    fixture.candidate().execution.path = "escape.json".into();
    assert!(fixture
        .compare()
        .unwrap_err()
        .to_string()
        .contains("escapes"));
}

#[test]
fn memory_summary_cannot_supply_a_peak_missing_from_raw_samples() {
    let mut fixture = Fixture::new();
    fixture.replace_metal(|rows| rows.last_mut().unwrap().peak_allocated_bytes += 100);
    assert!(fixture
        .compare()
        .unwrap_err()
        .to_string()
        .contains("raw samples"));
}

#[test]
fn wrong_process_and_regressed_memory_clock_are_rejected() {
    let mut fixture = Fixture::new();
    fixture.replace_metal(|rows| {
        for row in rows {
            row.pid += 100;
        }
    });
    assert!(fixture
        .compare()
        .unwrap_err()
        .to_string()
        .contains("identity mismatch"));
    let mut fixture = Fixture::new();
    fixture.replace_metal(|rows| rows[2].elapsed_ns = 0);
    assert!(fixture
        .compare()
        .unwrap_err()
        .to_string()
        .contains("regressed"));
}

#[test]
fn unfinished_or_absent_memory_stays_unknown_and_raw_peak_is_retained() {
    let mut fixture = Fixture::new();
    fixture.replace_metal(|rows| {
        rows.pop();
    });
    let report = fixture.compare().unwrap();
    assert_eq!(report.comparison.status, ComparisonStatus::Unknown);
    assert!(report
        .to_markdown(MarkdownLanguage::English)
        .contains("unverified max"));
    fixture.candidate().memory = MemoryArtifactRefs::default();
    assert_eq!(
        fixture.compare().unwrap().comparison.status,
        ComparisonStatus::Unknown
    );
}

#[test]
fn raw_external_observations_are_derived_and_sampler_errors_remain_unknown() {
    let mut fixture = Fixture::new();
    let refs = &fixture.manifest.cells[0].pairs[0].candidate;
    let mut samples: SampledMemoryArtifact =
        read(fixture.dir.path(), refs.memory.footprint.as_ref().unwrap());
    samples.measurement = MemoryMeasurement::SampledDeviceAllocation;
    samples.window = "pre_weight_load_to_shutdown".into();
    samples.interval_ms = 250;
    samples.observations = (0..=14)
        .map(|index| MemoryObservation {
            elapsed_ns: index * 250_000_000,
            bytes: Some(1_000 + index),
            error: None,
        })
        .collect();
    let file = save(
        fixture.dir.path(),
        "candidate-device-observations.json",
        &samples,
    );
    fixture.candidate().memory.device = Some(DeviceMemoryRef::SampledMemoryV1 { file });
    let report = fixture.compare().unwrap();
    assert_eq!(report.comparison.status, ComparisonStatus::Inconclusive);
    assert_eq!(
        report.comparison.cells[0].pairs[0]
            .candidate_memory
            .as_ref()
            .unwrap()
            .device_allocation
            .peak_bytes,
        Some(1_014)
    );
    samples.observations[1].bytes = None;
    samples.observations[1].error = Some("query failed".into());
    let file = save(
        fixture.dir.path(),
        "candidate-device-observations.json",
        &samples,
    );
    fixture.candidate().memory.device = Some(DeviceMemoryRef::SampledMemoryV1 { file });
    assert_eq!(
        fixture.compare().unwrap().comparison.status,
        ComparisonStatus::Unknown
    );
}

#[test]
fn missing_frozen_cells_are_retained_instead_of_reporting_only_the_winner() {
    let mut fixture = Fixture::new();
    fixture.manifest.cells.clear();
    let report = fixture.compare().unwrap();
    assert_eq!(report.comparison.status, ComparisonStatus::Unknown);
    assert_eq!(report.comparison.cells.len(), 1);
    assert_eq!(report.comparison.cells[0].pairs.len(), 1);
}

#[test]
fn jsonl_record_selection_uses_nonempty_rows_and_bounds_indices() {
    let mut fixture = Fixture::new();
    let candidate = &fixture.manifest.cells[0].pairs[0].candidate;
    let bytes = std::fs::read(fixture.dir.path().join(&candidate.sidecar.file.path)).unwrap();
    let mut joined = b"\n \n".to_vec();
    joined.extend_from_slice(&bytes);
    joined.extend_from_slice(b"\n");
    joined.extend_from_slice(&bytes);
    let file = save_bytes(fixture.dir.path(), "candidate-sidecar.jsonl", &joined);
    fixture.candidate().sidecar = SidecarRef {
        file,
        record_index: 1,
    };
    assert_eq!(
        fixture.compare().unwrap().comparison.status,
        ComparisonStatus::Inconclusive
    );
    fixture.candidate().sidecar.record_index = 2;
    assert!(fixture
        .compare()
        .unwrap_err()
        .to_string()
        .contains("out of range"));
}

#[test]
fn artifact_versions_and_unknown_manifest_fields_fail_closed() {
    let mut fixture = Fixture::new();
    fixture.manifest.schema_version = 2;
    assert!(fixture.compare().is_err());
    fixture.manifest.schema_version = 1;
    let mut value = serde_json::to_value(&fixture.manifest).unwrap();
    value["proof_pass"] = true.into();
    let path = fixture.dir.path().join("manifest.json");
    std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    assert!(compare_manifest(&path, &ArtifactLoadLimits::default())
        .unwrap_err()
        .to_string()
        .contains("unknown field"));
}
