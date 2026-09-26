use super::*;
use crate::slo_comparison::statistics::eligibility::tests::Fixture;
use crate::slo_comparison::tests::OwnedArm;
use serde::Serialize;
use tempfile::TempDir;

fn save<T: Serialize>(root: &Path, name: &str, value: &T) -> ArtifactFileRef {
    let bytes = serde_json::to_vec(value).unwrap();
    std::fs::write(root.join(name), &bytes).unwrap();
    ArtifactFileRef {
        path: name.into(),
        sha256: digest(&bytes),
        bytes: bytes.len() as u64,
    }
}

fn write_arm(
    root: &Path,
    contract: &FrozenComparisonContract,
    pair: usize,
    arm: &OwnedArm,
    pid: u32,
) -> ArmArtifactRefs {
    let label = arm.report.benchmark_run_id.as_ref().unwrap();
    let identity = RunArtifactIdentity {
        benchmark_run_id: label.clone(),
        cell_id: arm.report.cell_id.clone().unwrap(),
        repeat_index: 0,
        server_pid: pid,
    };
    let execution = ExecutionArtifact {
        schema_version: 1,
        identity: identity.clone(),
        shared: arm.execution.shared.clone(),
        server: arm.execution.server.clone(),
        capacity: arm.execution.capacity.clone(),
        measurement_started_unix_ns: arm.execution.measurement_started_unix_ns,
        measurement_ended_unix_ns: arm.execution.measurement_ended_unix_ns,
        device_registry_id: None,
    };
    let sidecar = SloSidecarCell {
        schema_version: 1,
        config_sha256: contract.shared.client_slo_config_sha256.clone(),
        legacy_benchmark: arm.report.clone(),
        repeats: vec![SloSidecarRepeat {
            repeat_index: 0,
            evaluation: arm.evaluation.clone(),
            arrivals: vec![SidecarArrival::default(); arm.evaluation.request_evidence.len()],
            request_start_window_s: None,
            observed_request_start_rate_rps: None,
            max_client_dispatch_backlog: None,
        }],
    };
    let anchor = execution.measurement_started_unix_ns - 250_000_000;
    let observations = |measurement, window: String| SampledMemoryArtifact {
        schema_version: 1,
        identity: identity.clone(),
        measurement,
        collector: "typed raw fixture sampler".into(),
        window,
        started_unix_ns: anchor,
        interval_ms: 100,
        observations: (0..=35)
            .map(|index| MemoryObservation {
                elapsed_ns: index * 100_000_000,
                bytes: Some(1_000_000 + index),
                error: None,
            })
            .collect(),
        finished_elapsed_ns: Some(3_500_000_000),
    };
    let device = observations(
        MemoryMeasurement::SampledDeviceAllocation,
        contract.memory.device_allocation.window.clone(),
    );
    let footprint = observations(
        MemoryMeasurement::SampledOsPhysicalFootprint,
        contract.memory.os_footprint.window().into(),
    );
    let rss = MaximumRssArtifact {
        schema_version: 1,
        identity,
        collector: "typed raw process fixture".into(),
        window: contract.memory.maximum_rss_window.clone(),
        process_started_unix_ns: anchor,
        process_ended_unix_ns: Some(anchor + 3_500_000_000),
        maximum_rss_bytes: Some(2_000_000),
        error: None,
    };
    ArmArtifactRefs {
        sidecar: SidecarRef {
            file: save(root, &format!("{label}-sidecar.jsonl"), &sidecar),
            record_index: 0,
        },
        repeat_index: 0,
        selection_sha256: contract.pairs[pair].selection.selection_sha256.clone(),
        execution: save(root, &format!("{label}-execution.json"), &execution),
        memory: MemoryArtifactRefs {
            device: Some(DeviceMemoryRef::SampledMemoryV1 {
                file: save(root, &format!("{label}-device.json"), &device),
            }),
            footprint: Some(save(root, &format!("{label}-footprint.json"), &footprint)),
            rss: Some(save(root, &format!("{label}-rss.json"), &rss)),
            process_lifetime: None,
        },
    }
}

struct ArtifactFixture {
    dir: TempDir,
    manifest: ComparisonArtifactManifest,
}
impl ArtifactFixture {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();
        let mut fixture = Fixture::new();
        let arms = |contract: &FrozenComparisonContract, pairs: &[(OwnedArm, OwnedArm)]| {
            vec![CellArtifactRefs {
                concurrency: 2,
                pairs: pairs
                    .iter()
                    .enumerate()
                    .map(|(index, (b, c))| PairArtifactRefs {
                        pair_id: contract.pairs[index].pair_id.clone(),
                        baseline: write_arm(root, contract, index, b, 100 + index as u32 * 2),
                        candidate: write_arm(root, contract, index, c, 101 + index as u32 * 2),
                    })
                    .collect(),
            }]
        };
        let pilot = ComparisonArtifactManifest {
            schema_version: 1,
            contract: save(root, "pilot-contract.json", &fixture.pilot_contract),
            cells: arms(&fixture.pilot_contract, &fixture.pilot),
            eligibility: None,
        };
        let pilot_manifest = save(root, "pilot-manifest.json", &pilot);
        let mut config = fixture
            .contract
            .uncertainty
            .as_ref()
            .unwrap()
            .paired_bootstrap
            .clone()
            .unwrap();
        config.declared_design.pilot_source_sha256 = pilot_manifest.sha256.clone();
        fixture.contract.uncertainty =
            Some(FrozenStatisticalMethod::paired_cluster_bootstrap(config).unwrap());
        let manifest = ComparisonArtifactManifest {
            schema_version: 1,
            contract: save(root, "main-contract.json", &fixture.contract),
            cells: arms(&fixture.contract, &fixture.main),
            eligibility: Some(EligibilityArtifactRefs {
                plan: save(root, "plan.json", &fixture.plan),
                pilot_manifest,
            }),
        };
        Self { dir, manifest }
    }
    fn compare(
        &self,
        limits: &ArtifactLoadLimits,
    ) -> Result<ArtifactComparisonReport, ArtifactError> {
        save(self.dir.path(), "manifest.json", &self.manifest);
        compare_manifest(&self.dir.path().join("manifest.json"), limits)
    }
}

#[test]
fn hash_pinned_raw_pilot_manifest_can_reach_conditional_proof_without_a_certificate() {
    let fixture = ArtifactFixture::new();
    let report = fixture.compare(&ArtifactLoadLimits::default()).unwrap();
    assert_eq!(
        report.comparison.status,
        ComparisonStatus::ProofPass,
        "{report:#?}"
    );
    assert!(report.comparison.inference_eligibility.is_some());
    assert!(report
        .to_markdown(MarkdownLanguage::English)
        .contains("approximate"));
    assert!(report
        .to_markdown(MarkdownLanguage::Chinese)
        .contains("假设"));
}

#[test]
fn omitted_pilot_never_passes_and_pilot_bytes_share_the_outer_total_budget() {
    let mut fixture = ArtifactFixture::new();
    let report = fixture.compare(&ArtifactLoadLimits::default()).unwrap();
    let total_bytes = report
        .verified_files
        .iter()
        .map(|file| file.bytes)
        .sum::<u64>();
    let limits = ArtifactLoadLimits {
        max_total_bytes: total_bytes - 1,
        ..Default::default()
    };
    assert!(fixture.compare(&limits).is_err());
    fixture.manifest.eligibility = None;
    let report = fixture.compare(&ArtifactLoadLimits::default()).unwrap();
    assert_eq!(
        report.comparison.status,
        ComparisonStatus::Inconclusive,
        "{report:#?}"
    );
}

#[test]
fn recursive_pilot_and_modified_planning_document_cannot_unlock_proof() {
    let mut fixture = ArtifactFixture::new();
    let refs = fixture.manifest.eligibility.as_mut().unwrap();
    let bytes = std::fs::read(fixture.dir.path().join(&refs.pilot_manifest.path)).unwrap();
    let mut pilot: ComparisonArtifactManifest = serde_json::from_slice(&bytes).unwrap();
    pilot.eligibility = Some(refs.clone());
    refs.pilot_manifest = save(fixture.dir.path(), "pilot-manifest.json", &pilot);
    assert!(fixture
        .compare(&ArtifactLoadLimits::default())
        .unwrap_err()
        .to_string()
        .contains("recursively"));
    let mut fixture = ArtifactFixture::new();
    let refs = fixture.manifest.eligibility.as_mut().unwrap();
    let bytes = std::fs::read(fixture.dir.path().join(&refs.plan.path)).unwrap();
    let mut plan: FrozenEligibilityPlan = serde_json::from_slice(&bytes).unwrap();
    plan.maximum_order_log_ratio_shift *= 2.0;
    refs.plan = save(fixture.dir.path(), "plan.json", &plan);
    let report = fixture.compare(&ArtifactLoadLimits::default()).unwrap();
    assert_eq!(report.comparison.status, ComparisonStatus::Inconclusive);
    assert!(report.comparison.inference_eligibility_failure.is_some());
}
