//! Family profile14 import in a fresh Rust process. Only artifact paths and
//! declared clocks cross the process boundary; the child builds future queries.
use super::*;
use crate::implementations::continuous::cost_model::structured_v2::NumericalFamilyKeyV1;
use std::process::Command;

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Job {
    profile: PathBuf,
    result: PathBuf,
    wall_ns: u64,
    monotonic_ns: u64,
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Prediction {
    rows: usize,
    planning_ns: u64,
    valid_until_ns: u64,
    remaining_ttl_ns: u64,
    monotonic_expiry_rejected: bool,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChildResult {
    child_pid: u32,
    schema_version: u32,
    predictions: [Prediction; 2],
    oldest_age_ns: u64,
    newest_age_ns: u64,
    numerical_family: NumericalFamilyKeyV1,
    parameters_sha256: [u8; 32],
    source_sha256: [u8; 32],
    capture_protocol: [u8; 32],
    source_bytes: u64,
    domain_sha256: [u8; 32],
}

fn predictions(catalog: &ImportedStructuredCatalogV14, now: u64) -> [Prediction; 2] {
    assert_eq!(catalog.children.len(), 1);
    let model = &catalog.children[0];
    let domain = model.workload_domain().unwrap();
    [1, 2].map(|rows| {
        let query = future(domain, rows, ActualWaveGraphState::Disabled);
        assert_eq!(query.owner().rows as usize, rows);
        assert_eq!(
            query.input().numerical_family_key().unwrap(),
            *model.numerical_family_key().unwrap()
        );
        let predicted = model
            .predict_query_local(&old::fingerprint(), &query, now)
            .unwrap();
        let remaining = predicted
            .valid_until_ns
            .checked_sub(model.model_now_ns(now).unwrap())
            .unwrap();
        let after_expiry = now
            .checked_add(remaining)
            .and_then(|v| v.checked_add(1))
            .unwrap();
        let monotonic_expiry_rejected = model
            .predict_query_local(&old::fingerprint(), &query, after_expiry)
            .is_err();
        assert!(monotonic_expiry_rejected);
        Prediction {
            rows,
            planning_ns: predicted.planning_ns,
            valid_until_ns: predicted.valid_until_ns,
            remaining_ttl_ns: remaining,
            monotonic_expiry_rejected,
        }
    })
}

fn child_job(files: &Files, wall_ns: u64, monotonic_ns: u64, ordinal: usize) -> ChildResult {
    let result = files.dir.join(format!("family-child-{ordinal}.json"));
    let job_path = files.dir.join(format!("family-job-{ordinal}.json"));
    std::fs::write(
        &job_path,
        serde_json::to_vec(&Job {
            profile: files.profile.clone(),
            result: result.clone(),
            wall_ns,
            monotonic_ns,
        })
        .unwrap(),
    )
    .unwrap();
    let name = format!(
        "{}::source7_family_profile14_import_child",
        module_path!().split_once("::").unwrap().1
    );
    let output = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", &name, "--ignored"])
        .env("FERRUM_TEST_FAMILY_PROFILE14_IMPORT_JOB", &job_path)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "independent family importer failed: {}; stdout={}; stderr={}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    // Requiring a typed result also rejects a harness that selected no test.
    let report: ChildResult = serde_json::from_slice(&std::fs::read(result).unwrap()).unwrap();
    assert_ne!(report.child_pid, std::process::id());
    report
}

#[test]
fn source7_family_profile14_independent_process_matches_both_future_widths_and_original_ttl() {
    let (bytes, collector, checkpoint, _) = collected_family();
    let source_sha256: [u8; 32] = Sha256::digest(&bytes).into();
    let source_bytes = bytes.len() as u64;
    let limits = CostProfileLoadLimits::default();
    let memory = checkpoint
        .activate_same_process_memory(paired(70_000), &limits)
        .unwrap();
    let expected = predictions(&memory, 70_000);
    let model = &memory.children[0];
    let family = *model.numerical_family_key().unwrap();
    let parameters_sha256 = model.parameters_signature();
    let domain_sha256 = *model.workload_domain().unwrap().sha256();
    let oldest_age_ns = model.provenance().oldest_imported_age_ns;
    let newest_age_ns = model.provenance().newest_imported_age_ns;
    let capture_protocol = memory.capture_protocol;
    let files = Files::new(&bytes);
    export_structured_profile_v14(
        &files.source,
        source_sha256,
        source_bytes,
        &files.profile,
        0,
        &limits,
    )
    .unwrap();
    drop(collector);
    drop(memory);

    // Unrelated child monotonic origins must preserve the original model clock.
    // The later import loses exactly the elapsed wall time from its remaining TTL.
    for (ordinal, elapsed, monotonic) in [(0, 0, 17), (1, 1_000, 900)] {
        let report = child_job(
            &files,
            paired(70_000 + elapsed).wall_unix_ns,
            monotonic,
            ordinal,
        );
        assert_eq!(report.schema_version, 14);
        assert_eq!(report.numerical_family, family);
        assert_eq!(report.parameters_sha256, parameters_sha256);
        assert_eq!(report.source_sha256, source_sha256);
        assert_eq!(report.capture_protocol, capture_protocol);
        assert_eq!(report.source_bytes, source_bytes);
        assert_eq!(report.domain_sha256, domain_sha256);
        assert_eq!(report.oldest_age_ns, oldest_age_ns + elapsed);
        assert_eq!(report.newest_age_ns, newest_age_ns + elapsed);
        for (actual, expected) in report.predictions.iter().zip(&expected) {
            assert_eq!(actual.rows, expected.rows);
            assert_eq!(actual.planning_ns, expected.planning_ns);
            assert_eq!(actual.valid_until_ns, expected.valid_until_ns);
            assert_eq!(actual.remaining_ttl_ns + elapsed, expected.remaining_ttl_ns);
            assert!(actual.monotonic_expiry_rejected);
        }
    }
}

#[test]
#[ignore = "subprocess-only family importer driven by a parent-owned typed job"]
fn source7_family_profile14_import_child() {
    let job_path = PathBuf::from(
        std::env::var_os("FERRUM_TEST_FAMILY_PROFILE14_IMPORT_JOB")
            .expect("Rust parent family import job"),
    );
    let job: Job = serde_json::from_slice(&std::fs::read(job_path).unwrap()).unwrap();
    let catalog = load_structured_profile_v14(
        &job.profile,
        &old::fingerprint(),
        &CostProfileLoadLimits::default(),
        ProfileLoadClock {
            wall_unix_ns: Some(job.wall_ns),
            wall_max_error_ns: Some(0),
            monotonic_now_ns: job.monotonic_ns,
        },
    )
    .unwrap();
    assert_eq!(catalog.children.len(), 1);
    let model = &catalog.children[0];
    let result = ChildResult {
        child_pid: std::process::id(),
        schema_version: model.provenance().schema_version,
        predictions: predictions(&catalog, job.monotonic_ns),
        oldest_age_ns: model.provenance().oldest_imported_age_ns,
        newest_age_ns: model.provenance().newest_imported_age_ns,
        numerical_family: *model.numerical_family_key().unwrap(),
        parameters_sha256: model.parameters_signature(),
        source_sha256: catalog.source_sha256,
        capture_protocol: catalog.capture_protocol,
        source_bytes: catalog.source_bytes,
        domain_sha256: *model.workload_domain().unwrap().sha256(),
    };
    std::fs::write(job.result, serde_json::to_vec(&result).unwrap()).unwrap();
}
