//! Real process separation for profile14. The child can access only artifacts
//! and declared clocks; no live collector/model is transferred. Test environment
//! keys select this ignored harness helper, never a product behavior.
use super::*;
use std::process::Command;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Case {
    Predict,
    WrongFingerprint,
    LargerPhysicalDomain,
    Expired,
    TamperedJournal,
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Job {
    profile: PathBuf,
    result: PathBuf,
    case: Case,
    wall_ns: u64,
    monotonic_ns: u64,
}
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "result", rename_all = "snake_case", deny_unknown_fields)]
enum Outcome {
    Predicted {
        schema_version: u32,
        planning_ns: u64,
        valid_until_ns: u64,
        oldest_age_ns: u64,
        newest_age_ns: u64,
        parameters_sha256: [u8; 32],
        source_sha256: [u8; 32],
        source_bytes: u64,
        journal_bytes: u64,
        domain_sha256: [u8; 32],
        monotonic_expiry_rejected: bool,
    },
    Rejected {
        case: Case,
    },
}
#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ChildResult {
    child_pid: u32,
    outcome: Outcome,
}
fn child_job(
    dir: &Path,
    profile: &Path,
    case: Case,
    wall_ns: u64,
    monotonic_ns: u64,
    ordinal: usize,
) -> ChildResult {
    let result = dir.join(format!("child-{ordinal}.json"));
    let job_path = dir.join(format!("job-{ordinal}.json"));
    let job = Job {
        profile: profile.to_path_buf(),
        result: result.clone(),
        case,
        wall_ns,
        monotonic_ns,
    };
    std::fs::write(&job_path, serde_json::to_vec(&job).unwrap()).unwrap();
    let name = format!(
        "{}::source7_profile14_import_child",
        module_path!().split_once("::").unwrap().1
    );
    let status = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", &name, "--ignored"])
        .env("FERRUM_TEST_PROFILE14_IMPORT_JOB", &job_path)
        .status()
        .unwrap();
    assert!(
        status.success(),
        "independent Rust import helper failed: {status}"
    );
    // Reading a structured child result also rejects a harness selecting no test.
    let report: ChildResult = serde_json::from_slice(&std::fs::read(result).unwrap()).unwrap();
    assert_ne!(report.child_pid, std::process::id());
    report
}
#[test]
fn source7_profile14_independent_process_import_predicts_and_preserves_original_age_identity() {
    check_independent_import(collected());
}

#[test]
fn empirical_fitted_profile_independent_process_preserves_prediction_age_and_identity() {
    use crate::implementations::continuous::cost_model::structured_v2::NonNegativePlanningEstimatorV1;
    let mut original = block_header();
    original
        .declaration
        .nonnegative_envelope
        .as_mut()
        .unwrap()
        .planning_estimator = NonNegativePlanningEstimatorV1::FittedResidualV1;
    // Mint the declaration and all records with the selected contract before
    // collection; never rewrite an already-qualified legacy journal.
    let header = StructuredServiceHeaderV7::new(
        original.capture_identity,
        original.generation,
        original.fingerprint,
        original.producer,
        original.opening,
        original.declaration,
        original.maximum_file_bytes,
    )
    .unwrap();
    check_independent_import(collected_with_header(header));
}

fn check_independent_import(
    collected: (
        Vec<u8>,
        StructuredServiceCollectorV7,
        StructuredServiceCheckpointV7,
        StructuredServiceClockV7,
    ),
) {
    let (mut source, mut collector, checkpoint, closing) = collected;
    let source_bytes = source.len() as u64;
    let source_sha256: [u8; 32] = Sha256::digest(&source).into();
    let limits = CostProfileLoadLimits::default();
    let memory = checkpoint
        .activate_same_process_memory(paired(70_000), &limits)
        .unwrap();
    let expected = memory.children[0]
        .predict_query_local(
            &old::fingerprint(),
            &old::future_query_with_domain(&domain()),
            70_000,
        )
        .unwrap();
    let parameters_sha256 = memory.children[0].parameters_signature();
    // Freeze a complete immutable prefix, then retain a later incomplete block
    // and failure in the final journal. The loader must bind both byte ranges.
    append(&mut source, &collector.open_block(65_999, 96).unwrap());
    let record = block_wave(33);
    collector.push(&record).unwrap();
    append(&mut source, &record);
    append(
        &mut source,
        &collector
            .fail(34, 102, 68_000, "original ticket lost")
            .unwrap(),
    );
    append(&mut source, &collector.stop(paired(68_001)).unwrap());
    let files = Files::new(&source);
    export_structured_profile_v14(
        &files.source,
        Sha256::digest(&source).into(),
        source_bytes,
        &files.profile,
        0,
        &limits,
    )
    .unwrap();
    drop(collector);
    drop(memory);
    assert!(closing.monotonic_ns < 70_000);
    let first = child_job(
        &files.dir,
        &files.profile,
        Case::Predict,
        paired(70_000).wall_unix_ns,
        17,
        0,
    );
    let second = child_job(
        &files.dir,
        &files.profile,
        Case::Predict,
        paired(71_000).wall_unix_ns,
        900,
        1,
    );
    let (first_age, second_age) = match (&first.outcome, &second.outcome) {
        (
            Outcome::Predicted {
                schema_version,
                planning_ns,
                valid_until_ns,
                oldest_age_ns,
                parameters_sha256: got_parameters,
                source_sha256: got_source,
                source_bytes: got_bytes,
                journal_bytes,
                domain_sha256,
                monotonic_expiry_rejected,
                ..
            },
            Outcome::Predicted {
                planning_ns: later_plan,
                valid_until_ns: later_expiry,
                oldest_age_ns: later_age,
                monotonic_expiry_rejected: later_rejected,
                ..
            },
        ) => {
            assert_eq!(*schema_version, 14);
            assert_eq!(*planning_ns, expected.planning_ns);
            assert_eq!(*later_plan, expected.planning_ns);
            assert_eq!(*valid_until_ns, expected.valid_until_ns);
            assert_eq!(*later_expiry, expected.valid_until_ns);
            assert_eq!(*got_parameters, parameters_sha256);
            assert_eq!(*got_source, source_sha256);
            assert_eq!(*got_bytes, source_bytes);
            assert_eq!(*journal_bytes, source.len() as u64);
            assert_eq!(domain_sha256, domain().sha256());
            assert!(*monotonic_expiry_rejected && *later_rejected);
            (*oldest_age_ns, *later_age)
        }
        other => panic!("unexpected typed child results: {other:?}"),
    };
    assert_eq!(second_age - first_age, 1_000);
    for (ordinal, case) in [
        (2, Case::WrongFingerprint),
        (3, Case::LargerPhysicalDomain),
        (4, Case::Expired),
    ] {
        let wall = if case == Case::Expired {
            paired(70_000 + block_header().declaration.settings.max_sample_age_ns).wall_unix_ns
        } else {
            paired(70_000).wall_unix_ns
        };
        assert_eq!(
            child_job(&files.dir, &files.profile, case, wall, 71, ordinal).outcome,
            Outcome::Rejected { case }
        );
    }
    // Tamper a later journal byte, beyond the immutable checkpoint. It still
    // invalidates the file artifact even though the qualified prefix is intact.
    let last = source.len() - 2;
    source[last] ^= 1;
    std::fs::write(&files.source, &source).unwrap();
    assert_eq!(
        child_job(
            &files.dir,
            &files.profile,
            Case::TamperedJournal,
            paired(70_000).wall_unix_ns,
            71,
            5
        )
        .outcome,
        Outcome::Rejected {
            case: Case::TamperedJournal
        }
    );
}
#[test]
#[ignore = "subprocess-only profile14 importer driven by a parent-owned typed job"]
fn source7_profile14_import_child() {
    let path = PathBuf::from(
        std::env::var_os("FERRUM_TEST_PROFILE14_IMPORT_JOB").expect("Rust test parent job"),
    );
    let job: Job = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    let mut fingerprint = old::fingerprint();
    if job.case == Case::WrongFingerprint {
        fingerprint.execution_config[0] ^= 1;
    }
    let loaded = load_structured_profile_v14(
        &job.profile,
        &fingerprint,
        &CostProfileLoadLimits::default(),
        ProfileLoadClock {
            wall_unix_ns: Some(job.wall_ns),
            wall_max_error_ns: Some(0),
            monotonic_now_ns: job.monotonic_ns,
        },
    );
    let outcome = match job.case {
        Case::WrongFingerprint => {
            assert!(matches!(loaded, Err(CostProfileError::FingerprintMismatch)));
            Outcome::Rejected { case: job.case }
        }
        Case::Expired => {
            assert!(matches!(loaded, Err(CostProfileError::Clock(_))));
            Outcome::Rejected { case: job.case }
        }
        Case::TamperedJournal => {
            assert!(matches!(loaded, Err(CostProfileError::Metadata(_))));
            Outcome::Rejected { case: job.case }
        }
        Case::LargerPhysicalDomain => {
            let catalog = loaded.unwrap();
            let original = domain();
            let mut limits = *original.limits();
            limits.maximum_context_tokens =
                NonZeroU32::new(limits.maximum_context_tokens.get() + 1).unwrap();
            let larger = CostWorkloadDomainV1::new_vnext(
                &ExecutorCostIdentity {
                    schema_version: EXECUTOR_COST_IDENTITY_SCHEMA,
                    model_weights: fingerprint.model_weights,
                    numerical_policy: fingerprint.numerical_policy,
                    device_runtime: fingerprint.device_runtime,
                    execution_config: fingerprint.execution_config,
                },
                limits,
            )
            .unwrap();
            // This is the real scheduler imported-query gate, not an emulation
            // of the separate engine activation-time runtime-domain check.
            let future = old::future_query_with_domain(&larger);
            assert!(matches!(
                catalog.children[0].predict_query_local(&fingerprint, &future, job.monotonic_ns),
                Err(StructuredUnknownV2::WrongDomain)
            ));
            Outcome::Rejected { case: job.case }
        }
        Case::Predict => {
            let catalog = loaded.unwrap();
            assert_eq!(catalog.children.len(), 1);
            let model = &catalog.children[0];
            let query = old::future_query_with_domain(&domain());
            let prediction = model
                .predict_query_local(&fingerprint, &query, job.monotonic_ns)
                .unwrap();
            let model_now = model.model_now_ns(job.monotonic_ns).unwrap();
            let remaining = prediction.valid_until_ns.checked_sub(model_now).unwrap();
            let after_expiry = job
                .monotonic_ns
                .checked_add(remaining)
                .and_then(|n| n.checked_add(1))
                .unwrap();
            let monotonic_expiry_rejected = model
                .predict_query_local(&fingerprint, &query, after_expiry)
                .is_err();
            assert!(monotonic_expiry_rejected);
            Outcome::Predicted {
                schema_version: model.provenance().schema_version,
                planning_ns: prediction.planning_ns,
                valid_until_ns: prediction.valid_until_ns,
                oldest_age_ns: model.provenance().oldest_imported_age_ns,
                newest_age_ns: model.provenance().newest_imported_age_ns,
                parameters_sha256: model.parameters_signature(),
                source_sha256: catalog.source_sha256,
                source_bytes: catalog.source_bytes,
                journal_bytes: catalog.journal_bytes,
                domain_sha256: *model.workload_domain().unwrap().sha256(),
                monotonic_expiry_rejected,
            }
        }
    };
    std::fs::write(
        job.result,
        serde_json::to_vec(&ChildResult {
            child_pid: std::process::id(),
            outcome,
        })
        .unwrap(),
    )
    .unwrap();
}
