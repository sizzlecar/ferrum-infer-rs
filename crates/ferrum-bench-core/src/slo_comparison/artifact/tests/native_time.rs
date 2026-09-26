use super::*;

const NATIVE_REPORT: &[u8] = b" 4.68 real 2.79 user 0.44 sys\n 477102080 maximum resident set size\n 0 swaps\n 523634920 peak memory footprint\n";

fn memory_table_cell<'a>(markdown: &'a str, arm: &str, column: &str) -> &'a str {
    let mut lines = markdown.lines();
    let header = lines
        .find(|line| line.starts_with("| Scope |") || line.starts_with("| 范围 |"))
        .expect("primary evidence table");
    let headings: Vec<_> = header.split('|').map(str::trim).collect();
    let arm_column = headings
        .iter()
        .position(|heading| matches!(*heading, "Implementation" | "实现"))
        .expect("implementation column");
    let value_column = headings
        .iter()
        .position(|heading| *heading == column)
        .expect("memory column");
    lines
        .take_while(|line| line.starts_with('|'))
        .find_map(|line| {
            let values: Vec<_> = line.split('|').map(str::trim).collect();
            (values.get(arm_column) == Some(&arm))
                .then(|| *values.get(value_column).expect("memory value"))
        })
        .unwrap_or_else(|| panic!("no table row for {arm}: {markdown}"))
}

fn fixture() -> Fixture {
    let mut fixture = Fixture::new();
    let mut contract: FrozenComparisonContract =
        read(fixture.dir.path(), &fixture.manifest.contract);
    contract.memory.os_footprint =
        OsFootprintPolicy::ProcessLifetime(ProcessLifetimeMemoryPolicy {
            kind: ProcessLifetimeMemoryKind::MacosTimeLV1,
        });
    contract.memory.maximum_rss_window = "process_lifetime".into();
    fixture.manifest.contract = save(fixture.dir.path(), "contract.json", &contract);
    let pair = &mut fixture.manifest.cells[0].pairs[0];
    for (label, arm) in [
        ("baseline", &mut pair.baseline),
        ("candidate", &mut pair.candidate),
    ] {
        let execution: ExecutionArtifact = read(fixture.dir.path(), &arm.execution);
        let capture = MacosTimeCaptureArtifact {
            schema_version: 1,
            identity: execution.identity,
            subject: MacosTimeSubject::DeclaredDirectServerChild,
            process_started_unix_ns: execution.measurement_started_unix_ns - 250_000_000,
            process_ended_unix_ns: Some(execution.measurement_ended_unix_ns + 250_000_000),
            time_output: save_bytes(fixture.dir.path(), &format!("{label}.time"), NATIVE_REPORT),
            exit_status: Some(save_bytes(
                fixture.dir.path(),
                &format!("{label}.exit"),
                b"0\n",
            )),
            launch_evidence: Some(save_bytes(
                fixture.dir.path(),
                &format!("{label}.command"),
                b"/usr/bin/time -l actual-server serve\n",
            )),
        };
        arm.memory.footprint = None;
        arm.memory.rss = None;
        arm.memory.process_lifetime = Some(ProcessLifetimeMemoryRef::MacosTimeLV1 {
            capture: save(
                fixture.dir.path(),
                &format!("{label}-capture.json"),
                &capture,
            ),
        });
    }
    fixture
}

fn edit_capture(fixture: &mut Fixture, edit: impl FnOnce(&Path, &mut MacosTimeCaptureArtifact)) {
    let Some(ProcessLifetimeMemoryRef::MacosTimeLV1 { capture }) = &fixture.manifest.cells[0].pairs
        [0]
    .candidate
    .memory
    .process_lifetime
    else {
        panic!("native fixture");
    };
    let mut value: MacosTimeCaptureArtifact = read(fixture.dir.path(), capture);
    edit(fixture.dir.path(), &mut value);
    let capture = save(fixture.dir.path(), "candidate-capture.json", &value);
    fixture.candidate().memory.process_lifetime =
        Some(ProcessLifetimeMemoryRef::MacosTimeLV1 { capture });
}

#[test]
fn original_native_report_drives_both_columns_without_sampling_claims() {
    let fixture = fixture();
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
    assert_eq!(
        memory.os_footprint.measurement,
        MemoryMeasurement::ProcessPeakPhysicalFootprint
    );
    assert_eq!(memory.os_footprint.peak_bytes, Some(523_634_920));
    assert_eq!(memory.maximum_rss.peak_bytes, Some(477_102_080));
    for peak in [&memory.os_footprint, &memory.maximum_rss] {
        assert!(peak.complete);
        assert_eq!(peak.source_sha256, digest(NATIVE_REPORT));
        assert_eq!(
            (peak.sample_count, peak.interval_ms, peak.max_sample_gap_ns),
            (None, None, None)
        );
    }
    let en = report.to_markdown(MarkdownLanguage::English);
    let zh = report.to_markdown(MarkdownLanguage::Chinese);
    assert!(en.contains("whole process lifetime peaks"));
    assert!(zh.contains("整个进程寿命峰值"));
    for markdown in [&en, &zh] {
        for arm in [
            &report.comparison.contract.baseline.implementation,
            &report.comparison.contract.candidate.implementation,
        ] {
            assert_eq!(
                memory_table_cell(markdown, arm, "OS footprint GiB"),
                "0.488",
                "{markdown}"
            );
            assert_eq!(
                memory_table_cell(markdown, arm, "RSS GiB"),
                "0.444",
                "{markdown}"
            );
        }
    }
    assert!(report
        .verified_files
        .iter()
        .any(|file| file.path == Path::new("candidate.time")));
    assert!(report
        .verified_files
        .iter()
        .any(|file| file.path == Path::new("candidate.exit")));
    assert!(report
        .verified_files
        .iter()
        .any(|file| file.path == Path::new("candidate.command")));
}

#[test]
fn incomplete_capture_or_unsuccessful_exit_keeps_raw_peaks_unknown() {
    for case in 0..4 {
        let mut fixture = fixture();
        edit_capture(&mut fixture, |dir, capture| match case {
            0 => capture.exit_status = Some(save_bytes(dir, "candidate.exit", b"130\n")),
            1 => capture.exit_status = None,
            2 => capture.process_ended_unix_ns = None,
            _ => {
                capture.time_output =
                    save_bytes(dir, "candidate.time", b"523634920 peak memory footprint\n")
            }
        });
        let report = fixture.compare().unwrap();
        assert_eq!(
            report.comparison.status,
            ComparisonStatus::Unknown,
            "case {case}: {:#?}",
            report.comparison
        );
        let memory = report.comparison.cells[0].pairs[0]
            .candidate_memory
            .as_ref()
            .unwrap();
        assert_eq!(memory.os_footprint.peak_bytes, Some(523_634_920));
        assert!(!memory.os_footprint.complete);
        assert_eq!(memory.os_footprint.error_count, u64::from(case == 0));
        let markdown = report.to_markdown(MarkdownLanguage::English);
        let footprint = memory_table_cell(
            &markdown,
            &report.comparison.contract.candidate.implementation,
            "OS footprint GiB",
        );
        assert!(
            footprint.starts_with("Unknown (unverified max 0.488;"),
            "case {case}: {markdown}"
        );
    }
}

#[test]
fn process_window_must_enclose_measurement_and_policy_must_match() {
    let mut fixture = fixture();
    edit_capture(&mut fixture, |_, capture| {
        capture.process_started_unix_ns += 500_000_000
    });
    let report = fixture.compare().unwrap();
    assert_eq!(
        report.comparison.status,
        ComparisonStatus::Unknown,
        "{:#?}",
        report.comparison
    );

    let mut fixture = self::fixture();
    let mut contract: FrozenComparisonContract =
        read(fixture.dir.path(), &fixture.manifest.contract);
    contract.memory.os_footprint = SampledMemoryPolicy {
        window: "process_lifetime".into(),
        interval_ms: 100,
        max_sample_gap_ns: 500_000_000,
    }
    .into();
    fixture.manifest.contract = save(fixture.dir.path(), "contract.json", &contract);
    let report = fixture.compare().unwrap();
    assert_eq!(
        report.comparison.status,
        ComparisonStatus::Unknown,
        "{:#?}",
        report.comparison
    );
}

#[test]
fn identity_conflicts_invalid_lifetime_and_manual_values_are_load_errors() {
    for case in 0..3 {
        let mut fixture = fixture();
        edit_capture(&mut fixture, |_, capture| match case {
            0 => capture.identity.server_pid += 1,
            1 => capture.identity.repeat_index += 1,
            _ => capture.process_ended_unix_ns = Some(capture.process_started_unix_ns),
        });
        assert!(fixture
            .compare()
            .unwrap_err()
            .0
            .contains("identity/lifetime"));
    }
    let fixture = fixture();
    let Some(ProcessLifetimeMemoryRef::MacosTimeLV1 { capture }) = &fixture.manifest.cells[0].pairs
        [0]
    .candidate
    .memory
    .process_lifetime
    else {
        panic!("fixture");
    };
    let mut value: serde_json::Value = read(fixture.dir.path(), capture);
    value["peak_bytes"] = 1.into();
    assert!(serde_json::from_value::<MacosTimeCaptureArtifact>(value).is_err());
}

#[test]
fn native_sources_cannot_be_overridden_or_escape_hash_and_total_limits() {
    let mut fixture = fixture();
    let manual = save_bytes(fixture.dir.path(), "manual.json", b"{}");
    fixture.candidate().memory.rss = Some(manual);
    assert!(fixture.compare().unwrap_err().0.contains("overrides"));

    let mut fixture = self::fixture();
    edit_capture(&mut fixture, |_, capture| {
        capture.time_output.sha256 = "0".repeat(64)
    });
    assert!(fixture.compare().is_err());

    let fixture = self::fixture();
    let report = fixture.compare().unwrap();
    let verified_bytes: u64 = report.verified_files.iter().map(|file| file.bytes).sum();
    let limits = ArtifactLoadLimits {
        max_total_bytes: verified_bytes - 1,
        ..Default::default()
    };
    assert!(compare_manifest(&fixture.write_manifest(), &limits).is_err());
}

#[test]
fn missing_peak_does_not_become_complete_through_declaration() {
    let mut fixture = fixture();
    edit_capture(&mut fixture, |dir, capture| {
        capture.time_output = save_bytes(dir, "candidate.time", b"4.68 real 2.79 user 0.44 sys\n");
    });
    let report = fixture.compare().unwrap();
    assert_eq!(
        report.comparison.status,
        ComparisonStatus::Unknown,
        "{:#?}",
        report.comparison
    );
    let memory = report.comparison.cells[0].pairs[0]
        .candidate_memory
        .as_ref()
        .unwrap();
    assert_eq!(memory.os_footprint.peak_bytes, None);
    assert_eq!(memory.maximum_rss.peak_bytes, None);
}

#[test]
fn lifetime_evidence_cannot_claim_samples_or_a_different_rss_window() {
    let fixture = fixture();
    let report = fixture.compare().unwrap();
    let pair = &report.comparison.cells[0].pairs[0];
    let mut peak = pair.candidate_memory.as_ref().unwrap().os_footprint.clone();
    peak.sample_count = Some(1);
    let issues = crate::slo_comparison::evidence::memory_peak_issues(
        &peak,
        MemoryMeasurement::ProcessPeakPhysicalFootprint,
        &pair.candidate_source.as_ref().unwrap().execution,
        &report.comparison.contract.memory,
    );
    assert!(!issues.is_empty());
    let mut contract = report.comparison.contract;
    contract.memory.maximum_rss_window = "measurement_only".into();
    assert!(contract.validate().is_err());
}
