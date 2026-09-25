use super::*;
mod fixture;
use fixture::*;
struct Files {
    dir: PathBuf,
    source: PathBuf,
    profile: PathBuf,
}
impl Files {
    fn new(bytes: &[u8]) -> Self {
        let dir =
            std::env::temp_dir().join(format!("ferrum-structured-v9-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&dir).unwrap();
        let source = dir.join("source.jsonl");
        let profile = dir.join("profile.json");
        std::fs::write(&source, bytes).unwrap();
        Self {
            dir,
            source,
            profile,
        }
    }
}
impl Drop for Files {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}
fn limits() -> CostProfileLoadLimits {
    CostProfileLoadLimits {
        max_file_bytes: NonZeroUsize::new(4 * 1024 * 1024).unwrap(),
        ..Default::default()
    }
}
fn exported(source: &Source) -> Files {
    let files = Files::new(&source.bytes);
    export_structured_profile_v9(
        &files.source,
        Sha256::digest(&source.bytes).into(),
        &files.profile,
        0,
        &limits(),
    )
    .unwrap();
    files
}
fn load_clock(s: &Source) -> ProfileLoadClock {
    ProfileLoadClock {
        wall_unix_ns: Some(s.closing.wall_unix_ns + 100),
        wall_max_error_ns: Some(0),
        monotonic_now_ns: 7,
    }
}
fn mutate(bytes: &[u8], mut f: impl FnMut(&mut serde_json::Value)) -> Vec<u8> {
    let mut out = Vec::new();
    for line in bytes.split(|b| *b == b'\n').filter(|l| !l.is_empty()) {
        let mut value: serde_json::Value = serde_json::from_slice(line).unwrap();
        f(&mut value["record"]);
        serde_json::to_writer(&mut out, &value).unwrap();
        out.push(b'\n');
    }
    out
}
#[test]
fn structured_v9_replays_all_original_phases_and_preserves_clock_epoch() {
    let s = source();
    let files = exported(&s);
    let model =
        load_structured_profile_v9(&files.profile, &fingerprint(), &limits(), load_clock(&s))
            .unwrap();
    assert_eq!(
        model
            .provenance()
            .phases
            .iter()
            .map(|p| p.members)
            .collect::<Vec<_>>(),
        [8, 8, 8]
    );
    assert_eq!(model.provenance().offered_attempts, 25);
    assert_eq!(model.provenance().reserved_members, 24);
    assert_eq!(model.provenance().phases[2].accepted_fifo_cutoff, 34);
    assert_eq!(model.model_now_ns(7).unwrap(), s.closing.monotonic_ns + 100);
    assert_eq!(
        model.model_now_ns(17).unwrap(),
        s.closing.monotonic_ns + 110
    );
    let predicted = model.predict_input(&fingerprint(), &s.query, 17).unwrap();
    assert!(predicted.planning_ns >= predicted.fitted_ns);
    assert!(predicted.valid_until_ns > model.model_now_ns(17).unwrap());
    assert!(matches!(
        model.model_now_ns(6),
        Err(StructuredUnknown::Clock)
    ));
    assert!(matches!(
        model.predict_input(&fingerprint(), &s.query, 2_000_000_000),
        Err(StructuredUnknown::Stale)
    ));
    let mut changed = fingerprint();
    changed.execution_config = [9; 32];
    assert!(matches!(
        model.predict_input(&changed, &s.query, 7),
        Err(StructuredUnknown::WrongFingerprint)
    ));
}
#[test]
fn structured_v9_export_rejects_failed_reserved_member_despite_success_footer() {
    let s = source();
    let bytes = mutate(&s.bytes, |r| {
        if r["kind"] == "completed" && r["member"] == 1 {
            r["conversion_error"] = "MissingEvidence".into();
        }
    });
    let f = Files::new(&bytes);
    assert!(export_structured_profile_v9(
        &f.source,
        Sha256::digest(&bytes).into(),
        &f.profile,
        0,
        &limits()
    )
    .is_err());
    assert!(!f.profile.exists());
}
#[test]
fn structured_v9_rejects_omitted_member_fifo_gap_and_policy_exchange() {
    let s = source();
    for field in ["member", "fifo", "recipe", "numeric", "protocol", "freeze"] {
        let bytes = mutate(&s.bytes, |r| {
            if r["kind"] == "reserved" && r["offered"] == 2 && field == "member" {
                r["member"] = serde_json::Value::Null;
            }
            if r["kind"] == "completed" && r["offered"] == 2 {
                match field {
                    "fifo" => r["queue"]["accepted_ordinal"] = 100.into(),
                    "recipe" => {
                        r["selected_structured_capture"]["Ok"]["device"]["ordered_template"][0] =
                            99.into()
                    }
                    "numeric" => r["numeric"]["basis"][0] = 2.into(),
                    _ => {}
                }
            }
            if r.get("artifact_type").is_some() && field == "protocol" {
                r["protocol"][0] = 99.into();
            }
            if r["kind"] == "phase_freeze" && r["receipt"]["phase"] == "fit" && field == "freeze" {
                r["receipt"]["parameters_sha256"][0] = 99.into();
            }
        });
        assert!(
            replay::replay_source(&bytes, &limits()).is_err(),
            "mutation {field}"
        );
    }
}
#[test]
fn structured_v9_rejects_modified_stage_receipt_cleanup_and_time() {
    let s = source();
    for field in ["wall", "cleanup", "clock", "binding"] {
        let bytes = mutate(&s.bytes, |r| {
            if r["kind"] == "completed" && r["member"] == 2 {
                match field {
                    "wall" => r["host_stages"]["full_wall_ns"] = 1.into(),
                    "cleanup" => {
                        r["host_stages"]["rows"][0]["terminal"]["cache_completion_work"] =
                            "unknown".into()
                    }
                    "clock" => r["host_stages"]["rows"][0]["settled_at_ns"] = 1.into(),
                    "binding" => {
                        r["host_stages"]["structured_evidence"]["Ok"]["stage_binding"][0] =
                            99.into()
                    }
                    _ => {}
                }
            }
        });
        assert!(
            replay::replay_source(&bytes, &limits()).is_err(),
            "mutation {field}"
        );
    }
}
#[test]
fn structured_v9_import_rejects_source_mutation_limits_clock_and_unknown_version() {
    let s = source();
    let f = exported(&s);
    let mut small = limits();
    small.max_file_bytes = NonZeroUsize::new(1024).unwrap();
    assert!(
        load_structured_profile_v9(&f.profile, &fingerprint(), &small, load_clock(&s)).is_err()
    );
    let mut clock = load_clock(&s);
    clock.wall_max_error_ns = None;
    assert!(load_structured_profile_v9(&f.profile, &fingerprint(), &limits(), clock).is_err());
    clock = load_clock(&s);
    clock.wall_unix_ns = Some(s.closing.wall_unix_ns + 2_000_000_000);
    assert!(load_structured_profile_v9(&f.profile, &fingerprint(), &limits(), clock).is_err());
    let mut bytes = s.bytes.clone();
    bytes.push(b'\n');
    std::fs::write(&f.source, bytes).unwrap();
    assert!(
        load_structured_profile_v9(&f.profile, &fingerprint(), &limits(), load_clock(&s)).is_err()
    );
    std::fs::write(&f.source, &s.bytes).unwrap();
    let mut envelope: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&f.profile).unwrap()).unwrap();
    envelope["schema_version"] = 8.into();
    std::fs::write(&f.profile, serde_json::to_vec(&envelope).unwrap()).unwrap();
    assert!(matches!(
        load_structured_profile_v9(&f.profile, &fingerprint(), &limits(), load_clock(&s)),
        Err(CostProfileError::UnsupportedVersion(8))
    ));
}
#[test]
fn structured_v9_export_does_not_replace_existing_artifact() {
    let s = source();
    let f = Files::new(&s.bytes);
    std::fs::write(&f.profile, b"original").unwrap();
    assert!(export_structured_profile_v9(
        &f.source,
        Sha256::digest(&s.bytes).into(),
        &f.profile,
        0,
        &limits()
    )
    .is_err());
    assert_eq!(std::fs::read(&f.profile).unwrap(), b"original");
}

#[test]
fn structured_v9_combined_clock_error_must_fit_declared_budget() {
    let s = source();
    let f = Files::new(&s.bytes);
    let mut bound = limits();
    bound.max_clock_error_ns = 10;
    export_structured_profile_v9(
        &f.source,
        Sha256::digest(&s.bytes).into(),
        &f.profile,
        6,
        &bound,
    )
    .unwrap();
    let mut clock = load_clock(&s);
    clock.wall_max_error_ns = Some(6);
    assert!(load_structured_profile_v9(&f.profile, &fingerprint(), &bound, clock).is_err());
    clock.wall_max_error_ns = Some(4);
    let imported = load_structured_profile_v9(&f.profile, &fingerprint(), &bound, clock).unwrap();
    assert_eq!(imported.provenance().conservative_clock_error_ns, 10);
}

#[test]
fn structured_v9_checks_source_header_bound_before_deserializing_metadata() {
    // This source fits the file budget; its first record alone exceeds the same
    // bound already applied to every observation record. It must fail on that
    // bound before attempting to parse or allocate the oversized JSON value.
    let mut bytes = vec![b' '; replay::MAX_SOURCE_RECORD_BYTES + 1];
    bytes.push(b'\n');
    let mut bound = limits();
    bound.max_file_bytes = NonZeroUsize::new(bytes.len()).unwrap();
    assert!(matches!(
        replay::replay_source(&bytes, &bound),
        Err(CostProfileError::Limit("structured source record bytes"))
    ));
}

#[test]
fn structured_v9_producer_metadata_is_typed_bounded_and_rejects_unknown_fields() {
    let s = source();
    let first = s.bytes.split(|b| *b == b'\n').next().unwrap();
    let raw: serde_json::Value = serde_json::from_slice(first).unwrap();
    let header: Header = serde_json::from_value(raw["record"].clone()).unwrap();
    let bound = limits();
    replay::header_valid(&header, s.bytes.len(), &bound).unwrap();
    let mut exact_bound = header.clone();
    exact_bound.producer["source_revision"] = "r".repeat(bound.max_source_field_bytes.get()).into();
    replay::header_valid(&exact_bound, s.bytes.len(), &bound).unwrap();
    for (field, value) in [
        (
            "source_revision",
            serde_json::json!("r".repeat(bound.max_source_field_bytes.get() + 1)),
        ),
        ("source_revision", serde_json::json!(7)),
        ("source_revision", serde_json::json!("")),
        ("source_revision", serde_json::json!("rev\n")),
        ("executable_bytes", serde_json::json!("1")),
        ("executable_bytes", serde_json::json!(0)),
        ("executable_sha256", serde_json::json!("z".repeat(64))),
        (
            "package_version",
            serde_json::json!({"unexpected": "metadata"}),
        ),
        ("extra", serde_json::json!({"nested": [1, 2, 3]})),
    ] {
        let mut changed = header.clone();
        changed.producer[field] = value;
        assert!(
            replay::header_valid(&changed, s.bytes.len(), &bound).is_err(),
            "producer field {field}"
        );
    }
}
