use super::*;
use serde_json::{json, Value};

struct SourceFixture {
    dir: PathBuf,
    artifact: CalibrationProfileArtifact,
    fingerprint: profile::ProfileFingerprint,
}
impl SourceFixture {
    fn new(schema: u32, body: Vec<Value>) -> Self {
        let dir =
            std::env::temp_dir().join(format!("ferrum-reference-source-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir(&dir).unwrap();
        let fingerprint = profile::ProfileFingerprint {
            model_weights: [1; 32],
            numerical_policy: [2; 32],
            device_runtime: [3; 32],
            execution_config: [4; 32],
        };
        let mut rows = vec![json!({
            "kind":"header", "artifact_type":"ferrum.cost-training-cut", "schema_version":schema,
            "accepted_ordinal":3, "fingerprint":fingerprint, "settings":{}, "producer":{},
            "opening":{"wall_unix_ns":1000,"monotonic_ns":100},
            "declared_clock_max_error_ns":0,"coverage":"fixture",
        })];
        let observations = body
            .iter()
            .filter(|row| row["kind"] == "observation")
            .count();
        rows.extend(body);
        rows.push(json!({"kind":"summary","accepted_ordinal":3,
            "closing":{"wall_unix_ns":1300,"monotonic_ns":400},
            "counts":{},"sink":{},"training":{},"coverage":"fixture"}));
        let mut bytes = Vec::new();
        for row in rows {
            serde_json::to_writer(&mut bytes, &row).unwrap();
            bytes.push(b'\n');
        }
        let digest: [u8; 32] = Sha256::digest(&bytes).into();
        let source = dir.join("source.jsonl");
        std::fs::write(&source, &bytes).unwrap();
        let artifact = CalibrationProfileArtifact {
            accepted_ordinal: 3,
            profile: dir.join("unused-profile.json"),
            profile_sha256: String::new(),
            profile_bytes: 0,
            source,
            source_sha256: hex(&digest),
            source_digest: digest,
            source_bytes: bytes.len() as u64,
            retained_samples: observations as u64,
            raw_retained_observations: observations as u64,
        };
        Self {
            dir,
            artifact,
            fingerprint,
        }
    }
    fn read(&self) -> Result<Joined> {
        read(&self.artifact, &self.fingerprint, &BTreeMap::new())
    }
}
impl Drop for SourceFixture {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn observation() -> Value {
    let shape = profile::v2::ProfileWaveShapeV2 {
        exact: profile::ProfileWaveShape {
            kind: profile::ProfileWaveKind::Decode,
            path: profile::ProfileExecutionPath::PlanRuntime,
            provider_signature: [5; 32],
            output_policy_signature: [6; 32],
            graph_state: profile::ProfileGraphState::Disabled,
            order: profile::ProfileBatchOrder::Ordered,
            decode_kv_tokens: vec![128],
            prefill_chunks: vec![],
            recurrent_state_bytes: 0,
            restore_bytes: 0,
            maintenance_bytes: 0,
            maintenance_units: 0,
        },
        numeric_features: None,
    };
    let sample = ProfileSampleV2 {
        source_record: 0,
        measured_unix_ns: 1020,
        shape,
        boundary: profile::ProfileCostBoundary::PreparationToCommit,
        outcome: profile::ProfileObservationOutcome::Completed {},
        timing: profile::ProfileWaveTiming {
            wall_total_ns: 20,
            device_elapsed_ns: None,
            stages: Default::default(),
        },
    };
    json!({"kind":"observation","accepted_ordinal":1,"observed_at_monotonic_ns":120,
        "fingerprint":profile::ProfileFingerprint{model_weights:[1;32],numerical_policy:[2;32],device_runtime:[3;32],execution_config:[4;32]},
        "training":{"status":"recorded"},"pre_update_prediction":{},"sample":sample})
}
fn auxiliary(ordinal: u64, source: Option<u64>) -> Value {
    json!({"kind":"host_stages_v1","accepted_ordinal":ordinal,"source_record":source,
        "legacy_rejection":"composite","evidence":{"schema_version":1,"completeness":"missing_evidence"}})
}

#[test]
fn legacy_cut_and_new_auxiliary_suffix_preserve_original_sample_population() {
    let old = SourceFixture::new(1, vec![observation()]);
    assert!(old.read().is_ok());
    let new = SourceFixture::new(
        2,
        vec![observation(), auxiliary(1, Some(0)), auxiliary(3, None)],
    );
    let joined = new.read().unwrap();
    assert!(
        joined.samples.is_empty(),
        "auxiliary evidence cannot satisfy any requested measurement"
    );
    assert_eq!(joined.generated_unix_ns.get(), 1300);
}

#[test]
fn auxiliary_records_cannot_cross_join_duplicate_or_enter_legacy_schema() {
    for (schema, body) in [
        (1, vec![observation(), auxiliary(1, Some(0))]),
        (2, vec![observation(), auxiliary(1, Some(1))]),
        (2, vec![observation(), auxiliary(2, Some(0))]),
        (2, vec![observation(), auxiliary(1, None)]),
        (
            2,
            vec![observation(), auxiliary(1, Some(0)), auxiliary(1, Some(0))],
        ),
        (2, vec![auxiliary(1, Some(0))]),
    ] {
        assert!(SourceFixture::new(schema, body).read().is_err());
    }
}

#[test]
fn host_content_training_is_a_separate_v3_suffix_and_never_a_reference_measurement() {
    let training = |ordinal, source| {
        json!({"kind":"host_content_training_v1", "accepted_ordinal":ordinal,
        "host_source_record":source, "evaluation":{"training":{"status":"recorded"}}, "sample":null})
    };
    let v3 = SourceFixture::new(
        3,
        vec![
            observation(),
            auxiliary(1, Some(0)),
            training(1, 0),
            auxiliary(3, None),
            training(3, 1),
        ],
    );
    assert!(v3.read().unwrap().samples.is_empty());
    for (version, body) in [
        (
            2,
            vec![observation(), auxiliary(1, Some(0)), training(1, 0)],
        ),
        (3, vec![observation(), training(1, 0)]),
        (
            3,
            vec![observation(), auxiliary(1, Some(0)), training(2, 0)],
        ),
        (
            3,
            vec![
                observation(),
                auxiliary(1, Some(0)),
                training(1, 0),
                training(1, 0),
            ],
        ),
        (
            3,
            vec![
                observation(),
                auxiliary(1, Some(0)),
                training(1, 0),
                auxiliary(3, None),
                training(3, 0),
            ],
        ),
    ] {
        assert!(SourceFixture::new(version, body).read().is_err());
    }
}

#[test]
fn reference_source_checks_its_actual_receipt_length_before_parsing_or_retention() {
    let mut source = SourceFixture::new(2, vec![observation(), auxiliary(1, Some(0))]);
    assert!(source.read().is_ok());
    source.artifact.source_bytes -= 1;
    assert!(source
        .read()
        .err()
        .unwrap()
        .to_string()
        .contains("length differs from its receipt"));
    source.artifact.source_bytes += 2;
    assert!(source
        .read()
        .err()
        .unwrap()
        .to_string()
        .contains("length differs from its receipt"));
    // A forged large receipt is still rejected before allocating/reading the
    // file. The reference reader shares the import/export hard file ceiling.
    source.artifact.source_bytes = MAX_CUT_BYTES + 1;
    assert!(source
        .read()
        .err()
        .unwrap()
        .to_string()
        .contains("invalid reference cut source receipt"));
}

#[test]
fn row_multiset_cut_v4_auxiliary_cannot_upgrade_legacy_witnesses_or_cross_join() {
    let training = |ordinal, source| {
        json!({"kind":"host_row_multiset_training_v2",
        "accepted_ordinal":ordinal,"host_source_record":source,
        "evaluation":{"training":{"status":"recorded"}},"sample":null})
    };
    let body = vec![
        observation(),
        auxiliary(1, Some(0)),
        training(1, 0),
        auxiliary(3, None),
        training(3, 1),
    ];
    assert!(SourceFixture::new(4, body.clone())
        .read()
        .unwrap()
        .samples
        .is_empty());
    assert!(SourceFixture::new(3, body).read().is_err());
    for body in [
        vec![observation(), auxiliary(1, Some(0)), training(2, 0)],
        vec![observation(), training(1, 0)],
        vec![auxiliary(1, None), training(1, 0), training(1, 0)],
        vec![
            auxiliary(1, None),
            training(1, 0),
            auxiliary(3, None),
            training(3, 0),
        ],
        vec![
            auxiliary(1, None),
            json!({"kind":"host_content_training_v1","accepted_ordinal":1,
            "host_source_record":0,"evaluation":{},"sample":null}),
        ],
    ] {
        assert!(SourceFixture::new(4, body).read().is_err());
    }
}
