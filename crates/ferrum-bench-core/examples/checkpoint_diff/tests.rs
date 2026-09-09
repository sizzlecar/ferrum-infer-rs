use super::*;

fn fixture(directory: &Path, dtype: &str, values: &[f32]) -> Wave {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|&value| match dtype {
            "f16" => half::f16::from_f32(value).to_le_bytes().to_vec(),
            "bf16" => half::bf16::from_f32(value).to_le_bytes().to_vec(),
            "f32" => value.to_le_bytes().to_vec(),
            _ => panic!("unsupported fixture dtype"),
        })
        .collect();
    fs::write(directory.join("values.bin"), &bytes).unwrap();
    Wave {
        schema_version: 3,
        wave_kind: "prefill".into(),
        participant_count: 1,
        records: vec![Record {
            value: Some(Identity {
                value_id: "hidden".into(),
                tensor: Tensor {
                    dimensions: vec![values.len() as u64],
                },
            }),
            output_mode: None,
            participant_index: 0,
            token_span: json!({"immediate_tokens":1,"fingerprint":"same-token-span"}),
            output_layout: Layout {
                element_type: dtype.into(),
                element_count: values.len() as u64,
            },
            raw_file: "values.bin".into(),
            raw_bytes: bytes.len() as u64,
            raw_sha256: format!("{:x}", Sha256::digest(&bytes)),
        }],
        product_outputs: vec![],
    }
}

fn write_wave(directory: &Path, wave: &Wave) -> PathBuf {
    let path = directory.join("wave.json");
    fs::write(&path, serde_json::to_vec(wave).unwrap()).unwrap();
    path
}

#[test]
fn compares_actual_values_across_float_storage_types() {
    let left = tempfile::tempdir().unwrap();
    let right = tempfile::tempdir().unwrap();
    let a = fixture(left.path(), "f16", &[1.0, 2.0]);
    let b = fixture(right.path(), "f32", &[1.0, 3.0]);
    let report = compare(&write_wave(left.path(), &a), &write_wave(right.path(), &b)).unwrap();
    let row = &report["comparisons"][0];
    assert_eq!(row["nmse"], 0.2);
    assert_eq!(row["max_abs"], 1.0);
    assert_eq!(row["equal_f32_bits"], 1);
    assert_eq!(report["release_approved"], false);
    let b = fixture(right.path(), "bf16", &[1.0, 2.0]);
    let report = compare(&write_wave(left.path(), &a), &write_wave(right.path(), &b)).unwrap();
    assert_eq!(report["comparisons"][0]["nmse"], 0.0);
    assert_eq!(report["comparisons"][0]["equal_f32_bits"], 2);
}

#[test]
fn refuses_corrupted_extents_hashes_and_nonfinite_values() {
    let directory = tempfile::tempdir().unwrap();
    let mut wave = fixture(directory.path(), "f32", &[1.0, 2.0]);
    fs::write(directory.path().join("values.bin"), [0_u8; 8]).unwrap();
    assert!(array(directory.path(), &wave.records[0])
        .unwrap_err()
        .contains("SHA-256"));
    fs::write(directory.path().join("values.bin"), [0_u8; 4]).unwrap();
    assert!(array(directory.path(), &wave.records[0])
        .unwrap_err()
        .contains("length"));
    wave.records[0].output_layout.element_count = u64::MAX;
    assert!(array(directory.path(), &wave.records[0])
        .unwrap_err()
        .contains("extent"));
    for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let wave = fixture(directory.path(), "f32", &[value]);
        assert!(array(directory.path(), &wave.records[0])
            .unwrap_err()
            .contains("non-finite"));
    }
}

#[test]
fn refuses_duplicate_values_and_different_token_spans() {
    let left = tempfile::tempdir().unwrap();
    let right = tempfile::tempdir().unwrap();
    let a = fixture(left.path(), "f32", &[1.0]);
    let mut b = fixture(right.path(), "f32", &[1.0]);
    b.records[0].token_span["fingerprint"] = "other-token-span".into();
    let error = compare(&write_wave(left.path(), &a), &write_wave(right.path(), &b)).unwrap_err();
    assert!(error.contains("spans or shapes"));
    b.records
        .push(fixture(right.path(), "f32", &[1.0]).records.remove(0));
    assert!(index(&b).err().unwrap().contains("duplicate"));
}

#[test]
fn product_logits_are_ranked_by_value_with_stable_ties() {
    let directory = tempfile::tempdir().unwrap();
    let mut wave = fixture(directory.path(), "f32", &[-2.0, 4.0, 4.0]);
    let mut record = wave.records.remove(0);
    record.value = None;
    record.output_mode = Some("full-logits".into());
    wave.product_outputs.push(record);
    let path = write_wave(directory.path(), &wave);
    let report = compare(&path, &path).unwrap();
    let row = &report["comparisons"][0];
    assert_eq!(row["reference_top_logits"][0]["index"], 1);
    assert_eq!(row["candidate_top_logits"][1]["index"], 2);
    assert_eq!(row["reference_top_logits"][2]["value"], -2.0);
}

#[test]
fn failure_is_recorded_without_overwriting_existing_evidence() {
    let directory = tempfile::tempdir().unwrap();
    let output = directory.path().join("report.json");
    let args = || Args {
        reference: directory.path().join("missing.json"),
        candidate: directory.path().join("also-missing.json"),
        output: output.clone(),
    };
    assert!(run(args()).is_err());
    let bytes = fs::read(&output).unwrap();
    let report: Value = serde_json::from_slice(&bytes).unwrap();
    assert!(report["error"].as_str().unwrap().contains("reference wave"));
    assert!(run(args()).unwrap_err().contains("reserve"));
    assert_eq!(fs::read(&output).unwrap(), bytes);
}
