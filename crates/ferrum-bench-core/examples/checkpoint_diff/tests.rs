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
        teacher_forced_decision: None,
        records: vec![Record {
            value: Some(Identity {
                value_id: "hidden".into(),
                tensor: Tensor {
                    dimensions: vec![values.len() as u64],
                },
            }),
            output_mode: None,
            participant_index: 0,
            token_span: json!({"immediate_tokens":1,"full_input_tokens":1,"fit_input_tokens":1,
                "immediate_start_token":0,"immediate_end_token":1,
                "fingerprint":format!("{:x}",Sha256::digest(b"same-token-history"))}),
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
    b.records[0].token_span["fingerprint"] =
        format!("{:x}", Sha256::digest(b"other-token-history")).into();
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
    assert_eq!(row["distribution"]["kl_reference_to_candidate_nats"], 0.0);
    assert!(row["distribution"]["teacher_forced"].is_null());
}

fn logits_fixture(directory: &Path, values: &[f32], target: Option<u32>) -> Wave {
    let mut wave = fixture(directory, "f32", values);
    let mut record = wave.records.remove(0);
    record.value = None;
    record.output_mode = Some("full-logits".into());
    wave.product_outputs.push(record);
    if let Some(token_id) = target {
        wave.schema_version = 4;
        wave.teacher_forced_decision = Some(TeacherForcedDecision {
            token_index: 0,
            token_id,
        });
    }
    wave
}

fn assert_close(actual: &Value, expected: f64, tolerance: f64) {
    let actual = actual.as_f64().unwrap();
    assert!(
        (actual - expected).abs() <= tolerance,
        "{actual} differs from expected {expected}"
    );
}

#[test]
fn teacher_forced_logits_measure_directional_kl_and_target_nll_in_nats() {
    let left = tempfile::tempdir().unwrap();
    let right = tempfile::tempdir().unwrap();
    let a = logits_fixture(left.path(), &[3.0_f32.ln(), 0.0], Some(1));
    let b = logits_fixture(right.path(), &[0.0, 0.0], Some(1));
    let left_path = write_wave(left.path(), &a);
    let right_path = write_wave(right.path(), &b);
    let report = compare(&left_path, &right_path).unwrap();
    let metrics = &report["comparisons"][0]["distribution"];
    let expected = 0.75 * 1.5_f64.ln() + 0.25 * 0.5_f64.ln();
    assert_close(&metrics["kl_reference_to_candidate_nats"], expected, 1e-8);
    assert_close(
        &metrics["teacher_forced"]["reference_nll_nats"],
        4.0_f64.ln(),
        3e-8,
    );
    assert_close(
        &metrics["teacher_forced"]["candidate_nll_nats"],
        2.0_f64.ln(),
        1e-15,
    );
    assert_close(
        &metrics["teacher_forced"]["delta_nll_nats"],
        -2.0_f64.ln(),
        3e-8,
    );
    assert_eq!(metrics["teacher_forced"]["decision"]["token_id"], 1);
    assert_eq!(metrics["vocabulary_size"], 2);
    assert_eq!(report["input_and_weight_identity_verified"], false);
    let reverse = compare(&right_path, &left_path).unwrap();
    assert_close(
        &reverse["comparisons"][0]["distribution"]["kl_reference_to_candidate_nats"],
        0.5 * (4.0_f64 / 3.0).ln(),
        1e-8,
    );
}

#[test]
fn schema_three_can_compare_with_teacher_schema_without_inventing_a_target() {
    let left = tempfile::tempdir().unwrap();
    let right = tempfile::tempdir().unwrap();
    let a = logits_fixture(left.path(), &[1.0, 2.0], None);
    let b = logits_fixture(right.path(), &[1.0, 2.0], Some(0));
    let old_json = serde_json::to_value(&a).unwrap();
    assert!(old_json.get("teacher_forced_decision").is_none());
    let parsed: Wave = serde_json::from_value(old_json).unwrap();
    assert_eq!(parsed.teacher_forced_decision, None);
    let report = compare(&write_wave(left.path(), &a), &write_wave(right.path(), &b)).unwrap();
    assert!(report["comparisons"][0]["distribution"]["teacher_forced"].is_null());
}

#[test]
fn teacher_metrics_require_matching_targets_history_and_valid_vocabulary() {
    let left = tempfile::tempdir().unwrap();
    let right = tempfile::tempdir().unwrap();
    let a = logits_fixture(left.path(), &[1.0, 2.0], Some(1));
    let mut b = logits_fixture(right.path(), &[1.0, 2.0], Some(0));
    let left_path = write_wave(left.path(), &a);
    assert!(compare(&left_path, &write_wave(right.path(), &b))
        .unwrap_err()
        .contains("decisions differ"));
    b.teacher_forced_decision = a.teacher_forced_decision;
    b.teacher_forced_decision.as_mut().unwrap().token_index = 1;
    assert!(compare(&left_path, &write_wave(right.path(), &b))
        .unwrap_err()
        .contains("decisions differ"));
    b.teacher_forced_decision = a.teacher_forced_decision;
    b.product_outputs[0].token_span["full_input_tokens"] = 2.into();
    assert!(compare(&left_path, &write_wave(right.path(), &b))
        .unwrap_err()
        .contains("spans or shapes"));
    let mut b = logits_fixture(right.path(), &[1.0, 2.0], Some(2));
    let path = write_wave(right.path(), &b);
    assert!(compare(&path, &path).unwrap_err().contains("vocabulary"));
    b.teacher_forced_decision = None;
    assert!(index(&b).err().unwrap().contains("schema"));
    let mut b = logits_fixture(right.path(), &[1.0, 2.0], Some(1));
    b.participant_count = 2;
    assert!(index(&b).err().unwrap().contains("one participant"));
    b.participant_count = 1;
    b.product_outputs[0].token_span = json!({});
    assert!(index(&b).err().unwrap().contains("history fingerprint"));
}

#[test]
fn full_vocabulary_metrics_stay_finite_for_extreme_logits_and_underflowed_targets() {
    let decision = Some(TeacherForcedDecision {
        token_index: 0,
        token_id: 1,
    });
    let result =
        distribution_metrics(&[f32::MAX, -f32::MAX], &[-f32::MAX, f32::MAX], decision).unwrap();
    let gap = 2.0 * f64::from(f32::MAX);
    assert_close(&result["kl_reference_to_candidate_nats"], gap, 0.0);
    assert_close(&result["teacher_forced"]["reference_nll_nats"], gap, 0.0);
    assert_close(&result["teacher_forced"]["candidate_nll_nats"], 0.0, 0.0);
    let same =
        distribution_metrics(&[f32::MAX, f32::MAX], &[f32::MAX, f32::MAX], decision).unwrap();
    assert_close(
        &same["teacher_forced"]["reference_nll_nats"],
        2.0_f64.ln(),
        1e-15,
    );
    assert_close(&same["kl_reference_to_candidate_nats"], 0.0, 0.0);
    let base = distribution_metrics(&[1.0, -3.0, 2.0], &[2.0, -2.0, 1.0], decision).unwrap();
    let shifted = distribution_metrics(
        &[10001.0, 9997.0, 10002.0],
        &[-9998.0, -10002.0, -9999.0],
        decision,
    )
    .unwrap();
    assert_eq!(base, shifted);
    // The changed index is outside the top ten; the measurement uses every logit.
    let mut candidate = [-2.0; 12];
    candidate[11] = -3.0;
    let tail = distribution_metrics(&[-2.0; 12], &candidate, None).unwrap();
    assert_close(
        &tail["kl_reference_to_candidate_nats"],
        1.0 / 12.0 + ((11.0 + (-1.0_f64).exp()) / 12.0).ln(),
        1e-14,
    );
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
