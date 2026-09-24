use super::*;
#[path = "tests/fixture.rs"]
mod fixture;
#[path = "tests/native_work.rs"]
mod native_work_tests;
use fixture::{Pair, BASE_LOGITS};
#[path = "tests/comparison_shape.rs"]
mod comparison_shape_tests;
#[path = "tests/numerical_transition.rs"]
mod numerical_transition_tests;
#[path = "tests/transition.rs"]
mod transition_tests;

fn assert_insufficient(report: &Value, code: i32, reason: &str) {
    assert_eq!(code, 2, "{report:#}");
    assert_eq!(report["status"], "evidence_insufficient");
    assert_eq!(report["quality_passed"], false);
    assert!(
        report["errors"]
            .as_array()
            .unwrap()
            .iter()
            .any(|error| error["message"].as_str().unwrap().contains(reason)),
        "missing {reason}: {report:#}"
    );
}

#[test]
fn physical_b4_is_per_node_owner_width_and_joins_all_histories() {
    let pair = Pair::new();
    let (report, code) = pair.compare();
    assert_eq!(code, 0, "{report:#}");
    assert_eq!(report["summary"]["all"]["decision_count"], 12);
    assert_eq!(report["summary"]["prefill"]["decision_count"], 4);
    assert_eq!(report["summary"]["decode"]["decision_count"], 8);
    assert_eq!(report["provenance"]["logical_decode_width"]["candidate"], 4);
    assert_eq!(report["provenance"]["source_location_difference"], true);
    assert_eq!(report["release_approved"], false);
    let wave = pair
        .candidate
        .manifest
        .waves
        .iter()
        .find(|wave| wave.kind == "decode")
        .unwrap();
    let file = wave.completion_receipt.as_ref().unwrap();
    let receipt: receipt::Completion =
        serde_json::from_slice(&fs::read(pair.candidate.directory.join(&file.file)).unwrap())
            .unwrap();
    assert_eq!(receipt.participants.len(), 8, "two nodes × four owners");
    assert_eq!(
        receipt.submission.batch_identity.nodes[0]
            .participants
            .len(),
        4
    );
}

#[test]
fn owner_and_decision_record_order_does_not_define_correspondence() {
    let mut pair = Pair::with_width(2);
    pair.candidate.manifest.owners.reverse();
    pair.candidate.manifest.decisions.reverse();
    for wave in &mut pair.candidate.manifest.waves {
        wave.participants.reverse();
    }
    pair.candidate.persist();
    let (report, code) = pair.compare();
    assert_eq!(code, 0, "{report:#}");
}

#[test]
fn full_vocabulary_tail_regression_fails_even_with_identical_argmax() {
    let mut pair = Pair::new();
    pair.candidate
        .set_logits("owner-0", 1, &[0.0, 2.0, 1.0, -8.0]);
    let (report, code) = pair.compare();
    assert_eq!(code, 1, "{report:#}");
    assert_eq!(report["status"], "quality_failed");
    assert_eq!(report["evidence_complete"], true);
    assert_eq!(report["summary"]["all"]["argmax_agreement_count"], 12);
    assert_eq!(report["summary"]["prefill"]["quality_passed"], true);
    assert_eq!(report["summary"]["decode"]["quality_passed"], false);
}

#[test]
fn separate_decode_budget_cannot_be_hidden_by_prefill_average() {
    let mut pair = Pair::new();
    pair.candidate
        .set_logits("owner-0", 2, &[0.0, 2.0, 1.0, -1.04]);
    let mut args = pair.args();
    args.budgets = Budgets {
        mean_delta_nll_limit: 1.0,
        max_delta_nll_limit: 1.0,
        mean_kl_limit: 1.0,
        max_kl_limit: 1.0,
    };
    let (initial, _) = compare(&args);
    let all = initial["summary"]["all"]["checks"][0]["observed"]
        .as_f64()
        .unwrap();
    let decode = initial["summary"]["decode"]["checks"][0]["observed"]
        .as_f64()
        .unwrap();
    assert!(decode > all && all > 0.0);
    args.budgets.mean_delta_nll_limit = (all + decode) / 2.0;
    let (report, code) = compare(&args);
    assert_eq!(code, 1, "{report:#}");
    assert_eq!(report["summary"]["all"]["quality_passed"], true);
    assert_eq!(report["summary"]["decode"]["quality_passed"], false);
}

#[test]
fn recomputed_logit_file_hash_does_not_hide_post_readback_modification() {
    let mut pair = Pair::new();
    let logits = &mut pair.candidate.manifest.decisions[0].logits;
    let bytes: Vec<_> = [0.0_f32, 2.0, 1.0, -9.0]
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    fs::write(pair.candidate.directory.join(&logits.file), &bytes).unwrap();
    logits.sha256 = sha256(&bytes);
    pair.candidate.persist();
    let (report, code) = pair.compare();
    assert_insufficient(&report, code, "independent raw dtype conversion");
}

#[test]
fn raw_dtype_conversion_supports_f32_f16_and_bf16_exactly() {
    let half = [0x0000_u16, 0x8000, 0x0001, 0x8001, 0x3c00, 0x7bff];
    let bytes: Vec<_> = half.iter().flat_map(|bits| bits.to_le_bytes()).collect();
    let decoded = files::decode(&bytes, "f16", half.len()).unwrap();
    let expected = [
        0.0_f32,
        -0.0,
        2.0_f32.powi(-24),
        -2.0_f32.powi(-24),
        1.0,
        65504.0,
    ];
    assert_eq!(
        decoded.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    for dtype in ["f16", "bf16"] {
        let mut pair = Pair::with_width(2);
        // One immutable plan fixes the raw dtype for every owner/wave and both
        // arms. The f32 files remain untouched, testing real widening equality.
        pair.reference.set_raw_dtype(dtype);
        pair.candidate.set_raw_dtype(dtype);
        let (report, code) = pair.compare();
        assert_eq!(code, 0, "{dtype}: {report:#}");
    }
    for (dtype, bytes) in [
        ("f16", 0x7c00_u16.to_le_bytes().to_vec()),
        ("bf16", 0x7fc0_u16.to_le_bytes().to_vec()),
        ("f32", f32::NEG_INFINITY.to_le_bytes().to_vec()),
    ] {
        assert!(files::decode(&bytes, dtype, 1).is_err());
    }
    assert!(files::decode(&[0; 3], "f32", 1).is_err());
    assert!(files::decode(&[0; 4], "i32", 1).is_err());
}

#[test]
fn a_sequence_of_serial_waves_cannot_be_relabelled_batched() {
    let mut pair = Pair::new();
    pair.candidate.directory = pair.reference.directory.clone();
    pair.candidate.manifest = pair.reference.manifest.clone();
    pair.candidate.manifest.mode = "batched".into();
    // Keep reference artifacts independent while reusing the typed serial
    // fixture as an invalid candidate claim.
    let candidate_dir = pair.root.path().join("pretend-batched");
    fs::create_dir(&candidate_dir).unwrap();
    for entry in fs::read_dir(&pair.reference.directory).unwrap() {
        let entry = entry.unwrap();
        fs::copy(entry.path(), candidate_dir.join(entry.file_name())).unwrap();
    }
    pair.candidate.directory = candidate_dir;
    pair.candidate.persist();
    let (report, code) = pair.compare();
    assert_insufficient(&report, code, "actual wave width differs");
}

#[test]
fn node_owner_projection_cannot_be_swapped_even_with_updated_file_and_node_hash() {
    let mut pair = Pair::new();
    let wave = pair
        .candidate
        .manifest
        .waves
        .iter()
        .position(|wave| wave.kind == "decode")
        .unwrap();
    pair.candidate.edit_completion(wave, |completion| {
        let node = &mut completion.submission.batch_identity.nodes[0];
        node.participants[0].identity.request_id = node.participants[1].identity.request_id.clone();
        node.fingerprint = receipt::node_fingerprint(node).unwrap();
    });
    let (report, code) = pair.compare();
    assert_insufficient(&report, code, "another request owner");
}

#[test]
fn complete_flattened_count_does_not_replace_per_node_owner_coverage() {
    let mut pair = Pair::new();
    let wave = pair
        .candidate
        .manifest
        .waves
        .iter()
        .position(|wave| wave.kind == "decode")
        .unwrap();
    pair.candidate.edit_completion(wave, |completion| {
        completion.submission.batch_identity.nodes[0]
            .participants
            .pop();
    });
    let (report, code) = pair.compare();
    assert_insufficient(&report, code, "physical node width differs");
}

#[test]
fn missing_decisions_and_incomplete_capture_are_evidence_failures() {
    let mut pair = Pair::with_width(2);
    pair.candidate.manifest.decisions.pop();
    pair.candidate.manifest.complete = false;
    pair.candidate.persist();
    let (report, code) = pair.compare();
    assert_insufficient(&report, code, "canonical decision missing");
    assert_insufficient(&report, code, "capture incomplete");
}

#[test]
fn exact_binary_and_history_are_reopened_and_overrides_are_verified() {
    let mut pair = Pair::with_width(2);
    let binary = PathBuf::from(
        &pair
            .reference
            .manifest
            .identity
            .as_ref()
            .unwrap()
            .binary
            .path,
    );
    pair.candidate
        .manifest
        .identity
        .as_mut()
        .unwrap()
        .binary
        .path = "/unavailable/remote/binary".into();
    pair.candidate.persist();
    let (report, code) = pair.compare();
    assert_insufficient(&report, code, "missing external provenance file");
    let mut args = pair.args();
    args.candidate_binary = Some(binary);
    let (report, code) = compare(&args);
    assert_eq!(code, 0, "{report:#}");
    let history = pair.root.path().join("wrong-history.json");
    fs::write(&history, b"{\"schema_version\":1,\"owners\":[]}").unwrap();
    args.history_file = Some(history);
    let (report, code) = compare(&args);
    assert_insufficient(&report, code, "external provenance file");
}

#[test]
fn different_candidate_binary_is_allowed_only_with_its_own_valid_identity() {
    let mut pair = Pair::with_width(2);
    let path = pair.root.path().join("candidate-binary");
    let bytes = b"distinct candidate binary, same numerical and plan contract";
    fs::write(&path, bytes).unwrap();
    pair.candidate.manifest.identity.as_mut().unwrap().binary = VNextTeacherFileIdentity {
        path: path.to_string_lossy().into_owned(),
        bytes: bytes.len() as u64,
        sha256: sha256(bytes),
    };
    pair.candidate.persist();
    let (report, code) = pair.compare();
    assert_eq!(code, 0, "{report:#}");
    assert_eq!(report["provenance"]["same_binary"], false);
    fs::write(&path, vec![0; bytes.len()]).unwrap();
    let (report, code) = pair.compare();
    assert_insufficient(&report, code, "external provenance file SHA-256 differs");
}

#[test]
fn configuration_profile_and_history_changes_are_not_quality_passes() {
    let mut pair = Pair::with_width(2);
    pair.candidate.manifest.configuration["max_model_len"] = json!(32);
    pair.candidate.persist();
    let (report, code) = pair.compare();
    assert_insufficient(&report, code, "configuration differs");
    pair.candidate.manifest.configuration["max_model_len"] = json!(16);
    pair.candidate
        .manifest
        .identity
        .as_mut()
        .unwrap()
        .numerical_profile = "another.profile".into();
    pair.candidate.persist();
    let (report, code) = pair.compare();
    assert_insufficient(&report, code, "numerical/KV/resolved plan identity differs");
    pair.candidate
        .manifest
        .identity
        .as_mut()
        .unwrap()
        .numerical_profile = "fixture.exact".into();
    pair.candidate.manifest.decisions[0].evidence.history_sha256 = sha256(b"another-prefix");
    pair.candidate.persist();
    let (report, code) = pair.compare();
    assert_insufficient(&report, code, "decision target/history binding differs");
}

#[test]
fn artifact_paths_and_full_vocab_sizes_are_verified() {
    let mut pair = Pair::with_width(2);
    pair.candidate.manifest.decisions[0].logits.file = "../outside.f32le".into();
    pair.candidate.persist();
    let (report, code) = pair.compare();
    assert_insufficient(&report, code, "portable relative path");
    pair.candidate.manifest.decisions[0].logits.elements = BASE_LOGITS.len() - 1;
    pair.candidate.persist();
    let (report, code) = pair.compare();
    assert_insufficient(&report, code, "drops vocabulary");
}

#[test]
fn all_four_quality_budgets_are_required_and_finite() {
    let args = [
        "compare",
        "--reference-dir",
        "reference",
        "--candidate-dir",
        "candidate",
        "--output-file",
        "report.json",
    ];
    assert!(Args::try_parse_from(args).is_err());
    assert!(Budgets {
        mean_delta_nll_limit: f64::NAN,
        max_delta_nll_limit: 0.1,
        mean_kl_limit: 0.001,
        max_kl_limit: 0.01
    }
    .validate()
    .is_err());
}

#[test]
fn source_location_normalization_preserves_every_execution_option() {
    let mut pair = Pair::with_width(2);
    let (report, code) = pair.compare();
    assert_eq!(code, 0, "{report:#}");
    let locations = &report["provenance"]["configuration_source_locations"];
    assert_ne!(
        locations["reference_model_path"],
        locations["candidate_model_path"]
    );
    pair.candidate.manifest.configuration["backend"]["backend_options"]["numerical_route"] =
        json!("changed-route");
    pair.candidate.persist();
    let (report, code) = pair.compare();
    assert_insufficient(&report, code, "configuration differs");
    pair.candidate.manifest.configuration["backend"]["backend_options"]["numerical_route"] =
        json!("fixture.exact");
    pair.candidate.manifest.configuration["backend"]["backend_options"]["model_path"] =
        json!("/unaccounted/model");
    pair.candidate.persist();
    let (report, code) = pair.compare();
    assert_insufficient(
        &report,
        code,
        "not covered by the captured source inventory",
    );
}

#[test]
fn readback_cannot_move_to_another_executed_node_resource_or_offset() {
    for (field, value) in [
        ("node_id", json!("node.0")),
        ("resource_id", json!("resource.other")),
        ("logical_offset_bytes", json!(4)),
    ] {
        let mut pair = Pair::with_width(2);
        // Change all readbacks in this candidate so within-wave and
        // within-capture checks alone cannot hide the cross-arm mismatch.
        for wave in &mut pair.candidate.manifest.waves {
            for raw in &mut wave.readbacks {
                raw.request[field] = value.clone();
            }
            wave.receipt_fingerprint =
                receipt::readback_fingerprint(&wave.completion_fingerprint, &wave.readbacks)
                    .unwrap();
        }
        pair.candidate.persist();
        let (report, code) = pair.compare();
        assert_insufficient(&report, code, "product output binding differs between arms");
    }
}

#[test]
fn matching_float_values_do_not_allow_different_plan_output_dtypes() {
    let mut pair = Pair::with_width(2);
    pair.candidate.set_raw_dtype("f16");
    let (report, code) = pair.compare();
    assert_insufficient(&report, code, "product output binding differs between arms");
}

#[test]
fn owner_and_wave_output_binding_changes_are_rejected() {
    let mut pair = Pair::new();
    let wave = pair
        .candidate
        .manifest
        .waves
        .iter_mut()
        .find(|wave| wave.kind == "decode")
        .unwrap();
    wave.readbacks[0].request["resource_id"] = json!("resource.other");
    wave.receipt_fingerprint =
        receipt::readback_fingerprint(&wave.completion_fingerprint, &wave.readbacks).unwrap();
    pair.candidate.persist();
    let (report, code) = pair.compare();
    assert_insufficient(
        &report,
        code,
        "physical owners have different product output bindings",
    );
    let mut pair = Pair::new();
    let wave = &mut pair.candidate.manifest.waves[1];
    wave.readbacks[0].request["resource_id"] = json!("resource.other");
    wave.receipt_fingerprint =
        receipt::readback_fingerprint(&wave.completion_fingerprint, &wave.readbacks).unwrap();
    pair.candidate.persist();
    let (report, code) = pair.compare();
    assert_insufficient(
        &report,
        code,
        "product output binding changed between physical waves",
    );
}

#[test]
fn one_physical_completion_cannot_prove_two_different_decode_steps() {
    let mut pair = Pair::new();
    let waves = &mut pair.candidate.manifest.waves;
    let first = waves.iter().position(|wave| wave.kind == "decode").unwrap();
    let prior = waves[first].clone();
    let next = &mut waves[first + 1];
    next.completion_receipt = prior.completion_receipt;
    next.completion_fingerprint = prior.completion_fingerprint;
    next.receipt_fingerprint =
        receipt::readback_fingerprint(&next.completion_fingerprint, &next.readbacks).unwrap();
    pair.candidate.persist();
    let (report, code) = pair.compare();
    assert_insufficient(&report, code, "actual physical completion was reused");
}

#[test]
fn unique_completions_cannot_be_swapped_between_canonical_decode_steps() {
    let mut pair = Pair::new();
    let waves = &mut pair.candidate.manifest.waves;
    let first = waves.iter().position(|wave| wave.kind == "decode").unwrap();
    let earlier = waves[first].clone();
    let later = waves[first + 1].clone();
    waves[first].completion_receipt = later.completion_receipt;
    waves[first].completion_fingerprint = later.completion_fingerprint;
    waves[first + 1].completion_receipt = earlier.completion_receipt;
    waves[first + 1].completion_fingerprint = earlier.completion_fingerprint;
    for wave in &mut waves[first..=first + 1] {
        wave.receipt_fingerprint =
            receipt::readback_fingerprint(&wave.completion_fingerprint, &wave.readbacks).unwrap();
    }
    pair.candidate.persist();
    let (report, code) = pair.compare();
    assert_insufficient(
        &report,
        code,
        "actual physical teacher wave order was replayed or reversed",
    );
}
