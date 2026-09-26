use super::*;

fn batched_args(pair: &Pair) -> Args {
    let mut args = pair.args();
    args.comparison_shape = ComparisonShape::BatchedToBatched;
    args.require_bitwise_logits = true;
    args
}

#[test]
fn two_real_eight_owner_captures_cover_every_decode_and_original_prefill() {
    let pair = Pair::with_batched_width(8);
    let (report, code) = compare(&batched_args(&pair));
    assert_eq!(code, 0, "{report:#}");
    assert_eq!(report["comparison_shape"], "batched_to_batched");
    assert_eq!(report["comparison_passed"], true);
    assert_eq!(report["quality_passed"], true);
    assert_eq!(report["summary"]["prefill"]["decision_count"], 8);
    assert_eq!(report["summary"]["decode"]["decision_count"], 16);
    assert_eq!(
        report["provenance"]["logical_decode_width"],
        json!({"reference":8,"candidate":8})
    );
    assert_eq!(report["bitwise_logits"]["checked_logit_count"], 96);
    assert_eq!(report["bitwise_logits"]["mismatch_count"], 0);
    assert_eq!(report["bitwise_logits"]["passed"], true);
    for arm in [&pair.reference, &pair.candidate] {
        assert!(arm
            .manifest
            .waves
            .iter()
            .filter(|wave| wave.kind == "prefill")
            .all(|wave| wave.participant_count == 1));
        assert!(arm
            .manifest
            .waves
            .iter()
            .filter(|wave| wave.kind == "decode")
            .all(|wave| wave.participant_count == 8));
    }
    // The new shape is explicit, not an automatic fallback from the default.
    let (report, code) = pair.compare();
    assert_insufficient(&report, code, "real serial reference");
}

#[test]
fn comparison_shape_cli_is_typed_and_default_is_unchanged() {
    let base = [
        "compare",
        "--reference-dir",
        "r",
        "--candidate-dir",
        "c",
        "--output-file",
        "out",
        "--mean-delta-nll-limit",
        "0.01",
        "--max-delta-nll-limit",
        "0.1",
        "--mean-kl-limit",
        "0.001",
        "--max-kl-limit",
        "0.01",
    ];
    let args = Args::try_parse_from(base).unwrap();
    assert_eq!(args.comparison_shape, ComparisonShape::SerialToBatched);
    assert!(!args.require_bitwise_logits);
    let args = Args::try_parse_from(base.into_iter().chain([
        "--comparison-shape",
        "batched-to-batched",
        "--require-bitwise-logits",
    ]))
    .unwrap();
    assert_eq!(args.comparison_shape, ComparisonShape::BatchedToBatched);
    assert!(args.require_bitwise_logits);
    let args = Args::try_parse_from(
        base.into_iter()
            .chain(["--comparison-shape", "serial-to-serial"]),
    )
    .unwrap();
    assert_eq!(args.comparison_shape, ComparisonShape::SerialToSerial);
    assert!(Args::try_parse_from(base.into_iter().chain(["--comparison-shape", "auto"])).is_err());
}

#[test]
fn serial_to_serial_covers_every_owner_with_single_participant_decode_receipts() {
    for width in [1, 3] {
        let pair = Pair::with_serial_width(width);
        let mut args = pair.args();
        args.comparison_shape = ComparisonShape::SerialToSerial;
        args.require_bitwise_logits = true;
        let (report, code) = compare(&args);
        assert_eq!(code, 0, "{report:#}");
        assert_eq!(report["comparison_shape"], "serial_to_serial");
        assert_eq!(
            report["provenance"]["logical_decode_width"],
            json!({"reference":1,"candidate":1})
        );
        assert_eq!(report["summary"]["prefill"]["decision_count"], width);
        assert_eq!(report["summary"]["decode"]["decision_count"], 2 * width);
        assert_eq!(report["bitwise_logits"]["passed"], true);
        for arm in [&pair.reference, &pair.candidate] {
            assert!(arm
                .manifest
                .waves
                .iter()
                .all(|wave| wave.participant_count == 1));
        }
        let (report, code) = pair.compare();
        assert_insufficient(&report, code, "real serial reference and batched candidate");
    }
}

#[test]
fn serial_comparison_cannot_relabel_batched_waves_or_omit_a_canonical_decision() {
    let mut pair = Pair::with_width(3);
    let mut args = pair.args();
    args.comparison_shape = ComparisonShape::SerialToSerial;
    let (report, code) = compare(&args);
    assert_insufficient(&report, code, "two real serial captures");
    pair.candidate.manifest.mode = "serial".into();
    pair.candidate.persist();
    let (report, code) = compare(&args);
    assert_insufficient(&report, code, "actual wave width differs");
    let mut pair = Pair::with_serial_width(3);
    pair.candidate.manifest.decisions.pop();
    pair.candidate.persist();
    let mut args = pair.args();
    args.comparison_shape = ComparisonShape::SerialToSerial;
    let (report, code) = compare(&args);
    assert_insufficient(&report, code, "canonical decision missing");
}

#[test]
fn batch_width_is_actual_positive_inventory_not_a_fixed_eight_owner_rule() {
    for width in [1, 3] {
        let pair = Pair::with_batched_width(width);
        let (report, code) = compare(&batched_args(&pair));
        assert_eq!(code, 0, "{report:#}");
        assert_eq!(
            report["provenance"]["logical_decode_width"]["reference"],
            width
        );
    }
    let mut pair = Pair::with_batched_width(8);
    let mut other = Pair::with_batched_width(4);
    std::mem::swap(&mut pair.candidate, &mut other.candidate);
    let (report, code) = compare(&batched_args(&pair));
    assert_insufficient(&report, code, "must cover the same owners");
}

#[test]
fn batching_is_not_inferred_from_owner_inventory_or_relabelled_serial_waves() {
    let mut pair = Pair::with_width(8);
    let (report, code) = compare(&batched_args(&pair));
    assert_insufficient(&report, code, "two real batched captures");
    // Keep real per-owner receipts: relabelling cannot create an eight-owner wave.
    pair.reference.manifest.mode = "batched".into();
    pair.reference.persist();
    let (report, code) = compare(&batched_args(&pair));
    assert_insufficient(&report, code, "actual wave width differs");
    assert_eq!(report["bitwise_logits"]["passed"], false);
}

#[test]
fn same_batch_requires_complete_physical_node_coverage_and_every_decision() {
    let mut pair = Pair::with_batched_width(8);
    transition_tests::edit_arm_receipts(&mut pair.reference, |batch| {
        if batch.nodes[0].participants.len() > 1 {
            batch.nodes[0].participants.pop();
        }
    });
    let (report, code) = compare(&batched_args(&pair));
    assert_insufficient(&report, code, "physical node width differs");

    let mut pair = Pair::with_batched_width(8);
    pair.candidate.manifest.decisions.pop();
    pair.candidate.persist();
    let (report, code) = compare(&batched_args(&pair));
    assert_insufficient(&report, code, "canonical decision missing");
    assert_eq!(report["comparison_passed"], false);
    assert_eq!(report["bitwise_logits"]["passed"], false);
}

#[test]
fn bitwise_requirement_detects_signed_zero_even_when_all_quality_budgets_pass() {
    let mut pair = Pair::with_batched_width(2);
    let mut values = BASE_LOGITS;
    values[0] = -0.0;
    pair.candidate.set_logits("owner-1", 2, &values);
    let mut args = batched_args(&pair);
    let (report, code) = compare(&args);
    assert_eq!(code, 1, "{report:#}");
    assert_eq!(report["status"], "bitwise_failed");
    assert_eq!(report["evidence_complete"], true);
    assert_eq!(report["quality_passed"], true);
    assert_eq!(report["comparison_passed"], false);
    assert_eq!(report["bitwise_logits"]["mismatch_count"], 1);
    assert_eq!(report["summary"]["all"]["max_abs_logit_error"], 0.0);
    args.require_bitwise_logits = false;
    let (report, code) = compare(&args);
    assert_eq!(code, 0, "{report:#}");
    assert_eq!(report["comparison_passed"], true);
    assert_eq!(report["bitwise_logits"]["passed"], false);
}

#[test]
fn required_bits_preserve_quality_failure_and_independent_raw_validation() {
    let mut pair = Pair::with_batched_width(3);
    pair.candidate
        .set_logits("owner-0", 1, &[0.0, 2.0, 1.0, -8.0]);
    let (report, code) = compare(&batched_args(&pair));
    assert_eq!(code, 1, "{report:#}");
    assert_eq!(report["status"], "quality_failed");
    assert_eq!(report["evidence_complete"], true);
    assert_eq!(report["summary"]["prefill"]["quality_passed"], true);
    assert_eq!(report["summary"]["decode"]["quality_passed"], false);
    assert_eq!(report["bitwise_logits"]["passed"], false);

    // Matching saved f32 files cannot erase a differing raw device readback.
    let row = &mut pair
        .candidate
        .manifest
        .decisions
        .iter_mut()
        .find(|row| row.evidence.owner_id == "owner-0" && row.evidence.decision_index == 1)
        .unwrap()
        .logits;
    let bytes: Vec<_> = BASE_LOGITS
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    fs::write(pair.candidate.directory.join(&row.file), &bytes).unwrap();
    row.sha256 = sha256(&bytes);
    pair.candidate.persist();
    let (report, code) = compare(&batched_args(&pair));
    assert_insufficient(&report, code, "independent raw dtype conversion");
    assert_eq!(report["bitwise_logits"]["passed"], false);
}

#[test]
fn same_batch_keeps_exact_implementation_pins_and_no_undeclared_node_change() {
    let mut pair = Pair::with_batched_width(8);
    let bytes = b"different exact fixture executable, not real GPU evidence";
    let path = pair.root.path().join("new-binary");
    fs::write(&path, bytes).unwrap();
    pair.candidate.manifest.identity.as_mut().unwrap().binary = VNextTeacherFileIdentity {
        path: path.to_string_lossy().into_owned(),
        bytes: bytes.len() as u64,
        sha256: sha256(bytes),
    };
    transition_tests::edit_arm_receipts(&mut pair.candidate, |batch| {
        for node in &mut batch.nodes {
            node.provider_implementation_fingerprint = sha256(b"new provider");
        }
    });
    let pin = |arm: &fixture::Arm| {
        let checked = receipt::validate(&arm.directory, &arm.manifest.waves[0]).unwrap();
        let identity = arm.manifest.identity.as_ref().unwrap();
        transition::ExecutionPin {
            binary_sha256: identity.binary.sha256.clone(),
            resolved_plan_fingerprint: identity.resolved_plan_fingerprint.clone(),
            plan_id: checked.plan_id,
            plan_hash: checked.plan_hash,
            runtime_implementation_fingerprint: checked.runtime,
        }
    };
    let mut declaration = transition::ImplementationTransition {
        schema_version: 1,
        reference: pin(&pair.reference),
        candidate: pin(&pair.candidate),
        provider_changes: vec![transition::ProviderChange {
            provider_id: "provider.fixture".into(),
            reference_fingerprint: sha256(b"provider"),
            candidate_fingerprint: sha256(b"new provider"),
        }],
    };
    let mut args = batched_args(&pair);
    let (report, code) = compare(&args);
    assert_insufficient(
        &report,
        code,
        "actual executed plan/provider/runtime differs",
    );
    let path = pair.root.path().join("transition.json");
    args.implementation_transition = Some(path.clone());
    fs::write(&path, serde_json::to_vec(&declaration).unwrap()).unwrap();
    let (report, code) = compare(&args);
    assert_eq!(code, 0, "{report:#}");
    assert_eq!(
        report["provenance"]["execution_qualification"]["mode"],
        "pinned_implementation_transition"
    );
    let original = declaration.candidate.clone();
    declaration.candidate.plan_hash = sha256(b"wrong pin");
    fs::write(&path, serde_json::to_vec(&declaration).unwrap()).unwrap();
    let (report, code) = compare(&args);
    assert_insufficient(
        &report,
        code,
        "candidate execution identity does not match exact transition pin",
    );
    declaration.candidate = original;
    fs::write(&path, serde_json::to_vec(&declaration).unwrap()).unwrap();
    transition_tests::edit_arm_receipts(&mut pair.candidate, |batch| {
        batch.nodes[0].operation_id = "operation.changed".into();
    });
    let (report, code) = compare(&args);
    assert_insufficient(&report, code, "topology or execution semantics");
}
