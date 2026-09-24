//! Typed fixture construction only: no captured production artifact is rewritten.
use super::*;
use crate::numerical_transition::OutputResourceMapping;

fn edit_bindings(arm: &mut fixture::Arm, edit: impl Fn(&mut receipt::ReadbackRequest)) {
    for wave in &mut arm.manifest.waves {
        for raw in &mut wave.readbacks {
            let mut request: receipt::ReadbackRequest =
                serde_json::from_value(raw.request.clone()).unwrap();
            edit(&mut request);
            raw.request = serde_json::to_value(request).unwrap();
        }
        // Construct a valid fixture receipt, so rejection tests reach the
        // actual binding contract rather than merely failing its checksum.
        wave.receipt_fingerprint =
            receipt::readback_fingerprint(&wave.completion_fingerprint, &wave.readbacks).unwrap();
    }
    arm.persist();
}

fn mapped_pair() -> (Pair, NumericalTransition) {
    let mut pair = migrated_pair(1);
    edit_bindings(&mut pair.candidate, |request| {
        request.resource_id = "resource.activation.q8-program".into();
    });
    let mut d = declaration(&pair);
    d.schema_version = 2;
    d.output_resource_mapping = Some(OutputResourceMapping {
        reference_resource_id: pair.reference.manifest.waves[0].readbacks[0].request["resource_id"]
            .as_str()
            .unwrap()
            .into(),
        candidate_resource_id: "resource.activation.q8-program".into(),
    });
    (pair, d)
}

#[test]
fn numerical_output_mapping_qualifies_exact_names_without_rewriting_raw_or_receipts() {
    let (pair, d) = mapped_pair();
    let originals: Vec<_> = [&pair.reference, &pair.candidate]
        .into_iter()
        .map(|arm| {
            (
                fs::read(arm.directory.join("manifest.json")).unwrap(),
                arm.manifest
                    .waves
                    .iter()
                    .map(|wave| {
                        let file = wave.completion_receipt.as_ref().unwrap();
                        fs::read(arm.directory.join(&file.file)).unwrap()
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    let (report, code) = compare(&args(&pair, &d));
    assert_eq!(code, 0, "{report:#}");
    assert_eq!(report["evidence_complete"], true);
    let provenance = &report["provenance"];
    assert!(provenance["fixed_product_output_binding"].is_null());
    assert_eq!(
        provenance["execution_qualification"]["output_resource_mapping_verified"],
        true
    );
    let bindings = &provenance["product_output_bindings"];
    let mapping = d.output_resource_mapping.unwrap();
    assert_eq!(
        bindings["reference"]["resource_id"],
        mapping.reference_resource_id
    );
    assert_eq!(
        bindings["candidate"]["resource_id"],
        mapping.candidate_resource_id
    );
    for (arm, (manifest, receipts)) in [&pair.reference, &pair.candidate]
        .into_iter()
        .zip(originals)
    {
        assert_eq!(
            fs::read(arm.directory.join("manifest.json")).unwrap(),
            manifest
        );
        for (wave, original) in arm.manifest.waves.iter().zip(receipts) {
            assert_eq!(
                fs::read(
                    arm.directory
                        .join(&wave.completion_receipt.as_ref().unwrap().file)
                )
                .unwrap(),
                original
            );
        }
    }
}

#[test]
fn numerical_output_mapping_requires_declaration_and_exact_pins_in_both_arms() {
    let (pair, d) = mapped_pair();
    let undeclared = declaration(&pair);
    let (report, code) = compare(&args(&pair, &undeclared));
    assert_insufficient(&report, code, "product output binding differs");
    for reference in [true, false] {
        let mut wrong = d.clone();
        let mapping = wrong.output_resource_mapping.as_mut().unwrap();
        if reference {
            mapping.reference_resource_id = "wrong.reference".into();
        } else {
            mapping.candidate_resource_id = "wrong.candidate".into();
        }
        let (report, code) = compare(&args(&pair, &wrong));
        assert_insufficient(&report, code, "exact numerical transition mapping");
    }
    // Default mode still rejects a resource-only change with identical plans.
    let mut strict = Pair::new();
    edit_bindings(&mut strict.candidate, |request| {
        request.resource_id = "new.resource".into()
    });
    let (report, code) = strict.compare();
    assert_insufficient(&report, code, "product output binding differs");
}

#[test]
fn numerical_output_mapping_schema_is_explicit_strict_and_nonempty() {
    let (pair, d) = mapped_pair();
    for mutation in 0..6 {
        let mut wire = serde_json::to_value(&d).unwrap();
        let reason = match mutation {
            0 => {
                wire["schema_version"] = json!(1);
                "requires schema 2"
            }
            1 => {
                wire.as_object_mut()
                    .unwrap()
                    .remove("output_resource_mapping");
                "requires output resource mapping"
            }
            2 => {
                wire["output_resource_mapping"]["reference_resource_id"] = json!(" ");
                "distinct nonempty"
            }
            3 => {
                wire["output_resource_mapping"]["candidate_resource_id"] =
                    wire["output_resource_mapping"]["reference_resource_id"].clone();
                "distinct nonempty"
            }
            4 => {
                wire["output_resource_mapping"]["allow_layout_change"] = json!(true);
                "unknown field"
            }
            _ => {
                wire["schema_version"] = json!(3);
                "unsupported numerical transition schema"
            }
        };
        let a = args(&pair, &wire);
        let error = crate::numerical_transition::DeclaredNumericalTransition::read(
            a.numerical_profile_transition.as_deref().unwrap(),
        )
        .err()
        .unwrap();
        assert!(format!("{error:#}").contains(reason), "{error:#}");
    }
}

#[test]
fn numerical_output_mapping_preserves_node_usage_offset_dtype_and_vocabulary() {
    for mutation in 0..5 {
        let (mut pair, d) = mapped_pair();
        edit_bindings(&mut pair.candidate, |request| match mutation {
            0 => request.node_id = "node.0".into(), // A real other node, not an unknown ID.
            1 => request.expected_usage = "weights".into(),
            2 => request.logical_offset_bytes += 16,
            3 => request.output_layout.element_type = "f16".into(),
            _ => request.output_layout.element_count += 1,
        });
        let (report, code) = compare(&args(&pair, &d));
        let reason = match mutation {
            1 => "activation output node",
            3 => "raw readback layout byte count differs",
            4 => "readback is not the complete vocabulary",
            _ => "product output binding differs beyond declared resource ID mapping",
        };
        assert_insufficient(&report, code, reason);
    }
}

#[test]
fn numerical_output_mapping_does_not_hide_later_wave_owner_or_receipt_changes() {
    for mutation in 0..3 {
        let (mut pair, d) = mapped_pair();
        let wave = pair.candidate.manifest.waves.last_mut().unwrap();
        if mutation == 0 {
            for raw in &mut wave.readbacks {
                raw.request["resource_id"] = json!("different.later-wave");
            }
        } else {
            wave.readbacks[0].request["resource_id"] = json!("different.one-owner");
        }
        // The third case deliberately leaves the original checksum intact.
        if mutation != 2 {
            wave.receipt_fingerprint =
                receipt::readback_fingerprint(&wave.completion_fingerprint, &wave.readbacks)
                    .unwrap();
        }
        pair.candidate.persist();
        let (report, code) = compare(&args(&pair, &d));
        let reason = match mutation {
            0 => "product output binding changed between physical waves",
            1 => "physical owners have different product output bindings",
            _ => "readback receipt fingerprint differs",
        };
        assert_insufficient(&report, code, reason);
    }
}

#[test]
fn numerical_output_mapping_still_fails_unchanged_prefill_quality_budget() {
    let (mut pair, d) = mapped_pair();
    pair.candidate
        .set_logits("owner-0", 0, &[0.0, 2.0, 1.0, -8.0]);
    let (report, code) = compare(&args(&pair, &d));
    assert_eq!(code, 1, "{report:#}");
    assert_eq!(report["evidence_complete"], true);
    assert_eq!(report["summary"]["prefill"]["quality_passed"], false);
    assert_eq!(report["quality_passed"], false);
}
