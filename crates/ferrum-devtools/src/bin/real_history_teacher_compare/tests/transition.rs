use super::*;
use ferrum_interfaces::vnext::ProviderExecutionSemantics;

fn migrate(pair: &mut Pair) {
    let identity = pair.candidate.manifest.identity.as_mut().unwrap();
    let bytes = b"candidate actual executable bytes";
    let path = pair.root.path().join("migrated-binary");
    fs::write(&path, bytes).unwrap();
    identity.binary = VNextTeacherFileIdentity {
        path: path.to_string_lossy().into_owned(),
        bytes: bytes.len() as u64,
        sha256: sha256(bytes),
    };
    identity.resolved_plan_fingerprint = sha256(b"new resolved plan");
    edit_all_receipts(pair, |batch| {
        batch.plan_id = "plan.migrated".into();
        batch.plan_hash = sha256(b"new actual plan");
        batch.runtime_implementation_fingerprint = sha256(b"new runtime");
        for node in &mut batch.nodes {
            node.provider_implementation_fingerprint = sha256(b"new provider");
        }
    });
}

/// Fixtures intentionally reconstruct every typed envelope after a declared change.
/// This is test construction only; real capture files are never rewritten by the tool.
fn edit_all_receipts(pair: &mut Pair, edit: impl Fn(&mut receipt::Batch)) {
    edit_arm_receipts(&mut pair.candidate, edit);
}

pub(super) fn edit_arm_receipts(arm: &mut fixture::Arm, edit: impl Fn(&mut receipt::Batch)) {
    for wave in &mut arm.manifest.waves {
        let file = wave.completion_receipt.as_mut().unwrap();
        let mut completion: receipt::Completion =
            serde_json::from_slice(&fs::read(arm.directory.join(&file.file)).unwrap()).unwrap();
        let batch = &mut completion.submission.batch_identity;
        edit(batch);
        let mut participant_index = 0;
        for (node_index, node) in batch.nodes.iter_mut().enumerate() {
            node.node_index = node_index;
            for part in &mut node.participants {
                part.participant_index = participant_index;
                participant_index += 1;
                part.node_key.node_id = node.node_id.clone();
                part.identity.node_id = Some(serde_json::from_value(json!(node.node_id)).unwrap());
                part.identity.provider_id =
                    Some(serde_json::from_value(json!(node.provider_id)).unwrap());
                part.identity.plan_id = Some(serde_json::from_value(json!(batch.plan_id)).unwrap());
                part.identity.plan_hash =
                    Some(serde_json::from_value(json!(batch.plan_hash)).unwrap());
                part.identity.runtime_implementation_fingerprint =
                    Some(batch.runtime_implementation_fingerprint.clone());
                part.identity.operation_id =
                    Some(serde_json::from_value(json!(node.operation_id)).unwrap());
            }
            node.fingerprint = receipt::node_fingerprint(node).unwrap();
        }
        batch.participants = batch
            .nodes
            .iter()
            .flat_map(|node| node.participants.clone())
            .collect();
        batch.fingerprint = sha256(&serde_json::to_vec(&batch).unwrap());
        let submission = &mut completion.submission;
        submission.fingerprint = receipt::submission_fingerprint(submission).unwrap();
        submission.participants = submission
            .batch_identity
            .participants
            .iter()
            .map(|part| receipt::SubmissionParticipant {
                slot_id: submission.slot_id.clone(),
                participant_index: part.participant_index,
                identity: part.identity.clone(),
                batch_submission_fingerprint: submission.fingerprint.clone(),
            })
            .collect();
        completion.fingerprint = receipt::completion_fingerprint(&completion).unwrap();
        completion.participants = completion
            .submission
            .participants
            .iter()
            .cloned()
            .map(|submission| receipt::CompletionParticipant {
                submission,
                disposition: json!({"status":"succeeded"}),
                batch_completion_fingerprint: completion.fingerprint.clone(),
            })
            .collect();
        let bytes = serde_json::to_vec(&completion).unwrap();
        fs::write(arm.directory.join(&file.file), &bytes).unwrap();
        file.bytes = bytes.len() as u64;
        file.sha256 = sha256(&bytes);
        wave.completion_fingerprint = completion.fingerprint;
        wave.receipt_fingerprint =
            receipt::readback_fingerprint(&wave.completion_fingerprint, &wave.readbacks).unwrap();
    }
    arm.persist();
}

fn declaration(pair: &Pair) -> transition::ImplementationTransition {
    let pin = |arm: &fixture::Arm| {
        let identity = arm.manifest.identity.as_ref().unwrap();
        let receipt = receipt::validate(&arm.directory, &arm.manifest.waves[0]).unwrap();
        transition::ExecutionPin {
            binary_sha256: identity.binary.sha256.clone(),
            resolved_plan_fingerprint: identity.resolved_plan_fingerprint.clone(),
            plan_id: receipt.plan_id,
            plan_hash: receipt.plan_hash,
            runtime_implementation_fingerprint: receipt.runtime,
        }
    };
    transition::ImplementationTransition {
        schema_version: 1,
        reference: pin(&pair.reference),
        candidate: pin(&pair.candidate),
        provider_changes: vec![transition::ProviderChange {
            provider_id: "provider.fixture".into(),
            reference_fingerprint: sha256(b"provider"),
            candidate_fingerprint: sha256(b"new provider"),
        }],
    }
}
fn args(pair: &Pair, declaration: &transition::ImplementationTransition) -> Args {
    let mut args = pair.args();
    let path = pair.root.path().join("transition.json");
    fs::write(&path, serde_json::to_vec(declaration).unwrap()).unwrap();
    args.implementation_transition = Some(path);
    args
}

#[test]
fn declared_transition_requires_exact_pins_and_keeps_strict_default() {
    let mut pair = Pair::new();
    migrate(&mut pair);
    let (report, code) = pair.compare();
    assert_insufficient(&report, code, "resolved plan identity differs");
    let mut d = declaration(&pair);
    let (report, code) = compare(&args(&pair, &d));
    assert_eq!(code, 0, "{report:#}");
    assert_eq!(
        report["provenance"]["execution_qualification"]["mode"],
        "pinned_implementation_transition"
    );
    assert_eq!(
        report["provenance"]["execution_qualification"]["matched_provider_node_counts"]
            ["provider.fixture"],
        2
    );
    d.candidate.plan_hash = sha256(b"wrong actual plan");
    let (report, code) = compare(&args(&pair, &d));
    assert_insufficient(&report, code, "exact transition pin");
}

#[test]
fn transition_rejects_unlisted_changed_and_unused_providers() {
    let mut pair = Pair::new();
    migrate(&mut pair);
    let mut d = declaration(&pair);
    d.provider_changes[0].provider_id = "provider.other".into();
    let (report, code) = compare(&args(&pair, &d));
    assert_insufficient(&report, code, "undeclared provider");
    d = declaration(&pair);
    d.provider_changes.push(transition::ProviderChange {
        provider_id: "provider.unused".into(),
        reference_fingerprint: sha256(b"a"),
        candidate_fingerprint: sha256(b"b"),
    });
    let (report, code) = compare(&args(&pair, &d));
    assert_insufficient(&report, code, "unused provider exemption");
    d = declaration(&pair);
    d.provider_changes[0].candidate_fingerprint = sha256(b"wrong provider");
    let (report, code) = compare(&args(&pair, &d));
    assert_insufficient(&report, code, "exact declared transition");
}

#[test]
fn transition_does_not_authorize_changed_topology_or_execution_semantics() {
    for semantics in [false, true] {
        let mut pair = Pair::new();
        migrate(&mut pair);
        let d = declaration(&pair);
        edit_all_receipts(&mut pair, |batch| {
            if semantics {
                batch.nodes[0].provider_execution_semantics =
                    ProviderExecutionSemantics::bitwise_eager_and_replay();
            } else {
                batch.nodes[0].operation_id = "operation.different".into();
            }
        });
        let (report, code) = compare(&args(&pair, &d));
        assert_insufficient(&report, code, "topology or execution semantics");
    }
}

#[test]
fn transition_preserves_source_precision_configuration_history_and_output_contracts() {
    for change in 0..6 {
        let mut pair = Pair::new();
        migrate(&mut pair);
        let d = declaration(&pair);
        let reason = match change {
            0 => {
                pair.candidate
                    .manifest
                    .identity
                    .as_mut()
                    .unwrap()
                    .numerical_profile = "different.profile".into();
                "model/numerical/KV"
            }
            1 => {
                pair.candidate.manifest.configuration["max_model_len"] = json!(32);
                "configuration differs"
            }
            2 => {
                pair.candidate.manifest.decisions[0].evidence.history_sha256 = sha256(b"different");
                "history binding differs"
            }
            3 => {
                pair.candidate
                    .manifest
                    .identity
                    .as_mut()
                    .unwrap()
                    .family_fingerprint = sha256(b"different model");
                "model/numerical/KV"
            }
            _ => {
                for wave in &mut pair.candidate.manifest.waves {
                    for raw in &mut wave.readbacks {
                        if change == 4 {
                            raw.request["logical_offset_bytes"] = json!(16);
                        } else {
                            raw.request["resource_id"] = json!("undeclared.output.resource");
                        }
                    }
                    wave.receipt_fingerprint = receipt::readback_fingerprint(
                        &wave.completion_fingerprint,
                        &wave.readbacks,
                    )
                    .unwrap();
                }
                "output binding differs"
            }
        };
        pair.candidate.persist();
        let (report, code) = compare(&args(&pair, &d));
        assert_insufficient(&report, code, reason);
    }
}

#[test]
fn declared_transition_still_fails_full_vocabulary_tail_quality_and_corrupt_binary() {
    let mut pair = Pair::new();
    migrate(&mut pair);
    let d = declaration(&pair);
    pair.candidate
        .set_logits("owner-0", 1, &[0.0, 2.0, 1.0, -8.0]);
    let (report, code) = compare(&args(&pair, &d));
    assert_eq!(code, 1, "{report:#}");
    assert_eq!(report["evidence_complete"], true);
    assert_eq!(report["summary"]["all"]["argmax_agreement_count"], 12);
    fs::write(
        &pair
            .candidate
            .manifest
            .identity
            .as_ref()
            .unwrap()
            .binary
            .path,
        b"corrupt",
    )
    .unwrap();
    let (report, code) = compare(&args(&pair, &d));
    assert_insufficient(&report, code, "external provenance file");
}
