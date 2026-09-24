use super::*;
use crate::numerical_transition::{
    NodeChange, NodeImplementation, NumericalPin, NumericalScope, NumericalTransition,
};
use ferrum_interfaces::vnext::{ProductModelSourceIdentity, ProviderExecutionSemantics};
use transition_tests::edit_arm_receipts;

#[path = "numerical_transition/input_sum.rs"]
mod input_sum;
#[path = "numerical_transition/output_resource.rs"]
mod output_resource;
#[path = "numerical_transition/stream_mmq.rs"]
mod stream_mmq;

fn migrated_pair(ffn_nodes: usize) -> Pair {
    let mut pair = Pair::new();
    for (arm, profile, candidate) in [
        (&mut pair.reference, "qwen3_5.f32-master", false),
        (&mut pair.candidate, "qwen3_5.f32-master.q8-swiglu", true),
    ] {
        let identity = arm.manifest.identity.as_mut().unwrap();
        identity.numerical_profile = profile.into();
        if candidate {
            identity.family_fingerprint = sha256(b"q8 family");
            identity.program_fingerprint = sha256(b"q8 program");
            identity.resolved_plan_fingerprint = sha256(b"q8 resolved");
        }
        arm.manifest.configuration["numerical_execution"] = json!({"require":profile});
        edit_arm_receipts(arm, |batch| {
            if candidate {
                batch.plan_id = "plan.q8-ffn".into();
                batch.plan_hash = sha256(b"q8 plan");
            }
            for node in batch.nodes.iter_mut().take(ffn_nodes) {
                node.operation_id = if candidate {
                    "operation.dense_swiglu.q8-f32scale"
                } else {
                    "operation.dense_swiglu"
                }
                .into();
                node.provider_id = if candidate {
                    "provider.q8-ffn"
                } else {
                    "provider.ffn"
                }
                .into();
                node.provider_implementation_fingerprint = sha256(node.provider_id.as_bytes());
            }
        });
    }
    pair
}

fn declaration(pair: &Pair) -> NumericalTransition {
    let pin = |arm: &fixture::Arm| {
        let identity = arm.manifest.identity.as_ref().unwrap();
        let receipt = receipt::validate(&arm.directory, &arm.manifest.waves[0]).unwrap();
        NumericalPin {
            numerical_profile: identity.numerical_profile.parse().unwrap(),
            family_fingerprint: identity.family_fingerprint.clone(),
            program_fingerprint: identity.program_fingerprint.clone(),
            execution: transition::ExecutionPin {
                binary_sha256: identity.binary.sha256.clone(),
                resolved_plan_fingerprint: identity.resolved_plan_fingerprint.clone(),
                plan_id: receipt.plan_id,
                plan_hash: receipt.plan_hash,
                runtime_implementation_fingerprint: receipt.runtime,
            },
        }
    };
    let r =
        receipt::validate(&pair.reference.directory, &pair.reference.manifest.waves[0]).unwrap();
    let c =
        receipt::validate(&pair.candidate.directory, &pair.candidate.manifest.waves[0]).unwrap();
    let implementation = |node: &receipt::NodeSignature| NodeImplementation {
        operation_id: node.operation_id.clone(),
        provider_id: node.provider_id.clone(),
        implementation_fingerprint: node.implementation_fingerprint.clone(),
    };
    NumericalTransition {
        schema_version: 1,
        scope: NumericalScope::DenseSwiGluQ8F32ScaleV1,
        reference: pin(&pair.reference),
        candidate: pin(&pair.candidate),
        ordered_node_count: r.node_signature.len(),
        output_resource_mapping: None,
        node_changes: r
            .node_signature
            .iter()
            .zip(&c.node_signature)
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .map(|(node_index, (a, b))| NodeChange {
                node_index,
                node_id: a.node_id.clone(),
                execution_semantics: serde_json::from_value(
                    serde_json::to_value(a.execution_semantics).unwrap(),
                )
                .unwrap(),
                reference: implementation(a),
                candidate: implementation(b),
            })
            .collect(),
    }
}

fn args(pair: &Pair, declaration: &impl Serialize) -> Args {
    let mut args = pair.args();
    let path = pair.root.path().join("numerical-transition.json");
    fs::write(&path, serde_json::to_vec(declaration).unwrap()).unwrap();
    args.numerical_profile_transition = Some(path);
    args
}

#[test]
fn numerical_transition_same_elf_requires_exact_declaration_and_reports_both_profiles() {
    let pair = migrated_pair(1);
    let (report, code) = pair.compare();
    assert_insufficient(&report, code, "model/numerical/KV");
    let d = declaration(&pair);
    assert_eq!(
        d.reference.execution.binary_sha256,
        d.candidate.execution.binary_sha256
    );
    let a = args(&pair, &d);
    let (report, code) = compare(&a);
    assert_eq!(code, 0, "{report:#}");
    assert_eq!(report["release_approved"], false);
    assert!(report["provenance"]["numerical_profile"].is_null());
    assert_eq!(
        report["provenance"]["numerical_profiles"],
        json!({
            "reference":"qwen3_5.f32-master", "candidate":"qwen3_5.f32-master.q8-swiglu"
        })
    );
    assert_eq!(report["provenance"]["same_binary"], true);
    let qualified = &report["provenance"]["execution_qualification"];
    assert_eq!(qualified["mode"], "pinned_numerical_profile_transition");
    assert_eq!(qualified["changed_node_count"], 1);
    assert_eq!(
        qualified["declaration"]["sha256"],
        sha256(&fs::read(a.numerical_profile_transition.unwrap()).unwrap())
    );
    for (group, count) in [("all", 12), ("prefill", 4), ("decode", 8)] {
        assert_eq!(report["summary"][group]["decision_count"], count);
        assert_eq!(report["summary"][group]["quality_passed"], true);
    }
}

#[test]
fn numerical_transition_modes_are_mutually_exclusive_in_cli_and_direct_calls() {
    let parsed = Args::try_parse_from([
        "compare",
        "--reference-dir",
        "r",
        "--candidate-dir",
        "c",
        "--output-file",
        "o",
        "--mean-delta-nll-limit",
        "0.01",
        "--max-delta-nll-limit",
        "0.1",
        "--mean-kl-limit",
        "0.001",
        "--max-kl-limit",
        "0.01",
        "--implementation-transition",
        "i",
        "--numerical-profile-transition",
        "n",
    ]);
    assert!(parsed.is_err());
    let pair = migrated_pair(1);
    let mut a = args(&pair, &declaration(&pair));
    a.implementation_transition = a.numerical_profile_transition.clone();
    let (report, code) = compare(&a);
    assert_insufficient(&report, code, "mutually exclusive");
}

#[test]
fn numerical_transition_rejects_wrong_family_program_and_execution_pins() {
    let pair = migrated_pair(1);
    for arm in ["reference", "candidate"] {
        for field in [
            "family_fingerprint",
            "program_fingerprint",
            "execution/binary_sha256",
            "execution/resolved_plan_fingerprint",
            "execution/plan_hash",
            "execution/runtime_implementation_fingerprint",
            "execution/plan_id",
        ] {
            let mut d = serde_json::to_value(declaration(&pair)).unwrap();
            *d.pointer_mut(&format!("/{arm}/{field}")).unwrap() =
                json!(if field == "execution/plan_id" {
                    "plan.wrong".into()
                } else {
                    sha256(b"wrong pin")
                });
            let (report, code) = compare(&args(&pair, &d));
            assert_insufficient(&report, code, "exact numerical transition pin");
        }
    }
}

#[test]
fn numerical_transition_rejects_auto_empty_equal_reverse_and_unknown_profile_pairs() {
    let pair = migrated_pair(1);
    for (reference, candidate) in [
        ("", "qwen3_5.f32-master.q8-swiglu"),
        ("auto", "qwen3_5.f32-master.q8-swiglu"),
        ("qwen3_5.f32-master", "qwen3_5.f32-master"),
        ("qwen3_5.f32-master.q8-swiglu", "qwen3_5.f32-master"),
        ("qwen3_5.f32-master", "qwen3_5.f32-master.q8-gdn"),
    ] {
        let mut d = serde_json::to_value(declaration(&pair)).unwrap();
        d["reference"]["numerical_profile"] = json!(reference);
        d["candidate"]["numerical_profile"] = json!(candidate);
        let a = args(&pair, &d);
        assert!(
            crate::numerical_transition::DeclaredNumericalTransition::read(
                a.numerical_profile_transition.as_deref().unwrap()
            )
            .is_err()
        );
        let (report, code) = compare(&a);
        assert_eq!(code, 2, "{report:#}");
    }
}

#[test]
fn numerical_transition_denies_unknown_fields_at_every_declaration_level() {
    let pair = migrated_pair(1);
    for pointer in [
        "",
        "/reference",
        "/candidate/execution",
        "/node_changes/0",
        "/node_changes/0/reference",
        "/node_changes/0/candidate",
        "/node_changes/0/execution_semantics",
        "/node_changes/0/execution_semantics/contract_version",
    ] {
        let mut d = serde_json::to_value(declaration(&pair)).unwrap();
        d.pointer_mut(pointer)
            .unwrap()
            .as_object_mut()
            .unwrap()
            .insert("undeclared".into(), json!(true));
        let a = args(&pair, &d);
        let error = crate::numerical_transition::DeclaredNumericalTransition::read(
            a.numerical_profile_transition.as_deref().unwrap(),
        )
        .err()
        .unwrap();
        assert!(format!("{error:#}").contains("unknown field"), "{error:#}");
    }
}

#[test]
fn numerical_transition_requires_complete_ordered_node_changes_and_exact_tuples() {
    let pair = migrated_pair(2);
    let base = declaration(&pair);
    let (report, code) = compare(&args(&pair, &base));
    assert_eq!(code, 0, "{report:#}");
    for mutation in 0..9 {
        let mut d = base.clone();
        let reason = match mutation {
            0 => {
                d.node_changes.pop();
                "undeclared numerical transition"
            }
            1 => {
                d.node_changes.clear();
                "no node changes"
            }
            2 => {
                d.node_changes.swap(0, 1);
                "unordered, duplicated"
            }
            3 => {
                d.node_changes[1].node_index = 0;
                "unordered, duplicated"
            }
            4 => {
                d.node_changes[1].node_id = d.node_changes[0].node_id.clone();
                "unordered, duplicated"
            }
            5 => {
                d.ordered_node_count += 1;
                "ordered node inventory"
            }
            6 => {
                d.node_changes[0].reference.provider_id = "provider.wrong".into();
                "exact declared numerical transition"
            }
            7 => {
                d.node_changes[0].candidate.implementation_fingerprint = sha256(b"wrong");
                "exact declared numerical transition"
            }
            _ => {
                d.node_changes[0].node_id = "node.unused".into();
                "exact declared numerical transition"
            }
        };
        let (report, code) = compare(&args(&pair, &d));
        assert_insufficient(&report, code, reason);
    }
}

#[test]
fn numerical_transition_cannot_declare_gdn_head_or_undeclared_provider_changes() {
    for operation in ["operation.gated_delta", "operation.logits_head"] {
        let mut pair = migrated_pair(1);
        let d = declaration(&pair);
        edit_arm_receipts(&mut pair.candidate, |batch| {
            batch.nodes[1].operation_id = operation.into();
            batch.nodes[1].provider_id = "provider.q8-extra".into();
            batch.nodes[1].provider_implementation_fingerprint = sha256(b"extra");
        });
        let (report, code) = compare(&args(&pair, &d));
        assert_insufficient(&report, code, "undeclared numerical transition");
        let (report, code) = compare(&args(&pair, &declaration(&pair)));
        assert_insufficient(&report, code, "outside dense SwiGLU Q8 scope");
    }
}

#[test]
fn numerical_transition_cannot_reorder_nodes_or_change_execution_semantics() {
    for semantic_change in [false, true] {
        let mut pair = migrated_pair(2);
        let d = declaration(&pair);
        edit_arm_receipts(&mut pair.candidate, |batch| {
            if semantic_change {
                batch.nodes[0].provider_execution_semantics =
                    ProviderExecutionSemantics::bitwise_eager_and_replay();
            } else {
                batch.nodes.swap(0, 1);
            }
        });
        let (report, code) = compare(&args(&pair, &d));
        assert_insufficient(&report, code, "topology or execution semantics");
    }
}

#[test]
fn numerical_transition_only_normalizes_explicit_require_and_preserves_configuration() {
    for mutation in 0..8 {
        let mut pair = migrated_pair(1);
        let d = declaration(&pair);
        let config = &mut pair.candidate.manifest.configuration;
        let reason = match mutation {
            0 => {
                config["numerical_execution"] = json!("auto");
                "numerical_execution.require"
            }
            1 => {
                config
                    .as_object_mut()
                    .unwrap()
                    .remove("numerical_execution");
                "numerical_execution.require"
            }
            2 => {
                config["numerical_execution"]["require"] = json!("qwen3_5.f32-master");
                "numerical_execution.require"
            }
            3 => {
                config["numerical_execution"]["ignored"] = json!(true);
                "numerical_execution.require"
            }
            4 => {
                config["runtime_memory_budget_bytes"] = json!(1024);
                "configuration differs"
            }
            5 => {
                config["max_model_len"] = json!(32);
                "configuration differs"
            }
            6 => {
                config["sequence_slots"] = json!(8);
                "configuration differs"
            }
            _ => {
                config["backend"]["backend_options"]["numerical_route"] = json!("other");
                "configuration differs"
            }
        };
        pair.candidate.persist();
        let (report, code) = compare(&args(&pair, &d));
        assert_insufficient(&report, code, reason);
    }
}

#[test]
fn numerical_transition_retains_kv_source_history_output_and_raw_file_checks() {
    for mutation in 0..6 {
        let mut pair = migrated_pair(1);
        let d = declaration(&pair);
        let reason = match mutation {
            0 => {
                pair.candidate
                    .manifest
                    .identity
                    .as_mut()
                    .unwrap()
                    .kv_storage = "int8".into();
                "F16 KV policy"
            }
            1 => {
                pair.candidate.manifest.decisions[0].evidence.history_sha256 =
                    sha256(b"wrong history");
                "history binding differs"
            }
            2 => {
                for wave in &mut pair.candidate.manifest.waves {
                    for raw in &mut wave.readbacks {
                        raw.request["logical_offset_bytes"] = json!(16);
                    }
                    wave.receipt_fingerprint = receipt::readback_fingerprint(
                        &wave.completion_fingerprint,
                        &wave.readbacks,
                    )
                    .unwrap();
                }
                "output binding differs"
            }
            3 => {
                let file = &pair.candidate.manifest.decisions[0].logits.file;
                fs::write(pair.candidate.directory.join(file), b"corrupt").unwrap();
                "byte count differs"
            }
            4 => {
                let identity = pair.candidate.manifest.identity.as_mut().unwrap();
                let mut source: ProductModelSourceIdentity =
                    serde_json::from_value(identity.model_source.clone()).unwrap();
                source.resolved_sources.weights.files[0].sha256 = sha256(b"alternate weights");
                source.validate().unwrap();
                identity.model_source = serde_json::to_value(source).unwrap();
                "source content differs"
            }
            _ => {
                let path = &pair
                    .candidate
                    .manifest
                    .identity
                    .as_ref()
                    .unwrap()
                    .binary
                    .path;
                fs::write(path, b"corrupt").unwrap();
                "external provenance file"
            }
        };
        pair.candidate.persist();
        let (report, code) = compare(&args(&pair, &d));
        assert_insufficient(&report, code, reason);
    }
}

#[test]
fn numerical_transition_retains_quality_budgets_for_prefill_decode_and_full_vocabulary() {
    for decision in [0, 1] {
        let mut pair = migrated_pair(1);
        let d = declaration(&pair);
        pair.candidate
            .set_logits("owner-0", decision, &[0.0, 2.0, 1.0, -8.0]);
        let (report, code) = compare(&args(&pair, &d));
        assert_eq!(code, 1, "{report:#}");
        assert_eq!(report["evidence_complete"], true);
        assert_eq!(report["quality_passed"], false);
        assert_eq!(report["summary"]["all"]["argmax_agreement_count"], 12);
        let group = if decision == 0 { "prefill" } else { "decode" };
        assert_eq!(report["summary"][group]["quality_passed"], false);
        let checks = report["summary"][group]["checks"].as_array().unwrap();
        assert_eq!(checks.len(), 4);
        assert!(checks.iter().any(|check| check["passed"] == false));
    }
}
