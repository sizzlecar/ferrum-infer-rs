//! Typed evidence-contract fixtures, not hardware/performance claims.
use super::*;
use ferrum_interfaces::vnext::{ModelArtifactSourceRole, ProductModelSourceIdentity};

const MODEL_BYTES: u64 = 4_000_000;

fn inventory() -> Value {
    let mut offset = 32_u64;
    let mut tensors = Vec::new();
    let mut counts = BTreeMap::<String, usize>::new();
    for layer in 0..2 {
        for role in ["gate", "up", "down"] {
            let q6 = layer == 0 && role == "down";
            let (dtype, id, bytes_per_block, format) = if q6 {
                ("Q6K", 14, 210, "quantization.gguf.q6-k")
            } else {
                ("Q4K", 12, 144, "quantization.gguf.q4-k")
            };
            let bytes = 1024 * 1024 / 256 * bytes_per_block;
            tensors.push(json!({"name":format!("blk.{layer}.ffn_{role}.weight"),"dtype":dtype,
                "ggml_type":id,"candle_dtype_available":true,"quantization_format":format,
                "dimensions":[1024,1024],"block_axis":1,"logical_values_per_block":256,
                "bytes_per_block":bytes_per_block,"elements":1024*1024,"absolute_offset":offset,"bytes":bytes}));
            offset += bytes;
            *counts.entry(dtype.into()).or_default() += 1;
        }
    }
    json!({"file":"weights.gguf","file_length_source":"local_file","local_file_bytes":MODEL_BYTES,
        "payload_verified":false,"tensor_payloads_materialized":false,
        "inventory":{"schema_version":1,"architecture":"qwen35","quantization_version":2,
            "declared_file_bytes":MODEL_BYTES,"tensor_data_offset":32,"tensor_payload_bytes":offset-32,
            "split":null,"tensor_counts_by_dtype":counts,"tensors":tensors}})
}

fn write_artifact(directory: &Path, name: &str, value: &impl Serialize) -> Value {
    let bytes = serde_json::to_vec(value).unwrap();
    fs::write(directory.join(name), &bytes).unwrap();
    json!({"file":name,"bytes":bytes.len(),"sha256":sha256(&bytes)})
}

fn command(node: &receipt::Node, wave: &VNextTeacherWaveEvidence, candidate: bool) -> Value {
    let rows: usize = wave
        .participants
        .iter()
        .map(|p| p.immediate_end - p.immediate_start)
        .sum();
    let count =
        if candidate && node.node_id == "node.layer.0.feed_forward" && (5..=7).contains(&rows) {
            6
        } else {
            4
        };
    json!({"command_index":node.node_index,"node_index":node.node_index,"command_phase":"compute",
        "native_op_id":"vnext_dense_swiglu","execution_path":"eager",
        "batching_form":if wave.participant_count==1 {"scalar"}else{"packed"},
        "participant_start":0,"participant_count":wave.participant_count,"token_count":rows,
        "compute_dispatch_count":count,"transfer_command_count":0,"reusable_graph_node_count":null,
        "statistical_evidence":{"schema_version":1,"family_signature":vec![9_u8;32],"token_count":rows,
            "compute_dispatches":count,"transfer_commands":0,"work":{
                "logical_units":rows*4096,"padded_units":rows*4096,"inner_work_units":rows*(3*1024*1024+1024),
                "grid_blocks":1,"peak_scratch_bytes":rows*6144,"staged_weight_bytes":0,
                "host_to_device_bytes":0,"device_to_host_bytes":0,"device_to_device_bytes":0,"fill_bytes":0}}})
}

fn native_sidecars(arm: &fixture::Arm, candidate: bool) {
    let mut entries = Vec::new();
    for wave in &arm.manifest.waves {
        let receipt_file = wave.completion_receipt.as_ref().unwrap();
        let completion: receipt::Completion =
            serde_json::from_slice(&fs::read(arm.directory.join(&receipt_file.file)).unwrap())
                .unwrap();
        let commands: Vec<_> = completion
            .submission
            .batch_identity
            .nodes
            .iter()
            .map(|node| command(node, wave, candidate))
            .collect();
        let record = json!({"schema_version":1,"artifact_type":"ferrum.teacher.completed_native_work",
            "wave_index":wave.wave_index,"kind":wave.kind,"participant_count":wave.participant_count,
            "coverage":"backend_reported_only_no_inferred_node_coverage",
            "completion_fingerprint":wave.completion_fingerprint,"receipt_fingerprint":wave.receipt_fingerprint,
            "submission_fingerprint":completion.submission.fingerprint,
            "batch_identity_fingerprint":completion.submission.batch_identity.fingerprint,
            "device":{"commands":commands,"replayed_segments":[]}});
        let artifact = write_artifact(
            &arm.directory,
            &format!("native-work-{:06}.json", wave.wave_index),
            &record,
        );
        entries.push(json!({"wave_index":wave.wave_index,"completion_fingerprint":wave.completion_fingerprint,
            "receipt_fingerprint":wave.receipt_fingerprint,"artifact":artifact}));
    }
    write_artifact(
        &arm.directory,
        "native-work-manifest.json",
        &json!({
        "schema_version":1,"artifact_type":"ferrum.teacher.completed_native_work_index","complete":true,
        "identity":arm.manifest.identity,"file_limit_bytes":16*1024*1024,"waves":entries,"errors":[]}),
    );
}

fn native_pair(width: usize) -> (Pair, Args) {
    let mut pair = Pair::with_batched_width(width);
    for arm in [&mut pair.reference, &mut pair.candidate] {
        let identity = arm.manifest.identity.as_mut().unwrap();
        let mut source: ProductModelSourceIdentity =
            serde_json::from_value(identity.model_source.clone()).unwrap();
        source.resolved_sources.weights.files[0].sha256 =
            sha256(b"explicit fixture full-model pin");
        source.resolved_sources.weights.files[0].size_bytes = MODEL_BYTES;
        // Weight source is not semantic/tokenizer storage; their pins stay intact.
        assert_eq!(
            source
                .resolved_sources
                .for_role(ModelArtifactSourceRole::Weights)
                .files
                .len(),
            1
        );
        identity.model_source = serde_json::to_value(source).unwrap();
        for wave in &mut arm.manifest.waves {
            for output in &mut wave.readbacks {
                output.request["node_id"] = json!("node.layer.1.feed_forward");
            }
        }
        transition_tests::edit_arm_receipts(arm, |batch| {
            for (layer, node) in batch.nodes.iter_mut().enumerate() {
                node.node_id = format!("node.layer.{layer}.feed_forward");
                node.operation_id = "operation.dense_swiglu".into();
            }
        });
    }
    native_sidecars(&pair.reference, false);
    native_sidecars(&pair.candidate, true);
    let artifact = write_artifact(pair.root.path(), "inventory.json", &inventory());
    let config = json!({"schema_version":1,"scope":"metal_paired_q4_gate_up_shared_tail_v1",
        "inventory":artifact,"model_sha256":sha256(b"explicit fixture full-model pin"),"model_bytes":MODEL_BYTES});
    let config_path = pair.root.path().join("native-audit.json");
    fs::write(&config_path, serde_json::to_vec(&config).unwrap()).unwrap();
    let mut args = pair.args();
    args.comparison_shape = ComparisonShape::BatchedToBatched;
    args.require_bitwise_logits = true;
    args.native_work_audit = Some(config_path);
    (pair, args)
}

fn edit_native(arm: &fixture::Arm, wave: usize, edit: impl FnOnce(&mut Value)) {
    let index_path = arm.directory.join("native-work-manifest.json");
    let mut index: Value = serde_json::from_slice(&fs::read(&index_path).unwrap()).unwrap();
    let name = index["waves"][wave]["artifact"]["file"]
        .as_str()
        .unwrap()
        .to_owned();
    let mut record: Value =
        serde_json::from_slice(&fs::read(arm.directory.join(&name)).unwrap()).unwrap();
    edit(&mut record);
    index["waves"][wave]["artifact"] = write_artifact(&arm.directory, &name, &record);
    fs::write(index_path, serde_json::to_vec(&index).unwrap()).unwrap();
}
fn first_decode(pair: &Pair) -> usize {
    pair.candidate
        .manifest
        .waves
        .iter()
        .find(|w| w.kind == "decode")
        .unwrap()
        .wave_index
}

#[test]
fn native_work_audit_all_tail_widths_use_complete_eligible_and_control_populations() {
    for width in 5..=7 {
        let (_pair, args) = native_pair(width);
        let (report, code) = compare(&args);
        assert_eq!(code, 0, "{report:#}");
        let audit = &report["native_work_audit"];
        assert_eq!(audit["passed"], true);
        assert_eq!(audit["physical_decode_width"], width);
        assert_eq!(audit["decode_waves"], 2);
        assert_eq!(audit["hit_node_waves_per_arm"], 2);
        assert_eq!(audit["all_ffn_layer_count"], 2);
        assert_eq!(report["comparison_passed"], true);
        assert_eq!(report["summary"]["all"]["decision_count"], width * 3);
    }
}

#[test]
fn native_work_audit_missing_duplicate_cross_wave_or_wrong_physical_rows_fail() {
    for case in 0..6 {
        let (pair, args) = native_pair(5);
        let wave = first_decode(&pair);
        edit_native(&pair.candidate, wave, |record| match case {
            0 => {
                record["device"]["commands"].as_array_mut().unwrap().pop();
            }
            1 => {
                let mut duplicate = record["device"]["commands"][0].clone();
                duplicate["command_index"] = json!(2);
                record["device"]["commands"]
                    .as_array_mut()
                    .unwrap()
                    .push(duplicate);
            }
            2 => {
                record["submission_fingerprint"] = json!(sha256(b"another actual wave"));
            }
            3 => {
                record["device"]["commands"][0]["participant_start"] = json!(1);
            }
            4 => {
                record["device"]["replayed_segments"] = json!([{"logical_nodes":32}]);
            }
            _ => {
                record["device"]["commands"][0]["statistical_evidence"] = Value::Null;
            }
        });
        let (report, code) = compare(&args);
        assert_eq!(code, 2, "{report:#}");
        assert_eq!(report["native_work_audit"]["passed"], false);
    }
}

#[test]
fn native_work_audit_target_count_control_count_and_full_work_are_required() {
    for case in 0..3 {
        let (pair, args) = native_pair(7);
        let wave = first_decode(&pair);
        edit_native(&pair.candidate, wave, |record| {
            let c = &mut record["device"]["commands"][usize::from(case == 1)];
            if case == 2 {
                c["statistical_evidence"]["work"]["inner_work_units"] = json!(1);
            } else {
                let count = if case == 0 { 4 } else { 6 };
                c["compute_dispatch_count"] = json!(count);
                c["statistical_evidence"]["compute_dispatches"] = json!(count);
            }
        });
        let (report, code) = compare(&args);
        assert_eq!(code, 2, "{report:#}");
        assert_eq!(report["native_work_audit"]["passed"], false);
    }
}

#[test]
fn native_work_audit_pins_inventory_and_refuses_omitted_ffn_leaf() {
    for wrong_model in [true, false] {
        let (pair, args) = native_pair(6);
        let path = args.native_work_audit.as_ref().unwrap();
        let mut config: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        if wrong_model {
            config["model_sha256"] = json!(sha256(b"wrong model"));
        } else {
            let mut inv = inventory();
            let removed = inv["inventory"]["tensors"]
                .as_array_mut()
                .unwrap()
                .pop()
                .unwrap();
            let dtype = removed["dtype"].as_str().unwrap();
            let n = inv["inventory"]["tensor_counts_by_dtype"][dtype]
                .as_u64()
                .unwrap();
            inv["inventory"]["tensor_counts_by_dtype"][dtype] = json!(n - 1);
            inv["inventory"]["tensor_payload_bytes"] = json!(
                inv["inventory"]["tensor_payload_bytes"].as_u64().unwrap()
                    - removed["bytes"].as_u64().unwrap()
            );
            config["inventory"] = write_artifact(pair.root.path(), "inventory.json", &inv);
        }
        fs::write(path, serde_json::to_vec(&config).unwrap()).unwrap();
        let (report, code) = compare(&args);
        assert_eq!(code, 2, "{report:#}");
    }
}

#[test]
fn native_work_audit_requires_sidecar_integrity_and_preserves_default_and_quality_gates() {
    let (pair, mut args) = native_pair(5);
    let wave = first_decode(&pair);
    let path = pair
        .candidate
        .directory
        .join(format!("native-work-{wave:06}.json"));
    fs::write(&path, b"{}").unwrap();
    let (report, code) = compare(&args);
    assert_eq!(code, 2, "{report:#}");
    args.native_work_audit = None;
    let (report, code) = compare(&args);
    assert_eq!(code, 0, "{report:#}");
    assert!(report.get("native_work_audit").is_none());
    let (mut pair, args) = native_pair(5);
    pair.candidate
        .set_logits("owner-0", 1, &[0.0, 2.0, 1.0, -8.0]);
    native_sidecars(&pair.candidate, true);
    let (report, code) = compare(&args);
    assert_eq!(code, 1, "{report:#}");
    assert_eq!(report["native_work_audit"]["passed"], true);
    assert_eq!(report["comparison_passed"], false);
    assert_eq!(report["quality_passed"], false);
}
