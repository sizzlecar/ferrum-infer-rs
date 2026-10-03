use super::*;

fn axis(byte: u8) -> Value {
    json!({"signature":([byte; 32]),"kind":0})
}
fn candidate(epoch: u64, algorithms: Vec<Value>) -> Value {
    json!({"model_version":epoch,"child":"external-child","capture":"external-capture",
        "universe":{"revision":1,"workload_domain":([9u8; 32]),"algorithms":algorithms}})
}
fn manifest(trace: &Trace, candidates: Vec<Value>) -> Value {
    json!({"schema":"ferrum.required-query-universe-input.v1","run_id":"fixture",
        "input_jsonl_sha256":format!("{:x}",Sha256::digest(trace.bytes())),"candidates":candidates})
}
fn compare(trace: &Trace, candidates: Vec<Value>, options: AuditOptions) -> Value {
    let audit = audit_required_queries(Cursor::new(trace.bytes()), options).unwrap();
    assert!(
        audit.integrity.trace_content_complete,
        "{:?}",
        audit.integrity.issues
    );
    let input = read_universe_comparison_input(Cursor::new(
        serde_json::to_vec(&manifest(trace, candidates)).unwrap(),
    ))
    .unwrap();
    serde_json::to_value(
        compare_query_universes(Cursor::new(trace.bytes()), &audit, input).unwrap(),
    )
    .unwrap()
}
fn trace_with_queries(epochs: &[u64]) -> Trace {
    let mut trace = Trace::new();
    for (i, epoch) in epochs.iter().enumerate() {
        let tx = i as u64 + 1;
        trace.begin(tx);
        trace.snapshot(tx, *epoch, 1);
        trace.attempt(tx, 1, "Search", 1);
        trace.query(
            tx,
            1,
            json!({"kind":"structured_unknown","reason":"WrongDomain"}),
        );
        let index = trace.records.len() - 2;
        trace.records[index]["data"]["algorithm_axes"] = json!([axis(2)]);
        trace.records[index]["data"]["identity"] = json!({"physical_domain_signature":([9u8; 32])});
        trace.records[index]["data"]["prebound_algorithm_universe"] = Value::Null;
        trace.end_attempt(tx, 1, 1, 1);
        trace.end(tx, "cost_unavailable");
    }
    trace.footer();
    trace
}
fn query_data(trace: &mut Trace, tx: u64) -> &mut Value {
    &mut trace
        .records
        .iter_mut()
        .find(|r| r["event"] == "query_constructed" && r["transaction"] == tx)
        .unwrap()["data"]
}

#[test]
fn universe_comparison_checks_all_same_epoch_candidates_and_preserves_original_audit() {
    let mut trace = trace_with_queries(&[4]);
    let old = trace.audit();
    let report = compare(
        &trace,
        vec![
            candidate(3, vec![axis(2)]),
            candidate(4, vec![axis(1)]),
            candidate(4, vec![axis(1), axis(2)]),
        ],
        AuditOptions::default(),
    );
    assert_eq!(report["constructed_queries"], 1);
    assert_eq!(report["candidate_constructed_comparisons"][0], json!({}));
    assert_eq!(
        report["candidate_constructed_comparisons"][1]["missing_algorithms"],
        1
    );
    assert_eq!(
        report["candidate_constructed_comparisons"][2]["no_missing_algorithms"],
        1
    );
    let example = &report["examples"][0];
    assert_eq!(example["transaction"], 1);
    assert_eq!(example["attempt"], 1);
    assert_eq!(example["alternative"], 0);
    assert_eq!(example["snapshot_model_version"], 4);
    assert!(example["ordinal"].as_u64().unwrap() > 0);
    assert!(example["line"].as_u64().unwrap() > 0);
    assert_eq!(example["missing_algorithms"], json!([axis(2)]));
    assert!(!old.integrity.successful_close_attested);
    assert_eq!(
        serde_json::to_value(old).unwrap(),
        serde_json::to_value(trace.audit()).unwrap()
    );
    query_data(&mut trace, 1)["prebound_algorithm_universe"] =
        report["candidates"][2]["universe_signature"].clone();
    let bound = compare(
        &trace,
        vec![
            candidate(4, vec![axis(1)]),
            candidate(4, vec![axis(1), axis(2)]),
        ],
        AuditOptions::default(),
    );
    assert_eq!(
        bound["candidate_constructed_comparisons"][0]["prebound_universe_mismatch"],
        1
    );
    assert_eq!(
        bound["candidate_constructed_comparisons"][1]["no_missing_algorithms"],
        1
    );
}

#[test]
fn universe_comparison_old_missing_fields_domain_prebinding_and_epoch_stay_distinct() {
    let mut trace = trace_with_queries(&[4, 4, 4, 4, 4, 5, 4]);
    query_data(&mut trace, 1)
        .as_object_mut()
        .unwrap()
        .remove("algorithm_axes");
    query_data(&mut trace, 2)["identity"] = Value::Null;
    query_data(&mut trace, 3)
        .as_object_mut()
        .unwrap()
        .remove("prebound_algorithm_universe");
    query_data(&mut trace, 4)["identity"]["physical_domain_signature"] = json!(([8u8; 32]));
    query_data(&mut trace, 5)["prebound_algorithm_universe"] = json!(([7u8; 32]));
    query_data(&mut trace, 7)["input_unknown"] = json!("WrongDomain");
    let report = compare(
        &trace,
        vec![candidate(4, vec![axis(2)])],
        AuditOptions::default(),
    );
    for reason in [
        "missing_algorithm_roster",
        "missing_physical_domain",
        "missing_prebound_evidence",
        "no_same_epoch_candidate_evidence",
        "input_unknown",
    ] {
        assert_eq!(report["query_evidence"][reason], 1, "{reason}");
    }
    assert_eq!(
        report["candidate_constructed_comparisons"][0],
        json!({"physical_domain_mismatch":1,"prebound_universe_mismatch":1})
    );
}

#[test]
fn universe_comparison_cuts_bind_complete_transactions_and_attempt_snapshot() {
    let mut trace = trace_with_queries(&[4, 5]);
    // A later snapshot cannot relabel the already constructed query's attempt.
    let at = trace
        .records
        .iter()
        .position(|r| r["event"] == "query_constructed")
        .unwrap();
    let mut record = trace.records[at - 2].clone();
    record["data"]["cost_model_version"] = json!(99);
    trace.records.insert(at, record);
    trace.ordinal += 1;
    let mut ordinal = 0u64;
    for r in &mut trace.records {
        if r.get("ordinal").is_some() {
            ordinal += 1;
            r["ordinal"] = json!(ordinal);
            r["retained_ordinal"] = json!(ordinal);
        }
    }
    trace.records.last_mut().unwrap()["statistics"]["offered"] = json!(ordinal);
    trace.records.last_mut().unwrap()["statistics"]["accepted"] = json!(ordinal);
    trace.records.last_mut().unwrap()["statistics"]["written"] = json!(ordinal);
    let options = AuditOptions {
        through_transaction: Some(1),
        ..AuditOptions::default()
    };
    let report = compare(
        &trace,
        vec![candidate(4, vec![axis(2)]), candidate(99, vec![axis(2)])],
        options,
    );
    assert_eq!(report["constructed_queries"], 1);
    assert_eq!(
        report["candidate_constructed_comparisons"][0]["no_missing_algorithms"],
        1
    );
    assert_eq!(report["candidate_constructed_comparisons"][1], json!({}));
    let cut = trace
        .records
        .iter()
        .find(|r| r["event"] == "transaction_end")
        .unwrap()["ordinal"]
        .as_u64()
        .unwrap();
    let report = compare(
        &trace,
        vec![candidate(4, vec![axis(2)])],
        AuditOptions {
            before_ordinal: Some(cut),
            ..AuditOptions::default()
        },
    );
    assert_eq!(report["constructed_queries"], 0);
}

#[test]
fn universe_comparison_rejects_invalid_declarations_and_bounds_examples() {
    let trace = trace_with_queries(&[4]);
    let valid = candidate(4, vec![axis(1), axis(2)]);
    let mut bads = vec![];
    for (field, value) in [
        ("revision", json!(2)),
        ("workload_domain", json!(([0u8; 32]))),
        ("algorithms", json!([])),
        ("algorithms", json!([axis(2), axis(1)])),
        ("algorithms", json!([axis(1), axis(1)])),
        ("algorithms", json!([{"signature":([1u8; 32]),"kind":6}])),
    ] {
        let mut bad = valid.clone();
        bad["universe"][field] = value;
        bads.push(bad);
    }
    let mut bad = valid.clone();
    bad["universe_signature"] = json!(([1u8; 32]));
    bads.push(bad);
    for bad in bads {
        assert!(read_universe_comparison_input(Cursor::new(
            serde_json::to_vec(&manifest(&trace, vec![bad])).unwrap()
        ))
        .is_err());
    }
    assert!(read_universe_comparison_input(Cursor::new(
        serde_json::to_vec(&manifest(&trace, vec![valid.clone(); 257])).unwrap()
    ))
    .is_err());
    let axes = (1u16..=373)
        .map(|n| {
            let mut signature = [0u8; 32];
            signature[..2].copy_from_slice(&n.to_be_bytes());
            json!({"signature":signature,"kind":0})
        })
        .collect::<Vec<_>>();
    for (count, accepted) in [(372, true), (373, false)] {
        let bytes = serde_json::to_vec(&manifest(
            &trace,
            vec![candidate(4, axes[..count].to_vec())],
        ))
        .unwrap();
        assert_eq!(
            read_universe_comparison_input(Cursor::new(bytes)).is_ok(),
            accepted
        );
    }
    let report = compare(&trace, vec![valid; 256], AuditOptions::default());
    assert_eq!(
        report["candidate_constructed_comparisons"]
            .as_array()
            .unwrap()
            .len(),
        256
    );
    assert_eq!(report["examples"].as_array().unwrap().len(), 128);
    assert_eq!(report["examples_truncated"], true);
    let mut replayed = candidate(4, vec![axis(1), axis(2)]);
    replayed["universe_signature"] = report["candidates"][0]["universe_signature"].clone();
    assert_eq!(
        compare(&trace, vec![replayed], AuditOptions::default())
            ["candidate_constructed_comparisons"][0]["no_missing_algorithms"],
        1
    );
}

#[test]
fn universe_comparison_fail_closed_on_wrong_binding_changed_trace_and_invalid_roster() {
    let mut trace = trace_with_queries(&[4]);
    let audit = trace.audit();
    let original = manifest(&trace, vec![candidate(4, vec![axis(2)])]);
    let input = read_universe_comparison_input(Cursor::new(serde_json::to_vec(&original).unwrap()))
        .unwrap();
    query_data(&mut trace, 1)["algorithm_axes"] = json!([axis(1)]);
    assert!(
        compare_query_universes(Cursor::new(trace.bytes()), &audit, input)
            .unwrap_err()
            .to_string()
            .contains("changed")
    );
    let mut bad = original.clone();
    bad["run_id"] = json!("another-run");
    let input =
        read_universe_comparison_input(Cursor::new(serde_json::to_vec(&bad).unwrap())).unwrap();
    assert!(
        compare_query_universes(Cursor::new(trace.bytes()), &audit, input)
            .unwrap_err()
            .to_string()
            .contains("run or original trace hash")
    );
    query_data(&mut trace, 1)["algorithm_axes"] = json!([axis(2), axis(1)]);
    let audit = trace.audit(); // Legacy/default audit deliberately ignores extensions.
    assert!(audit.integrity.trace_content_complete);
    let input = read_universe_comparison_input(Cursor::new(
        serde_json::to_vec(&manifest(&trace, vec![candidate(4, vec![axis(2)])])).unwrap(),
    ))
    .unwrap();
    assert!(
        compare_query_universes(Cursor::new(trace.bytes()), &audit, input)
            .unwrap_err()
            .to_string()
            .contains("unsorted")
    );
    let mut audit = audit;
    audit.integrity.trace_content_complete = false;
    let input = read_universe_comparison_input(Cursor::new(
        serde_json::to_vec(&manifest(&trace, vec![])).unwrap(),
    ))
    .unwrap();
    assert!(
        compare_query_universes(Cursor::new(trace.bytes()), &audit, input)
            .unwrap_err()
            .to_string()
            .contains("complete original")
    );
}

#[test]
fn universe_comparison_counts_constructed_unqueried_alternatives_explicitly() {
    let mut trace = Trace::new();
    trace.begin(1);
    trace.snapshot(1, 4, 1);
    trace.attempt(1, 1, "Search", 1);
    trace.event(1,"query_constructed",json!({"attempt":1,"alternative":0,"input_unknown":null,"demand_error":null,
        "identity":{"physical_domain_signature":([9u8;32])},"algorithm_axes":[axis(2)],"prebound_algorithm_universe":null}));
    trace.end_attempt(1, 1, 1, 0);
    trace.end(1, "planner_budget_exhausted");
    trace.footer();
    let report = compare(
        &trace,
        vec![candidate(4, vec![axis(2)])],
        AuditOptions::default(),
    );
    assert_eq!(report["constructed_queries"], 1);
    assert_eq!(
        report["candidate_constructed_comparisons"][0]["no_missing_algorithms"],
        1
    );
    assert!(trace.records.iter().all(|r| r["event"] != "query_lookup"));
}

#[test]
fn universe_comparison_keeps_different_missing_rosters_for_one_candidate() {
    let mut trace = trace_with_queries(&[4, 4]);
    query_data(&mut trace, 2)["algorithm_axes"] = json!([axis(3)]);
    let report = compare(
        &trace,
        vec![candidate(4, vec![axis(1)])],
        AuditOptions::default(),
    );
    assert_eq!(
        report["candidate_constructed_comparisons"][0]["missing_algorithms"],
        2
    );
    assert_eq!(report["examples"].as_array().unwrap().len(), 2);
    assert_eq!(report["examples"][0]["transaction"], 1);
    assert_eq!(
        report["examples"][0]["missing_algorithms"],
        json!([axis(2)])
    );
    assert_eq!(report["examples"][1]["transaction"], 2);
    assert_eq!(
        report["examples"][1]["missing_algorithms"],
        json!([axis(3)])
    );
}
