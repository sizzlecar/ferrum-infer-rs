use super::*;
use std::collections::BTreeSet;

fn exchange(facts: &Value) -> Value {
    json!({"payload": {"type": "tool_exchange_committed", "assistant": {"content": [
        {"type": "tool_call", "call_id": "call-1", "name": "file_read", "arguments": {"path": DATA_FILE}}
    ]}, "tool": {"content": [{"type": "tool_result", "call_id": "call-1", "is_error": false,
        "result": {"path": DATA_FILE, "content": serde_json::to_string(facts).unwrap()}}]}}})
}

#[test]
fn exact_semantic_values_allow_prose_but_reject_substrings_and_wrong_values() {
    let expected = json!({"tracking_code": "cargo-012abc", "parcel_count": 17});
    journal::validate_answer(
        "The tracking code is cargo-012abc, and there are 17 parcels.",
        &expected,
    )
    .unwrap();
    journal::validate_answer(
        "{\"tracking_code\":\"cargo-012abc\",\"parcel_count\":17}",
        &expected,
    )
    .unwrap();
    for text in [
        "cargo-012abc has 117 parcels",
        "prefix-cargo-012abc has 17 parcels",
        "cargo-012abc-extra has 17 parcels",
        "cargo-012abd has 17 parcels",
    ] {
        assert!(journal::validate_answer(text, &expected).is_err(), "{text}");
    }
}

#[test]
fn tool_read_requires_native_identity_success_and_actual_file_content() {
    let facts = json!({"tracking_code": "cargo-abc", "parcel_count": 17});
    let mut snapshot = journal::Snapshot {
        session: vec![exchange(&facts)],
        ..Default::default()
    };
    snapshot.validate_file_read(DATA_FILE, &facts).unwrap();
    snapshot.session[0]["payload"]["tool"]["content"][0]["call_id"] = json!("different-call");
    assert!(snapshot.validate_file_read(DATA_FILE, &facts).is_err());
    snapshot.session[0] = exchange(&facts);
    snapshot.session[0]["payload"]["tool"]["content"][0]["is_error"] = json!(true);
    assert!(snapshot.validate_file_read(DATA_FILE, &facts).is_err());
    snapshot.session[0] = exchange(&facts);
    assert!(snapshot
        .validate_file_read(DATA_FILE, &json!({"tracking_code": "wrong"}))
        .is_err());
    snapshot.session = vec![
        json!({"payload": {"type": "run_output_committed", "message": "I called file_read and got cargo-abc, 17"}}),
    ];
    assert!(snapshot.validate_file_read(DATA_FILE, &facts).is_err());
}

fn delivery(text: &str) -> Value {
    json!({"event": {"run_id": "run-1", "payload": {"type": "delivery_committed", "delivery": {
        "final_response": {"media_type": "text/plain", "body": {"kind": "inline", "value": text}}
    }}}})
}

#[test]
fn final_delivery_rejects_partial_failure_and_duplicate_terminals() {
    let mut snapshot = journal::Snapshot::default();
    snapshot
        .runs
        .insert("run-1".into(), vec![delivery("cargo-abc")]);
    assert_eq!(
        snapshot.new_delivery(&BTreeSet::new()).unwrap(),
        "cargo-abc"
    );
    assert!(snapshot
        .new_delivery(&BTreeSet::from(["run-1".into()]))
        .is_err());
    snapshot
        .runs
        .get_mut("run-1")
        .unwrap()
        .push(delivery("cargo-abc"));
    assert!(snapshot.new_delivery(&BTreeSet::new()).is_err());
    snapshot.runs.insert(
        "run-1".into(),
        vec![json!({"event": {"run_id": "run-1", "payload": {"type": "run_incomplete"}}})],
    );
    assert!(snapshot.new_delivery(&BTreeSet::new()).is_err());
}

#[test]
fn actual_kv_resolution_and_failure_counters_are_required() {
    let format = ferrum_types::KvStorageFormat::Int8PerTokenHeadF32ScaleV1;
    let mut health = json!({"status": "healthy", "kv_storage": {
        "source": "resolved_model_plan", "requested": format, "selected": format
    }, "scheduler": {"failed_requests": 0}});
    health_evidence(&health, "int8").unwrap();
    assert!(health_evidence(&health, "fp16").is_err());
    health["scheduler"]["failed_requests"] = json!(1);
    assert!(health_evidence(&health, "int8").is_err());
    health["scheduler"]["failed_requests"] = Value::Null;
    assert!(health_evidence(&health, "int8").is_err());
}
