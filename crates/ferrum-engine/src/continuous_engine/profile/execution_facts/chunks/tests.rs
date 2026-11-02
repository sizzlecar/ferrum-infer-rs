use super::*;

fn event(entrypoint: ProfileEntrypoint, rows: usize, participants: usize) -> FerrumProfileEvent {
    FerrumProfileEvent {
        schema_version: OBSERVABILITY_PROFILE_SCHEMA_VERSION,
        ts_unix_nanos: 1,
        event_id: "evt-execution-facts-completion".to_owned(),
        request_id: "request.startup".to_owned(),
        correlation_id: Some("submission".to_owned()),
        entrypoint,
        backend: "actual".to_owned(),
        runtime_preset_hash: ENGINE_RUNTIME_TRACE_PRESET_HASH.to_owned(),
        phase: "vnext.execution_facts".to_owned(),
        event_kind: ProfileEventKind::Instant,
        timestamp: chrono::DateTime::from_timestamp_nanos(1),
        status: ProfileStatus::Ok,
        model: Some("fixture".to_owned()),
        duration_us: None,
        memory: None,
        resource: None,
        error: None,
        replay: None,
        shape: BTreeMap::from([("node_count".to_owned(), json!(rows))]),
        backend_detail: None,
        attributes: BTreeMap::from([
            ("completion_fingerprint".to_owned(), json!("completion")),
            ("submission_fingerprint".to_owned(), json!("submission")),
            (
                "submitted_node_work".to_owned(),
                json!((0..rows)
                    .map(|node| json!({
                        "node_index": node,
                        "participants": (0..participants).map(|participant| json!({
                            "participant_index": participant,
                            "request_id": format!("request.{participant}"),
                            "source_tokens": {"start": 0, "end": 1},
                            "identity/escaped~": "文本".repeat(40),
                        })).collect::<Vec<_>>(),
                    }))
                    .collect::<Vec<_>>()),
            ),
            (
                "device_execution".to_owned(),
                json!({
                    "batch_command_count": rows,
                    "commands": (0..rows).map(|index| json!({"command":index})).collect::<Vec<_>>(),
                    "projection_rows": [],
                }),
            ),
        ]),
    }
}

fn reassemble(records: &[FerrumProfileEvent]) -> Value {
    let mut restored = Value::Null;
    for (index, record) in records.iter().enumerate() {
        let fragment = &record.attributes["execution_facts_fragment"];
        assert_eq!(fragment["index"], index);
        assert_eq!(fragment["count"], records.len());
        assert_eq!(
            record.shape,
            BTreeMap::from([("fragment".to_owned(), json!(true))]),
            "no repeated work counts in envelopes"
        );
        for piece in fragment["pieces"].as_array().unwrap() {
            let path = piece["path"].as_str().unwrap();
            let value = piece["value"].clone();
            if path.is_empty() {
                restored = value;
            } else {
                let (parent, leaf) = path.rsplit_once('/').unwrap();
                let leaf = leaf.replace("~1", "/").replace("~0", "~");
                match restored
                    .pointer_mut(parent)
                    .expect("container precedes child")
                {
                    Value::Object(object) => {
                        assert!(object.insert(leaf, value).is_none());
                    }
                    Value::Array(array) => {
                        assert_eq!(leaf.parse::<usize>().unwrap(), array.len());
                        array.push(value);
                    }
                    _ => panic!("piece parent is not a container"),
                }
            }
        }
    }
    restored
}

#[test]
fn execution_facts_preserve_normal_wire_and_losslessly_split_nested_values() {
    let normal = event(ProfileEntrypoint::Run, 1, 1);
    let expected = serde_json::to_vec(&normal).unwrap();
    let records = bounded_records(normal, 1024 * 1024).unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(serde_json::to_vec(&records[0]).unwrap(), expected);

    let large = event(ProfileEntrypoint::Serve, 4, 32);
    let expected = serde_json::to_value(&large).unwrap();
    let records = bounded_records(large, 4096).unwrap();
    assert!(records.len() > 1);
    for record in &records {
        record.validate().unwrap();
        assert!(serde_json::to_vec(record).unwrap().len() + 1 <= 4096);
    }
    assert_eq!(reassemble(&records), expected);

    let mut indivisible = event(ProfileEntrypoint::Run, 1, 1);
    indivisible
        .attributes
        .insert("large_scalar".to_owned(), json!("x".repeat(8192)));
    assert!(bounded_records(indivisible, 4096).is_err());
}

#[test]
fn execution_facts_run_and_serve_journals_keep_large_fact_sets_complete() {
    for entrypoint in [ProfileEntrypoint::Run, ProfileEntrypoint::Serve] {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("facts.jsonl");
        let config = ferrum_types::ObservabilityJournalConfig::default();
        let journal = TestTraceJournal::create(path.clone(), config).unwrap();
        let mut engine = EngineConfig::default();
        engine.runtime.profile_detail = ObservabilityProfileDetail::Debug;
        let sink = VNextProfileExecutionEventSink::new(journal.clone(), entrypoint, &engine);
        let original = event(entrypoint, 256, 64);
        assert!(encoded_bytes(&original).unwrap() > config.max_batch_bytes);
        let expected = serde_json::to_value(&original).unwrap();
        sink.enqueue_execution_facts_event(original).unwrap();
        journal.flush().unwrap();
        let stats = journal.delivery_stats();
        assert_eq!((stats.dropped, stats.failed), (0, 0));
        assert_eq!(stats.accepted, stats.flushed);
        drop(sink);
        journal.close().unwrap();
        let text = std::fs::read_to_string(path).unwrap();
        let records: Vec<FerrumProfileEvent> = text
            .lines()
            .map(|line| {
                assert!(line.len() + 1 <= config.max_record_bytes);
                serde_json::from_str(line).unwrap()
            })
            .collect();
        assert_eq!(reassemble(&records), expected);
    }
}
