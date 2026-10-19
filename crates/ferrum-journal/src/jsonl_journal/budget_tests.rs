use super::*;

fn path(label: &str) -> PathBuf {
    static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    std::env::temp_dir().join(format!(
        "ferrum-journal-budget-{label}-{}-{}.jsonl",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ))
}

fn typed_open_error<T: Serialize + Send + 'static>(
    result: io::Result<OwnedJsonlJournal<T>>,
) -> JsonlJournalError {
    let error = match result {
        Ok(_) => panic!("expected a typed journal open error"),
        Err(error) => error,
    };
    error
        .get_ref()
        .and_then(|source| source.downcast_ref::<JsonlJournalError>())
        .expect("open must retain the original typed cause")
        .clone()
}

#[test]
fn declared_journal_budget_flushes_large_batch_and_rejects_budget_minus_one_atomically() {
    let rows: Vec<String> = (0..1025)
        .map(|index| format!("{index}:{}", "x".repeat(4096)))
        .collect();
    let mut expected = Vec::new();
    let mut record_bytes = 0;
    for row in &rows {
        let start = expected.len();
        serde_json::to_writer(&mut expected, row).unwrap();
        expected.push(b'\n');
        record_bytes = record_bytes.max(expected.len() - start);
    }
    let defaults = ferrum_types::ObservabilityJournalConfig::default();
    assert!(rows.len() > defaults.max_batch_records);
    assert!(expected.len() > defaults.max_batch_bytes);
    let declared = ferrum_types::ObservabilityJournalConfig {
        queue_capacity: 2,
        payload_capacity_bytes: expected.len(),
        max_record_bytes: record_bytes,
        max_batch_bytes: expected.len(),
        max_batch_records: rows.len(),
    };
    declared.validate().unwrap();
    let output = path("large");
    let mut journal =
        OwnedJsonlJournal::open(&output, JsonlJournalOpenMode::Truncate, declared.into()).unwrap();
    journal.try_enqueue_batch(rows.clone()).unwrap();
    journal.close().unwrap().result().unwrap();
    assert_eq!(std::fs::read(&output).unwrap(), expected);
    let stats = journal.delivery_stats();
    assert_eq!(
        (stats.accepted, stats.written, stats.flushed),
        (rows.len() as u64, rows.len() as u64, rows.len() as u64)
    );
    assert_eq!(
        (stats.dropped, stats.failed, stats.pending_payload_bytes),
        (0, 0, 0)
    );
    assert!(stats.worker_finished);
    drop(journal);
    let _ = std::fs::remove_file(&output);

    for limited in [
        ferrum_types::ObservabilityJournalConfig {
            max_batch_bytes: expected.len() - 1,
            ..declared
        },
        ferrum_types::ObservabilityJournalConfig {
            max_batch_records: rows.len() - 1,
            ..declared
        },
    ] {
        let output = path("rejected");
        let mut journal =
            OwnedJsonlJournal::open(&output, JsonlJournalOpenMode::Truncate, limited.into())
                .unwrap();
        assert_eq!(
            journal.try_enqueue_batch(rows.clone()).unwrap_err().kind(),
            JsonlJournalErrorKind::Limit
        );
        journal.close().unwrap().result().unwrap();
        assert!(std::fs::read(&output).unwrap().is_empty());
        let stats = journal.delivery_stats();
        assert_eq!(
            (
                stats.accepted,
                stats.written,
                stats.flushed,
                stats.pending_payload_bytes
            ),
            (0, 0, 0, 0)
        );
        assert_eq!(stats.dropped, rows.len() as u64);
        drop(journal);
        let _ = std::fs::remove_file(output);
    }

    // Borrowed tool delivery uses the same declared batch boundary, without
    // splitting or truncating an existing artifact before validation succeeds.
    let output = path("tool");
    std::fs::write(&output, "existing\n").unwrap();
    let limited = ferrum_types::ObservabilityJournalConfig {
        max_batch_bytes: expected.len() - 1,
        ..declared
    };
    assert_eq!(
        write_jsonl_records_with_config(
            &output,
            JsonlJournalOpenMode::Truncate,
            &rows,
            limited.into()
        )
        .unwrap_err()
        .kind(),
        JsonlJournalErrorKind::Limit
    );
    assert_eq!(std::fs::read(&output).unwrap(), b"existing\n");
    write_jsonl_records_with_config(
        &output,
        JsonlJournalOpenMode::Truncate,
        &rows,
        declared.into(),
    )
    .unwrap();
    assert_eq!(std::fs::read(&output).unwrap(), expected);
    let _ = std::fs::remove_file(output);
}

#[test]
fn explicit_journal_budget_conflict_rejects_and_owner_scopes_use_original_writer() {
    let output = path("shared");
    let alias = output
        .parent()
        .unwrap()
        .join(".")
        .join(output.file_name().unwrap());
    let declared = JsonlJournalConfig {
        queue_capacity: 2,
        max_batch_records: 2048,
        ..Default::default()
    };
    let mut original = OwnedJsonlJournal::<serde_json::Value>::open(
        &output,
        JsonlJournalOpenMode::Truncate,
        declared,
    )
    .unwrap();
    assert_eq!(
        typed_open_error(OwnedJsonlJournal::<serde_json::Value>::open(
            &alias,
            JsonlJournalOpenMode::Append,
            declared
        ))
        .kind(),
        JsonlJournalErrorKind::Config
    );
    let mut sibling = original.owner.producer_scope().unwrap();
    let same = sibling.producer::<serde_json::Value>();
    let implicit = original.scope.producer::<serde_json::Value>();
    assert!(Arc::ptr_eq(&original.producer.inner, &same.inner));
    assert!(Arc::ptr_eq(&original.producer.inner, &implicit.inner));
    assert_eq!(implicit.config(), declared);
    let mismatch = JsonlJournalConfig {
        max_batch_records: 2049,
        ..declared
    };
    assert_eq!(
        typed_open_error(OwnedJsonlJournal::<serde_json::Value>::open(
            &alias,
            JsonlJournalOpenMode::Append,
            mismatch
        ))
        .kind(),
        JsonlJournalErrorKind::Config
    );
    let invalid = JsonlJournalConfig {
        queue_capacity: 0,
        ..declared
    };
    assert_eq!(
        typed_open_error(OwnedJsonlJournal::<serde_json::Value>::open(
            &alias,
            JsonlJournalOpenMode::Append,
            invalid
        ))
        .kind(),
        JsonlJournalErrorKind::Config
    );
    original
        .try_enqueue(serde_json::json!({"original": true}))
        .unwrap();
    write_jsonl_records_to(&implicit, &[serde_json::json!({"implicit": true})])
        .unwrap()
        .result()
        .unwrap();
    sibling.finish().result().unwrap();
    original.close().unwrap().result().unwrap();
    let rows: Vec<serde_json::Value> = std::fs::read_to_string(&output)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(
        rows,
        vec![
            serde_json::json!({"original": true}),
            serde_json::json!({"implicit": true})
        ]
    );
    assert_eq!(original.delivery_stats().failure_kind, None);
    drop((original, same, implicit));
    let _ = std::fs::remove_file(output);
}

#[test]
fn live_failed_writer_keeps_io_cause_and_independent_owners_are_contract_conflicts() {
    let output = path("io-conflict");
    drop(File::create(&output).unwrap());
    let config = JsonlJournalConfig {
        queue_capacity: 2,
        ..Default::default()
    };
    let mut original =
        super::delivery_tests::readonly_journal::<serde_json::Value>(&output, config);
    // Register this actual read-only File writer through the same canonical
    // lookup used by production. Its error is produced by the real flush.
    let canonical = normalize_jsonl_path(&output).unwrap();
    lock_journal_registry(JSONL_JOURNAL_REGISTRY.get_or_init(|| Mutex::new(HashMap::new())))
        .unwrap()
        .insert(canonical, Arc::downgrade(&original.producer.inner));
    original
        .try_enqueue(serde_json::json!({"io": "failure"}))
        .unwrap();
    let failure = original.flush().result().unwrap_err();
    assert_eq!(failure.kind(), JsonlJournalErrorKind::Worker);
    assert!(!original.producer.inner.closed.load(Ordering::Acquire));
    let other = JsonlJournalConfig {
        queue_capacity: 3,
        ..config
    };
    assert_eq!(
        typed_open_error(OwnedJsonlJournal::<serde_json::Value>::open(
            &output,
            JsonlJournalOpenMode::Append,
            other
        ))
        .kind(),
        JsonlJournalErrorKind::Config
    );
    assert_eq!(
        typed_open_error(OwnedJsonlJournal::<serde_json::Value>::open(
            &output,
            JsonlJournalOpenMode::Append,
            config
        ))
        .kind(),
        JsonlJournalErrorKind::Config
    );
    assert_eq!(
        typed_open_error(OwnedJsonlJournal::<serde_json::Value>::append(&output)).kind(),
        JsonlJournalErrorKind::Config
    );
    assert_eq!(original.flush().result().unwrap_err(), failure);
    assert_eq!(original.close().unwrap().result().unwrap_err(), failure);
    assert!(original.delivery_stats().worker_finished);
    assert_eq!(original.delivery_stats().pending_payload_bytes, 0);
    drop(original);
    let _ = std::fs::remove_file(output);
}

struct InvalidRecord;
impl Serialize for InvalidRecord {
    fn serialize<S: serde::Serializer>(&self, _: S) -> Result<S::Ok, S::Error> {
        Err(serde::ser::Error::custom(
            "actual serializer contract failure",
        ))
    }
}

#[test]
fn closed_failed_journal_append_preserves_cause_until_explicit_new_epoch() {
    let output = path("failed-append");
    let mut original = OwnedJsonlJournal::<InvalidRecord>::create(&output).unwrap();
    original.enqueue(InvalidRecord).unwrap();
    let failure = original.close().unwrap().result().unwrap_err();
    assert_eq!(failure.kind(), JsonlJournalErrorKind::Encoding);
    assert_eq!(
        typed_open_error(OwnedJsonlJournal::<serde_json::Value>::append(&output)),
        failure
    );
    assert_eq!(
        typed_open_error(OwnedJsonlJournal::<serde_json::Value>::open(
            &output,
            JsonlJournalOpenMode::Append,
            Default::default()
        )),
        failure
    );
    let mut fresh = OwnedJsonlJournal::<serde_json::Value>::open(
        &output,
        JsonlJournalOpenMode::Truncate,
        Default::default(),
    )
    .unwrap();
    assert!(!Arc::ptr_eq(
        &original.producer.inner,
        &fresh.producer.inner
    ));
    fresh
        .try_enqueue(serde_json::json!({"new_epoch": true}))
        .unwrap();
    fresh.close().unwrap().result().unwrap();
    assert_eq!(original.close().unwrap().result().unwrap_err(), failure);
    assert_eq!(
        std::fs::read_to_string(&output).unwrap(),
        "{\"new_epoch\":true}\n"
    );
    drop((original, fresh));
    // Without a live failure owner, file contents do not establish a journal
    // failure: tools may append after unrelated external records.
    std::fs::write(&output, "external history\n").unwrap();
    let mut unknown = OwnedJsonlJournal::<serde_json::Value>::append(&output).unwrap();
    unknown
        .try_enqueue(serde_json::json!({"appended": true}))
        .unwrap();
    unknown.close().unwrap().result().unwrap();
    assert_eq!(
        std::fs::read_to_string(&output).unwrap(),
        "external history\n{\"appended\":true}\n"
    );
    drop(unknown);
    let _ = std::fs::remove_file(output);
}
