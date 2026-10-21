use super::*;

#[test]
fn sharegpt_heldout_excludes_ids_and_normalized_prompts_before_sampling() {
    let records = [
        record("seen-id", 4, 4),
        record("new-id-same-text", 5, 4),
        record("fresh", 6, 4),
    ];
    let mut fixture = Fixture::records(&records);
    let mut same_text = record("other-id", 5, 4);
    same_text["conversations"][0]["value"] = json!("\ta\u{2003}a  a\na a  ");
    let history = fixture._directory.path().join("history.jsonl");
    let bytes = format!("{}\n{}\n", json!({"id":"seen-id"}), same_text);
    std::fs::write(&history, &bytes).unwrap();
    fixture.command.sharegpt.sharegpt_exclude_path = vec![history.clone()];
    let (prompts, evidence) = load(&fixture.command, &fixture.tokenizer, 0, 1).unwrap();
    assert_eq!(prompts[0].text, text(6));
    assert_eq!(evidence.counts.excluded_by_history, 2);
    assert_eq!(evidence.counts.eligible, 1);
    let history_evidence = evidence.exclusions.unwrap();
    assert_eq!(
        history_evidence.sources[0].source_sha256,
        sha256_hex(bytes.as_bytes())
    );
    assert_eq!(history_evidence.sources[0].records, 2);
    assert_eq!(history_evidence.unique_original_ids, 2);
    assert_eq!(history_evidence.unique_normalized_prompts, 1);
    assert!(load(&fixture.command, &fixture.tokenizer, 0, 2).is_err());
}

#[test]
fn sharegpt_heldout_rejects_unreadable_or_unusable_history() {
    let mut fixture = Fixture::records(&[record("fresh", 4, 4)]);
    let history = fixture._directory.path().join("history.jsonl");
    fixture.command.sharegpt.sharegpt_exclude_path = vec![history.clone()];
    assert!(load(&fixture.command, &fixture.tokenizer, 0, 1).is_err());
    for bytes in ["bad-json\n", "{}\n"] {
        std::fs::write(&history, bytes).unwrap();
        assert!(load(&fixture.command, &fixture.tokenizer, 0, 1).is_err());
    }
    fixture.command.dataset = "synthetic".into();
    assert!(validate(&fixture.command).is_err());
}

#[test]
fn sharegpt_heldout_cache_is_bound_to_history_content() {
    let records = [record("first", 4, 4), record("second", 5, 4)];
    let mut fixture = Fixture::records(&records);
    fixture.command.num_prompts = 1;
    fixture.command.warmup_requests = 0;
    fixture.command.n_repeats = 1;
    fixture.command.sharegpt.sharegpt_prepared_cache_dir =
        Some(fixture._directory.path().join("cache"));
    let history = fixture._directory.path().join("history.json");
    fixture.command.sharegpt.sharegpt_exclude_path = vec![history.clone()];
    std::fs::write(&history, serde_json::to_vec(&records[..1]).unwrap()).unwrap();
    let first = prepare(&fixture.command).unwrap().unwrap();
    assert_eq!(first[0].prompts[0].text, text(5));
    let hit = prepare(&fixture.command).unwrap().unwrap();
    assert_eq!(first[0].evidence, hit[0].evidence);
    std::fs::write(&history, serde_json::to_vec(&records[1..]).unwrap()).unwrap();
    let second = prepare(&fixture.command).unwrap().unwrap();
    assert_eq!(second[0].prompts[0].text, text(4));
    assert_ne!(first[0].evidence.exclusions, second[0].evidence.exclusions);
    let mut combined = Some(first[0].evidence.clone());
    assert!(append_evidence(&mut combined, second[0].evidence.clone()).is_err());
}
