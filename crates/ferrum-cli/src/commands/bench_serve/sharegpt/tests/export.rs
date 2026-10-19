use super::super::super::{execute, validate_command};
use super::super::export as pool_export;
use super::*;
use crate::config::CliConfig;
use std::io::Write;

fn fixture() -> (Fixture, Vec<Value>) {
    let records = (0..40)
        .map(|index| {
            let mut value = record(&format!("original-{index}"), 4 + index % 7, 4 + index % 11);
            value["metadata"] = json!({
                "nested": [null, {"original_index": index, "u64": u64::MAX - 7}],
                "untouched": "later-turn metadata"
            });
            value
        })
        .collect::<Vec<_>>();
    let mut fixture = Fixture::records(&records);
    fixture.command.num_prompts = 2;
    fixture.command.warmup_requests = 1;
    fixture.command.sharegpt.sharegpt_selection_pool_size = Some(7);
    fixture.command.sharegpt.sharegpt_fixed_output_tokens = Some(11);
    fixture
        .command
        .sharegpt
        .sharegpt_chat_template_reserve_tokens = 3;
    fixture.command.sharegpt.export_sharegpt_pool = Some(fixture._directory.path().join("export"));
    fixture.command.sharegpt.sharegpt_source_sha256 = Some(sha256_hex(
        &std::fs::read(fixture.command.sharegpt_path.as_ref().unwrap()).unwrap(),
    ));
    fixture.command.enable_thinking = Some(false);
    fixture.command.sampling.temperature = 0.7;
    fixture.command.sampling.top_p = Some(0.9);
    fixture.command.sampling.repetition_penalty = Some(1.0);
    fixture.command.sampling.sampling_seed = Some(u64::MAX - 3);
    (fixture, records)
}

fn target(fixture: &Fixture) -> &std::path::Path {
    fixture
        .command
        .sharegpt
        .export_sharegpt_pool
        .as_deref()
        .unwrap()
}

fn read_export(fixture: &Fixture) -> (Vec<Value>, Value) {
    let values =
        serde_json::from_slice(&std::fs::read(target(fixture).join("pool.json")).unwrap()).unwrap();
    let receipt =
        serde_json::from_slice(&std::fs::read(target(fixture).join("pool.receipt.json")).unwrap())
            .unwrap();
    (values, receipt)
}

#[test]
fn sharegpt_pool_export_preserves_original_records_provenance_and_u64_policy() {
    let (fixture, original) = fixture();
    let mut parent_command = fixture.command.clone();
    parent_command.sharegpt.export_sharegpt_pool = None;
    parent_command.sharegpt.sharegpt_source_sha256 = None;
    parent_command.num_prompts = 7;
    parent_command.warmup_requests = 0;
    let (_, parent_evidence) = load(&parent_command, &fixture.tokenizer, 0, 7).unwrap();
    pool_export::run(&fixture.command).unwrap();
    let (pool, receipt) = read_export(&fixture);
    assert_eq!(
        pool.len(),
        7,
        "export the reservoir, not the three-request prefix"
    );
    assert_eq!(receipt["selection_semantics"], "new_fixed_derived_pool");
    assert_eq!(
        receipt["serialization_order"],
        "parent_source_record_index_ascending"
    );
    assert_eq!(
        receipt["parent"],
        serde_json::to_value(&parent_evidence).unwrap()
    );
    assert_eq!(receipt["parent"]["prompt_seed"].as_u64(), Some(42));
    assert_eq!(
        receipt["parent"]["repeats"][0]["rng_seed"].as_u64(),
        Some(repeat_seed(42, 0))
    );
    assert_eq!(
        receipt["request_policy"]["sampling"]["seed"].as_u64(),
        Some(u64::MAX - 3)
    );
    let sampling: ferrum_bench_core::env::HttpRequestSampling =
        serde_json::from_value(receipt["request_policy"]["sampling"].clone()).unwrap();
    assert_eq!(sampling, fixture.command.sampling.request_sampling());
    assert_eq!(receipt["request_policy"]["ignore_eos"], true);
    assert_eq!(receipt["request_policy"]["enable_thinking"], false);
    assert_eq!(receipt["parent"]["filter"]["fixed_output_tokens"], 11);
    assert_eq!(
        receipt["parent"]["filter"]["chat_template_reserve_tokens"],
        3
    );
    assert_eq!(
        receipt["tokenizer"]["sha256"],
        parent_evidence.tokenizer_sha256
    );
    let normalized =
        serde_json::to_value(unbounded_tokenizer(&fixture.tokenizer).unwrap()).unwrap();
    assert_eq!(
        receipt["tokenizer"]["effective_tokenizer_sha256"],
        sha256_hex(&serde_json::to_vec(&normalized).unwrap())
    );
    let mut expected_indices = parent_evidence.repeats[0]
        .samples
        .iter()
        .map(|sample| sample.source_record_index)
        .collect::<Vec<_>>();
    expected_indices.sort_unstable();
    let mappings = receipt["records"].as_array().unwrap();
    assert_eq!(mappings.len(), expected_indices.len());
    for (child, &index) in expected_indices.iter().enumerate() {
        assert_eq!(
            pool[child], original[index as usize],
            "all conversations and metadata must survive"
        );
        assert_eq!(mappings[child]["child_record_index"], child);
        assert_eq!(mappings[child]["parent_source_record_index"], index);
        let sample = parent_evidence.repeats[0]
            .samples
            .iter()
            .find(|sample| sample.source_record_index == index)
            .unwrap();
        assert_eq!(
            mappings[child]["original_id"],
            serde_json::to_value(&sample.original_id).unwrap()
        );
        assert_eq!(mappings[child]["prompt_sha256"], sample.prompt_sha256);
        assert_eq!(mappings[child]["assistant_sha256"], sample.assistant_sha256);
        assert_eq!(mappings[child]["input_tokens"], sample.input_tokens);
        assert_eq!(
            mappings[child]["reference_output_tokens"],
            sample.reference_output_tokens
        );
        assert_eq!(mappings[child]["requested_output_tokens"], 11);
    }
    assert!(
        expected_indices
            .windows(2)
            .any(|indices| indices[1] > indices[0] + 1),
        "fixture must test a selected subset, not a source prefix"
    );
    let bytes = std::fs::read(target(&fixture).join("pool.json")).unwrap();
    assert_eq!(receipt["child"]["source_sha256"], sha256_hex(&bytes));
    assert_eq!(receipt["child"]["source_bytes"], bytes.len());
    assert_eq!(receipt["child"]["records"], 7);
    assert_eq!(std::fs::read_dir(target(&fixture)).unwrap().count(), 2);
}

#[test]
fn sharegpt_pool_export_child_reselection_keeps_its_own_identity_across_concurrency() {
    let (fixture, _) = fixture();
    pool_export::run(&fixture.command).unwrap();
    let (pool, receipt) = read_export(&fixture);
    let mut child_command = fixture.command.clone();
    child_command.sharegpt.export_sharegpt_pool = None;
    child_command.sharegpt.sharegpt_source_sha256 = None;
    child_command.sharegpt_path = Some(target(&fixture).join("pool.json"));
    child_command.concurrency = 4;
    let first = prepare(&child_command).unwrap().unwrap();
    child_command.concurrency = 8;
    child_command.model = "other-server-artifact".into();
    let second = prepare(&child_command).unwrap().unwrap();
    assert_eq!(first[0].evidence, second[0].evidence);
    assert_eq!(
        first[0].evidence.source_sha256,
        receipt["child"]["source_sha256"]
    );
    assert_ne!(
        first[0].evidence.source_sha256,
        receipt["parent"]["source_sha256"]
    );
    assert_eq!(first[0].evidence.counts.records, 7);
    assert_eq!(first[0].evidence.counts.eligible, 7);
    assert_eq!(first[0].prompts.len(), 3);
    for ((a, b), sample) in first[0]
        .prompts
        .iter()
        .zip(&second[0].prompts)
        .zip(&first[0].evidence.repeats[0].samples)
    {
        assert_eq!(
            (&a.text, a.input_tokens, &a.sha256, a.output_budget),
            (&b.text, b.input_tokens, &b.sha256, b.output_budget)
        );
        let original_child = &pool[sample.source_record_index as usize];
        assert_eq!(a.text, original_child["conversations"][0]["value"]);
        assert_eq!(
            a.input_tokens,
            token_length(&fixture.tokenizer, &a.text).unwrap()
        );
        assert_eq!(a.output_budget, Some(11));
    }
    assert_eq!(
        first[0].evidence.repeats[0].samples[0].phase,
        BenchmarkPhase::Warmup
    );
    assert_eq!(
        first[0].evidence.repeats[0].samples[1].phase,
        BenchmarkPhase::Measured
    );
}

#[tokio::test]
async fn sharegpt_pool_export_execute_returns_before_http_or_benchmark_report() {
    let (mut fixture, _) = fixture();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    fixture.command.base_url = format!("http://{}", listener.local_addr().unwrap());
    let report = fixture._directory.path().join("not-a-benchmark.json");
    fixture.command.out = Some(report.clone());
    execute(fixture.command.clone(), CliConfig::default())
        .await
        .unwrap();
    assert_eq!(
        listener.accept().unwrap_err().kind(),
        std::io::ErrorKind::WouldBlock
    );
    assert!(
        !report.exists(),
        "an export does not emit benchmark success/timing evidence"
    );
    assert_eq!(read_export(&fixture).0.len(), 7);
}

#[test]
fn sharegpt_pool_export_rejects_changed_parent_and_removes_partial_output() {
    let (fixture, _) = fixture();
    let selected = pool_export::select_pool(&fixture.command).unwrap();
    // The same selected JSON records still match; only the complete raw source
    // hash changes. This catches implementations checking selected pairs only.
    let mut source = std::fs::OpenOptions::new()
        .append(true)
        .open(fixture.command.sharegpt_path.as_ref().unwrap())
        .unwrap();
    source.write_all(b" \n").unwrap();
    drop(source);
    let error = pool_export::publish_pool(target(&fixture), &selected).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("changed between selection and export"),
        "{error}"
    );
    assert!(
        !target(&fixture).exists(),
        "partial pool and receipt must be removed"
    );
}

#[test]
fn sharegpt_pool_export_pin_io_and_existing_destination_fail_without_publishing() {
    let (mut fixture, _) = fixture();
    let correct_pin = fixture.command.sharegpt.sharegpt_source_sha256.clone();
    fixture.command.sharegpt.sharegpt_source_sha256 = Some("0".repeat(64));
    let error = pool_export::run(&fixture.command).unwrap_err();
    assert!(error.to_string().contains("differs from the required pin"));
    assert!(!target(&fixture).exists());
    fixture.command.sharegpt.sharegpt_source_sha256 = correct_pin;
    let selected = pool_export::select_pool(&fixture.command).unwrap();
    std::fs::remove_file(fixture.command.sharegpt_path.as_ref().unwrap()).unwrap();
    let error = pool_export::publish_pool(target(&fixture), &selected).unwrap_err();
    assert!(
        error.to_string().contains("reopen parent source"),
        "{error}"
    );
    assert!(!target(&fixture).exists());
    std::fs::create_dir(target(&fixture)).unwrap();
    let sentinel = target(&fixture).join("original.txt");
    std::fs::write(&sentinel, b"owned before export").unwrap();
    let error = pool_export::run(&fixture.command).unwrap_err();
    assert!(error.to_string().contains("already exists"), "{error}");
    assert_eq!(std::fs::read(sentinel).unwrap(), b"owned before export");
}

#[test]
fn sharegpt_pool_export_requires_explicit_pinned_offline_settings() {
    let (fixture, _) = fixture();
    let mutations: &[fn(&mut BenchServeCommand)] = &[
        |cmd| cmd.dataset = "random".into(),
        |cmd| cmd.sharegpt_path = None,
        |cmd| cmd.seed = None,
        |cmd| cmd.n_repeats = 2,
        |cmd| cmd.sharegpt.sharegpt_selection_pool_size = None,
        |cmd| cmd.sharegpt.sharegpt_selection_pool_size = Some(0),
        |cmd| cmd.sharegpt.sharegpt_source_sha256 = None,
        |cmd| cmd.sharegpt.sharegpt_source_sha256 = Some("g".repeat(64)),
        |cmd| cmd.sharegpt.sharegpt_prepared_cache_dir = Some("cache".into()),
        |cmd| cmd.sharegpt.export_sharegpt_pool = None,
    ];
    for (index, mutate) in mutations.iter().enumerate() {
        let mut command = fixture.command.clone();
        mutate(&mut command);
        assert!(
            validate_command(&command).is_err(),
            "invalid export settings accepted at case {index}"
        );
    }
    assert!(!target(&fixture).exists());
    let mut ordinary = fixture.command.clone();
    ordinary.sharegpt.export_sharegpt_pool = None;
    ordinary.sharegpt.sharegpt_source_sha256 = None;
    validate_command(&ordinary).unwrap();
    assert!(!export_pool_if_requested(&ordinary).unwrap());
}

#[test]
fn sharegpt_pool_export_cli_requires_both_flags_without_truncating_seed() {
    use clap::Parser;
    #[derive(Parser)]
    struct Arguments {
        #[command(flatten)]
        command: BenchServeCommand,
    }
    let base = [
        "bench",
        "--base-url",
        "http://127.0.0.1:1",
        "--model",
        "offline",
        "--tokenizer",
        ".",
    ];
    assert!(
        Arguments::try_parse_from(base.into_iter().chain(["--export-sharegpt-pool", "out"]))
            .is_err()
    );
    assert!(
        Arguments::try_parse_from(base.into_iter().chain(["--sharegpt-source-sha256", "pin"]))
            .is_err()
    );
    let pin = "a".repeat(64);
    let parsed = Arguments::try_parse_from(base.into_iter().chain([
        "--dataset",
        "sharegpt",
        "--sharegpt-path",
        "source.json",
        "--export-sharegpt-pool",
        "out",
        "--sharegpt-source-sha256",
        pin.as_str(),
        "--sharegpt-selection-pool-size",
        "208",
        "--seed",
        "18446744073709551614",
    ]))
    .unwrap()
    .command;
    assert_eq!(parsed.seed, Some(u64::MAX - 1));
    validate_command(&parsed).unwrap();
}

#[test]
fn sharegpt_pool_export_preserves_raw_large_integer_metadata() {
    let (mut fixture, _) = fixture();
    let source = fixture.command.sharegpt_path.as_ref().unwrap();
    // Value cannot represent this integer or decimal exactly. The export must
    // copy the original selected record bytes rather than serialize Value.
    let record = r#"{"id":"raw-source","conversations":[{"from":"human","value":"a a a a"},{"from":"gpt","value":"a a a a"}],"metadata":{"large":18446744073709551617,"decimal":0.1234567890123456789012345,"same":1,"same":2}}"#;
    std::fs::write(source, format!("[\n{record}\n]\n")).unwrap();
    fixture.command.sharegpt.sharegpt_selection_pool_size = Some(1);
    fixture.command.sharegpt.sharegpt_source_sha256 =
        Some(sha256_hex(&std::fs::read(source).unwrap()));
    pool_export::run(&fixture.command).unwrap();
    let bytes = std::fs::read(target(&fixture).join("pool.json")).unwrap();
    assert_eq!(bytes, format!("[\n{record}\n]\n").as_bytes());
    let (_, receipt) = read_export(&fixture);
    assert_eq!(receipt["child"]["source_sha256"], sha256_hex(&bytes));
    assert_eq!(receipt["records"][0]["parent_source_record_index"], 0);
}
