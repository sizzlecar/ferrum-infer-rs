use super::super::{execute_cell, tests::test_command, Cell, RunContext};
use super::*;
use serde_json::json;
use std::sync::{Arc, Mutex};

struct Fixture {
    _directory: tempfile::TempDir,
    tokenizer: tokenizers::Tokenizer,
    command: BenchServeCommand,
}

fn text(tokens: usize) -> String {
    std::iter::repeat_n("a", tokens)
        .collect::<Vec<_>>()
        .join(" ")
}

fn record(id: &str, input: usize, output: usize) -> Value {
    json!({"id":id,"conversations":[
        {"from":"human","value":text(input)},
        {"from":"gpt","value":text(output)},
        {"from":"human","value":"unused later turn"}
    ]})
}

impl Fixture {
    fn new(bytes: &[u8]) -> Self {
        let directory = tempfile::tempdir().unwrap();
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(
                [("[UNK]".to_owned(), 0), ("a".to_owned(), 1)]
                    .into_iter()
                    .collect(),
            )
            .unk_token("[UNK]".into())
            .build()
            .unwrap();
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(
            tokenizers::pre_tokenizers::whitespace::WhitespaceSplit,
        ));
        tokenizer
            .save(directory.path().join("tokenizer.json"), false)
            .unwrap();
        let source = directory.path().join("sharegpt.json");
        std::fs::write(&source, bytes).unwrap();
        let mut command = test_command();
        command.dataset = "sharegpt".into();
        command.sharegpt_path = Some(source);
        command.tokenizer = directory.path().to_owned();
        command.seed = Some(42);
        command.ignore_eos = true;
        Self {
            _directory: directory,
            tokenizer,
            command,
        }
    }

    fn records(records: &[Value]) -> Self {
        Self::new(&serde_json::to_vec_pretty(records).unwrap())
    }
}

#[test]
fn sharegpt_array_and_jsonl_preserve_first_pair_text_and_variable_lengths() {
    let mut records = vec![
        record("one", 4, 5),
        record("two", 7, 9),
        record("three", 6, 4),
    ];
    records[0]["conversations"][0]["value"] = json!("  a\na  a a  ");
    let array = Fixture::records(&records);
    let lines = records
        .iter()
        .map(Value::to_string)
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    let jsonl = Fixture::new(lines.as_bytes());
    let (a, evidence_a) = load(&array.command, &array.tokenizer, 0, 3).unwrap();
    let (b, evidence_b) = load(&jsonl.command, &jsonl.tokenizer, 0, 3).unwrap();
    assert_eq!(evidence_a.source_format, "json_array");
    assert_eq!(evidence_b.source_format, "jsonl");
    assert_ne!(evidence_a.source_sha256, evidence_b.source_sha256);
    assert_eq!(evidence_a.repeats, evidence_b.repeats);
    assert_eq!(evidence_b.source_sha256, sha256_hex(lines.as_bytes()));
    for ((actual, second), sample) in a.iter().zip(&b).zip(&evidence_a.repeats[0].samples) {
        let original = &records[sample.source_record_index as usize];
        assert_eq!(
            actual.text,
            original["conversations"][0]["value"].as_str().unwrap()
        );
        assert_eq!(actual.text, second.text);
        assert_eq!(actual.input_tokens, sample.input_tokens);
        assert_eq!(
            actual.output_budget,
            Some(sample.reference_output_tokens as usize)
        );
    }
}

#[test]
fn sharegpt_filters_count_exclusions_without_truncation_and_include_boundaries() {
    let fixture = Fixture::records(&[
        record("minimum", 4, 4),
        record("boundary", 8, 7),
        record("input-short", 3, 4),
        record("input-long", 9, 4),
        record("output-short", 4, 3),
        record("output-long", 4, 9),
        record("total-long", 8, 8),
        json!({"id":"missing","conversations":[{"from":"human","value":"a a a a"}]}),
        json!({"id":"empty","conversations":[{"from":"human","value":"  "},{"from":"gpt","value":"a a a a"}]}),
    ]);
    let mut cmd = fixture.command.clone();
    cmd.sharegpt.sharegpt_max_input_tokens = 8;
    cmd.sharegpt.sharegpt_max_output_tokens = Some(8);
    cmd.sharegpt.sharegpt_max_total_tokens = 16;
    cmd.sharegpt.sharegpt_chat_template_reserve_tokens = 1;
    let (prompts, evidence) = load(&cmd, &fixture.tokenizer, 0, 2).unwrap();
    assert_eq!(
        evidence.counts,
        ShareGptCounts {
            records: 9,
            eligible: 2,
            missing_first_pair: 1,
            empty_text: 1,
            input_too_short: 1,
            input_too_long: 1,
            output_too_short: 1,
            output_too_long: 1,
            total_too_long: 1,
            ..Default::default()
        }
    );
    assert!(prompts
        .iter()
        .any(|p| p.input_tokens == 8 && p.output_budget == Some(7)));
    let error = load(&cmd, &fixture.tokenizer, 0, 3)
        .err()
        .unwrap()
        .to_string();
    assert!(error.contains("2 eligible records"));
    assert!(error.contains("source_sha256="));
    assert!(error.contains("input_too_long"));
}

#[test]
fn sharegpt_selection_is_without_replacement_and_identical_across_concurrency() {
    let records = (0..40)
        .map(|i| record(&i.to_string(), 4 + i % 7, 4 + i % 11))
        .collect::<Vec<_>>();
    let mut fixture = Fixture::records(&records);
    fixture.command.warmup_requests = 3;
    let (_, baseline) = load(&fixture.command, &fixture.tokenizer, 1, 12).unwrap();
    let samples = &baseline.repeats[0].samples;
    let unique = samples
        .iter()
        .map(|s| s.source_record_index)
        .collect::<std::collections::HashSet<_>>();
    assert_eq!(unique.len(), 12);
    assert_eq!(samples[0].phase, BenchmarkPhase::Warmup);
    assert_eq!(samples[2].request_index, 2);
    assert_eq!(samples[3].phase, BenchmarkPhase::Measured);
    assert_eq!(samples[3].request_index, 0);
    for concurrency in [1, 4, 8, 16, 32] {
        fixture.command.concurrency = concurrency;
        let (_, result) = load(&fixture.command, &fixture.tokenizer, 1, 12).unwrap();
        assert_eq!(baseline, result);
    }
    assert_ne!(repeat_seed(42, 0), repeat_seed(42, 1));
}

#[test]
fn sharegpt_explicit_fixed_output_changes_only_the_declared_budget_and_filter() {
    let mut fixture = Fixture::records(&[record("long-answer", 4, 20)]);
    fixture.command.sharegpt.sharegpt_max_total_tokens = 12;
    assert!(load(&fixture.command, &fixture.tokenizer, 0, 1).is_err());
    fixture.command.sharegpt.sharegpt_fixed_output_tokens = Some(8);
    let (prompts, evidence) = load(&fixture.command, &fixture.tokenizer, 0, 1).unwrap();
    assert_eq!(prompts[0].output_budget, Some(8));
    assert_eq!(evidence.filter.fixed_output_tokens, Some(8));
    assert_eq!(evidence.repeats[0].samples[0].reference_output_tokens, 20);
    assert_eq!(evidence.repeats[0].samples[0].requested_output_tokens, 8);
}

#[test]
fn sharegpt_malformed_jsonl_and_missing_pairs_are_counted_but_invalid_array_fails() {
    let lines = format!(
        "{{bad\n{}\n{}\n{}\n",
        record("good", 4, 5),
        json!({"input":"legacy prompt with no reference answer"}),
        json!({"conversations":[{"from":"gpt","value":"a a a a"},{"from":"human","value":"a a a a"}]})
    );
    let fixture = Fixture::new(lines.as_bytes());
    let (_, evidence) = load(&fixture.command, &fixture.tokenizer, 0, 1).unwrap();
    assert_eq!(evidence.counts.records, 4);
    assert_eq!(evidence.counts.malformed_json, 1);
    assert_eq!(evidence.counts.missing_first_pair, 2);
    assert_eq!(evidence.counts.missing_original_id, 2);
    let broken = Fixture::new(b"[{\"conversations\":");
    assert!(load(&broken.command, &broken.tokenizer, 0, 1)
        .err()
        .unwrap()
        .to_string()
        .contains("parse ShareGPT JSON array"));
}

#[test]
fn sharegpt_tokenizer_padding_and_truncation_cannot_hide_real_lengths() {
    let mut fixture = Fixture::records(&[record("unmodified", 9, 11)]);
    fixture
        .tokenizer
        .with_truncation(Some(tokenizers::TruncationParams {
            max_length: 2,
            ..Default::default()
        }))
        .unwrap();
    fixture
        .tokenizer
        .with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(16),
            ..Default::default()
        }));
    fixture
        .tokenizer
        .save(fixture.command.tokenizer.join("tokenizer.json"), false)
        .unwrap();
    let (prompts, evidence) = load(&fixture.command, &fixture.tokenizer, 0, 1).unwrap();
    assert_eq!(prompts[0].input_tokens, 9);
    assert_eq!(prompts[0].output_budget, Some(11));
    assert_eq!(evidence.repeats[0].samples[0].reference_output_tokens, 11);
}

#[test]
fn sharegpt_requires_seed_and_rejects_invalid_bounds() {
    let mut fixture = Fixture::records(&[record("one", 4, 4)]);
    fixture.command.seed = None;
    assert!(validate(&fixture.command).is_err());
    fixture.command.seed = Some(42);
    fixture.command.sharegpt.sharegpt_min_output_tokens = 0;
    assert!(validate(&fixture.command).is_err());
    fixture.command.sharegpt = ShareGptArgs::default();
    fixture.command.sharegpt.sharegpt_fixed_output_tokens = Some(0);
    assert!(validate(&fixture.command).is_err());
}

#[tokio::test]
async fn sharegpt_http_uses_each_reference_budget_and_reports_variable_shape() {
    use axum::{extract::State, http::header, routing::post, Json, Router};
    let captured = Arc::new(Mutex::new(Vec::<Value>::new()));
    let app = Router::new().route("/v1/chat/completions", post(
        |State(captured): State<Arc<Mutex<Vec<Value>>>>, Json(body): Json<Value>| async move {
            let output = body["max_tokens"].as_u64().unwrap();
            captured.lock().unwrap().push(body);
            let mut stream = String::new();
            for _ in 0..output { stream.push_str("data: {\"id\":\"test\",\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\n"); }
            stream.push_str(&format!("data: {{\"id\":\"test\",\"choices\":[],\"usage\":{{\"prompt_tokens\":4,\"completion_tokens\":{output}}}}}\n\ndata: [DONE]\n\n"));
            ([(header::CONTENT_TYPE, "text/event-stream")], stream)
        }
    )).with_state(Arc::clone(&captured));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base_url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });
    let mut fixture = Fixture::records(&[record("a", 4, 4), record("b", 4, 7), record("c", 4, 11)]);
    fixture.command.num_prompts = 2;
    fixture.command.warmup_requests = 1;
    fixture.command.random_output_len = 128;
    let context = RunContext {
        client: Arc::new(reqwest::Client::new()),
        base_url: Arc::new(base_url),
        model: Arc::new("test".into()),
        max_out: 128,
        ignore_eos: true,
        enable_thinking: Some(false),
        reasoning_effort: None,
        sampling: fixture.command.sampling.request_sampling(),
        timeout_s: 5.0,
        benchmark_run_id: Arc::new("sharegpt-test".into()),
        capture_slo: false,
    };
    let prepared = prepare(&fixture.command).unwrap().unwrap();
    let (report, _) = execute_cell(
        &fixture.command,
        &context,
        Cell::Closed(2),
        "cell",
        Some(&prepared),
        None,
    )
    .await
    .unwrap();
    server.abort();
    assert_eq!(report.n_prompt, 0);
    assert_eq!(report.n_gen, 0);
    assert_eq!(report.completed_per_run, vec![2]);
    assert_eq!(report.errored_per_run, vec![0]);
    let captured = captured.lock().unwrap();
    let samples = &report.dataset_evidence.as_ref().unwrap().repeats[0].samples;
    let mut observed = captured
        .iter()
        .map(|b| b["max_tokens"].as_u64().unwrap())
        .collect::<Vec<_>>();
    observed.sort();
    assert_eq!(observed, vec![4, 7, 11]);
    for body in captured.iter() {
        assert_eq!(body["ignore_eos"], true);
        assert_eq!(body["messages"][0]["content"], text(4));
    }
    assert_eq!(
        report.output_tokens_per_request.as_ref().unwrap()[0],
        samples
            .iter()
            .filter(|s| s.phase == BenchmarkPhase::Measured)
            .map(|s| s.requested_output_tokens)
            .collect::<Vec<_>>()
    );
    let mut serialized = serde_json::to_value(&report).unwrap();
    serialized
        .as_object_mut()
        .unwrap()
        .remove("dataset_evidence");
    assert!(
        serde_json::from_value::<ferrum_bench_core::BenchReport>(serialized)
            .unwrap()
            .dataset_evidence
            .is_none()
    );
}

#[test]
fn sharegpt_natural_eos_requires_usage_within_each_request_budget() {
    for usage in [None, Some(0), Some(3), Some(7), Some(8)] {
        let mut state = super::super::StreamState::new(std::time::Instant::now(), 4);
        state.maximum_completion_tokens = Some(7);
        state
            .handle_payload(r#"{"choices":[{"delta":{"content":"answer"}}]}"#)
            .unwrap();
        state.usage_completion_tokens = usage;
        state.done_count = 1;
        assert_eq!(
            state.finish().success,
            matches!(usage, Some(1..=7)),
            "{usage:?}"
        );
    }
}
