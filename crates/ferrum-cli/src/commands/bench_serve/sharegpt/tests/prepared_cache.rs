use super::super::cache::PreparedCache;
use super::*;
use std::cell::Cell as Flag;
use std::path::PathBuf;

fn fixture() -> Fixture {
    let records = (0..40)
        .map(|index| record(&format!("id-{index:02}"), 4 + index % 7, 4 + index % 11))
        .collect::<Vec<_>>();
    let mut fixture = Fixture::records(&records);
    fixture.command.num_prompts = 3;
    fixture.command.warmup_requests = 1;
    fixture.command.sharegpt.sharegpt_selection_pool_size = Some(12);
    fixture.command.sharegpt.sharegpt_prepared_cache_dir =
        Some(fixture._directory.path().join("prepared"));
    fixture
}

fn cache_for(cmd: &BenchServeCommand, tokenizer: &tokenizers::Tokenizer) -> PreparedCache {
    PreparedCache::for_command(
        cmd,
        tokenizer,
        (cmd.num_prompts + cmd.warmup_requests) as usize,
    )
    .unwrap()
    .unwrap()
}

fn load_repeat(
    cmd: &BenchServeCommand,
    tokenizer: &tokenizers::Tokenizer,
    repeat: u32,
) -> Result<PreparedRepeat> {
    let (prompts, evidence) = load(
        cmd,
        tokenizer,
        repeat,
        (cmd.num_prompts + cmd.warmup_requests) as usize,
    )?;
    Ok(PreparedRepeat { prompts, evidence })
}

fn assert_same(actual: &PreparedRepeat, expected: &PreparedRepeat) {
    assert_eq!(actual.evidence, expected.evidence);
    let prompt_values = |entry: &PreparedRepeat| {
        entry
            .prompts
            .iter()
            .map(|prompt| {
                (
                    prompt.text.clone(),
                    prompt.sha256.clone(),
                    prompt.input_tokens,
                    prompt.output_budget,
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(prompt_values(actual), prompt_values(expected));
}

fn only_cache_file(fixture: &Fixture) -> PathBuf {
    let paths = std::fs::read_dir(
        fixture
            .command
            .sharegpt
            .sharegpt_prepared_cache_dir
            .as_ref()
            .unwrap(),
    )
    .unwrap()
    .map(|entry| entry.unwrap().path())
    .collect::<Vec<_>>();
    assert_eq!(
        paths.len(),
        1,
        "one prepared repeat should publish one complete entry"
    );
    paths.into_iter().next().unwrap()
}

#[test]
fn sharegpt_prepared_cache_cli_is_explicit_and_sharegpt_only() {
    use clap::Parser;
    #[derive(Parser)]
    struct Arguments {
        #[command(flatten)]
        sharegpt: ShareGptArgs,
    }
    assert!(Arguments::parse_from(["bench-serve"])
        .sharegpt
        .sharegpt_prepared_cache_dir
        .is_none());
    let args = Arguments::parse_from(["bench-serve", "--sharegpt-prepared-cache-dir", "prepared"]);
    assert_eq!(
        args.sharegpt.sharegpt_prepared_cache_dir,
        Some(PathBuf::from("prepared"))
    );
    let mut fixture = fixture();
    fixture.command.dataset = "random".into();
    fixture.command.sharegpt.sharegpt_selection_pool_size = None;
    assert!(validate(&fixture.command).is_err());
}

#[test]
fn sharegpt_prepared_cache_cold_and_hit_preserve_all_requests_and_evidence() {
    let fixture = fixture();
    let mut no_cache = fixture.command.clone();
    no_cache.sharegpt.sharegpt_prepared_cache_dir = None;
    let expected = prepare(&no_cache).unwrap().unwrap();
    let cold = prepare(&fixture.command).unwrap().unwrap();
    assert_same(&cold[0], &expected[0]);
    let hot = cache_for(&fixture.command, &fixture.tokenizer)
        .get_or_prepare(0, || {
            panic!("valid cache must skip selection and tokenization")
        })
        .unwrap();
    assert_same(&hot, &expected[0]);
    let second_process = prepare(&fixture.command).unwrap().unwrap();
    assert_same(&second_process[0], &expected[0]);
    assert_eq!(
        hot.evidence.repeats[0].samples[0].phase,
        BenchmarkPhase::Warmup
    );
    assert_eq!(
        hot.evidence.repeats[0].samples[1].phase,
        BenchmarkPhase::Measured
    );
}

#[test]
fn sharegpt_prepared_cache_binds_workload_settings_but_not_concurrency() {
    let fixture = fixture();
    let expected = prepare(&fixture.command).unwrap().unwrap();
    let mut different_concurrency = fixture.command.clone();
    different_concurrency.concurrency = 32;
    let hot = cache_for(&different_concurrency, &fixture.tokenizer)
        .get_or_prepare(0, || {
            panic!("concurrency does not change ShareGPT selection")
        })
        .unwrap();
    assert_same(&hot, &expected[0]);

    let mutations: &[fn(&mut BenchServeCommand)] = &[
        |cmd| cmd.num_prompts += 1,
        |cmd| cmd.warmup_requests = 0,
        |cmd| cmd.seed = Some(43),
        |cmd| cmd.sharegpt.sharegpt_selection_pool_size = Some(13),
        |cmd| cmd.sharegpt.sharegpt_selection_pool_size = None,
        |cmd| cmd.sharegpt.sharegpt_min_input_tokens = 5,
        |cmd| cmd.sharegpt.sharegpt_fixed_output_tokens = Some(9),
        |cmd| cmd.ignore_eos = false,
        |cmd| cmd.enable_thinking = Some(false),
    ];
    for mutate in mutations {
        let mut changed = fixture.command.clone();
        mutate(&mut changed);
        let called = Flag::new(false);
        let rebuilt = cache_for(&changed, &fixture.tokenizer)
            .get_or_prepare(0, || {
                called.set(true);
                load_repeat(&changed, &fixture.tokenizer, 0)
            })
            .unwrap();
        assert!(called.get(), "changed selection contract must miss");
        assert_same(
            &rebuilt,
            &load_repeat(&changed, &fixture.tokenizer, 0).unwrap(),
        );
    }
    let called = Flag::new(false);
    let next_repeat = cache_for(&fixture.command, &fixture.tokenizer)
        .get_or_prepare(1, || {
            called.set(true);
            load_repeat(&fixture.command, &fixture.tokenizer, 1)
        })
        .unwrap();
    assert!(called.get());
    assert_same(
        &next_repeat,
        &load_repeat(&fixture.command, &fixture.tokenizer, 1).unwrap(),
    );
}

#[test]
fn sharegpt_prepared_cache_rehashes_content_even_with_unchanged_size_and_mtime() {
    let fixture = fixture();
    let first = prepare(&fixture.command).unwrap().unwrap();
    let source = fixture.command.sharegpt_path.as_ref().unwrap();
    let old_metadata = std::fs::metadata(source).unwrap();
    let bytes = std::fs::read_to_string(source)
        .unwrap()
        .replacen("id-00", "id-XX", 1);
    std::fs::write(source, bytes).unwrap();
    std::fs::File::options()
        .write(true)
        .open(source)
        .unwrap()
        .set_times(std::fs::FileTimes::new().set_modified(old_metadata.modified().unwrap()))
        .unwrap();
    assert_eq!(std::fs::metadata(source).unwrap().len(), old_metadata.len());
    let called = Flag::new(false);
    let changed = cache_for(&fixture.command, &fixture.tokenizer)
        .get_or_prepare(0, || {
            called.set(true);
            load_repeat(&fixture.command, &fixture.tokenizer, 0)
        })
        .unwrap();
    assert!(called.get());
    assert_ne!(
        first[0].evidence.source_sha256,
        changed.evidence.source_sha256
    );
    assert_same(
        &changed,
        &load_repeat(&fixture.command, &fixture.tokenizer, 0).unwrap(),
    );

    let tokenizer_path = fixture.command.tokenizer.join("tokenizer.json");
    let original = std::fs::read(&tokenizer_path).unwrap();
    let mut changed_bytes = original.clone();
    changed_bytes.push(b'\n');
    std::fs::write(&tokenizer_path, changed_bytes).unwrap();
    let called = Flag::new(false);
    cache_for(&fixture.command, &fixture.tokenizer)
        .get_or_prepare(0, || {
            called.set(true);
            load_repeat(&fixture.command, &fixture.tokenizer, 0)
        })
        .unwrap();
    assert!(called.get(), "raw tokenizer content is independently bound");

    // The file is unchanged for this probe; the actual runtime config differs.
    let mut tokenizer = fixture.tokenizer.clone();
    tokenizer.with_pre_tokenizer(Some(
        tokenizers::pre_tokenizers::byte_level::ByteLevel::default(),
    ));
    let called = Flag::new(false);
    let result = cache_for(&fixture.command, &tokenizer).get_or_prepare(0, || {
        called.set(true);
        Err(FerrumError::model(
            "stop after observing a changed-tokenizer cache miss",
        ))
    });
    assert!(result.is_err());
    assert!(
        called.get(),
        "actual runtime tokenizer config must also be bound"
    );
}

#[test]
fn sharegpt_prepared_cache_rebuilds_truncated_mismatched_and_inconsistent_entries() {
    let fixture = fixture();
    let expected = prepare(&fixture.command).unwrap().unwrap();
    let path = only_cache_file(&fixture);
    let original: Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
    let mut wrong_contract = original.clone();
    wrong_contract["identity"]["selector_contract"] = json!("incompatible-selector");
    let mut wrong_digest = original.clone();
    wrong_digest["payload"]["prompts"][0]["text"] = json!("changed prompt");
    let mut wrong_phase = original.clone();
    wrong_phase["payload"]["evidence"]["repeats"][0]["samples"][0]["phase"] = json!("measured");
    // Recompute through the typed serialization used by the reader below.
    // Even a consistent checksum must not bypass phase/ordinal validation.
    super::super::cache::refresh_test_payload_checksum(&mut wrong_phase);
    for bytes in [
        b"{".to_vec(),
        serde_json::to_vec(&wrong_contract).unwrap(),
        serde_json::to_vec(&wrong_digest).unwrap(),
        serde_json::to_vec(&wrong_phase).unwrap(),
    ] {
        std::fs::write(&path, bytes).unwrap();
        let called = Flag::new(false);
        let rebuilt = cache_for(&fixture.command, &fixture.tokenizer)
            .get_or_prepare(0, || {
                called.set(true);
                load_repeat(&fixture.command, &fixture.tokenizer, 0)
            })
            .unwrap();
        assert!(called.get());
        assert_same(&rebuilt, &expected[0]);
        cache_for(&fixture.command, &fixture.tokenizer)
            .get_or_prepare(0, || panic!("rebuilt cache must be reusable"))
            .unwrap();
    }
}
