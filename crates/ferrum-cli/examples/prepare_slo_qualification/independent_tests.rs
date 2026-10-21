//! Independent contract tests for the offline preparer; no model or network.
use super::*;
use std::cell::Cell;

#[tokio::test]
async fn exact_geometry_uses_real_rendering_and_add_special_tokens_encoding() {
    use ferrum_server::{
        chat_template::{render_chat_prompt_with_model_template_options, ChatTemplateOptions},
        openai::ChatMessage,
    };
    use ferrum_tokenizer::Tokenizer;
    let (snapshot, tokenizer) = local_tokenizer_fixture().await;
    let mut recipe = small_recipe();
    recipe.model_snapshot = snapshot.path().to_owned();
    let template_bytes = std::fs::read(snapshot.path().join("tokenizer_config.json")).unwrap();
    let (template, _) = selected_template(&recipe, &template_bytes, None).unwrap();
    let stage = &recipe.stages[0];
    let target = stage.groups[0].target_rendered_tokens;
    let mut budget = GenerationBudget::new(recipe.limits.max_candidate_bytes);
    let prepared = generation::prepare_prompt(
        &recipe,
        &stage.name,
        0,
        target,
        &tokenizer,
        &template,
        &mut budget,
    )
    .unwrap();
    let message: ChatMessage =
        serde_json::from_value(json!({"role":"user","content":&prepared.prompt})).unwrap();
    let rendered = render_chat_prompt_with_model_template_options(
        &[message],
        &recipe.client.model,
        Some(&template),
        &ChatTemplateOptions {
            enable_thinking: recipe.client.request_policy.enable_thinking,
            reasoning_effort: recipe.client.request_policy.reasoning_effort,
            now_override: None,
        },
    )
    .unwrap();
    let actual: Vec<u32> = tokenizer
        .encode(&rendered, true)
        .unwrap()
        .into_iter()
        .map(u32::from)
        .collect();
    let without_specials = tokenizer.encode(&rendered, false).unwrap();
    let raw = tokenizer.encode(&prepared.prompt, false).unwrap();
    assert_eq!(prepared.rendered_prompt, rendered);
    assert_eq!(prepared.token_ids, actual);
    assert_eq!(prepared.token_ids.len(), target);
    assert_eq!(
        prepared.token_ids.first(),
        Some(&1),
        "actual post-processor BOS"
    );
    assert_eq!(without_specials.len() + 1, actual.len());
    assert_eq!(prepared.input_tokens_raw, raw.len());
    assert!(
        raw.len() < target,
        "raw user tokens must not substitute for rendered tokens"
    );
    assert!(prepared
        .prompt
        .starts_with(&nonce(recipe.seed, &stage.name, 0)));
    assert_eq!(prepared.evaluations, budget.evaluations);

    let mut renamed = recipe.clone();
    renamed.client.model = "gpt-oss-display-name".into();
    let same = generation::prepare_prompt(
        &renamed,
        &stage.name,
        0,
        target,
        &tokenizer,
        &template,
        &mut GenerationBudget::new(recipe.limits.max_candidate_bytes),
    )
    .unwrap();
    assert_eq!(same.prompt, prepared.prompt);
    assert_eq!(same.rendered_prompt, prepared.rendered_prompt);
    assert_eq!(
        same.token_ids, prepared.token_ids,
        "explicit template, not alias inference"
    );
}

#[tokio::test]
async fn identical_recipe_is_deterministic_and_distinct_raw_nonces_do_not_hide_token_prefixes() {
    let (snapshot, tokenizer) = local_tokenizer_fixture().await;
    let mut recipe = small_recipe();
    recipe.model_snapshot = snapshot.path().to_owned();
    let template_bytes = std::fs::read(snapshot.path().join("tokenizer_config.json")).unwrap();
    let (template, _) = selected_template(&recipe, &template_bytes, None).unwrap();
    let stage = &recipe.stages[0];
    let build = || {
        prepare_stage(
            &recipe,
            stage,
            &tokenizer,
            &template,
            "same-recipe-hash",
            &mut GenerationBudget::new(recipe.limits.max_candidate_bytes),
            &mut 0,
            &mut Vec::new(),
        )
        .unwrap()
    };
    let first = build();
    let second = build();
    assert_eq!(
        serde_json::to_vec(&first.client).unwrap(),
        serde_json::to_vec(&second.client).unwrap()
    );
    assert_eq!(
        serde_json::to_vec(&first.audit).unwrap(),
        serde_json::to_vec(&second.audit).unwrap()
    );
    assert_eq!(first.client.protocol, "chat_completions");
    assert_eq!(first.client.requests.len(), stage.groups[0].count);
    for (index, request) in first.client.requests.iter().enumerate() {
        assert!(request
            .prompt
            .starts_with(&nonce(recipe.seed, &stage.name, index)));
        assert_eq!(request.source["kind"], "synthetic_public_geometry");
        assert_eq!(request.source["index"], index);
        assert_eq!(request.max_tokens, stage.groups[0].max_tokens);
    }
    assert_ne!(
        first.client.requests[0].prompt,
        first.client.requests[1].prompt
    );
    assert_ne!(
        first.client.requests[0].source["source_id"],
        first.client.requests[1].source["source_id"]
    );
    // This real tiny tokenizer maps the different digest words to [UNK].
    // The audit must expose their identical rendered tokens despite different raw text.
    let left = &first.audit.requests[0];
    let right = &first.audit.requests[1];
    assert_eq!(left.token_ids, right.token_ids);
    assert_eq!(
        right.maximum_prior_shared_prefix_tokens,
        right.token_ids.len()
    );
    assert_eq!(
        right.maximum_prior_potential_reusable_blocks,
        (right.token_ids.len() - 1) / recipe.kv_block_tokens,
        "potential reuse must retain one real target token and never claim a cache hit",
    );
}

#[tokio::test]
async fn prompt_generation_rejects_raw_rendered_and_total_candidate_byte_exhaustion() {
    let (snapshot, tokenizer) = local_tokenizer_fixture().await;
    let mut recipe = small_recipe();
    recipe.model_snapshot = snapshot.path().to_owned();
    let template_bytes = std::fs::read(snapshot.path().join("tokenizer_config.json")).unwrap();
    let (template, _) = selected_template(&recipe, &template_bytes, None).unwrap();
    let stage = recipe.stages[0].clone();
    let target = stage.groups[0].target_rendered_tokens;
    assert!(generation::prepare_prompt(
        &recipe,
        &stage.name,
        0,
        target,
        &tokenizer,
        &template,
        &mut GenerationBudget::new(1),
    )
    .is_err());

    let mut raw_limited = recipe.clone();
    raw_limited.limits.max_raw_prompt_bytes = 1;
    assert!(generation::prepare_prompt(
        &raw_limited,
        &stage.name,
        0,
        target,
        &tokenizer,
        &template,
        &mut GenerationBudget::new(recipe.limits.max_candidate_bytes),
    )
    .is_err());

    let mut rendered_limited = recipe.clone();
    rendered_limited.limits.max_rendered_bytes = 1;
    assert!(generation::prepare_prompt(
        &rendered_limited,
        &stage.name,
        0,
        target,
        &tokenizer,
        &template,
        &mut GenerationBudget::new(recipe.limits.max_candidate_bytes),
    )
    .is_err());
    // These are explicit payload/search limits; the renderer itself has no fuel
    // sandbox, so this test makes no hard RSS or wall-clock claim.
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[tokio::test]
async fn complete_local_preparation_is_repeatable_and_never_overwrites_a_published_bundle() {
    let (snapshot, _) = local_tokenizer_fixture().await;
    let root = tempfile::tempdir().unwrap();
    let mut recipe = small_recipe();
    recipe.model_snapshot = snapshot.path().to_owned();
    let recipe_path = root.path().join("input.json");
    std::fs::write(&recipe_path, serde_json::to_vec(&recipe).unwrap()).unwrap();
    let first = root.path().join("first");
    let second = root.path().join("second");
    prepare(&recipe_path, &first).await.unwrap();
    prepare(&recipe_path, &second).await.unwrap();
    let mut names: Vec<_> = std::fs::read_dir(&first)
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    names.sort();
    for name in &names {
        assert_eq!(
            std::fs::read(first.join(name)).unwrap(),
            std::fs::read(second.join(name)).unwrap()
        );
    }
    let saved_receipt = std::fs::read(first.join("receipt.json")).unwrap();
    let receipt: Value = serde_json::from_slice(&saved_receipt).unwrap();
    assert_eq!(receipt["kind"], "synthetic_public_geometry_qualification");
    assert_eq!(receipt["encode_add_special_tokens"], true);
    assert!(prepare(&recipe_path, &first).await.is_err());
    assert_eq!(
        std::fs::read(first.join("receipt.json")).unwrap(),
        saved_receipt
    );
}

fn small_recipe() -> Recipe {
    Recipe {
        schema_version: 1,
        model_snapshot: std::path::PathBuf::from("/unused-local-fixture"),
        template_source: TemplateSource::TokenizerConfig,
        output_protocol: ferrum_types::ModelOutputProtocol::Text,
        context_tokens: 64,
        kv_block_tokens: 8,
        seed: 17,
        instruction: "Explain this fixture.".into(),
        padding_unit: " pad".into(),
        client: ClientSettings {
            base_url: "http://127.0.0.1:9".into(),
            model: "unrelated-display-name".into(),
            deadline_ms: 1000,
            request_policy: RequestPolicy {
                sampling: Sampling {
                    temperature: 0.0,
                    top_k: None,
                    top_p: Some(1.0),
                    repetition_penalty: Some(1.0),
                    seed: Some(17),
                },
                ignore_eos: true,
                enable_thinking: Some(false),
                reasoning_effort: None,
            },
        },
        limits: Limits {
            max_requests: 8,
            max_repeats: 64,
            max_evaluations_per_request: 32,
            ..Limits::default()
        },
        stages: vec![Stage {
            name: "tiny-wave".into(),
            output_dir: std::path::PathBuf::from("/unused-runtime-output/tiny-wave"),
            groups: vec![Group {
                count: 2,
                target_rendered_tokens: 24,
                max_tokens: 4,
                first_offset_ms: 0,
                interval_ms: 3,
            }],
        }],
    }
}

#[test]
fn recipe_rejects_empty_work_and_excess_requests_before_generation() {
    let valid = small_recipe();
    valid.validate().unwrap();

    let mut empty = valid.clone();
    empty.stages.clear();
    assert!(empty.validate().is_err());
    empty = valid.clone();
    empty.stages[0].groups.clear();
    assert!(empty.validate().is_err());
    empty = valid.clone();
    empty.stages[0].groups[0].count = 0;
    assert!(empty.validate().is_err());

    let mut excess = valid;
    excess.stages[0].groups[0].count = excess.limits.max_requests + 1;
    assert!(excess.validate().is_err());
}

#[test]
fn context_is_rendered_tokens_plus_output_cap_with_checked_arithmetic() {
    let mut recipe = small_recipe();
    let group = &mut recipe.stages[0].groups[0];
    group.target_rendered_tokens = recipe.context_tokens - group.max_tokens as usize;
    recipe.validate().unwrap();
    recipe.stages[0].groups[0].target_rendered_tokens += 1;
    assert!(recipe.validate().is_err());

    recipe.stages[0].groups[0].target_rendered_tokens = usize::MAX;
    assert!(
        recipe.validate().is_err(),
        "addition overflow must be rejected"
    );
}

#[test]
fn arrival_and_deadline_overflow_are_rejected_without_wrapping() {
    let mut recipe = small_recipe();
    recipe.stages[0].groups[0].first_offset_ms = u64::MAX;
    recipe.stages[0].groups[0].interval_ms = 1;
    assert!(recipe.validate().is_err(), "last arrival would wrap");

    // One arrival itself is representable; adding its positive deadline is not.
    recipe.stages[0].groups[0].count = 1;
    recipe.stages[0].groups[0].interval_ms = 0;
    assert!(recipe.validate().is_err(), "absolute deadline would wrap");
}

#[test]
fn two_stages_cannot_publish_client_results_to_the_same_runtime_directory() {
    let mut recipe = small_recipe();
    let mut second = recipe.stages[0].clone();
    second.name = "second-wave".into();
    recipe.stages.push(second);
    assert!(recipe.validate().is_err());

    recipe.stages[1].output_dir = std::path::PathBuf::from("/unused-runtime-output/second-wave");
    recipe.validate().unwrap();
    recipe.stages[1].output_dir = recipe.stages[0].output_dir.join("nested");
    assert!(
        recipe.validate().is_err(),
        "a later stage cannot write inside an earlier stage's result directory"
    );
    recipe.stages[1].output_dir = std::path::PathBuf::from("/unused-runtime-output");
    assert!(
        recipe.validate().is_err(),
        "reversing the parent-child order must not permit overlapping outputs"
    );
    recipe.stages[1].output_dir =
        std::path::PathBuf::from("/unused-runtime-output/other/../tiny-wave");
    assert!(
        recipe.validate().is_err(),
        "parent-path aliases cannot be exported"
    );
}

#[test]
fn template_source_is_explicit_and_empty_or_missing_selected_templates_fail() {
    let mut recipe = small_recipe();
    let config = br#"{"chat_template":"CONFIG {{ messages[0].content }}"}"#;
    let jinja = b"FILE {{ messages[0].content }}";
    let (selected, _) = selected_template(&recipe, config, Some(jinja)).unwrap();
    assert!(selected.template.starts_with("CONFIG "));
    recipe.template_source = TemplateSource::ChatTemplateJinja;
    let (selected, _) = selected_template(&recipe, config, Some(jinja)).unwrap();
    assert!(selected.template.starts_with("FILE "));
    assert!(selected_template(&recipe, config, None).is_err());
    assert!(selected_template(&recipe, config, Some(b" \n ")).is_err());

    recipe.template_source = TemplateSource::TokenizerConfig;
    assert!(selected_template(&recipe, br#"{}"#, Some(jinja)).is_err());
    assert!(selected_template(&recipe, br#"{"chat_template":""}"#, Some(jinja)).is_err());
    let mut serialized = serde_json::to_value(&recipe).unwrap();
    serialized
        .as_object_mut()
        .unwrap()
        .remove("template_source");
    assert!(serde_json::from_value::<Recipe>(serialized).is_err());
}

async fn local_tokenizer_fixture() -> (tempfile::TempDir, HuggingFaceTokenizer) {
    use tokenizers::{
        models::wordlevel::WordLevel, pre_tokenizers::whitespace::WhitespaceSplit,
        processors::template::TemplateProcessing,
    };
    let model = WordLevel::builder()
        .vocab(
            [
                ("[UNK]", 0),
                ("<s>", 1),
                ("</s>", 2),
                ("BEGIN", 3),
                ("END", 4),
                ("pad", 5),
            ]
            .into_iter()
            .map(|(word, id)| (word.to_owned(), id))
            .collect(),
        )
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut hf = tokenizers::Tokenizer::new(model);
    hf.with_pre_tokenizer(Some(WhitespaceSplit));
    hf.with_post_processor(Some(
        TemplateProcessing::builder()
            .try_single("<s> $A")
            .unwrap()
            .special_tokens(vec![("<s>", 1)])
            .build()
            .unwrap(),
    ));
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("tokenizer.json"),
        hf.to_string(false).unwrap(),
    )
    .unwrap();
    std::fs::write(
        dir.path().join("tokenizer_config.json"),
        br#"{"chat_template":"BEGIN {{ messages[0].content }} END","bos_token":"<s>","eos_token":"</s>"}"#,
    )
    .unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        br#"{"model_type":"fixture"}"#,
    )
    .unwrap();
    let tokenizer = HuggingFaceTokenizer::new(hf).await.unwrap();
    (dir, tokenizer)
}

#[tokio::test]
async fn preparation_failure_leaves_no_published_directory_or_staging_files() {
    let (snapshot, _) = local_tokenizer_fixture().await;
    let root = tempfile::tempdir().unwrap();
    let mut recipe = small_recipe();
    recipe.model_snapshot = snapshot.path().to_owned();
    recipe.limits.max_output_bytes = 1;
    let recipe_path = root.path().join("input.json");
    std::fs::write(&recipe_path, serde_json::to_vec(&recipe).unwrap()).unwrap();
    let destination = root.path().join("new-output");
    assert!(prepare(&recipe_path, &destination).await.is_err());
    assert!(!destination.exists());
    let entries: Vec<_> = std::fs::read_dir(root.path())
        .unwrap()
        .map(|entry| entry.unwrap().file_name())
        .collect();
    assert_eq!(entries, vec![std::ffi::OsString::from("input.json")]);
}

#[test]
fn nonmonotone_search_never_invents_an_exact_target_and_respects_attempt_budget() {
    let calls = Cell::new(0usize);
    let outcome = bounded_search(13, 31, 7, |repeats| {
        assert!(repeats <= 31);
        calls.set(calls.get() + 1);
        // Deliberately discontinuous and nonmonotone;13 is absent everywhere.
        Ok([9, 19, 4, 17, 6][repeats % 5])
    });
    assert!(outcome.is_err());
    assert!(calls.get() > 0 && calls.get() <= 7);
}

#[test]
fn bounded_search_success_is_an_observed_equality_not_a_slope_estimate() {
    let mut observed = Vec::new();
    let result = bounded_search(11, 64, 12, |repeats| {
        // The measured slope points at 24, but that candidate is not exact.
        // This nonmonotone map only qualifies the separately observed 25.
        let actual = match repeats {
            0 => 5,
            16 => 9,
            24 => 12,
            25 => 11,
            _ => 7,
        };
        observed.push((repeats, actual));
        Ok(actual)
    })
    .unwrap();
    assert!(observed.contains(&(24, 12)));
    assert_eq!(result.repeats, 25);
    assert!(observed.contains(&(result.repeats, 11)));
    assert_eq!(result.evaluations, observed.len());
    assert!(observed.len() <= 12);
}

#[test]
fn exhausted_or_failed_renderer_does_not_become_a_search_hit() {
    let calls = Cell::new(0usize);
    assert!(bounded_search(5, 64, 0, |_| {
        calls.set(calls.get() + 1);
        Ok(5)
    })
    .is_err());
    assert_eq!(
        calls.get(),
        0,
        "zero evaluation budget must not call renderer"
    );
    assert!(bounded_search(5, 64, 3, |_| {
        calls.set(calls.get() + 1);
        Err(anyhow::anyhow!("deterministic rendering failure"))
    })
    .is_err());
    assert_eq!(
        calls.get(),
        1,
        "rendering failure must be returned, not filled in"
    );
}

#[test]
fn nonce_is_deterministic_and_separates_each_recipe_identity_component() {
    let original = nonce(42, "stage-a", 0);
    assert_eq!(original, nonce(42, "stage-a", 0));
    assert_ne!(original, nonce(43, "stage-a", 0));
    assert_ne!(original, nonce(42, "stage-b", 0));
    assert_ne!(original, nonce(42, "stage-a", 1));
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn atomic_directory_publish_preserves_existing_file_and_empty_directory() {
    let root = tempfile::tempdir().unwrap();
    let staging = root.path().join("staging");
    std::fs::create_dir(&staging).unwrap();
    std::fs::write(staging.join("client.json"), b"complete client payload").unwrap();
    let existing_file = root.path().join("existing-file");
    std::fs::write(&existing_file, b"do not replace").unwrap();
    assert!(publish_new_directory(&staging, &existing_file).is_err());
    assert_eq!(std::fs::read(&existing_file).unwrap(), b"do not replace");
    assert_eq!(
        std::fs::read(staging.join("client.json")).unwrap(),
        b"complete client payload"
    );

    let existing_empty_directory = root.path().join("existing-empty-directory");
    std::fs::create_dir(&existing_empty_directory).unwrap();
    assert!(publish_new_directory(&staging, &existing_empty_directory).is_err());
    assert_eq!(
        std::fs::read_dir(&existing_empty_directory)
            .unwrap()
            .count(),
        0
    );
    assert!(staging.join("client.json").is_file());

    let fresh = root.path().join("fresh");
    publish_new_directory(&staging, &fresh).unwrap();
    assert!(!staging.exists());
    assert_eq!(
        std::fs::read(fresh.join("client.json")).unwrap(),
        b"complete client payload"
    );
}

#[cfg(any(target_os = "linux", target_os = "macos"))]
#[test]
fn atomic_directory_publish_does_not_follow_or_replace_destination_symlink() {
    use std::os::unix::fs::symlink;
    let root = tempfile::tempdir().unwrap();
    let target = root.path().join("original");
    std::fs::create_dir(&target).unwrap();
    std::fs::write(target.join("sentinel"), b"original").unwrap();
    let destination = root.path().join("link");
    symlink(&target, &destination).unwrap();
    let staging = root.path().join("staging");
    std::fs::create_dir(&staging).unwrap();
    std::fs::write(staging.join("new"), b"new").unwrap();

    assert!(publish_new_directory(&staging, &destination).is_err());
    assert_eq!(std::fs::read_link(&destination).unwrap(), target);
    assert_eq!(std::fs::read(target.join("sentinel")).unwrap(), b"original");
    assert!(staging.join("new").is_file());
}
