use super::*;
use ferrum_bench_core::{
    dataset::{
        ShareGptCounts, ShareGptDatasetEvidence, ShareGptFilter, ShareGptSample, ShareGptSelection,
    },
    env::HttpRequestSampling,
    BenchmarkPhase,
};
use ferrum_server::chat_template::ModelChatTemplate;
use std::num::{NonZeroU32, NonZeroU64, NonZeroUsize};

struct Fixture {
    directory: tempfile::TempDir,
    manifest: manifest::Manifest,
    inputs: inputs::PreparedInputs,
}

impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let tokenizer_path = directory.path().join("tokenizer.json");
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
        tokenizer.save(&tokenizer_path, false).unwrap();
        let mut dataset = Vec::new();
        let mut samples = Vec::new();
        for (index, maximum) in [24, 786, 82, 200].into_iter().enumerate() {
            let prompt = format!("  {}\na  ", "a ".repeat(index + 1));
            let assistant = "a ".repeat(maximum as usize);
            let record = serde_json::json!({"id":format!("source-{index}"), "conversations":[
                {"from":"human","value":prompt}, {"from":"gpt","value":assistant}
            ]});
            serde_json::to_writer(&mut dataset, &record).unwrap();
            dataset.push(b'\n');
            samples.push(ShareGptSample {
                source_record_index: index as u64,
                original_id: Some(format!("source-{index}")),
                phase: BenchmarkPhase::Warmup,
                request_index: index as u32,
                prompt_sha256: sharegpt::hex(&Sha256::digest(prompt.as_bytes())),
                assistant_sha256: sharegpt::hex(&Sha256::digest(assistant.as_bytes())),
                input_tokens: index as u32 + 2,
                reference_output_tokens: maximum,
                requested_output_tokens: maximum,
            });
        }
        let dataset_path = directory.path().join("dataset.jsonl");
        std::fs::write(&dataset_path, &dataset).unwrap();
        let selection_sha256: [u8; 32] =
            Sha256::digest(serde_json::to_vec(&samples).unwrap()).into();
        let policy = sharegpt::RequestPolicy {
            requested_model_name: "public\"model".into(),
            sampling: HttpRequestSampling::default(),
            ignore_eos: true,
            enable_thinking: Some(false),
            reasoning_effort: None,
            server_default_enable_thinking: None,
            interleaved_system_coalescing: true,
        };
        let evidence = ShareGptDatasetEvidence {
            dataset: "sharegpt".into(),
            source_path: dataset_path.display().to_string(),
            source_sha256: sharegpt::hex(&Sha256::digest(&dataset)),
            source_format: "jsonl".into(),
            tokenizer_sha256: sharegpt::hex(&Sha256::digest(
                std::fs::read(&tokenizer_path).unwrap(),
            )),
            filter: ShareGptFilter {
                min_input_tokens: 1,
                max_input_tokens: None,
                min_output_tokens: 1,
                max_output_tokens: None,
                max_total_tokens: None,
                chat_template_reserve_tokens: 0,
                fixed_output_tokens: None,
            },
            counts: ShareGptCounts {
                records: 4,
                eligible: 4,
                ..Default::default()
            },
            prompt_seed: 17,
            sampling: "without_replacement".into(),
            ignore_eos: true,
            enable_thinking: Some(false),
            repeats: vec![ShareGptSelection {
                repeat_index: 0,
                rng_seed: 17,
                selection_sha256: sharegpt::hex(&selection_sha256),
                samples,
            }],
        };
        let report_path = directory.path().join("selection.json");
        let report = serde_json::to_vec(&serde_json::json!({"dataset_evidence": evidence,
            "env":{"http_request_sampling":policy.sampling}}))
        .unwrap();
        std::fs::write(&report_path, &report).unwrap();
        let mut manifest = super::super::tests::manifest();
        manifest.schema_version = 2;
        manifest.prompts.clear();
        manifest.sharegpt = Some(sharegpt::FrozenShareGpt {
            report_path,
            report_sha256: Sha256::digest(&report).into(),
            dataset_path,
            tokenizer_path,
            repeat_index: 0,
            selection_sha256,
            request_policy: policy,
            read_limits: Default::default(),
        });
        manifest.protocol.maximum_requests = NonZeroUsize::new(5).unwrap();
        manifest.protocol.output = manifest::Codec::ChatSse {
            include_usage: true,
        };
        manifest.validation_model = manifest::ValidationSource::ExportedProfile {
            profile: directory.path().join("cut-profile.json"),
            source: directory.path().join("cut-source.jsonl"),
        };
        manifest.training[0].prompts = vec![3, 0, 1, 0, 2];
        manifest.training[0].repetitions = NonZeroUsize::new(2).unwrap();
        manifest.validation[0].prompts = vec![0, 1, 2, 3];
        manifest.reference = Some(reference::ReferenceConfig {
            graph_routes: Default::default(),
            piecewise: None,
            request_policy: serde_json::from_value(
                serde_json::json!({"kind":"fixed_reference_output",
                "max_tokens":3, "eos":"ignore"}),
            )
            .unwrap(),
            revision: NonZeroU64::new(1).unwrap(),
            artifact_path: directory.path().join("reference.json"),
            frozen_plan_path: directory.path().join("reference-plan.json"),
            warmup: vec![],
            curve_prompt_indices: vec![0, 1, 2, 3],
            granule_tokens: NonZeroU32::new(1).unwrap(),
            repetitions: NonZeroUsize::new(1).unwrap(),
            decode_unit: reference::DecodeUnit {
                prompt_index: 0,
                generated_before: NonZeroU32::new(1).unwrap(),
            },
            limits: Default::default(),
        });
        manifest.validate().unwrap();
        let inputs = inputs::PreparedInputs::prepare(&manifest, ModelChatTemplate::new(
            "{% for message in messages %}{{ message['role'] }}={{ message['content'] }};{% endfor %}assistant=",
            "fixture",
        )).unwrap();
        Self {
            directory,
            manifest,
            inputs,
        }
    }
}

#[test]
fn loaded_original_bodies_preserve_budgets_hashes_and_repetition_order_without_reference_override()
{
    let fixture = Fixture::new();
    let directory = fixture.directory.path().join("http");
    let receipt = export(&fixture.inputs, &fixture.manifest, &directory).unwrap();
    let bytes = std::fs::read(directory.join("manifest.json")).unwrap();
    let metadata: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(
        receipt.manifest_sha256,
        sharegpt::hex(&Sha256::digest(&bytes))
    );
    assert_eq!(metadata["reference_policy_applied"], false);
    assert_eq!(
        metadata["training"][0]["prompts"],
        serde_json::json!([3, 0, 1, 0, 2])
    );
    assert_eq!(metadata["training"][0]["repetitions"], 2);
    assert_eq!(metadata["frozen_selection"], fixture.inputs.provenance());
    assert_eq!(receipt.requests, 4);
    assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 5);
    let model = ferrum_types::ModelId("resolved-internal".into());
    let mut written = bytes.len() as u64;
    for (index, maximum) in [24, 786, 82, 200].into_iter().enumerate() {
        let row = &metadata["requests"][index];
        let body = std::fs::read(directory.join(row["body_file"].as_str().unwrap())).unwrap();
        written += body.len() as u64;
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();
        let (_, actual) = fixture
            .inputs
            .request(
                &fixture.manifest,
                index,
                &model,
                ferrum_types::ModelOutputProtocol::Text,
            )
            .unwrap();
        assert_eq!(row["body_sha256"], sharegpt::hex(&Sha256::digest(&body)));
        assert_eq!(row["body_sha256"], actual["chat_body_sha256"]);
        assert_eq!(parsed["max_tokens"], maximum);
        assert_eq!(parsed["ignore_eos"], true);
        assert_eq!(parsed["model"], "public\"model");
        assert_eq!(parsed["stream_options"]["include_usage"], true);
        let (reference, _) = fixture
            .inputs
            .request_for_phase(
                &fixture.manifest,
                index,
                &model,
                ferrum_types::ModelOutputProtocol::Text,
                report::Phase::Reference,
            )
            .unwrap();
        assert_eq!(reference.sampling_params.max_tokens, 3);
    }
    assert_eq!(receipt.total_bytes, written);
    assert!(export(&fixture.inputs, &fixture.manifest, &directory).is_err());
    assert_eq!(
        std::fs::read(directory.join("manifest.json")).unwrap(),
        bytes
    );
}

#[test]
fn bounded_partial_export_retains_written_body_without_publishing_manifest_or_overwriting() {
    let fixture = Fixture::new();
    let directory = fixture.directory.path().join("partial");
    let first = serde_json::to_vec(&fixture.inputs.original_http_body(0).unwrap()).unwrap();
    assert!(export_with_limit(
        &fixture.inputs,
        &fixture.manifest,
        &directory,
        first.len() as u64
    )
    .is_err());
    assert_eq!(
        std::fs::read(directory.join("request-0000.json")).unwrap(),
        first
    );
    assert!(!directory.join("request-0001.json").exists());
    assert!(!directory.join("manifest.json").exists());
    assert!(export(&fixture.inputs, &fixture.manifest, &directory).is_err());
    assert_eq!(
        std::fs::read(directory.join("request-0000.json")).unwrap(),
        first
    );
}

#[test]
fn unsupported_input_or_missing_selection_is_rejected_before_creating_outputs() {
    let fixture = Fixture::new();
    let directory = fixture.directory.path().join("invalid");
    assert!(export(
        &inputs::PreparedInputs::Rendered,
        &super::super::tests::manifest(),
        &directory
    )
    .is_err());
    assert!(!directory.exists());
    let mut manifest = fixture.manifest.clone();
    manifest.training[0].prompts[0] = 1000;
    assert!(export(&fixture.inputs, &manifest, &directory).is_err());
    assert!(!directory.exists());
}

#[cfg(unix)]
#[test]
fn existing_directory_symlink_and_file_are_never_replaced() {
    let fixture = Fixture::new();
    let target = fixture.directory.path().join("target");
    std::fs::create_dir(&target).unwrap();
    std::fs::write(target.join("keep"), b"existing").unwrap();
    let alias = fixture.directory.path().join("alias");
    std::os::unix::fs::symlink(&target, &alias).unwrap();
    let file = fixture.directory.path().join("file");
    std::fs::write(&file, b"existing-file").unwrap();
    for path in [&target, &alias, &file] {
        assert!(export(&fixture.inputs, &fixture.manifest, path).is_err());
    }
    assert_eq!(std::fs::read(target.join("keep")).unwrap(), b"existing");
    assert_eq!(std::fs::read(&file).unwrap(), b"existing-file");
}

#[test]
fn typed_cli_export_is_optional_and_its_destination_cannot_alias_another_output() {
    use clap::Parser;
    #[derive(Parser)]
    struct Command {
        #[command(flatten)]
        args: CalibrateSloCommand,
    }
    let base = [
        "ferrum",
        "fixture",
        "--manifest",
        "input.json",
        "--slo-config",
        "observe.toml",
        "--observations",
        "raw.jsonl",
        "--out",
        "report.json",
        "--startup-usage",
        "serve",
    ];
    assert!(Command::try_parse_from(base)
        .unwrap()
        .args
        .export_http_inputs
        .is_none());
    let mut cmd = Command::try_parse_from(base.into_iter().chain(["--export-http-inputs", "http"]))
        .unwrap()
        .args;
    let fixture = Fixture::new();
    cmd.out = fixture.directory.path().join("report.json");
    cmd.observations = fixture.directory.path().join("raw.jsonl");
    cmd.export_http_inputs = Some(fixture.directory.path().join("http"));
    paths::validate(paths::outputs(&cmd, &fixture.manifest)).unwrap();
    cmd.export_http_inputs = Some(cmd.out.clone());
    assert!(paths::validate(paths::outputs(&cmd, &fixture.manifest)).is_err());
}
