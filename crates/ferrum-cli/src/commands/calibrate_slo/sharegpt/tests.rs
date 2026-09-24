use super::*;
use ferrum_bench_core::dataset::{ShareGptCounts, ShareGptFilter, ShareGptSelection};
use ferrum_bench_core::BenchmarkPhase;

struct Fixture {
    _directory: tempfile::TempDir,
    source: FrozenShareGpt,
    evidence: ShareGptDatasetEvidence,
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
        let prompt = "  a a\na  ";
        let assistant = "a a a a";
        // Blank rows do not consume indices; malformed nonblank rows do.
        let data = format!(
            "\n{{bad\n{}\n",
            serde_json::json!({"id":"selected", "conversations":[
                {"from":"human", "value":prompt}, {"from":"gpt", "value":assistant}
            ]})
        );
        let dataset_path = directory.path().join("dataset.jsonl");
        std::fs::write(&dataset_path, data.as_bytes()).unwrap();
        let sample = ShareGptSample {
            source_record_index: 1,
            original_id: Some("selected".into()),
            phase: BenchmarkPhase::Measured,
            request_index: 0,
            prompt_sha256: hex(&Sha256::digest(prompt.as_bytes())),
            assistant_sha256: hex(&Sha256::digest(assistant.as_bytes())),
            input_tokens: 3,
            reference_output_tokens: 4,
            requested_output_tokens: 7,
        };
        let samples = vec![sample];
        let selection_hash: [u8; 32] = Sha256::digest(serde_json::to_vec(&samples).unwrap()).into();
        let policy = RequestPolicy {
            requested_model_name: "public-model".into(),
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
            source_sha256: hex(&Sha256::digest(data.as_bytes())),
            source_format: "jsonl".into(),
            tokenizer_sha256: hex(&Sha256::digest(std::fs::read(&tokenizer_path).unwrap())),
            filter: ShareGptFilter {
                min_input_tokens: 1,
                max_input_tokens: Some(8),
                min_output_tokens: 1,
                max_output_tokens: Some(8),
                max_total_tokens: Some(16),
                chat_template_reserve_tokens: 1,
                fixed_output_tokens: Some(7),
            },
            counts: ShareGptCounts {
                records: 2,
                malformed_json: 1,
                eligible: 1,
                ..Default::default()
            },
            prompt_seed: 3,
            sampling: "without_replacement".into(),
            ignore_eos: policy.ignore_eos,
            enable_thinking: policy.enable_thinking,
            repeats: vec![ShareGptSelection {
                repeat_index: 0,
                rng_seed: 3,
                selection_sha256: hex(&selection_hash),
                samples,
            }],
        };
        let source = FrozenShareGpt {
            report_path: directory.path().join("report.json"),
            report_sha256: [1; 32],
            dataset_path,
            tokenizer_path,
            repeat_index: 0,
            selection_sha256: selection_hash,
            request_policy: policy,
            read_limits: ShareGptReadLimits::default(),
        };
        let mut fixture = Self {
            _directory: directory,
            source,
            evidence,
        };
        fixture.write_report();
        fixture
    }

    fn write_report(&mut self) {
        let selection = &mut self.evidence.repeats[0];
        self.source.selection_sha256 =
            Sha256::digest(serde_json::to_vec(&selection.samples).unwrap()).into();
        selection.selection_sha256 = hex(&self.source.selection_sha256);
        let bytes = serde_json::to_vec(&serde_json::json!({
            "dataset_evidence":self.evidence,
            "env":{"http_request_sampling":self.source.request_policy.sampling}
        }))
        .unwrap();
        self.source.report_sha256 = Sha256::digest(&bytes).into();
        std::fs::write(&self.source.report_path, bytes).unwrap();
    }

    fn error(&self) -> String {
        match load(&self.source) {
            Err(error) => error.to_string(),
            Ok(_) => panic!("invalid frozen selection was accepted"),
        }
    }
}

#[test]
fn replay_preserves_selected_raw_text_indices_and_full_output_budget() {
    let fixture = Fixture::new();
    let recovered = load(&fixture.source).unwrap();
    assert_eq!(recovered.prompts.len(), 1);
    assert_eq!(recovered.prompts[0].text, "  a a\na  ");
    assert_eq!(
        recovered.prompts[0].sample,
        fixture.evidence.repeats[0].samples[0]
    );
    assert_eq!(recovered.prompts[0].sample.requested_output_tokens, 7);
    assert_eq!(
        recovered.provenance["historical_server_token_digest"],
        "unavailable_unless_separately_recorded"
    );
}

#[test]
fn replay_checks_all_pinned_files_and_request_policy() {
    let mut fixture = Fixture::new();
    fixture.source.request_policy.ignore_eos = false;
    assert!(fixture.error().contains("request policy differs"));
    fixture.source.request_policy.ignore_eos = true;
    fixture.source.report_sha256[0] ^= 1;
    assert!(fixture.error().contains("report digest differs"));
    fixture.write_report();
    let original = std::fs::read(&fixture.source.tokenizer_path).unwrap();
    std::fs::write(&fixture.source.tokenizer_path, b"{}").unwrap();
    assert!(fixture.error().contains("tokenizer digest differs"));
    std::fs::write(&fixture.source.tokenizer_path, original).unwrap();
    let mut data = std::fs::read(&fixture.source.dataset_path).unwrap();
    data.push(b'\n');
    std::fs::write(&fixture.source.dataset_path, data).unwrap();
    assert!(fixture
        .error()
        .contains("dataset bytes, format or record count differ"));
}

#[test]
fn replay_rejects_inconsistent_output_budget_and_duplicate_source_owners() {
    let mut fixture = Fixture::new();
    fixture.evidence.repeats[0].samples[0].requested_output_tokens = 2;
    fixture.write_report();
    assert!(fixture
        .error()
        .contains("lengths/filter/output policy differ"));
    let mut fixture = Fixture::new();
    let mut duplicate = fixture.evidence.repeats[0].samples[0].clone();
    duplicate.request_index = 1;
    fixture.evidence.repeats[0].samples.push(duplicate);
    fixture.write_report();
    assert!(fixture
        .error()
        .contains("duplicate or unordered request identities"));
}

#[test]
fn request_policy_requires_explicit_optional_fields() {
    let fixture = Fixture::new();
    let value = serde_json::to_value(&fixture.source.request_policy).unwrap();
    serde_json::from_value::<RequestPolicy>(value.clone()).unwrap();
    for field in [
        "enable_thinking",
        "requested_model_name",
        "reasoning_effort",
        "server_default_enable_thinking",
    ] {
        let mut missing = value.clone();
        missing.as_object_mut().unwrap().remove(field);
        assert!(
            serde_json::from_value::<RequestPolicy>(missing).is_err(),
            "{field}"
        );
    }
}

#[test]
fn tagged_report_label_never_replaces_explicit_wire_model() {
    let fixture = Fixture::new();
    let mut value: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&fixture.source.report_path).unwrap()).unwrap();
    value["model"] = "another-model#arbitrary-tag".into();
    let bytes = serde_json::to_vec(&value).unwrap();
    std::fs::write(&fixture.source.report_path, &bytes).unwrap();
    let mut source = fixture.source;
    source.report_sha256 = Sha256::digest(&bytes).into();
    let recovered = load(&source).unwrap();
    assert_eq!(
        recovered.provenance["report_model_label"],
        "another-model#arbitrary-tag"
    );
    assert_eq!(
        recovered.provenance["request_policy"]["requested_model_name"],
        "public-model"
    );
    assert_eq!(
        recovered.provenance["requested_model_name_evidence"],
        "explicit_manifest_declaration_report_label_may_include_tag"
    );
    source.request_policy.requested_model_name.clear();
    assert!(source.validate().is_err());
    source.request_policy.requested_model_name = " padded ".into();
    assert!(source.validate().is_err());
    source.request_policy.requested_model_name = "m".repeat(4097);
    assert!(source.validate().is_err());
}

#[test]
fn actual_chat_conversion_and_codec_keep_wire_name_separate_from_engine_model() {
    use ferrum_interfaces::output_flow::{OutputProjectionContract, RequestOutputPlan};
    use ferrum_server::chat_template::ModelChatTemplate;
    use std::sync::Arc;
    let mut fixture = Fixture::new();
    // Equal raw byte length to internal, but two extra JSON escaping bytes.
    fixture.source.request_policy.requested_model_name = "public\"\\".into();
    let mut manifest = super::super::tests::manifest();
    manifest.schema_version = 2;
    manifest.sharegpt = Some(fixture.source.clone());
    manifest.prompts.clear();
    manifest.protocol.output = manifest::Codec::ChatSse {
        include_usage: true,
    };
    manifest.validate().unwrap();
    let template = ModelChatTemplate::new(
        "{% for message in messages %}{{ message['role'] }}={{ message['content'] }};{% endfor %}assistant=",
        "typed-fixture",
    );
    let inputs = inputs::PreparedInputs::prepare(&manifest, template).unwrap();
    let model_id = ferrum_types::ModelId("internal".into());
    let (request, evidence) = inputs
        .request(
            &manifest,
            0,
            &model_id,
            ferrum_types::ModelOutputProtocol::Text,
        )
        .unwrap();
    assert_eq!(request.model_id, model_id);
    assert_eq!(request.prompt, "user=  a a\na  ;assistant=");
    assert_eq!(request.sampling_params.max_tokens, 7);
    assert_eq!(request.metadata["ferrum_ignore_eos"], true);
    assert!(matches!(
        request.api_request.as_ref(),
        Some(ferrum_types::ApiRequest::Chat(_))
    ));
    assert_eq!(evidence["requested_model_name"], "public\"\\");
    assert_eq!(evidence["resolved_engine_model_id"], "internal");
    let body = super::super::super::chat_request::chat_completion_body(
        "public\"\\",
        "  a a\na  ",
        7,
        true,
        Some(false),
        None,
        fixture.source.request_policy.sampling,
    );
    assert_eq!(
        evidence["chat_body_sha256"],
        hex(&Sha256::digest(serde_json::to_vec(&body).unwrap()))
    );
    let tokenizer = EnvelopeTokenizer(ferrum_types::SpecialTokens::default());
    let descriptor = |contract| {
        RequestOutputPlan::derive(Arc::new(contract), &tokenizer, &request, 3)
            .unwrap()
            .codec_descriptor()
            .unwrap()
    };
    let actual = descriptor(inputs.output_contract(manifest.protocol.output, &request));
    let http = descriptor(OutputProjectionContract::chat_sse(
        request.id.to_string(),
        "public\"\\".into(),
        true,
    ));
    let wrong_internal = descriptor(OutputProjectionContract::chat_sse(
        request.id.to_string(),
        "internal".into(),
        true,
    ));
    assert_eq!(actual, http);
    assert_eq!(
        actual.max_data_envelope_bytes,
        wrong_internal.max_data_envelope_bytes + 2
    );
    assert_ne!(actual, wrong_internal);
    // Schema 1 continues to use its original internal name and codec path.
    let rendered = inputs::PreparedInputs::Rendered;
    let raw_manifest = super::super::tests::manifest();
    let (raw, _) = rendered
        .request(
            &raw_manifest,
            0,
            &model_id,
            ferrum_types::ModelOutputProtocol::Text,
        )
        .unwrap();
    assert_eq!(raw.sampling_params.max_tokens, 73);
    assert!(raw.api_request.is_none());
    let raw_plan = RequestOutputPlan::derive(
        Arc::new(rendered.output_contract(raw_manifest.protocol.output, &raw)),
        &tokenizer,
        &raw,
        3,
    )
    .unwrap();
    assert_eq!(
        raw_plan.codec_descriptor().unwrap().kind,
        ferrum_interfaces::output_flow::OutputCodecKind::CliText
    );
}

/// Metadata-only tokenizer for the actual envelope/credit derivation. No
/// inference or decoding is simulated by this test.
struct EnvelopeTokenizer(ferrum_types::SpecialTokens);
impl ferrum_interfaces::Tokenizer for EnvelopeTokenizer {
    fn encode(&self, _: &str, _: bool) -> Result<Vec<ferrum_types::TokenId>> {
        unreachable!()
    }
    fn decode(&self, _: &[ferrum_types::TokenId], _: bool) -> Result<String> {
        unreachable!()
    }
    fn decode_incremental(
        &self,
        _: &[ferrum_types::TokenId],
        _: ferrum_types::TokenId,
    ) -> Result<String> {
        unreachable!()
    }
    fn vocab_size(&self) -> usize {
        2
    }
    fn special_tokens(&self) -> &ferrum_types::SpecialTokens {
        &self.0
    }
    fn token_id(&self, _: &str) -> Option<ferrum_types::TokenId> {
        None
    }
    fn token_text(&self, _: ferrum_types::TokenId) -> Option<&str> {
        None
    }
    fn info(&self) -> ferrum_interfaces::tokenizer::TokenizerInfo {
        unreachable!()
    }
    fn bounded_decode_bound(&self) -> Option<ferrum_interfaces::tokenizer::BoundedDecodeBound> {
        Some(ferrum_interfaces::tokenizer::BoundedDecodeBound::new(
            std::num::NonZeroUsize::new(1).unwrap(),
            2,
        ))
    }
    fn bounded_token_bytes_bound(&self) -> Option<std::num::NonZeroUsize> {
        std::num::NonZeroUsize::new(1)
    }
}
