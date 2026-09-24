use super::super::super::{inputs, manifest, report, sharegpt};
use super::*;
use ferrum_types::{ModelId, ModelOutputProtocol};

fn policy() -> ReferenceRequestPolicy {
    ReferenceRequestPolicy::FixedReferenceOutput {
        max_tokens: NonZeroUsize::new(5).unwrap(),
        eos: ReferenceEosPolicy::Ignore,
    }
}
fn manifest_with_reference() -> manifest::Manifest {
    let mut manifest = super::super::tests::manifest();
    let mut config = super::super::tests::config();
    config.request_policy = policy();
    manifest.reference = Some(config);
    manifest
}
fn evidence() -> CalibrationRequestEvidence {
    CalibrationRequestEvidence {
        original_input_tokens: 11,
        original_input_tokens_sha256: [3; 32],
    }
}

#[test]
fn fixed_policy_is_explicit_bounded_and_legacy_default_remains_strict() {
    let mut wire = serde_json::to_value(super::super::tests::config()).unwrap();
    wire.as_object_mut().unwrap().remove("request_policy");
    let legacy: ReferenceConfig = serde_json::from_value(wire).unwrap();
    assert!(matches!(
        legacy.request_policy,
        ReferenceRequestPolicy::OriginalInput {}
    ));
    for invalid in [
        serde_json::json!({"kind":"fixed_reference_output","max_tokens":5}),
        serde_json::json!({"kind":"fixed_reference_output","eos":"ignore"}),
        serde_json::json!({"kind":"fixed_reference_output","max_tokens":0,"eos":"ignore"}),
        serde_json::json!({"kind":"fixed_reference_output","max_tokens":5,"eos":"inherit"}),
        serde_json::json!({"kind":"original_input","max_tokens":5}),
    ] {
        assert!(serde_json::from_value::<ReferenceRequestPolicy>(invalid).is_err());
    }
    assert!(ReferenceRequestPolicy::FixedReferenceOutput {
        max_tokens: NonZeroUsize::new(1_048_577).unwrap(),
        eos: ReferenceEosPolicy::Respect,
    }
    .validate()
    .is_err());

    let mut manifest = manifest_with_reference();
    manifest.prompts.push(manifest.prompts[0].clone());
    manifest.prompts[1].sampling.max_tokens = 74;
    let config = manifest.reference.as_mut().unwrap();
    config.curve_prompt_indices.push(1);
    let config = config.clone();
    config
        .validate(&manifest, &inputs::PreparedInputs::Rendered)
        .unwrap();
    let mut legacy = config.clone();
    legacy.request_policy = ReferenceRequestPolicy::OriginalInput {};
    assert!(legacy
        .validate(&manifest, &inputs::PreparedInputs::Rendered)
        .is_err());
    let mut unreachable = config;
    unreachable.decode_unit.generated_before = NonZeroU32::new(5).unwrap();
    assert!(unreachable
        .validate(&manifest, &inputs::PreparedInputs::Rendered)
        .is_err());
}

fn check_phases(inputs: &inputs::PreparedInputs, manifest: &manifest::Manifest) {
    let model = ModelId("actual-model".into());
    for (index, original_maximum) in [(0, 73), (1, 74)] {
        let (original, _) = inputs
            .request(manifest, index, &model, ModelOutputProtocol::Text)
            .unwrap();
        for phase in [
            report::Phase::Warmup,
            report::Phase::Discovery,
            report::Phase::Reference,
            report::Phase::Training,
            report::Phase::Heldout,
        ] {
            let (actual, source) = inputs
                .request_for_phase(manifest, index, &model, ModelOutputProtocol::Text, phase)
                .unwrap();
            assert_eq!(actual.prompt, original.prompt);
            if policy().independent(phase) {
                assert_eq!(actual.sampling_params.max_tokens, 5);
                assert_eq!(actual.metadata["ferrum_ignore_eos"], true);
                assert_eq!(source["source_maximum_output_tokens"], original_maximum);
                assert_eq!(source["reference_maximum_output_tokens"], 5);
                assert_eq!(source["kind"], "independent_reference_request_v1");
                assert_ne!(actual.id, original.id);
            } else {
                assert_eq!(actual.sampling_params.max_tokens, original_maximum);
                assert_eq!(
                    actual.metadata.get("ferrum_ignore_eos"),
                    original.metadata.get("ferrum_ignore_eos")
                );
                assert_ne!(source["kind"], "independent_reference_request_v1");
            }
        }
    }
}

#[test]
fn rendered_reference_uses_new_budget_but_service_73_74_and_sampling_are_unchanged() {
    let mut manifest = manifest_with_reference();
    manifest.prompts.push(manifest.prompts[0].clone());
    manifest.prompts[1].sampling.max_tokens = 74;
    check_phases(&inputs::PreparedInputs::Rendered, &manifest);
    assert_eq!(manifest.prompts[0].sampling.max_tokens, 73);
    assert_eq!(manifest.prompts[1].sampling.max_tokens, 74);
}

#[test]
fn chat_reference_reuses_real_template_and_keeps_actual_api_body_separate() {
    use ferrum_bench_core::{dataset::ShareGptSample, env::HttpRequestSampling, BenchmarkPhase};
    use ferrum_server::chat_template::ModelChatTemplate;
    let prompts = [(0, 73), (1, 74)]
        .into_iter()
        .map(|(index, requested)| sharegpt::RecoveredPrompt {
            text: format!("same pinned source {index}"),
            sample: ShareGptSample {
                source_record_index: u64::from(index),
                original_id: Some(format!("source-{index}")),
                phase: BenchmarkPhase::Measured,
                request_index: index,
                prompt_sha256: "typed factory fixture".into(),
                assistant_sha256: "typed factory fixture".into(),
                input_tokens: 3,
                reference_output_tokens: requested,
                requested_output_tokens: requested,
            },
        })
        .collect();
    let inputs = inputs::PreparedInputs::ShareGpt {
        // This tests the product factory, not recovery/hash validation; that has
        // separate real-file tests. It does not rewrite a production selection.
        recovered: sharegpt::Recovered { prompts, provenance: serde_json::json!({"kind":"typed_factory_fixture"}) },
        template: ModelChatTemplate::new(
            "{% for message in messages %}{{ message['role'] }}={{ message['content'] }};{% endfor %}assistant=",
            "typed-reference-fixture"),
        policy: sharegpt::RequestPolicy {
            requested_model_name: "public-model".into(),
            sampling: HttpRequestSampling::default(), ignore_eos: false,
            enable_thinking: Some(false), reasoning_effort: None,
            server_default_enable_thinking: None, interleaved_system_coalescing: true,
        },
    };
    let manifest = manifest_with_reference();
    check_phases(&inputs, &manifest);
    let (request, source) = inputs
        .request_for_phase(
            &manifest,
            0,
            &ModelId("actual-model".into()),
            ModelOutputProtocol::Text,
            report::Phase::Reference,
        )
        .unwrap();
    assert_eq!(request.prompt, "user=same pinned source 0;assistant=");
    assert!(matches!(
        request.api_request,
        Some(ferrum_types::ApiRequest::Chat(_))
    ));
    assert_ne!(
        source["source_request"]["chat_body_sha256"],
        source["reference_request"]["chat_body_sha256"]
    );
    let (original, _) = inputs
        .request(
            &manifest,
            0,
            &ModelId("actual-model".into()),
            ModelOutputProtocol::Text,
        )
        .unwrap();
    assert_eq!(original.sampling_params.max_tokens, 73);
    assert_eq!(original.metadata.get("ferrum_ignore_eos"), None);
}

#[test]
fn actual_input_ledger_checks_length_digest_and_distinguishes_unobserved_original_policy() {
    let mut ledger = InputIdentityLedger::new(2).unwrap();
    ledger.observe(0, evidence(), true).unwrap();
    let state = serde_json::to_value(&ledger).unwrap();
    assert_eq!(state["entries"][0]["independent_reference_observed"], true);
    assert_eq!(state["entries"][0]["original_policy_observed"], false);
    for changed in [
        CalibrationRequestEvidence {
            original_input_tokens: 12,
            ..evidence()
        },
        CalibrationRequestEvidence {
            original_input_tokens_sha256: [4; 32],
            ..evidence()
        },
    ] {
        assert!(ledger.observe(0, changed, false).is_err());
        assert!(!ledger.entries[0].original_policy_observed);
    }
    ledger.observe(0, evidence(), false).unwrap();
    assert!(ledger.entries[0].original_policy_observed);
    ledger.observe(1, evidence(), true).unwrap();
    assert!(ledger.observe(2, evidence(), true).is_err());
    assert_eq!(ledger.entries.len(), 2);
    assert!(InputIdentityLedger::new(4097).is_err());
}

#[test]
fn explicit_reference_eos_does_not_clear_user_stops_or_rewrite_sampling() {
    let mut manifest = manifest_with_reference();
    manifest.prompts[0].sampling.stop_sequences = vec!["explicit-user-stop".into()];
    manifest.prompts[0].sampling.temperature = 0.25;
    let inputs = inputs::PreparedInputs::Rendered;
    for (eos, ignored) in [
        (ReferenceEosPolicy::Respect, false),
        (ReferenceEosPolicy::Ignore, true),
    ] {
        manifest.reference.as_mut().unwrap().request_policy =
            ReferenceRequestPolicy::FixedReferenceOutput {
                max_tokens: NonZeroUsize::new(5).unwrap(),
                eos,
            };
        let (request, _) = inputs
            .request_for_phase(
                &manifest,
                0,
                &ModelId("actual-model".into()),
                ModelOutputProtocol::Text,
                report::Phase::Reference,
            )
            .unwrap();
        assert_eq!(request.metadata["ferrum_ignore_eos"], ignored);
        assert_eq!(
            request.sampling_params.stop_sequences,
            ["explicit-user-stop"]
        );
        assert_eq!(request.sampling_params.temperature, 0.25);
        assert_eq!(request.sampling_params.max_tokens, 5);
    }
    assert_eq!(manifest.prompts[0].sampling.max_tokens, 73);
}
