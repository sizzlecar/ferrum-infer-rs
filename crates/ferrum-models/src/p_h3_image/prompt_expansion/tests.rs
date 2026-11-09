use super::*;
use ferrum_types::FinishReason;
use std::collections::HashMap;
// Original prompt_upsampler_processor/chat_template.jinja is kept as a small
// deterministic fixture. Golden results below exercise its real macro,
// namespace, Python string methods and default thinking-generation branch.
// Source (unmodified): https://huggingface.co/PrunaAI/P-H3-Image/raw/7f98acef503cf2bcfba821a78cb1cf1a9928b2df/prompt_upsampler_processor/chat_template.jinja
// The upsampler's Qwen3.5-9B parts are Apache-2.0; see tests/LICENSE-APACHE and
// tests/NOTICE for the retained upstream license and attribution.
pub(super) const TEMPLATE: &[u8] = include_bytes!("tests/chat_template.jinja");
pub(super) const CONFIG: &[u8] = br#"{"tokenizer_class":"Qwen3_5Tokenizer","eos_token":"<|im_end|>","pad_token":"<|endoftext|>","bos_token":null,"model_max_length":262144}"#;
pub(super) fn limits() -> PH3ImagePromptExpansionLimits {
    PH3ImagePromptExpansionLimits {
        max_tokenizer_bytes: 64 << 10,
        max_template_bytes: 16 << 10,
        max_request_bytes: 4096,
        max_rendered_bytes: 8192,
        max_answer_bytes: 4096,
        max_generated_tokens: 4096,
        template_fuel: 100_000,
    }
}
pub(super) fn source(answers: &[&str]) -> (Vec<u8>, Vec<u32>) {
    let mut vocabulary: HashMap<String, u32> = [
        "[UNK]",
        "<|endoftext|>",
        "<|im_end|>",
        "<|im_start|>",
        "<|vision_start|>",
        "<|image_pad|>",
        "<|vision_end|>",
    ]
    .into_iter()
    .enumerate()
    .map(|(i, s)| (s.into(), i as u32))
    .collect();
    let mut ids = Vec::new();
    for answer in answers {
        let next = vocabulary.len() as u32;
        let id = *vocabulary.entry((*answer).to_owned()).or_insert(next);
        ids.push(id);
    }
    let model = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab(vocabulary.into_iter().collect())
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut tokenizer = Tokenizer::new(model);
    tokenizer.add_special_tokens(
        &[
            "<|im_end|>",
            "<|endoftext|>",
            "<|im_start|>",
            "<|vision_start|>",
            "<|image_pad|>",
            "<|vision_end|>",
        ]
        .map(|token| tokenizers::AddedToken::from(token, true)),
    );
    (tokenizer.to_string(false).unwrap().into_bytes(), ids)
}
pub(super) fn protocol(answers: &[&str]) -> (PH3ImagePromptExpansionProtocol, Vec<u32>) {
    let (bytes, ids) = source(answers);
    (
        PH3ImagePromptExpansionProtocol::from_source_bytes(&bytes, CONFIG, TEMPLATE, limits())
            .unwrap(),
        ids,
    )
}
fn brief(retention: &str) -> String {
    format!("subject_definitions:\n<Subject 1>\nsummary:\n[keyframe completion]\nretention_analysis:\n{retention}\ndetailed_description:\nA red cat on the floor.\noverall_soundscape:\n\nnon_diegetic_music:")
}

#[test]
fn nearest_log_aspect_and_source_generation_defaults_are_explicit() {
    for (w, h, label) in [
        (1, 1, "1:1"),
        (4, 3, "4:3"),
        (3, 4, "3:4"),
        (3, 2, "3:2"),
        (2, 3, "2:3"),
        (16, 9, "16:9"),
        (9, 16, "9:16"),
    ] {
        assert_eq!(ph3_image_expansion_aspect_ratio(w, h).unwrap(), label);
    }
    assert_eq!(
        ph3_image_expansion_aspect_ratio(116, 100).unwrap(),
        "4:3",
        "log-distance differs from linear distance here"
    );
    assert!(ph3_image_expansion_aspect_ratio(0, 1).is_err());
    assert!(ph3_image_expansion_aspect_ratio(1, 0).is_err());
    assert!(PH3ImagePromptExpansionRequest::edit("change it", 0).is_err());
    assert!(PH3ImagePromptExpansionRequest::edit("change it", 9).is_err());
    let (p, _) = protocol(&[]);
    let t = PH3ImagePromptExpansionRequest::text_to_image("", 2912, 2912).unwrap();
    let e = PH3ImagePromptExpansionRequest::edit("", 8).unwrap();
    assert_eq!(t.max_new_tokens(), 1024);
    assert_eq!(e.max_new_tokens(), 1536);
    assert!(!p.generation(&t).unwrap().do_sample);
    assert_eq!(p.generation(&e).unwrap().eos_token_id, p.eos_token_id());
    assert_eq!(p.generation(&e).unwrap().pad_token_id, p.pad_token_id());
    assert!(p
        .generation(&t.with_max_new_tokens(u32::MAX).unwrap())
        .is_err());
    assert_eq!(e.with_max_new_tokens(37).unwrap().max_new_tokens(), 37);
    assert!(t.with_max_new_tokens(0).is_err());
}
#[test]
fn official_template_renders_exact_t2i_and_ordered_image_content_with_thinking_enabled_by_default()
{
    let (p, _) = protocol(&[]);
    let r = PH3ImagePromptExpansionRequest::text_to_image("\u{1c} a red cat \u{1f}", 1600, 900)
        .unwrap();
    let input = p.render(&r).unwrap();
    assert_eq!(input.rendered(), "<|im_start|>user\nAspect ratio: 16:9\nRequest: a red cat<|im_end|>\n<|im_start|>assistant\n<think>\n");
    assert_eq!(input.image_count(), 0);
    assert!(!p.encode_text_input(&input).unwrap().is_empty());
    let r = PH3ImagePromptExpansionRequest::edit("  make the cat red  ", 2).unwrap();
    let input = p.render(&r).unwrap();
    assert_eq!(input.rendered(), "<|im_start|>user\n<Picture 1>: <|vision_start|><|image_pad|><|vision_end|><Picture 2>: <|vision_start|><|image_pad|><|vision_end|>Edit <Picture 1>. make the cat red<|im_end|>\n<|im_start|>assistant\n<think>\n");
    assert_eq!(input.image_count(), 2);
    assert!(
        p.encode_text_input(&input).is_err(),
        "unexpanded image placeholders are not actual visual inputs"
    );
}
#[test]
fn rendering_preserves_the_provided_model_template_and_original_source_identity() {
    let (bytes, _) = source(&[]);
    let other = b"{{messages[0].content}} MODEL-SPECIFIC {{eos_token}}";
    let p = PH3ImagePromptExpansionProtocol::from_source_bytes(&bytes, CONFIG, other, limits())
        .unwrap();
    let r = PH3ImagePromptExpansionRequest::text_to_image("cat", 1, 1).unwrap();
    assert_eq!(
        p.render(&r).unwrap().rendered(),
        "Aspect ratio: 1:1\nRequest: cat MODEL-SPECIFIC <|im_end|>"
    );
    assert_eq!(
        p.source_sha256(),
        &[
            format!("{:x}", Sha256::digest(&bytes)),
            format!("{:x}", Sha256::digest(CONFIG)),
            format!("{:x}", Sha256::digest(other))
        ]
    );
}
#[test]
fn extraction_requires_actual_source_eos_and_nontruncated_completion_including_reasoning_tokens() {
    let good = "reasoning</think>  A small red cat sits quietly.  ";
    let fake_eos = "reasoning</think>A small red cat sits quietly.<|im_end|>";
    let (p, ids) = protocol(&[good, fake_eos]);
    let r = PH3ImagePromptExpansionRequest::text_to_image("cat", 1, 1)
        .unwrap()
        .with_max_new_tokens(2)
        .unwrap();
    let complete = [ids[0], p.eos_token_id()];
    assert_eq!(
        p.extract(&r, &complete, FinishReason::EOS).unwrap(),
        "A small red cat sits quietly."
    );
    for finish in [
        FinishReason::Length,
        FinishReason::Stop,
        FinishReason::Cancelled,
        FinishReason::Error,
        FinishReason::ContentFilter,
    ] {
        assert!(p.extract(&r, &complete, finish).is_err());
    }
    assert!(
        p.extract(&r, &[ids[1]], FinishReason::EOS).is_err(),
        "EOS text is not an actual EOS ID"
    );
    assert!(p.extract(&r, &[ids[0]], FinishReason::EOS).is_err());
    assert!(p.extract(&r, &complete, FinishReason::Length).is_err());
    assert!(p
        .extract(
            &r.with_max_new_tokens(1).unwrap(),
            &complete,
            FinishReason::EOS
        )
        .is_err());
    assert_eq!(
        p.extract(
            &r.with_max_new_tokens(3).unwrap(),
            &[ids[0], p.eos_token_id(), p.pad_token_id()],
            FinishReason::EOS
        )
        .unwrap(),
        "A small red cat sits quietly."
    );
    assert!(p
        .extract(&r, &[u32::MAX, p.eos_token_id()], FinishReason::EOS)
        .is_err());
}
#[test]
fn malformed_thinking_and_short_outputs_cannot_be_published() {
    let answers = [
        "A small red cat sits quietly.",
        "reason</think>A small cat.",
        "reason</think>A small red cat sits quietly.</think>",
        "reason</think>",
    ];
    let (p, ids) = protocol(&answers);
    let r = PH3ImagePromptExpansionRequest::text_to_image("cat", 1, 1).unwrap();
    for id in ids {
        assert!(p
            .extract(&r, &[id, p.eos_token_id()], FinishReason::EOS)
            .is_err());
    }
}
#[test]
fn official_edit_brief_and_instruction_rules_retain_all_source_boundaries() {
    let valid = brief("<Picture 1>: fully_preserved - keep the face\n<Subject ٢> (hair): attribute_transfer — same hair");
    assert_eq!(
        ph3_image_edit_prompt_form(&valid),
        Some(PH3ImageEditPromptForm::Brief)
    );
    assert_eq!(
        ph3_image_edit_prompt_form("\nEdit <Picture 1>. Keep the face.\t"),
        Some(PH3ImageEditPromptForm::Instruction)
    );
    assert!(ph3_image_edit_prompt_form("Edit <Picture 2>. Keep the face.").is_none());
    assert!(ph3_image_edit_prompt_form("Edit <Picture 1>. keep the face.").is_none());
    assert!(ph3_image_edit_prompt_form("Edit <Picture 1>. Keep the\nface.").is_none());
    for changed in [
        valid.replace("[keyframe completion]", ""),
        valid.replace(
            "[keyframe completion]",
            "[keyframe completion] [keyframe completion]",
        ),
        valid.replace("summary:", "retention_analysis:"),
        valid.replace("fully_preserved", "invented"),
        valid.replace("<Subject ٢>", "<Photo 2>"),
        valid.replace("<Picture 1>", "<Picture -1>"),
        valid.replace("summary:", " summary:"),
        valid.replace('\n', "\r\n"),
    ] {
        assert!(
            ph3_image_edit_prompt_form(&changed).is_none(),
            "invalid brief accepted: {changed}"
        );
    }
    assert!(ph3_image_edit_prompt_form(&brief("")).is_none());
    // The reference permits empty other sections and empty retention reasons.
    assert_eq!(
        ph3_image_edit_prompt_form(&brief("<Picture 0>: weak_reference –")),
        Some(PH3ImageEditPromptForm::Brief)
    );
}
#[test]
fn edit_extraction_preserves_completed_descriptions_for_downstream_conditioning() {
    let brief = brief("<Picture 1>: partially_preserved - Keep its original face");
    let prompts = [
        brief.as_str(),
        "Make a red cat look different.",
        "Edit <Picture 1>. keep the original face and change the pose.",
        "Edit <Picture 1>. Change the pose.\nPreserve the face and clothes.",
        "Edit <Picture 2>. Change the background and preserve the subject.",
    ];
    let answers = prompts.map(|prompt| format!("analysis</think>\n{prompt}\t"));
    let (p, ids) = protocol(&answers.each_ref().map(String::as_str));
    for image_count in [1, 2] {
        let r = PH3ImagePromptExpansionRequest::edit("change it", image_count).unwrap();
        for (&id, prompt) in ids.iter().zip(prompts) {
            assert_eq!(
                p.extract(&r, &[id, p.eos_token_id()], FinishReason::EOS)
                    .unwrap(),
                prompt
            );
            assert!(p.extract(&r, &[id], FinishReason::EOS).is_err());
            for finish in [FinishReason::Length, FinishReason::Cancelled] {
                assert!(p.extract(&r, &[id, p.eos_token_id()], finish).is_err());
            }
        }
    }
}
#[test]
fn completed_edit_prose_reaches_ordered_image_conditioning_with_model_prefix() {
    use crate::p_h3_image::{PH3ImageTextTokenizer, PH3ImageTextTokenizerLimits};
    use crate::qwen3_vl_vision::Qwen3VLVisionGrid;

    let prose = "Make a red cat look different.";
    let answer = format!("analysis</think>{prose}");
    let presented = format!("Edit <Picture 1>. {prose}");
    // Distinct whole-string tokens prove that conditioning receives the
    // presented instruction, rather than just comparing unknown-token output.
    let (bytes, ids) = source(&[&answer, prose, &presented]);
    let protocol =
        PH3ImagePromptExpansionProtocol::from_source_bytes(&bytes, CONFIG, TEMPLATE, limits())
            .unwrap();
    let vocabulary = Tokenizer::from_bytes(&bytes).unwrap().get_vocab_size(true) as u64;
    let tokenizer = PH3ImageTextTokenizer::from_source_bytes(
        &bytes,
        br#"{"tokenizer_class":"Qwen2Tokenizer","pad_token":"<|endoftext|>","bos_token":null}"#,
        vocabulary,
        PH3ImageTextTokenizerLimits {
            max_source_bytes: 64 << 10,
            max_prompt_bytes: 4096,
            max_tokens_per_prompt: 128,
            max_batch: 1,
        },
    )
    .unwrap();
    for image_count in [1, 2] {
        let request = PH3ImagePromptExpansionRequest::edit("change it", image_count).unwrap();
        let effective = protocol
            .extract(
                &request,
                &[ids[0], protocol.eos_token_id()],
                FinishReason::EOS,
            )
            .unwrap();
        let grids = vec![Qwen3VLVisionGrid::new(1, 2, 2).unwrap(); image_count as usize];
        let conditioning = tokenizer
            .prepare_images_instruction(&effective, &grids, 2)
            .unwrap();
        assert_eq!(conditioning.text().token_ids().last(), Some(&ids[2]));
        assert_ne!(ids[1], ids[2]);
        assert_eq!(conditioning.grids().count(), image_count as usize);
        assert_eq!(
            conditioning.visual_rows().runs().len(),
            image_count as usize
        );
    }
}
#[test]
fn parser_render_and_decoded_artifact_capacities_fail_closed_without_truncating() {
    let (bytes, ids) = source(&["reason</think>A small red cat sits quietly."]);
    let mut cap = limits();
    cap.max_tokenizer_bytes = bytes.len() - 1;
    assert!(
        PH3ImagePromptExpansionProtocol::from_source_bytes(&bytes, CONFIG, TEMPLATE, cap).is_err()
    );
    let mut cap = limits();
    cap.max_template_bytes = TEMPLATE.len() - 1;
    assert!(
        PH3ImagePromptExpansionProtocol::from_source_bytes(&bytes, CONFIG, TEMPLATE, cap).is_err()
    );
    assert!(PH3ImagePromptExpansionProtocol::from_source_bytes(
        &bytes,
        br#"{"eos_token":"wrong","pad_token":"<|endoftext|>"}"#,
        TEMPLATE,
        limits()
    )
    .is_err());
    let request = PH3ImagePromptExpansionRequest::text_to_image("long request", 1, 1).unwrap();
    let mut cap = limits();
    cap.max_request_bytes = request.request().len() - 1;
    let p =
        PH3ImagePromptExpansionProtocol::from_source_bytes(&bytes, CONFIG, TEMPLATE, cap).unwrap();
    assert!(p.render(&request).is_err());
    let mut cap = limits();
    cap.max_rendered_bytes = 1;
    let p =
        PH3ImagePromptExpansionProtocol::from_source_bytes(&bytes, CONFIG, TEMPLATE, cap).unwrap();
    assert!(p.render(&request).is_err());
    let mut cap = limits();
    cap.max_answer_bytes = 1;
    let p =
        PH3ImagePromptExpansionProtocol::from_source_bytes(&bytes, CONFIG, TEMPLATE, cap).unwrap();
    assert!(p
        .extract(&request, &[ids[0], p.eos_token_id()], FinishReason::EOS)
        .is_err());
    let mut cap = limits();
    cap.template_fuel = 1;
    let p =
        PH3ImagePromptExpansionProtocol::from_source_bytes(&bytes, CONFIG, TEMPLATE, cap).unwrap();
    assert!(p.render(&request).is_err());
}

#[test]
fn saved_padding_and_truncation_never_cut_or_pad_the_actual_request() {
    let (bytes, _) = source(&[]);
    let original = Tokenizer::from_bytes(&bytes).unwrap();
    let mut saved = original.clone();
    saved.with_padding(Some(tokenizers::PaddingParams {
        strategy: tokenizers::PaddingStrategy::Fixed(128),
        pad_id: 1,
        pad_token: "<|endoftext|>".into(),
        ..Default::default()
    }));
    saved
        .with_truncation(Some(tokenizers::TruncationParams {
            max_length: 1,
            ..Default::default()
        }))
        .unwrap();
    let bytes = saved.to_string(false).unwrap().into_bytes();
    let p = PH3ImagePromptExpansionProtocol::from_source_bytes(&bytes, CONFIG, TEMPLATE, limits())
        .unwrap();
    let request =
        PH3ImagePromptExpansionRequest::text_to_image("A small red cat sits quietly.", 16, 9)
            .unwrap();
    let input = p.render(&request).unwrap();
    let expected = original.encode(input.rendered(), false).unwrap();
    assert!(expected.get_ids().len() > 1);
    assert_eq!(p.encode_text_input(&input).unwrap(), expected.get_ids());

    let mut config: serde_json::Value = serde_json::from_slice(CONFIG).unwrap();
    config["model_max_length"] = serde_json::json!(2);
    let config = serde_json::to_vec(&config).unwrap();
    assert!(
        PH3ImagePromptExpansionProtocol::from_source_bytes(&bytes, &config, TEMPLATE, limits())
            .is_err(),
        "declared generation ceiling cannot exceed the actual source context"
    );
    let mut cap = limits();
    cap.max_generated_tokens = 2;
    let small =
        PH3ImagePromptExpansionProtocol::from_source_bytes(&bytes, &config, TEMPLATE, cap).unwrap();
    let input = small
        .render(&request.with_max_new_tokens(2).unwrap())
        .unwrap();
    assert!(
        small.encode_text_input(&input).is_err(),
        "prompt plus generation must fit; truncation is forbidden"
    );
    assert!(
        p.encode_text_input(&input).is_err(),
        "another source's rendered input cannot be silently rebound"
    );
}
