use super::*;
use crate::qwen3_vl_text::{Qwen3VLTextArchitecture, Qwen3VLTextEncoderFamily};
use ferrum_quantization::SafetensorsArchive;
use ferrum_runtime::*;
use half::bf16;
use serde_json::json;

// Synthetic tokenizer and a deliberately different implementation of the same
// system-only Jinja semantics. Neither fixture is a deployed tokenizer oracle.
fn tokenizer_sources() -> (Vec<u8>, Vec<u8>, Vec<u8>) {
    let vocabulary = [
        ("[UNK]", 0),
        ("system", 1),
        ("Comprehend", 2),
        ("and", 3),
        ("analyze", 4),
        ("the", 5),
        ("provided", 6),
        ("prompt.", 7),
        ("user", 8),
        ("assistant", 9),
        ("cat", 10),
        ("blue", 11),
        ("你好", 12),
        ("é", 13),
    ]
    .into_iter()
    .map(|(s, n)| (s.to_string(), n))
    .collect();
    let model = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab(vocabulary)
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut engine = tokenizers::Tokenizer::new(model);
    engine.with_pre_tokenizer(Some(
        tokenizers::pre_tokenizers::whitespace::WhitespaceSplit,
    ));
    engine.with_post_processor(Some(
        tokenizers::pre_tokenizers::byte_level::ByteLevel::default(),
    ));
    engine.add_special_tokens(
        &[
            "<|endoftext|>",
            "<|im_start|>",
            "<|im_end|>",
            "<|image_pad|>",
            "<|vision_start|>",
            "<|vision_end|>",
        ]
        .map(|t| tokenizers::AddedToken::from(t, true)),
    );
    let config = json!({"tokenizer_class":"Qwen2Tokenizer","processor_class":"Qwen3VLProcessor","bos_token":null,
        "pad_token":"<|endoftext|>","eos_token":"<|im_end|>","add_prefix_space":false,"split_special_tokens":false});
    let template = "{%- for m in messages -%}{{ '<|im_start|>' + m.role + '\\n' }}{%- for c in m.content -%}{{ c.text }}{%- endfor -%}{{ '<|im_end|>\\n' }}{%- endfor -%}";
    (
        engine.to_string(false).unwrap().into_bytes(),
        serde_json::to_vec(&config).unwrap(),
        template.as_bytes().to_vec(),
    )
}
fn tokenizer() -> QwenImage21T2iTokenizer {
    let (t, c, j) = tokenizer_sources();
    QwenImage21T2iTokenizer::from_source_bytes(&t, &c, &j).unwrap()
}
fn architecture() -> Qwen3VLTextArchitecture {
    architecture_with_layers(1)
}
fn architecture_with_layers(layers: u64) -> Qwen3VLTextArchitecture {
    Qwen3VLTextArchitecture::from_hf_value(&json!({"model_type":"qwen3_vl","architectures":["Qwen3VLForConditionalGeneration"],
        "text_config":{"model_type":"qwen3_vl_text","dtype":"bfloat16","hidden_act":"silu","attention_bias":false,"attention_dropout":0.0,
            "num_hidden_layers":layers,"hidden_size":4,"intermediate_size":8,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
            "vocab_size":32,"max_position_embeddings":1024,"rms_norm_eps":0.000001,"rope_theta":100,
            "rope_scaling":{"rope_type":"default","mrope_interleaved":true,"mrope_section":[1,1,0]}}})).unwrap()
}

#[test]
fn edit_prompt_expands_only_the_single_image_run_and_reports_cropped_slots() {
    let tokenizer = tokenizer();
    let small = tokenizer.prepare_edit("cat", 64, 4, 4, 2).unwrap();
    let large = tokenizer.prepare_edit("cat", 64, 8, 8, 2).unwrap();
    assert_eq!(small.image_slots(), 4);
    assert_eq!(large.image_slots(), 16);
    assert_eq!(small.image_slot_start(), large.image_slot_start());
    let start = small.prompts().drop_prefix_tokens() + small.image_slot_start() as usize;
    let small_ids = small.prompts().input_ids();
    let large_ids = large.prompts().input_ids();
    assert_eq!(&small_ids[..start], &large_ids[..start]);
    assert!(small_ids[start..start + 4]
        .iter()
        .all(|&v| v == small_ids[start]));
    assert!(large_ids[start..start + 16]
        .iter()
        .all(|&v| v == small_ids[start]));
    assert_eq!(&small_ids[start + 4..], &large_ids[start + 16..]);
    let config = small.text_config(&architecture_with_layers(3)).unwrap();
    assert_eq!(config.deepstack().unwrap().runs()[0].start(), start as u64);
    assert_eq!(
        small.prompts().feature_sequence_length() + 12,
        large.prompts().feature_sequence_length()
    );
    assert!(tokenizer
        .prepare_edit("cat <|image_pad|>", 64, 4, 4, 2)
        .is_err());
    assert!(tokenizer.prepare_edit("cat", 1, 4, 4, 2).is_err());
    assert!(tokenizer.prepare_edit("cat", 64, 3, 4, 2).is_err());
    assert!(tokenizer.prepare_edit("cat", 64, u64::MAX, 4, 2).is_err());
}

#[test]
fn edit_inputs_copy_output_backings_under_one_host_and_preserve_visual_rows() {
    use crate::qwen3_vl_text::Qwen3VLTextEncoderFamily as Text;
    use crate::qwen3_vl_vision::Qwen3VLVisionFamily as Vision;
    let prompt = tokenizer().prepare_edit("cat", 64, 4, 4, 2).unwrap();
    let a = architecture_with_layers(3);
    let rows = prompt.prompts().sequence_length();
    let visual_bytes = 4 * 4 * 2;
    let input_bytes = rows * 4 * 2 * 3 + visual_bytes * 3;
    let host = account(input_bytes + visual_bytes * 4);
    let spec = ResolvedTensorSpec::new(
        vec![4, 4],
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    let mut outputs = Vec::new();
    for (index, name) in std::iter::once(Vision::OUTPUT_ID)
        .chain(Vision::DEEPSTACK_IDS)
        .enumerate()
    {
        let mut bytes = host
            .allocate_zeroed(HostMemoryClass::Output, visual_bytes)
            .unwrap();
        for (j, b) in bytes.as_mut().chunks_exact_mut(2).enumerate() {
            b.copy_from_slice(&(0x3f00_u16 + index as u16 * 32 + j as u16).to_le_bytes());
        }
        outputs.push((
            ProgramValueId::new(name).unwrap(),
            spec.clone(),
            bytes.freeze(),
        ));
    }
    let table = (0..32 * 4)
        .flat_map(|_| 0x3e80_u16.to_le_bytes())
        .collect::<Vec<_>>();
    let parts = outputs
        .iter()
        .map(|(id, t, b)| (id, t, b))
        .collect::<Vec<_>>();
    let occupied = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 1)
        .unwrap();
    assert!(prompt
        .managed_text_parts(&host, &a, &table, &parts)
        .is_err());
    assert_eq!(
        host.usage().unwrap().committed_bytes(),
        visual_bytes * 4 + 1
    );
    drop(occupied);
    let inputs = prompt
        .managed_text_parts(&host, &a, &table, &parts)
        .unwrap();
    assert_eq!(inputs.len(), 6);
    let start = prompt.prompts().drop_prefix_tokens() + prompt.image_slot_start() as usize;
    let embedding = inputs
        .iter()
        .find(|t| t.value_id().as_str() == Text::EMBEDDINGS_ID)
        .unwrap();
    assert_eq!(
        &embedding.bytes().bytes()[start * 8..(start + 4) * 8],
        outputs[0].2.bytes()
    );
    assert_eq!(&embedding.bytes().bytes()[..2], &0x3e80_u16.to_le_bytes());
    for (input, source) in inputs[3..].iter().zip(&outputs[1..]) {
        assert_eq!(input.bytes().bytes(), source.2.bytes());
        assert_eq!(input.bytes().class(), HostMemoryClass::InputPreprocess);
        assert!(!input.bytes().same_region(&source.2));
    }
    assert_eq!(
        host.usage().unwrap().allocated_bytes(),
        input_bytes + visual_bytes * 4
    );
    let foreign = account(input_bytes + visual_bytes * 4);
    assert!(prompt
        .managed_text_parts(&foreign, &a, &table, &parts)
        .is_err());
    assert_eq!(foreign.usage().unwrap().committed_bytes(), 0);
    drop(parts);
    drop(outputs);
    assert_eq!(host.usage().unwrap().allocated_bytes(), input_bytes);
    drop(inputs);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
#[test]
fn edit_batch_keeps_each_single_reference_span_after_left_padding() {
    let tokenizer = tokenizer();
    let texts = ["cat", "blue cat"];
    let batch = tokenizer.prepare_edit_batch(&texts, 64, 4, 4, 2).unwrap();
    assert_eq!(batch.prompts().batch(), 2);
    let sequence = batch.prompts().sequence_length();
    let runs = batch.text_config(&architecture_with_layers(3)).unwrap();
    let runs = runs.deepstack().unwrap().runs();
    assert_eq!(runs.len(), 2);
    assert!(sequence > batch.prompts().valid_input_lengths()[0]);
    for (index, text) in texts.into_iter().enumerate() {
        let single = tokenizer.prepare_edit(text, 64, 4, 4, 2).unwrap();
        let length = single.prompts().sequence_length();
        let left = sequence - length;
        assert_eq!(
            &batch.prompts().input_ids()[index * sequence + left..(index + 1) * sequence],
            single.prompts().input_ids()
        );
        assert_eq!(batch.image_slot_start(), single.image_slot_start());
        assert_eq!(batch.image_slots(), single.image_slots());
        assert_eq!(
            runs[index].start(),
            (index * sequence
                + left
                + single.prompts().drop_prefix_tokens()
                + single.image_slot_start() as usize) as u64
        );
        assert_eq!(runs[index].length(), 4);
        let explicit_single = tokenizer.prepare_edit_batch(&[text], 64, 4, 4, 2).unwrap();
        assert_eq!(
            explicit_single.prompts().input_ids(),
            single.prompts().input_ids()
        );
        assert_eq!(explicit_single.spans, single.spans);
    }
    assert!(tokenizer.prepare_edit_batch(&[], 64, 4, 4, 2).is_err());
    assert!(tokenizer
        .prepare_edit_batch(
            &["cat"; STRUCTURED_ATTENTION_BF16_MAX_BATCH + 1],
            64,
            4,
            4,
            2
        )
        .is_err());
    assert!(tokenizer
        .prepare_edit_batch(&["cat", "<|image_pad|>"], 64, 4, 4, 2)
        .is_err());
    let oversized = "x".repeat(512 * 1024 + 1);
    assert!(tokenizer
        .prepare_edit_batch(&[&oversized, &oversized], 8192, 4, 4, 2)
        .is_err());
}

type EditVisionPart = (ProgramValueId, ResolvedTensorSpec, ManagedHostBytes);

fn edit_vision_outputs(host: &HostMemoryAccount) -> Vec<Vec<EditVisionPart>> {
    use crate::qwen3_vl_vision::Qwen3VLVisionFamily as Vision;
    (0..2)
        .map(|sample| {
            std::iter::once(Vision::OUTPUT_ID)
                .chain(Vision::DEEPSTACK_IDS)
                .enumerate()
                .map(|(part, id)| {
                    let mut bytes = host.allocate_zeroed(HostMemoryClass::Output, 32).unwrap();
                    for (element, b) in bytes.as_mut().chunks_exact_mut(2).enumerate() {
                        b.copy_from_slice(
                            &(0x3f00_u16 + sample * 256 + part as u16 * 32 + element as u16)
                                .to_le_bytes(),
                        );
                    }
                    (
                        ProgramValueId::new(id).unwrap(),
                        ResolvedTensorSpec::new(
                            vec![4, 4],
                            ElementType::Bf16,
                            ResolvedTensorLayout::Contiguous,
                        )
                        .unwrap(),
                        bytes.freeze(),
                    )
                })
                .collect()
        })
        .collect()
}

fn edit_vision_parts(
    outputs: &[Vec<EditVisionPart>],
) -> Vec<Vec<(&ProgramValueId, &ResolvedTensorSpec, &ManagedHostBytes)>> {
    outputs
        .iter()
        .map(|sample| {
            sample
                .iter()
                .map(|(id, spec, bytes)| (id, spec, bytes))
                .collect()
        })
        .collect()
}

#[test]
fn edit_batch_bridge_keeps_distinct_images_rotary_and_host_capacity() {
    use crate::qwen3_vl_text::Qwen3VLTextEncoderFamily as Text;
    let tokenizer = tokenizer();
    let texts = ["cat", "blue cat"];
    let prompt = tokenizer.prepare_edit_batch(&texts, 64, 4, 4, 2).unwrap();
    let a = architecture_with_layers(3);
    let sequence = prompt.prompts().sequence_length();
    let source_bytes = 2 * 4 * 32;
    let input_bytes = 2 * sequence * 4 * 2 * 3 + 2 * 32 * 3;
    let host = account(source_bytes + input_bytes);
    let outputs = edit_vision_outputs(&host);
    let mut parts = edit_vision_parts(&outputs);
    parts[1].reverse(); // Output IDs, not incidental Vec order, select each level.
    let table = embedding_bytes();
    let occupied = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 1)
        .unwrap();
    assert!(prompt
        .managed_text_batch_parts(&host, &a, &table, &parts)
        .is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), source_bytes + 1);
    drop(occupied);
    let inputs = prompt
        .managed_text_batch_parts(&host, &a, &table, &parts)
        .unwrap();
    assert_eq!(inputs.len(), 6);
    let embedding = inputs
        .iter()
        .find(|t| t.value_id().as_str() == Text::EMBEDDINGS_ID)
        .unwrap();
    for (row, token) in prompt.prompts().input_ids().iter().enumerate() {
        let sample = row / sequence;
        let start = prompt.spans[sample].visual_run().start() as usize;
        let expected = if (start..start + 4).contains(&row) {
            &outputs[sample][0].2.bytes()[(row - start) * 8..(row - start + 1) * 8]
        } else {
            &table[*token as usize * 8..(*token as usize + 1) * 8]
        };
        assert_eq!(&embedding.bytes().bytes()[row * 8..(row + 1) * 8], expected);
    }
    for (part, input) in inputs[3..].iter().enumerate() {
        assert_eq!(input.tensor().dimensions(), [8, 4]);
        for sample in 0..2 {
            assert_eq!(
                &input.bytes().bytes()[sample * 32..(sample + 1) * 32],
                outputs[sample][part + 1].2.bytes()
            );
            assert!(!input.bytes().same_region(&outputs[sample][part + 1].2));
        }
    }
    for (sample, text) in texts.into_iter().enumerate() {
        let single = tokenizer.prepare_edit(text, 64, 4, 4, 2).unwrap();
        let single_host = account(4096);
        let rotary = single
            .text_config(&a)
            .unwrap()
            .managed_edit_mrope(&single_host, &single.spans)
            .unwrap();
        let length = single.prompts().sequence_length();
        let start = (sample * sequence + sequence - length) * 8;
        for row in sample * sequence..sample * sequence + sequence - length {
            assert_eq!(
                bits(&inputs[1].bytes().bytes()[row * 8..(row + 1) * 8]),
                [0x3f80; 4]
            );
            assert_eq!(
                bits(&inputs[2].bytes().bytes()[row * 8..(row + 1) * 8]),
                [0; 4]
            );
        }
        for part in 0..2 {
            assert_eq!(
                &inputs[part + 1].bytes().bytes()[start..(sample + 1) * sequence * 8],
                rotary[part].bytes().bytes()
            );
        }
        drop(rotary);
        assert_eq!(single_host.usage().unwrap().committed_bytes(), 0);
    }
    assert_eq!(
        host.usage().unwrap().allocated_bytes(),
        source_bytes + input_bytes
    );
    drop(parts);
    drop(outputs);
    assert_eq!(host.usage().unwrap().committed_bytes(), input_bytes);
    drop(inputs);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn edit_batch_bridge_rejects_invalid_groups_before_publishing_inputs() {
    let prompt = tokenizer()
        .prepare_edit_batch(&["cat", "blue cat"], 64, 4, 4, 2)
        .unwrap();
    let host = account(16384);
    let a = architecture_with_layers(3);
    let outputs = edit_vision_outputs(&host);
    let table = embedding_bytes();
    let baseline = host.usage().unwrap().committed_bytes();
    let check = |bad: &[Vec<EditVisionPart>]| {
        let before = host.usage().unwrap();
        assert!(prompt
            .managed_text_batch_parts(&host, &a, &table, &edit_vision_parts(bad))
            .is_err());
        assert_eq!(host.usage().unwrap(), before);
    };
    check(&outputs[..1]);
    let mut bad = outputs.clone();
    bad[1].pop();
    check(&bad);
    bad = outputs.clone();
    bad[1][3].0 = bad[1][2].0.clone();
    check(&bad);
    bad = outputs.clone();
    bad[1][0].1 = ResolvedTensorSpec::new(
        vec![2, 8],
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    check(&bad);
    let foreign = account(32);
    bad = outputs.clone();
    bad[1][0].2 = foreign
        .allocate_zeroed(HostMemoryClass::Output, 32)
        .unwrap()
        .freeze();
    check(&bad);
    bad = outputs.clone();
    bad[1][0].2 = host
        .allocate_zeroed(HostMemoryClass::InputPreprocess, 32)
        .unwrap()
        .freeze();
    check(&bad);
    bad = outputs.clone();
    let mut nonfinite = host.allocate_zeroed(HostMemoryClass::Output, 32).unwrap();
    nonfinite.as_mut()[..2].copy_from_slice(&0x7f80_u16.to_le_bytes());
    bad[1][0].2 = nonfinite.freeze();
    check(&bad);
    drop(bad);
    let parts = edit_vision_parts(&outputs);
    assert!(prompt
        .managed_text_parts(&host, &a, &table, &parts[0])
        .is_err());
    let nonfinite_table = vec![0xff; table.len()];
    assert!(prompt
        .managed_text_batch_parts(&host, &a, &nonfinite_table, &parts)
        .is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), baseline);
    drop(parts);
    drop(outputs);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert_eq!(foreign.usage().unwrap().committed_bytes(), 0);
}

fn account(bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: bytes,
                transfer_staging_bytes: bytes,
                output_bytes: bytes,
                weight_backing_bytes: bytes,
            },
        )
        .unwrap(),
    )
}
fn bytes(bits: impl IntoIterator<Item = u16>) -> Vec<u8> {
    bits.into_iter().flat_map(u16::to_le_bytes).collect()
}
fn bits(data: &[u8]) -> Vec<u16> {
    data.chunks_exact(2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .collect()
}
fn embedding_bytes() -> Vec<u8> {
    bytes((0..128).map(|n| bf16::from_f32(n as f32).to_bits()))
}

#[test]
fn actual_tokenizer_calls_keep_raw_template_dynamic_crop_and_left_padding() {
    let t = tokenizer();
    let batch = t.prepare(&["cat", "blue cat"], 8192).unwrap();
    // Synthetic vocabulary: added markers receive IDs 14(PAD),15(start),16(end).
    assert_eq!(t.drop_prefix_tokens(), 9);
    assert_eq!(batch.valid_input_lengths(), [15, 16]);
    assert_eq!(batch.sequence_length(), 16);
    assert_eq!(
        &batch.input_ids()[..16],
        &[14, 15, 1, 2, 3, 4, 5, 6, 7, 16, 15, 8, 10, 16, 15, 9]
    );
    assert_eq!(batch.valid_feature_lengths(), [6, 7]);
    assert_eq!(batch.is_valid_key(0, 0), Some(false));
    assert_eq!(batch.is_valid_key(0, 1), Some(true));
    assert_eq!(batch.position_ids(0, 1), Some([1, 1, 1]));
    assert_eq!(batch.position_ids(1, 1), Some([1, 1, 1]));
    assert_eq!(batch.position_ids(0, 16), None);
    assert_eq!(
        t.prepare(&[""], 8192).unwrap().input_ids(),
        t.prepare(&[" "], 8192).unwrap().input_ids()
    );
    let literal = t.prepare(&["<|endoftext|>"], 8192).unwrap();
    let index = literal.input_ids().iter().position(|&id| id == 14).unwrap();
    assert_eq!(literal.is_valid_key(0, index), Some(true));
    assert!(t.prepare(&["你好 é"], 8192).is_ok());
    assert!(t.prepare(&["blue cat"], 1).is_err());
    assert!(t.prepare(&["<|image_pad|>"], 8192).is_err());
    assert!(t.prepare(&["cat"], 0).is_err());
}

#[test]
fn canonical_token_batch_preserves_rows_and_stable_ties_without_weakening_limits() {
    let t = tokenizer();
    let prompts = ["blue cat", "cat", "blue", "cat blue cat"];
    let original = t.prepare(&prompts, 8192).unwrap();
    let (canonical, order) = t.prepare_canonical(&prompts, 8192).unwrap();
    assert_eq!(order, [1, 2, 0, 3]);
    assert_eq!(canonical.valid_input_lengths(), [15, 15, 16, 17]);
    let sequence = original.sequence_length();
    for (execution_index, &request_index) in order.iter().enumerate() {
        assert_eq!(
            &canonical.input_ids()[execution_index * sequence..(execution_index + 1) * sequence],
            &original.input_ids()[request_index * sequence..(request_index + 1) * sequence]
        );
        assert_eq!(
            canonical.layout().key_runs()[execution_index],
            original.layout().key_runs()[request_index]
        );
    }
    let (reordered, _) = t
        .prepare_canonical(&["cat blue cat", "blue", "blue cat", "cat"], 8192)
        .unwrap();
    assert_eq!(canonical.layout(), reordered.layout());
    assert_eq!(
        canonical.valid_feature_lengths(),
        reordered.valid_feature_lengths()
    );
    assert_eq!(
        canonical.drop_prefix_tokens(),
        original.drop_prefix_tokens()
    );
    assert!(t.prepare_canonical(&["blue cat"], 1).is_err());
    assert!(t
        .prepare_canonical(&["cat", "<|image_pad|>"], 8192)
        .is_err());
    assert!(t.prepare_canonical(&[], 8192).is_err());
    assert!(t.prepare_canonical(&["cat"; 17], 8192).is_err());
}

#[test]
fn real_jinja_semantics_accept_equivalent_templates_and_reject_unknown_calls_or_wrong_crop() {
    let (t, c, j) = tokenizer_sources();
    let mut equivalent = b"{# independently formatted equivalent template #}".to_vec();
    equivalent.extend(j);
    assert!(QwenImage21T2iTokenizer::from_source_bytes(&t, &c, &equivalent).is_ok());
    for template in [
        "{{ unknown_function() }}",
        "{{ raise_exception('unsupported') }}",
        "<|im_start|>system\\nwrong<|im_end|>\\n",
    ] {
        assert!(QwenImage21T2iTokenizer::from_source_bytes(&t, &c, template.as_bytes()).is_err());
    }
    let mut config: serde_json::Value = serde_json::from_slice(&c).unwrap();
    config["split_special_tokens"] = json!(true);
    assert!(QwenImage21T2iTokenizer::from_source_bytes(
        &t,
        &serde_json::to_vec(&config).unwrap(),
        &equivalent
    )
    .is_err());
}

#[test]
fn managed_lookup_is_exact_and_rotary_uses_padded_columns_with_duplicate_halves() {
    let batch =
        QwenImage21T2iPromptBatch::from_encoded(vec![vec![2, 3], vec![4, 5, 6]], 1, 0).unwrap();
    let host = account(4096);
    let input = batch
        .prepare_bytes(&host, &architecture(), &embedding_bytes())
        .unwrap();
    assert_eq!(input.config().layout(), batch.layout());
    assert_eq!(
        bits(input.tensors()[0].bytes().bytes()),
        [
            0., 1., 2., 3., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21.,
            22., 23., 24., 25., 26., 27.
        ]
        .map(|x| bf16::from_f32(x).to_bits())
    );
    // Column zero is padding in sample 0 and valid in sample 1; both get p=0.
    let cosine = bits(input.tensors()[1].bytes().bytes());
    let sine = bits(input.tensors()[2].bytes().bytes());
    assert_eq!(&cosine[..4], &[0x3f80; 4]);
    assert_eq!(&sine[..4], &[0; 4]);
    assert_eq!(&cosine[4..8], &[0x3f0a, 0x3f7f, 0x3f0a, 0x3f7f]);
    assert_eq!(&sine[4..8], &[0x3f57, 0x3dcc, 0x3f57, 0x3dcc]);
    assert_eq!(&cosine[..12], &cosine[12..]);
    assert_eq!(&sine[..12], &sine[12..]);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 6 * 4 * 2 * 3);
    drop(input);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn all_input_capacity_is_reserved_atomically_and_failures_release_every_backing() {
    let batch = QwenImage21T2iPromptBatch::from_encoded(vec![vec![2, 3]], 1, 0).unwrap();
    let size = 2 * 4 * 2 * 3;
    let host = account(size - 1);
    assert!(batch
        .prepare_bytes(&host, &architecture(), &embedding_bytes())
        .is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    let host = account(size);
    let mut corrupt = embedding_bytes();
    corrupt[3 * 8..3 * 8 + 2].copy_from_slice(&0x7f80_u16.to_le_bytes());
    assert!(batch
        .prepare_bytes(&host, &architecture(), &corrupt)
        .is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    assert!(batch
        .prepare_bytes(&host, &architecture(), &corrupt[..corrupt.len() - 1])
        .is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    let invalid = QwenImage21T2iPromptBatch::from_encoded(vec![vec![2, 32]], 1, 0).unwrap();
    assert!(invalid
        .prepare_bytes(&host, &architecture(), &embedding_bytes())
        .is_err());
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}

#[test]
fn borrowed_archive_lookup_validates_name_dtype_and_shape_before_host_admission() {
    let directory = tempfile::tempdir().unwrap();
    let data = embedding_bytes();
    let view =
        safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, vec![32, 4], &data).unwrap();
    safetensors::tensor::serialize_to_file(
        vec![(Qwen3VLTextEncoderFamily::EMBEDDING_EXTERNAL_NAME, view)],
        &None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let source = SafetensorsArchive::open(directory.path()).unwrap();
    let batch = QwenImage21T2iPromptBatch::from_encoded(vec![vec![2, 3]], 1, 0).unwrap();
    let host = account(4096);
    let selected = batch
        .managed_inputs(
            &host,
            &architecture(),
            &ferrum_quantization::TensorArchive::tensor(
                &source,
                Qwen3VLTextEncoderFamily::EMBEDDING_EXTERNAL_NAME,
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(selected.tensors()[0].bytes().len(), 16);
    drop(selected);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    for (name, dtype, shape) in [
        ("wrong.weight", safetensors::Dtype::BF16, vec![32, 4]),
        (
            Qwen3VLTextEncoderFamily::EMBEDDING_EXTERNAL_NAME,
            safetensors::Dtype::F16,
            vec![32, 4],
        ),
        (
            Qwen3VLTextEncoderFamily::EMBEDDING_EXTERNAL_NAME,
            safetensors::Dtype::BF16,
            vec![16, 8],
        ),
    ] {
        let d = tempfile::tempdir().unwrap();
        let view = safetensors::tensor::TensorView::new(dtype, shape, &data).unwrap();
        safetensors::tensor::serialize_to_file(
            vec![(name, view)],
            &None,
            &d.path().join("model.safetensors"),
        )
        .unwrap();
        let source = SafetensorsArchive::open(d.path()).unwrap();
        assert!(batch
            .managed_inputs(
                &host,
                &architecture(),
                &ferrum_quantization::TensorArchive::tensor(&source, name).unwrap()
            )
            .is_err());
        assert_eq!(host.usage().unwrap().committed_bytes(), 0);
    }
}

#[test]
fn crop_removes_left_padding_then_system_and_right_pads_without_changing_source() {
    let batch =
        QwenImage21T2iPromptBatch::from_encoded(vec![vec![2, 3], vec![4, 5, 6]], 1, 0).unwrap();
    let host = account(1024);
    let source_bits = [
        0x7fc0, 0x7fc0, 0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0, 0x40e0, 0x4100, 0x4110,
        0x4120,
    ];
    let original = bytes(source_bits);
    let mut source = host
        .allocate_zeroed(HostMemoryClass::Output, original.len())
        .unwrap();
    source.as_mut().copy_from_slice(&original);
    let source = source.freeze();
    let spec = ResolvedTensorSpec::new(
        vec![6, 2],
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    let output = batch.crop_bytes(&host, 2, &spec, &source).unwrap();
    assert_eq!(output.valid_lengths(), [1, 2]);
    assert_eq!(output.sequence_length(), 2);
    assert_eq!(
        bits(output.tensor().bytes().bytes()),
        [0x4040, 0x4080, 0, 0, 0x40e0, 0x4100, 0x4110, 0x4120]
    );
    assert_eq!(source.bytes(), original);
    assert!(batch.crop_bytes(&account(1024), 2, &spec, &source).is_err());
    let wrong = ResolvedTensorSpec::new(
        vec![3, 4],
        ElementType::Bf16,
        ResolvedTensorLayout::Contiguous,
    )
    .unwrap();
    assert!(batch.crop_bytes(&host, 2, &wrong, &source).is_err());
    drop(source);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 16);
    drop(output);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
