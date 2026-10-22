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
    Qwen3VLTextArchitecture::from_hf_value(&json!({"model_type":"qwen3_vl","architectures":["Qwen3VLForConditionalGeneration"],
        "text_config":{"model_type":"qwen3_vl_text","dtype":"bfloat16","hidden_act":"silu","attention_bias":false,"attention_dropout":0.0,
            "num_hidden_layers":1,"hidden_size":4,"intermediate_size":8,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":4,
            "vocab_size":32,"max_position_embeddings":1024,"rms_norm_eps":0.000001,"rope_theta":100,
            "rope_scaling":{"rope_type":"default","mrope_interleaved":true,"mrope_section":[1,1,0]}}})).unwrap()
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
            &source
                .tensor(Qwen3VLTextEncoderFamily::EMBEDDING_EXTERNAL_NAME)
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
            .managed_inputs(&host, &architecture(), &source.tensor(name).unwrap())
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
