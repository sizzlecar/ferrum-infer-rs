//! Runner ownership/transport fixture. Zero branch projections make the
//! residual identity analytically checkable; this is not a full-TE numeric test.
use super::*;
use ferrum_models::{
    qwen3_vl_text::Qwen3VLTextArchitecture, qwen_image21::t2i_prompt::QwenImage21T2iTokenizer,
};
use half::bf16;
use std::collections::BTreeMap;

#[test]
fn one_account_survives_real_text_execution_crop_and_device_retirement() {
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
    let mut tokenizer = tokenizers::Tokenizer::new(model);
    tokenizer.with_pre_tokenizer(Some(
        tokenizers::pre_tokenizers::whitespace::WhitespaceSplit,
    ));
    tokenizer.with_post_processor(Some(
        tokenizers::pre_tokenizers::byte_level::ByteLevel::default(),
    ));
    tokenizer.add_special_tokens(
        &["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
            .map(|t| tokenizers::AddedToken::from(t, true)),
    );
    let token_config = json!({"tokenizer_class":"Qwen2Tokenizer","processor_class":"Qwen3VLProcessor","bos_token":null,"pad_token":"<|endoftext|>","eos_token":"<|im_end|>","add_prefix_space":false,"split_special_tokens":false});
    let template =
        b"{{ '<|im_start|>system\\n' }}{{ messages[0].content[0].text }}{{ '<|im_end|>\\n' }}";
    let tokenizer = QwenImage21T2iTokenizer::from_source_bytes(
        tokenizer.to_string(false).unwrap().as_bytes(),
        &serde_json::to_vec(&token_config).unwrap(),
        template,
    )
    .unwrap();
    let prompts = tokenizer.prepare(&["cat"], 8192).unwrap();
    let architecture=Qwen3VLTextArchitecture::from_hf_value(&json!({"model_type":"qwen3_vl","architectures":["Qwen3VLForConditionalGeneration"],
        "text_config":{"model_type":"qwen3_vl_text","dtype":"bfloat16","hidden_act":"silu","attention_bias":false,"attention_dropout":0.0,
            "num_hidden_layers":1,"hidden_size":4,"intermediate_size":8,"num_attention_heads":2,"num_key_value_heads":1,"head_dim":2,
            "vocab_size":32,"max_position_embeddings":1024,"rms_norm_eps":0.000001,"rope_theta":100,
            "rope_scaling":{"rope_type":"default","mrope_interleaved":true,"mrope_section":[1,0,0]}}})).unwrap();
    let config = Qwen3VLTextConfig::new(architecture.clone(), prompts.layout().clone()).unwrap();
    let budget = HostBudget::new(&config, &prompts, 65536, 1024 * 1024).unwrap();
    assert!(HostBudget::new(&config, &prompts, 65536, budget.total - 1).is_err());
    let host = budget.account().unwrap();
    let directory = tempfile::tempdir().unwrap();
    let mut payloads = BTreeMap::new();
    let weights = config.expected_weights().unwrap();
    let mut weight_bytes = 0;
    for weight in &weights {
        let value = if weight.dimensions().len() == 1 {
            0x3f80_u16
        } else {
            0
        };
        let bytes = std::iter::repeat_n(value, (weight.bytes() / 2) as usize)
            .flat_map(u16::to_le_bytes)
            .collect::<Vec<_>>();
        payloads.insert(
            weight.external_name().to_string(),
            (
                weight
                    .dimensions()
                    .iter()
                    .map(|&x| x as usize)
                    .collect::<Vec<_>>(),
                bytes,
            ),
        );
        weight_bytes += weight.bytes();
    }
    let embed = (0..32)
        .flat_map(|token| {
            (0..4)
                .map(move |channel| bf16::from_f32(token as f32 + channel as f32 / 16.0).to_bits())
        })
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    payloads.insert(
        Family::EMBEDDING_EXTERNAL_NAME.to_string(),
        (vec![32, 4], embed),
    );
    let views = payloads
        .iter()
        .map(|(name, (shape, bytes))| {
            (
                name.as_str(),
                safetensors::tensor::TensorView::new(
                    safetensors::Dtype::BF16,
                    shape.clone(),
                    bytes,
                )
                .unwrap(),
            )
        })
        .collect::<Vec<_>>();
    safetensors::tensor::serialize_to_file(
        views,
        &None,
        &directory.path().join("model.safetensors"),
    )
    .unwrap();
    let source = SafetensorsArchive::open(directory.path()).unwrap();
    let inputs = prompts
        .managed_inputs(
            &host,
            &architecture,
            &ferrum_quantization::TensorArchive::tensor(&source, Family::EMBEDDING_EXTERNAL_NAME)
                .unwrap(),
        )
        .unwrap();
    let args = Arguments {
        model_dir: directory.path().to_owned(),
        artifact_dir: directory.path().join("unused"),
        prompt: "cat".into(),
        maximum_user_tokens: 8192,
        maximum_host_bytes: 1024 * 1024,
        reference_memory_bytes: 64 * 1024 * 1024,
        backend: Backend::Reference,
    };
    let result = run(
        &config,
        &prompts,
        inputs,
        &host,
        &budget,
        &source,
        &args,
        weight_bytes,
    )
    .unwrap();
    let expected = [15_u32, 8, 10, 16, 15, 9]
        .into_iter()
        .flat_map(|token| {
            (0..4)
                .map(move |channel| bf16::from_f32(token as f32 + channel as f32 / 16.0).to_bits())
        })
        .flat_map(u16::to_le_bytes)
        .collect::<Vec<_>>();
    assert_eq!(result.features.tensor().bytes().bytes(), expected);
    assert_eq!(result.features.valid_lengths(), [6]);
    assert_eq!(result.retirement["released_static_resources"], 1);
    assert_eq!(host.usage().unwrap().allocated_bytes(), expected.len());
    drop(result);
    assert_eq!(host.usage().unwrap().allocated_bytes(), 0);
    assert_eq!(host.usage().unwrap().committed_bytes(), 0);
}
