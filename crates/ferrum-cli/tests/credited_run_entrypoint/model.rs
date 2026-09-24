//! Tiny real PlanRuntime model. Llama's cache-recovery fixture is legacy and
//! cannot exercise credited admission. These shapes follow the CPU Qwen3.5
//! fixture in ferrum-engine::registry, including its required hybrid layers.
use safetensors::tensor::{serialize, Dtype, TensorView};
use serde_json::json;
use std::{collections::BTreeMap, fs, path::Path};

pub fn write(directory: &Path) {
    fs::create_dir_all(directory).unwrap();
    let config = json!({
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5", "vocab_size": 3,
        "max_position_embeddings": 16, "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0, "tie_word_embeddings": false,
        "text_config": {
            "model_type": "qwen3_5_text", "hidden_size": 2,
            "intermediate_size": 2, "num_hidden_layers": 2,
            "layer_types": ["linear_attention", "full_attention"],
            "linear_num_key_heads": 1, "linear_num_value_heads": 1,
            "linear_key_head_dim": 1, "linear_value_head_dim": 1,
            "linear_conv_kernel_dim": 2, "mamba_ssm_dtype": "float32",
            "head_dim": 2, "num_attention_heads": 1, "num_key_value_heads": 1,
            "vocab_size": 3, "max_position_embeddings": 16,
            "tie_word_embeddings": false
        }
    });
    fs::write(
        directory.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let mut tensors = BTreeMap::<String, (Vec<usize>, Vec<f32>)>::new();
    // All embeddings carry one nonzero coordinate. Zero residual projections
    // preserve it; Qwen's zero norm coefficients have the declared +1 offset.
    tensors.insert(
        "model.embed_tokens.weight".into(),
        (vec![3, 2], vec![1., 0., 1., 0., 1., 0.]),
    );
    tensors.insert(
        "model.lm_head.weight".into(),
        (vec![3, 2], vec![0., 0., 8., 0., 0., 0.]),
    );
    tensors.insert("model.norm.weight".into(), (vec![2], vec![0.; 2]));
    for layer in 0..2 {
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            tensors.insert(
                format!("model.layers.{layer}.{norm}.weight"),
                (vec![2], vec![0.; 2]),
            );
        }
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            tensors.insert(
                format!("model.layers.{layer}.mlp.{projection}.weight"),
                (vec![2, 2], vec![0.; 4]),
            );
        }
    }
    // Real W3 hybrid contract: use the existing registry fixture's recurrent
    // geometry and finite coefficients. Its zero output projection preserves
    // the known residual stream without bypassing recurrent execution.
    for (name, shape, values) in [
        (
            "in_proj_qkv.weight",
            vec![3, 2],
            vec![1., 0., 0., 1., 1., 1.],
        ),
        ("in_proj_z.weight", vec![1, 2], vec![1., -1.]),
        ("in_proj_b.weight", vec![1, 2], vec![0.5, 0.25]),
        ("in_proj_a.weight", vec![1, 2], vec![-0.25, 0.75]),
        ("conv1d.weight", vec![3, 1, 2], vec![0., 1., 0., 1., 0., 1.]),
        ("A_log", vec![1], vec![0.]),
        ("dt_bias", vec![1], vec![0.]),
        ("norm.weight", vec![1], vec![1.]),
        ("out_proj.weight", vec![2, 1], vec![0., 0.]),
    ] {
        tensors.insert(
            format!("model.layers.0.linear_attn.{name}"),
            (shape, values),
        );
    }
    for projection in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        tensors.insert(
            format!("model.layers.1.self_attn.{projection}.weight"),
            (vec![2, 2], vec![0.; 4]),
        );
    }
    for norm in ["q_norm", "k_norm"] {
        tensors.insert(
            format!("model.layers.1.self_attn.{norm}.weight"),
            (vec![2], vec![1.; 2]),
        );
    }
    let storage: Vec<_> = tensors
        .into_iter()
        .map(|(name, (shape, values))| {
            let bytes: Vec<_> = values.into_iter().flat_map(f32::to_le_bytes).collect();
            (name, shape, bytes)
        })
        .collect();
    let views = storage.iter().map(|(name, shape, bytes)| {
        (
            name.as_str(),
            TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
        )
    });
    fs::write(
        directory.join("model.safetensors"),
        serialize(views, &None).unwrap(),
    )
    .unwrap();
    // The credited codec requires a *direct* ByteLevel decoder with a complete
    // unambiguous ID table. WordLevel without a decoder has no bounded proof.
    // U+0120 is ByteLevel's space byte: each generated token is " hello" and
    // repeated decoding has an exact, monotonically extending UTF-8 prefix.
    let mut tokenizer = tokenizers::Tokenizer::new(
        tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(
                [
                    ("<unk>".to_owned(), 0),
                    ("Ġhello".to_owned(), 1),
                    ("<eos>".to_owned(), 2),
                ]
                .into_iter()
                .collect(),
            )
            .unk_token("<unk>".to_owned())
            .build()
            .unwrap(),
    );
    tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
    tokenizer.with_decoder(Some(tokenizers::decoders::byte_level::ByteLevel::default()));
    assert_eq!(tokenizer.get_vocab_size(false), 3);
    assert_eq!(
        tokenizer.decode(&[1, 1, 1], true).unwrap(),
        " hello hello hello"
    );
    fs::write(
        directory.join("tokenizer.json"),
        tokenizer.to_string(false).unwrap(),
    )
    .unwrap();
    fs::write(directory.join("tokenizer_config.json"), br#"{"chat_template":"{% for message in messages %}{{ message['content'] }}{% endfor %}","eos_token_id":2,"unk_token":"<unk>"}"#).unwrap();
}
