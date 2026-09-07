//! A real, tiny Llama checkpoint whose only preferred next token is `restored`.
//! Zero attention/MLP projections retain a nonzero embedding through the
//! residual stream; the output head selects the expected vocabulary row.

use safetensors::tensor::{serialize, Dtype, TensorView};
use serde_json::json;
use std::collections::BTreeMap;

pub const REPO: &str = "fixture/cache-recovery";
pub const EXPECTED_TOKEN: &str = "restored";
pub const FIRST_SHARD: &str = "model-00001-of-00002.safetensors";
pub const LAST_SHARD: &str = "model-00002-of-00002.safetensors";
pub const INDEX: &str = "model.safetensors.index.json";
const HIDDEN: usize = 16;
const INTERMEDIATE: usize = 32;
const VOCAB: usize = 4;

#[derive(Clone, Copy)]
enum TemplateLocation {
    TokenizerConfig,
    Standalone,
}

struct Tensor {
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

impl Tensor {
    fn filled(shape: &[usize], value: f32) -> Self {
        let values = vec![value; shape.iter().product()];
        Self::values(shape, &values)
    }

    fn values(shape: &[usize], values: &[f32]) -> Self {
        assert_eq!(shape.iter().product::<usize>(), values.len());
        Self {
            shape: shape.to_vec(),
            bytes: values
                .iter()
                .flat_map(|value| value.to_le_bytes())
                .collect(),
        }
    }
}

fn shard(tensors: &BTreeMap<String, Tensor>) -> Vec<u8> {
    let views = tensors.iter().map(|(name, tensor)| {
        (
            name.as_str(),
            TensorView::new(Dtype::F32, tensor.shape.clone(), &tensor.bytes).unwrap(),
        )
    });
    serialize(views, &None).unwrap()
}

pub fn files() -> BTreeMap<String, Vec<u8>> {
    checkpoint(TemplateLocation::TokenizerConfig)
}

pub fn standalone_template_files() -> BTreeMap<String, Vec<u8>> {
    checkpoint(TemplateLocation::Standalone)
}

fn checkpoint(template: TemplateLocation) -> BTreeMap<String, Vec<u8>> {
    let tokenizer = tokenizers::Tokenizer::new(
        tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(
                ["[UNK]", "[BOS]", "[EOS]", EXPECTED_TOKEN]
                    .into_iter()
                    .enumerate()
                    .map(|(index, token)| (token.to_owned(), index as u32))
                    .collect(),
            )
            .unk_token("[UNK]".into())
            .build()
            .unwrap(),
    );
    let mut embedding = vec![0.0; VOCAB * HIDDEN];
    for row in embedding.chunks_mut(HIDDEN) {
        row[0] = 1.0;
    }
    if matches!(template, TemplateLocation::Standalone) {
        // Only the standalone template produces this input token. A missing
        // template that falls back to the raw prompt must fail the inference
        // assertion, even if model loading itself succeeds.
        embedding[3 * HIDDEN] = 0.0;
        embedding[3 * HIDDEN + 1] = 1.0;
    }
    let mut first = BTreeMap::from([
        (
            "model.embed_tokens.weight".to_owned(),
            Tensor::values(&[VOCAB, HIDDEN], &embedding),
        ),
        (
            "model.norm.weight".to_owned(),
            Tensor::filled(&[HIDDEN], 1.0),
        ),
    ]);
    for name in ["input_layernorm", "post_attention_layernorm"] {
        first.insert(
            format!("model.layers.0.{name}.weight"),
            Tensor::filled(&[HIDDEN], 1.0),
        );
    }
    for name in ["q_proj", "k_proj", "v_proj", "o_proj"] {
        first.insert(
            format!("model.layers.0.self_attn.{name}.weight"),
            Tensor::filled(&[HIDDEN, HIDDEN], 0.0),
        );
    }
    for name in ["gate_proj", "up_proj"] {
        first.insert(
            format!("model.layers.0.mlp.{name}.weight"),
            Tensor::filled(&[INTERMEDIATE, HIDDEN], 0.0),
        );
    }
    first.insert(
        "model.layers.0.mlp.down_proj.weight".to_owned(),
        Tensor::filled(&[HIDDEN, INTERMEDIATE], 0.0),
    );
    let mut output_head = vec![0.0; VOCAB * HIDDEN];
    match template {
        TemplateLocation::TokenizerConfig => output_head[3 * HIDDEN] = 8.0,
        TemplateLocation::Standalone => {
            output_head[0] = 8.0;
            output_head[3 * HIDDEN + 1] = 8.0;
        }
    }
    let last = BTreeMap::from([(
        "lm_head.weight".to_owned(),
        Tensor::values(&[VOCAB, HIDDEN], &output_head),
    )]);
    let weight_map: BTreeMap<_, _> = first
        .keys()
        .map(|name| (name.clone(), FIRST_SHARD))
        .chain(last.keys().map(|name| (name.clone(), LAST_SHARD)))
        .collect();
    let mut files: BTreeMap<String, Vec<u8>> = [
        (FIRST_SHARD.to_owned(), shard(&first)),
        (LAST_SHARD.to_owned(), shard(&last)),
        (
            INDEX.to_owned(),
            serde_json::to_vec(&json!({"weight_map": weight_map})).unwrap(),
        ),
        (
            "config.json".to_owned(),
            serde_json::to_vec(&json!({
                "architectures": ["LlamaForCausalLM"], "model_type": "llama",
                "hidden_size": HIDDEN, "intermediate_size": INTERMEDIATE,
                "num_hidden_layers": 1, "num_attention_heads": 2,
                "num_key_value_heads": 2, "vocab_size": VOCAB,
                "max_position_embeddings": 64, "rms_norm_eps": 0.00001,
                "rope_theta": 10000.0, "bos_token_id": 1, "eos_token_id": 2,
                "tie_word_embeddings": false
            }))
            .unwrap(),
        ),
        (
            "tokenizer.json".to_owned(),
            tokenizer.to_string(false).unwrap().into_bytes(),
        ),
    ]
    .into();
    let mut tokenizer_config = json!({
        "bos_token": "[BOS]", "eos_token": "[EOS]", "unk_token": "[UNK]"
    });
    match template {
        TemplateLocation::TokenizerConfig => {
            tokenizer_config["chat_template"] =
                json!("{% for message in messages %}{{ message.content }}{% endfor %}");
        }
        TemplateLocation::Standalone => {
            files.insert(
                "chat_template.jinja".to_owned(),
                b"{{ 'restored' }}".to_vec(),
            );
        }
    }
    files.insert(
        "tokenizer_config.json".to_owned(),
        serde_json::to_vec(&tokenizer_config).unwrap(),
    );
    files
}
