use super::*;
use ferrum_kernels::backend::cpu::{
    vnext_ops::CpuVNextComposition, vnext_runtime::CpuDeviceRuntime,
};
use ferrum_runtime::{DeviceId, HostMemoryClassLimits, HostMemoryLimits};
use safetensors::tensor::{Dtype, TensorView};

pub(in crate::image_generation::p_h3::expansion) const ANSWER_ID: u32 = 7;
pub(in crate::image_generation::p_h3::expansion) const EOS_ID: u32 = 2;
pub(in crate::image_generation::p_h3::expansion) const ANSWER: &str =
    "</think> A small red cat sits quietly.";
pub(in crate::image_generation::p_h3::expansion) const SECOND_ANSWER_ID: u32 = 8;

pub(in crate::image_generation::p_h3::expansion) fn protocol_limits(
) -> PH3ImagePromptExpansionLimits {
    PH3ImagePromptExpansionLimits {
        max_tokenizer_bytes: 64 << 10,
        max_template_bytes: 16 << 10,
        max_request_bytes: 4096,
        max_rendered_bytes: 8192,
        max_answer_bytes: 4096,
        max_generated_tokens: 16,
        template_fuel: 100_000,
    }
}

pub(in crate::image_generation::p_h3::expansion) struct Fixture {
    pub(in crate::image_generation::p_h3::expansion) directory: std::path::PathBuf,
    expansion: Option<NativeTextExpansion>,
}
impl Drop for Fixture {
    fn drop(&mut self) {
        drop(self.expansion.take());
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}
impl Fixture {
    pub(in crate::image_generation::p_h3::expansion) fn take_expansion(
        &mut self,
    ) -> Arc<NativeTextExpansion> {
        Arc::new(self.expansion.take().unwrap())
    }
    pub(in crate::image_generation::p_h3::expansion) fn expansion(&self) -> &NativeTextExpansion {
        self.expansion.as_ref().unwrap()
    }
    pub(in crate::image_generation::p_h3::expansion) fn new() -> Self {
        Self::build(false)
    }
    pub(in crate::image_generation::p_h3::expansion) fn new_two_answers() -> Self {
        Self::build(true)
    }
    fn build(two_answers: bool) -> Self {
        let directory = crate::registry::tests::write_qwen35_fixture_model_dir();
        let config_path = directory.join("config.json");
        let mut config: serde_json::Value =
            serde_json::from_slice(&std::fs::read(&config_path).unwrap()).unwrap();
        for key in [
            "vocab_size",
            "max_position_embeddings",
            "eos_token_id",
            "pad_token_id",
        ] {
            let value = match key {
                "vocab_size" => {
                    if two_answers {
                        9
                    } else {
                        8
                    }
                }
                "max_position_embeddings" => 64,
                _ => 1,
            };
            config[key] = value.into();
            config["text_config"][key] = value.into();
        }
        std::fs::write(config_path, serde_json::to_vec(&config).unwrap()).unwrap();

        // A genuine tiny Qwen program with zero attention/MLP projections.
        // Its residual stream preserves the last input embedding: ordinary
        // prompt tokens select the answer, then that answer selects actual EOS.
        // No mocked executor or injected completion is used by these tests.
        let weights_path = directory.join("model.safetensors");
        let original = std::fs::read(&weights_path).unwrap();
        let tensors = safetensors::SafeTensors::deserialize(&original).unwrap();
        let owned = tensors
            .tensors()
            .into_iter()
            .map(|(name, tensor)| {
                let embedding = name == "model.embed_tokens.weight";
                let head = name == "model.lm_head.weight";
                let shape = if embedding || head {
                    vec![if two_answers { 9 } else { 8 }, 2]
                } else {
                    tensor.shape().to_vec()
                };
                let mut values = vec![0.0f32; shape.iter().product()];
                if embedding {
                    for (row, values) in values.chunks_exact_mut(2).enumerate() {
                        values.copy_from_slice(if row == ANSWER_ID as usize {
                            &[-1.0, 1.0]
                        } else {
                            &[1.0, -1.0]
                        });
                    }
                }
                if head {
                    values[ANSWER_ID as usize * 2..ANSWER_ID as usize * 2 + 2]
                        .copy_from_slice(&[1.0, -1.0]);
                    values[EOS_ID as usize * 2..EOS_ID as usize * 2 + 2]
                        .copy_from_slice(&[-1.0, 1.0]);
                }
                if two_answers && embedding {
                    for (row, values) in values.chunks_exact_mut(2).enumerate() {
                        values.copy_from_slice(match row as u32 {
                            ANSWER_ID | SECOND_ANSWER_ID => &[-1.0, 0.0],
                            0 => &[0.0, 1.0],
                            _ => &[1.0, 0.0],
                        });
                    }
                }
                if two_answers && head {
                    values[ANSWER_ID as usize * 2..ANSWER_ID as usize * 2 + 2]
                        .copy_from_slice(&[1.0, 0.0]);
                    values[EOS_ID as usize * 2..EOS_ID as usize * 2 + 2]
                        .copy_from_slice(&[-1.0, 0.0]);
                    values[SECOND_ANSWER_ID as usize * 2..SECOND_ANSWER_ID as usize * 2 + 2]
                        .copy_from_slice(&[0.0, 1.0]);
                }
                let bytes = values
                    .into_iter()
                    .flat_map(|v| half::bf16::from_f32(v).to_le_bytes())
                    .collect::<Vec<_>>();
                (name, shape, bytes)
            })
            .collect::<Vec<_>>();
        let views = owned
            .iter()
            .map(|(name, shape, bytes)| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::BF16, shape.clone(), bytes).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        safetensors::tensor::serialize_to_file(views, &None, &weights_path).unwrap();

        let mut vocabulary = vec![
            "[UNK]",
            "<|endoftext|>",
            "<|im_end|>",
            "<|im_start|>",
            "<|vision_start|>",
            "<|image_pad|>",
            "<|vision_end|>",
            ANSWER,
        ];
        if two_answers {
            vocabulary.push("</think> A blue dog runs quickly.");
        }
        let vocabulary = vocabulary
            .into_iter()
            .enumerate()
            .map(|(i, token)| (token.to_owned(), i as u32))
            .collect();
        let model = tokenizers::models::wordlevel::WordLevel::builder()
            .vocab(vocabulary)
            .unk_token("[UNK]".to_owned())
            .build()
            .unwrap();
        let mut tokenizer = tokenizers::Tokenizer::new(model);
        tokenizer.add_special_tokens(
            &[
                "<|endoftext|>",
                "<|im_end|>",
                "<|im_start|>",
                "<|vision_start|>",
                "<|image_pad|>",
                "<|vision_end|>",
            ]
            .map(|token| tokenizers::AddedToken::from(token, true)),
        );
        std::fs::write(
            directory.join("tokenizer.json"),
            tokenizer.to_string(false).unwrap(),
        )
        .unwrap();
        let tokenizer_config = serde_json::json!({
            "tokenizer_class": "Qwen3_5Tokenizer",
            "eos_token": "<|im_end|>",
            "pad_token": "<|endoftext|>",
            "bos_token": null,
            "model_max_length": 64,
            "added_tokens_decoder": {
                "1": {"content": "<|endoftext|>", "special": true},
                "2": {"content": "<|im_end|>", "special": true}
            }
        });
        std::fs::write(
            directory.join("tokenizer_config.json"),
            serde_json::to_vec(&tokenizer_config).unwrap(),
        )
        .unwrap();
        // Reuse the small unmodified upstream template and its retained license
        // attribution from the protocol tests; no generated/model assets added.
        std::fs::write(
            directory.join("chat_template.jinja"),
            include_bytes!(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../ferrum-models/src/p_h3_image/prompt_expansion/tests/chat_template.jinja"
            )),
        )
        .unwrap();
        let sources =
            Arc::new(ProductionModelSourceBundle::open_colocated_safetensors(&directory).unwrap());
        let expansion = NativeTextExpansion::from_sources(sources, protocol_limits()).unwrap();
        Self {
            directory,
            expansion: Some(expansion),
        }
    }
}

pub(in crate::image_generation::p_h3::expansion) fn composition(
) -> (RuntimeComposition<CpuDeviceRuntime>, Arc<CpuDeviceRuntime>) {
    let (runtime, registry, materializers, _, catalog) = CpuVNextComposition::create(
        DeviceId::new("device.cpu.native-prompt-expansion").unwrap(),
        16 << 20,
    )
    .unwrap()
    .into_parts()
    .unwrap();
    (
        RuntimeComposition::new(Arc::clone(&runtime), registry, materializers, catalog),
        runtime,
    )
}
pub(in crate::image_generation::p_h3::expansion) fn host() -> HostMemoryAccount {
    HostMemoryAccount::with_allocation_alignment(
        HostMemoryLimits::new(
            4 << 20,
            HostMemoryClassLimits {
                input_preprocess_bytes: 1 << 20,
                output_bytes: 1 << 20,
                transfer_staging_bytes: 1 << 20,
                weight_backing_bytes: 1 << 20,
            },
        )
        .unwrap(),
        16,
    )
    .unwrap()
}
pub(in crate::image_generation::p_h3::expansion) fn prepared(
    expansion: &NativeTextExpansion,
    max_new_tokens: u32,
) -> PreparedTextExpansion {
    let mut engine = EngineConfig::default();
    engine.backend.device = ferrum_types::Device::CPU;
    expansion
        .prepare_text("draw a cat", 1024, 1024, max_new_tokens, 2, engine)
        .unwrap()
}
