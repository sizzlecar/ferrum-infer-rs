use super::test_fixture::floats;
pub(super) use super::test_fixture::{archive, config, limits, original};
use super::*;
use half::bf16;
use safetensors::Dtype;

#[cfg(feature = "cuda")]
mod cuda;

fn id<T>(s: impl Into<String>) -> T
where
    T: TryFrom<String>,
    T::Error: std::fmt::Debug,
{
    T::try_from(s.into()).unwrap()
}
pub(super) fn prompt() -> PH3ImageTextPromptBatch {
    PH3ImageTextPromptBatch::from_rows(vec![vec![0, 1], vec![1]], 15, 16).unwrap()
}
fn host(bytes: usize) -> HostMemoryAccount {
    HostMemoryAccount::new(
        HostMemoryLimits::new(
            bytes,
            HostMemoryClassLimits {
                input_preprocess_bytes: bytes,
                output_bytes: bytes,
                transfer_staging_bytes: bytes,
                weight_backing_bytes: bytes,
            },
        )
        .unwrap(),
    )
}

#[test]
fn declared_connector_loads_actual_scaled_five_tap_mixed_quantized_checkpoint() {
    let c = config(prompt().layout().clone());
    let tensors = original(&c);
    let (_directory, source) = archive(&tensors);
    let connector = c.connector_config().unwrap();
    let loaded = PH3ImageTextConditioningConfig::from_comfy_checkpoint_with_connector(
        c.encoder().text().architecture().clone(),
        c.encoder().text().layout().clone(),
        &source,
        &connector,
    )
    .unwrap();
    assert_eq!(loaded, c);
    assert_eq!(
        loaded.encoder().text().hidden_state_indices(),
        [1, 2, 3, 4, 5]
    );
    assert_eq!(loaded.connector_output_dim(), 4);
    assert_eq!(loaded.encoder().projections(), c.encoder().projections());
    PH3ImageTextConditioningCheckpoint::new(&loaded, &source, limits()).unwrap();
    PH3ImageTextConnectorCheckpoint::new_with_layout(
        &connector,
        source.archive(),
        PH3ImageTextConnectorProjectionLimits {
            max_projection_bytes: 1 << 20,
        },
        PH3ImageTextConnectorCheckpointLayout::ComfyMergedTextEncoder,
    )
    .unwrap();
    assert!(PH3ImageTextConditioningConfig::from_comfy_checkpoint(
        c.encoder().text().architecture().clone(),
        c.encoder().text().layout().clone(),
        &source
    )
    .is_err());
}

#[test]
fn declared_connector_rejects_hidden_tap_and_missing_source_contracts_before_execution() {
    let c = config(prompt().layout().clone());
    let (_directory, source) = archive(&original(&c));
    for (indices, hidden) in [
        (vec![1, 2, 3, 5], 256),
        (vec![1, 2, 2, 4, 5], 256),
        (vec![0, 1, 2, 3, 5], 256),
        (vec![1, 3, 2, 4, 5], 256),
        (vec![1, 2, 3, 4, 6], 256),
        (vec![1, 2, 3, 4, 4], 256),
        (vec![1, 2, 3, 4, 5], 128),
    ] {
        let connector=PH3ImageTextConnectorConfig::from_hf_json(&serde_json::to_vec(&serde_json::json!({"_class_name":"PH3ImageTextConnector","hidden_state_indices":indices,"text_hidden_size":hidden,"out_channels":4})).unwrap()).unwrap();
        assert!(
            PH3ImageTextConditioningConfig::from_comfy_checkpoint_with_connector(
                c.encoder().text().architecture().clone(),
                c.encoder().text().layout().clone(),
                &source,
                &connector
            )
            .is_err()
        );
    }
    let mut tensors = original(&c);
    assert!(tensors
        .remove("model.language_model.layers.4.mlp.down_proj.comfy_quant")
        .is_some());
    let (_directory, source) = archive(&tensors);
    let error = PH3ImageTextConditioningConfig::from_comfy_checkpoint_with_connector(
        c.encoder().text().architecture().clone(),
        c.encoder().text().layout().clone(),
        &source,
        &c.connector_config().unwrap(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("quantization metadata"));
}

#[test]
fn embedding_and_all_decoder_layers_share_one_stored_program_and_only_five_readbacks() {
    let p = prompt();
    let c = config(p.layout().clone());
    let family = PH3ImageTextConditioningFamily::new(limits()).unwrap();
    let profiles = family.numerical_profiles(&c).unwrap();
    let profile = profiles
        .resolve(&id(PH3ImageTextConditioningFamily::CUDA_PROFILE_ID))
        .unwrap();
    let program = family.semantic_program(&c, profile).unwrap();
    assert_eq!(
        program.outputs(),
        (1..=5)
            .map(|i| Qwen3VLTextComfyFamily::hidden_state_output_id(i).unwrap())
            .collect::<Vec<_>>()
    );
    assert!(!program
        .inputs()
        .contains(&id(Qwen3VLTextComfyFamily::EMBEDDINGS_ID)));
    assert_eq!(
        program.blocks()[0].nodes[0].operation_id.as_str(),
        COMFY_CONVROT_INT8_EMBEDDING_F32_OPERATION_ID
    );
    assert_eq!(
        program.blocks()[0].nodes[0].attributes[&id("dequantized_dtype")],
        SemanticValue::Text("bf16_then_f32".into())
    );
    assert_eq!(
        program.blocks().last().unwrap().nodes[0]
            .operation_id
            .as_str(),
        HEAD_RMS_NORM_F32_OPERATION_ID
    );
    let weights = family.weights(&c).unwrap();
    weights
        .stored_schema()
        .unwrap()
        .validate_program_references(family.family_id(), &program)
        .unwrap();
    let (_directory, source) = archive(&original(&c));
    let checkpoint = PH3ImageTextConditioningCheckpoint::new(&c, &source, limits()).unwrap();
    for component in &weights.stored_schema().unwrap().components {
        checkpoint.component(component).unwrap();
    }
    let peak = c
        .host_stage_bytes(PH3ImageConditioningPrecision::Bf16)
        .unwrap();
    assert_eq!(
        peak,
        [4 * 5 * 256 * 8, 4 * 5 * 256 * 4 + 4 * 4 * 4, 4 * 4 * 6]
    );
    let mut low = limits();
    low.max_stored_bytes = 4096;
    assert!(PH3ImageTextConditioningFamily::new(low)
        .unwrap()
        .weights(&c)
        .is_err());
}

#[test]
fn checkpoint_rejects_embedding_shape_and_norm_dtype_without_expansion() {
    let p = prompt();
    let c = config(p.layout().clone());
    let mut tensors = original(&c);
    let norm = tensors
        .get_mut("model.language_model.layers.0.input_layernorm.weight")
        .unwrap();
    norm.0 = Dtype::F32;
    norm.2 = floats(std::iter::repeat_n(1.0, 256));
    let (_directory, source) = archive(&tensors);
    assert!(PH3ImageTextConditioningCheckpoint::new(&c, &source, limits()).is_err());
    let mut tensors = original(&c);
    let table = tensors
        .get_mut(Qwen3VLTextEncoderFamily::EMBEDDING_EXTERNAL_NAME)
        .unwrap();
    table.1[0] = 15;
    table.2.truncate(15 * 256);
    let scales = tensors
        .get_mut("model.language_model.embed_tokens.weight_scale")
        .unwrap();
    scales.1[0] = 15;
    scales.2.truncate(15 * 4);
    let (_directory, source) = archive(&tensors);
    assert!(PH3ImageTextConditioningCheckpoint::new(&c, &source, limits()).is_err());
    let mut raw = serde_json::to_value(&c).unwrap();
    raw["encoder"]["text"]["hidden_state_indices"] = serde_json::json!([1, 2, 3, 5]);
    assert!(serde_json::from_value::<PH3ImageTextConditioningConfig>(raw).is_err());
    let left = Qwen3VLTextLayout::new(
        2,
        vec![vec![
            Qwen3VLTextKeyRun::new(1, false),
            Qwen3VLTextKeyRun::new(1, true),
        ]],
    )
    .unwrap();
    assert!(PH3ImageTextConditioningConfig::new(
        Qwen3VLTextComfyConfig::new(
            Qwen3VLTextConfig::new(c.encoder().text().architecture().clone(), left)
                .unwrap()
                .with_hidden_state_indices(vec![1, 2, 3, 4, 5])
                .unwrap(),
            c.encoder().projections().clone()
        )
        .unwrap(),
        4
    )
    .is_err());
}

#[test]
fn raw_tokenizer_binds_source_pad_and_rejects_vision_truncation_and_invalid_ids() {
    let vocab = [
        ("zero", 0),
        ("one", 1),
        ("[UNK]", 2),
        ("<|image_pad|>", 14),
        ("<|endoftext|>", 15),
    ]
    .into_iter()
    .map(|(s, id)| (s.to_owned(), id))
    .collect();
    let model = tokenizers::models::wordlevel::WordLevel::builder()
        .vocab(vocab)
        .unk_token("[UNK]".into())
        .build()
        .unwrap();
    let mut engine = tokenizers::Tokenizer::new(model);
    engine.with_pre_tokenizer(Some(
        tokenizers::pre_tokenizers::whitespace::WhitespaceSplit,
    ));
    engine.add_special_tokens(&[
        tokenizers::AddedToken::from("<|endoftext|>", true),
        tokenizers::AddedToken::from("<|image_pad|>", true),
    ]);
    let source = engine.to_string(false).unwrap();
    let tokenizer = PH3ImageTextTokenizer::from_source_bytes(
        source.as_bytes(),
        br#"{"tokenizer_class":"Qwen2Tokenizer","pad_token":"<|endoftext|>","bos_token":null}"#,
        16,
        PH3ImageTextTokenizerLimits {
            max_source_bytes: 1 << 20,
            max_prompt_bytes: 128,
            max_tokens_per_prompt: 2,
            max_batch: 2,
        },
    )
    .unwrap();
    assert_eq!(tokenizer.pad_token_id(), 15);
    let p = tokenizer.prepare(&["zero one", ""]).unwrap();
    assert_eq!(p.token_ids(), [0, 1, 15, 15]);
    assert_eq!(p.valid_lengths(), [2, 1]);
    assert!(tokenizer.prepare(&["<|image_pad|>"]).is_err());
    assert!(tokenizer.prepare(&["zero one zero"]).is_err());
    assert!(tokenizer.prepare(&["embedding:unbound"]).is_err());
    assert!(PH3ImageTextPromptBatch::from_rows(vec![vec![16]], 15, 16).is_err());
}

#[test]
fn managed_ids_and_f32_text_rotary_use_atomic_host_admission_and_preserve_prefix_positions() {
    let p = prompt();
    let c = config(p.layout().clone());
    let tiny = host(4);
    assert!(p.managed_inputs(&c, &tiny).is_err());
    assert_eq!(tiny.usage().unwrap().committed_bytes(), 0);
    let h = host(1 << 20);
    let inputs = p.managed_inputs(&c, &h).unwrap();
    let cosine = inputs
        .iter()
        .find(|i| i.value_id().as_str() == Qwen3VLTextComfyFamily::COSINE_ID)
        .unwrap();
    assert_eq!(cosine.tensor().element_type(), ElementType::F32);
    let value =
        |i: usize| f32::from_le_bytes(cosine.bytes().bytes()[i * 4..i * 4 + 4].try_into().unwrap());
    assert_eq!(value(0), 1.0);
    assert_eq!(value(128), libm::cosf(1.0));
    assert_eq!(value(3 * 128), 1.0); // right padding, never a valid DiT text row
    assert_ne!(bf16::from_f32(value(128)).to_f32(), value(128));
    drop(inputs);
    assert_eq!(h.usage().unwrap().committed_bytes(), 0);
}
