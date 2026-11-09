//! A real small BF16 archive follows the same registered source/definition/
//! preparation path as an explicitly selected conditional-model checkpoint.
use super::*;
use sha2::{Digest, Sha256};

mod managed;
mod native_bf16;

struct Fixture {
    _root: tempfile::TempDir,
    semantic: PathBuf,
    tokenizer: PathBuf,
    selected: PathBuf,
    weights: Vec<FamilyWeight>,
}

impl Fixture {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let semantic = root.path().join("semantic");
        let tokenizer = root.path().join("tokenizer");
        let physical = root.path().join("physical");
        for directory in [&semantic, &tokenizer, &physical] {
            std::fs::create_dir_all(directory).unwrap();
        }
        let mut config = test_config();
        config.hf_config["tie_word_embeddings"] = Value::Bool(false);
        config.hf_config["text_config"]["tie_word_embeddings"] = Value::Bool(false);
        for weight in &mut config.weights {
            weight.external_name =
                weight
                    .external_name
                    .replacen("model.", "model.language_model.", 1);
        }
        config.weights.push(FamilyWeight {
            layer_index: None,
            expert_index: None,
            role: "lm_head".to_owned(),
            external_name: "lm_head.weight".to_owned(),
            dimensions: vec![32, 16],
            source_encoding: FamilyWeightSourceEncoding::Dense {
                element_type: ElementType::Bf16,
            },
        });
        std::fs::write(
            semantic.join("config.json"),
            serde_json::to_vec(&config.hf_config).unwrap(),
        )
        .unwrap();
        std::fs::write(tokenizer.join("tokenizer.json"), br#"{"version":"1.0"}"#).unwrap();
        std::fs::write(
            tokenizer.join("tokenizer_config.json"),
            br#"{"chat_template":"{{ messages }}","bos_token_id":1,"eos_token_id":2,"pad_token_id":0}"#,
        )
        .unwrap();
        // Same-directory candidates must not affect the selected artifact.
        std::fs::write(
            physical.join("model.safetensors"),
            b"unrelated invalid archive",
        )
        .unwrap();
        std::fs::write(
            physical.join("model.safetensors.index.json"),
            b"unrelated invalid index",
        )
        .unwrap();
        std::fs::write(
            physical.join("config.json"),
            br#"{"architectures":["OtherModel"]}"#,
        )
        .unwrap();
        let fixture = Self {
            _root: root,
            semantic,
            tokenizer,
            selected: physical.join("official-bf16.safetensors"),
            weights: config.weights,
        };
        fixture.write(None, false);
        fixture
    }

    fn write(&self, wrong_shape: Option<&str>, omit_late_norm: bool) {
        let mut tensors = self
            .weights
            .iter()
            .filter_map(|weight| {
                if omit_late_norm
                    && weight.external_name
                        == "model.language_model.layers.3.self_attn.q_norm.weight"
                {
                    return None;
                }
                let shape = if wrong_shape == Some(weight.external_name.as_str()) {
                    vec![1, 1]
                } else {
                    weight
                        .dimensions
                        .iter()
                        .map(|dimension| *dimension as usize)
                        .collect::<Vec<_>>()
                };
                let value = if weight.role.contains("norm") {
                    0.125
                } else {
                    0.375
                };
                let bytes = bf16::from_f32(value)
                    .to_bits()
                    .to_le_bytes()
                    .repeat(shape.iter().product());
                Some((weight.external_name.clone(), shape, bytes))
            })
            .collect::<Vec<_>>();
        // Nonexecuted modalities are inventoried, not accidentally included
        // in the language weight schema or used as replacement text tensors.
        tensors.push((
            "model.visual.pos_embed.weight".to_owned(),
            vec![1],
            bf16::from_f32(0.5).to_bits().to_le_bytes().to_vec(),
        ));
        tensors.push((
            "mtp.norm.weight".to_owned(),
            vec![1],
            bf16::from_f32(0.5).to_bits().to_le_bytes().to_vec(),
        ));
        let views = tensors
            .iter()
            .map(|(name, shape, bytes)| {
                (
                    name.as_str(),
                    TensorView::new(Dtype::BF16, shape.clone(), bytes).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(views, &None, &self.selected).unwrap();
    }

    fn open(&self) -> ferrum_types::Result<ProductionModelSourceBundle> {
        let local = |kind, path: &Path| OriginalModelSource {
            kind,
            location: path.display().to_string(),
            requested_revision: None,
        };
        crate::vnext::open_registered_product_sources(
            &self.semantic,
            &self.tokenizer,
            ProductionWeightArtifact::safetensors_file(&self.selected),
            OriginalModelSources {
                semantic: local(ModelSourceKind::LocalDirectory, &self.semantic),
                tokenizer: local(ModelSourceKind::LocalDirectory, &self.tokenizer),
                weights: local(ModelSourceKind::LocalFile, &self.selected),
            },
        )
    }
}

#[test]
fn selected_bf16_tokenizer_added_tokens_prepare_with_frozen_source_identity() {
    let fixture = Fixture::new();
    let config = br#"{
        "bos_token":null,"eos_token":"<|im_end|>",
        "pad_token":"<|endoftext|>","chat_template":null
    }"#;
    let tokenizer = br#"{
        "version":"1.0","added_tokens":[
            {"id":5,"content":"<|endoftext|>","special":true},
            {"id":7,"content":"<|im_end|>","special":true}
        ]
    }"#;
    let template = b"{{ messages[0].content }}<|im_end|>\n";
    std::fs::write(fixture.tokenizer.join("tokenizer_config.json"), config).unwrap();
    std::fs::write(fixture.tokenizer.join("tokenizer.json"), tokenizer).unwrap();
    std::fs::write(fixture.tokenizer.join("chat_template.jinja"), template).unwrap();
    let sources = Arc::new(fixture.open().unwrap());
    let resolved = sources.resolved_sources().clone();
    let registration = crate::vnext::resolve_registered_model_from_sources(&sources)
        .unwrap()
        .into_required()
        .unwrap();
    let defined = registration.define_from_sources(sources.clone()).unwrap();
    let prepared = defined
        .prepare(&F16_NUMERICAL_PROFILE_ID.parse().unwrap())
        .unwrap();
    let metadata = prepared.family().metadata().causal_language().unwrap();
    assert_eq!(metadata.special_tokens.bos_token_id, None);
    assert_eq!(metadata.special_tokens.pad_token_id, Some(5));
    assert_eq!(metadata.special_tokens.eos_token_ids, BTreeSet::from([7]));
    assert_eq!(metadata.template.template.as_bytes(), template);
    assert_eq!(metadata.template.source_file, "chat_template.jinja");
    assert_eq!(
        metadata.template.sha256,
        format!("{:x}", Sha256::digest(template))
    );
    assert!(Arc::ptr_eq(prepared.sources(), &sources));
    assert_eq!(prepared.sources().resolved_sources(), &resolved);
    assert_eq!(prepared.sources().tokenizer_json(), tokenizer);
    assert_eq!(
        prepared.sources().tokenizer_config_json(),
        Some(config.as_slice())
    );

    // Changing files after opening cannot redirect the frozen model package.
    std::fs::write(
        fixture.tokenizer.join("tokenizer.json"),
        br#"{"added_tokens":[{"id":8,"content":"<|im_end|>"}]}"#,
    )
    .unwrap();
    assert_eq!(prepared.sources().tokenizer_json(), tokenizer);
    assert_eq!(prepared.sources().resolved_sources(), &resolved);
}

#[test]
fn selected_bf16_conditional_archive_prepares_through_the_registered_family() {
    let fixture = Fixture::new();
    let sources = Arc::new(fixture.open().unwrap());
    let archive = sources.weights().open_safetensors_archive().unwrap();
    let text = Qwen35TextConfig::from_hf_config_value(
        &serde_json::from_slice::<Value>(sources.config_json()).unwrap(),
    )
    .unwrap();
    let inventory = Qwen35WeightInventory::from_names(archive.tensor_names());
    let plan = inventory.detect_prefix_and_resolve(&text).unwrap();
    assert_eq!(plan.prefix, "model.language_model");
    let partition = inventory.partition_resolved_plan(&plan).unwrap();
    assert_eq!(
        partition.vision_non_executed,
        ["model.visual.pos_embed.weight"]
    );
    assert_eq!(partition.mtp_non_executed, ["mtp.norm.weight"]);
    partition.require_no_unknown().unwrap();
    drop(archive);

    let registration = crate::vnext::resolve_registered_model_from_sources(&sources)
        .unwrap()
        .into_required()
        .unwrap();
    let defined = registration.define_from_sources(sources.clone()).unwrap();
    let prepared = defined
        .prepare(&F32_MASTER_NUMERICAL_PROFILE_ID.parse().unwrap())
        .unwrap();
    let schema = prepared.family().stored_weight_schema().unwrap();
    let embedding = schema
        .components
        .iter()
        .find(|component| component.external_names == ["model.language_model.embed_tokens.weight"])
        .unwrap();
    let payload = prepared.weights().component(embedding).unwrap();
    // The source is BF16; preparation preserves the family's existing typed
    // F16 static-weight ABI even when activations use the F32-master profile.
    assert_eq!(embedding.dense_element_type(), Some(ElementType::F16));
    assert_eq!(
        payload.bytes(),
        f16::from_f32(0.375).to_bits().to_le_bytes().repeat(32 * 16)
    );
    let norm = schema
        .components
        .iter()
        .find(|component| component.external_names == ["model.language_model.norm.weight"])
        .unwrap();
    assert_eq!(
        prepared.weights().component(norm).unwrap().bytes(),
        f16::from_f32(1.125).to_bits().to_le_bytes().repeat(16)
    );
    assert!(schema
        .components
        .iter()
        .any(|component| component.external_names == ["lm_head.weight"]));
    assert!(schema
        .components
        .iter()
        .flat_map(|component| &component.external_names)
        .all(|name| !name.starts_with("model.visual.") && !name.starts_with("mtp.")));
    assert_eq!(prepared.sources().resolved_sources().weights.files.len(), 1);
    assert_eq!(
        prepared.sources().resolved_sources().weights.files[0].relative_path,
        "official-bf16.safetensors"
    );
    assert!(prepared.sources().weight_config_json().is_none());
    prepared
        .sources()
        .product_source_identity(
            "fixture",
            "fixture",
            "tokenizer_config.json",
            "{{ messages }}",
        )
        .unwrap();
}

#[test]
fn selected_archive_rejects_late_shape_and_missing_weight_errors_without_neighbor_fallback() {
    let fixture = Fixture::new();
    for missing in [false, true] {
        fixture.write(
            (!missing).then_some("model.language_model.layers.3.self_attn.o_proj.weight"),
            missing,
        );
        let sources = Arc::new(fixture.open().unwrap());
        let registration = crate::vnext::resolve_registered_model_from_sources(&sources)
            .unwrap()
            .into_required()
            .unwrap();
        let error = registration
            .define_from_sources(sources)
            .unwrap_err()
            .to_string();
        if missing {
            assert!(error.contains("q_norm"), "{error}");
        } else {
            assert!(
                error.contains("dimension") || error.contains("shape"),
                "{error}"
            );
        }
    }
}

#[test]
fn registered_architecture_preflight_precedes_single_file_access() {
    let fixture = Fixture::new();
    let mut semantic: Value =
        serde_json::from_slice(&std::fs::read(fixture.semantic.join("config.json")).unwrap())
            .unwrap();
    semantic["text_config"]["model_type"] = Value::from("incompatible_text_architecture");
    std::fs::write(
        fixture.semantic.join("config.json"),
        serde_json::to_vec(&semantic).unwrap(),
    )
    .unwrap();
    std::fs::remove_file(&fixture.selected).unwrap();
    let error = fixture.open().unwrap_err().to_string();
    assert!(
        error.contains("unsupported Qwen3.5 text model_type"),
        "{error}"
    );
    assert!(!error.contains("weight source is not a file"), "{error}");
}
