use super::*;

struct LegacyFixture {
    _directory: tempfile::TempDir,
    sources: Arc<ProductionModelSourceBundle>,
    requested_model: String,
}

fn legacy_fixture(gguf: bool, template_file: &str) -> LegacyFixture {
    let directory = tempfile::tempdir().unwrap();
    let snapshot = |repo: &str, revision: &str| {
        directory
            .path()
            .join(format!("models--{}", repo.replace('/', "--")))
            .join("snapshots")
            .join(revision)
    };
    let metadata_revision = "a".repeat(40);
    let weight_revision = "b".repeat(40);
    let metadata = snapshot("fixture/metadata", &metadata_revision);
    let weights = snapshot("fixture/weights", &weight_revision);
    for path in [&metadata, &weights] {
        std::fs::create_dir_all(path).unwrap();
    }
    std::fs::write(
        metadata.join("config.json"),
        serde_json::to_vec(&serde_json::json!({
            "architectures": ["LlamaForCausalLM"], "model_type": "llama"
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(metadata.join("tokenizer.json"), b"{}").unwrap();
    let template = "{{ messages[0].content }}";
    let template_bytes = if template_file == "chat_template.jinja" {
        template.as_bytes().to_vec()
    } else {
        serde_json::to_vec(&serde_json::json!({"chat_template": template})).unwrap()
    };
    std::fs::write(metadata.join(template_file), template_bytes).unwrap();
    let weight_file = weights.join(if gguf {
        "model.gguf"
    } else {
        "model.safetensors"
    });
    std::fs::write(&weight_file, b"unused fixture weight bytes").unwrap();
    let original = |repo: &str, revision: &str| OriginalModelSource {
        kind: ModelSourceKind::Repository,
        location: repo.to_owned(),
        requested_revision: Some(revision.to_owned()),
    };
    let sources = Arc::new(
        ProductionModelSourceBundle::open(
            &metadata,
            &metadata,
            if gguf {
                ProductionWeightArtifact::gguf_file(weight_file)
            } else {
                ProductionWeightArtifact::safetensors_directory(weights)
            },
            OriginalModelSources {
                semantic: original("fixture/metadata", &metadata_revision),
                tokenizer: original("fixture/metadata", &metadata_revision),
                weights: original("fixture/weights", &weight_revision),
            },
        )
        .unwrap(),
    );
    LegacyFixture {
        _directory: directory,
        sources,
        requested_model: format!("fixture/weights@{weight_revision}"),
    }
}

#[test]
fn legacy_source_identity_binds_actual_roles_and_retained_template() {
    for gguf in [false, true] {
        for template_file in [
            "chat_template.jinja",
            "chat_template.json",
            "tokenizer_config.json",
        ] {
            let fixture = legacy_fixture(gguf, template_file);
            let defined = define_registered_product_model(
                Some(&fixture.sources),
                &ferrum_types::NumericalExecutionPolicy::default(),
                ferrum_types::KvCacheDtype::Fp16,
            )
            .unwrap();
            assert!(defined.is_none(), "fixture must exercise the legacy path");
            let selected = load_product_chat_template(&fixture.sources).unwrap();
            let container = std::fs::read(&selected.source).unwrap();
            // The runtime and identity must retain the same lease even if a
            // source file changes after resolution.
            std::fs::write(&selected.source, "changed after source resolution").unwrap();
            let identity = product_source_identity(
                defined.as_deref(),
                Some(&fixture.sources),
                &fixture.requested_model,
                "fixture/weights",
                Some(&selected),
            )
            .unwrap()
            .unwrap();
            assert_eq!(identity.requested_model, fixture.requested_model);
            assert_eq!(identity.resolved_model, "fixture/weights");
            assert_eq!(
                identity.resolved_sources,
                *fixture.sources.resolved_sources()
            );
            assert_eq!(
                identity.original_sources,
                *fixture.sources.original_sources()
            );
            assert_eq!(identity.template.source_file, template_file);
            assert_eq!(
                identity.template.container_sha256,
                format!("{:x}", Sha256::digest(&container))
            );
            assert_eq!(
                identity.template.content_sha256,
                Some(format!(
                    "{:x}",
                    Sha256::digest(selected.template.as_bytes())
                ))
            );
            // Both entrypoints publish through this same artifact writer.
            let config = ferrum_types::FerrumConfigBuilder::new(
                RuntimeConfigSnapshot::from_entries(Vec::new()),
            )
            .with_execution_resource_authority(
                ferrum_types::ExecutionResourceAuthority::LegacyEngine,
            )
            .resolve()
            .unwrap();
            let output = fixture._directory.path().join("effective.json");
            crate::commands::serve::write_startup_config_artifacts(
                &config,
                Some(&identity),
                &ferrum_types::NumericalExecutionPolicy::default(),
                Some(&output),
                None,
            )
            .unwrap();
            let document: serde_json::Value =
                serde_json::from_slice(&std::fs::read(output).unwrap()).unwrap();
            assert_eq!(
                document["resolution_evidence"],
                serde_json::to_value(identity).unwrap()
            );
        }
    }
}

#[test]
fn legacy_source_identity_rejects_changed_or_foreign_selected_templates() {
    let fixture = legacy_fixture(true, "tokenizer_config.json");
    let selected = load_product_chat_template(&fixture.sources).unwrap();
    let mut changed = selected.clone();
    changed.template.push_str("different");
    let mut foreign = selected.clone();
    foreign.source = fixture
        ._directory
        .path()
        .join("tokenizer_config.json")
        .display()
        .to_string();
    let mut absent = selected.clone();
    absent.source = fixture
        .sources
        .tokenizer_root()
        .join("chat_template.jinja")
        .display()
        .to_string();
    for invalid in [changed, foreign, absent] {
        assert!(product_source_identity(
            None,
            Some(&fixture.sources),
            &fixture.requested_model,
            "fixture/weights",
            Some(&invalid),
        )
        .is_err());
    }
}

#[test]
fn incomplete_legacy_sources_do_not_fabricate_identity() {
    let fixture = legacy_fixture(false, "tokenizer_config.json");
    let selected = load_product_chat_template(&fixture.sources).unwrap();
    let policy = ferrum_types::NumericalExecutionPolicy::default();
    let config =
        ferrum_types::FerrumConfigBuilder::new(RuntimeConfigSnapshot::from_entries(Vec::new()))
            .resolve()
            .unwrap();
    let mut expected = config.effective_config_document();
    expected["numerical_execution"] = serde_json::json!({"requested": policy});
    for (index, (sources, template)) in [
        (None, Some(&selected)),
        (Some(fixture.sources.as_ref()), None),
        (None, None),
    ]
    .into_iter()
    .enumerate()
    {
        let identity = product_source_identity(
            None,
            sources,
            &fixture.requested_model,
            "fixture/weights",
            template,
        )
        .unwrap();
        assert!(identity.is_none());
        let output = fixture
            ._directory
            .path()
            .join(format!("incomplete-{index}.json"));
        crate::commands::serve::write_startup_config_artifacts(
            &config,
            identity.as_ref(),
            &policy,
            Some(&output),
            None,
        )
        .unwrap();
        let document: serde_json::Value =
            serde_json::from_slice(&std::fs::read(output).unwrap()).unwrap();
        assert!(document.get("resolution_evidence").is_none());
        assert_eq!(
            document, expected,
            "unrelated startup configuration is preserved"
        );
    }
}
