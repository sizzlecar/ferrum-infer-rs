use super::*;
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

struct Fixture {
    root: tempfile::TempDir,
    semantic: PathBuf,
    tokenizer: PathBuf,
    selected: PathBuf,
}

fn write_tensor_file(path: &Path, value: f32) {
    let bytes = [value; 6]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let tensor = TensorView::new(Dtype::F32, vec![2, 3], &bytes).unwrap();
    serialize_to_file([("weight", tensor)], &None, path).unwrap();
}

impl Fixture {
    fn new() -> Self {
        let root = tempfile::tempdir().unwrap();
        let semantic = root.path().join("semantic");
        let tokenizer = root.path().join("tokenizer");
        let weights = root.path().join("weights");
        for path in [&semantic, &tokenizer, &weights] {
            std::fs::create_dir_all(path).unwrap();
        }
        std::fs::write(
            semantic.join("config.json"),
            br#"{"architectures":["ThirdPartyDecoder"]}"#,
        )
        .unwrap();
        std::fs::write(tokenizer.join("tokenizer.json"), br#"{"version":"1.0"}"#).unwrap();
        std::fs::write(
            tokenizer.join("tokenizer_config.json"),
            br#"{"chat_template":"fixture"}"#,
        )
        .unwrap();
        let selected = weights.join("official-native.safetensors");
        write_tensor_file(&selected, 0.375);
        Self {
            root,
            semantic,
            tokenizer,
            selected,
        }
    }

    fn original_sources(&self) -> OriginalModelSources {
        let local = |kind, path: &Path| OriginalModelSource {
            kind,
            location: path.display().to_string(),
            requested_revision: None,
        };
        OriginalModelSources {
            semantic: local(ModelSourceKind::LocalDirectory, &self.semantic),
            tokenizer: local(ModelSourceKind::LocalDirectory, &self.tokenizer),
            weights: local(ModelSourceKind::LocalFile, &self.selected),
        }
    }

    fn open(&self) -> Result<ProductionModelSourceBundle> {
        ProductionModelSourceBundle::open(
            &self.semantic,
            &self.tokenizer,
            ProductionWeightArtifact::safetensors_file(&self.selected),
            self.original_sources(),
        )
    }
}

#[test]
fn selected_safetensors_file_maps_and_fingerprints_without_reading_neighbors() {
    let fixture = Fixture::new();
    let parent = fixture.selected.parent().unwrap();
    // Neither a conflicting config nor malformed default weights/index belong
    // to an explicitly selected file source.
    for (name, bytes) in [
        (
            "config.json",
            br#"{"architectures":["OtherDecoder"]}"#.as_slice(),
        ),
        (
            "model.safetensors",
            b"invalid unrelated checkpoint".as_slice(),
        ),
        (
            "model.safetensors.index.json",
            b"invalid unrelated index".as_slice(),
        ),
    ] {
        std::fs::write(parent.join(name), bytes).unwrap();
    }
    let bundle = fixture.open().unwrap();
    assert!(matches!(
        bundle.weights(),
        ProductionWeightArtifact::SafetensorsFile(_)
    ));
    assert!(bundle.weight_config_json().is_none());
    let files = &bundle.resolved_sources().weights.files;
    assert_eq!(files.len(), 1);
    assert_eq!(files[0].relative_path, "official-native.safetensors");
    assert_eq!(files[0].sha256, hash_file(&fixture.selected).unwrap());

    let archive = bundle.weights().open_safetensors_archive().unwrap();
    assert_eq!(archive.tensor_names().collect::<Vec<_>>(), ["weight"]);
    let tensor = archive.tensor("weight").unwrap();
    assert_eq!(tensor.source_file(), "official-native.safetensors");
    assert_eq!(tensor.shape(), [2, 3]);
    assert_eq!(tensor.bytes(), 0.375_f32.to_le_bytes().repeat(6));
    let retained = tensor.retained_host_memory().clone();
    drop(tensor);
    drop(archive);
    // The actual mmap owner outlives the archive through the tensor lease.
    assert_eq!(retained.bytes(), 0.375_f32.to_le_bytes().repeat(6));
}

#[test]
fn single_file_source_keeps_semantic_tokenizer_and_weight_identity_independent() {
    let fixture = Fixture::new();
    let before = fixture.open().unwrap();
    let identity = before
        .product_source_identity("requested", "resolved", "tokenizer_config.json", "fixture")
        .unwrap();
    assert_eq!(identity.original_sources, fixture.original_sources());
    assert!(identity.weight_config.is_none());
    assert_eq!(
        identity.semantic_config.role,
        ModelArtifactSourceRole::Semantic
    );
    assert_eq!(identity.tokenizer.role, ModelArtifactSourceRole::Tokenizer);

    std::fs::write(
        fixture.selected.parent().unwrap().join("other.safetensors"),
        b"other",
    )
    .unwrap();
    let neighbor_changed = fixture.open().unwrap();
    assert_eq!(
        before.resolved_sources(),
        neighbor_changed.resolved_sources()
    );
    std::fs::write(
        fixture.tokenizer.join("tokenizer.json"),
        br#"{"version":"changed"}"#,
    )
    .unwrap();
    let tokenizer_changed = fixture.open().unwrap();
    assert_ne!(
        before.resolved_sources().tokenizer,
        tokenizer_changed.resolved_sources().tokenizer
    );
    assert_eq!(
        before.resolved_sources().weights,
        tokenizer_changed.resolved_sources().weights
    );
    write_tensor_file(&fixture.selected, -0.5);
    let selected_changed = fixture.open().unwrap();
    assert_ne!(
        before.resolved_sources().weights,
        selected_changed.resolved_sources().weights
    );
    assert_eq!(
        before.resolved_sources().semantic,
        selected_changed.resolved_sources().semantic
    );
}

#[test]
fn single_file_selection_rejects_missing_bad_names_and_invalid_selected_archive() {
    let mut fixture = Fixture::new();
    write_tensor_file(
        &fixture.selected.parent().unwrap().join("model.safetensors"),
        1.0,
    );
    for name in [
        "missing.safetensors",
        "checkpoint.json",
        ".safetensors",
        "bad:name.safetensors",
    ] {
        fixture.selected = fixture.selected.parent().unwrap().join(name);
        if name != "missing.safetensors" {
            std::fs::write(&fixture.selected, b"not a selected safetensors archive").unwrap();
        }
        assert!(fixture.open().is_err(), "accepted {name}");
    }
    fixture.selected = fixture
        .selected
        .parent()
        .unwrap()
        .join("corrupt.safetensors");
    std::fs::write(&fixture.selected, b"corrupt selected archive").unwrap();
    let bundle = fixture.open().unwrap();
    assert!(bundle.weights().open_safetensors_archive().is_err());
}

#[test]
fn semantic_architecture_is_checked_before_a_missing_single_file() {
    let fixture = Fixture::new();
    std::fs::write(
        fixture.semantic.join("config.json"),
        br#"{"architectures":["FirstDecoder","SecondDecoder"]}"#,
    )
    .unwrap();
    std::fs::remove_file(&fixture.selected).unwrap();
    let error = fixture.open().unwrap_err().to_string();
    assert!(
        error.contains("architecture") && error.contains("exactly one"),
        "{error}"
    );
    assert!(!error.contains("weight source is not a file"), "{error}");
}

#[cfg(unix)]
#[test]
fn single_file_lfs_source_preserves_nested_snapshot_identity_and_actual_bytes() {
    use std::os::unix::fs::symlink;

    let fixture = Fixture::new();
    let repository = fixture.root.path().join("models--fixture--weights");
    let revision = "a".repeat(40);
    let snapshot = repository.join("snapshots").join(&revision);
    let selected = snapshot.join("weights/native/official-native.safetensors");
    std::fs::create_dir_all(selected.parent().unwrap()).unwrap();
    std::fs::create_dir_all(repository.join("blobs")).unwrap();
    let blob = repository.join("blobs/not-a-content-sha256");
    write_tensor_file(&blob, 0.75);
    symlink(&blob, &selected).unwrap();
    let mut original = fixture.original_sources();
    original.weights = OriginalModelSource {
        kind: ModelSourceKind::Repository,
        location: "fixture/weights".to_owned(),
        requested_revision: Some(revision.clone()),
    };
    let bundle = ProductionModelSourceBundle::open(
        &fixture.semantic,
        &fixture.tokenizer,
        ProductionWeightArtifact::safetensors_file(&selected),
        original,
    )
    .unwrap();
    let resolved = &bundle.resolved_sources().weights;
    assert_eq!(resolved.canonical_location, "fixture/weights");
    assert_eq!(resolved.resolved_revision, revision);
    assert_eq!(
        resolved.files[0].relative_path,
        "weights/native/official-native.safetensors"
    );
    assert_eq!(resolved.files[0].sha256, hash_file(&blob).unwrap());
    assert!(bundle
        .weights()
        .path()
        .starts_with(snapshot.canonicalize().unwrap()));
    bundle
        .product_source_identity("requested", "resolved", "tokenizer_config.json", "fixture")
        .unwrap();
    let archive = bundle.weights().open_safetensors_archive().unwrap();
    assert_eq!(
        archive.tensor("weight").unwrap().bytes(),
        0.75_f32.to_le_bytes().repeat(6)
    );
}
