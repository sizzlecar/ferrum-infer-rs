use super::*;
use safetensors::tensor::{serialize_to_file, TensorView};
use serde_json::json;

fn weight(path: &Path) {
    let bytes = [0x80, 0x3f, 0x00, 0xc0]; // BF16 +1, -2.
    serialize_to_file(
        [(
            "weight",
            TensorView::new(Dtype::BF16, vec![2], &bytes).unwrap(),
        )],
        &None,
        path,
    )
    .unwrap();
}
fn index(path: &Path, shard: &str) {
    fs::write(
        path,
        serde_json::to_vec(&json!({"weight_map":{"weight":shard}})).unwrap(),
    )
    .unwrap();
}
fn open(root: &Path, naming: SafetensorsNaming) -> Result<SafetensorsArchive> {
    SafetensorsArchive::open_standard_component(root, "component", naming, 4096)
}
fn failure(result: Result<SafetensorsArchive>) -> String {
    result
        .err()
        .expect("expected an explicit source error")
        .to_string()
}

#[test]
fn standard_names_select_single_or_index_without_changing_payload() {
    for naming in [
        SafetensorsNaming::Transformers,
        SafetensorsNaming::Diffusers,
    ] {
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().join("component");
        fs::create_dir(&directory).unwrap();
        let (single, map) = naming.names();
        weight(&directory.join(single));
        let archive = open(root.path(), naming).unwrap();
        assert_eq!(archive.tensor_count(), 1);
        assert_eq!(
            archive.tensor("weight").unwrap().bytes(),
            &[0x80, 0x3f, 0, 0xc0]
        );
        drop(archive);
        fs::rename(directory.join(single), directory.join("shard.safetensors")).unwrap();
        index(&directory.join(map), "shard.safetensors");
        let archive = open(root.path(), naming).unwrap();
        assert_eq!(archive.tensor("weight").unwrap().shape(), &[2]);
        assert_eq!(
            archive.tensor("weight").unwrap().bytes(),
            &[0x80, 0x3f, 0, 0xc0]
        );
        assert_eq!(
            archive.tensor("weight").unwrap().source_file(),
            "shard.safetensors"
        );
    }
}

#[test]
fn missing_conflicting_and_damaged_archives_never_fall_back() {
    let root = tempfile::tempdir().unwrap();
    let directory = root.path().join("component");
    fs::create_dir(&directory).unwrap();
    weight(&directory.join("model.safetensors"));
    assert!(failure(open(root.path(), SafetensorsNaming::Diffusers))
        .contains("missing component archive"));
    let index_path = directory.join("model.safetensors.index.json");
    index(&index_path, "missing.safetensors");
    assert!(failure(open(root.path(), SafetensorsNaming::Transformers)).contains("ambiguous"));
    fs::remove_file(directory.join("model.safetensors")).unwrap();
    assert!(failure(open(root.path(), SafetensorsNaming::Transformers)).contains("missing"));
    weight(&directory.join("other.safetensors"));
    fs::write(&index_path, b"{").unwrap();
    assert!(failure(open(root.path(), SafetensorsNaming::Transformers)).contains("parse"));
    index(&index_path, "other.safetensors");
    assert!(failure(SafetensorsArchive::open_standard_component(
        root.path(),
        "component",
        SafetensorsNaming::Transformers,
        2
    ))
    .contains("read limit"));
    index(&index_path, "../other.safetensors");
    assert!(failure(open(root.path(), SafetensorsNaming::Transformers))
        .contains("invalid safetensors shard path"));
    for bad in ["../component", "/component", "a\\b", "C:component", ""] {
        assert!(SafetensorsArchive::open_standard_component(
            root.path(),
            bad,
            SafetensorsNaming::Transformers,
            4096
        )
        .is_err());
    }
}

#[cfg(unix)]
#[test]
fn package_boundary_covers_component_single_index_and_every_shard() {
    use std::os::unix::fs::symlink;
    let root = tempfile::tempdir().unwrap();
    let outside = tempfile::tempdir().unwrap();
    let directory = root.path().join("component");
    weight(&outside.path().join("model.safetensors"));
    symlink(outside.path(), &directory).unwrap();
    assert!(failure(open(root.path(), SafetensorsNaming::Transformers)).contains("outside"));
    fs::remove_file(&directory).unwrap();
    fs::create_dir(&directory).unwrap();
    symlink(
        outside.path().join("model.safetensors"),
        directory.join("model.safetensors"),
    )
    .unwrap();
    assert!(failure(open(root.path(), SafetensorsNaming::Transformers)).contains("outside"));
    fs::remove_file(directory.join("model.safetensors")).unwrap();
    index(
        &outside.path().join("model.safetensors.index.json"),
        "model.safetensors",
    );
    symlink(
        outside.path().join("model.safetensors.index.json"),
        directory.join("model.safetensors.index.json"),
    )
    .unwrap();
    assert!(failure(open(root.path(), SafetensorsNaming::Transformers)).contains("outside"));
    fs::remove_file(directory.join("model.safetensors.index.json")).unwrap();
    index(
        &directory.join("model.safetensors.index.json"),
        "shard.safetensors",
    );
    symlink(
        outside.path().join("model.safetensors"),
        directory.join("shard.safetensors"),
    )
    .unwrap();
    assert!(failure(open(root.path(), SafetensorsNaming::Transformers)).contains("outside"));
    fs::remove_file(directory.join("shard.safetensors")).unwrap();
    weight(&root.path().join("inside.safetensors"));
    symlink(
        root.path().join("inside.safetensors"),
        directory.join("shard.safetensors"),
    )
    .unwrap();
    assert_eq!(
        open(root.path(), SafetensorsNaming::Transformers)
            .unwrap()
            .tensor_count(),
        1
    );
}
