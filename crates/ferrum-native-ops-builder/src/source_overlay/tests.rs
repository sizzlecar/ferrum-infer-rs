use super::*;

fn fixture(root: &Path) -> (std::path::PathBuf, std::path::PathBuf, std::path::PathBuf) {
    let base = root.join("base");
    let repo = root.join("repo");
    fs::create_dir(&base).unwrap();
    fs::create_dir(&repo).unwrap();
    fs::write(base.join("kernel.cu"), b"baseline kernel").unwrap();
    fs::write(base.join("header.h"), b"retained header").unwrap();
    fs::write(repo.join("override.cu"), b"indexed kernel").unwrap();
    let members = vec![
        NativeOperatorSourceFileLock {
            path: "header.h".into(),
            sha256: sha256_bytes(b"retained header"),
        },
        NativeOperatorSourceFileLock {
            path: "kernel.cu".into(),
            sha256: sha256_bytes(b"baseline kernel"),
        },
    ];
    write_json(
        &repo.join("bundle.json"),
        &serde_json::json!({ "members": members }),
    )
    .unwrap();
    let spec = NativeOperatorSourceOverlay {
        schema_version: 1,
        base_bundle_manifest: NativeOperatorSourceFileLock {
            path: "bundle.json".into(),
            sha256: sha256_bytes(&fs::read(repo.join("bundle.json")).unwrap()),
        },
        files: members
            .into_iter()
            .map(|member| NativeOperatorSourceOverlayFile {
                repository_override: (member.path == "kernel.cu").then(|| {
                    NativeOperatorSourceFileLock {
                        path: "override.cu".into(),
                        sha256: sha256_bytes(b"indexed kernel"),
                    }
                }),
                path: member.path,
                base_sha256: member.sha256,
            })
            .collect(),
    };
    let path = root.join("overlay.json");
    write_json(&path, &spec).unwrap();
    (path, base, repo)
}

#[test]
fn source_overlay_materializes_exact_override_without_mutating_inputs() {
    let root = tempfile::tempdir().unwrap();
    let (spec, base, repo) = fixture(root.path());
    let output = root.path().join("out");
    materialize_native_operator_source_overlay(&spec, &base, &repo, &output).unwrap();
    assert_eq!(
        fs::read(output.join("kernel.cu")).unwrap(),
        b"indexed kernel"
    );
    assert_eq!(
        fs::read(output.join("header.h")).unwrap(),
        b"retained header"
    );
    assert_eq!(
        fs::read(base.join("kernel.cu")).unwrap(),
        b"baseline kernel"
    );
    assert_eq!(
        fs::read(repo.join("override.cu")).unwrap(),
        b"indexed kernel"
    );
    let receipt: NativeOperatorSourceOverlay =
        read_json(&output.join("source-overlay.json")).unwrap();
    assert_eq!(
        receipt,
        read_json::<NativeOperatorSourceOverlay>(&spec).unwrap()
    );
}

#[test]
fn source_overlay_rejects_baseline_or_override_drift_before_writing() {
    for drift in ["base", "override", "manifest"] {
        let root = tempfile::tempdir().unwrap();
        let (spec, base, repo) = fixture(root.path());
        let changed = match drift {
            "base" => base.join("kernel.cu"),
            "override" => repo.join("override.cu"),
            _ => repo.join("bundle.json"),
        };
        fs::write(changed, b"different").unwrap();
        let output = root.path().join("out");
        assert!(materialize_native_operator_source_overlay(&spec, &base, &repo, &output).is_err());
        assert!(!output.exists());
    }
}

#[test]
fn source_overlay_rejects_escape_duplicate_and_existing_output() {
    let root = tempfile::tempdir().unwrap();
    let (path, base, repo) = fixture(root.path());
    let original: NativeOperatorSourceOverlay = read_json(&path).unwrap();
    for case in ["escape", "duplicate", "foreign_member"] {
        let mut spec = original.clone();
        match case {
            "escape" => {
                spec.files[1].repository_override.as_mut().unwrap().path = "../escape.cu".into()
            }
            "duplicate" => spec.files.push(spec.files[1].clone()),
            _ => spec.files[1].base_sha256 = sha256_bytes(b"not the bundle member"),
        }
        write_json(&path, &spec).unwrap();
        let output = root.path().join("out");
        assert!(materialize_native_operator_source_overlay(&path, &base, &repo, &output).is_err());
        assert!(!output.exists());
    }
    write_json(&path, &original).unwrap();
    assert!(
        materialize_native_operator_source_overlay(&path, &base, &repo, &base.join("nested"))
            .is_err()
    );
    let output = root.path().join("out");
    fs::create_dir(&output).unwrap();
    fs::write(output.join("keep"), b"existing").unwrap();
    assert!(materialize_native_operator_source_overlay(&path, &base, &repo, &output).is_err());
    assert_eq!(fs::read(output.join("keep")).unwrap(), b"existing");
}
