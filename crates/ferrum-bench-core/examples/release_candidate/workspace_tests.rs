use super::*;

fn edit(path: &Path, before: &str, after: &str) -> Edit {
    fs::write(path, before).unwrap();
    Edit {
        path: path.into(),
        original: before.into(),
        replacement: after.into(),
    }
}

#[test]
fn failed_validation_restores_all_manifests_and_lockfile() {
    let dir = tempfile::tempdir().unwrap();
    let first = dir.path().join("Cargo.toml");
    let second = dir.path().join("Cargo.lock");
    let edits = [
        edit(&first, "old manifest", "new manifest"),
        edit(&second, "old lock", "new lock"),
    ];
    let result = apply(&edits, || Err("dependency resolution failed".into()));
    assert!(result.unwrap_err().contains("dependency resolution failed"));
    assert_eq!(read(&first).unwrap(), "old manifest");
    assert_eq!(read(&second).unwrap(), "old lock");
    assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 2);
}

#[test]
fn changed_inputs_are_preserved_without_partial_apply() {
    let dir = tempfile::tempdir().unwrap();
    let first = dir.path().join("Cargo.toml");
    let second = dir.path().join("Cargo.lock");
    let edits = [
        edit(&first, "old manifest", "new manifest"),
        edit(&second, "old lock", "new lock"),
    ];
    fs::write(&second, "user edit").unwrap();
    assert!(apply(&edits, || Ok(())).is_err());
    assert_eq!(read(&first).unwrap(), "old manifest");
    assert_eq!(read(&second).unwrap(), "user edit");
}

#[test]
fn rollback_keeps_a_concurrent_edit_and_reports_it() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("Cargo.toml");
    let edits = [edit(&path, "old", "new")];
    let result = apply(&edits, || {
        fs::write(&path, "user edit after preparation").unwrap();
        Err("validation failed".into())
    });
    assert!(result.unwrap_err().contains("preserving concurrent edits"));
    assert_eq!(read(&path).unwrap(), "user edit after preparation");
}

#[test]
fn cargo_workspace_prepare_updates_real_resolution_and_dry_run_writes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    fs::create_dir_all(root.join("core/src")).unwrap();
    fs::create_dir_all(root.join("app/src")).unwrap();
    fs::write(
        root.join("Cargo.toml"),
        r#"[workspace]
members = ["core", "app"]
resolver = "2"
[workspace.package]
version = "1.2.3"
[workspace.dependencies]
fixture-core = { path = "core", version = "=1.2.3" }
"#,
    )
    .unwrap();
    fs::write(
        root.join("core/Cargo.toml"),
        "[package]\nname = \"fixture-core\"\nversion.workspace = true\nedition = \"2021\"\n",
    )
    .unwrap();
    fs::write(root.join("app/Cargo.toml"), "[package]\nname = \"fixture-app\"\nversion.workspace = true\nedition = \"2021\"\n[dependencies]\nfixture-core.workspace = true\n").unwrap();
    for member in ["core", "app"] {
        fs::write(root.join(member).join("src/lib.rs"), "").unwrap();
    }
    let output = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()))
        .current_dir(root)
        .args(["generate-lockfile", "--offline"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let before = read(&root.join("Cargo.toml")).unwrap();
    let before_lock = read(&root.join("Cargo.lock")).unwrap();
    prepare(root, "1.3.0", true).unwrap();
    assert_eq!(read(&root.join("Cargo.toml")).unwrap(), before);
    assert_eq!(read(&root.join("Cargo.lock")).unwrap(), before_lock);
    prepare(root, "1.3.0", false).unwrap();
    let info = cargo_metadata(root, false).unwrap();
    staging::validate_workspace_versions(&info, "1.3.0").unwrap();
    let dependency = &info["packages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|package| package["name"] == "fixture-app")
        .unwrap()["dependencies"][0];
    assert_eq!(dependency["req"], "=1.3.0");
    assert!(!read(&root.join("Cargo.lock")).unwrap().contains("1.2.3"));
    assert!(prepare(root, "1.2.0", false).is_err());
    staging::validate_workspace_versions(&cargo_metadata(root, false).unwrap(), "1.3.0").unwrap();
}
