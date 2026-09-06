use super::*;

fn fixture() -> BTreeMap<String, String> {
    [
        ("Cargo.toml", r#"[workspace]
members = ["crates/ferrum-bench-core", "crates/app"]
resolver = "2"
[workspace.package]
version = "1.0.0"
[workspace.dependencies]
ferrum-bench-core = { path = "crates/ferrum-bench-core", version = "1.0.0" }
"#),
        ("crates/ferrum-bench-core/Cargo.toml", "[package]\nname = 'ferrum-bench-core'\nversion.workspace = true\n[dependencies]\nserde = '1'\n"),
        ("crates/app/Cargo.toml", "[package]\nname = 'app'\nversion.workspace = true\n[dependencies]\nferrum-bench-core.workspace = true\nserde = '1'\nsemver = '1'\ntoml_edit = '0.22'\n"),
        ("Cargo.lock", r#"version = 4
[[package]]
name = "app"
version = "1.0.0"
dependencies = ["ferrum-bench-core", "serde", "semver", "toml_edit"]
[[package]]
name = "ferrum-bench-core"
version = "1.0.0"
dependencies = ["serde"]
[[package]]
name = "serde"
version = "1.0.1"
source = "registry+https://example.invalid/index"
checksum = "serde-checksum"
[[package]]
name = "semver"
version = "1.0.9"
source = "registry+https://example.invalid/index"
checksum = "semver-checksum"
[[package]]
name = "toml_edit"
version = "0.22.27"
source = "registry+https://example.invalid/index"
checksum = "toml-checksum"
"#),
    ].into_iter().map(|(path, text)| (path.into(), text.into())).collect()
}
fn append(files: &mut BTreeMap<String, String>, path: &str, text: &str) {
    files.get_mut(path).unwrap().push_str(text);
}
fn add_dev(files: &mut BTreeMap<String, String>) {
    append(
        files,
        "crates/ferrum-bench-core/Cargo.toml",
        "[dev-dependencies]\nfixture_test = '1'\n",
    );
    let lock = files.get_mut("Cargo.lock").unwrap();
    *lock = lock.replace(
        "dependencies = [\"serde\"]",
        "dependencies = [\"serde\", \"fixture_test\"]",
    );
    lock.push_str("[[package]]\nname = 'fixture_test'\nversion = '1.0.0'\nsource = 'registry+https://example.invalid/index'\nchecksum = 'fixture-checksum'\n");
}

#[test]
fn pure_dev_addition_can_introduce_only_its_new_locked_closure() {
    let before = fixture();
    let mut after = before.clone();
    add_dev(&mut after);
    let result = validation_dependency_paths(&before, &after).unwrap();
    assert!(!result.coordinated_version);
    assert!(result.validation_runtime_dependencies.is_empty());
    assert!(result.paths.contains(&"Cargo.lock".into()));
    let mut unexplained = after.clone();
    append(&mut unexplained, "Cargo.lock", "[[package]]\nname = 'unexplained'\nversion = '1.0.0'\nsource = 'registry+https://example.invalid/index'\n");
    assert!(validation_dependency_paths(&before, &unexplained).is_err());
    let mut modified_runtime = after;
    let lock = modified_runtime.get_mut("Cargo.lock").unwrap();
    *lock = lock.replace("serde-checksum", "different-checksum");
    assert!(validation_dependency_paths(&before, &modified_runtime).is_err());
}

#[test]
fn known_release_tool_additions_remain_build_changes_and_never_allow_other_runtime_edges() {
    let before = fixture();
    let mut after = before.clone();
    append(
        &mut after,
        "crates/ferrum-bench-core/Cargo.toml",
        "semver = '1'\ntoml_edit = '0.22'\n",
    );
    let lock = after.get_mut("Cargo.lock").unwrap();
    *lock = lock.replace(
        "dependencies = [\"serde\"]",
        "dependencies = [\"serde\", \"semver\", \"toml_edit\"]",
    );
    let result = validation_dependency_paths(&before, &after).unwrap();
    assert!(result
        .validation_runtime_dependencies
        .iter()
        .any(|edge| edge.ends_with(":semver")));
    assert!(result
        .validation_runtime_dependencies
        .iter()
        .any(|edge| edge.ends_with(":toml_edit")));
    for addition in [
        "other_runtime = '1'\n",
        "[features]\nnew_runtime = []\n",
        "[build-dependencies]\nserde = '1'\n",
    ] {
        let mut changed = after.clone();
        append(
            &mut changed,
            "crates/ferrum-bench-core/Cargo.toml",
            addition,
        );
        assert!(validation_dependency_paths(&before, &changed).is_err());
    }
    let mut features = after;
    let manifest = features
        .get_mut("crates/ferrum-bench-core/Cargo.toml")
        .unwrap();
    *manifest = manifest.replace(
        "semver = '1'",
        "semver = { version = '1', features = ['serde'] }",
    );
    assert!(validation_dependency_paths(&before, &features).is_err());
}

#[test]
fn dev_dependency_cannot_hide_a_changed_runtime_version_or_registry_feature_edge() {
    let before = fixture();
    let mut after = before.clone();
    append(
        &mut after,
        "crates/app/Cargo.toml",
        "[dev-dependencies]\nserde = '2'\n",
    );
    let lock = after.get_mut("Cargo.lock").unwrap();
    *lock = lock.replace(
        "\"ferrum-bench-core\", \"serde\"",
        "\"ferrum-bench-core\", \"serde 2.0.0\"",
    );
    lock.push_str("[[package]]\nname = 'serde'\nversion = '2.0.0'\nsource = 'registry+https://example.invalid/index'\n");
    assert!(validation_dependency_paths(&before, &after).is_err());
    let mut after = before.clone();
    add_dev(&mut after);
    let lock = after.get_mut("Cargo.lock").unwrap();
    *lock = lock.replace(
        "checksum = \"serde-checksum\"",
        "checksum = \"serde-checksum\"\ndependencies = [\"fixture_test\"]",
    );
    assert!(validation_dependency_paths(&before, &after).is_err());
}

#[test]
fn target_dev_dependencies_and_formatting_are_semantic_but_unknown_workspace_changes_are_not() {
    let before = fixture();
    let mut after = before.clone();
    append(
        &mut after,
        "crates/ferrum-bench-core/Cargo.toml",
        "# harmless documentation\n[target.'cfg(unix)'.dev-dependencies]\nsemver = '1'\n",
    );
    let lock = after.get_mut("Cargo.lock").unwrap();
    *lock = lock.replace(
        "dependencies = [\"serde\"]",
        "dependencies = [\"serde\", \"semver\"]",
    );
    validation_dependency_paths(&before, &after).unwrap();
    for change in ["resolver = \"1\"", "resolver = \"invalid\""] {
        let mut changed = after.clone();
        let manifest = changed.get_mut("Cargo.toml").unwrap();
        *manifest = manifest.replace("resolver = \"2\"", change);
        assert!(validation_dependency_paths(&before, &changed).is_err());
    }
    let mut missing = after;
    missing.remove("crates/app/Cargo.toml");
    assert!(validation_dependency_paths(&before, &missing).is_err());
}

#[test]
fn coordinated_version_and_dev_dependency_edits_compose_without_erasing_other_changes() {
    let before = fixture();
    let mut after = before.clone();
    let members = names(&parse(&before, "Cargo.toml").unwrap(), &before).unwrap();
    let manifests: Vec<_> = members
        .iter()
        .map(|(name, path)| MemberManifest {
            package_name: name.clone(),
            manifest_path: path.clone(),
            text: before[path].clone(),
        })
        .collect();
    let update = plan_version_update(
        &before["Cargo.toml"],
        &manifests,
        &before["Cargo.lock"],
        "1.0.1",
    )
    .unwrap();
    after.insert("Cargo.toml".into(), update.workspace_manifest);
    after.insert("Cargo.lock".into(), update.lockfile);
    for member in update.members {
        after.insert(member.manifest_path, member.text);
    }
    add_dev(&mut after);
    let result = validation_dependency_paths(&before, &after).unwrap();
    assert!(result.coordinated_version);
    let mut unrelated = after;
    append(
        &mut unrelated,
        "crates/app/Cargo.toml",
        "new_runtime = '1'\n",
    );
    assert!(validation_dependency_paths(&before, &unrelated).is_err());
}
