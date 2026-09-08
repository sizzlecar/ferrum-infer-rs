use super::*;

pub(super) fn fixture() -> BTreeMap<String, String> {
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
fn syntax_visitor_is_a_scoped_tool_feature_not_permission_to_change_parser_resolution() {
    let mut before = fixture();
    append(&mut before, "crates/app/Cargo.toml", "syn = '2'\n");
    let lock = before.get_mut("Cargo.lock").unwrap();
    *lock = lock.replace(
        "\"serde\", \"semver\", \"toml_edit\"]",
        "\"serde\", \"semver\", \"toml_edit\", \"syn\"]",
    );
    lock.push_str("[[package]]\nname = 'syn'\nversion = '2.0.1'\nsource = 'registry+https://example.invalid/index'\nchecksum = 'syntax-checksum'\n");
    let mut full = before.clone();
    append(
        &mut full,
        "crates/ferrum-bench-core/Cargo.toml",
        "syn = { version = '2', features = ['full'] }\n",
    );
    let lock = full.get_mut("Cargo.lock").unwrap();
    *lock = lock.replace(
        "dependencies = [\"serde\"]",
        "dependencies = [\"serde\", \"syn\"]",
    );
    let mut visitor = full.clone();
    let manifest = visitor
        .get_mut("crates/ferrum-bench-core/Cargo.toml")
        .unwrap();
    *manifest = manifest.replace("['full']", "['full', 'visit-mut']");
    for base in [&before, &full] {
        assert!(validation_dependency_paths(base, &visitor).is_ok());
        for mutation in [
            "syn = { version = '3', features = ['full', 'visit-mut'] }",
            "syn = { version = '2', features = ['full', 'visit-mut', 'extra-traits'] }",
            "syn = { version = '2', features = ['full', 'visit-mut'], default-features = false }",
        ] {
            let mut changed = visitor.clone();
            let manifest = changed
                .get_mut("crates/ferrum-bench-core/Cargo.toml")
                .unwrap();
            *manifest = manifest.replace(
                "syn = { version = '2', features = ['full', 'visit-mut'] }",
                mutation,
            );
            // An original addition may specify a different compatible version
            // range; an existing dependency cannot change its declaration.
            if base == &full || !mutation.contains("version = '3'") {
                assert!(validation_dependency_paths(base, &changed).is_err());
            }
        }
        let mut changed = visitor.clone();
        let lock = changed.get_mut("Cargo.lock").unwrap();
        *lock = lock.replace("syntax-checksum", "different-checksum");
        assert!(validation_dependency_paths(base, &changed).is_err());
    }
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

fn add_private_tool(files: &mut BTreeMap<String, String>) {
    let root = files.get_mut("Cargo.toml").unwrap();
    *root = root.replace(
        "\"crates/app\"]",
        "\"crates/app\", \"crates/ferrum-devtools\"]",
    );
    files.insert(
        DEVTOOLS_MANIFEST.into(),
        r#"[package]
name = "ferrum-devtools"
version.workspace = true
edition = "2021"
publish = false
[[bin]]
name = "release_delivery"
path = "src/bin/release_delivery.rs"
[[bin]]
name = "contract_checks"
path = "src/bin/contract_checks.rs"
[dependencies]
ferrum-bench-core.workspace = true
serde = "1"
"#
        .into(),
    );
    append(
        files,
        "Cargo.lock",
        r#"[[package]]
name = "ferrum-devtools"
version = "1.0.0"
dependencies = ["ferrum-bench-core", "serde"]
"#,
    );
}

#[test]
fn isolated_private_tool_addition_and_moved_development_edge_keep_runtime_resolution() {
    let mut before = fixture();
    add_dev(&mut before);
    let mut after = before.clone();
    add_private_tool(&mut after);
    let bench = after
        .get_mut("crates/ferrum-bench-core/Cargo.toml")
        .unwrap();
    *bench = bench.replace("[dev-dependencies]\nfixture_test = '1'\n", "");
    append(
        &mut after,
        DEVTOOLS_MANIFEST,
        "fixture_test = '1'\ntool_support = '1'\n",
    );
    let lock = after.get_mut("Cargo.lock").unwrap();
    *lock = lock
        .replace(
            "dependencies = [\"serde\", \"fixture_test\"]",
            "dependencies = [\"serde\"]",
        )
        .replace(
            "dependencies = [\"ferrum-bench-core\", \"serde\"]",
            "dependencies = [\"ferrum-bench-core\", \"serde\", \"fixture_test\", \"tool_support\"]",
        );
    lock.push_str("[[package]]\nname = 'tool_support'\nversion = '1.0.0'\nsource = 'registry+https://example.invalid/index'\nchecksum = 'tool-support-checksum'\ndependencies = ['serde']\n");
    let result = validation_dependency_paths(&before, &after).unwrap();
    assert_eq!(result.private_tool_members, [DEVTOOLS]);
    assert!(result.paths.contains(&DEVTOOLS_MANIFEST.into()));
    assert!(result.paths.contains(&"Cargo.toml".into()));
    assert!(result.validation_runtime_dependencies.is_empty());
    let mut drift = after;
    let lock = drift.get_mut("Cargo.lock").unwrap();
    *lock = lock.replace("serde-checksum", "changed-runtime-checksum");
    assert!(validation_dependency_paths(&before, &drift).is_err());
}

#[test]
fn any_existing_member_or_locked_reverse_edge_into_private_tools_prevents_refinement() {
    let before = fixture();
    let mut after = before.clone();
    add_private_tool(&mut after);
    validation_dependency_paths(&before, &after).unwrap();
    for edge in [
        "release_tool = { package = 'ferrum-devtools', path = '../ferrum-devtools' }\n",
        "[build-dependencies]\nferrum-devtools = { path = '../ferrum-devtools' }\n",
        "[dev-dependencies]\nferrum-devtools = { path = '../ferrum-devtools' }\n",
        "[target.'cfg(unix)'.dependencies]\nrelease_tool = { package = 'ferrum-devtools', path = '../ferrum-devtools' }\n",
        "[target.'cfg(windows)'.build-dependencies]\nferrum-devtools = { path = '../ferrum-devtools' }\n",
    ] {
        let mut reversed = after.clone();
        append(&mut reversed, "crates/app/Cargo.toml", edge);
        assert!(validation_dependency_paths(&before, &reversed).unwrap_err().contains("depends on private devtools"));
    }
    let mut inherited = after.clone();
    append(
        &mut inherited,
        "Cargo.toml",
        "release_tool = { package = 'ferrum-devtools', path = 'crates/ferrum-devtools' }\n",
    );
    append(
        &mut inherited,
        "crates/app/Cargo.toml",
        "release_tool.workspace = true\n",
    );
    assert!(validation_dependency_paths(&before, &inherited)
        .unwrap_err()
        .contains("depends on private devtools"));
    let mut lock_only = after;
    let lock = lock_only.get_mut("Cargo.lock").unwrap();
    *lock = lock.replace(
        "checksum = \"serde-checksum\"",
        "checksum = \"serde-checksum\"\ndependencies = ['ferrum-devtools']",
    );
    assert!(validation_dependency_paths(&before, &lock_only)
        .unwrap_err()
        .contains("depends on private devtools"));
}

#[test]
fn private_tool_name_does_not_allow_publishing_build_hooks_unknown_members_or_lock_drift() {
    let before = fixture();
    let mut after = before.clone();
    add_private_tool(&mut after);
    for (old, new) in [
        ("publish = false", "publish = true"),
        ("publish = false", "publish = ['private-registry']"),
        ("publish = false", "publish = false\nbuild = 'build.rs'"),
        (
            "serde = \"1\"",
            "serde = { git = 'https://example.invalid/serde' }",
        ),
        ("[dependencies]", "[features]\nextra = []\n[dependencies]"),
        ("src/bin/contract_checks.rs", "src/lib.rs"),
    ] {
        let mut invalid = after.clone();
        let manifest = invalid.get_mut(DEVTOOLS_MANIFEST).unwrap();
        *manifest = manifest.replace(old, new);
        assert!(
            validation_dependency_paths(&before, &invalid).is_err(),
            "{new}"
        );
    }
    let mut added_member = after.clone();
    added_member.insert(
        "crates/unreviewed/Cargo.toml".into(),
        "[package]\nname='unreviewed'\nversion='1.0.0'\npublish=false\n".into(),
    );
    assert!(validation_dependency_paths(&before, &added_member).is_err());
    let mut unexplained = after.clone();
    append(&mut unexplained, "Cargo.lock", "[[package]]\nname='unexplained'\nversion='1.0.0'\nsource='registry+https://example.invalid/index'\n");
    assert!(validation_dependency_paths(&before, &unexplained).is_err());
    let mut bad_lock = after;
    let lock = bad_lock.get_mut("Cargo.lock").unwrap();
    *lock = lock.replace(
        "dependencies = [\"ferrum-bench-core\", \"serde\"]",
        "dependencies = [\"serde\"]",
    );
    assert!(validation_dependency_paths(&before, &bad_lock)
        .unwrap_err()
        .contains("manifest and locked dependency names differ"));
}

const TESTKIT_MANIFEST: &str = "crates/ferrum-testkit/Cargo.toml";
fn lock_edge(files: &mut BTreeMap<String, String>, package: &str, dependency: &str) {
    let mut lock = parse(files, "Cargo.lock").unwrap();
    let record = lock["package"]
        .as_array_of_tables_mut()
        .unwrap()
        .iter_mut()
        .find(|record| record["name"].as_str() == Some(package))
        .unwrap();
    if record.get("dependencies").is_none() {
        record["dependencies"] = toml_edit::value(toml_edit::Array::new());
    }
    record["dependencies"]
        .as_array_mut()
        .unwrap()
        .push(dependency);
    files.insert("Cargo.lock".into(), lock.to_string());
}
fn validation_tools_fixture(private_tool: bool) -> BTreeMap<String, String> {
    let mut files = fixture();
    if private_tool {
        add_private_tool(&mut files);
    }
    let root = files.get_mut("Cargo.toml").unwrap();
    *root = root.replace("members = [", "members = [\"crates/ferrum-testkit\", ");
    root.push_str("ferrum-testkit = {path='crates/ferrum-testkit',version='1.0.0'}\nchrono = {version='0.4',features=['serde']}\n");
    files.insert(TESTKIT_MANIFEST.into(),"[package]\nname='ferrum-testkit'\nversion.workspace=true\n[dependencies]\nserde='1'\n[features]\ndefault=[]\nmetal=[]\n".into());
    append(&mut files,"crates/app/Cargo.toml", "chrono.workspace=true\nzip={version='7.2.0',default-features=false}\nflate2='1'\n[dev-dependencies]\nferrum-testkit.workspace=true\n");
    append(
        &mut files,
        "Cargo.lock",
        r#"
[[package]]
name = "ferrum-testkit"
version = "1.0.0"
dependencies = ["serde"]
[[package]]
name = "chrono"
version = "0.4.42"
source = "registry+https://example.invalid/index"
checksum = "chrono-checksum"
[[package]]
name = "zip"
version = "7.2.0"
source = "registry+https://example.invalid/index"
checksum = "zip-checksum"
dependencies = ["crc32fast"]
[[package]]
name = "flate2"
version = "1.1.2"
source = "registry+https://example.invalid/index"
checksum = "flate-checksum"
dependencies = ["crc32fast"]
[[package]]
name = "crc32fast"
version = "1.5.0"
source = "registry+https://example.invalid/index"
checksum = "crc-checksum"
"#,
    );
    for dependency in ["chrono", "zip", "flate2", TESTKIT] {
        lock_edge(&mut files, "app", dependency);
    }
    files
}
fn add_numerical_sharing(files: &mut BTreeMap<String, String>) {
    let manifest = files.get_mut(TESTKIT_MANIFEST).unwrap();
    *manifest = manifest.replace(
        "[dependencies]",
        "[dependencies]\nferrum-bench-core.workspace=true",
    );
    lock_edge(files, TESTKIT, BENCH_CORE);
}
fn add_artifact_tools(files: &mut BTreeMap<String, String>) {
    append(files,DEVTOOLS_MANIFEST,"chrono.workspace=true\nzip={version='7.2.0',default-features=false,features=['deflate-flate2']}\nflate2='1'\n");
    for dependency in ["chrono", "zip", "flate2"] {
        lock_edge(files, DEVTOOLS, dependency);
    }
    lock_edge(files, "zip", "flate2");
}

#[test]
fn dev_only_testkit_can_share_existing_local_numerics_without_changing_product_edges() {
    let before = validation_tools_fixture(true);
    let mut after = before.clone();
    add_numerical_sharing(&mut after);
    let result = validation_dependency_paths(&before, &after).unwrap();
    assert_eq!(
        result.validation_runtime_dependencies,
        [format!("{TESTKIT_MANIFEST}:{BENCH_CORE}")]
    );
    for section in [
        "dependencies",
        "build-dependencies",
        "target.'cfg(unix)'.dependencies",
        "target.'cfg(windows)'.build-dependencies",
    ] {
        let mut runtime_before = before.clone();
        let manifest = runtime_before.get_mut("crates/app/Cargo.toml").unwrap();
        *manifest = manifest.replace("[dev-dependencies]", &format!("[{section}]"));
        // The normal dependency table already exists: insert a renamed edge
        // there instead of creating invalid TOML for the normal case.
        if section == "dependencies" {
            *manifest = before["crates/app/Cargo.toml"]
                .replace("[dev-dependencies]\nferrum-testkit.workspace=true\n", "")
                .replace("[dependencies]",
                    "[dependencies]\nrenamed_testkit={package='ferrum-testkit',path='../ferrum-testkit'}");
        }
        let mut invalid = runtime_before.clone();
        add_numerical_sharing(&mut invalid);
        // The runtime consumer already existed at the base: only the new
        // testkit edge changes, so manifest equality alone cannot detect reach.
        assert!(
            validation_dependency_paths(&runtime_before, &invalid)
                .unwrap_err()
                .contains("runtime/build dependency on testkit"),
            "{section}"
        );
    }
    for bad in [
        "ferrum-bench-core='1'",
        "ferrum-bench-core={workspace=true,features=['extra']}",
    ] {
        let mut invalid = after.clone();
        let manifest = invalid.get_mut(TESTKIT_MANIFEST).unwrap();
        *manifest = manifest.replace("ferrum-bench-core.workspace=true", bad);
        assert!(validation_dependency_paths(&before, &invalid).is_err());
    }
    let mut invalid = after;
    let mut lock = parse(&invalid, "Cargo.lock").unwrap();
    let record = lock["package"]
        .as_array_of_tables_mut()
        .unwrap()
        .iter_mut()
        .find(|record| record["name"].as_str() == Some(TESTKIT))
        .unwrap();
    let edges = record["dependencies"].as_array_mut().unwrap();
    let index = edges
        .iter()
        .position(|edge| edge.as_str() == Some(BENCH_CORE))
        .unwrap();
    edges.remove(index);
    invalid.insert("Cargo.lock".into(), lock.to_string());
    assert!(validation_dependency_paths(&before, &invalid).is_err());
}

#[test]
fn private_artifact_readers_extend_only_reviewed_locked_zip_feature_edge() {
    for added_tool in [false, true] {
        let before = validation_tools_fixture(!added_tool);
        let mut after = before.clone();
        if added_tool {
            add_private_tool(&mut after);
        }
        add_artifact_tools(&mut after);
        add_numerical_sharing(&mut after);
        let result = validation_dependency_paths(&before, &after).unwrap();
        assert!(result
            .validation_runtime_dependencies
            .contains(&format!("{DEVTOOLS_MANIFEST}:zip/deflate-flate2")));
        assert!(result
            .validation_runtime_dependencies
            .contains(&format!("{DEVTOOLS_MANIFEST}:chrono")));
        for (old, new) in [
            (
                "default-features=false,features=['deflate-flate2']",
                "default-features=true,features=['deflate-flate2']",
            ),
            (
                "features=['deflate-flate2']",
                "features=['deflate-flate2','aes-crypto']",
            ),
            (
                "version='7.2.0',default-features=false,features=['deflate-flate2']",
                "version='8',default-features=false,features=['deflate-flate2']",
            ),
            (
                "chrono.workspace=true",
                "chrono={workspace=true,features=['unstable-locales']}",
            ),
            ("flate2='1'", "flate2={version='1',default-features=false}"),
        ] {
            let mut invalid = after.clone();
            let manifest = invalid.get_mut(DEVTOOLS_MANIFEST).unwrap();
            *manifest = manifest.replace(old, new);
            assert!(
                validation_dependency_paths(&before, &invalid).is_err(),
                "{new}"
            );
        }
    }
}

#[test]
fn private_feature_exception_does_not_hide_shared_registry_or_product_changes() {
    let before = validation_tools_fixture(true);
    let mut after = before.clone();
    add_artifact_tools(&mut after);
    for (old, new) in [
        ("zip-checksum", "changed"),
        ("flate-checksum", "changed"),
        ("version = \"7.2.0\"", "version = \"7.2.1\""),
        (
            "registry+https://example.invalid/index",
            "registry+https://other.invalid/index",
        ),
    ] {
        let mut invalid = after.clone();
        let lock = invalid.get_mut("Cargo.lock").unwrap();
        *lock = lock.replace(old, new);
        assert!(
            validation_dependency_paths(&before, &invalid).is_err(),
            "{old}"
        );
    }
    for (package, dependency) in [("zip", "serde"), ("flate2", "serde"), ("serde", "flate2")] {
        let mut invalid = after.clone();
        lock_edge(&mut invalid, package, dependency);
        assert!(
            validation_dependency_paths(&before, &invalid).is_err(),
            "{package} -> {dependency}"
        );
    }
    let mut invalid = after.clone();
    let manifest = invalid.get_mut("crates/app/Cargo.toml").unwrap();
    *manifest = manifest.replace("default-features=false", "default-features=true");
    assert!(validation_dependency_paths(&before, &invalid).is_err());
    let mut invalid = after.clone();
    append(&mut invalid,"crates/app/Cargo.toml","[target.'cfg(unix)'.build-dependencies]\nprivate_tool={package='ferrum-devtools',path='../ferrum-devtools'}\n");
    assert!(validation_dependency_paths(&before, &invalid)
        .unwrap_err()
        .contains("depends on private devtools"));
    let mut invalid = after;
    let manifest = invalid.get_mut(DEVTOOLS_MANIFEST).unwrap();
    *manifest = manifest.replace("publish = false", "publish = true");
    assert!(validation_dependency_paths(&before, &invalid).is_err());
}

fn model_runner_migration_fixture() -> (BTreeMap<String, String>, BTreeMap<String, String>) {
    let mut before = fixture();
    add_private_tool(&mut before);
    let root = before.get_mut("Cargo.toml").unwrap();
    *root = root.replace("members = [", "members = [\"crates/ferrum-types\", ");
    root.push_str("ferrum-types = {path='crates/ferrum-types',version='1.0.0'}\nanyhow = '1'\n");
    before.insert(
        "crates/ferrum-types/Cargo.toml".into(),
        "[package]\nname='ferrum-types'\nversion.workspace=true\n".into(),
    );
    append(
        &mut before,
        "crates/app/Cargo.toml",
        "anyhow.workspace=true\n",
    );
    append(
        &mut before,
        DEVTOOLS_MANIFEST,
        "[dev-dependencies]\nferrum-types.workspace=true\n",
    );
    append(&mut before, "Cargo.lock", "[[package]]\nname='ferrum-types'\nversion='1.0.0'\n[[package]]\nname='anyhow'\nversion='1.0.100'\nsource='registry+https://example.invalid/index'\nchecksum='anyhow-checksum'\n");
    lock_edge(&mut before, "app", "anyhow");
    lock_edge(&mut before, DEVTOOLS, "ferrum-types");
    let mut after = before.clone();
    let manifest = after.get_mut(DEVTOOLS_MANIFEST).unwrap();
    *manifest = manifest.replace("[dependencies]",
        "[[bin]]\nname = \"model_regression\"\npath = \"src/bin/model_regression.rs\"\n[dependencies]\nanyhow.workspace=true\nferrum-types.workspace=true")
        .replace("[dev-dependencies]\nferrum-types.workspace=true\n", "");
    lock_edge(&mut after, DEVTOOLS, "anyhow");
    (before, after)
}

#[test]
fn model_runner_migration_preserves_product_and_locked_dependency_identities() {
    let (before, after) = model_runner_migration_fixture();
    // Both the original private two-bin layout and the migrated layout remain
    // accepted; no product normal/build dependency or shared package changes.
    validation_dependency_paths(&before, &before).unwrap();
    let result = validation_dependency_paths(&before, &after).unwrap();
    assert_eq!(result.paths, ["Cargo.lock", DEVTOOLS_MANIFEST]);
    assert!(result
        .validation_runtime_dependencies
        .contains(&format!("{DEVTOOLS_MANIFEST}:anyhow")));
    assert!(result
        .validation_runtime_dependencies
        .contains(&format!("{DEVTOOLS_MANIFEST}:ferrum-types")));
    validation_dependency_paths(&after, &after).unwrap();
}

#[test]
fn model_runner_migration_cannot_remove_existing_tools_or_add_unreviewed_targets_and_hooks() {
    let (before, after) = model_runner_migration_fixture();
    for (old, new) in [
        (
            "[[bin]]\nname = \"release_delivery\"\npath = \"src/bin/release_delivery.rs\"\n",
            "",
        ),
        (
            "[[bin]]\nname = \"contract_checks\"\npath = \"src/bin/contract_checks.rs\"\n",
            "",
        ),
        ("model_regression", "unknown_runner"),
        ("src/bin/model_regression.rs", "src/lib.rs"),
        ("publish = false", "publish = true"),
        ("publish = false", "publish = false\nbuild = 'build.rs'"),
        (
            "[dependencies]",
            "[[bin]]\nname='unreviewed'\npath='src/bin/unreviewed.rs'\n[dependencies]",
        ),
        (
            "[dependencies]",
            "[target.'cfg(unix)'.dependencies]\nserde='1'\n[dependencies]",
        ),
    ] {
        let mut invalid = after.clone();
        let manifest = invalid.get_mut(DEVTOOLS_MANIFEST).unwrap();
        assert!(manifest.contains(old), "missing fixture mutation {old}");
        *manifest = manifest.replace(old, new);
        assert!(
            validation_dependency_paths(&before, &invalid).is_err(),
            "{old} -> {new}"
        );
    }
    for edge in [
        "private_tool={package='ferrum-devtools',path='../ferrum-devtools'}\n",
        "[target.'cfg(unix)'.build-dependencies]\nprivate_tool={package='ferrum-devtools',path='../ferrum-devtools'}\n",
    ] {
        let mut invalid = after.clone();
        append(&mut invalid, "crates/app/Cargo.toml", edge);
        assert!(validation_dependency_paths(&before, &invalid).unwrap_err().contains("depends on private devtools"));
    }
    let mut invalid = after;
    lock_edge(&mut invalid, "serde", DEVTOOLS);
    assert!(validation_dependency_paths(&before, &invalid)
        .unwrap_err()
        .contains("depends on private devtools"));
}

#[test]
fn model_runner_migration_does_not_relax_shared_flags_versions_or_lock_edges() {
    let (before, after) = model_runner_migration_fixture();
    for (old, new) in [
        (
            "anyhow.workspace=true",
            "anyhow={workspace=true,default-features=false}",
        ),
        (
            "ferrum-types.workspace=true",
            "ferrum-types={workspace=true,features=['extra']}",
        ),
        ("ferrum-types.workspace=true", "ferrum-types='1'"),
        ("serde = \"1\"", "serde = '2'"),
    ] {
        let mut invalid = after.clone();
        let manifest = invalid.get_mut(DEVTOOLS_MANIFEST).unwrap();
        *manifest = manifest.replace(old, new);
        assert!(
            validation_dependency_paths(&before, &invalid).is_err(),
            "{new}"
        );
    }
    for (old, new) in [
        ("anyhow-checksum", "different-checksum"),
        ("1.0.100", "1.0.101"),
        (
            "registry+https://example.invalid/index",
            "registry+https://elsewhere.invalid/index",
        ),
    ] {
        let mut invalid = after.clone();
        let lock = invalid.get_mut("Cargo.lock").unwrap();
        assert!(lock.contains(old));
        *lock = lock.replace(old, new);
        assert!(
            validation_dependency_paths(&before, &invalid).is_err(),
            "{new}"
        );
    }
    for (package, dependency) in [("anyhow", "serde"), ("app", "ferrum-types")] {
        let mut invalid = after.clone();
        lock_edge(&mut invalid, package, dependency);
        assert!(
            validation_dependency_paths(&before, &invalid).is_err(),
            "{package} -> {dependency}"
        );
    }
    let mut incomplete = after;
    let mut lock = parse(&incomplete, "Cargo.lock").unwrap();
    let tool = lock["package"]
        .as_array_of_tables_mut()
        .unwrap()
        .iter_mut()
        .find(|record| record["name"].as_str() == Some(DEVTOOLS))
        .unwrap();
    let dependencies = tool["dependencies"].as_array_mut().unwrap();
    let index = dependencies
        .iter()
        .position(|value| value.as_str() == Some("anyhow"))
        .unwrap();
    dependencies.remove(index);
    incomplete.insert("Cargo.lock".into(), lock.to_string());
    let error = validation_dependency_paths(&before, &incomplete).unwrap_err();
    assert!(
        error.contains("missing private tool lock edge to anyhow"),
        "{error}"
    );
}
