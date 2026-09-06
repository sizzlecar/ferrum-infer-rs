use super::*;

fn manifests(version: &str) -> BTreeMap<String, String> {
    BTreeMap::from([
        (
            "Cargo.toml".into(),
            format!(
                r#"[workspace]
members = ["crates/a", "crates/b"]
resolver = "2"
[workspace.package]
version = "{version}"
[workspace.dependencies]
a = {{ path = "crates/a", version = "^{version}", features = ["small"] }}
core-foundation-sys = "0.8.7"
[profile.release]
opt-level = 3
"#
            ),
        ),
        (
            "crates/a/Cargo.toml".into(),
            "[package]\nname = \"a\"\nversion.workspace = true\n[features]\nsmall = []\nlarge = []\n".into(),
        ),
        (
            "crates/b/Cargo.toml".into(),
            format!(
                r#"[package]
name = "b"
version = "{version}"
[dependencies]
a = {{ path = "../a", version = "={version}", optional = true }}
core-foundation-sys = {{ workspace = true }}
"#
            ),
        ),
        (
            "Cargo.lock".into(),
            format!(
                r#"version = 4
[[package]]
name = "a"
version = "{version}"
[[package]]
name = "b"
version = "{version}"
dependencies = ["a {version}", "core-foundation-sys"]
[[package]]
name = "core-foundation-sys"
version = "0.8.7"
source = "registry+https://github.com/rust-lang/crates.io-index"
"#
            ),
        ),
    ])
}

#[test]
fn recognizes_coordinated_version_paths_without_touching_registry_package() {
    let before = manifests("0.8.7");
    let after = manifests("0.8.8");
    assert_eq!(
        coordinated_version_paths(&before, &after).unwrap(),
        vec!["Cargo.lock", "Cargo.toml", "crates/b/Cargo.toml"]
    );
    assert!(after["Cargo.lock"].contains("name = \"core-foundation-sys\"\nversion = \"0.8.7\""));
}

#[test]
fn recognizes_explicit_root_package_alongside_members() {
    let with_root = |version| {
        let mut files = manifests(version);
        files.get_mut("Cargo.toml").unwrap().push_str(&format!(
            "[package]\nname = \"root\"\nversion = \"{version}\"\n"
        ));
        files.get_mut("Cargo.lock").unwrap().push_str(&format!(
            "[[package]]\nname = \"root\"\nversion = \"{version}\"\n"
        ));
        files
    };
    assert!(coordinated_version_paths(&with_root("0.8.7"), &with_root("0.8.8")).is_ok());
}

#[test]
fn rejects_dependency_flags_build_and_registry_changes_with_version_bump() {
    for (path, old, new) in [
        (
            "Cargo.toml",
            "core-foundation-sys = \"0.8.7\"",
            "core-foundation-sys = \"0.8.8\"",
        ),
        (
            "Cargo.toml",
            "features = [\"small\"]",
            "features = [\"large\"]",
        ),
        ("crates/b/Cargo.toml", "optional = true", "optional = false"),
        ("Cargo.toml", "resolver = \"2\"", "resolver = \"3\""),
        ("Cargo.toml", "opt-level = 3", "opt-level = 2"),
        (
            "Cargo.lock",
            "registry+https://github.com/rust-lang/crates.io-index",
            "registry+https://example.invalid/index",
        ),
        ("crates/b/Cargo.toml", "name = \"b\"", "name = \"renamed\""),
    ] {
        let mut after = manifests("0.8.8");
        let text = after.get_mut(path).unwrap();
        assert!(text.contains(old));
        *text = text.replace(old, new);
        assert!(
            coordinated_version_paths(&manifests("0.8.7"), &after).is_err(),
            "accepted {path}: {new}"
        );
    }
}

#[test]
fn rejects_incomplete_snapshots_and_membership_changes() {
    let before = manifests("0.8.7");
    for missing in ["Cargo.toml", "Cargo.lock", "crates/a/Cargo.toml"] {
        let mut incomplete_before = before.clone();
        let mut incomplete_after = manifests("0.8.8");
        incomplete_before.remove(missing);
        incomplete_after.remove(missing);
        assert!(
            coordinated_version_paths(&incomplete_before, &incomplete_after).is_err(),
            "accepted missing {missing}"
        );
    }
    let mut after = manifests("0.8.8");
    after.remove("crates/a/Cargo.toml");
    assert!(coordinated_version_paths(&before, &after).is_err());
    let mut after = manifests("0.8.8");
    after.insert(
        "crates/new/Cargo.toml".into(),
        "[package]\nname = \"new\"\nversion = \"0.8.8\"\n".into(),
    );
    assert!(coordinated_version_paths(&before, &after).is_err());
    let mut after = manifests("0.8.8");
    *after.get_mut("Cargo.toml").unwrap() = after["Cargo.toml"].replace(
        "members = [\"crates/a\", \"crates/b\"]",
        "members = [\"crates/a\"]",
    );
    assert!(coordinated_version_paths(&before, &after).is_err());
}

#[test]
fn rejects_implicit_local_packages_and_unsupported_member_paths() {
    for member in ["crates/*", "../external", "./crates/a", "/crates/a"] {
        let mut before = manifests("0.8.7");
        let mut after = manifests("0.8.8");
        for files in [&mut before, &mut after] {
            *files.get_mut("Cargo.toml").unwrap() =
                files["Cargo.toml"].replace("\"crates/a\"", &format!("\"{member}\""));
        }
        assert!(
            coordinated_version_paths(&before, &after).is_err(),
            "accepted {member}"
        );
    }
    let mut before = manifests("0.8.7");
    let mut after = manifests("0.8.8");
    for files in [&mut before, &mut after] {
        files
            .get_mut("Cargo.lock")
            .unwrap()
            .push_str("[[package]]\nname = \"implicit-path-package\"\nversion = \"1.0.0\"\n");
    }
    assert!(coordinated_version_paths(&before, &after).is_err());
}
