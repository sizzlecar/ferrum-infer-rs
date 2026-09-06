//! Pure, coordinated workspace version edits. Metadata discovery and filesystem
//! application belong to the caller; this module neither writes nor publishes.
use semver::{Version, VersionReq};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use toml_edit::{DocumentMut, Item, TableLike, Value};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MemberManifest {
    /// Actual package name from Cargo metadata, not the dependency's alias.
    pub package_name: String,
    /// Workspace-relative manifest path, e.g. crates/types/Cargo.toml.
    pub manifest_path: String,
    pub text: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VersionUpdatePlan {
    pub previous_version: String,
    pub target_version: String,
    pub workspace_manifest: String,
    pub members: Vec<MemberManifest>,
    pub lockfile: String,
}

fn parse_document(text: &str, label: &str) -> Result<DocumentMut, String> {
    text.parse()
        .map_err(|error| format!("invalid {label}: {error}"))
}

fn get_string(item: Option<&Item>, label: &str) -> Result<String, String> {
    item.and_then(Item::as_str)
        .map(str::to_owned)
        .ok_or_else(|| format!("{label} must be a string"))
}

fn replace_string(item: &mut Item, replacement: &str, label: &str) -> Result<(), String> {
    let value = item
        .as_value_mut()
        .ok_or_else(|| format!("{label} must be a string"))?;
    replace_value_string(value, replacement, label)
}

fn replace_value_string(value: &mut Value, replacement: &str, label: &str) -> Result<(), String> {
    if value.as_str().is_none() {
        return Err(format!("{label} must be a string"));
    }
    let decor = value.decor().clone();
    *value = Value::from(replacement);
    *value.decor_mut() = decor;
    Ok(())
}

fn normalized_path(base: &str, path: &str) -> Result<String, String> {
    if path.is_empty()
        || path.starts_with('/')
        || path.contains(['\\', '\0'])
        || path
            .split('/')
            .next()
            .is_some_and(|part| part.contains(':'))
    {
        return Err(format!(
            "unsupported non-relative dependency/manifest path: {path:?}"
        ));
    }
    let mut parts: Vec<&str> = base.split('/').filter(|part| !part.is_empty()).collect();
    for part in path.split('/') {
        match part {
            "" | "." => {}
            ".." => {
                if parts.pop().is_none() {
                    return Err(format!("path escapes workspace: {path:?}"));
                }
            }
            _ => parts.push(part),
        }
    }
    Ok(parts.join("/"))
}

fn parent(path: &str) -> &str {
    path.rsplit_once('/').map_or("", |(directory, _)| directory)
}

fn next_constraint(original: &str, previous: &Version, target: &str) -> Result<String, String> {
    let trimmed = original.trim();
    let (operator, version) = match trimmed.as_bytes().first() {
        Some(b'^') => ("^", trimmed[1..].trim()),
        Some(b'~') => ("~", trimmed[1..].trim()),
        Some(b'=') => ("=", trimmed[1..].trim()),
        _ => ("", trimmed),
    };
    Version::parse(version).map_err(|_| format!(
        "unsupported coordinated dependency constraint {original:?}; expected a full version with optional ^, ~ or ="))?;
    let requirement = VersionReq::parse(trimmed)
        .map_err(|error| format!("invalid dependency constraint {original:?}: {error}"))?;
    if !requirement.matches(previous) {
        return Err(format!("internal dependency constraint {original:?} does not accept current workspace version {previous}"));
    }
    Ok(format!("{operator}{target}"))
}

struct WorkspacePackages {
    /// Manifest-directory identity is needed to distinguish a path dependency
    /// from an unrelated registry package with the same name.
    by_directory: BTreeMap<String, String>,
    names: BTreeSet<String>,
    inherited_dependencies: BTreeSet<String>,
}

fn update_dependency_table(
    table: &mut dyn TableLike,
    manifest: &str,
    packages: &WorkspacePackages,
    previous: &Version,
    target: &str,
) -> Result<(), String> {
    for (key, item) in table.iter_mut() {
        let alias = key.get().to_owned();
        let Some(dependency) = item.as_table_like_mut() else {
            // A plain version string denotes a registry dependency, even if its
            // package name happens to match a workspace member.
            if item.as_str().is_some() {
                continue;
            }
            return Err(format!(
                "{manifest}: dependency {alias} is not a string or table"
            ));
        };
        if let Some(inherited) = dependency.get("workspace") {
            if inherited.as_bool() != Some(true) {
                return Err(format!("{manifest}: {alias}.workspace must be true"));
            }
            if !packages.inherited_dependencies.contains(&alias) {
                return Err(format!("{manifest}: inherited dependency {alias} is absent from workspace.dependencies"));
            }
            if dependency.contains_key("path")
                || dependency.contains_key("version")
                || dependency.contains_key("package")
            {
                return Err(format!(
                    "{manifest}: inherited dependency {alias} overrides path/version/package"
                ));
            }
            continue;
        }
        let Some(path_item) = dependency.get("path") else {
            continue;
        };
        let path = get_string(Some(path_item), &format!("{manifest}: {alias}.path"))?;
        let package_name = match dependency.get("package") {
            Some(item) => get_string(Some(item), &format!("{manifest}: {alias}.package"))?,
            None => alias.clone(),
        };
        let resolved = normalized_path(parent(manifest), &path);
        let resolved = match resolved {
            Ok(path) => path,
            Err(error) if packages.names.contains(&package_name) => {
                return Err(format!("{manifest}: {alias}: {error}"))
            }
            Err(_) => continue, // External path dependency is outside coordinated membership.
        };
        let Some(expected_name) = packages.by_directory.get(&resolved) else {
            if packages.names.contains(&package_name) {
                return Err(format!("{manifest}: path dependency {alias} names workspace package {package_name} but does not resolve to its metadata member directory"));
            }
            continue;
        };
        if expected_name != &package_name {
            return Err(format!("{manifest}: dependency {alias} names {package_name}, but path {path:?} identifies {expected_name}; use package for renames"));
        }
        if dependency.contains_key("git")
            || dependency.contains_key("branch")
            || dependency.contains_key("rev")
            || dependency.contains_key("tag")
        {
            return Err(format!(
                "{manifest}: internal path dependency {alias} also declares a Git source"
            ));
        }
        match dependency.get_mut("version") {
            Some(version) => {
                let original = get_string(Some(version), &format!("{manifest}: {alias}.version"))?;
                let replacement = next_constraint(&original, previous, target)
                    .map_err(|error| format!("{manifest}: {alias}: {error}"))?;
                replace_string(
                    version,
                    &replacement,
                    &format!("{manifest}: {alias}.version"),
                )?;
            }
            None => {
                dependency.insert("version", toml_edit::value(target));
            }
        }
    }
    Ok(())
}

fn update_dependencies(
    document: &mut DocumentMut,
    manifest: &str,
    packages: &WorkspacePackages,
    previous: &Version,
    target: &str,
) -> Result<(), String> {
    const DEPENDENCY_TABLES: &[&str] = &[
        "dependencies",
        "build-dependencies",
        "dev-dependencies",
        "build_dependencies",
        "dev_dependencies",
    ];
    fn section(
        item: &mut Item,
        label: &str,
        manifest: &str,
        packages: &WorkspacePackages,
        previous: &Version,
        target: &str,
    ) -> Result<(), String> {
        let table = item
            .as_table_like_mut()
            .ok_or_else(|| format!("{manifest}: {label} must be a table"))?;
        update_dependency_table(table, manifest, packages, previous, target)
    }
    for name in DEPENDENCY_TABLES {
        if let Some(item) = document.get_mut(name) {
            section(item, name, manifest, packages, previous, target)?;
        }
    }
    if let Some(item) = document.get_mut("target") {
        let targets = item
            .as_table_like_mut()
            .ok_or_else(|| format!("{manifest}: target must be a table"))?;
        for (_, item) in targets.iter_mut() {
            let configuration = item
                .as_table_like_mut()
                .ok_or_else(|| format!("{manifest}: target configuration must be a table"))?;
            for name in DEPENDENCY_TABLES {
                if let Some(item) = configuration.get_mut(name) {
                    section(item, name, manifest, packages, previous, target)?;
                }
            }
        }
    }
    if manifest == "Cargo.toml" {
        if let Some(dependencies) = document
            .get_mut("workspace")
            .and_then(Item::as_table_like_mut)
            .and_then(|workspace| workspace.get_mut("dependencies"))
        {
            section(
                dependencies,
                "workspace.dependencies",
                manifest,
                packages,
                previous,
                target,
            )?;
        }
    }
    // Path patches can refer to the very same workspace package; keep their
    // version predicates coordinated without touching unrelated registry patches.
    if let Some(item) = document.get_mut("patch") {
        let registries = item
            .as_table_like_mut()
            .ok_or_else(|| format!("{manifest}: patch must be a table"))?;
        for (_, item) in registries.iter_mut() {
            section(item, "patch", manifest, packages, previous, target)?;
        }
    }
    if document.get("replace").is_some() {
        return Err(format!("{manifest}: legacy [replace] constraints need explicit migration before coordinated version editing"));
    }
    Ok(())
}

fn update_lock(
    lockfile: &str,
    names: &BTreeSet<String>,
    previous: &Version,
    target: &str,
) -> Result<String, String> {
    let mut document = parse_document(lockfile, "Cargo.lock")?;
    let packages = document
        .get_mut("package")
        .and_then(Item::as_array_of_tables_mut)
        .ok_or("Cargo.lock must contain package records")?;
    let mut external_versions = BTreeSet::new();
    for package in packages.iter() {
        if package.contains_key("source") {
            external_versions.insert((
                get_string(package.get("name"), "Cargo.lock package.name")?,
                get_string(package.get("version"), "Cargo.lock package.version")?,
            ));
        }
    }
    let mut seen = BTreeSet::new();
    for package in packages.iter_mut() {
        let name = get_string(package.get("name"), "Cargo.lock package.name")?;
        if package.contains_key("source") || !names.contains(&name) {
            continue;
        }
        if !seen.insert(name.clone()) {
            return Err(format!(
                "Cargo.lock has multiple source-less records for workspace package {name}"
            ));
        }
        let old = get_string(package.get("version"), "Cargo.lock package.version")?;
        if Version::parse(&old).map_err(|error| format!("Cargo.lock {name}: {error}"))? != *previous
        {
            return Err(format!(
                "Cargo.lock local package {name} has version {old}, expected {previous}"
            ));
        }
        replace_string(
            package.get_mut("version").expect("checked version exists"),
            target,
            "Cargo.lock package.version",
        )?;
    }
    if let Some(name) = names.difference(&seen).next() {
        return Err(format!(
            "Cargo.lock is missing source-less workspace package {name}"
        ));
    }
    for package in packages.iter_mut() {
        if let Some(item) = package.get_mut("dependencies") {
            let dependencies = item
                .as_array_mut()
                .ok_or("Cargo.lock package.dependencies must be an array")?;
            for dependency in dependencies.iter_mut() {
                let original = dependency
                    .as_str()
                    .ok_or("Cargo.lock dependency reference must be a string")?
                    .to_owned();
                let fields: Vec<_> = original.split_whitespace().collect();
                // A qualified registry/git source is intentionally left intact.
                // Bare names remain valid; only local name+version references move.
                if fields.len() == 2 && names.contains(fields[0]) {
                    let old = Version::parse(fields[1])
                        .map_err(|error| format!("Cargo.lock dependency {original:?}: {error}"))?;
                    if old != *previous {
                        if external_versions.contains(&(fields[0].to_owned(), fields[1].to_owned()))
                        {
                            if fields[1] == target {
                                return Err(format!("Cargo.lock external dependency reference {original:?} would collide with the new local version; qualify its source before coordinated version editing"));
                            }
                            continue;
                        }
                        return Err(format!("Cargo.lock local dependency reference {original:?} does not match {previous}"));
                    }
                    replace_value_string(
                        dependency,
                        &format!("{} {target}", fields[0]),
                        "Cargo.lock dependency",
                    )?;
                }
            }
        }
    }
    Ok(document.to_string())
}

/// Produce a complete text plan from actual Cargo metadata members. All supplied
/// members participate in one coordinated release; inconsistent pre-existing
/// versions fail rather than silently changing an independently versioned crate.
/// The caller must supply workspace-relative paths resolved from metadata; this
/// pure module cannot resolve symlink aliases or discover missing members.
pub fn plan_version_update(
    workspace_manifest: &str,
    members: &[MemberManifest],
    lockfile: &str,
    target_version: &str,
) -> Result<VersionUpdatePlan, String> {
    let target = Version::parse(target_version)
        .map_err(|error| format!("invalid target version: {error}"))?;
    if !target.pre.is_empty() || !target.build.is_empty() {
        return Err(
            "target version must be a formal semantic version without prerelease or build metadata"
                .into(),
        );
    }
    let mut root = parse_document(workspace_manifest, "workspace Cargo.toml")?;
    let previous_text = get_string(
        root.get("workspace")
            .and_then(Item::as_table_like)
            .and_then(|workspace| workspace.get("package"))
            .and_then(Item::as_table_like)
            .and_then(|package| package.get("version")),
        "workspace.package.version",
    )?;
    let previous = Version::parse(&previous_text)
        .map_err(|error| format!("invalid current workspace version: {error}"))?;
    if target <= previous {
        return Err(format!(
            "target version {target} must increase current workspace version {previous}"
        ));
    }
    if members.is_empty() {
        return Err("Cargo metadata returned no workspace members".into());
    }
    let target = target.to_string();
    let inherited_dependencies = root
        .get("workspace")
        .and_then(Item::as_table_like)
        .and_then(|workspace| workspace.get("dependencies"))
        .and_then(Item::as_table_like)
        .map(|dependencies| {
            dependencies
                .iter()
                .map(|(name, _)| name.to_owned())
                .collect()
        })
        .unwrap_or_default();
    let mut packages = WorkspacePackages {
        by_directory: BTreeMap::new(),
        names: BTreeSet::new(),
        inherited_dependencies,
    };
    let mut documents = BTreeMap::<String, DocumentMut>::new();
    for member in members {
        if member.package_name.trim().is_empty()
            || member.package_name.trim() != member.package_name
        {
            return Err("metadata package names must be nonblank and unpadded".into());
        }
        let path = normalized_path("", &member.manifest_path)?;
        if !path.ends_with("Cargo.toml") || path.rsplit('/').next() != Some("Cargo.toml") {
            return Err(format!(
                "metadata member manifest must identify Cargo.toml: {path}"
            ));
        }
        if documents.contains_key(&path) || !packages.names.insert(member.package_name.clone()) {
            return Err("metadata member paths and package names must be unique".into());
        }
        let document = parse_document(&member.text, &path)?;
        let actual = get_string(
            document
                .get("package")
                .and_then(Item::as_table_like)
                .and_then(|package| package.get("name")),
            &format!("{path}: package.name"),
        )?;
        if actual != member.package_name {
            return Err(format!(
                "metadata member {} disagrees with manifest package {actual}",
                member.package_name
            ));
        }
        if path == "Cargo.toml" && member.text != workspace_manifest {
            return Err("root package member text differs from supplied workspace manifest".into());
        }
        packages.by_directory.insert(parent(&path).into(), actual);
        documents.insert(path, document);
    }
    if let Some(root_member) = documents.remove("Cargo.toml") {
        root = root_member;
    }
    documents.insert("Cargo.toml".into(), root);
    for member in members {
        let path = normalized_path("", &member.manifest_path)?;
        let package = documents
            .get_mut(&path)
            .expect("member was parsed")
            .get_mut("package")
            .and_then(Item::as_table_like_mut)
            .ok_or_else(|| format!("{path}: package must be a table"))?;
        let version = package.get_mut("version").ok_or_else(|| {
            format!("{path}: package.version is required for a coordinated release")
        })?;
        if let Some(original) = version.as_str() {
            let parsed = Version::parse(original)
                .map_err(|error| format!("{path}: invalid package.version: {error}"))?;
            if parsed != previous {
                return Err(format!("{path}: explicit version {original} differs from coordinated workspace version {previous}"));
            }
            replace_string(version, &target, &format!("{path}: package.version"))?;
        } else if version
            .as_table_like()
            .and_then(|value| value.get("workspace"))
            .and_then(Item::as_bool)
            != Some(true)
        {
            return Err(format!(
                "{path}: package.version must be a version string or inherit workspace=true"
            ));
        }
    }
    let root = documents.get_mut("Cargo.toml").expect("root is present");
    let version = root
        .get_mut("workspace")
        .and_then(Item::as_table_like_mut)
        .and_then(|workspace| workspace.get_mut("package"))
        .and_then(Item::as_table_like_mut)
        .and_then(|package| package.get_mut("version"))
        .expect("workspace version was checked");
    replace_string(version, &target, "workspace.package.version")?;
    for (path, document) in &mut documents {
        update_dependencies(document, path, &packages, &previous, &target)?;
    }
    let updated_lock = update_lock(lockfile, &packages.names, &previous, &target)?;
    Ok(VersionUpdatePlan {
        previous_version: previous.to_string(),
        target_version: target,
        workspace_manifest: documents["Cargo.toml"].to_string(),
        members: members
            .iter()
            .map(|member| {
                Ok(MemberManifest {
                    package_name: member.package_name.clone(),
                    manifest_path: member.manifest_path.clone(),
                    text: documents[&normalized_path("", &member.manifest_path)?].to_string(),
                })
            })
            .collect::<Result<_, String>>()?,
        lockfile: updated_lock,
    })
}

#[cfg(test)]
#[path = "version_tests.rs"]
mod tests;
