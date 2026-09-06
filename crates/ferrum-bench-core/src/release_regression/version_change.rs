//! Recognize only complete coordinated version edits in Git manifest snapshots.
//! Unsupported membership or extra edits retain the caller's conservative impact.
use crate::release_candidate::version::{plan_version_update, MemberManifest};
use std::collections::{BTreeMap, BTreeSet};
use toml_edit::{DocumentMut, Item};

fn document(text: &str, path: &str) -> Result<DocumentMut, String> {
    text.parse()
        .map_err(|error| format!("invalid {path}: {error}"))
}

fn snapshot<'a>(files: &'a BTreeMap<String, String>, path: &str) -> Result<&'a str, String> {
    files
        .get(path)
        .map(String::as_str)
        .ok_or_else(|| format!("incomplete version snapshot: missing {path}"))
}

fn member_paths(root: &DocumentMut) -> Result<BTreeSet<String>, String> {
    let workspace = root
        .get("workspace")
        .and_then(Item::as_table_like)
        .ok_or("workspace must be a table")?;
    if let Some(exclude) = workspace.get("exclude") {
        if !exclude.as_array().is_some_and(|paths| paths.is_empty()) {
            return Err("workspace exclusions require conservative impact analysis".into());
        }
    }
    let members = workspace
        .get("members")
        .and_then(Item::as_array)
        .ok_or("workspace.members must be an explicit array")?;
    let mut paths = BTreeSet::new();
    for member in members {
        let member = member.as_str().ok_or("workspace member must be a string")?;
        if member.contains(['\\', ':', '*', '?', '[', ']', '{', '}'])
            || member.chars().any(char::is_control)
            || member
                .split('/')
                .any(|part| matches!(part, "" | "." | ".."))
        {
            return Err(format!("unsupported workspace member path: {member:?}"));
        }
        if !paths.insert(format!("{member}/Cargo.toml")) {
            return Err(format!("duplicate workspace member: {member}"));
        }
    }
    if root.get("package").is_some() {
        paths.insert("Cargo.toml".into());
    }
    Ok(paths)
}

fn local_lock_names(lock: &str) -> Result<BTreeSet<String>, String> {
    let lock = document(lock, "Cargo.lock")?;
    let packages = lock
        .get("package")
        .and_then(Item::as_array_of_tables)
        .ok_or("Cargo.lock must contain package records")?;
    let mut names = BTreeSet::new();
    for package in packages.iter() {
        if package.contains_key("source") {
            continue;
        }
        let name = package
            .get("name")
            .and_then(Item::as_str)
            .ok_or("Cargo.lock source-less package must have a name")?;
        if !names.insert(name.to_owned()) {
            return Err(format!("duplicate source-less Cargo.lock package: {name}"));
        }
    }
    Ok(names)
}

/// Return changed manifest/lock paths only when the complete after snapshot is
/// exactly the existing release preparer's coordinated version plan. The maps
/// must contain root Cargo.toml, Cargo.lock and all tracked member manifests;
/// unchanged extra manifests are allowed. This never discovers Git or writes.
/// Errors deliberately mean that no version-only refinement is safe.
pub fn coordinated_version_paths(
    before: &BTreeMap<String, String>,
    after: &BTreeMap<String, String>,
) -> Result<Vec<String>, String> {
    if !before.keys().eq(after.keys()) {
        return Err("version snapshots added or removed manifest/lock paths".into());
    }
    let root_text = snapshot(before, "Cargo.toml")?;
    let root = document(root_text, "Cargo.toml")?;
    let next_root = document(snapshot(after, "Cargo.toml")?, "after Cargo.toml")?;
    let target = next_root
        .get("workspace")
        .and_then(Item::as_table_like)
        .and_then(|workspace| workspace.get("package"))
        .and_then(Item::as_table_like)
        .and_then(|package| package.get("version"))
        .and_then(Item::as_str)
        .ok_or("after workspace.package.version must be a string")?;
    let mut members = Vec::new();
    for path in member_paths(&root)? {
        let text = snapshot(before, &path)?;
        let manifest = document(text, &path)?;
        let name = manifest
            .get("package")
            .and_then(Item::as_table_like)
            .and_then(|package| package.get("name"))
            .and_then(Item::as_str)
            .ok_or_else(|| format!("{path}: package.name must be a string"))?;
        members.push(MemberManifest {
            package_name: name.to_owned(),
            manifest_path: path,
            text: text.to_owned(),
        });
    }
    let lock = snapshot(before, "Cargo.lock")?;
    let names: BTreeSet<_> = members
        .iter()
        .map(|member| member.package_name.clone())
        .collect();
    if local_lock_names(lock)? != names {
        return Err(
            "source-less Cargo.lock packages differ from explicit workspace members; implicit or external path members need conservative analysis".into(),
        );
    }
    let plan = plan_version_update(root_text, &members, lock, target)?;
    let mut expected = before.clone();
    expected.insert("Cargo.toml".into(), plan.workspace_manifest);
    expected.insert("Cargo.lock".into(), plan.lockfile);
    for member in plan.members {
        expected.insert(member.manifest_path, member.text);
    }
    if let Some(path) = expected
        .iter()
        .find_map(|(path, text)| (after.get(path) != Some(text)).then_some(path))
    {
        return Err(format!(
            "{path} contains changes beyond the coordinated version plan"
        ));
    }
    Ok(expected
        .into_iter()
        .filter_map(|(path, text)| (before.get(&path) != Some(&text)).then_some(path))
        .collect())
}

#[cfg(test)]
#[path = "version_change_tests.rs"]
mod tests;
