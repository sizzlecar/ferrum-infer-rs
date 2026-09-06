//! Cargo discovers membership; edits preserve formatting and unrelated files.
use ferrum_bench_core::release_candidate::{
    staging,
    version::{self, MemberManifest},
};
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
};

fn cargo_metadata(directory: &Path, no_deps: bool) -> Result<Value, String> {
    let mut command = Command::new(std::env::var_os("CARGO").unwrap_or_else(|| "cargo".into()));
    command
        .current_dir(directory)
        .args(["metadata", "--locked", "--format-version", "1"]);
    if no_deps {
        command.arg("--no-deps");
    }
    let output = command
        .output()
        .map_err(|error| format!("run cargo metadata: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("decode Cargo metadata: {error}"))
}

pub fn metadata(directory: &Path) -> Result<Value, String> {
    cargo_metadata(directory, true)
}

fn read(path: &Path) -> Result<String, String> {
    fs::read_to_string(path).map_err(|error| format!("read {}: {error}", path.display()))
}

fn regular_file(path: &Path) -> Result<(), String> {
    let meta = fs::symlink_metadata(path)
        .map_err(|error| format!("inspect {}: {error}", path.display()))?;
    if !meta.file_type().is_file() {
        return Err(format!(
            "expected a regular file, without a symlink: {}",
            path.display()
        ));
    }
    Ok(())
}

#[derive(Debug)]
struct Edit {
    path: PathBuf,
    original: String,
    replacement: String,
}

fn replace(path: &Path, text: &str) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or("edit path must have a parent directory")?;
    let permissions = fs::metadata(path).map_err(|e| e.to_string())?.permissions();
    let mut temporary = tempfile::NamedTempFile::new_in(parent).map_err(|e| e.to_string())?;
    temporary
        .as_file()
        .set_permissions(permissions)
        .map_err(|e| e.to_string())?;
    temporary
        .write_all(text.as_bytes())
        .and_then(|()| temporary.as_file().sync_all())
        .map_err(|error| format!("stage {}: {error}", path.display()))?;
    temporary
        .persist(path)
        .map_err(|error| format!("replace {}: {error}", path.display()))?;
    Ok(())
}

/// Each file replacement is atomic. Multi-file failure rolls back our own edits;
/// process termination is not a filesystem-wide transaction. Concurrent writers
/// must not share the preparation checkout.
fn apply(edits: &[Edit], validate: impl FnOnce() -> Result<(), String>) -> Result<(), String> {
    for edit in edits {
        regular_file(&edit.path)?;
        if read(&edit.path)? != edit.original {
            return Err(format!(
                "file changed since preparation: {}",
                edit.path.display()
            ));
        }
    }
    let mut applied: Vec<&Edit> = Vec::new();
    let result = (|| {
        for edit in edits {
            // Check again immediately before replacement, not global git status.
            if read(&edit.path)? != edit.original {
                return Err(format!(
                    "file changed during preparation: {}",
                    edit.path.display()
                ));
            }
            replace(&edit.path, &edit.replacement)?;
            applied.push(edit);
        }
        validate()
    })();
    match result {
        Ok(()) => Ok(()),
        Err(mut error) => {
            for edit in applied.into_iter().rev() {
                let rollback = (|| {
                    if read(&edit.path)? != edit.replacement {
                        return Err(
                            "changed after our write; preserving concurrent edits".to_string()
                        );
                    }
                    replace(&edit.path, &edit.original)
                })();
                if let Err(reason) = rollback {
                    error.push_str(&format!("; rollback {}: {reason}", edit.path.display()));
                }
            }
            Err(error)
        }
    }
}

pub fn prepare(directory: &Path, target: &str, dry_run: bool) -> Result<(), String> {
    let info = metadata(directory)?;
    let root = info["workspace_root"]
        .as_str()
        .ok_or("Cargo metadata missing workspace_root")?;
    let root = Path::new(root)
        .canonicalize()
        .map_err(|error| error.to_string())?;
    let root_path = root.join("Cargo.toml");
    let lock_path = root.join("Cargo.lock");
    regular_file(&root_path)?;
    regular_file(&lock_path)?;
    let root_text = read(&root_path)?;
    let lock_text = read(&lock_path)?;
    let ids = info["workspace_members"]
        .as_array()
        .ok_or("Cargo metadata missing workspace_members")?;
    let packages = info["packages"]
        .as_array()
        .ok_or("Cargo metadata missing packages")?;
    let mut members = Vec::new();
    for id in ids {
        let package = packages
            .iter()
            .find(|package| package.get("id") == Some(id))
            .ok_or("workspace member is absent from packages")?;
        let path = Path::new(
            package["manifest_path"]
                .as_str()
                .ok_or("package missing manifest_path")?,
        );
        regular_file(path)?;
        let path = path.canonicalize().map_err(|error| error.to_string())?;
        let relative = path
            .strip_prefix(&root)
            .map_err(|_| format!("member manifest is outside workspace: {}", path.display()))?;
        let relative = relative
            .to_str()
            .ok_or("workspace manifest path must be UTF-8")?
            .replace('\\', "/");
        members.push(MemberManifest {
            package_name: package["name"]
                .as_str()
                .ok_or("package missing name")?
                .into(),
            manifest_path: relative,
            text: read(&path)?,
        });
    }
    let plan = version::plan_version_update(&root_text, &members, &lock_text, target)?;
    let mut by_path = BTreeMap::new();
    by_path.insert(
        root_path.clone(),
        Edit {
            path: root_path,
            original: root_text,
            replacement: plan.workspace_manifest,
        },
    );
    for (before, after) in members.iter().zip(plan.members) {
        let path = root.join(&before.manifest_path);
        // Cargo can include the root package; it is one document, written once.
        by_path.insert(
            path.clone(),
            Edit {
                path,
                original: before.text.clone(),
                replacement: after.text,
            },
        );
    }
    by_path.insert(
        lock_path.clone(),
        Edit {
            path: lock_path,
            original: lock_text,
            replacement: plan.lockfile,
        },
    );
    let edits: Vec<_> = by_path
        .into_values()
        .filter(|edit| edit.original != edit.replacement)
        .collect();
    println!(
        "Workspace version {} -> {}",
        plan.previous_version, plan.target_version
    );
    for edit in &edits {
        println!(
            "{} {}",
            if dry_run { "would update" } else { "update" },
            edit.path
                .strip_prefix(&root)
                .unwrap_or(&edit.path)
                .display()
        );
    }
    if !dry_run {
        apply(&edits, || {
            // Resolve dependencies with --locked, rather than assuming that
            // no-deps metadata checked all lockfile references.
            let actual = cargo_metadata(&root, false)?;
            staging::validate_workspace_versions(&actual, target)
        })?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "workspace_tests.rs"]
mod tests;
