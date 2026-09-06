//! Cargo owns dependency ordering and index barriers. This module owns exact
//! version/checksum reconciliation, including Cargo's final warning-only timeout.
use super::{http_client, sha256, write_json, AcceptedRelease, PublishArgs};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
    process::Stdio,
    time::{Duration, Instant},
};
use tokio::process::Command;

#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
struct CrateArtifact {
    name: String,
    version: String,
    sha256: String,
}
#[derive(Deserialize, Serialize)]
struct Prepared {
    version: String,
    candidate_sha: String,
    source_sha256: String,
    cargo_toolchain: String,
    verification: String,
    crates: Vec<CrateArtifact>,
}
#[derive(Debug, Serialize, PartialEq, Eq)]
enum RegistryStatus {
    Missing,
    Identical,
}

pub(super) async fn publish_crates(
    args: &PublishArgs,
    accepted: &AcceptedRelease,
) -> Result<(), String> {
    let head = git_output(&accepted.workspace, &["rev-parse", "HEAD"]).await?;
    if String::from_utf8_lossy(&head).trim() != accepted.candidate_sha {
        return Err("publication workspace is not the accepted candidate commit".into());
    }
    let metadata = cargo_output(
        args,
        accepted,
        &["metadata", "--no-deps", "--format-version", "1", "--locked"],
        false,
    )
    .await?;
    let metadata: Value =
        serde_json::from_slice(&metadata).map_err(|_| "invalid cargo metadata")?;
    let names = publishable_packages(&metadata, &accepted.version)?;
    let inputs = package_inputs(args, accepted, &metadata, &names).await?;
    let source_hash = source_fingerprint(accepted, &inputs).await?;
    let package_dir = PathBuf::from(
        metadata["target_directory"]
            .as_str()
            .ok_or("Cargo metadata lacks target directory")?,
    )
    .join("package");
    let path = args.state_dir.join("crates.json");
    let prepared = if path.exists() {
        let saved: Prepared = serde_json::from_slice(&fs::read(&path).map_err(|e| e.to_string())?)
            .map_err(|_| "invalid crate checkpoint")?;
        if saved.version != accepted.version
            || saved.candidate_sha != accepted.candidate_sha
            || saved.source_sha256 != source_hash
            || saved.cargo_toolchain != args.cargo_toolchain
            || saved.verification != "cargo-publish-dry-run-package-build"
        {
            return Err(
                "crate checkpoint belongs to different candidate inputs or verification settings"
                    .into(),
            );
        }
        if saved
            .crates
            .iter()
            .map(|c| c.name.clone())
            .collect::<BTreeSet<_>>()
            != names
            || saved.crates.len() != names.len()
            || saved
                .crates
                .iter()
                .any(|c| c.version != accepted.version || !valid_checksum(&c.sha256))
        {
            return Err(
                "crate checkpoint does not match the publishable workspace inventory".into(),
            );
        }
        // Archived bytes are retained so a checkpoint never merely asserts passed.
        for artifact in &saved.crates {
            verify_saved_archive(args, artifact)?;
        }
        saved
    } else {
        // Only the first attempt performs package verification of the full set.
        // --allow-dirty permits the caller's controlled version preparation;
        // actual inputs are fingerprinted, not accepted based on Git cleanliness.
        cargo_output(
            args,
            accepted,
            &[
                "publish",
                "--workspace",
                "--registry",
                "crates-io",
                "--locked",
                "--allow-dirty",
                "--dry-run",
            ],
            false,
        )
        .await?;
        require_same_source(accepted, &inputs, &source_hash).await?;
        let archives = args.state_dir.join("crates");
        fs::create_dir_all(&archives).map_err(|e| e.to_string())?;
        let mut crates = Vec::new();
        for name in names {
            let filename = format!("{name}-{}.crate", accepted.version);
            let bytes = fs::read(package_dir.join(&filename))
                .map_err(|e| format!("read Cargo package {name}: {e}"))?;
            let artifact = CrateArtifact {
                name,
                version: accepted.version.clone(),
                sha256: sha256(&bytes),
            };
            let destination = archives.join(filename);
            if destination.exists() && fs::read(&destination).map_err(|e| e.to_string())? != bytes {
                return Err(
                    "existing staged crate archive conflicts with newly verified bytes".into(),
                );
            }
            fs::write(destination, bytes).map_err(|e| e.to_string())?;
            crates.push(artifact);
        }
        let prepared = Prepared {
            version: accepted.version.clone(),
            candidate_sha: accepted.candidate_sha.clone(),
            source_sha256: source_hash.clone(),
            cargo_toolchain: args.cargo_toolchain.clone(),
            verification: "cargo-publish-dry-run-package-build".into(),
            crates,
        };
        write_json(&path, &prepared)?;
        prepared
    };
    let client = http_client()?;
    let missing = reconcile(&client, "https://index.crates.io", &prepared.crates).await?;
    write_json(
        &args.state_dir.join("crates-observed.json"),
        &json!({"version": accepted.version, "missing": missing, "complete": missing.is_empty()}),
    )?;
    if !missing.is_empty() {
        // Re-package only missing crates without repeating their previous build.
        // This also detects changed lock resolution before the uploading command.
        let mut command = vec![
            "publish".to_string(),
            "--registry".into(),
            "crates-io".into(),
            "--locked".into(),
            "--allow-dirty".into(),
            "--no-verify".into(),
        ];
        for name in &missing {
            command.extend(["-p".into(), name.clone()]);
        }
        let mut dry = command.clone();
        dry.push("--dry-run".into());
        cargo_output(
            args,
            accepted,
            &dry.iter().map(String::as_str).collect::<Vec<_>>(),
            false,
        )
        .await?;
        verify_current_archives(&package_dir, &prepared.crates, &missing)?;
        require_same_source(accepted, &inputs, &source_hash).await?;
        for asset in &accepted.assets {
            super::verified_asset_bytes(asset)?;
        }
        // Cargo still re-packages here: this is not an upload-existing-archive API.
        let result = cargo_output(
            args,
            accepted,
            &command.iter().map(String::as_str).collect::<Vec<_>>(),
            true,
        )
        .await;
        verify_current_archives(&package_dir, &prepared.crates, &missing)?;
        require_same_source(accepted, &inputs, &source_hash).await?;
        if result.is_err() {
            let observed = reconcile(&client, "https://index.crates.io", &prepared.crates).await;
            match observed {
                Ok(missing) => write_json(
                    &args.state_dir.join("crates-observed.json"),
                    &json!({"version": accepted.version, "missing": missing, "complete": missing.is_empty(), "cargo_process_succeeded": false}),
                )?,
                Err(_) => write_json(
                    &args.state_dir.join("crates-observed.json"),
                    &json!({"version": accepted.version, "status": "unknown_after_cargo_failure"}),
                )?,
            }
            return Err("Cargo publication did not complete successfully; remote progress is retained, resume after resolving the failure".into());
        }
    }
    let deadline = Instant::now()
        .checked_add(Duration::from_secs(args.index_timeout_secs))
        .ok_or("registry timeout is outside the supported range")?;
    loop {
        let missing = reconcile(&client, "https://index.crates.io", &prepared.crates).await?;
        write_json(
            &args.state_dir.join("crates-observed.json"),
            &json!({"version": accepted.version, "missing": missing, "complete": missing.is_empty()}),
        )?;
        if missing.is_empty() {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(
                "registry visibility deadline expired; publication remains incomplete".into(),
            );
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

fn valid_checksum(checksum: &str) -> bool {
    checksum.len() == 64
        && checksum
            .bytes()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
}

fn publishable_packages(metadata: &Value, version: &str) -> Result<BTreeSet<String>, String> {
    let packages = metadata["packages"]
        .as_array()
        .ok_or("Cargo metadata lacks packages")?;
    let members = metadata["workspace_members"]
        .as_array()
        .ok_or("Cargo metadata lacks workspace members")?;
    let mut names = BTreeSet::new();
    for member in members {
        let package = packages
            .iter()
            .find(|p| p.get("id") == Some(member))
            .ok_or("missing workspace package metadata")?;
        let publish = package
            .get("publish")
            .ok_or("Cargo metadata lacks publish policy")?;
        if let Some(registries) = publish.as_array() {
            if !registries.iter().any(|r| r.as_str() == Some("crates-io")) {
                continue;
            }
        } else if !publish.is_null() {
            return Err("invalid publish registry metadata".into());
        }
        let name = package["name"].as_str().ok_or("package missing name")?;
        if !super::safe_component(name) || package["version"].as_str() != Some(version) {
            return Err(format!(
                "publishable package {name} does not match the accepted version"
            ));
        }
        if !names.insert(name.into()) {
            return Err("duplicate publishable package name".into());
        }
    }
    if names.is_empty() {
        return Err("candidate has no publishable workspace members".into());
    }
    Ok(names)
}

fn verify_saved_archive(args: &PublishArgs, artifact: &CrateArtifact) -> Result<(), String> {
    let path = args
        .state_dir
        .join("crates")
        .join(format!("{}-{}.crate", artifact.name, artifact.version));
    if sha256(&fs::read(path).map_err(|e| format!("read verified crate archive: {e}"))?)
        != artifact.sha256
    {
        return Err(format!("verified crate archive changed: {}", artifact.name));
    }
    Ok(())
}

fn verify_current_archives(
    directory: &Path,
    artifacts: &[CrateArtifact],
    selected: &[String],
) -> Result<(), String> {
    for artifact in artifacts.iter().filter(|a| selected.contains(&a.name)) {
        let bytes =
            fs::read(directory.join(format!("{}-{}.crate", artifact.name, artifact.version)))
                .map_err(|e| e.to_string())?;
        if sha256(&bytes) != artifact.sha256 {
            return Err(format!("Cargo re-packaged different bytes for {}; reconcile candidate before publishing further", artifact.name));
        }
    }
    Ok(())
}

fn index_path(name: &str) -> Result<String, String> {
    if !super::safe_component(name) {
        return Err("invalid crate name".into());
    }
    let n = name.to_ascii_lowercase();
    Ok(match n.len() {
        1 => format!("1/{n}"),
        2 => format!("2/{n}"),
        3 => format!("3/{}/{n}", &n[..1]),
        _ => format!("{}/{}/{n}", &n[..2], &n[2..4]),
    })
}

async fn registry_status(
    client: &reqwest::Client,
    base: &str,
    artifact: &CrateArtifact,
) -> Result<RegistryStatus, String> {
    let response = client
        .get(format!(
            "{}/{}",
            base.trim_end_matches('/'),
            index_path(&artifact.name)?
        ))
        .header(reqwest::header::CACHE_CONTROL, "no-cache")
        .send()
        .await
        .map_err(|_| {
            format!(
                "registry query failed for {} (status unknown)",
                artifact.name
            )
        })?;
    if response.status() == reqwest::StatusCode::NOT_FOUND {
        return Ok(RegistryStatus::Missing);
    }
    if response.status() != reqwest::StatusCode::OK {
        return Err(format!(
            "registry query for {} returned HTTP {}; not treating it as missing",
            artifact.name,
            response.status()
        ));
    }
    let text = response
        .text()
        .await
        .map_err(|_| "registry response was not readable")?;
    let mut found = None;
    let mut any = false;
    for line in text.lines().filter(|line| !line.is_empty()) {
        any = true;
        let row: Value =
            serde_json::from_str(line).map_err(|_| "malformed registry index response")?;
        if row["name"].as_str() != Some(artifact.name.as_str())
            || row["vers"].as_str().is_none()
            || row["cksum"].as_str().is_none()
            || row["yanked"].as_bool().is_none()
        {
            return Err("registry index row is missing required identity/checksum fields".into());
        }
        if row["vers"].as_str() == Some(artifact.version.as_str()) {
            if found.is_some() {
                return Err("registry contains duplicate entries for one version".into());
            }
            if row["yanked"].as_bool() != Some(false) {
                return Err(format!("{}@{} is yanked", artifact.name, artifact.version));
            }
            if row["cksum"].as_str() != Some(artifact.sha256.as_str()) {
                return Err(format!(
                    "registry checksum conflict for {}@{}",
                    artifact.name, artifact.version
                ));
            }
            found = Some(RegistryStatus::Identical);
        }
    }
    if !any {
        return Err("registry returned an empty successful index response".into());
    }
    Ok(found.unwrap_or(RegistryStatus::Missing))
}

async fn reconcile(
    client: &reqwest::Client,
    base: &str,
    artifacts: &[CrateArtifact],
) -> Result<Vec<String>, String> {
    let mut missing = Vec::new();
    for artifact in artifacts {
        if registry_status(client, base, artifact).await? == RegistryStatus::Missing {
            missing.push(artifact.name.clone());
        }
    }
    Ok(missing)
}

async fn cargo_output(
    args: &PublishArgs,
    accepted: &AcceptedRelease,
    arguments: &[&str],
    upload: bool,
) -> Result<Vec<u8>, String> {
    let mut command = Command::new("cargo");
    command
        .arg(format!("+{}", args.cargo_toolchain))
        .args(arguments)
        .current_dir(&accepted.workspace)
        .kill_on_drop(true)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .env_remove("GITHUB_TOKEN")
        .env_remove(&args.tap_token_env);
    command.env_remove("CARGO_REGISTRIES_CRATES_IO_TOKEN");
    if upload {
        let token = std::env::var("CARGO_REGISTRY_TOKEN")
            .map_err(|_| "CARGO_REGISTRY_TOKEN is required for publication")?;
        if token.is_empty() {
            return Err("CARGO_REGISTRY_TOKEN is empty".into());
        }
        command
            .env("CARGO_REGISTRY_TOKEN", &token)
            .env("CARGO_REGISTRIES_CRATES_IO_TOKEN", &token);
    } else {
        command.env_remove("CARGO_REGISTRY_TOKEN");
    }
    let output = tokio::time::timeout(Duration::from_secs(7200), command.output())
        .await
        .map_err(|_| "Cargo operation timed out")?
        .map_err(|_| "could not start Cargo")?;
    // Do not echo subprocess stderr: credential providers/remote errors can contain secrets.
    if !output.status.success() {
        return Err(format!(
            "Cargo {} failed with {}; raw credential-bearing process output is suppressed",
            arguments[0], output.status
        ));
    }
    Ok(output.stdout)
}

async fn git_output(workspace: &Path, arguments: &[&str]) -> Result<Vec<u8>, String> {
    let output = Command::new("git")
        .args(arguments)
        .current_dir(workspace)
        .kill_on_drop(true)
        .stdin(Stdio::null())
        .output()
        .await
        .map_err(|_| "cannot inspect candidate Git inputs")?;
    if !output.status.success() {
        return Err("cannot inspect candidate Git inputs".into());
    }
    Ok(output.stdout)
}

async fn package_inputs(
    args: &PublishArgs,
    accepted: &AcceptedRelease,
    metadata: &Value,
    names: &BTreeSet<String>,
) -> Result<BTreeMap<String, PathBuf>, String> {
    let workspace = accepted
        .workspace
        .canonicalize()
        .map_err(|e| e.to_string())?;
    let mut files = BTreeMap::new();
    let mut add = |path: PathBuf| -> Result<(), String> {
        let path = path
            .canonicalize()
            .map_err(|_| "Cargo package input is not a readable regular file")?;
        let relative = path
            .strip_prefix(&workspace)
            .map_err(|_| "Cargo package input is outside the accepted workspace")?
            .to_str()
            .ok_or("package input path is not UTF-8")?
            .replace('\\', "/");
        files.insert(relative, path);
        Ok(())
    };
    add(workspace.join("Cargo.toml"))?;
    add(workspace.join("Cargo.lock"))?;
    for config in [".cargo/config", ".cargo/config.toml"] {
        if workspace.join(config).exists() {
            add(workspace.join(config))?;
        }
    }
    for package in metadata["packages"]
        .as_array()
        .ok_or("missing Cargo packages")?
    {
        let Some(name) = package["name"].as_str().filter(|n| names.contains(*n)) else {
            continue;
        };
        let manifest = Path::new(
            package["manifest_path"]
                .as_str()
                .ok_or("package lacks manifest path")?,
        );
        let directory = manifest.parent().ok_or("manifest has no directory")?;
        let output = cargo_output(
            args,
            accepted,
            &["package", "--list", "--locked", "--allow-dirty", "-p", name],
            false,
        )
        .await?;
        let list = std::str::from_utf8(&output).map_err(|_| "Cargo package list is not UTF-8")?;
        if list.trim().is_empty() {
            return Err("Cargo returned an empty package input list".into());
        }
        for entry in list.lines() {
            // Cargo generates these files; their original inputs are separately checked.
            if matches!(entry, ".cargo_vcs_info.json" | "Cargo.toml.orig") {
                continue;
            }
            if Path::new(entry)
                .components()
                .any(|c| !matches!(c, std::path::Component::Normal(_)))
            {
                return Err("invalid Cargo package path".into());
            }
            let path = directory.join(entry);
            if path.exists() {
                add(path)?;
            } else if entry == "Cargo.lock" {
                add(workspace.join("Cargo.lock"))?;
            } else {
                let source = ["readme", "license_file"]
                    .iter()
                    .filter_map(|field| package[*field].as_str())
                    .map(|path| directory.join(path))
                    .find(|path| path.file_name() == Path::new(entry).file_name())
                    .ok_or_else(|| {
                        format!(
                            "cannot map generated package member {name}/{entry} to accepted source"
                        )
                    })?;
                add(source)?;
            }
        }
    }
    Ok(files)
}

async fn source_fingerprint(
    accepted: &AcceptedRelease,
    inputs: &BTreeMap<String, PathBuf>,
) -> Result<String, String> {
    let mut hash = Sha256::new();
    for (relative, path) in inputs {
        let bytes =
            fs::read(path).map_err(|_| format!("cannot read Cargo package input {relative}"))?;
        let committed = git_output(
            &accepted.workspace,
            &["show", &format!("{}:{relative}", accepted.candidate_sha)],
        )
        .await?;
        if committed != bytes {
            return Err(format!(
                "Cargo package input differs from the accepted candidate: {relative}"
            ));
        }
        hash.update((relative.len() as u64).to_le_bytes());
        hash.update(relative.as_bytes());
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    Ok(format!("{:x}", hash.finalize()))
}
async fn require_same_source(
    accepted: &AcceptedRelease,
    inputs: &BTreeMap<String, PathBuf>,
    hash: &str,
) -> Result<(), String> {
    if source_fingerprint(accepted, inputs).await? != hash {
        return Err("candidate package inputs changed during publication".into());
    }
    Ok(())
}

#[cfg(test)]
#[path = "registry_tests.rs"]
mod tests;
