//! Windows build/packaging evidence adds assets to the existing publisher.
//! Driverless staging remains byte-only; model and upgrade QA are separate.
use super::super::{installation, portable, AcceptedAsset};
use serde::Deserialize;
use std::{collections::BTreeSet, fs, path::Path};

const ARCHIVE: &str = "ferrum-windows-x86_64-cuda-sm89.zip";
const LAUNCHER: &str = "ferrum-windows-launcher-v1.exe";

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct FileIdentity {
    name: String,
    sha256: String,
    size_bytes: u64,
}

#[derive(Deserialize)]
struct Staging {
    schema_version: u32,
    version: String,
    candidate_sha: String,
    candidate_tag: String,
    staging_label: String,
    workflow_run_id: u64,
    workflow_run_attempt: u64,
    scope: String,
    binary_sha256: String,
    launcher_sha256: String,
    manifest_sha256: String,
    archive: FileIdentity,
    setup: FileIdentity,
    launcher: FileIdentity,
}

#[derive(Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case", deny_unknown_fields)]
enum LauncherOrigin {
    FirstIssuance {
        source_commit: String,
        sha256: String,
        dependencies: Vec<String>,
    },
    Released {
        url: String,
        sha256: String,
        dependencies: Vec<String>,
    },
}

#[derive(Debug)]
pub(in super::super) struct VerifiedWindows {
    pub assets: Vec<AcceptedAsset>,
    pub attempt: u64,
}

pub(in super::super) fn verify(
    directory: &Path,
    version: &str,
    candidate: &str,
    run: u64,
    baseline_tag: &str,
    repo: &str,
) -> Result<VerifiedWindows, String> {
    let stage: Staging = super::read(&directory.join("windows-staging.json"))?;
    let setup = format!("ferrum-{version}-windows-x86_64-cuda-sm89-setup.exe");
    let candidate_prefix = format!("v{version}-rc.");
    if stage.schema_version != 1
        || stage.version != version
        || stage.candidate_sha != candidate
        || stage.workflow_run_id != run
        || run == 0
        || stage.workflow_run_attempt == 0
        || stage.staging_label != format!("run-{run}-{}", stage.workflow_run_attempt)
        || stage
            .candidate_tag
            .strip_prefix(&candidate_prefix)
            .is_none_or(|n| {
                n.is_empty() || n.starts_with('0') || !n.bytes().all(|b| b.is_ascii_digit())
            })
        || stage.archive.name != ARCHIVE
        || stage.setup.name != setup
        || stage.launcher.name != LAUNCHER
    {
        return Err("Windows staging receipt differs from this formal candidate/run".into());
    }
    let receipt_path = directory.join(format!("{ARCHIVE}.receipt.json"));
    let receipt_hash = installation::sha256(&receipt_path)?;
    let portable = portable::verify_staged(
        &directory.join(ARCHIVE),
        &receipt_path,
        &directory.join("portable.staging-inspection.json"),
        version,
        candidate,
    )?;
    if stage.archive.sha256 != portable.archive_sha256
        || stage.archive.size_bytes != portable.archive_size_bytes
        || stage.binary_sha256 != portable.binary_sha256
        || stage.manifest_sha256 != portable.manifest_sha256
        || stage.launcher.sha256 != stage.launcher_sha256
        || stage.scope
            != if portable.startup_executed {
                "staged_bytes_and_startup"
            } else {
                "staged_bytes_only"
            }
    {
        return Err("Windows setup inputs do not match the verified portable payload".into());
    }
    let origin: LauncherOrigin = super::read(&directory.join("launcher-origin.json"))?;
    let (sha, dependencies) = match origin {
        LauncherOrigin::FirstIssuance {
            source_commit,
            sha256,
            dependencies,
        } => {
            if source_commit != candidate {
                return Err("first launcher issuance has another source".into());
            }
            (sha256, dependencies)
        }
        LauncherOrigin::Released {
            url,
            sha256,
            dependencies,
        } => {
            if url
                != format!("https://github.com/{repo}/releases/download/{baseline_tag}/{LAUNCHER}")
            {
                return Err("stable launcher must reuse the formal baseline asset".into());
            }
            (sha256, dependencies)
        }
    };
    if sha != stage.launcher.sha256
        || dependencies.is_empty()
        || dependencies.iter().any(|name| {
            let name = name.to_ascii_lowercase();
            !name.ends_with(".dll")
                || !super::file_name(&name)
                || ["vcruntime", "msvcp", "msvcr", "concrt", "vcomp"]
                    .iter()
                    .any(|prefix| {
                        name.strip_prefix(prefix)
                            .is_some_and(|tail| tail.starts_with(|c: char| c.is_ascii_digit()))
                    })
        })
    {
        return Err("stable launcher byte/runtime identity is invalid".into());
    }
    portable::verify_launcher(&directory.join(LAUNCHER))?;
    let mut assets = Vec::new();
    for identity in [&stage.archive, &stage.setup, &stage.launcher] {
        let path = directory.join(&identity.name);
        let metadata = fs::symlink_metadata(&path).map_err(|e| e.to_string())?;
        if !metadata.is_file()
            || metadata.len() != identity.size_bytes
            || identity.size_bytes == 0
            || !super::hex(&identity.sha256, 64)
            || installation::sha256(&path)? != identity.sha256
        {
            return Err(format!(
                "Windows staged asset identity mismatch: {}",
                identity.name
            ));
        }
        let checksum_name = format!("{}.sha256", identity.name);
        let checksum = directory.join(&checksum_name);
        let text = fs::read_to_string(&checksum).map_err(|e| e.to_string())?;
        let lines: Vec<_> = text.lines().collect();
        if lines.len() != 1 || lines[0] != format!("{}  {}", identity.sha256, identity.name) {
            return Err(format!("Windows checksum does not bind {}", identity.name));
        }
        assets.push(AcceptedAsset {
            path,
            name: identity.name.clone(),
            sha256: identity.sha256.clone(),
        });
        assets.push(AcceptedAsset {
            path: checksum.clone(),
            name: checksum_name,
            sha256: installation::sha256(&checksum)?,
        });
    }
    if installation::sha256(&receipt_path)? != receipt_hash {
        return Err("portable receipt changed during verification".into());
    }
    assets.push(AcceptedAsset {
        path: receipt_path,
        name: format!("{ARCHIVE}.receipt.json"),
        sha256: receipt_hash,
    });
    let mut names = BTreeSet::new();
    if assets.iter().any(|asset| !names.insert(&asset.name)) {
        return Err("duplicate Windows public asset".into());
    }
    println!("Verified Windows staged bytes; CUDA startup {}. Model/upgrade QA is not asserted by this asset gate.", if portable.startup_executed { "passed" } else { "deferred (no driver on staging host)" });
    Ok(VerifiedWindows {
        assets,
        attempt: stage.workflow_run_attempt,
    })
}
