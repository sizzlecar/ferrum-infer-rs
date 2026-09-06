//! Build-time asset metadata. Hashes identify bytes; these manifests do not
//! certify runtime correctness, installation, or authorization to publish.

use semver::Version;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

#[path = "staging_audit.rs"]
mod audit;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CandidateInput {
    pub version: String,
    pub release_candidate_sha: String,
    pub release_candidate_tag: String,
    pub staging_label: String,
    pub workflow_run_id: String,
    pub workflow_run_attempt: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Backend {
    Cpu,
    Metal,
    Cuda,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AbiInput {
    pub backend: Backend,
    pub target_triple: String,
    pub cargo_features: Vec<String>,
    pub cuda_compute_capability: Option<String>,
    pub cuda_toolkit_image: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StagedManifests {
    pub asset_checksum: String,
    pub binary_checksum: String,
    pub version: Value,
    pub dependency: Value,
    pub abi: Value,
}

fn safe_label(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"._-".contains(&byte))
}

fn safe_file_name(value: &str) -> bool {
    safe_label(value) && value != "." && value != ".."
}

fn release_version(value: &str) -> Result<Version, String> {
    let version =
        Version::parse(value).map_err(|error| format!("invalid release version: {error}"))?;
    if !version.pre.is_empty() || !version.build.is_empty() {
        return Err("target release version must not contain prerelease or build metadata".into());
    }
    Ok(version)
}

/// Validate identity syntax and the RC/version relationship, not code quality.
pub fn validate_candidate(input: &CandidateInput) -> Result<(), String> {
    let version = release_version(&input.version)?;
    let prefix = format!("v{version}-rc.");
    let number = input
        .release_candidate_tag
        .strip_prefix(&prefix)
        .ok_or_else(|| format!("RC tag must match {prefix}N for the target version"))?;
    if number.is_empty()
        || number.starts_with('0')
        || !number.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err("RC number must be a positive decimal integer without leading zeroes".into());
    }
    if !matches!(input.release_candidate_sha.len(), 40 | 64)
        || !input
            .release_candidate_sha
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(
            "release candidate commit must be a full lowercase hexadecimal object ID".into(),
        );
    }
    for (name, value) in [
        ("staging_label", &input.staging_label),
        ("workflow_run_id", &input.workflow_run_id),
        ("workflow_run_attempt", &input.workflow_run_attempt),
    ] {
        if !safe_label(value) {
            return Err(format!(
                "{name} must contain only nonempty ASCII letters, digits, '.', '_' or '-'"
            ));
        }
    }
    Ok(())
}

/// Check Cargo's actual workspace membership, including members other than the
/// CLI. Non-workspace dependency versions are deliberately irrelevant.
pub fn validate_workspace_versions(metadata: &Value, expected: &str) -> Result<(), String> {
    let expected = release_version(expected)?;
    let members = metadata["workspace_members"]
        .as_array()
        .filter(|members| !members.is_empty())
        .ok_or("Cargo metadata must contain nonempty workspace_members")?;
    let packages = metadata["packages"]
        .as_array()
        .ok_or("Cargo metadata must contain packages")?;
    let mut seen = BTreeSet::new();
    for member in members {
        let member = member
            .as_str()
            .ok_or("workspace member ID must be a string")?;
        if member.is_empty() || !seen.insert(member) {
            return Err("workspace member IDs must be nonempty and unique".into());
        }
        let mut matches = packages
            .iter()
            .filter(|package| package["id"].as_str() == Some(member));
        let package = matches
            .next()
            .ok_or_else(|| format!("workspace member {member:?} is absent from packages"))?;
        if matches.next().is_some() {
            return Err(format!("duplicate Cargo package ID {member:?}"));
        }
        let value = package["version"]
            .as_str()
            .ok_or("package version must be a string")?;
        let actual = Version::parse(value)
            .map_err(|error| format!("invalid workspace package version: {error}"))?;
        if actual != expected {
            return Err(format!(
                "workspace member {member:?} has version {actual}; expected {expected}"
            ));
        }
    }
    Ok(())
}

fn validate_abi(abi: &AbiInput) -> Result<audit::Format, String> {
    let format = audit::Format::for_target(abi.backend, &abi.target_triple)?;
    let mut features = BTreeSet::new();
    for feature in &abi.cargo_features {
        if !safe_label(feature) || !features.insert(feature.as_str()) {
            return Err("Cargo feature names must be nonempty, safe and unique".into());
        }
    }
    match abi.backend {
        Backend::Cpu if features.contains("cuda") || features.contains("metal") => {
            return Err("CPU asset declares an accelerator Cargo feature".into());
        }
        Backend::Metal => {
            if !features.contains("metal") || features.contains("cuda") {
                return Err("Metal asset requires the metal feature and an Apple Darwin target, without cuda".into());
            }
        }
        Backend::Cuda => {
            if !features.contains("cuda") || features.contains("metal") {
                return Err("CUDA asset requires the cuda feature without metal".into());
            }
            let capability = abi
                .cuda_compute_capability
                .as_deref()
                .ok_or("CUDA compute capability is required")?;
            if capability.is_empty()
                || !capability.bytes().all(|byte| byte.is_ascii_alphanumeric())
                || !capability.as_bytes()[0].is_ascii_digit()
            {
                return Err("invalid CUDA compute capability".into());
            }
            let image = abi
                .cuda_toolkit_image
                .as_deref()
                .ok_or("CUDA toolkit image is required")?;
            if image.is_empty()
                || !image
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || b"._-/:@".contains(&byte))
            {
                return Err("invalid CUDA toolkit image identifier".into());
            }
        }
        _ => {}
    }
    if abi.backend != Backend::Cuda
        && (abi.cuda_compute_capability.is_some() || abi.cuda_toolkit_image.is_some())
    {
        return Err("CUDA ABI fields are only valid for a CUDA asset".into());
    }
    Ok(format)
}

fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

/// Hash supplied bytes and generate the existing schema-v1 adjacent manifests.
/// Filesystem creation, fresh-path enforcement, and obtaining authentic command
/// output are the caller's responsibilities. This function never runs a command.
#[allow(clippy::too_many_arguments)]
pub fn generate_manifests(
    input: &CandidateInput,
    abi_input: &AbiInput,
    asset_name: &str,
    asset_bytes: &[u8],
    binary_bytes: &[u8],
    audit_name: &str,
    audit_text: &str,
) -> Result<StagedManifests, String> {
    validate_candidate(input)?;
    let audit_format = validate_abi(abi_input)?;
    if !safe_file_name(asset_name) || !safe_file_name(audit_name) {
        return Err("asset and audit names must be safe file basenames".into());
    }
    if asset_bytes.is_empty() || binary_bytes.is_empty() {
        return Err("asset and binary bytes must be nonempty".into());
    }
    let audit = audit::inspect(audit_text, abi_input.backend, audit_format)?;
    let asset_sha = digest(asset_bytes);
    let binary_sha = digest(binary_bytes);
    let audit_sha = digest(audit_text.as_bytes());
    let common = json!({
        "schema_version": 1,
        "asset_name": asset_name,
        "asset_sha256": asset_sha,
        "binary_name": "ferrum",
        "binary_sha256": binary_sha,
        "release_candidate_sha": input.release_candidate_sha,
        "release_candidate_tag": input.release_candidate_tag,
        "staging_label": input.staging_label,
        "workflow_run_id": input.workflow_run_id,
        "workflow_run_attempt": input.workflow_run_attempt,
    });
    let mut version = common.clone();
    version["version"] = json!(input.version);
    let mut dependency = common.clone();
    dependency["audit_file"] = json!(audit_name);
    dependency["audit_sha256"] = json!(audit_sha);
    dependency["forbidden_runtime_linkage"] = json!(["python", "torch", "vllm"]);
    dependency["forbidden_runtime_linkage_found"] = json!(false);
    dependency["runtime_libraries"] = json!(audit.runtime_libraries);
    dependency["unresolved_runtime_libraries"] = json!(audit.unresolved_runtime_libraries);
    dependency["deferred_runtime_dependencies"] = json!(audit.deferred_runtime_dependencies);
    dependency["audit_scope"] = json!("build_time_dynamic_linkage");
    let mut abi = common;
    abi["target_triple"] = json!(abi_input.target_triple);
    abi["backend"] = json!(abi_input.backend);
    abi["cargo_features"] = json!(abi_input.cargo_features);
    abi["dependency_audit_sha256"] = json!(audit_sha);
    if let Some(capability) = &abi_input.cuda_compute_capability {
        abi["cuda_compute_capability"] = json!(capability);
    }
    if let Some(image) = &abi_input.cuda_toolkit_image {
        abi["cuda_toolkit_image"] = json!(image);
    }
    Ok(StagedManifests {
        asset_checksum: format!("{asset_sha}  {asset_name}\n"),
        binary_checksum: format!("{binary_sha}  ferrum\n"),
        version,
        dependency,
        abi,
    })
}

#[cfg(test)]
#[path = "staging_tests.rs"]
mod tests;
