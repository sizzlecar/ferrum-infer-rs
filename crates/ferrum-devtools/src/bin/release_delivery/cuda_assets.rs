//! Private release-package identities; these do not change inference schemas.
use ferrum_bench_core::release_candidate::staging::{AbiInput, Backend};
use std::{collections::BTreeSet, path::Path, process::Stdio};
use tokio::process::Command;

const WORKFLOW_PATH: &str = ".github/workflows/release-cuda.yml";

/// Read the immutable candidate blob, never the tool's checkout or available assets.
pub(super) async fn candidate_requires_sm120(
    workspace: &Path,
    candidate: &str,
) -> Result<bool, String> {
    if !matches!(candidate.len(), 40 | 64)
        || !candidate
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err("CUDA delivery capability requires a complete candidate commit".into());
    }
    async fn git(workspace: &Path, args: &[&str]) -> Result<Vec<u8>, String> {
        let output = Command::new("git")
            .arg("-C")
            .arg(workspace)
            .args(args)
            .env_remove("GIT_DIR")
            .env_remove("GIT_WORK_TREE")
            .env_remove("GIT_INDEX_FILE")
            .stdin(Stdio::null())
            .kill_on_drop(true)
            .output()
            .await
            .map_err(|_| "cannot read candidate CUDA delivery declaration")?;
        if !output.status.success() {
            return Err("cannot read candidate CUDA delivery declaration".into());
        }
        Ok(output.stdout)
    }
    let tree = git(
        workspace,
        &["ls-tree", "-z", candidate, "--", WORKFLOW_PATH],
    )
    .await?;
    if tree.is_empty() {
        return Ok(false);
    }
    let spec = format!("{candidate}:{WORKFLOW_PATH}");
    workflow_requires_sm120(&git(workspace, &["cat-file", "blob", &spec]).await?)
}

pub(super) fn workflow_requires_sm120(bytes: &[u8]) -> Result<bool, String> {
    use serde_yaml::{Mapping, Value};
    fn mapping<'a>(value: &'a Value, field: &str) -> Result<&'a Mapping, String> {
        let mapping = value
            .as_mapping()
            .ok_or_else(|| format!("{field} must be a YAML mapping"))?;
        if mapping.keys().any(|key| !key.is_string()) {
            return Err(format!("{field} must have string keys"));
        }
        Ok(mapping)
    }
    fn get<'a>(map: &'a Mapping, key: &str) -> Option<&'a Value> {
        map.get(Value::String(key.into()))
    }
    let workflow: Value = serde_yaml::from_slice(bytes)
        .map_err(|error| format!("invalid candidate CUDA workflow YAML: {error}"))?;
    let root = mapping(&workflow, "workflow")?;
    let on = mapping(get(root, "on").ok_or("CUDA workflow omits on")?, "on")?;
    let Some(call) = get(on, "workflow_call") else {
        return Ok(false);
    };
    let Some(outputs) = get(mapping(call, "workflow_call")?, "outputs") else {
        return Ok(false);
    };
    let Some(declaration) = get(mapping(outputs, "outputs")?, "cuda_sm120_asset_id") else {
        return Ok(false);
    };
    let value = get(mapping(declaration, "cuda_sm120_asset_id")?, "value")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty());
    if value.is_none() {
        return Err("SM120 output requires a nonempty scalar value".into());
    }
    Ok(true)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum CudaVariant {
    Sm89,
    Sm120,
}

impl CudaVariant {
    pub(super) fn from_abi(name: &str, abi: &AbiInput) -> Result<Option<Self>, String> {
        if abi.backend != Backend::Cuda {
            if name.contains("-cuda-") {
                return Err("CUDA asset name disagrees with its backend".into());
            }
            return Ok(None);
        }
        let variant = match name {
            "ferrum-linux-x86_64-cuda-sm89.tar.gz" => Self::Sm89,
            "ferrum-linux-x86_64-cuda-sm120.tar.gz" => Self::Sm120,
            _ => return Err("unrecognized CUDA distribution asset name".into()),
        };
        let (capability, toolkit, features): (_, _, &[&str]) = match variant {
            Self::Sm89 => (
                "89",
                "nvidia/cuda:12.4.0-devel-ubuntu22.04",
                &["cuda", "vllm-moe-marlin", "vllm-paged-attn-v2"],
            ),
            Self::Sm120 => (
                "120",
                "nvidia/cuda:12.8.1-devel-ubuntu24.04@sha256:520292dbb4f755fd360766059e62956e9379485d9e073bbd2f6e3c20c270ed66",
                &["cuda", "cuda-production"],
            ),
        };
        let actual: BTreeSet<_> = abi.cargo_features.iter().map(String::as_str).collect();
        if abi.target_triple != "x86_64-unknown-linux-gnu"
            || abi.cuda_compute_capability.as_deref() != Some(capability)
            || abi.cuda_toolkit_image.as_deref() != Some(toolkit)
            || actual != features.iter().copied().collect()
            || actual.len() != abi.cargo_features.len()
        {
            return Err(
                "CUDA asset name, target, capability, toolkit and production features disagree"
                    .into(),
            );
        }
        Ok(Some(variant))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const DECLARED: &str = "on:\n  workflow_call:\n    outputs:\n      cuda_sm120_asset_id:\n        value: '${{ jobs.sm120.outputs.asset_id }}'\n";

    #[test]
    fn candidate_workflow_declaration_is_parsed_not_inferred_from_text() {
        assert!(workflow_requires_sm120(DECLARED.as_bytes()).unwrap());
        assert!(workflow_requires_sm120(
            b"'on': {workflow_call: {outputs: {cuda_sm120_asset_id: {value: 'asset'}}}}"
        )
        .unwrap());
        for legacy in [
            "on:\n  workflow_call:\n    outputs:\n      cuda_asset_id: {value: legacy}\n",
            "on: {workflow_call: {}}\n# cuda_sm120_asset_id: present only in a comment\n",
            "on: {workflow_dispatch: {}}\njobs: {sm120: {name: cuda_sm120_asset_id}}\n",
        ] {
            assert!(!workflow_requires_sm120(legacy.as_bytes()).unwrap());
        }
        for invalid in [
            "on: [workflow_call]",
            "on: {workflow_call: {outputs: {cuda_sm120_asset_id: {}}}}",
            "on: {workflow_call: {outputs: {cuda_sm120_asset_id: {value: '  '}}}}",
            "on: {workflow_call: {outputs: {cuda_sm120_asset_id: {value: [asset]}}}}",
            "on: {workflow_call: {outputs: {cuda_sm120_asset_id: null}}}",
            "on: {workflow_call: {}}\non: {workflow_dispatch: {}}",
            "on: {workflow_call: {outputs: {cuda_sm120_asset_id: {value: a}, cuda_sm120_asset_id: {value: b}}}}",
        ] {
            assert!(workflow_requires_sm120(invalid.as_bytes()).is_err(), "{invalid}");
        }
    }

    #[tokio::test]
    async fn cuda_capability_uses_candidate_blob_instead_of_current_checkout() {
        let directory = tempfile::tempdir().unwrap();
        let git = |arguments: &[&str]| {
            let output = std::process::Command::new("git")
                .arg("-C")
                .arg(directory.path())
                .args(arguments)
                .env_remove("GIT_DIR")
                .env_remove("GIT_WORK_TREE")
                .env_remove("GIT_INDEX_FILE")
                .env("GIT_AUTHOR_NAME", "Fixture")
                .env("GIT_AUTHOR_EMAIL", "fixture@example.invalid")
                .env("GIT_COMMITTER_NAME", "Fixture")
                .env("GIT_COMMITTER_EMAIL", "fixture@example.invalid")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            String::from_utf8(output.stdout).unwrap().trim().to_owned()
        };
        git(&["init", "--quiet"]);
        git(&[
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "--allow-empty",
            "-m",
            "legacy",
        ]);
        let legacy = git(&["rev-parse", "HEAD"]);
        let path = directory.path().join(WORKFLOW_PATH);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, DECLARED).unwrap();
        git(&["add", WORKFLOW_PATH]);
        git(&[
            "-c",
            "commit.gpgsign=false",
            "commit",
            "--quiet",
            "-m",
            "declare SM120",
        ]);
        let current = git(&["rev-parse", "HEAD"]);
        // An uncommitted older declaration must not downgrade this candidate.
        std::fs::write(&path, "on: {workflow_call: {}}\n").unwrap();
        assert!(candidate_requires_sm120(directory.path(), &current)
            .await
            .unwrap());
        assert!(!candidate_requires_sm120(directory.path(), &legacy)
            .await
            .unwrap());
        assert!(candidate_requires_sm120(directory.path(), "HEAD")
            .await
            .is_err());
        assert!(candidate_requires_sm120(directory.path(), &"0".repeat(40))
            .await
            .is_err());
    }
}
