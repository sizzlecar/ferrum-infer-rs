//! Native operator artifact manifest types.

use serde::{Deserialize, Serialize};

pub const NATIVE_OPERATOR_MANIFEST_SCHEMA_VERSION: u32 = 1;
pub const FERRUM_NATIVE_OPERATOR_ABI_VERSION: &str = "1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeOperatorBackend {
    Cuda,
    Metal,
    Cpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NativeOperatorLinkage {
    Static,
    Dynamic,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperatorSourcePackage {
    pub kind: String,
    pub revision: String,
    pub sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperatorBuildSummary {
    pub builder_sha: String,
    pub elapsed_ms: u64,
    pub nvcc_version: Option<String>,
    pub host_compiler: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NativeOperatorManifest {
    pub schema_version: u32,
    pub operator: String,
    pub operator_abi_version: String,
    pub ferrum_native_abi_version: String,
    pub backend: NativeOperatorBackend,
    pub cuda_toolkit: Option<String>,
    pub cuda_runtime_min: Option<String>,
    #[serde(default)]
    pub compute_capabilities: Vec<String>,
    pub source_package: NativeOperatorSourcePackage,
    pub inputs_sha256: String,
    pub binary_sha256: String,
    pub linkage: NativeOperatorLinkage,
    #[serde(default)]
    pub exports: Vec<String>,
    #[serde(default)]
    pub license_files: Vec<String>,
    pub build_summary: NativeOperatorBuildSummary,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeOperatorRequirement {
    pub operator: String,
    pub backend: NativeOperatorBackend,
    pub operator_abi_version: String,
    pub ferrum_native_abi_version: String,
    pub compute_capability: Option<String>,
    pub source_package_sha256: Option<String>,
    pub inputs_sha256: Option<String>,
    pub binary_sha256: Option<String>,
}

impl NativeOperatorRequirement {
    pub fn cuda(operator: impl Into<String>, compute_capability: impl Into<String>) -> Self {
        Self {
            operator: operator.into(),
            backend: NativeOperatorBackend::Cuda,
            operator_abi_version: FERRUM_NATIVE_OPERATOR_ABI_VERSION.to_string(),
            ferrum_native_abi_version: FERRUM_NATIVE_OPERATOR_ABI_VERSION.to_string(),
            compute_capability: Some(compute_capability.into()),
            source_package_sha256: None,
            inputs_sha256: None,
            binary_sha256: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeOperatorResolution {
    pub operator: String,
    pub backend: NativeOperatorBackend,
    pub linkage: NativeOperatorLinkage,
    pub binary_sha256: String,
}

impl NativeOperatorManifest {
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.schema_version != NATIVE_OPERATOR_MANIFEST_SCHEMA_VERSION {
            return Err(format!(
                "schema_version must be {NATIVE_OPERATOR_MANIFEST_SCHEMA_VERSION}"
            ));
        }
        require_non_empty("operator", &self.operator)?;
        require_non_empty("operator_abi_version", &self.operator_abi_version)?;
        require_non_empty("ferrum_native_abi_version", &self.ferrum_native_abi_version)?;
        require_non_empty("source_package.kind", &self.source_package.kind)?;
        require_non_empty("source_package.revision", &self.source_package.revision)?;
        require_sha256("source_package.sha256", &self.source_package.sha256)?;
        require_sha256("inputs_sha256", &self.inputs_sha256)?;
        require_sha256("binary_sha256", &self.binary_sha256)?;
        require_non_empty("build_summary.builder_sha", &self.build_summary.builder_sha)?;
        require_non_empty(
            "build_summary.host_compiler",
            &self.build_summary.host_compiler,
        )?;
        if self.backend == NativeOperatorBackend::Cuda {
            if self.compute_capabilities.is_empty() {
                return Err(
                    "cuda native operator manifest requires compute_capabilities".to_string(),
                );
            }
            for capability in &self.compute_capabilities {
                if !capability.starts_with("sm_") {
                    return Err("compute_capabilities entries must use sm_xx form".to_string());
                }
            }
        }
        if !self
            .exports
            .iter()
            .any(|export| export == "ferrum_native_op_init")
        {
            return Err("exports must include ferrum_native_op_init".to_string());
        }
        Ok(())
    }
}

pub fn resolve_native_operator_manifest(
    manifest: Option<&NativeOperatorManifest>,
    requirement: &NativeOperatorRequirement,
) -> std::result::Result<NativeOperatorResolution, String> {
    let manifest = manifest.ok_or_else(|| "native operator manifest is missing".to_string())?;
    manifest.validate()?;
    if manifest.operator != requirement.operator {
        return Err(format!(
            "native operator mismatch: manifest={} required={}",
            manifest.operator, requirement.operator
        ));
    }
    if manifest.backend != requirement.backend {
        return Err(format!(
            "native operator backend mismatch: manifest={:?} required={:?}",
            manifest.backend, requirement.backend
        ));
    }
    if manifest.operator_abi_version != requirement.operator_abi_version {
        return Err(format!(
            "native operator ABI mismatch: manifest={} required={}",
            manifest.operator_abi_version, requirement.operator_abi_version
        ));
    }
    if manifest.ferrum_native_abi_version != requirement.ferrum_native_abi_version {
        return Err(format!(
            "Ferrum native ABI mismatch: manifest={} required={}",
            manifest.ferrum_native_abi_version, requirement.ferrum_native_abi_version
        ));
    }
    if let Some(required_capability) = requirement.compute_capability.as_deref() {
        if !manifest
            .compute_capabilities
            .iter()
            .any(|capability| capability == required_capability)
        {
            return Err(format!(
                "compute capability mismatch: manifest={:?} required={}",
                manifest.compute_capabilities, required_capability
            ));
        }
    }
    if let Some(expected) = requirement.source_package_sha256.as_deref() {
        require_expected_sha256(
            "source_package.sha256",
            &manifest.source_package.sha256,
            expected,
        )?;
    }
    if let Some(expected) = requirement.inputs_sha256.as_deref() {
        require_expected_sha256("inputs_sha256", &manifest.inputs_sha256, expected)?;
    }
    if let Some(expected) = requirement.binary_sha256.as_deref() {
        require_expected_sha256("binary_sha256", &manifest.binary_sha256, expected)?;
    }
    Ok(NativeOperatorResolution {
        operator: manifest.operator.clone(),
        backend: manifest.backend,
        linkage: manifest.linkage,
        binary_sha256: manifest.binary_sha256.clone(),
    })
}

fn require_non_empty(field: &str, value: &str) -> std::result::Result<(), String> {
    if value.trim().is_empty() {
        Err(format!("{field} must be non-empty"))
    } else {
        Ok(())
    }
}

fn require_sha256(field: &str, value: &str) -> std::result::Result<(), String> {
    if is_sha256_digest(value) {
        Ok(())
    } else {
        Err(format!("{field} must be a lowercase hex sha256 digest"))
    }
}

fn require_expected_sha256(
    field: &str,
    actual: &str,
    expected: &str,
) -> std::result::Result<(), String> {
    require_sha256(field, actual)?;
    require_sha256(&format!("expected {field}"), expected)?;
    if actual.eq_ignore_ascii_case(expected) {
        Ok(())
    } else {
        Err(format!(
            "{field} mismatch: manifest={actual} expected={expected}"
        ))
    }
}

pub fn is_sha256_digest(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(ch: char) -> String {
        std::iter::repeat(ch).take(64).collect()
    }

    fn manifest() -> NativeOperatorManifest {
        NativeOperatorManifest {
            schema_version: NATIVE_OPERATOR_MANIFEST_SCHEMA_VERSION,
            operator: "fa2".to_string(),
            operator_abi_version: "1".to_string(),
            ferrum_native_abi_version: "1".to_string(),
            backend: NativeOperatorBackend::Cuda,
            cuda_toolkit: Some("12.4".to_string()),
            cuda_runtime_min: Some("12.4".to_string()),
            compute_capabilities: vec!["sm_89".to_string()],
            source_package: NativeOperatorSourcePackage {
                kind: "external_archive".to_string(),
                revision: "rev".to_string(),
                sha256: digest('a'),
            },
            inputs_sha256: digest('b'),
            binary_sha256: digest('c'),
            linkage: NativeOperatorLinkage::Static,
            exports: vec!["ferrum_native_op_init".to_string()],
            license_files: vec!["LICENSE".to_string()],
            build_summary: NativeOperatorBuildSummary {
                builder_sha: "builder".to_string(),
                elapsed_ms: 1,
                nvcc_version: Some("12.4".to_string()),
                host_compiler: "clang".to_string(),
            },
        }
    }

    #[test]
    fn validates_required_hashes_and_cuda_capability() {
        manifest().validate().unwrap();

        let mut missing_hash = manifest();
        missing_hash.binary_sha256.clear();
        assert!(missing_hash.validate().is_err());

        let mut bad_capability = manifest();
        bad_capability.compute_capabilities = vec!["rtx4090".to_string()];
        assert!(bad_capability.validate().is_err());
    }

    #[test]
    fn resolver_fails_closed_for_missing_or_mismatched_manifest() {
        let mut requirement = NativeOperatorRequirement::cuda("fa2", "sm_89");
        requirement.source_package_sha256 = Some(digest('a'));
        requirement.inputs_sha256 = Some(digest('b'));
        requirement.binary_sha256 = Some(digest('c'));

        let resolution = resolve_native_operator_manifest(Some(&manifest()), &requirement).unwrap();
        assert_eq!(resolution.operator, "fa2");
        assert_eq!(resolution.binary_sha256, digest('c'));

        assert!(resolve_native_operator_manifest(None, &requirement).is_err());

        let mut bad_binary = requirement.clone();
        bad_binary.binary_sha256 = Some(digest('d'));
        assert!(resolve_native_operator_manifest(Some(&manifest()), &bad_binary).is_err());

        let mut bad_abi = manifest();
        bad_abi.operator_abi_version = "2".to_string();
        assert!(resolve_native_operator_manifest(Some(&bad_abi), &requirement).is_err());

        let bad_capability = NativeOperatorRequirement::cuda("fa2", "sm_90");
        assert!(resolve_native_operator_manifest(Some(&manifest()), &bad_capability).is_err());

        let wrong_operator = NativeOperatorRequirement::cuda("dummy", "sm_89");
        assert!(resolve_native_operator_manifest(Some(&manifest()), &wrong_operator).is_err());
    }
}
