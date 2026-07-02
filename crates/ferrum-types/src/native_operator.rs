//! Native operator artifact manifest types.

use serde::{Deserialize, Serialize};

pub const NATIVE_OPERATOR_MANIFEST_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
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
}
