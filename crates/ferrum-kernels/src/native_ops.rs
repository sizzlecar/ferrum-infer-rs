//! Kernel-side adapter for Ferrum native operator artifacts.
//!
//! This module is intentionally path/config driven. It does not read
//! environment variables or select an accelerator path by itself; product
//! entrypoints must pass typed manifest/artifact choices into this layer.

use std::path::PathBuf;

use ferrum_native_ops::{
    NativeOperatorArtifactFormat, NativeOperatorResolveError, NativeOperatorResolveRequest,
    NativeOperatorResolver,
};
use ferrum_types::{
    resolve_native_operator_manifest, NativeOperatorBackend, NativeOperatorLinkage,
    NativeOperatorRequirement,
};

pub const FA2_NATIVE_OPERATOR: &str = "fa2";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeOperatorArtifactSpec {
    pub operator: String,
    pub backend: NativeOperatorBackend,
    pub compute_capability: Option<String>,
    pub manifest_path: PathBuf,
    pub artifact_path: PathBuf,
    pub source_package_sha256: Option<String>,
    pub inputs_sha256: Option<String>,
    pub binary_sha256: Option<String>,
}

impl NativeOperatorArtifactSpec {
    pub fn cuda_fa2(
        manifest_path: impl Into<PathBuf>,
        artifact_path: impl Into<PathBuf>,
        compute_capability: impl Into<String>,
    ) -> Self {
        Self {
            operator: FA2_NATIVE_OPERATOR.to_string(),
            backend: NativeOperatorBackend::Cuda,
            compute_capability: Some(compute_capability.into()),
            manifest_path: manifest_path.into(),
            artifact_path: artifact_path.into(),
            source_package_sha256: None,
            inputs_sha256: None,
            binary_sha256: None,
        }
    }

    pub fn with_source_package_sha256(mut self, sha256: impl Into<String>) -> Self {
        self.source_package_sha256 = Some(sha256.into());
        self
    }

    pub fn with_inputs_sha256(mut self, sha256: impl Into<String>) -> Self {
        self.inputs_sha256 = Some(sha256.into());
        self
    }

    pub fn with_binary_sha256(mut self, sha256: impl Into<String>) -> Self {
        self.binary_sha256 = Some(sha256.into());
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeOperatorRuntimeSelection {
    pub operator: String,
    pub backend: NativeOperatorBackend,
    pub compute_capability: Option<String>,
    pub linkage: NativeOperatorLinkage,
    pub manifest_path: PathBuf,
    pub artifact_path: PathBuf,
    pub binary_sha256: String,
    pub source_package_sha256: String,
    pub inputs_sha256: String,
    pub artifact_format: NativeOperatorArtifactFormat,
    pub archive_members: Vec<String>,
    pub required_exports: Vec<String>,
    pub matched_exports: Vec<String>,
}

pub fn resolve_native_operator_artifact(
    spec: &NativeOperatorArtifactSpec,
) -> Result<NativeOperatorRuntimeSelection, NativeOperatorResolveError> {
    let mut request = NativeOperatorResolveRequest::new(
        spec.operator.clone(),
        spec.backend,
        spec.manifest_path.clone(),
        spec.artifact_path.clone(),
    );
    if let Some(compute_capability) = spec.compute_capability.clone() {
        request = request.with_compute_capability(compute_capability);
    }

    let resolved = NativeOperatorResolver.resolve(&request)?;
    let mut requirement = NativeOperatorRequirement {
        operator: spec.operator.clone(),
        backend: spec.backend,
        operator_abi_version: resolved.manifest.operator_abi_version.clone(),
        ferrum_native_abi_version: resolved.manifest.ferrum_native_abi_version.clone(),
        compute_capability: spec.compute_capability.clone(),
        source_package_sha256: spec.source_package_sha256.clone(),
        inputs_sha256: spec.inputs_sha256.clone(),
        binary_sha256: spec
            .binary_sha256
            .clone()
            .or_else(|| Some(resolved.artifact_sha256.clone())),
    };
    if requirement.source_package_sha256.is_none() {
        requirement.source_package_sha256 = Some(resolved.manifest.source_package.sha256.clone());
    }
    if requirement.inputs_sha256.is_none() {
        requirement.inputs_sha256 = Some(resolved.manifest.inputs_sha256.clone());
    }
    resolve_native_operator_manifest(Some(&resolved.manifest), &requirement)
        .map_err(NativeOperatorResolveError::ManifestInvalid)?;

    Ok(NativeOperatorRuntimeSelection {
        operator: resolved.manifest.operator.clone(),
        backend: resolved.manifest.backend,
        compute_capability: spec.compute_capability.clone(),
        linkage: resolved.manifest.linkage,
        manifest_path: resolved.manifest_path,
        artifact_path: resolved.artifact_path,
        binary_sha256: resolved.artifact_sha256,
        source_package_sha256: resolved.manifest.source_package.sha256,
        inputs_sha256: resolved.manifest.inputs_sha256,
        artifact_format: resolved.binary_validation.format,
        archive_members: resolved.binary_validation.archive_members,
        required_exports: resolved.binary_validation.required_exports,
        matched_exports: resolved.binary_validation.matched_exports,
    })
}

pub fn resolve_cuda_fa2_native_operator(
    spec: &NativeOperatorArtifactSpec,
) -> Result<NativeOperatorRuntimeSelection, NativeOperatorResolveError> {
    if spec.operator != FA2_NATIVE_OPERATOR || spec.backend != NativeOperatorBackend::Cuda {
        return Err(NativeOperatorResolveError::ManifestInvalid(
            "FA2 native operator selection requires operator=fa2 backend=cuda".to_string(),
        ));
    }
    resolve_native_operator_artifact(spec)
}
