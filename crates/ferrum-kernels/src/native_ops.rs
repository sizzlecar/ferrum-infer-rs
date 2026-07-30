//! Kernel-side adapter for Ferrum native operator artifacts.
//!
//! This module is intentionally path/config driven. It does not read
//! environment variables or select an accelerator path by itself; product
//! entrypoints must pass typed manifest/artifact choices into this layer.

use std::path::PathBuf;
use std::sync::OnceLock;

use ferrum_native_ops::{
    NativeOperatorArtifactFormat, NativeOperatorResolveError, NativeOperatorResolveRequest,
    NativeOperatorResolver,
};
use ferrum_types::{
    resolve_native_operator_manifest, NativeOperatorBackend, NativeOperatorBinding,
    NativeOperatorLinkage, NativeOperatorProviderCatalog, NativeOperatorRequirement,
    NATIVE_OPERATOR_MANIFEST_SCHEMA_VERSION,
};

pub use ferrum_types::CompiledNativeOperatorIdentity as CompiledNativeOperatorArtifact;

pub const FA2_NATIVE_OPERATOR: &str = "fa2";
pub const CUDA_NATIVE_SOURCE_BUNDLE_ID: &str = "ferrum-native-cuda-v1+sha256.\
7f6f35f91a85df6ea5374d5597f7f8ca4c159b5e567c1c1ef122a0ab88657613";

pub fn compiled_native_operator_artifacts() -> &'static [CompiledNativeOperatorArtifact] {
    static COMPILED: OnceLock<Vec<CompiledNativeOperatorArtifact>> = OnceLock::new();
    COMPILED
        .get_or_init(|| {
            serde_json::from_str(
                option_env!("FERRUM_COMPILED_NATIVE_OPERATOR_SET_JSON").unwrap_or("[]"),
            )
            .expect("build.rs emitted invalid native operator inventory JSON")
        })
        .as_slice()
}

pub fn validate_compiled_native_operator_provider_catalog(
    catalog: &NativeOperatorProviderCatalog,
    artifacts: &[CompiledNativeOperatorArtifact],
) -> Result<(), String> {
    catalog.validate()?;
    if artifacts.is_empty() {
        return Ok(());
    }
    let catalog_sha256 = catalog.canonical_sha256()?;
    let mut binding_count = 0_usize;
    for artifact in artifacts {
        if artifact.schema_version != NATIVE_OPERATOR_MANIFEST_SCHEMA_VERSION {
            return Err(format!(
                "compiled native operator {} uses schema {}, expected {}",
                artifact.operator, artifact.schema_version, NATIVE_OPERATOR_MANIFEST_SCHEMA_VERSION
            ));
        }
        if artifact.backend != catalog.backend {
            return Err(format!(
                "compiled native operator {} backend {:?} differs from live catalog {:?}",
                artifact.operator, artifact.backend, catalog.backend
            ));
        }
        if artifact.g03_catalog_sha256.as_deref() != Some(catalog_sha256.as_str()) {
            return Err(format!(
                "compiled native operator {} is stale for the live provider catalog",
                artifact.operator
            ));
        }
        for binding in &artifact.operation_bindings {
            binding_count = binding_count.checked_add(1).ok_or_else(|| {
                "compiled native operator binding count overflows usize".to_string()
            })?;
            let live = catalog
                .providers
                .iter()
                .find(|provider| {
                    provider.operation_id == binding.operation_id
                        && provider.provider_id == binding.provider_id
                })
                .ok_or_else(|| {
                    format!(
                        "compiled native operator {} binds missing live provider {}/{}",
                        artifact.operator, binding.operation_id, binding.provider_id
                    )
                })?;
            if live.operation_contract_version != binding.operation_contract_version
                || live.provider_version != binding.provider_version
                || live.provider_implementation_fingerprint
                    != binding.provider_implementation_fingerprint
            {
                return Err(format!(
                    "compiled native operator {} binding {}/{} differs from the live provider identity",
                    artifact.operator, binding.operation_id, binding.provider_id
                ));
            }
        }
    }
    if binding_count == 0 {
        return Err(
            "compiled native operator set does not bind any live G03 operation/provider"
                .to_string(),
        );
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CompiledFa2NativeOperatorArtifact {
    pub manifest_path: String,
    pub artifact_path: String,
    pub source_package_sha256: String,
    pub inputs_sha256: String,
    pub binary_sha256: String,
}

pub fn compiled_fa2_native_operator_artifact_linked() -> bool {
    compiled_fa2_native_operator_artifact().is_some()
}

pub fn compiled_fa2_native_operator_artifact_state() -> &'static str {
    option_env!("FERRUM_FA2_NATIVE_ARTIFACT_COMPILE").unwrap_or("not_configured")
}

pub fn compiled_fa2_native_operator_artifact() -> Option<CompiledFa2NativeOperatorArtifact> {
    Some(CompiledFa2NativeOperatorArtifact {
        manifest_path: option_env!("FERRUM_COMPILED_FA2_NATIVE_MANIFEST")?.to_string(),
        artifact_path: option_env!("FERRUM_COMPILED_FA2_NATIVE_ARTIFACT")?.to_string(),
        source_package_sha256: option_env!("FERRUM_COMPILED_FA2_NATIVE_SOURCE_SHA256")?.to_string(),
        inputs_sha256: option_env!("FERRUM_COMPILED_FA2_NATIVE_INPUTS_SHA256")?.to_string(),
        binary_sha256: option_env!("FERRUM_COMPILED_FA2_NATIVE_BINARY_SHA256")?.to_string(),
    })
}

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
    pub schema_version: u32,
    pub operator: String,
    pub operator_abi_version: String,
    pub ferrum_native_abi_version: String,
    pub backend: NativeOperatorBackend,
    pub compute_capability: Option<String>,
    pub linkage: NativeOperatorLinkage,
    pub manifest_path: PathBuf,
    pub artifact_path: PathBuf,
    pub binary_sha256: String,
    pub source_package_sha256: String,
    pub inputs_sha256: String,
    pub g03_catalog_sha256: Option<String>,
    pub abi_contract_sha256: Option<String>,
    pub descriptor_export: Option<String>,
    pub operation_bindings: Vec<NativeOperatorBinding>,
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
        g03_catalog_sha256: resolved.manifest.g03_catalog_sha256.clone(),
        abi_contract_sha256: resolved.manifest.abi_contract_sha256.clone(),
        descriptor_export: resolved.manifest.descriptor_export.clone(),
        required_exports: resolved.manifest.exports.clone(),
        operation_bindings: Some(resolved.manifest.operation_bindings.clone()),
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
        schema_version: resolved.manifest.schema_version,
        operator: resolved.manifest.operator.clone(),
        operator_abi_version: resolved.manifest.operator_abi_version.clone(),
        ferrum_native_abi_version: resolved.manifest.ferrum_native_abi_version.clone(),
        backend: resolved.manifest.backend,
        compute_capability: spec.compute_capability.clone(),
        linkage: resolved.manifest.linkage,
        manifest_path: resolved.manifest_path,
        artifact_path: resolved.artifact_path,
        binary_sha256: resolved.artifact_sha256,
        source_package_sha256: resolved.manifest.source_package.sha256,
        inputs_sha256: resolved.manifest.inputs_sha256,
        g03_catalog_sha256: resolved.manifest.g03_catalog_sha256,
        abi_contract_sha256: resolved.manifest.abi_contract_sha256,
        descriptor_export: resolved.manifest.descriptor_export,
        operation_bindings: resolved.manifest.operation_bindings,
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
