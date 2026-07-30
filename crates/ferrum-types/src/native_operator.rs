//! Native operator artifact manifest types.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const LEGACY_NATIVE_OPERATOR_MANIFEST_SCHEMA_VERSION: u32 = 1;
pub const PROVIDER_BOUND_NATIVE_OPERATOR_MANIFEST_SCHEMA_VERSION: u32 = 2;
pub const NATIVE_OPERATOR_MANIFEST_SCHEMA_VERSION: u32 = 3;
pub const NATIVE_OPERATOR_PROVIDER_CATALOG_SCHEMA_VERSION: u32 = 1;
pub const NATIVE_OPERATOR_ABI_CONTRACT_SCHEMA_VERSION: u32 = 1;
pub const LEGACY_FERRUM_NATIVE_OPERATOR_ABI_VERSION: &str = "1";
pub const FERRUM_NATIVE_OPERATOR_ABI_VERSION: &str = "2";
pub const DEFAULT_NATIVE_OPERATOR_ABI_VERSION: &str = "1";

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

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct NativeOperatorBinding {
    pub operation_id: String,
    pub operation_contract_version: NativeOperatorContractVersion,
    pub provider_id: String,
    pub provider_version: NativeOperatorContractVersion,
    pub provider_implementation_fingerprint: String,
    #[serde(default)]
    pub entrypoints: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct NativeOperatorContractVersion {
    pub major: u16,
    pub minor: u16,
}

impl NativeOperatorContractVersion {
    pub const fn new(major: u16, minor: u16) -> Self {
        Self { major, minor }
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum NativeOperatorContractVersionWire {
    Version { major: u16, minor: u16 },
    LegacyMajor(u32),
}

impl<'de> Deserialize<'de> for NativeOperatorContractVersion {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        match NativeOperatorContractVersionWire::deserialize(deserializer)? {
            NativeOperatorContractVersionWire::Version { major, minor } => {
                Ok(Self { major, minor })
            }
            NativeOperatorContractVersionWire::LegacyMajor(major) => {
                let major = u16::try_from(major)
                    .map_err(|_| serde::de::Error::custom("legacy contract major exceeds u16"))?;
                Ok(Self { major, minor: 0 })
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeOperatorProviderCatalog {
    pub schema_version: u32,
    pub backend: NativeOperatorBackend,
    pub providers: Vec<NativeOperatorProviderCatalogRow>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeOperatorProviderCatalogRow {
    pub operation_id: String,
    pub operation_contract_version: NativeOperatorContractVersion,
    pub operation_fingerprint: String,
    pub provider_id: String,
    pub provider_version: NativeOperatorContractVersion,
    pub provider_implementation_fingerprint: String,
}

impl NativeOperatorProviderCatalog {
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.schema_version != NATIVE_OPERATOR_PROVIDER_CATALOG_SCHEMA_VERSION {
            return Err(format!(
                "native operator provider catalog schema_version must be {NATIVE_OPERATOR_PROVIDER_CATALOG_SCHEMA_VERSION}"
            ));
        }
        if self.providers.is_empty() {
            return Err("native operator provider catalog must not be empty".to_string());
        }
        let provider_prefix = match self.backend {
            NativeOperatorBackend::Cuda => "provider.cuda.",
            NativeOperatorBackend::Metal => "provider.metal.",
            NativeOperatorBackend::Cpu => "provider.cpu.",
        };
        let mut previous_key: Option<(&str, &str)> = None;
        for (index, provider) in self.providers.iter().enumerate() {
            let label = format!("providers[{index}]");
            require_contract_identifier(
                &format!("{label}.operation_id"),
                &provider.operation_id,
                "operation.",
            )?;
            require_contract_identifier(
                &format!("{label}.provider_id"),
                &provider.provider_id,
                "provider.",
            )?;
            if !provider.provider_id.starts_with(provider_prefix) {
                return Err(format!(
                    "{label}.provider_id must match catalog backend {:?}",
                    self.backend
                ));
            }
            if provider.operation_contract_version.major == 0
                || provider.provider_version.major == 0
            {
                return Err(format!("{label} contract major versions must be positive"));
            }
            require_sha256(
                &format!("{label}.operation_fingerprint"),
                &provider.operation_fingerprint,
            )?;
            require_sha256(
                &format!("{label}.provider_implementation_fingerprint"),
                &provider.provider_implementation_fingerprint,
            )?;
            let key = (
                provider.operation_id.as_str(),
                provider.provider_id.as_str(),
            );
            if previous_key.is_some_and(|previous| previous >= key) {
                return Err(
                    "native operator provider catalog rows must be sorted and unique by operation_id/provider_id"
                        .to_string(),
                );
            }
            previous_key = Some(key);
        }
        Ok(())
    }

    pub fn canonical_json_bytes(&self) -> std::result::Result<Vec<u8>, String> {
        self.validate()?;
        canonical_json_bytes(self, "native operator provider catalog")
    }

    pub fn canonical_sha256(&self) -> std::result::Result<String, String> {
        Ok(format!(
            "{:x}",
            Sha256::digest(self.canonical_json_bytes()?)
        ))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeOperatorAbiContract {
    pub schema_version: u32,
    pub ferrum_native_abi_version: String,
    pub descriptor_struct: String,
    pub descriptor_symbol_policy: String,
    pub descriptor_fields: Vec<NativeOperatorAbiField>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NativeOperatorAbiField {
    pub name: String,
    pub c_type: String,
}

impl NativeOperatorAbiContract {
    pub fn validate(&self) -> std::result::Result<(), String> {
        if self.schema_version != NATIVE_OPERATOR_ABI_CONTRACT_SCHEMA_VERSION {
            return Err(format!(
                "native ABI contract schema_version must be {NATIVE_OPERATOR_ABI_CONTRACT_SCHEMA_VERSION}"
            ));
        }
        if self.ferrum_native_abi_version != FERRUM_NATIVE_OPERATOR_ABI_VERSION
            || self.descriptor_struct != "FerrumNativeOperatorDescriptorV2"
            || self.descriptor_symbol_policy != "operator_namespaced"
        {
            return Err(
                "native ABI contract version, descriptor, or symbol policy is unsupported"
                    .to_string(),
            );
        }
        let expected = [
            ("struct_size", "uint32_t"),
            ("ferrum_native_abi_version", "uint32_t"),
            ("operator_name", "const char *"),
            ("operator_abi_version", "const char *"),
            ("g03_catalog_sha256", "const char *"),
            ("abi_contract_sha256", "const char *"),
        ];
        if self.descriptor_fields.len() != expected.len()
            || self
                .descriptor_fields
                .iter()
                .zip(expected)
                .any(|(actual, (name, c_type))| actual.name != name || actual.c_type != c_type)
        {
            return Err(
                "native ABI descriptor fields differ from FerrumNativeOperatorDescriptorV2"
                    .to_string(),
            );
        }
        Ok(())
    }

    pub fn canonical_json_bytes(&self) -> std::result::Result<Vec<u8>, String> {
        self.validate()?;
        canonical_json_bytes(self, "native operator ABI contract")
    }

    pub fn canonical_sha256(&self) -> std::result::Result<String, String> {
        Ok(format!(
            "{:x}",
            Sha256::digest(self.canonical_json_bytes()?)
        ))
    }
}

fn canonical_json_bytes(
    value: &impl Serialize,
    label: &str,
) -> std::result::Result<Vec<u8>, String> {
    let mut bytes =
        serde_json::to_vec_pretty(value).map_err(|error| format!("serialize {label}: {error}"))?;
    bytes.push(b'\n');
    Ok(bytes)
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
    pub g03_catalog_sha256: Option<String>,
    #[serde(default)]
    pub abi_contract_sha256: Option<String>,
    #[serde(default)]
    pub descriptor_export: Option<String>,
    #[serde(default)]
    pub operation_bindings: Vec<NativeOperatorBinding>,
    #[serde(default)]
    pub exports: Vec<String>,
    #[serde(default)]
    pub license_files: Vec<String>,
    pub build_summary: NativeOperatorBuildSummary,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompiledNativeOperatorIdentity {
    pub schema_version: u32,
    pub operator: String,
    pub operator_abi_version: String,
    pub ferrum_native_abi_version: String,
    pub backend: NativeOperatorBackend,
    pub linkage: NativeOperatorLinkage,
    pub g03_catalog_sha256: Option<String>,
    pub abi_contract_sha256: Option<String>,
    pub descriptor_export: Option<String>,
    pub operation_bindings: Vec<NativeOperatorBinding>,
    pub exports: Vec<String>,
    pub source_package_sha256: String,
    pub inputs_sha256: String,
    pub binary_sha256: String,
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
    pub g03_catalog_sha256: Option<String>,
    pub abi_contract_sha256: Option<String>,
    pub descriptor_export: Option<String>,
    pub required_exports: Vec<String>,
    pub operation_bindings: Option<Vec<NativeOperatorBinding>>,
}

impl NativeOperatorRequirement {
    pub fn cuda(operator: impl Into<String>, compute_capability: impl Into<String>) -> Self {
        Self {
            operator: operator.into(),
            backend: NativeOperatorBackend::Cuda,
            operator_abi_version: DEFAULT_NATIVE_OPERATOR_ABI_VERSION.to_string(),
            ferrum_native_abi_version: FERRUM_NATIVE_OPERATOR_ABI_VERSION.to_string(),
            compute_capability: Some(compute_capability.into()),
            source_package_sha256: None,
            inputs_sha256: None,
            binary_sha256: None,
            g03_catalog_sha256: None,
            abi_contract_sha256: None,
            descriptor_export: None,
            required_exports: Vec::new(),
            operation_bindings: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NativeOperatorResolution {
    pub operator: String,
    pub backend: NativeOperatorBackend,
    pub linkage: NativeOperatorLinkage,
    pub binary_sha256: String,
    pub g03_catalog_sha256: Option<String>,
    pub abi_contract_sha256: Option<String>,
}

impl NativeOperatorManifest {
    pub fn validate(&self) -> std::result::Result<(), String> {
        if ![
            LEGACY_NATIVE_OPERATOR_MANIFEST_SCHEMA_VERSION,
            PROVIDER_BOUND_NATIVE_OPERATOR_MANIFEST_SCHEMA_VERSION,
            NATIVE_OPERATOR_MANIFEST_SCHEMA_VERSION,
        ]
        .contains(&self.schema_version)
        {
            return Err(format!(
                "schema_version must be {LEGACY_NATIVE_OPERATOR_MANIFEST_SCHEMA_VERSION}, \
                 {PROVIDER_BOUND_NATIVE_OPERATOR_MANIFEST_SCHEMA_VERSION}, or \
                 {NATIVE_OPERATOR_MANIFEST_SCHEMA_VERSION}"
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
        match self.schema_version {
            LEGACY_NATIVE_OPERATOR_MANIFEST_SCHEMA_VERSION => self.validate_legacy_v1()?,
            PROVIDER_BOUND_NATIVE_OPERATOR_MANIFEST_SCHEMA_VERSION => {
                self.validate_versioned(false)?
            }
            NATIVE_OPERATOR_MANIFEST_SCHEMA_VERSION => self.validate_versioned(true)?,
            _ => unreachable!("schema version checked above"),
        }
        Ok(())
    }

    fn validate_legacy_v1(&self) -> std::result::Result<(), String> {
        if self.ferrum_native_abi_version != LEGACY_FERRUM_NATIVE_OPERATOR_ABI_VERSION {
            return Err(format!(
                "legacy schema v1 requires ferrum_native_abi_version={LEGACY_FERRUM_NATIVE_OPERATOR_ABI_VERSION}"
            ));
        }
        if !self
            .exports
            .iter()
            .any(|export| export == "ferrum_native_op_init")
        {
            return Err("legacy schema v1 exports must include ferrum_native_op_init".to_string());
        }
        if !self
            .exports
            .iter()
            .any(|export| export == "ferrum_native_op_descriptor")
        {
            return Err(
                "legacy schema v1 exports must include ferrum_native_op_descriptor".to_string(),
            );
        }
        if self.g03_catalog_sha256.is_some()
            || self.abi_contract_sha256.is_some()
            || self.descriptor_export.is_some()
            || !self.operation_bindings.is_empty()
        {
            return Err("legacy schema v1 must not contain schema v2 identity fields".to_string());
        }
        Ok(())
    }

    fn validate_versioned(&self, allow_unbound_component: bool) -> std::result::Result<(), String> {
        let schema_label = format!("schema v{}", self.schema_version);
        let catalog_sha256 = self
            .g03_catalog_sha256
            .as_deref()
            .ok_or_else(|| format!("{schema_label} requires g03_catalog_sha256"))?;
        require_sha256("g03_catalog_sha256", catalog_sha256)?;
        let abi_contract_sha256 = self
            .abi_contract_sha256
            .as_deref()
            .ok_or_else(|| format!("{schema_label} requires abi_contract_sha256"))?;
        require_sha256("abi_contract_sha256", abi_contract_sha256)?;

        require_sorted_unique_symbols("exports", &self.exports)?;
        let descriptor_export = self
            .descriptor_export
            .as_deref()
            .ok_or_else(|| format!("{schema_label} requires descriptor_export"))?;
        require_native_symbol("descriptor_export", descriptor_export)?;
        if matches!(
            descriptor_export,
            "ferrum_native_op_init" | "ferrum_native_op_descriptor"
        ) {
            return Err(format!(
                "{schema_label} descriptor_export must be namespaced per native operator"
            ));
        }
        if !self
            .exports
            .iter()
            .any(|export| export == descriptor_export)
        {
            return Err(format!(
                "{schema_label} exports must include descriptor_export"
            ));
        }
        if !allow_unbound_component && self.operation_bindings.is_empty() {
            return Err("schema v2 requires at least one operation_binding".to_string());
        }
        if self.license_files.is_empty() {
            return Err(format!(
                "{schema_label} requires at least one license_files entry"
            ));
        }
        if self.license_files.windows(2).any(|pair| pair[0] >= pair[1])
            || self.license_files.iter().any(|path| {
                path.is_empty()
                    || path.starts_with('/')
                    || path.split('/').any(|component| component == "..")
            })
        {
            return Err(format!(
                "{schema_label} license_files must be sorted, unique, non-empty relative paths"
            ));
        }
        if !is_git_oid(&self.build_summary.builder_sha) {
            return Err(format!(
                "{schema_label} build_summary.builder_sha must be a lowercase 40- or 64-hex git object id"
            ));
        }
        if self.backend == NativeOperatorBackend::Cuda {
            require_non_empty(
                "cuda_toolkit",
                self.cuda_toolkit.as_deref().unwrap_or_default(),
            )?;
            require_non_empty(
                "cuda_runtime_min",
                self.cuda_runtime_min.as_deref().unwrap_or_default(),
            )?;
            require_non_empty(
                "build_summary.nvcc_version",
                self.build_summary
                    .nvcc_version
                    .as_deref()
                    .unwrap_or_default(),
            )?;
        }

        let mut previous_key: Option<(&str, &str)> = None;
        let mut keys = BTreeSet::new();
        for (index, binding) in self.operation_bindings.iter().enumerate() {
            let label = format!("operation_bindings[{index}]");
            require_contract_identifier(
                &format!("{label}.operation_id"),
                &binding.operation_id,
                "operation.",
            )?;
            require_contract_identifier(
                &format!("{label}.provider_id"),
                &binding.provider_id,
                "provider.",
            )?;
            if binding.operation_contract_version.major == 0 {
                return Err(format!(
                    "{label}.operation_contract_version major must be positive"
                ));
            }
            if binding.provider_version.major == 0 {
                return Err(format!("{label}.provider_version major must be positive"));
            }
            require_sha256(
                &format!("{label}.provider_implementation_fingerprint"),
                &binding.provider_implementation_fingerprint,
            )?;
            require_sorted_unique_symbols(&format!("{label}.entrypoints"), &binding.entrypoints)?;
            for entrypoint in &binding.entrypoints {
                if !self.exports.iter().any(|export| export == entrypoint) {
                    return Err(format!(
                        "{label}.entrypoints contains {entrypoint}, which is missing from exports"
                    ));
                }
            }
            let key = (binding.operation_id.as_str(), binding.provider_id.as_str());
            if let Some(previous) = previous_key {
                if previous >= key {
                    return Err(
                        "operation_bindings must be sorted and unique by operation_id/provider_id"
                            .to_string(),
                    );
                }
            }
            if !keys.insert((binding.operation_id.clone(), binding.provider_id.clone())) {
                return Err(
                    "operation_bindings contains a duplicate operation/provider".to_string()
                );
            }
            previous_key = Some(key);
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
    if let Some(expected) = requirement.g03_catalog_sha256.as_deref() {
        require_expected_optional_sha256(
            "g03_catalog_sha256",
            manifest.g03_catalog_sha256.as_deref(),
            expected,
        )?;
    }
    if let Some(expected) = requirement.abi_contract_sha256.as_deref() {
        require_expected_optional_sha256(
            "abi_contract_sha256",
            manifest.abi_contract_sha256.as_deref(),
            expected,
        )?;
    }
    if let Some(expected) = requirement.descriptor_export.as_deref() {
        if manifest.descriptor_export.as_deref() != Some(expected) {
            return Err(format!(
                "descriptor_export mismatch: manifest={:?} expected={expected}",
                manifest.descriptor_export
            ));
        }
    }
    for required_export in &requirement.required_exports {
        if !manifest
            .exports
            .iter()
            .any(|export| export == required_export)
        {
            return Err(format!(
                "required export is missing from manifest: {required_export}"
            ));
        }
    }
    if let Some(expected) = requirement.operation_bindings.as_ref() {
        if &manifest.operation_bindings != expected {
            return Err("operation_bindings mismatch".to_string());
        }
    }
    Ok(NativeOperatorResolution {
        operator: manifest.operator.clone(),
        backend: manifest.backend,
        linkage: manifest.linkage,
        binary_sha256: manifest.binary_sha256.clone(),
        g03_catalog_sha256: manifest.g03_catalog_sha256.clone(),
        abi_contract_sha256: manifest.abi_contract_sha256.clone(),
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

fn require_expected_optional_sha256(
    field: &str,
    actual: Option<&str>,
    expected: &str,
) -> std::result::Result<(), String> {
    let actual = actual.ok_or_else(|| format!("{field} is missing"))?;
    require_expected_sha256(field, actual, expected)
}

fn require_sorted_unique_symbols(
    field: &str,
    symbols: &[String],
) -> std::result::Result<(), String> {
    if symbols.is_empty() {
        return Err(format!("{field} must be non-empty"));
    }
    let mut previous: Option<&str> = None;
    for symbol in symbols {
        require_native_symbol(field, symbol)?;
        if previous.is_some_and(|value| value >= symbol.as_str()) {
            return Err(format!("{field} must be sorted and unique"));
        }
        previous = Some(symbol);
    }
    Ok(())
}

fn require_native_symbol(field: &str, value: &str) -> std::result::Result<(), String> {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return Err(format!("{field} must be non-empty"));
    };
    if !(first == '_' || first.is_ascii_alphabetic())
        || !chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
    {
        return Err(format!(
            "{field} contains an invalid native symbol: {value}"
        ));
    }
    Ok(())
}

fn require_contract_identifier(
    field: &str,
    value: &str,
    prefix: &str,
) -> std::result::Result<(), String> {
    if !value.starts_with(prefix)
        || !value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-'))
    {
        return Err(format!(
            "{field} must start with {prefix} and contain only canonical identifier characters"
        ));
    }
    Ok(())
}

pub fn is_sha256_digest(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn is_git_oid(value: &str) -> bool {
    matches!(value.len(), 40 | 64)
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
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
            ferrum_native_abi_version: FERRUM_NATIVE_OPERATOR_ABI_VERSION.to_string(),
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
            g03_catalog_sha256: Some(digest('d')),
            abi_contract_sha256: Some(digest('e')),
            descriptor_export: Some("ferrum_native_fa2_descriptor_v2".to_string()),
            operation_bindings: vec![NativeOperatorBinding {
                operation_id: "operation.causal_paged_attention".to_string(),
                operation_contract_version: NativeOperatorContractVersion::new(1, 0),
                provider_id: "provider.cuda.fa2".to_string(),
                provider_version: NativeOperatorContractVersion::new(1, 0),
                provider_implementation_fingerprint: digest('f'),
                entrypoints: vec!["ferrum_native_fa2_execute_v1".to_string()],
            }],
            exports: vec![
                "ferrum_native_fa2_descriptor_v2".to_string(),
                "ferrum_native_fa2_execute_v1".to_string(),
            ],
            license_files: vec!["LICENSE".to_string()],
            build_summary: NativeOperatorBuildSummary {
                builder_sha: digest('7'),
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
        requirement.g03_catalog_sha256 = Some(digest('d'));
        requirement.abi_contract_sha256 = Some(digest('e'));
        requirement.descriptor_export = Some("ferrum_native_fa2_descriptor_v2".to_string());
        requirement.required_exports = vec!["ferrum_native_fa2_execute_v1".to_string()];
        requirement.operation_bindings = Some(manifest().operation_bindings);

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

    #[test]
    fn versioned_schema_rejects_legacy_shared_descriptor_symbols() {
        let mut invalid = manifest();
        invalid.descriptor_export = Some("ferrum_native_op_descriptor".to_string());
        invalid.exports = vec![
            "ferrum_native_fa2_execute_v1".to_string(),
            "ferrum_native_op_descriptor".to_string(),
        ];
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn schema_v3_allows_a_native_leaf_without_a_g03_consumer() {
        let mut component = manifest();
        component.operation_bindings.clear();
        component.validate().unwrap();

        component.schema_version = PROVIDER_BOUND_NATIVE_OPERATOR_MANIFEST_SCHEMA_VERSION;
        assert!(component
            .validate()
            .unwrap_err()
            .contains("schema v2 requires at least one operation_binding"));
    }

    #[test]
    fn provider_catalog_and_abi_contract_validate_exact_versioned_identity() {
        let version = NativeOperatorContractVersion::new(1, 2);
        let mut catalog = NativeOperatorProviderCatalog {
            schema_version: NATIVE_OPERATOR_PROVIDER_CATALOG_SCHEMA_VERSION,
            backend: NativeOperatorBackend::Cuda,
            providers: vec![NativeOperatorProviderCatalogRow {
                operation_id: "operation.alpha".to_string(),
                operation_contract_version: version,
                operation_fingerprint: digest('1'),
                provider_id: "provider.cuda.alpha".to_string(),
                provider_version: version,
                provider_implementation_fingerprint: digest('2'),
            }],
        };
        catalog.validate().unwrap();
        let canonical = catalog.canonical_json_bytes().unwrap();
        assert_eq!(
            catalog.canonical_sha256().unwrap(),
            format!("{:x}", Sha256::digest(&canonical))
        );
        catalog.backend = NativeOperatorBackend::Metal;
        assert!(catalog.validate().is_err());
        catalog.backend = NativeOperatorBackend::Cuda;
        catalog.providers[0].provider_implementation_fingerprint = "not-a-digest".to_string();
        assert!(catalog.validate().is_err());

        let mut abi = NativeOperatorAbiContract {
            schema_version: NATIVE_OPERATOR_ABI_CONTRACT_SCHEMA_VERSION,
            ferrum_native_abi_version: FERRUM_NATIVE_OPERATOR_ABI_VERSION.to_string(),
            descriptor_struct: "FerrumNativeOperatorDescriptorV2".to_string(),
            descriptor_symbol_policy: "operator_namespaced".to_string(),
            descriptor_fields: [
                ("struct_size", "uint32_t"),
                ("ferrum_native_abi_version", "uint32_t"),
                ("operator_name", "const char *"),
                ("operator_abi_version", "const char *"),
                ("g03_catalog_sha256", "const char *"),
                ("abi_contract_sha256", "const char *"),
            ]
            .into_iter()
            .map(|(name, c_type)| NativeOperatorAbiField {
                name: name.to_string(),
                c_type: c_type.to_string(),
            })
            .collect(),
        };
        abi.validate().unwrap();
        assert_eq!(
            abi.canonical_sha256().unwrap(),
            format!("{:x}", Sha256::digest(abi.canonical_json_bytes().unwrap()))
        );
        abi.descriptor_fields.swap(0, 1);
        assert!(abi.validate().is_err());
    }

    #[test]
    fn legacy_schema_v1_remains_read_only_compatible() {
        let mut legacy = manifest();
        legacy.schema_version = LEGACY_NATIVE_OPERATOR_MANIFEST_SCHEMA_VERSION;
        legacy.ferrum_native_abi_version = LEGACY_FERRUM_NATIVE_OPERATOR_ABI_VERSION.to_string();
        legacy.g03_catalog_sha256 = None;
        legacy.abi_contract_sha256 = None;
        legacy.descriptor_export = None;
        legacy.operation_bindings.clear();
        legacy.exports = vec![
            "ferrum_native_op_init".to_string(),
            "ferrum_native_op_descriptor".to_string(),
        ];
        legacy.validate().unwrap();
    }
}
