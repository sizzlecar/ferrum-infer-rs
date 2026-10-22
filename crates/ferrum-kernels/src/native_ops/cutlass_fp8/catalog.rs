//! Source declaration for first-artifact packaging, without CUDA initialization.
//! This hashes real source and the real operation contract. It is not a claim
//! that an artifact is installed, executable or numerically qualified. The
//! executable provider additionally binds the actual compiled artifact identity.

use ferrum_runtime::{dense_linear_fp8_bf16_contract, OperationContract};
use ferrum_types::{
    NativeOperatorBackend, NativeOperatorContractVersion, NativeOperatorProviderCatalog,
    NativeOperatorProviderCatalogRow, NATIVE_OPERATOR_PROVIDER_CATALOG_SCHEMA_VERSION,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const CUTLASS_FP8_PROVIDER_ID: &str = "provider.cuda.dense_linear_fp8.bf16.cutlass.sm120";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CutlassFp8CatalogConfig {
    pub opaque_memory_bytes: u64,
    pub maximum_plans: usize,
    pub opaque_memory_profile: String,
}

impl CutlassFp8CatalogConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.opaque_memory_bytes == 0
            || self.maximum_plans == 0
            || self
                .maximum_plans
                .checked_mul(std::mem::size_of::<super::PreparedCutlassFp8>())
                .is_none_or(|bytes| bytes > isize::MAX as usize)
            || self.opaque_memory_profile.trim().is_empty()
            || self.opaque_memory_profile.len() > 256
            || !self.opaque_memory_profile.is_ascii()
            || self
                .opaque_memory_profile
                .bytes()
                .any(|b| b.is_ascii_control())
        {
            return Err(
                "FP8 declaration requires positive bounded capacity/plans and an ASCII profile"
                    .into(),
            );
        }
        Ok(())
    }
}

/// A source-derived identity also consumed by the executable CUDA provider.
/// No runtime, artifact lookup, device query, FFI or environment read occurs.
pub fn source_declaration_fingerprint() -> String {
    digest(&[
        b"ferrum.cutlass-fp8.source-declaration.v1",
        super::CUTLASS_FP8_NUMERICAL_IDENTITY.as_bytes(),
        include_bytes!("catalog.rs"),
        include_bytes!("../cutlass_fp8.rs"),
        include_bytes!("../../backend/cuda/vnext_ops/tensor_dense_fp8_bf16.rs"),
        include_bytes!("../../backend/cuda/vnext_ops/tensor_dense_fp8_bf16/weights.rs"),
        include_bytes!("../../backend/cuda/vnext_runtime/tensor_eager/cutlass_owner.rs"),
        include_bytes!("../../../../../native-operators/cuda/cutlass-fp8/abi.h"),
        include_bytes!("../../../../../native-operators/cuda/cutlass-fp8/gemm.cuh"),
        include_bytes!("../../../../../native-operators/cuda/cutlass-fp8/quantize.cuh"),
        include_bytes!("../../../../../native-operators/cuda/cutlass-fp8/boundary.cu"),
        include_bytes!(
            "../../../../../native-operators/cuda/source-definitions/cutlass-fp8.seed.json"
        ),
    ])
}

/// Declare the provider's real source/contract/configuration before its first
/// native artifact exists. Packaging records this provenance; execution still
/// requires its own actual artifact, runtime and prepared implementation checks.
pub fn provider_catalog_declaration(
    config: &CutlassFp8CatalogConfig,
) -> Result<NativeOperatorProviderCatalog, String> {
    config.validate()?;
    let contract = dense_linear_fp8_bf16_contract().map_err(|error| error.to_string())?;
    let operation = contract.descriptor();
    let config_bytes = serde_json::to_vec(config).map_err(|error| error.to_string())?;
    let catalog = NativeOperatorProviderCatalog {
        schema_version: NATIVE_OPERATOR_PROVIDER_CATALOG_SCHEMA_VERSION,
        backend: NativeOperatorBackend::Cuda,
        providers: vec![NativeOperatorProviderCatalogRow {
            operation_id: operation.id.to_string(),
            operation_contract_version: NativeOperatorContractVersion::new(
                operation.version.major,
                operation.version.minor,
            ),
            operation_fingerprint: operation.fingerprint().map_err(|error| error.to_string())?,
            provider_id: CUTLASS_FP8_PROVIDER_ID.into(),
            provider_version: NativeOperatorContractVersion::new(1, 0),
            provider_implementation_fingerprint: digest(&[
                b"ferrum.cutlass-fp8.provider-source-declaration.v1",
                source_declaration_fingerprint().as_bytes(),
                &config_bytes,
            ]),
        }],
    };
    catalog.validate()?;
    Ok(catalog)
}

fn digest(parts: &[&[u8]]) -> String {
    let mut hash = Sha256::new();
    for part in parts {
        hash.update((part.len() as u64).to_le_bytes());
        hash.update(part);
    }
    format!("{:x}", hash.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn declaration_uses_real_contract_and_changes_with_explicit_configuration() {
        let mut config = CutlassFp8CatalogConfig {
            opaque_memory_bytes: 64 << 20,
            maximum_plans: 16,
            opaque_memory_profile: "experimental-test-v1".into(),
        };
        let first = provider_catalog_declaration(&config).unwrap();
        assert_eq!(
            first.providers[0].operation_fingerprint,
            dense_linear_fp8_bf16_contract()
                .unwrap()
                .descriptor()
                .fingerprint()
                .unwrap()
        );
        assert_eq!(
            first.canonical_json_bytes().unwrap(),
            provider_catalog_declaration(&config)
                .unwrap()
                .canonical_json_bytes()
                .unwrap()
        );
        config.maximum_plans += 1;
        assert_ne!(
            first.canonical_sha256().unwrap(),
            provider_catalog_declaration(&config)
                .unwrap()
                .canonical_sha256()
                .unwrap()
        );
        config.opaque_memory_bytes = 0;
        assert!(provider_catalog_declaration(&config).is_err());
    }
}
