//! CPU-only declaration of the automatic GPT-OSS attention implementation.
//! This identifies source and contracts, not an installed/qualified artifact.

use ferrum_runtime::{gpt_oss_causal_paged_attention_contract, OperationContract};
use ferrum_types::{
    NativeOperatorBackend, NativeOperatorContractVersion, NativeOperatorProviderCatalog,
    NativeOperatorProviderCatalogRow, NATIVE_OPERATOR_PROVIDER_CATALOG_SCHEMA_VERSION,
};
use sha2::{Digest, Sha256};

pub const GPT_OSS_ATTENTION_PROVIDER_ID: &str = "provider.cuda.gpt_oss.causal_paged_attention.f16";

/// The executable provider uses this exact source identity. The selected
/// device, workspace estimate and prepared native plan are separately bound to
/// estimator/command identities, so this does not require a CUDA bootstrap.
pub fn provider_source_fingerprint() -> String {
    let native_source = super::source_declaration_fingerprint();
    let mut digest = Sha256::new();
    for bytes in [
        b"ferrum.gpt-oss-attention.auto-xqa.source.v1".as_slice(),
        GPT_OSS_ATTENTION_PROVIDER_ID.as_bytes(),
        native_source.as_bytes(),
        include_bytes!("catalog.rs"),
        include_bytes!("../../backend/cuda/vnext_ops/transformer/gpt_oss_attention.rs"),
        include_bytes!("../../backend/cuda/vnext_ops/transformer/gpt_oss_attention/xqa.rs"),
        include_bytes!("../../../kernels/gpt_oss_attention.cu"),
        include_bytes!("../../../kernels/rms_norm.cu"),
        include_bytes!("../../../kernels/sandwich_norm.cu"),
    ] {
        digest.update((bytes.len() as u64).to_le_bytes());
        digest.update(bytes);
    }
    format!("{:x}", digest.finalize())
}

pub fn provider_catalog_declaration() -> Result<NativeOperatorProviderCatalog, String> {
    let contract = gpt_oss_causal_paged_attention_contract().map_err(|error| error.to_string())?;
    let operation = contract.descriptor();
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
            provider_id: GPT_OSS_ATTENTION_PROVIDER_ID.to_owned(),
            provider_version: NativeOperatorContractVersion::new(1, 0),
            provider_implementation_fingerprint: provider_source_fingerprint(),
        }],
    };
    catalog.validate()?;
    Ok(catalog)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn declaration_uses_the_executable_source_identity_and_real_contract() {
        let catalog = provider_catalog_declaration().unwrap();
        let row = &catalog.providers[0];
        assert_eq!(
            row.provider_implementation_fingerprint,
            provider_source_fingerprint()
        );
        assert_eq!(
            row.operation_fingerprint,
            gpt_oss_causal_paged_attention_contract()
                .unwrap()
                .descriptor()
                .fingerprint()
                .unwrap()
        );
        assert_eq!(
            catalog.canonical_json_bytes().unwrap(),
            provider_catalog_declaration()
                .unwrap()
                .canonical_json_bytes()
                .unwrap()
        );
    }
}
