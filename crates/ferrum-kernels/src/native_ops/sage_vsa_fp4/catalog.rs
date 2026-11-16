//! Source declaration only; payload binding and native qualification are separate.
use ferrum_runtime::{
    sparse_fp4_attention_checked_host_contract, sparse_fp4_attention_region_topk_contract,
    sparse_fp4_qkv_quantize_bf16_contract, OperationContract,
};
use ferrum_types::{
    NativeOperatorBackend, NativeOperatorContractVersion, NativeOperatorProviderCatalog,
    NativeOperatorProviderCatalogRow, NATIVE_OPERATOR_PROVIDER_CATALOG_SCHEMA_VERSION,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
pub const CAPABILITY_ID: &str = "capability.cuda.sage_vsa_fp4.checked_host.sm120";
pub const PROVIDER_ID: &str = "provider.cuda.sparse_fp4_attention.checked_host.bf16.sage.sm120";
pub const REGION_CAPABILITY_ID: &str = "capability.cuda.sage_vsa_fp4.region_topk.sm120";
pub const REGION_PROVIDER_ID: &str =
    "provider.cuda.sparse_fp4_attention.region_topk.bf16.sage.sm120";
pub const QUANTIZE_CAPABILITY_ID: &str = "capability.cuda.sage_vsa_fp4.quantize.sm120";
pub const QUANTIZE_PROVIDER_ID: &str = "provider.cuda.sparse_fp4_qkv_quantize.bf16.sage.sm120";

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SageVsaFp4Config {
    pub opaque_memory_bytes: u64,
    pub opaque_memory_profile: String,
}
impl SageVsaFp4Config {
    pub fn validate(&self) -> Result<(), String> {
        if self.opaque_memory_bytes == 0
            || self.opaque_memory_profile.trim().is_empty()
            || self.opaque_memory_profile.len() > 256
            || !self.opaque_memory_profile.is_ascii()
            || self
                .opaque_memory_profile
                .bytes()
                .any(|b| b.is_ascii_control())
        {
            return Err(
                "VSA requires positive opaque allowance and bounded ASCII memory profile".into(),
            );
        }
        Ok(())
    }
}
pub fn source_declaration_fingerprint() -> String {
    digest(&[
        b"ferrum.sage-vsa-fp4.checked-host.source.v1",
        super::SAGE_VSA_FP4_NUMERICAL_IDENTITY.as_bytes(),
        super::regions::REGION_SELECTION_NUMERICAL_IDENTITY.as_bytes(),
        super::quantize::NUMERICAL_IDENTITY.as_bytes(),
        include_bytes!("catalog.rs"),
        include_bytes!("host_pattern.rs"),
        include_bytes!("regions.rs"),
        include_bytes!("regions/kernel_upload.rs"),
        include_bytes!("quantize.rs"),
        include_bytes!("../sage_vsa_fp4.rs"),
        include_bytes!("../../backend/cuda/vnext_ops/tensor_sparse_fp4_attention.rs"),
        include_bytes!("../../backend/cuda/vnext_ops/tensor_sparse_fp4_attention/profile.rs"),
        include_bytes!("../../backend/cuda/vnext_ops/tensor_sparse_fp4_attention/module.rs"),
        include_bytes!("../../backend/cuda/vnext_ops/tensor_sparse_fp4_quantize.rs"),
        include_bytes!("layout.rs"),
        include_bytes!("../../backend/cuda/vnext_ops/tensor_storage.rs"),
        include_bytes!("../../backend/cuda/vnext_runtime/pinned_host.rs"),
        include_bytes!("../../backend/cuda/vnext_runtime/tensor_eager/native_module_owner.rs"),
        include_bytes!("../../backend/cuda/vnext_runtime/tensor_eager/native_module_capacity.rs"),
        include_bytes!("../../../native/sage-vsa-fp4/abi.h"),
        include_bytes!("../../../native/sage-vsa-fp4/boundary.cu"),
        include_bytes!("../../../native/sage-vsa-fp4/boundary_checks.cuh"),
        include_bytes!("../../../native/sage-vsa-fp4/region_abi.h"),
        include_bytes!("../../../native/sage-vsa-fp4/region_boundary.cuh"),
        include_bytes!("../../../native/sage-vsa-fp4/region_kernel_boundary.cuh"),
        include_bytes!("../../../native/sage-vsa-fp4/region_select.cuh"),
        include_bytes!("../../../native/sage-vsa-fp4/quantize_abi.h"),
        include_bytes!("../../../native/sage-vsa-fp4/quantize_core.cuh"),
        include_bytes!("../../../native/sage-vsa-fp4/quantize_boundary.cuh"),
        include_bytes!("../../../native/sage-vsa-fp4/launch.cuh"),
        include_bytes!("../../../native/sage-vsa-fp4/params.h"),
        include_bytes!("../../../native/sage-vsa-fp4/source.seed.json"),
        include_bytes!("../../../../ferrum-native-ops/src/sage_vsa_fp4.rs"),
        include_bytes!("../../../../ferrum-native-ops/src/sage_vsa_fp4/lists.rs"),
        include_bytes!("../../../../ferrum-native-ops/src/sage_vsa_fp4/selection.rs"),
        include_bytes!("../../../../ferrum-native-ops/src/sage_vsa_fp4/selection/kernel_upload.rs"),
        include_bytes!("../../../../ferrum-native-ops/src/sage_vsa_fp4/quantize.rs"),
        include_bytes!(
            "../../../../ferrum-runtime/src/standard_operations/sparse_fp4_attention.rs"
        ),
        include_bytes!(
            "../../../../ferrum-runtime/src/standard_operations/sparse_fp4_qkv_quantize.rs"
        ),
    ])
}
pub fn provider_catalog_declaration(
    config: &SageVsaFp4Config,
) -> Result<NativeOperatorProviderCatalog, String> {
    config.validate()?;
    let contracts = [
        (
            sparse_fp4_attention_checked_host_contract().map_err(|e| e.to_string())?,
            PROVIDER_ID,
        ),
        (
            sparse_fp4_attention_region_topk_contract().map_err(|e| e.to_string())?,
            REGION_PROVIDER_ID,
        ),
        (
            sparse_fp4_qkv_quantize_bf16_contract().map_err(|e| e.to_string())?,
            QUANTIZE_PROVIDER_ID,
        ),
    ];
    let result = NativeOperatorProviderCatalog {
        schema_version: NATIVE_OPERATOR_PROVIDER_CATALOG_SCHEMA_VERSION,
        backend: NativeOperatorBackend::Cuda,
        providers: contracts
            .iter()
            .map(|(contract, provider)| {
                let d = contract.descriptor();
                Ok(NativeOperatorProviderCatalogRow {
                    operation_id: d.id.to_string(),
                    operation_contract_version: NativeOperatorContractVersion::new(
                        d.version.major,
                        d.version.minor,
                    ),
                    operation_fingerprint: d.fingerprint().map_err(|e| e.to_string())?,
                    provider_id: (*provider).into(),
                    provider_version: NativeOperatorContractVersion::new(1, 0),
                    provider_implementation_fingerprint: digest(&[
                        source_declaration_fingerprint().as_bytes(),
                        &serde_json::to_vec(config).map_err(|e| e.to_string())?,
                    ]),
                })
            })
            .collect::<Result<Vec<_>, String>>()?,
    };
    result.validate()?;
    Ok(result)
}
fn digest(parts: &[&[u8]]) -> String {
    let mut hash = Sha256::new();
    for part in parts {
        hash.update((part.len() as u64).to_le_bytes());
        hash.update(part);
    }
    format!("{:x}", hash.finalize())
}
