//! CPU-only source declaration shared by packaging and the executable ConvRot
//! providers. Artifact availability and native geometry are checked separately.
use ferrum_runtime::*;
use ferrum_types::{
    NativeOperatorBackend, NativeOperatorContractVersion, NativeOperatorProviderCatalog,
    NativeOperatorProviderCatalogRow, NATIVE_OPERATOR_PROVIDER_CATALOG_SCHEMA_VERSION,
};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;

pub const CAPABILITY_ID: &str = "capability.cuda.convrot.cutlass_int8_fused_output";
pub const OPERATION_IDS: [&str; 4] = [
    COMFY_CONVROT_W4A8_BF16_OPERATION_ID,
    COMFY_CONVROT_W8A8_BF16_OPERATION_ID,
    COMFY_CONVROT_SWIGLU_F32_W4A8_BF16_OPERATION_ID,
    COMFY_CONVROT_SWIGLU_F32_W8A8_BF16_OPERATION_ID,
];

/// Exact geometry decision used before scratch admission. F32 output and
/// unsupported vector/accumulator boundaries keep the existing implementation.
pub(crate) fn native_shape(
    output: ElementType,
    rows: u64,
    outputs: u64,
    inputs: u64,
) -> Option<ferrum_native_ops::cutlass_int8_fused_output::CutlassInt8FusedOutputShapeV1> {
    (output == ElementType::Bf16)
        .then(|| {
            ferrum_native_ops::cutlass_int8_fused_output::CutlassInt8FusedOutputShapeV1::new(
                rows, outputs, inputs,
            )
            .ok()
        })
        .flatten()
}

pub fn source_declaration_fingerprint() -> String {
    digest(&[
        b"ferrum.convrot.cutlass-int8-fused-output.provider-source.v1",
        include_bytes!("cutlass_int8_fused_output.rs"),
        include_bytes!("../backend/cuda/vnext_ops/tensor_silu/comfy_convrot.rs"),
        include_bytes!("../backend/cuda/vnext_ops/tensor_silu/comfy_convrot/launch.rs"),
        include_bytes!("../backend/cuda/vnext_ops/tensor_silu/comfy_convrot/fused_native.rs"),
        include_bytes!("../backend/cuda/vnext_runtime/tensor_eager/convrot_fused_owner.rs"),
        include_bytes!("../../kernels/vnext_comfy_convrot.cuh"),
        include_bytes!("../../native/cutlass-int8-fused-output/abi.h"),
        include_bytes!("../../native/cutlass-int8-fused-output/boundary.cu"),
        include_bytes!("../../native/cutlass-int8-fused-output/source.seed.json"),
        include_bytes!("../../native/cutlass-int8-fused-output/source.build.seed.json"),
        include_bytes!("../../build.rs"),
        include_bytes!("../../../ferrum-native-ops/src/cutlass_int8_fused_output.rs"),
        include_bytes!(
            "../../../ferrum-runtime/src/standard_operations/comfy_convrot_w4a8_bf16.rs"
        ),
        include_bytes!(
            "../../../ferrum-runtime/src/standard_operations/comfy_convrot_w8a8_bf16.rs"
        ),
    ])
}

pub fn provider_descriptor(
    device: &DeviceDescriptor,
    operation_id: &str,
) -> Result<OperationProviderDescriptor, VNextError> {
    let descriptor = descriptor_for(&device.id, operation_id)?;
    if !descriptor.capabilities().is_subset(&device.capabilities) {
        return Err(invalid("ConvRot fused output capability is absent"));
    }
    Ok(descriptor)
}

fn declaration(
    operation_id: &str,
) -> Result<(StandardOperationContract, &'static str, &'static str, u32), VNextError> {
    match operation_id {
        COMFY_CONVROT_W4A8_BF16_OPERATION_ID => Ok((
            comfy_convrot_w4a8_bf16_contract()?,
            "provider.cuda.comfy_convrot_codebook_w4a8.bf16.cutlass_int8_fused_output",
            "resource-estimator.cuda.comfy_convrot_codebook_w4a8.bf16.cutlass_int8_fused_output",
            5,
        )),
        COMFY_CONVROT_W8A8_BF16_OPERATION_ID => Ok((
            comfy_convrot_w8a8_bf16_contract()?,
            "provider.cuda.comfy_convrot_w8a8.bf16.cutlass_int8_fused_output",
            "resource-estimator.cuda.comfy_convrot_w8a8.bf16.cutlass_int8_fused_output",
            3,
        )),
        COMFY_CONVROT_SWIGLU_F32_W4A8_BF16_OPERATION_ID => Ok((
            comfy_convrot_swiglu_f32_w4a8_bf16_contract()?,
            "provider.cuda.comfy_convrot_w4a8.swiglu_f32.bf16.cutlass_int8_fused_output",
            "resource-estimator.cuda.comfy_convrot_w4a8.swiglu_f32.bf16.cutlass_int8_fused_output",
            5,
        )),
        COMFY_CONVROT_SWIGLU_F32_W8A8_BF16_OPERATION_ID => Ok((
            comfy_convrot_swiglu_f32_w8a8_bf16_contract()?,
            "provider.cuda.comfy_convrot_w8a8.swiglu_f32.bf16.cutlass_int8_fused_output",
            "resource-estimator.cuda.comfy_convrot_w8a8.swiglu_f32.bf16.cutlass_int8_fused_output",
            3,
        )),
        _ => Err(invalid(
            "fused output descriptor requires a BF16 ConvRot operation",
        )),
    }
}

fn descriptor_for(
    device: &DeviceId,
    operation_id: &str,
) -> Result<OperationProviderDescriptor, VNextError> {
    let (contract, provider, estimator, inputs) = declaration(operation_id)?;
    let operation_capability = match operation_id {
        COMFY_CONVROT_W4A8_BF16_OPERATION_ID => COMFY_CONVROT_W4A8_BF16_CAPABILITY_ID,
        COMFY_CONVROT_W8A8_BF16_OPERATION_ID => COMFY_CONVROT_W8A8_BF16_CAPABILITY_ID,
        COMFY_CONVROT_SWIGLU_F32_W4A8_BF16_OPERATION_ID => {
            COMFY_CONVROT_SWIGLU_F32_W4A8_BF16_CAPABILITY_ID
        }
        _ => COMFY_CONVROT_SWIGLU_F32_W8A8_BF16_CAPABILITY_ID,
    };
    let implementation = digest(&[
        source_declaration_fingerprint().as_bytes(),
        operation_id.as_bytes(),
    ]);
    let bindings = (0..inputs)
        .map(|n| {
            ProviderStorageBindingRequirement::new(
                ResolvedValueRole::Input,
                n,
                DynamicStorageRequirement::contiguous(),
            )
        })
        .chain(std::iter::once(ProviderStorageBindingRequirement::new(
            ResolvedValueRole::Output,
            0,
            DynamicStorageRequirement::contiguous(),
        )))
        .collect();
    OperationProviderDescriptor::new(
        ProviderId::new(provider)?,
        contract.descriptor().id.clone(),
        contract.descriptor().fingerprint()?,
        implementation.clone(),
        ProviderExecutionSemantics::bitwise_eager_only(),
        ContractVersion::new(1, 0),
        device.clone(),
        BTreeSet::from([
            CapabilityId::new(operation_capability)?,
            CapabilityId::new(CAPABILITY_ID)?,
        ]),
        BTreeSet::from([
            WeightFormatId::new("weight-format.comfy.convrot.canonical.v1")?,
            WeightFormatId::new("weight-format.safetensors.dense")?,
        ]),
        BTreeSet::new(),
        bindings,
        estimator,
        ContractVersion::new(1, 0),
        digest(&[estimator.as_bytes(), implementation.as_bytes()]),
    )
    .map(|d| d.with_fixed_tensor_work(true))
}

pub fn provider_catalog_declaration() -> Result<NativeOperatorProviderCatalog, String> {
    let device = DeviceId::new("device.cuda.source_declaration").map_err(|e| e.to_string())?;
    let mut providers = OPERATION_IDS
        .into_iter()
        .map(|operation| {
            let d = descriptor_for(&device, operation).map_err(|e| e.to_string())?;
            let (contract, ..) = declaration(operation).map_err(|e| e.to_string())?;
            Ok(NativeOperatorProviderCatalogRow {
                operation_id: operation.into(),
                operation_contract_version: NativeOperatorContractVersion::new(
                    contract.descriptor().version.major,
                    contract.descriptor().version.minor,
                ),
                operation_fingerprint: d.operation_fingerprint().into(),
                provider_id: d.provider_id().to_string(),
                provider_version: NativeOperatorContractVersion::new(
                    d.version().major,
                    d.version().minor,
                ),
                provider_implementation_fingerprint: d.provider_implementation_fingerprint().into(),
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    providers
        .sort_by(|a, b| (&a.operation_id, &a.provider_id).cmp(&(&b.operation_id, &b.provider_id)));
    let catalog = NativeOperatorProviderCatalog {
        schema_version: NATIVE_OPERATOR_PROVIDER_CATALOG_SCHEMA_VERSION,
        backend: NativeOperatorBackend::Cuda,
        providers,
    };
    catalog.validate()?;
    Ok(catalog)
}
fn invalid(reason: &str) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
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
    fn native_selection_preserves_f32_and_unsupported_bf16_fallbacks() {
        assert!(native_shape(ElementType::Bf16, 3, 8, 256).is_some());
        assert!(native_shape(ElementType::Bf16, 3, 4, 256).is_none());
        assert!(native_shape(ElementType::Bf16, 3, 8, 131072).is_none());
        assert!(native_shape(ElementType::F32, 3, 8, 256).is_none());
    }

    #[test]
    fn fused_prefix_retains_all_preprocessing_spans_and_removes_only_i32_output() {
        let tensor = |shape: &[u64], dtype| {
            ResolvedTensorSpec::new(shape.to_vec(), dtype, ResolvedTensorLayout::Contiguous)
                .unwrap()
        };
        for (m, k, n) in [
            (1, 256, 8),
            (17, 512, 24),
            (6296, 5376, 28672),
            (19389, 5376, 28672),
        ] {
            assert!(native_shape(ElementType::Bf16, m, n, k).is_some());
            let output = tensor(&[m, n], ElementType::Bf16);
            let x = tensor(&[m, k], ElementType::Bf16);
            let w4 = ComfyConvRotW4A8Geometry::new(
                &[
                    &x,
                    &tensor(&[n, k / 2], ElementType::I8),
                    &tensor(&[n, k / 16], ElementType::U8),
                    &tensor(&[n], ElementType::F32),
                    &tensor(&[16], ElementType::F32),
                ],
                &output,
                k,
                n,
            )
            .unwrap();
            let w8 = ComfyConvRotW8A8Geometry::new(
                &[
                    &x,
                    &tensor(&[n, k], ElementType::I8),
                    &tensor(&[n, 1], ElementType::F32),
                ],
                &output,
                k,
                n,
            )
            .unwrap();
            let a = w4.scratch();
            let b = w8.scratch();
            assert_eq!(
                a.total_bytes - w4.scratch_without_accumulator_bytes(),
                m * n * 4
            );
            assert_eq!(
                b.total_bytes - w8.scratch_without_accumulator_bytes(),
                m * n * 4
            );
            assert!(a.decoded_weight + n * k <= w4.scratch_without_accumulator_bytes());
            assert!(b.row_scales + m * 4 <= w8.scratch_without_accumulator_bytes());
            assert!(a.rotated_activation + m * k * 4 <= a.quantized_activation);
            assert!(a.quantized_activation + m * k <= a.row_scales);
            assert!(a.row_scales + m * 4 <= a.decoded_weight);
            assert_eq!(a.rotated_activation, b.rotated_activation);
            assert_eq!(a.quantized_activation, b.quantized_activation);
            assert_eq!(a.row_scales, b.row_scales);
        }
    }

    #[test]
    fn catalog_keeps_existing_contracts_and_distinct_implementations() {
        let catalog = provider_catalog_declaration().unwrap();
        for operation in OPERATION_IDS {
            let row = catalog
                .providers
                .iter()
                .find(|row| row.operation_id == operation)
                .unwrap();
            let d =
                descriptor_for(&DeviceId::new("device.cuda.fixture").unwrap(), operation).unwrap();
            assert_eq!(
                row.operation_fingerprint,
                declaration(operation)
                    .unwrap()
                    .0
                    .descriptor()
                    .fingerprint()
                    .unwrap()
            );
            assert_eq!(
                row.provider_implementation_fingerprint,
                d.provider_implementation_fingerprint()
            );
            assert!(d.supports_fixed_tensor_work());
        }
        assert!(descriptor_for(
            &DeviceId::new("device.cuda.fixture").unwrap(),
            COMFY_CONVROT_W4A8_F32_OPERATION_ID
        )
        .is_err());
    }
}
