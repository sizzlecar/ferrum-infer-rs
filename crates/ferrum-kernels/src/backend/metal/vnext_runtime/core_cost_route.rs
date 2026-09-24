//! Pure metadata shared by the actual core transfer encoder and its future
//! declaration. The encoder still validates and retains the real buffers.
use super::*;
use ferrum_interfaces::execution_cost::CoreReadbackRoute;
use ferrum_interfaces::vnext::DeviceCoreCostCapabilities;

pub(super) fn capabilities() -> DeviceCoreCostCapabilities {
    DeviceCoreCostCapabilities {
        upload_native_operation: HOST_UPLOAD_NATIVE_OPERATION_ID.as_str(),
        zero_native_operation: DEVICE_ZERO_NATIVE_OPERATION_ID.as_str(),
        single_transfer_commands: true,
        preserves_program_bindings: true,
        staged_host_readback_without_commands: true,
        staged_host_readback_native_operation: None,
        fallback_readback: CoreReadbackRoute::HostSynchronized,
    }
}

/// These are the real encode_upload/copy/zero blit paths: contiguous Shared
/// buffers, one blit command, no conversion or compute dispatch. Byte counts
/// are work coordinates, never algorithm-family identity.
pub(super) fn transfer_evidence(
    kind: ferrum_interfaces::execution_cost::StatisticalTransferKindV1,
    bytes: u64,
    tokens: u64,
) -> Option<ferrum_interfaces::execution_cost::SelectedCommandCostEvidenceV1> {
    use ferrum_interfaces::execution_cost::{
        SelectedAlgorithmClassV1, SelectedCommandCostBuilderV1, StatisticalTransferKindV1 as K,
    };
    use sha2::{Digest, Sha256};
    use std::sync::OnceLock;
    let (entry, abi) = match kind {
        K::HostToDevice => (
            "MTLBlit.copy.shared_staging_to_shared_buffer",
            b"one_contiguous_shared_upload".as_slice(),
        ),
        K::DeviceToDevice => (
            "MTLBlit.copy.shared_buffer_to_shared_buffer",
            b"one_contiguous_shared_copy".as_slice(),
        ),
        K::Fill => (
            "MTLBlit.fill.shared_buffer.zero",
            b"one_contiguous_shared_zero".as_slice(),
        ),
        K::DeviceToHost => return None, // shared host staging emits no device command
    };
    static NUMERICAL: OnceLock<[u8; 32]> = OnceLock::new();
    let numerical = *NUMERICAL.get_or_init(|| Sha256::digest(b"metal.blit.byte_exact.v1").into());
    let layout = Sha256::digest(abi).into();
    let class = SelectedAlgorithmClassV1::new(entry, 1, numerical, layout).ok()?;
    let mut builder = SelectedCommandCostBuilderV1::new(tokens);
    builder.transfer(class, kind, bytes).ok()?;
    builder.finish().ok()
}
