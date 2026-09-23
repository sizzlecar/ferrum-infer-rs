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
