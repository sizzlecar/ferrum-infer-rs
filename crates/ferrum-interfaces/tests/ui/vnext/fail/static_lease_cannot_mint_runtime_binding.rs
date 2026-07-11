use ferrum_interfaces::vnext::{DeviceRuntime, StaticProvisioningLease};

fn mint<R: DeviceRuntime>(lease: &StaticProvisioningLease<R>) {
    let _binding = lease.trusted_runtime_binding();
}

fn main() {}
