use ferrum_interfaces::vnext::*;

fn raw_view<R: DeviceRuntime>(lease: &StaticProvisioningLease<R>, id: &ResourceId, generation: u64) {
    let _ = lease.view(id, generation);
}

fn main() {}
