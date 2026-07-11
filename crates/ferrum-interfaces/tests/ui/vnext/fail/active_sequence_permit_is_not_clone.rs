use ferrum_interfaces::vnext::{ActiveSequencePermit, DeviceRuntime};

fn duplicate<R: DeviceRuntime>(permit: ActiveSequencePermit<'_, '_, R>) {
    let _duplicate = permit.clone();
}

fn main() {}
