use ferrum_runtime::DeviceAllocationReceipt;

fn stash<'commit>(
    slot: &mut Option<DeviceAllocationReceipt<'static>>,
    receipt: DeviceAllocationReceipt<'commit>,
) {
    *slot = Some(receipt);
}

fn main() {}
