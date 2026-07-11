use ferrum_interfaces::vnext::DeviceAllocationReceipt;

fn raw_buffer(receipt: &DeviceAllocationReceipt<'_>) {
    let _ = receipt.buffer();
}

fn main() {}
