use ferrum_runtime::DeviceAllocationReceipt;

fn raw_buffer(receipt: &DeviceAllocationReceipt<'_>) {
    let _ = receipt.buffer();
}

fn main() {}
