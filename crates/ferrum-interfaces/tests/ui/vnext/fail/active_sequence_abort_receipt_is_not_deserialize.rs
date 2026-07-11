use ferrum_interfaces::vnext::ActiveSequenceAbortReceipt;

fn decode(bytes: &[u8]) {
    let _: ActiveSequenceAbortReceipt = serde_json::from_slice(bytes).unwrap();
}

fn main() {}
