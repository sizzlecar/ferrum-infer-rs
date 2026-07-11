use ferrum_interfaces::vnext::ActiveSequenceCompletionReceipt;

fn decode(bytes: &[u8]) {
    let _: ActiveSequenceCompletionReceipt = serde_json::from_slice(bytes).unwrap();
}

fn main() {}
