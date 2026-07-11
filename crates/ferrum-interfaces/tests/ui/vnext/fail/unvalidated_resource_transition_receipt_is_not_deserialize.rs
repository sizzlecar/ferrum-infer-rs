use ferrum_interfaces::vnext::UnvalidatedResourceTransitionReceipt;

fn decode(bytes: &[u8]) {
    let _: UnvalidatedResourceTransitionReceipt = serde_json::from_slice(bytes).unwrap();
}

fn main() {}
