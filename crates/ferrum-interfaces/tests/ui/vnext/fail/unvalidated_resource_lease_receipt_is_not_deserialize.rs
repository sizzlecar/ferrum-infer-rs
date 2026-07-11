use ferrum_interfaces::vnext::UnvalidatedResourceLeaseTransitionReceipt;

fn decode(bytes: &[u8]) {
    let _: UnvalidatedResourceLeaseTransitionReceipt = serde_json::from_slice(bytes).unwrap();
}

fn main() {}
