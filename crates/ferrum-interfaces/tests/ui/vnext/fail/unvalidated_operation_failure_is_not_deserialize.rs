use ferrum_interfaces::vnext::UnvalidatedOperationFailure;

fn decode(bytes: &[u8]) {
    let _: UnvalidatedOperationFailure = serde_json::from_slice(bytes).unwrap();
}

fn main() {}
