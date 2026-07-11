use ferrum_interfaces::vnext::UnvalidatedReplayIdentity;

fn decode(bytes: &[u8]) {
    let _: UnvalidatedReplayIdentity = serde_json::from_slice(bytes).unwrap();
}

fn main() {}
