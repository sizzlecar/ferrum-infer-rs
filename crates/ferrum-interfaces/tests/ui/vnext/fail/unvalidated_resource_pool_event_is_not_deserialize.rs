use ferrum_interfaces::vnext::UnvalidatedResourcePoolEvent;

fn decode(bytes: &[u8]) {
    let _: UnvalidatedResourcePoolEvent = serde_json::from_slice(bytes).unwrap();
}

fn main() {}
