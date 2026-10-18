use ferrum_runtime::UnvalidatedResourcePoolEvent;

fn decode(bytes: &[u8]) {
    let _: UnvalidatedResourcePoolEvent = serde_json::from_slice(bytes).unwrap();
}

fn main() {}
