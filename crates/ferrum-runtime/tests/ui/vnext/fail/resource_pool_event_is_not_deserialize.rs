use ferrum_runtime::ResourcePoolEvent;

fn decode(bytes: &[u8]) {
    let _: ResourcePoolEvent = serde_json::from_slice(bytes).unwrap();
}

fn main() {}
