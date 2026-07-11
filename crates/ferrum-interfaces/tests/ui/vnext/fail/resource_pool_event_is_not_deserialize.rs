use ferrum_interfaces::vnext::ResourcePoolEvent;

fn decode(bytes: &[u8]) {
    let _: ResourcePoolEvent = serde_json::from_slice(bytes).unwrap();
}

fn main() {}
