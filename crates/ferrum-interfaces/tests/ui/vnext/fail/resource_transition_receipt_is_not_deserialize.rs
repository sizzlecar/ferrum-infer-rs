use ferrum_interfaces::vnext::ResourceTransitionReceipt;

fn main() {
    let _: ResourceTransitionReceipt = serde_json::from_str("{}").unwrap();
}
