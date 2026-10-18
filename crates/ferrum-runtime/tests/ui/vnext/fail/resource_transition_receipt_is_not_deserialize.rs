use ferrum_runtime::ResourceTransitionReceipt;

fn main() {
    let _: ResourceTransitionReceipt = serde_json::from_str("{}").unwrap();
}
