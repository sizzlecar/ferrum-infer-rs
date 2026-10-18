use ferrum_runtime::ResourceLeaseTransitionReceipt;

fn main() {
    let _: ResourceLeaseTransitionReceipt = serde_json::from_str("{}").unwrap();
}
