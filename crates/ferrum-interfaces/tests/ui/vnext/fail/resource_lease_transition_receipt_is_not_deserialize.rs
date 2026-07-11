use ferrum_interfaces::vnext::ResourceLeaseTransitionReceipt;

fn main() {
    let _: ResourceLeaseTransitionReceipt = serde_json::from_str("{}").unwrap();
}
