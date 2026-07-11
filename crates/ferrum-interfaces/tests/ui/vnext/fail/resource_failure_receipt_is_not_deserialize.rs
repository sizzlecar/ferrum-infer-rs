use ferrum_interfaces::vnext::ResourceFailureReceipt;

fn main() {
    let _: ResourceFailureReceipt = serde_json::from_str("{}").unwrap();
}
