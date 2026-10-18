use ferrum_runtime::ResourceFailureReceipt;

fn main() {
    let _: ResourceFailureReceipt = serde_json::from_str("{}").unwrap();
}
