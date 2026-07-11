use ferrum_interfaces::vnext::OperationFailure;

fn main() {
    let _: OperationFailure = serde_json::from_str("{}").unwrap();
}
