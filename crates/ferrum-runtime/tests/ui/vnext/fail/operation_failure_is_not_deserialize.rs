use ferrum_runtime::OperationFailure;

fn main() {
    let _: OperationFailure = serde_json::from_str("{}").unwrap();
}
