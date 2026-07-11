use ferrum_interfaces::vnext::OperationOracleRequest;

fn main() {
    let _: OperationOracleRequest = serde_json::from_str("{}").unwrap();
}
