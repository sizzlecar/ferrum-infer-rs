use ferrum_runtime::OperationOracleRequest;

fn main() {
    let _: OperationOracleRequest = serde_json::from_str("{}").unwrap();
}
