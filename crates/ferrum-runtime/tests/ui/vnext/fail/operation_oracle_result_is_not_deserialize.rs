use ferrum_runtime::OperationOracleResult;

fn main() {
    let _: OperationOracleResult = serde_json::from_str("{}").unwrap();
}
