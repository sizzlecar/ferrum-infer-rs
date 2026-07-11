use ferrum_interfaces::vnext::OperationOracleResult;

fn main() {
    let _: OperationOracleResult = serde_json::from_str("{}").unwrap();
}
