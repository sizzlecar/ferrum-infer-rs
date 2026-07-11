use ferrum_interfaces::vnext::UnvalidatedOperationOracleResult;

fn main() {
    let _: UnvalidatedOperationOracleResult = serde_json::from_str("{}").unwrap();
}
