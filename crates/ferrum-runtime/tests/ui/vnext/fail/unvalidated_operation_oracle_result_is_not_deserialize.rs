use ferrum_runtime::UnvalidatedOperationOracleResult;

fn main() {
    let _: UnvalidatedOperationOracleResult = serde_json::from_str("{}").unwrap();
}
