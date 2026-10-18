use ferrum_runtime::UnvalidatedOperationOracleRequest;

fn main() {
    let _: UnvalidatedOperationOracleRequest = serde_json::from_str("{}").unwrap();
}
