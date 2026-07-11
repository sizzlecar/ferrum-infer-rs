use ferrum_interfaces::vnext::UnvalidatedOperationOracleRequest;

fn main() {
    let _: UnvalidatedOperationOracleRequest = serde_json::from_str("{}").unwrap();
}
