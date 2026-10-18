use ferrum_runtime::UnvalidatedOperationOracleDescriptor;

fn main() {
    let _: UnvalidatedOperationOracleDescriptor = serde_json::from_str("{}").unwrap();
}
