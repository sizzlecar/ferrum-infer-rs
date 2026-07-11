use ferrum_interfaces::vnext::UnvalidatedOperationOracleDescriptor;

fn main() {
    let _: UnvalidatedOperationOracleDescriptor = serde_json::from_str("{}").unwrap();
}
