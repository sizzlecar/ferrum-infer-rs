use ferrum_runtime::OperationOracleDescriptor;

fn main() {
    let _: OperationOracleDescriptor = serde_json::from_str("{}").unwrap();
}
