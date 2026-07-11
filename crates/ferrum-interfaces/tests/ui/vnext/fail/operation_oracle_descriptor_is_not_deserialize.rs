use ferrum_interfaces::vnext::OperationOracleDescriptor;

fn main() {
    let _: OperationOracleDescriptor = serde_json::from_str("{}").unwrap();
}
