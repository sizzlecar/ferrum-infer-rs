use ferrum_interfaces::vnext::UnvalidatedOracleTensor;

fn main() {
    let _: UnvalidatedOracleTensor = serde_json::from_str("{}").unwrap();
}
