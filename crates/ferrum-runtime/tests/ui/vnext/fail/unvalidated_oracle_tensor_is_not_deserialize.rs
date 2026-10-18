use ferrum_runtime::UnvalidatedOracleTensor;

fn main() {
    let _: UnvalidatedOracleTensor = serde_json::from_str("{}").unwrap();
}
