use ferrum_runtime::UnvalidatedPreparedModelFamily;

fn main() {
    let _: UnvalidatedPreparedModelFamily = serde_json::from_str("{}").unwrap();
}
