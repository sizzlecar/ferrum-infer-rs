use ferrum_interfaces::vnext::UnvalidatedPreparedModelFamily;

fn main() {
    let _: UnvalidatedPreparedModelFamily = serde_json::from_str("{}").unwrap();
}
