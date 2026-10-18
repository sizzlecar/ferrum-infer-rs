use ferrum_runtime::PreparedModelFamily;

fn main() {
    let _: PreparedModelFamily = serde_json::from_str("{}").unwrap();
}
