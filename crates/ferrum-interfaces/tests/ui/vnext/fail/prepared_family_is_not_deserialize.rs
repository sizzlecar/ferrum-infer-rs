use ferrum_interfaces::vnext::PreparedModelFamily;

fn main() {
    let _: PreparedModelFamily = serde_json::from_str("{}").unwrap();
}
