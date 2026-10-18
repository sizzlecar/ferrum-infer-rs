use ferrum_runtime::ReplayIdentity;

fn main() {
    let _: ReplayIdentity = serde_json::from_str("{}").unwrap();
}
