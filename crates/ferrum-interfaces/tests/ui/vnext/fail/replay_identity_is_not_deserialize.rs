use ferrum_interfaces::vnext::ReplayIdentity;

fn main() {
    let _: ReplayIdentity = serde_json::from_str("{}").unwrap();
}
