use ferrum_interfaces::vnext::ProviderResourcePlan;

fn main() {
    let _: ProviderResourcePlan = serde_json::from_str("{}").unwrap();
}
