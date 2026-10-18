use ferrum_runtime::ProviderResourcePlan;

fn main() {
    let _: ProviderResourcePlan = serde_json::from_str("{}").unwrap();
}
