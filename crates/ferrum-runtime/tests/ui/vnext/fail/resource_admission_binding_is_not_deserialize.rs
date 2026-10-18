use ferrum_runtime::StaticProvisioningBinding;

fn main() {
    let _: StaticProvisioningBinding = serde_json::from_str("{}").unwrap();
}
