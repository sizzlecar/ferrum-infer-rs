use ferrum_interfaces::vnext::StaticProvisioningBinding;

fn main() {
    let _: StaticProvisioningBinding = serde_json::from_str("{}").unwrap();
}
