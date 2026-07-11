use ferrum_interfaces::vnext::TrustedActiveSequenceBinding;

fn main() {
    let _: TrustedActiveSequenceBinding = serde_json::from_str("{}").unwrap();
}
