use ferrum_runtime::TrustedActiveSequenceBinding;

fn main() {
    let _: TrustedActiveSequenceBinding = serde_json::from_str("{}").unwrap();
}
