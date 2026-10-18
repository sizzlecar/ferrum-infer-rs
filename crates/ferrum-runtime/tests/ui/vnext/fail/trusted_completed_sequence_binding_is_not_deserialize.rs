use ferrum_runtime::TrustedCompletedSequenceBinding;

fn decode(bytes: &[u8]) {
    let _: TrustedCompletedSequenceBinding = serde_json::from_slice(bytes).unwrap();
}

fn main() {}
