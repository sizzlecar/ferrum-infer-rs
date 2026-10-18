use ferrum_runtime::TrustedAbortedSequenceBinding;

fn decode(bytes: &[u8]) {
    let _: TrustedAbortedSequenceBinding = serde_json::from_slice(bytes).unwrap();
}

fn main() {}
