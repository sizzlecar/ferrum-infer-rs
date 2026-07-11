use ferrum_interfaces::vnext::TrustedAbortedSequenceBinding;

fn decode(bytes: &[u8]) {
    let _: TrustedAbortedSequenceBinding = serde_json::from_slice(bytes).unwrap();
}

fn main() {}
