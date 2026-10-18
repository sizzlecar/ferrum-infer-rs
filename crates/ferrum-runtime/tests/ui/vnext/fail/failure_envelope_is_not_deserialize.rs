use ferrum_runtime::FailureEnvelope;

fn main() {
    let _: FailureEnvelope = serde_json::from_str("{}").unwrap();
}
