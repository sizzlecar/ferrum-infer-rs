use ferrum_interfaces::vnext::FailureEnvelope;

fn main() {
    let _: FailureEnvelope = serde_json::from_str("{}").unwrap();
}
