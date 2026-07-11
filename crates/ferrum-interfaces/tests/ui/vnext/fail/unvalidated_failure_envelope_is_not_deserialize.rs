use ferrum_interfaces::vnext::UnvalidatedFailureEnvelope;

fn main() {
    let _: UnvalidatedFailureEnvelope = serde_json::from_str("{}").unwrap();
}
