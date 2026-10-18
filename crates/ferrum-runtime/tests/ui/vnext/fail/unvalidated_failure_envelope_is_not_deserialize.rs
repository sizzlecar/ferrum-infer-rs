use ferrum_runtime::UnvalidatedFailureEnvelope;

fn main() {
    let _: UnvalidatedFailureEnvelope = serde_json::from_str("{}").unwrap();
}
