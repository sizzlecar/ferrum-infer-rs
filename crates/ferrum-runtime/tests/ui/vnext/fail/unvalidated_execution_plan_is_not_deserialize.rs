use ferrum_runtime::UnvalidatedExecutionPlan;

fn decode(bytes: &[u8]) {
    let _: UnvalidatedExecutionPlan = serde_json::from_slice(bytes).unwrap();
}

fn main() {}
