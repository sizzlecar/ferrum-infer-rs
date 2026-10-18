use ferrum_runtime::UnvalidatedExecutionEvent;

fn decode(bytes: &[u8]) {
    let _: UnvalidatedExecutionEvent = serde_json::from_slice(bytes).unwrap();
}

fn main() {}
