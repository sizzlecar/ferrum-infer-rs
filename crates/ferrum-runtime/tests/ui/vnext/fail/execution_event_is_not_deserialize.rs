use ferrum_runtime::ExecutionEvent;

fn main() {
    let _: ExecutionEvent = serde_json::from_str("{}").unwrap();
}
