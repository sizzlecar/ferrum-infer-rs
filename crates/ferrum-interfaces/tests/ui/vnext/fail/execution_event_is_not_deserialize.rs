use ferrum_interfaces::vnext::ExecutionEvent;

fn main() {
    let _: ExecutionEvent = serde_json::from_str("{}").unwrap();
}
