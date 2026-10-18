use ferrum_runtime::ExecutionPlan;

fn main() {
    let _: ExecutionPlan = serde_json::from_str("{}").unwrap();
}
