use ferrum_interfaces::vnext::ExecutionPlan;

fn main() {
    let _: ExecutionPlan = serde_json::from_str("{}").unwrap();
}
