use ferrum_interfaces::vnext::ResolvedModelPlan;

fn main() {
    let _: ResolvedModelPlan = serde_json::from_str("{}").unwrap();
}
