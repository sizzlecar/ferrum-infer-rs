use ferrum_runtime::ResolvedModelPlan;

fn main() {
    let _: ResolvedModelPlan = serde_json::from_str("{}").unwrap();
}
