use ferrum_interfaces::vnext::UnvalidatedResolvedModelPlan;

fn main() {
    let _: UnvalidatedResolvedModelPlan = serde_json::from_str("{}").unwrap();
}
