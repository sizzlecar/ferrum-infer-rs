use ferrum_runtime::UnvalidatedResolvedModelPlan;

fn main() {
    let _: UnvalidatedResolvedModelPlan = serde_json::from_str("{}").unwrap();
}
