use ferrum_runtime::ExecutionPlan;

fn mutate(plan: &ExecutionPlan) {
    plan.payload().schema = ferrum_runtime::PlanSchemaVersion::new(2, 0);
}

fn main() {}
