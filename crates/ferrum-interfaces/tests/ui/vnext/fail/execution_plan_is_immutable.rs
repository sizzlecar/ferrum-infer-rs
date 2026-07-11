use ferrum_interfaces::vnext::ExecutionPlan;

fn mutate(plan: &ExecutionPlan) {
    plan.payload().schema = ferrum_interfaces::vnext::PlanSchemaVersion::new(2, 0);
}

fn main() {}
