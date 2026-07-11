use ferrum_interfaces::vnext::*;

fn forge(registry: &dyn OperationPlanningRegistry) {
    let _ = OperationPlanningHandle { registry };
}

fn main() {}
