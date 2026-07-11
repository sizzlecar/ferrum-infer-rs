use ferrum_interfaces::vnext::OperationPlanningHandle;

fn inspect(handle: &OperationPlanningHandle<'_>) {
    let _ = handle.authority();
}

fn main() {}
