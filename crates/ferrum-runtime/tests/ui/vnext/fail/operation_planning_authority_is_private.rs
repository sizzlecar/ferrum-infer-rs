use ferrum_runtime::OperationPlanningHandle;

fn inspect(handle: &OperationPlanningHandle<'_>) {
    let _ = handle.authority();
}

fn main() {}
