use ferrum_runtime::{DeviceRuntime, PlanRuntimeResources};

fn inspect<R: DeviceRuntime>(resources: PlanRuntimeResources<R>) {
    let PlanRuntimeResources { phase, .. } = resources;
    let _ = phase;
}

fn main() {}
