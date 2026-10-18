use ferrum_runtime::{
    DeviceRuntime, PlanRuntimeResources, TrustedPlanRuntimeBinding,
};
use std::sync::Arc;

fn construct<R: DeviceRuntime>(
    resources: Arc<PlanRuntimeResources<R>>,
) -> TrustedPlanRuntimeBinding<R> {
    TrustedPlanRuntimeBinding { resources }
}

fn main() {}
