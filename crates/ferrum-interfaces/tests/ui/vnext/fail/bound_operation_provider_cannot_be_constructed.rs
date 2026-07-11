use ferrum_interfaces::vnext::*;

fn forge<R: DeviceRuntime>(
    provider: &'static dyn OperationProvider<R>,
    plan_id: PlanId,
    plan_hash: PlanHash,
    node_id: NodeId,
) {
    let _ = BoundOperationProvider {
        provider,
        plan_id,
        plan_hash,
        node_id,
    };
}

fn main() {}
