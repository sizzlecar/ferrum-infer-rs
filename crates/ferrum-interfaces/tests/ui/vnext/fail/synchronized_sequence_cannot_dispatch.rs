use ferrum_interfaces::vnext::*;
use std::sync::Arc;

#[allow(clippy::too_many_arguments)]
fn dispatch_after_synchronize<R: DeviceRuntime>(
    provider: &BoundOperationProvider<'_, R>,
    resolved: &ResolvedModelPlan,
    batch_identity: &BatchOperationIdentity,
    synchronized: SynchronizedSequencePermit<'_, '_, R>,
    invocation: InvocationResourceLease<R>,
    lane: &Arc<ExecutionLane<R>>,
    reaper: &Arc<CompletionReaper<R>>,
) {
    let _ = OperationDispatch::encode_and_submit(
        provider,
        resolved,
        batch_identity,
        std::slice::from_ref(&synchronized),
        invocation,
        lane,
        reaper,
    );
}

fn main() {}
