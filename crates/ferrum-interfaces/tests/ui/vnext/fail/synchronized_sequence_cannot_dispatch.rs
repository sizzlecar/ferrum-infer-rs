use ferrum_interfaces::vnext::*;
use std::sync::Arc;

#[allow(clippy::too_many_arguments)]
fn dispatch_after_synchronize<R: DeviceRuntime>(
    provider: &BoundOperationProvider<'_, R>,
    resolved: &ResolvedModelPlan,
    identity: &ExecutionIdentityEnvelope,
    frame_id: &ExecutionFrameId,
    node_invocation_id: &NodeInvocationId,
    node_id: &NodeId,
    mut synchronized: SynchronizedSequencePermit<'_, '_, R>,
    invocation: InvocationResourceLease<R>,
    lane: &Arc<ExecutionLane<R>>,
    reaper: &Arc<CompletionReaper<R>>,
) {
    let _ = OperationDispatch::encode_and_submit(
        provider,
        resolved,
        identity,
        frame_id,
        node_invocation_id,
        node_id,
        &mut synchronized,
        invocation,
        lane,
        reaper,
    );
}

fn main() {}
