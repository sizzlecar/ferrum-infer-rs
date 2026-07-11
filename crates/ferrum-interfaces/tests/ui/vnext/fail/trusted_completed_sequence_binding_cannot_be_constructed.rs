use ferrum_interfaces::vnext::*;

fn forge(
    plan: TrustedPlanRuntimeEvidence,
    coordinator_id: LogicalAdmissionCoordinatorId,
    sequence_authority: SequenceAuthorityId,
    run_id: RunId,
    request_id: RequestIdentity,
) {
    let _ = TrustedCompletedSequenceBinding {
        plan,
        coordinator_id,
        sequence_authority,
        run_id,
        request_id,
        activation_epoch: 1,
        runtime_implementation_fingerprint: "1".repeat(64),
        active_sequence_fingerprint: "2".repeat(64),
        fingerprint: "3".repeat(64),
    };
}

fn main() {}
