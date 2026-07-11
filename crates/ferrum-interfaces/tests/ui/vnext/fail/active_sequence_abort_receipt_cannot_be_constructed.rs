use ferrum_interfaces::vnext::*;

fn forge(
    plan: TrustedPlanRuntimeEvidence,
    sequence_authority: SequenceAuthorityId,
    run_id: RunId,
    request_id: RequestIdentity,
) {
    let _ = ActiveSequenceAbortReceipt {
        plan,
        sequence_authority,
        run_id,
        request_id,
        activation_epoch: 1,
        runtime_implementation_fingerprint: "0".repeat(64),
        disposition: ActiveSequenceAbortDisposition::SynchronizedAndPoisoned,
    };
}

fn main() {}
