//! Immutable identity for a new live structured capture, created before calls.
//! This is not a file digest, phase label, qualified receipt or profile import.
use super::*;
use ferrum_scheduler::implementations::continuous::cost_model::{
    structured::{StructuredUnknown, MODEL_REVISION},
    ExecutionFingerprint,
};
use sha2::{Digest, Sha256};

#[derive(Debug)]
pub(in crate::continuous_engine::inner) struct StructuredCaptureSessionBinding {
    identity: [u8; 32],
    protocol: [u8; 32],
    fingerprint: ExecutionFingerprint,
    opened_at_ns: u64,
}

impl StructuredCaptureSessionBinding {
    pub(in crate::continuous_engine::inner) fn new(
        protocol: [u8; 32],
        fingerprint: ExecutionFingerprint,
        clock: &dyn CostObservationClock,
    ) -> Result<Self, StructuredUnknown> {
        if protocol == [0; 32] {
            return Err(StructuredUnknown::WrongProtocol);
        }
        let opened_at_ns = clock.now_ns().ok_or(StructuredUnknown::Clock)?;
        let mut digest = Sha256::new();
        digest.update(b"ferrum.structured-capture-session.v1\0");
        digest.update(MODEL_REVISION.as_bytes());
        digest.update(uuid::Uuid::new_v4().as_bytes());
        digest.update(protocol);
        digest.update(fingerprint.model_weights);
        digest.update(fingerprint.numerical_policy);
        digest.update(fingerprint.device_runtime);
        digest.update(fingerprint.execution_config);
        digest.update(opened_at_ns.to_le_bytes());
        Ok(Self {
            identity: digest.finalize().into(),
            protocol,
            fingerprint,
            opened_at_ns,
        })
    }

    pub(in crate::continuous_engine::inner) fn identity(&self) -> [u8; 32] {
        self.identity
    }
    pub(in crate::continuous_engine::inner) fn protocol(&self) -> [u8; 32] {
        self.protocol
    }
    pub(in crate::continuous_engine::inner) fn fingerprint(&self) -> &ExecutionFingerprint {
        &self.fingerprint
    }
    pub(in crate::continuous_engine::inner) fn opened_at_ns(&self) -> u64 {
        self.opened_at_ns
    }
}
