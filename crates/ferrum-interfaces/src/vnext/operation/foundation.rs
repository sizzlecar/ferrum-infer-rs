use std::io::{self, Write};

use serde::Serialize;
use sha2::{Digest, Sha256};

use super::super::{NodeId, VNextError};

pub(super) fn invalid_operation(reason: impl Into<String>) -> VNextError {
    VNextError::InvalidExecutionPlan {
        reason: reason.into(),
    }
}

pub(super) fn operation_error_for_node(error: VNextError, node_id: &NodeId) -> VNextError {
    match error {
        VNextError::UnsupportedOperation {
            node_id: None,
            operation_id,
            device_id,
            reason,
        } => VNextError::UnsupportedOperation {
            node_id: Some(node_id.to_string()),
            operation_id,
            device_id,
            reason,
        },
        VNextError::IncompatibleOperationVersion {
            node_id: None,
            operation_id,
            required_major,
            required_minor,
            available_major,
            available_minor,
        } => VNextError::IncompatibleOperationVersion {
            node_id: Some(node_id.to_string()),
            operation_id,
            required_major,
            required_minor,
            available_major,
            available_minor,
        },
        error => error,
    }
}

struct OperationFingerprintWriter<'a>(&'a mut Sha256);

impl Write for OperationFingerprintWriter<'_> {
    fn write(&mut self, buffer: &[u8]) -> io::Result<usize> {
        self.0.update(buffer);
        Ok(buffer.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub(super) fn canonical_operation_fingerprint(
    value: &impl Serialize,
    failure_context: &'static str,
) -> Result<String, VNextError> {
    let mut digest = Sha256::new();
    serde_json::to_writer(OperationFingerprintWriter(&mut digest), value)
        .map_err(|error| invalid_operation(format!("{failure_context}: {error}")))?;
    Ok(format!("{:x}", digest.finalize()))
}

pub(super) fn canonical_sha256(value: &str) -> bool {
    value.len() == 64
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}
