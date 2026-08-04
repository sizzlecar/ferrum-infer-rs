use serde::Serialize;

use crate::vnext::{
    BufferUsage, DeviceCommandPhase, DeviceComputePathRequirement, DeviceExecutionPath,
    ElementType, ExecutionDeterminismWitnessKind, TensorAccess, VNextError,
};

use super::foundation::invalid_operation;
use super::SubmissionWaveDeterminismEvidence;

const MAX_ARTIFACT_EXECUTION_ID_BYTES: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SubmissionWaveDeterminismArtifactInitializationIdentity {
    input_sha256: String,
    rng_sha256: String,
    initial_state_sha256: String,
}

impl SubmissionWaveDeterminismArtifactInitializationIdentity {
    pub fn input_sha256(&self) -> &str {
        &self.input_sha256
    }

    pub fn rng_sha256(&self) -> &str {
        &self.rng_sha256
    }

    pub fn initial_state_sha256(&self) -> &str {
        &self.initial_state_sha256
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SubmissionWaveDeterminismArtifactPhysicalCommand {
    command_index: u32,
    node_id: Option<String>,
    command_phase: String,
    native_op_id: String,
    execution_path: String,
    batching_form: String,
    participant_count: u32,
    token_count: u64,
    compute_dispatch_count: u64,
    transfer_command_count: u64,
    reusable_graph_node_count: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SubmissionWaveDeterminismArtifactLogicalCommand {
    logical_command_ordinal: u32,
    node_id: String,
    native_op_id: String,
    batching_form: String,
    participant_count: u32,
    token_count: u64,
    compute_dispatch_count: u64,
    transfer_command_count: u64,
    reusable_graph_node_count: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SubmissionWaveDeterminismArtifactReplayedSegment {
    physical_command_index: u32,
    reusable_program_fingerprint: String,
    reusable_executable_fingerprint: String,
    logical_commands: Vec<SubmissionWaveDeterminismArtifactLogicalCommand>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SubmissionWaveDeterminismArtifactAttribution {
    batch_identity_fingerprint: String,
    submission_fingerprint: String,
    physical_commands: Vec<SubmissionWaveDeterminismArtifactPhysicalCommand>,
    replayed_segments: Vec<SubmissionWaveDeterminismArtifactReplayedSegment>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SubmissionWaveDeterminismArtifactWitness {
    kind: String,
    semantic_id: String,
    node_id: String,
    resource_id: String,
    access: String,
    participant_index: u32,
    logical_offset_bytes: u64,
    length_bytes: u64,
    element_type: String,
    raw_sha256: String,
}

impl SubmissionWaveDeterminismArtifactWitness {
    pub fn kind(&self) -> &str {
        &self.kind
    }

    pub fn semantic_id(&self) -> &str {
        &self.semantic_id
    }

    pub fn node_id(&self) -> &str {
        &self.node_id
    }

    pub fn resource_id(&self) -> &str {
        &self.resource_id
    }

    pub fn access(&self) -> &str {
        &self.access
    }

    pub const fn participant_index(&self) -> u32 {
        self.participant_index
    }

    pub const fn logical_offset_bytes(&self) -> u64 {
        self.logical_offset_bytes
    }

    pub const fn length_bytes(&self) -> u64 {
        self.length_bytes
    }

    pub fn element_type(&self) -> &str {
        &self.element_type
    }

    pub fn raw_sha256(&self) -> &str {
        &self.raw_sha256
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SubmissionWaveDeterminismArtifactExecution {
    execution_id: String,
    mode: String,
    compute_path_requirement: String,
    reusable_program_fingerprint: Option<String>,
    declared_eager_boundary_node_ids: Vec<String>,
    restore_sha256: String,
    initialization_identity: SubmissionWaveDeterminismArtifactInitializationIdentity,
    submission_fingerprint: String,
    receipt_fingerprint: String,
    attribution: SubmissionWaveDeterminismArtifactAttribution,
    witnesses: Vec<SubmissionWaveDeterminismArtifactWitness>,
}

impl SubmissionWaveDeterminismArtifactExecution {
    pub fn execution_id(&self) -> &str {
        &self.execution_id
    }

    pub fn mode(&self) -> &str {
        &self.mode
    }

    pub fn compute_path_requirement(&self) -> &str {
        &self.compute_path_requirement
    }

    pub fn reusable_program_fingerprint(&self) -> Option<&str> {
        self.reusable_program_fingerprint.as_deref()
    }

    pub fn declared_eager_boundary_node_ids(&self) -> &[String] {
        &self.declared_eager_boundary_node_ids
    }

    pub fn restore_sha256(&self) -> &str {
        &self.restore_sha256
    }

    pub const fn initialization_identity(
        &self,
    ) -> &SubmissionWaveDeterminismArtifactInitializationIdentity {
        &self.initialization_identity
    }

    pub fn witnesses(&self) -> &[SubmissionWaveDeterminismArtifactWitness] {
        &self.witnesses
    }

    pub fn replayed_segments(&self) -> &[SubmissionWaveDeterminismArtifactReplayedSegment] {
        &self.attribution.replayed_segments
    }
}

impl SubmissionWaveDeterminismEvidence {
    pub fn into_artifact_execution(
        self,
        execution_id: impl Into<String>,
    ) -> Result<SubmissionWaveDeterminismArtifactExecution, VNextError> {
        let execution_id = execution_id.into();
        if execution_id.is_empty()
            || execution_id.len() > MAX_ARTIFACT_EXECUTION_ID_BYTES
            || !execution_id.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b':' | b'/' | b'-')
            })
        {
            return Err(invalid_operation(
                "determinism artifact execution id is empty, too long, or non-portable",
            ));
        }

        let attribution = self.attribution();
        let batch_identity = attribution.batch_identity();
        let physical_commands = attribution
            .device()
            .commands()
            .iter()
            .map(|command| {
                let node_id = command
                    .node_index()
                    .map(|index| {
                        usize::try_from(index)
                            .ok()
                            .and_then(|index| batch_identity.node_id_at(index))
                            .map(ToString::to_string)
                            .ok_or_else(|| {
                                invalid_operation(
                                    "determinism physical command has an unknown plan node index",
                                )
                            })
                    })
                    .transpose()?;
                Ok(SubmissionWaveDeterminismArtifactPhysicalCommand {
                    command_index: command.command_index(),
                    node_id,
                    command_phase: command_phase_label(command.command_phase()).to_owned(),
                    native_op_id: command.native_op_id().to_owned(),
                    execution_path: command.execution_path().as_str().to_owned(),
                    batching_form: command.batching_form().as_str().to_owned(),
                    participant_count: command.participant_count(),
                    token_count: command.token_count(),
                    compute_dispatch_count: command.compute_dispatch_count(),
                    transfer_command_count: command.transfer_command_count(),
                    reusable_graph_node_count: command.reusable_graph_node_count(),
                })
            })
            .collect::<Result<Vec<_>, VNextError>>()?;
        let replayed_segments = attribution
            .device()
            .replayed_segments()
            .iter()
            .map(|segment| {
                let logical_commands = segment
                    .logical_commands()
                    .iter()
                    .map(|command| {
                        let node_index = usize::try_from(command.node_index()).map_err(|_| {
                            invalid_operation("determinism replay logical node index exceeds usize")
                        })?;
                        let node_id = batch_identity.node_id_at(node_index).ok_or_else(|| {
                            invalid_operation(
                                "determinism replay logical command has an unknown plan node index",
                            )
                        })?;
                        Ok(SubmissionWaveDeterminismArtifactLogicalCommand {
                            logical_command_ordinal: command.logical_command_ordinal(),
                            node_id: node_id.to_string(),
                            native_op_id: command.native_op_id().to_owned(),
                            batching_form: command.batching_form().as_str().to_owned(),
                            participant_count: command.participant_count(),
                            token_count: command.token_count(),
                            compute_dispatch_count: command.compute_dispatch_count(),
                            transfer_command_count: command.transfer_command_count(),
                            reusable_graph_node_count: command.reusable_graph_node_count(),
                        })
                    })
                    .collect::<Result<Vec<_>, VNextError>>()?;
                Ok(SubmissionWaveDeterminismArtifactReplayedSegment {
                    physical_command_index: segment.physical_command_index(),
                    reusable_program_fingerprint: segment.program_id().fingerprint(),
                    reusable_executable_fingerprint: segment
                        .reusable_executable_fingerprint()
                        .to_owned(),
                    logical_commands,
                })
            })
            .collect::<Result<Vec<_>, VNextError>>()?;

        let mut witnesses = self
            .witnesses()
            .iter()
            .map(|witness| {
                let physical_index =
                    usize::try_from(witness.physical_readback_index()).map_err(|_| {
                        invalid_operation(
                            "determinism witness physical readback index exceeds usize",
                        )
                    })?;
                let physical = self
                    .physical_readbacks()
                    .get(physical_index)
                    .ok_or_else(|| {
                        invalid_operation(
                            "determinism witness references an absent physical readback",
                        )
                    })?;
                let request = physical.request();
                let spec = witness.witness();
                if request.node_id() != spec.node_id()
                    || request.resource_id() != spec.resource_id()
                    || request.participant_index() != witness.participant_index()
                    || request.expected_usage() != spec.location().usage()
                {
                    return Err(invalid_operation(
                        "determinism witness differs from its exact physical readback request",
                    ));
                }
                let (kind, semantic_id, access) = match spec.kind() {
                    ExecutionDeterminismWitnessKind::Output { value_id, .. } => {
                        ("declared_output", value_id.to_string(), TensorAccess::Write)
                    }
                    ExecutionDeterminismWitnessKind::StateEffect {
                        state_id, access, ..
                    } => ("state_effect", state_id.to_string(), *access),
                };
                if kind == "state_effect" && request.expected_usage() != BufferUsage::State {
                    return Err(invalid_operation(
                        "determinism state witness readback does not use state backing",
                    ));
                }
                Ok(SubmissionWaveDeterminismArtifactWitness {
                    kind: kind.to_owned(),
                    semantic_id,
                    node_id: spec.node_id().to_string(),
                    resource_id: spec.resource_id().to_string(),
                    access: tensor_access_label(access).to_owned(),
                    participant_index: witness.participant_index(),
                    logical_offset_bytes: request.logical_offset_bytes(),
                    length_bytes: request.output_layout().byte_len()?,
                    element_type: element_type_label(spec.element_type()).to_owned(),
                    raw_sha256: physical.raw_sha256().to_owned(),
                })
            })
            .collect::<Result<Vec<_>, VNextError>>()?;
        witnesses.sort_by(|left, right| {
            (
                &left.kind,
                &left.semantic_id,
                &left.node_id,
                &left.resource_id,
                &left.access,
                left.participant_index,
                left.logical_offset_bytes,
                left.length_bytes,
                &left.element_type,
            )
                .cmp(&(
                    &right.kind,
                    &right.semantic_id,
                    &right.node_id,
                    &right.resource_id,
                    &right.access,
                    right.participant_index,
                    right.logical_offset_bytes,
                    right.length_bytes,
                    &right.element_type,
                ))
        });

        let initialization = self.initialization_identity();
        let reusable_program_fingerprint = self.reusable_program_fingerprint();
        let declared_eager_boundary_node_ids = self
            .declared_eager_boundary_node_ids()
            .iter()
            .map(ToString::to_string)
            .collect();
        Ok(SubmissionWaveDeterminismArtifactExecution {
            execution_id,
            mode: match self.expected_execution_path() {
                DeviceExecutionPath::Eager => "eager",
                DeviceExecutionPath::Replayed => "replay",
            }
            .to_owned(),
            compute_path_requirement: compute_path_requirement_label(
                self.expected_compute_path_requirement(),
            )
            .to_owned(),
            reusable_program_fingerprint,
            declared_eager_boundary_node_ids,
            restore_sha256: self.restore_fingerprint().to_owned(),
            initialization_identity: SubmissionWaveDeterminismArtifactInitializationIdentity {
                input_sha256: initialization.input_sha256().to_owned(),
                rng_sha256: initialization.rng_sha256().to_owned(),
                initial_state_sha256: initialization.initial_state_sha256().to_owned(),
            },
            submission_fingerprint: self.submission_receipt_fingerprint().to_owned(),
            receipt_fingerprint: self.terminal_receipt_fingerprint().to_owned(),
            attribution: SubmissionWaveDeterminismArtifactAttribution {
                batch_identity_fingerprint: batch_identity.fingerprint().to_owned(),
                submission_fingerprint: attribution.submission_fingerprint().to_owned(),
                physical_commands,
                replayed_segments,
            },
            witnesses,
        })
    }
}

const fn compute_path_requirement_label(value: DeviceComputePathRequirement) -> &'static str {
    match value {
        DeviceComputePathRequirement::Adaptive => "adaptive",
        DeviceComputePathRequirement::EagerOnly => "eager_only",
        DeviceComputePathRequirement::ReplayedOnly => "replayed_only",
        DeviceComputePathRequirement::ReplayedWithDeclaredEagerBoundaries => {
            "replayed_with_declared_eager_boundaries"
        }
    }
}

const fn command_phase_label(value: DeviceCommandPhase) -> &'static str {
    match value {
        DeviceCommandPhase::Initialization => "initialization",
        DeviceCommandPhase::DynamicBinding => "dynamic_binding",
        DeviceCommandPhase::Compute => "compute",
        DeviceCommandPhase::ResultBinding => "result_binding",
    }
}

const fn tensor_access_label(value: TensorAccess) -> &'static str {
    match value {
        TensorAccess::Read => "read",
        TensorAccess::Write => "write",
        TensorAccess::ReadWrite => "read_write",
    }
}

const fn element_type_label(value: ElementType) -> &'static str {
    match value {
        ElementType::Bool => "bool",
        ElementType::U8 => "u8",
        ElementType::U32 => "u32",
        ElementType::I8 => "i8",
        ElementType::I32 => "i32",
        ElementType::F16 => "f16",
        ElementType::Bf16 => "bf16",
        ElementType::F32 => "f32",
    }
}
