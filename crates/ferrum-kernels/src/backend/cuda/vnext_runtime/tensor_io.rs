//! Submission-owned tensor transfers. No payload is copied into the legacy
//! upload packet or a replayable executable.

use super::*;
use ferrum_runtime::{BufferUsage, HostMemoryClass};

/// Armed before the first possible enqueue. Any unwind, including timing or
/// bookkeeping callbacks, preserves commands instead of dropping DMA owners.
pub(super) struct SubmissionGuard<'a> {
    runtime: &'a CudaDeviceRuntime,
    stream_id: u64,
    stream: Arc<CudaStream>,
    blas: Arc<CudaBlas>,
    state: Arc<CudaStreamState>,
    commands: Vec<CudaDeviceCommand>,
    armed: bool,
}

impl<'a> SubmissionGuard<'a> {
    pub(super) fn new(
        runtime: &'a CudaDeviceRuntime,
        stream: &CudaDeviceStream,
        commands: Vec<CudaDeviceCommand>,
    ) -> Self {
        Self {
            runtime,
            stream_id: stream.id,
            stream: Arc::clone(&stream.stream),
            blas: Arc::clone(&stream.blas),
            state: Arc::clone(&stream.state),
            commands,
            armed: false,
        }
    }
    pub(super) fn arm(&mut self) {
        self.armed = true;
    }
    pub(super) fn take(&mut self) -> Vec<CudaDeviceCommand> {
        self.armed = false;
        std::mem::take(&mut self.commands)
    }
}

impl std::ops::Deref for SubmissionGuard<'_> {
    type Target = Vec<CudaDeviceCommand>;
    fn deref(&self) -> &Self::Target {
        &self.commands
    }
}
impl std::ops::DerefMut for SubmissionGuard<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.commands
    }
}
impl Drop for SubmissionGuard<'_> {
    fn drop(&mut self) {
        if self.armed && !self.commands.is_empty() {
            self.state.fail();
            self.runtime.quarantined().push(QuarantinedSubmission {
                stream_id: self.stream_id,
                _stream: Arc::clone(&self.stream),
                _blas: Arc::clone(&self.blas),
                _commands: std::mem::take(&mut self.commands),
            });
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum UploadPurpose {
    Input,
    StaticWeight,
}

impl UploadPurpose {
    pub(super) fn new(
        class: HostMemoryClass,
        usage: BufferUsage,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        match (class, usage) {
            (HostMemoryClass::InputPreprocess, BufferUsage::Activations) => Ok(Self::Input),
            (HostMemoryClass::TransferStaging, BufferUsage::Weights) => Ok(Self::StaticWeight),
            _ => Err(CudaDeviceRuntimeError::contract(
                "managed CUDA upload class differs from destination usage",
            )),
        }
    }

    fn phase(self) -> DeviceCommandPhase {
        match self {
            Self::Input => DeviceCommandPhase::DynamicBinding,
            Self::StaticWeight => DeviceCommandPhase::Initialization,
        }
    }

    fn host_class(self) -> HostMemoryClass {
        match self {
            Self::Input => HostMemoryClass::InputPreprocess,
            Self::StaticWeight => HostMemoryClass::TransferStaging,
        }
    }
}

pub(super) struct ManagedUpload {
    source: ManagedHostBytes,
    destination: CudaBufferRegion,
    layout: HostTransferLayout,
    purpose: UploadPurpose,
}

impl ManagedUpload {
    pub(super) fn enqueue(&self, stream: &CudaStream) -> Result<(), CudaDeviceRuntimeError> {
        // Ordinary accounted host storage may be pageable. CUDA may stage or
        // synchronize this transfer; retaining the original owner proves its
        // lifetime, not pinned DMA or a bound on opaque driver staging.
        unsafe {
            cudarc::driver::result::memcpy_htod_async(
                self.destination.device_ptr,
                self.source.bytes(),
                stream.cu_stream(),
            )
        }
        .map_err(|error| CudaDeviceRuntimeError::driver("managed host upload", error))
    }

    fn validate(&self, runtime_instance: u64) -> Result<(), CudaDeviceRuntimeError> {
        self.layout
            .validate_bytes(self.source.len())
            .map_err(|error| CudaDeviceRuntimeError::contract(error.to_string()))?;
        if self.source.is_empty()
            || self.source.class() != self.purpose.host_class()
            || self.destination.runtime_instance != runtime_instance
            || self.layout.element_type() != self.destination.element_type
            || self.destination.length_bytes != self.source.len() as u64
            || self.destination.backing_byte_offset() % self.layout.element_type().size_bytes() != 0
        {
            return Err(CudaDeviceRuntimeError::contract(
                "invalid managed CUDA upload region",
            ));
        }
        Ok(())
    }
}

/// Typed eager transfers never enter the legacy executable/upload packet.
/// A static zero has no host payload and is admitted only in Initialization.
pub(super) enum TensorTransfer {
    Upload(ManagedUpload),
    StaticZero(CudaBufferRegion),
    StateCopy {
        source: CudaBufferRegion,
        destination: CudaBufferRegion,
    },
}

impl TensorTransfer {
    pub(super) fn enqueue(&self, stream: &CudaStream) -> Result<(), CudaDeviceRuntimeError> {
        match self {
            Self::Upload(upload) => upload.enqueue(stream),
            Self::StateCopy {
                source,
                destination,
            } => {
                let bytes = checked_usize(source.length_bytes, "CUDA state copy length")?;
                unsafe {
                    cudarc::driver::result::memcpy_dtod_async(
                        destination.device_ptr,
                        source.device_ptr,
                        bytes,
                        stream.cu_stream(),
                    )
                }
                .map_err(|error| CudaDeviceRuntimeError::driver("fixed tensor state copy", error))
            }
            Self::StaticZero(destination) => {
                let bytes = checked_usize(destination.length_bytes, "CUDA static zero length")?;
                unsafe {
                    cudarc::driver::result::memset_d8_async(
                        destination.device_ptr,
                        0,
                        bytes,
                        stream.cu_stream(),
                    )
                }
                .map_err(|error| CudaDeviceRuntimeError::driver("static zero", error))
            }
        }
    }

    fn validate(&self, runtime_instance: u64) -> Result<(), CudaDeviceRuntimeError> {
        match self {
            Self::Upload(upload) => upload.validate(runtime_instance),
            Self::StateCopy {
                source,
                destination,
            } => {
                let source_end = source.device_ptr.checked_add(source.length_bytes);
                let destination_end = destination.device_ptr.checked_add(destination.length_bytes);
                if source.runtime_instance != runtime_instance
                    || destination.runtime_instance != runtime_instance
                    || source.length_bytes == 0
                    || source.length_bytes != destination.length_bytes
                    || source.element_type != destination.element_type
                    || source._core_retention.is_none()
                    || destination._core_retention.is_none()
                    || source_end.is_none()
                    || destination_end.is_none()
                    || (source.device_ptr < destination_end.unwrap_or(0)
                        && destination.device_ptr < source_end.unwrap_or(0))
                {
                    return Err(CudaDeviceRuntimeError::contract(
                        "invalid retained CUDA state copy",
                    ));
                }
                checked_usize(source.length_bytes, "CUDA state copy length")?;
                Ok(())
            }
            Self::StaticZero(destination) => {
                if destination.runtime_instance != runtime_instance || destination.length_bytes == 0
                {
                    return Err(CudaDeviceRuntimeError::contract(
                        "invalid CUDA static zero region",
                    ));
                }
                checked_usize(destination.length_bytes, "CUDA static zero length")?;
                Ok(())
            }
        }
    }

    fn phase(&self) -> DeviceCommandPhase {
        match self {
            Self::Upload(upload) => upload.purpose.phase(),
            Self::StaticZero(_) => DeviceCommandPhase::Initialization,
            Self::StateCopy { .. } => DeviceCommandPhase::ResultBinding,
        }
    }

    fn operation(&self) -> &'static str {
        match self {
            Self::Upload(_) => HOST_UPLOAD_NATIVE_OPERATION_ID.as_str(),
            Self::StaticZero(_) => DEVICE_ZERO_NATIVE_OPERATION_ID.as_str(),
            Self::StateCopy { .. } => DEVICE_COPY_NATIVE_OPERATION_ID.as_str(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn managed_upload_roles_do_not_turn_weights_into_request_inputs() {
        let input =
            UploadPurpose::new(HostMemoryClass::InputPreprocess, BufferUsage::Activations).unwrap();
        assert_eq!(input.phase(), DeviceCommandPhase::DynamicBinding);
        let weights =
            UploadPurpose::new(HostMemoryClass::TransferStaging, BufferUsage::Weights).unwrap();
        assert_eq!(weights.phase(), DeviceCommandPhase::Initialization);
        for (class, usage) in [
            (HostMemoryClass::InputPreprocess, BufferUsage::Weights),
            (HostMemoryClass::TransferStaging, BufferUsage::Activations),
            (HostMemoryClass::Output, BufferUsage::Weights),
            (HostMemoryClass::WeightBacking, BufferUsage::Weights),
            (HostMemoryClass::TransferStaging, BufferUsage::Persistent),
        ] {
            assert!(UploadPurpose::new(class, usage).is_err());
        }
    }

    #[test]
    fn tensor_compute_cannot_be_submitted_as_initialization_or_input_upload() {
        let command = tensor_command();
        assert!(command
            .validate_tensor_phase(DeviceCommandPhase::Compute)
            .is_ok());
        for phase in [
            DeviceCommandPhase::Initialization,
            DeviceCommandPhase::DynamicBinding,
            DeviceCommandPhase::ResultBinding,
        ] {
            assert!(command.validate_tensor_phase(phase).is_err());
        }
    }

    #[test]
    fn managed_pieces_use_exact_length_and_nonzero_destination_offsets() {
        let layout = HostTransferLayout::new(ElementType::Bf16, 3).unwrap();
        assert_eq!(
            transfer_range(layout, 16, 20, CopyRegion::new(4, 10, 6).unwrap()).unwrap(),
            (4..10, 10..16)
        );
        assert!(transfer_range(layout, 16, 15, CopyRegion::new(4, 10, 6).unwrap()).is_err());
        assert!(transfer_range(layout, 16, 20, CopyRegion::new(3, 10, 6).unwrap()).is_err());
        assert!(transfer_range(layout, 16, 20, CopyRegion::new(4, 9, 6).unwrap()).is_err());
        assert!(transfer_range(layout, 16, 20, CopyRegion::new(4, 10, 4).unwrap()).is_err());
        assert!(CopyRegion::new(u64::MAX - 3, 0, 6).is_err());
    }

    fn tensor_command() -> CudaDeviceCommand {
        // Metadata only: this command is never submitted to a CUDA stream.
        CudaDeviceCommand::transfer(
            1,
            "test_tensor",
            Vec::new(),
            Vec::new(),
            Box::new(|_, _, _, _| Ok(())),
        )
        .with_tensor_work(8, 32, 1, 0)
        .unwrap()
    }

    #[test]
    fn tensor_attribution_preserves_elements_without_synthetic_tokens() {
        let command = tensor_command();
        let attribution = cuda_submission_attribution(
            &[DeviceCommandPhase::Compute],
            &[Some(2)],
            &[command],
            &[DeviceExecutionPath::Eager],
            None,
            Vec::new(),
            None,
        )
        .unwrap();
        let row = &attribution.commands()[0];
        assert_eq!(row.token_count(), None);
        assert_eq!(
            row.logical_work(),
            ferrum_runtime::DeviceLogicalWork::Tensor {
                element_count: 8,
                arithmetic_operations: 32
            }
        );
        assert_eq!(row.compute_dispatch_count(), 1);
    }

    #[test]
    fn tensor_metadata_rejects_replay_and_attribution_replacement() {
        let mut command = tensor_command();
        command.reusable_address_scope = Some(DeviceReusableAddressScope::Plan);
        assert!(validate_tensor_command(&command).is_err());
        assert!(tensor_command()
            .with_work_attribution(DeviceBatchingForm::Scalar, 1, 1, 1, 0)
            .is_err());
        let mut command = tensor_command();
        command.participant_count = 2;
        assert!(validate_tensor_command(&command).is_err());
    }
}

pub(super) fn validate_tensor_command(
    command: &CudaDeviceCommand,
) -> Result<(), CudaDeviceRuntimeError> {
    let Some(work) = command.tensor_work else {
        return Ok(());
    };
    if work.token_count().is_some()
        || command.token_count != 0
        || command.batching_form != work.batching_form()
        || command.participant_start != work.participant_start()
        || command.participant_count != work.participant_count()
        || command.replay_key.is_some()
        || command.reusable_address_scope.is_some()
        || command.reusable_execution.is_some()
        || command.program_binding_patch.is_some()
        || command.binding_program.is_some()
        || command.work_declaration() != CudaWorkDeclaration::Static
        || !command.completion_checks.is_empty()
    {
        return Err(CudaDeviceRuntimeError::contract(
            "CUDA tensor commands must remain eager and token-free",
        ));
    }
    Ok(())
}

impl CudaDeviceCommand {
    pub(super) fn managed_upload(
        runtime_instance: u64,
        source: ManagedHostBytes,
        destination: CudaBufferRegion,
        layout: HostTransferLayout,
        purpose: UploadPurpose,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        let upload = ManagedUpload {
            source,
            destination,
            layout,
            purpose,
        };
        Self::tensor_transfer_command(runtime_instance, TensorTransfer::Upload(upload))
    }

    pub(super) fn static_zero(
        runtime_instance: u64,
        destination: CudaBufferRegion,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::tensor_transfer_command(runtime_instance, TensorTransfer::StaticZero(destination))
    }

    pub(super) fn tensor_state_copy(
        runtime_instance: u64,
        source: CudaBufferRegion,
        destination: CudaBufferRegion,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        Self::tensor_transfer_command(
            runtime_instance,
            TensorTransfer::StateCopy {
                source,
                destination,
            },
        )
    }

    fn tensor_transfer_command(
        runtime_instance: u64,
        transfer: TensorTransfer,
    ) -> Result<Self, CudaDeviceRuntimeError> {
        transfer.validate(runtime_instance)?;
        Ok(Self {
            completion_checks: Vec::new(),
            runtime_instance,
            operation: transfer.operation(),
            batching_form: DeviceBatchingForm::Scalar,
            participant_start: 0,
            participant_count: 0,
            token_count: 0,
            compute_dispatch_count: 0,
            transfer_command_count: 1,
            executable: None,
            fence_dependencies: Vec::new(),
            binding_owner_tables: Vec::new(),
            replay_key: None,
            reusable_address_scope: None,
            replay_gap_reason: None,
            program_binding_patch: None,
            binding_program: None,
            reusable_execution: None,
            tensor_work: None,
            tensor_transfer: Some(transfer),
            tensor_flight: None,
            execution_facts: None,
        })
    }

    pub(super) fn validate_managed_transfer(&self) -> Result<(), CudaDeviceRuntimeError> {
        let Some(upload) = &self.tensor_transfer else {
            return Ok(());
        };
        upload.validate(self.runtime_instance)?;
        if self.operation != upload.operation()
            || self.executable.is_some()
            || self.replay_key.is_some()
            || self.reusable_address_scope.is_some()
            || self.reusable_execution.is_some()
            || self.program_binding_patch.is_some()
            || self.binding_program.is_some()
            || self.tensor_work.is_some()
            || self.participant_count != 0
            || self.participant_start != 0
            || self.token_count != 0
            || self.compute_dispatch_count != 0
            || self.transfer_command_count != 1
            || !self.completion_checks.is_empty()
            || !self.binding_owner_tables.is_empty()
            || !self.fence_dependencies.is_empty()
        {
            return Err(CudaDeviceRuntimeError::contract(
                "managed CUDA transfer cannot enter a cached executable",
            ));
        }
        Ok(())
    }

    pub(super) fn validate_tensor_phase(
        &self,
        phase: DeviceCommandPhase,
    ) -> Result<(), CudaDeviceRuntimeError> {
        let expected = if let Some(transfer) = &self.tensor_transfer {
            transfer.phase()
        } else if self.tensor_work.is_some() {
            DeviceCommandPhase::Compute
        } else {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA tensor eager runtime rejects token and unmanaged commands",
            ));
        };
        if phase != expected {
            return Err(CudaDeviceRuntimeError::contract(
                "CUDA tensor command differs from its required submission phase",
            ));
        }
        Ok(())
    }
}

pub(super) fn transfer_range(
    layout: HostTransferLayout,
    source_bytes: u64,
    destination_bytes: u64,
    region: CopyRegion,
) -> Result<(Range<usize>, Range<usize>), CudaDeviceRuntimeError> {
    let bytes = layout
        .byte_len()
        .map_err(|error| CudaDeviceRuntimeError::contract(error.to_string()))?;
    let element_bytes = layout.element_type().size_bytes();
    if bytes != region.length_bytes()
        || bytes == 0
        || region.source_offset_bytes() % element_bytes != 0
        || region.destination_offset_bytes() % element_bytes != 0
    {
        return Err(CudaDeviceRuntimeError::contract(
            "managed CUDA transfer splits an element or differs from its layout",
        ));
    }
    let source_end = checked_end(
        region.source_offset_bytes(),
        bytes,
        source_bytes,
        "managed CUDA source",
    )?;
    let destination_end = checked_end(
        region.destination_offset_bytes(),
        bytes,
        destination_bytes,
        "managed CUDA destination",
    )?;
    Ok((
        checked_usize(region.source_offset_bytes(), "managed CUDA source offset")?
            ..checked_usize(source_end, "managed CUDA source end")?,
        checked_usize(
            region.destination_offset_bytes(),
            "managed CUDA destination offset",
        )?..checked_usize(destination_end, "managed CUDA destination end")?,
    ))
}
