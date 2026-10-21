//! Submission-owned tensor transfers. No payload is copied into the legacy
//! upload packet or a replayable executable.

use super::*;

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

pub(super) struct ManagedUpload {
    source: ManagedHostBytes,
    destination: CudaBufferRegion,
    layout: HostTransferLayout,
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
            || self.source.class() != ferrum_runtime::HostMemoryClass::InputPreprocess
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

#[cfg(test)]
mod tests {
    use super::*;

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
    ) -> Result<Self, CudaDeviceRuntimeError> {
        let upload = ManagedUpload {
            source,
            destination,
            layout,
        };
        upload.validate(runtime_instance)?;
        Ok(Self {
            completion_checks: Vec::new(),
            runtime_instance,
            operation: HOST_UPLOAD_NATIVE_OPERATION_ID.as_str(),
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
            managed_upload: Some(upload),
            tensor_flight: None,
            execution_facts: None,
        })
    }

    pub(super) fn validate_managed_transfer(&self) -> Result<(), CudaDeviceRuntimeError> {
        let Some(upload) = &self.managed_upload else {
            return Ok(());
        };
        upload.validate(self.runtime_instance)?;
        if self.executable.is_some()
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
                "managed CUDA upload cannot enter a cached executable",
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
