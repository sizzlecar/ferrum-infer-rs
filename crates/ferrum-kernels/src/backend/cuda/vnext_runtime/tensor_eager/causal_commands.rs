//! Eager token commands admitted by the closed additional-module profile.
use super::*;
use ferrum_runtime::DeviceReusableExecutionProgramId;

impl CudaDeviceCommand {
    pub(crate) fn prepare_bounded_causal_phase(
        &mut self,
        phase: DeviceCommandPhase,
        replay_enabled: bool,
        capture: Option<&DeviceReusableExecutionProgramId>,
    ) -> Result<(), CudaDeviceRuntimeError> {
        if self.tensor_work.is_some() || self.tensor_transfer.is_some() {
            return self.validate_tensor_phase(phase);
        }
        if self.reusable_execution.is_some()
            && replay_enabled
            && phase == DeviceCommandPhase::Compute
        {
            // The original stream catalog validates this exact program and retains its native owners.
            return Ok(());
        }
        let admitted = self
            .executable
            .as_ref()
            .is_some_and(|e| e._causal_modules.is_some());
        let binding = self.binding_program.is_some() || self.program_binding_patch.is_some();
        if self.reusable_execution.is_some()
            || self
                .executable
                .as_ref()
                .is_some_and(|e| e.host_upload_layout.is_some())
            || (phase == DeviceCommandPhase::Compute
                && (!admitted
                    || (self.compute_dispatch_count == 0 && self.transfer_command_count == 0)))
            || (phase != DeviceCommandPhase::Compute && !admitted && !binding)
        {
            return Err(CudaDeviceRuntimeError::contract(format!(
                "bounded causal command rejected: operation={}, phase={phase:?}, admitted_modules={admitted}, reusable={}, unmanaged_upload={}, compute_dispatches={}, transfers={}, binding={binding}",
                self.operation,
                self.reusable_execution.is_some(),
                self.executable
                    .as_ref()
                    .is_some_and(|e| e.host_upload_layout.is_some()),
                self.compute_dispatch_count,
                self.transfer_command_count,
            )));
        }
        // A compute-phase graph operation may consist solely of device copies
        // (for example, concatenating token rows). Its real transfer count is
        // work; do not invent a kernel launch. The original module, region and
        // managed-transfer checks still apply.
        // F16 providers also serve the unbounded legacy runtime. Their existing
        // replay keys authorize capture only in the explicitly bounded decode
        // route. Regions, functions, bindings and the module quota stay owned.
        if !crate::backend::cuda_tensor_causal_policy::permits_command_replay(
            replay_enabled,
            phase,
            self.participant_count,
            self.token_count,
            capture.map(|id| (id.immediate_sequences(), id.immediate_tokens())),
        ) {
            self.replay_key = None;
            self.reusable_address_scope = None;
        }
        Ok(())
    }
}
