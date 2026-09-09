use ferrum_interfaces::vnext::{
    DeviceBatchingForm, DeviceCommandLogicalWork, DeviceCommandPhase, DeviceExecutionPath,
    DeviceNativeOperationId, DeviceNativeWorkAttribution,
};

use super::memory::{CpuBufferRegion, Storage};
use super::CpuRuntimeError;

/// CPU providers own their validated bindings until synchronous execution has
/// reached a terminal. Validation must not write any of those bindings.
pub(crate) trait CpuKernelLaunch: Send {
    fn validate_runtime(&self, instance: u64) -> Result<(), CpuRuntimeError>;
    fn execute(&self) -> Result<(), CpuRuntimeError>;
}

pub(super) enum CommandKind {
    Copy {
        source: CpuBufferRegion,
        destination: CpuBufferRegion,
    },
    Upload {
        source: Storage,
        destination: CpuBufferRegion,
    },
    Zero {
        destination: CpuBufferRegion,
    },
    Compute {
        launches: Vec<Box<dyn CpuKernelLaunch>>,
    },
}

pub struct CpuDeviceCommand {
    operation: &'static str,
    batching_form: DeviceBatchingForm,
    participant_start: u32,
    participant_count: u32,
    token_count: u64,
    kind: CommandKind,
}

impl CpuDeviceCommand {
    pub(super) fn transfer(operation: &'static str, kind: CommandKind) -> Self {
        Self {
            operation,
            batching_form: DeviceBatchingForm::Scalar,
            participant_start: 0,
            participant_count: 0,
            token_count: 0,
            kind,
        }
    }

    pub(crate) fn compute(
        operation: &'static str,
        launches: Vec<Box<dyn CpuKernelLaunch>>,
        batching_form: DeviceBatchingForm,
        participants: u32,
        tokens: u64,
    ) -> Result<Self, CpuRuntimeError> {
        if launches.is_empty()
            || participants == 0
            || tokens == 0
            || DeviceNativeOperationId::new(operation).is_none()
        {
            return Err(CpuRuntimeError::new(
                "CPU compute command has invalid work attribution",
            ));
        }
        Ok(Self {
            operation,
            batching_form,
            participant_start: 0,
            participant_count: participants,
            token_count: tokens,
            kind: CommandKind::Compute { launches },
        })
    }

    pub(super) fn bind_logical_work(
        mut self,
        work: DeviceCommandLogicalWork,
    ) -> Result<Self, CpuRuntimeError> {
        if self.participant_count != 0 || self.token_count != 0 {
            return Err(CpuRuntimeError::new(
                "CPU core logical work cannot replace provider attribution",
            ));
        }
        self.batching_form = work.batching_form();
        self.participant_start = work.participant_start();
        self.participant_count = work.participant_count();
        self.token_count = work.token_count();
        Ok(self)
    }

    pub(super) fn attribution(
        &self,
        index: u32,
        node: Option<u32>,
        phase: DeviceCommandPhase,
    ) -> Result<DeviceNativeWorkAttribution, CpuRuntimeError> {
        let (compute, transfer) = match &self.kind {
            CommandKind::Compute { launches } => (launches.len() as u64, 0),
            _ => (0, 1),
        };
        DeviceNativeWorkAttribution::with_participant_range(
            index,
            node,
            phase,
            DeviceNativeOperationId::new(self.operation)
                .ok_or_else(|| CpuRuntimeError::new("invalid CPU native operation ID"))?,
            DeviceExecutionPath::Eager,
            self.batching_form,
            self.participant_start,
            self.participant_count,
            self.token_count,
            compute,
            transfer,
            None,
        )
        .ok_or_else(|| CpuRuntimeError::new("invalid CPU native work attribution"))
    }

    pub(super) fn validate_runtime(&self, instance: u64) -> Result<(), CpuRuntimeError> {
        match &self.kind {
            CommandKind::Copy {
                source,
                destination,
            } => {
                source.validate_runtime(instance)?;
                destination.validate_runtime(instance)
            }
            CommandKind::Upload { destination, .. } | CommandKind::Zero { destination } => {
                destination.validate_runtime(instance)
            }
            CommandKind::Compute { launches } => {
                for launch in launches {
                    launch.validate_runtime(instance)?;
                }
                Ok(())
            }
        }
    }

    pub(super) fn execute(&self) -> Result<(), CpuRuntimeError> {
        match &self.kind {
            CommandKind::Copy {
                source,
                destination,
            } => source.copy_to(destination),
            CommandKind::Upload {
                source,
                destination,
            } => destination.write(source.bytes()),
            CommandKind::Zero { destination } => destination.zero(),
            CommandKind::Compute { launches } => {
                for launch in launches {
                    launch.execute()?;
                }
                Ok(())
            }
        }
    }
}
