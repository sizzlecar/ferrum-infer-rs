//! Optional work by the actual selected algorithm. This is input evidence,
//! never a duration, executable permission, or qualified training sample.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AlgorithmWorkKindV1 {
    Kernel,
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
    Fill,
    /// One documented device-library API call, not one native kernel.
    /// Appended to preserve the existing wire/digest discriminants 0..=4.
    LibraryCall,
}
impl AlgorithmWorkKindV1 {
    pub fn is_compute(self) -> bool {
        matches!(self, Self::Kernel | Self::LibraryCall)
    }

    /// Kind-specific numeric validation shared with numerical source replay.
    /// A library call supplies no claim about its private native launch grid.
    pub fn validate_work(
        self,
        work: DeviceNumericWorkV1,
    ) -> Result<(), StatisticalEvidenceUnknown> {
        if work.padded_units < work.logical_units {
            return Err(StatisticalEvidenceUnknown::InvalidWork);
        }
        let valid = match self {
            Self::Kernel => {
                work.logical_units > 0 && work.inner_work_units > 0 && work.grid_blocks > 0
            }
            Self::LibraryCall => {
                work.logical_units > 0
                    && work.padded_units == work.logical_units
                    && work.inner_work_units > 0
                    && work.grid_blocks == 0
                    && work.peak_scratch_bytes == 0
                    && work.staged_weight_bytes == 0
                    && work.host_to_device_bytes == 0
                    && work.device_to_host_bytes == 0
                    && work.device_to_device_bytes == 0
                    && work.fill_bytes == 0
            }
            _ => {
                work.logical_units == 0
                    && work.padded_units == 0
                    && work.inner_work_units == 0
                    && work.grid_blocks == 0
            }
        };
        if valid {
            Ok(())
        } else {
            Err(StatisticalEvidenceUnknown::InvalidWork)
        }
    }
}
impl From<StatisticalTransferKindV1> for AlgorithmWorkKindV1 {
    fn from(value: StatisticalTransferKindV1) -> Self {
        match value {
            StatisticalTransferKindV1::HostToDevice => Self::HostToDevice,
            StatisticalTransferKindV1::DeviceToHost => Self::DeviceToHost,
            StatisticalTransferKindV1::DeviceToDevice => Self::DeviceToDevice,
            StatisticalTransferKindV1::Fill => Self::Fill,
        }
    }
}

/// Work units have meaning only together with this exact algorithm class and
/// command kind. The surrounding command recipe still retains execution order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AlgorithmNumericWorkV1 {
    algorithm: SelectedAlgorithmClassV1,
    kind: AlgorithmWorkKindV1,
    commands: u64,
    work: DeviceNumericWorkV1,
}
impl AlgorithmNumericWorkV1 {
    pub fn algorithm(&self) -> SelectedAlgorithmClassV1 {
        self.algorithm
    }
    pub fn kind(&self) -> AlgorithmWorkKindV1 {
        self.kind
    }
    pub fn commands(&self) -> u64 {
        self.commands
    }
    pub fn work(&self) -> DeviceNumericWorkV1 {
        self.work
    }
    fn key(&self) -> (&[u8; 32], AlgorithmWorkKindV1) {
        (self.algorithm.signature(), self.kind)
    }
}

/// Built in the same pass as a selected command. No deserialization or public
/// constructor can synthesize missing per-algorithm inputs from aggregate work.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct SelectedAlgorithmWorkEvidenceV1 {
    protocol: &'static str,
    command_binding: [u8; 32],
    entries: Vec<AlgorithmNumericWorkV1>,
}
impl SelectedAlgorithmWorkEvidenceV1 {
    /// Already checked by the private accumulator before attachment to its
    /// immutable selected command. This is passive identity, not a permit.
    pub(super) fn command_binding(&self) -> [u8; 32] {
        self.command_binding
    }

    pub fn entries(&self) -> &[AlgorithmNumericWorkV1] {
        &self.entries
    }

    pub fn validate_command(
        &self,
        command: &SelectedCommandCostEvidenceV1,
    ) -> Result<(), StatisticalEvidenceUnknown> {
        if self.protocol != "ferrum.selected-algorithm-work.v1"
            || self.entries.is_empty()
            || self.entries.capacity() > MAX_COST_COMMANDS
            || self.command_binding != command.algorithm_work_binding()?
        {
            return Err(StatisticalEvidenceUnknown::CommandMismatch);
        }
        let mut aggregate = DeviceNumericWorkV1::default();
        let (mut compute, mut transfers) = (0_u64, 0_u64);
        let mut prior = None;
        for entry in &self.entries {
            if entry.commands == 0 || prior.is_some_and(|key| key >= entry.key()) {
                return Err(StatisticalEvidenceUnknown::InvalidWork);
            }
            prior = Some(entry.key());
            entry.kind.validate_work(entry.work)?;
            aggregate = aggregate.checked_add(entry.work)?;
            let count = if entry.kind.is_compute() {
                &mut compute
            } else {
                &mut transfers
            };
            *count = count
                .checked_add(entry.commands)
                .ok_or(StatisticalEvidenceUnknown::Overflow)?;
        }
        command.validate_command(command.token_count(), compute, transfers)?;
        if aggregate != command.work() {
            return Err(StatisticalEvidenceUnknown::CommandMismatch);
        }
        Ok(())
    }

    /// Allocation retained by an Arc to this value, including its counters and
    /// unused vector capacity. The caller must separately account its owner.
    pub fn retained_bytes(&self) -> Result<usize, StatisticalEvidenceUnknown> {
        self.entries
            .capacity()
            .checked_mul(std::mem::size_of::<AlgorithmNumericWorkV1>())
            .and_then(|n| n.checked_add(std::mem::size_of::<Self>()))
            .and_then(|n| n.checked_add(2 * std::mem::size_of::<usize>()))
            .ok_or(StatisticalEvidenceUnknown::Overflow)
    }
}

pub(super) struct AlgorithmWorkAccumulator {
    entries: Vec<AlgorithmNumericWorkV1>,
    failure: Option<StatisticalEvidenceUnknown>,
    assignment: Sha256,
}
impl AlgorithmWorkAccumulator {
    pub(super) fn new() -> Self {
        let mut assignment = Sha256::new();
        bytes(
            &mut assignment,
            b"ferrum.selected-algorithm-work-assignment.v1",
        );
        Self {
            entries: Vec::new(),
            failure: None,
            assignment,
        }
    }
    pub(super) fn assignment_signature(&self) -> Option<[u8; 32]> {
        self.failure
            .is_none()
            .then(|| self.assignment.clone().finalize().into())
    }
    pub(super) fn observe(
        &mut self,
        algorithm: SelectedAlgorithmClassV1,
        kind: AlgorithmWorkKindV1,
        work: DeviceNumericWorkV1,
    ) {
        if self.failure.is_none() {
            if let Err(error) = self.append(algorithm, kind, work) {
                self.failure = Some(error);
            }
        }
    }
    fn append(
        &mut self,
        algorithm: SelectedAlgorithmClassV1,
        kind: AlgorithmWorkKindV1,
        work: DeviceNumericWorkV1,
    ) -> Result<(), StatisticalEvidenceUnknown> {
        let key = (algorithm.signature(), kind);
        match self.entries.binary_search_by(|entry| entry.key().cmp(&key)) {
            Ok(index) => {
                let entry = &mut self.entries[index];
                let next = entry.work.checked_add(work)?;
                let commands = entry
                    .commands
                    .checked_add(1)
                    .ok_or(StatisticalEvidenceUnknown::Overflow)?;
                entry.work = next;
                entry.commands = commands;
            }
            Err(index) => {
                if self.entries.len() == MAX_COST_COMMANDS {
                    return Err(StatisticalEvidenceUnknown::Capacity);
                }
                if self.entries.len() == self.entries.capacity() {
                    self.entries
                        .try_reserve_exact((MAX_COST_COMMANDS - self.entries.len()).min(8))
                        .map_err(|_| StatisticalEvidenceUnknown::Capacity)?;
                }
                if self.entries.capacity() > MAX_COST_COMMANDS {
                    return Err(StatisticalEvidenceUnknown::Capacity);
                }
                self.entries.insert(
                    index,
                    AlgorithmNumericWorkV1 {
                        algorithm,
                        kind,
                        commands: 1,
                        work,
                    },
                );
            }
        }
        // Retain numeric assignment order beside the sparse sums. Equal totals
        // cannot exchange evidence across different algorithms or occurrences.
        self.assignment.update(algorithm.signature());
        number(&mut self.assignment, kind as u64);
        for n in [
            work.logical_units,
            work.padded_units,
            work.inner_work_units,
            work.grid_blocks,
            work.peak_scratch_bytes,
            work.staged_weight_bytes,
            work.host_to_device_bytes,
            work.device_to_host_bytes,
            work.device_to_device_bytes,
            work.fill_bytes,
        ] {
            number(&mut self.assignment, n);
        }
        Ok(())
    }
    pub(super) fn finish(
        self,
        command: &SelectedCommandCostEvidenceV1,
    ) -> Result<SelectedAlgorithmWorkEvidenceV1, StatisticalEvidenceUnknown> {
        if let Some(error) = self.failure {
            return Err(error);
        }
        let value = SelectedAlgorithmWorkEvidenceV1 {
            protocol: "ferrum.selected-algorithm-work.v1",
            command_binding: command.algorithm_work_binding()?,
            entries: self.entries,
        };
        value.validate_command(command)?;
        Ok(value)
    }
}

#[cfg(test)]
mod tests;
