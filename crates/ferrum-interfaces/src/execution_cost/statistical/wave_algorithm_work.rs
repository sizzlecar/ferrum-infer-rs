//! Complete command-bound algorithm work, collected only by the structured
//! opt-in builder. These numerical inputs confer no execution or fit authority.
use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct WaveAlgorithmNumericWorkV1 {
    algorithm: SelectedAlgorithmClassV1,
    kind: AlgorithmWorkKindV1,
    commands: u64,
    work: DeviceNumericWorkV1,
}
impl WaveAlgorithmNumericWorkV1 {
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeviceAlgorithmWorkEvidenceV1 {
    protocol: &'static str,
    exact_binding: [u8; 32],
    ordered_command_binding: [u8; 32],
    physical_commands: u32,
    selected_commands: u64,
    entries: Vec<WaveAlgorithmNumericWorkV1>,
    aggregate_work: DeviceNumericWorkV1,
}
impl DeviceAlgorithmWorkEvidenceV1 {
    pub fn entries(&self) -> &[WaveAlgorithmNumericWorkV1] {
        &self.entries
    }
    pub fn physical_command_count(&self) -> u32 {
        self.physical_commands
    }
    pub fn selected_command_count(&self) -> u64 {
        self.selected_commands
    }
    pub fn ordered_command_binding(&self) -> &[u8; 32] {
        &self.ordered_command_binding
    }
    pub fn aggregate_work(&self) -> DeviceNumericWorkV1 {
        self.aggregate_work
    }
    pub fn validate_exact(
        &self,
        exact: &CanonicalWaveCostShape,
    ) -> Result<(), StatisticalEvidenceUnknown> {
        self.validate_binding(super::wave::exact_binding(exact)?)
    }
    /// An exact canonical shape alone intentionally has the old statistical
    /// semantics. This also checks the NEW ordered assignment binding against
    /// the actual/proposed structural recipe, rejecting equal-total exchanges.
    pub fn validate_structure(
        &self,
        recipe: &UnsettledStructuredWaveEvidenceV1,
    ) -> Result<(), StatisticalEvidenceUnknown> {
        let expected = recipe.algorithm_work()?;
        self.validate_binding(expected.exact_binding)?;
        if self.ordered_command_binding != expected.ordered_command_binding
            || self.physical_commands != expected.physical_commands
            || self.aggregate_work != recipe.device().aggregate_work()
        {
            return Err(StatisticalEvidenceUnknown::CommandMismatch);
        }
        Ok(())
    }
    pub(super) fn validate_binding(
        &self,
        binding: [u8; 32],
    ) -> Result<(), StatisticalEvidenceUnknown> {
        if self.protocol != "ferrum.device-algorithm-work.v1"
            || self.exact_binding != binding
            || self.physical_commands == 0
            || self.selected_commands == 0
            || self.entries.is_empty()
            || self.physical_commands as usize > MAX_COST_COMMANDS
            || self.selected_commands > MAX_COST_COMMANDS as u64
            || self.entries.capacity() > MAX_COST_COMMANDS
        {
            return Err(StatisticalEvidenceUnknown::ExactBindingMismatch);
        }
        self.retained_dynamic_bytes()?;
        Ok(())
    }
    /// Dynamic allocation only. Self is inline in the structured recipe.
    /// No command Arc or duplicate per-command numeric payload is retained.
    pub fn retained_dynamic_bytes(&self) -> Result<usize, StatisticalEvidenceUnknown> {
        self.entries
            .capacity()
            .checked_mul(std::mem::size_of::<WaveAlgorithmNumericWorkV1>())
            .ok_or(StatisticalEvidenceUnknown::Overflow)
    }
}

struct Pending {
    commands: usize,
    entries: Vec<WaveAlgorithmNumericWorkV1>,
    selected_commands: u64,
    aggregate: DeviceNumericWorkV1,
    ordered: Sha256,
}
pub(super) struct WaveAlgorithmAccumulator {
    pending: Result<Pending, StatisticalEvidenceUnknown>,
}
impl WaveAlgorithmAccumulator {
    pub(super) fn new() -> Self {
        let mut ordered = Sha256::new();
        bytes(&mut ordered, b"ferrum.wave-algorithm-assignment.v1");
        Self {
            pending: Ok(Pending {
                commands: 0,
                entries: Vec::new(),
                selected_commands: 0,
                aggregate: Default::default(),
                ordered,
            }),
        }
    }
    pub(super) fn observe(&mut self, command: CostPhysicalCommand<'_>) {
        let result = match &mut self.pending {
            Ok(pending) => pending.append(command),
            Err(_) => return,
        };
        if let Err(error) = result {
            // Release the allocated sparse table on failure. An Err
            // sidecar may not hide uncharged partial allocations.
            self.pending = Err(error);
        }
    }
    pub(super) fn finish(
        self,
        shape: &CanonicalWaveCostShape,
        count: usize,
        aggregate: DeviceNumericWorkV1,
    ) -> Result<DeviceAlgorithmWorkEvidenceV1, StatisticalEvidenceUnknown> {
        let pending = self.pending?;
        if pending.commands == 0 || pending.commands != count || pending.aggregate != aggregate {
            return Err(StatisticalEvidenceUnknown::CommandMismatch);
        }
        let value = DeviceAlgorithmWorkEvidenceV1 {
            protocol: "ferrum.device-algorithm-work.v1",
            exact_binding: super::wave::exact_binding(shape)?,
            ordered_command_binding: pending.ordered.finalize().into(),
            physical_commands: pending.commands as u32,
            selected_commands: pending.selected_commands,
            entries: pending.entries,
            aggregate_work: pending.aggregate,
        };
        value.validate_exact(shape)?;
        Ok(value)
    }
}
fn reserve<T>(values: &mut Vec<T>) -> Result<(), StatisticalEvidenceUnknown> {
    if values.len() >= MAX_COST_COMMANDS {
        return Err(StatisticalEvidenceUnknown::Capacity);
    }
    if values.len() == values.capacity() {
        values
            .try_reserve_exact((MAX_COST_COMMANDS - values.len()).min(8))
            .map_err(|_| StatisticalEvidenceUnknown::Capacity)?;
    }
    if values.capacity() > MAX_COST_COMMANDS {
        return Err(StatisticalEvidenceUnknown::Capacity);
    }
    Ok(())
}
impl Pending {
    fn append(
        &mut self,
        command: CostPhysicalCommand<'_>,
    ) -> Result<(), StatisticalEvidenceUnknown> {
        if command.path != CostCommandPath::Eager || command.reusable_graph_node_count.is_some() {
            return Err(StatisticalEvidenceUnknown::UnsupportedReplay);
        }
        let selected = command
            .statistical_evidence
            .ok_or(StatisticalEvidenceUnknown::MissingProducer)?;
        selected.validate_command(
            command.token_count,
            command.compute_dispatch_count,
            command.transfer_command_count,
        )?;
        let evidence = selected
            .algorithm_work()
            .ok_or(StatisticalEvidenceUnknown::MissingProducer)??;
        evidence.validate_command(selected)?;
        let dispatches = command
            .compute_dispatch_count
            .checked_add(command.transfer_command_count)
            .ok_or(StatisticalEvidenceUnknown::Overflow)?;
        let selected_commands = self
            .selected_commands
            .checked_add(dispatches)
            .ok_or(StatisticalEvidenceUnknown::Overflow)?;
        if selected_commands > MAX_COST_COMMANDS as u64 {
            return Err(StatisticalEvidenceUnknown::Capacity);
        }
        let aggregate = self.aggregate.checked_add(selected.work())?;
        if self.commands >= MAX_COST_COMMANDS {
            return Err(StatisticalEvidenceUnknown::Capacity);
        }
        for entry in evidence.entries() {
            let key = (entry.algorithm().signature().to_owned(), entry.kind());
            match self
                .entries
                .binary_search_by(|value| value.key().cmp(&(&key.0, key.1)))
            {
                Ok(index) => {
                    let old = &mut self.entries[index];
                    old.commands = old
                        .commands
                        .checked_add(entry.commands())
                        .ok_or(StatisticalEvidenceUnknown::Overflow)?;
                    old.work = old.work.checked_add(entry.work())?;
                }
                Err(index) => {
                    reserve(&mut self.entries)?;
                    self.entries.insert(
                        index,
                        WaveAlgorithmNumericWorkV1 {
                            algorithm: entry.algorithm(),
                            kind: entry.kind(),
                            commands: entry.commands(),
                            work: entry.work(),
                        },
                    );
                }
            }
        }
        number(&mut self.ordered, self.commands as u64);
        super::wave::append_command_identity(&mut self.ordered, command);
        self.ordered.update(selected.algorithm_work_binding()?);
        self.commands += 1;
        self.selected_commands = selected_commands;
        self.aggregate = aggregate;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
