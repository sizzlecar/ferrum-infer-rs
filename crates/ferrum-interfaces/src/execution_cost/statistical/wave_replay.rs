//! Numerical replay accounting from the original physical/segment/ordinal
//! stream. This contains no executable capability and no captured old work.
use super::*;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct StructuredReplayWorkV1 {
    protocol: &'static str,
    exact_binding: [u8; 32],
    resident_binding: [u8; 32],
    replayed_segments: u32,
    logical_commands: u32,
    native_graph_nodes: u64,
}
impl StructuredReplayWorkV1 {
    pub fn resident_binding(&self) -> &[u8; 32] {
        &self.resident_binding
    }
    pub fn replayed_segments(&self) -> u32 {
        self.replayed_segments
    }
    pub fn logical_commands(&self) -> u32 {
        self.logical_commands
    }
    pub fn native_graph_nodes(&self) -> u64 {
        self.native_graph_nodes
    }
    pub(super) fn validate_binding(
        &self,
        binding: [u8; 32],
    ) -> Result<(), StatisticalEvidenceUnknown> {
        if self.protocol != "ferrum.structured-replay-work.v1"
            || self.exact_binding != binding
            || self.resident_binding == [0; 32]
            || self.replayed_segments == 0
            || self.logical_commands == 0
            || self.replayed_segments as usize > MAX_COST_COMMANDS
            || self.logical_commands as usize > MAX_COST_COMMANDS
            || self.native_graph_nodes == 0
        {
            return Err(StatisticalEvidenceUnknown::ExactBindingMismatch);
        }
        Ok(())
    }
}

pub(super) struct ReplayWaveAccumulator {
    launches: u32,
    segments: u32,
    logical: u32,
    physical_nodes: u64,
    logical_nodes: u64,
    remaining: usize,
    next_ordinal: u32,
    physical_index: u32,
    resident: Sha256,
}
impl ReplayWaveAccumulator {
    pub(super) fn new() -> Self {
        let mut resident = Sha256::new();
        bytes(
            &mut resident,
            b"ferrum.structured-replay-resident-binding.v1",
        );
        Self {
            launches: 0,
            segments: 0,
            logical: 0,
            physical_nodes: 0,
            logical_nodes: 0,
            remaining: 0,
            next_ordinal: 0,
            physical_index: 0,
            resident,
        }
    }
    pub(super) fn physical(
        &mut self,
        command: CostPhysicalCommand<'_>,
    ) -> Result<(), StatisticalEvidenceUnknown> {
        if command.path != CostCommandPath::Replayed
            || command.compute_dispatch_count != 1
            || command.transfer_command_count != 0
            || command.statistical_evidence.is_some()
        {
            return Err(StatisticalEvidenceUnknown::CommandMismatch);
        }
        let nodes = command
            .reusable_graph_node_count
            .filter(|n| *n != 0)
            .ok_or(StatisticalEvidenceUnknown::UnsupportedReplay)?;
        if self.launches as usize >= MAX_COST_COMMANDS {
            return Err(StatisticalEvidenceUnknown::Capacity);
        }
        self.launches += 1;
        self.physical_nodes = self
            .physical_nodes
            .checked_add(nodes)
            .ok_or(StatisticalEvidenceUnknown::Overflow)?;
        number(&mut self.resident, u64::from(command.command_index));
        number(&mut self.resident, nodes);
        Ok(())
    }
    pub(super) fn segment(
        &mut self,
        physical: u32,
        fingerprint: &str,
        count: usize,
    ) -> Result<(), StatisticalEvidenceUnknown> {
        if self.remaining != 0
            || self.segments >= self.launches
            || !text_valid(fingerprint)
            || count == 0
        {
            return Err(StatisticalEvidenceUnknown::CommandMismatch);
        }
        if count > MAX_COST_COMMANDS
            || (self.logical as usize)
                .checked_add(count)
                .is_none_or(|n| n > MAX_COST_COMMANDS)
        {
            return Err(StatisticalEvidenceUnknown::Capacity);
        }
        self.remaining = count;
        self.next_ordinal = 0;
        self.physical_index = physical;
        self.segments += 1;
        number(&mut self.resident, u64::from(physical));
        bytes(&mut self.resident, fingerprint.as_bytes());
        number(&mut self.resident, count as u64);
        Ok(())
    }
    pub(super) fn logical(
        &mut self,
        command: CostLogicalCommand<'_>,
    ) -> Result<u32, StatisticalEvidenceUnknown> {
        if self.remaining == 0
            || command.logical_command_ordinal != self.next_ordinal
            || command.reusable_graph_node_count == 0
        {
            return Err(StatisticalEvidenceUnknown::CommandMismatch);
        }
        let selected = command
            .statistical_evidence
            .ok_or(StatisticalEvidenceUnknown::MissingProducer)?;
        selected.validate_command(
            command.token_count,
            command.compute_dispatch_count,
            command.transfer_command_count,
        )?;
        // A replay producer must describe the actual fixed launch ABI as well
        // as current work. The resident binding path has already checked it
        // against the sealed successful-capture template.
        selected.replay_fixed_launch_signature()?;
        number(
            &mut self.resident,
            u64::from(command.logical_command_ordinal),
        );
        number(&mut self.resident, u64::from(command.node_index));
        self.resident.update(selected.algorithm_work_binding()?);
        self.logical_nodes = self
            .logical_nodes
            .checked_add(command.reusable_graph_node_count)
            .ok_or(StatisticalEvidenceUnknown::Overflow)?;
        self.logical += 1;
        self.next_ordinal += 1;
        self.remaining -= 1;
        Ok(self.physical_index)
    }
    pub(super) fn finish(
        &self,
        shape: &CanonicalWaveCostShape,
    ) -> Result<Option<StructuredReplayWorkV1>, StatisticalEvidenceUnknown> {
        if self.launches == 0 {
            return if matches!(
                shape.graph,
                ActualWaveGraphState::Disabled | ActualWaveGraphState::ConfiguredEager
            ) {
                Ok(None)
            } else {
                Err(StatisticalEvidenceUnknown::UnsupportedReplay)
            };
        }
        if shape.graph != ActualWaveGraphState::Warm
            || self.remaining != 0
            || self.segments != self.launches
            || self.logical_nodes != self.physical_nodes
        {
            return Err(StatisticalEvidenceUnknown::UnsupportedReplay);
        }
        let value = StructuredReplayWorkV1 {
            protocol: "ferrum.structured-replay-work.v1",
            exact_binding: super::wave::exact_binding(shape)?,
            resident_binding: self.resident.clone().finalize().into(),
            replayed_segments: self.segments,
            logical_commands: self.logical,
            native_graph_nodes: self.logical_nodes,
        };
        value.validate_binding(value.exact_binding)?;
        Ok(Some(value))
    }
}

pub(super) fn append_logical_identity(
    hash: &mut Sha256,
    physical: u32,
    command: CostLogicalCommand<'_>,
) {
    // Physical ordinal and logical provider classes are shared model topology.
    // Resident program fingerprints/addresses are deliberately absent here.
    bytes(hash, b"ferrum.structured-logical-command.v1");
    number(hash, u64::from(physical));
    number(hash, u64::from(command.logical_command_ordinal));
    number(hash, u64::from(command.node_index));
    bytes(hash, command.native_op_id.as_bytes());
    bytes(hash, command.batching_form.as_bytes());
    bytes(hash, command.provider.provider_id.as_bytes());
    bytes(hash, command.provider.implementation_fingerprint.as_bytes());
    bytes(hash, command.provider.operation_fingerprint.as_bytes());
    number(hash, u64::from(command.participant_count));
    number(hash, command.compute_dispatch_count);
    number(hash, command.transfer_command_count);
}

#[cfg(test)]
mod tests;
