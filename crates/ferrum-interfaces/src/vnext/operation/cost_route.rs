//! Read-only route declarations from the provider selected by a compiled plan.
//! No buffer, request identity, allocator or submission authority is exposed.
mod program_bindings;
mod rows;
mod selected;
pub use program_bindings::{
    coalesce_sorted_program_binding_writes, ProgramBindingCostPatch, ProgramBindingCostWrite,
    ProgramBindingTransferLayout,
};
pub(crate) use rows::{ValidatedCostRows, ValidatedCostRowsBuilder};
pub use selected::SelectedEagerCostRoute;
use std::{collections::BTreeMap, num::NonZeroU64};

use super::{
    foundation::invalid_operation, resolved_value::resource_uses_packed_batch_coordinates,
    AttributeId, ResolvedValueBinding, ResolvedValueRole,
};
use crate::execution_cost::{
    CostCommandPath, CostPhysicalCommand, CostProviderIdentity, MAX_COST_COMMANDS, MAX_COST_ROWS,
};
use crate::vnext::{
    DeviceBatchingForm, DeviceCommandPhase, DeviceNativeOperationId, MemoryPlan, NodeId,
    OperationId, PlanNode, SemanticValue, VNextError,
};

/// Numerical projection only. Constructing one does not admit a sequence or
/// reserve its KV, workspace, output, or Step slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OperationCostWorkRow {
    pub offset: u64,
    pub count: NonZeroU64,
    pub full_input_tokens: NonZeroU64,
}

pub struct OperationCostRouteRequest<'a> {
    node: &'a PlanNode,
    memory: &'a MemoryPlan,
    rows: &'a [OperationCostWorkRow],
    immediate_tokens: u64,
    packed_starts: &'a [u64],
    physical_ranges: Option<&'a crate::vnext::ResourceCostRangeProof>,
}

impl<'a> OperationCostRouteRequest<'a> {
    pub(super) fn new(
        node: &'a PlanNode,
        memory: &'a MemoryPlan,
        rows: &'a ValidatedCostRows<'a>,
    ) -> Self {
        Self {
            node,
            memory,
            rows: rows.rows(),
            immediate_tokens: rows.immediate_tokens(),
            packed_starts: rows.packed_starts(),
            physical_ranges: None,
        }
    }

    pub(super) fn with_physical_ranges(
        mut self,
        ranges: Option<&'a crate::vnext::ResourceCostRangeProof>,
    ) -> Self {
        self.physical_ranges = ranges;
        self
    }

    /// The exact participant-local byte range mapped through the same Step
    /// coordinate translation as actual core readback. Numeric evidence only;
    /// plan-only queries and unproven/request-paged storage return None.
    pub fn binding_physical_range(
        &self,
        role: ResolvedValueRole,
        ordinal: u32,
        component_index: usize,
        participant_index: usize,
        semantic_range: std::ops::Range<u64>,
    ) -> Result<Option<crate::vnext::DeviceCostBufferRange>, VNextError> {
        let binding = self
            .bindings()
            .iter()
            .find(|value| value.role() == role && value.ordinal() == ordinal)
            .ok_or_else(|| invalid_operation("cost range requested unknown binding"))?;
        let component = binding
            .storage()
            .components()
            .get(component_index)
            .ok_or_else(|| invalid_operation("cost range requested unknown component"))?;
        let row = self
            .rows
            .get(participant_index)
            .ok_or_else(|| invalid_operation("cost range participant is out of bounds"))?;
        let Some(base) = self
            .physical_ranges
            .and_then(|ranges| ranges.get(component.resource_id()))
        else {
            return Ok(None);
        };
        let descriptor = self.memory.dynamic_descriptor(component.resource_id());
        let range = if let Some(descriptor) = descriptor {
            if descriptor.lifetime() != crate::vnext::AllocationLifetime::Step {
                return Ok(None);
            }
            let packed_start = self.packed_starts[participant_index];
            super::buffer_view::translate_step_participant_numeric_range(
                descriptor.demand(),
                self.rows.len() as u32,
                self.immediate_tokens,
                participant_index,
                row.offset..row.offset + row.count.get(),
                packed_start,
                semantic_range,
                super::buffer_view::StepParticipantRangeCoordinates::ParticipantLocal,
            )?
        } else {
            semantic_range
        };
        Ok(base.slice(
            range.start,
            range
                .end
                .checked_sub(range.start)
                .ok_or_else(|| invalid_operation("cost range is inverted"))?,
        ))
    }

    /// Complete resident sequence window as page-aligned extents in actual
    /// physical-row order. Adjacent pages stay coalesced: extent count is bounded
    /// by the captured resource limits, never multiplied by context length. Numeric
    /// evidence only: no sequence identity, address or page grants execution.
    /// Missing, partial, or non-paged evidence stays unavailable.
    pub fn binding_sequence_ranges(
        &self,
        role: ResolvedValueRole,
        ordinal: u32,
        participant_index: usize,
        expected_bytes: u64,
        page_bytes: u64,
    ) -> Result<Option<Vec<crate::vnext::DeviceCostBufferRange>>, VNextError> {
        let binding = self
            .bindings()
            .iter()
            .find(|v| v.role() == role && v.ordinal() == ordinal)
            .ok_or_else(|| invalid_operation("cost pages requested unknown binding"))?;
        let [component] = binding.storage().components() else {
            return Ok(None);
        };
        if participant_index >= self.rows.len()
            || expected_bytes == 0
            || page_bytes == 0
            || !expected_bytes.is_multiple_of(page_bytes)
            || component.offset_bytes() != 0
        {
            return Ok(None);
        }
        let Some(descriptor) = self.memory.dynamic_descriptor(component.resource_id()) else {
            return Ok(None);
        };
        if descriptor.lifetime() != crate::vnext::AllocationLifetime::Sequence
            || descriptor.storage().profile().view()
                != (crate::vnext::DynamicStorageView::PagedRegions {
                    block_bytes: page_bytes,
                })
            || descriptor.element_type() != component.element_type()
        {
            return Ok(None);
        }
        let row = self.rows[participant_index];
        let end = row
            .offset
            .checked_add(row.count.get())
            .ok_or_else(|| invalid_operation("cost page frontier overflows"))?;
        if descriptor.evaluate_request_bytes_for_shape(
            crate::vnext::DynamicResourceShape::from_validated(1, end, 0),
        )? != expected_bytes
        {
            return Ok(None);
        }
        let Some(ranges) = self
            .physical_ranges
            .and_then(|proof| proof.sequence(participant_index, component.resource_id()))
        else {
            return Ok(None);
        };
        let mut bytes = 0_u64;
        for range in ranges {
            if !range.length().is_multiple_of(page_bytes) {
                return Ok(None);
            }
            bytes = bytes
                .checked_add(range.length())
                .ok_or_else(|| invalid_operation("cost page extent overflows"))?;
            if bytes > expected_bytes {
                return Ok(None);
            }
        }
        Ok((bytes == expected_bytes && !ranges.is_empty()).then(|| ranges.to_vec()))
    }

    pub fn node_id(&self) -> &NodeId {
        self.node.id()
    }
    pub fn operation_id(&self) -> &OperationId {
        self.node.operation_id()
    }
    pub fn attributes(&self) -> &BTreeMap<AttributeId, SemanticValue> {
        self.node.attributes()
    }
    pub fn bindings(&self) -> &[ResolvedValueBinding] {
        self.node.values()
    }
    pub fn rows(&self) -> &[OperationCostWorkRow] {
        self.rows
    }
    pub fn immediate_tokens(&self) -> u64 {
        self.immediate_tokens
    }

    /// Declared alignment of a contiguous dynamic resource's logical origin.
    /// The resource layer enforces this on backing slices; this is neither a
    /// live allocation nor evidence that two bindings alias or are disjoint.
    /// The caller must account for component and row offsets separately.
    pub fn binding_contiguous_base_alignment(
        &self,
        role: ResolvedValueRole,
        ordinal: u32,
    ) -> Result<Option<NonZeroU64>, VNextError> {
        let binding = self
            .bindings()
            .iter()
            .find(|binding| binding.role() == role && binding.ordinal() == ordinal)
            .ok_or_else(|| invalid_operation("cost alignment requested an unknown binding"))?;
        let [component] = binding.storage().components() else {
            return Ok(None);
        };
        let descriptor = self.memory.dynamic_descriptor(component.resource_id());
        Ok(descriptor
            .filter(|descriptor| {
                matches!(
                    descriptor.storage().profile().view(),
                    crate::vnext::DynamicStorageView::Contiguous
                )
            })
            .and_then(|descriptor| NonZeroU64::new(descriptor.alignment_bytes())))
    }

    /// Same declared coordinate ownership used by actual invocation creation.
    /// This proves a layout choice, never live physical storage or aliasing.
    pub fn binding_uses_packed_batch_coordinates(
        &self,
        role: ResolvedValueRole,
        ordinal: u32,
    ) -> Result<bool, VNextError> {
        let binding = self
            .bindings()
            .iter()
            .find(|binding| binding.role() == role && binding.ordinal() == ordinal)
            .ok_or_else(|| invalid_operation("cost route requested an unknown value binding"))?;
        let [component] = binding.storage().components() else {
            return Err(invalid_operation(
                "cost route coordinate ownership needs one resource component",
            ));
        };
        resource_uses_packed_batch_coordinates(self.memory, component.resource_id())
    }
}

#[cfg(test)]
fn validate_work_rows(rows: &[OperationCostWorkRow]) -> Result<u64, VNextError> {
    ValidatedCostRows::new(rows).map(|rows| rows.immediate_tokens())
}

/// One physical encoder command in an eager route. Host-only bindings are
/// retained as zero-work slots so later physical command indices stay exact.
#[derive(Debug, Clone)]
pub struct OperationCostCommand {
    native_operation: &'static str,
    phase: DeviceCommandPhase,
    batching: DeviceBatchingForm,
    participant_start: u32,
    participant_count: u32,
    token_count: u64,
    compute_dispatch_count: u64,
    transfer_command_count: u64,
    statistical_evidence: Option<crate::execution_cost::SelectedCommandCostEvidenceV1>,
}
// Equality retains the legacy exact contract. Passive statistics must be
// compared explicitly and can never alter an execution/route equality gate.
impl PartialEq for OperationCostCommand {
    fn eq(&self, other: &Self) -> bool {
        self.native_operation == other.native_operation
            && self.phase == other.phase
            && self.batching == other.batching
            && self.participant_start == other.participant_start
            && self.participant_count == other.participant_count
            && self.token_count == other.token_count
            && self.compute_dispatch_count == other.compute_dispatch_count
            && self.transfer_command_count == other.transfer_command_count
    }
}
impl Eq for OperationCostCommand {}

impl OperationCostCommand {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        native_operation: &'static str,
        phase: DeviceCommandPhase,
        batching: DeviceBatchingForm,
        participant_start: u32,
        participant_count: u32,
        token_count: u64,
        compute_dispatch_count: u64,
        transfer_command_count: u64,
    ) -> Result<Self, VNextError> {
        if DeviceNativeOperationId::new(native_operation).is_none()
            || participant_count == 0
            || participant_start.checked_add(participant_count).is_none()
            || token_count == 0
        {
            return Err(invalid_operation(
                "cost route command has invalid identity or work",
            ));
        }
        Ok(Self {
            native_operation,
            phase,
            batching,
            participant_start,
            participant_count,
            token_count,
            compute_dispatch_count,
            transfer_command_count,
            statistical_evidence: None,
        })
    }
    /// Provider declaration from the same selected launch used by the encoder.
    pub fn with_statistical_evidence(
        mut self,
        evidence: crate::execution_cost::SelectedCommandCostEvidenceV1,
    ) -> Result<Self, crate::execution_cost::StatisticalEvidenceUnknown> {
        evidence.validate_command(
            self.token_count,
            self.compute_dispatch_count,
            self.transfer_command_count,
        )?;
        self.statistical_evidence = Some(evidence);
        Ok(self)
    }
    pub fn statistical_evidence(
        &self,
    ) -> Option<&crate::execution_cost::SelectedCommandCostEvidenceV1> {
        self.statistical_evidence.as_ref()
    }
    pub fn native_operation(&self) -> &'static str {
        self.native_operation
    }
    pub fn phase(&self) -> DeviceCommandPhase {
        self.phase
    }
    pub fn batching(&self) -> DeviceBatchingForm {
        self.batching
    }
    pub fn participant_start(&self) -> u32 {
        self.participant_start
    }
    pub fn participant_count(&self) -> u32 {
        self.participant_count
    }
    pub fn token_count(&self) -> u64 {
        self.token_count
    }
    pub fn compute_dispatch_count(&self) -> u64 {
        self.compute_dispatch_count
    }
    pub fn transfer_command_count(&self) -> u64 {
        self.transfer_command_count
    }
    pub fn host_only(&self) -> bool {
        self.compute_dispatch_count == 0 && self.transfer_command_count == 0
    }
    /// None is precisely a host-only slot; callers must still advance the
    /// physical command index. This is a declaration, not observed evidence.
    pub fn canonical_command<'a>(
        &'a self,
        command_index: u32,
        node_index: u32,
        provider: CostProviderIdentity<'a>,
    ) -> Option<CostPhysicalCommand<'a>> {
        (!self.host_only()).then_some(CostPhysicalCommand {
            native_op_id: self.native_operation,
            command_index,
            node_index: Some(node_index),
            command_phase: self.phase,
            provider: Some(provider),
            path: CostCommandPath::Eager,
            participant_start: self.participant_start,
            participant_count: self.participant_count,
            token_count: self.token_count,
            batching_form: self.batching.as_str(),
            compute_dispatch_count: self.compute_dispatch_count,
            transfer_command_count: self.transfer_command_count,
            reusable_graph_node_count: None,
            statistical_evidence: self.statistical_evidence.as_ref(),
        })
    }
}

/// A provider may decline with None when future resource-dependent choices
/// cannot be established. An eager declaration is not a replay/capture claim.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OperationCostRoute {
    commands: Vec<OperationCostCommand>,
    relocatable_binding: Option<usize>,
    program_binding_writes: Option<Vec<ProgramBindingCostWrite>>,
}

impl OperationCostRoute {
    pub fn new(commands: Vec<OperationCostCommand>) -> Result<Self, VNextError> {
        if commands.is_empty()
            || commands.len() > MAX_COST_COMMANDS
            || commands.iter().all(OperationCostCommand::host_only)
        {
            return Err(invalid_operation(
                "cost route must contain bounded physical device work",
            ));
        }
        // EncodedDeviceOperation has exactly one compute slot, preceded by
        // dynamic bindings and followed by result bindings. Core initialization
        // and coalesced program bindings are outside this per-node route.
        let mut compute_seen = false;
        for command in &commands {
            match command.phase() {
                DeviceCommandPhase::DynamicBinding if !compute_seen => {}
                DeviceCommandPhase::Compute if !compute_seen => compute_seen = true,
                DeviceCommandPhase::ResultBinding if compute_seen => {}
                _ => return Err(invalid_operation("cost route has invalid operation phases")),
            }
        }
        if !compute_seen {
            return Err(invalid_operation("cost route has no compute slot"));
        }
        Ok(Self {
            commands,
            relocatable_binding: None,
            program_binding_writes: None,
        })
    }

    /// Identifies the same command the actual encoder passes to
    /// `BatchedOperationInvocation::attach_binding_command`. A compiled
    /// program-binding slot moves this command into the whole-wave prelude;
    /// without that slot it remains a per-node dynamic binding. This is not
    /// permission to infer a binding from an arbitrary command's phase.
    pub fn with_relocatable_binding(mut self, index: usize) -> Result<Self, VNextError> {
        if self.relocatable_binding.is_some()
            || self
                .commands
                .get(index)
                .is_none_or(|command| command.phase() != DeviceCommandPhase::DynamicBinding)
        {
            return Err(invalid_operation(
                "cost route has an invalid relocatable binding",
            ));
        }
        self.relocatable_binding = Some(index);
        Ok(self)
    }

    pub fn commands(&self) -> &[OperationCostCommand] {
        &self.commands
    }

    /// Exact slot-relative writes of the actual typed binding encoder. A
    /// missing declaration cannot be inferred from transfer count or byte cap.
    pub fn with_program_binding_writes(
        mut self,
        writes: Vec<ProgramBindingCostWrite>,
    ) -> Result<Self, VNextError> {
        if self.relocatable_binding.is_none()
            || self.program_binding_writes.is_some()
            || writes.is_empty()
            || writes.len() > MAX_COST_COMMANDS
        {
            return Err(invalid_operation(
                "program binding writes require one bounded relocatable declaration",
            ));
        }
        self.program_binding_writes = Some(writes);
        Ok(self)
    }
    pub(super) fn validate_participants(&self, rows: usize) -> Result<(), VNextError> {
        if self.commands.iter().any(|command| {
            u64::from(command.participant_start) + u64::from(command.participant_count)
                > rows as u64
        }) {
            return Err(invalid_operation(
                "cost route command exceeds projected participants",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn work(offset: u64, count: u64, total: u64) -> OperationCostWorkRow {
        OperationCostWorkRow {
            offset,
            count: NonZeroU64::new(count).unwrap(),
            full_input_tokens: NonZeroU64::new(total).unwrap(),
        }
    }

    #[test]
    fn route_work_checks_real_span_and_storage_boundaries_before_querying_a_provider() {
        assert_eq!(
            validate_work_rows(&[work(1, 4, 5), work(7, 1, 8)]).unwrap(),
            5
        );
        assert!(validate_work_rows(&[work(2, 4, 5)]).is_err());
        assert!(validate_work_rows(&[work(u64::MAX, 1, u64::MAX)]).is_err());
        assert!(validate_work_rows(&[work(0, u64::MAX, u64::MAX), work(0, 1, 1)]).is_err());
        assert_eq!(
            validate_work_rows(&vec![work(0, 1, 1); MAX_COST_ROWS]).unwrap(),
            MAX_COST_ROWS as u64
        );
        assert!(validate_work_rows(&vec![work(0, 1, 1); MAX_COST_ROWS + 1]).is_err());
        assert!(validate_work_rows(&[]).is_err());
    }

    #[test]
    fn route_retains_host_slots_but_never_promotes_them_to_device_work() {
        let host = OperationCostCommand::new(
            "binding",
            DeviceCommandPhase::DynamicBinding,
            DeviceBatchingForm::Scalar,
            0,
            1,
            1,
            0,
            0,
        )
        .unwrap();
        let compute = OperationCostCommand::new(
            "compute",
            DeviceCommandPhase::Compute,
            DeviceBatchingForm::ParticipantLoop,
            0,
            2,
            2,
            2,
            0,
        )
        .unwrap();
        assert!(OperationCostRoute::new(vec![host.clone()]).is_err());
        let route = OperationCostRoute::new(vec![host, compute]).unwrap();
        assert!(route.validate_participants(1).is_err());
        route.validate_participants(2).unwrap();
        let provider = CostProviderIdentity {
            provider_id: "p",
            implementation_fingerprint: "i",
            operation_fingerprint: "o",
        };
        let declared = route
            .commands()
            .iter()
            .enumerate()
            .filter_map(|(index, command)| command.canonical_command(index as u32, 3, provider))
            .collect::<Vec<_>>();
        assert_eq!(declared.len(), 1);
        assert_eq!(declared[0].command_index, 1);
        assert_eq!(declared[0].node_index, Some(3));
        assert_eq!(declared[0].compute_dispatch_count, 2);
    }
}

impl super::ReusableExecutionTopologyView for OperationCostRouteRequest<'_> {
    fn operation_id(&self) -> &OperationId {
        self.node.operation_id()
    }
    fn attributes(&self) -> &BTreeMap<AttributeId, SemanticValue> {
        self.node.attributes()
    }
    fn bindings(&self) -> &[ResolvedValueBinding] {
        self.node.values()
    }
    fn memory_plan(&self) -> &MemoryPlan {
        self.memory
    }
    fn participant_count(&self) -> usize {
        self.rows.len()
    }
    fn immediate_tokens(&self) -> u64 {
        self.immediate_tokens
    }
    fn token_row(&self, index: usize) -> Option<OperationCostWorkRow> {
        self.rows.get(index).copied()
    }
    fn workspace_resource(
        &self,
        workspace: super::ReusableExecutionWorkspaceAddress,
    ) -> Option<&crate::vnext::ResourceId> {
        use super::ReusableExecutionWorkspaceAddress as W;
        match workspace {
            W::Scratch => self.node.scratch_resource(),
            W::Binding => self.node.binding_resource(),
            W::Persistent => self.node.persistent_resource(),
        }
    }
    fn resource_reusable_address_scope(
        &self,
        resource: &crate::vnext::ResourceId,
    ) -> Result<Option<crate::vnext::DeviceReusableAddressScope>, VNextError> {
        if self
            .memory
            .static_allocations()
            .binary_search_by(|a| a.resource_id().cmp(resource))
            .is_ok()
        {
            return Ok(Some(crate::vnext::DeviceReusableAddressScope::Plan));
        }
        if self
            .memory
            .dynamic_descriptors()
            .binary_search_by(|d| d.base_resource_id().cmp(resource))
            .is_err()
        {
            return Err(invalid_operation(
                "future topology references unknown memory resource",
            ));
        }
        Ok(self
            .physical_ranges
            .and_then(|proof| proof.reusable_scope(resource)))
    }
}
