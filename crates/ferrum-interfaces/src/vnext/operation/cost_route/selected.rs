//! Provider-only portion of an eager wave, in the actual immutable plan order.
use super::*;
use crate::execution_cost::{CanonicalCostError, CanonicalWaveCostBuilder, CostLogicalCommand};
use crate::vnext::{BoundOperationProviderSet, DeviceRuntime, ExecutablePlanView};

struct NodeRoute<'a> {
    identity: CostProviderIdentity<'a>,
    route: OperationCostRoute,
}

/// Complete declarations for the selected providers. This does not include
/// core initialization, uploads, coalesced program bindings, or readbacks.
/// The owner must establish those, eager selection, and live capacity before
/// constructing a complete wave prediction. No historical attribution is used.
pub struct SelectedEagerCostRoute<'a> {
    nodes: Vec<NodeRoute<'a>>,
    physical_slots: u32,
}

impl SelectedEagerCostRoute<'_> {
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Includes host-only commands omitted from device attribution.
    pub fn physical_slots(&self) -> u32 {
        self.physical_slots
    }

    pub fn program_binding_patches(
        &self,
        binding_nodes: &[usize],
    ) -> Result<Vec<ProgramBindingCostPatch<'_>>, CanonicalCostError> {
        if binding_nodes.len() > self.nodes.len()
            || binding_nodes.windows(2).any(|pair| pair[0] >= pair[1])
        {
            return Err(CanonicalCostError::InvalidRoute);
        }
        let mut total_writes = 0_usize;
        binding_nodes
            .iter()
            .map(|&node_index| {
                let node = self
                    .nodes
                    .get(node_index)
                    .ok_or(CanonicalCostError::InvalidRoute)?;
                let index = node
                    .route
                    .relocatable_binding
                    .ok_or(CanonicalCostError::InvalidRoute)?;
                let writes = node
                    .route
                    .program_binding_writes
                    .as_deref()
                    .ok_or(CanonicalCostError::InvalidRoute)?;
                total_writes = total_writes
                    .checked_add(writes.len())
                    .filter(|n| *n <= MAX_COST_COMMANDS)
                    .ok_or(CanonicalCostError::Capacity)?;
                Ok(ProgramBindingCostPatch {
                    node_index,
                    command: &node.route.commands()[index],
                    writes,
                })
            })
            .collect()
    }

    /// The runtime supplies its actual merged prelude declaration. All moved
    /// slots disappear from their provider nodes and exactly one merged slot
    /// precedes them; host-only slots elsewhere keep their original position.
    pub fn append_canonical_with_coalesced_program_binding(
        &self,
        canonical: &mut CanonicalWaveCostBuilder,
        first_command_index: u32,
        binding_nodes: &[usize],
        merged: &OperationCostCommand,
    ) -> Result<u32, CanonicalCostError> {
        if binding_nodes.is_empty()
            || merged.phase() != DeviceCommandPhase::DynamicBinding
            || merged.host_only()
        {
            return Err(CanonicalCostError::InvalidRoute);
        }
        self.program_binding_patches(binding_nodes)?;
        let end = first_command_index
            .checked_add(self.physical_slots)
            .and_then(|end| end.checked_sub(binding_nodes.len() as u32))
            .and_then(|end| end.checked_add(1))
            .ok_or(CanonicalCostError::Capacity)?;
        let mut command = merged
            .canonical_command(first_command_index, 0, self.nodes[0].identity)
            .ok_or(CanonicalCostError::InvalidRoute)?;
        command.node_index = None;
        command.provider = None;
        canonical.physical_command(command)?;
        let mut index = first_command_index + 1;
        for (node_index, node) in self.nodes.iter().enumerate() {
            let relocated = binding_nodes.binary_search(&node_index).is_ok();
            for (command_index, command) in node.route.commands().iter().enumerate() {
                if relocated && node.route.relocatable_binding == Some(command_index) {
                    continue;
                }
                if let Some(command) =
                    command.canonical_command(index, node_index as u32, node.identity)
                {
                    canonical.physical_command(command)?;
                }
                index += 1;
            }
        }
        debug_assert_eq!(index, end);
        Ok(end)
    }

    /// Append after the caller's actual core-prefix declaration. Returns the
    /// first following slot, for result readbacks. Host-only slots advance the
    /// index, but never become invented physical device work.
    pub fn append_canonical(
        &self,
        canonical: &mut CanonicalWaveCostBuilder,
        first_command_index: u32,
    ) -> Result<u32, CanonicalCostError> {
        self.append_canonical_with_program_bindings(canonical, first_command_index, &[])
    }

    /// Append the exact unmerged program-binding prelude followed by the
    /// per-node commands. The caller must prove these nodes have compiled
    /// binding slots and that its runtime preserves binding command order and
    /// cardinality. Moving a host-only command preserves its physical slot.
    pub fn append_canonical_with_program_bindings(
        &self,
        canonical: &mut CanonicalWaveCostBuilder,
        first_command_index: u32,
        binding_nodes: &[usize],
    ) -> Result<u32, CanonicalCostError> {
        let end = first_command_index
            .checked_add(self.physical_slots)
            .ok_or(CanonicalCostError::Capacity)?;
        if binding_nodes.len() > self.nodes.len()
            || binding_nodes.windows(2).any(|pair| pair[0] >= pair[1])
            || binding_nodes.iter().any(|&node| {
                self.nodes
                    .get(node)
                    .is_none_or(|node| node.route.relocatable_binding.is_none())
            })
        {
            return Err(CanonicalCostError::InvalidRoute);
        }
        let mut index = first_command_index;
        for &node_index in binding_nodes {
            let node = &self.nodes[node_index];
            let command = &node.route.commands()[node.route.relocatable_binding.unwrap()];
            if let Some(mut command) =
                command.canonical_command(index, node_index as u32, node.identity)
            {
                // Actual core uses push_dynamic_binding for the prelude,
                // outside any provider node attribution boundary.
                command.node_index = None;
                command.provider = None;
                canonical.physical_command(command)?;
            }
            index += 1;
        }
        for (node_index, node) in self.nodes.iter().enumerate() {
            let relocated = binding_nodes.binary_search(&node_index).is_ok();
            for (command_index, command) in node.route.commands().iter().enumerate() {
                if relocated && node.route.relocatable_binding == Some(command_index) {
                    continue;
                }
                if let Some(command) =
                    command.canonical_command(index, node_index as u32, node.identity)
                {
                    canonical.physical_command(command)?;
                }
                index += 1;
            }
        }
        Ok(end)
    }
}

impl<R: DeviceRuntime> BoundOperationProviderSet<R> {
    /// Query every selected node, fail closed on any unavailable declaration,
    /// and retain no partial route. Polls surround each provider callback; the
    /// callback itself remains synchronous, bounded and free of I/O.
    pub fn eager_cost_route<'a>(
        &'a self,
        resolved: &dyn ExecutablePlanView,
        rows: &[OperationCostWorkRow],
        poll_budget: &mut dyn FnMut() -> Result<(), VNextError>,
    ) -> Result<Option<SelectedEagerCostRoute<'a>>, VNextError> {
        self.eager_cost_route_with_ranges(resolved, rows, None, poll_budget)
    }

    pub(crate) fn eager_cost_route_with_ranges<'a>(
        &'a self,
        resolved: &dyn ExecutablePlanView,
        rows: &[OperationCostWorkRow],
        physical_ranges: Option<&crate::vnext::ResourceCostRangeProof>,
        poll_budget: &mut dyn FnMut() -> Result<(), VNextError>,
    ) -> Result<Option<SelectedEagerCostRoute<'a>>, VNextError> {
        validate_work_rows(rows)?;
        let nodes = resolved.execution_plan().payload().nodes();
        if nodes.is_empty() || nodes.len() != self.len() || nodes.len() > MAX_COST_COMMANDS {
            return Err(invalid_operation(
                "cost route requires a bounded complete selected plan",
            ));
        }
        // Check all bindings before invoking any provider. A partial plan or
        // a reordered provider list must not be silently reindexed.
        for (provider, node) in self.providers().iter().zip(nodes) {
            poll_budget()?;
            provider.validate_binding(resolved, node.id())?;
        }
        let mut result = Vec::new();
        result.try_reserve_exact(nodes.len()).map_err(|_| {
            invalid_operation("selected cost route allocation capacity unavailable")
        })?;
        let mut physical_slots = 0_usize;
        for (provider, node) in self.providers().iter().zip(nodes) {
            poll_budget()?;
            let route = match provider.eager_cost_route_with_ranges(resolved, rows, physical_ranges)
            {
                Ok(route) => route,
                Err(error) => {
                    tracing::trace!(
                        node = %node.id(),
                        provider = %provider.descriptor().provider_id(),
                        error = %error,
                        "future selected provider eager route returned error"
                    );
                    return Err(error);
                }
            };
            poll_budget()?;
            let Some(route) = route else {
                tracing::trace!(
                    node = %node.id(),
                    provider = %provider.descriptor().provider_id(),
                    "future selected provider eager route unavailable"
                );
                return Ok(None);
            };
            physical_slots = physical_slots
                .checked_add(route.commands().len())
                .filter(|slots| *slots <= MAX_COST_COMMANDS)
                .ok_or_else(|| invalid_operation("selected cost route exceeds command capacity"))?;
            let descriptor = provider.descriptor();
            result.push(NodeRoute {
                identity: CostProviderIdentity {
                    provider_id: descriptor.provider_id().as_str(),
                    implementation_fingerprint: descriptor.provider_implementation_fingerprint(),
                    operation_fingerprint: descriptor.operation_fingerprint(),
                },
                route,
            });
        }
        Ok(Some(SelectedEagerCostRoute {
            nodes: result,
            physical_slots: physical_slots as u32,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_cost::*;

    fn command(phase: DeviceCommandPhase, dispatches: u64) -> OperationCostCommand {
        OperationCostCommand::new(
            "test.command",
            phase,
            DeviceBatchingForm::Scalar,
            0,
            1,
            1,
            dispatches,
            0,
        )
        .unwrap()
    }

    #[test]
    fn operation_phases_match_actual_encoded_operation_contract() {
        let compute = command(DeviceCommandPhase::Compute, 1);
        let binding = command(DeviceCommandPhase::DynamicBinding, 0);
        let result = command(DeviceCommandPhase::ResultBinding, 0);
        assert!(
            OperationCostRoute::new(vec![binding.clone(), compute.clone(), result.clone()]).is_ok()
        );
        assert!(OperationCostRoute::new(vec![result, compute.clone()]).is_err());
        assert!(OperationCostRoute::new(vec![compute.clone(), binding]).is_err());
        assert!(OperationCostRoute::new(vec![compute.clone(), compute.clone()]).is_err());
        assert!(OperationCostRoute::new(vec![
            command(DeviceCommandPhase::Initialization, 1),
            compute
        ])
        .is_err());
    }

    #[test]
    fn composed_route_preserves_host_only_slots_and_rejects_index_overflow() {
        let route = SelectedEagerCostRoute {
            physical_slots: 3,
            nodes: vec![NodeRoute {
                identity: CostProviderIdentity {
                    provider_id: "fixture",
                    implementation_fingerprint:
                        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                    operation_fingerprint:
                        "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                },
                route: OperationCostRoute::new(vec![
                    command(DeviceCommandPhase::DynamicBinding, 0),
                    command(DeviceCommandPhase::Compute, 1),
                    command(DeviceCommandPhase::ResultBinding, 0),
                ])
                .unwrap(),
            }],
        };
        let mut builder =
            CanonicalWaveCostBuilder::new(0, crate::execution_cost::CostProductOutput::GreedyToken);
        assert_eq!(
            route.append_canonical(&mut builder, u32::MAX - 1),
            Err(CanonicalCostError::Capacity)
        );
        assert_eq!(route.append_canonical(&mut builder, 5), Ok(8));
        // Only slot 6 was attributed; slot 7 was host-only. Appending at 6
        // again is rejected, while the returned slot remains available.
        assert_eq!(
            route.append_canonical(&mut builder, 5),
            Err(CanonicalCostError::InvalidCommand)
        );
    }

    fn identity() -> CostProviderIdentity<'static> {
        CostProviderIdentity {
            provider_id: "fixture",
            implementation_fingerprint:
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            operation_fingerprint:
                "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        }
    }

    fn finish(mut builder: CanonicalWaveCostBuilder) -> CanonicalWaveCostShape {
        builder
            .row(CanonicalCostRow {
                work: ActualRowWork::Decode { kv_tokens: 1 },
                host_policy_signature: [3; 32],
                host_features: None,
                mask_upload_required: false,
                output: CostRowOutput::Decode {
                    requires_full_logits: false,
                    repetition_tokens: 0,
                    repetition_penalty_bits: 1.0_f32.to_bits(),
                },
            })
            .unwrap();
        builder
            .finish(
                ActualWaveKind::Decode,
                ActualWavePath::PlanRuntime,
                ActualWaveGraphState::Disabled,
                ActualWaveRowOrder::Ordered,
                0,
            )
            .unwrap()
    }

    fn two_node_route(binding: OperationCostCommand) -> SelectedEagerCostRoute<'static> {
        SelectedEagerCostRoute {
            physical_slots: 3,
            nodes: vec![
                NodeRoute {
                    identity: identity(),
                    route: OperationCostRoute::new(vec![command(DeviceCommandPhase::Compute, 1)])
                        .unwrap(),
                },
                NodeRoute {
                    identity: identity(),
                    route: OperationCostRoute::new(vec![
                        binding,
                        command(DeviceCommandPhase::Compute, 1),
                    ])
                    .unwrap()
                    .with_relocatable_binding(0)
                    .unwrap(),
                },
            ],
        }
    }

    #[test]
    fn workspace_binding_prelude_moves_host_only_slot_without_fabricating_work() {
        let route = two_node_route(command(DeviceCommandPhase::DynamicBinding, 0));
        let new_builder = || CanonicalWaveCostBuilder::new(0, CostProductOutput::GreedyToken);
        let mut predicted = new_builder();
        assert_eq!(
            route.append_canonical_with_program_bindings(&mut predicted, 5, &[1]),
            Ok(8)
        );
        let mut actual_order = new_builder();
        // Actual core pushes the host binding at slot 5 before both provider
        // computes. The host-only slot contributes no physical attribution.
        for (slot, node) in [(6, 0), (7, 1)] {
            actual_order
                .physical_command(
                    command(DeviceCommandPhase::Compute, 1)
                        .canonical_command(slot, node, identity())
                        .unwrap(),
                )
                .unwrap();
        }
        let predicted = finish(predicted);
        assert_eq!(predicted, finish(actual_order));
        let mut unbound = new_builder();
        route.append_canonical(&mut unbound, 5).unwrap();
        assert_ne!(predicted, finish(unbound));
    }

    #[test]
    fn physical_binding_prelude_retains_work_without_provider_node_attribution() {
        let binding = OperationCostCommand::new(
            "test.binding",
            DeviceCommandPhase::DynamicBinding,
            DeviceBatchingForm::Scalar,
            0,
            1,
            1,
            0,
            1,
        )
        .unwrap();
        let route = two_node_route(binding.clone());
        let new_builder = || CanonicalWaveCostBuilder::new(0, CostProductOutput::GreedyToken);
        let mut predicted = new_builder();
        route
            .append_canonical_with_program_bindings(&mut predicted, 5, &[1])
            .unwrap();
        let mut actual_order = new_builder();
        let mut prelude = binding.canonical_command(5, 1, identity()).unwrap();
        prelude.node_index = None;
        prelude.provider = None;
        actual_order.physical_command(prelude).unwrap();
        for (slot, node) in [(6, 0), (7, 1)] {
            actual_order
                .physical_command(
                    command(DeviceCommandPhase::Compute, 1)
                        .canonical_command(slot, node, identity())
                        .unwrap(),
                )
                .unwrap();
        }
        assert_eq!(finish(predicted), finish(actual_order));
    }

    #[test]
    fn workspace_prelude_rejects_missing_duplicate_and_invalid_binding_declarations() {
        let route = two_node_route(command(DeviceCommandPhase::DynamicBinding, 0));
        for nodes in [&[0][..], &[1, 1], &[2], &[1, 0]] {
            let mut builder = CanonicalWaveCostBuilder::new(0, CostProductOutput::GreedyToken);
            assert_eq!(
                route.append_canonical_with_program_bindings(&mut builder, 0, nodes),
                Err(CanonicalCostError::InvalidRoute)
            );
            // Validation precedes mutation; a correct declaration can reuse it.
            assert_eq!(
                route.append_canonical_with_program_bindings(&mut builder, 0, &[1]),
                Ok(3)
            );
        }
        assert!(
            OperationCostRoute::new(vec![command(DeviceCommandPhase::Compute, 1)])
                .unwrap()
                .with_relocatable_binding(0)
                .is_err()
        );
        assert!(route.nodes[1]
            .route
            .clone()
            .with_relocatable_binding(0)
            .is_err());
    }

    #[test]
    fn merged_binding_keeps_one_core_slot_then_exact_provider_and_host_slots() {
        let binding = OperationCostCommand::new(
            "test.binding",
            DeviceCommandPhase::DynamicBinding,
            DeviceBatchingForm::ParticipantLoop,
            0,
            1,
            1,
            0,
            1,
        )
        .unwrap();
        let route = SelectedEagerCostRoute {
            physical_slots: 5,
            nodes: (0..2)
                .map(|index| NodeRoute {
                    identity: identity(),
                    route: OperationCostRoute::new(if index == 0 {
                        vec![
                            binding.clone(),
                            command(DeviceCommandPhase::Compute, 1),
                            command(DeviceCommandPhase::ResultBinding, 0),
                        ]
                    } else {
                        vec![binding.clone(), command(DeviceCommandPhase::Compute, 1)]
                    })
                    .unwrap()
                    .with_relocatable_binding(0)
                    .unwrap()
                    .with_program_binding_writes(vec![ProgramBindingCostWrite::new(0, 4).unwrap()])
                    .unwrap(),
                })
                .collect(),
        };
        let merged = OperationCostCommand::new(
            "test.coalesced",
            DeviceCommandPhase::DynamicBinding,
            DeviceBatchingForm::ParticipantLoop,
            0,
            1,
            1,
            0,
            2,
        )
        .unwrap();
        let mut predicted = CanonicalWaveCostBuilder::new(0, CostProductOutput::GreedyToken);
        assert_eq!(
            route.append_canonical_with_coalesced_program_binding(
                &mut predicted,
                5,
                &[0, 1],
                &merged
            ),
            Ok(9)
        );
        let mut actual = CanonicalWaveCostBuilder::new(0, CostProductOutput::GreedyToken);
        let mut prelude = merged.canonical_command(5, 0, identity()).unwrap();
        prelude.node_index = None;
        prelude.provider = None;
        actual.physical_command(prelude).unwrap();
        for (slot, node) in [(6, 0), (8, 1)] {
            actual
                .physical_command(
                    command(DeviceCommandPhase::Compute, 1)
                        .canonical_command(slot, node, identity())
                        .unwrap(),
                )
                .unwrap();
        }
        assert_eq!(finish(predicted), finish(actual));
        let patches = route.program_binding_patches(&[0, 1]).unwrap();
        assert_eq!(
            patches.iter().map(|p| p.node_index).collect::<Vec<_>>(),
            [0, 1]
        );
        assert_eq!(
            patches[0].writes,
            &[ProgramBindingCostWrite::new(0, 4).unwrap()]
        );
    }

    #[test]
    fn merged_binding_requires_explicit_write_evidence_before_mutating_canonical() {
        let binding = OperationCostCommand::new(
            "test.binding",
            DeviceCommandPhase::DynamicBinding,
            DeviceBatchingForm::Scalar,
            0,
            1,
            1,
            0,
            1,
        )
        .unwrap();
        let route = two_node_route(binding.clone());
        let mut builder = CanonicalWaveCostBuilder::new(0, CostProductOutput::GreedyToken);
        assert_eq!(
            route.append_canonical_with_coalesced_program_binding(&mut builder, 0, &[1], &binding),
            Err(CanonicalCostError::InvalidRoute)
        );
        assert_eq!(route.append_canonical(&mut builder, 0), Ok(3));
        assert!(
            OperationCostRoute::new(vec![command(DeviceCommandPhase::Compute, 1)])
                .unwrap()
                .with_program_binding_writes(vec![ProgramBindingCostWrite::new(0, 4).unwrap()])
                .is_err()
        );
    }
}

impl<R: DeviceRuntime> BoundOperationProviderSet<R> {
    /// Proves every provider excludes its complete compute node from reusable
    /// segments using the same selector as actual dispatch. This numerical
    /// result holds only under the caller's live resource-view fence.
    pub(crate) fn future_has_only_eager_boundaries(
        &self,
        resolved: &dyn ExecutablePlanView,
        rows: &[OperationCostWorkRow],
        physical_ranges: Option<&crate::vnext::ResourceCostRangeProof>,
        poll: &mut dyn FnMut() -> Result<(), VNextError>,
    ) -> Result<bool, VNextError> {
        validate_work_rows(rows)?;
        let plan = resolved.execution_plan();
        let nodes = plan.payload().nodes();
        if nodes.len() != self.len() || nodes.is_empty() || nodes.len() > MAX_COST_COMMANDS {
            return Err(invalid_operation(
                "future eager boundary requires the complete provider plan",
            ));
        }
        for (provider, node) in self.providers().iter().zip(nodes) {
            poll()?;
            provider.validate_binding(resolved, node.id())?;
            let request = OperationCostRouteRequest::new(node, plan.payload().memory(), rows)?
                .with_physical_ranges(physical_ranges);
            let selected = provider
                .provider()
                .reusable_execution_cost_topology(request)?;
            poll()?;
            if selected != Some(super::super::ReusableExecutionTopology::EagerBoundary) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Derives the actual program ID from numerical work and the selected
    /// Invocation arena. No claimed Step or execution authority is fabricated.
    pub(crate) fn future_reusable_program_id(
        &self,
        resolved: &dyn ExecutablePlanView,
        rows: &[OperationCostWorkRow],
        physical_ranges: Option<&crate::vnext::ResourceCostRangeProof>,
        layout: &crate::vnext::ProgramBindingLayout,
        slot: &crate::vnext::LaneStableArenaSlotIdentity,
        lane: crate::vnext::ExecutionLaneId,
        poll: &mut dyn FnMut() -> Result<(), VNextError>,
    ) -> Result<Option<(crate::vnext::DeviceReusableExecutionProgramId, Vec<u32>)>, VNextError>
    {
        use crate::vnext::*;
        let tokens = validate_work_rows(rows)?;
        let plan = resolved.execution_plan();
        let nodes = plan.payload().nodes();
        if nodes.len() != self.len()
            || nodes.is_empty()
            || nodes.len() > MAX_COST_COMMANDS
            || slot.lane_id() != lane
            || slot.reusable_execution_bucket_id() != layout.reusable_execution_bucket_id()
            || slot.lifetime() != AllocationLifetime::Invocation
        {
            return Err(invalid_operation(
                "future program identity differs from actual selected arena",
            ));
        }
        let mut digest = super::super::topology_digest::ReusableTopologyDigest::new(nodes.len())?;
        let mut eager = Vec::new();
        for (index, (provider, node)) in self.providers().iter().zip(nodes).enumerate() {
            poll()?;
            provider.validate_binding(resolved, node.id())?;
            let request = OperationCostRouteRequest::new(node, plan.payload().memory(), rows)?
                .with_physical_ranges(physical_ranges);
            let topology = match provider
                .provider()
                .reusable_execution_cost_topology(request)
            {
                Ok(topology) => topology,
                Err(error) => {
                    tracing::trace!(
                        node = %node.id(),
                        provider = %provider.descriptor().provider_id(),
                        error = %error,
                        "future selected provider graph topology returned error"
                    );
                    return Err(error);
                }
            };
            let Some(topology) = topology else {
                tracing::trace!(
                    node = %node.id(),
                    provider = %provider.descriptor().provider_id(),
                    "future selected provider graph topology unavailable"
                );
                return Ok(None);
            };
            poll()?;
            if topology == ReusableExecutionTopology::EagerBoundary {
                eager.push(
                    u32::try_from(index)
                        .map_err(|_| invalid_operation("topology node overflow"))?,
                );
            }
            digest.append(
                index,
                index,
                node.id(),
                provider.descriptor().provider_id(),
                node.provider_execution_semantics().replay_equivalence(),
                &topology,
            )?;
        }
        let id = DeviceReusableExecutionProgramId::new(
            plan.plan_hash().clone(),
            resolved.device().runtime_implementation_fingerprint.clone(),
            lane,
            layout.reusable_execution_bucket_id().clone(),
            layout.fingerprint().to_owned(),
            slot.layout_fingerprint().to_owned(),
            slot.slot_id(),
            u32::try_from(rows.len()).map_err(|_| invalid_operation("row count overflow"))?,
            tokens,
            0,
        )?
        .with_topology_fingerprint(digest.finish());
        Ok(Some((id, eager)))
    }
}

impl SelectedEagerCostRoute<'_> {
    /// Mirror core's actual uploaded-segment dispatch: shared binding prelude,
    /// all segment dynamic bindings, one replay, then segment result bindings.
    /// Every logical compute row must independently match the current provider
    /// selector. A copied catalog row alone is never a future-work declaration.
    pub(crate) fn append_canonical_warm_program(
        &self,
        canonical: &mut CanonicalWaveCostBuilder,
        first: u32,
        binding_nodes: &[usize],
        merged: Option<&OperationCostCommand>,
        graph: &crate::vnext::DeviceCostGraphProgram,
        native_operation: &str,
        participants: u32,
        tokens: u64,
        poll: &mut dyn FnMut() -> Result<(), VNextError>,
    ) -> Result<(u32, Vec<u32>), VNextError> {
        use crate::vnext::*;
        let invalid =
            || invalid_operation("future warm program differs from selected provider route");
        let append_error = |_| invalid_operation("future warm canonical route exceeds contract");
        let program = graph.program();
        if program.node_count() as usize != self.nodes.len()
            || !program.is_determinism_ready()
            || graph.uploaded_segments().len() != program.segments().len()
        {
            return Err(invalid());
        }
        let mut index = first;
        let mut replay_indices = Vec::new();
        self.program_binding_patches(binding_nodes)
            .map_err(append_error)?;
        if let Some(merged) = merged {
            let mut command = merged
                .canonical_command(index, 0, self.nodes[0].identity)
                .ok_or_else(invalid)?;
            command.node_index = None;
            command.provider = None;
            canonical.physical_command(command).map_err(append_error)?;
            index = index.checked_add(1).ok_or_else(invalid)?;
        } else {
            for &n in binding_nodes {
                poll()?;
                let node = self.nodes.get(n).ok_or_else(invalid)?;
                let c = &node.route.commands[node.route.relocatable_binding.ok_or_else(invalid)?];
                if let Some(mut c) = c.canonical_command(index, n as u32, node.identity) {
                    c.node_index = None;
                    c.provider = None;
                    canonical.physical_command(c).map_err(append_error)?;
                }
                index = index.checked_add(1).ok_or_else(invalid)?;
            }
        }
        let mut n = 0_usize;
        let mut segment_index = 0_usize;
        while n < self.nodes.len() {
            poll()?;
            let segment = graph
                .uploaded_segments()
                .get(segment_index)
                .filter(|s| s.segment().start_node_index() as usize == n);
            if let Some(uploaded) = segment {
                let end = uploaded.segment().end_node_index() as usize;
                let logical = uploaded.logical_commands();
                if end > self.nodes.len()
                    || logical.len() != end - n
                    || program.segments().get(segment_index) != Some(uploaded.segment())
                {
                    return Err(invalid());
                }
                let mut graph_nodes = 0_u64;
                for (offset, row) in logical.iter().enumerate() {
                    poll()?;
                    let node_index = n + offset;
                    let node = &self.nodes[node_index];
                    let compute = node
                        .route
                        .commands
                        .iter()
                        .find(|c| c.phase() == DeviceCommandPhase::Compute)
                        .ok_or_else(invalid)?;
                    if row.node_index() != node_index as u32
                        || row.logical_command_ordinal() != offset as u32
                        || compute.host_only()
                        || compute.participant_start() != 0
                        || row.native_op_id() != compute.native_operation()
                        || row.batching_form() != compute.batching()
                        || row.participant_count() != compute.participant_count()
                        || row.token_count() != compute.token_count()
                        || row.compute_dispatch_count() != compute.compute_dispatch_count()
                        || row.transfer_command_count() != compute.transfer_command_count()
                    {
                        return Err(invalid());
                    }
                    graph_nodes = graph_nodes
                        .checked_add(row.reusable_graph_node_count())
                        .ok_or_else(invalid)?;
                    let binding_member = program
                        .per_wave_binding_node_indices()
                        .binary_search(&(node_index as u32))
                        .is_ok();
                    if !binding_member && node.route.commands.len() != 1 {
                        return Err(invalid());
                    }
                    for (ci, c) in node.route.commands.iter().enumerate() {
                        if c.phase() != DeviceCommandPhase::DynamicBinding
                            || (binding_nodes.binary_search(&node_index).is_ok()
                                && node.route.relocatable_binding == Some(ci))
                        {
                            continue;
                        }
                        if let Some(c) =
                            c.canonical_command(index, n as u32, self.nodes[n].identity)
                        {
                            canonical.physical_command(c).map_err(append_error)?;
                        }
                        index = index.checked_add(1).ok_or_else(invalid)?;
                    }
                }
                let replay_index = index;
                canonical
                    .physical_command(CostPhysicalCommand {
                        native_op_id: native_operation,
                        command_index: index,
                        node_index: Some(n as u32),
                        command_phase: DeviceCommandPhase::Compute,
                        provider: Some(self.nodes[n].identity),
                        path: CostCommandPath::Replayed,
                        participant_start: 0,
                        participant_count: participants,
                        token_count: tokens,
                        batching_form: DeviceBatchingForm::ParticipantLoop.as_str(),
                        compute_dispatch_count: 1,
                        transfer_command_count: 0,
                        reusable_graph_node_count: Some(graph_nodes),
                        statistical_evidence: None, // Logical selected work is appended separately.
                    })
                    .map_err(append_error)?;
                index = index.checked_add(1).ok_or_else(invalid)?;
                replay_indices.push(replay_index);
                for ni in n..end {
                    let node = &self.nodes[ni];
                    for c in &node.route.commands {
                        if c.phase() == DeviceCommandPhase::ResultBinding {
                            if let Some(c) =
                                c.canonical_command(index, n as u32, self.nodes[n].identity)
                            {
                                canonical.physical_command(c).map_err(append_error)?;
                            }
                            index = index.checked_add(1).ok_or_else(invalid)?;
                        }
                    }
                }
                n = end;
                segment_index += 1;
            } else {
                if program
                    .eager_boundary_node_indices()
                    .binary_search(&(n as u32))
                    .is_err()
                {
                    return Err(invalid());
                }
                let node = &self.nodes[n];
                for (ci, c) in node.route.commands.iter().enumerate() {
                    if binding_nodes.binary_search(&n).is_ok()
                        && node.route.relocatable_binding == Some(ci)
                    {
                        continue;
                    }
                    if let Some(c) = c.canonical_command(index, n as u32, node.identity) {
                        canonical.physical_command(c).map_err(append_error)?;
                    }
                    index = index.checked_add(1).ok_or_else(invalid)?;
                }
                n += 1;
            }
            if index as usize > MAX_COST_COMMANDS {
                return Err(invalid());
            }
        }
        if segment_index != graph.uploaded_segments().len() {
            return Err(invalid());
        }
        Ok((index, replay_indices))
    }
}

#[cfg(test)]
mod warm_tests;

impl SelectedEagerCostRoute<'_> {
    /// Canonical wire requires all physical commands, including staged
    /// readbacks, before any logical replay rows. Keep that ordering explicit.
    pub(crate) fn append_warm_logical(
        &self,
        canonical: &mut CanonicalWaveCostBuilder,
        graph: &crate::vnext::DeviceCostGraphProgram,
        indices: &[u32],
        poll: &mut dyn FnMut() -> Result<(), VNextError>,
    ) -> Result<(), VNextError> {
        let invalid = || invalid_operation("warm logical continuation differs from physical route");
        if indices.len() != graph.uploaded_segments().len() {
            return Err(invalid());
        }
        for (uploaded, &index) in graph.uploaded_segments().iter().zip(indices) {
            poll()?;
            let logical = uploaded.logical_commands();
            canonical
                .replay_segment(
                    index,
                    uploaded.reusable_executable_fingerprint(),
                    logical.len(),
                )
                .map_err(|_| invalid())?;
            for row in logical {
                poll()?;
                let node = self
                    .nodes
                    .get(row.node_index() as usize)
                    .ok_or_else(invalid)?;
                let mut commands = node
                    .route
                    .commands
                    .iter()
                    .filter(|command| command.phase() == DeviceCommandPhase::Compute);
                let compute = commands.next().ok_or_else(invalid)?;
                if commands.next().is_some() {
                    return Err(invalid());
                }
                let mut projected = CostLogicalCommand::from_attribution(row, node.identity);
                // Fresh provider route work must match the resident fixed
                // geometry/scalars. Do not replay captured context work.
                projected.statistical_evidence =
                    row.bind_current_cost_evidence(compute.statistical_evidence());
                canonical
                    .logical_command(projected)
                    .map_err(|_| invalid())?;
            }
        }
        Ok(())
    }
}
