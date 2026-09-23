//! Provider-only portion of an eager wave, in the actual immutable plan order.
use super::*;
use crate::execution_cost::{CanonicalCostError, CanonicalWaveCostBuilder};
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
        for provider in self.providers() {
            poll_budget()?;
            let route = provider.eager_cost_route_with_ranges(resolved, rows, physical_ranges)?;
            poll_budget()?;
            let Some(route) = route else {
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
