//! Per-projection provider selection. A result never accepts replacement query
//! inputs and is dropped before this projection returns to search or replay.
use super::*;
use crate::vnext::*;

#[cfg(test)]
mod tests;

pub(crate) struct SelectedFutureCostRoute<'query, R: DeviceRuntime> {
    providers: &'query BoundOperationProviderSet<R>,
    resolved: &'query dyn ExecutablePlanView,
    rows: &'query ValidatedCostRows<'query>,
    route: SelectedEagerCostRoute<'query>,
    topologies: Option<Vec<Option<ReusableExecutionTopology>>>,
}

impl<R: DeviceRuntime> BoundOperationProviderSet<R> {
    pub(crate) fn future_cost_route_with_ranges<'query>(
        &'query self,
        resolved: &'query dyn ExecutablePlanView,
        rows: &'query ValidatedCostRows<'query>,
        physical_ranges: Option<&ResourceCostRangeProof>,
        topology: OperationCostTopologyRequirement,
        poll: &mut dyn FnMut() -> Result<(), VNextError>,
    ) -> Result<Option<SelectedFutureCostRoute<'query, R>>, VNextError> {
        poll()?;
        let nodes = self.prepared_cost_nodes(resolved)?;
        let mut selected = Vec::new();
        selected
            .try_reserve_exact(nodes.len())
            .map_err(|_| invalid_operation("selected future cost route allocation unavailable"))?;
        let mut topologies = match topology {
            OperationCostTopologyRequirement::NotRequested => None,
            OperationCostTopologyRequirement::Required => {
                let mut values = Vec::new();
                values.try_reserve_exact(nodes.len()).map_err(|_| {
                    invalid_operation("selected future topology allocation unavailable")
                })?;
                Some(values)
            }
        };
        let mut physical_slots = 0_usize;
        for (provider, node) in self.providers().iter().zip(nodes) {
            poll()?;
            let request = OperationCostRouteRequest::new(
                node,
                resolved.execution_plan().payload().memory(),
                rows,
                provider.dispatch(),
                provider.cost_data(),
            )
            .with_physical_ranges(physical_ranges);
            let selection = match provider
                .provider()
                .future_cost_selection(request, topology, poll)
            {
                Ok(selection) => selection,
                Err(error) => {
                    tracing::trace!(
                        node = %node.id(),
                        provider = %provider.descriptor().provider_id(),
                        error = %error,
                        "future selected provider cost/topology selection returned error"
                    );
                    return Err(error);
                }
            };
            poll()?;
            let Some(selection) = selection else {
                tracing::trace!(
                    node = %node.id(),
                    provider = %provider.descriptor().provider_id(),
                    "future selected provider cost/topology selection unavailable"
                );
                return Ok(None);
            };
            selection.route.validate_participants(rows.rows().len())?;
            physical_slots = physical_slots
                .checked_add(selection.route.commands().len())
                .filter(|slots| *slots <= MAX_COST_COMMANDS)
                .ok_or_else(|| invalid_operation("selected future cost route exceeds capacity"))?;
            if let Some(values) = &mut topologies {
                values.push(selection.topology);
            }
            let descriptor = provider.descriptor();
            selected.push(NodeRoute {
                identity: CostProviderIdentity {
                    provider_id: descriptor.provider_id().as_str(),
                    implementation_fingerprint: descriptor.provider_implementation_fingerprint(),
                    operation_fingerprint: descriptor.operation_fingerprint(),
                },
                route: selection.route,
            });
        }
        Ok(Some(SelectedFutureCostRoute {
            providers: self,
            resolved,
            rows,
            route: SelectedEagerCostRoute {
                nodes: selected,
                physical_slots: physical_slots as u32,
            },
            topologies,
        }))
    }
}

impl<R: DeviceRuntime> SelectedFutureCostRoute<'_, R> {
    pub(crate) fn route(&self) -> &SelectedEagerCostRoute<'_> {
        &self.route
    }

    pub(crate) fn only_eager_boundaries(
        &self,
        poll: &mut dyn FnMut() -> Result<(), VNextError>,
    ) -> Result<bool, VNextError> {
        poll()?;
        let topologies = self
            .topologies
            .as_ref()
            .ok_or_else(|| invalid_operation("future route did not request topology evidence"))?;
        for topology in topologies {
            poll()?;
            if *topology != Some(ReusableExecutionTopology::EagerBoundary) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// Only current arena/layout facts enter after selection. Provider, plan,
    /// work and topology are inseparable from this private query-bound result.
    pub(crate) fn reusable_program_id(
        &self,
        layout: &ProgramBindingLayout,
        slot: &LaneStableArenaSlotIdentity,
        lane: ExecutionLaneId,
        poll: &mut dyn FnMut() -> Result<(), VNextError>,
    ) -> Result<Option<(DeviceReusableExecutionProgramId, Vec<u32>)>, VNextError> {
        poll()?;
        if slot.lane_id() != lane
            || slot.reusable_execution_bucket_id() != layout.reusable_execution_bucket_id()
            || slot.lifetime() != AllocationLifetime::Invocation
        {
            return Err(invalid_operation(
                "future program identity differs from actual selected arena",
            ));
        }
        let topologies = self
            .topologies
            .as_ref()
            .ok_or_else(|| invalid_operation("future route did not request topology evidence"))?;
        let mut digest =
            super::super::super::topology_digest::ReusableTopologyDigest::new(topologies.len())?;
        let mut eager = Vec::new();
        for (index, topology) in topologies.iter().enumerate() {
            poll()?;
            let Some(topology) = topology else {
                return Ok(None);
            };
            if *topology == ReusableExecutionTopology::EagerBoundary {
                eager.push(
                    u32::try_from(index)
                        .map_err(|_| invalid_operation("topology node overflow"))?,
                );
            }
            digest.append_prepared(self.providers.prepared_topology_node(index), topology)?;
        }
        let plan = self.resolved.execution_plan();
        let id = DeviceReusableExecutionProgramId::new(
            plan.plan_hash().clone(),
            self.resolved
                .device()
                .runtime_implementation_fingerprint
                .clone(),
            lane,
            layout.reusable_execution_bucket_id().clone(),
            layout.fingerprint().to_owned(),
            slot.layout_fingerprint().to_owned(),
            slot.slot_id(),
            u32::try_from(self.rows.rows().len())
                .map_err(|_| invalid_operation("row count overflow"))?,
            self.rows.immediate_tokens(),
            0,
        )?
        .with_topology_fingerprint(digest.finish());
        Ok(Some((id, eager)))
    }
}
