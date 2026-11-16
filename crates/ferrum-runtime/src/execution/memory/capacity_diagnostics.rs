//! Read-only explanations of a compiled plan. These observations do
//! not replace its capacity calculation, storage proof or original error.
use super::*;

/// A sequential node-order census explains simultaneous payloads separately
/// from charged slot sizes. It is not a proof of reuse: completion ordering,
/// aliasing, alignment, and retained ownership remain the planner's authority.
/// Include accepted plans too; their physical admission can still encounter
/// other live device owners that the static plan cannot account for.
pub(super) fn trace_step_liveness(
    pools: &[DynamicBackingPoolSpec],
    descriptors: &[DynamicResourceDescriptor],
    nodes: &[PlanNode],
    retained: &BTreeSet<ResourceId>,
) {
    if nodes.is_empty() || !tracing::enabled!(tracing::Level::DEBUG) {
        return;
    }
    struct Access<'a> {
        first: usize,
        last: usize,
        external: bool,
        values: BTreeSet<&'a str>,
    }
    let produced = nodes
        .iter()
        .flat_map(|node| &node.values)
        .filter(|value| value.role() == ResolvedValueRole::Output)
        .map(|value| value.value_id())
        .collect::<BTreeSet<_>>();
    let mut accesses = BTreeMap::<&ResourceId, Access<'_>>::new();
    for (index, node) in nodes.iter().enumerate() {
        for resource in &node.resources {
            let access = accesses.entry(resource).or_insert_with(|| Access {
                first: index,
                last: index,
                external: false,
                values: BTreeSet::new(),
            });
            access.last = index;
        }
        for value in &node.values {
            for component in value.storage().components() {
                if let Some(access) = accesses.get_mut(component.resource_id()) {
                    access.external |= !produced.contains(value.value_id());
                    access.values.insert(value.value_id().as_str());
                }
            }
        }
    }
    let by_id = descriptors
        .iter()
        .map(|descriptor| (&descriptor.base_resource_id, descriptor))
        .collect::<BTreeMap<_, _>>();
    for pool in pools {
        let shared = pool
            .step_resource_slots()
            .iter()
            .filter(|slot| slot.kind() == StepResourceSlotKind::OrderedSingleFenceStepWave)
            .flat_map(|slot| slot.resource_ids())
            .collect::<BTreeSet<_>>();
        let intervals = descriptors
            .iter()
            .filter(|descriptor| {
                descriptor.pool_id == *pool.pool_id()
                    && descriptor.lifetime == AllocationLifetime::Step
            })
            .filter_map(|descriptor| {
                let access = accesses.get(&descriptor.base_resource_id)?;
                let bytes = descriptor.minimum_request_bytes().ok()?;
                let first = if access.external { 0 } else { access.first };
                let last = if (access.external && !shared.contains(&descriptor.base_resource_id))
                    || retained.contains(&descriptor.base_resource_id)
                {
                    nodes.len() - 1
                } else {
                    access.last
                };
                Some((descriptor, bytes, first, last, access))
            })
            .collect::<Vec<_>>();
        if intervals.is_empty() {
            continue;
        }
        let mut deltas = vec![0_i128; nodes.len() + 1];
        for (_, bytes, first, last, _) in &intervals {
            deltas[*first] += i128::from(*bytes);
            deltas[*last + 1] -= i128::from(*bytes);
        }
        let (mut live, mut peak, mut peak_index) = (0_i128, 0_i128, 0);
        for (index, delta) in deltas.iter().take(nodes.len()).enumerate() {
            live += delta;
            if live > peak {
                peak = live;
                peak_index = index;
            }
        }
        tracing::debug!(
            node_count = nodes.len(),
            pool = ?pool.pool_id(),
            element_type = ?pool.compatibility().element_type(),
            usage = ?pool.compatibility().usage(),
            step_bytes = pool.minimum_step_bytes(),
            sequential_live_peak_bytes = %peak,
            peak_node = %nodes[peak_index].id,
            "step interval census"
        );
        let mut peak_members = intervals
            .iter()
            .filter(|(_, _, first, last, _)| *first <= peak_index && peak_index <= *last)
            .collect::<Vec<_>>();
        peak_members.sort_by_key(|(_, bytes, _, _, _)| std::cmp::Reverse(*bytes));
        for (descriptor, bytes, first, last, access) in peak_members.into_iter().take(12) {
            tracing::debug!(
                pool = ?pool.pool_id(),
                bytes,
                resource = %descriptor.base_resource_id,
                values = ?access.values.iter().take(8).collect::<Vec<_>>(),
                external = access.external,
                retained = retained.contains(&descriptor.base_resource_id),
                first_node = %nodes[*first].id,
                last_node = %nodes[*last].id,
                "step interval peak member"
            );
        }
        let mut slots = pool
            .step_resource_slots()
            .iter()
            .filter_map(|slot| {
                let (bytes, descriptor) = slot
                    .resource_ids()
                    .iter()
                    .filter_map(|id| by_id.get(id).copied())
                    .filter_map(|descriptor| {
                        Some((descriptor.minimum_request_bytes().ok()?, descriptor))
                    })
                    .max_by_key(|(bytes, _)| *bytes)?;
                Some((bytes, descriptor, slot))
            })
            .collect::<Vec<_>>();
        slots.sort_by_key(|(bytes, _, _)| std::cmp::Reverse(*bytes));
        for (bytes, descriptor, slot) in slots.into_iter().take(12) {
            let Some(access) = accesses.get(&descriptor.base_resource_id) else {
                continue;
            };
            tracing::debug!(
                pool = ?pool.pool_id(),
                bytes,
                kind = ?slot.kind(),
                member_count = slot.resource_ids().len(),
                resource = %descriptor.base_resource_id,
                representative_values = ?access.values.iter().take(8).collect::<Vec<_>>(),
                first_use = %nodes[access.first].id,
                last_use = %nodes[access.last].id,
                "step interval charged slot"
            );
        }
    }
}

pub(super) fn trace_shortfall(
    pools: &[DynamicBackingPoolSpec],
    descriptors: &[DynamicResourceDescriptor],
    nodes: &[PlanNode],
) {
    if !tracing::enabled!(tracing::Level::DEBUG) {
        return;
    }
    let by_id = descriptors
        .iter()
        .map(|descriptor| (&descriptor.base_resource_id, descriptor))
        .collect::<BTreeMap<_, _>>();
    for pool in pools {
        let mut slots = pool
            .step_resource_slots()
            .iter()
            .filter_map(|slot| {
                let largest = slot
                    .resource_ids()
                    .iter()
                    .filter_map(|id| by_id.get(id).copied())
                    .filter_map(|descriptor| {
                        descriptor
                            .minimum_request_bytes()
                            .ok()
                            .map(|bytes| (bytes, descriptor))
                    })
                    .max_by_key(|(bytes, _)| *bytes)?;
                Some((largest.0, largest.1, slot))
            })
            .collect::<Vec<_>>();
        slots.sort_by_key(|(bytes, _, _)| std::cmp::Reverse(*bytes));
        tracing::debug!(
            pool = ?pool.pool_id(),
            element_type = ?pool.compatibility().element_type(),
            usage = ?pool.compatibility().usage(),
            alignment_bytes = pool.compatibility().alignment_bytes(),
            storage_profile = ?pool.compatibility().profile(),
            layout_fingerprint = pool.compatibility().logical_layout_fingerprint(),
            request_bytes = pool.minimum_request_bytes(),
            sequence_bytes = pool.minimum_sequence_bytes(),
            step_bytes = pool.minimum_step_bytes(),
            invocation_peak_bytes = pool.minimum_invocation_peak_bytes(),
            slot_count = slots.len(),
            "capacity shortfall pool"
        );
        // Bound diagnostic output independently of model size. Each row is a
        // real selected slot; its representative is its largest charged member.
        for (bytes, descriptor, slot) in slots.into_iter().take(12) {
            let values =
                nodes
                    .iter()
                    .flat_map(|node| &node.values)
                    .filter(|value| {
                        value.storage().components().iter().any(|component| {
                            component.resource_id() == &descriptor.base_resource_id
                        })
                    })
                    .map(|value| value.value_id())
                    .collect::<BTreeSet<_>>();
            tracing::debug!(
                pool = ?pool.pool_id(),
                bytes,
                kind = ?slot.kind(),
                member_count = slot.resource_ids().len(),
                resource = %descriptor.base_resource_id,
                representative_values = ?values.iter().take(8).collect::<Vec<_>>(),
                "capacity shortfall step slot"
            );
        }
        if let Some((bytes, descriptor)) = descriptors
            .iter()
            .filter(|descriptor| {
                descriptor.pool_id == *pool.pool_id()
                    && descriptor.lifetime == AllocationLifetime::Invocation
            })
            .filter_map(|descriptor| {
                descriptor
                    .minimum_request_bytes()
                    .ok()
                    .map(|bytes| (bytes, descriptor))
            })
            .max_by_key(|(bytes, _)| *bytes)
        {
            tracing::debug!(
                pool = ?pool.pool_id(),
                bytes,
                kind = ?descriptor.kind,
                resource = %descriptor.base_resource_id,
                "capacity shortfall largest invocation resource"
            );
        }
    }
}
