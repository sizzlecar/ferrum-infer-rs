//! Stronger base alignment can let compatible activations share storage. Keep
//! the original partition unless the complete fixed-shape slot plan is smaller,
//! including padding and completion retention. Variable demands and provider
//! workspaces retain their original contracts.
use super::*;

pub(super) fn coalesce(
    descriptors: &mut [DynamicResourceDescriptor],
    nodes: &[PlanNode],
    retained: &BTreeSet<ResourceId>,
) -> Result<(), VNextError> {
    let mut groups = BTreeMap::<DynamicBackingPoolId, (PoolCompatibilityKey, Vec<usize>)>::new();
    for (index, descriptor) in descriptors.iter().enumerate() {
        if descriptor.usage != BufferUsage::Activations {
            continue;
        }
        let key = PoolCompatibilityKey::new(
            &descriptor.storage,
            descriptor.usage,
            descriptor.element_type,
            1,
        )?;
        let entry = groups
            .entry(DynamicBackingPoolId::from_compatibility(&key)?)
            .or_insert_with(|| (key.clone(), Vec::new()));
        if entry.0 != key {
            return Err(invalid_plan(
                "alignment group has incompatible storage contracts",
            ));
        }
        entry.1.push(index);
    }
    for (_, indices) in groups.into_values() {
        if indices.iter().any(|&index| {
            let descriptor = &descriptors[index];
            descriptor.kind != AllocationKind::Value
                || descriptor.lifetime != AllocationLifetime::Step
                || !matches!(descriptor.demand, DynamicResourceDemand::Fixed { .. })
        }) {
            continue;
        }
        let mut original = BTreeMap::<u64, Vec<&DynamicResourceDescriptor>>::new();
        for &index in &indices {
            let descriptor = &descriptors[index];
            original
                .entry(descriptor.alignment_bytes)
                .or_default()
                .push(descriptor);
        }
        if original.len() < 2 {
            continue;
        }
        let alignment = *original.last_key_value().unwrap().0;
        let original_bytes = original.values().try_fold(0_u64, |sum, group| {
            sum.checked_add(slot_bytes(group, nodes, retained)?)
                .ok_or_else(|| invalid_plan("alignment group total overflows u64"))
        })?;
        let candidate = indices
            .iter()
            .map(|&index| {
                let mut descriptor = descriptors[index].clone();
                descriptor.alignment_bytes = alignment;
                descriptor.pool_id =
                    DynamicBackingPoolId::from_compatibility(&PoolCompatibilityKey::new(
                        &descriptor.storage,
                        descriptor.usage,
                        descriptor.element_type,
                        alignment,
                    )?)?;
                Ok(descriptor)
            })
            .collect::<Result<Vec<_>, VNextError>>()?;
        let candidate_bytes = slot_bytes(&candidate.iter().collect::<Vec<_>>(), nodes, retained)?;
        if candidate_bytes < original_bytes {
            tracing::debug!(
                original_alignments = ?original.keys().collect::<Vec<_>>(),
                alignment, original_bytes, candidate_bytes,
                resource_count = candidate.len(),
                "coalesced compatible activation alignments"
            );
            for (index, descriptor) in indices.into_iter().zip(candidate) {
                descriptors[index] = descriptor;
            }
        }
    }
    Ok(())
}

fn slot_bytes(
    descriptors: &[&DynamicResourceDescriptor],
    nodes: &[PlanNode],
    retained: &BTreeSet<ResourceId>,
) -> Result<u64, VNextError> {
    let slots = MemoryPlan::derive_pool_step_slots(nodes, descriptors, retained)?;
    MemoryPlan::step_slot_bytes(&slots, descriptors, false, "alignment group step minimum")
}
