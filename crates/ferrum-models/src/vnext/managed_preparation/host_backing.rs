use super::*;
use ferrum_runtime::{BufferUsage, ExecutionWeightPlan};
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn selected_groups<'plan>(
    source: &ManagedProductionWeightSource<'_>,
    plan: &'plan ExecutionPlan,
) -> Result<BTreeSet<&'plan Vec<ferrum_runtime::WeightId>>, VNextError> {
    let family = source.prepared.family();
    let payload = plan.payload();
    if payload.family_id() != family.family_id()
        || payload.prepared_family_fingerprint() != family.fingerprint()?
        || payload.program_fingerprint() != family.program().fingerprint()?
    {
        return Err(invalid(
            "source backing requires the exact prepared compiled plan",
        ));
    }
    let execution = payload.execution_weights().stored()?;
    if execution != &ExecutionWeightPlan::identity(family)? {
        return Err(invalid(
            "source backing requires the core identity materializer without static transforms",
        ));
    }
    let schema = execution.schema();
    let components = schema
        .components
        .iter()
        .map(|c| (&c.id, c))
        .collect::<BTreeMap<_, _>>();
    let allocations = payload
        .memory()
        .static_allocations()
        .iter()
        .map(|a| (a.resource_id(), a))
        .collect::<BTreeMap<_, _>>();
    let mut placements = BTreeMap::new();
    for node in payload.nodes() {
        for binding in node
            .values()
            .iter()
            .filter(|v| v.usage() == BufferUsage::Weights)
        {
            for resolved in binding.storage().components() {
                let id = resolved
                    .component_id()
                    .ok_or_else(|| invalid("static placement has no component identity"))?;
                let component = components
                    .get(id)
                    .ok_or_else(|| invalid("placed weight is absent from schema"))?;
                let allocation = allocations
                    .get(resolved.resource_id())
                    .ok_or_else(|| invalid("placed weight has no static allocation"))?;
                let end = resolved
                    .offset_bytes()
                    .checked_add(resolved.length_bytes())
                    .ok_or_else(|| invalid("static placement overflows"))?;
                if resolved.length_bytes() != component.physical_bytes()?
                    || resolved.element_type() != component.physical_element_type()
                    || allocation.usage() != BufferUsage::Weights
                    || allocation.element_type() != resolved.element_type()
                    || end > allocation.size_bytes()
                {
                    return Err(invalid(
                        "static placement differs from its physical weight schema",
                    ));
                }
                let placement = (
                    resolved.resource_id(),
                    resolved.offset_bytes(),
                    resolved.length_bytes(),
                    resolved.element_type(),
                );
                if let Some(old) = placements.insert(id.clone(), placement) {
                    if old != placement {
                        return Err(invalid("component has inconsistent static placements"));
                    }
                }
            }
        }
    }
    if schema
        .components
        .iter()
        .any(|c| c.required && !placements.contains_key(&c.id))
    {
        return Err(invalid("required weights lack actual static placements"));
    }
    // Runtime prepares all sources of one ordered materialization group before
    // it uploads its outputs. Sum individual aligned owners within that group,
    // then take the largest group; never sum every converted model component.
    let mut groups = BTreeSet::new();
    for component in &schema.components {
        if placements.contains_key(&component.id) {
            groups.insert(
                execution
                    .component_sources()
                    .get(&component.id)
                    .ok_or_else(|| invalid("placed weight has no source mapping"))?,
            );
        }
    }
    Ok(groups)
}

pub(super) fn maximum(
    source: &ManagedProductionWeightSource<'_>,
    plan: &ExecutionPlan,
) -> Result<usize, VNextError> {
    let groups = selected_groups(source, plan)?;
    let sources = source.prepared.family().stored_weight_schema()?;
    let mut maximum = 0usize;
    for ids in groups {
        let mut group = 0usize;
        for id in ids {
            let component = sources
                .components
                .iter()
                .find(|c| &c.id == id)
                .ok_or_else(|| invalid("static group references an unknown source"))?;
            let bytes = source.component_backing_bytes(component)?;
            if bytes != 0 {
                let bytes = usize::try_from(bytes)
                    .map_err(|_| invalid("source backing exceeds address space"))?;
                let allocation = source
                    .host
                    .backing_bytes(bytes)
                    .map_err(|error| invalid(error.to_string()))?;
                group = group
                    .checked_add(allocation)
                    .ok_or_else(|| invalid("source group backing overflows"))?;
            }
        }
        maximum = maximum.max(group);
    }
    Ok(maximum)
}
