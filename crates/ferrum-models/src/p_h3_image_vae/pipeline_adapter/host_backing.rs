//! Extra source-owned WeightBacking during one actual static initialization.
//! Upload staging and the retained materializer's own preparation remain the
//! executor's separate charges. This path currently supports the exact core
//! identity materializer; it makes no lifetime guess for other materializers.
use super::*;
use std::collections::{BTreeMap, BTreeSet};

pub(super) fn maximum(
    family: &PreparedModelFamily,
    plan: &ExecutionPlan,
    host: &HostMemoryAccount,
    source_bytes: impl Fn(&WeightComponentSpec) -> Result<u64>,
) -> Result<usize> {
    let payload = plan.payload();
    ensure!(
        payload.family_id() == family.family_id()
            && payload.prepared_family_fingerprint() == family.fingerprint()?
            && payload.program_fingerprint() == family.program().fingerprint()?,
        "VAE source backing requires the exact prepared compiled plan"
    );
    let execution = payload.execution_weights().stored()?;
    ensure!(execution==&ExecutionWeightPlan::identity(family)?,
        "VAE source backing currently requires the core identity materializer without static transforms");
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
                    .context("static weight placement has no component identity")?;
                let component = components
                    .get(id)
                    .context("placed weight is absent from execution schema")?;
                let allocation = allocations
                    .get(resolved.resource_id())
                    .context("placed weight has no static allocation")?;
                let end = resolved
                    .offset_bytes()
                    .checked_add(resolved.length_bytes())
                    .context("static weight placement overflows")?;
                ensure!(
                    resolved.length_bytes() == component.physical_bytes()?
                        && resolved.element_type() == component.physical_element_type()
                        && allocation.usage() == BufferUsage::Weights
                        && allocation.element_type() == resolved.element_type()
                        && end <= allocation.size_bytes(),
                    "static placement differs from its physical weight schema"
                );
                let placement = (
                    resolved.resource_id(),
                    resolved.offset_bytes(),
                    resolved.length_bytes(),
                    resolved.element_type(),
                );
                if let Some(old) = placements.insert(id.clone(), placement) {
                    ensure!(
                        old == placement,
                        "one component has inconsistent static placements"
                    );
                }
            }
        }
    }
    ensure!(
        schema
            .components
            .iter()
            .all(|c| !c.required || placements.contains_key(&c.id)),
        "required VAE weights lack actual static placements"
    );
    // These are the same placed-output groups as static_initialization:
    // equal ordered source_ids share one materialization call. Do not sum the
    // entire model, count duplicate node uses, or assume one component/group.
    let mut groups = BTreeSet::new();
    for component in &schema.components {
        if placements.contains_key(&component.id) {
            groups.insert(
                execution
                    .component_sources()
                    .get(&component.id)
                    .context("placed weight has no ordered source mapping")?,
            );
        }
    }
    let source_schema = family.stored_weight_schema()?;
    let mut maximum = 0usize;
    for source_ids in groups {
        let mut group = 0usize;
        for id in source_ids {
            let index = source_schema
                .components
                .binary_search_by(|c| c.id.cmp(id))
                .map_err(|_| {
                    anyhow::anyhow!("static group references unknown source component {id}")
                })?;
            let bytes = source_bytes(&source_schema.components[index])?;
            if bytes != 0 {
                // One actual allocation per converted source component.
                // Group-sum rounding would undercount distinct backings.
                group = group
                    .checked_add(host.backing_bytes(usize::try_from(bytes)?)?)
                    .context("VAE static-group Host backing overflows")?;
            }
        }
        maximum = maximum.max(group);
    }
    Ok(maximum)
}
