//! Retain the complete checked algorithm seed without merging startup sources.
//! Each numerical source keeps its original checked family and complete phase
//! obligations. The global seed grants no fitted prediction or wider support.
use super::*;
use ferrum_scheduler::implementations::continuous::cost_model::structured_v2::{
    DeclaredAlgorithmUniverseBuilderV1, DeclaredAlgorithmUniverseV1,
    StructuredCostTemplatePolicyV1, StructuredPopulationPolicyV1, StructuredUnknownV2,
};

/// Reuse the original bounded declaration builder. This also permits a
/// terminal inventory failure to retain its already checked algorithm seed
/// without selecting a partial numerical source or projecting old facts.
pub(super) fn freeze_declared_algorithms(
    inventory: &CheckedCaseInventory,
    population: &StructuredServiceDeclarationV7,
    maximum_bytes: usize,
) -> Result<Option<DeclaredAlgorithmUniverseV1>> {
    let Some(contract) = &population.nonnegative_envelope else {
        return Ok(None);
    };
    if contract.population_policy != StructuredPopulationPolicyV1::HomogeneousOrdinaryDecodeV1
        || contract.template_policy != StructuredCostTemplatePolicyV1::InstalledAlgorithmSetV1
    {
        return Ok(None);
    }
    let retained = inventory
        .retained_payload_bytes()
        .ok_or_else(|| error("algorithm inventory capacity overflow"))?;
    let available = maximum_bytes
        .checked_sub(retained)
        .ok_or_else(|| error("algorithm inventory capacity exhausted"))?;
    let mut builder =
        DeclaredAlgorithmUniverseBuilderV1::new(population.settings.max_axes, available)
            .map_err(|reason| error(format!("algorithm inventory builder: {reason:?}")))?;
    for input in inventory
        .inputs
        .iter()
        .flatten()
        .map(|facts| {
            facts
                .original
                .as_deref()
                .ok_or_else(|| error("algorithm inventory already frozen"))
        })
        .chain(
            inventory
                .algorithm_inputs
                .iter()
                .map(|input| Ok(input.as_ref())),
        )
    {
        match builder.observe(input?) {
            Ok(()) | Err(StructuredUnknownV2::UnsupportedScope) => {}
            Err(reason) => return Err(error(format!("algorithm inventory input: {reason:?}"))),
        }
    }
    if builder.is_empty() {
        return Ok(None);
    }
    let universe = builder
        .finish()
        .map_err(|reason| error(format!("algorithm inventory freeze: {reason:?}")))?;
    require(
        retained.checked_add(
            universe
                .retained_payload_bytes()
                .ok_or_else(|| error("algorithm universe capacity overflow"))?,
        ),
        maximum_bytes,
    )?;
    Ok(Some(universe))
}

/// The online discovery seed and a numerical source have different scopes.
/// Keep every checked algorithm for later discovery, while startup selection
/// uses the unchanged raw family keys, axes, alternatives and fresh-member
/// floors. A source can therefore finish without qualifying unrelated kernels.
/// No retained slot means this standalone plan owns no global seed.
pub(super) fn freeze_seed(
    inventory: &CheckedCaseInventory,
    population: &StructuredServiceDeclarationV7,
    maximum_bytes: usize,
    retained_seed: Option<&mut Option<DeclaredAlgorithmUniverseV1>>,
) -> Result<usize> {
    if population
        .nonnegative_envelope
        .as_ref()
        .is_some_and(|contract| contract.algorithm_universe.is_some())
    {
        return Err(error(
            "startup numerical source requires original checked domains",
        ));
    }
    let Some(slot) = retained_seed else {
        return Ok(0);
    };
    let old_bytes = slot
        .as_ref()
        .map_or(Some(0), DeclaredAlgorithmUniverseV1::retained_payload_bytes)
        .ok_or_else(|| error("cold algorithm seed retained capacity overflow"))?;
    // The old seed remains alive during the complete new declaration build.
    // Its charge cannot be borrowed as scratch or released before replacement.
    let available = maximum_bytes
        .checked_sub(old_bytes)
        .ok_or_else(|| error("cold algorithm seed exceeds retained capacity"))?;
    let Some(universe) = freeze_declared_algorithms(inventory, population, available)? else {
        return Ok(old_bytes);
    };
    let seed_bytes = universe
        .retained_payload_bytes()
        .ok_or_else(|| error("algorithm universe capacity overflow"))?;
    match slot.as_ref() {
        Some(old) if old.contains_universe(&universe) => return Ok(old_bytes),
        Some(old) if !universe.contains_universe(old) => {
            return Err(error(
                "cold algorithm prefix is not a monotone checked universe",
            ));
        }
        _ => {}
    }
    tracing::info!(
        algorithms = universe.algorithm_count(),
        universe = ?universe.signature(),
        retained_bytes = seed_bytes,
        source_domains = "original_checked_families",
        "Automatic complete algorithm seed retained independently of numerical sources"
    );
    *slot = Some(universe);
    Ok(seed_bytes)
}

fn require(bytes: Option<usize>, maximum: usize) -> Result<()> {
    if bytes.is_none_or(|bytes| bytes > maximum) {
        Err(error(
            "algorithm inventory shared retained capacity exhausted",
        ))
    } else {
        Ok(())
    }
}
