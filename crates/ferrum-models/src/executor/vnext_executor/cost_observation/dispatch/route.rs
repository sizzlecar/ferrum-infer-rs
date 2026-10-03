//! Pure projection of supplied actual dispatch evidence. Historical attribution
//! never certifies that the same route will be selected on a future wave.
use super::*;

pub(super) struct ObservedRoute {
    pub canonical: CanonicalWaveCostBuilder,
    pub graph: ActualWaveGraphState,
}

/// Actual command structure survives a numerical graph-population rejection.
/// Earlier command/replay errors still reject the entire projection.
pub(super) struct ObservedRouteComponents {
    pub canonical: CanonicalWaveCostBuilder,
    pub graph: std::result::Result<ActualWaveGraphState, ActualWaveEvidenceUnknown>,
}

#[cfg(test)]
pub(super) fn actual_route<'a>(
    attribution: Option<&DeviceSubmissionAttribution>,
    provider_at: impl Fn(u32) -> Option<CostProviderIdentity<'a>>,
    graph_capability: DeviceCostGraphCaptureCapability,
    product: CostProductOutput,
    retries: u32,
) -> std::result::Result<ObservedRoute, ActualWaveEvidenceUnknown> {
    actual_route_with_capture(
        attribution,
        provider_at,
        graph_capability,
        product,
        retries,
        false,
    )
}

pub(super) fn actual_route_with_capture<'a>(
    attribution: Option<&DeviceSubmissionAttribution>,
    provider_at: impl Fn(u32) -> Option<CostProviderIdentity<'a>>,
    graph_capability: DeviceCostGraphCaptureCapability,
    product: CostProductOutput,
    retries: u32,
    structured_capture: bool,
) -> std::result::Result<ObservedRoute, ActualWaveEvidenceUnknown> {
    actual_route_projection(
        attribution,
        provider_at,
        graph_capability,
        product,
        retries,
        structured_capture,
        true,
    )
}

pub(super) fn actual_route_exact<'a>(
    attribution: Option<&DeviceSubmissionAttribution>,
    provider_at: impl Fn(u32) -> Option<CostProviderIdentity<'a>>,
    graph_capability: DeviceCostGraphCaptureCapability,
    product: CostProductOutput,
    retries: u32,
) -> std::result::Result<ObservedRoute, ActualWaveEvidenceUnknown> {
    actual_route_projection(
        attribution,
        provider_at,
        graph_capability,
        product,
        retries,
        false,
        false,
    )
}

pub(super) fn actual_route_projection<'a>(
    attribution: Option<&DeviceSubmissionAttribution>,
    provider_at: impl Fn(u32) -> Option<CostProviderIdentity<'a>>,
    graph_capability: DeviceCostGraphCaptureCapability,
    product: CostProductOutput,
    retries: u32,
    structured_capture: bool,
    statistics: bool,
) -> std::result::Result<ObservedRoute, ActualWaveEvidenceUnknown> {
    let parts = actual_route_components(
        attribution,
        provider_at,
        graph_capability,
        product,
        retries,
        structured_capture,
        statistics,
    )?;
    Ok(ObservedRoute {
        canonical: parts.canonical,
        graph: parts.graph?,
    })
}

pub(super) fn actual_route_components<'a>(
    attribution: Option<&DeviceSubmissionAttribution>,
    provider_at: impl Fn(u32) -> Option<CostProviderIdentity<'a>>,
    graph_capability: DeviceCostGraphCaptureCapability,
    product: CostProductOutput,
    retries: u32,
    structured_capture: bool,
    statistics: bool,
) -> std::result::Result<ObservedRouteComponents, ActualWaveEvidenceUnknown> {
    let attribution = attribution.ok_or(ActualWaveEvidenceUnknown::GraphPath)?;
    let commands = attribution.commands();
    if commands.is_empty() || commands.len() > MAX_COST_COMMANDS {
        return Err(ActualWaveEvidenceUnknown::ProviderPath);
    }
    let mut canonical = if !statistics {
        CanonicalWaveCostBuilder::new_exact(retries, product)
    } else if structured_capture {
        CanonicalWaveCostBuilder::new_with_structured_statistics(retries, product)
    } else {
        CanonicalWaveCostBuilder::new(retries, product)
    };
    let replayed = append_actual_commands(&mut canonical, attribution, provider_at, None)?;
    let graph = super::graph_state(graph_capability, replayed, attribution.graph_evidence());
    Ok(ObservedRouteComponents { canonical, graph })
}

/// A separate exact-only reconstruction after the original numerical route
/// failed. Every compact replay still requires its full sealed logical ledger.
pub(super) fn actual_physical_route<'a>(
    attribution: &DeviceSubmissionAttribution,
    provider_at: impl Fn(u32) -> Option<CostProviderIdentity<'a>>,
    product: CostProductOutput,
    retries: u32,
    direct_operation: ferrum_interfaces::vnext::DeviceNativeOperationId,
) -> std::result::Result<CanonicalWaveCostBuilder, ActualWaveEvidenceUnknown> {
    let mut canonical = CanonicalWaveCostBuilder::new_exact(retries, product);
    append_actual_commands(
        &mut canonical,
        attribution,
        provider_at,
        Some(direct_operation),
    )?;
    Ok(canonical)
}

fn append_actual_commands<'a>(
    canonical: &mut CanonicalWaveCostBuilder,
    attribution: &DeviceSubmissionAttribution,
    provider_at: impl Fn(u32) -> Option<CostProviderIdentity<'a>>,
    direct_operation: Option<ferrum_interfaces::vnext::DeviceNativeOperationId>,
) -> std::result::Result<bool, ActualWaveEvidenceUnknown> {
    let commands = attribution.commands();
    if commands.is_empty() || commands.len() > MAX_COST_COMMANDS {
        return Err(ActualWaveEvidenceUnknown::ProviderPath);
    }
    let mut replayed = false;
    for command in commands {
        let provider = command
            .node_index()
            .map(|index| provider_at(index).ok_or(ActualWaveEvidenceUnknown::ProviderPath))
            .transpose()?;
        replayed |= command.execution_path() == DeviceExecutionPath::Replayed;
        let projected = CostPhysicalCommand::from_attribution(command, provider);
        match direct_operation.filter(|direct| {
            command.execution_path() == DeviceExecutionPath::Replayed
                && command.native_op_id() != direct.as_str()
        }) {
            Some(direct) => canonical.original_replay_command(projected, direct),
            None => canonical.physical_command(projected),
        }
        .map_err(|_| ActualWaveEvidenceUnknown::ProviderPath)?;
    }
    for segment in attribution.replayed_segments() {
        canonical
            .replay_segment(
                segment.physical_command_index(),
                segment.reusable_executable_fingerprint(),
                segment.logical_commands().len(),
            )
            .map_err(|_| ActualWaveEvidenceUnknown::GraphPath)?;
        for command in segment.logical_commands() {
            let provider =
                provider_at(command.node_index()).ok_or(ActualWaveEvidenceUnknown::ProviderPath)?;
            canonical
                .logical_command(CostLogicalCommand::from_attribution(command, provider))
                .map_err(|_| ActualWaveEvidenceUnknown::ProviderPath)?;
        }
    }
    Ok(replayed)
}

#[cfg(test)]
mod tests;
