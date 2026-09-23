//! Pure projection of supplied actual dispatch evidence. Historical attribution
//! never certifies that the same route will be selected on a future wave.
use super::*;

pub(super) struct ObservedRoute {
    pub canonical: CanonicalWaveCostBuilder,
    pub graph: ActualWaveGraphState,
}

pub(super) fn actual_route<'a>(
    attribution: Option<&DeviceSubmissionAttribution>,
    provider_at: impl Fn(u32) -> Option<CostProviderIdentity<'a>>,
    graph_capability: DeviceCostGraphCaptureCapability,
    product: CostProductOutput,
    retries: u32,
) -> std::result::Result<ObservedRoute, ActualWaveEvidenceUnknown> {
    let attribution = attribution.ok_or(ActualWaveEvidenceUnknown::GraphPath)?;
    let commands = attribution.commands();
    if commands.is_empty() || commands.len() > MAX_COST_COMMANDS {
        return Err(ActualWaveEvidenceUnknown::ProviderPath);
    }
    let mut canonical = CanonicalWaveCostBuilder::new(retries, product);
    let mut replayed = false;
    for command in commands {
        let provider = command
            .node_index()
            .map(|index| provider_at(index).ok_or(ActualWaveEvidenceUnknown::ProviderPath))
            .transpose()?;
        replayed |= command.execution_path() == DeviceExecutionPath::Replayed;
        canonical
            .physical_command(CostPhysicalCommand::from_attribution(command, provider))
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
    let graph = super::graph_state(graph_capability, replayed, attribution.graph_evidence())?;
    Ok(ObservedRoute { canonical, graph })
}

#[cfg(test)]
mod tests;
