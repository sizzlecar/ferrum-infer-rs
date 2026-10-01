//! Owned passive observations. No constructor in this module grants execution
//! authority. Only the explicit worker resolver evaluates a CPU template.
use super::*;
use crate::execution_cost::{SelectedCommandCostEvidenceV1, StatisticalEvidenceUnknown};
use std::ops::Range;
mod budget;
pub use budget::*;

/// Immutable, validated logical population owned by one resident segment.
/// Validation and payload accounting happen at sealing, not on each launch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceReplayedCommandCatalogue {
    pub(super) segment: DeviceReusableExecutionSegment,
    pub(super) commands: Arc<[DeviceReplayedLogicalCommandAttribution]>,
    pub(super) payload_bytes: usize,
    pub(super) graph_nodes: u64,
    pub(super) participants: u32,
    pub(super) has_statistical_evidence: bool,
}
impl DeviceReplayedCommandCatalogue {
    pub fn new(
        segment: DeviceReusableExecutionSegment,
        commands: Arc<[DeviceReplayedLogicalCommandAttribution]>,
    ) -> Option<Self> {
        let participants = commands.first()?.participant_count();
        if commands.len() > crate::execution_cost::MAX_COST_COMMANDS
            || commands.len() != segment.logical_command_count() as usize
            || segment
                .start_node_index()
                .checked_add(segment.logical_command_count())
                != Some(segment.end_node_index())
            || commands.iter().enumerate().any(|(ordinal, row)| {
                u32::try_from(ordinal).ok() != Some(row.logical_command_ordinal())
                    || segment
                        .start_node_index()
                        .checked_add(row.logical_command_ordinal())
                        != Some(row.node_index())
                    || row.participant_count() != participants
            })
        {
            return None;
        }
        let payload_bytes = logical_payload_bytes(&commands)?;
        let graph_nodes = commands.iter().try_fold(0u64, |n, row| {
            n.checked_add(row.reusable_graph_node_count())
        })?;
        let has_statistical_evidence = commands
            .iter()
            .any(|row| row.statistical_evidence.is_some());
        Some(Self {
            segment,
            commands,
            payload_bytes,
            graph_nodes,
            participants,
            has_statistical_evidence,
        })
    }
    pub fn logical_commands(&self) -> &[DeviceReplayedLogicalCommandAttribution] {
        &self.commands
    }
    pub fn retained_payload_bytes(&self) -> Option<usize> {
        self.payload_bytes.checked_add(std::mem::size_of::<Self>())
    }
    pub fn logical_graph_node_count(&self) -> u64 {
        self.graph_nodes
    }
}
impl std::ops::Deref for DeviceReplayedCommandCatalogue {
    type Target = [DeviceReplayedLogicalCommandAttribution];
    fn deref(&self) -> &Self::Target {
        &self.commands
    }
}
fn logical_payload_bytes(rows: &[DeviceReplayedLogicalCommandAttribution]) -> Option<usize> {
    rows.iter().try_fold(
        rows.len()
            .checked_mul(std::mem::size_of::<DeviceReplayedLogicalCommandAttribution>())?,
        |bytes, row| {
            bytes.checked_add(
                row.statistical_evidence
                    .as_ref()
                    .map_or(Some(0), |evidence| evidence.retained_payload_bytes())?,
            )
        },
    )
}

/// Complete small numerical state for one physical wave. It has no dependency
/// on the preceding wave, so queue loss cannot silently corrupt later work.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FrozenObservationInput {
    tokens: u64,
    immediate: Arc<[Range<u64>]>,
    source: Arc<[Range<u64>]>,
}
impl FrozenObservationInput {
    pub fn from_work_shape(shape: &super::super::BatchWorkShape) -> Option<Self> {
        let rows = shape.participant_token_ranges();
        if rows.is_empty() || rows.len() > crate::execution_cost::MAX_COST_ROWS {
            return None;
        }
        let mut end = 0;
        for row in rows {
            let r = row.immediate_token_range();
            if r.start != end || r.end <= r.start {
                return None;
            }
            end = r.end;
        }
        if end != shape.immediate_tokens() {
            return None;
        }
        Some(Self {
            tokens: end,
            immediate: rows.iter().map(|r| r.immediate_token_range()).collect(),
            source: rows.iter().map(|r| r.source_token_range()).collect(),
        })
    }
    /// Encoder-local work for commands whose CPU recipe already owns all row
    /// facts (including zero-token transfers). This carries no live identity.
    pub fn command(tokens: u64) -> Self {
        Self {
            tokens,
            immediate: Arc::from([]),
            source: Arc::from([]),
        }
    }
    pub fn tokens(&self) -> u64 {
        self.tokens
    }
    pub fn participant_ranges(&self) -> &[Range<u64>] {
        &self.immediate
    }
    pub fn source_ranges(&self) -> &[Range<u64>] {
        &self.source
    }
    pub fn replay_cost_work(&self) -> Option<DeviceReplayCostWork> {
        DeviceReplayCostWork::from_observation(self.tokens, Arc::clone(&self.immediate))
    }
    pub fn retained_payload_bytes(&self) -> Option<usize> {
        self.immediate
            .len()
            .checked_add(self.source.len())?
            .checked_mul(std::mem::size_of::<Range<u64>>())?
            .checked_add(std::mem::size_of::<Self>())
    }
}

/// Implement with an explicit backend CPU-data type. Implementations must not
/// retain GPU buffers, leases, runtime/stream/pipeline objects, arbitrary
/// closures, or mutable execution state. There is deliberately no Fn adapter.
/// The resolver may neither choose a provider nor consult current runtime state.
pub trait DeviceObservationTemplate: Send + Sync + 'static {
    fn command_count(&self) -> usize;
    fn retained_payload_bytes(&self) -> Option<usize>;
    fn projection_retained_bytes_upper_bound(&self) -> Option<usize>;
    fn project(
        &self,
        input: &FrozenObservationInput,
    ) -> Result<Vec<Option<SelectedCommandCostEvidenceV1>>, StatisticalEvidenceUnknown>;
}

#[derive(Clone)]
pub struct DeviceObservationPacket {
    template: Arc<dyn DeviceObservationTemplate>,
    input: FrozenObservationInput,
    retained: usize,
    call_owned: usize,
    projected: usize,
    retention: Option<RetainedDeviceObservationTemplate>,
}
impl std::fmt::Debug for DeviceObservationPacket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceObservationPacket")
            .field("commands", &self.template.command_count())
            .field("retained_payload_bytes", &self.retained)
            .field("projected_payload_upper_bound", &self.projected)
            .finish()
    }
}
// Passive observation must never alter the pre-existing execution equality
// used by cache/guard comparisons. Explicit resolution validates its content.
impl PartialEq for DeviceReplayedSegmentAttribution {
    fn eq(&self, other: &Self) -> bool {
        self.physical_command_index == other.physical_command_index
            && self.program_id == other.program_id
            && self.segment == other.segment
            && self.reusable_executable_fingerprint == other.reusable_executable_fingerprint
            && self.logical_commands == other.logical_commands
    }
}
impl Eq for DeviceReplayedSegmentAttribution {}
impl DeviceObservationPacket {
    fn new(
        template: Arc<dyn DeviceObservationTemplate>,
        input: FrozenObservationInput,
    ) -> Result<Self, StatisticalEvidenceUnknown> {
        if template.command_count() == 0
            || template.command_count() > crate::execution_cost::MAX_COST_COMMANDS
        {
            return Err(StatisticalEvidenceUnknown::Capacity);
        }
        let call_owned = input
            .retained_payload_bytes()
            .and_then(|n| n.checked_add(std::mem::size_of::<Self>()))
            .ok_or(StatisticalEvidenceUnknown::Capacity)?;
        let retained = template
            .retained_payload_bytes()
            .and_then(|n| n.checked_add(call_owned))
            .ok_or(StatisticalEvidenceUnknown::Capacity)?;
        let projected = template
            .projection_retained_bytes_upper_bound()
            .ok_or(StatisticalEvidenceUnknown::Capacity)?;
        Ok(Self {
            template,
            input,
            retained,
            call_owned,
            projected,
            retention: None,
        })
    }
    pub fn retained_payload_bytes(&self) -> usize {
        self.retained
    }
    /// Payload charged to this observation. Only the private retained handle
    /// proves that a separate lease owns the template and reservation overhead.
    /// Unleased packets keep their complete conservative charge. Dynamic ranges
    /// stay charged even when another packet happens to share their Arc.
    pub fn call_owned_payload_bytes(&self) -> usize {
        if self.retention.is_some() {
            self.call_owned
        } else {
            self.retained
        }
    }
    /// Immutable metadata access only. This never projects or reads a runtime.
    pub fn template(&self) -> &dyn DeviceObservationTemplate {
        self.template.as_ref()
    }
    pub fn input(&self) -> &FrozenObservationInput {
        &self.input
    }
    pub fn budget(&self) -> Option<&Arc<DeviceObservationTemplateBudget>> {
        self.retention
            .as_ref()
            .map(RetainedDeviceObservationTemplate::budget)
    }
    pub fn projection_retained_bytes_upper_bound(&self) -> usize {
        self.projected
    }
    fn resolve(
        self,
    ) -> Result<Vec<Option<SelectedCommandCostEvidenceV1>>, StatisticalEvidenceUnknown> {
        let result = self.template.project(&self.input)?;
        if result.len() != self.template.command_count()
            || result.capacity() > crate::execution_cost::MAX_COST_COMMANDS
        {
            return Err(StatisticalEvidenceUnknown::CommandMismatch);
        }
        let payload = result
            .iter()
            .try_fold(
                result
                    .capacity()
                    .checked_mul(std::mem::size_of::<Option<SelectedCommandCostEvidenceV1>>())
                    .ok_or(StatisticalEvidenceUnknown::Capacity)?,
                |n, row| {
                    n.checked_add(
                        row.as_ref()
                            .map_or(Some(0), |v| v.retained_payload_bytes())?,
                    )
                },
            )
            .ok_or(StatisticalEvidenceUnknown::Capacity)?;
        if payload > self.projected {
            return Err(StatisticalEvidenceUnknown::Capacity);
        }
        Ok(result)
    }
}

impl DeviceNativeWorkAttribution {
    pub fn with_observation(mut self, observation: DeviceObservationPacket) -> Option<Self> {
        if observation.template.command_count() != 1
            || self.execution_path != DeviceExecutionPath::Eager
        {
            return None;
        }
        self.statistical_evidence = None;
        self.observation = Some(observation);
        Some(self)
    }
}
impl DeviceReplayedSegmentAttribution {
    /// Reuse the exact immutable population certified at registration. This
    /// preserves ordinal, node, participant and native-work checks without a
    /// second catalogue traversal at each physical launch.
    pub fn from_catalogue(
        physical_command_index: u32,
        program_id: DeviceReusableExecutionProgramId,
        segment: DeviceReusableExecutionSegment,
        reusable_executable_fingerprint: String,
        catalogue: &DeviceReplayedCommandCatalogue,
    ) -> Option<Self> {
        if segment != catalogue.segment
            || reusable_executable_fingerprint.len() != 64
            || !reusable_executable_fingerprint
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
        {
            return None;
        }
        Some(Self {
            physical_command_index,
            program_id,
            segment,
            reusable_executable_fingerprint,
            logical_commands: Arc::clone(&catalogue.commands),
            observation: None,
            logical_payload_bytes: catalogue.payload_bytes,
            logical_graph_nodes: catalogue.graph_nodes,
            logical_participants: catalogue.participants,
            logical_has_statistical_evidence: catalogue.has_statistical_evidence,
        })
    }
    pub fn logical_graph_node_count(&self) -> u64 {
        self.logical_graph_nodes
    }
    /// Runtime attaches this only to the actual launched segment, never a
    /// candidate/preview. Original executable and logical ordinal checks stay.
    pub fn with_observation(mut self, observation: DeviceObservationPacket) -> Option<Self> {
        if observation.template.command_count() != self.logical_commands.len()
            || self.logical_has_statistical_evidence
        {
            return None;
        }
        self.observation = Some(observation);
        Some(self)
    }
}
impl DeviceSubmissionAttribution {
    pub fn has_unresolved_observation(&self) -> bool {
        self.commands.iter().any(|row| row.observation.is_some())
            || self
                .replayed_segments
                .iter()
                .any(|row| row.observation.is_some())
    }

    /// Explicit worker-only boundary. No getter, Clone, Debug or Serialize
    /// calls it. The caller must first join the original completed settlement.
    pub fn resolve_observation(mut self) -> Result<Self, StatisticalEvidenceUnknown> {
        if !self.has_unresolved_observation() {
            return Ok(self);
        }
        for row in Arc::make_mut(&mut self.commands) {
            if let Some(observation) = row.observation.take() {
                let mut projected = observation.resolve()?;
                let evidence = projected
                    .pop()
                    .ok_or(StatisticalEvidenceUnknown::CommandMismatch)?;
                row.statistical_evidence = match evidence {
                    Some(evidence) => {
                        evidence.validate_command(
                            row.token_count,
                            row.compute_dispatch_count,
                            row.transfer_command_count,
                        )?;
                        Some(evidence)
                    }
                    None => None,
                };
            }
        }
        for segment in Arc::make_mut(&mut self.replayed_segments) {
            if let Some(observation) = segment.observation.take() {
                let projected = observation.resolve()?;
                if projected.len() != segment.logical_commands.len() {
                    return Err(StatisticalEvidenceUnknown::CommandMismatch);
                }
                for (row, evidence) in Arc::make_mut(&mut segment.logical_commands)
                    .iter_mut()
                    .zip(projected)
                {
                    row.statistical_evidence = match evidence {
                        Some(evidence) => Some(
                            row.bind_current_cost_evidence(Some(&evidence))
                                .ok_or(StatisticalEvidenceUnknown::ExactBindingMismatch)?
                                .clone(),
                        ),
                        None => None,
                    };
                }
                segment.logical_payload_bytes = logical_payload_bytes(&segment.logical_commands)
                    .ok_or(StatisticalEvidenceUnknown::Capacity)?;
                segment.logical_has_statistical_evidence = segment
                    .logical_commands
                    .iter()
                    .any(|row| row.statistical_evidence.is_some());
            }
        }
        Ok(self)
    }

    pub fn retained_payload_bytes(&self) -> Option<usize> {
        self.payload_bytes(true)
    }
    /// Conservative call payload, excluding only templates with a live private
    /// budget lease. Exact logical catalogues have no such lease and remain
    /// charged, including storage which worker COW resolution may duplicate.
    pub fn call_owned_payload_bytes(&self) -> Option<usize> {
        self.payload_bytes(false)
    }
    fn payload_bytes(&self, include_leased_templates: bool) -> Option<usize> {
        let mut bytes = std::mem::size_of::<Self>()
            .checked_add(
                self.commands
                    .len()
                    .checked_mul(std::mem::size_of::<DeviceNativeWorkAttribution>())?,
            )?
            .checked_add(
                self.replayed_segments
                    .len()
                    .checked_mul(std::mem::size_of::<DeviceReplayedSegmentAttribution>())?,
            )?;
        for command in self.commands.iter() {
            if let Some(v) = &command.statistical_evidence {
                bytes = bytes.checked_add(v.retained_payload_bytes()?)?;
            }
            if let Some(v) = &command.observation {
                bytes = bytes.checked_add(if include_leased_templates {
                    v.retained_payload_bytes()
                } else {
                    v.call_owned_payload_bytes()
                })?;
            }
        }
        for segment in self.replayed_segments.iter() {
            bytes = bytes
                .checked_add(segment.reusable_executable_fingerprint.capacity())?
                .checked_add(segment.logical_payload_bytes)?;
            // Program identity owns numerical signatures and plan/layout text.
            for text in [
                &segment.program_id.runtime_implementation_fingerprint,
                &segment.program_id.program_binding_layout_fingerprint,
                &segment.program_id.lane_stable_layout_fingerprint,
            ] {
                bytes = bytes.checked_add(text.capacity())?;
            }
            bytes = bytes.checked_add(segment.program_id.plan_hash.as_str().len())?;
            bytes = bytes.checked_add(segment.program_id.bucket_id.retained_text_bytes())?;
            if let Some(v) = &segment.observation {
                bytes = bytes.checked_add(if include_leased_templates {
                    v.retained_payload_bytes()
                } else {
                    v.call_owned_payload_bytes()
                })?;
            }
        }
        Some(bytes)
    }
    pub fn projection_retained_bytes_upper_bound(&self) -> Option<usize> {
        // A queued immutable raw owner can remain alive while COW resolution
        // allocates current exact rows. Include both populations conservatively.
        self.working_bytes_from(self.retained_payload_bytes()?)
    }
    /// Simultaneous raw/COW payload plus original producer projection scratch.
    /// Shared template storage stays in its independent lease budget. This is
    /// a checked bound only; it neither resolves nor reserves/allocates memory.
    pub fn maximum_working_bytes(&self) -> Option<usize> {
        self.working_bytes_from(self.call_owned_payload_bytes()?)
    }
    fn working_bytes_from(&self, raw_bytes: usize) -> Option<usize> {
        let mut bytes = raw_bytes.checked_mul(2)?;
        for command in self.commands.iter() {
            if let Some(v) = &command.observation {
                bytes = bytes.checked_add(v.projection_retained_bytes_upper_bound())?;
            }
        }
        for segment in self.replayed_segments.iter() {
            if let Some(v) = &segment.observation {
                bytes = bytes.checked_add(v.projection_retained_bytes_upper_bound())?;
            }
        }
        Some(bytes)
    }
    pub fn maximum_resolved_bytes(&self) -> Option<usize> {
        self.projection_retained_bytes_upper_bound()
    }
}

pub(super) fn serialize_commands<S: serde::Serializer>(
    v: &Arc<[DeviceNativeWorkAttribution]>,
    s: S,
) -> Result<S::Ok, S::Error> {
    v.as_ref().serialize(s)
}
pub(super) fn serialize_segments<S: serde::Serializer>(
    v: &Arc<[DeviceReplayedSegmentAttribution]>,
    s: S,
) -> Result<S::Ok, S::Error> {
    v.as_ref().serialize(s)
}

#[cfg(test)]
mod tests;
