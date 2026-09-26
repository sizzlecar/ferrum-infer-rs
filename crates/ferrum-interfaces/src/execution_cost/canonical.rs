//! One request-independent key protocol for actual and predicted wave shapes.
//! A canonical key identifies supplied evidence; it does not prove that a
//! provider will choose that route on a future invocation.
use super::{
    project_host_cost_features, ActualRowWork, ActualWaveGraphState, ActualWaveKind,
    ActualWavePath, ActualWaveRowOrder, CanonicalWaveCostFeatures, CoreReadbackRoute,
    CostRowNumericFeatures, CostSamplingHistoryScope, HostContentCostFeaturesV1,
    HostCostFeaturesV1, HostRowMultisetCostFeaturesV2, HostRowStaticCostFeaturesV2,
    COST_NUMERIC_FEATURE_SCHEMA_V1, HOST_CONTENT_FEATURE_SCHEMA_V1,
};
use crate::vnext::{
    DeviceCommandPhase, DeviceExecutionPath, DeviceNativeWorkAttribution,
    DeviceReplayedLogicalCommandAttribution,
};
use sha2::{Digest, Sha256};

mod row_multiset;

pub const CANONICAL_WAVE_COST_SCHEMA: u32 = 1;
pub const MAX_COST_COMMANDS: usize = 8192;
pub const MAX_COST_ROWS: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CanonicalCostError {
    EmptyWave,
    Capacity,
    InvalidCommand,
    InvalidRow,
    InvalidRoute,
    EvidenceMismatch,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostCommandPath {
    Eager,
    Replayed,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostProductOutput {
    FullLogits,
    GreedyToken,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CostProviderIdentity<'a> {
    pub provider_id: &'a str,
    pub implementation_fingerprint: &'a str,
    pub operation_fingerprint: &'a str,
}
#[derive(Debug, Clone, Copy)]
pub struct CostPhysicalCommand<'a> {
    pub native_op_id: &'a str,
    pub command_index: u32,
    pub node_index: Option<u32>,
    pub command_phase: DeviceCommandPhase,
    pub provider: Option<CostProviderIdentity<'a>>,
    pub path: CostCommandPath,
    pub participant_start: u32,
    pub participant_count: u32,
    pub token_count: u64,
    pub batching_form: &'a str,
    pub compute_dispatch_count: u64,
    pub transfer_command_count: u64,
    pub reusable_graph_node_count: Option<u64>,
    pub statistical_evidence: Option<&'a super::SelectedCommandCostEvidenceV1>,
}
// Equality retains the legacy exact contract. Passive statistics must be
// compared explicitly and can never alter an execution/route equality gate.
impl PartialEq for CostPhysicalCommand<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.native_op_id == other.native_op_id
            && self.command_index == other.command_index
            && self.node_index == other.node_index
            && self.command_phase == other.command_phase
            && self.provider == other.provider
            && self.path == other.path
            && self.participant_start == other.participant_start
            && self.participant_count == other.participant_count
            && self.token_count == other.token_count
            && self.batching_form == other.batching_form
            && self.compute_dispatch_count == other.compute_dispatch_count
            && self.transfer_command_count == other.transfer_command_count
            && self.reusable_graph_node_count == other.reusable_graph_node_count
    }
}
impl Eq for CostPhysicalCommand<'_> {}

#[derive(Debug, Clone, Copy)]
pub struct CostLogicalCommand<'a> {
    pub native_op_id: &'a str,
    pub logical_command_ordinal: u32,
    pub node_index: u32,
    pub provider: CostProviderIdentity<'a>,
    pub participant_count: u32,
    pub token_count: u64,
    pub batching_form: &'a str,
    pub compute_dispatch_count: u64,
    pub transfer_command_count: u64,
    pub reusable_graph_node_count: u64,
    pub statistical_evidence: Option<&'a super::SelectedCommandCostEvidenceV1>,
}
// Preserve the original exact command comparison; passive work is checked separately.
impl PartialEq for CostLogicalCommand<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.native_op_id == other.native_op_id
            && self.logical_command_ordinal == other.logical_command_ordinal
            && self.node_index == other.node_index
            && self.provider == other.provider
            && self.participant_count == other.participant_count
            && self.token_count == other.token_count
            && self.batching_form == other.batching_form
            && self.compute_dispatch_count == other.compute_dispatch_count
            && self.transfer_command_count == other.transfer_command_count
            && self.reusable_graph_node_count == other.reusable_graph_node_count
    }
}
impl Eq for CostLogicalCommand<'_> {}
impl<'a> CostPhysicalCommand<'a> {
    /// Lossless projection of actual work evidence, not a future route promise.
    pub fn from_attribution(
        command: &'a DeviceNativeWorkAttribution,
        provider: Option<CostProviderIdentity<'a>>,
    ) -> Self {
        Self {
            native_op_id: command.native_op_id(),
            command_index: command.command_index(),
            node_index: command.node_index(),
            command_phase: command.command_phase(),
            provider,
            path: match command.execution_path() {
                DeviceExecutionPath::Eager => CostCommandPath::Eager,
                DeviceExecutionPath::Replayed => CostCommandPath::Replayed,
            },
            participant_start: command.participant_start(),
            participant_count: command.participant_count(),
            token_count: command.token_count(),
            batching_form: command.batching_form().as_str(),
            compute_dispatch_count: command.compute_dispatch_count(),
            transfer_command_count: command.transfer_command_count(),
            reusable_graph_node_count: command.reusable_graph_node_count(),
            statistical_evidence: command.statistical_evidence(),
        }
    }
}
impl<'a> CostLogicalCommand<'a> {
    pub fn from_attribution(
        command: &'a DeviceReplayedLogicalCommandAttribution,
        provider: CostProviderIdentity<'a>,
    ) -> Self {
        Self {
            native_op_id: command.native_op_id(),
            logical_command_ordinal: command.logical_command_ordinal(),
            node_index: command.node_index(),
            provider,
            participant_count: command.participant_count(),
            token_count: command.token_count(),
            batching_form: command.batching_form().as_str(),
            compute_dispatch_count: command.compute_dispatch_count(),
            transfer_command_count: command.transfer_command_count(),
            reusable_graph_node_count: command.reusable_graph_node_count(),
            statistical_evidence: command.statistical_evidence(),
        }
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CostRowOutput {
    Prefill {
        final_logits: bool,
    },
    Decode {
        requires_full_logits: bool,
        repetition_tokens: u64,
        repetition_penalty_bits: u32,
    },
}
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CanonicalCostRow {
    pub work: ActualRowWork,
    /// Already includes actual generated-history length under the host policy
    /// protocol. A missing policy cannot be represented by a zero placeholder.
    pub host_policy_signature: [u8; 32],
    pub host_features: Option<HostCostFeaturesV1>,
    pub mask_upload_required: bool,
    pub output: CostRowOutput,
}

/// Correlation IDs and allocation addresses are intentionally absent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CanonicalWaveCostShape {
    pub kind: ActualWaveKind,
    pub path: ActualWavePath,
    pub graph: ActualWaveGraphState,
    pub row_order: ActualWaveRowOrder,
    pub provider_signature: [u8; 32],
    pub output_policy_signature: [u8; 32],
    pub numeric_features: Option<CanonicalWaveCostFeatures>,
    pub host_content_features: Option<HostContentCostFeaturesV1>,
    pub row_multiset_features: Option<HostRowMultisetCostFeaturesV2>,
    pub rows: Vec<ActualRowWork>,
    pub recurrent_state_bytes: u64,
}

/// Incremental bounded hashing avoids building a second large command/JSON
/// representation on the actual observation path. All errors are sticky.
pub struct CanonicalWaveCostBuilder {
    statistical: super::statistical::StatisticalWaveAccumulator,
    structured_host: Option<super::statistical::StructuredHostAccumulator>,
    provider: Sha256,
    output: Sha256,
    numeric_output: Sha256,
    content_output: Option<Sha256>,
    row_multiset_rows: Option<Vec<HostRowStaticCostFeaturesV2>>,
    row_multiset_product: CostProductOutput,
    numeric_rows: Option<Vec<CostRowNumericFeatures>>,
    core_readback: Option<CoreReadbackRoute>,
    role_order: Sha256,
    commands: usize,
    last_command_index: Option<u32>,
    logical_commands: usize,
    rows: Vec<ActualRowWork>,
    replayed: bool,
    replayed_commands: Vec<(u32, Option<u64>)>,
    segments: usize,
    segment_remaining: usize,
    segment_total: usize,
    segment_graph_nodes: u64,
    expected_graph_nodes: Option<u64>,
    last_segment: Option<u32>,
    failed: Option<CanonicalCostError>,
}
fn number(hash: &mut Sha256, n: u64) {
    hash.update(n.to_le_bytes());
}
fn bytes(hash: &mut Sha256, value: &[u8]) {
    number(hash, value.len() as u64);
    hash.update(value);
}
fn text_valid(value: &str) -> bool {
    !value.is_empty() && value.len() <= 1024
}
fn provider_identity(
    hash: &mut Sha256,
    provider: CostProviderIdentity<'_>,
) -> Result<(), CanonicalCostError> {
    if !text_valid(provider.provider_id)
        || !text_valid(provider.implementation_fingerprint)
        || !text_valid(provider.operation_fingerprint)
    {
        return Err(CanonicalCostError::InvalidCommand);
    }
    bytes(hash, provider.provider_id.as_bytes());
    bytes(hash, provider.implementation_fingerprint.as_bytes());
    bytes(hash, provider.operation_fingerprint.as_bytes());
    Ok(())
}
impl CanonicalWaveCostBuilder {
    pub fn new(retries: u32, product: CostProductOutput) -> Self {
        let mut provider = Sha256::new();
        bytes(&mut provider, b"ferrum.canonical-wave.providers.v1");
        number(&mut provider, u64::from(CANONICAL_WAVE_COST_SCHEMA));
        number(&mut provider, u64::from(retries));
        let mut output = Sha256::new();
        bytes(&mut output, b"ferrum.canonical-wave.output.v1");
        number(
            &mut output,
            match product {
                CostProductOutput::FullLogits => 0,
                CostProductOutput::GreedyToken => 1,
            },
        );
        let mut role_order = Sha256::new();
        bytes(&mut role_order, b"ferrum.canonical-wave.ordered-roles.v1");
        let mut numeric_output = Sha256::new();
        bytes(
            &mut numeric_output,
            b"ferrum.canonical-wave.numeric-output.v1",
        );
        number(
            &mut numeric_output,
            u64::from(COST_NUMERIC_FEATURE_SCHEMA_V1),
        );
        number(
            &mut numeric_output,
            match product {
                CostProductOutput::FullLogits => 0,
                CostProductOutput::GreedyToken => 1,
            },
        );
        let mut content_output = Sha256::new();
        bytes(
            &mut content_output,
            b"ferrum.canonical-wave.empirical-host-content.v1",
        );
        number(
            &mut content_output,
            match product {
                CostProductOutput::FullLogits => 0,
                CostProductOutput::GreedyToken => 1,
            },
        );
        Self {
            statistical: super::statistical::StatisticalWaveAccumulator::new(),
            structured_host: None,
            provider,
            output,
            numeric_output,
            content_output: Some(content_output),
            row_multiset_rows: Some(Vec::new()),
            row_multiset_product: product,
            numeric_rows: Some(Vec::new()),
            core_readback: None,
            role_order,
            commands: 0,
            last_command_index: None,
            logical_commands: 0,
            rows: Vec::new(),
            replayed: false,
            replayed_commands: Vec::new(),
            segments: 0,
            segment_remaining: 0,
            segment_total: 0,
            segment_graph_nodes: 0,
            expected_graph_nodes: None,
            last_segment: None,
            failed: None,
        }
    }
    /// Explicit passive collection; the normal constructor does not retain
    /// another row representation. This does not enable a new predictor.
    pub fn new_with_structured_statistics(retries: u32, product: CostProductOutput) -> Self {
        let mut value = Self::new(retries, product);
        value.structured_host = Some(super::statistical::StructuredHostAccumulator::new(retries));
        value.statistical.capture_algorithm_work();
        value
    }
    /// Must describe the real complete-wave disposition (including a staging
    /// rollback). No call or Unknown leaves numeric evidence unavailable.
    pub fn core_readback_route(
        &mut self,
        route: CoreReadbackRoute,
    ) -> Result<(), CanonicalCostError> {
        self.guard(|this| {
            if this.core_readback.is_some() {
                return Err(CanonicalCostError::EvidenceMismatch);
            }
            this.core_readback = Some(route);
            Ok(())
        })
    }
    fn guard<T>(
        &mut self,
        f: impl FnOnce(&mut Self) -> Result<T, CanonicalCostError>,
    ) -> Result<T, CanonicalCostError> {
        if let Some(error) = self.failed {
            return Err(error);
        }
        let result = f(self);
        if let Err(error) = &result {
            self.failed = Some(*error);
        }
        result
    }
    pub fn physical_command(
        &mut self,
        command: CostPhysicalCommand<'_>,
    ) -> Result<(), CanonicalCostError> {
        self.guard(|this| {
            if this.commands >= MAX_COST_COMMANDS {
                return Err(CanonicalCostError::Capacity);
            }
            if this.segments > 0
                // Device evidence omits host-only argument-binding commands.
                // Preserve their index gaps; never renumber the physical trace.
                || this.last_command_index.is_some_and(|last| command.command_index <= last)
                || !text_valid(command.native_op_id)
                || !text_valid(command.batching_form)
                || command.node_index.is_some() != command.provider.is_some()
                || (command.node_index.is_some() && command.participant_count == 0)
                || (command.node_index.is_none() && command.participant_start != 0)
                || (command.compute_dispatch_count == 0 && command.transfer_command_count == 0)
                || (command.reusable_graph_node_count.is_some()
                    && command.path != CostCommandPath::Replayed)
                || command
                    .participant_start
                    .checked_add(command.participant_count)
                    .is_none()
            {
                return Err(CanonicalCostError::InvalidCommand);
            }
            bytes(&mut this.provider, command.native_op_id.as_bytes());
            number(&mut this.provider, u64::from(command.command_index));
            match command.node_index {
                Some(index) => {
                    number(&mut this.provider, 1);
                    number(&mut this.provider, u64::from(index));
                }
                None => number(&mut this.provider, 0),
            }
            number(
                &mut this.provider,
                match command.command_phase {
                    DeviceCommandPhase::Initialization => 0,
                    DeviceCommandPhase::DynamicBinding => 1,
                    DeviceCommandPhase::Compute => 2,
                    DeviceCommandPhase::ResultBinding => 3,
                },
            );
            match command.provider {
                Some(provider) => {
                    number(&mut this.provider, 1);
                    provider_identity(&mut this.provider, provider)?;
                }
                None => number(&mut this.provider, 0),
            }
            number(
                &mut this.provider,
                match command.path {
                    CostCommandPath::Eager => 0,
                    CostCommandPath::Replayed => {
                        this.replayed = true;
                        this.replayed_commands
                            .push((command.command_index, command.reusable_graph_node_count));
                        1
                    }
                },
            );
            number(&mut this.provider, u64::from(command.participant_start));
            number(&mut this.provider, u64::from(command.participant_count));
            number(&mut this.provider, command.token_count);
            bytes(&mut this.provider, command.batching_form.as_bytes());
            number(&mut this.provider, command.compute_dispatch_count);
            number(&mut this.provider, command.transfer_command_count);
            match command.reusable_graph_node_count {
                Some(count) => {
                    number(&mut this.provider, 1);
                    number(&mut this.provider, count);
                }
                None => number(&mut this.provider, 0),
            }
            // Passive evidence never changes exact validation or authorization.
            this.statistical.observe(command);
            this.commands += 1;
            this.last_command_index = Some(command.command_index);
            Ok(())
        })
    }
    pub fn replay_segment(
        &mut self,
        physical_command: u32,
        executable_fingerprint: &str,
        logical_count: usize,
    ) -> Result<(), CanonicalCostError> {
        self.guard(|this| {
            if this.segment_remaining != 0
                || !this.replayed
                || this
                    .last_segment
                    .is_some_and(|last| last >= physical_command)
                || !text_valid(executable_fingerprint)
                || logical_count == 0
            {
                return Err(CanonicalCostError::InvalidRoute);
            }
            let Some(&(expected_command, expected_nodes)) =
                this.replayed_commands.get(this.segments)
            else {
                return Err(CanonicalCostError::InvalidRoute);
            };
            if physical_command != expected_command || expected_nodes.is_none_or(|nodes| nodes == 0)
            {
                return Err(CanonicalCostError::InvalidRoute);
            }
            if this
                .commands
                .checked_add(this.logical_commands)
                .and_then(|count| count.checked_add(logical_count))
                .is_none_or(|count| count > MAX_COST_COMMANDS)
            {
                return Err(CanonicalCostError::Capacity);
            }
            number(&mut this.provider, u64::from(physical_command));
            bytes(&mut this.provider, executable_fingerprint.as_bytes());
            number(&mut this.provider, logical_count as u64);
            this.segment_remaining = logical_count;
            this.segment_total = logical_count;
            this.segment_graph_nodes = 0;
            this.expected_graph_nodes = expected_nodes;
            this.last_segment = Some(physical_command);
            this.segments += 1;
            this.statistical.replay_segment(
                physical_command,
                executable_fingerprint,
                logical_count,
            );
            Ok(())
        })
    }
    pub fn logical_command(
        &mut self,
        command: CostLogicalCommand<'_>,
    ) -> Result<(), CanonicalCostError> {
        self.guard(|this| {
            if this.segment_remaining == 0
                || !text_valid(command.native_op_id)
                || command.participant_count == 0
                || !text_valid(command.batching_form)
                || (command.compute_dispatch_count == 0 && command.transfer_command_count == 0)
                || command.reusable_graph_node_count == 0
                || command.logical_command_ordinal as usize
                    != this.segment_total - this.segment_remaining
            {
                return Err(CanonicalCostError::InvalidCommand);
            }
            provider_identity(&mut this.provider, command.provider)?;
            bytes(&mut this.provider, command.native_op_id.as_bytes());
            number(
                &mut this.provider,
                u64::from(command.logical_command_ordinal),
            );
            number(&mut this.provider, u64::from(command.node_index));
            number(&mut this.provider, u64::from(command.participant_count));
            number(&mut this.provider, command.token_count);
            bytes(&mut this.provider, command.batching_form.as_bytes());
            number(&mut this.provider, command.compute_dispatch_count);
            number(&mut this.provider, command.transfer_command_count);
            number(&mut this.provider, command.reusable_graph_node_count);
            this.logical_commands += 1;
            this.segment_remaining -= 1;
            this.segment_graph_nodes = this
                .segment_graph_nodes
                .checked_add(command.reusable_graph_node_count)
                .ok_or(CanonicalCostError::InvalidRoute)?;
            if this.segment_remaining == 0
                && this.expected_graph_nodes != Some(this.segment_graph_nodes)
            {
                return Err(CanonicalCostError::EvidenceMismatch);
            }
            this.statistical.logical_command(command);
            Ok(())
        })
    }
    pub fn row(&mut self, row: CanonicalCostRow) -> Result<(), CanonicalCostError> {
        self.guard(|this| {
            if this.rows.len() >= MAX_COST_ROWS {
                return Err(CanonicalCostError::Capacity);
            }
            let role = match (row.work, row.output) {
                (
                    ActualRowWork::Decode { .. },
                    CostRowOutput::Decode {
                        repetition_penalty_bits,
                        ..
                    },
                ) if f32::from_bits(repetition_penalty_bits).is_finite()
                    && f32::from_bits(repetition_penalty_bits) > 0.0 =>
                {
                    0
                }
                (
                    ActualRowWork::Prefill {
                        offset,
                        count,
                        total_prompt_tokens,
                    },
                    CostRowOutput::Prefill { final_logits },
                ) if count > 0
                    && offset.checked_add(count).is_some_and(|end| {
                        end <= total_prompt_tokens && (end == total_prompt_tokens) == final_logits
                    }) =>
                {
                    1
                }
                _ => return Err(CanonicalCostError::InvalidRow),
            };
            this.role_order.update([role]);
            if let Some(host) = row.host_features {
                let numeric = project_host_cost_features(host, row.work, row.output)
                    .map_err(|_| CanonicalCostError::InvalidRow)?;
                if let Some(static_row) = row_multiset::static_row(row, numeric) {
                    if let Some(rows) = &mut this.row_multiset_rows {
                        rows.push(static_row);
                    }
                } else {
                    this.row_multiset_rows = None;
                }
                if host.supports_empirical_plain_text_content() {
                    if let Some(hash) = &mut this.content_output {
                        number(hash, role.into());
                        bytes(hash, &host.policy.categorical_signature);
                        number(hash, host.policy.decoder_text_bytes_per_token);
                        number(hash, host.policy.decoder_scratch_bytes_per_token);
                        number(hash, host.policy.raw_token_bytes_bound);
                        number(hash, u64::from(host.state.generated_tokens_before == 0));
                        number(hash, u64::from(row.mask_upload_required));
                        number(
                            hash,
                            u64::from(
                                numeric.decoded_prefix_tokens > numeric.generated_tokens_before
                                    && numeric.decoded_prefix_tokens
                                        == numeric.maximum_output_tokens,
                            ),
                        );
                        // UTF-8 bytes/retries and which peer forced full logits
                        // are latent host content here. The real whole-wave
                        // product mode was hashed above, not inferred per row.
                        match row.output {
                            CostRowOutput::Prefill { final_logits } => {
                                number(hash, u64::from(final_logits))
                            }
                            CostRowOutput::Decode {
                                repetition_penalty_bits,
                                ..
                            } => {
                                number(hash, 2);
                                number(hash, u64::from(repetition_penalty_bits));
                            }
                        }
                    }
                } else {
                    this.content_output = None;
                }
                if let Some(rows) = &mut this.numeric_rows {
                    rows.push(numeric);
                }
                // All row roles and branch evidence remain ordered. Numeric
                // work counts, including max_tokens, are deliberately absent.
                number(&mut this.numeric_output, role.into());
                bytes(&mut this.numeric_output, &host.policy.categorical_signature);
                number(
                    &mut this.numeric_output,
                    host.policy.decoder_text_bytes_per_token,
                );
                number(
                    &mut this.numeric_output,
                    host.policy.decoder_scratch_bytes_per_token,
                );
                number(&mut this.numeric_output, host.policy.raw_token_bytes_bound);
                number(
                    &mut this.numeric_output,
                    u64::from(host.state.generated_tokens_before == 0),
                );
                number(
                    &mut this.numeric_output,
                    u64::from(host.state.pending_decoded_utf8),
                );
                number(
                    &mut this.numeric_output,
                    match host.state.sampling_history_scope {
                        CostSamplingHistoryScope::FullGeneration => 0,
                        CostSamplingHistoryScope::HiddenStructuredOutput => 1,
                        CostSamplingHistoryScope::VisibleStructuredOutput => 2,
                    },
                );
                bytes(
                    &mut this.numeric_output,
                    &host.state.completion_state_signature,
                );
                number(
                    &mut this.numeric_output,
                    u64::from(row.mask_upload_required),
                );
                number(
                    &mut this.numeric_output,
                    u64::from(
                        numeric.decoded_prefix_tokens > numeric.generated_tokens_before
                            && numeric.decoded_prefix_tokens == numeric.maximum_output_tokens,
                    ),
                );
                match row.output {
                    CostRowOutput::Prefill { final_logits } => {
                        number(&mut this.numeric_output, u64::from(final_logits));
                    }
                    CostRowOutput::Decode {
                        requires_full_logits,
                        repetition_penalty_bits,
                        ..
                    } => {
                        number(&mut this.numeric_output, 2);
                        number(&mut this.numeric_output, u64::from(requires_full_logits));
                        number(&mut this.numeric_output, u64::from(repetition_penalty_bits));
                    }
                }
            } else {
                this.numeric_rows = None;
                this.content_output = None;
                this.row_multiset_rows = None;
            }
            bytes(&mut this.output, &row.host_policy_signature);
            number(&mut this.output, u64::from(row.mask_upload_required));
            match row.output {
                CostRowOutput::Prefill { final_logits } => {
                    number(&mut this.output, u64::from(final_logits))
                }
                CostRowOutput::Decode {
                    requires_full_logits,
                    repetition_tokens,
                    repetition_penalty_bits,
                } => {
                    number(&mut this.output, 2);
                    number(&mut this.output, u64::from(requires_full_logits));
                    number(&mut this.output, repetition_tokens);
                    number(&mut this.output, u64::from(repetition_penalty_bits));
                }
            }
            if let Some(structured) = &mut this.structured_host {
                structured.observe(row);
            }
            this.rows.push(row.work);
            Ok(())
        })
    }
    /// Completes both views from one receipt; invalid or missing producer data
    /// never changes the legacy exact shape or its validation result.
    pub fn finish_with_statistics(
        mut self,
        kind: ActualWaveKind,
        path: ActualWavePath,
        graph: ActualWaveGraphState,
        row_order: ActualWaveRowOrder,
        recurrent_state_bytes: u64,
    ) -> Result<super::CanonicalStatisticalWave, CanonicalCostError> {
        let accumulator = std::mem::replace(
            &mut self.statistical,
            super::statistical::StatisticalWaveAccumulator::new(),
        );
        let exact = self.finish(kind, path, graph, row_order, recurrent_state_bytes)?;
        let statistical = accumulator.finish(&exact);
        Ok(super::CanonicalStatisticalWave { exact, statistical })
    }

    /// Transport opt-in structure beside existing statistics. Default builders
    /// take the original path without allocating structured rows or hashing them.
    pub fn finish_with_captured_structure(
        self,
        kind: ActualWaveKind,
        path: ActualWavePath,
        graph: ActualWaveGraphState,
        row_order: ActualWaveRowOrder,
        recurrent_state_bytes: u64,
    ) -> Result<super::CanonicalStatisticalWave, CanonicalCostError> {
        if self.structured_host.is_none() {
            return self.finish_with_statistics(
                kind,
                path,
                graph,
                row_order,
                recurrent_state_bytes,
            );
        }
        let result =
            self.finish_with_structure(kind, path, graph, row_order, recurrent_state_bytes)?;
        let statistical = result
            .statistical
            .map(|value| value.attach_structured_capture(result.structured, &result.exact));
        Ok(super::CanonicalStatisticalWave {
            exact: result.exact,
            statistical,
        })
    }

    /// Completes an opt-in, pre-host-settlement structural recipe. All legacy
    /// canonical validation still runs. Missing structure cannot turn invalid
    /// execution into valid execution or produce a trainable observation.
    pub fn finish_with_structure(
        mut self,
        kind: ActualWaveKind,
        path: ActualWavePath,
        graph: ActualWaveGraphState,
        row_order: ActualWaveRowOrder,
        recurrent_state_bytes: u64,
    ) -> Result<super::CanonicalStructuredWave, CanonicalCostError> {
        let host = self.structured_host.take();
        let product = self.row_multiset_product;
        let readback = self.core_readback;
        let mut accumulator = std::mem::replace(
            &mut self.statistical,
            super::statistical::StatisticalWaveAccumulator::new(),
        );
        let exact = self.finish(kind, path, graph, row_order, recurrent_state_bytes)?;
        let structured = host
            .ok_or(super::StatisticalEvidenceUnknown::MissingProducer)
            .and_then(|rows| {
                let device =
                    accumulator.structured_device(&exact, product, readback, rows.retries)?;
                super::UnsettledStructuredWaveEvidenceV1::finish(rows, device, &exact)
            });
        let statistical = accumulator.finish(&exact);
        Ok(super::CanonicalStructuredWave {
            exact,
            statistical,
            structured,
        })
    }

    pub fn finish(
        mut self,
        kind: ActualWaveKind,
        path: ActualWavePath,
        graph: ActualWaveGraphState,
        row_order: ActualWaveRowOrder,
        recurrent_state_bytes: u64,
    ) -> Result<CanonicalWaveCostShape, CanonicalCostError> {
        if let Some(error) = self.failed {
            return Err(error);
        }
        if self.commands == 0 || self.rows.is_empty() {
            return Err(CanonicalCostError::EmptyWave);
        }
        if self.segment_remaining != 0
            || (self.replayed && self.segments != self.replayed_commands.len())
        {
            return Err(CanonicalCostError::InvalidRoute);
        }
        let decode = self
            .rows
            .iter()
            .any(|row| matches!(row, ActualRowWork::Decode { .. }));
        let prefill = self
            .rows
            .iter()
            .any(|row| matches!(row, ActualRowWork::Prefill { .. }));
        if !matches!(
            (kind, decode, prefill),
            (ActualWaveKind::Decode, true, false)
                | (ActualWaveKind::Prefill, false, true)
                | (ActualWaveKind::Mixed, true, true)
        ) {
            return Err(CanonicalCostError::InvalidRow);
        }
        let raw: [u8; 32] = self.provider.finalize().into();
        let provider_signature = if row_order == ActualWaveRowOrder::Ordered {
            let mut order = self.role_order;
            order.update(raw);
            order.finalize().into()
        } else {
            raw
        };
        let numeric_features = match (self.numeric_rows, self.core_readback) {
            (Some(rows), Some(route)) if route != CoreReadbackRoute::Unknown => {
                number(
                    &mut self.numeric_output,
                    match route {
                        CoreReadbackRoute::SubmissionStaged => 0,
                        CoreReadbackRoute::HostSynchronized => 1,
                        CoreReadbackRoute::NoReadback => 2,
                        CoreReadbackRoute::SubmissionFallbackSynchronized => 3,
                        CoreReadbackRoute::Unknown => unreachable!(),
                    },
                );
                number(
                    &mut self.numeric_output,
                    match row_order {
                        ActualWaveRowOrder::Ordered => 0,
                        ActualWaveRowOrder::IndependentRows => 1,
                    },
                );
                let features = CanonicalWaveCostFeatures {
                    schema_version: COST_NUMERIC_FEATURE_SCHEMA_V1,
                    output_policy_signature: self.numeric_output.finalize().into(),
                    rows,
                };
                features
                    .validate(self.rows.len())
                    .map_err(|_| CanonicalCostError::InvalidRow)?;
                Some(features)
            }
            _ => None,
        };
        let host_content_features = self
            .content_output
            .filter(|_| numeric_features.is_some())
            .map(|mut hash| {
                number(
                    &mut hash,
                    match self.core_readback.expect("numeric readback checked") {
                        CoreReadbackRoute::SubmissionStaged => 0,
                        CoreReadbackRoute::HostSynchronized => 1,
                        CoreReadbackRoute::NoReadback => 2,
                        CoreReadbackRoute::SubmissionFallbackSynchronized => 3,
                        CoreReadbackRoute::Unknown => unreachable!(),
                    },
                );
                number(
                    &mut hash,
                    match row_order {
                        ActualWaveRowOrder::Ordered => 0,
                        ActualWaveRowOrder::IndependentRows => 1,
                    },
                );
                HostContentCostFeaturesV1 {
                    schema_version: HOST_CONTENT_FEATURE_SCHEMA_V1,
                    output_policy_signature: hash.finalize().into(),
                }
            });
        let row_multiset_features = self
            .row_multiset_rows
            .filter(|_| numeric_features.is_some())
            .map(|rows| {
                row_multiset::finish(
                    rows,
                    self.row_multiset_product,
                    self.core_readback.expect("numeric readback checked"),
                    row_order,
                )
            });
        Ok(CanonicalWaveCostShape {
            kind,
            path,
            graph,
            row_order,
            provider_signature,
            output_policy_signature: self.output.finalize().into(),
            numeric_features,
            host_content_features,
            row_multiset_features,
            rows: self.rows,
            recurrent_state_bytes,
        })
    }
}

/// The engine and hypothetical row projection must use this same protocol.
pub fn host_history_cost_signature(cached_policy: [u8; 32], generated_tokens: u64) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"ferrum.engine.host-history-policy.v1\0");
    hash.update(cached_policy);
    hash.update(generated_tokens.to_le_bytes());
    hash.finalize().into()
}

#[cfg(test)]
mod tests;
