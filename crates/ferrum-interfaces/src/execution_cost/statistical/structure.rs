//! Passive, opt-in inputs for a future structured model, NOT a qualified sample.
//! No profile loader/predictor consumes this protocol. Host settlement and
//! per-algorithm numeric work are never invented from aggregate work.
use super::*;

pub const STRUCTURED_COST_INPUT_PROTOCOL_V1: &str = "ferrum.structured-cost-input.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum StructuredCostProductV1 {
    FullLogits,
    GreedyToken,
}
impl From<CostProductOutput> for StructuredCostProductV1 {
    fn from(value: CostProductOutput) -> Self {
        match value {
            CostProductOutput::FullLogits => Self::FullLogits,
            CostProductOutput::GreedyToken => Self::GreedyToken,
        }
    }
}

/// Output-capacity facts before host processing, not actual finish reasons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum HostTerminalExpectationV1 {
    /// A non-final prefill does not produce a token in this wave.
    NoTokenProduced,
    /// A token is produced below the length limit. EOS/stop may still terminate.
    TokenMayTerminate,
    /// This token reaches the declared length boundary, but cleanup is unobserved.
    LengthBoundary,
}

/// Physical position is NOT HostStageEvidenceV1::host_processing_ordinal.
/// The latter becomes available only from real post-execution host processing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct StructuredHostRowV1 {
    pub physical_position: u32,
    pub role: HostRowRoleV2,
    pub installed_policy: HostCostPolicyV2,
    pub no_generated_history: bool,
    pub pending_decoded_utf8: bool,
    pub initial_prefill: bool,
    pub final_prefill: bool,
    pub mask_upload_required: bool,
    /// None for prefill: final-prefill output is not a FullLogits product claim.
    pub decode_requires_full_logits: Option<bool>,
    pub repetition_penalty_bits: Option<u32>,
    pub terminal_expectation: HostTerminalExpectationV1,
}

/// The source is the same checked selected-command stream used by canonical
/// actual/future builders. No hash is decoded or reconstructed from a profile.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct DeviceRouteTemplateV1 {
    ordered_template: [u8; 32],
    /// Only existing provider-declared independent attention row groups are
    /// normalized. Ordinary commands retain order. This is not wave/host
    /// permutation authority, nor qualification of a new statistical family.
    provider_grouped_template: Option<[u8; 32]>,
    physical_commands: u32,
    product: StructuredCostProductV1,
    readback: CoreReadbackRoute,
    retries: u32,
    /// Existing total launch work; this is NOT per-algorithm regression work.
    aggregate_work: DeviceNumericWorkV1,
    algorithm_work: Result<DeviceAlgorithmWorkEvidenceV1, StatisticalEvidenceUnknown>,
}
impl DeviceRouteTemplateV1 {
    pub fn ordered_template(&self) -> &[u8; 32] {
        &self.ordered_template
    }
    pub fn provider_grouped_template(&self) -> Option<&[u8; 32]> {
        self.provider_grouped_template.as_ref()
    }
    pub fn physical_commands(&self) -> u32 {
        self.physical_commands
    }
    pub fn product(&self) -> StructuredCostProductV1 {
        self.product
    }
    pub fn readback(&self) -> CoreReadbackRoute {
        self.readback
    }
    pub fn retries(&self) -> u32 {
        self.retries
    }
    /// Complete command-bound inputs, or explicit missing/partial evidence.
    pub fn algorithm_work(
        &self,
    ) -> Result<&DeviceAlgorithmWorkEvidenceV1, StatisticalEvidenceUnknown> {
        self.algorithm_work.as_ref().map_err(|error| *error)
    }
    pub fn aggregate_work(&self) -> DeviceNumericWorkV1 {
        self.aggregate_work
    }
    pub(super) fn from_selected_stream(
        ordered: [u8; 32],
        grouped: Option<[u8; 32]>,
        physical_commands: usize,
        work: DeviceNumericWorkV1,
        shape: &CanonicalWaveCostShape,
        product: CostProductOutput,
        readback: Option<CoreReadbackRoute>,
        retries: u32,
        algorithm_work: Result<DeviceAlgorithmWorkEvidenceV1, StatisticalEvidenceUnknown>,
    ) -> Result<Self, StatisticalEvidenceUnknown> {
        if physical_commands == 0 || physical_commands > MAX_COST_COMMANDS {
            return Err(StatisticalEvidenceUnknown::MissingProducer);
        }
        // Sealed current-wave Graph recipes require their own adapter. Never
        // reclassify a physical replay as an eager selected stream.
        if shape.path != ActualWavePath::PlanRuntime
            || shape.graph != ActualWaveGraphState::Disabled
        {
            return Err(StatisticalEvidenceUnknown::UnsupportedReplay);
        }
        let readback = readback
            .filter(|route| *route != CoreReadbackRoute::Unknown)
            .ok_or(StatisticalEvidenceUnknown::MissingHostDomain)?;
        let bind = |commands: [u8; 32]| {
            let mut hash = Sha256::new();
            bytes(&mut hash, b"ferrum.structured-device-route.v1");
            hash.update(commands);
            for value in [
                shape.kind as u64,
                shape.path as u64,
                shape.graph as u64,
                shape.row_order as u64,
                shape.recurrent_state_bytes,
                u64::from(product == CostProductOutput::GreedyToken),
                match readback {
                    CoreReadbackRoute::SubmissionStaged => 0,
                    CoreReadbackRoute::SubmissionFallbackSynchronized => 1,
                    CoreReadbackRoute::HostSynchronized => 2,
                    CoreReadbackRoute::NoReadback => 3,
                    CoreReadbackRoute::Unknown => unreachable!(),
                },
                u64::from(retries),
            ] {
                number(&mut hash, value);
            }
            hash.finalize().into()
        };
        Ok(Self {
            ordered_template: bind(ordered),
            provider_grouped_template: grouped.map(bind),
            physical_commands: physical_commands as u32,
            product: product.into(),
            readback,
            retries,
            aggregate_work: work,
            algorithm_work,
        })
    }
}

/// Serialize-only passive recipe. It cannot be imported as profile8/9 or
/// passed to a fitter. No public bool/setter can claim successful settlement.
/// Future host behavior (including EOS) and extra terminal work still need
/// their real typed domain/receipt before any new model can accept a sample.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct UnsettledStructuredWaveEvidenceV1 {
    protocol: &'static str,
    exact_binding: [u8; 32],
    device: DeviceRouteTemplateV1,
    physical_host_rows: Vec<StructuredHostRowV1>,
}
impl UnsettledStructuredWaveEvidenceV1 {
    /// Actual physical-host Vec capacity; independent of retention accounting units.
    pub fn physical_host_capacity(&self) -> usize {
        self.physical_host_rows.capacity()
    }
    /// Conservative row-equivalent retention units, NOT actual host rows.
    /// Auxiliary bytes round up by CostRowNumericFeatures, a row size already
    /// covered by recorder/FIFO/export accounting. The existing outer recipe Arc
    /// is conservatively charged for each retained reference by its consumers.
    pub fn retained_rows(&self) -> usize {
        self.retained_units().unwrap_or(usize::MAX)
    }
    pub fn retained_units(&self) -> Result<usize, StatisticalEvidenceUnknown> {
        let bytes = self
            .device
            .algorithm_work
            .as_ref()
            .map_or(Ok(0), |work| work.retained_dynamic_bytes())?;
        let unit = std::mem::size_of::<CostRowNumericFeatures>();
        let units = bytes
            .checked_add(unit - 1)
            .ok_or(StatisticalEvidenceUnknown::Overflow)?
            / unit;
        self.physical_host_capacity()
            .checked_add(units)
            .ok_or(StatisticalEvidenceUnknown::Overflow)
    }
    /// Backing allocation of one retained recipe Arc, including unused Vec
    /// capacity and both Arc counters. Shared references may conservatively
    /// charge this amount more than once. No per-command Arc is retained here.
    pub fn retained_bytes(&self) -> Result<usize, StatisticalEvidenceUnknown> {
        let auxiliary = self
            .device
            .algorithm_work
            .as_ref()
            .map_or(Ok(0), |work| work.retained_dynamic_bytes())?;
        self.physical_host_capacity()
            .checked_mul(std::mem::size_of::<StructuredHostRowV1>())
            .and_then(|bytes| bytes.checked_add(auxiliary))
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<Self>()))
            .and_then(|bytes| bytes.checked_add(2 * std::mem::size_of::<usize>()))
            .ok_or(StatisticalEvidenceUnknown::Overflow)
    }
    pub fn algorithm_work(
        &self,
    ) -> Result<&DeviceAlgorithmWorkEvidenceV1, StatisticalEvidenceUnknown> {
        self.device.algorithm_work()
    }
    pub fn validate_actual(
        &self,
        shape: &ActualWaveShape,
    ) -> Result<(), StatisticalEvidenceUnknown> {
        if shape.restore_bytes != 0 || shape.maintenance_bytes != 0 || shape.maintenance_units != 0
        {
            return Err(StatisticalEvidenceUnknown::UnsupportedWave);
        }
        let digest = super::wave::exact_binding_parts(
            shape.kind,
            shape.path,
            shape.graph,
            shape.row_order,
            shape.provider_signature,
            shape.output_policy_signature,
            shape.recurrent_state_bytes,
            shape.numeric_features.as_ref(),
            shape.row_multiset_features.as_ref(),
            shape.rows.iter().map(|row| row.work),
        )?;
        if self.protocol != STRUCTURED_COST_INPUT_PROTOCOL_V1
            || self.exact_binding != digest
            || self.physical_host_rows.len() != shape.rows.len()
            || self.physical_host_capacity() > MAX_COST_ROWS
        {
            return Err(StatisticalEvidenceUnknown::ExactBindingMismatch);
        }
        self.retained_units()?;
        if let Ok(work) = self.algorithm_work() {
            work.validate_binding(digest)?;
        }
        Ok(())
    }
    pub fn device(&self) -> &DeviceRouteTemplateV1 {
        &self.device
    }
    pub fn physical_host_rows(&self) -> &[StructuredHostRowV1] {
        &self.physical_host_rows
    }
    pub fn validate_exact(
        &self,
        shape: &CanonicalWaveCostShape,
    ) -> Result<(), StatisticalEvidenceUnknown> {
        if self.protocol != STRUCTURED_COST_INPUT_PROTOCOL_V1
            || self.physical_host_rows.len() != shape.rows.len()
            || self.physical_host_rows.capacity() > MAX_COST_ROWS
            || self.exact_binding != super::wave::exact_binding(shape)?
        {
            return Err(StatisticalEvidenceUnknown::ExactBindingMismatch);
        }
        self.retained_units()?;
        if let Ok(work) = self.algorithm_work() {
            work.validate_binding(self.exact_binding)?;
        }
        Ok(())
    }
    pub(in crate::execution_cost) fn finish(
        rows: StructuredHostAccumulator,
        device: DeviceRouteTemplateV1,
        shape: &CanonicalWaveCostShape,
    ) -> Result<Self, StatisticalEvidenceUnknown> {
        if let Some(error) = rows.failure {
            return Err(error);
        }
        let value = Self {
            protocol: STRUCTURED_COST_INPUT_PROTOCOL_V1,
            exact_binding: super::wave::exact_binding(shape)?,
            device,
            physical_host_rows: rows.rows,
        };
        value.validate_exact(shape)?;
        Ok(value)
    }
}

/// Separate result type leaves legacy exact/statistical fields, equality and
/// serialization unchanged. A successful `structured` is still UNSETTLED.
#[derive(Debug, Clone)]
pub struct CanonicalStructuredWave {
    pub exact: CanonicalWaveCostShape,
    pub statistical: Result<StatisticalWaveEvidenceV1, StatisticalEvidenceUnknown>,
    pub structured: Result<UnsettledStructuredWaveEvidenceV1, StatisticalEvidenceUnknown>,
}

pub(in crate::execution_cost) struct StructuredHostAccumulator {
    pub retries: u32,
    rows: Vec<StructuredHostRowV1>,
    failure: Option<StatisticalEvidenceUnknown>,
}
impl StructuredHostAccumulator {
    pub fn new(retries: u32) -> Self {
        Self {
            retries,
            rows: Vec::new(),
            failure: None,
        }
    }
    pub fn observe(&mut self, row: CanonicalCostRow) {
        if self.failure.is_none() {
            if let Err(error) = self.append(row) {
                self.failure = Some(error);
            }
        }
    }
    fn append(&mut self, row: CanonicalCostRow) -> Result<(), StatisticalEvidenceUnknown> {
        if self.rows.len() == MAX_COST_ROWS {
            return Err(StatisticalEvidenceUnknown::Capacity);
        }
        let host = row
            .host_features
            .filter(HostCostFeaturesV1::supports_empirical_plain_text_content)
            .ok_or(StatisticalEvidenceUnknown::MissingHostDomain)?;
        let numeric = project_host_cost_features(host, row.work, row.output)
            .map_err(|_| StatisticalEvidenceUnknown::MissingHostDomain)?;
        let (role, initial_prefill, final_prefill, decode_requires_full_logits, penalty) =
            match (row.work, row.output) {
                (
                    ActualRowWork::Prefill { offset, .. },
                    CostRowOutput::Prefill { final_logits },
                ) => (
                    HostRowRoleV2::Prefill,
                    offset == 0,
                    final_logits,
                    None,
                    None,
                ),
                (
                    ActualRowWork::Decode { .. },
                    CostRowOutput::Decode {
                        requires_full_logits,
                        repetition_penalty_bits,
                        ..
                    },
                ) => (
                    HostRowRoleV2::Decode,
                    false,
                    false,
                    Some(requires_full_logits),
                    Some(repetition_penalty_bits),
                ),
                _ => return Err(StatisticalEvidenceUnknown::MissingHostDomain),
            };
        let terminal_expectation =
            if numeric.decoded_prefix_tokens == numeric.generated_tokens_before {
                HostTerminalExpectationV1::NoTokenProduced
            } else if numeric.decoded_prefix_tokens == numeric.maximum_output_tokens {
                HostTerminalExpectationV1::LengthBoundary
            } else {
                HostTerminalExpectationV1::TokenMayTerminate
            };
        // The limit is an existing physical-row contract, not an arbitrary
        // fixed shape. No large allocation on the default builder path.
        if self.rows.len() == self.rows.capacity() {
            self.rows
                .try_reserve_exact((MAX_COST_ROWS - self.rows.len()).min(8))
                .map_err(|_| StatisticalEvidenceUnknown::Capacity)?;
        }
        if self.rows.capacity() > MAX_COST_ROWS {
            return Err(StatisticalEvidenceUnknown::Capacity);
        }
        self.rows.push(StructuredHostRowV1 {
            physical_position: self.rows.len() as u32,
            role,
            installed_policy: host.policy,
            no_generated_history: host.state.generated_tokens_before == 0,
            pending_decoded_utf8: host.state.pending_decoded_utf8,
            initial_prefill,
            final_prefill,
            mask_upload_required: row.mask_upload_required,
            decode_requires_full_logits,
            repetition_penalty_bits: penalty,
            terminal_expectation,
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests;
