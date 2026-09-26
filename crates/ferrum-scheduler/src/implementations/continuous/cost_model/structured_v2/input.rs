use super::super::statistical::StatisticalModelInputV1;
use super::*;
use ferrum_interfaces::execution_cost::{
    AlgorithmWorkKindV1, DeviceNumericWorkV1, HostRowRoleV2, HostTerminalExpectationV1,
};
use sha2::{Digest, Sha256};

#[derive(Debug, Clone, PartialEq)]
pub struct StructuredInputV2 {
    pub(super) owner: StructuredOwnerKeyV2,
    pub(super) domain: [u8; 32],
    pub(super) basis: Vec<f64>,
    pub(super) support: Vec<u64>,
    pub(super) physical_host_rows: Vec<StructuredHostRowV1>,
    pub(super) pending_positions: Vec<u32>,
    pub(super) length_positions: Vec<u32>,
    pub(super) pending_basis_offset: usize,
    pub(super) pending_support_offset: usize,
}
#[derive(Debug, Clone)]
pub struct StructuredQueryV2 {
    pub(super) input: StructuredInputV2,
    /// None preserves Exact versus conditional-empty Unresolved.
    pub(super) pending: Option<PendingQuery>,
}
#[derive(Debug, Clone)]
pub(super) struct PendingQuery {
    pub eligible: Vec<u32>,
    pub constraint: HostPendingConstraintV2,
}
impl StructuredInputV2 {
    /// Private numerical reconstruction, not a deserializer for a live receipt
    /// or forecast. Original Prepared and settled wire bindings are checked by
    /// the profile10 importer before invoking this shared projection.
    pub(in crate::implementations::continuous) fn from_replay_parts(
        exact: &CanonicalWaveCostShape,
        selected: &StatisticalWaveEvidenceV1,
        ordered: [u8; 32],
        grouped: Option<[u8; 32]>,
        product: StructuredProductV2,
        readback: CoreReadbackRoute,
        rows: &[StructuredHostRowV1],
        algorithms: &[([u8; 32], AlgorithmWorkKindV1, u64, DeviceNumericWorkV1)],
        replay_counts: Option<[u64; 3]>,
    ) -> Result<(Self, StructuredOwnerFactsV2)> {
        match (exact.graph, replay_counts) {
            (
                ferrum_interfaces::execution_cost::ActualWaveGraphState::Disabled
                | ferrum_interfaces::execution_cost::ActualWaveGraphState::ConfiguredEager,
                None,
            )
            | (ferrum_interfaces::execution_cost::ActualWaveGraphState::Warm, Some(_)) => {}
            _ => return Err(StructuredUnknown::MissingEvidence),
        }
        let facts = StructuredOwnerFactsV2::from_replay_parts(
            rows, product, readback, ordered, grouped, algorithms,
        )?;
        let owner = facts.owner_key()?;
        let old = StatisticalModelInputV1::from_future_structured_v2(exact, selected)
            .map_err(|_| StructuredUnknown::MissingEvidence)?;
        let input = Self::project_numeric(
            owner,
            &old,
            rows,
            algorithms.iter().map(|(_, k, n, w)| (*k, *n, *w)),
            replay_counts,
        )?;
        Ok((input, facts))
    }
    pub fn owner_for(
        exact: &CanonicalWaveCostShape,
        selected: &StatisticalWaveEvidenceV1,
        recipe: &Arc<UnsettledStructuredWaveEvidenceV1>,
    ) -> Result<StructuredOwnerKeyV2> {
        StructuredOwnerFactsV2::from_prepared(exact, selected, recipe)?.owner_key()
    }
    pub fn from_actual(
        exact: &CanonicalWaveCostShape,
        selected: &StatisticalWaveEvidenceV1,
        recipe: &Arc<UnsettledStructuredWaveEvidenceV1>,
    ) -> Result<Self> {
        let owner = Self::owner_for(exact, selected, recipe)?;
        let old = StatisticalModelInputV1::from_future_structured_v2(exact, selected)
            .map_err(|_| StructuredUnknown::MissingEvidence)?;
        let algorithms = recipe
            .algorithm_work()
            .map_err(|_| StructuredUnknown::MissingEvidence)?;
        Self::project_numeric(
            owner,
            &old,
            recipe.physical_host_rows(),
            algorithms
                .entries()
                .iter()
                .map(|a| (a.kind(), a.commands(), a.work())),
            recipe.device().replay_work().map(|r| {
                [
                    u64::from(r.replayed_segments()),
                    u64::from(r.logical_commands()),
                    r.native_graph_nodes(),
                ]
            }),
        )
    }
    pub fn owner(&self) -> &StructuredOwnerKeyV2 {
        &self.owner
    }
    pub fn domain_signature(&self) -> &[u8; 32] {
        &self.domain
    }
    pub fn regression_axes(&self) -> &[f64] {
        &self.basis
    }
    pub fn joint_support_coordinates(&self) -> &[u64] {
        &self.support
    }
    pub fn physical_host_rows(&self) -> &[StructuredHostRowV1] {
        &self.physical_host_rows
    }

    fn project_numeric(
        owner: StructuredOwnerKeyV2,
        old: &StatisticalModelInputV1,
        rows: &[StructuredHostRowV1],
        algorithms: impl Iterator<Item = (AlgorithmWorkKindV1, u64, DeviceNumericWorkV1)>,
        replay_counts: Option<[u64; 3]>,
    ) -> Result<Self> {
        if rows.len() != owner.rows as usize || rows.is_empty() || rows.len() > 128 {
            return Err(StructuredUnknown::InvalidInput);
        }
        let mut pending_positions = Vec::new();
        let mut length_positions = Vec::new();
        // Counts describe actual work classes. Physical position remains audit
        // evidence; none of these bits is interpreted as observed host order.
        let mut host_counts = [0u64; 7];
        for (position, row) in rows.iter().enumerate() {
            if row.physical_position as usize != position {
                return Err(StructuredUnknown::InvalidInput);
            }
            match row.role {
                HostRowRoleV2::Decode => {
                    if row.decode_requires_full_logits.is_none()
                        || row.repetition_penalty_bits.is_none()
                        || row.initial_prefill
                        || row.final_prefill
                    {
                        return Err(StructuredUnknown::InvalidInput);
                    }
                    host_counts[0] += 1;
                }
                HostRowRoleV2::Prefill => {
                    if row.decode_requires_full_logits.is_some()
                        || row.repetition_penalty_bits.is_some()
                    {
                        return Err(StructuredUnknown::InvalidInput);
                    }
                    host_counts[1] += 1;
                }
            }
            host_counts[2] += u64::from(row.no_generated_history);
            host_counts[3] += u64::from(row.initial_prefill);
            host_counts[4] += u64::from(row.final_prefill);
            host_counts[5] += u64::from(row.mask_upload_required);
            match row.terminal_expectation {
                HostTerminalExpectationV1::NoTokenProduced => {
                    if row.role != HostRowRoleV2::Prefill || row.final_prefill {
                        return Err(StructuredUnknown::InvalidInput);
                    }
                }
                HostTerminalExpectationV1::TokenMayTerminate => host_counts[6] += 1,
                HostTerminalExpectationV1::LengthBoundary => {
                    host_counts[6] += 1;
                    length_positions.push(position as u32);
                }
            }
            if row.pending_decoded_utf8 {
                pending_positions.push(position as u32);
            }
        }
        let mut domain = Sha256::new();
        domain.update(MODEL_REVISION_V2.as_bytes());
        let owner_bytes =
            serde_json::to_vec(&owner).map_err(|_| StructuredUnknown::InvalidInput)?;
        domain.update(owner_bytes);
        let mut basis = vec![1.0];
        let mut support = Vec::new();
        for (kind, commands, work) in algorithms {
            kind.validate_work(work)
                .map_err(|_| StructuredUnknown::InvalidInput)?;
            support.push(commands);
            support.extend(device_coordinates(work));
            basis.push(commands as f64);
            if kind == AlgorithmWorkKindV1::Kernel {
                basis.extend([
                    work.inner_work_units as f64,
                    work.padded_units as f64,
                    work.grid_blocks as f64,
                ]);
            } else if kind == AlgorithmWorkKindV1::LibraryCall {
                // API output elements and reduction work, never a native grid.
                basis.extend([work.logical_units as f64, work.inner_work_units as f64]);
            } else {
                let bytes = work
                    .host_to_device_bytes
                    .checked_add(work.device_to_host_bytes)
                    .and_then(|n| n.checked_add(work.device_to_device_bytes))
                    .and_then(|n| n.checked_add(work.fill_bytes))
                    .ok_or(StructuredUnknown::InvalidInput)?;
                basis.push(bytes as f64);
            }
        }
        if let Some(counts) = replay_counts {
            if counts.iter().any(|n| *n == 0)
                || counts[0] > ferrum_interfaces::execution_cost::MAX_COST_COMMANDS as u64
                || counts[1] > ferrum_interfaces::execution_cost::MAX_COST_COMMANDS as u64
                || counts[1] < counts[0]
                || counts[2] < counts[1]
            {
                return Err(StructuredUnknown::MissingEvidence);
            }
            // Shared topology/domain excludes resident binding. Actual graph
            // work, including launch and logical counts, remains numeric and
            // must be inside the same joint support as all other coordinates.
            domain.update(b"ferrum.structured-replay-coordinates.v1\0");
            support.extend(counts);
            basis.extend(counts.map(|n| n as f64));
        }
        let h = old.host_and_sequence();

        support.extend([
            h.rows,
            h.prefill_tokens,
            h.attention_pairs,
            h.kv_tokens_sum,
            h.kv_tokens_max,
            h.prompt_tokens_sum,
            h.prompt_tokens_max,
            h.generated_tokens_sum,
            h.sampling_history_sum,
            h.sampling_history_max,
            h.repetition_tokens_sum,
            h.decoded_prefix_sum,
            h.decoded_text_bytes_sum,
            h.decode_scratch_bytes_sum,
            h.recurrent_bytes,
        ]);
        basis.extend([
            h.prefill_tokens as f64,
            h.attention_pairs as f64,
            h.sampling_history_sum as f64,
            h.decoded_text_bytes_sum as f64,
            h.decode_scratch_bytes_sum as f64,
        ]);
        support.extend(host_counts);
        basis.extend(host_counts.map(|x| x as f64));
        let length = position_moments(&length_positions)?;
        support.extend(length);
        basis.extend(length.map(|x| x as f64));
        let pending_basis_offset = basis.len();
        let pending_support_offset = support.len();
        let pending = position_moments(&pending_positions)?;
        support.extend(pending);
        basis.extend(pending.map(|x| x as f64));
        let value = Self {
            owner,
            domain: domain.finalize().into(),
            basis,
            support,
            physical_host_rows: rows.to_vec(),
            pending_positions,
            length_positions,
            pending_basis_offset,
            pending_support_offset,
        };
        let construction_limits = StructuredSettingsV2 {
            max_axes: 4096,
            ..Default::default()
        };
        value.validate(&construction_limits)?;
        Ok(value)
    }
    pub(super) fn validate(&self, settings: &StructuredSettingsV2) -> Result<()> {
        if self.owner.rows as usize != self.physical_host_rows.len()
            || self.owner.rows == 0
            || self.domain == [0; 32]
            || self.basis.first() != Some(&1.0)
            || self
                .basis
                .iter()
                .any(|v| !v.is_finite() || *v < 0. || *v > (1u64 << 53) as f64)
            || self.support.iter().any(|v| *v > (1u64 << 53))
            || self.pending_basis_offset < 3
            || self.pending_support_offset < 3
            || self.pending_basis_offset.checked_add(3) != Some(self.basis.len())
            || self.pending_support_offset.checked_add(3) != Some(self.support.len())
        {
            return Err(StructuredUnknown::InvalidInput);
        }
        if self.basis.len() > settings.max_axes || self.support.len() > settings.max_axes {
            return Err(StructuredUnknown::Capacity);
        }
        let pending = position_moments(&self.pending_positions)?;
        let lengths = position_moments(&self.length_positions)?;
        if self.support[self.pending_support_offset..] != pending
            || self.basis[self.pending_basis_offset..] != pending.map(|v| v as f64)
            || self.support[self.pending_support_offset - 3..self.pending_support_offset] != lengths
            || self.basis[self.pending_basis_offset - 3..self.pending_basis_offset]
                != lengths.map(|v| v as f64)
        {
            return Err(StructuredUnknown::InvalidInput);
        }
        Ok(())
    }
    pub(super) fn same_domain(&self, other: &Self) -> Result<()> {
        if self.owner != other.owner
            || self.domain != other.domain
            || self.basis.len() != other.basis.len()
            || self.support.len() != other.support.len()
        {
            return Err(StructuredUnknown::WrongDomain);
        }
        Ok(())
    }
}
impl StructuredQueryV2 {
    pub fn exact(input: StructuredInputV2) -> Self {
        Self {
            input,
            pending: None,
        }
    }
    pub fn from_future(
        exact: &CanonicalWaveCostShape,
        selected: &StatisticalWaveEvidenceV1,
        recipe: &Arc<UnsettledStructuredWaveEvidenceV1>,
        forecast: &HostContentForecastV2,
    ) -> Result<Self> {
        forecast
            .validate(exact, recipe)
            .map_err(|_| StructuredUnknown::MissingEvidence)?;
        let input = StructuredInputV2::from_actual(exact, selected, recipe)?;
        let pending = match forecast {
            HostContentForecastV2::Exact => None,
            HostContentForecastV2::Unresolved(p) => Some(PendingQuery {
                eligible: p.eligible_positions().to_vec(),
                constraint: p.constraint(),
            }),
        };
        Ok(Self { input, pending })
    }
    pub fn owner(&self) -> &StructuredOwnerKeyV2 {
        self.input.owner()
    }
    pub fn domain_signature(&self) -> &[u8; 32] {
        self.input.domain_signature()
    }
}
pub(super) fn position_moments(positions: &[u32]) -> Result<[u64; 3]> {
    if positions.len() > 128
        || positions.windows(2).any(|v| v[0] >= v[1])
        || positions.iter().any(|p| *p >= 128)
    {
        return Err(StructuredUnknown::InvalidInput);
    }
    let mut out = [0; 3];
    for p in positions {
        let p = u64::from(*p) + 1;
        out[0] += 1;
        out[1] += p;
        out[2] += p * p;
    }
    Ok(out)
}
fn device_coordinates(d: DeviceNumericWorkV1) -> [u64; 10] {
    [
        d.logical_units,
        d.padded_units,
        d.inner_work_units,
        d.grid_blocks,
        d.peak_scratch_bytes,
        d.staged_weight_bytes,
        d.host_to_device_bytes,
        d.device_to_host_bytes,
        d.device_to_device_bytes,
        d.fill_bytes,
    ]
}
